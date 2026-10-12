"""Phase 17: frozen-backbone probe. Per-image embeddings from a frozen pretrained network (inference only, every image
independently; nothing is fitted on test), then Ridge under the shared RepeatedKFold(5x3, seed 42).
Embedding = [CLS, mean, std] of the patch tokens of the last N blocks (ViT) or [GAP, std] of the last stages (CNN), at
one or more input sizes. Weights: vendored in weights/ (sha256 in src/cnn2.py / weights/MANIFEST_p17.json).
Pre-declared candidate set (user decision, Phase 17): DINOv2 {Small, Base, Large}, frozen only, all reported; sizes 126/168/224/252/448
reported for each, 252 (~1:1 with the 256 px images) declared primary before the Base/Large runs.
Usage: python src/p17_dino_probe.py --arch dinos --sizes 252,448 [--threads 4]
"""
import argparse, json, time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image

from common import C, D, REP, ROOT, cv, load_meta, ridge, sha256, splits

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="dinos"); ap.add_argument("--sizes", default="252,448")
ap.add_argument("--blocks", type=int, default=4); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--device", default="cpu")
ap.add_argument("--d4", action="store_true", help="embed all 8 D4 views; Ridge on the view mean and on views stacked as augmented rows")
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(42)
WTS = json.load(open(ROOT / "weights" / "MANIFEST_p17.json"))
ARCH = {"dinos": ("vit_small_patch14_dinov2", "vit_small_patch14_dinov2.lvd142m"), "dinob": ("vit_base_patch14_dinov2", "vit_base_patch14_dinov2.lvd142m"),
        "dinol": ("vit_large_patch14_dinov2", "vit_large_patch14_dinov2.lvd142m"), "cnxt": ("convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k"),
        "r50tv2": ("resnet50", "resnet50.tv2_in1k"), "ev2s": ("tf_efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k")}[args.arch]
wf = ROOT / "weights" / f"{ARCH[1]}.safetensors"; assert sha256(wf) == WTS[ARCH[1]]["sha256"]
dev = torch.device(args.device); M, itr, ite, y = load_meta(); ids = M.ID.values
IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids)]))[:, None].float() / 255
mean, std = 0.449, 0.226                                                   # ImageNet mean/std averaged over RGB (in_chans=1 stem)


def d4(x, k, f):
    x = torch.rot90(x, k, (2, 3)); return torch.flip(x, (3,)) if f else x


@torch.no_grad()
def embed(size, view=(0, 0)):
    vit = ARCH[0].startswith("vit"); kw = dict(img_size=size) if vit else dict(features_only=True, out_indices=(-2, -1))
    m = timm.create_model(ARCH[0], pretrained=True, in_chans=1, num_classes=0, pretrained_cfg_overlay=dict(file=str(wf)), **kw).eval().to(dev)
    out = []
    for i in range(0, len(IMG), 25):
        x = Fn.interpolate((d4(IMG[i:i + 25], *view) - mean) / std, size=(size, size), mode="bilinear", align_corners=False).to(dev)
        if vit:
            h = m.patch_embed(x); h = m._pos_embed(h); h = m.norm_pre(h); feats = []
            for bi, blk in enumerate(m.blocks):
                h = blk(h)
                if bi >= len(m.blocks) - args.blocks:
                    z = m.norm(h) if bi == len(m.blocks) - 1 else h; p = z[:, m.num_prefix_tokens:]
                    feats += [z[:, 0], p.mean(1), p.std(1)]
        else:
            feats = sum(([f.mean((2, 3)), f.std((2, 3))] for f in m(x)), [])
        out.append(torch.cat(feats, 1).cpu().numpy())
    return np.concatenate(out)


def ridge_stacked(V, a, b, alphas=np.logspace(-3, 5, 33)):
    """Ridge on the 8 D4 views of the training images as augmented rows; alpha by an inner 5-fold CV that keeps all views
    of an image in the same inner fold (sklearn's LOO would leak across views); prediction = mean over the 8 views."""
    def fitpred(tr, te_list, al):
        Xt = np.concatenate([V[v, tr] for v in range(8)]); yt = np.tile(y[tr], 8); mu, sd = Xt.mean(0), Xt.std(0) + 1e-8
        Z = (Xt - mu) / sd; ym = yt.mean(); U, S, Vt = np.linalg.svd(Z, full_matrices=False); Uy = U.T @ (yt - ym)
        out = []
        for te in te_list:
            P = [((V[v, te] - mu) / sd) @ Vt.T for v in range(8)]
            out.append(np.stack([np.mean([p @ (S / (S ** 2 + al_) * Uy) for p in P], 0) + ym for al_ in al]))
        return out
    ia = np.array_split(np.random.default_rng(0).permutation(a), 5); err = np.zeros(len(alphas))
    for k in range(5):
        tr = np.concatenate([ia[j] for j in range(5) if j != k]); err += ((fitpred(tr, [ia[k]], alphas)[0] - y[ia[k]]) ** 2).sum(1)
    best = alphas[err.argmin()]; return fitpred(a, [b], [best])[0][0], best


if __name__ == "__main__":
    sp = splits(len(y)); res = {}; allE = []
    if args.d4:
        for s in [int(v) for v in args.sizes.split(",")]:
            t0 = time.time(); V = np.stack([embed(s, (k, f)) for k in range(4) for f in (0, 1)]); np.save(C / "p12" / f"emb17_{args.arch}_{s}_d4.npy", V.astype(np.float32))
            print(f"[{args.arch} @{s} D4] {V.shape}, {time.time() - t0:.0f}s", flush=True)
            o = cv(ridge, V.mean(0)[itr], y, sp); res[f"ridge_{s}_d4mean"] = (o["mean"], o["std"], o["r2"]); np.save(C / "oof" / f"p17_{args.arch}_{s}_d4mean_oof.npy", o["oof"])
            print(f"  ridge @{s} D4 view-mean: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}", flush=True)
            oof = np.zeros((len(sp) // 5, len(y))); al = []
            for fi, (a, b) in enumerate(sp):
                pr, bst = ridge_stacked(V[:, itr], a, b); oof[fi // 5, b] = pr; al.append(bst)
            f_ = [np.sqrt(np.mean((oof[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)]
            res[f"ridge_{s}_d4stack"] = (float(np.mean(f_)), float(np.std(f_)), None); np.save(C / "oof" / f"p17_{args.arch}_{s}_d4stack_oof.npy", oof)
            print(f"  ridge @{s} D4 stacked views: {np.mean(f_):.3f} +/- {np.std(f_):.3f}  (alphas {sorted(set(np.round(al, 0)))})", flush=True)
        json.dump(res, open(REP / f"p17_probe_{args.arch}_d4.json", "w"), indent=1); raise SystemExit
    for s in [int(v) for v in args.sizes.split(",")]:
        t0 = time.time(); E = embed(s); allE.append(E); print(f"[{args.arch} @{s}] {E.shape[1]} dims, {time.time() - t0:.0f}s", flush=True)
        np.save(C / "p12" / f"emb17_{args.arch}_{s}.npy", E)
        o = cv(ridge, E[itr], y, sp); res[f"ridge_{s}"] = (o["mean"], o["std"], o["r2"]); print(f"  ridge @{s}: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}", flush=True)
        np.save(C / "oof" / f"p17_{args.arch}_{s}_oof.npy", o["oof"])
    if len(allE) > 1:
        E = np.hstack(allE); o = cv(ridge, E[itr], y, sp); res["ridge_all"] = (o["mean"], o["std"], o["r2"])
        print(f"  ridge all sizes: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}"); np.save(C / "oof" / f"p17_{args.arch}_all_oof.npy", o["oof"])
    json.dump(res, open(REP / f"p17_probe_{args.arch}.json", "w"), indent=1)
