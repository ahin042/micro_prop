"""Phase 17: frozen-backbone probe. Per-image embeddings from a frozen pretrained network (inference only, every image
independently; nothing is fitted on test), then Ridge under the shared RepeatedKFold(5x3, seed 42).
Embedding = [CLS, mean, std] of the patch tokens of the last N blocks (ViT) or [GAP, std] of the last stages (CNN), at
one or more input sizes. Weights: vendored in weights/ (sha256 in src/cnn2.py / weights/MANIFEST_p17.json).
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
args = ap.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(42)
WTS = json.load(open(ROOT / "weights" / "MANIFEST_p17.json"))
ARCH = {"dinos": ("vit_small_patch14_dinov2", "vit_small_patch14_dinov2.lvd142m"), "cnxt": ("convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k"),
        "r50tv2": ("resnet50", "resnet50.tv2_in1k"), "ev2s": ("tf_efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k")}[args.arch]
wf = ROOT / "weights" / f"{ARCH[1]}.safetensors"; assert sha256(wf) == WTS[ARCH[1]]["sha256"]
dev = torch.device(args.device); M, itr, ite, y = load_meta(); ids = M.ID.values
IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids)]))[:, None].float() / 255
mean, std = 0.449, 0.226                                                   # ImageNet mean/std averaged over RGB (in_chans=1 stem)


@torch.no_grad()
def embed(size):
    vit = ARCH[0].startswith("vit"); kw = dict(img_size=size) if vit else dict(features_only=True, out_indices=(-2, -1))
    m = timm.create_model(ARCH[0], pretrained=True, in_chans=1, num_classes=0, pretrained_cfg_overlay=dict(file=str(wf)), **kw).eval().to(dev)
    out = []
    for i in range(0, len(IMG), 25):
        x = Fn.interpolate((IMG[i:i + 25] - mean) / std, size=(size, size), mode="bilinear", align_corners=False).to(dev)
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


if __name__ == "__main__":
    sp = splits(len(y)); res = {}; allE = []
    for s in [int(v) for v in args.sizes.split(",")]:
        t0 = time.time(); E = embed(s); allE.append(E); print(f"[{args.arch} @{s}] {E.shape[1]} dims, {time.time() - t0:.0f}s", flush=True)
        np.save(C / "p12" / f"emb17_{args.arch}_{s}.npy", E)
        o = cv(ridge, E[itr], y, sp); res[f"ridge_{s}"] = (o["mean"], o["std"], o["r2"]); print(f"  ridge @{s}: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}", flush=True)
        np.save(C / "oof" / f"p17_{args.arch}_{s}_oof.npy", o["oof"])
    if len(allE) > 1:
        E = np.hstack(allE); o = cv(ridge, E[itr], y, sp); res["ridge_all"] = (o["mean"], o["std"], o["r2"])
        print(f"  ridge all sizes: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}"); np.save(C / "oof" / f"p17_{args.arch}_all_oof.npy", o["oof"])
    json.dump(res, open(REP / f"p17_probe_{args.arch}.json", "w"), indent=1)
