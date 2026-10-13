"""Phase 19 / MicroNet step 1: frozen NASA MicroNet ResNet encoders (MIT; allowed by the official Q&A of 2026-10-13).

Per image (inference only, nothing fitted on test): grayscale replicated to 3 channels, ImageNet normalization, 256 px;
for layer2/3/4 the ReLU output pooled as [mean, std, GeM3]; averaged over the 8 D4 views. Pre-declared grid, all reported:
layer sets {L4, L3+4, L2+3+4} x stats {mean+std, +GeM} x heads {Ridge, RBF kernel ridge} (heads/CV as in src/p19_heads.py),
nested configuration selection for the honest gate number; blends with 005's components (replace its DINOv2 Ridge / add 4th).
Usage: python src/p19_micronet.py --model resnet18_imagenet_micronet [--threads 5]
"""
import argparse, json, time

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image
from scipy import stats

from common import C, D, REP, ROOT, load_meta, sha256, splits
import p19_lib as ph

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--threads", type=int, default=5); args, _ = ap.parse_known_args()
torch.set_num_threads(args.threads); MAN = json.load(open(ROOT / "src" / "weights_manifest.json")); name = f"micronet_{args.model}"
wf = ROOT / "weights" / f"{name}.pth"; assert sha256(wf) == MAN[name]["sha256"], "missing or modified weights"
arch = args.model.split("_")[0]; M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y)); nrep = len(sp) // 5
out_f = C / "p12" / f"emb19_{name}_256_d4mean.npz"
if not out_f.exists():
    sd = torch.load(wf, map_location="cpu", weights_only=True); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "").replace("encoder.", ""): v for k, v in sd.items()}
    m = timm.create_model(arch, pretrained=False, num_classes=0, features_only=True, out_indices=(2, 3, 4))
    miss, unexp = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    print(f"[{name}] loaded: missing {len(miss)} {miss[:4]} | unexpected {len(unexp)} {unexp[:4]}"); assert not [k for k in miss if "num_batches" not in k], "weights do not match"
    m.eval(); IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids)]))[:, None].float() / 255
    mean, std = torch.tensor([.485, .456, .406])[None, :, None, None], torch.tensor([.229, .224, .225])[None, :, None, None]
    d4 = lambda x, k, f: (lambda z: torch.flip(z, (3,)) if f else z)(torch.rot90(x, k, (2, 3)))
    acc, t0 = None, time.time()
    with torch.no_grad():
        for k in range(4):
            for f in (0, 1):
                feats = []
                for i in range(0, len(IMG), 50):
                    x = (d4(IMG[i:i + 50], k, f).repeat(1, 3, 1, 1) - mean) / std
                    feats.append([torch.stack([z.mean((2, 3)), z.std((2, 3)), z.clamp_min(0).pow(3).mean((2, 3)).pow(1 / 3)], 1).numpy() for z in m(x)])
                lay = [np.concatenate([b[j] for b in feats]) for j in range(3)]      # per layer: (n, 3 stats, channels)
                acc = lay if acc is None else [a + l for a, l in zip(acc, lay)]
        print(f"[{name}] 8 views in {time.time() - t0:.0f}s", flush=True)
    np.savez(out_f, *[a / 8 for a in acc])
Z = np.load(out_f); LAY = [Z[f"arr_{j}"][itr] for j in range(3)]                    # layer2, layer3, layer4
LSETS = {"L4": [2], "L34": [1, 2], "L234": [0, 1, 2]}; STS = {"ms": [0, 1], "msg": [0, 1, 2]}
X_of = lambda l, s: np.hstack([LAY[j][:, STS[s]].reshape(len(y), -1) for j in LSETS[l]])
HEADS = {"ridge": ph.h_ridge, "krr": ph.kernel_head("krr")}
cfgs = {f"{l}/{s}/{h}": (l, s, h) for l in LSETS for s in STS for h in HEADS}
oof = {k: np.zeros((nrep, len(y))) for k in cfgs}; isc = {k: np.zeros(len(sp)) for k in cfgs}
for fi, (a, b) in enumerate(sp):
    for k, (l, s, h) in cfgs.items():
        Xa, Xb = ph.std_fit(X_of(l, s)[a], X_of(l, s)[b]); p, sc = HEADS[h](Xa, y[a], Xb); oof[k][fi // 5, b] = p
        isc[k][fi] = sc if sc is not None else ph.ridge_inner(Xa, y[a])
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
L = lambda n: np.load(C / "oof" / n); c18, rf, ds = L("cnn_r18_final_a_oof.npy"), L("p17_ridge004_oof.npy"), L("p17_dinos_252_oof.npy"); b005 = fr((c18 + rf + ds) / 3)
res = {}
for k in cfgs:
    f1, fa, frp = fr(oof[k]), fr((c18 + rf + ds + oof[k]) / 4), fr((c18 + rf + oof[k]) / 3)
    res[k] = dict(single=f1.mean(), add=fa.mean(), d_add=(fa - b005).mean(), wins_add=int((fa < b005).sum()), replace=frp.mean(), d_replace=(frp - b005).mean())
sel = [min(cfgs, key=lambda k: isc[k][fi]) for fi in range(len(sp))]; nest = np.zeros((nrep, len(y)))
for fi, (a, b) in enumerate(sp): nest[fi // 5, b] = oof[sel[fi]][fi // 5, b]
fa = fr((c18 + rf + ds + nest) / 4); d = fa - b005
print(f"[{name}] best single " + ", ".join(f"{k} {v['single']:.3f}" for k, v in sorted(res.items(), key=lambda kv: kv[1]["single"])[:3]))
print(f"[{name}] nested: single {fr(nest).mean():.3f} | add as 4th {fa.mean():.3f}  vs 005 d {d.mean():+.3f} p {stats.ttest_1samp(d, 0).pvalue:.2g} wins {(d < 0).sum()}/15 | gate {'PASS' if d.mean() <= -.25 else 'fail'}")
np.save(C / "oof" / f"p19_{name}_nested_oof.npy", nest)
json.dump(dict(configs=res, nested=dict(single=float(fr(nest).mean()), add=float(fa.mean()), d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))),
          open(REP / f"p19_{name}.json", "w"), indent=1, default=float)
