"""Phase 19 / priority 1: embedding-level D4 augmentation on frozen DINOv2 features (train only for fitting; CPU).

Per-view features (8 D4 views, blocks 4-11, [CLS, mean, std, GeM3]) are extracted once (inference only, every image independently)
and cached as float16. Head: Ridge trained on the 8 views of every training image as augmented rows (src/p19_lib.ridge_stacked_views),
prediction = mean over the 8 views. Pre-declared configs: mid4/base, mid4/gem, last4/base, mid4+last4/gem; nested selection
across them for the honest number; blends with 005's components (replace its DINOv2 Ridge / add as 4th).
Usage: python src/p19_d4aug.py --arch dinos [--threads 5]
"""
import argparse, json, time

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image
from scipy import stats

from common import C, D, REP, ROOT, load_meta, sha256, splits
from p19_lib import ridge_stacked_views

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="dinos", choices=["dinos", "dinob"]); ap.add_argument("--threads", type=int, default=5); args = ap.parse_args()
torch.set_num_threads(args.threads); M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y)); nrep = len(sp) // 5
f_cache = C / "p12" / f"emb19_{args.arch}_252_views_b4to11.npy"
if not f_cache.exists():
    ARCH = {"dinos": ("vit_small_patch14_dinov2", "vit_small_patch14_dinov2.lvd142m"), "dinob": ("vit_base_patch14_dinov2", "vit_base_patch14_dinov2.lvd142m")}[args.arch]
    wf = ROOT / "weights" / f"{ARCH[1]}.safetensors"; assert sha256(wf) == json.load(open(ROOT / "src" / "weights_manifest.json"))[ARCH[1]]["sha256"]
    m = timm.create_model(ARCH[0], pretrained=True, in_chans=1, num_classes=0, img_size=252, pretrained_cfg_overlay=dict(file=str(wf))).eval()
    IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids)]))[:, None].float() / 255
    d4 = lambda x, k, f: (lambda z: torch.flip(z, (3,)) if f else z)(torch.rot90(x, k, (2, 3)))
    V = np.zeros((len(IMG), 8, 8, 4, m.embed_dim), np.float16); t0 = time.time()
    with torch.no_grad():
        for vi, (k, f) in enumerate([(k, f) for k in range(4) for f in (0, 1)]):
            for i in range(0, len(IMG), 25):
                x = Fn.interpolate((d4(IMG[i:i + 25], k, f) - .449) / .226, size=(252, 252), mode="bilinear", align_corners=False)
                h = m.norm_pre(m._pos_embed(m.patch_embed(x)))
                for bi, blk in enumerate(m.blocks):
                    h = blk(h)
                    if bi >= 4:
                        z = m.norm(h) if bi == len(m.blocks) - 1 else h; p = z[:, m.num_prefix_tokens:]
                        V[i:i + 25, vi, bi - 4] = torch.stack([z[:, 0], p.mean(1), p.std(1), p.abs().pow(3).mean(1).pow(1 / 3)], 1).numpy().astype(np.float16)
            print(f"view {vi + 1}/8 ({time.time() - t0:.0f}s)", flush=True)
    np.save(f_cache, V)
V = np.load(f_cache, mmap_mode="r")[itr]                                    # (500, 8 views, 8 blocks [4..11], 4 stats, dim)
CF = {"mid4/base": ([0, 1, 2, 3], [0, 1, 2]), "mid4/gem": ([0, 1, 2, 3], [0, 1, 2, 3]), "last4/base": ([4, 5, 6, 7], [0, 1, 2]), "mid4+last4/gem": (list(range(8)), [0, 1, 2, 3])}
X_of = lambda c: np.asarray(V[:, :, CF[c][0]][:, :, :, CF[c][1]], dtype=np.float32).reshape(len(y), 8, -1)
oof = {c: np.zeros((nrep, len(y))) for c in CF}; isc = {c: np.zeros(len(sp)) for c in CF}; t0 = time.time()
for c in CF:
    X = X_of(c)
    for fi, (a, b) in enumerate(sp):
        p, s = ridge_stacked_views(X[a], y[a], X[b]); oof[c][fi // 5, b] = p; isc[c][fi] = s
    print(f"{c} done ({time.time() - t0:.0f}s)", flush=True)
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
L = lambda n: np.load(C / "oof" / n); c18, rf, ds = L("cnn_r18_final_a_oof.npy"), L("p17_ridge004_oof.npy"), L("p17_dinos_252_oof.npy"); b005 = fr((c18 + rf + ds) / 3)
res = {}
for c in CF:
    f1, fa, frp = fr(oof[c]), fr((c18 + rf + ds + oof[c]) / 4), fr((c18 + rf + oof[c]) / 3)
    res[c] = dict(single=f1.mean(), add=fa.mean(), d_add=(fa - b005).mean(), replace=frp.mean(), d_replace=(frp - b005).mean())
    print(f"  {c:16s} single {f1.mean():.3f} | replace 005's DINOv2 {frp.mean():.3f} ({(frp - b005).mean():+.3f}) | add as 4th {fa.mean():.3f} ({(fa - b005).mean():+.3f})")
sel = [min(CF, key=lambda c: isc[c][fi]) for fi in range(len(sp))]; nest = np.zeros((nrep, len(y)))
for fi, (a, b) in enumerate(sp): nest[fi // 5, b] = oof[sel[fi]][fi // 5, b]
out = dict(configs=res)
for nm, o in [("replace", (c18 + rf + nest) / 3), ("add", (c18 + rf + ds + nest) / 4)]:
    f = fr(o); d = f - b005; out[f"nested_{nm}"] = dict(cv=float(f.mean()), d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    print(f"nested ({nm}): {f.mean():.3f}  vs 005 d {d.mean():+.3f} p {stats.ttest_1samp(d, 0).pvalue:.2g} wins {(d < 0).sum()}/15 | gate {'PASS' if d.mean() <= -.25 else 'fail'}")
print(f"nested single {fr(nest).mean():.3f}; chosen {dict(zip(*np.unique(sel, return_counts=True)))}")
np.save(C / "oof" / f"p19_d4aug_{args.arch}_nested_oof.npy", nest)
for c in CF: np.save(C / "oof" / f"p19_d4aug_{args.arch}_{c.replace('/', '_').replace('+', 'p')}_oof.npy", oof[c])
json.dump(out, open(REP / f"p19_d4aug_{args.arch}.json", "w"), indent=1, default=float)
