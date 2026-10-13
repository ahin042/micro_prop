"""Phase 19 / step 2a: multi-layer frozen DINOv2 features (inference only, every image independently; nothing fitted on test).

For every block of a frozen DINOv2 ViT at 252 px: [CLS, patch-token mean, patch-token std, GeM3 = (mean |token|^3)^(1/3)],
for the identity view and averaged over the 8 D4 views (TTA). Saved as cache/p12/emb19_<arch>_252_{view0,d4mean}.npy,
shape (n_images, n_blocks, 4, dim), float32. CPU by default.
Usage: python src/p19_dino_feats.py --arch dinos [--threads 5]
"""
import argparse, json, time

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image

from common import C, D, ROOT, load_meta, sha256

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="dinos", choices=["dinos", "dinob"]); ap.add_argument("--threads", type=int, default=5)
ap.add_argument("--size", type=int, default=252); args = ap.parse_args(); torch.set_num_threads(args.threads)
ARCH = {"dinos": ("vit_small_patch14_dinov2", "vit_small_patch14_dinov2.lvd142m"), "dinob": ("vit_base_patch14_dinov2", "vit_base_patch14_dinov2.lvd142m")}[args.arch]
wf = ROOT / "weights" / f"{ARCH[1]}.safetensors"; assert sha256(wf) == json.load(open(ROOT / "src" / "weights_manifest.json"))[ARCH[1]]["sha256"]
M, itr, ite, y = load_meta(); ids = M.ID.values
IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids)]))[:, None].float() / 255
m = timm.create_model(ARCH[0], pretrained=True, in_chans=1, num_classes=0, img_size=args.size, pretrained_cfg_overlay=dict(file=str(wf))).eval()
d4 = lambda x, k, f: (lambda z: torch.flip(z, (3,)) if f else z)(torch.rot90(x, k, (2, 3)))


@torch.no_grad()
def feats(view):
    out = []
    for i in range(0, len(IMG), 25):
        x = Fn.interpolate((d4(IMG[i:i + 25], *view) - .449) / .226, size=(args.size, args.size), mode="bilinear", align_corners=False)
        h = m.norm_pre(m._pos_embed(m.patch_embed(x))); per = []
        for bi, blk in enumerate(m.blocks):
            h = blk(h); z = m.norm(h) if bi == len(m.blocks) - 1 else h; p = z[:, m.num_prefix_tokens:]
            per.append(torch.stack([z[:, 0], p.mean(1), p.std(1), p.abs().pow(3).mean(1).pow(1 / 3)], 1))
        out.append(torch.stack(per, 1).numpy())
    return np.concatenate(out).astype(np.float32)                           # (n, blocks, 4, dim)


t0 = time.time(); V = []
for k in range(4):
    for f in (0, 1):
        V.append(feats((k, f))); print(f"view {len(V)}/8 done ({time.time() - t0:.0f}s)", flush=True)
np.save(C / "p12" / f"emb19_{args.arch}_{args.size}_view0.npy", V[0]); np.save(C / "p12" / f"emb19_{args.arch}_{args.size}_d4mean.npy", np.mean(V, 0))
print("saved", V[0].shape)
