"""Phase 12 / Step 1: montages on ONE fixed global contrast scale (no per-image normalization)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import C, D, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
oof = np.load(C / "oof" / "003_recovered_oof.npy").mean(0)           # best submitted blend (LB 13.2852)
imgs = np.stack([np.asarray(Image.open(D / "train" / f"{i}.png")) for i in ids]).astype(np.float32)
lo, hi = np.percentile(imgs, [0.5, 99.5])                             # one global stretch for every tile
print(f"global display scale: [{lo:.0f}, {hi:.0f}] -> [0, 255]")
OUT = REP / "p12"; OUT.mkdir(exist_ok=True)


def montage(idx, name, title, cols=6, t=256, cap=18):
    W = Image.new("L", (cols * t, ((len(idx) + cols - 1) // cols) * (t + cap) + 22), 255); d = ImageDraw.Draw(W)
    d.text((4, 4), title, fill=0)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols); x0, y0 = c * t, 22 + r * (t + cap)
        W.paste(Image.fromarray(np.clip((imgs[i] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)), (x0, y0))
        d.text((x0 + 3, y0 + t + 3), f"{ids[i][-4:]} y={y[i]:.0f} p={oof[i]:.0f} mu={imgs[i].mean():.0f}", fill=0)
    W.save(OUT / f"{name}.png"); return OUT / f"{name}.png"


o = np.argsort(y); med = np.argsort(np.abs(y - np.median(y)))[:24]
montage(o[:24], "low_y", "24 LOWEST hardness"); montage(o[-24:][::-1], "high_y", "24 HIGHEST hardness")
montage(med[np.argsort(y[med])], "median_y", "24 around the median"); montage(np.argsort(-np.abs(y - oof))[:24], "worst_resid", "24 largest |OOF residual| (blend 003)")
g = pd.DataFrame({"y": y, "mean": imgs.mean((1, 2)), "std": imgs.std((1, 2)), "p1": np.percentile(imgs, 1, axis=(1, 2)),
                  "p99": np.percentile(imgs, 99, axis=(1, 2))})
print(g.groupby(pd.qcut(g.y, 5)).mean().round(2).to_string())
