"""Phase 12: SIGNED orientation features. Earlier features used |cos 2θ| / |sin 2θ| (and the CNN used D4 augmentation),
which erase the difference between horizontal/vertical and between the two diagonals. Here orientation is kept signed:
c = coh*cos2φ, s = coh*sin2φ of the elongation direction φ, from the structure tensor at several scales and the ACF lobe.
Image coordinates: x to the right, y DOWN (row index), so φ < 0 means the grains descend from upper-left to lower-right.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, D, REP, cv, lgbm, load_meta, ridge, splits


def orient(split, id_):
    f = np.asarray(Image.open(D / split / f"{id_}.png")).astype(np.float64); r = {"ID": id_}
    for s in (1.5, 3, 6):
        g = ndi.gaussian_filter(f, s); gy, gx = np.gradient(g)                   # gy = d/drow (down)
        for w in (s * 2,):
            Jxx, Jyy, Jxy = (ndi.gaussian_filter(v, w).mean() for v in (gx * gx, -gy * -gy, gx * -gy))   # flip y -> math axes (y up)
        tr = Jxx + Jyy; cg, sg = (Jxx - Jyy) / tr, 2 * Jxy / tr                  # gradient-direction doubled-angle vector
        r[f"or_s{s}_c"], r[f"or_s{s}_s"] = -cg, -sg                               # elongation direction = gradient + 90°  (negate)
        r[f"or_s{s}_coh"] = np.hypot(cg, sg); r[f"or_s{s}_deg"] = np.degrees(0.5 * np.arctan2(-sg, -cg))
    x = f - f.mean(); A = np.real(np.fft.ifft2(np.abs(np.fft.fft2(x, s=(512, 512))) ** 2)); A = np.fft.fftshift(A)[128:384, 128:384]; A /= A.max()
    yy, xx = np.indices(A.shape) - 128; m = A > .3; w = A[m]; X, Y = xx[m], -yy[m]                 # math axes
    cxx, cyy, cxy = (w * X * X).sum() / w.sum(), (w * Y * Y).sum() / w.sum(), (w * X * Y).sum() / w.sum()
    tr = cxx + cyy; r["or_acf_c"], r["or_acf_s"] = (cxx - cyy) / tr, 2 * cxy / tr; r["or_acf_deg"] = np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy))
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); (C / "p12").mkdir(parents=True, exist_ok=True); out = C / "p12" / "orient.parquet"
    if not out.exists():
        items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
        pd.DataFrame(Parallel(n_jobs=8)(delayed(orient)(s, i) for s, i in items)).to_parquet(out)
    O = pd.read_parquet(out).set_index("ID").loc[M.ID.values[itr]].reset_index()
    cols = [c for c in O.columns if c.startswith("or_")]
    print(pd.DataFrame({c: dict(pearson=np.corrcoef(O[c], y)[0, 1], spearman=stats.spearmanr(O[c], y)[0]) for c in cols}).T.round(3).to_string())
    deg = O.or_acf_deg.values; b = np.digitize(deg, np.arange(-90, 91, 15)) - 1
    print("mean y by ACF elongation angle (deg, math axes; -30 = descending to the right):")
    for k in range(12):
        mk = b == k
        if mk.sum(): print(f"  [{-90 + 15 * k:+4d},{-75 + 15 * k:+4d})  n={mk.sum():3d}  y={y[mk].mean():6.1f}  coh={O.or_acf_c.values[mk].std():.2f}")
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    G = pd.read_parquet(C / "p12" / "phase.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    P = pd.read_parquet(C / "p12" / "pixphase.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    S1 = sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")); PH = [c for c in G.columns if c.startswith("ph_")]; PX = [c for c in P.columns if c.startswith("px_")]
    base = pd.concat([F[S1], G[PH], P[PX]], axis=1); base = base.fillna(base.median())
    sp = splits(len(y)); res = {}
    for nm, Z in [("orient", O[cols].values), ("s1+phase+pix", base.values), ("s1+phase+pix+orient", np.hstack([base.values, O[cols].values]))]:
        for mn, fn in [("ridge", ridge), ("lgbm", lgbm)]:
            o = cv(fn, Z.astype(float), y, sp); res[f"{mn}_{nm}"] = (o["mean"], o["std"], o["r2"]); np.save(C / "p12" / f"oof_{mn}_{nm}.npy", o["oof"])
            print(f"  {mn}_{nm}: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}", flush=True)
    json.dump(res, open(REP / "p12_orient.json", "w"), indent=1)
