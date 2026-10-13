"""Phase 18 / step 3: label variation among nearest neighbours and error vs image quality (train only, CPU).

(a) Variogram: semivariance g(d) = mean((y_i - y_j)^2) / 2 over clean training pairs, binned by distance d in a representation;
    extrapolating to d -> 0 gives a nugget = label noise + whatever the representation does not capture: only an UPPER bound on
    label noise given this representation (missing drivers look like noise there); the true noise level is unknown.
(b) Error vs image quality: RMSE of the best blend's OOF by quality decile; the clean end bounds (noise + model error).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; r = np.load(C / "oof" / "p18_residual_005.npy")
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F = rd("features.parquet"); E = np.load(C / "p12" / "emb17_dinos_252.npy")[itr]
q = stats.rankdata(F.q_noise.values) + stats.rankdata(F.m_sigma_px.values); cl = q <= np.median(q)
reps = {"DINOv2-S (20 PCs)": StandardScaler().fit_transform(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(E))),
        "OOF prediction of 005 (1-D)": (y - r)[:, None]}
out = {}
for nm, R in reps.items():
    d = pdist(R[cl]); yy = y[cl]; dy2 = pdist(yy[:, None], "sqeuclidean") / 2
    qs = np.quantile(d, [0, .005, .01, .02, .05, .1, .25, .5, 1]); g = [dy2[(d >= a) & (d < b)].mean() for a, b in zip(qs[:-1], qs[1:])]
    mid = [np.median(d[(d >= a) & (d < b)]) for a, b in zip(qs[:-1], qs[1:])]
    sl, ic = np.polyfit(mid[:4], g[:4], 1)                          # linear extrapolation of the 4 closest bins to d = 0
    out[nm] = dict(bins=[float(x) for x in g], nugget=float(ic), sqrt_nugget=float(np.sqrt(max(ic, 0))), sill=float(dy2.mean()))
    print(f"{nm:30s} semivariance by distance bin (closest→all): {' '.join(f'{x:.0f}' for x in g)} | nugget {ic:.0f} → sd {np.sqrt(max(ic, 0)):.1f} | sill {dy2.mean():.0f} (sd {np.sqrt(dy2.mean()):.1f})")
dec = pd.qcut(q, 10, labels=False); rm = [float(np.sqrt(np.mean(r[dec == k] ** 2))) for k in range(10)]
print("OOF RMSE of 005 by image-quality decile (0 = cleanest):", " ".join(f"{x:.1f}" for x in rm))
out["rmse_by_quality_decile"] = rm; out["label_sd"] = float(y.std())
json.dump(out, open(REP / "p18_ceiling.json", "w"), indent=1)
