"""Phase 19 / step 1: learning curves of the two Ridge components of 005 (train only, CPU).

Sizes 100/200/300/400: the training part of every RepeatedKFold(5x3, seed 42) fold is subsampled (fixed seed per fold); every
size is scored on the SAME held-out 100 images, so the points are paired. 450 / 475: RepeatedKFold(10x3) / (20x3), seed 42.
Power law RMSE(n) = a + b * n^-c fitted on the paired 5-fold points -> extrapolated RMSE at 500 and the asymptote a.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import RepeatedKFold

from common import C, CHEAP, REP, load_meta, ridge, splits

M, itr, ite, y = load_meta(); ids = M.ID.values
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd("p13/latnet_pred.parquet")
cols = sorted(set([c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")] + CHEAP + [c for c in G if c.startswith("ph_")] + [c for c in P if c.startswith("px_")] +
                  [c for c in O if c.endswith("_coh")] + [c for c in Dd if c.startswith(("d_", "p_"))] + [c for c in LN if c.startswith("ln_")]))
A = pd.concat([F, M[CHEAP].reset_index(drop=True), G, P, O, Dd, LN], axis=1); A = A.loc[:, ~A.columns.duplicated()]
Xf = A[cols].iloc[itr].values.astype(float)                                  # NaNs imputed with the training-subset median inside fit()
Xd = np.load(C / "p12" / "emb17_dinos_252.npy")[itr]


def fit(X, a, b):
    med = np.nan_to_num(np.nanmedian(X[a], 0)); Z = np.where(np.isfinite(X), X, med); return ridge(Z[a], y[a], Z[b])


comps = {"Ridge[features + latent net]": [Xf], "Ridge[DINOv2-S]": [Xd], "mean of both": [Xf, Xd]}
res = {k: {} for k in comps}; sp = splits(len(y))
for n in (100, 200, 300, 400):
    for k, Xs in comps.items():
        r = []
        for fi, (a, b) in enumerate(sp):
            aa = np.random.default_rng(1000 + fi).permutation(a)[:n] if n < len(a) else a
            p = np.mean([fit(X, aa, b) for X in Xs], 0); r.append(np.sqrt(np.mean((p - y[b]) ** 2)))
        res[k][n] = (float(np.mean(r)), float(np.std(r) / np.sqrt(len(r))))
for n, k_split in ((450, 10), (475, 20)):
    rk = list(RepeatedKFold(n_splits=k_split, n_repeats=3, random_state=42).split(np.arange(len(y))))
    for k, Xs in comps.items():
        r = [np.sqrt(np.mean((np.mean([fit(X, a, b) for X in Xs], 0) - y[b]) ** 2)) for a, b in rk]; res[k][n] = (float(np.mean(r)), float(np.std(r) / np.sqrt(len(r))))
law = lambda n, a, b, c: a + b * n ** (-c); out = {}
print(f"{'component':30s} " + " ".join(f"{n:>13d}" for n in (100, 200, 300, 400, 450, 475)) + "   fit: RMSE(500)  asymptote  slope 400→500")
for k in comps:
    ns = np.array([100, 200, 300, 400]); m = np.array([res[k][n][0] for n in ns])
    try:
        (a_, b_, c_), _ = curve_fit(law, ns, m, p0=(m[-1] - 1, 50, .5), bounds=([0, 0, .05], [30, 1e5, 3]), maxfev=20000)
        e500, slope = law(500, a_, b_, c_), law(500, a_, b_, c_) - law(400, a_, b_, c_)
    except Exception:
        a_ = e500 = slope = np.nan
    out[k] = dict(points={str(n): v for n, v in res[k].items()}, rmse500=float(e500), asymptote=float(a_), slope_400_500=float(slope))
    print(f"{k:30s} " + " ".join(f"{res[k][n][0]:7.3f}±{res[k][n][1]:.2f}" for n in (100, 200, 300, 400, 450, 475)) + f"   {e500:9.3f}  {a_:9.3f}  {slope:+.3f}")
json.dump(out, open(REP / "p19_learning_curve.json", "w"), indent=1)
