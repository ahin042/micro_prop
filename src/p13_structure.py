"""Phase 13 / Part 2c: rule-of-mixtures darkness descriptors + structure of y vs latents on the clean half.

Matched pairs suggested hardness rises with the area-weighted RELATIVE darkness of grains measured from the brightest
phase (not from the matrix mode), and with darker darkest-phase levels. Descriptors (per image, from per-grain raw means):
  top_ref    = 90th pct of area-weighted grain means (brightest population)
  dk_top     = sum(area * clip(top_ref - mu, 0, 80)) / sum(area)        (absolute gray levels)
  dk_top_g{g}= area fraction with mu < top_ref - g
  gl_iqr / gl_sd / gl_p10gap = spread of grain gray levels
Then: scatter y vs dark fraction coloured by size/aspect/alignment; linear vs GBDT vs additive-spline (GAM) vs power law.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import C, D, REP, load_meta
from features import disk, extract_array


def mix(split, id_):
    a = np.asarray(Image.open(D / split / f"{id_}.png")); f = a.astype(np.float64)
    o, m = extract_array(a, id_, masks=True, ext=False); lab = m["lab"]
    inner = ~ndi.binary_dilation(m["bm"] | m["pores"], disk(2)); idx = np.unique(lab[inner])
    ain = ndi.sum(inner, lab, idx); ar = ndi.sum(np.ones_like(f), lab, idx); mu = ndi.mean(f, lab * inner, idx); k = ain >= 30
    r = {"ID": id_}
    if k.sum() < 6: return r
    mu, ar = mu[k], ar[k]; o_ = np.argsort(mu); cw = np.cumsum(ar[o_]) / ar.sum()
    q = lambda p: mu[o_][np.searchsorted(cw, p)]
    top = q(.9); r.update(mx_top=top, mx_p50=q(.5), mx_p10=q(.1), mx_dk_top=float((ar * np.clip(top - mu, 0, 80)).sum() / ar.sum()),
                          mx_gl_iqr=q(.75) - q(.25), mx_gl_sd=float(np.sqrt(np.cov(mu, aweights=ar))), mx_p10gap=top - q(.1), mx_p50gap=top - q(.5))
    for g in (10, 20, 30, 40): r[f"mx_dk_top_g{g}"] = float(ar[mu < top - g].sum() / ar.sum())
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; out = C / "p13" / "mix.parquet"
    if not out.exists():
        import p13_structure as me
        items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
        pd.DataFrame(Parallel(n_jobs=6)(delayed(me.mix)(s, i) for s, i in items)).to_parquet(out)
    rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
    F, Dd, O, X_ = rd("features.parquet"), rd("p13/desc.parquet"), rd("p12/orient.parquet"), rd("p13/mix.parquet")
    q_ = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = (q_ <= np.median(q_)) & Dd.d_fd_abs20.notna().values & X_.mx_dk_top.notna().values
    print("rho with y (all-seg / clean half):")
    for c in [c for c in X_.columns if c.startswith("mx_")] + ["d_fd_abs20", "p_fd_abs10"]:
        v = (X_ if c in X_ else Dd)[c].values; ok = ~np.isnan(v)
        print(f"  {c:14s} {stats.spearmanr(v[ok], y[ok])[0]:+.3f} / {stats.spearmanr(v[clean], y[clean])[0]:+.3f}")
    LAT = pd.DataFrame({"fdark": Dd.d_fd_abs20, "pfd10": Dd.p_fd_abs10, "dk_top": X_.mx_dk_top, "gl_iqr": X_.mx_gl_iqr, "align": O.or_s3_coh,
                        "size": Dd.d_eqd_mean, "aspect": Dd.d_asp_aw, "pores": Dd.d_pore_frac, "gbdens": Dd.d_gb_density, "eqd_skew": Dd.d_eqd_skew})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a_, c in zip(ax, ["size", "aspect", "align"]):
        s = a_.scatter(LAT.fdark[clean], y[clean], c=LAT[c][clean], s=12, cmap="viridis"); plt.colorbar(s, ax=a_, label=c)
        a_.set(xlabel="dark-phase fraction (abs20)", ylabel="hardness y", title=f"clean half, colour = {c}")
    fig.tight_layout(); fig.savefig(REP / "p13" / "y_vs_fdark.png", dpi=95)
    Z = LAT[clean].values; yy = y[clean]; res = {}
    models = {"linear (ridge)": make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 4, 25))),
              "additive splines (GAM-like)": make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), RidgeCV(np.logspace(-1, 4, 25))),
              "GBDT (hist, shallow)": HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=300, min_samples_leaf=15, random_state=0)}
    for sets in [["fdark", "align", "size", "aspect", "pores", "gbdens"], list(LAT.columns)]:
        for nm, mdl in models.items():
            r2 = []
            for a, b in RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(Z):
                cols_i = [LAT.columns.get_loc(c) for c in sets]; mdl.fit(Z[a][:, cols_i], yy[a]); p = mdl.predict(Z[b][:, cols_i])
                r2.append(1 - np.mean((p - yy[b]) ** 2) / np.var(yy[b]))
            res[f"{nm} | {len(sets)} latents"] = (float(np.mean(r2)), float(np.std(r2))); print(f"  {nm:30s} {len(sets):2d} latents: CV R2 {np.mean(r2):.3f} +/- {np.std(r2):.3f}")
    pos = ["pfd10", "dk_top", "gl_iqr", "align", "size", "aspect"]; Lg = np.log(np.clip(LAT[pos][clean].values, 1e-3, None))
    r2 = []
    for a, b in RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(Lg):
        mdl = RidgeCV(np.logspace(-4, 2, 13)).fit(Lg[a], np.log(yy[a])); p = np.exp(mdl.predict(Lg[b])); r2.append(1 - np.mean((p - yy[b]) ** 2) / np.var(yy[b]))
    ex = RidgeCV(np.logspace(-4, 2, 13)).fit(Lg, np.log(yy)).coef_
    print(f"  power law log y ~ sum a_i log x_i: CV R2 {np.mean(r2):.3f}; exponents", dict(zip(pos, np.round(ex, 3))))
    res["power law"] = (float(np.mean(r2)), dict(zip(pos, ex.tolist())))
    json.dump(res, open(REP / "p13_structure.json", "w"), indent=1)
