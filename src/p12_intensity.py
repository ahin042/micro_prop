"""Phase 12: per-phase RAW intensity levels and affine-invariant contrast ratios.

Hypothesis: the generator assigns each phase a gray level and then applies a random brightness/contrast (affine)
transform. Absolute levels are then scrambled, but ratios such as (matrix - dark) / (matrix - pore) are invariant.
Segmentation (labels, pores, boundaries) comes from features.extract_array; every level is measured on the raw image.
"""
import json, sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats
from skimage import filters

from common import C, D, REP, cv, lgbm, load_meta, ridge, splits
from features import disk, extract_array


def levels(split, id_):
    a = np.asarray(Image.open(D / split / f"{id_}.png")); f = ndi.median_filter(a.astype(np.float64), 3)
    o, m = extract_array(a, id_, masks=True, ext=False)
    lab, bm, pores = m["lab"], m["bm"], m["pores"]
    inner = ~ndi.binary_dilation(bm | pores, disk(1))
    idx = np.unique(lab[inner]); area = ndi.sum(inner, lab, idx); mu = ndi.mean(f, lab * inner, idx)
    ok = (area >= 20) & np.isfinite(mu); mu, area = mu[ok], area[ok]
    r = dict(ID=id_)
    if len(mu) >= 4:
        t = filters.threshold_otsu(mu); dk = mu < t
        w = lambda v, ww: float(np.average(v, weights=ww)) if len(v) else np.nan
        r.update(L_matrix=w(mu[~dk], area[~dk]), L_dark=w(mu[dk], area[dk]) if dk.any() else np.nan,
                 L_grain_med=float(np.median(mu)), L_grain_sd=float(np.sqrt(np.cov(mu, aweights=area))), L_grain_iqr=float(np.subtract(*np.percentile(mu, [75, 25]))),
                 f_dark_area=float(area[dk].sum() / area.sum()), n_levels_grain=int(len(np.unique(np.round(mu / 4)))))
    r["L_pore"] = float(np.median(f[pores])) if pores.sum() >= 10 else np.nan
    r["L_bnd"] = float(np.median(f[bm])) if bm.any() else np.nan
    r["L_p01"], r["L_p50"], r["L_p99"] = np.percentile(f, [0.1, 50, 99]).tolist()
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta()
    out = C / "p12" / "intensity.parquet"
    if not out.exists():
        rows = Parallel(n_jobs=8)(delayed(levels)("train", i) for i in M.ID.values[itr]); pd.DataFrame(rows).to_parquet(out)
    L = pd.read_parquet(out).set_index("ID").loc[M.ID.values[itr]].reset_index()
    ref = L.L_pore.fillna(L.L_p01)                                   # internal black reference: pores, else 0.1th pct
    L["R_dark"] = (L.L_matrix - L.L_dark) / (L.L_matrix - ref)        # affine-invariant contrast ratios
    L["R_bnd"] = (L.L_matrix - L.L_bnd) / (L.L_matrix - ref)
    L["R_spread"] = L.L_grain_sd / (L.L_matrix - ref)
    L["R_iqr"] = L.L_grain_iqr / (L.L_matrix - ref)
    L["D_dark"] = L.L_matrix - L.L_dark                               # absolute gaps (gain-dependent)
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.quantile(q, .25)
    cols = [c for c in L.columns if c != "ID"]
    T = pd.DataFrame({c: dict(rho_all=stats.spearmanr(L[c], y, nan_policy="omit")[0], rho_clean=stats.spearmanr(L[c][clean], y[clean], nan_policy="omit")[0],
                              nan=int(L[c].isna().sum())) for c in cols}).T.sort_values("rho_clean", key=abs, ascending=False)
    print(T.round(3).to_string())
    X = L[cols].fillna(L[cols].median()).values; S1 = [c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")]
    sp = splits(len(y)); res = {}
    for nm, Z in [("intensity", X), ("intensity+s1", np.hstack([X, F[sorted(S1)].values.astype(float)]))]:
        for mn, fn in [("ridge", ridge), ("lgbm", lgbm)]:
            o = cv(fn, Z, y, sp); res[f"{mn}_{nm}"] = (o["mean"], o["std"], o["r2"]); print(f"  {mn}_{nm}: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}")
    json.dump(dict(rho=T.to_dict("index"), cv=res), open(REP / "p12_intensity.json", "w"), indent=1, default=float)
