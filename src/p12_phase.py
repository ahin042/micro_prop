"""Phase 12: mode-aware dark-phase fraction from per-grain (noise-averaged) raw intensities.

Otsu always splits a distribution in two, so on single-phase images it invents ~40% "dark phase". Here the matrix level
is the KDE mode of area-weighted grain means, its spread a MAD around the mode, and a grain counts as dark phase only if
it is darker than the mode by max(k*MAD, gap). Cached for all 1500 images in cache/p12/phase.parquet.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, D, REP, cv, lgbm, load_meta, ridge, splits
from features import disk, extract_array


def phase_feats(split, id_):
    a = np.asarray(Image.open(D / split / f"{id_}.png")); f = a.astype(np.float64)
    o, m = extract_array(a, id_, masks=True, ext=False); lab, bm, pores = m["lab"], m["bm"], m["pores"]
    inner = ~ndi.binary_dilation(bm | pores, disk(2))
    idx = np.unique(lab[inner]); ain = ndi.sum(inner, lab, idx); mu = ndi.mean(f, lab * inner, idx); afull = ndi.sum(np.ones_like(f), lab, idx)
    ok = ain >= 30; mu, ar = mu[ok], afull[ok]; r = {"ID": id_, "ph_ngrains": int(ok.sum())}
    if ok.sum() < 5: return r
    grid = np.linspace(mu.min() - 5, mu.max() + 5, 400); kde = stats.gaussian_kde(mu, bw_method=.15, weights=ar)(grid)
    mode = grid[kde.argmax()]; near = np.abs(mu - mode) < 10
    mad = 1.4826 * np.median(np.abs(mu[near] - mode)) if near.sum() >= 3 else 3.0
    pore = np.median(f[pores]) if pores.sum() >= 10 else np.percentile(f, .1)
    tot = ar.sum(); dark_rel = np.clip(mode - mu, 0, None)
    r.update(ph_mode=mode, ph_mad=mad, ph_pore=pore, ph_range=mode - pore,
             ph_darkness=float((ar * dark_rel).sum() / tot), ph_darkness_rel=float((ar * dark_rel).sum() / tot / max(mode - pore, 1)))
    for k in (3, 5, 8):
        d = mu < mode - max(k * mad, 6); r[f"ph_fdark_k{k}"] = float(ar[d].sum() / tot); r[f"ph_ndark_k{k}"] = int(d.sum())
    for g in (10, 15, 20, 30):
        d = mu < mode - g; r[f"ph_fdark_g{g}"] = float(ar[d].sum() / tot)
        r[f"ph_gap_g{g}"] = float(mode - np.average(mu[d], weights=ar[d])) if d.any() else 0.0
    d = mu < mode - max(5 * mad, 6)
    r["ph_dark_level_rel"] = float((mode - np.average(mu[d], weights=ar[d])) / max(mode - pore, 1)) if d.any() else 0.0
    r["ph_bright_frac"] = float(ar[mu > mode + max(5 * mad, 6)].sum() / tot)
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); (C / "p12").mkdir(parents=True, exist_ok=True); out = C / "p12" / "phase.parquet"
    if not out.exists():
        items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
        pd.DataFrame(Parallel(n_jobs=8)(delayed(phase_feats)(s, i) for s, i in items)).to_parquet(out)
    P = pd.read_parquet(out).set_index("ID").loc[M.ID.values[itr]].reset_index()
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.quantile(q, .25)
    cols = [c for c in P.columns if c.startswith("ph_")]
    T = pd.DataFrame({c: dict(rho_all=stats.spearmanr(P[c], y, nan_policy="omit")[0], rho_clean=stats.spearmanr(P[c][clean], y[clean], nan_policy="omit")[0],
                              nan=int(P[c].isna().sum())) for c in cols}).T.sort_values("rho_all", key=abs, ascending=False)
    print(T.round(3).to_string())
    X = P[cols].fillna(P[cols].median()).values; S1 = sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_"))
    sp = splits(len(y)); res = {}
    for nm, Z in [("phase", X), ("phase+s1", np.hstack([X, F[S1].values.astype(float)]))]:
        for mn, fn in [("ridge", ridge), ("lgbm", lgbm)]:
            o = cv(fn, Z, y, sp); res[f"{mn}_{nm}"] = (o["mean"], o["std"], o["r2"]); np.save(C / "p12" / f"oof_{mn}_{nm}.npy", o["oof"])
            print(f"  {mn}_{nm}: {o['mean']:.3f} +/- {o['std']:.3f}  R2 {o['r2']:.3f}", flush=True)
    json.dump(dict(rho=T.to_dict("index"), cv=res), open(REP / "p12_phase.json", "w"), indent=1, default=float)
