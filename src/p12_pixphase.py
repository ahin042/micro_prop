"""Phase 12: segmentation-free dark-phase fraction (works on blurry/noisy images where grain segmentation fails).

Smooth the raw image (grains are much larger than the noise correlation length), take the KDE mode as the matrix level,
and count pixels darker than mode - g but not as dark as pores (pore cut = mode - pore_gap). No per-image normalization.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, D, load_meta


def pix_feats(split, id_):
    f = np.asarray(Image.open(D / split / f"{id_}.png")).astype(np.float64); r = {"ID": id_}
    for s in (2, 3, 4):
        g = ndi.gaussian_filter(f, s); v = g[::2, ::2].ravel()
        grid = np.linspace(v.min(), v.max(), 512); kde = stats.gaussian_kde(v[::4], bw_method=.1)(grid); mode = grid[kde.argmax()]
        hi = v[v >= mode]; sd_up = np.sqrt(np.mean((hi - mode) ** 2))       # right half-width of the matrix peak (noise + grain spread)
        r[f"px_s{s}_mode"], r[f"px_s{s}_sdup"] = mode, sd_up
        for gap in (12, 16, 20, 25, 30):
            r[f"px_s{s}_fd{gap}"] = float(np.mean((v < mode - gap) & (v > mode - 70)))
        for k in (2.5, 3.5, 5):
            r[f"px_s{s}_fk{k}"] = float(np.mean((v < mode - k * sd_up) & (v > mode - 70)))
        r[f"px_s{s}_dark_mass"] = float(np.mean(np.clip(mode - v, 0, 60)))
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); out = C / "p12" / "pixphase.parquet"
    if not out.exists():
        items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
        pd.DataFrame(Parallel(n_jobs=8)(delayed(pix_feats)(s, i) for s, i in items)).to_parquet(out)
    P = pd.read_parquet(out).set_index("ID").loc[M.ID.values[itr]].reset_index()
    G = pd.read_parquet(C / "p12" / "phase.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID.values[itr]].reset_index()
    q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.quantile(q, .25); seg_fail = G.ph_fdark_g20.isna().values
    cols = [c for c in P.columns if c.startswith("px_")]
    T = pd.DataFrame({c: dict(rho_all=stats.spearmanr(P[c], y)[0], rho_clean=stats.spearmanr(P[c][clean], y[clean])[0],
                              rho_segfail=stats.spearmanr(P[c][seg_fail], y[seg_fail])[0],
                              rho_vs_grainbased=stats.spearmanr(P[c][~seg_fail], G.ph_fdark_g20[~seg_fail])[0]) for c in cols}).T
    print(T.sort_values("rho_all", key=abs, ascending=False).head(15).round(3).to_string())
