"""Phase 13 / Part 2: images WITHOUT dark phase (clean half, abs20 fraction < 0.03): what drives y when the dominant latent is absent?"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats

from common import C, CHEAP, D, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, Dd, O, X_, P = rd("features.parquet"), rd("p13/desc.parquet"), rd("p12/orient.parquet"), rd("p13/mix.parquet"), rd("p12/pixphase.parquet")
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px)
nd = np.flatnonzero((q <= np.median(q)) & (Dd.d_fd_abs20.values < .03))
print(f"no-dark clean images: n={len(nd)}, y range {y[nd].min():.0f}-{y[nd].max():.0f}, sd {y[nd].std():.1f} (all-image sd {y.std():.1f})")
A = pd.concat([F, Dd, O, X_, P, M[CHEAP].iloc[itr].reset_index(drop=True)], axis=1); A = A.loc[:, ~A.columns.duplicated()].select_dtypes("number")
rr = pd.Series({c: stats.spearmanr(A[c].values[nd], y[nd], nan_policy="omit")[0] for c in A.columns if A[c].values[nd].std() > 0}).dropna()
print(rr.reindex(rr.abs().sort_values(ascending=False).index).head(25).round(3).to_string())
o = nd[np.argsort(y[nd])]; sel = list(o[:8]) + list(o[-8:])
t, cap = 256, 30; W = Image.new("L", (8 * t, 2 * (t + cap) + 20), 255); d = ImageDraw.Draw(W)
d.text((4, 2), "NO dark phase (clean): TOP = 8 lowest y, BOTTOM = 8 highest y | fixed global contrast [84,194]", fill=0)
for k, i in enumerate(sel):
    r_, c_ = divmod(k, 8); x0, y0 = c_ * t, 20 + r_ * (t + cap); im = np.asarray(Image.open(D / "train" / f"{ids[i]}.png")).astype(float)
    W.paste(Image.fromarray(np.clip((im - 84) / 110 * 255, 0, 255).astype(np.uint8)), (x0, y0))
    d.text((x0 + 3, y0 + t + 2), f"{ids[i][-4:]} y={y[i]:.0f} sz={Dd.d_eqd_mean[i]:.0f} ar={Dd.d_asp_aw[i]:.1f}", fill=0)
    d.text((x0 + 3, y0 + t + 15), f"al={O.or_s3_coh[i]:.2f} po={Dd.d_pore_frac[i]:.3f} glsd={X_.mx_gl_sd[i]:.1f}", fill=0)
W.save(REP / "p13" / "nodark_extremes.png")
