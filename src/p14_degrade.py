"""Phase 14 / Step 6: stability of the grain-level aggregates under realistic degradation (CPU, numpy).

100 clean train images are degraded with (blur, contrast gain, brightness, noise) taken jointly from a random dirty-half
image, adding only the missing amount (same rule as src/p13_latnet.py::degrade). Report corr(clean value, degraded value)
per aggregate and how often segmentation fails after degradation.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, CHEAP, D, REP, SEED, load_meta
from p14_grains import aggregate, grain_table

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[ids].reset_index(drop=True)
AG = pd.read_parquet(C / "p14" / "agg.parquet").set_index("ID").loc[ids].reset_index(drop=True)
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); rng = np.random.default_rng(SEED)
clean = np.flatnonzero((q <= np.quantile(q, .4)) & AG["km2_all_p0_f"].notna().values); dirty = np.flatnonzero(q > np.median(q))
pick = rng.choice(clean, 100, replace=False); src = rng.choice(dirty, 100); seeds = rng.integers(0, 2 ** 31, 100)
NB, NN, SP, MU = F.m_sigma_px.values, F.q_noise.values, F.q_span99.values, M["mean"].values[itr]


def degraded(i, j, s):
    x = np.asarray(Image.open(D / "train" / f"{ids[i]}.png")).astype(np.float64); r = np.random.default_rng(s)
    sb = np.sqrt(max(NB[j] ** 2 - NB[i] ** 2, 0)); gain = SP[j] / SP[i]; sn = np.sqrt(max(NN[j] ** 2 - (NN[i] * gain) ** 2, 0))
    x = ndi.gaussian_filter(x, sb) if sb > .05 else x
    x = (x - MU[i]) * gain + MU[j] + r.normal(0, sn, x.shape)
    a = np.clip(np.round(x), 0, 255).astype(np.uint8); return aggregate(grain_table(a, ids[i]), ids[i]), dict(sb=sb, sn=sn, gain=gain)


out = Parallel(n_jobs=6)(delayed(degraded)(i, j, s) for i, j, s in zip(pick, src, seeds))
DG = pd.DataFrame([o[0] for o in out]).set_index("ID"); NU = pd.DataFrame([o[1] for o in out])
CL = AG.set_index(pd.Index(ids)).loc[ids[pick]]
fail = DG.reindex(CL.index)["km2_all_p0_f"].isna()
print(f"degradation applied: blur sigma_add median {NU.sb.median():.2f}, noise sigma_add median {NU.sn.median():.1f}, gain median {NU.gain.median():.2f}")
print(f"segmentation failure after degradation: {int(fail.sum())}/100 images (all 100 segmented when clean)")
KEY = [c for c in AG.columns if c.endswith(("_f", "_fdm05")) and "_all_" in c] + ["g_all_dm05", "g_all_nb"]
rows = {}
for c in KEY:
    a, b = CL[c].values, DG.reindex(CL.index)[c].values; m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() > 10 and np.std(a[m]) > 0: rows[c] = dict(corr=float(np.corrcoef(a[m], b[m])[0, 1]), n=int(m.sum()), rel_shift=float(np.median((b[m] - a[m]) / (np.abs(a[m]) + 1e-9))))
R = pd.DataFrame(rows).T.sort_values("corr"); pd.set_option("display.width", 200); print(R.round(3).to_string())
print("median corr by scheme:", {s: round(R[R.index.str.startswith(s)]["corr"].median(), 3) for s in ("bins_all", "km2_all", "km3_all", "g_all")})
json.dump(dict(fail=int(fail.sum()), stability=R.to_dict("index"), nuisance=NU.median().to_dict()), open(REP / "p14_degrade.json", "w"), indent=1, default=float)
