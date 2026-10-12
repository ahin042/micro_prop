"""Phase 13 / Part 2a-b: matched-pair differential inspection on the clean half.

Pairs that are nearly identical in standardized latent space (dark fraction, alignment, grain size, aspect, porosity,
boundary density) but differ by |dy| > 25. Montage (fixed global contrast, full resolution) + paired tests of ~30
candidate descriptors (within each pair: high-y member minus low-y member).
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats

from common import C, D, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, Dd, O = rd("features.parquet"), rd("p13/desc.parquet"), rd("p12/orient.parquet")
LAT = {"fdark": Dd.d_fd_abs20, "align": O["or_s3_coh"], "size": Dd.d_eqd_mean, "aspect": Dd.d_asp_aw, "pores": Dd.d_pore_frac, "gbdens": Dd.d_gb_density}
q = pd.Series(stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px)); ok = (q <= q.median()).values & Dd.d_fd_abs20.notna().values
idx = np.flatnonzero(ok); Z = np.column_stack([v.values[idx] for v in LAT.values()]); Z = (Z - Z.mean(0)) / Z.std(0)
dist = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); dy = np.abs(y[idx][:, None] - y[idx][None])
iu = np.triu_indices(len(idx), 1); cand = np.flatnonzero(dy[iu] > 25); thr = np.quantile(dist[iu], .05)
cand = cand[dist[iu][cand] < thr]; cand = cand[np.argsort(dist[iu][cand])]
pairs = [(idx[iu[0][c]], idx[iu[1][c]]) for c in cand]
pairs = [(a, b) if y[a] < y[b] else (b, a) for a, b in pairs]                  # (low-y, high-y)
print(f"clean half n={len(idx)}; latent-distance 5th pct = {thr:.2f}; matched pairs with |dy|>25: {len(pairs)}")

# montage of the 16 closest distinct pairs
seen, show = set(), []
for a, b in pairs:
    if a in seen or b in seen: continue
    show.append((a, b)); seen |= {a, b}
    if len(show) == 16: break
t, cap = 256, 44; W = Image.new("L", (8 * t + 3 * 12, 4 * (t + cap) + 24), 255); d = ImageDraw.Draw(W)
d.text((4, 4), "matched pairs (near-identical latents): LEFT = low y, RIGHT = high y | fixed global contrast [84,194]", fill=0)
for k, (a, b) in enumerate(show):
    r_, c_ = divmod(k, 4); x0 = c_ * (2 * t + 12); y0 = 24 + r_ * (t + cap)
    for j, i in enumerate((a, b)):
        im = np.asarray(Image.open(D / "train" / f"{ids[i]}.png")).astype(float)
        W.paste(Image.fromarray(np.clip((im - 84) / 110 * 255, 0, 255).astype(np.uint8)), (x0 + j * t, y0))
        d.text((x0 + j * t + 3, y0 + t + 2), f"{ids[i][-4:]} y={y[i]:.0f}", fill=0)
        d.text((x0 + j * t + 3, y0 + t + 15), f"fd={LAT['fdark'][i]:.2f} al={LAT['align'][i]:.2f} sz={LAT['size'][i]:.0f}", fill=0)
        d.text((x0 + j * t + 3, y0 + t + 28), f"ar={LAT['aspect'][i]:.1f} po={LAT['pores'][i]:.3f}", fill=0)
(REP / "p13").mkdir(exist_ok=True); W.save(REP / "p13" / "matched_pairs.png")

# paired tests over all matched pairs
DESC = [c for c in Dd.columns if c.startswith("d_") and not c.startswith(("d_fd_abs", "d_fd_mad", "d_fd_rng"))] + \
       [c for c in Dd.columns if c.startswith("p_fd_")] + ["q_noise", "m_sigma_px", "q_span95", "q_span99", "n_grain_mu_std", "n_otsu_c0", "n_otsu_c1", "n_otsu_c2"]
A = pd.concat([Dd, F], axis=1); A = A.loc[:, ~A.columns.duplicated()]
lo, hi = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); res = {}
for c in DESC:
    v = A[c].values.astype(float); dlt = v[hi] - v[lo]; m = ~np.isnan(dlt)
    if m.sum() < 10 or np.nanstd(v) == 0: continue
    s = stats.wilcoxon(dlt[m]) if np.any(dlt[m] != 0) else None
    res[c] = dict(frac_hi_gt_lo=float((dlt[m] > 0).mean()), median_delta_z=float(np.median(dlt[m]) / np.nanstd(v)), p=float(s.pvalue) if s else 1.0, n=int(m.sum()))
R = pd.DataFrame(res).T.sort_values("p"); pd.set_option("display.width", 200); print(R.head(25).round(4).to_string())
json.dump(dict(n_pairs=len(pairs), dist_thr=float(thr), tests=R.to_dict("index")), open(REP / "p13_pairs.json", "w"), indent=1)
