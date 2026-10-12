"""Phase 15 / D: interaction test (dark phase x shape/alignment) with the degradation-robust latents (ln_*).
Hypothesis from the CNN-unique montage: dark phase adds more hardness when grains are elongated/aligned laths."""
import itertools, json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, REP, cv, load_meta, ridge, splits

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; n = len(y)
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, Dd, LN = (rd(p) for p in ["features.parquet", "p12/phase.parquet", "p12/pixphase.parquet", "p12/orient.parquet", "p13/desc.parquet", "p13/latnet_pred.parquet"])
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q); sp = splits(n)
L = LN[[c for c in LN if c.startswith("ln_")]]; Z = pd.DataFrame(StandardScaler().fit_transform(L), columns=L.columns)   # y-free scaling
pairs = {f"{a}*{b}": Z[a] * Z[b] for a, b in itertools.combinations(Z.columns, 2)}; sq = {f"{a}^2": Z[a] ** 2 for a in Z.columns}
I = pd.DataFrame(pairs); Q = pd.DataFrame(sq)
key = ["ln_fd20*ln_aspect", "ln_fd20*ln_align", "ln_pfd10*ln_aspect", "ln_pfd10*ln_align", "ln_align*ln_aspect", "ln_fd20*ln_logsize"]
fam = {"BASE": sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")) + CHEAP, "PH": [c for c in G if c.startswith("ph_")], "PX": [c for c in P if c.startswith("px_")],
       "COH": [c for c in O if c.endswith("_coh")], "DESC": [c for c in Dd if c.startswith(("d_", "p_"))], "LN": list(L.columns), "KEY": key, "PAIRS": list(I.columns), "SQ": list(Q.columns)}
A = pd.concat([F, M[CHEAP].iloc[itr].reset_index(drop=True), G, P, O, Dd, LN, I, Q], axis=1); A = A.loc[:, ~A.columns.duplicated()]
num = A.select_dtypes("number").columns; A[num] = A[num].fillna(A[num].median())
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
r2 = lambda o, m: float(1 - np.mean((o.mean(0)[m] - y[m]) ** 2) / np.var(y[m]))
print("Spearman of key interactions with y (all / clean) and with residual of LN-only ridge:")
ln_oof = cv(ridge, A[fam["LN"]].values, y, sp)["oof"].mean(0); res_ln = y - ln_oof
for c in key: print(f"  {c:22s} {stats.spearmanr(I[c], y)[0]:+.3f} / {stats.spearmanr(I[c][clean], y[clean])[0]:+.3f} | resid {stats.spearmanr(I[c], res_ln)[0]:+.3f}")
best = ["BASE", "PH", "PX", "COH", "DESC", "LN"]; ref = None; res = {}
for nm, fs in [("LN", ["LN"]), ("LN + KEY", ["LN", "KEY"]), ("LN + all pairs + sq", ["LN", "PAIRS", "SQ"]), ("best", best), ("best + KEY", best + ["KEY"]), ("best + all pairs + sq", best + ["PAIRS", "SQ"])]:
    cols = sorted(set(sum((fam[f] for f in fs), []))); o = cv(ridge, A[cols].values.astype(float), y, sp); f_ = fr(o["oof"])
    if nm in ("LN", "best"): ref = f_
    d = f_ - ref; res[nm] = dict(cv=float(f_.mean()), r2_clean=r2(o["oof"], clean), r2_dirty=r2(o["oof"], ~clean), d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins=int((d < 0).sum()))
    np.save(C / "p15" / f"oof_int_{nm.replace(' ', '_').replace('+', 'p')}.npy", o["oof"])
    r = res[nm]; print(f"  {nm:22s} dim {len(cols):3d} CV {r['cv']:.3f} | R2 clean {r['r2_clean']:.3f} dirty {r['r2_dirty']:.3f} | vs its base d {r['d']:+.3f} p {r['p']:.2g} wins {r['wins']}/15", flush=True)
json.dump(res, open(REP / "p15_interact.json", "w"), indent=1)
