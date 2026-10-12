"""Phase 15 / analysis of A (darkness spectrum), B (pores) and C (what the CNN knows that the features do not)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from common import C, CHEAP, REP, SEED, cv, load_meta, ridge, splits

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; n = len(y)
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, Dd, LN, S = (rd(p) for p in ["features.parquet", "p12/phase.parquet", "p12/pixphase.parquet", "p12/orient.parquet", "p13/desc.parquet",
                                          "p13/latnet_pred.parquet", "p15/darkspec.parquet"])
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q); sp = splits(n)
fam = {"BASE": sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")) + CHEAP, "PH": [c for c in G if c.startswith("ph_")],
       "PX": [c for c in P if c.startswith("px_")], "COH": [c for c in O if c.endswith("_coh")], "DESC": [c for c in Dd if c.startswith(("d_", "p_"))],
       "LN": [c for c in LN if c.startswith("ln_")], "SPEC1": [c for c in S if c.startswith("s1_b") and c[4] in "+-"], "SPEC2": [c for c in S if c.startswith("s2_b") and c[4] in "+-"],
       "SPEC3": [c for c in S if c.startswith("s3_b") and c[4] in "+-"], "PORE": [c for c in S if c.startswith("pore")]}
A = pd.concat([F, M[CHEAP].iloc[itr].reset_index(drop=True), G, P, O, Dd, LN, S], axis=1); A = A.loc[:, ~A.columns.duplicated()]
num = A.select_dtypes("number").columns; A[num] = A[num].fillna(A[num].median())
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
r2 = lambda o, m: float(1 - np.mean((o.mean(0)[m] - y[m]) ** 2) / np.var(y[m]))
best_fs = ["BASE", "PH", "PX", "COH", "DESC", "LN"]; res = {}; ref = None
print("== A: darkness spectrum ==")
for nm, fs in [("best (A+DESC+LN)", best_fs), ("SPEC2 only", ["SPEC2"]), ("SPEC1+2+3 only", ["SPEC1", "SPEC2", "SPEC3"]), ("LN + SPEC2", ["LN", "SPEC2"]),
               ("best + SPEC2", best_fs + ["SPEC2"]), ("best + SPEC1+2+3", best_fs + ["SPEC1", "SPEC2", "SPEC3"]), ("best + PORE", best_fs + ["PORE"]),
               ("best + SPEC1+2+3 + PORE", best_fs + ["SPEC1", "SPEC2", "SPEC3", "PORE"])]:
    cols = sorted(set(sum((fam[f] for f in fs), []))); o = cv(ridge, A[cols].values.astype(float), y, sp); f_ = fr(o["oof"])
    ref = f_ if ref is None else ref; d = f_ - ref
    res[nm] = dict(cv=float(f_.mean()), sd=float(f_.std()), r2_clean=r2(o["oof"], clean), r2_dirty=r2(o["oof"], ~clean), d=float(d.mean()),
                   p=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins=int((d < 0).sum()))
    np.save(C / "p15" / f"oof_{nm.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')}.npy", o["oof"])
    r = res[nm]; print(f"  {nm:26s} dim {len(cols):3d} CV {r['cv']:.3f}±{r['sd']:.3f} | R2 clean {r['r2_clean']:.3f} dirty {r['r2_dirty']:.3f} | vs best d {r['d']:+.3f} p {r['p']:.2g} wins {r['wins']}/15", flush=True)

# coefficient curve: hardness contribution per darkness band (sigma 2), Ridge on raw fractions, bootstrap CI
Xs = S[fam["SPEC2"]].values; lo = np.array([int(c.split("_b")[1]) for c in fam["SPEC2"]]); rng = np.random.default_rng(SEED)
fit = lambda ix, m: Ridge(alpha=1e-3).fit(Xs[ix], y[ix]).coef_
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for a_, (nm, m) in zip(ax, [("all 500", np.ones(n, bool)), ("clean half", clean)]):
    ix0 = np.flatnonzero(m); co = fit(ix0, m); B = np.array([fit(rng.choice(ix0, len(ix0)), m) for _ in range(300)])
    co_rel = co - co[lo == 0][0]; Bl = B - B[:, [list(lo).index(0)]]
    a_.fill_between(lo + 2.5, np.percentile(Bl, 5, 0), np.percentile(Bl, 95, 0), alpha=.25); a_.plot(lo + 2.5, co_rel, "o-")
    a_.axhline(0, c="k", lw=.5); a_.axvline(0, c="r", lw=.5, ls="--")
    a_.set(xlabel="gray offset from matrix mode (sigma=2 smoothed pixels)", ylabel="HV per unit area fraction (relative to matrix band)",
           title=f"darkness spectrum coefficients, {nm} (90% bootstrap band)"); a_.set_ylim(np.percentile(Bl, 3), np.percentile(Bl, 97))
    res[f"coef_{nm}"] = dict(zip(lo.tolist(), co_rel.round(1).tolist()))
fig.tight_layout(); (REP / "p15").mkdir(exist_ok=True); fig.savefig(REP / "p15" / "darkspec_coef.png", dpi=95)
print("coefficient (rel. to matrix band, all 500):", res["coef_all 500"])

print("== B: pores ==")
best_oof = np.load(C / "p15" / "oof_best_ApDESCpLN.npy").mean(0); resid = y - best_oof
for c in fam["PORE"] + ["d_pore_frac", "d_pore_n", "n_pore_frac", "n_pore_count", "ln_pores"]:
    v = A[c].values; print(f"  {c:14s} rho y all {stats.spearmanr(v, y)[0]:+.3f} clean {stats.spearmanr(v[clean], y[clean])[0]:+.3f} | rho residual(best) {stats.spearmanr(v, resid)[0]:+.3f}")
json.dump(res, open(REP / "p15_analyze.json", "w"), indent=1, default=float)
