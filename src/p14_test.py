"""Phase 14 / Steps 4-5: do per-phase Hall-Petch / rule-of-mixtures aggregates beat the 6 global latents?

(a) 6 current latents: dark fraction, alignment, grain size, aspect, porosity, boundary density
(b) (a) + per-phase Hall-Petch terms of one scheme: f_p (p>=1) and f_p * d_p^-0.5 (all p)
(c) (a) + every aggregate of one scheme
Schemes: {bins, km2, km3} x {all grains, interior grains}. The scheme is also chosen INSIDE each training fold by inner
5-fold CV ("nested"), which is the honest number. Evaluated (i) on all 500 with the usual 5x3 folds (OOF R^2 reported on
the clean / dirty half) and (ii) by CV restricted to the clean half. Step 5: OLS physical form with bootstrap CIs.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold

from common import C, REP, SEED, load_meta, ridge, splits

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; n = len(y)
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, Dd, O, AG = rd("features.parquet"), rd("p13/desc.parquet"), rd("p12/orient.parquet"), rd("p14/agg.parquet")
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q)
LAT6 = pd.DataFrame({"fdark": Dd.d_fd_abs20, "align": O.or_s3_coh, "size": Dd.d_eqd_mean, "aspect": Dd.d_asp_aw, "pores": Dd.d_pore_frac, "gbdens": Dd.d_gb_density})
SCH = [f"{s}_{b}" for s in ("bins", "km2", "km3") for b in ("all", "inn")]
hp_cols = lambda s: [c for c in AG.columns if c.startswith(s + "_") and (c.endswith("_fdm05") or (c.endswith("_f") and not c.endswith("_p0_f")))]
all_cols = lambda s: [c for c in AG.columns if c.startswith(s + "_")] + [c for c in AG.columns if c.startswith("g_" + s.split("_")[1])]
X0 = pd.concat([LAT6, AG], axis=1); X0["seg_ok"] = AG["km2_all_p0_f"].notna().astype(float)
X0 = X0.fillna(X0.median(numeric_only=True))                              # y-free imputation; seg_ok flags the imputed rows
design = lambda cols: X0[list(LAT6.columns) + ["seg_ok"] + cols].values.astype(float)


def run_cv(get_cols, rows, sp):
    """get_cols(train_idx) -> feature columns (lets the scheme be chosen inside the fold). Returns fold RMSEs and OOF."""
    oof = np.full((3, n), np.nan); fr = []
    for i, (a, b) in enumerate(sp):
        a, b = rows[a], rows[b]; Z = design(get_cols(a)); p = ridge(Z[a], y[a], Z[b]); oof[i // 5, b] = p; fr.append(np.sqrt(np.mean((p - y[b]) ** 2)))
    return np.array(fr), oof


def nested(colfn):
    def pick(a):
        best, be = None, np.inf
        for s in SCH:
            Z = design(colfn(s)); e = sum(np.sum((ridge(Z[a[ia]], y[a[ia]], Z[a[ib]]) - y[a[ib]]) ** 2) for ia, ib in KFold(5, shuffle=True, random_state=SEED).split(a))
            if e < be: be, best = e, s
        picks.append(best); return colfn(best)
    return pick


r2 = lambda o, m: float(1 - np.nanmean((np.nanmean(o, 0)[m] - y[m]) ** 2) / np.var(y[m]))
res = {}; picks = []
for setting, rows, sp in [("all500", np.arange(n), splits(n)),
                          ("clean_only", np.flatnonzero(clean), list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.flatnonzero(clean))))]:
    cfg = {"(a) 6 latents": lambda a: []}
    for s in SCH: cfg[f"(b) +HP {s}"] = (lambda s: lambda a: hp_cols(s))(s); cfg[f"(c) +all {s}"] = (lambda s: lambda a: all_cols(s))(s)
    cfg["(b) +HP nested-scheme"] = nested(hp_cols); cfg["(c) +all nested-scheme"] = nested(all_cols)
    base = None
    for nm, fn in cfg.items():
        picks.clear(); fr, oof = run_cv(fn, rows, sp); base = fr if base is None else base; d = fr - base
        m_eval = clean if setting == "all500" else np.zeros(n, bool)
        if setting != "all500": m_eval[rows] = True
        res[(setting, nm)] = dict(rmse=float(fr.mean()), rmse_sd=float(fr.std()), r2_clean=r2(oof, clean & ~np.isnan(oof).all(0)),
                                  r2_dirty=r2(oof, ~clean) if setting == "all500" else np.nan, r2_all=r2(oof, ~np.isnan(oof).all(0)),
                                  d_vs_a=float(d.mean()), p_vs_a=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins=int((d < 0).sum()),
                                  picks=pd.Series(picks).value_counts().to_dict() if picks else None)
        r = res[(setting, nm)]
        print(f"[{setting:10s}] {nm:26s} RMSE {r['rmse']:.3f}±{r['rmse_sd']:.3f} | R2 clean {r['r2_clean']:.3f} dirty {r['r2_dirty']:.3f} all {r['r2_all']:.3f} "
              f"| vs (a) d {r['d_vs_a']:+.3f} p {r['p_vs_a']:.2g} wins {r['wins']}/15" + (f" | picks {r['picks']}" if r["picks"] else ""), flush=True)

ok = AG["km2_all_p0_f"].notna().values
rho = pd.DataFrame({c: dict(rho_clean=stats.spearmanr(AG[c][clean & ok], y[clean & ok], nan_policy="omit")[0], rho_all=stats.spearmanr(AG[c][ok], y[ok], nan_policy="omit")[0])
                    for c in AG.columns if c != "ID" and AG[c][ok].std() > 0}).T.sort_values("rho_clean", key=abs, ascending=False)
print("top aggregates by |Spearman| on the clean half (segmented images):"); print(rho.head(15).round(3).to_string())

# Step 5: y ~ H0 + sum_p f_p (a_p + k_p d_p^-0.5) + c * porosity  (OLS; f_0 dropped because sum f_p = 1 is absorbed by H0)
phys = {}
for s in ("bins_all", "km3_all", "km2_all", "bins_inn", "km3_inn", "km2_inn", "LAT6-reference"):
    P = 0 if s.startswith("LAT6") else 4 if s.startswith("bins") else int(s[2])
    cols = [f"{s}_p{p}_f" for p in range(1, P)] + [f"{s}_p{p}_fdm05" for p in range(P)]
    for sub, m in (("clean", clean & ok), ("all_seg", ok)):
        if s.startswith("LAT6"): Z, names_ = LAT6[m].fillna(LAT6.median()).values, list(LAT6.columns)
        else: Z, names_ = np.c_[AG.loc[m, cols].fillna(0).values, Dd.d_pore_frac.values[m]], None
        yy = y[m]; Z = np.nan_to_num(Z)
        coef = LinearRegression().fit(Z, yy).coef_; rng = np.random.default_rng(SEED)
        boot = np.array([LinearRegression().fit(Z[ix], yy[ix]).coef_ for ix in (rng.integers(0, len(yy), len(yy)) for _ in range(500))])
        cvr2 = []
        for a, b in RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(Z):
            p = LinearRegression().fit(Z[a], yy[a]).predict(Z[b]); cvr2.append(1 - np.mean((p - yy[b]) ** 2) / np.var(yy[b]))
        names = names_ or [c.replace(s + "_", "") for c in cols] + ["porosity"]
        phys[(s, sub)] = dict(n=int(m.sum()), cv_r2=float(np.mean(cvr2)), coef={k: (float(v), float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5)))
                                                                                 for j, (k, v) in enumerate(zip(names, coef))})
        kp = {k: v for k, v in phys[(s, sub)]["coef"].items() if k.endswith("fdm05")}
        print(f"[phys {s} {sub} n={m.sum()}] CV R2 {np.mean(cvr2):.3f} | k_p (Hall-Petch, HV*px^0.5) with 95% CI: " +
              ", ".join(f"{k.split('_')[0]} {v[0]:+.0f} [{v[1]:+.0f},{v[2]:+.0f}]" for k, v in kp.items()))
json.dump(dict(cv={f"{a}|{b}": v for (a, b), v in res.items()}, phys={f"{a}|{b}": v for (a, b), v in phys.items()}, rho=rho.head(40).to_dict("index")),
          open(REP / "p14_test.json", "w"), indent=1, default=float)
