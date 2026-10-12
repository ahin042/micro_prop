"""Stage 1 report: estimator validity (rho with y vs rho with blur), top features, CV vs the Stage 0 floor."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, cv, lgbm, load_meta, ridge, splits, write_sub

M, itr, ite, y = load_meta()
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
FEAT = [c for c in F.columns if c.startswith(("q_", "r_", "n_", "b_"))]
X = F[FEAT].values[itr].astype(float)
BLUR = ["q_crete_blur", "q_lapvar_dn", "r_blur_r0.01"]
rs = lambda a, b: stats.spearmanr(a, b)[0]
T = pd.DataFrame({"rho_y": [rs(F[c].values[itr], y) for c in FEAT],
                  **{f"rho_{b}": [rs(F[c].values[itr], F[b].values[itr]) for c in FEAT] for b in BLUR},
                  "rho_noise": [rs(F[c].values[itr], F.q_noise.values[itr]) for c in FEAT]}, index=FEAT)
T["rho_blur_max"] = T[[f"rho_{b}" for b in BLUR]].abs().max(1)
T["blur_dominated"] = (T.rho_blur_max > T.rho_y.abs()) & ~T.index.str.startswith("q_")
print("blur estimates inter-correlation:\n", F[BLUR].iloc[itr].corr("spearman").round(2).to_string())

GS = ["n_mli", "n_ws_d_planimetric", "n_ws_eqd_mean", "n_ws_eqd_med", "n_ws_minor_mean", "n_gran_mean", "n_gran_median",
      "r_spec_peak_wl", "n_spec_peak_wl", "b_spec_peak_wl", "r_spec_r50", "n_spec_r50", "r_acf_len50", "n_acf_len50", "n_acf_len37"]
pd.set_option("display.width", 200)
print("\n== S2 grain-size estimators ==\n", T.loc[GS, ["rho_y", "rho_blur_max", "rho_noise", "blur_dominated"]].round(3).to_string())
top = T.reindex(T.rho_y.abs().sort_values(ascending=False).index)
print("\n== S5 top 15 |rho_y| ==\n", top.head(15)[["rho_y", "rho_blur_max", "rho_noise", "blur_dominated"]].round(3).to_string())
print(f"\nblur-dominated (excluding q_*): {int(T.blur_dominated.sum())}/{len(FEAT)}:", T.index[T.blur_dominated].tolist())

sp = splits(len(y)); Xc = M[CHEAP].values[itr]; res = {}
for name, Z in [("stage1", X), ("stage1+cheap", np.hstack([X, Xc]))]:
    for mn, fn in [("ridge", ridge), ("lgbm", lgbm)]:
        r = cv(fn, Z, y, sp); res[f"{mn}_{name}"] = r
        print(f"  {mn}_{name:14s} RMSE {r['mean']:.3f} +/- {r['std']:.3f}  R2 {r['r2']:+.3f}")
        np.save(C / "oof" / f"s1_{mn}_{name}_oof.npy", r["oof"])
Xall = np.hstack([F[FEAT].values.astype(float), M[CHEAP].values]); pt = lgbm(Xall[itr], y, Xall[ite])
np.save(C / "oof" / "002_s1_lgbm_test.npy", pt); write_sub(pt, M.ID.values[ite], "002_s1_lgbm.csv", y)
T.to_csv(REP / "stage1_feature_rho.csv")
json.dump({k: {m: v[m] for m in ("mean", "std", "r2")} for k, v in res.items()}, open(REP / "stage1_cv.json", "w"), indent=1)
