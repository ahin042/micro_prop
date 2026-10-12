"""Part B report: do blur-corrected estimators (m_ MTF, d_ deconvolved, x_ scale-space) break blur domination? + CV."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, cv, lgbm, load_meta, ridge, splits

M, itr, ite, y = load_meta()
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
S1 = [c for c in F.columns if c.startswith(("q_", "r_", "n_", "b_"))]; NEW = [c for c in F.columns if c.startswith(("m_", "d_", "x_"))]
rs = lambda c, v: stats.spearmanr(F[c].values[itr], v)[0]
BLUR = ["q_crete_blur", "q_lapvar_dn", "r_blur_r0.01", "m_sigma_px"]
row = lambda c: dict(rho_y=rs(c, y), rho_sigma=rs(c, F.m_sigma_px.values[itr]) if c != "m_sigma_px" else 1.0,
                     rho_blur_max=max(abs(rs(c, F[b].values[itr])) for b in BLUR if b != c), rho_noise=rs(c, F.q_noise.values[itr]))
print("m_sigma_px vs other blur estimates:", {b: round(rs("m_sigma_px", F[b].values[itr]), 3) for b in BLUR[:3]}, "| vs noise", round(rs("m_sigma_px", F.q_noise.values[itr]), 3))
PAIRS = [("r_spec_peak_wl", "m_spec_peak_wl"), ("r_spec_r50", "m_spec_r50"), ("r_spec_centroid", "m_spec_centroid"), ("r_acf_len50", "m_knee_wl"),
         ("n_mli", "d_mli"), ("n_mli", "x_mli0"), ("n_ws_eqd_mean", "d_ws_eqd_mean"), ("n_ws_eqd_mean", "x_ws_eqd_mean0"),
         ("n_gran_mean", "d_gran_mean"), ("n_gran_mean", "x_gran_mean0"), ("n_ws_minor_mean", "d_ws_minor_mean"), ("n_ws_minor_mean", "x_ws_minor_mean0"),
         ("n_ws_d_planimetric", "d_ws_d_planimetric"), ("n_otsu_c0", "d_otsu_c0"), ("n_dark_frac", "d_dark_frac"), ("n_pore_frac", "d_pore_frac")]
T = pd.DataFrame({f"{o} -> {nw}": {**{f"old_{k}": v for k, v in row(o).items()}, **{f"new_{k}": v for k, v in row(nw).items()}} for o, nw in PAIRS}).T
T["new_breaks_blur"] = T.new_rho_y.abs() > T.new_rho_blur_max
pd.set_option("display.width", 250)
print(T[["old_rho_y", "old_rho_blur_max", "new_rho_y", "new_rho_blur_max", "new_rho_sigma", "new_rho_noise", "new_breaks_blur"]].round(3).to_string())
A = pd.DataFrame({c: row(c) for c in NEW}).T; A["breaks"] = A.rho_y.abs() > A.rho_blur_max
print(f"\nnew features breaking blur domination: {int(A.breaks.sum())}/{len(NEW)} ->", A[A.breaks].rho_y.round(3).sort_values(key=abs, ascending=False).to_dict())

sp = splits(len(y)); Xc = M[CHEAP].values[itr]; base = np.load(C / "oof" / "s1_lgbm_stage1+cheap_oof.npy"); res = {}
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
for mn, fn in [("ridge", ridge), ("lgbm", lgbm)]:
    r = cv(fn, np.hstack([F[S1 + NEW].values[itr].astype(float), Xc]), y, sp); d = fr(r["oof"]) - fr(base)
    res[mn] = dict(mean=r["mean"], std=r["std"], r2=r["r2"], d_vs_14574=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    np.save(C / "oof" / f"b_{mn}_s1+cheap+new_oof.npy", r["oof"])
    print(f"  {mn} stage1+cheap+new: RMSE {r['mean']:.3f} +/- {r['std']:.3f}  R2 {r['r2']:.3f} | vs lgbm s1+cheap 14.574: d {d.mean():+.3f} p {res[mn]['p']:.3g} wins {res[mn]['wins']}/15")
T.to_csv(REP / "partb_pairs.csv"); json.dump(dict(cv=res, n_breaking=int(A.breaks.sum())), open(REP / "partb.json", "w"), indent=1)
