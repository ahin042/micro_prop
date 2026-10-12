"""Stage 2: formula recovery. T1 power law in logs, T2 enumerated small-formula search, T3 residual diagnostics."""
import itertools, json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, REP, load_meta, splits

M, itr, ite, y = load_meta()
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index().iloc[itr].reset_index(drop=True)
rs = lambda a, b: stats.spearmanr(a, b)[0]
GROUPS = {"d": ["n_ws_eqd_mean", "n_ws_eqd_aw", "n_ws_eqd_med", "n_ws_minor_mean", "n_mli", "n_gran_mean", "r_spec_peak_wl"],
          "f_dark": ["n_otsu_c0", "n_dark_t1_frac", "n_dark_frac"], "P": ["n_pore_frac"],
          "aspect": ["n_ws_aspect_mean", "n_acf_aspect", "n_mli_aniso", "r_acf_aspect"], "bnd": ["n_bnd_len", "n_bnd_frac", "r_canny"]}
LAT = {g: max(c, key=lambda k: abs(rs(F[k], y))) for g, c in GROUPS.items()}   # best estimator per latent (diagnostic)
print("latent -> estimator (rho_y):", {g: f"{k} ({rs(F[k], y):+.3f})" for g, k in LAT.items()})
x = {g: F[k].values.astype(float) for g, k in LAT.items()}
sp = splits(len(y)); eps = 1e-3


def cv_lin(Z, yt, inv):
    """OLS with intercept; CV RMSE on the original scale + mean fold R^2 in the fitted space."""
    Z1 = np.column_stack([np.ones(len(yt)), Z]); r, r2 = [], []
    for a, b in sp:
        w = np.linalg.lstsq(Z1[a], yt[a], rcond=None)[0]; p = Z1[b] @ w
        r.append(np.sqrt(np.mean((inv(p) - y[b]) ** 2))); r2.append(1 - np.mean((p - yt[b]) ** 2) / np.var(yt[b]))
    w = np.linalg.lstsq(Z1, yt, rcond=None)[0]
    return np.mean(r), np.std(r), np.mean(r2), w, 1 - np.mean((Z1 @ w - yt) ** 2) / np.var(yt)


# T1: log y = c + sum a_i log x_i  (+ additive Hall-Petch-style alternative)
L = np.column_stack([np.log(x["d"]), np.log(x["f_dark"] + eps), np.log(1 - x["P"]), np.log(x["aspect"]), np.log(x["bnd"] + eps)])
m, s, r2cv, w, r2in = cv_lin(L, np.log(y), np.exp)
print(f"\n== T1 power law: CV RMSE {m:.3f} +/- {s:.3f} | R2(log y) CV {r2cv:.3f} in-sample {r2in:.3f}")
print("   exponents:", dict(zip(["d", "f_dark", "(1-P)", "aspect", "bnd"], np.round(w[1:], 3))))
A = np.column_stack([x["d"] ** -.5, x["f_dark"], x["P"], x["aspect"], x["bnd"]])
m2, s2, r2cv2, w2, r2in2 = cv_lin(A, y, lambda p: p)
print(f"   additive y = H0 + k/sqrt(d) + ...: CV RMSE {m2:.3f} +/- {s2:.3f} | R2 CV {r2cv2:.3f} in-sample {r2in2:.3f}; coefs {np.round(w2, 2)}")

# T2: enumerate y = H0 + C * d^a * aspect^b * bnd^c * (1 + k f_dark) * g(P); 2-param OLS per fold (vectorized)
E = [-1, -.5, 0, .5, 1]; K = [0, .5, 1, 2, 4]
PF = {"1": np.ones(len(y)), **{f"(1-P)^{n}": (1 - x["P"]) ** n for n in (1, 2, 4)}, **{f"exp(-{b}P)": np.exp(-b * x["P"]) for b in (3, 10)}}
names, Z = [], []
for a, b, c, k, (pn, pv) in itertools.product(E, E, E, K, PF.items()):
    if a == b == c == k == 0 and pn == "1": continue
    names.append(f"d^{a} asp^{b} bnd^{c} (1+{k}f) {pn}")
    Z.append(x["d"] ** a * x["aspect"] ** b * (x["bnd"] + eps) ** c * (1 + k * x["f_dark"]) * pv)
Z = np.array(Z); Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-12)
se = np.zeros(len(Z))
for a, b in sp:
    za, zb = Z[:, a], Z[:, b]; mz = za.mean(1, keepdims=True); my = y[a].mean()
    beta = ((za - mz) * (y[a] - my)).sum(1, keepdims=True) / ((za - mz) ** 2).sum(1, keepdims=True)
    se += np.sqrt((((my + beta * (zb - mz)) - y[b]) ** 2).mean(1))
se /= len(sp); o = np.argsort(se)
print(f"\n== T2 searched {len(names)} formulas (H0 + C*term). Top 8 by CV RMSE:")
for i in o[:8]: print(f"   {se[i]:.3f}  {names[i]}")

# T3: residuals of the best formula vs latents and nuisance
zb = Z[o[0]]; beta = np.polyfit(zb, y, 1); res = y - np.polyval(beta, zb)
NUIS = ["q_noise", "q_lapvar", "q_lapvar_dn", "q_crete_blur", "r_blur_r0.01", "q_span99", "q_snr"]
rr = pd.Series({k: rs(res, F[k]) for k in list(LAT.values()) + NUIS}).round(3)
print("\n== T3 Spearman(residual of best formula, .):\n", rr.to_string())
oof = np.load(C / "oof" / "s1_lgbm_stage1+cheap_oof.npy").mean(0); rg = y - oof
rg_s = pd.Series({k: rs(rg, F[k]) for k in NUIS}).round(3)
print("   Spearman(OOF residual of lgbm stage1+cheap, nuisance):", rg_s.to_dict())
json.dump(dict(latents=LAT, T1=dict(cv_rmse=m, r2_log_cv=r2cv, r2_log_in=r2in, exponents=w[1:].tolist()),
               T1_additive=dict(cv_rmse=m2, r2_cv=r2cv2), T2_top=[(names[i], float(se[i])) for i in o[:20]],
               T3=rr.to_dict(), T3_gbdt=rg_s.to_dict()), open(REP / "stage2.json", "w"), indent=1)
