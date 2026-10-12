"""Part A: ceiling diagnostics. A1/A2 feature-space twin test, A3 residual structure of the best model."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, REP, cv, lgbm, load_meta, splits

M, itr, ite, y = load_meta(); n = len(y)
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
S1 = [c for c in F.columns if c.startswith(("q_", "r_", "n_", "b_"))]
T = pd.read_csv(REP / "stage1_feature_rho.csv", index_col=0)
Q = [c for c in S1 if c.startswith("q_")]
CLEAN = [c for c in S1 if not T.loc[c, "blur_dominated"] and c not in Q]
BLURD = [c for c in S1 if T.loc[c, "blur_dominated"]]
X = pd.concat([F[S1], M[CHEAP]], axis=1).iloc[itr].astype(float); X = X[sorted(X.columns)]   # canonical column order
gmd = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * np.sort(y)) / (n * (n - 1))
iu = np.triu_indices(n, 1); dy = np.abs(y[iu[0]] - y[iu[1]])


def twin(cols, whiten):
    Z = StandardScaler().fit_transform(X[cols].values)
    if whiten: Z = PCA(n_components=0.95, whiten=True, random_state=42).fit_transform(Z)
    G = (Z ** 2).sum(1); d = (G[:, None] + G[None] - 2 * Z @ Z.T)[iu]; o = np.argsort(d)
    nn = np.sqrt(np.clip(G[:, None] + G[None] - 2 * Z @ Z.T, 0, None)); np.fill_diagonal(nn, np.inf); j = nn.argmin(1)
    return {**{f"top{p}%": float(dy[o[:int(len(o) * p / 100)]].mean()) for p in (1, 5, 25)},
            "nn1": float(np.abs(y - y[j]).mean()), "dims": Z.shape[1], "rho_dist_dy": float(stats.spearmanr(d, dy)[0])}


R = {"gmd": gmd, "sets": {"all(s1+cheap)": len(X.columns), "clean_structural": len(CLEAN), "blur_dominated": len(BLURD), "q_nuisance": len(Q)}}
print(f"Gini mean difference {gmd:.3f}; set sizes {R['sets']}")
for name, cols in [("all(s1+cheap)", list(X.columns)), ("clean_structural", CLEAN), ("blur_dominated", BLURD), ("q_nuisance", Q)]:
    for w in (False, True):
        k = f"{name}|{'pca-white95' if w else 'zscore'}"; R[k] = twin(cols, w)
        print(f"  {k:36s} " + "  ".join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}" for a, b in R[k].items()))

# A3: residual structure of the best model (0.5 ridge-cheap + 0.5 lgbm-s1+cheap)
oof = .5 * (np.load(C / "oof" / "001_cheap_gbdt_oof.npy") + np.load(C / "oof" / "s1_lgbm_stage1+cheap_oof.npy")); res = (y - oof).ravel(); pred = oof.ravel(); r1 = y - oof.mean(0)
sh = stats.shapiro(r1)
R["A3"] = dict(res_sd=float(r1.std()), skew=float(stats.skew(r1)), exkurt=float(stats.kurtosis(r1)), shapiro_p=float(sh.pvalue),
               rho_absres_pred=float(stats.spearmanr(np.abs(res), pred)[0]), rho_res_pred=float(stats.spearmanr(res, pred)[0]),
               rho_res_y=float(stats.spearmanr(r1, y)[0]))
rr = pd.Series({c: stats.spearmanr(r1, X[c])[0] for c in X.columns})
pv = pd.Series({c: stats.spearmanr(r1, X[c])[1] for c in X.columns}) * len(X.columns)
R["A3"].update(max_abs_rho_res_feat=float(rr.abs().max()), argmax=rr.abs().idxmax(), n_sig_bonf=int((pv < .05).sum()))
q = np.quantile(r1, [.1, .5, .9]); R["A3"]["res_sd_by_pred_tercile"] = [float(r1[b].std()) for b in np.array_split(np.argsort(oof.mean(0)), 3)]
rcv = cv(lgbm, X.values, r1, splits(n)); R["A3"]["residual_learnable_R2"] = rcv["r2"]
print("A3:", json.dumps(R["A3"], default=float))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
stats.probplot(r1, plot=ax[0]); ax[0].set_title("QQ of OOF residual (blend 003)")
ax[1].scatter(oof.mean(0), r1, s=6, alpha=.5); ax[1].axhline(0, c="k", lw=.5); ax[1].set(xlabel="OOF pred", ylabel="residual", title="residual vs predicted")
ax[2].scatter(y, r1, s=6, alpha=.5); ax[2].axhline(0, c="k", lw=.5); ax[2].set(xlabel="y", ylabel="residual", title="residual vs y")
fig.tight_layout(); fig.savefig(REP / "a3_residuals.png", dpi=110)
json.dump(R, open(REP / "ceiling.json", "w"), indent=1, default=float)
