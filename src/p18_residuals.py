"""Phase 18 / step 1: what separates high- from low-residual training images under the current best blend (005)?

Residual r = y - OOF prediction of 005 = mean(CNN r18, Ridge004, Ridge[DINOv2-S]) averaged over the 3 CV repeats (train only).
For every feature family: Spearman(feature, r) with BH-FDR, top-vs-bottom residual quartile effect size, and whether the
residual is learnable from the family (CV R^2 of Ridge / LightGBM under RepeatedKFold(5x3, seed 7), a different split from the CV).
CPU only. Output: reports/p18_residuals.json and a printed summary.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold

from common import C, CHEAP, REP, lgbm, load_meta, ridge

M, itr, ite, y = load_meta(); ids = M.ID.values
L = lambda n: np.load(C / "oof" / n)
oof = (L("cnn_r18_final_a_oof.npy") + L("p17_ridge004_oof.npy") + L("p17_dinos_252_oof.npy")) / 3
pred = oof.mean(0); r = y - pred
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids[itr]].reset_index(drop=True)
F, G, P, O, Dd, X_, LN = (rd(p) for p in ["features.parquet", "p12/phase.parquet", "p12/pixphase.parquet", "p12/orient.parquet",
                                          "p13/desc.parquet", "p13/mix.parquet", "p13/latnet_pred.parquet"])
E = np.load(C / "p12" / "emb17_dinos_252.npy"); Z = PCA(50, random_state=0).fit_transform(E[itr])   # PCA fit on train only
fam = {"features (174)": F[[c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_", "m_", "d_", "x_")]],
       "cheap global stats": M[CHEAP].iloc[itr].reset_index(drop=True), "phase (ph_)": G[[c for c in G if c.startswith("ph_")]],
       "pixel phase (px_)": P[[c for c in P if c.startswith("px_")]], "orientation (or_)": O[[c for c in O if c.startswith("or_")]],
       "descriptors (d_/p_)": Dd[[c for c in Dd if c.startswith(("d_", "p_"))]], "mixture (mx_)": X_[[c for c in X_ if c.startswith("mx_")]],
       "latent net (ln_)": LN[[c for c in LN if c.startswith("ln_")]], "DINOv2-S PCs": pd.DataFrame(Z, columns=[f"dino_pc{k}" for k in range(50)])}
q = stats.rankdata(F.q_noise.values) + stats.rankdata(F.m_sigma_px.values); clean = q <= np.median(q)

print(f"residual sd {r.std():.2f} | Spearman(r, y) {stats.spearmanr(r, y)[0]:+.3f} (shrinkage) | slope y~pred {np.polyfit(pred, y, 1)[0]:.3f}")
print(f"|r| vs noise {stats.spearmanr(np.abs(r), F.q_noise)[0]:+.3f} | |r| vs blur {stats.spearmanr(np.abs(r), F.m_sigma_px)[0]:+.3f} | "
      f"RMSE clean half {np.sqrt(np.mean(r[clean] ** 2)):.2f} vs dirty half {np.sqrt(np.mean(r[~clean] ** 2)):.2f}")

rows = []
for fn, D in fam.items():
    for c in D.columns:
        v = D[c].values.astype(float); ok = np.isfinite(v)
        if ok.sum() < 100 or np.nanstd(v) == 0: continue
        rho, p = stats.spearmanr(v[ok], r[ok]); hi, lo = r >= np.quantile(r, .75), r <= np.quantile(r, .25)
        es = (np.nanmean(v[hi & ok]) - np.nanmean(v[lo & ok])) / (np.nanstd(v[ok]) + 1e-12)
        rows.append(dict(family=fn, feature=c, rho_r=rho, p=p, rho_y=stats.spearmanr(v[ok], y[ok])[0], es_q4_q1=es, rho_r_clean=stats.spearmanr(v[ok & clean], r[ok & clean])[0]))
T = pd.DataFrame(rows); T["q_bh"] = stats.false_discovery_control(T.p.values)
T = T.sort_values("p"); pd.set_option("display.width", 220)
print(f"\n{len(T)} features tested; BH q < .05: {(T.q_bh < .05).sum()}, q < .10: {(T.q_bh < .10).sum()}")
print(T.head(15).round(3).to_string(index=False))

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=7); res = {}
def learn(fn, X, model):
    X = X.astype(float); r2 = []
    for a, b in rkf.split(X):
        med = np.nan_to_num(np.nanmedian(X[a], 0)); Xa = np.where(np.isfinite(X), X, med)          # train-fold medians
        p = model(Xa[a], r[a], Xa[b]); r2.append(1 - np.mean((p - r[b]) ** 2) / np.mean((r[b] - r[a].mean()) ** 2))
    return float(np.mean(r2)), float(stats.ttest_1samp(r2, 0).pvalue)
print("\nIs the residual learnable? CV R^2 of r (RepeatedKFold 5x3, seed 7), mean over folds; p = t-test of fold R^2 vs 0")
for fn, D in fam.items():
    res[fn] = learn(fn, D.values.astype(float), ridge); print(f"  ridge {fn:22s} R2 {res[fn][0]:+.3f}  p {res[fn][1]:.2g}")
ALL = pd.concat(list(fam.values()), axis=1).values.astype(float)
res["ALL ridge"] = learn("all", ALL, ridge); res["ALL lgbm"] = learn("all", ALL, lgbm)
print(f"  ridge ALL families        R2 {res['ALL ridge'][0]:+.3f}  p {res['ALL ridge'][1]:.2g}")
print(f"  lgbm  ALL families        R2 {res['ALL lgbm'][0]:+.3f}  p {res['ALL lgbm'][1]:.2g}")
json.dump(dict(residual_sd=float(r.std()), n_tested=len(T), n_q05=int((T.q_bh < .05).sum()), top=T.head(30).to_dict("records"), learnable=res),
          open(REP / "p18_residuals.json", "w"), indent=1, default=float)
np.save(C / "oof" / "p18_residual_005.npy", r)
