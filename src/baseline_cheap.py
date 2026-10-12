"""P1-P3: anchor table (+LightGBM), target-transform choice by CV, first submission from cached cheap stats."""
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from common import C, CHEAP, REP, SEED, cv, dist, lgbm, load_meta, ridge, splits, target, write_sub

M, itr, ite, y = load_meta()
X, Xte = M[CHEAP].values[itr], M[CHEAP].values[ite]
sp = splits(len(y)); res = {}

# P1: anchor table (const / ridge / kNN reproduce stage0.json exactly; LightGBM added)
th, sg = np.load(C / "stage0_thumbs.npy"), np.load(C / "stage0_sigs.npz")
res["const"] = cv(lambda a, b, c: np.full(len(c), b.mean()), X, y, sp)
res["ridge_cheap"] = cv(ridge, X, y, sp)
for k, A, dih in [("raw32", th, False), ("struct64", sg["struct64"], True), ("radial", sg["radial"], False)]:
    Dt = dist(A, A, dih)[np.ix_(itr, itr)]; np.fill_diagonal(Dt, 0); I = np.arange(len(y))[:, None]
    res[f"knn5_{k}"] = cv(lambda a, b, c: KNeighborsRegressor(5, metric="precomputed").fit(Dt[np.ix_(a[:, 0], a[:, 0])], b)
                          .predict(Dt[np.ix_(c[:, 0], a[:, 0])]), I, y, sp)
res["lgbm_cheap"] = cv(lgbm, X, y, sp)
s0 = json.load(open(REP / "stage0.json"))["baseline"]
for k in s0: assert abs(s0[k]["mean"] - res[k]["mean"]) < 1e-9, (k, s0[k]["mean"], res[k]["mean"])
print("== P1 anchors: RepeatedKFold(5x3, seed 42) ==")
for k, v in res.items(): print(f"  {k:16s} RMSE {v['mean']:.3f} +/- {v['std']:.3f}   R2 {v['r2']:+.3f}")

# P2: target transform, decided by CV RMSE on the original scale
tt = {}
for mname, fn in [("ridge", ridge), ("lgbm", lgbm)]:
    for kind in ["raw", "log", "boxcox"]:
        tt[f"{mname}_{kind}"] = res[f"{mname}_cheap"] if kind == "raw" else cv(target(fn, kind), X, y, sp)
print("== P2 target transform ==")
for k, v in tt.items(): print(f"  {k:14s} RMSE {v['mean']:.3f} +/- {v['std']:.3f}   R2 {v['r2']:+.3f}")
best = min(tt, key=lambda k: tt[k]["mean"]); mname, kind = best.split("_")
print("winner:", best)

# P3: refit on all 500, predict test, clip, validate, write
fn = target(dict(ridge=ridge, lgbm=lgbm)[mname], kind)
pt = fn(X, y, Xte)
out = write_sub(pt, M.ID.values[ite], "001_cheap_gbdt.csv", y)
print(out.head().to_string(index=False))
print(f"pred mean {out.hardness.mean():.3f} std {out.hardness.std():.3f} | train mean {y.mean():.3f} std {y.std(ddof=1):.3f}")

# P4: expected public LB (300 rows): bootstrap OOF squared errors in 300-row subsets
oof = tt[best]["oof"]; e2 = ((oof - y) ** 2).ravel(); rng = np.random.default_rng(SEED)
lb = np.sqrt([rng.choice(e2, 300).mean() for _ in range(5000)])
print(f"expected public LB RMSE {np.sqrt(e2.mean()):.3f}  (95% band for a 300-row public split: {np.percentile(lb, 2.5):.2f}-{np.percentile(lb, 97.5):.2f})")
(C / "oof").mkdir(exist_ok=True)
np.save(C / "oof" / "001_cheap_gbdt_oof.npy", oof); np.save(C / "oof" / "001_cheap_gbdt_test.npy", pt)
json.dump({k: {m: v[m] for m in ("mean", "std", "r2")} for k, v in {**res, **tt}.items()} | {"winner": best},
          open(REP / "baseline_cheap.json", "w"), indent=1)
