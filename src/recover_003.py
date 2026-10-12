"""Reconstruct the exact submission 003 that scored public LB 13.2852 (first upload).

Root cause of the drift: the Part-B refactor of features.py moved the six b_ columns (position 46 -> after the
segmentation block). LightGBM's feature_fraction samples columns BY POSITION, so the same features in a different
order give a different model. This script restores the pre-refactor column order (src/legacy_s1_columns.json) and the
pre-lockdown LightGBM parameters, then rebuilds 001-ridge + 002-lgbm and their 50/50 blend.
Outputs: submissions/003_recovered.csv, cache/oof/003_recovered_{oof,test}.npy (baseline for paired tests).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, ROOT, SEED, cv, load_meta, splits, write_sub

LEGACY = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, deterministic=True, force_row_wise=True)


def lgbm_legacy(Xtr, ytr, Xte, seeds=(0, 1, 2)):
    ds = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    h = lgb.cv(dict(LEGACY, seed=SEED), ds, num_boost_round=3000, nfold=5, stratified=False, seed=SEED,
               callbacks=[lgb.early_stopping(200, verbose=False)])
    nr = len(next(iter(h.values())))
    return np.mean([lgb.train(dict(LEGACY, seed=s), ds, num_boost_round=nr).predict(Xte) for s in seeds], 0)


def ridge_legacy(Xtr, ytr, Xte):
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 4, 29))).fit(Xtr, ytr).predict(Xte)


if __name__ == "__main__":
    M, itr, ite, y = load_meta()
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
    cols = json.load(open(ROOT / "src" / "legacy_s1_columns.json"))
    Xc = M[CHEAP].values.astype(float); X = np.hstack([F[cols].values.astype(float), Xc])
    p001, p002 = ridge_legacy(Xc[itr], y, Xc[ite]), lgbm_legacy(X[itr], y, X[ite])
    out = write_sub(.5 * (p001 + p002), M.ID.values[ite], "003_recovered.csv", y)
    sp = splits(len(y)); o1, o2 = cv(ridge_legacy, Xc[itr], y, sp), cv(lgbm_legacy, X[itr], y, sp)
    oof = .5 * (o1["oof"] + o2["oof"]); (C / "oof").mkdir(exist_ok=True)
    np.save(C / "oof" / "003_recovered_oof.npy", oof); np.save(C / "oof" / "003_recovered_test.npy", .5 * (p001 + p002))
    fr = [np.sqrt(np.mean((oof[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)]
    print(f"003_recovered: ridge CV {o1['mean']:.3f}, lgbm CV {o2['mean']:.3f}, blend CV {np.mean(fr):.3f} +/- {np.std(fr):.3f}; "
          f"test mean {out.hardness.mean():.3f}")
