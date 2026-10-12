"""Shared paths, CV protocol, target transforms and models (all seeds fixed)."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D, C, REP, SUB = ROOT / "data", ROOT / "cache", ROOT / "reports", ROOT / "submissions"
SEED = 42
CHEAP = [f"pct{q}" for q in [1, 5, 25, 50, 75, 95, 99]] + [f"band{k}" for k in range(8)] + [
    "mean", "std", "skew", "kurt", "entropy", "n_levels", "gap_bins", "sat0", "sat255", "log_lapvar", "hf_std", "grad_mean", "fsize"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, deterministic=True,
                  force_row_wise=True)


def load_meta(path=C / "stage0_meta.parquet"):
    M = pd.read_parquet(path).merge(pd.read_csv(D / "train.csv"), on="ID", how="left")
    itr, ite = np.flatnonzero(M.split.values == "train"), np.flatnonzero(M.split.values == "test")
    return M, itr, ite, M.hardness.values[itr]


def splits(n):
    return list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.arange(n)))


def cv(fn, X, y, sp=None):
    """fn(Xtr, ytr, Xva) -> pred. Returns fold RMSE mean/std, mean fold R^2 and OOF (n_repeats, n)."""
    sp = sp or splits(len(y)); oof = np.zeros((len(sp) // 5, len(y))); r, r2 = [], []
    for f, (a, b) in enumerate(sp):
        p = fn(X[a], y[a], X[b]); oof[f // 5, b] = p
        r.append(np.sqrt(np.mean((p - y[b]) ** 2))); r2.append(1 - np.mean((p - y[b]) ** 2) / np.var(y[b]))
    return dict(mean=float(np.mean(r)), std=float(np.std(r)), r2=float(np.mean(r2)), oof=oof)


def target(fn, kind):
    """Wrap a model so it is trained on T(y) and predicts on the original scale (Box-Cox lambda fit on train fold only)."""
    def g(Xtr, ytr, Xte):
        if kind == "raw": return fn(Xtr, ytr, Xte)
        if kind == "log": return np.exp(fn(Xtr, np.log(ytr), Xte))
        yt, lam = stats.boxcox(ytr); return inv_boxcox(fn(Xtr, yt, Xte), lam)
    return g


def ridge(Xtr, ytr, Xte):
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 4, 29))).fit(Xtr, ytr).predict(Xte)


def lgbm(Xtr, ytr, Xte, seeds=(0, 1, 2), params=LGB_PARAMS):
    """Rounds chosen by inner 5-fold lgb.cv on the training data only; then seed-averaged refit."""
    ds = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    h = lgb.cv(dict(params, seed=SEED), ds, num_boost_round=3000, nfold=5, stratified=False, seed=SEED,
               callbacks=[lgb.early_stopping(200, verbose=False)])
    nr = len(next(iter(h.values())))
    return np.mean([lgb.train(dict(params, seed=s), ds, num_boost_round=nr).predict(Xte) for s in seeds], 0)


def zn(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), 1e-6)


def dist(A, B, dih=False):
    """Same Gram-trick distance as stage0_recon.py (dih=True: min over the 8 D4 transforms of B)."""
    A, B = zn(A), zn(B)
    if not dih: return np.clip(2 - 2 * A @ B.T / A.shape[1], 0, None)
    s = int(np.sqrt(B.shape[1])); Bi = B.reshape(-1, s, s); G = None
    for Bf in (Bi, Bi[:, :, ::-1]):
        for k in range(4):
            g = A @ np.rot90(Bf, k, (1, 2)).reshape(len(B), -1).T
            G = g if G is None else np.maximum(G, g)
    return np.clip(2 - 2 * G / A.shape[1], 0, None)


def write_sub(pred_test, ids_test, name, y_train):
    """Clip to train range, align to sample_submission ID order, validate, save."""
    ss = pd.read_csv(D / "sample_submission.csv")
    s = pd.Series(np.clip(pred_test, y_train.min(), y_train.max()), index=ids_test).loc[ss.ID]
    out = pd.DataFrame({ss.columns[0]: ss.ID.values, ss.columns[1]: s.values})
    assert len(out) == len(ss) == 1000 and out.iloc[:, 1].notna().all() and (out.ID.values == ss.ID.values).all()
    assert list(out.columns) == list(ss.columns)
    SUB.mkdir(exist_ok=True); out.to_csv(SUB / name, index=False)
    return out
