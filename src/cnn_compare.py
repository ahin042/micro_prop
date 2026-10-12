"""Paired comparison of CNN runs (repeat-1 or full) vs 003 / lgbm_s1 on the same folds; CNN+003 blend check."""
import glob, json, sys

import numpy as np
from scipy import stats

from common import C, REP, load_meta, splits

M, itr, ite, y = load_meta(); sp = splits(len(y)); O = C / "oof"
b003 = .5 * (np.load(O / "001_cheap_gbdt_oof.npy") + np.load(O / "s1_lgbm_stage1+cheap_oof.npy"))
fr = lambda o, n: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp[:n])])
for p in sorted(glob.glob(str(O / "cnn_*_oof.npy"))):
    o = np.load(p); n = 5 * o.shape[0]; tag = p.split("cnn_")[1][:-8]
    if not np.isfinite(o).all(): continue
    o3 = np.vstack([o, np.zeros((3 - o.shape[0], len(y)))]) if o.shape[0] < 3 else o
    c, base = fr(o3, n), fr(b003, n); d = c - base; bl = fr(.5 * (o3 + b003), n); db = bl - base
    print(f"{tag:26s} folds {n:2d} | CNN {c.mean():.3f} vs 003 {base.mean():.3f}: d {d.mean():+.3f} p {stats.ttest_1samp(d, 0).pvalue:.3g} wins {(d < 0).sum()}/{n}"
          f" | 0.5*CNN+0.5*003 {bl.mean():.3f}: d {db.mean():+.3f} p {stats.ttest_1samp(db, 0).pvalue:.3g} wins {(db < 0).sum()}/{n}"
          f" | corr(CNN,003) {np.corrcoef(o3[:o.shape[0]].ravel(), b003[:o.shape[0]].ravel())[0, 1]:.3f}")
