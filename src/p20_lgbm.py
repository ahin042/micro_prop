"""Phase 20 / experiment C: LightGBM on concatenated features, per fold (train images only; test = mean of the 15 fold models).

Features per image: (a) the 174 hand-crafted features (features.parquet, prefixes q_ r_ n_ b_ m_ d_ x_, as in p18_residuals.py),
(b) frozen DINOv2-S block 11 patch-token mean (384; cache/p12/emb19_dinos_252_d4mean.npy, D4 mean),
(c) the pre-head embedding of the fine-tuned DINOv2-S (384) from each fold's own model (cnn4.py --dump_emb, 007's recipe; D4 mean).
Caveat (by construction): for the training images of a fold, (c) comes from a model trained on them (train RMSE ~5 vs ~13.5 on
held-out images), so LightGBM learns on memorised embeddings and is applied to held-out ones.
LightGBM: n_estimators 500, learning_rate 0.05, max_depth 6, num_leaves 31 (user spec), seed 42, deterministic.
Evaluation: standalone 15-fold CV; added to 007 as a 4th equal-weight component; paired tests vs 007 and 005; gate = 007 - 0.25.
Usage: python src/p20_lgbm.py [--emb_tag x_dinos_ft_e]
"""
import argparse, json, shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from common import C, REP, ROOT, SEED, load_meta, splits

ap = argparse.ArgumentParser(); ap.add_argument("--emb_tag", default="x_dinos_ft_e"); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y)); n = len(y)
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[ids].reset_index(drop=True)
HC = F[[c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_", "m_", "d_", "x_")]].values.astype(float)
FZ = np.load(C / "p12" / "emb19_dinos_252_d4mean.npy")[:, 11, 1, :]                # rows = M order
base = np.hstack([HC, FZ]); btr, bte = base[itr], base[ite]
print(f"features: hand-crafted {HC.shape[1]}, frozen DINOv2-S block 11 mean {FZ.shape[1]}", flush=True)
P = dict(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=SEED, deterministic=True, force_row_wise=True,
         n_jobs=args.threads, verbose=-1)
oof, tv = np.full((3, n), np.nan), []
for fi, (a, b) in enumerate(sp):
    E = np.load(C / "oof" / "cnn_parts" / f"{args.emb_tag}_emb_f{fi}.npy")     # rows: train images (itr order), then test images
    Xtr, Xte = np.hstack([btr, E[:n]]), np.hstack([bte, E[n:]])
    m = lgb.LGBMRegressor(**P).fit(Xtr[a], y[a])
    oof[fi // 5, b] = m.predict(Xtr[b]); tv.append(m.predict(Xte))
    print(f"  fold {fi:2d} rmse {np.sqrt(np.mean((oof[fi // 5, b] - y[b]) ** 2)):.3f}", flush=True)
np.save(C / "oof" / "p20_lgbm_oof.npy", oof); np.save(C / "oof" / "p20_lgbm_test.npy", np.mean(tv, 0))
(ROOT / "predictions").mkdir(exist_ok=True)
for f in ("p20_lgbm_oof.npy", "p20_lgbm_test.npy"): shutil.copyfile(C / "oof" / f, ROOT / "predictions" / f)

L = lambda f: np.load(C / "oof" / f)
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
c18, rf, ds, ft = (L(f) for f in ["cnn_r18_final_a_oof.npy", "p17_ridge004_oof.npy", "p17_dinos_252_oof.npy", "cnn_x_dinos_ft_t_oof.npy"])
ref = {"007": fr((ft + rf + ds) / 3), "005": fr((c18 + rf + ds) / 3)}
res = dict(emb_tag=args.emb_tag, params={k: v for k, v in P.items() if k != "n_jobs"}, gate=float(ref["007"].mean() - .25))
for nm, o in [("standalone", oof), ("007 + LightGBM as 4th", (ft + rf + ds + oof) / 4)]:
    f = fr(o); r = res[nm] = dict(cv=float(f.mean()), sd=float(f.std()), expected_lb=[round(f.mean() - 1.17 - .3, 2), round(f.mean() - 1.17 + .3, 2)])
    line = f"{nm:22s} {f.mean():.3f} ± {f.std():.3f}"
    for k, rf_ in ref.items():
        d = f - rf_; r[f"vs_{k}"] = dict(d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
        line += f" | vs {k} d {d.mean():+.3f} p {r[f'vs_{k}']['p']:.2g} {r[f'vs_{k}']['wins']}/15"
    print(line)
res["passes_gate"] = bool(res["007 + LightGBM as 4th"]["cv"] <= res["gate"])
print(f"007 {ref['007'].mean():.3f} | gate {res['gate']:.3f} | {'PASS' if res['passes_gate'] else 'fail'}")
json.dump(res, open(REP / "p20_lgbm.json", "w"), indent=1)
