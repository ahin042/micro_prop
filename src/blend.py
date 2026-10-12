"""Part D: OOF blending across model families. Equal weights vs NNLS (honestly cross-fitted per fold); paired test vs best."""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from common import C, CHEAP, REP, lgbm, load_meta, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", default=""); ap.add_argument("--out", default="")
args = ap.parse_args()
M, itr, ite, y = load_meta(); sp = splits(len(y)); O = C / "oof"
if not (O / "b_lgbm_s1new_test.npy").exists():                    # test preds for the Part B family
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
    cols = [c for c in F.columns if c.startswith(("q_", "r_", "n_", "b_", "m_", "d_", "x_"))]
    X = np.hstack([F[cols].values.astype(float), M[CHEAP].values]); np.save(O / "b_lgbm_s1new_test.npy", lgbm(X[itr], y, X[ite]))
FAM = {"ridge_cheap": ("001_cheap_gbdt_oof.npy", "001_cheap_gbdt_test.npy"),
       "lgbm_s1": ("s1_lgbm_stage1+cheap_oof.npy", "002_s1_lgbm_test.npy"),
       "lgbm_s1new": ("b_lgbm_s1+cheap+new_oof.npy", "b_lgbm_s1new_test.npy")}
if args.cnn: FAM["cnn"] = (f"cnn_{args.cnn}_oof.npy", f"cnn_{args.cnn}_test.npy")
OOF = {k: np.load(O / a) for k, (a, _) in FAM.items()}; TE = {k: np.load(O / b) for k, (_, b) in FAM.items()}
assert all(v.shape == (3, len(y)) and np.isfinite(v).all() for v in OOF.values()), {k: v.shape for k, v in OOF.items()}
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
best = .5 * (OOF["ridge_cheap"] + OOF["lgbm_s1"])                  # current best = submission 003


def crossfit_nnls(keys):
    """Weights fit on the other 4 folds of the same repeat, applied to the held-out fold (no optimism)."""
    out = np.zeros((3, len(y)))
    for i, (a, b) in enumerate(sp):
        r = i // 5; A = np.column_stack([OOF[k][r, a] for k in keys]); w, _ = nnls(A, y[a]); w /= w.sum()
        out[r, b] = np.column_stack([OOF[k][r, b] for k in keys]) @ w
    return out


rows = {}
for k in OOF: rows[k] = OOF[k]
keys = list(OOF)
for sub in [keys[:2], keys, [k for k in keys if k != "lgbm_s1new"]] + ([["ridge_cheap", "cnn"], ["lgbm_s1", "cnn"]] if args.cnn else []):
    if len(sub) < 2: continue
    rows["eq(" + "+".join(sub) + ")"] = np.mean([OOF[k] for k in sub], 0)
    rows["nnls_cf(" + "+".join(sub) + ")"] = crossfit_nnls(sub)
res = {}
for k, o in rows.items():
    d = fr(o) - fr(best)
    res[k] = dict(mean=float(fr(o).mean()), std=float(fr(o).std()), d_vs_003=float(d.mean()),
                  p=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() > 0 else 1.0, wins=int((d < 0).sum()))
    print(f"  {k:55s} RMSE {res[k]['mean']:.3f} +/- {res[k]['std']:.3f} | vs 003 d {res[k]['d_vs_003']:+.3f} p {res[k]['p']:.3g} wins {res[k]['wins']}/15")
A = np.column_stack([OOF[k].ravel() for k in keys]); w, _ = nnls(A, np.tile(y, 3)); w /= w.sum()
print("NNLS weights on pooled OOF (all families):", dict(zip(keys, np.round(w, 3))))
print("OOF correlation:\n", pd.DataFrame({k: OOF[k].ravel() for k in keys}).corr().round(3).to_string())
if args.out:                                                         # args.out = "name.csv|fam1+fam2" (equal weights)
    name, fams = args.out.split("|"); fams = fams.split("+")
    out = write_sub(np.mean([TE[k] for k in fams], 0), M.ID.values[ite], name, y)
    print(f"wrote {name}: mean {out.hardness.mean():.3f} std {out.hardness.std():.3f}")
json.dump(res, open(REP / "blend.json", "w"), indent=1)
