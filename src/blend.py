"""Part D / Step 5: OOF blending across model families.

Methods: equal weights, NNLS and stacked Ridge. NNLS/Ridge are scored honestly by cross-fitting (weights fit on the other
4 folds of the same repeat, applied to the held-out fold). Every blend is compared fold-by-fold with submission 003.
Selection rule: improvement vs 003 with p < 0.1, or delta <= 0 with > 50% of the 15 folds improved; lowest RMSE wins.
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

from common import C, CHEAP, REP, design, lgbm, load_meta, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", default="", help="comma list of CNN tags")
ap.add_argument("--out", default="", help="'name.csv|fam1+fam2' equal-weight submission (e.g. 003)")
ap.add_argument("--final", default="", help="write the selected blend to this file name (e.g. 004_....csv)")
args = ap.parse_args()
M, itr, ite, y = load_meta(); sp = splits(len(y)); O = C / "oof"
if not (O / "b_lgbm_s1new_test.npy").exists():                    # test preds for the Part B family
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index()
    cols = [c for c in F.columns if c.startswith(("q_", "r_", "n_", "b_", "m_", "d_", "x_"))]
    X = design(pd.concat([F[cols], M[CHEAP]], axis=1), cols + CHEAP); np.save(O / "b_lgbm_s1new_test.npy", lgbm(X[itr], y, X[ite]))
FAM = {"ridge_cheap": ("001_cheap_gbdt_oof.npy", "001_cheap_gbdt_test.npy"),
       "lgbm_s1": ("s1_lgbm_stage1+cheap_oof.npy", "002_s1_lgbm_test.npy"),
       "lgbm_s1new": ("b_lgbm_s1+cheap+new_oof.npy", "b_lgbm_s1new_test.npy")}
CNNS = [c for c in args.cnn.split(",") if c]
for c in CNNS: FAM[f"cnn_{c}"] = (f"cnn_{c}_oof.npy", f"cnn_{c}_test.npy")
OOF = {k: np.load(O / a) for k, (a, _) in FAM.items()}; TE = {k: np.load(O / b) for k, (_, b) in FAM.items()}
assert all(v.shape == (3, len(y)) and np.isfinite(v).all() for v in OOF.values()), {k: v.shape for k, v in OOF.items()}
assert all(v.shape == (len(ite),) and np.isfinite(v).all() for v in TE.values())
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
best = .5 * (OOF["ridge_cheap"] + OOF["lgbm_s1"])                  # current best = submission 003


def fit_w(A, t, how):
    if how == "nnls": w, _ = nnls(A, t); return ("w", w / w.sum())
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(A, t); return ("ridge", m)


def apply_w(W, A):
    return A @ W[1] if W[0] == "w" else W[1].predict(A)


def crossfit(keys, how):
    out = np.zeros((3, len(y)))
    for i, (a, b) in enumerate(sp):
        r = i // 5; W = fit_w(np.column_stack([OOF[k][r, a] for k in keys]), y[a], how)
        out[r, b] = apply_w(W, np.column_stack([OOF[k][r, b] for k in keys]))
    return out


base = ["ridge_cheap", "lgbm_s1"]; allb = base + ["lgbm_s1new"]; cn = [f"cnn_{c}" for c in CNNS]
sets = [allb] + ([base + [c] for c in cn] + [allb + [c] for c in cn] + ([allb + cn, base + cn] if len(cn) > 1 else [])) if cn else [allb]
rows, recipe = {k: OOF[k] for k in OOF}, {}
for s in sets:
    nm = "+".join(s); rows[f"eq({nm})"] = np.mean([OOF[k] for k in s], 0); recipe[f"eq({nm})"] = (s, "eq")
    for how in ("nnls", "stack_ridge"):
        rows[f"{how}_cf({nm})"] = crossfit(s, how); recipe[f"{how}_cf({nm})"] = (s, how)
res = {}
for k, o in rows.items():
    d = fr(o) - fr(best)
    res[k] = dict(mean=float(fr(o).mean()), std=float(fr(o).std()), d_vs_003=float(d.mean()),
                  p=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() > 0 else 1.0, wins=int((d < 0).sum()))
    print(f"  {k:72s} RMSE {res[k]['mean']:.3f} +/- {res[k]['std']:.3f} | vs 003 d {res[k]['d_vs_003']:+.3f} p {res[k]['p']:.3g} wins {res[k]['wins']}/15")
print("OOF correlation:\n", pd.DataFrame({k: OOF[k].ravel() for k in OOF}).corr().round(3).to_string())
ok = {k: v for k, v in res.items() if k in recipe and ((v["d_vs_003"] < 0 and v["p"] < .1) or (v["d_vs_003"] <= 0 and v["wins"] > 7))}
sel = min(ok, key=lambda k: res[k]["mean"]) if ok else None
print("eligible under the selection rule:", len(ok), "| selected:", sel)
if sel:
    keys, how = recipe[sel]; A = np.column_stack([OOF[k].ravel() for k in keys])
    W = ("w", np.full(len(keys), 1 / len(keys))) if how == "eq" else fit_w(A, np.tile(y, 3), "nnls" if how == "nnls" else "ridge")
    desc = dict(zip(keys, np.round(W[1], 4).tolist())) if W[0] == "w" else dict(zip(keys, np.round(W[1].coef_, 4).tolist()), intercept=round(float(W[1].intercept_), 3))
    print("final weights (fit on pooled OOF):", desc); res["_selected"] = dict(name=sel, weights=desc)
    if args.final:
        pt = apply_w(W, np.column_stack([TE[k] for k in keys])); out = write_sub(pt, M.ID.values[ite], args.final, y)
        print(f"wrote {args.final}: mean {out.hardness.mean():.3f} std {out.hardness.std():.3f} | OOF blend mean {rows[sel].mean():.3f} std {rows[sel].std():.3f}")
if args.out:
    name, fams = args.out.split("|"); out = write_sub(np.mean([TE[k] for k in fams.split("+")], 0), M.ID.values[ite], name, y)
    print(f"wrote {name}: mean {out.hardness.mean():.3f} std {out.hardness.std():.3f}")
json.dump(res, open(REP / "blend.json", "w"), indent=1)
