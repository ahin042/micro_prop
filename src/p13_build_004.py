"""Phase 13 / Part 1: build submission 004 = eq(CNN, Ridge_p12).

Ridge_p12 = Ridge on S1 + cheap + ph_* + px_* + coherence (pre-registered in Phase 12; its honest nested CV is reported
alongside). CNN = winning config r18 / D4+blur+noise / global norm, 15 folds, deterministic re-run (--cnn tag).
Usage: python src/p13_build_004.py --cnn r18_final_a [--name 004_....csv]
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, SUB, cv, load_meta, ridge, sha256, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", required=True); ap.add_argument("--name", default="004_cnn_r18_d4bn_plus_ridge_p12.csv")
args = ap.parse_args()
M, itr, ite, y = load_meta(); ids_all = M.ID.values
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids_all].reset_index(drop=True)
F, G, P, O = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet")
cols = sorted(set([c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")] + CHEAP + [c for c in G if c.startswith("ph_")] +
                  [c for c in P if c.startswith("px_")] + [c for c in O if c.endswith("_coh")]))
A = pd.concat([F, M[CHEAP].reset_index(drop=True), G, P, O], axis=1); A = A.loc[:, ~A.columns.duplicated()]
med = A[cols].iloc[itr].median(); X = A[cols].fillna(med).values.astype(float)            # impute with TRAIN medians only
sp = splits(len(y)); o_r = cv(ridge, X[itr], y, sp); p_r = ridge(X[itr], y, X[ite])
o_c, p_c = np.load(C / "oof" / f"cnn_{args.cnn}_oof.npy"), np.load(C / "oof" / f"cnn_{args.cnn}_test.npy")
assert o_c.shape == (3, len(y)) and np.isfinite(o_c).all() and p_c.shape == (len(ite),)
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
base = fr(np.load(C / "oof" / "003_recovered_oof.npy")); res = {}
nested = np.load(C / "p13" / "oof_ridge_p12_nested.npy")
for nm, o in [("ridge_p12", o_r["oof"]), ("cnn", o_c), ("eq(cnn, ridge_p12)", .5 * (o_c + o_r["oof"])), ("eq(cnn, ridge_p12 nested-honest)", .5 * (o_c + nested))]:
    d = fr(o) - base; res[nm] = dict(mean=float(fr(o).mean()), std=float(fr(o).std()), d_vs_003=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    print(f"  {nm:34s} {res[nm]['mean']:.3f} +/- {res[nm]['std']:.3f} | vs 003(LB 13.2852) d {d.mean():+.3f} p {res[nm]['p']:.2g} wins {res[nm]['wins']}/15")
out = write_sub(.5 * (p_c + p_r), ids_all[ite], args.name, y)
f = SUB / args.name if (SUB / args.name).read_text(encoding="utf-8") == out.to_csv(index=False) else SUB / "rerun" / args.name
print(f"wrote {f.relative_to(SUB.parent)}  sha256 {sha256(f)}  mean {out.hardness.mean():.3f} std {out.hardness.std():.3f} | "
      f"corr with 003_recovered {np.corrcoef(out.hardness, pd.read_csv(SUB / '003_recovered.csv').hardness)[0, 1]:.4f}")
json.dump(dict(cnn=args.cnn, cv=res, file=str(f.name), sha256=sha256(f)), open(REP / f"p13_build_004_{args.cnn}.json", "w"), indent=1)
