"""Phase 16: downstream CV with latent-net v2 (ln2_*), paired tests vs the 004 recipe, optional submission.

Reference = eq(CNN, Ridge[S1+cheap+PH+PX+COH+DESC+LN]) (the 004 recipe). Same RepeatedKFold(5x3, seed 42); y only in folds.
Usage: python src/p16_cv.py --cnn r18_final_a [--ln p13/latnet_cf_pred.parquet] [--ln2 p16/latnet2cf_pred.parquet] [--write 005_....csv]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, cv, load_meta, ridge, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", default="r18_final_a"); ap.add_argument("--write", default=""); ap.add_argument("--ln2", default="p16/latnet2_pred.parquet"); ap.add_argument("--ln", default="p13/latnet_pred.parquet")
args = ap.parse_args()
M, itr, ite, y = load_meta(); ids_all = M.ID.values; n = len(y)
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids_all].reset_index(drop=True)
F, G, P, O, Dd, LN, L2 = (rd(p) for p in ["features.parquet", "p12/phase.parquet", "p12/pixphase.parquet", "p12/orient.parquet", "p13/desc.parquet",
                                          args.ln, args.ln2])
fam = {"BASE": sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")) + CHEAP, "PH": [c for c in G if c.startswith("ph_")], "PX": [c for c in P if c.startswith("px_")],
       "COH": [c for c in O if c.endswith("_coh")], "DESC": [c for c in Dd if c.startswith(("d_", "p_"))], "LN": [c for c in LN if c.startswith("ln_")], "LN2": [c for c in L2 if c.startswith("ln2_") and c != "ln2_oof"]}
A = pd.concat([F, M[CHEAP].reset_index(drop=True), G, P, O, Dd, LN, L2], axis=1); A = A.loc[:, ~A.columns.duplicated()]
q = stats.rankdata(F.q_noise.values[itr]) + stats.rankdata(F.m_sigma_px.values[itr]); clean = q <= np.median(q); sp = splits(n)
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
r2 = lambda o, m: float(1 - np.mean((o.mean(0)[m] - y[m]) ** 2) / np.var(y[m]))
cnn_o, cnn_t = np.load(C / "oof" / f"cnn_{args.cnn}_oof.npy"), np.load(C / "oof" / f"cnn_{args.cnn}_test.npy")
b003 = fr(np.load(C / "oof" / "003_recovered_oof.npy")); ref = None; res = {}; keep = {}
CONF = [("004 recipe: A+DESC+LN", ["BASE", "PH", "PX", "COH", "DESC", "LN"]), ("LN2 only", ["LN2"]), ("LN+LN2", ["LN", "LN2"]),
        ("A+DESC+LN2", ["BASE", "PH", "PX", "COH", "DESC", "LN2"]), ("A+DESC+LN+LN2", ["BASE", "PH", "PX", "COH", "DESC", "LN", "LN2"])]
for nm, fs in CONF:
    cols = sorted(set(sum((fam[f] for f in fs), []))); med = A[cols].iloc[itr].median(); X = A[cols].fillna(med).values.astype(float)
    o = cv(ridge, X[itr], y, sp); bl = .5 * (cnn_o + o["oof"]); f_ = fr(bl); ref = f_ if ref is None else ref; d = f_ - ref; d3 = f_ - b003
    res[nm] = dict(ridge_cv=o["mean"], r2_clean=r2(o["oof"], clean), r2_dirty=r2(o["oof"], ~clean), blend_cv=float(f_.mean()), blend_sd=float(f_.std()), d_vs_004=float(d.mean()),
                   p_vs_004=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins=int((d < 0).sum()), d_vs_003=float(d3.mean()))
    keep[nm] = (X, o["oof"]); r = res[nm]
    print(f"  {nm:24s} ridge {r['ridge_cv']:.3f} (R2 clean {r['r2_clean']:.3f} dirty {r['r2_dirty']:.3f}) | eq(CNN,·) {r['blend_cv']:.3f}±{r['blend_sd']:.3f} "
          f"vs 004 recipe d {r['d_vs_004']:+.3f} p {r['p_vs_004']:.2g} wins {r['wins']}/15 | vs 003 d {r['d_vs_003']:+.3f}", flush=True)
json.dump(res, open(REP / f"p16_cv_{args.cnn}_{Path(args.ln).stem}_{Path(args.ln2).stem}.json", "w"), indent=1)
if args.write:                                                              # best config by the selection rule: p < .1 or >= 8/15 wins with d <= 0
    ok = [k for k, v in res.items() if k != CONF[0][0] and v["d_vs_004"] <= 0 and (v["p_vs_004"] < .1 or v["wins"] >= 8)]
    pick = min(ok, key=lambda k: res[k]["blend_cv"]) if ok else CONF[0][0]; X, _ = keep[pick]
    pt = .5 * (cnn_t + ridge(X[itr], y, X[ite])); out = write_sub(pt, ids_all[ite], args.write, y)
    print(f"selected '{pick}' -> {args.write}: mean {out.hardness.mean():.3f} std {out.hardness.std():.3f}")
