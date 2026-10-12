"""Phase 13 / Part 3 (ii): downstream CV with the degradation-robust predicted latents (ln_*) added.

Same RepeatedKFold(5x3, seed 42); y is only used inside folds (the latent net never saw y). Paired tests vs the current
best blend eq(CNN, Ridge_p12) and vs submission 003.
Usage: python src/p13_latnet_cv.py --cnn r18_final_a
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, cv, lgbm, load_meta, ridge, splits

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", default="r18_d4blurnoise_global"); args = ap.parse_args()
M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd("p13/latnet_pred.parquet")
fam = {"BASE": sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")) + CHEAP, "PH": [c for c in G if c.startswith("ph_")], "PX": [c for c in P if c.startswith("px_")],
       "COH": [c for c in O if c.endswith("_coh")], "DESC": [c for c in Dd if c.startswith(("d_", "p_"))], "LN": [c for c in LN if c.startswith("ln_")]}
A = pd.concat([F, M[CHEAP].iloc[itr].reset_index(drop=True), G, P, O, Dd, LN], axis=1); A = A.loc[:, ~A.columns.duplicated()]
num = A.select_dtypes("number").columns; A[num] = A[num].fillna(A[num].median())
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q); sp = splits(len(y))
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
cnn = np.load(C / "oof" / f"cnn_{args.cnn}_oof.npy"); b003 = fr(np.load(C / "oof" / "003_recovered_oof.npy"))
ref = None; res = {}
for nm, fs, mdl in [("ridge A (Ridge_p12)", ["BASE", "PH", "PX", "COH"], ridge), ("ridge A+DESC", ["BASE", "PH", "PX", "COH", "DESC"], ridge),
                    ("ridge LN only", ["LN"], ridge), ("ridge A+LN", ["BASE", "PH", "PX", "COH", "LN"], ridge), ("ridge A+DESC+LN", ["BASE", "PH", "PX", "COH", "DESC", "LN"], ridge),
                    ("lgbm A+DESC+LN", ["BASE", "PH", "PX", "COH", "DESC", "LN"], lgbm)]:
    cols = sorted(set(sum((fam[f] for f in fs), []))); o = cv(mdl, A[cols].values.astype(float), y, sp); oo = o["oof"].mean(0)
    bl = .5 * (cnn + o["oof"]); ref = fr(bl) if ref is None else ref; d = fr(bl) - ref; d3 = fr(bl) - b003
    res[nm] = dict(cv=o["mean"], r2_clean=float(1 - np.mean((oo[clean] - y[clean]) ** 2) / np.var(y[clean])), r2_dirty=float(1 - np.mean((oo[~clean] - y[~clean]) ** 2) / np.var(y[~clean])),
                   blend=float(fr(bl).mean()), d_vs_ref=float(d.mean()), p_vs_ref=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins_vs_ref=int((d < 0).sum()),
                   d_vs_003=float(d3.mean()), p_vs_003=float(stats.ttest_1samp(d3, 0).pvalue))
    np.save(C / "p13" / f"oof_{nm.replace(' ', '_').replace('(', '').replace(')', '')}.npy", o["oof"])
    r = res[nm]; print(f"  {nm:22s} CV {r['cv']:.3f} | R2 clean {r['r2_clean']:.3f} dirty {r['r2_dirty']:.3f} | eq(CNN,·) {r['blend']:.3f} "
                       f"vs eq(CNN,Ridge_p12) d {r['d_vs_ref']:+.3f} p {r['p_vs_ref']:.2g} wins {r['wins_vs_ref']}/15 | vs 003 d {r['d_vs_003']:+.3f}", flush=True)
json.dump(res, open(REP / f"p13_latnet_cv_{args.cnn}.json", "w"), indent=1)
