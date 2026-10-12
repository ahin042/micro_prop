"""Phase 17: equal-weight blend of CNN(s), the 004 feature Ridge, and Ridge on frozen DINOv2 embeddings; paired fold tests
vs the 004 recipe and 003; optional submission. No blend weights are fitted (equal weights), so the only selection is
the pre-declared DINOv2 backbone choice, reported in reports/p17_probe_*.json.
Usage: python src/p17_build.py --cnn r18_final_a [--cnn r18up2_final:full] --dino dinos:252 [--dino dinob:252 ...] [--no_feat] [--name 005_....csv]
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, SUB, cv, load_meta, ridge, sha256, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", action="append", default=[]); ap.add_argument("--dino", action="append", default=[])
ap.add_argument("--no_feat", action="store_true"); ap.add_argument("--name", default=""); args = ap.parse_args()
M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y))
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
comp = {}                                                                   # name -> (oof (3, n), test (n_test,))
for t in args.cnn:                                                          # "tag" = fold-model test mean; "tag:full" = full-data seeds (cnn2.py --full)
    tg, mode = (t.split(":") + ["folds"])[:2]
    comp[f"cnn[{t}]"] = (np.load(C / "oof" / f"cnn_{tg}_oof.npy"), np.load(C / "oof" / (f"cnn_{tg}_full_test.npy" if mode == "full" else f"cnn_{tg}_test.npy")))
if not args.no_feat:                                                        # the 004 feature Ridge, exactly as src/p13_build_004.py
    rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
    F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd("p13/latnet_pred.parquet")
    cols = sorted(set([c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")] + CHEAP + [c for c in G if c.startswith("ph_")] + [c for c in P if c.startswith("px_")] +
                      [c for c in O if c.endswith("_coh")] + [c for c in Dd if c.startswith(("d_", "p_"))] + [c for c in LN if c.startswith("ln_")]))
    A = pd.concat([F, M[CHEAP].reset_index(drop=True), G, P, O, Dd, LN], axis=1); A = A.loc[:, ~A.columns.duplicated()]
    X = A[cols].fillna(A[cols].iloc[itr].median()).values.astype(float)
    comp["ridge004"] = (cv(ridge, X[itr], y, sp)["oof"], ridge(X[itr], y, X[ite]))
for d in args.dino:                                                         # frozen embeddings cached by src/p17_dino_probe.py
    arch, size = d.split(":"); E = np.load(C / "p12" / f"emb17_{arch}_{size}.npy")
    comp[f"dino[{d}]"] = (cv(ridge, E[itr], y, sp)["oof"], ridge(E[itr], y, E[ite]))
ref = fr(.5 * (np.load(C / "oof" / "cnn_r18_final_a_oof.npy") + cv(ridge, X[itr], y, sp)["oof"])) if not args.no_feat else None
b003 = fr(np.load(C / "oof" / "003_recovered_oof.npy")); res = {}
blend_o = np.mean([o for o, _ in comp.values()], 0); blend_t = np.mean([t for _, t in comp.values()], 0)
for nm, o in list((k, v[0]) for k, v in comp.items()) + [("BLEND eq(" + ", ".join(comp) + ")", blend_o)]:
    f = fr(o); r = res[nm] = dict(mean=float(f.mean()), std=float(f.std()), d_vs_003=float((f - b003).mean()))
    if ref is not None:
        d = f - ref; r.update(d_vs_004=float(d.mean()), p_vs_004=float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0, wins_vs_004=int((d < 0).sum()))
    print(f"  {nm:60s} {r['mean']:.3f} ± {r['std']:.3f}" + (f" | vs 004 recipe d {r['d_vs_004']:+.3f} p {r['p_vs_004']:.2g} wins {r['wins_vs_004']}/15" if ref is not None else ""))
tag = "_".join([*[t.replace(":", "") for t in args.cnn], *(["feat"] if not args.no_feat else []), *[d.replace(":", "") for d in args.dino]])
out = dict(components=list(comp), cv=res)
if args.name:
    sub = write_sub(blend_t, ids[ite], args.name, y); f = SUB / args.name if (SUB / args.name).read_text(encoding="utf-8") == sub.to_csv(index=False) else SUB / "rerun" / args.name
    c004 = np.corrcoef(sub.hardness, pd.read_csv(SUB / "004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv").hardness)[0, 1]
    print(f"wrote {f.relative_to(SUB.parent)} sha256 {sha256(f)} mean {sub.hardness.mean():.3f} std {sub.hardness.std():.3f} corr with 004 {c004:.4f}")
    out.update(file=f.name, sha256=sha256(f))
json.dump(out, open(REP / f"p17_build_{tag}.json", "w"), indent=1)
