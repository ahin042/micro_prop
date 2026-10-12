"""Phase 17: equal-weight blend of CNN(s), the 004 feature Ridge, and Ridge on frozen DINOv2 embeddings; paired fold tests
vs the 004 recipe and 003; optional submission. No blend weights are fitted (equal weights), so the only selection is
the pre-declared DINOv2 backbone choice, reported in reports/p17_probe_*.json.
--ln p13/latnet_cf_pred.parquet = honest variant: the latent-net features of the clean train images are cross-fitted (out-of-fold), so
they are as noisy as the test features (the final latent net memorises its own training images; see experiments.md, Phase 16).
Selection rule vs 004: d <= 0 and (p < .1 or >= 8/15 fold wins). Expected LB = CV - 1.16 (reports/lb_tracking.json).
Usage: python src/p17_build.py --cnn r18_final_a [--cnn r18up2_final:full] --dino dinos:252 [--ln p13/latnet_cf_pred.parquet] [--no_feat] [--name 00x_....csv]
"""
import argparse, json

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, SUB, cv, load_meta, ridge, sha256, splits, write_sub

ap = argparse.ArgumentParser(); ap.add_argument("--cnn", action="append", default=[]); ap.add_argument("--dino", action="append", default=[])
ap.add_argument("--no_feat", action="store_true"); ap.add_argument("--name", default=""); ap.add_argument("--ln", default="p13/latnet_pred.parquet")
args = ap.parse_args(); LN0 = "p13/latnet_pred.parquet"
M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y))
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
comp = {}                                                                   # name -> (oof (3, n), test (n_test,))
for t in args.cnn:                                                          # "tag" = fold-model test mean; "tag:full" = full-data seeds (cnn2.py --full)
    tg, mode = (t.split(":") + ["folds"])[:2]
    comp[f"cnn[{t}]"] = (np.load(C / "oof" / f"cnn_{tg}_oof.npy"), np.load(C / "oof" / (f"cnn_{tg}_full_test.npy" if mode == "full" else f"cnn_{tg}_test.npy")))
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)


def feat_X(ln):                                                             # the 004 feature matrix, exactly as src/p13_build_004.py
    F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd(ln)
    cols = sorted(set([c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")] + CHEAP + [c for c in G if c.startswith("ph_")] + [c for c in P if c.startswith("px_")] +
                      [c for c in O if c.endswith("_coh")] + [c for c in Dd if c.startswith(("d_", "p_"))] + [c for c in LN if c.startswith("ln_")]))
    A = pd.concat([F, M[CHEAP].reset_index(drop=True), G, P, O, Dd, LN], axis=1); A = A.loc[:, ~A.columns.duplicated()]
    return A[cols].fillna(A[cols].iloc[itr].median()).values.astype(float)


if not args.no_feat:
    X = feat_X(args.ln); comp["ridge004" + ("" if args.ln == LN0 else "[cf]")] = (cv(ridge, X[itr], y, sp)["oof"], ridge(X[itr], y, X[ite]))
for d in args.dino:                                                         # frozen embeddings cached by src/p17_dino_probe.py
    arch, size = d.split(":"); E = np.load(C / "p12" / f"emb17_{arch}_{size}.npy")
    comp[f"dino[{d}]"] = (cv(ridge, E[itr], y, sp)["oof"], ridge(E[itr], y, E[ite]))
c004 = np.load(C / "oof" / "cnn_r18_final_a_oof.npy"); X0 = feat_X(LN0)
refs = {"004 as reported": fr(.5 * (c004 + cv(ridge, X0[itr], y, sp)["oof"]))}       # the uploaded 004 (CV 13.529, LB 12.3733)
if args.ln != LN0: refs["004 honest (same LN)"] = fr(.5 * (c004 + cv(ridge, X[itr], y, sp)["oof"]))
b003 = fr(np.load(C / "oof" / "003_recovered_oof.npy")); res = {}
for k, v in refs.items(): print(f"  reference {k:22s} {v.mean():.3f} ± {v.std():.3f}")
blend_o = np.mean([o for o, _ in comp.values()], 0); blend_t = np.mean([t for _, t in comp.values()], 0)
for nm, o in list((k, v[0]) for k, v in comp.items()) + [("BLEND eq(" + ", ".join(comp) + ")", blend_o)]:
    f = fr(o); r = res[nm] = dict(mean=float(f.mean()), std=float(f.std()), expected_lb=float(f.mean() - 1.16), d_vs_003=float((f - b003).mean()), vs={})
    line = f"  {nm:58s} {r['mean']:.3f} ± {r['std']:.3f} (exp. LB {r['expected_lb']:.2f})"
    for k, rf in refs.items():
        d = f - rf; p_ = float(stats.ttest_1samp(d, 0).pvalue) if np.abs(d).sum() else 1.0; w = int((d < 0).sum())
        ok = bool(d.mean() <= 0 and (p_ < .1 or w >= 8)); r["vs"][k] = dict(d=float(d.mean()), p=p_, wins=w, passes_rule=ok)
        line += f" | vs {k}: d {d.mean():+.3f} p {p_:.2g} {w}/15 {'PASS' if ok else 'fail'}"
    print(line)
tag = "_".join([*[t.replace(":", "") for t in args.cnn], *(["feat" + ("" if args.ln == LN0 else "cf")] if not args.no_feat else []), *[d.replace(":", "") for d in args.dino]])
out = dict(components=list(comp), ln=args.ln, cv=res)
if args.name:
    sub = write_sub(blend_t, ids[ite], args.name, y); f = SUB / args.name if (SUB / args.name).read_text(encoding="utf-8") == sub.to_csv(index=False) else SUB / "rerun" / args.name
    c004 = np.corrcoef(sub.hardness, pd.read_csv(SUB / "004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv").hardness)[0, 1]
    print(f"wrote {f.relative_to(SUB.parent)} sha256 {sha256(f)} mean {sub.hardness.mean():.3f} std {sub.hardness.std():.3f} corr with 004 {c004:.4f}")
    out.update(file=f.name, sha256=sha256(f))
json.dump(out, open(REP / f"p17_build_{tag}.json", "w"), indent=1)
