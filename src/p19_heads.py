"""Phase 19 / step 2b: heads on frozen multi-layer DINOv2 features (train only, CPU).

Pre-declared grid (all reported): layers {last4 = 8-11, mid4 = 4-7, all12} x stats {base = CLS+mean+std, gem = base+GeM3}
x heads {Ridge, RBF kernel ridge, RBF SVR, PLS} on the D4-mean features, plus Ridge on [DINOv2 features + the 004 hand features].
Hyperparameters: inner 5-fold CV inside each outer training fold (Ridge: built-in LOO). Outer protocol: RepeatedKFold(5x3, seed 42).
Honest gate number: nested selection of the configuration (best inner score) inside every outer fold.
Blends: equal weight with 005's components, (i) replacing 005's DINOv2 Ridge, (ii) added as a 4th component; paired vs 005.
Usage: python src/p19_heads.py --arch dinos
"""
import argparse, json, time

import numpy as np
import pandas as pd
from scipy import stats

from common import C, CHEAP, REP, load_meta, splits

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="dinos"); ap.add_argument("--view", default="d4mean"); args = ap.parse_args()
M, itr, ite, y = load_meta(); ids = M.ID.values; sp = splits(len(y)); nrep = len(sp) // 5
V = np.load(C / "p12" / f"emb19_{args.arch}_252_{args.view}.npy")[itr]          # (500, blocks, 4, dim)
nb = V.shape[1]; LAYERS = {"last4": list(range(nb - 4, nb)), "mid4": list(range(nb // 3, nb // 3 + 4)), "all12": list(range(nb))}
STATS = {"base": [0, 1, 2], "gem": [0, 1, 2, 3]}
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids[itr]].reset_index(drop=True)
F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd("p13/latnet_pred.parquet")
hcols = sorted(set([c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")] + CHEAP + [c for c in G if c.startswith("ph_")] + [c for c in P if c.startswith("px_")] +
                   [c for c in O if c.endswith("_coh")] + [c for c in Dd if c.startswith(("d_", "p_"))] + [c for c in LN if c.startswith("ln_")]))
H = pd.concat([F, M[CHEAP].iloc[itr].reset_index(drop=True), G, P, O, Dd, LN], axis=1); H = H.loc[:, ~H.columns.duplicated()][hcols].values.astype(float)


from p19_lib import h_pls, h_ridge, kernel_head, ridge_inner, std_fit


HEADS = {"ridge": h_ridge, "krr": kernel_head("krr"), "svr": kernel_head("svr"), "pls": h_pls}
cfgs = {f"{l}/{s}/{h}": (l, s, h, False) for l in LAYERS for s in STATS for h in HEADS} | {f"{l}/{s}/ridge+hand": (l, s, "ridge", True) for l in LAYERS for s in STATS}
X_of = lambda l, s: V[:, LAYERS[l]][:, :, STATS[s]].reshape(len(V), -1)
oof = {k: np.zeros((nrep, len(y))) for k in cfgs}; inner_score = {k: np.zeros(len(sp)) for k in cfgs}; t0 = time.time()
for fi, (a, b) in enumerate(sp):
    for k, (l, s, h, hand) in cfgs.items():
        X = np.hstack([X_of(l, s), H]) if hand else X_of(l, s)
        Xa, Xb = std_fit(X[a], X[b])
        if hand:                                                            # equal total weight for the two blocks after standardization
            nd = X_of(l, s).shape[1]; Xa[:, nd:] *= np.sqrt(nd / H.shape[1]); Xb[:, nd:] *= np.sqrt(nd / H.shape[1])
        p, sc = HEADS[h](Xa, y[a], Xb); oof[k][fi // 5, b] = p
        inner_score[k][fi] = sc if sc is not None else ridge_inner(Xa, y[a])
    print(f"outer fold {fi + 1}/15 done ({time.time() - t0:.0f}s)", flush=True)

fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
L = lambda n: np.load(C / "oof" / n); c18, rf, ds = L("cnn_r18_final_a_oof.npy"), L("p17_ridge004_oof.npy"), L("p17_dinos_252_oof.npy")
b005 = fr((c18 + rf + ds) / 3); rows = []
for k in cfgs:
    f1 = fr(oof[k]); fr_rep, fr_add = fr((c18 + rf + oof[k]) / 3), fr((c18 + rf + ds + oof[k]) / 4)
    rows.append(dict(config=k, single=f1.mean(), blend_replace=fr_rep.mean(), d_replace=(fr_rep - b005).mean(), wins_replace=int((fr_rep < b005).sum()),
                     blend_add=fr_add.mean(), d_add=(fr_add - b005).mean(), wins_add=int((fr_add < b005).sum())))
T = pd.DataFrame(rows).sort_values("single"); pd.set_option("display.width", 220)
print(f"\nreference: 005 = {b005.mean():.3f}; 005's DINOv2 Ridge (last4/base, single view) = {fr(ds).mean():.3f}; gate = 005 - 0.25 = {b005.mean() - .25:.3f}")
print(T.round(3).to_string(index=False))
# honest nested selection: in each outer fold use the config with the best inner score (single model, and blend-replace)
sel = np.array([min(cfgs, key=lambda k: inner_score[k][fi]) for fi in range(len(sp))])
nested = np.zeros((nrep, len(y)))
for fi, (a, b) in enumerate(sp): nested[fi // 5, b] = oof[sel[fi]][fi // 5, b]
fn = fr(nested); fb = fr((c18 + rf + nested) / 3); d = fb - b005
print(f"\nnested selection (configs chosen: {pd.Series(sel).value_counts().to_dict()})")
print(f"  single {fn.mean():.3f} | blend (replace 005's DINOv2) {fb.mean():.3f}  vs 005 d {d.mean():+.3f} p {stats.ttest_1samp(d, 0).pvalue:.2g} wins {(d < 0).sum()}/15 | gate (≤ -0.25): {'PASS' if d.mean() <= -.25 else 'fail'}")
np.save(C / "oof" / f"p19_heads_{args.arch}_nested_oof.npy", nested)
for k in cfgs: np.save(C / "oof" / f"p19_{args.arch}_{k.replace('/', '_').replace('+', 'p')}_oof.npy", oof[k])
json.dump(dict(table=T.to_dict("records"), nested=dict(single=float(fn.mean()), blend=float(fb.mean()), d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue),
               wins=int((d < 0).sum()), chosen=pd.Series(sel).value_counts().to_dict())), open(REP / f"p19_heads_{args.arch}.json", "w"), indent=1, default=float)
