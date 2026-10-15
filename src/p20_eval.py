"""Phase 20: 15-fold evaluation of a new fine-tuned component (cnn4.py tag) against 007, per the user's blending rule (train only).

Two pre-declared variants: (a) "replace" = drop-in replacement of 007's fine-tuned DINOv2-S component -> mean(new, Ridge004, Ridge[DINOv2-S]);
(b) "add as 4th" = mean(FT-S, Ridge004, Ridge[DINOv2-S], new). Equal weights, nothing fitted on OOF. The better variant is picked
per outer fold on the training images of that fold (nested) -> selection-adjusted CV. Fold 0 was the screening fold, so folds 1-14
(and repeats 1-2) are also reported separately. Paired tests vs 007 and 005; gate = 007 - 0.25; expected LB = CV - 1.17 +- 0.3.
Usage: python src/p20_eval.py --tag x_dinob_ft      (the submission file itself is built with src/p17_build.py, as for 005-007)
"""
import argparse, json

import numpy as np
from scipy import stats

from common import C, REP, load_meta, splits

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); args = ap.parse_args()
M, itr, ite, y = load_meta(); sp = splits(len(y)); L = lambda f: np.load(C / "oof" / f)
c18, rf, ds, ft = (L(f) for f in ["cnn_r18_final_a_oof.npy", "p17_ridge004_oof.npy", "p17_dinos_252_oof.npy", "cnn_x_dinos_ft_t_oof.npy"])
nw = L(f"cnn_{args.tag}_oof.npy"); assert nw.shape == (3, len(y)) and np.isfinite(nw).all(), f"{args.tag}: OOF incomplete (need all 15 folds)"
r = lambda o, b: float(np.sqrt(np.mean((o[b] - y[b]) ** 2)))
fr = lambda o: np.array([r(o[i // 5], b) for i, (a, b) in enumerate(sp)])
ref = {"007": fr((ft + rf + ds) / 3), "005": fr((c18 + rf + ds) / 3)}; gate = ref["007"].mean() - .25
V = {"replace": (nw + rf + ds) / 3, "add as 4th": (ft + rf + ds + nw) / 4}
out = dict(tag=args.tag, single=float(fr(nw).mean()), single_sd=float(fr(nw).std()), ref_007=float(ref["007"].mean()), gate=float(gate))
print(f"{args.tag} single 15 folds {fr(nw).mean():.3f} ± {fr(nw).std():.3f} (FT-S {fr(ft).mean():.3f}) | 007 {ref['007'].mean():.3f} | 005 {ref['005'].mean():.3f} | gate {gate:.3f}")


def paired(f, label):
    res = {}
    for k, rf_ in ref.items():
        d = f - rf_; res[f"vs_{k}"] = dict(d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    s = " | ".join(f"vs {k} d {res[f'vs_{k}']['d']:+.3f} p {res[f'vs_{k}']['p']:.2g} {res[f'vs_{k}']['wins']}/15" for k in ref)
    print(f"  {label:22s} {f.mean():.3f} ± {f.std():.3f} | {s}"); return res


for nm, o in V.items():
    f = fr(o); out[nm] = dict(cv=float(f.mean()), sd=float(f.std()), **paired(f, nm))
    d7 = f - ref["007"]; out[nm]["folds_1_14_vs_007"] = float(d7[1:].mean()); out[nm]["by_repeat_vs_007"] = [float(d7[k * 5:(k + 1) * 5].mean()) for k in range(3)]
    print(f"      folds 1-14 vs 007 {d7[1:].mean():+.3f} | by repeat {' / '.join(f'{v:+.3f}' for v in out[nm]['by_repeat_vs_007'])}")
nest, ch = np.zeros(len(sp)), []
for i, (a, b) in enumerate(sp):
    k = min(V, key=lambda k: r(V[k][i // 5], a)); ch.append(k); nest[i] = r(V[k][i // 5], b)
out["selection_adjusted"] = dict(cv=float(nest.mean()), choices={k: ch.count(k) for k in V}, **paired(nest, "selection-adjusted"))
best = min(V, key=lambda k: out[k]["cv"]); out["better_variant"] = best
out["passes_gate"] = bool(out["selection_adjusted"]["cv"] <= gate)
out["expected_lb"] = [round(nest.mean() - 1.17 - .3, 2), round(nest.mean() - 1.17 + .3, 2)]
print(f"  choices {out['selection_adjusted']['choices']} | better variant: {best} | gate {'PASS' if out['passes_gate'] else 'fail'} | exp. LB {out['expected_lb'][0]}-{out['expected_lb'][1]}")
json.dump(out, open(REP / f"p20_eval_{args.tag}.json", "w"), indent=1)
