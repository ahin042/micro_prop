"""Phase 21: seed averaging of the fine-tuned DINOv2 components (train only; nothing fitted on OOF).

S_42 / B_42 = fine-tuned DINOv2-S / -B of Phases 19/20 (seed 42); S_0 / B_0 = the same recipes with --seed 0 (cnn4.py).
Five pre-declared equal-weight blends (user, Phase 21), each with Ridge004 and Ridge[frozen DINOv2-S]:
  1 avg(S_42, S_0) + R + F      2 avg(S_42, S_0) + avg(B_42, B_0) + R + F      3 S_42 + S_0 + B_42 + B_0 + R + F (6 parts)
  4 avg(B_42, B_0) + R + F      5 avg(S_42, S_0) + B_42 + R + F
Selection-adjusted CV: per outer fold the blend with the lowest RMSE on that fold's training images (same repeat) is scored on the
fold. Paired tests vs 007 and 005; gate = selection-adjusted CV <= 007 - 0.25 (12.540); expected LB = CV - 1.17 +- 0.3.
--write k: save blend k's component arrays as cnn_p21_avg* so src/p17_build.py can build the file.
Usage: python src/p21_eval.py [--s0 x_dinos_ft_seed0] [--b0 x_dinob_ft_seed0] [--write 2]
"""
import argparse, json

import numpy as np
from scipy import stats

from common import C, REP, load_meta, splits

ap = argparse.ArgumentParser(); ap.add_argument("--s0", default="x_dinos_ft_seed0"); ap.add_argument("--b0", default="x_dinob_ft_seed0")
ap.add_argument("--write", type=int, default=0); args = ap.parse_args()
M, itr, ite, y = load_meta(); sp = splits(len(y)); L = lambda f: np.load(C / "oof" / f)
oo = {k: L(f"cnn_{t}_oof.npy") for k, t in dict(S42="x_dinos_ft_t", S0=args.s0, B42="x_dinob_ft", B0=args.b0).items()}
te = {k: L(f"cnn_{t}_test.npy") for k, t in dict(S42="x_dinos_ft_t", S0=args.s0, B42="x_dinob_ft", B0=args.b0).items()}
for k, v in oo.items(): assert v.shape == (3, len(y)) and np.isfinite(v).all(), f"{k}: OOF incomplete"
c18, R, F = L("cnn_r18_final_a_oof.npy"), L("p17_ridge004_oof.npy"), L("p17_dinos_252_oof.npy")
avg = lambda d, *ks: np.mean([d[k] for k in ks], 0)
parts = {1: [("S", ("S42", "S0"))], 2: [("S", ("S42", "S0")), ("B", ("B42", "B0"))], 3: [(k, (k,)) for k in ("S42", "S0", "B42", "B0")],
         4: [("B", ("B42", "B0"))], 5: [("S", ("S42", "S0")), ("B42", ("B42",))]}
V = {k: np.mean([avg(oo, *ks) for _, ks in p] + [R, F], 0) for k, p in parts.items()}
r = lambda o, b: float(np.sqrt(np.mean((o[b] - y[b]) ** 2)))
fr = lambda o: np.array([r(o[i // 5], b) for i, (a, b) in enumerate(sp)])
ref = {"007": fr((oo["S42"] + R + F) / 3), "005": fr((c18 + R + F) / 3)}; gate = ref["007"].mean() - .25
print("single models: " + " | ".join(f"{k} {fr(v).mean():.3f}" for k, v in oo.items()) + f" | avg S {fr(avg(oo, 'S42', 'S0')).mean():.3f} | avg B {fr(avg(oo, 'B42', 'B0')).mean():.3f}")
print(f"007 {ref['007'].mean():.3f} | 005 {ref['005'].mean():.3f} | gate {gate:.3f}")


def paired(f, label):
    res = {}
    for k, rf_ in ref.items():
        d = f - rf_; res[f"vs_{k}"] = dict(d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    print(f"  {label:20s} {f.mean():.3f} ± {f.std():.3f} | " + " | ".join(f"vs {k} d {res[f'vs_{k}']['d']:+.3f} p {res[f'vs_{k}']['p']:.2g} {res[f'vs_{k}']['wins']}/15" for k in ref))
    return res


out = dict(single={k: float(fr(v).mean()) for k, v in oo.items()}, ref_007=float(ref["007"].mean()), gate=float(gate))
for k, o in V.items():
    f = fr(o); out[f"blend{k}"] = dict(cv=float(f.mean()), sd=float(f.std()), **paired(f, f"blend {k}"))
nest, ch = np.zeros(len(sp)), []
for i, (a, b) in enumerate(sp):
    k = min(V, key=lambda k: r(V[k][i // 5], a)); ch.append(k); nest[i] = r(V[k][i // 5], b)
out["selection_adjusted"] = dict(cv=float(nest.mean()), choices={str(k): ch.count(k) for k in V}, **paired(nest, "selection-adjusted"))
out["passes_gate"] = bool(nest.mean() <= gate); out["expected_lb"] = [round(nest.mean() - 1.17 - .3, 2), round(nest.mean() - 1.17 + .3, 2)]
print(f"  choices {out['selection_adjusted']['choices']} | gate {'PASS' if out['passes_gate'] else 'fail'} | exp. LB {out['expected_lb'][0]}-{out['expected_lb'][1]}")
if args.write:                                                          # component arrays for p17_build.py (one cnn tag per fine-tuned part)
    for nm, ks in parts[args.write]:
        np.save(C / "oof" / f"cnn_p21_{nm}_oof.npy", avg(oo, *ks)); np.save(C / "oof" / f"cnn_p21_{nm}_test.npy", avg(te, *ks))
    out["written"] = [f"p21_{nm}" for nm, _ in parts[args.write]]; print("wrote components", out["written"])
json.dump(out, open(REP / "p21_eval.json", "w"), indent=1)
