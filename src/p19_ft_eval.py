"""Phase 19 / priority 3 verification: DINOv2-S fine-tuning (cnn3.py tag x_dinos_ft) on all 15 folds (train only).

Exactly two pre-declared variants (user decision after the rep-0 screen):
  A "replace CNN" = mean(DINOv2-FT, Ridge004, Ridge[DINOv2-S])      B "add as 4th" = mean(CNN r18, Ridge004, Ridge[DINOv2-S], DINOv2-FT)
Gate: honest 15-fold CV at least 0.25 better than 005. Paired fold tests vs 005 and 006; expected LB = CV - 1.17 (+-0.3).
"""
import json

import numpy as np
from scipy import stats

from common import C, REP, load_meta, splits

M, itr, ite, y = load_meta(); sp = splits(len(y))
L = lambda n: np.load(C / "oof" / n)
c18, cup, rf, ds, ft = (L(f) for f in ["cnn_r18_final_a_oof.npy", "cnn_r18up2_final_oof.npy", "p17_ridge004_oof.npy", "p17_dinos_252_oof.npy", "cnn_x_dinos_ft_oof.npy"])
assert ft.shape == (3, len(y)) and np.isfinite(ft).all(), "x_dinos_ft OOF incomplete (need reps 0-2)"
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
ref = {"005": fr((c18 + rf + ds) / 3), "006": fr((cup + rf + ds) / 3)}
print(f"DINOv2-FT single 15 folds {fr(ft).mean():.3f} ± {fr(ft).std():.3f} | 005 {ref['005'].mean():.3f} | 006 {ref['006'].mean():.3f} | gate {ref['005'].mean() - .25:.3f}")
out = dict(single=float(fr(ft).mean()))
for nm, o in [("A replace CNN", (ft + rf + ds) / 3), ("B add as 4th", (c18 + rf + ds + ft) / 4)]:
    f = fr(o); r = out[nm] = dict(cv=float(f.mean()), sd=float(f.std()), expected_lb=[round(f.mean() - 1.17 - .3, 2), round(f.mean() - 1.17 + .3, 2)])
    for k, rf_ in ref.items():
        d = f - rf_; r[f"vs_{k}"] = dict(d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    r["gate"] = bool(r["vs_005"]["d"] <= -.25)
    print(f"  {nm:14s} {f.mean():.3f} ± {f.std():.3f} | vs 005 d {r['vs_005']['d']:+.3f} p {r['vs_005']['p']:.2g} {r['vs_005']['wins']}/15 | "
          f"vs 006 d {r['vs_006']['d']:+.3f} p {r['vs_006']['p']:.2g} {r['vs_006']['wins']}/15 | exp. LB {r['expected_lb'][0]}-{r['expected_lb'][1]} | gate {'PASS' if r['gate'] else 'fail'}")
    for rep in range(3):
        print(f"      repeat {rep}: d vs 005 {(f - ref['005'])[rep * 5:(rep + 1) * 5].mean():+.3f}")
json.dump(out, open(REP / "p19_ft_eval.json", "w"), indent=1)
