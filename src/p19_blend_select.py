"""Phase 19 / priority 4: equal-weight blend of a component set chosen by nested CV (no fitted weights; train only).

Pool (pre-declared): every honest component OOF of Phases 17-19 present in cache/oof (fixed recipes or nested-selected configs).
For each outer fold (repeat r, validation fold b) the subset of the pool with the lowest RMSE of its equal-weight mean on the
training images of that fold (OOF of the same repeat) is chosen and scored on b -> selection-adjusted CV (not the best of the
2^k subsets). Caveat: a training image's OOF comes from a model that saw fold b, so the selection step is not perfectly clean;
with an aggregate criterion over 400 images this effect is small. Paired tests vs 005 and 006; gate = 005 - 0.25.
"""
import itertools, json

import numpy as np
from scipy import stats

from common import C, REP, load_meta, splits

M, itr, ite, y = load_meta(); sp = splits(len(y))
POOL = {"cnn_r18": "cnn_r18_final_a_oof.npy", "cnn_r18_up2": "cnn_r18up2_final_oof.npy", "ridge004": "p17_ridge004_oof.npy",
        "dinoS_last4": "p17_dinos_252_oof.npy", "dinoS_mid_nested": "p19_heads_dinos_nested_oof.npy", "dinoB_mid_nested": "p19_heads_dinob_nested_oof.npy",
        "dinoS_d4aug_nested": "p19_d4aug_dinos_nested_oof.npy"}
for m in ("resnet18_micronet", "resnet18_imagenet_micronet", "resnet34_micronet", "resnet34_imagenet_micronet", "resnet50_micronet", "resnet50_imagenet_micronet"):
    POOL[f"micronet_{m}"] = f"p19_micronet_{m}_nested_oof.npy"
P = {k: np.load(C / "oof" / f) for k, f in POOL.items() if (C / "oof" / f).exists()}; names = list(P)
print(f"pool ({len(names)}): {names}")
subsets = [s for r in range(1, len(names) + 1) for s in itertools.combinations(range(len(names)), r)]
A = np.stack([P[k] for k in names])                                          # (k, 3, n)
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
out_f, chosen = np.zeros(len(sp)), []
for fi, (a, b) in enumerate(sp):
    r = fi // 5; Ar = A[:, r]
    best = min(subsets, key=lambda s: np.mean((Ar[list(s)][:, a].mean(0) - y[a]) ** 2))
    chosen.append(tuple(names[j] for j in best)); out_f[fi] = np.sqrt(np.mean((Ar[list(best)][:, b].mean(0) - y[b]) ** 2))
L = lambda n: np.load(C / "oof" / n)
b005 = fr((L("cnn_r18_final_a_oof.npy") + L("p17_ridge004_oof.npy") + L("p17_dinos_252_oof.npy")) / 3)
b006 = fr((L("cnn_r18up2_final_oof.npy") + L("p17_ridge004_oof.npy") + L("p17_dinos_252_oof.npy")) / 3)
best_overall = min(subsets, key=lambda s: fr(A[list(s)].mean(0)).mean())                     # winner's-curse reference only
res = dict(pool=names, selection_adjusted_cv=float(out_f.mean()), best_of_all_subsets_cv=float(fr(A[list(best_overall)].mean(0)).mean()),
           best_of_all_subsets=[names[j] for j in best_overall], chosen_counts={",".join(c): chosen.count(c) for c in set(chosen)})
print(f"selection-adjusted CV {out_f.mean():.3f}   (best of all {len(subsets)} subsets, in-sample: {res['best_of_all_subsets_cv']:.3f} = {res['best_of_all_subsets']})")
for nm, ref in [("005", b005), ("006", b006)]:
    d = out_f - ref; res[f"vs_{nm}"] = dict(d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), wins=int((d < 0).sum()))
    print(f"  vs {nm}: d {d.mean():+.3f} p {stats.ttest_1samp(d, 0).pvalue:.2g} wins {(d < 0).sum()}/15")
print(f"  gate (vs 005 ≤ -0.25): {'PASS' if res['vs_005']['d'] <= -.25 else 'fail'}")
for c, n in sorted(res["chosen_counts"].items(), key=lambda kv: -kv[1]): print(f"  chosen {n:2d}/15: {c}")
json.dump(res, open(REP / "p19_blend_select.json", "w"), indent=1)
