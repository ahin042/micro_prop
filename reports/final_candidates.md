# Final candidates

CV = RepeatedKFold(5x3, seed 42) fold-mean RMSE on the 500 training images (paired over the same 15 folds).
"Honest CV" recomputes the feature Ridge with cross-fitted latent-net features; it changes nothing material.
Public LB = 30% of the test set (300 images). The final ranking uses the other 70%.

| file | recipe (equal weights, nothing fitted on OOF) | CV (honest) | public LB | LB − CV | pretrained weights used | sha256 |
|---|---|---|---|---|---|---|
| `004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv` | mean(CNN ResNet18, Ridge[features + latent net]) | 13.529 (13.529) | 12.3733 | −1.156 | `resnet18.a1_in1k` (Apache-2.0) | `1cabcbcf…` |
| `005_eq3_cnn_r18_ridge004_dinov2s.csv` | mean(CNN ResNet18, Ridge[features + latent net], Ridge[frozen DINOv2-S]) | 13.113 (13.114) | **11.9317** | −1.181 | `resnet18.a1_in1k`, `vit_small_patch14_dinov2.lvd142m` (both Apache-2.0; DINOv2 frozen) | `d082a949…` |
| `006_eq3_cnn_r18up2full_ridge004_dinov2s.csv` | as 005, CNN = ResNet18 on 2x-upsampled input (test: 5 full-data seeds) | **12.988** (12.987) | 12.0015 | −0.987 | same as 005 | `fdf94275…` |

Pairwise CV tests (15 paired folds): 005 vs 004 −0.416 (p 1e-5, 14/15) · 006 vs 004 −0.542 (p 1e-6, 15/15) · 006 vs 005 −0.125 (p .002, 13/15).
Public LB: 005 vs 004 −0.442 (matches CV) · 006 vs 005 **+0.070** (CV said −0.125).

Notes for the decision
- 006 vs 005: the two signals disagree. The 006 public gap is unexplained, possibly public-subset noise (about 6 candidate comparisons so far, 300 public images). Ruled out as causes: the full-data CNN models (they match the fold ensembles), ensemble size, train/test distribution shift (none detected), and most of the selection effect (the gain holds on the folds not used for screening). The spread of the 006−005 difference over random 300-image subsets is sd ≈ 0.055.
- 004 is the fallback without DINOv2 weights.
- Shared by all three: the feature pipeline, the latent net (no labels used), and `resnet18.a1_in1k`.
- Reproducibility: `run.py --mode fast --target all` reproduces all three byte-identically in an empty folder. The full mode, which retrains the CNNs, is checked after the experiment freeze; the result goes here.
