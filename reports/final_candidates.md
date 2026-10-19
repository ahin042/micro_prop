# Final candidates

CV = average RMSE over the same 15 folds (5 folds × 3 repeats, seed 42) on the 500 training images.
Public LB = score on 30% of the test set (300 images). The final ranking uses the other 70%.
Every file is a plain average of its parts (equal weights, nothing fitted).

| file | parts | CV | public LB | LB − CV |
|---|---|---|---|---|
| `004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv` | ResNet18 CNN, Ridge on hand-made features | 13.529 | 12.373 | −1.16 |
| `005_eq3_cnn_r18_ridge004_dinov2s.csv` | + Ridge on frozen DINOv2-S features | 13.113 | 11.932 | −1.18 |
| `006_eq3_cnn_r18up2full_ridge004_dinov2s.csv` | as 005, CNN on 2×-upscaled images | 12.988 | 12.002 | −0.99 |
| `007_eq3_dinov2s_ft_ridge004_dinov2s.csv` | fine-tuned DINOv2-S, Ridge on hand-made features, Ridge on frozen DINOv2-S | 12.790 | 11.677 | −1.11 |
| `008_eq4_dinov2s_ft_dinov2b_ft_ridge004_dinov2s.csv` | 007 + fine-tuned DINOv2-B | **12.604** | **11.497** | −1.11 |

## Step-by-step comparison (same 15 folds)

| change | CV difference | folds better | public LB difference |
|---|---|---|---|
| 004 → 005 | −0.416 (p 0.00001) | 14 of 15 | −0.442 |
| 005 → 006 | −0.125 (p 0.002) | 13 of 15 | **+0.070** (worse) |
| 005 → 007 | −0.323 (p 0.0015) | 12 of 15 | −0.255 |
| 007 → 008 | −0.186 (p 0.0007) | 13 of 15 | −0.180 |

Only 006 disagreed with its CV. We checked the usual causes (models trained on all data, ensemble size, train/test differences) and found none. It may be chance in the 300 public test images.

## Notes

- Pretrained weights: `resnet18.a1_in1k` (all files), DINOv2-S (005–008), DINOv2-B (008). All Apache-2.0 and public before the competition.
- 004 is the only file without DINOv2.
- 008 did not reach our own adoption rule (CV at least 0.25 better than 007; it was 0.19 better), but it has the best CV and the best public LB.
- Rebuilding in an empty folder gives byte-identical files on the reference Mac: 004–006 with `run.py --mode fast` and `--mode full`, 007 and 008 with `--mode fast`.

SHA-256:

- 004 `1cabcbcfcd2b8ac0b094d224a380611172d4c15a247b961b491ce9f53e8fd1b8`
- 005 `d082a9496e74e7f3c2d779d45f9828766688aa8d5063ad105c661265d946bc72`
- 006 `fdf9427587b3a7c29f48419e8fd6c3de3b021e8e9c2de5667b19d69e9eab9476`
- 007 `93292368f2f44b9d63918f800fa9766f16712b7d82a0d78fed0a0c9b0377cff6`
- 008 `2c383d076343630423a358fab18227dbe31fb701197b17957f4bc88597999da9`
