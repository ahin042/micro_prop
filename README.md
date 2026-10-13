# Microstructure → hardness regression

Language: English. (`README.ko.md` / `README.ja.md` and `experiments.ko.md` / `experiments.ja.md` are frozen at Phase 11 and out of date.)

Image regression: 256×256 grayscale synthetic microstructure → synthetic hardness proxy (HV), metric RMSE.
500 train / 1000 test images. Full experiment log with every CV number: [experiments.md](experiments.md).
Machine-readable results: `reports/*.json`, public-LB tracking: `reports/lb_tracking.json`.

## Final candidates

| file | recipe (equal-weight mean, nothing fitted on OOF) | CV RMSE | public LB | pretrained weights |
|---|---|---|---|---|
| `004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv` | CNN ResNet18 + Ridge[features + latent net] | 13.529 | 12.3733 | resnet18.a1_in1k |
| `005_eq3_cnn_r18_ridge004_dinov2s.csv` | + Ridge[frozen DINOv2-S embedding] | 13.113 | **11.9317** | resnet18.a1_in1k, DINOv2-S |
| `006_eq3_cnn_r18up2full_ridge004_dinov2s.csv` | as 005, CNN on 2x-upsampled input | **12.988** | 12.0015 | resnet18.a1_in1k, DINOv2-S |

CV = RepeatedKFold(5×3, seed 42) fold-mean RMSE on the training images. Details, pairwise tests and notes: [reports/final_candidates.md](reports/final_candidates.md).

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # exact versions; full environment: requirements-lock.txt
brew install libomp                               # macOS only: LightGBM's OpenMP runtime
.venv/bin/python src/run.py --mode full           # everything from raw images, incl. all CNN training
.venv/bin/python src/run.py --mode fast           # same, but reuses the cached CNN predictions in ./cache/oof
```

- Data: `./data/train/*.png`, `./data/test/*.png`, `./data/train.csv`, `./data/sample_submission.csv` (not in git).
- Weights: `./weights/<name>.safetensors` (not in git; see the table below). Every script checks the SHA-256 and refuses a different file. No network access is needed (`HF_HUB_OFFLINE=1`).
- `--target {004,005,006,all}` selects the submissions (default: all). `HARDNESS_OUT=<dir>` writes every output (`cache/`, `reports/`, `submissions/`) to a fresh folder instead of the repository.
- All paths are relative to the repository root. Outputs: `submissions/*.csv`, `cache/` (features, OOF/test predictions), `reports/`.
- Expected SHA-256 of the outputs: 004 `1cabcbcf…`, 005 `d082a949…`, 006 `fdf94275…` (full hashes in `reports/lb_tracking.json`).
- Run time on the reference machine: about 15 min in fast mode. Full mode adds the CNN training (ResNet18 15 folds ≈ 35 min; 2x-upsampled ResNet18 15 folds ≈ 2 h for the CV report, plus 5 full-data seeds ≈ 45 min for the 006 test predictions).

## Environment (as run)

| item | version |
|---|---|
| OS / hardware | macOS 26.3.1 (Darwin 25.3.0), Apple M5, 10 cores, 16 GB, PyTorch MPS backend |
| Python | 3.13.12 |
| core | numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, joblib 1.6.0, pillow 12.3.0, pyarrow 26.0.0 |
| models | lightgbm 4.7.0 (libomp 23.1.3), torch 2.14.1, torchvision 0.29.1, timm 1.0.30, safetensors 0.8.0, huggingface_hub 2.2.0 |
| reporting | matplotlib 3.11.2, reportlab 5.0.1 |

## Pretrained weights and licenses

All weights are ImageNet / self-supervised checkpoints with permissive licenses, loaded through timm (Apache-2.0).
Source, commit, SHA-256 and license of every file: `src/weights_manifest.json` (license = Hugging Face model card and timm
pretrained config; they agree for every file). "Screening only" weights were evaluated and rejected; no submission uses them.

| weights | used by | source | commit | SHA-256 | license |
|---|---|---|---|---|---|
| `resnet18.a1_in1k` | **004, 005, 006** (CNN, latent net) | huggingface.co/timm/resnet18.a1_in1k | `491b427b45c9` | `80c49dee3da4822c…` | apache-2.0 |
| `vit_small_patch14_dinov2.lvd142m` | **005, 006** (frozen embedding) | huggingface.co/timm/vit_small_patch14_dinov2.lvd142m | `4610ca143709` | `04d27f3400d059fc…` | apache-2.0 |
| `vit_base_patch14_dinov2.lvd142m` | screening only | huggingface.co/timm/vit_base_patch14_dinov2.lvd142m | `4685c99dabff` | `55cbb5d887b336d4…` | apache-2.0 |
| `vit_large_patch14_dinov2.lvd142m` | screening only | huggingface.co/timm/vit_large_patch14_dinov2.lvd142m | `4741e1cafbf4` | `0424a5d1b515278c…` | apache-2.0 |
| `convnext_tiny.fb_in22k_ft_in1k` | screening only | huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k | `bc48a87f119b` | `77b619e187de71bc…` | apache-2.0 |
| `tf_efficientnetv2_s.in21k_ft_in1k` | screening only | huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k | `ea9abc143ea2` | `6f1933fb6c0d760e…` | apache-2.0 |
| `resnet50.tv2_in1k` | screening only | huggingface.co/timm/resnet50.tv2_in1k | `10eea9779fc8` | `30f57ada2f100011…` | bsd-3-clause |
| `efficientnet_b0.ra_in1k` | screening only | huggingface.co/timm/efficientnet_b0.ra_in1k | `1b5383e5f79c` | `d569899762ea9b13…` | apache-2.0 |
| `resnet18.tv_in1k` | screening only | https://download.pytorch.org/models/resnet18-f37072fd.pth (torchvision IMAGENET1K_V1) | `n/a (content-addressed URL)` | `f37072fd47e89c5e…` | bsd-3-clause |

To restore `weights/`: download `model.safetensors` from the listed repository at the listed commit and save it as
`weights/<name>.safetensors` (the torchvision file as `weights/resnet18.tv_in1k.pth`).

## Competition rules

- Python only, no remote APIs, every model runs locally.
- No external data. Pretrained weights only (table above). No microstructure-pretrained models.
- The test images are never used for training in any form. Per-image features and frozen embeddings are computed independently for each image. Every fitted object (scalers, Ridge, CNNs, the latent net, imputation medians) is fitted on training images only. The latent net learns from training images only (y-free pseudo-labels on the cleanest train images, re-degraded with blur/noise drawn from the train nuisance distribution).
- Seeds are fixed and outputs are byte-reproducible (next section).

## Reproducibility

- One CV protocol for every model: `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)` over images (`src/common.py`), so OOF predictions compare and blend fold by fold.
- LightGBM: `deterministic=True`, one thread, explicit seeds, alphabetically sorted design-matrix columns (`common.design`). Column order used to change the model; see `src/recover_003.py`.
- PyTorch: fixed seeds, `torch.use_deterministic_algorithms(True)`, fixed data order. Verified bitwise-identical re-runs on MPS: the ResNet18 CNN (15 folds, OOF and test), the latent net, and the 2x-upsampled CNN fold models (OOF and test, one run interrupted by machine sleep). The 006 full-data seeds are re-run in the full-mode check. CNN steps cap MPS memory (`PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO` = 0.6/0.5, set by `run.py`); this does not change the numerics.
- Verified: `run.py --mode fast --target all` in an empty `HARDNESS_OUT` folder reproduces 004, 005 and 006 byte-identically. The full-mode check is recorded in `experiments.md`.
- `submissions/001` and `003` were produced before the determinism lockdown. `src/recover_003.py` reproduces the uploaded 003 byte-identically; the regenerated 001 differs from the uploaded 001 and is not a candidate.

## Pipeline (`src/`)

| script | role |
|---|---|
| `run.py` | single-command reproduction of 004 / 005 / 006 |
| `common.py` | paths, CV protocol, Ridge / LightGBM, submission writer (refuses to overwrite a different file) |
| `stage0_recon.py`, `baseline_cheap.py` | data audit (labels, duplicates, PNG forensics, train/test shift), anchor models, 001 |
| `features.py` | per-image nuisance-aware features (`q_`, `r_`, `n_`, `b_`, blur-corrected `m_`, `d_`, `x_`) |
| `recover_003.py` | exact reconstruction of submission 003 (reference for paired tests) |
| `p12_phase.py`, `p12_pixphase.py`, `p12_orient.py` | mode-aware dark-phase fractions (grain- and pixel-level), orientation coherence |
| `p13_descriptors.py`, `p13_structure.py` | grain-level descriptors, dark-phase mixture statistics |
| `p13_latnet.py` | degradation-robust latent estimator (ResNet18 multi-head, y-free pseudo-labels, train images only) |
| `cnn.py` | CNN regressor: ResNet18, full 256×256 input, D4 + blur + noise augmentation, EMA, D4 TTA, 15 folds |
| `cnn2.py` | `cnn.py` + input upsampling and more backbones; `--full K` trains K seeds on all training images |
| `p17_dino_probe.py` | frozen DINOv2 / CNN embeddings (inference only) → Ridge probes |
| `p13_build_004.py`, `p17_build.py` | blends, paired fold tests, submissions 004 / 005 / 006 |
| `repro_check.py` | double clean run of the feature chain, artifact digests compared |
| `writeup.py` | solution PDF (`reports/solution.pdf`); every number is read from `reports/` |
| `stage1_report.py`, `stage2_formula.py`, `ceiling.py`, `partb_report.py`, `stationarity.py`, `blend.py`, `p12_*`, `p13_*`, `p14_*`, `p15_*`, `p16_*` | analyses behind experiments.md (not needed for the submissions) |

## Main findings

1. **Measurement bias.** Blur, noise and contrast vary strongly per image and are label-independent. Classic grain-size estimators track the blur more than the hardness. Analytic blur correction did not fix this.
2. **Mode-aware phase fractions.** A dark-phase fraction measured against the matrix gray-level mode, in absolute gray units, is the strongest hand-crafted signal.
3. **Latent net.** A CNN trained to read y-free latent measurements under synthetic degradation measures them far more robustly on dirty images than the classical pipeline.
4. **Frozen self-supervised features.** A Ridge on frozen DINOv2-S embeddings (13.31 CV) beats every hand-crafted Ridge and every fine-tuned CNN. Pre-declared size comparison at 252 px: Small 13.31 < Base 13.69 < Large 14.06. Fine-tuned CNNs are limited by generalisation from 400 training images, not by fit.
5. **006 vs 005.** 006 is better on every CV view but worse on the public LB (+0.070). This is unexplained, possibly public-subset noise. Full-data models, ensemble size and train/test shift were ruled out.

Solution write-up: `reports/solution.pdf` (`python src/writeup.py`).
