# Microstructure → hardness regression

Image regression (256×256 grayscale microstructure → synthetic hardness proxy, metric RMSE).
Full experiment log: [experiments.md](experiments.md). Machine-readable results: `reports/*.json`.

## Reproduce (single command)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp            # macOS only: LightGBM's OpenMP runtime
.venv/bin/python src/run.py --mode full     # everything, incl. CNN training (~1.5 h on Apple M5)
.venv/bin/python src/run.py --mode fast     # same, reusing cached CNN OOF/test predictions
```

Data must be placed at `./data/{train,test}/*.png`, `./data/train.csv`, `./data/sample_submission.csv`.
All paths are relative to the repository root. Outputs: `submissions/*.csv`, `cache/` (features, OOF), `reports/`.

## Environment (as run)

| item | version |
|---|---|
| OS | macOS 26.3.1 (Darwin 25.3.0), Apple M5, 10 cores, 16 GB, PyTorch MPS backend |
| Python | 3.13.12 |
| core | numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, joblib 1.6.0, pillow 12.3.0, pyarrow 26.0.0 |
| models | lightgbm 4.7.0 (libomp 23.1.3), torch 2.14.1, torchvision 0.29.1, timm 1.0.30 |

Every package is pinned in `requirements.txt` (`pip freeze`).

## Pretrained weights and licenses

| weights | source | license |
|---|---|---|
| `timm/resnet18.a1_in1k` | Hugging Face Hub via `timm.create_model(..., pretrained=True)` | Apache-2.0 (model card) |
| `timm/efficientnet_b0.ra_in1k` (ablation only) | Hugging Face Hub via timm | Apache-2.0 (model card) |
| timm library | PyPI | Apache-2.0 |

No other pretrained weights, no external data, and no remote APIs. Test images are never used for training:
per-image features are computed independently, and every fitted statistic (scalers, Box-Cox λ, CNN input
normalization, target standardization, blend weights) is fitted on training rows only.

## Reproducibility

- All seeds are fixed (`SEED = 42`). The CV protocol is `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)`,
  and every model uses the same image folds, so OOF predictions blend and compare fold by fold.
- Stage 0 output (`reports/stage0.json`) is bit-identical across `n_jobs` and the cache. LightGBM runs with `deterministic=True`
  and a fixed thread count.
- Some PyTorch MPS kernels are not bit-deterministic, so CNN OOF can differ in the last digits between runs.

## Pipeline (`src/`)

| script | purpose |
|---|---|
| `stage0_recon.py` | label structure, duplicate search with a positive control and a robust null, PNG forensics, baseline anchors |
| `baseline_cheap.py` | anchor table (const / ridge / kNN / LightGBM), target-transform choice by CV, submission 001 |
| `features.py` | 174 per-image features: nuisance (`q_`), raw/normalized estimators (`r_`/`n_`), MTF-corrected (`m_`), deconvolved (`d_`), scale-space (`x_`) |
| `stage1_report.py` | estimator validity (rho with y vs rho with blur), Stage 1 CV, submission 002 |
| `stage2_formula.py` | power-law test, enumerated formula search, residual diagnostics |
| `ceiling.py` | feature-space twin test, residual-structure probe |
| `partb_report.py` | blur-correction evaluation and CV |
| `stationarity.py` | quadrant ICC: are crops label-equivalent? |
| `cnn.py` | timm CNN on full images, D4 + nuisance augmentation, same folds |
| `blend.py` | equal-weight vs cross-fitted NNLS blends, paired fold tests, submissions |
| `common.py` | paths, CV protocol, models, submission writer/validator |

## Central finding: measurement bias, not model capacity

The images carry strong, label-independent nuisance: blur, noise (σ 4–20) and contrast (span 46–131). Classic
microstructure estimators measure the nuisance rather than the material. 84 of 114 structural features correlate
more with blur or noise than with hardness. The ASTM E112 mean linear intercept has rho with y −0.03 but rho with
noise 0.76. Fitting a Hall-Petch power law gives a grain-size exponent of +0.02 instead of −0.5.
Analytic blur correction (MTF deconvolution of the spectrum, Richardson-Lucy before segmentation, scale-space
extrapolation) did not break the domination of any grain-size estimator. See `experiments.md` and the PDF write-up.
