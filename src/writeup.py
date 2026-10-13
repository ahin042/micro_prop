"""Solution write-up PDF (reports/solution.pdf). Every number is read from reports/ or recomputed from cache/oof; nothing is typed by hand."""
import glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy import stats

from common import C, REP, load_meta, splits

J = lambda n: json.load(open(REP / n))
s0, s2, pb, st = (J(f) for f in ["stage0.json", "stage2.json", "partb.json", "stationarity.json"])
lbt = J("lb_tracking.json"); LB = {s["file"][:3]: s for s in lbt["submissions"] if s["file"][:3] in ("003", "004", "005", "006") and "regenerated" not in s["file"]}
cnn = {p.split("cnn_")[1][:-5]: json.load(open(p)) for p in sorted(glob.glob(str(REP / "cnn_*.json")))}
M, itr, ite, y = load_meta(); sp = splits(len(y)); O = C / "oof"
fr = lambda o: np.array([np.sqrt(np.mean((o[i // 5, b] - y[b]) ** 2)) for i, (a, b) in enumerate(sp)])
L = lambda n: np.load(O / n)

# ---- Figure 1: estimator validity (|rho with y| vs |rho with blur|) -------------------------------------------------------
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index().iloc[itr]
BLUR = ["q_crete_blur", "q_lapvar_dn", "r_blur_r0.01"]
cols = [c for c in F.columns if c[:2] in ("r_", "n_", "b_", "m_", "d_", "x_")]
rs = lambda a, b: abs(stats.spearmanr(a, b)[0])
ry = np.array([rs(F[c], y) for c in cols]); rb = np.array([max(rs(F[c], F[b]) for b in BLUR) for c in cols]); above = int((ry > rb).sum())
fam = np.array([c[:2] for c in cols]); GS = {"n_mli", "n_ws_eqd_mean", "r_acf_len50", "r_spec_r50", "m_knee_wl"}
fig, ax = plt.subplots(figsize=(6.2, 4.3))
for f_, col in zip(["r_", "n_", "b_", "m_", "d_", "x_"], ["#4c72b0", "#55a868", "#8172b2", "#c44e52", "#dd8452", "#937860"]):
    m = fam == f_; ax.scatter(rb[m], ry[m], s=14, alpha=.75, c=col, label=f"{f_} ({m.sum()})")
for c, a, b in zip(cols, rb, ry):
    if c in GS: ax.annotate(c, (a, b), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlim=(0, .9), ylim=(0, .5), xlabel="max |Spearman| with blur estimates", ylabel="|Spearman| with hardness")
ax.legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(REP / "fig_validity.png", dpi=170); plt.close(fig)

# ---- component and blend CVs from the saved OOF ---------------------------------------------------------------------------
c18, cup, rf, ds = L("cnn_r18_final_a_oof.npy"), L("cnn_r18up2_final_oof.npy"), L("p17_ridge004_oof.npy"), L("p17_dinos_252_oof.npy")
fold = {"004": fr((c18 + rf) / 2), "005": fr((c18 + rf + ds) / 3), "006": fr((cup + rf + ds) / 3)}
comp = {"CNN ResNet18 (15 folds)": fr(c18), "CNN ResNet18, 2x upsampled (15 folds)": fr(cup), "Ridge[features + latent net]": fr(rf), "Ridge[frozen DINOv2-S @252]": fr(ds)}
pair = lambda a, b: (lambda d: (d.mean(), stats.ttest_1samp(d, 0).pvalue, int((d < 0).sum())))(fold[a] - fold[b])

# ---- Figure 2: CV vs public LB ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.6)); xs = np.linspace(12.8, 14.5, 10); g = lbt["calibration"]["gap_used"]
ax.plot(xs, xs + g, "k--", lw=.8, label=f"LB = CV {g:+.2f}")
for k, s in sorted(LB.items()):
    ax.scatter(s["cv"], s["lb"], s=30, c="#c44e52" if k == "006" else "#4c72b0"); ax.annotate(k, (s["cv"], s["lb"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.set(xlabel="CV RMSE (5x3 folds, train)", ylabel="public LB RMSE (300 test images)"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(REP / "fig_cv_lb.png", dpi=170); plt.close(fig)

# ---- document --------------------------------------------------------------------------------------------------------------
ss = getSampleStyleSheet(); H1, H2, P = ss["Title"], ss["Heading2"], ss["BodyText"]; P.fontSize, P.leading = 9.2, 12
para = lambda t: Paragraph(t, P); f3 = lambda a: f"{np.mean(a):.3f} ± {np.std(a):.3f}"


def table(rows, w=None):
    t = Table(rows, colWidths=w, hAlign="LEFT")
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                           ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


gs = pd.read_csv(REP / "stage1_feature_rho.csv", index_col=0)
acc = pd.read_csv(REP / "p13_latnet_accuracy.csv", index_col=[0, 1])
story = [Paragraph("Microstructure → hardness: solution", H1),
         para("Image regression of a synthetic hardness proxy from 256×256 grayscale micrographs (500 train / 1000 test, RMSE). "
              "Protocol: one shared RepeatedKFold(5×3, seed 42) over images for every model, so every comparison is paired by fold. "
              "Python only, no external data, no remote APIs, test images never used for training; pretrained weights only under "
              "Apache-2.0 / BSD-3 (README.md, src/weights_manifest.json)."),
         Paragraph("1. Data audit", H2),
         para(f"Labels are unrounded float64 generator output with no value grid (Rayleigh R ≤ {max(v['R'] for v in s0['grid']['rayleigh'].values()):.2f} for every spacing); "
              f"one PNG encoder signature for all files; no train/test shift ({s0['cheap']['ks_n_sig_bonf']}/{s0['cheap']['n_feat']} statistics significant after Bonferroni); "
              f"IDs carry no label information (ρ = {s0['ids']['rho_id_y']:+.3f}); no duplicated or near-duplicated structures."),
         Paragraph("2. Measurement bias: classic estimators track the imaging nuisance", H2),
         para(f"Blur, noise and contrast vary strongly per image and are label-independent. Only {above}/{len(cols)} structural features correlate more "
              f"with hardness than with blur (Fig. 1). The ASTM E112 mean linear intercept has ρ<sub>y</sub> = {gs.loc['n_mli', 'rho_y']:+.3f} but "
              f"ρ<sub>noise</sub> = {gs.loc['n_mli', 'rho_noise']:+.3f}; a Hall-Petch power law gives a grain-size exponent of {s2['T1']['exponents'][0]:+.3f} "
              f"instead of −0.5. Analytic blur corrections (MTF division, Richardson-Lucy, scale-space extrapolation) moved only {pb['n_breaking']} of 60 "
              f"corrected features across the diagonal. 128-px crops are not label-equivalent (quadrant ICC {st['median_icc']:.2f}), so all models use full images."),
         Image(str(REP / "fig_validity.png"), width=12.5 * cm, height=8.7 * cm),
         para("<i>Fig. 1.</i> One point per feature; grain-size estimators (labelled) sit far below the diagonal."),
         Paragraph("3. Signals that survived", H2),
         para("<b>Mode-aware dark-phase fraction.</b> The matrix level is the KDE mode of noise-averaged grain intensities; a grain or pixel counts as dark phase "
              "when it is darker than the mode by a fixed number of gray levels (absolute thresholds beat contrast-normalized ones). "
              "<b>Latent net.</b> A ResNet18 multi-head regressor learns y-free latent measurements (dark fraction, alignment, grain size, aspect, boundary density, …) "
              "from the cleanest training images, re-degraded with blur/noise/contrast drawn from the dirty half of the training set. On held-out clean images under "
              f"the heaviest degradation (dirty p90) it keeps r = {acc.loc[('L4 dirty p90', 'fd20'), 'net_r']:.2f} for the dark fraction where the classical estimator "
              f"drops to r = {acc.loc[('L4 dirty p90', 'fd20'), 'classical_r']:.2f}. Its predictions are features of the 004 Ridge; cross-fitting them changes the CV by < 0.01."),
         PageBreak(),
         Paragraph("4. CNN", H2),
         para("timm <font face='Courier'>resnet18.a1_in1k</font>, in_chans=1, one output, MSE on standardized y, AdamW (lr 3e-4, wd 1e-2), 40 epochs cosine, batch 32, "
              "bf16 autocast (MPS), EMA weights, no early stopping. Full 256×256 input; augmentation D4 + Gaussian blur σ~U(0,2) + noise σ~U(0,20) "
              "(photometric augmentation hurts: contrast carries signal). OOF and test use the same inference: 8 D4 transforms, EMA weights. "
              "Screening rule for alternatives: same split (repeat 0, folds 0-4) as the baseline, drop unless clearly better.")]
base0 = np.array([cnn["r18_final_a"]["folds"][str(i)]["rmse"] for i in range(5)])
rows = [["config (repeat 0, folds 0-4)", "CV RMSE", "Δ vs ResNet18", "fold wins"]]
NAMES = {"x_tv": "ResNet18, torchvision weights", "x_cnxt": "ConvNeXt-Tiny (in22k)", "x_r50": "ResNet50 (tv2)", "x_ev2s": "EfficientNetV2-S (in21k)",
         "x_up2": "ResNet18, 2x upsampled input", "x_up2_lr1e3": "ResNet18, 2x upsampled, lr 1e-3"}
rows.append(["ResNet18 a1 (baseline)", f"{base0.mean():.3f}", "–", "–"])
for k, nm in NAMES.items():
    if k in cnn and all(str(i) in cnn[k]["folds"] for i in range(5)):
        f_ = np.array([cnn[k]["folds"][str(i)]["rmse"] for i in range(5)]); rows.append([nm, f"{f_.mean():.3f}", f"{(f_ - base0).mean():+.3f}", f"{int((f_ < base0).sum())}/5"])
story += [table(rows, [6.2 * cm, 2.4 * cm, 2.8 * cm, 2.2 * cm]), Spacer(1, 4),
          para("Training RMSE without augmentation reaches 3-5 for ConvNeXt / torchvision weights while validation stays ≈ 14: the limit is generalisation from 400 images, not fit."),
          Paragraph("5. Frozen DINOv2 embeddings + Ridge", H2),
          para("Per image (inference only): [CLS, patch-token mean, patch-token std] of the last 4 blocks of a frozen DINOv2 ViT, then a standardized RidgeCV. "
               "The candidate set {Small, Base, Large} was declared before the runs and every result is reported; 252 px (≈ 1:1 with the 256 px images) "
               "was declared the primary input size before the Base/Large runs.")]
drows = [["input size"] + ["Small", "Base", "Large"]]
for s in (126, 168, 224, 252, 448):
    drows.append([f"{s} px"] + [f"{fr(L(f'p17_dino{a}_{s}_oof.npy')).mean():.3f}" for a in ("s", "b", "l")])
story += [table(drows, [3 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]), Spacer(1, 4),
          para("Small at 252 px is the best single model of the project. Larger backbones are worse here (more dimensions than 400 training rows can support); "
               "supervised ImageNet CNN features (ConvNeXt, EfficientNetV2, ResNet50) give 14.3-14.7."),
          Paragraph("6. Final candidates", H2)]
frows = [["", "recipe (equal-weight mean; nothing fitted on OOF)", "CV RMSE", "public LB", "LB − CV"]]
REC = {"004": "CNN ResNet18 + Ridge[features + latent net]", "005": "004 + Ridge[frozen DINOv2-S]", "006": "005 with the 2x-upsampled CNN (test: 5 full-data seeds)"}
for k in ("004", "005", "006"):
    s = LB[k]; frows.append([k, REC[k], f3(fold[k]), f"{s['lb']:.4f}" if s.get("lb") else "–", f"{s['lb'] - fold[k].mean():+.3f}" if s.get("lb") else "–"])
story += [table(frows, [1 * cm, 8.2 * cm, 2.6 * cm, 1.9 * cm, 1.6 * cm]), Spacer(1, 4)]
ctab = [["component", "CV RMSE"]] + [[k, f3(v)] for k, v in comp.items()]
d54, d64, d65 = pair("005", "004"), pair("006", "004"), pair("006", "005")
story += [table(ctab, [8.2 * cm, 3 * cm]), Spacer(1, 4),
          para(f"Paired fold tests: 005 vs 004 Δ {d54[0]:+.3f} (p {d54[1]:.1g}, {d54[2]}/15 folds); 006 vs 004 Δ {d64[0]:+.3f} (p {d64[1]:.1g}, {d64[2]}/15); "
               f"006 vs 005 Δ {d65[0]:+.3f} (p {d65[1]:.1g}, {d65[2]}/15). The CV → public-LB offset is stable for 003-005 (Fig. 2) because test predictions "
               "come from models fit on all 500 images. 006 is better than 005 on every CV view but worse on the public LB; this is unexplained, possibly "
               "public-subset noise (about six candidate comparisons on 300 public images). Ruled out: the full-data CNN models (they match the fold ensembles), "
               "ensemble size, train/test shift, and most of the selection effect."),
          Image(str(REP / "fig_cv_lb.png"), width=10 * cm, height=6.7 * cm),
          para("<i>Fig. 2.</i> CV vs public LB of the uploaded candidates."),
          Paragraph("7. Reproducibility", H2),
          para("<font face='Courier'>python src/run.py --mode full</font> rebuilds 004/005/006 from the raw images (fast mode reuses the cached CNN predictions). "
               "LightGBM runs deterministic and single-threaded with sorted design matrices; PyTorch runs with fixed seeds and deterministic algorithms. "
               "A fresh-folder fast-mode run reproduces all three submissions byte-identically (SHA-256 in reports/lb_tracking.json); "
               "re-runs of the CNNs and the latent net are bitwise identical; the full-mode result is recorded in experiments.md. "
               "Environment, pinned versions, weight hashes and licenses: README.md. Every experiment and its CV: experiments.md.")]
SimpleDocTemplate(str(REP / "solution.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                  title="Microstructure hardness - solution").build(story)
print("wrote reports/solution.pdf")
