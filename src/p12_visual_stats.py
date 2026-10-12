"""Phase 12 / Step 1 support: global raw-intensity and histogram-shape summaries.

Uses original 8-bit images only. No per-image normalization.
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi, stats
from sklearn.mixture import GaussianMixture

from common import C, D, REP, load_meta


def entropy_from_hist(h):
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def image_stats(id_, y, pred):
    f = np.asarray(Image.open(D / "train" / f"{id_}.png")).astype(np.float64)
    h = np.bincount(f.astype(np.uint8).ravel(), minlength=256).astype(float)
    px = np.random.default_rng(42).choice(f.ravel(), 20000, replace=False)[:, None]
    g2 = GaussianMixture(2, random_state=42).fit(px)
    order = np.argsort(g2.means_[:, 0])
    means = g2.means_[order, 0]
    sigmas = np.sqrt(g2.covariances_.reshape(2)[order])
    weights = g2.weights_[order]
    gx = ndi.sobel(f, axis=1)
    gy = ndi.sobel(f, axis=0)
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    c2 = np.mean(mag * np.cos(2 * angle))
    s2 = np.mean(mag * np.sin(2 * angle))
    orient_strength = float(np.hypot(c2, s2) / max(mag.mean(), 1e-9))
    orient_angle_deg = float((0.5 * np.arctan2(s2, c2)) * 180 / np.pi)
    return dict(
        ID=id_,
        y=float(y),
        pred=float(pred),
        residual=float(y - pred),
        mean=float(f.mean()),
        std=float(f.std()),
        p1=float(np.percentile(f, 1)),
        p50=float(np.percentile(f, 50)),
        p99=float(np.percentile(f, 99)),
        span99=float(np.percentile(f, 99) - np.percentile(f, 1)),
        skew=float(stats.skew(f.ravel())),
        kurt=float(stats.kurtosis(f.ravel())),
        entropy=entropy_from_hist(h),
        dark_frac=float((f < 80).mean()),
        bright_frac=float((f > 190).mean()),
        gmm2_low_mean=float(means[0]),
        gmm2_high_mean=float(means[1]),
        gmm2_sep=float(means[1] - means[0]),
        gmm2_low_sigma=float(sigmas[0]),
        gmm2_high_sigma=float(sigmas[1]),
        gmm2_low_weight=float(weights[0]),
        orient_strength=orient_strength,
        orient_angle_deg=orient_angle_deg,
        grad_mean=float(mag.mean()),
    )


if __name__ == "__main__":
    M, itr, _, y = load_meta()
    ids = M.ID.values[itr]
    pred = np.load(C / "oof" / "003_recovered_oof.npy").mean(0)
    rows = [image_stats(i, yy, pp) for i, yy, pp in zip(ids, y, pred)]
    df = pd.DataFrame(rows)
    out = REP / "p12"; out.mkdir(exist_ok=True)
    df.to_csv(out / "visual_stats.csv", index=False)

    corr = {}
    for c in df.columns:
        if c in {"ID"}:
            continue
        corr[c] = float(stats.spearmanr(df[c], df.y).statistic)
    quint = df.assign(y_bin=pd.qcut(df.y, 5)).groupby("y_bin", observed=True)[
        ["y", "pred", "mean", "std", "span99", "dark_frac", "bright_frac", "gmm2_sep", "gmm2_low_weight", "orient_strength", "grad_mean"]
    ].mean()
    summary = {
        "spearman_with_y": {k: round(v, 4) for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))},
        "quintile_means": json.loads(quint.round(4).to_json(orient="index")),
        "worst_residuals": df.reindex(df.residual.abs().sort_values(ascending=False).index).head(24)[
            ["ID", "y", "pred", "residual", "mean", "std", "gmm2_sep", "orient_strength"]
        ].round(4).to_dict("records"),
    }
    (out / "visual_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:6000])
