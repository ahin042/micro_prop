"""Stage 0 (v2): label structure, duplicates, PNG forensics, baseline RMSE anchors.
Usage: python src/stage0_recon.py [--quick] [--force] [--jobs N] -> cache/stage0_*, reports/stage0.{json,md}"""
import argparse, hashlib, io, json, os, time, warnings, zlib
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage, stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
D, C, REP = ROOT / "data", ROOT / "cache", ROOT / "reports"
SEED, N, PCTS = 42, 256, [1, 5, 25, 50, 75, 95, 99]
RIDX = np.hypot(*(np.indices((N, N)) - N // 2)).astype(int); RCNT = np.bincount(RIDX.ravel())  # radius per fftshift-ed bin
EDGES = [1, 2, 4, 8, 16, 32, 64, 96, 129]                     # 8 radial FFT bands
HANN = np.outer(np.hanning(N), np.hanning(N)).astype(np.float32)
SIGS = ["raw32", "struct64", "radial"]                        # D2 signatures (raw32 = box thumbnail)
L, R = [], {}
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


def log(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)


def sigs(a):
    """Signatures of one uint8 image: box32, old bilinear32, structure64, log radial |FFT|, radial power sum."""
    f = a.astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2((f - f.mean()) * HANN)))
    return (f.reshape(32, 8, 32, 8).mean((1, 3)).ravel(),
            np.asarray(Image.fromarray(a).resize((32, 32), Image.BILINEAR), np.float32).ravel(),
            ndimage.gaussian_filter(f, 2).reshape(64, 4, 64, 4).mean((1, 3)).ravel(),
            np.log(np.bincount(RIDX.ravel(), F.ravel())[1:64] / RCNT[1:64] + 1e-6).astype(np.float32),
            np.bincount(RIDX.ravel(), (F.astype(np.float64) ** 2).ravel())[:129])


def png_info(b):
    """Raw PNG chunk walk: layout, IDAT segmentation, zlib level, row filters, trailing bytes."""
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    i, seq, idat, ihdr = 8, [], [], b""
    while i < len(b):
        n, t = int.from_bytes(b[i:i + 4], "big"), b[i + 4:i + 8].decode("latin1")
        ihdr = b[i + 8:i + 8 + n] if t == "IHDR" else ihdr
        if t == "IDAT": idat.append(b[i + 8:i + 8 + n])
        seq.append(t); i += 12 + n
        if t == "IEND": break
    z = b"".join(idat); raw = zlib.decompress(z)
    assert len(raw) == N * (N + 1), "expect 8-bit gray, non-interlaced"
    filt = np.bincount(np.frombuffer(raw[::N + 1], np.uint8), minlength=5)
    return dict(layout=" ".join(k if (m := len(list(g))) == 1 else f"{k}x{m}" for k, g in groupby(seq)),
                layout_norm=" ".join(k if k != "IDAT" else "IDAT+" for k, _ in groupby(seq)),
                n_idat=len(idat), idat0=len(idat[0]) if len(idat) > 1 else -1, ihdr=ihdr.hex(), zlevel=z[1] >> 6,
                has_text=any(t in seq for t in ("tEXt", "zTXt", "iTXt")), has_phys="pHYs" in seq,
                has_time="tIME" in seq, trailing=len(b) - i, filt_mode=int(filt.argmax()), n_filt=int((filt > 0).sum()))


def per_image(split, id_):
    b = (D / split / f"{id_}.png").read_bytes()
    im = Image.open(io.BytesIO(b)); a = np.asarray(im)
    assert im.mode == "L" and a.dtype == np.uint8 and a.ndim == 2 and a.shape == (N, N), id_
    f = a.astype(np.float64); h = np.bincount(a.ravel(), minlength=256); p = h / h.sum()
    s32, b32, s64, rad, pw = sigs(a)
    lo, hi = int(a.min()), int(a.max())
    m = dict(ID=id_, split=split, md5_px=hashlib.md5(a.tobytes()).hexdigest(), md5_file=hashlib.md5(b).hexdigest(),
             vmin=lo, vmax=hi, **{f"pct{q}": v for q, v in zip(PCTS, np.percentile(f, PCTS))},
             mean=f.mean(), std=f.std(), skew=stats.skew(f.ravel()), kurt=stats.kurtosis(f.ravel()),
             entropy=-(p[p > 0] * np.log2(p[p > 0])).sum(), n_levels=int((h > 0).sum()),
             gap_bins=int((h[lo:hi + 1] == 0).sum()), sat0=p[0], sat255=p[255],
             **{f"band{k}": np.log10(pw[EDGES[k]:EDGES[k + 1]].sum() / pw[1:129].sum()) for k in range(8)},
             log_lapvar=np.log(ndimage.laplace(f).var()), hf_std=(f - ndimage.gaussian_filter(f, 1)).std(),
             grad_mean=np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1)).mean(), fsize=len(b), **png_info(b))
    return m, s32, b32, s64, rad


def zn(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), 1e-6)


def dist(A, B, dih=False):
    """Mean squared distance of z-normed rows via Gram trick; dih=True takes the min over the 8 D4 transforms of B."""
    A, B = zn(A), zn(B)
    if not dih: return np.clip(2 - 2 * A @ B.T / A.shape[1], 0, None)
    s = int(np.sqrt(B.shape[1])); Bi = B.reshape(-1, s, s); G = None
    for Bf in (Bi, Bi[:, :, ::-1]):
        for k in range(4):
            g = A @ np.rot90(Bf, k, (1, 2)).reshape(len(B), -1).T
            G = g if G is None else np.maximum(G, g)
    return np.clip(2 - 2 * G / A.shape[1], 0, None)


def q(x, qs=(0, .001, .01, .05, .5)):
    return dict(zip([f"q{v:g}" for v in qs], np.round(np.quantile(x, qs), 4).tolist()))


ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--jobs", type=int, default=-1)
args = ap.parse_args()
rng = np.random.default_rng(SEED)
tr, ss = pd.read_csv(D / "train.csv"), pd.read_csv(D / "sample_submission.csv")
items = [("train", i) for i in tr.ID] + [("test", i) for i in ss.ID]
if args.quick: items = [items[k] for k in np.sort(rng.choice(len(items), 200, replace=False))]
sfx = "_quick" if args.quick else ""
cp = [C / f"stage0_meta{sfx}.parquet", C / f"stage0_thumbs{sfx}.npy", C / f"stage0_sigs{sfx}.npz"]
data_mtime = max(os.path.getmtime(p) for p in [D / "train.csv", D / "sample_submission.csv", *D.glob("*/*.png"), Path(__file__)])
t0 = time.time()
if not args.force and all(p.exists() for p in cp) and min(os.path.getmtime(p) for p in cp) > data_mtime:
    M = pd.read_parquet(cp[0]); T = np.load(cp[1]); z = np.load(cp[2]); B32, S64, RAD = z["bil32"], z["struct64"], z["radial"]
    log(f"[cache] loaded {len(M)} images in {time.time() - t0:.1f}s")
else:
    out = Parallel(n_jobs=args.jobs)(delayed(per_image)(s, i) for s, i in items)
    M = pd.DataFrame([o[0] for o in out]); T, B32, S64, RAD = (np.stack([o[k] for o in out]) for k in range(1, 5))
    C.mkdir(exist_ok=True); M.to_parquet(cp[0]); np.save(cp[1], T); np.savez(cp[2], bil32=B32, struct64=S64, radial=RAD)
    log(f"[extract] {len(M)} images, n_jobs={args.jobs}: {time.time() - t0:.1f}s wall")
M = M.merge(tr, on="ID", how="left")
itr, ite = np.flatnonzero(M.split.values == "train"), np.flatnonzero(M.split.values == "test")
y = M.hardness.values[itr]; n = len(y)
CHEAP = [c for c in M.columns if c.startswith(("pct", "band"))] + ["mean", "std", "skew", "kurt", "entropy", "n_levels",
         "gap_bins", "sat0", "sat255", "log_lapvar", "hf_std", "grad_mean", "fsize"]

log("\n== 1. labels ==")
s = pd.read_csv(D / "train.csv", dtype={"hardness": str}).set_index("ID").hardness.loc[M.ID.values[itr]]
dec = s.str.split(".").str[1].str.len(); sig = s.str.replace(".", "", regex=False).str.lstrip("0").str.len()
R["labels"] = dict(n=n, mean=y.mean(), std=y.std(ddof=1), min=y.min(), max=y.max(), unique=int(np.unique(y).size),
                   dec_digits=[int(dec.min()), float(dec.median()), int(dec.max())], sig_digits=[int(sig.min()), int(sig.max())],
                   shortest_repr_frac=float(np.mean([repr(float(t)) == t for t in s])),
                   pandas_vs_python_parse_mismatch=int((y != s.map(float).values).sum()))
log(R["labels"])
ray = {}
for g in [1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
    Rl = abs(np.exp(2j * np.pi * np.mod(y / g, 1)).mean()); ray[str(g)] = dict(R=round(Rl, 4), p=float(np.exp(-n * Rl ** 2)))
dif = np.diff(np.sort(np.unique(y))); dmin = dif.min()
R["grid"] = dict(rayleigh=ray, dmin=dmin, dmin_expected_continuous=(y.max() - y.min()) / n ** 2,
                 rayleigh_diffs_over_dmin=round(abs(np.exp(2j * np.pi * np.mod(dif / dmin, 1)).mean()), 4))
log("grid Rayleigh R (p):", {g: f"{v['R']} ({v['p']:.2g})" for g, v in ray.items()})
log(f"min spacing {dmin:.3e} vs ~{R['grid']['dmin_expected_continuous']:.1e} expected if continuous; "
    f"Rayleigh(diffs/dmin mod 1) R={R['grid']['rayleigh_diffs_over_dmin']}")
yb, lam, ci = stats.boxcox(y, alpha=0.05)
assert abs(lam - stats.boxcox_normmax(y, method="mle")) < 1e-3
R["boxcox"] = dict(lam=lam, ci95=list(ci))
R["moments"] = {k: dict(skew=stats.skew(v), exkurt=stats.kurtosis(v), shapiro_p=stats.shapiro(v).pvalue)
                for k, v in dict(raw=y, log=np.log(y), sqrt=np.sqrt(y), boxcox=yb).items()}
log(f"Box-Cox lambda {lam:.3f} 95%CI [{ci[0]:.3f}, {ci[1]:.3f}]  (0=log, 0.5=sqrt, 1=raw) -> target-transform candidate only")
for k, v in R["moments"].items(): log(f"  {k:7s} skew {v['skew']:+.3f}  exkurt {v['exkurt']:+.3f}  shapiro p {v['shapiro_p']:.3g}")
hc, he = np.histogram(y, 20)
for c_, e in zip(hc, he): log(f"  {e:7.1f} {c_:3d} {'#' * int(round(40 * c_ / hc.max()))}")

log("\n== 2. images: range, saturation, quantization, hashes ==")
M["sat_any"] = (M.vmin == 0) | (M.vmax == 255)
R["images"] = dict(vmin=q(M.vmin, (0, .5, 1)), vmax=q(M.vmax, (0, .5, 1)), n_min0=int((M.vmin == 0).sum()),
                   n_max255=int((M.vmax == 255).sum()), sat_px_frac_mean=float((M.sat0 + M.sat255).mean()),
                   sat_px_frac_max=float((M.sat0 + M.sat255).max()), n_levels=q(M.n_levels, (0, .5, 1)),
                   gap_bins=q(M.gap_bins, (0, .5, 1)), dup_px=int(M.md5_px.duplicated(keep=False).sum()),
                   dup_file=int(M.md5_file.duplicated(keep=False).sum()),
                   reencode=int((M.groupby("md5_px").md5_file.nunique() > 1).sum()))
R["images"]["rho_y"] = {c: stats.spearmanr(M[c].values[itr], y)[0] for c in ["n_levels", "gap_bins", "sat_any"]}
log(R["images"])

log("\n== 3. PNG chunk forensics ==")
M["png_sig"] = M.layout_norm + "|" + M.ihdr + "|z" + M.zlevel.astype(str) + "|idat0=" + M.idat0.astype(str)
R["png"] = dict(signatures={k: v.to_dict() for k, v in M.groupby("png_sig").split.value_counts().unstack(fill_value=0).iterrows()},
                layouts=M.layout.value_counts().head(5).to_dict(), any_text=int(M.has_text.sum()), any_phys=int(M.has_phys.sum()),
                any_time=int(M.has_time.sum()), trailing_bytes=int(M.trailing.sum()), filt_mode=M.filt_mode.value_counts().to_dict(),
                rho_y={c: stats.spearmanr(M[c].values[itr], y)[0] for c in ["n_idat", "n_filt", "fsize"]},
                ks_test_vs_train_n_idat=stats.ks_2samp(M.n_idat.values[itr], M.n_idat.values[ite]).pvalue)
g = [y[M.png_sig.values[itr] == k] for k in np.unique(M.png_sig.values[itr])]
R["png"]["kruskal_y_by_sig_p"] = stats.kruskal(*g).pvalue if len(g) > 1 else None
log(json.dumps(R["png"], indent=1, default=str))

log("\n== 4. cheap stats: vs y, train/test KS (Bonferroni), ID structure ==")
rho = pd.Series({c: stats.spearmanr(M[c].values[itr], y)[0] for c in CHEAP}).sort_values(key=abs, ascending=False)
ks = pd.Series({c: stats.ks_2samp(M[c].values[itr], M[c].values[ite]).pvalue for c in CHEAP}).sort_values()
num = M.ID.str.extract(r"(\d+)$")[0].astype(int).values
idr = {sp: [tuple(stats.spearmanr(num[ix], M[c].values[ix])) for c in CHEAP] for sp, ix in [("train", itr), ("test", ite)]}
R["cheap"] = dict(rho_y_top8=rho.head(8).round(3).to_dict(), ks_min_p=ks.iloc[0], ks_min_feat=ks.index[0],
                  ks_n_sig_bonf=int((ks < 0.05 / len(ks)).sum()), n_feat=len(CHEAP))
R["ids"] = dict(train_range=[int(num[itr].min()), int(num[itr].max())], test_range=[int(num[ite].min()), int(num[ite].max())],
                train_gaps=int(num[itr].max() - num[itr].min() + 1 - len(itr)), test_gaps=int(num[ite].max() - num[ite].min() + 1 - len(ite)),
                id_overlap=len(set(num[itr]) & set(num[ite])), rho_id_y=stats.spearmanr(num[itr], y)[0],
                **{f"{sp}_max_abs_rho_id_stat": np.nanmax([abs(r[0]) for r in v]) for sp, v in idr.items()},
                **{f"{sp}_min_p_bonf": min(1, np.nanmin([r[1] for r in v]) * len(v)) for sp, v in idr.items()})
log(R["cheap"]); log(R["ids"])

log("\n== 5. distances, duplicates, smoothness ==")
sub = rng.choice(len(T), min(200, len(T)), replace=False); Z = zn(T[sub])
dev = np.abs(((Z[:, None] - Z[None]) ** 2).mean(-1) - dist(T[sub], T[sub])).max()
assert dev < 1e-4, dev
log(f"A1 Gram vs brute force on {len(sub)} imgs: max|dev| = {dev:.2e}")
gmd = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * np.sort(y)) / (n * (n - 1))   # analytic Gini mean difference
perm = np.mean([np.abs(y - y[rng.permutation(n)]).mean() for _ in range(1000)])
R["gmd"], R["perm1000"] = gmd, perm
log(f"random |dy| baseline: Gini mean difference {gmd:.3f}  (1000-perm avg {perm:.3f})")
A = dict(raw32=T, struct64=S64, radial=RAD); DIH = dict(raw32=False, struct64=True, radial=False)
Dm = {k: dist(A[k], A[k], DIH[k]) for k in SIGS}; Dm["bil32_old"] = dist(B32, B32)
csub = rng.choice(itr, min(100, n), replace=False); ctl = {False: [], True: []}   # positive control: photometric / +D4
for k in csub:
    a0 = np.asarray(Image.open(D / "train" / f"{M.ID[k]}.png"), np.float64)
    fl_, rot, sg, gain, off, nz = rng.random() < .5, rng.integers(4), rng.uniform(.5, 2), rng.uniform(.6, 1.4), rng.uniform(-25, 25), rng.uniform(3, 20)
    for geo in (False, True):
        a = np.rot90(a0[:, ::-1] if fl_ else a0, rot) if geo else a0
        a = ndimage.gaussian_filter(a, sg) * gain + off + rng.normal(0, nz, a.shape)
        ctl[geo].append(sigs(np.clip(np.round(a), 0, 255).astype(np.uint8)))
ctlS = {g: dict(zip(SIGS, (np.stack([c[i] for c in v]) for i in (0, 2, 3)))) for g, v in ctl.items()}
rthr = lambda d, z=5: (lambda m: m - z * 1.4826 * np.median(m - d[d < m]))(np.median(d))  # left-tail robust: median - z*MAD(lower half)
R["dist"] = {}
for k in SIGS + ["bil32_old"]:
    X = Dm[k]; np.fill_diagonal(X, np.inf)
    tt, trtr = X[np.ix_(ite, itr)], X[np.ix_(itr, itr)]
    nn, j = trtr.min(1), trtr.argmin(1); dy = np.abs(y - y[j])
    close = nn <= np.quantile(nn, .05); bs = [rng.choice(dy[close], close.sum()).mean() for _ in range(2000)]
    r = dict(test_train_nn=q(tt.min(1)), train_train_nn=q(nn), closest5pct_dy=dy[close].mean(),
             closest5pct_dy_ci95=np.percentile(bs, [2.5, 97.5]).tolist(), rho_nn_dy=stats.spearmanr(nn, dy)[0],
             dy_by_nn_quintile=[round(dy[b].mean(), 2) for b in np.array_split(np.argsort(nn), 5)])
    if k in SIGS:
        iu = np.triu_indices(n, 1); thr_tt, thr_trtr = rthr(tt.ravel()), rthr(trtr[iu])
        cd = np.diag(dist(ctlS[DIH[k]][k], A[k][csub], DIH[k])); fl = trtr[iu] < thr_trtr
        r.update(null_thr_test_train=thr_tt, null_thr_train_train=thr_trtr, n_test_train_flagged=int((tt < thr_tt).sum()),
                 n_train_train_flagged=int(fl.sum()), dy_flagged_train_pairs=float(np.abs(y[iu[0][fl]] - y[iu[1][fl]]).mean()) if fl.any() else None,
                 ctl_family="photometric+D4" if DIH[k] else "photometric", ctl_dist=q(cd, (.5, .99)),
                 ctl_power=float((cd < thr_tt).mean()), ctl_valid=bool((cd < thr_tt).mean() >= .9))   # control = power check only
        if r["n_test_train_flagged"]:
            log(f"!!!!! LEAK ALERT [{k}]: {r['n_test_train_flagged']} test-train pairs > 5 robust sigma below bulk (power {r['ctl_power']:.2f}) !!!!!")
    R["dist"][k] = r
    log(f"[{k}]", json.dumps(r, default=float))

log("\n== 6. baseline RMSE anchors: RepeatedKFold(5x3, seed 42) ==")
rkf = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(y))
cv = lambda fn: (lambda r: dict(mean=np.mean(r), std=np.std(r)))([np.sqrt(np.mean((fn(a, b) - y[b]) ** 2)) for a, b in rkf])
F = M[CHEAP].values[itr]
R["baseline"] = dict(const=cv(lambda a, b: np.full(len(b), y[a].mean())),
                     ridge_cheap=cv(lambda a, b: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 4, 29))).fit(F[a], y[a]).predict(F[b])))
for k in SIGS:
    Dt = Dm[k][np.ix_(itr, itr)].copy(); np.fill_diagonal(Dt, 0)
    R["baseline"][f"knn5_{k}"] = cv(lambda a, b: KNeighborsRegressor(5, metric="precomputed").fit(Dt[np.ix_(a, a)], y[a]).predict(Dt[np.ix_(b, a)]))
for k, v in R["baseline"].items(): log(f"  {k:16s} RMSE {v['mean']:.3f} +/- {v['std']:.3f}")

REP.mkdir(exist_ok=True); conv = lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))
(REP / f"stage0{sfx}.json").write_text(json.dumps(R, indent=1, default=conv, sort_keys=True))
(REP / f"stage0{sfx}.md").write_text("# Stage 0 report (auto-generated by src/stage0_recon.py)\n\n```\n" + "\n".join(L) + "\n```\n")
log(f"\n[done] total {time.time() - t0:.1f}s -> reports/stage0{sfx}.json/.md")
