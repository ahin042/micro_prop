"""Stage 1: nuisance-aware physical features -> cache/features.parquet (joblib-parallel, cached).

Prefixes: q_ = image-quality nuisance, r_ = estimator on raw image, n_ = same estimator on the
quality-normalized image (percentile stretch + noise-adaptive NLM). Segmentation-based estimators
(intercept, watershed, granulometry, phases, pores) need denoising, so they exist only on n_.
Part B (blur-corrected): m_ = MTF-corrected spectrum, d_ = segmentation chain on the Richardson-Lucy
deconvolved image, x_ = scale-space extrapolation of grain size to zero blur.
"""
import argparse, os, time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, optimize, stats
from skimage import feature, filters, measure, restoration, segmentation
from skimage.morphology import skeletonize

from common import C, D

IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)       # Immerkaer noise operator, L2 norm = 6
S1_PREFIX, B_PREFIX = ("q_", "r_", "n_", "b_"), ("m_", "d_", "x_")
SADD = [0.5, 1.0, 1.5, 2.0]                                          # B3 added-blur levels (0 = the n_ chain)


@lru_cache(None)
def grids(n):
    """Hann window, integer radius map, ring counts and k = 1..n/2-1 for an n x n image."""
    ri = np.hypot(*(np.indices((n, n)) - n // 2)).astype(int)
    return np.outer(np.hanning(n), np.hanning(n)), ri, np.bincount(ri.ravel()), np.arange(1, n // 2)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return x * x + y * y <= r * r


def drop_small(m, k):
    lab, n = ndi.label(m); keep = np.bincount(lab.ravel()) >= k; keep[0] = False
    return keep[lab]


def radial(P):
    _, ri, rc, _ = grids(P.shape[0]); h = P.shape[0] // 2
    return np.bincount(ri.ravel(), P.ravel())[:h] / rc[:h]


def noise_sigma(f):
    """MAD of the Immerkaer residual restricted to the flattest 30% of pixels (grain interiors)."""
    g = ndi.gaussian_gradient_magnitude(f, 2); v = ndi.convolve(f, IMM, mode="reflect")[g < np.percentile(g, 30)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) / 6


def spec(f, sn, p):
    """Windowed radial power spectrum with the white-noise floor removed."""
    n = f.shape[0]; H, _, _, RR = grids(n)
    P = np.abs(np.fft.fftshift(np.fft.fft2((f - f.mean()) * H))) ** 2
    ps = np.clip(radial(P)[1:] - sn ** 2 * (H ** 2).sum(), 1e-12, None); w = ps * RR
    cum = np.cumsum(w) / w.sum(); pk = ps.max()
    o = {f"{p}spec_r50": RR[np.searchsorted(cum, .5)], f"{p}spec_r90": RR[min(np.searchsorted(cum, .9), len(RR) - 1)],
         f"{p}spec_centroid": (RR * w).sum() / w.sum(), f"{p}spec_peak_wl": n / RR[np.argmax(ps * RR ** 2)]}
    for fr in (1e-2, 1e-3):
        o[f"{p}blur_r{fr:g}"] = RR[np.argmax(ps < fr * pk)] if (ps < fr * pk).any() else n // 2
    return o


def mtf(f, sn, K=40):
    """B1: fit log P = log A - beta log k - c k^2 on the SNR>2 band (k>=4), divide out exp(-c k^2), recompute
    spectral grain-size stats on a FIXED band k<=K; plus a knee model A/(1+(k/kc)^2)^g * exp(-c k^2)."""
    n = f.shape[0]; H, _, _, RR = grids(n); fl = sn ** 2 * (H ** 2).sum()
    ps = radial(np.abs(np.fft.fftshift(np.fft.fft2((f - f.mean()) * H))) ** 2)[1:] - fl
    ok = (RR >= 4) & (ps > 2 * fl) & (RR <= .8 * n / 2)
    if ok.sum() < 6: ok = (RR >= 4) & (RR <= n // 4) & (ps > 0)
    Z = np.column_stack([np.ones(ok.sum()), -np.log(RR[ok]), -RR[ok] ** 2.0])
    la, beta, c = np.linalg.lstsq(Z, np.log(np.clip(ps[ok], 1e-12, None)), rcond=None)[0]; c = max(c, 0.0)
    sig = np.sqrt(c) * n / (2 * np.pi)                                  # power MTF exp(-4 pi^2 s^2 (k/n)^2)
    kk = RR[:K]; pc = np.clip(ps[:K], 1e-12, None) * np.exp(np.minimum(c * kk ** 2.0, np.log(1e3))); w = pc * kk
    cum = np.cumsum(w) / w.sum()
    o = {"m_sigma_px": sig, "m_beta": beta, "m_logA": la, "m_spec_r50": kk[np.searchsorted(cum, .5)],
         "m_spec_centroid": (kk * w).sum() / w.sum(), "m_spec_peak_wl": n / kk[np.argmax(pc * kk ** 2)]}
    fit = (RR <= .8 * n / 2) & (ps > 2 * fl); fit[:2] = True
    lp = np.log(np.clip(ps[fit], 1e-12, None)); kf = RR[fit].astype(float)
    res = lambda q: q[0] - q[2] * np.log1p((kf / np.exp(q[1])) ** 2) - q[3] * kf ** 2 - lp
    q = optimize.least_squares(res, [lp[0], np.log(8.0), 1.5, max(c, 1e-5)],
                               bounds=([-np.inf, 0, .3, 0], [np.inf, np.log(n), 4, .05]), loss="soft_l1").x
    o.update(m_knee_wl=n / np.exp(q[1]), m_knee_gamma=q[2], m_knee_sigma_px=np.sqrt(q[3]) * n / (2 * np.pi))
    return o


def acf(f, p):
    """Linear autocorrelation: radial decay lengths + anisotropy of the central >0.5 lobe."""
    N = f.shape[0]; x = f - f.mean(); A = np.real(np.fft.ifft2(np.abs(np.fft.fft2(x, s=(2 * N, 2 * N))) ** 2))
    A = np.fft.fftshift(A)[N // 2:N // 2 + N, N // 2:N // 2 + N]
    A = A / radial(A)[1]; ra = radial(A)                # normalize at lag 1: white noise only inflates lag 0
    o = {f"{p}acf_len{t}": float(np.interp(-t_, -ra[:N // 4], np.arange(N // 4))) for t, t_ in (("50", .5), ("37", 1 / np.e))}
    lab, _ = ndi.label(A > .5); m = lab == lab[N // 2, N // 2]
    if m.sum() < 5: m = np.zeros_like(m); m[N // 2 - 1:N // 2 + 2, N // 2 - 1:N // 2 + 2] = True
    yy, xx = np.nonzero(m); w = np.clip(A[m], 1e-6, None); yy = yy - N // 2; xx = xx - N // 2
    cov = np.cov(np.vstack([xx, yy]), aweights=w); ev, evec = np.linalg.eigh(cov)
    th = np.arctan2(evec[1, 1], evec[0, 1])
    o.update({f"{p}acf_aspect": np.sqrt(ev[1] / max(ev[0], 1e-9)), f"{p}acf_area": m.sum(),
              f"{p}acf_cos2": abs(np.cos(2 * th)), f"{p}acf_sin2": abs(np.sin(2 * th))})
    return o


def tensor(f, p):
    Arr, Arc, Acc = feature.structure_tensor(f, sigma=2, order="rc")
    J = [Arr.mean(), Arc.mean(), Acc.mean()]; th = 0.5 * np.arctan2(2 * J[1], J[0] - J[2])
    gy, gx = np.gradient(f); m = np.hypot(gx, gy)
    return {f"{p}st_coh": np.hypot(J[0] - J[2], 2 * J[1]) / (J[0] + J[2]),
            f"{p}st_coh_local": np.average(np.hypot(Arr - Acc, 2 * Arc) / (Arr + Acc + 1e-12), weights=Arr + Acc),
            f"{p}gori_R": np.abs((m * np.exp(2j * np.arctan2(gy, gx))).sum()) / m.sum(),
            f"{p}st_cos2": abs(np.cos(2 * th)), f"{p}st_sin2": abs(np.sin(2 * th))}


def mli(bm):
    """ASTM E112 mean linear intercept over 8 directions on the boundary mask."""
    Ls, th = [], np.arange(8) * 22.5
    for a in th:
        b = ndi.rotate(bm.astype(np.uint8), a, order=0, reshape=True); v = ndi.rotate(np.ones_like(bm, np.uint8), a, order=0, reshape=True)
        b = b & v; cross = ((b[:, 1:] == 1) & (b[:, :-1] == 0) & (v[:, :-1] == 1)).sum()
        Ls.append(v.sum() / max(cross, 1))
    Ls = np.array(Ls); amax = np.deg2rad(th[Ls.argmax()])
    return {"n_mli": len(th) / np.sum(1 / Ls), "n_mli_aniso": Ls.max() / Ls.min(),
            "n_mli_cos2": abs(np.cos(2 * amax)), "n_mli_sin2": abs(np.sin(2 * amax))}


def granulometry(interior):
    """Binary openings by disks r=1..20 via two EDTs each; pattern spectrum moments."""
    A = [interior.sum()]; dt = ndi.distance_transform_edt(interior)
    for r in range(1, 21):
        er = dt > r; A.append(int((ndi.distance_transform_edt(~er) <= r).sum()) if er.any() else 0)
    A = np.array(A, float); ps = np.clip(-np.diff(A), 0, None); r = np.arange(1, 21); s = ps.sum() + 1e-9
    o = {"n_gran_mean": (r * ps).sum() / s, "n_gran_median": r[np.searchsorted(np.cumsum(ps) / s, .5)],
         "n_gran_std": np.sqrt((ps * (r - (r * ps).sum() / s) ** 2).sum() / s), "n_gran_resid20": A[-1] / max(A[0], 1)}
    o.update({f"n_ps{lo}_{hi}": ps[lo - 1:hi].sum() / s for lo, hi in ((1, 2), (3, 4), (5, 7), (8, 12), (13, 20))})
    return o


def segment(g):
    R = filters.sato(g, sigmas=[1, 2], black_ridges=True)
    bm = drop_small(R > filters.threshold_otsu(R), 10)
    interior = ~ndi.binary_dilation(bm, disk(1))
    mk, _ = ndi.label(drop_small(interior, 8))
    return R, bm, interior, segmentation.watershed(R, mk)


def phases(g, bm, interior, lab):
    o, t1, t2 = {}, *filters.threshold_multiotsu(g, 3)
    for k, m in enumerate([g < t1, (g >= t1) & (g < t2), g >= t2]): o[f"n_otsu_c{k}"] = m.mean()
    # pores: compact dark blobs that are NOT a whole grain. After stretching, dark-phase grains are as dark as pores,
    # so use topology: a pore covers <50% of its watershed cell, or is its own cell with <=2 neighbours (enclosed).
    cand = ndi.binary_opening(g < t1, disk(1)); pl, _ = ndi.label(cand); pores = np.zeros_like(cand); pp = []
    amed = np.median(np.bincount(lab.ravel())[1:])                   # pores must be < half a typical grain
    for r in measure.regionprops(pl):
        if not (16 <= r.area <= .5 * amed and r.solidity >= .8 and r.axis_minor_length >= 3 and 4 * np.pi * r.area / max(r.perimeter, 1) ** 2 >= .5):
            continue
        sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in r.slice); c = pl[sl] == r.label; L = lab[sl]
        k = np.bincount(L[c]).argmax(); cell = lab == k; cov = (c & (L == k)).sum() / cell.sum()
        nb = np.setdiff1d(np.unique(lab[ndi.binary_dilation(cell, disk(2))]), [k]).size
        if cov < .5 or nb <= 2: pp.append(r); pores[sl][c] = True
    eq = np.array([r.equivalent_diameter_area for r in pp]) if pp else np.zeros(1)
    o.update(n_pore_frac=pores.mean(), n_pore_count=len(pp), n_pore_eqd_mean=eq.mean(), n_pore_eqd_med=np.median(eq),
             n_pore_eqd_std=eq.std(), n_pore_aspect=np.mean([r.axis_major_length / max(r.axis_minor_length, 1) for r in pp]) if pp else 0)
    # dark phase: Otsu split of per-grain interior means (pores and boundaries excluded)
    inner = interior & ~ndi.binary_dilation(pores, disk(1))
    idx = np.unique(lab[inner]); area = ndi.sum(np.ones_like(g), lab, idx); mu = ndi.mean(g, lab * inner, idx)
    ok = np.isfinite(mu) & (area >= 8); mu, area = mu[ok], area[ok]
    if len(mu) >= 4:
        t = filters.threshold_otsu(mu); dk = mu < t
        sep = (mu[~dk].mean() - mu[dk].mean()) / (np.sqrt((mu[dk].var() * dk.sum() + mu[~dk].var() * (~dk).sum()) / len(mu)) + 1e-9) if dk.any() and (~dk).any() else 0
        o.update(n_dark_frac=area[dk].sum() / area.sum(), n_dark_sep=sep, n_dark_t1_frac=area[mu < t1].sum() / area.sum(),
                 n_grain_mu_std=mu.std())
    else:
        o.update(n_dark_frac=0, n_dark_sep=0, n_dark_t1_frac=0, n_grain_mu_std=0)
    return o, pores


def grains(lab):
    rp = [r for r in measure.regionprops(lab) if r.area >= 8]; N = lab.shape[0]
    bd = np.array([r.bbox[0] == 0 or r.bbox[1] == 0 or r.bbox[2] == N or r.bbox[3] == N for r in rp])
    inn = [r for r, b in zip(rp, bd) if not b] or rp
    eq = np.array([r.equivalent_diameter_area for r in inn]); ar = np.array([r.area for r in inn], float)
    asp = np.array([r.axis_major_length / max(r.axis_minor_length, 1) for r in inn])
    th = np.angle(np.sum(ar * np.exp(2j * np.array([r.orientation for r in inn]))))  # doubled-angle mean
    NA = ((~bd).sum() + 0.5 * bd.sum()) / N ** 2                                       # Jeffries planimetric
    return {"n_ws_count": len(rp), "n_ws_d_planimetric": 1 / np.sqrt(NA), "n_ws_eqd_mean": eq.mean(), "n_ws_eqd_med": np.median(eq),
            "n_ws_eqd_std": eq.std(), "n_ws_eqd_skew": stats.skew(eq) if len(eq) > 2 else 0, "n_ws_eqd_aw": np.average(eq, weights=ar),
            "n_ws_minor_mean": np.mean([r.axis_minor_length for r in inn]), "n_ws_aspect_mean": asp.mean(), "n_ws_aspect_med": np.median(asp),
            "n_ws_aspect_aw": np.average(asp, weights=ar), "n_ws_cos2": abs(np.cos(th)), "n_ws_sin2": abs(np.sin(th)),
            "n_ws_solidity": np.mean([r.solidity for r in inn])}


def texture(g):
    q8 = np.clip(g * 31, 0, 31).astype(np.uint8)
    M = feature.graycomatrix(q8, [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    o = {f"n_glcm_{p}_d{d}": v for p in ("contrast", "homogeneity", "energy", "correlation")
         for d, v in zip((1, 3, 5), feature.graycoprops(M, p).mean(1))}
    h = np.bincount(feature.local_binary_pattern(q8, 8, 1, "uniform").astype(int).ravel(), minlength=10) / q8.size
    o.update({f"n_lbp{k}": h[k] for k in range(10)})
    return o


def seg_chain(g, p="n_", full=True):
    """Segmentation-based estimators on a [0,1] image; keys renamed n_ -> p."""
    R, bm, interior, lab = segment(g)
    o = dict(n_bnd_frac=bm.mean(), n_bnd_len=skeletonize(bm).sum() / bm.size, n_ridge_mean=R.mean())
    o.update(mli(bm)); o.update(granulometry(interior)); o.update(grains(lab)); pores = None
    if full: ph, pores = phases(g, bm, interior, lab); o.update(ph)
    return {p + k[2:]: v for k, v in o.items()}, R, bm, lab, pores


def blur_fixes(g, sig, o):
    """B2 deconvolve (Richardson-Lucy, Gaussian PSF from the MTF fit) + B3 scale-space extrapolation to zero blur."""
    s = float(np.clip(sig, 0, 4)); out = {}
    if s >= .3:
        r = int(np.ceil(3 * s)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s)); psf /= psf.sum()
        gd = restoration.richardson_lucy(g, psf, num_iter=20, clip=True)
    else:
        gd = g
    out.update(seg_chain(gd, "d_")[0])
    E = {k: [o[f"n_{k}"]] for k in ("ws_eqd_mean", "mli", "gran_mean", "ws_minor_mean")}
    for sa in SADD:
        e = seg_chain(ndi.gaussian_filter(g, sa), "n_", full=False)[0]
        for k in E: E[k].append(e[f"n_{k}"])
    s2 = s ** 2 + np.array([0] + SADD) ** 2                               # total blur variance per level
    for k, v in E.items():
        b, a = np.polyfit(s2, np.array(v, float), 1); out[f"x_{k}0"] = a; out[f"x_{k}_slope"] = b
    return out


def extract(split, id_, masks=False, ext=True):
    return extract_array(np.asarray(Image.open(D / split / f"{id_}.png")), id_, masks, ext)


def extract_array(a, id_=None, masks=False, ext=True):
    f = a.astype(np.float64)
    p1, p5, p95, p99 = np.percentile(f, [1, 5, 95, 99]); sn = noise_sigma(f)
    o = {"ID": id_, "q_noise": sn, "q_lapvar": np.log(ndi.laplace(f).var()), "q_span99": p99 - p1, "q_span95": p95 - p5,
         "q_snr": (p99 - p1) / max(sn, 1e-3), "q_lapvar_dn": np.log(max(ndi.laplace(f).var() - 20 * sn ** 2, 1e-3))}
    g0 = np.clip((f - p1) / max(p99 - p1, 1), 0, 1); sg = sn / max(p99 - p1, 1)
    g = restoration.denoise_nl_means(g0, h=0.8 * sg, sigma=sg, patch_size=5, patch_distance=6, fast_mode=True)
    o["q_crete_blur"] = (lambda dv, db: (dv.sum() - np.clip(dv - db, 0, None).sum()) / dv.sum())(
        np.abs(np.diff(g, axis=0)), np.abs(np.diff(ndi.uniform_filter1d(g, 9, axis=0), axis=0)))
    for p, img, s in (("r_", f, sn), ("n_", g, 0.0)):
        o.update(spec(img, s, p)); o.update(acf(img, p)); o.update(tensor(img, p))
        o[f"{p}canny"] = feature.canny(img, sigma=2 if p == "r_" else 1).mean()
    so, R, bm, lab, pores = seg_chain(g); o.update(so); o.update(spec(R, 0, "b_"))
    o.update(texture(g))
    if ext:
        o.update(mtf(f, sn)); o.update(blur_fixes(g, o["m_sigma_px"], o))
    return (o, dict(g=g, bm=bm, pores=pores, lab=lab)) if masks else o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--n", type=int, default=0, help="debug: first n images only")
    args = ap.parse_args()
    out = C / "features.parquet"
    items = [("train", i) for i in pd.read_csv(D / "train.csv").ID] + [("test", i) for i in pd.read_csv(D / "sample_submission.csv").ID]
    items = items[:args.n] if args.n else items
    newest = max(os.path.getmtime(p) for p in [*D.glob("*/*.png"), Path(__file__)])
    if out.exists() and not args.force and not args.n and os.path.getmtime(out) > newest:
        print("[cache] features.parquet is up to date"); raise SystemExit
    t0 = time.time()
    import features as _f                                            # workers pickle by reference (lru_cache in __main__ is not picklable)
    F = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(_f.extract)(s, i) for s, i in items))
    F["split"] = [s for s, _ in items]
    print(f"[extract] {len(F)} images x {F.shape[1] - 2} features in {time.time() - t0:.1f}s; NaN cols: {F.columns[F.isna().any()].tolist()}")
    if not args.n: F.to_parquet(out)
