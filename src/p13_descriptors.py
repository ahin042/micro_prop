"""Phase 13: one segmentation pass -> candidate latent descriptors for all 1500 images (cache/p13/desc.parquet).

Dark-phase fraction in three unit systems (the task varies contrast deliberately, so absolute gray thresholds are
confounded with contrast):
  abs  : grain mean < mode - g            (g in gray levels)
  mad  : grain mean < mode - k * MAD      (k in units of the matrix-peak spread)
  rng  : grain mean < mode - a * range    (a = fraction of the matrix-to-pore intensity range; scales with contrast)
  vly  : KDE valley between the matrix mode and the darkest secondary peak (data-driven separation)
plus phase count, per-phase brightness/spread, dark-particle size/connectivity/clustering (Clark-Evans), boundary
brightness/width, intra-grain texture, pores, grain-size distribution tails, aspect, alignment. Uses NO labels.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage import measure

from common import C, D, load_meta
from features import disk, extract_array, noise_sigma

ABS, MADK, RNG = (10, 15, 20, 25, 30), (2, 3, 4, 5, 6), (.08, .12, .16, .20, .25, .30)


def desc(split, id_):
    return desc_array(np.asarray(Image.open(D / split / f"{id_}.png")), id_)


def desc_array(a, id_=None):
    f = a.astype(np.float64); sn = noise_sigma(f)
    o, m = extract_array(a, id_, masks=True, ext=False); lab, bm, pores = m["lab"], m["bm"], m["pores"]
    r = {"ID": id_}
    inner = ~ndi.binary_dilation(bm | pores, disk(2))
    rp = [p for p in measure.regionprops(lab, intensity_image=f) if p.area >= 20]
    idx = np.array([p.label for p in rp]); ain = ndi.sum(inner, lab, idx) if len(idx) else np.zeros(0)
    keep = ain >= 30; rp = [p for p, k in zip(rp, keep) if k]; idx = idx[keep]
    r["d_ngr"] = len(rp)
    pore_lvl = np.median(f[pores]) if pores.sum() >= 10 else np.percentile(ndi.gaussian_filter(f, 1), .5)
    if len(rp) >= 6:
        mu = ndi.mean(f, lab * inner, idx); var = ndi.variance(f, lab * inner, idx); ar = np.array([p.area for p in rp], float)
        eqd = np.array([p.equivalent_diameter_area for p in rp]); asp = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in rp])
        cen = np.array([p.centroid for p in rp]); ori = np.array([p.orientation for p in rp])
        grid = np.linspace(mu.min() - 8, mu.max() + 8, 512); kde = stats.gaussian_kde(mu, bw_method=.12, weights=ar)(grid)
        mode = grid[kde.argmax()]; near = np.abs(mu - mode) < 10
        mad = max(1.4826 * np.median(np.abs(mu[near] - mode)) if near.sum() >= 3 else 3.0, 1.0); rng = max(mode - pore_lvl, 5.0); tot = ar.sum()
        r.update(d_mode=mode, d_mad=mad, d_range=rng, d_pore_lvl=pore_lvl, d_mad_rel=mad / rng)
        for g in ABS: r[f"d_fd_abs{g}"] = ar[mu < mode - g].sum() / tot
        for k in MADK: r[f"d_fd_mad{k}"] = ar[mu < mode - k * mad].sum() / tot
        for q in RNG: r[f"d_fd_rng{int(q * 100)}"] = ar[mu < mode - q * rng].sum() / tot
        pk, pr = find_peaks(kde, prominence=kde.max() * .05)
        r["d_npeaks"] = len(pk); r["d_3phase"] = float(len(pk) >= 3)
        lower = pk[grid[pk] < mode - max(2 * mad, 4)]
        if len(lower):
            dp = lower[np.argmax(kde[lower])]; seg = slice(dp, kde.argmax()); vly = grid[seg][np.argmin(kde[seg])] if dp < kde.argmax() else mode - 3 * mad
            r["d_fd_vly"] = ar[mu < vly].sum() / tot; r["d_dark_peak_rel"] = (mode - grid[dp]) / rng
        else:
            r["d_fd_vly"] = 0.0; r["d_dark_peak_rel"] = 0.0
        dk = mu < mode - .16 * rng                                          # contrast-normalized dark-phase definition
        r["d_bright_rel"] = (np.average(mu[mu > mode + 2 * mad], weights=ar[mu > mode + 2 * mad]) - mode) / rng if (mu > mode + 2 * mad).any() else 0.0
        r["d_matrix_sd_rel"] = np.sqrt(np.cov(mu[~dk], aweights=ar[~dk])) / rng if (~dk).sum() > 2 else 0.0
        r["d_dark_sd_rel"] = np.sqrt(np.cov(mu[dk], aweights=ar[dk])) / rng if dk.sum() > 2 else 0.0
        r["d_dark_lvl_rel"] = (mode - np.average(mu[dk], weights=ar[dk])) / rng if dk.any() else 0.0
        r["d_dark_eqd_ratio"] = eqd[dk].mean() / eqd[~dk].mean() if dk.any() and (~dk).any() else 1.0
        r["d_dark_asp_ratio"] = asp[dk].mean() / asp[~dk].mean() if dk.any() and (~dk).any() else 1.0
        if dk.sum() >= 3:
            dmask = np.isin(lab, idx[dk]); dmask = ndi.binary_closing(dmask, disk(2)); ncomp = ndi.label(dmask)[1]
            r["d_dark_conn"] = 1 - ncomp / dk.sum()                          # 0 = isolated dark grains, ->1 = clustered
            dd = cKDTree(cen[dk]).query(cen[dk], 2)[0][:, 1]; dens = dk.sum() / lab.size
            r["d_dark_clark_evans"] = dd.mean() / (0.5 / np.sqrt(dens))      # <1 clustered, 1 random, >1 dispersed
        else:
            r["d_dark_conn"], r["d_dark_clark_evans"] = 0.0, 1.0
        r["d_intra_tex_rel"] = np.sqrt(max(np.average(var, weights=ar) - sn ** 2, 0)) / rng   # within-grain sub-structure beyond noise
        r.update(d_eqd_mean=eqd.mean(), d_eqd_cv=eqd.std() / eqd.mean(), d_eqd_skew=stats.skew(eqd), d_eqd_p90p10=np.percentile(eqd, 90) / np.percentile(eqd, 10),
                 d_eqd_tail=float((eqd > 2 * np.median(eqd)).mean()), d_eqd_logsd=np.log(eqd).std(), d_area_cv=ar.std() / ar.mean(),
                 d_asp_aw=np.average(asp, weights=ar), d_asp_p90=np.percentile(asp, 90),
                 d_align=np.abs(np.average(np.exp(2j * ori), weights=ar * (asp - 1 + 1e-6))))
    sk = bm & ~ndi.binary_erosion(bm)
    r["d_gb_lvl_rel"] = (r.get("d_mode", np.median(f)) - np.median(f[bm])) / r.get("d_range", 50.0) if bm.any() else 0.0
    from skimage.morphology import skeletonize
    sl = skeletonize(bm).sum(); r["d_gb_width"] = bm.sum() / max(sl, 1); r["d_gb_density"] = sl / lab.size
    pl = measure.label(pores); pr_ = measure.regionprops(pl)
    pe = np.array([p.equivalent_diameter_area for p in pr_]) if pr_ else np.zeros(0)
    r.update(d_pore_n=len(pr_), d_pore_frac=pores.mean(), d_pore_eqd=pe.mean() if len(pe) else 0.0, d_pore_eqd_sd=pe.std() if len(pe) > 1 else 0.0)
    g2 = ndi.gaussian_filter(f, 2); v = g2[::2, ::2].ravel(); gr = np.linspace(v.min(), v.max(), 512)
    md = gr[stats.gaussian_kde(v[::4], bw_method=.1)(gr).argmax()]; rg2 = max(md - np.percentile(v, .5), 5.0)
    sdu = np.sqrt(np.mean((v[v >= md] - md) ** 2))
    for g in ABS: r[f"p_fd_abs{g}"] = np.mean((v < md - g) & (v > md - 70))
    for k in (2, 3, 4): r[f"p_fd_sd{k}"] = np.mean((v < md - k * sdu) & (v > md - .8 * rg2))
    for q in RNG: r[f"p_fd_rng{int(q * 100)}"] = np.mean((v < md - q * rg2) & (v > md - .8 * rg2))
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); (C / "p13").mkdir(exist_ok=True); out = C / "p13" / "desc.parquet"
    items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
    import p13_descriptors as me                                          # workers pickle by reference
    pd.DataFrame(Parallel(n_jobs=8)(delayed(me.desc)(s, i) for s, i in items)).to_parquet(out); print("wrote", out)
