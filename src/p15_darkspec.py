"""Phase 15 / A+B: mode-relative darkness spectrum (continuous rule of mixtures) and a direct pore re-measurement.

A  For each image, smooth (sigma 1/2/3), take the KDE mode of the pixels as the matrix level and histogram the offsets
   (pixel - mode) in 5-gray-level bins from -90 to +45. Ridge on these bins learns how much each darkness band adds to y;
   the coefficient curve is the data-driven "hardness per phase gray level".
B  Pores as very dark compact blobs: pixels below mode - {45,55,65} on the sigma-1 image, components >= 8 px.
No labels used here; cache/p15/darkspec.parquet for all 1500 images.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, D, load_meta

EDGES = np.arange(-90, 46, 5)


def spec(split, id_):
    f = np.asarray(Image.open(D / split / f"{id_}.png")).astype(np.float64); r = {"ID": id_}
    for s in (1, 2, 3):
        g = ndi.gaussian_filter(f, s); v = g[::2, ::2].ravel(); gr = np.linspace(v.min(), v.max(), 600)
        mode = gr[stats.gaussian_kde(v[::3], bw_method=.08)(gr).argmax()]; h = np.histogram(v - mode, EDGES)[0] / v.size
        r[f"s{s}_mode"] = mode; r[f"s{s}_below"] = float(np.mean(v - mode < EDGES[0])); r[f"s{s}_above"] = float(np.mean(v - mode >= EDGES[-1]))
        for lo, val in zip(EDGES[:-1], h): r[f"s{s}_b{lo:+d}"] = val
        if s == 1:
            for cut in (45, 55, 65):
                lab, nc = ndi.label(g < mode - cut); sz = np.bincount(lab.ravel())[1:]; sz = sz[sz >= 8]
                r[f"pore{cut}_n"], r[f"pore{cut}_frac"] = len(sz), float(sz.sum() / f.size); r[f"pore{cut}_mean"] = float(sz.mean()) if len(sz) else 0.0
    return r


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); (C / "p15").mkdir(exist_ok=True)
    import p15_darkspec as me
    items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
    pd.DataFrame(Parallel(n_jobs=6)(delayed(me.spec)(s, i) for s, i in items)).to_parquet(C / "p15" / "darkspec.parquet"); print("done")
