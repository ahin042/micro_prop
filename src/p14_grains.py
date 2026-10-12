"""Phase 14 / Steps 1-3: per-grain table and per-phase Hall-Petch / rule-of-mixtures aggregates (CPU only, no labels).

Step 1  cache/p14/grains.parquet : one row per grain (area, eqd, minor/major, aspect, RAW mean gray with boundary and pore
        pixels excluded, within-grain std, neighbour count, touches-border flag). No per-image normalization.
Step 2  phase assignment per image, three schemes:
          km2 / km3 : k-means on grain mean gray, weighted by grain area; phases sorted bright (p0 = matrix) -> dark
          bins      : fixed offsets from the matrix mode (KDE of area-weighted grain means):
                      p0 >= mode-10 > p1 >= mode-20 > p2 >= mode-30 > p3
Step 3  cache/p14/agg.parquet : per image, scheme, phase (area-weighted): f_p, d_p^-0.5, d_p^-1, minor_p^-0.5, count, aspect,
        products f_p*d_p^-0.5 (zero when the phase is absent), plus global sum_p f_p*d_p^-0.5 and sum_p f_p*(1+aspect_p).
        Every aggregate is computed with all grains ("_all") and without border grains ("_inn").
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats
from skimage import measure
from sklearn.cluster import KMeans

from common import C, D, load_meta
from features import disk, extract_array

N = 256


def grain_table(a, id_):
    """Step 1 on a uint8 array."""
    f = a.astype(np.float64); _, m = extract_array(a, id_, masks=True, ext=False); lab, bm, pores = m["lab"], m["bm"], m["pores"]
    inner = ~ndi.binary_dilation(bm | pores, disk(2))
    pairs = np.vstack([np.c_[lab[:, :-1].ravel(), lab[:, 1:].ravel()], np.c_[lab[:-1].ravel(), lab[1:].ravel()]])
    pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], 1), axis=0)
    nb = np.bincount(pairs.ravel(), minlength=lab.max() + 1)
    rows = []
    for p in measure.regionprops(lab):
        if p.area < 20: continue
        sl = p.slice; msk = (lab[sl] == p.label) & inner[sl]
        if msk.sum() < 15: continue
        v = f[sl][msk]
        rows.append((id_, p.label, p.area, p.equivalent_diameter_area, p.axis_minor_length, p.axis_major_length,
                     p.axis_major_length / max(p.axis_minor_length, 1), v.mean(), v.std(), int(nb[p.label]),
                     bool(p.bbox[0] == 0 or p.bbox[1] == 0 or p.bbox[2] == N or p.bbox[3] == N)))
    return pd.DataFrame(rows, columns=["ID", "label", "area", "eqd", "minor", "major", "aspect", "gray", "gray_sd", "n_nb", "border"])


def phases(g, scheme):
    """Phase index per grain (0 = brightest / matrix)."""
    if scheme in ("km2", "km3"):
        k = int(scheme[-1]); km = KMeans(k, n_init=10, random_state=0).fit(g.gray.values[:, None], sample_weight=g.area.values)
        rank = np.argsort(np.argsort(-km.cluster_centers_[:, 0])); return rank[km.labels_]
    grid = np.linspace(g.gray.min() - 8, g.gray.max() + 8, 400)
    mode = grid[stats.gaussian_kde(g.gray, bw_method=.12, weights=g.area)(grid).argmax()]
    return np.digitize(mode - g.gray.values, [10, 20, 30])


def aggregate(g, id_):
    """Step 3 for one image's grain table."""
    r = {"ID": id_}
    for sub, gg in (("all", g), ("inn", g[~g.border])):
        if len(gg) < 6: continue
        for sch in ("km2", "km3", "bins"):
            ph = phases(gg, sch); A = gg.area.values; tot = A.sum(); P = 4 if sch == "bins" else int(sch[-1]); hp, ra = 0.0, 0.0
            for p in range(P):
                k = ph == p; t = f"{sch}_{sub}_p{p}"; fp = A[k].sum() / tot; r[f"{t}_f"] = fp; r[f"{t}_n"] = int(k.sum())
                if k.any():
                    w = A[k]; d = gg.eqd.values[k]
                    r[f"{t}_dm05"] = np.average(d ** -.5, weights=w); r[f"{t}_dm1"] = np.average(d ** -1, weights=w)
                    r[f"{t}_minm05"] = np.average(np.maximum(gg.minor.values[k], 1) ** -.5, weights=w); r[f"{t}_asp"] = np.average(gg.aspect.values[k], weights=w)
                    r[f"{t}_gray"] = np.average(gg.gray.values[k], weights=w)
                    hp += fp * r[f"{t}_dm05"]; ra += fp * (1 + r[f"{t}_asp"])
                r[f"{t}_fdm05"] = fp * r.get(f"{t}_dm05", 0.0); r[f"{t}_fdm1"] = fp * r.get(f"{t}_dm1", 0.0)
            r[f"{sch}_{sub}_sum_fdm05"] = hp; r[f"{sch}_{sub}_sum_fasp"] = ra
        A = gg.area.values; r[f"g_{sub}_dm05"] = np.average(gg.eqd.values ** -.5, weights=A); r[f"g_{sub}_nb"] = np.average(gg.n_nb.values, weights=A)
    return r


def one(split, id_):
    g = grain_table(np.asarray(Image.open(D / split / f"{id_}.png")), id_); return g, aggregate(g, id_)


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); (C / "p14").mkdir(exist_ok=True)
    items = [("train", i) for i in M.ID.values[itr]] + [("test", i) for i in M.ID.values[ite]]
    import p14_grains as me                                              # workers pickle by reference
    out = Parallel(n_jobs=6)(delayed(me.one)(s, i) for s, i in items)
    G = pd.concat([o[0] for o in out], ignore_index=True); A = pd.DataFrame([o[1] for o in out])
    G.to_parquet(C / "p14" / "grains.parquet"); A.to_parquet(C / "p14" / "agg.parquet")
    print(f"grains: {len(G)} rows over {G.ID.nunique()} images (median {G.groupby('ID').size().median():.0f}/image); "
          f"aggregates: {A.shape[1] - 1} columns, images without enough grains: {int(A['km2_all_p0_f'].isna().sum())}")
