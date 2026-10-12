"""Phase 12: per-grain (noise-averaged) raw intensity distributions — do grains fall on discrete phase levels?"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import C, D, REP, load_meta
from features import disk, extract_array


def grain_levels(id_):
    a = np.asarray(Image.open(D / "train" / f"{id_}.png")); f = a.astype(np.float64)
    o, m = extract_array(a, id_, masks=True, ext=False)
    inner = ~ndi.binary_dilation(m["bm"] | m["pores"], disk(2)); lab = m["lab"]
    idx = np.unique(lab[inner]); ar = ndi.sum(inner, lab, idx); mu = ndi.mean(f, lab * inner, idx); ok = ar >= 30
    return id_, mu[ok], ar[ok]


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[ids].reset_index()
    q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); cl = np.flatnonzero(q <= np.quantile(q, .2))
    pick = cl[np.argsort(y[cl])][np.linspace(0, len(cl) - 1, 16).astype(int)]          # 16 clean images spanning y
    res = Parallel(n_jobs=8)(delayed(grain_levels)(ids[i]) for i in pick)
    fig, ax = plt.subplots(4, 4, figsize=(15, 11))
    for a_, (i, (id_, mu, ar)) in zip(ax.ravel(), zip(pick, res)):
        a_.hist(mu, bins=np.arange(60, 220, 2), weights=ar / ar.sum(), color="k")
        a_.set_title(f"{id_[-4:]}  y={y[i]:.0f}  grains={len(mu)}", fontsize=9); a_.set_xlim(60, 215)
    fig.suptitle("area-weighted per-grain mean RAW intensity, 16 cleanest-quintile images ordered by y", fontsize=11)
    fig.tight_layout(); fig.savefig(REP / "p12" / "grain_levels.png", dpi=90)
