"""C1: is a 128x128 quadrant label-equivalent to the full image? ICC(1) of the top-20 predictive features over 4 quadrants."""
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from common import C, D, REP, load_meta
from features import extract_array

warnings.filterwarnings("ignore")
M, itr, ite, y = load_meta()
T = pd.read_csv(REP / "stage1_feature_rho.csv", index_col=0)
TOP = T.rho_y.abs().sort_values(ascending=False).index[:20].tolist()


def quads(id_):
    a = np.asarray(Image.open(D / "train" / f"{id_}.png"))
    return [extract_array(a[i:i + 128, j:j + 128], id_, ext=False) for i in (0, 128) for j in (0, 128)]


Q = pd.DataFrame([o for r in Parallel(n_jobs=-1)(delayed(quads)(i) for i in M.ID.values[itr]) for o in r])
k = 4; out = {}
for c in TOP:
    v = Q[c].values.astype(float).reshape(-1, k); gm = v.mean(1)
    msb = k * ((gm - v.mean()) ** 2).sum() / (len(v) - 1); msw = ((v - gm[:, None]) ** 2).sum() / (len(v) * (k - 1))
    out[c] = dict(icc=(msb - msw) / (msb + (k - 1) * msw), rho_y_full=T.loc[c, "rho_y"])
R = pd.DataFrame(out).T
print(R.round(3).to_string()); print(f"median ICC (top-20 predictive): {R.icc.median():.3f}")
json.dump(dict(median_icc=float(R.icc.median()), per_feature=R.to_dict("index")), open(REP / "stationarity.json", "w"), indent=1)
