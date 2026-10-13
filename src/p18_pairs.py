"""Phase 18 / step 2: matched pairs on the CLEAN half of the training set (train only, CPU).

Representation = standardized [DINOv2-S PCs (PCA fit on train) + the 004 feature matrix PCs]; pairs:
 (a) nearest neighbours in the representation with the largest label gap  -> what differs although everything we measure agrees?
 (b) far apart in the representation with nearly equal labels             -> what can differ without changing the label?
Writes montages (competition images: never committed, see .gitignore) to the folder given by --out, and prints the pair table.
Usage: python src/p18_pairs.py --out <dir>
"""
import argparse, json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, D, REP, load_meta

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=4); args = ap.parse_args()
M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; r = np.load(C / "oof" / "p18_residual_005.npy"); pred = y - r
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, Dd, LN = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p13/desc.parquet"), rd("p13/latnet_pred.parquet")
A = pd.concat([F[[c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")]], M[CHEAP].iloc[itr].reset_index(drop=True), G.filter(like="ph_"), P.filter(like="px_"),
               O.filter(like="_coh"), Dd[[c for c in Dd if c.startswith(("d_", "p_"))]], LN.filter(like="ln_")], axis=1)
A = A.fillna(A.median()).values.astype(float)
E = np.load(C / "p12" / "emb17_dinos_252.npy")[itr]
R = np.hstack([StandardScaler().fit_transform(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(E))),
               StandardScaler().fit_transform(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(A)))])
q = stats.rankdata(F.q_noise.values) + stats.rankdata(F.m_sigma_px.values); cl = np.flatnonzero(q <= np.median(q))
Dm = squareform(pdist(R[cl])); iu = np.triu_indices(len(cl), 1); d, dy = Dm[iu], np.abs(y[cl][iu[0]] - y[cl][iu[1]])
close, far = d <= np.quantile(d, .01), d >= np.quantile(d, .90)
print(f"clean images {len(cl)}, pairs {len(d)}; mean |dy|: closest 1% {dy[close].mean():.1f} | all {dy.mean():.1f} | farthest 10% {dy[far].mean():.1f}")
print(f"Spearman(distance, |dy|) over all clean pairs {stats.spearmanr(d, dy)[0]:+.3f}")


def pick(mask, key, k):
    idx = np.flatnonzero(mask)[np.argsort(key[mask])][:k * 3]; used, out = set(), []
    for t in idx:
        a, b = cl[iu[0][t]], cl[iu[1][t]]
        if a in used or b in used: continue
        used |= {a, b}; out.append((a, b, d[t]))
        if len(out) == k: break
    return out


def montage(pairs, path, title):
    W, H = 2 * 256 + 30, len(pairs) * (256 + 34) + 30; im = Image.new("L", (W, H), 255); dr = ImageDraw.Draw(im); dr.text((8, 8), title, fill=0)
    for k, (a, b, dist) in enumerate(pairs):
        for j, t in enumerate((a, b)):
            x0, y0 = 10 + j * 266, 30 + k * 290
            im.paste(Image.open(D / "train" / f"{ids[t]}.png"), (x0, y0 + 16))
            dr.text((x0, y0), f"{ids[t][-4:]}  y {y[t]:.0f}  pred {pred[t]:.0f}  noise {F.q_noise[t]:.1f}  blur {F.m_sigma_px[t]:.2f}", fill=0)
    im.save(path)


rows = {}
for tag, mask, key, title in [("a_close_bigdy", close, -dy, "(a) nearly identical in DINOv2+features, very different labels"),
                              ("b_far_samey", far & (dy < 2), -d, "(b) very different in DINOv2+features, nearly equal labels")]:
    pairs = pick(mask, key, args.k); montage(pairs, f"{args.out}/p18_{tag}.png", title)
    rows[tag] = [dict(a=ids[a], b=ids[b], dist=float(dd), ya=float(y[a]), yb=float(y[b]), pred_a=float(pred[a]), pred_b=float(pred[b]),
                      **{f"{c}_{s}": float(F[c][t]) for c in ("q_noise", "m_sigma_px", "q_span99") for s, t in (("a", a), ("b", b))}) for a, b, dd in pairs]
    print(f"\n{title}"); print(pd.DataFrame(rows[tag]).round(2).to_string(index=False))
json.dump(dict(rows=rows, close_dy=float(dy[close].mean()), all_dy=float(dy.mean()), far_dy=float(dy[far].mean())), open(REP / "p18_pairs.json", "w"), indent=1)
