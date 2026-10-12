"""Phase 12 extra probes after montage inspection.

Still no per-image normalization. These probes test whether spatial layout,
orientation texture, or segmented phase shape carries the missing signal.
"""
import json
import time

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from scipy.fft import dctn
from skimage import feature, filters, measure, transform
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import C, D, REP, SEED, cv, load_meta, splits


def qstats(x):
    if len(x) == 0:
        return np.zeros(8)
    return np.array([len(x), np.mean(x), np.std(x), np.min(x), *np.percentile(x, [25, 50, 75]), np.max(x)], float)


def phase_shape(f):
    t = filters.threshold_multiotsu(f, 3)
    labs = np.digitize(f, t)
    feats = []
    for ph in range(3):
        m = labs == ph
        feats.append(m.mean())
        cc = measure.label(m, connectivity=1)
        props = measure.regionprops(cc)
        areas = np.array([p.area for p in props if p.area >= 8], float)
        ecc = np.array([p.eccentricity for p in props if p.area >= 8], float)
        sol = np.array([p.solidity for p in props if p.area >= 8], float)
        ar = np.array([p.major_axis_length / max(p.minor_axis_length, 1e-6) for p in props if p.area >= 8], float)
        orient = np.array([p.orientation for p in props if p.area >= 8], float)
        feats.extend(qstats(areas))
        feats.extend(qstats(ecc))
        feats.extend(qstats(sol))
        feats.extend(qstats(np.clip(ar, 0, 20)))
        if len(orient):
            feats.extend([np.mean(np.cos(2 * orient)), np.mean(np.sin(2 * orient))])
        else:
            feats.extend([0, 0])
    return np.array(feats, float)


def one(id_):
    f = np.asarray(Image.open(D / "train" / f"{id_}.png")).astype(np.float32)
    r64 = transform.resize(f, (64, 64), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    d64 = dctn(r64, type=2, norm="ortho")
    hog = feature.hog(f, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                      block_norm="L2-Hys", feature_vector=True)
    lbps = []
    for radius, points in [(1, 8), (2, 16), (3, 24)]:
        lbp = feature.local_binary_pattern(f, points, radius, method="uniform")
        h = np.bincount(lbp.astype(int).ravel(), minlength=points + 2).astype(float)
        lbps.append(h / h.sum())
    return dict(raw64=r64.ravel(), dct64=d64.ravel(), hog=hog, lbp=np.concatenate(lbps), phase_shape=phase_shape(f))


def ridge(Xa, ya, Xb):
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 6, 33))).fit(Xa, ya).predict(Xb)


def krr(Xa, ya, Xb):
    s = StandardScaler().fit(Xa)
    Za, Zb = s.transform(Xa), s.transform(Xb)
    g0 = 1 / Za.shape[1]
    gs = GridSearchCV(
        KernelRidge(kernel="rbf"),
        {"alpha": [.03, .3, 3], "gamma": [g0 * .1, g0 * .3, g0, g0 * 3]},
        cv=KFold(3, shuffle=True, random_state=SEED),
        scoring="neg_mean_squared_error",
    ).fit(Za, ya - ya.mean())
    return gs.predict(Zb) + ya.mean()


if __name__ == "__main__":
    M, itr, _, y = load_meta()
    ids = M.ID.values[itr]
    P = C / "p12"
    P.mkdir(exist_ok=True)
    if not (P / "phase_shape.npy").exists():
        t0 = time.time()
        rows = Parallel(n_jobs=2)(delayed(one)(i) for i in ids)
        for k in rows[0]:
            np.save(P / f"{k}.npy", np.stack([r[k] for r in rows]))
        print(f"[extra feature extraction] {time.time() - t0:.1f}s", flush=True)

    fams = {k: np.load(P / f"{k}.npy") for k in ["raw64", "dct64", "hog", "lbp", "phase_shape"]}
    fams["hog+phase_shape"] = np.hstack([fams["hog"], fams["phase_shape"]])
    sp = splits(len(y))
    var = y.var()
    res = {}
    for name, X in fams.items():
        X = np.nan_to_num(X.astype(float))
        r = {"dim": int(X.shape[1])}
        for mn, fn in [("ridge", ridge), ("krr", krr)]:
            o = cv(fn, X, y, sp)
            r[mn] = [round(o["mean"], 3), round(o["std"], 3)]
        curve, rmse_curve = {}, {}
        for k in [1, 2, 3, 5, 8, 12, 20, 30]:
            if k > min(X.shape[1], 300):
                continue
            o = cv(lambda a, b, c, kk=k: PLSRegression(kk, scale=True).fit(a, b).predict(c).ravel(), X, y, sp)
            rmse_curve[k] = [round(o["mean"], 3), round(o["std"], 3)]
            curve[k] = round(1 - o["mean"] ** 2 / var, 3)
        best_k = min(rmse_curve, key=lambda kk: rmse_curve[kk][0])
        r["pls"] = dict(best_k=int(best_k), rmse=rmse_curve[best_k], rmse_curve=rmse_curve, r2_curve=curve)
        res[name] = r
        print(f"{name:15s} dim {r['dim']:5d} | ridge {r['ridge'][0]:.3f}±{r['ridge'][1]:.3f} | "
              f"krr {r['krr'][0]:.3f}±{r['krr'][1]:.3f} | PLS best k={best_k} {r['pls']['rmse'][0]:.3f}±{r['pls']['rmse'][1]:.3f}",
              flush=True)
    (REP / "p12_extra_probes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
