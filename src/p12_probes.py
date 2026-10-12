"""Phase 12 / Step 3: high-capacity representation probes on RAW intensities (no per-image normalization).

Families (train images only; cached in cache/p12/):
  a spec   : Hann-windowed 2D log-power spectrum, 64 radial x 8 angular bins
  b hist   : smoothed 256-bin gray histogram + GMM (K=2,3,4) parameters
  c corr   : radial autocovariance/autocorrelation (64 lags) + two-point S2 and lineal-path L (dark/bright phase, x/y)
  d gabor  : Gabor bank (5 scales x 8 orientations) mean/std + second-order (scattering-like) means
  e emb_*  : frozen timm backbones (Apache-2.0), multi-stage mean+std pooled, full 256x256 input
Models: Ridge (inner alpha), PLS curve RMSE/R^2 vs n_components, RBF kernel ridge (inner alpha/gamma); shared 5x3 folds.
"""
import json, time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, D, REP, SEED, cv, load_meta, splits

N = 256; yy, xx = np.indices((N, N)) - N // 2
RAD = np.hypot(yy, xx); ANG = np.mod(np.arctan2(yy, xx), np.pi)
RB = np.minimum((RAD / 2).astype(int), 63); AB = np.minimum((ANG / (np.pi / 8)).astype(int), 7)
OK = (RAD >= 1) & (RAD < 128); HANN = np.outer(np.hanning(N), np.hanning(N))
FR = np.fft.fftfreq(N); FY, FX = np.meshgrid(FR, FR, indexing="ij"); FRAD = np.hypot(FY, FX); FANG = np.arctan2(FY, FX)


def gabor_bank():
    G = []
    for j, f0 in enumerate([1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]):
        for t in range(8):
            th = t * np.pi / 8; d = np.angle(np.exp(1j * (FANG - th)))
            G.append(np.exp(-(np.log(np.maximum(FRAD, 1e-9) / f0) ** 2) / (2 * 0.55 ** 2)) * np.exp(-d ** 2 / (2 * (np.pi / 10) ** 2)) * (FRAD > 0))
    return np.array(G)


GB = gabor_bank()


def lineal(B):
    """Lineal-path L(r), r=1..64, along rows of a boolean mask."""
    L = np.zeros(64)
    for row in B:
        e = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        for ln in e[1::2] - e[::2]:
            r = np.arange(1, min(ln, 64) + 1); L[r - 1] += ln - r + 1
    return L / B.size


def feats(i):
    f = np.asarray(Image.open(D / "train" / f"{i}.png")).astype(np.float64); x = f - f.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * HANN))) ** 2
    spec = np.log(np.bincount((RB * 8 + AB)[OK], P[OK], minlength=512) / np.maximum(np.bincount((RB * 8 + AB)[OK], minlength=512), 1) + 1e-6)
    h = ndi.gaussian_filter1d(np.bincount(f.astype(int).ravel(), minlength=256) / f.size, 2)
    px = np.random.default_rng(SEED).choice(f.ravel(), 20000, replace=False)[:, None]; gm = []
    for k in (2, 3, 4):
        g = GaussianMixture(k, random_state=0).fit(px); o = np.argsort(g.means_[:, 0])
        gm += [*g.means_[o, 0], *np.sqrt(g.covariances_.ravel()[o]), *g.weights_[o], g.bic(px) / len(px)]
    A = np.real(np.fft.ifft2(np.abs(np.fft.fft2(x, s=(2 * N, 2 * N))) ** 2)) / f.size
    A = np.fft.fftshift(A)[N // 2:N // 2 + N, N // 2:N // 2 + N]; ri = RAD.astype(int)
    acov = (np.bincount(ri.ravel(), A.ravel()) / np.bincount(ri.ravel()))[:64]
    t = filters.threshold_multiotsu(ndi.gaussian_filter(f, 1), 3)[0]; B = ndi.gaussian_filter(f, 1) < t
    b = B - B.mean(); S = np.real(np.fft.ifft2(np.abs(np.fft.fft2(b, s=(2 * N, 2 * N))) ** 2)) / f.size
    S = np.fft.fftshift(S)[N // 2:N // 2 + N, N // 2:N // 2 + N]; s2 = (np.bincount(ri.ravel(), S.ravel()) / np.bincount(ri.ravel()))[:64] + B.mean() ** 2
    corr = np.r_[acov, acov / acov[0], s2, lineal(B), lineal(B.T), lineal(~B), lineal(~B.T), B.mean()]
    F0 = np.fft.fft2(x); U = np.abs(np.fft.ifft2(F0[None] * GB))
    g1 = np.r_[U.mean((1, 2)), U.std((1, 2))]; g2 = []
    for a in range(40):
        ja = a // 8; Fu = np.fft.fft2(U[a] - U[a].mean())
        for b2 in range(40):
            if b2 // 8 > ja: g2.append(np.abs(np.fft.ifft2(Fu * GB[b2])).mean())
    return dict(spec=spec, hist=np.r_[h, gm], corr=corr, gabor=np.r_[g1, g2])


def embeddings(ids, name, ckpt=None):
    import timm, torch
    torch.set_num_threads(2)
    kw = dict(pretrained_cfg_overlay=dict(file=ckpt)) if ckpt else {}
    m = timm.create_model(name, pretrained=True, features_only=True, **kw).eval()
    mean, std = np.array(m.pretrained_cfg["mean"]), np.array(m.pretrained_cfg["std"]); out = []
    with torch.no_grad():
        for k in range(0, len(ids), 16):
            a = np.stack([np.asarray(Image.open(D / "train" / f"{i}.png")) for i in ids[k:k + 16]]).astype(np.float32) / 255
            x = torch.from_numpy(((a[:, None] - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32))
            out.append(torch.cat([torch.cat([z.mean((2, 3)), z.std((2, 3))], 1) for z in m(x)], 1).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; P = C / "p12"; P.mkdir(exist_ok=True); t0 = time.time()
    if not (P / "gabor.npy").exists():
        rows = Parallel(n_jobs=8)(delayed(feats)(i) for i in ids)
        for k in rows[0]: np.save(P / f"{k}.npy", np.stack([r[k] for r in rows]))
        print(f"[feats a-d] {time.time() - t0:.0f}s", flush=True)
    local_backbones = [
        ("resnet18.a1_in1k", C.parent / "weights" / "resnet18.a1_in1k.safetensors"),
        ("efficientnet_b0.ra_in1k", C.parent / "weights" / "efficientnet_b0.ra_in1k.safetensors"),
    ]
    for nm, ck in local_backbones:
        fn = P / f"emb_{nm.split('.')[0]}.npy"
        if not fn.exists():
            if not ck.exists():
                print(f"[skip emb {nm}] missing local weights: {ck}", flush=True)
                continue
            np.save(fn, embeddings(ids, nm, str(ck)))
            print(f"[emb {nm}] {time.time() - t0:.0f}s", flush=True)
    F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[M.ID].reset_index().iloc[itr]
    fams = {k: np.load(P / f"{k}.npy") for k in ["spec", "hist", "corr", "gabor"]}
    for k in ["emb_resnet18", "emb_efficientnet_b0"]:
        fn = P / f"{k}.npy"
        if fn.exists(): fams[k] = np.load(fn)
    fams["s1+cheap (ref)"] = np.hstack([F[[c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")]].values.astype(float), M[CHEAP].values[itr]])
    fams["a+b+c+d"] = np.hstack([fams[k] for k in ["spec", "hist", "corr", "gabor"]])
    sp = splits(len(y)); var = y.var(); res = {}

    def ridge(Xa, ya, Xb): return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 6, 33))).fit(Xa, ya).predict(Xb)

    def krr(Xa, ya, Xb):
        s = StandardScaler().fit(Xa); Za, Zb = s.transform(Xa), s.transform(Xb); g0 = 1 / Za.shape[1]
        gs = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [.03, .3, 3], "gamma": [g0 * .1, g0 * .3, g0, g0 * 3]}, cv=KFold(3, shuffle=True, random_state=SEED),
                          scoring="neg_mean_squared_error").fit(Za, ya - ya.mean())
        return gs.predict(Zb) + ya.mean()

    for name, X in fams.items():
        X = np.nan_to_num(X.astype(float)); r = {"dim": X.shape[1]}
        for mn, fn in [("ridge", ridge), ("krr", krr)]:
            o = cv(fn, X, y, sp); r[mn] = (round(o["mean"], 3), round(o["std"], 3)); np.save(P / f"oof_{name.split(' ')[0]}_{mn}.npy", o["oof"])
        curve, rmse_curve = {}, {}
        for k in [1, 2, 3, 5, 8, 12, 20, 30]:
            if k > min(X.shape[1], 300): continue
            o = cv(lambda a, b_, c, kk=k: PLSRegression(kk, scale=True).fit(a, b_).predict(c).ravel(), X, y, sp)
            rmse_curve[k] = (round(o["mean"], 3), round(o["std"], 3))
            curve[k] = round(1 - o["mean"] ** 2 / var, 3)
        best_k = min(rmse_curve, key=lambda kk: rmse_curve[kk][0])
        r["pls"] = dict(best_k=int(best_k), rmse=rmse_curve[best_k], rmse_curve=rmse_curve, r2_curve=curve)
        res[name] = r
        print(f"{name:18s} dim {r['dim']:5d} | ridge {r['ridge'][0]:.3f}±{r['ridge'][1]:.3f} | "
              f"krr {r['krr'][0]:.3f}±{r['krr'][1]:.3f} | PLS best k={best_k} "
              f"{rmse_curve[best_k][0]:.3f}±{rmse_curve[best_k][1]:.3f} | PLS R2 {curve}", flush=True)
    json.dump(res, open(REP / "p12_probes.json", "w"), indent=1)
