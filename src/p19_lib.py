"""Phase 19 shared heads: train-fold standardization, Ridge, RBF kernel ridge / SVR and PLS with inner 5-fold tuning."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.svm import SVR


def std_fit(Xa, Xb):
    med = np.nan_to_num(np.nanmedian(Xa, 0)); Xa, Xb = np.where(np.isfinite(Xa), Xa, med), np.where(np.isfinite(Xb), Xb, med)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-8; return (Xa - mu) / sd, (Xb - mu) / sd


def h_ridge(Xa, ya, Xb):
    return RidgeCV(np.logspace(-2, 6, 33)).fit(Xa, ya).predict(Xb), None


def kernel_head(kind):
    grid = [(c, p) for c in (.5, 1, 2) for p in ((.01, .1, 1) if kind == "krr" else (3, 10, 30))]
    def fit_pred(Ka, ya, Kb, c, p):
        ym, ys = ya.mean(), ya.std()
        m = KernelRidge(alpha=p, kernel="precomputed") if kind == "krr" else SVR(kernel="precomputed", C=p, epsilon=.1)
        return m.fit(Ka, (ya - ym) / ys).predict(Kb) * ys + ym
    def head(Xa, ya, Xb):
        D2a = cdist(Xa, Xa, "sqeuclidean"); med = np.median(D2a[np.triu_indices(len(Xa), 1)]); D2b = cdist(Xb, Xa, "sqeuclidean")
        sc = np.zeros(len(grid)); inner = list(KFold(5, shuffle=True, random_state=0).split(Xa))
        for i, (c, p) in enumerate(grid):
            Kf = np.exp(-c * D2a / med)
            for a, b in inner: sc[i] += np.sum((fit_pred(Kf[np.ix_(a, a)], ya[a], Kf[np.ix_(b, a)], c, p) - ya[b]) ** 2)
        c, p = grid[int(sc.argmin())]; return fit_pred(np.exp(-c * D2a / med), ya, np.exp(-c * D2b / med), c, p), sc.min() / len(ya)
    return head


def h_pls(Xa, ya, Xb):
    grid = (2, 4, 8, 16); inner = list(KFold(5, shuffle=True, random_state=0).split(Xa)); sc = np.zeros(len(grid))
    for i, k in enumerate(grid):
        for a, b in inner: sc[i] += np.sum((PLSRegression(k, scale=False).fit(Xa[a], ya[a]).predict(Xa[b]).ravel() - ya[b]) ** 2)
    k = grid[int(sc.argmin())]; return PLSRegression(k, scale=False).fit(Xa, ya).predict(Xb).ravel(), sc.min() / len(ya)


def ridge_inner(Xa, ya):                                                   # inner 5-fold score for nested config selection (Ridge)
    s = 0.0
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xa): s += np.sum((h_ridge(Xa[a], ya[a], Xa[b])[0] - ya[b]) ** 2)
    return s / len(ya)




def ridge_stacked_views(Va, ya, Vb, alphas=np.logspace(-1, 6, 29), seed=0):
    """Ridge with the D4 views of every training image as augmented rows. Va: (n_train, n_views, d), Vb: (n_test, n_views, d).
    Standardization on the stacked training rows; alpha by an inner 5-fold CV that keeps all views of an image in one fold
    (dual / kernel form, n_rows < d); prediction = mean over the views of each test image."""
    nv = Va.shape[1]
    def fitpred(tr, te_V, als):
        Xt = Va[tr].reshape(-1, Va.shape[2]).astype(np.float64); yt = np.repeat(ya[tr], nv); mu, sd = Xt.mean(0), Xt.std(0) + 1e-8
        Z = (Xt - mu) / sd; ym = yt.mean(); K = Z @ Z.T; w, U = np.linalg.eigh(K); Uy = U.T @ (yt - ym)
        Zt = (te_V.reshape(-1, te_V.shape[2]) - mu) / sd; Kt = Zt @ Z.T
        out = [];
        for al in als:
            coef = U @ (Uy / (w + al)); out.append((Kt @ coef + ym).reshape(len(te_V), nv).mean(1))
        return np.array(out)
    idx = np.random.default_rng(seed).permutation(len(Va)); folds = np.array_split(idx, 5); err = np.zeros(len(alphas))
    for k in range(5):
        tr = np.concatenate([folds[j] for j in range(5) if j != k]); err += ((fitpred(tr, Va[folds[k]], alphas) - ya[folds[k]]) ** 2).sum(1)
    best = alphas[int(err.argmin())]; return fitpred(np.arange(len(Va)), Vb, [best])[0], err.min() / len(Va)
