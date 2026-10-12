"""Phase 13 / Part 1: selection-bias audit of Ridge_p12 + absolute vs contrast-normalized dark-phase thresholds.

(A) reported Ridge_p12: S1 + cheap + all ph_* + all px_* + coherence (no threshold picked: every variant enters the ridge)
(B) nested family selection: base = S1 + cheap; the subset of {INT, PH, PX, COH, SIGNED} is chosen by inner 5-fold CV
    on each outer training fold only, then scored on the outer fold (this replays the Phase-12 family decisions honestly)
(C) one dark-fraction variant chosen inside each training fold by |Spearman| (out of all variants looked at)
"""
import itertools, json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import C, CHEAP, REP, SEED, load_meta, splits

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]; n = len(y)
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, G, P, O, L, Dd = rd("features.parquet"), rd("p12/phase.parquet"), rd("p12/pixphase.parquet"), rd("p12/orient.parquet"), rd("p12/intensity.parquet"), rd("p13/desc.parquet")
FAM = {"BASE": sorted(c for c in F.columns if c[:2] in ("q_", "r_", "n_", "b_")) + CHEAP,
       "INT": [c for c in L.columns if c.startswith("L_")], "PH": [c for c in G.columns if c.startswith("ph_")],
       "PX": [c for c in P.columns if c.startswith("px_")], "COH": [c for c in O.columns if c.endswith("_coh")],
       "SIGNED": [c for c in O.columns if c.endswith(("_c", "_s")) and "acf" not in c]}
X = pd.concat([F, M[CHEAP].iloc[itr].reset_index(drop=True), L, G, P, O, Dd], axis=1); X = X.loc[:, ~X.columns.duplicated()]
num = X.select_dtypes("number").columns; X[num] = X[num].fillna(X[num].median())
cols = lambda fams: sorted(set(sum((FAM[f] for f in fams), [])))
sp = splits(n)


def ridge_fit_pred(Xa, ya, Xb):
    return make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 4, 29))).fit(Xa, ya).predict(Xb)


def cv_rmse(fn):
    oof = np.zeros((3, n)); r = []
    for i, (a, b) in enumerate(sp): oof[i // 5, b] = fn(a, b); r.append(np.sqrt(np.mean((oof[i // 5, b] - y[b]) ** 2)))
    return np.array(r), oof


res = {}
# (A) as reported
A, oofA = cv_rmse(lambda a, b: ridge_fit_pred(X[cols(["BASE", "PH", "PX", "COH"])].values[a], y[a], X[cols(["BASE", "PH", "PX", "COH"])].values[b]))
# (B) nested family subset selection
OPT = ["INT", "PH", "PX", "COH", "SIGNED"]; subsets = [s for k in range(len(OPT) + 1) for s in itertools.combinations(OPT, k)]
chosen = []


def nested(a, b):
    best, bs = None, np.inf
    for s in subsets:
        Z = X[cols(["BASE", *s])].values; e = 0
        for ia, ib in KFold(5, shuffle=True, random_state=SEED).split(a):
            e += np.sum((ridge_fit_pred(Z[a[ia]], y[a[ia]], Z[a[ib]]) - y[a[ib]]) ** 2)
        if e < bs: bs, best = e, s
    chosen.append("+".join(best) or "none"); Z = X[cols(["BASE", *best])].values
    return ridge_fit_pred(Z[a], y[a], Z[b])


B, oofB = cv_rmse(nested)
# (C) single dark-fraction variant chosen in-fold
VAR = [c for c in G.columns if c.startswith(("ph_fdark", "ph_darkness"))] + [c for c in P.columns if "_fd" in c or "_fk" in c or "dark_mass" in c]
cv_pick = []


def single(a, b):
    v = max(VAR, key=lambda c: abs(stats.spearmanr(X[c].values[a], y[a])[0])); cv_pick.append(v)
    Z = X[cols(["BASE", "COH"]) + [v]].values; return ridge_fit_pred(Z[a], y[a], Z[b])


Cc, oofC = cv_rmse(single)
base_r, _ = cv_rmse(lambda a, b: ridge_fit_pred(X[cols(["BASE"])].values[a], y[a], X[cols(["BASE"])].values[b]))
for k, r in [("base S1+cheap", base_r), ("(A) reported Ridge_p12", A), ("(B) nested family selection", B), ("(C) in-fold single variant", Cc)]:
    res[k] = dict(mean=float(r.mean()), std=float(r.std())); print(f"{k:32s} {r.mean():.3f} +/- {r.std():.3f}")
print("(B) families chosen per outer fold:", pd.Series(chosen).value_counts().to_dict())
print("(C) variant chosen per outer fold:", pd.Series(cv_pick).value_counts().to_dict())
np.save(C / "p13" / "oof_ridge_p12_nested.npy", oofB)

# absolute vs contrast-normalized dark-phase thresholds: clean vs dirty half
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q)
V = [c for c in Dd.columns if c.startswith(("d_fd_", "p_fd_"))]
T = {}
for c in V:
    v = Dd[c].values; ok = ~np.isnan(v)
    T[c] = dict(unit="abs" if "abs" in c else "rng" if "rng" in c else "vly" if "vly" in c else "mad/sd",
                rho_clean=stats.spearmanr(v[ok & clean], y[ok & clean])[0], rho_dirty=stats.spearmanr(v[ok & ~clean], y[ok & ~clean])[0],
                n_dirty=int((ok & ~clean).sum()))
T = pd.DataFrame(T).T; pd.set_option("display.width", 200); print(T.round(3).to_string())
for lvl in ("d_fd_", "p_fd_"):
    t = T[T.index.str.startswith(lvl)]
    print(f"{lvl}: best |rho| dirty half by unit:", t.groupby("unit").rho_dirty.apply(lambda s: round(s.abs().max(), 3)).to_dict(),
          "| clean:", t.groupby("unit").rho_clean.apply(lambda s: round(s.abs().max(), 3)).to_dict())
json.dump(dict(cv=res, chosen_families=chosen, chosen_variant=cv_pick, variants=VAR, thresholds=T.to_dict("index")), open(REP / "p13_audit.json", "w"), indent=1, default=float)
