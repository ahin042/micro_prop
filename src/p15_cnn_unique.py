"""Phase 15 / C: what does the CNN know that the feature model does not?

r_f = y - feature_model_oof (what the features miss), u = cnn_oof - feature_model_oof (where the CNN disagrees).
If corr(u, r_f) > 0 the CNN's disagreements carry signal. Descriptors that correlate with u (but not with the feature
model's inputs linearly) point at the missing latent; images where the CNN is right and the features are wrong are shown.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats

from common import C, CHEAP, D, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids].reset_index(drop=True)
F, Dd, O, LN, S, X_ = (rd(p) for p in ["features.parquet", "p13/desc.parquet", "p12/orient.parquet", "p13/latnet_pred.parquet", "p15/darkspec.parquet", "p13/mix.parquet"])
feat = np.load(C / "p15" / "oof_best_ApDESCpLN.npy").mean(0); cnn = np.load(C / "oof" / "cnn_r18_final_a_oof.npy").mean(0)
rf, u = y - feat, cnn - feat
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = q <= np.median(q)
w = np.polyfit(u, rf, 1)[0]
print(f"corr(u, r_f) = {np.corrcoef(u, rf)[0, 1]:.3f} (clean {np.corrcoef(u[clean], rf[clean])[0, 1]:.3f}, dirty {np.corrcoef(u[~clean], rf[~clean])[0, 1]:.3f}); "
      f"least-squares CNN weight on top of features = {w:.2f}")
print(f"RMSE feature model {np.sqrt(np.mean(rf ** 2)):.2f} | CNN {np.sqrt(np.mean((y - cnn) ** 2)):.2f} | clean: feat {np.sqrt(np.mean(rf[clean] ** 2)):.2f} cnn {np.sqrt(np.mean((y - cnn)[clean] ** 2)):.2f} "
      f"| dirty: feat {np.sqrt(np.mean(rf[~clean] ** 2)):.2f} cnn {np.sqrt(np.mean((y - cnn)[~clean] ** 2)):.2f}")
A = pd.concat([F, Dd, O, LN, S, X_, M[CHEAP].iloc[itr].reset_index(drop=True)], axis=1); A = A.loc[:, ~A.columns.duplicated()].select_dtypes("number")
good = (u * rf > 0)                                                        # CNN moved the prediction in the right direction
rr = pd.DataFrame({c: dict(rho_u=stats.spearmanr(A[c], u, nan_policy="omit")[0], rho_rf=stats.spearmanr(A[c], rf, nan_policy="omit")[0])
                   for c in A.columns if A[c].std() > 0}).T.dropna()
print("descriptors most correlated with the CNN's disagreement u (and with what features miss, r_f):")
print(rr.reindex(rr.rho_u.abs().sort_values(ascending=False).index).head(20).round(3).to_string())
sc = u * rf; up = np.argsort(-np.where((u > 0) & (rf > 0), sc, 0))[:8]; dn = np.argsort(-np.where((u < 0) & (rf < 0), sc, 0))[:8]
t, cap = 256, 30; W = Image.new("L", (8 * t, 2 * (t + cap) + 20), 255); d = ImageDraw.Draw(W)
d.text((4, 2), "CNN right, features wrong. TOP: features too low, CNN higher.  BOTTOM: features too high, CNN lower. [fixed contrast 84-194]", fill=0)
for k, i in enumerate(list(up) + list(dn)):
    r_, c_ = divmod(k, 8); x0, y0 = c_ * t, 20 + r_ * (t + cap); im = np.asarray(Image.open(D / "train" / f"{ids[i]}.png")).astype(float)
    W.paste(Image.fromarray(np.clip((im - 84) / 110 * 255, 0, 255).astype(np.uint8)), (x0, y0))
    d.text((x0 + 3, y0 + t + 2), f"{ids[i][-4:]} y={y[i]:.0f} feat={feat[i]:.0f} cnn={cnn[i]:.0f}", fill=0)
    d.text((x0 + 3, y0 + t + 15), f"noise={F.q_noise[i]:.0f} blur={F.m_sigma_px[i]:.1f} fd={LN.ln_fd20[i]:.2f}", fill=0)
W.save(REP / "p15" / "cnn_unique.png"); print("saved montage")
