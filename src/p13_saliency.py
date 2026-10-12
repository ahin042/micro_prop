"""Phase 13 / Part 2d: occlusion maps of the winning CNN config on high-|residual| clean images.

The 8 clean images with the largest |OOF residual| of the best blend are held out; one model is trained on the other 492
with exactly the code of src/cnn.py (train_one), then each held-out image is occluded with a 32x32 patch (filled with a
local Gaussian blur, i.e. structure removed but brightness kept) at stride 16. Map = prediction change (HV); D4-TTA mean.
"""
import runpy, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi, stats

sys.argv = ["cnn.py", "--tag", "zz_saliency", "--arch", "resnet18", "--aug", "d4,blur,noise", "--norm", "global", "--epochs", "40", "--folds", "0", "--reps", "0"]
g = runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("cnn.py")))      # defines functions; trains nothing
from common import C, REP, load_meta

M, itr, ite, y = load_meta(); ids = M.ID.values[itr]
F = pd.read_parquet(C / "features.parquet").set_index("ID").loc[ids].reset_index()
q = stats.rankdata(F.q_noise) + stats.rankdata(F.m_sigma_px); clean = np.flatnonzero(q <= np.median(q))
oof = .5 * (np.load(C / "oof" / "cnn_r18_d4blurnoise_global_oof.npy") + np.load(C / "p12" / "oof_ridge_+phase+pix+coh.npy")); res = y - oof.mean(0)
pick = clean[np.argsort(-np.abs(res[clean]))[:8]]; train = np.setdiff1d(np.arange(len(y)), pick)
model, mu, sd, _ = g["train_one"](train, pick, 42); model.eval(); X = g["Xtr"]; norm, d4 = g["norm"], g["d4"]


@torch.no_grad()
def pred(batch):                                       # batch: (B,1,H,W) float in [0,1]
    x = norm(batch); return torch.stack([model(d4(x, k, f)).float().squeeze(1) for k in range(4) for f in (0, 1)]).mean(0).cpu().numpy() * sd + mu


fig, ax = plt.subplots(4, 4, figsize=(15, 15))
for j, i in enumerate(pick):
    img = X[i:i + 1].float() / 255; base = pred(img)[0]
    blur = torch.from_numpy(ndi.gaussian_filter(img[0, 0].cpu().numpy(), 6)).to(img.device)
    H = np.zeros((15, 15)); B = []
    for r in range(15):
        for c in range(15):
            z = img.clone(); z[0, 0, r * 16:r * 16 + 32, c * 16:c * 16 + 32] = blur[r * 16:r * 16 + 32, c * 16:c * 16 + 32]; B.append(z)
    P = np.concatenate([pred(torch.cat(B[k:k + 25])) for k in range(0, len(B), 25)]); H[:] = (P - base).reshape(15, 15)
    a1, a2 = ax[(j // 2), (j % 2) * 2], ax[(j // 2), (j % 2) * 2 + 1]
    a1.imshow(X[i, 0].cpu().numpy(), cmap="gray", vmin=84, vmax=194); a1.set_title(f"{ids[i][-4:]} y={y[i]:.0f} blend={oof.mean(0)[i]:.0f} cnn={base:.0f}", fontsize=9)
    v = np.abs(H).max(); a2.imshow(X[i, 0].cpu().numpy(), cmap="gray", vmin=84, vmax=194)
    a2.imshow(np.kron(H, np.ones((16, 16)))[:256, :256], cmap="RdBu_r", vmin=-v, vmax=v, alpha=.55)
    a2.set_title(f"occlusion: pred change (red = removing it raises pred), max |{v:.1f}| HV", fontsize=8)
    for a_ in (a1, a2): a_.axis("off")
fig.tight_layout(); fig.savefig(REP / "p13" / "occlusion.png", dpi=80); print("saved", [ids[i] for i in pick])
