"""Phase 13 / Part 3: degradation-robust latent estimator (multi-head CNN), trained on train images only, uses NO y.

Pseudo-labels: 10 latents measured by the classical pipeline on the cleanest train images.
Inputs: the same clean images re-degraded with (blur, noise, contrast gain, brightness) drawn jointly from the empirical
nuisance distribution of the dirty half (only the missing amount is added: sigma_add = sqrt(target^2 - own^2)).
(i) measurement accuracy on held-out clean images at fixed degradation levels vs the classical estimators re-run on the
    same degraded images; (ii) predicted latents for all 1500 images -> cache/p13/latnet_pred.parquet for downstream CV.
Usage: python src/p13_latnet.py [--epochs 100] [--final_only] [--crossfit 5 --out latnet_cf_pred.parquet]
"""
import argparse, json, math, time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image
from scipy import ndimage as ndi, stats

from common import C, D, REP, ROOT, SEED, load_meta, sha256
from p13_descriptors import desc_array

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--out", default="latnet_pred.parquet"); ap.add_argument("--final_only", action="store_true")
ap.add_argument("--crossfit", type=int, default=0, help="K>0: cross-fitted predictions (clean train images get out-of-fold predictions)")
args = ap.parse_args()
torch.use_deterministic_algorithms(True, warn_only=True); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
M, itr, ite, y = load_meta(); ids_all = M.ID.values
rd = lambda p: pd.read_parquet(C / p).set_index("ID").loc[ids_all].reset_index(drop=True)
F, Dd, O, X_ = rd("features.parquet"), rd("p13/desc.parquet"), rd("p12/orient.parquet"), rd("p13/mix.parquet")
LAT = {"fd20": Dd.d_fd_abs20, "pfd10": Dd.p_fd_abs10, "align": O.or_s3_coh, "logsize": np.log(Dd.d_eqd_mean), "aspect": Dd.d_asp_aw,
       "pores": Dd.d_pore_frac, "gbdens": Dd.d_gb_density, "glsd": X_.mx_gl_sd, "skew": Dd.d_eqd_skew, "msd": Dd.d_matrix_sd_rel}
L = pd.DataFrame(LAT).values.astype(np.float32); K = L.shape[1]
qa = stats.rankdata(F.q_noise.values[itr]) + stats.rankdata(F.m_sigma_px.values[itr])
cl = itr[(qa <= np.quantile(qa, .4)) & np.isfinite(L[itr]).all(1)]; dirty = itr[qa > np.median(qa)]
mu_L, sd_L = L[cl].mean(0), L[cl].std(0); T = (L - mu_L) / sd_L
perm = rng.permutation(cl); hold, fit = perm[:len(cl) // 5], perm[len(cl) // 5:]
print(f"clean pseudo-label images: {len(cl)} (fit {len(fit)}, held-out {len(hold)}); dirty nuisance pool: {len(dirty)}")
NB, NN, SPAN, MEAN = (F.m_sigma_px.values.astype(np.float32), F.q_noise.values.astype(np.float32), F.q_span99.values.astype(np.float32), M["mean"].values.astype(np.float32))
IMG = torch.from_numpy(np.stack([np.asarray(Image.open(D / s / f"{i}.png")) for s, i in zip(M.split, ids_all)]))[:, None].to(dev)
GM, GS = (IMG[itr].float() / 255).mean().item(), (IMG[itr].float() / 255).std().item()


def gblur(x, s):
    r = 8; t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t[None] ** 2 / (2 * s[:, None].clamp_min(1e-3) ** 2)); k = k / k.sum(1, keepdim=True); B = x.shape[0]; z = x.transpose(0, 1)
    z = Fn.conv2d(Fn.pad(z, (0, 0, r, r), mode="reflect"), k[:, None, :, None], groups=B)
    return Fn.conv2d(Fn.pad(z, (r, r, 0, 0), mode="reflect"), k[:, None, None, :], groups=B).transpose(0, 1)


def degrade(idx, x, g, src=None, keep=.15):
    """x: (B,1,H,W) in gray levels. src: dirty images whose nuisance to imitate (random from the pool if None)."""
    B = len(idx); src = dirty[torch.randint(0, len(dirty), (B,), generator=g).numpy()] if src is None else src
    t = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
    sb = t(np.sqrt(np.clip(NB[src] ** 2 - NB[idx] ** 2, 0, None))); gain = t(SPAN[src] / SPAN[idx])
    sn = t(np.sqrt(np.clip(NN[src] ** 2 - (NN[idx] * SPAN[src] / SPAN[idx]) ** 2, 0, None)))
    on = (torch.rand(B, generator=g) >= keep).to(dev).float()
    z = torch.where((sb * on > .05)[:, None, None, None], gblur(x, sb * on + 1e-3), x)
    mi, mj = t(MEAN[idx])[:, None, None, None], t(MEAN[src])[:, None, None, None]; gn = (gain - 1) * on + 1
    z = (z - mi) * gn[:, None, None, None] + mi + (mj - mi) * on[:, None, None, None]
    z = z + torch.randn(z.shape, generator=torch.Generator(device=dev).manual_seed(int(torch.randint(0, 2**31, (1,), generator=g))), device=dev) * (sn * on)[:, None, None, None]
    return z.clamp(0, 255).round()


def d4(x, k, f):
    x = torch.rot90(x, k, (2, 3)); return torch.flip(x, (3,)) if f else x


def make_model():
    wf = ROOT / "weights" / "resnet18.a1_in1k.safetensors"
    ov = dict(file=str(wf)) if wf.exists() and sha256(wf) == "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612" else None
    return timm.create_model("resnet18", pretrained=True, in_chans=1, num_classes=K, pretrained_cfg_overlay=ov).to(dev)


def train(idx):
    torch.manual_seed(SEED); g = torch.Generator().manual_seed(SEED); m = make_model(); tgt = torch.tensor(T[idx], device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2); spe = math.ceil(len(idx) / args.bs); tot = spe * args.epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / spe) * .5 * (1 + math.cos(math.pi * s / tot)))
    for ep in range(args.epochs):
        m.train(); pr = torch.randperm(len(idx), generator=g)
        for i in range(spe):
            b = pr[i * args.bs:(i + 1) * args.bs].numpy()
            if len(b) < 2: continue
            x = degrade(idx[b], IMG[idx[b]].float(), g); ks, fs = torch.randint(0, 4, (1,), generator=g).item(), torch.randint(0, 2, (1,), generator=g).item()
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                loss = Fn.mse_loss(m((d4(x, ks, fs) / 255 - GM) / GS).float(), tgt[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return m.eval()


@torch.no_grad()
def predict(m, x):
    out = []
    for i in range(0, len(x), 50):
        z = (x[i:i + 50] / 255 - GM) / GS; out.append(torch.stack([m(d4(z, k, f)).float() for k in range(4) for f in (0, 1)]).mean(0).cpu().numpy())
    return np.concatenate(out)


def classical(a):
    r = desc_array(a); f = a.astype(np.float64); gq = ndi.gaussian_filter(f, 3); gy, gx = np.gradient(gq)
    J = [ndi.gaussian_filter(v, 6).mean() for v in (gx * gx, gy * gy, -gx * gy)]
    return [r.get("d_fd_abs20", np.nan), r.get("p_fd_abs10", np.nan), np.hypot(J[0] - J[1], 2 * J[2]) / (J[0] + J[1]), np.log(r.get("d_eqd_mean", np.nan)),
            r.get("d_asp_aw", np.nan), r.get("d_pore_frac", np.nan), r.get("d_gb_density", np.nan), np.nan, r.get("d_eqd_skew", np.nan), r.get("d_matrix_sd_rel", np.nan)]


if __name__ == "__main__":
    if args.crossfit:                                                       # see src/p16_latnet2.py: removes the in-sample advantage of clean train images
        fo = np.random.default_rng(SEED + 100).permutation(len(cl)) % args.crossfit
        P, Pk = np.zeros((len(ids_all), K)), []
        for k in range(args.crossfit):
            t0 = time.time(); pk = predict(train(cl[fo != k]), IMG.float()); Pk.append(pk); P[cl[fo == k]] = pk[cl[fo == k]]
            print(f"[crossfit fold {k}] trained on {(fo != k).sum()} clean images, {time.time() - t0:.0f}s", flush=True)
        rest = np.setdiff1d(np.arange(len(ids_all)), cl); P[rest] = np.mean(Pk, 0)[rest]; P = P * sd_L + mu_L
        pd.DataFrame(P, columns=[f"ln_{k}" for k in LAT]).assign(ID=ids_all).to_parquet(C / "p13" / args.out); print("wrote cache/p13/" + args.out); raise SystemExit
    if args.final_only:
        m = train(cl); P = predict(m, IMG.float()) * sd_L + mu_L
        pd.DataFrame(P, columns=[f"ln_{k}" for k in LAT]).assign(ID=ids_all).to_parquet(C / "p13" / args.out); print("wrote", args.out); raise SystemExit
    t0 = time.time(); m = train(fit); print(f"[split model] trained {time.time() - t0:.0f}s", flush=True)
    names = list(LAT); qs = {"L0 none": None, "L1 dirty p25": .25, "L2 dirty p50": .5, "L3 dirty p75": .75, "L4 dirty p90": .9}; acc = {}
    for lv, qq in qs.items():
        if qq is None: x = IMG[hold].float()
        else:
            tb, tn = np.quantile(NB[dirty], qq), np.quantile(NN[dirty], qq)
            sb = torch.tensor(np.sqrt(np.clip(tb ** 2 - NB[hold] ** 2, 0, None)), device=dev, dtype=torch.float32) + 1e-3
            sn = torch.tensor(np.sqrt(np.clip(tn ** 2 - NN[hold] ** 2, 0, None)), device=dev, dtype=torch.float32)
            x = (gblur(IMG[hold].float(), sb) + torch.randn(IMG[hold].shape, generator=torch.Generator(device=dev).manual_seed(7), device=dev) * sn[:, None, None, None]).clamp(0, 255).round()
        P = predict(m, x) * sd_L + mu_L; Cl = np.array([classical(x[i, 0].cpu().numpy().astype(np.uint8)) for i in range(len(hold))])
        for k, nm in enumerate(names):
            tr = L[hold, k]; okc = np.isfinite(Cl[:, k])
            acc[(lv, nm)] = dict(net_r=float(np.corrcoef(P[:, k], tr)[0, 1]), net_rmse_sd=float(np.sqrt(np.mean((P[:, k] - tr) ** 2)) / sd_L[k]),
                                 classical_r=float(np.corrcoef(Cl[okc, k], tr[okc])[0, 1]) if okc.sum() > 5 else np.nan, classical_nan=int((~okc).sum()))
    A = pd.DataFrame(acc).T; pd.set_option("display.width", 220)
    print("Pearson r with clean pseudo-labels on held-out clean images (net | classical re-measured on the degraded image):")
    print(A.net_r.unstack(0).round(2).to_string()); print(A.classical_r.unstack(0).round(2).to_string())
    A.to_csv(REP / "p13_latnet_accuracy.csv")
    t0 = time.time(); m = train(cl); print(f"[final model on all {len(cl)} clean] trained {time.time() - t0:.0f}s", flush=True)
    P = predict(m, IMG.float()) * sd_L + mu_L
    pd.DataFrame(P, columns=[f"ln_{k}" for k in names]).assign(ID=ids_all).to_parquet(C / "p13" / args.out); print("wrote cache/p13/" + args.out)
