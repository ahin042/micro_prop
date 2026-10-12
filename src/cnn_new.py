"""Part C: CNN regression on full 256x256 images (C1: 128px patches are NOT label-equivalent, median ICC 0.60).

timm backbone, in_chans=1, ImageNet-1k pretrained (timm/resnet18.a1_in1k, timm/efficientnet_b0.ra_in1k: Apache-2.0).
Same RepeatedKFold(5x3, seed 42) image folds as every other model; OOF saved as (n_repeats, n) for blending.
No early stopping on the reported fold (that would leak): fixed cosine schedule, final EMA weights, D4 TTA.
"""
import argparse, json, math, time

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image

from common import C, D, REP, SEED, load_meta, splits

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="resnet18"); ap.add_argument("--norm", default="global", choices=["global", "image"])
ap.add_argument("--aug", default="full", help="comma list of d4,blur,noise,photo | full | none")
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--reps", default="0", help="repeat indices to train, e.g. 0,1,2 (results merge with earlier runs of the same tag)")
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--tag", required=True); ap.add_argument("--no_test", action="store_true")
args = ap.parse_args()
AUG = {"full": {"d4", "blur", "noise", "photo"}, "none": set()}.get(args.aug, set(args.aug.split(",")))
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
M, itr, ite, y = load_meta()
load = lambda split, ids: torch.from_numpy(np.stack([np.asarray(Image.open(D / split / f"{i}.png")) for i in ids]))[:, None]
Xtr, Xte = load("train", M.ID.values[itr]).to(dev), load("test", M.ID.values[ite]).to(dev)
GM, GS = (Xtr.float() / 255).mean().item(), (Xtr.float() / 255).std().item()


def d4(x, k, f):
    x = torch.rot90(x, k, (2, 3)); return torch.flip(x, (3,)) if f else x


def gblur(x, s):
    """Per-sample Gaussian blur (sigma vector s) as two grouped separable convolutions."""
    r = 6; t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t[None] ** 2 / (2 * s[:, None].clamp_min(1e-3) ** 2)); k = k / k.sum(1, keepdim=True)
    B = x.shape[0]; z = x.transpose(0, 1)                                   # (1, B, H, W)
    z = Fn.conv2d(Fn.pad(z, (0, 0, r, r), mode="reflect"), k[:, None, :, None], groups=B)
    z = Fn.conv2d(Fn.pad(z, (r, r, 0, 0), mode="reflect"), k[:, None, None, :], groups=B)
    return z.transpose(0, 1)


def augment(x, g):
    """x: (B,1,H,W) in [0,1]. Label-preserving nuisance augmentation; ranges from Stage 0/1 measurements."""
    B = x.shape[0]; u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi, generator=g)
    on = lambda p: (torch.rand(B, device=dev, generator=g) < p).float()
    if "d4" in AUG:
        ks = torch.randint(0, 4, (B,), generator=g, device=dev).tolist(); fs = torch.randint(0, 2, (B,), generator=g, device=dev).tolist()
        x = torch.stack([d4(x[i:i + 1], ks[i], fs[i])[0] for i in range(B)])
    if "blur" in AUG:
        s = u(0, 2) * on(.5); x = torch.where((s > 0)[:, None, None, None], gblur(x, s), x)
    if "photo" in AUG:
        m = x.mean((1, 2, 3), keepdim=True); x = x.clamp(1e-4, 1) ** torch.exp(u(math.log(.7), math.log(1.4)))[:, None, None, None]
        x = (x - m) * u(.7, 1.3)[:, None, None, None] + m + u(-.08, .08)[:, None, None, None]
    if "noise" in AUG:
        x = x + torch.randn(x.shape, device=dev, generator=g) * (u(0, 20) * on(.5) / 255)[:, None, None, None]
    return x.clamp(0, 1)


def norm(x):
    if args.norm == "image": return (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-4)
    return (x - GM) / GS


@torch.no_grad()
def predict(model, X, bs=50):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs].float() / 255)
        out.append(torch.stack([model(d4(x, k, f)).float().squeeze(1) for k in range(4) for f in (0, 1)]).mean(0))
    return torch.cat(out).cpu().numpy()


def train_one(a, b, seed):
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    mu, sd = y[a].mean(), y[a].std(); yt = torch.tensor((y[a] - mu) / sd, dtype=torch.float32, device=dev)
    model = timm.create_model(args.arch, pretrained=True, in_chans=1, num_classes=1, drop_rate=0.1).to(dev)
    ema = timm.create_model(args.arch, pretrained=False, in_chans=1, num_classes=1).to(dev); ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    spe = math.ceil(len(a) / args.bs); tot = spe * args.epochs; warm = spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (tot - warm))))
    step = 0; curve = []
    for ep in range(args.epochs):
        model.train(); ls = []; perm = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed * 1000 + ep))
        for i in range(spe):
            b = perm[i * args.bs:(i + 1) * args.bs]
            if len(b) < 2: continue
            xb = norm(augment(Xtr[a[b.numpy()]].float() / 255, g))
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                loss = Fn.mse_loss(model(xb).float().squeeze(1), yt[b.to(dev)])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1; ls.append(loss.item())
            dcy = min(0.995, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    pe.copy_(pe * dcy + pm.detach() * (1 - dcy)) if pe.dtype.is_floating_point else pe.copy_(pm)
        if (ep + 1) % 5 == 0:                                            # diagnostics only, never used for selection
            ema.eval()
            with torch.no_grad(): pv = torch.cat([ema(norm(Xtr[b[i:i + 50]].float() / 255)).float().squeeze(1) for i in range(0, len(b), 50)]).cpu().numpy()
            curve.append([ep + 1, float(np.sqrt(np.mean(ls)) * sd), float(np.sqrt(np.mean((pv * sd + mu - y[b]) ** 2)))])
    return ema, mu, sd, curve


# OOF and test use the SAME predict(): full 256x256 image, 8 D4 transforms, mean of de-standardized outputs, EMA weights
reps = [int(r) for r in args.reps.split(",")]; PART = C / "oof" / "cnn_parts"; PART.mkdir(parents=True, exist_ok=True)
fo, fj = C / "oof" / f"cnn_{args.tag}_oof.npy", REP / f"cnn_{args.tag}.json"
oof = np.full((3, len(y)), np.nan); prev = json.load(open(fj)) if fj.exists() else {}
if fo.exists(): o = np.load(fo); oof[:len(o)] = o                       # merge with earlier repeats of this tag
folds = prev.get("folds") or {str(i): {"rmse": v} for i, v in enumerate(prev.get("fold_rmse", []))}
t0 = time.time()
for fi, (a, b) in [(i, ab) for i, ab in enumerate(splits(len(y))) if i // 5 in reps and i % 5 < args.folds]:
    pv, tv, cv_, trr = [], [], [], []
    for s in range(args.seeds):
        model, mu, sd, curve = train_one(a, b, SEED + s); pv.append(predict(model, Xtr[b]) * sd + mu); cv_.append(curve)
        trr.append(float(np.sqrt(np.mean((predict(model, Xtr[a]) * sd + mu - y[a]) ** 2))))
        if not args.no_test: tv.append(predict(model, Xte) * sd + mu)
    oof[fi // 5, b] = np.mean(pv, 0); rm = float(np.sqrt(np.mean((oof[fi // 5, b] - y[b]) ** 2)))
    if tv: np.save(PART / f"{args.tag}_test_f{fi}.npy", np.mean(tv, 0))
    folds[str(fi)] = dict(rmse=rm, train_rmse=float(np.mean(trr)), curve=cv_[0])
    print(f"[{args.tag}] fold {fi} rmse {rm:.3f} train {np.mean(trr):.3f} ({time.time() - t0:.0f}s)", flush=True)
if folds:                                                            # nothing to save when no fold ran
    fr = [folds[k]["rmse"] for k in sorted(folds, key=int)]
    res = dict(config={**prev.get("config", {}), **vars(args)}, aug=sorted(AUG), device=str(dev), folds=folds, fold_rmse=fr,
               mean=float(np.mean(fr)), std=float(np.std(fr)), n_folds=len(fr), sec=prev.get("sec", 0) + time.time() - t0)
    print(f"[{args.tag}] RMSE {res['mean']:.3f} +/- {res['std']:.3f} over {len(fr)} folds")
    np.save(fo, oof[:max(int(k) for k in folds) // 5 + 1])
    parts = sorted(PART.glob(f"{args.tag}_test_f*.npy"))
    if parts: np.save(C / "oof" / f"cnn_{args.tag}_test.npy", np.mean([np.load(q) for q in parts], 0))
    json.dump(res, open(fj, "w"), indent=1)
