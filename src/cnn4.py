"""Phase 20: CNN v4 = src/cnn3.py (frozen: it reproduces 007) + DINOv2-B (--arch dinob), --head mlp, --dump_emb (pre-head embedding
of every image from each fold's own model, D4 mean), --fis (run selected global folds 0-14 only), automatic batch fallback 32 -> 16 -> 8
on MPS out-of-memory, cache release between folds, and a copy of the OOF/test arrays in predictions/. With the default options the
numerics are those of cnn3.py. Original cnn3 docstring:
Phase 19: CNN v3 = src/cnn2.py (frozen: it reproduces 006) + MicroNet backbones (MIT, allowed by the official Q&A of 2026-10-13) and
DINOv2 fine-tuning options (--llrd layer-wise LR decay, --pool avg). Original cnn2 docstring:
Phase 17: CNN v2 = src/cnn.py (frozen, it reproduces 004) + options for the underfitting diagnosis:
--up (bilinear input upsampling; the stem downsamples 4x and boundaries are 1-3 px wide), --amp fp32|bf16, --wd, --drop,
and more backbones (vendored in weights/, sha256-checked):
  r18a1 = timm/resnet18.a1_in1k (Apache-2.0) | r18tv = torchvision resnet18 IMAGENET1K_V1 (BSD-3-Clause) | eb0 = timm/efficientnet_b0.ra_in1k (Apache-2.0).
Same RepeatedKFold(5x3, seed 42) image folds as every other model; OOF saved as (n_repeats, n) for blending.
No early stopping on the reported fold (that would leak): fixed cosine schedule, final EMA weights, D4 TTA.
"""
import argparse, gc, json, math, random, shutil, time

import numpy as np
import timm
import torch
import torch.nn.functional as Fn
from PIL import Image

from common import C, D, REP, ROOT, SEED, load_meta, sha256, splits

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="r18a1"); ap.add_argument("--up", type=float, default=1); ap.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--llrd", type=float, default=1.0, help="ViT layer-wise LR decay (1 = off)"); ap.add_argument("--pool", default="", help="ViT global_pool override, e.g. avg"); ap.add_argument("--size", type=int, default=0, help="resize input to size x size (overrides --up)"); ap.add_argument("--drop", type=float, default=.1); ap.add_argument("--norm", default="global", choices=["global", "image"])
ap.add_argument("--aug", default="full", help="comma list of d4,blur,noise,photo | full | none")
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--reps", default="0", help="repeat indices to train, e.g. 0,1,2 (results merge with earlier runs of the same tag)")
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--full", type=int, default=0, help="K>0: also train K seeds on ALL training images -> cnn_{tag}_full_test.npy (no OOF; the learning curve is steep)");
ap.add_argument("--tag", required=True); ap.add_argument("--no_test", action="store_true")
ap.add_argument("--head", default="linear", choices=["linear", "mlp"], help="mlp = Linear(d,256)-GELU-Dropout(.1)-Linear(256,1), trained end-to-end at the head lr")
ap.add_argument("--dump_emb", action="store_true", help="save the pre-head embedding (D4 mean) of all train+test images from each fold's model")
ap.add_argument("--fis", default="", help="comma list of global fold indices 0-14 to run (overrides --reps/--folds); results merge with earlier runs")
ap.add_argument("--seed", type=int, default=SEED, help="base seed (default 42 = the seed of 004-007); fold model s uses seed + s, full-data seed s uses seed + 100 + s")
ap.add_argument("--skip_done", action="store_true", help="skip folds already recorded in reports/cnn_{tag}.json (resume after a crash)")
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)   # global seeds; every model re-seeds in train_one (no DataLoader / workers)
AUG = {"full": {"d4", "blur", "noise", "photo"}, "none": set()}.get(args.aug, set(args.aug.split(",")))
torch.use_deterministic_algorithms(True, warn_only=True)            # deterministic kernels wherever the backend has one
WEIGHTS = {"r18a1": ("resnet18", "resnet18.a1_in1k.safetensors", "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612"),
           "r18tv": ("resnet18", "resnet18.tv_in1k.pth", "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
           "eb0": ("efficientnet_b0", "efficientnet_b0.ra_in1k.safetensors", "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"),
           "cnxt": ("convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k.safetensors", "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a"),
           "ev2s": ("tf_efficientnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k.safetensors", "6f1933fb6c0d760eae03863ad0110570393e16c0610f8fe94dc9f978e52ec59c"),
           "r50tv2": ("resnet50", "resnet50.tv2_in1k.safetensors", "30f57ada2f100011a4bd38000df906d6354688a8ca599be580337151c3c5f6a3"),
           "mn18": ("resnet18", "micronet_resnet18_micronet.pth", "2433454b3a5003b7f3b3beffa15d11bbbb2591dd23759b10d0939853ceac6dd3"),
           "mn18in": ("resnet18", "micronet_resnet18_imagenet_micronet.pth", "f9ba47f142a81585f8980bd08c85da807f46dcf2db449c439954adbeb44ba82d"),
           "mn50in": ("resnet50", "micronet_resnet50_imagenet_micronet.pth", "1eb701119532e8259b98169bdbdacee24efb8e5cbc62b13b407e40ff4244d42b"),
           "dinos": ("vit_small_patch14_dinov2", "vit_small_patch14_dinov2.lvd142m.safetensors", "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"),
           "dinob": ("vit_base_patch14_dinov2", "vit_base_patch14_dinov2.lvd142m.safetensors", "55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b")}
# licenses / HF commits: src/weights_manifest.json and README (convnext, effnetv2, dinov2: Apache-2.0; resnet50.tv2, resnet18.tv: BSD-3-Clause)
ARCH, wn, wh = WEIGHTS[args.arch]; wf = ROOT / "weights" / wn
assert wf.exists() and sha256(wf) == wh, f"missing or modified weights file {wf}"
OVL = dict(file=str(wf))
AMP = torch.bfloat16 if args.amp == "bf16" else None
KW = dict(img_size=args.size or int(256 * args.up), **(dict(global_pool=args.pool) if args.pool else {})) if ARCH.startswith("vit") else {}
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if dev.type == "cuda" and AMP is not None and torch.cuda.get_device_capability()[0] < 8:   # e.g. T4 / P100: no native bf16 -> fp32
    print(f"[{args.tag}] {torch.cuda.get_device_name()} has no native bf16 -> running fp32", flush=True); AMP = None; args.amp = "fp32"
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
    x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-4) if args.norm == "image" else (x - GM) / GS
    if args.size: return Fn.interpolate(x, size=(args.size, args.size), mode="bilinear", align_corners=False)
    return Fn.interpolate(x, scale_factor=args.up, mode="bilinear", align_corners=False) if args.up != 1 else x


def make_model(pretrained, **kw):
    m = timm.create_model(ARCH, pretrained=pretrained, in_chans=1, num_classes=1, **({"pretrained_cfg_overlay": OVL} if pretrained else {}), **kw, **KW).to(dev)
    if args.head == "mlp":
        d = m.num_features; m.head = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1)).to(dev)
    return m


@torch.no_grad()
def embed(model, X, bs=50):
    """Pre-head embedding (pooled tokens after fc_norm), mean over the 8 D4 views, same input pipeline as predict()."""
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs].float() / 255)
        out.append(torch.stack([model.forward_head(model.forward_features(d4(x, k, f)), pre_logits=True).float() for k in range(4) for f in (0, 1)]).mean(0))
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def predict(model, X, bs=50):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs].float() / 255)
        out.append(torch.stack([model(d4(x, k, f)).float().squeeze(1) for k in range(4) for f in (0, 1)]).mean(0))
    return torch.cat(out).cpu().numpy()


def train_one(a, b, seed):
    torch.manual_seed(seed); np.random.seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    mu, sd = y[a].mean(), y[a].std(); yt = torch.tensor((y[a] - mu) / sd, dtype=torch.float32, device=dev)
    model = make_model(True, drop_rate=args.drop)
    if ARCH.startswith("vit") and args.pool == "avg":                     # avg pooling uses a fresh fc_norm: start it from the pretrained final norm
        from safetensors.torch import load_file; sdw = load_file(str(wf))
        with torch.no_grad(): model.fc_norm.weight.copy_(sdw["norm.weight"].to(dev)); model.fc_norm.bias.copy_(sdw["norm.bias"].to(dev))
    ema = make_model(False); ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if ARCH.startswith("vit") and args.llrd < 1:                          # layer-wise LR decay: head lr, block i lr * llrd^(depth - i), embeddings lowest
        nb = len(model.blocks); depth = lambda n: 0 if n.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token")) else (int(n.split(".")[1]) + 1 if n.startswith("blocks.") else nb + 1)
        by = {}
        for n, p_ in model.named_parameters(): by.setdefault(depth(n), []).append(p_)
        groups = [{"params": ps, "lr": args.lr * args.llrd ** (nb + 1 - d)} for d, ps in sorted(by.items())]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    spe = math.ceil(len(a) / args.bs); tot = spe * args.epochs; warm = spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (tot - warm))))
    step = 0; curve = []
    for ep in range(args.epochs):
        model.train(); ls = []; perm = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed * 1000 + ep))
        for i in range(spe):
            bi = perm[i * args.bs:(i + 1) * args.bs]                      # minibatch positions within a (b = val indices)
            if len(bi) < 2: continue
            xb = norm(augment(Xtr[a[bi.numpy()]].float() / 255, g))
            with torch.autocast(device_type=dev.type, dtype=AMP or torch.float32, enabled=AMP is not None):
                loss = Fn.mse_loss(model(xb).float().squeeze(1), yt[bi.to(dev)])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1; ls.append(loss.item())
            dcy = min(0.995, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                    pe.copy_(pe * dcy + pm.detach() * (1 - dcy)) if pe.dtype.is_floating_point else pe.copy_(pm)
        if (ep + 1) % 5 == 0:                                            # diagnostics only, never used for selection
            ema.eval()
            with torch.no_grad(): pv = torch.cat([ema(norm(Xtr[b[i:i + 50]].float() / 255)).float().squeeze(1) for i in range(0, len(b), 50)]).cpu().numpy()
            curve.append([ep + 1, float(np.sqrt(np.mean(ls)) * sd), float(np.sqrt(np.mean((pv * sd + mu - y[b]) ** 2)))])
        if dev.type == "mps": torch.mps.empty_cache()                    # return cached blocks (12 GB at --up 2 otherwise); no effect on numerics
    del model, opt, sch; gc.collect()
    if dev.type == "mps": torch.mps.empty_cache()
    return ema, mu, sd, curve


BS_TRIED = []


def train_fallback(a, b, seed):
    """train_one with batch fallback 32 -> 16 -> 8 on (MPS) out-of-memory; the batch used is logged in the report json."""
    for bs in [args.bs] + [v for v in (16, 8) if v < args.bs]:
        args.bs = bs
        try:
            r = train_one(a, b, seed); BS_TRIED.append(bs); return r
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            print(f"[{args.tag}] out of memory at batch {bs} -> retry smaller", flush=True); gc.collect()
            if dev.type == "mps": torch.mps.empty_cache()
    raise RuntimeError("out of memory at batch 8")


if args.full:                                                           # full-data models for the test predictions only
    allidx = np.arange(len(y)); tv = []
    for s in range(args.full):
        t0 = time.time(); model, mu, sd, _ = train_one(allidx, allidx[:50], args.seed + 100 + s); tv.append(predict(model, Xte) * sd + mu)
        np.save(C / "oof" / f"cnn_{args.tag}_full_test.npy", np.mean(tv, 0)); print(f"[{args.tag}] full-data seed {s} done ({time.time() - t0:.0f}s)", flush=True)
        del model; gc.collect()                                            # free this seed before the next one (MPS OOM otherwise); numerics unchanged
        if dev.type == "mps": torch.mps.empty_cache()
    raise SystemExit

# OOF and test use the SAME predict(): full 256x256 image, 8 D4 transforms, mean of de-standardized outputs, EMA weights
reps = [int(r) for r in args.reps.split(",")]; PART = C / "oof" / "cnn_parts"; PART.mkdir(parents=True, exist_ok=True)
fo, fj = C / "oof" / f"cnn_{args.tag}_oof.npy", REP / f"cnn_{args.tag}.json"
oof = np.full((3, len(y)), np.nan); prev = json.load(open(fj)) if fj.exists() else {}
if fo.exists(): o = np.load(fo); oof[:len(o)] = o                       # merge with earlier repeats of this tag
folds = prev.get("folds") or {str(i): {"rmse": v} for i, v in enumerate(prev.get("fold_rmse", []))}


def save_all(final=False):                                           # called after every fold: a crash loses at most the running fold
    fr = [folds[k]["rmse"] for k in sorted(folds, key=int)]
    res = dict(config={**prev.get("config", {}), **vars(args)}, aug=sorted(AUG), device=str(dev), batch_used=prev.get("batch_used", []) + BS_TRIED, folds=folds, fold_rmse=fr,
               mean=float(np.mean(fr)), std=float(np.std(fr)), n_folds=len(fr), sec=prev.get("sec", 0) + time.time() - t0)
    if final: print(f"[{args.tag}] RMSE {res['mean']:.3f} +/- {res['std']:.3f} over {len(fr)} folds")
    np.save(fo, oof[:max(int(k) for k in folds) // 5 + 1])
    parts = sorted(PART.glob(f"{args.tag}_test_f*.npy"))
    if parts: np.save(C / "oof" / f"cnn_{args.tag}_test.npy", np.mean([np.load(q) for q in parts], 0))
    json.dump(res, open(fj, "w"), indent=1)
    PRED = ROOT / "predictions"; PRED.mkdir(exist_ok=True)                 # copy of the arrays (the pipeline reads cache/oof)
    for f in (fo, C / "oof" / f"cnn_{args.tag}_test.npy"):
        if f.exists(): shutil.copyfile(f, PRED / f.name)


t0 = time.time(); FIS = {int(v) for v in args.fis.split(",")} if args.fis else None; Xall = torch.cat([Xtr, Xte])
for fi, (a, b) in [(i, ab) for i, ab in enumerate(splits(len(y))) if (i in FIS if FIS is not None else (i // 5 in reps and i % 5 < args.folds))]:
    if args.skip_done and str(fi) in folds: continue
    pv, tv, cv_, trr = [], [], [], []
    for s in range(args.seeds):
        model, mu, sd, curve = train_fallback(a, b, args.seed + s); pv.append(predict(model, Xtr[b]) * sd + mu); cv_.append(curve)
        trr.append(float(np.sqrt(np.mean((predict(model, Xtr[a]) * sd + mu - y[a]) ** 2))))
        if not args.no_test: tv.append(predict(model, Xte) * sd + mu)
        if args.dump_emb and s == 0: np.save(PART / f"{args.tag}_emb_f{fi}.npy", embed(model, Xall))      # rows: train images (itr order), then test
        del model; gc.collect()
        if dev.type == "mps": torch.mps.empty_cache()
    oof[fi // 5, b] = np.mean(pv, 0); rm = float(np.sqrt(np.mean((oof[fi // 5, b] - y[b]) ** 2)))
    if tv: np.save(PART / f"{args.tag}_test_f{fi}.npy", np.mean(tv, 0))
    folds[str(fi)] = dict(rmse=rm, train_rmse=float(np.mean(trr)), curve=cv_[0])
    print(f"[{args.tag}] fold {fi} rmse {rm:.3f} train {np.mean(trr):.3f} ({time.time() - t0:.0f}s)", flush=True); save_all()
if folds: save_all(final=True)
