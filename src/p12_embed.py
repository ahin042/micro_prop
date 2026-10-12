"""Phase 12 / Step 3e: frozen local timm embeddings, full 256x256 raw grayscale input.

The image is not per-image normalized. It is only mapped through the fixed ImageNet
channel mean/std expected by the pretrained backbone.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import C, D, load_meta

BACKBONES = {
    "resnet18": ("resnet18.a1_in1k", "resnet18.a1_in1k.safetensors"),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k", "efficientnet_b0.ra_in1k.safetensors"),
}


def extract(tag, batch_size):
    import timm
    import torch

    torch.set_num_threads(1)
    M, itr, _, _ = load_meta()
    ids = M.ID.values[itr]
    name, weight_name = BACKBONES[tag]
    ckpt = C.parent / "weights" / weight_name
    if not ckpt.exists():
        raise FileNotFoundError(f"missing local weights: {ckpt}")

    model = timm.create_model(name, pretrained=True, features_only=True,
                              pretrained_cfg_overlay={"file": str(ckpt)}).eval()
    mean = np.array(model.pretrained_cfg["mean"], dtype=np.float32)
    std = np.array(model.pretrained_cfg["std"], dtype=np.float32)
    out = []
    with torch.no_grad():
        for k in range(0, len(ids), batch_size):
            batch_ids = ids[k:k + batch_size]
            a = np.stack([np.asarray(Image.open(D / "train" / f"{i}.png")) for i in batch_ids]).astype(np.float32) / 255.0
            x = ((a[:, None] - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
            z = model(torch.from_numpy(x))
            pooled = torch.cat([torch.cat([u.mean((2, 3)), u.std((2, 3))], 1) for u in z], 1).cpu().numpy()
            out.append(pooled)
            print(f"{tag}: {min(k + batch_size, len(ids))}/{len(ids)}", flush=True)
    P = C / "p12"
    P.mkdir(exist_ok=True)
    np.save(P / f"emb_{tag}.npy", np.concatenate(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=sorted(BACKBONES), required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    extract(args.tag, args.batch_size)
