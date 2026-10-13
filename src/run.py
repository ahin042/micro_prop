"""Single-command reproduction of the final-candidate submissions 004 / 005 / 006.

  python src/run.py --mode full [--target all]   # everything from raw images, incl. all CNN training (~4 h on Apple M5 / MPS)
  python src/run.py --mode fast [--target all]   # same, but reuse the cached CNN OOF/test arrays from ./cache/oof

Set HARDNESS_OUT=<dir> to write every output (cache/, reports/, submissions/) into a fresh folder instead of the repo.
Reference outputs (sha256 on the reference setup):
  004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv  1cabcbcf...   0.5*CNN r18 + 0.5*Ridge[features + latent net]
  005_eq3_cnn_r18_ridge004_dinov2s.csv         d082a949...   mean(CNN r18, Ridge004, Ridge[frozen DINOv2-S])
  006_eq3_cnn_r18up2full_ridge004_dinov2s.csv  fdf94275...   mean(CNN r18 2x-upsampled [5 full-data seeds], Ridge004, Ridge[DINOv2-S])
Pipeline (all seeds fixed; pretrained weights loaded from ./weights, sha256-checked; no network needed):
  stage0_recon -> cheap global stats            features -> nuisance-aware features (+ blur-corrected variants)
  baseline_cheap / recover_003 -> 001 and the 003 baseline (reference for paired tests)
  p12_phase / p12_pixphase / p12_orient -> mode-aware dark-phase fractions, coherence
  p13_descriptors / p13_structure -> grain-level descriptors          p13_latnet -> degradation-robust latent estimates (no y)
  cnn -> resnet18, D4+blur+noise, global norm, 15 folds               p13_build_004 -> 004
  p17_dino_probe -> frozen DINOv2-S embedding (inference only)        p17_build -> 005, 006
  cnn2 -> resnet18 with 2x input upsampling: 15-fold OOF (CV only) + 5 full-data seeds (test predictions of 006)
"""
import argparse, os, shutil, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent; ROOT = SRC.parent
OUT = ROOT / os.environ.get("HARDNESS_OUT", ".")
CNN = ["cnn.py", "--tag", "r18_final", "--arch", "resnet18", "--aug", "d4,blur,noise", "--norm", "global", "--epochs", "40", "--seeds", "1", "--reps", "0,1,2"]
UP2 = ["cnn2.py", "--tag", "r18up2_final", "--arch", "r18a1", "--up", "2", "--aug", "d4,blur,noise", "--norm", "global"]
STEPS = [["stage0_recon.py"], ["baseline_cheap.py"], ["features.py"], ["recover_003.py"], ["p12_phase.py"], ["p12_pixphase.py"], ["p12_orient.py"],
         ["p13_descriptors.py"], ["p13_structure.py"], ["p13_latnet.py", "--final_only"]]
B004 = ["p13_build_004.py", "--cnn", "r18_final", "--feats", "p12_desc_ln", "--name", "004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv"]
DINO = ["p17_dino_probe.py", "--arch", "dinos", "--sizes", "252", "--threads", "4"]
B005 = ["p17_build.py", "--cnn", "r18_final", "--dino", "dinos:252", "--ref_cnn", "r18_final", "--name", "005_eq3_cnn_r18_ridge004_dinov2s.csv"]
B006 = ["p17_build.py", "--cnn", "r18up2_final:full", "--dino", "dinos:252", "--ref_cnn", "r18_final", "--name", "006_eq3_cnn_r18up2full_ridge004_dinov2s.csv"]
# cached CNN arrays reused by --mode fast: (source in ./cache/oof, name expected in the output root)
FAST = [("cnn_r18_final_a_oof.npy", "cnn_r18_final_oof.npy"), ("cnn_r18_final_a_test.npy", "cnn_r18_final_test.npy"),
        ("cnn_r18up2_final_oof.npy", "cnn_r18up2_final_oof.npy"), ("cnn_r18up2_final_full_test.npy", "cnn_r18up2_final_full_test.npy")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="full", choices=["full", "fast"])
    ap.add_argument("--target", default="all", choices=["004", "005", "006", "all"]); a = ap.parse_args()
    want = {"004", "005", "006"} if a.target == "all" else {a.target}
    plan = list(STEPS)
    if a.mode == "full":
        plan += [CNN]                                                       # always: 004's CNN is also the reference in every paired test
    else:                                                                   # reuse the verified CNN arrays under the canonical tags
        (OUT / "cache" / "oof").mkdir(parents=True, exist_ok=True)
        for src, dst in FAST:
            if not (OUT / "cache" / "oof" / dst).exists(): shutil.copyfile(ROOT / "cache" / "oof" / src, OUT / "cache" / "oof" / dst)
    plan += [B004] + ([DINO] if want & {"005", "006"} else []) + ([B005] if "005" in want else [])    # 004/005 never wait on the 006 CNN
    if "006" in want:
        plan += ([UP2 + ["--reps", "0,1,2", "--no_test"], UP2 + ["--full", "5"]] if a.mode == "full" else []) + [B006]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.6", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5"}  # MPS memory cap (16 GB machine)
    for s in plan:
        t0 = time.time(); print(f"\n>>> {' '.join(s)}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(SRC / s[0]), *s[1:]], check=True, cwd=ROOT, env=env)
        print(f"<<< {s[0]} done in {time.time() - t0:.0f}s", flush=True)
