"""Single-command reproduction of the selected submission (004).

  python src/run.py --mode full   # everything from raw images, incl. CNN training (~1.5-2 h on Apple M5 / MPS)
  python src/run.py --mode fast   # same, but reuse the cached CNN OOF/test arrays (cache/oof/cnn_r18_final_*)

Output: submissions/004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv  (sha256 1cabcbcf... when run on the reference setup)
Pipeline (all seeds fixed; pretrained weights loaded from ./weights, sha256-checked; no network needed):
  stage0_recon -> cheap global stats            features -> 114 nuisance-aware features (+ blur-corrected variants)
  baseline_cheap / recover_003 -> 001 and the 003 baseline (reference for paired tests)
  p12_phase / p12_pixphase / p12_orient -> mode-aware dark-phase fractions, coherence
  p13_descriptors / p13_structure -> grain-level descriptors          p13_latnet -> degradation-robust latent estimates (no y)
  cnn -> resnet18, D4+blur+noise, global norm, 15 folds               p13_build_004 -> 0.5*CNN + 0.5*Ridge -> submission
"""
import argparse, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent
CNN = ["cnn.py", "--tag", "r18_final", "--arch", "resnet18", "--aug", "d4,blur,noise", "--norm", "global", "--epochs", "40", "--seeds", "1", "--reps", "0,1,2"]
STEPS = [["stage0_recon.py"], ["baseline_cheap.py"], ["features.py"], ["recover_003.py"], ["p12_phase.py"], ["p12_pixphase.py"], ["p12_orient.py"],
         ["p13_descriptors.py"], ["p13_structure.py"], ["p13_latnet.py", "--final_only"]]
BUILD = ["p13_build_004.py", "--cnn", "r18_final", "--feats", "p12_desc_ln", "--name", "004_cnn_r18_d4bn_plus_ridge_desc_latnet.csv"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="full", choices=["full", "fast"]); a = ap.parse_args()
    if a.mode == "fast":                                                    # reuse the verified CNN arrays under the canonical tag
        import shutil; o = SRC.parent / "cache" / "oof"
        for k in ("oof", "test"):
            if not (o / f"cnn_r18_final_{k}.npy").exists(): shutil.copyfile(o / f"cnn_r18_final_a_{k}.npy", o / f"cnn_r18_final_{k}.npy")
    for s in STEPS + ([CNN] if a.mode == "full" else []) + [BUILD]:
        t0 = time.time(); print(f"\n>>> {' '.join(s)}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(SRC / s[0]), *s[1:]], check=True, cwd=SRC.parent, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
        print(f"<<< {s[0]} done in {time.time() - t0:.0f}s", flush=True)
