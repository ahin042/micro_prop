"""Single-command reproduction: python src/run.py --mode full   (fast = reuse cached CNN OOF/test predictions)."""
import argparse, subprocess, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent
CNN = "r18_final"   # final CNN config tag (see experiments.md); its arguments are in CNN_ARGS
CNN_ARGS = ["--arch", "resnet18", "--aug", "full", "--norm", "global", "--epochs", "40", "--repeats", "3", "--seeds", "3"]
STEPS = [["stage0_recon.py"], ["baseline_cheap.py"], ["features.py"], ["stage1_report.py"], ["stage2_formula.py"],
         ["ceiling.py"], ["partb_report.py"], ["stationarity.py"],
         ["blend.py", "--out", "003_blend_ridge_cheap_lgbm_s1.csv|ridge_cheap+lgbm_s1"]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="full", choices=["full", "fast"]); a = ap.parse_args()
    steps = STEPS + ([["cnn.py", "--tag", CNN, *CNN_ARGS]] if a.mode == "full" else []) + [["blend.py", "--cnn", CNN]]
    for s in steps:
        t0 = time.time(); print(f"\n>>> {' '.join(s)}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(SRC / s[0]), *s[1:]], check=True, cwd=SRC.parent)
        print(f"<<< {s[0]} done in {time.time() - t0:.0f}s", flush=True)
