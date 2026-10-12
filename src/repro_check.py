"""Bitwise reproducibility check: run the non-CNN pipeline twice into two fresh output roots and compare every artifact.

Usage: python src/repro_check.py <scratch_dir>     (both runs are written under <scratch_dir>/run_a and run_b)
"""
import hashlib, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
CHAIN = [["stage0_recon.py", "--force"], ["baseline_cheap.py"], ["features.py", "--force"], ["stage1_report.py"],
         ["partb_report.py"], ["blend.py", "--out", "003_blend_ridge_cheap_lgbm_s1.csv|ridge_cheap+lgbm_s1"]]


def run(out):
    out.mkdir(parents=True, exist_ok=True); env = dict(os.environ, HARDNESS_OUT=str(out)); t0 = time.time()
    for s in CHAIN:
        r = subprocess.run([sys.executable, "-W", "ignore", str(SRC / s[0]), *s[1:]], env=env, cwd=SRC.parent, capture_output=True, text=True)
        if r.returncode: raise SystemExit(f"{out.name}: {s[0]} failed\n{r.stderr[-2000:]}")
    return time.time() - t0


def digest(p):
    if p.suffix == ".parquet":                                  # compare parquet by content (file metadata may embed versions)
        return hashlib.sha256(pd.util.hash_pandas_object(pd.read_parquet(p), index=True).values.tobytes()).hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    base = Path(sys.argv[1]).resolve(); a, b = base / "run_a", base / "run_b"
    with ThreadPoolExecutor(2) as ex: ta, tb = ex.map(run, [a, b])
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.suffix in (".csv", ".npy", ".npz", ".parquet", ".json"))
    bad = [f for f in files if not (b / f).exists() or digest(a / f) != digest(b / f)]
    for f in files: print(f"{'OK  ' if f not in bad else 'DIFF'} {digest(a / f)[:16]}  {f}")
    print(f"\n{len(files) - len(bad)}/{len(files)} artifacts bitwise identical (run times {ta:.0f}s / {tb:.0f}s)")
    sys.exit(1 if bad else 0)
