"""Runs the feature-power step for real SAEs one layer at a time (one 46 GB activation cache on disk at once),
then the power-matched null job. Each stage is resumable on its own; a failed stage does not stop the next layer."""
import subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(*args):
    print("chain:", " ".join(map(str, args)), flush=True)
    return subprocess.run([PY, "-u", *map(str, args)], cwd=ROOT).returncode


def drop(layer):
    d = ROOT / "cache" / "acts" / "gemma-2-2b" / f"blocks.{layer}.hook_resid_post"
    for n in ("acts.npy", "acts.npy.part"):
        p = d / n
        if p.exists():
            p.unlink()
            print(f"chain: removed {p}", flush=True)


codes = []
for layer in (13, 0):
    drop(13), drop(0)
    codes.append(run(ROOT / "experiments" / "r2_job_robust.py", "--skip_g", "--skip_m", "--p_layers", layer))
drop(13), drop(0)
codes.append(run(ROOT / "experiments" / "r2_job_pmatch.py"))
print("chain: exit codes", codes, flush=True)
sys.exit(0 if codes[-1] == 0 else 1)
