"""Run the round-2 real-data job and then the second-model job in sequence, under one launcher and one watchdog:
  bash gpu/run_job.sh experiments/r2_job_all.py <max_hours>
Arguments are passed to r2_job_models.py (default --cut). Needs more than 32 GB of GPU memory: the layer-13
k = 128 SAE (batch 8192 x k x d decode) peaks near 33 GB and failed twice on a 32 GB card on 2026-11-11. r2_job_real.py resumes from
its rows; r2_job_models.py has no resume, so finished model rows are re-run and the summary keeps the last of each.
Exit code is the worst of the two, and the second job runs even if the first fails."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
codes = []
for job, args in (("r2_job_real.py", []), ("r2_job_models.py", sys.argv[1:] or ["--cut"])):
    print(f"[r2_job_all] starting {job} {' '.join(args)}", flush=True)
    codes.append(subprocess.run([sys.executable, "-u", str(ROOT / "experiments" / job), *args], cwd=ROOT).returncode)
    print(f"[r2_job_all] {job} exited {codes[-1]}", flush=True)
sys.exit(max(abs(c) for c in codes))
