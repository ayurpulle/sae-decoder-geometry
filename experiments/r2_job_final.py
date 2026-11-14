"""Run the codable sparse null job and then the Gemma-2-9b check in sequence, under one launcher and one watchdog:
  bash gpu/run_job.sh experiments/r2_job_final.py <max_hours>
On a 32 GB card: r2_job_sparse2.py --skip_k 128 (the k = 128 SAE peaks near 33 GB), then r2_job_models.py --skip_m1
(M2 resumes from its rows). Both resume, so a restart repeats nothing. Exit code is the worst of the two, and the
second job runs even if the first fails."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
codes = []
for job, args in (("r2_job_sparse2.py", sys.argv[1:] or ["--skip_k", "128"]), ("r2_job_models.py", ["--skip_m1"])):
    print(f"[r2_job_final] starting {job} {' '.join(args)}", flush=True)
    codes.append(subprocess.run([sys.executable, "-u", str(ROOT / "experiments" / job), *args], cwd=ROOT).returncode)
    print(f"[r2_job_final] {job} exited {codes[-1]}", flush=True)
sys.exit(max(abs(c) for c in codes))
