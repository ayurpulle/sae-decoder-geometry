"""Runs r2_job_pmatch2.py, then r2_job_ladder.py. The ladder starts only when the file ladder.GO exists in the repo
root (it is pushed with the ladder code); the chain waits up to 25 minutes for it. A failed first stage does not stop
the second."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    print("chain:", " ".join(map(str, args)), flush=True)
    return subprocess.run([sys.executable, "-u", *map(str, args)], cwd=ROOT).returncode


codes = [run(ROOT / "experiments" / "r2_job_pmatch2.py")]
t0 = time.time()
while not (ROOT / "ladder.GO").exists() and time.time() - t0 < 25 * 60:
    time.sleep(20)
if (ROOT / "ladder.GO").exists():
    extra = (ROOT / "ladder.GO").read_text().split()
    codes.append(run(ROOT / "experiments" / "r2_job_ladder.py", *extra))
else:
    print("chain: no ladder.GO after 25 min; ladder not run", flush=True)
print("chain: exit codes", codes, flush=True)
sys.exit(0 if all(c == 0 for c in codes) else 1)
