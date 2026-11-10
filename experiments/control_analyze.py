"""Score SAEs in a cache group at a fixed number of live latents, in gamma and raw decoder-slope units (runs on the pod).

The decoder eigen-slope of a cloud of n points is biased toward flat when n is small next to d (a cloud with true
gamma 0.5 measures 0.32 at n = 672, 0.41 at 4,786, 0.47 at 16,384), and the live count rises with k. So besides the
full live cloud, each SAE is scored on N_FIX randomly chosen live latents (mean of N_DRAWS draws), which fixes n
across k and across variants.

Per run and window: Sigma slope (training data's covariance), decoder slope (raw units) and gamma = decoder / Sigma,
for the full live cloud and the fixed-n cloud. Writes results/phase3/<out>.json.
Usage: python experiments/control_analyze.py --group saes_ctl --out control
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import _eigvals, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450)]
N_FIX, N_DRAWS = 8192, 4


def slopes(eig):
    return {str(w): slope_from_eigenvalues(eig, w)["slope"] for w in WINDOWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sig, rows = {}, []
    for tj in sorted(glob.glob(str(ROOT / "cache" / args.group / "*" / "train.json"))):
        rd = Path(tj).parent
        info = json.load(open(tj))
        c = info["config"]
        if c["model"] not in sig:
            cov = np.load(ROOT / "cache" / "acts" / c["model"] / c["hook"] / "stats.npz")["cov"]
            sig[c["model"]] = np.sort(np.linalg.eigvalsh(cov))[::-1]
        s_sig = slopes(sig[c["model"]])
        live = np.load(rd / "fires.npy") > 0
        W = np.load(rd / "W_dec.npy")[live]
        e_full = _eigvals(W, center=True)[0]
        rng = np.random.default_rng(c["k"] * 100 + c["seed"])
        draws = [_eigvals(W[rng.choice(len(W), N_FIX, replace=False)], center=True)[0]
                 for _ in range(N_DRAWS)] if len(W) >= N_FIX else []
        full = slopes(e_full)
        fixed = [slopes(e) for e in draws]
        row = {"data": c["model"].removeprefix("synth-"), "k": c["k"], "seed": c["seed"], "fvu": info["fvu_eval"],
               "alive": int(live.sum()), "sigma_slope": s_sig, "dec_full": full,
               "gamma_full": {w: full[w] / s_sig[w] for w in full},
               "dec_fixed": {w: float(np.mean([f[w] for f in fixed])) for w in full} if fixed else None,
               "dec_fixed_sd": {w: float(np.std([f[w] for f in fixed])) for w in full} if fixed else None,
               "eig_full": e_full[:600].tolist(),
               "eig_fixed": np.mean(draws, 0)[:600].tolist() if draws else None}
        row["gamma_fixed"] = {w: row["dec_fixed"][w] / s_sig[w] for w in full} if fixed else None
        rows.append(row)
        g, gf = row["gamma_full"]["(10, 150)"], row["gamma_fixed"]
        print(f"{row['data']:<12} k{c['k']:<4} s{c['seed']} alive {row['alive']:>5}  ranks 11-150: gamma full {g:.3f}"
              + (f", fixed-n {gf['(10, 150)']:.3f}" if gf else ", fixed-n n/a"), flush=True)
    gh = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not gh and (ROOT / ".git_hash").exists():  # the pod gets the code without .git (gpu/push.sh)
        gh = (ROOT / ".git_hash").read_text().strip()
    out = ROOT / "results" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"git": gh, "windows": WINDOWS, "n_fix": N_FIX, "n_draws": N_DRAWS,
               "sigma_eigenvalues": {m: e[:600].tolist() for m, e in sig.items()}, "rows": rows},
              open(out / f"{args.out}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
