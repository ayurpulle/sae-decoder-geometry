"""Phase 3, clustered planted directions at Gemma's settings (GPU): does cluster structure in the feature directions
restore Gemma's gamma(k) = 1.016 - 0.094·log2 k in ranks 11-150 (TopK, L13, d = 2304, width 16384)?

Phase 3's power-law-Sigma check (phase3_powerlaw.py) gave a flat real gamma(k) with random directions. Phase 4 found
Gemma's decoder clouds carry cluster structure beyond their covariance, tracking the top-window layer profile. Same
generator as the power-law check (alpha 0.5, beta 0, noise 0.05, --var_slope 1.0, 12M tokens x 3 epochs) plus
--n_clusters M/16 --cluster_cos rho: features split evenly into clusters of 16, each at cosine rho to a random centre.
Calibration (phase3_cluster_calib.py, feature level): Sigma slopes move by <= 0.03 in ranks 4-450; excess
between-cluster share over a covariance-matched cloud is +0.063 at rho = 0.4 (Gemma's decoders: +0.054) and +0.29 at
rho = 0.6. The rho = 0 control is the power-law run (results/phase3/powerlaw_match.json).
  stage 1: rho in {0.4, 0.6} x densities (4d, d/32), (8d, d/16), real, k in K_SWEEP, seed 0       -> 24 SAEs
  stage 2: seed 1 on the dataset whose real gamma(k), ranks 11-150, is nearest Gemma's (RMS)     ->  6 SAEs
One dataset at a time: generate, train, analyse, delete acts.npy (stats.npz stays for the analysis); a dataset is
regenerated identically if stage 2 needs it. Analysis: phase3_match_analyze.py --group saes_p3c --out clustered_match
--clusters (gamma per window plus the decoders' own cluster excess).
Pre-registered reading: clusters restore Gemma's law if real gamma in ranks 11-150 declines with k with RMS vs Gemma
< 0.15 (power-law run's best: 0.27) and per-doubling slope within 0.04 of -0.094; a raised but flat gamma means
clusters set gamma's level, not its k-law.
Usage (on the pod): python experiments/phase3_clustered.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, WIDTH = 2304, 16384
SETTINGS = [(4, 32, 0.4), (8, 16, 0.4), (4, 32, 0.6), (8, 16, 0.6)]  # (M/d, d/active, rho)
K_SWEEP = [4, 8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT = "saes_p3c", "clustered_match"
GEMMA = {4: 0.824, 8: 0.732, 16: 0.665, 40: 0.509, 128: 0.354}  # L13 TopK 16k, ranks 11-150 (analysis.json B)


def name(a, b, rho):
    return f"c{D}-m{a * D}-a{D // b}-r{round(rho * 100)}"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def acts(n):
    return ROOT / "cache" / "acts" / f"synth-{n}" / "blocks.0.hook_resid_post" / "acts.npy"


def generate(a, b, rho):
    if not acts(name(a, b, rho)).exists():
        py(ROOT / "synth.py", "--name", name(a, b, rho), "--n_tokens", TOKENS, "--d", D, "--M", a * D,
           "--mean_active", D // b, "--alpha", 0.5, "--beta", 0, "--noise", 0.05, "--var_slope", 1.0,
           "--n_clusters", a * D // 16, "--cluster_cos", rho)


def done(n, seed):
    return all((ROOT / "cache" / GROUP / f"synth-{n}__blocks.0.hook_resid_post__real__w{WIDTH}__k{k}__s{seed}"
                / "train.json").exists() for k in K_SWEEP)


def train(n, seed):
    for k in K_SWEEP:
        run = f"synth-{n}__blocks.0.hook_resid_post__real__w{WIDTH}__k{k}__s{seed}"
        if (ROOT / "cache" / GROUP / run / "train.json").exists():
            continue
        py(ROOT / "train_sae.py", "--model", f"synth-{n}", "--hook", "blocks.0.hook_resid_post", "--cond", "real",
           "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS,
           "--seed", seed, "--out", f"cache/{GROUP}")


def analyze():
    py(ROOT / "experiments" / "phase3_match_analyze.py", "--group", GROUP, "--out", OUT, "--clusters")
    return json.load(open(ROOT / "results" / "phase3" / f"{OUT}.json"))["rows"]


def rms_vs_gemma(rows, data):
    g = {r["k"]: r["gamma"]["(10, 150)"] for r in rows
         if r["data"] == data and r["cond"] == "real" and r["seed"] == 0 and "(10, 150)" in r["gamma"]}
    if not all(k in g for k in K_SWEEP):
        return float("inf")
    t = np.log2(40) - 5
    g[40] = (1 - t) * g[32] + t * g[64]
    return float(np.sqrt(np.mean([(g[k] - v) ** 2 for k, v in GEMMA.items()])))


def main():
    for a, b, rho in SETTINGS:
        n = name(a, b, rho)
        if not done(n, 0):
            generate(a, b, rho)
            train(n, 0)
        analyze()
        acts(n).unlink(missing_ok=True)
        print(f"DATASET DONE {n}", flush=True)
    rows = analyze()
    print("STAGE 1 DONE", flush=True)
    rms = {name(*s): rms_vs_gemma(rows, name(*s)) for s in SETTINGS}
    best = min(SETTINGS, key=lambda s: rms[name(*s)])
    print(f"stage 2: seed 1 on {name(*best)} (RMS vs Gemma, ranks 11-150: {rms})", flush=True)
    if not done(name(*best), 1):
        generate(*best)
        train(name(*best), 1)
    analyze()
    acts(name(*best)).unlink(missing_ok=True)
    print("CLUSTERED DONE", flush=True)


if __name__ == "__main__":
    main()
