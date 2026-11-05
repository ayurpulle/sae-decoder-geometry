"""Phase 3, power-law-Sigma check at Gemma's settings (GPU): with Sigma a power law from rank 1, does planted data
reproduce Gemma's gamma(k) = 1.016 - 0.094·log2 k in ranks 11-150 (TopK, L13, d = 2304, width 16384)?

The Gemma-matched check (phase3_gemma_match.py) could not test ranks 11-150: the P_MAX = 0.5 frequency cap saturated
the top 25-101 features, giving Sigma a plateau there. Here synth.py --var_slope 1.0 sets each feature's magnitude so its
variance ∝ i^-1, which gives Sigma eigen-slopes -0.95 / -0.90 / -0.82 / -0.79 in ranks 4-50 / 11-150 / 31-450 / 101-1000
at every density (Gemma: -0.98 / -0.95 / -0.75 / -0.70). Frequencies are unchanged (p_i ∝ i^-1.5, capped at 0.5).
Also 12M tokens x 3 epochs (3x the steps of the matched check) so small-k SAEs keep more latents alive.
  stage 1: real data, three densities (M/d, active/d) = (2, 1/64), (4, 1/32), (8, 1/16), k in K_SWEEP, seed 0  -> 18
  stage 2: Gaussian twin of the middle density (the twins' Sigma shape is the same at every density)          ->  6
  stage 3: seed 1 on the density whose real gamma(k) in ranks 11-150 is nearest Gemma's (RMS over k)          ->  6
Analysis: phase3_match_analyze.py --group saes_p3p --out powerlaw_match, after each stage.
Usage (on the pod): python experiments/phase3_powerlaw.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, WIDTH = 2304, 16384
DENSITIES = [(2, 64), (4, 32), (8, 16)]  # M = a·d features, d/b active per token
K_SWEEP = [4, 8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT = "saes_p3p", "powerlaw_match"
# Gemma L13 TopK 16k, ranks 11-150, mean of 2 seeds (results/gpu/analysis.json, section B); k = 40 compared with the
# planted gamma interpolated in log2 k between 32 and 64.
GEMMA = {4: 0.824, 8: 0.732, 16: 0.665, 40: 0.509, 128: 0.354}


def name(a, b):
    return f"p{D}-m{a * D}-a{D // b}"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def train(data, cond, seeds):
    for k in K_SWEEP:
        for seed in seeds:
            run = f"synth-{data}__blocks.0.hook_resid_post__{cond}__w{WIDTH}__k{k}__s{seed}"
            if (ROOT / "cache" / GROUP / run / "train.json").exists():
                continue
            py(ROOT / "train_sae.py", "--model", f"synth-{data}", "--hook", "blocks.0.hook_resid_post",
               "--cond", cond, "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL,
               "--epochs", EPOCHS, "--seed", seed, "--out", f"cache/{GROUP}")


def analyze():
    py(ROOT / "experiments" / "phase3_match_analyze.py", "--group", GROUP, "--out", OUT)
    return json.load(open(ROOT / "results" / "phase3" / f"{OUT}.json"))["rows"]


def rms_vs_gemma(rows, data):
    g = {r["k"]: r["gamma"]["(10, 150)"] for r in rows
         if r["data"] == data and r["cond"] == "real" and r["seed"] == 0 and "(10, 150)" in r["gamma"]}
    if not all(k in g for k in (4, 8, 16, 32, 64, 128)):
        return float("inf")
    t = (np.log2(40) - 5) / 1
    g[40] = (1 - t) * g[32] + t * g[64]
    return float(np.sqrt(np.mean([(g[k] - v) ** 2 for k, v in GEMMA.items()])))


def main():
    for a, b in DENSITIES:
        if not (ROOT / "cache" / "acts" / f"synth-{name(a, b)}" / "blocks.0.hook_resid_post" / "acts.npy").exists():
            py(ROOT / "synth.py", "--name", name(a, b), "--n_tokens", TOKENS, "--d", D, "--M", a * D,
               "--mean_active", D // b, "--alpha", 0.5, "--beta", 0, "--noise", 0.05, "--var_slope", 1.0)
    for a, b in DENSITIES:
        train(name(a, b), "real", [0])
    analyze()
    print("STAGE 1 DONE", flush=True)
    train(name(4, 32), "gaussian", [0])
    rows = analyze()
    print("STAGE 2 DONE", flush=True)
    rms = {name(a, b): rms_vs_gemma(rows, name(a, b)) for a, b in DENSITIES}
    best = min(rms, key=rms.get)
    print(f"stage 3: seed 1 on {best} (RMS vs Gemma, ranks 11-150: {rms})", flush=True)
    train(best, "real", [1])
    analyze()
    print("POWERLAW DONE", flush=True)


if __name__ == "__main__":
    main()
