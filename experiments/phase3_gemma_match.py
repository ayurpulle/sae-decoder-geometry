"""Phase 3, Gemma-matched planted check (GPU): at Gemma's d = 2304 and width 16384, does planted data reach Gemma's
gamma(k) = 1.016 - 0.094·log2 k (TopK, layer 13, ranks 11-150; log2 k0 = 10.8)?

Planted point features, beta = 0, p_i ∝ i^-1.5, noise 0.05 per coordinate, 6M tokens. The d-sweep found density and
d set gamma's amplitude A while width and d move k0 only slowly (extrapolated log2 k0 ≈ 8.5 here), so:
  stage 1: three densities bracketing Gemma's intercept, (M/d, active/d) = (2, 1/64), (4, 1/32), (8, 1/16),
           k in K_SWEEP (Gemma's range), real vs Gaussian twin, seed 0                         -> 36 SAEs
  stage 2: the density whose real intercept in ranks 11-150 is nearest 1.016, seeds 1 and 2   -> 24 SAEs
Analysis (phase3_match_analyze.py) runs after each stage, so W_dec files never leave the pod.
Usage (on the pod): python experiments/phase3_gemma_match.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D, WIDTH = 2304, 16384
DENSITIES = [(2, 64), (4, 32), (8, 16)]  # M = a·d features, d/b active per token
K_SWEEP = [4, 8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 6_000_000, 500_000, 2
GROUP = "saes_p3g"
GEMMA_A = 1.016


def name(a, b):
    return f"g{D}-m{a * D}-a{D // b}"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def train(data, seeds):
    for k in K_SWEEP:
        for seed in seeds:
            for cond in ["real", "gaussian"]:
                run = f"synth-{data}__blocks.0.hook_resid_post__{cond}__w{WIDTH}__k{k}__s{seed}"
                if (ROOT / "cache" / GROUP / run / "train.json").exists():
                    continue
                py(ROOT / "train_sae.py", "--model", f"synth-{data}", "--hook", "blocks.0.hook_resid_post",
                   "--cond", cond, "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL,
                   "--epochs", EPOCHS, "--seed", seed, "--out", f"cache/{GROUP}")


def main():
    for a, b in DENSITIES:
        if not (ROOT / "cache" / "acts" / f"synth-{name(a, b)}" / "blocks.0.hook_resid_post" / "acts.npy").exists():
            py(ROOT / "synth.py", "--name", name(a, b), "--n_tokens", TOKENS, "--d", D, "--M", a * D,
               "--mean_active", D // b, "--alpha", 0.5, "--beta", 0, "--noise", 0.05)
    for a, b in DENSITIES:
        train(name(a, b), [0])
    py(ROOT / "experiments" / "phase3_match_analyze.py")
    fits = json.load(open(ROOT / "results" / "phase3" / "gemma_match.json"))["fits"]
    best = min((name(a, b) for a, b in DENSITIES),
               key=lambda n: abs(fits[f"{n}|real|0|(10, 150)"]["intercept"] - GEMMA_A))
    print(f"stage 2: seeds 1, 2 on {best}", flush=True)
    train(best, [1, 2])
    py(ROOT / "experiments" / "phase3_match_analyze.py")
    print("MATCH DONE", flush=True)


if __name__ == "__main__":
    main()
