"""Phase 3, stage 2: sparse, mildly superposed planted data, where features are recoverable.

Stage 1 (4,096 features, 32 active, beta = 1) was too Gaussian-like: feature directions crowded into ~37 effective
dimensions, the Gaussian twin matched real, and few latents matched a feature. Here:
  d = 256, M = 512 features (M/d = 2), 4 active per token, p_i ∝ i^-1.5, isotropic directions (beta = 0), noise 0.05.
With near-orthogonal directions, Sigma ≈ sum_i p_i E[m^2] u_i u_i^T, so Sigma's spectrum is the frequency law itself
(slope ≈ -1.5) rather than direction anisotropy. If each feature f_i gets n_i latents at directions near u_i, the
decoder cloud's eigenvalues are ∝ n_i, so gamma equals the allocation exponent c in n_i ∝ p_i^c. That makes gamma
and the frequency-allocation law (2509.02565) the same measurement, testable directly by latent-to-feature matching.
  s2-point:  point features (one direction each). Expect one latent per feature: c ≈ 0, gamma ≈ 0.
  s2-circle: every feature a circle in span(u_i, v_i). Tiling a circle with n directions leaves mean squared error
             ∝ n^-2 (b = 2), so the allocation law predicts c = 1/(1 + b) = 1/3 (our derivation, k = 1 per feature).
SAEs: TopK, width 4096 (8 latents per feature), k ladder, real vs Gaussian twin, 2 epochs over 5.5M tokens, seed 0.
Usage: python experiments/phase3_stage2.py [--pilot]
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["--d", 256, "--M", 512, "--mean_active", 4, "--alpha", 0.5, "--beta", 0, "--noise", 0.05]
DATASETS = {"s2-point": COMMON, "s2-circle": COMMON + ["--frac_manifold", 1.0]}
K_SWEEP = [1, 2, 4, 8, 16, 32]
WIDTH, TOKENS, EVAL, EPOCHS = 4096, 6_000_000, 500_000, 2
GROUP = "saes_p3s2"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def main():
    pilot = "--pilot" in sys.argv
    ks = [4] if pilot else K_SWEEP
    for data, flags in DATASETS.items():
        if not (ROOT / "cache" / "acts" / f"synth-{data}" / "blocks.0.hook_resid_post" / "acts.npy").exists():
            py(ROOT / "synth.py", "--name", data, "--n_tokens", TOKENS, *flags)
    for k in ks:
        for data in DATASETS:
            for cond in ["real", "gaussian"]:
                name = f"synth-{data}__blocks.0.hook_resid_post__{cond}__w{WIDTH}__k{k}__s0"
                if (ROOT / "cache" / GROUP / name / "train.json").exists():
                    continue
                py(ROOT / "train_sae.py", "--model", f"synth-{data}", "--hook", "blocks.0.hook_resid_post",
                   "--cond", cond, "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL,
                   "--epochs", EPOCHS, "--seed", 0, "--out", f"cache/{GROUP}")
    py(ROOT / "experiments" / "phase3_analyze.py", GROUP, "--match")


if __name__ == "__main__":
    main()
