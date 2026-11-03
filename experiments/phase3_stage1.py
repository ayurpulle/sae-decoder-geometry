"""Phase 3, stage 1: does planted sparse-feature data reproduce the real-data gamma(k) law?

Dataset "base": d=256, 4096 point features, p_i ∝ i^-1.5, 32 active per token on average, direction anisotropy
beta=1 (Sigma slope ≈ -1.0, like Pythia's real Sigma). SAEs: TopK width 4096, k in K_SWEEP, real vs Gaussian
twin N(mu, Sigma), 2 epochs over 5.5M tokens, seed 0. Go/no-go: real-on-synthetic follows a smooth log-linear
gamma(k) with gamma ≈ 1 at small k while its Gaussian twin does not (the Gemma pattern).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = "base"
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]
WIDTH, TOKENS, EVAL, EPOCHS = 4096, 6_000_000, 500_000, 2


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT, env=os.environ | {"TRANSFORMERLENS_ALLOW_MPS": "1"})


def main():
    if not (ROOT / "cache" / "acts" / f"synth-{DATA}" / "blocks.0.hook_resid_post" / "acts.npy").exists():
        py(ROOT / "synth.py", "--name", DATA, "--n_tokens", TOKENS)
    for k in K_SWEEP:
        for cond in ["real", "gaussian"]:
            name = f"synth-{DATA}__blocks.0.hook_resid_post__{cond}__w{WIDTH}__k{k}__s0"
            if (ROOT / "cache" / "saes_p3" / name / "train.json").exists():
                continue
            py(ROOT / "train_sae.py", "--model", f"synth-{DATA}", "--hook", "blocks.0.hook_resid_post", "--cond", cond,
               "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS,
               "--seed", 0, "--out", "cache/saes_p3")
    py(ROOT / "experiments" / "phase3_analyze.py", "saes_p3", "--match")


if __name__ == "__main__":
    main()
