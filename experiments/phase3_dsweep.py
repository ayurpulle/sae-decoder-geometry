"""Phase 3, d-sweep: does the residual dimension d set the k-scale k0 of gamma(k)?

Density held fixed as ratios to d, matching i-m2048-a16 at d = 256: M = 8d features, d/16 active per token,
point features, beta = 0, p_i ∝ i^-1.5, noise 0.05 per coordinate (so SNR is constant in d). Two arms:
    scaled width (width = 16 d):  d = 64 / w1024, 128 / w2048, 256 / w4096 [saes_p3i], 512 / w8192
    fixed width (width = 4096):   d = 128, 256 [saes_p3i], 512, 1024
The width sweep found log2 k0 rises ~0.25 per doubling of width at d = 256, so the scaled arm mixes d with that
width effect and the fixed arm isolates d. k in K_SWEEP, real vs Gaussian twin, 2 epochs over 6M tokens, seed 0.
Cheapest configs first so partial results are usable. Usage: python experiments/phase3_dsweep.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ["--alpha", 0.5, "--beta", 0, "--noise", 0.05]
DS = [64, 128, 512, 1024]
RUNS = [(64, 1024), (128, 2048), (128, 4096), (512, 4096), (512, 8192), (1024, 4096)]  # (d, width)
K_SWEEP = [4, 8, 16, 32, 64]
TOKENS, EVAL, EPOCHS = 6_000_000, 500_000, 2
GROUP = "saes_p3d"


def name(d):
    return f"d{d}-m{8 * d}-a{d // 16}"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def main():
    for d in DS:
        if not (ROOT / "cache" / "acts" / f"synth-{name(d)}" / "blocks.0.hook_resid_post" / "acts.npy").exists():
            py(ROOT / "synth.py", "--name", name(d), "--n_tokens", TOKENS, "--d", d, "--M", 8 * d,
               "--mean_active", d // 16, *BASE)
    for d, w in RUNS:
        for k in K_SWEEP:
            for cond in ["real", "gaussian"]:
                run = f"synth-{name(d)}__blocks.0.hook_resid_post__{cond}__w{w}__k{k}__s0"
                if (ROOT / "cache" / GROUP / run / "train.json").exists():
                    continue
                py(ROOT / "train_sae.py", "--model", f"synth-{name(d)}", "--hook", "blocks.0.hook_resid_post",
                   "--cond", cond, "--width", w, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL,
                   "--epochs", EPOCHS, "--seed", 0, "--out", f"cache/{GROUP}")


if __name__ == "__main__":
    main()
