"""Matched-covariance control for the split test (GPU job). Written after an external review of the write-up found
that the split's verdict ("directions, not frequencies") flips in raw decoder-slope units, that F is near-circular
(with uniform firing, F's Sigma is the dictionary's own covariance), and that gamma is biased by the live-latent count.

Question: does the gamma(k) law need Gemma Scope's specific directions, or only their mean and covariance?
  D  Gemma Scope marginals, Gemma Scope rows permuted (reference; same data as the resynthesis run's D)
  G  Gemma Scope marginals, unit directions drawn from a Gaussian with W_dec rows' mean and covariance
  F  uniform marginals, Gemma Scope rows (reference; same data as the split run's F)
  H  uniform marginals, G's Gaussian-matched directions
Seed 0 only (SAE-seed spread was <= 0.003 per doubling in the split run). TopK width 16384, k in K_SWEEP,
12M tokens (last 500k held out), 3 epochs, as before. Needs only the Gemma Scope L12 params and codestats.npz.

Pre-registered reading (fixed before the run):
  Window ranks 11-150. Fit: per-doubling slope s over k = 8-128.
  Primary scoring: fixed live count, 8,192 random live latents per SAE (mean of 4 draws), from control_analyze.py.
  Units: s is computed both on gamma (decoder slope / Sigma slope) and on the raw decoder slope.
  retained_G = s_G / s_D and retained_H = s_H / s_F, in each unit.
  A variant keeps the law if retained >= 0.75 in both units, removes it if < 0.5 in both; anything else is partial
  (between the cuts, or the units disagree).
  H keeps: F's gamma(k) needs only the dictionary's covariance, not Gemma Scope's specific directions.
  H removes: structure in Gemma Scope's directions beyond their covariance carries it.
  G vs D answers the same question under Gemma Scope's own marginals. If G and H disagree, report both and claim
  neither.
  Checks: D and F should reproduce their earlier full-live-count gamma within 0.02 at each k (same data, same seed).
  A variant whose Sigma slope in ranks 11-150 is shallower than -0.3 is flagged as ill-conditioned in gamma units.
  Secondary (not a verdict): fixed-n minus full-live slopes for D and F, to size the live-count bias.
Usage (on the pod): python experiments/control_job.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYER, WIDTH = 12, 16384
K_SWEEP = [8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT = "saes_ctl", "control"
SAE = ROOT / "cache" / "gemmascope" / f"L{LAYER}" / "params.npz"
SRC = ROOT / "cache" / "acts" / "gemma-2-2b" / f"blocks.{LAYER}.hook_resid_post"  # holds codestats.npz only
PLAN = [("H", [0]), ("G", [0]), ("F", [0]), ("D", [0])]  # new variants first
HOOK = "blocks.0.hook_resid_post"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def run_dir(kind, k, seed):
    return ROOT / "cache" / GROUP / f"synth-rs{LAYER}-{kind}__{HOOK}__real__w{WIDTH}__k{k}__s{seed}"


def main():
    assert (SRC / "codestats.npz").exists(), "copy codestats.npz into " + str(SRC)
    for kind, seeds in PLAN:
        m = f"synth-rs{LAYER}-{kind}"
        if not all((run_dir(kind, k, s) / "train.json").exists() for k in K_SWEEP for s in seeds):
            acts = ROOT / "cache" / "acts" / m / HOOK / "acts.npy"
            if not acts.exists():
                py(ROOT / "resynth.py", "--kind", kind, "--src", SRC, "--sae", SAE, "--name", f"rs{LAYER}-{kind}",
                   "--n_tokens", TOKENS, "--eval_tokens", EVAL)
            for s in seeds:
                for k in K_SWEEP:
                    if not (run_dir(kind, k, s) / "train.json").exists():
                        py(ROOT / "train_sae.py", "--model", m, "--hook", HOOK, "--cond", "real", "--width", WIDTH,
                           "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", s,
                           "--out", f"cache/{GROUP}")
            acts.unlink(missing_ok=True)  # stats.npz (Sigma) stays next to it
        py(ROOT / "experiments" / "control_analyze.py", "--group", GROUP, "--out", OUT)
        print(f"VARIANT DONE {kind}", flush=True)
    print("CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
