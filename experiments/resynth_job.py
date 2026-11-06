"""Resynthesis dissection of Gemma's gamma(k) (GPU job). Which property of real activations produces the k-law?

Layer 12 of Gemma-2-2B (Gemma Scope 65k canonical SAE, L0 ~ 77, has full params locally; Phase 4 frequencies were
measured here). TopK width 16384, k in K_SWEEP, 12M tokens (last 500k held out), 3 epochs, as in Phase 3.
  R  real layer-12 activations (collect_acts.py), the in-pipeline reference                   seeds 0, 1
  A  Gemma Scope resynthesis z W_dec + b_dec                                                  seeds 0, 1
  C  real codes, decoder rows permuted across latents (frequency/co-occurrence-direction link removed)  seed 0
  B  independent per-latent firing with each latent's own frequency and values (co-occurrence removed) seed 0
  D  B and C together                                                                         seed 0
One synthetic dataset on disk at a time (acts deleted after its SAEs; stats.npz kept); the real cache stays until B's
code statistics exist. Analysis: phase3_match_analyze.py --group saes_rs --out resynth_match --clusters.
Pre-registered reading (ranks 11-150; per-doubling slope s from a straight-line fit of gamma on log2 k, 6 k values):
  Gate: R must show the law (s_R <= -0.05), and A must reproduce R (RMS(A - R) <= 0.15 and |s_A - s_R| <= 0.04).
        If the gate fails, B-D say nothing about Gemma; report them as descriptive only.
  Then for X in B, C, D: retained = s_X / s_A. < 0.5: removing that property removes the law; > 0.75: it does not;
        in between: partial. Seed spread on R and A sets the noise floor for these comparisons.
Usage (on the pod): python experiments/resynth_job.py --model_dir /workspace/models/gemma-2-2b
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYER, WIDTH = 12, 16384
HOOK = f"blocks.{LAYER}.hook_resid_post"
K_SWEEP = [4, 8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT = "saes_rs", "resynth_match"
SAE = ROOT / "cache" / "gemmascope" / f"L{LAYER}" / "params.npz"
REAL = ROOT / "cache" / "acts" / "gemma-2-2b" / HOOK
PLAN = [("R", [0, 1]), ("A", [0, 1]), ("C", [0]), ("B", [0]), ("D", [0])]


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def model_hook(kind):
    return ("gemma-2-2b", HOOK) if kind == "R" else (f"synth-rs{LAYER}-{kind}", "blocks.0.hook_resid_post")


def run_dir(kind, k, seed):
    m, h = model_hook(kind)
    return ROOT / "cache" / GROUP / f"{m}__{h}__real__w{WIDTH}__k{k}__s{seed}"


def done(kind, seeds):
    return all((run_dir(kind, k, s) / "train.json").exists() for k in K_SWEEP for s in seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="local Gemma-2-2B snapshot (dir named gemma-2-2b)")
    args = ap.parse_args()
    for kind, seeds in PLAN:
        if done(kind, seeds):
            print(f"VARIANT DONE {kind} (cached)", flush=True)
            continue
        m, h = model_hook(kind)
        acts = ROOT / "cache" / "acts" / m / h / "acts.npy"
        if kind == "R" or kind in "AC":
            if not (REAL / "acts.npy").exists():
                py(ROOT / "collect_acts.py", "--backend", "hf", "--model", args.model_dir, "--hook", HOOK,
                   "--ctx", 1024, "--batch", 16, "--dtype", "bfloat16", "--n_tokens", TOKENS)
        if kind != "R" and not acts.exists():
            py(ROOT / "resynth.py", "--kind", kind, "--src", REAL, "--sae", SAE, "--name", f"rs{LAYER}-{kind}",
               "--n_tokens", TOKENS, "--eval_tokens", EVAL)
        for s in seeds:
            for k in K_SWEEP:
                if (run_dir(kind, k, s) / "train.json").exists():
                    continue
                py(ROOT / "train_sae.py", "--model", m, "--hook", h, "--cond", "real", "--width", WIDTH, "--k", k,
                   "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", s,
                   "--out", f"cache/{GROUP}")
        py(ROOT / "experiments" / "phase3_match_analyze.py", "--group", GROUP, "--out", OUT, "--clusters")
        if kind != "R":
            acts.unlink(missing_ok=True)
        if kind == "B":
            (REAL / "acts.npy").unlink(missing_ok=True)  # D needs only the code statistics
        print(f"VARIANT DONE {kind}", flush=True)
    print("RESYNTH DONE", flush=True)


if __name__ == "__main__":
    main()
