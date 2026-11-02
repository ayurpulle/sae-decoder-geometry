"""GPU queue: Gemma-scale estimator nulls, small-k sweep, and de-massive runs. Runs unattended, skips work
already done, so it can be restarted after an interruption.

  0. Check Hugging Face access to Gemma-2-2B.
  1. Cache activations (collect_acts.py) for LAYERS: CACHE_TOKENS tokens each, last EVAL_TOKENS held out.
  2. (A) Phase 2 at Gemma scale: real / gaussian / scrambled x widths x seeds at every layer in LAYERS.
  3. (B) Small-k sweep at layer 13: real and gaussian, k in K_SWEEP. Does gamma -> 1 near L0 ~ 3-5?
  4. (C) De-massive at layer 13: real data with tokens whose |x_1170| is in the top DROP_FRACS removed.
  5. Summarize every run (gpu_summarize.py) into results/gpu/.
Layer choice: gamma's layer profile (Test 1) is low at 3, peaks at 13, low again at 23; dim 1170's kurtosis
is 7 / 642 / 57 there. All SAEs are TopK with k=40 (the L0=40 gamma curve) unless swept.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_MODEL, MODEL = "google/gemma-2-2b", "gemma-2-2b"
LAYERS = [3, 13, 23]
CACHE_TOKENS, EVAL_TOKENS, EPOCHS = 10_000_000, 500_000, 3
K = 40
P2_WIDTHS, P2_CONDS, P2_SEEDS = [16384, 65536], ["real", "gaussian", "scrambled"], [0, 1, 2]
K_SWEEP, K_LAYER, K_WIDTH, K_SEEDS = [1, 2, 4, 8, 16, 40, 128], 13, 16384, [0, 1]
DM_LAYER, DM_DIM, DM_FRACS, DM_WIDTHS, DM_SEEDS = 13, 1170, [0.001, 0.01], [16384], [0, 1, 2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT, env=os.environ)


def train(out, layer, cond, width, k, seed, extra=()):
    name = f"{MODEL}__blocks.{layer}.hook_resid_post__{cond}__w{width}__k{k}__s{seed}"
    if extra:
        name += f"__drop{extra[1]}q{extra[3]:g}"
    if (ROOT / out / name / "train.json").exists():
        return
    log(f"train {out}/{name}")
    sh([ROOT / "train_sae.py", "--model", MODEL, "--hook", f"blocks.{layer}.hook_resid_post", "--cond", cond,
        "--width", width, "--k", k, "--tokens", CACHE_TOKENS, "--eval_tokens", EVAL_TOKENS, "--epochs", EPOCHS,
        "--seed", seed, "--out", out, *extra])


def main():
    log("step 0: Hugging Face access")
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(HF_MODEL)
    except Exception as e:
        sys.exit(f"No access to {HF_MODEL} ({type(e).__name__}). Run gpu/hf_login.sh on the pod and accept the license.")

    log("step 1: activation caches")
    for layer in sorted(set(LAYERS + [K_LAYER, DM_LAYER])):
        d = ROOT / "cache" / "acts" / MODEL / f"blocks.{layer}.hook_resid_post"
        if (d / "acts.npy").exists():
            continue
        log(f"collect layer {layer}")
        sh([ROOT / "collect_acts.py", "--backend", "hf", "--model", HF_MODEL, "--hook", f"blocks.{layer}.hook_resid_post",
            "--ctx", 1024, "--batch", 16, "--dtype", "bfloat16", "--n_tokens", CACHE_TOKENS])

    log("step 2 (A): Phase 2 at Gemma scale")
    for seed in P2_SEEDS:
        for layer in LAYERS:
            for width in P2_WIDTHS:
                for cond in P2_CONDS:
                    train("cache/saes_gpu_p2", layer, cond, width, K, seed)

    log("step 3 (B): small-k sweep")
    for seed in K_SEEDS:
        for k in K_SWEEP:
            for cond in ["real", "gaussian"]:
                train("cache/saes_gpu_k", K_LAYER, cond, K_WIDTH, k, seed)

    log("step 4 (C): de-massive")
    for seed in DM_SEEDS:
        for width in DM_WIDTHS:
            for frac in DM_FRACS:
                train("cache/saes_gpu_dm", DM_LAYER, "real", width, K, seed, ("--drop_dim", DM_DIM, "--drop_frac", frac))

    log("step 5: summarize")
    sh([ROOT / "experiments" / "gpu_summarize.py"])
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
