"""Split test after the resynthesis dissection (GPU job): is Gemma's gamma(k) law carried by the dictionary's directions
or by the per-latent firing marginals? Needs only the Gemma Scope L12 params and its code statistics (codestats.npz,
measured on the previous pod from 4M real tokens); no real activations, no Gemma model.
  E  Gemma Scope per-latent marginals (independent firing) on random unit directions      seeds 0, 1
  F  Gemma Scope directions, every live latent with the mean frequency and pooled values  seeds 0, 1
  B  Gemma Scope marginals on Gemma Scope directions (as before), a second SAE seed       seed 1
TopK width 16384, k in K_SWEEP, 12M tokens (last 500k held out), 3 epochs, as in resynth_job.py.
Pre-registered reading (ranks 11-150; per-doubling slope s of gamma on log2 k):
  Reference: A from resynth_job.py (s_A = -0.089 over k 4-128, -0.095 over k 8-128).
  Primary fit: k 8-128 (the previous run showed the independent-firing variants unstable at k = 4); all-k reported too.
  retained = s_X / s_A on the same k set: < 0.5 removes the law, > 0.75 keeps it, partial between.
  E keeps & F removes: the marginals carry the law. E removes & F keeps: the dictionary geometry does.
  Both keep: either suffices. Both remove: it takes the two together.
  If a variant's Sigma slope in ranks 11-150 is shallower than -0.3, its gamma there is flagged as ill-conditioned.
Usage (on the pod): python experiments/resynth_split_job.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYER, WIDTH = 12, 16384
K_SWEEP = [4, 8, 16, 32, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT = "saes_rs2", "resynth_split"
SAE = ROOT / "cache" / "gemmascope" / f"L{LAYER}" / "params.npz"
SRC = ROOT / "cache" / "acts" / "gemma-2-2b" / f"blocks.{LAYER}.hook_resid_post"  # holds codestats.npz only
PLAN = [("E", [0, 1]), ("F", [0, 1]), ("B", [1])]
HOOK = "blocks.0.hook_resid_post"


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT)


def run_dir(kind, k, seed):
    return ROOT / "cache" / GROUP / f"synth-rs{LAYER}-{kind}__{HOOK}__real__w{WIDTH}__k{k}__s{seed}"


def main():
    assert (SRC / "codestats.npz").exists(), "copy codestats.npz into " + str(SRC)
    for kind, seeds in PLAN:
        if all((run_dir(kind, k, s) / "train.json").exists() for k in K_SWEEP for s in seeds):
            print(f"VARIANT DONE {kind} (cached)", flush=True)
            continue
        m = f"synth-rs{LAYER}-{kind}"
        acts = ROOT / "cache" / "acts" / m / HOOK / "acts.npy"
        if not acts.exists():
            py(ROOT / "resynth.py", "--kind", kind, "--src", SRC, "--sae", SAE, "--name", f"rs{LAYER}-{kind}",
               "--n_tokens", TOKENS, "--eval_tokens", EVAL)
        for s in seeds:
            for k in K_SWEEP:
                if not (run_dir(kind, k, s) / "train.json").exists():
                    py(ROOT / "train_sae.py", "--model", m, "--hook", HOOK, "--cond", "real", "--width", WIDTH, "--k", k,
                       "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", s,
                       "--out", f"cache/{GROUP}")
        py(ROOT / "experiments" / "phase3_match_analyze.py", "--group", GROUP, "--out", OUT, "--clusters")
        acts.unlink(missing_ok=True)
        print(f"VARIANT DONE {kind}", flush=True)
    print("SPLIT DONE", flush=True)


if __name__ == "__main__":
    main()
