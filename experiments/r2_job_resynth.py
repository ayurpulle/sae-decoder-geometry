"""Round-2 GPU job 2 (pod 2, no Gemma access needed): Q3, resynthesis variants D and G with three independent data
draws. Needs cache/gemmascope/L12/params.npz and cache/acts/gemma-2-2b/blocks.12.hook_resid_post/codestats.npz
(pushed by gpu/push_r2.sh ... job2). Unattended and restartable like r2_job_real.py; launch with gpu/run_job.sh.

PRE-REGISTRATION (fixed before any run; experiments/r2_summary.py implements it unchanged)
==========================================================================================
Data (resynth.py, as in control_job.py): Gemma Scope 65k L0-72 JumpReLU SAE at layer 12, per-latent firing
frequencies and 64-quantile value distributions from codestats.npz (4M real tokens).
  D  independent per-latent codes on Gemma Scope's decoder rows, randomly permuted across latents.
  G  the same marginals on unit directions drawn from a Gaussian with W_dec rows' mean and covariance.
Data seeds 0, 1, 2 (resynth.py --seed): the seed sets the code draws (torch generator, seed), D's row permutation
(seed + 2000) and G's Gaussian directions (seed + 5000), so each data seed is an independent draw of directions and
data. At a given data seed D and G share the same code draws (same generator, same marginals), so D vs G is paired.
Data seed 0 is the dataset control_job.py used (same seeds), which gives a reproduction check.
SAEs: TopK width 16,384, k in {8, 16, 24, 32, 48, 64, 128}, SAE seeds 0, 1, 12M tokens (last 500k held out),
3 epochs (84 SAEs). Each synthetic acts.npy (55 GB) is deleted after its 14 SAEs are trained.
Windows, Sigma (each dataset's own stats.npz), fixed-n scoring (8,192 live latents, mean of 4 draws), per-doubling
slope s and the window rule (primary ranks 11-150 plus at least two of the other three) are as in r2_job_real.py.

Q3  Question: does Gemma Scope's gamma(k) decline need its specific directions, or only their covariance, once
  the draw of directions and data is allowed to vary?
  Statement (iv): "G retains D's decline". Same rule as control_job.py, now with draw-level uncertainty:
    s over all seven k (8-128), in gamma and raw units, fixed-n primary. For each data seed j, s_X(j) is the mean
    over SAE seeds of the per-seed fits. Pooled: s_X = mean over j. retained = s_G / s_D in each unit.
    keeps   if retained >= 0.75 in both units,
    removes if retained < 0.5 in both units,
    partial otherwise.
  Draw-level uncertainty: per-draw retained_j = s_G(j) / s_D(j) (paired by data seed) and the sd over draws of
  s_D(j) and s_G(j). The pooled verdict is "firm" if all three per-draw verdicts equal it, otherwise it is reported
  as "<verdict>, draw-dependent". The window rule decides (iv): supported if "keeps" in the primary window and at
  least two others, contradicted if "removes" there; otherwise mixed.
  Checks (not verdicts): D and G at data seed 0, SAE seed 0 should reproduce results/phase3/control.json's fixed-n
  gamma at k = 8, 16, 32, 64, 128 within 0.02 (same data, same seed; CUDA nondeterminism allows small drift).
  A dataset whose Sigma slope in ranks 11-150 is shallower than -0.3 is flagged ill-conditioned in gamma units.
  Reported, not a verdict: full-live versions; SAE-seed sd; basis_gamma.
Usage (pod): python experiments/r2_job_resynth.py     smoke test (Mac): python experiments/r2_job_resynth.py --smoke
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "resynth"
LAYER, WIDTH = 12, 16384
KINDS, DATA_SEEDS, SAE_SEEDS = ["D", "G"], [0, 1, 2], [0, 1]
K_SWEEP = [8, 16, 24, 32, 48, 64, 128]
TOKENS, EVAL, EPOCHS = 12_000_000, 500_000, 3
GROUP, OUT, N_FIX, PREFIX = "saes_r2rs", ROOT / "results" / "r2", rc.N_FIX, ""
SAE = ROOT / "cache" / "gemmascope" / f"L{LAYER}" / "params.npz"
SRC = ROOT / "cache" / "acts" / "gemma-2-2b" / f"blocks.{LAYER}.hook_resid_post"  # holds codestats.npz only
HOOK = "blocks.0.hook_resid_post"


def smoke_config():
    global DATA_SEEDS, SAE_SEEDS, K_SWEEP, TOKENS, EVAL, EPOCHS, WIDTH, GROUP, OUT, N_FIX, PREFIX
    DATA_SEEDS, SAE_SEEDS, K_SWEEP = [0, 1], [0], [8, 16]
    TOKENS, EVAL, EPOCHS, WIDTH = 49_152, 16_384, 1, 512
    GROUP, OUT, N_FIX, PREFIX = "saes_r2rs_smoke", ROOT / "results" / "r2_smoke", 256, "smoke-"


def data_name(kind, ds):
    return f"{PREFIX}rs{LAYER}-{kind}-ds{ds}"


def act_dir(kind, ds):
    return ROOT / "cache" / "acts" / f"synth-{data_name(kind, ds)}" / HOOK


def run_name(kind, ds, k, seed):
    return f"synth-{data_name(kind, ds)}__{HOOK}__real__w{WIDTH}__k{k}__s{seed}"


def done(kind, ds, k, seed):
    return rc.row_path(OUT, JOB, run_name(kind, ds, k, seed)).exists()


def dataset(scorer, kind, ds):
    runs = [(k, s) for s in SAE_SEEDS for k in K_SWEEP]
    d = act_dir(kind, ds)
    if all(done(kind, ds, k, s) for k, s in runs):
        log(f"{data_name(kind, ds)}: all {len(runs)} runs done")
        return
    need = [r for r in runs if not done(kind, ds, *r)
            and not (ROOT / "cache" / GROUP / run_name(kind, ds, *r) / "train.json").exists()]
    if need and not (d / "acts.npy").exists():
        log(f"generate {data_name(kind, ds)} ({rc.disk_free_gb():.0f} GB free)")
        py(ROOT / "resynth.py", "--kind", kind, "--src", SRC, "--sae", SAE, "--name", data_name(kind, ds),
           "--n_tokens", TOKENS, "--eval_tokens", EVAL, "--seed", ds)
    sig = rc.Sigma(d / "stats.npz", data_name(kind, ds), OUT)
    for k, s in runs:
        if done(kind, ds, k, s):
            continue
        rd = ROOT / "cache" / GROUP / run_name(kind, ds, k, s)
        if not (rd / "train.json").exists():
            log(f"train {rd.name}")
            py(ROOT / "train_sae.py", "--model", f"synth-{data_name(kind, ds)}", "--hook", HOOK, "--cond", "real",
               "--width", WIDTH, "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", s,
               "--out", f"cache/{GROUP}")
        meta = {"step": "Q3", "kind": kind, "data_seed": ds, "layer": LAYER, "cond": "real",
                "dataset": data_name(kind, ds), "whitened": False}
        scorer.submit(rc.score, rd, sig, OUT, JOB, meta, n_fix=N_FIX)
    scorer.finish()
    (d / "acts.npy").unlink(missing_ok=True)  # stats.npz (Sigma) stays next to it
    log(f"deleted {d / 'acts.npy'}; {rc.disk_free_gb():.0f} GB free")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        smoke_config()
    OUT.mkdir(parents=True, exist_ok=True)
    assert SAE.exists(), f"missing {SAE} (gpu/push_r2.sh <host> <port> job2 pushes it)"
    assert (SRC / "codestats.npz").exists(), f"missing {SRC / 'codestats.npz'} (gpu/push_r2.sh ... job2 pushes it)"
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, out {OUT}, {rc.disk_free_gb():.0f} GB free")
    scorer = rc.Scorer()
    for ds in DATA_SEEDS:  # draw 0 of both variants first: a complete D/G pair as early as possible
        for kind in KINDS:
            dataset(scorer, kind, ds)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:
        log(f"r2_summary failed on the pod ({e}); rerun locally")
    log("JOB DONE resynth")


if __name__ == "__main__":
    main()
