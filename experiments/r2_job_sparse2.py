"""Round-2 GPU job 4: a sparse covariance-matched null that a TopK SAE can code ("sparse2"). Needs no model access:
it trains from each layer's stats.npz (mean and covariance of the real activations), which job 1 left on the volume.
Unattended and restartable like r2_job_real.py; rows go to results/r2/runs_sparse2.jsonl, decoders are kept.

Why a second sparse null. The first one (Q5 in r2_job_real.py: 65,536 sources, 64 active per token, normal values)
was not sparsely codable by a width-16,384 SAE. At every layer and every k its SAEs had the Gaussian null's eval FVU
to three decimals (layer 13, k = 40: 0.529 against 0.528; real 0.288) and the Gaussian null's slope ratios within
0.01. It therefore tested nothing the Gaussian null had not. Its rows stay in runs_real.jsonl and are reported.

Everything below was fixed before any sparse2 run on a GPU. A reduced-scale check on a laptop (layer-13 Sigma, width
8,192, 4,096 sources, 1M tokens, k = 40) preceded it: eval FVU 0.154 at rho = 0 against 0.564 for the Gaussian null at
the same scale, with bulk slope ratios 0.82 against 0.95-0.98. At rho = 0.3 the same check gave FVU 0.513: FVU rises
steeply with rho, which is why the first calibration step below starts at a quarter of the linear guess.

Generator (train_sae.py --cond sparse --sparse_values halfnormal --sparse_rho rho):
  x = mu + sqrt(1 - rho) (s - E s) A + sqrt(rho) g,   A = R Sigma^{1/2} (M x d), R^T R = I_d,   g ~ N(0, Sigma).
  M = 8,192 sources (half the SAE width, so every source can have its own latent), each active with probability
  p = L/M, L = 16 expected active sources per token; active values c|z|, z standard normal, c chosen so Var(s_i) = 1
  (one-signed values, so a ReLU latent can code a source without a second latent for the other sign). Cov(x) = Sigma
  in expectation for every rho in [0, 1). rho is the share of variance that is dense and Gaussian; rho = 0 is purely
  sparse and rho -> 1 is the Gaussian null. Data seed 0 for the dictionary and token streams; SAE seeds set only the
  SAE initialisation. Width, tokens, epochs and scoring as job 1 (16,384; 9.5M x 3; fixed-n and full-live slopes).

Calibration of rho, per layer, at k = 40, SAE seed 0, full training settings (a calibration run is an ordinary run):
  target T = mean eval FVU of job 1's three real k = 40 SAEs at that layer (TARGET below, from runs_real.jsonl);
  G = mean eval FVU of job 1's Gaussian k = 40 SAEs at that layer (GAUSS below).
  Step 0: train at rho = 0, giving F0. If F0 >= T - 0.03 the layer uses rho = 0 (the null cannot be made more
  codable; if F0 > T + 0.03 the layer is reported as unmatched).
  Step 1: rho_1 = 0.25 (T - F0) / (G - F0), rounded to two decimals (at least 0.01). Train, giving F1. Accept if
  |F1 - T| <= 0.03.
  Steps 2-4: secant step through the two tried rho values whose FVUs bracket T (or the two nearest to T if none
  bracket), rounded to two decimals, clipped to [0, 0.95]; accept as soon as |F - T| <= 0.03. After step 4 the tried
  rho with the smallest |F - T| is used and the layer is reported as unmatched if that exceeds 0.03.
  Rows of tried-but-unused rho values are scored and kept with step "CAL". The rho = 0 run is always among them or is
  the final run, so the purely sparse null is reported at every layer as a secondary comparison.
  The same rho is used for every k at layer 13: the null is matched to real at k = 40, and its FVU at other k is
  reported next to real's, not matched.

Runs: (a) step "S2a", layer 13, k in {8, 16, 24, 32, 40, 48, 64, 128}, SAE seeds 0, 1; (b) step "S2b", layers 0, 3, 6,
  9, 17, 21, 24 at k = 40, seeds 0, 1. 30 final SAEs plus the calibration runs (2 to 4 per layer). The ground-truth
  dictionary (unit rows of A) is scored once per layer.

Reading. Statement (v) of r2_job_real.py, word for word, with sparse2 in place of the first sparse null and the same
  rules and thresholds for S1 (flatter bulk), S2 (mid-stack profile of the top-window slope) and S3 (decline of gamma
  with k over 16-128); r2_summary.py reports it as v_sparse2. A signature that SUPPORTS (v) is one that real
  activations show beyond a sparse code with the same covariance at matched reconstruction error; one that
  CONTRADICTS is reproduced by sparsity plus Sigma. Unmatched layers are named with the verdict.

  python experiments/r2_job_sparse2.py            (GPU; the k = 128 run needs about 33 GB)
  python experiments/r2_job_sparse2.py --skip_k 128   (a 32 GB card; rerun without the flag on a larger one)
  python experiments/r2_job_sparse2.py --smoke    (laptop: layer 13 only, small width, two k values)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import r2_common as rc  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB, MODEL = "sparse2", "gemma-2-2b"
WIDTH, TOKENS, EVAL, EPOCHS = 16384, 9_500_000, 500_000, 3
SP_M, SP_L, SP_DS, VALUES = 8192, 16, 0, "halfnormal"
K_LAYER, K0, KS, SEEDS = 13, 40, [8, 16, 24, 32, 40, 48, 64, 128], [0, 1]
LAYERS = [13, 0, 3, 6, 9, 17, 21, 24]
TOL, MAX_STEPS, RHO_MAX = 0.03, 4, 0.95
# job 1, k = 40, mean eval FVU over three seeds (results/r2/runs_real.jsonl)
TARGET = {0: 0.114, 3: 0.204, 6: 0.216, 9: 0.250, 13: 0.288, 17: 0.252, 21: 0.252, 24: 0.269}
GAUSS = {0: 0.603, 3: 0.565, 6: 0.530, 9: 0.541, 13: 0.528, 17: 0.557, 21: 0.591, 24: 0.519}
GROUP, OUT, N_FIX = "saes_r2", ROOT / "results" / "r2", rc.N_FIX
EXTRA = []  # extra train_sae.py arguments (smoke only)


def smoke_config():
    global MODEL, WIDTH, TOKENS, EVAL, EPOCHS, SP_M, KS, LAYERS, MAX_STEPS, GROUP, OUT, N_FIX, EXTRA
    MODEL, WIDTH, TOKENS, EVAL, EPOCHS, SP_M = "sp2local", 4096, 300_000, 65_536, 1, 2304
    KS, LAYERS, MAX_STEPS, N_FIX = [16, 40], [13], 1, 2048
    GROUP, OUT, EXTRA = "sp2_smoke", ROOT / "cache" / "sp2_smoke_out", ["--batch", 2048]


def hook(layer):
    return f"blocks.{layer}.hook_resid_post"


def sigma(layer):
    return rc.Sigma(ROOT / "cache" / "acts" / MODEL / hook(layer) / "stats.npz", f"{MODEL}_L{layer}", OUT)


def run_name(layer, k, seed, rho):  # train_sae.py's naming
    return (f"{MODEL}__{hook(layer)}__sparse__w{WIDTH}__k{k}__s{seed}__ds{SP_DS}"
            f"__M{SP_M}__L{SP_L:g}__{VALUES}__rho{rho:.4f}")


def run_dir(*a):
    return ROOT / "cache" / GROUP / run_name(*a)


def train(layer, k, seed, rho):
    """Train if needed; return eval FVU from train.json."""
    rd = run_dir(layer, k, seed, rho)
    if not (rd / "train.json").exists():
        log(f"train {rd.name}")
        py(ROOT / "train_sae.py", "--model", MODEL, "--hook", hook(layer), "--cond", "sparse", "--width", WIDTH,
           "--k", k, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", seed,
           "--out", f"cache/{GROUP}", "--sparse_M", SP_M, "--sparse_L", SP_L, "--data_seed", SP_DS,
           "--sparse_values", VALUES, "--sparse_rho", rho, *EXTRA)
    return json.load(open(rd / "train.json"))["fvu_eval"]


def score(scorer, step, layer, k, seed, rho, extra=None):
    name = run_name(layer, k, seed, rho)
    if rc.row_path(OUT, JOB, name).exists():
        return
    meta = {"step": step, "layer": layer, "cond": "sparse2", "dataset": f"{MODEL}_L{layer}", "whitened": False,
            "rho": rho, "fvu_target": TARGET[layer], **(extra or {})}
    scorer.submit(rc.score, run_dir(layer, k, seed, rho), sigma(layer), OUT, JOB, meta, n_fix=N_FIX)


def next_rho(tried, T):
    """Secant step through the two tried (rho, F) points that bracket T, else the two nearest to T."""
    pts = sorted(tried.items())
    below = [p for p in pts if p[1] < T]
    above = [p for p in pts if p[1] >= T]
    if below and above:
        a, b = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
    else:
        a, b = sorted(pts, key=lambda p: abs(p[1] - T))[:2]
    if abs(b[1] - a[1]) < 1e-6:
        return None
    r = a[0] + (T - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
    return round(min(max(r, 0.0), RHO_MAX), 2)


def calibrate(layer):
    """The rule in the docstring. Returns (rho, tried, matched); reruns are free (finished runs are read back)."""
    T, tried = TARGET[layer], {}
    tried[0.0] = train(layer, K0, 0, 0.0)
    log(f"calibrate layer {layer}: target {T:.3f}, rho 0.00 -> FVU {tried[0.0]:.3f}")
    if tried[0.0] >= T - TOL:
        return 0.0, tried, tried[0.0] <= T + TOL
    rho = max(round(min(0.25 * (T - tried[0.0]) / (GAUSS[layer] - tried[0.0]), RHO_MAX), 2), 0.01)
    for step in range(1, MAX_STEPS + 1):
        if rho is None or rho in tried:
            break
        tried[rho] = train(layer, K0, 0, rho)
        log(f"calibrate layer {layer}: step {step}, rho {rho:.2f} -> FVU {tried[rho]:.3f}")
        if abs(tried[rho] - T) <= TOL:
            return rho, tried, True
        rho = next_rho(tried, T)
    best = min(tried, key=lambda r: abs(tried[r] - T))
    return best, tried, abs(tried[best] - T) <= TOL


def ground_truth(scorer, layer, rho):
    name = f"{MODEL}__{hook(layer)}__groundtruth__M{SP_M}__L{SP_L:g}__{VALUES}__ds{SP_DS}"
    if rc.row_path(OUT, JOB, name).exists():
        return
    from train_sae import sparse_dictionary
    sig = sigma(layer)
    settings = {"M": SP_M, "L": SP_L, "p": SP_L / SP_M, "data_seed": SP_DS, "values": VALUES, "rho": rho}
    meta = {"step": "GT", "layer": layer, "cond": "sparse2_dictionary", "dataset": f"{MODEL}_L{layer}",
            "whitened": False}
    scorer.submit(rc.score_dictionary, sparse_dictionary(sig.cov, SP_M, SP_DS), sig, OUT, JOB, name, meta, settings,
                  n_fix=N_FIX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_k", type=int, nargs="*", default=[], help="k values left for a later run (GPU memory)")
    a = ap.parse_args()
    if a.smoke:
        smoke_config()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, out {OUT}")
    scorer, cal = rc.Scorer(), {}
    cal_file = OUT / "sparse2_calibration.json"
    for layer in LAYERS:
        rho, tried, matched = calibrate(layer)
        cal[layer] = {"rho": rho, "target": TARGET[layer], "gaussian": GAUSS[layer], "matched": matched,
                      "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}}
        cal_file.write_text(json.dumps(cal, indent=1))
        log(f"layer {layer}: rho {rho:.2f} (FVU {tried[rho]:.3f}, target {TARGET[layer]:.3f}, "
            f"{'matched' if matched else 'UNMATCHED'})")
        step = "S2a" if layer == K_LAYER else "S2b"
        info = {"matched": matched}
        for r in tried:
            if r != rho:
                score(scorer, "CAL", layer, K0, 0, r, info)
        for k in (KS if layer == K_LAYER else [K0]):
            if k in a.skip_k:
                log(f"layer {layer}: k = {k} skipped on this run (--skip_k)")
                continue
            for seed in SEEDS:
                train(layer, k, seed, rho)
                score(scorer, step, layer, k, seed, rho, info)
        ground_truth(scorer, layer, rho)
        scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:  # rerun locally after the pull; never fail the job on it
        log(f"r2_summary failed ({e}); rerun locally")
    log("JOB DONE sparse2")


if __name__ == "__main__":
    main()
