"""Round-2 GPU job 5 ("robust"): three checks of the codable sparse null (r2_job_sparse2.py). Unattended and restartable
like the other jobs: a run whose row exists is skipped, a trained but unscored run is scored, calibration reads finished
runs back. Rows go to results/r2/runs_robust.jsonl; per-latent power arrays to results/r2/power/<run>.npz.

PRE-REGISTRATION (fixed before any run of this job; experiments/r2_summary.py, section "robust", implements it)
==============================================================================================================
Conventions as in r2_job_real.py: Gemma-2-2b windows are 0-based slices (3,50), (10,150), (30,450), (100,1000) = ranks
4-50, 11-150 (primary), 31-450, 101-1000; top windows = the first two, bulk windows = the last two. gamma = decoder slope /
Sigma slope; "gamma_fixed" = fixed-n unit scoring (8,192 live latents, 4 draws), Sigma = the layer's training covariance.
"Real" below always means the mean over job 1's three real SAEs (width 16,384, k = 40, results/r2/runs_real.jsonl):
            ranks 4-50   11-150   31-450   101-1000   eval FVU
  layer 0     0.315      0.304    0.375    0.623      0.114
  layer 13    0.466      0.484    0.595    0.863      0.288
  layer 21    0.400      0.416    0.450    0.650      0.252
  (other layers: 3 .304/.441/.636/.720, 6 .371/.459/.692/.785, 9 .448/.491/.637/.814, 17 .468/.464/.525/.750,
   24 .354/.343/.405/.660)
The codable null as run (M = 8,192 atoms, L = 16 active, half-normal values, dense share rho calibrated to real FVU),
seed 0, k = 40: layer 0 .407/.651/.906/1.030, layer 13 .243/.511/.930/1.156, layer 21 .245/.554/.948/1.101; over all 8
layers, bulk .88-.96 / 1.03-1.16 against real .38-.69 / .62-.86. Its calibration runs (rho = 0 and rho ~ 0.1, all
layers) gave bulk .87-.98 / .97-1.01 and .81-.95 / .99-1.02.

Calibration (all steps). Target T = real eval FVU above (GPT-2: job 3's real k = 32 FVU, below); G = the Gaussian null's
FVU at the same layer and k (the rho -> 1 limit); tolerance |F - T| <= 0.03 as in r2_job_sparse2.py. Routine of
r2_job_sparse2.py with one change made from its own data: the first guess is rho_1 = 0.8 (T - F0) / (G - F0) instead of
0.25 (...), because rho = 0.82 (F - F0) / (G - F0) fits all 16 sparse2 calibration points (per point 0.74-0.84), where F0
is the rho = 0 FVU. Step 0: rho = 0; if F0 >= T - 0.03 the setting uses rho = 0 (it is reported as unmatched, "not
codable at this FVU", if F0 > T + 0.03). Then rho_1, then secant steps through the bracketing (else nearest) pair,
rounded to 0.01 and clipped to [0, 0.95]; the first run within tolerance is used, else the tried rho closest to T,
flagged unmatched. Every tried run is scored; unused ones are kept with step "GCAL" / "MCAL".

STEP G  Generator sensitivity, Gemma-2-2b, width 16,384, k = 40, SAE seed 0, data seed 0, 9.5M tokens x 3 epochs.
  Question: is "real decoders are flatter than the null in the bulk" specific to (M, L) = (8192, 16) with a dense share?
  Settings at layers 13 and 0 (layer 21 only with --g_layers 13 0 21, see budget): (M, L) in {(4096, 16), (32768, 16),
  (8192, 8), (8192, 32)}, half-normal values, dense Gaussian share rho; and at layer 13 only "overflow": the dense share
  replaced by a second dictionary of 65,536 equal-power Sigma-shaped atoms (A2 = R2 Sigma^{1/2}, its own seed), 64 active
  per token, values N(0,1)/sqrt(p) (train_sae.py --sparse_overflow_M 65536 --sparse_overflow_L 64), its variance share
  playing the role of rho. With these settings alone this is job 1's first sparse null, whose width-16,384 SAEs had the
  Gaussian null's FVU to three decimals (layer 13: 0.529 vs 0.528), so it is sparse but cannot be held by the SAE. At
  share 0 the overflow generator is the M = 8192 null at rho = 0, so step 0 is that existing run (FVU 0.0135), not rerun.
  At most 3 SAEs per setting (step 0 counts). M = 32,768 at L = 16 may not be codable at width 16,384: if its rho = 0
  FVU exceeds T + 0.03 that is the result for the setting, recorded, and the job moves on.
  Rule (bulk windows ranks 31-450 and 101-1000; null = the setting's final run, real = mean of 3 seeds above):
    ROBUST TO THE GENERATOR if in both bulk windows real gamma_fixed <= null gamma_fixed - 0.15 at every (layer, setting)
      that reached matched FVU (and at least one did);
    GENERATOR-DEPENDENT if any matched (layer, setting) has null - real <= 0.05 in either bulk window;
    PARTIAL otherwise. Unmatched settings are listed and do not enter the rule.
  Reported per setting without a verdict: the top (4-50) and primary (11-150) windows, FVU, live count, check-sample
  slopes. Prediction: robust (every Sigma-shaped null tried so far has bulk gamma >= 0.81 against real <= 0.69 / 0.86).

STEP M  Codable sparse null on GPT-2 small (job 3's M1: same Sigma = stats.npz of 10M Pile tokens per layer, width 16,384,
  9.5M tokens x 3 epochs, 500k eval, k = 32 at every layer (M1 used k = 32 at all 12 layers), windows (1,17), (3,50)
  primary, (10,150), (33,333), the d-scaled images of the Gemma windows). All 12 layers, SAE seed 0, M = 8,192 (half the
  width), L = 16, half-normal, rho calibrated to the layer's real FVU (max 5 SAEs per layer: step 0 plus 4, as sparse2).
  Targets (real k = 32 FVU; layer 6 is the mean of 2 seeds) and Gaussian FVU, layers 0-11:
    T = .029 .034 .052 .076 .097 .119 .136 .151 .163 .174 .178 .069
    G = .213 .233 .277 .326 .361 .393 .414 .428 .437 .433 .399 .134
  At layers 0-1 (T < 0.035) the tolerance 0.03 is as large as the target: they will match at rho = 0 if F0 <= T + 0.03;
  this is stated with the result. Real gamma_fixed (k = 32) per layer is read from runs_models.jsonl; bulk (10,150)
  .39-.55, (33,333) .54-.82; the first, non-codable sparse null had .69-.76 / 1.40-1.46 there.
  Statements, each against what Gemma-2-2b showed (real 3-seed mean vs codable null, k = 40, 8 layers):
    M-bulk  per bulk window, the number of matched layers with real <= null - 0.15. Gemma: 8 of 8 in both (gaps
      0.23-0.55 in 31-450, 0.27-0.46 in 101-1000). REPLICATES if the fraction is >= 0.9 in both bulk windows; FAILS if
      < 0.5 in either; partial otherwise.
    M-primary  layers with |real - null| <= 0.06 in (3,50). Gemma: the null tracked real at layers 6, 9, 13, 17 (mid
      stack) and not at 0, 3, 21, 24 (real lower by 0.11-0.37). GPT-2 interior = layers 3-8, ends = 0, 1, 10, 11 (as in
      r2_summary.py). REPLICATES if it tracks at >= 2/3 of the matched interior layers and at <= 1 matched end layer;
      FAILS if it tracks at < 1/3 of the matched interior layers or at >= 3 end layers; partial otherwise.
    M-top  sign of real - null in (1,17). Gemma: real > null at 7 of 8 layers (+0.05 to +0.23; layer 0 -0.10), a sign
      that moved with rho. REPLICATES if real > null at >= 75% of matched layers; FAILS if real < null at >= 50%;
      partial otherwise.
  Prediction: M-bulk replicates; no prediction for M-primary and M-top.

STEP P  Feature power, no new nulls. Real Gemma-2-2b SAEs (width 16,384, k = 40, seed 0) and the codable null as run
  (M = 8192, L = 16, rho as calibrated: layer 0 .13, 3 .25, 6 .29, 9 .34, 13 .43, 17 .35, 21 .32, 24 .40), seed 0, at
  layers 13, 0, 21, 6, 24, 3, 9, 17 (in that order). train_sae.py kept only W_dec, so both are retrained with identical
  configs and seeds (train_sae.py --save_params adds the encoder); real tokens are collected again exactly as job 1 did
  (collect_acts.py's recipe, all 8 layers in one forward pass). Retraining reproduces each run up to GPU
  nondeterminism; the row records new and original FVU, live count and gamma_fixed, and the new Sigma against the saved
  one. Power sample: 1,000,000 tokens, real = the last 1M cached tokens (500k held out + 500k training), null = fresh
  tokens from its generator (stream 9_500_000 + 1000 * data seed). Per latent i: firing frequency, mean activation, mean
  squared activation, pi_i = E[a_i^2] ||d_i||^2 (input-scaled units; ||d_i|| = 1 for these SAEs), Rayleigh quotient
  rq_i = d_i^T Sigma d_i / (tr Sigma / d). Per run: gamma in the four windows of the unweighted cloud (gamma_fixed, live =
  fired in the power sample, the standard scoring) and of the power-weighted cloud sum_i pi_i d_i d_i^T over live latents
  (uncentred, normalised by sum pi; same slope code); share of power in the top 10 and top 1% of latents; effective
  sample size (sum pi)^2 / sum pi^2 as a fraction of live latents; median log10 rq of the top-10-power latents, of the
  rest, of the top 1% and of the other 99%; Spearman(pi, rq) over live latents.
  A local pilot (Gemma Scope 65k JumpReLU, 32,736 tokens; results/r2/pilot_power/) found power-weighted bulk gamma
  0.75-1.3 against unweighted 0.2-0.8, top-10 latents carrying 21-34% of power with median log10 rq 1.5-1.7, the rest
  near isotropic (-0.01 to +0.35), Spearman -0.02 to 0.21.
  Rule per layer, HEAD/BODY STRUCTURE PRESENT if (a) power-weighted gamma is within 0.25 of 1 in both bulk windows and
    unweighted gamma_fixed is at least 0.15 lower than power-weighted in both, AND (b) the median log10 rq of the
    latents outside the top 1% by power is below 0.5.
  What the equal-power null shows by construction: its atoms have equal power and are Sigma^{1/2}-shaped, so a codable
    SAE's latents carry similar power (weighted gamma close to unweighted, both near its matched values, about 0.9-1.15
    in the bulk) and its body directions are Sigma-shaped, median log10 rq about 1.0-1.4; it should fail (a) and (b).
  Verdict over layers: NOT SPECIFIC TO REAL if present at >= 50% of null layers with rows; else SUPPORTED if present at
    >= 75% of real layers and <= 25% of null layers; ABSENT if present at <= 25% of real layers; partial otherwise;
    insufficient with fewer than 4 real layers. Prediction: supported; the pilot met the rule at 2 of 3 layers (layer 24
    failed on weighted bulk 1.31).

Budget on one 32 GB card (measured from existing rows and logs):
  G  one codable-null SAE at width 16,384 takes 7.5-8.1 min wall when run alone (sparse2 calibration: train_seconds
     420-463 s plus the 1M-token check sample and start-up; scoring 92-104 s on the CPU, overlapped). Expected 2 SAEs per
     setting (step 0 and the first guess; on sparse2's own data the first guess is within 0.02 of the accepted rho at
     all 8 layers), 1-2 for
     overflow and for a non-codable setting: about 19 SAEs x 8.2 min = 2.6 h; worst case 26 SAEs = 3.6 h. Layer 21 adds
     8 settings (about 1.1 h, 3.7 h in all), over the 3-hour cap for this step, so it is cut by default.
  M  GPT-2 SAEs: 1.5 min wall (Gaussian) to 2.7 min (first sparse null, M = 65,536); about 2.0 min here, scoring 5-19 s.
     About 26 SAEs (layers 0-1 at rho = 0 only): 0.9 h.
  P  collection: collect_acts.py took 15-19 min per layer, mostly streaming and tokenising; one pass for 8 layers is
     estimated at 25-35 min and runs in a background process while the null SAEs train (null 8 x 8.2 min = 1.1 h).
     Real SAEs 4.0-5.8 min each (job 1 logs): 8 x 5 min = 0.7 h; power passes about 1 min each on the GPU; statistics on
     the CPU (about 2 min each) overlap the next SAE. Disk: 46 GB per collected layer (368 GB if all 8 fit at once).
     About 2.0 h.
  Total about 5.5 h expected (G 2.6 + M 0.9 + P 2.0); P is last and resumes on a second launch if the cap stops it.
Long CPU-only stretches are avoided (scoring runs in a thread while the next SAE trains; the collection overlaps null
training), because the pod watchdog stops a job whose GPU sits idle for 30 minutes.

  bash gpu/run_job.sh experiments/r2_job_robust.py 6.5       (pod; needs HF access to google/gemma-2-2b for step P)
  python experiments/r2_job_robust.py --skip_p                 (G and M only)
  python experiments/r2_job_robust.py --smoke                  (laptop: every step at toy scale)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "robust"
OUT, N_FIX = ROOT / "results" / "r2", rc.N_FIX
SIGMA_SRC = ROOT / "results" / "r2" / "sigma"  # saved training Sigma (stats.npz contents) of every earlier dataset
GROUP, PGROUP = "saes_r2r", "saes_r2p"
EXTRA = []  # extra train_sae.py arguments (smoke only)
SMOKE = None
TOL, RHO_MAX, C_GUESS = 0.03, 0.95, 0.8
SP_DS, VALUES = 0, "halfnormal"

# Gemma-2-2b (steps G and P)
MODEL, HF_MODEL, SIGMA_MODEL = "gemma-2-2b", "google/gemma-2-2b", "gemma-2-2b"
WIDTH, TOKENS, EVAL, EPOCHS, K = 16384, 10_000_000, 500_000, 3, 40
TARGET = {0: 0.114, 3: 0.204, 6: 0.216, 9: 0.250, 13: 0.288, 17: 0.252, 21: 0.252, 24: 0.269}  # as r2_job_sparse2.py
GAUSS = {0: 0.603, 3: 0.565, 6: 0.530, 9: 0.541, 13: 0.528, 17: 0.557, 21: 0.591, 24: 0.519}
SP2_RHO = {0: 0.13, 3: 0.25, 6: 0.29, 9: 0.34, 13: 0.43, 17: 0.35, 21: 0.32, 24: 0.40}  # sparse2_calibration.json
SP2_F0 = {0: 0.0165, 13: 0.0135, 21: 0.0156}  # rho = 0 FVU of the M = 8192 null (sparse2 CAL rows)
SP2_M, SP2_L = 8192, 16
G_LAYERS = [13, 0]
G_SETTINGS = [(4096, 16), (32768, 16), (8192, 8), (8192, 32)]
G_OVERFLOW, G_OVERFLOW_LAYERS, G_MAX = (65536, 64), [13], 3
# GPT-2 small (step M)
M_MODEL, M_SIGMA_MODEL, M_LAYERS, M_K, M_WIDTH = "gpt2", "gpt2", list(range(12)), 32, 16384
M_M, M_L, M_MAX = 8192, 16, 5
M_TARGET = {0: .029, 1: .034, 2: .052, 3: .076, 4: .097, 5: .119, 6: .136, 7: .151, 8: .163, 9: .174, 10: .178, 11: .069}
M_GAUSS = {0: .213, 1: .233, 2: .277, 3: .326, 4: .361, 5: .393, 6: .414, 7: .428, 8: .437, 9: .433, 10: .399, 11: .134}
M_WINDOWS = [(1, 17), (3, 50), (10, 150), (33, 333)]
GEMMA_WINDOWS = list(rc.WINDOWS)
# step P
P_LAYERS = [13, 0, 21, 6, 24, 3, 9, 17]
P_MODEL = MODEL  # smoke: a model name whose acts are symlinks
N_POWER, POWER_SEED = 1_000_000, 9_500_000
P_NULL_M = SP2_M
COLLECT_MARGIN_GB = 40


def smoke_config():
    """Toy scale on a laptop. Gemma and GPT-2 Sigma come from results/r2/sigma (the path the pod uses when stats.npz is
    missing); step P's real tokens are file symlinks to a local pythia-70m cache (never written or deleted)."""
    global SMOKE, MODEL, WIDTH, TOKENS, EVAL, EPOCHS, G_LAYERS, G_SETTINGS, G_OVERFLOW, G_MAX, GROUP, PGROUP, OUT
    global N_FIX, EXTRA, M_MODEL, M_LAYERS, M_WIDTH, M_M, M_MAX, P_LAYERS, P_MODEL, N_POWER, P_NULL_M
    SMOKE = {"acts": Path(os.environ.get("SMOKE_SRC", ROOT / "cache" / "acts" / "pythia-70m" / "blocks.3.hook_resid_post"))}
    if not (SMOKE["acts"] / "acts.npy").exists():
        SMOKE["acts"] = Path("/Users/ayurpulle/Desktop/SAE/cache/acts/pythia-70m/blocks.3.hook_resid_post")
    MODEL, WIDTH, TOKENS, EVAL, EPOCHS = "robust-smoke-g", 1024, 200_000, 16_384, 1
    G_LAYERS, G_SETTINGS, G_OVERFLOW, G_MAX = [13], [(2560, 16), (4096, 8)], (8192, 64), 2  # M >= d
    M_MODEL, M_LAYERS, M_WIDTH, M_M, M_MAX = "robust-smoke-gpt2", [3, 6], 1024, 1024, 2
    P_LAYERS, P_MODEL, N_POWER, P_NULL_M = [13], "robust-smoke-p", 32_768, 1024
    GROUP, PGROUP, OUT, N_FIX = "robust_smoke", "robust_smoke_p", ROOT / "cache" / "robust_smoke_out", 256
    EXTRA = ["--batch", 2048]


def hook(layer):
    return f"blocks.{layer}.hook_resid_post"


def act_dir(model, layer):
    return ROOT / "cache" / "acts" / model / hook(layer)


def job_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def done(name):
    return rc.row_path(OUT, JOB, name).exists()


def use_windows(w):
    """r2_common reads WINDOWS / PRIMARY at call time; switch only when no score is in flight."""
    rc.WINDOWS, rc.PRIMARY = list(w), w[1]


def ensure_stats(model, layer, label):
    """stats.npz for a stats-only run. Rebuilt from results/r2/sigma/<label>.npz (the same mean, covariance and n that
    the earlier jobs trained on) when the activation directory does not have it; checked against it when it does."""
    d = act_dir(model, layer)
    p, src = d / "stats.npz", SIGMA_SRC / f"{label}.npz"
    if p.exists():
        if src.exists() and not SMOKE:
            a, b = np.load(p)["cov"], np.load(src)["cov"]
            err = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            if err > 1e-9:
                log(f"WARNING {p} differs from {src}: relative Frobenius {err:.2e}")
        return p
    assert src.exists(), f"neither {p} nor {src} exists (push results/r2/sigma with gpu/push_r2.sh ... robust)"
    d.mkdir(parents=True, exist_ok=True)
    z = np.load(src)
    tmp = d / "stats.tmp.npz"
    np.savez(tmp, mean=z["mean"], cov=z["cov"], n=z["n"])
    os.replace(tmp, p)
    json.dump({"source": str(src.relative_to(ROOT)), "note": "stats rebuilt from a saved training Sigma"},
              open(d / "config.json", "w"), indent=1)
    log(f"rebuilt {p} from {src.name}")
    return p


def sigma(model, layer, label):
    return rc.Sigma(ensure_stats(model, layer, label), label, OUT)


def sp_name(model, layer, width, k, M, L, rho, overflow=None, seed=0):  # train_sae.py's naming
    s = (f"{model}__{hook(layer)}__sparse__w{width}__k{k}__s{seed}__ds{SP_DS}__M{M}__L{L:g}__{VALUES}"
         f"__rho{rho:.4f}")
    return s + (f"__OF{overflow[0]}x{overflow[1]:g}" if overflow else "")


def train_sparse(group, model, layer, width, k, M, L, rho, overflow=None, tokens=None):
    """Train a codable-null SAE if needed (stats-only, so n_train = tokens); return its eval FVU."""
    rd = ROOT / "cache" / group / sp_name(model, layer, width, k, M, L, rho, overflow)
    if not (rd / "train.json").exists():
        log(f"train {rd.name}")
        py(ROOT / "train_sae.py", "--model", model, "--hook", hook(layer), "--cond", "sparse", "--width", width,
           "--k", k, "--tokens", tokens or TOKENS - EVAL, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", 0,
           "--out", f"cache/{group}", "--sparse_M", M, "--sparse_L", L, "--data_seed", SP_DS,
           "--sparse_values", VALUES, "--sparse_rho", rho, "--save_params",
           *(["--sparse_overflow_M", overflow[0], "--sparse_overflow_L", overflow[1]] if overflow else []), *EXTRA)
    return json.load(open(rd / "train.json"))["fvu_eval"], rd


def next_rho(pts, T):
    """Secant step through the two tried (rho, F) points that bracket T, else the two nearest to T (r2_job_sparse2)."""
    pts = sorted(pts.items())
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


def calibrate(trainf, T, G, max_runs, f0=None):
    """The rule in the docstring. trainf(rho) -> FVU. f0: a known rho = 0 FVU (not retrained, not counted).
    Returns (rho, tried {rho: FVU of runs trained here}, matched, codable)."""
    tried = {}
    if f0 is None:
        tried[0.0] = f0 = trainf(0.0)
    pts = {0.0: f0}
    if f0 >= T - TOL:
        return 0.0, tried, f0 <= T + TOL, f0 <= T + TOL
    rho = max(round(min(C_GUESS * (T - f0) / (G - f0), RHO_MAX), 2), 0.01)
    while len(tried) < max_runs:
        if rho is None or rho in pts:
            break
        pts[rho] = tried[rho] = trainf(rho)
        if abs(tried[rho] - T) <= TOL:
            return rho, tried, True, True
        rho = next_rho(pts, T)
    best = min(tried, key=lambda r: abs(tried[r] - T))
    return best, tried, abs(tried[best] - T) <= TOL, True


def save_cal(key, entry):
    f = OUT / "robust_calibration.json"
    cal = json.load(open(f)) if f.exists() else {}
    cal[key] = entry
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=1))
    os.replace(tmp, f)


def score_runs(scorer, sig, tried, rho, step, meta):
    for r in sorted(tried):
        rd = tried[r][1]
        if not done(rd.name):
            scorer.submit(rc.score, rd, sig, OUT, JOB, {**meta, "step": step if r == rho else step + "CAL", "rho": r,
                                                        "job_code_sha256": job_sha()}, n_fix=N_FIX)


# ------------------------------------------------------------------------------------------------------------- G ---
def step_g(scorer, layers):
    use_windows(GEMMA_WINDOWS)
    for layer in layers:
        sig = sigma(MODEL, layer, f"{SIGMA_MODEL}_L{layer}")
        settings = [(M, L, None) for M, L in G_SETTINGS] + ([(SP2_M, SP2_L, G_OVERFLOW)]
                                                             if layer in G_OVERFLOW_LAYERS else [])
        for M, L, of in settings:
            key = f"G|L{layer}|M{M}_L{L:g}" + (f"_OF{of[0]}x{of[1]:g}" if of else "")
            runs = {}

            def trainf(rho):
                runs[rho] = train_sparse(GROUP, MODEL, layer, WIDTH, K, M, L, rho, of if rho > 0 else None)
                return runs[rho][0]
            rho, tried, matched, codable = calibrate(trainf, TARGET[layer], GAUSS[layer], G_MAX,
                                                     f0=SP2_F0.get(layer) if of else None)
            entry = {"layer": layer, "M": M, "L": L, "overflow": of, "rho": rho, "target": TARGET[layer],
                     "gaussian": GAUSS[layer], "matched": matched, "codable": codable,
                     "known_f0": SP2_F0.get(layer) if of else None,
                     "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}}
            save_cal(key, entry)
            log(f"{key}: rho {rho:.2f}, FVU {tried.get(rho, float('nan')):.3f}, target {TARGET[layer]:.3f}, "
                f"{'matched' if matched else 'UNMATCHED'}{'' if codable else ' (not codable at this FVU)'}")
            meta = {"layer": layer, "cond": "sparse_generator", "dataset": f"{SIGMA_MODEL}_L{layer}", "whitened": False,
                    "model": "gemma-2-2b", "setting": {"M": M, "L": L, "overflow": of}, "setting_key": key,
                    "fvu_target": TARGET[layer], "matched": matched, "codable": codable}
            score_runs(scorer, sig, runs, rho, "G", meta)
    scorer.finish()


# ------------------------------------------------------------------------------------------------------------- M ---
def step_m(scorer, layers):
    scorer.finish()
    use_windows(M_WINDOWS)
    for layer in layers:
        sig = sigma(M_MODEL, layer, f"{M_SIGMA_MODEL}_L{layer}")
        runs = {}

        def trainf(rho):
            runs[rho] = train_sparse(GROUP, M_MODEL, layer, M_WIDTH, M_K, M_M, M_L, rho)
            return runs[rho][0]
        rho, tried, matched, codable = calibrate(trainf, M_TARGET[layer], M_GAUSS[layer], M_MAX)
        key = f"M|L{layer}"
        save_cal(key, {"layer": layer, "M": M_M, "L": M_L, "rho": rho, "target": M_TARGET[layer],
                       "gaussian": M_GAUSS[layer], "matched": matched, "codable": codable,
                       "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}})
        log(f"{key}: rho {rho:.2f}, FVU {tried[rho]:.3f}, target {M_TARGET[layer]:.3f}, "
            f"{'matched' if matched else 'UNMATCHED'}")
        meta = {"layer": layer, "cond": "sparse2", "dataset": f"{M_SIGMA_MODEL}_L{layer}", "whitened": False,
                "model": "gpt2", "windows": [list(w) for w in M_WINDOWS], "n_layers": 12, "setting_key": key,
                "setting": {"M": M_M, "L": M_L, "overflow": None}, "fvu_target": M_TARGET[layer], "matched": matched,
                "codable": codable}
        score_runs(scorer, sig, runs, rho, "M", meta)
    scorer.finish()
    use_windows(GEMMA_WINDOWS)


# ------------------------------------------------------------------------------------------------------------- P ---
class _Stop(Exception):
    pass


class MultiResid:
    """collect_acts.HFResid for several layers at once: the model is cut after the deepest layer, every requested
    layer's resid_post is kept, and the pass stops there. Same dtype handling (bf16 weights, embedding on the CPU)."""

    def __init__(self, layers, dev=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = dev or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.dtype = dtype or torch.bfloat16
        self.layers, self.top = sorted(layers), max(layers)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16)
        model.model.layers = model.model.layers[:self.top + 1]
        model.lm_head = torch.nn.Identity()
        self.embed = model.model.embed_tokens.eval()
        model.model.embed_tokens = torch.nn.Embedding(1, model.config.hidden_size)
        self.model = model.to(self.dtype).to(self.dev).eval()
        self.d_model = model.config.hidden_size
        self._out = {}
        for L in self.layers:
            self.model.model.layers[L].register_forward_hook(self._hook(L))

    def _hook(self, L):
        def f(module, inputs, output):
            self._out[L] = output[0] if isinstance(output, tuple) else output
            if L == self.top:
                raise _Stop
        return f

    def __call__(self, toks):
        self._out = {}
        try:
            self.model.model(inputs_embeds=self.embed(toks).to(self.dtype).to(self.dev))
        except _Stop:
            pass
        return self._out


def collect_layers(layers):
    """Gemma-2-2b resid_post at several layers from one forward pass, exactly as collect_acts.py --backend hf --dtype
    bfloat16 --ctx 1024 --batch 16 --n_tokens 10M collects one layer (same model truncation, token stream, BOS drop,
    fp16 cache, fp64 accumulation of per-batch fp32 sums). Each layer's acts.npy appears only when complete."""
    layers = sorted(layers)
    for L in layers:
        d = act_dir(P_MODEL, L)
        d.mkdir(parents=True, exist_ok=True)
        if (d / "acts.npy.part").exists():
            (d / "acts.npy.part").unlink()
    if SMOKE:
        time.sleep(5)  # stands in for the pass; the main process trains a null meanwhile
        for L in layers:
            d = act_dir(P_MODEL, L)
            for f in ("acts.npy", "stats.npz"):
                if not (d / f).exists():
                    os.symlink((SMOKE["acts"] / f).resolve(), d / f)
        log(f"smoke: linked {layers}")
        return
    import torch
    from collect_acts import token_batches
    t0 = time.time()
    resid = MultiResid(layers)
    tok, d, box = resid.tokenizer, resid.d_model, None
    mm = {L: np.lib.format.open_memmap(act_dir(P_MODEL, L) / "acts.npy.part", mode="w+", dtype=np.float16,
                                       shape=(TOKENS, d)) for L in layers}
    s, G, n = {L: np.zeros(d) for L in layers}, {L: np.zeros((d, d)) for L in layers}, 0
    log(f"collect layers {layers}: {TOKENS:,} tokens, {rc.disk_free_gb():.0f} GB free")
    with torch.no_grad():
        for toks in token_batches(tok, 1024, 16, 0):
            box = resid(toks)
            m = 0
            for L in layers:
                a = box[L][:, 1:].reshape(-1, d).float()[: TOKENS - n]
                mm[L][n:n + len(a)] = a.cpu().numpy().astype(np.float16)
                s[L] += a.sum(0).cpu().double().numpy()
                G[L] += (a.T @ a).cpu().double().numpy()
                m = len(a)
            if (n + m) // 1_000_000 != n // 1_000_000:
                log(f"collect: {n + m:,} tokens, {time.time() - t0:.0f}s")
            n += m
            if n >= TOKENS:
                break
    for L in layers:
        mm[L].flush()
    del mm
    for L in layers:
        dd = act_dir(P_MODEL, L)
        mean = s[L] / n
        cov = (G[L] - n * np.outer(mean, mean)) / (n - 1)
        src = SIGMA_SRC / f"{SIGMA_MODEL}_L{L}.npz"
        chk = None
        if src.exists():
            ref = np.load(src)["cov"]
            chk = float(np.linalg.norm(cov - ref) / np.linalg.norm(ref))
        tmp = dd / "stats.tmp.npz"
        np.savez(tmp, mean=mean, cov=cov, n=n)
        os.replace(tmp, dd / "stats.npz")
        json.dump({"model": HF_MODEL, "hook": hook(L), "n_tokens": TOKENS, "ctx": 1024, "batch": 16, "seed": 0,
                   "dtype": "bfloat16", "backend": "hf (r2_job_robust.py, one pass for layers %s)" % layers,
                   "d_model": d, "n_tokens_written": n, "dataset": "monology/pile-uncopyrighted", "bos_dropped": True,
                   "seconds": time.time() - t0, "cov_rel_frob_vs_saved_sigma": chk},
                  open(dd / "config.json", "w"), indent=1)
        (dd / "acts.npy.part").rename(dd / "acts.npy")
        log(f"layer {L}: {n:,} tokens; covariance vs saved Sigma, relative Frobenius {chk}")
    log(f"collection done in {time.time() - t0:.0f}s")


def start_collect(layers):
    per = TOKENS * 2304 * 2 / 1e9
    fit = len(layers) if SMOKE else int((rc.disk_free_gb() - COLLECT_MARGIN_GB) // per)
    if fit < 1:
        log(f"P: not enough disk for one layer ({rc.disk_free_gb():.0f} GB free, need {per + COLLECT_MARGIN_GB:.0f})")
        return None, []
    sub = layers[:fit]
    logf = open(OUT / "robust_collect.log", "a")
    p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--collect", *map(str, sub),
                          *(["--smoke"] if SMOKE else [])], cwd=ROOT, env=os.environ, stdout=logf,
                         stderr=subprocess.STDOUT)
    log(f"P: collecting layers {sub} in the background (pid {p.pid}, log {OUT / 'robust_collect.log'})")
    return p, sub


def drop_acts(layer):
    p = act_dir(P_MODEL, layer) / "acts.npy"
    if p.exists() or p.is_symlink():
        if SMOKE:
            assert p.is_symlink(), f"smoke mode only removes symlinks; {p} is not one"
        p.unlink()
        log(f"deleted {p} (stats.npz kept)")


def p_sigma(layer):
    """Training Sigma of step P's layer: the saved Gemma Sigma (smoke: the stand-in tokens' own stats.npz)."""
    if SMOKE:
        d = act_dir(P_MODEL, layer)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "stats.npz").exists():
            os.symlink((SMOKE["acts"] / "stats.npz").resolve(), d / "stats.npz")
        return rc.Sigma(d / "stats.npz", f"{P_MODEL}_L{layer}", OUT)
    return sigma(P_MODEL, layer, f"{SIGMA_MODEL}_L{layer}")


def p_real_name(layer):
    return f"{P_MODEL}__{hook(layer)}__real__w{WIDTH}__k{K}__s0"


def p_null_name(layer):
    return sp_name(P_MODEL, layer, WIDTH, K, P_NULL_M, SP2_L, SP2_RHO[layer])


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def power_pass(rd, batches):
    """Per-latent firing count, sum of activations, sum of squares; FVU and second moments of the sample."""
    import torch
    dev = device()
    t = {p: torch.tensor(np.load(rd / f"{p}.npy"), device=dev) for p in ("W_enc", "b_enc", "b_dec", "W_dec")}
    k = json.load(open(rd / "train.json"))["config"]["k"]
    w, d = t["W_dec"].shape
    acc = {q: np.zeros(w) for q in ("fires", "s1", "s2")}
    sx, sxx, sse, n = np.zeros(d), 0.0, 0.0, 0
    with torch.no_grad():
        for x in batches:
            pre = (x - t["b_dec"]) @ t["W_enc"] + t["b_enc"]
            v, i = pre.topk(k, dim=-1)
            v = torch.relu(v)
            codes = torch.zeros_like(pre).scatter_(-1, i, v)
            acc["fires"] += (codes > 0).float().sum(0).cpu().double().numpy()  # MPS has no float64: cast on the CPU
            acc["s1"] += codes.sum(0).cpu().double().numpy()
            acc["s2"] += codes.pow(2).sum(0).cpu().double().numpy()
            recon = codes @ t["W_dec"] + t["b_dec"]
            sse += float((x - recon).pow(2).sum())
            sx += x.sum(0).cpu().double().numpy()
            sxx += float(x.pow(2).sum())
            n += len(x)
            del pre, codes, recon
    var = sxx / n - float(sx @ sx) / n ** 2
    return {**acc, "n": n, "fvu": sse / n / var, "total_var": var, "l0": float(acc["fires"].sum() / n),
            "W_dec": t["W_dec"].cpu().numpy()}


def acts_batches(layer, scale, batch=8192):
    import torch
    dev = device()
    a = np.load(act_dir(P_MODEL, layer) / "acts.npy", mmap_mode="r")
    for b in range(len(a) - N_POWER, len(a), batch):
        yield torch.from_numpy(np.asarray(a[b:min(b + batch, len(a))], dtype=np.float32)).to(dev) * scale


def null_batches(sig, layer, scale):
    from train_sae import sparse_batches, sparse_setup
    dev = device()
    sp = sparse_setup(sig.cov, sig.mean, P_NULL_M, SP2_L, SP_DS, dev, VALUES, SP2_RHO[layer])
    return sparse_batches(sp, N_POWER, 8192, POWER_SEED + 1000 * SP_DS, scale, dev)


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def power_row(rd, sig, pw, meta, ref_path):
    """CPU part of step P: per-latent arrays to power/<run>.npz and the row (definitions in the docstring)."""
    t0 = time.time()
    info = json.load(open(rd / "train.json"))
    c = info["config"]
    W = pw["W_dec"].astype(np.float64)
    norm = np.linalg.norm(W, axis=1)
    D = W / norm[:, None]
    n = pw["n"]
    freq, mean_act, msq = pw["fires"] / n, pw["s1"] / n, pw["s2"] / n
    pi = msq * norm ** 2
    live = pw["fires"] > 0
    live_eval = np.load(rd / "fires.npy") > 0
    tr = np.trace(sig.cov)
    rq = np.concatenate([((D[i:i + 4096] @ sig.cov) * D[i:i + 4096]).sum(1) for i in range(0, len(D), 4096)])
    rq /= tr / D.shape[1]
    Dl, pl, rql = D[live], pi[live], rq[live]
    Cw = (Dl * pl[:, None]).T @ Dl / pl.sum()
    Cu = Dl.T @ Dl / len(Dl)
    ew, eu = np.sort(np.linalg.eigvalsh(Cw))[::-1], np.sort(np.linalg.eigvalsh(Cu))[::-1]
    cs = rc._cloud_stats(D.astype(np.float32), live, c["k"], c["seed"], N_FIX, sig.slopes)
    cs_eval = rc._cloud_stats(D.astype(np.float32), live_eval, c["k"], c["seed"], N_FIX, sig.slopes)
    order = np.argsort(pl)[::-1]
    n1 = max(len(pl) // 100, 1)
    lq = np.log10(np.clip(rql, 1e-12, None))
    gw = rc.ratio(rc.slopes(ew), sig.slopes)
    gu = cs["gamma_fixed"]
    bulk = [str(w) for w in rc.WINDOWS[2:]]
    hb_a = all(abs(gw[w] - 1) <= 0.25 and gw[w] - gu[w] >= 0.15 for w in bulk)
    med_body = float(np.median(lq[order[n1:]]))
    ref = json.loads(ref_path.read_text()) if ref_path is not None and ref_path.exists() else None
    row = {"job": JOB, "run": rd.name, **meta, "sigma_label": sig.label, "config": c, "git_hash": rc.git_hash(),
           "code_sha256": rc.code_sha256(), "job_code_sha256": job_sha(), "width": c["width"], "k": c["k"],
           "seed": c["seed"], "fvu_eval": info["fvu_eval"], "n_live_eval": int(live_eval.sum()),
           "train_seconds": info["seconds"], "n_train_tokens": info["n_train_tokens"],
           "power_n_tokens": n, "power_fvu": pw["fvu"], "power_l0": pw["l0"], "n_live": int(live.sum()),
           "sigma_slope": sig.slopes, "sigma_n": sig.n,
           "unweighted": {k: cs[k] for k in ("gamma_fixed", "gamma_fixed_sd", "gamma_full", "dec_fixed", "dec_full",
                                             "n_fix_used")},
           "unweighted_uncentred_gamma": rc.ratio(rc.slopes(eu), sig.slopes),
           "power_weighted_gamma": gw, "power_weighted_slope": rc.slopes(ew),
           "power_weighted_eig": ew[:1100].tolist(),
           "power_share_top10": float(pl[order[:10]].sum() / pl.sum()),
           "power_share_top1pct": float(pl[order[:n1]].sum() / pl.sum()), "n_top1pct": int(n1),
           "power_ess_frac_live": float(pl.sum() ** 2 / (pl ** 2).sum() / len(pl)),
           "sum_pi_over_total_var": float(pl.sum() / pw["total_var"]),
           "rq_median_log10": {"top10": float(np.median(lq[order[:10]])), "rest": float(np.median(lq[order[10:]])),
                               "top1pct": float(np.median(lq[order[:n1]])), "rest_1pct": med_body,
                               "all_live": float(np.median(lq))},
           "rq_quantiles_log10": {str(q): float(np.quantile(lq, q)) for q in (.01, .1, .5, .9, .99)},
           "spearman_pi_rq": float(np.corrcoef(_rank(pl), _rank(rql))[0, 1]),
           "spearman_freq_rq": float(np.corrcoef(_rank(freq[live]), _rank(rql))[0, 1]),
           "head_body": {"a_weighted_near_1_and_above_unweighted": hb_a, "b_body_rq_below_0.5": med_body < 0.5,
                         "present": bool(hb_a and med_body < 0.5)},
           "reproduction": {"gamma_fixed_eval_live": cs_eval["gamma_fixed"], "n_live_eval": int(live_eval.sum()),
                            "fvu_eval": info["fvu_eval"],
                            "ref_run": ref["run"] if ref else None, "ref_job": ref["job"] if ref else None,
                            "ref_fvu_eval": ref["fvu_eval"] if ref else None,
                            "ref_n_live": ref["n_live"] if ref else None,
                            "ref_gamma_fixed": ref["gamma_fixed"] if ref else None},
           "stats_seconds": time.time() - t0, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "power").mkdir(parents=True, exist_ok=True)
    tmp = OUT / "power" / f"{rd.name}.tmp.npz"
    np.savez_compressed(tmp, fires=pw["fires"], freq=freq, mean_act=mean_act, mean_sq_act=msq, pi=pi, rayleigh=rq,
                        dec_norm=norm, live=live, live_eval=live_eval, n_tokens=n)
    os.replace(tmp, OUT / "power" / f"{rd.name}.npz")
    rc.write_row(OUT, JOB, row)
    b = bulk
    log(f"power {rd.name}: weighted bulk gamma {gw[b[0]]:.2f}/{gw[b[1]]:.2f}, unweighted {gu[b[0]]:.2f}/{gu[b[1]]:.2f}, "
        f"top10 {row['power_share_top10']:.2f}, body rq {med_body:+.2f}, head/body "
        f"{row['head_body']['present']} [{row['stats_seconds']:.0f}s]")
    return row


def do_real(scorer, layer):
    rd = ROOT / "cache" / PGROUP / p_real_name(layer)
    if not (rd / "train.json").exists() or not (rd / "W_enc.npy").exists():
        log(f"train P {rd.name}")
        py(ROOT / "train_sae.py", "--model", P_MODEL, "--hook", hook(layer), "--cond", "real", "--width", WIDTH,
           "--k", K, "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", 0,
           "--out", f"cache/{PGROUP}", "--save_params", *EXTRA)
    sig = p_sigma(layer)
    scale = json.load(open(rd / "train.json"))["input_scale"]
    pw = power_pass(rd, acts_batches(layer, scale))
    drop_acts(layer)
    meta = {"step": "P", "layer": layer, "cond": "real", "dataset": f"{SIGMA_MODEL}_L{layer}", "whitened": False,
            "model": "gemma-2-2b", "power_tokens": "last 1M cached tokens (500k held out + 500k training)"}
    cfgp = act_dir(P_MODEL, layer) / "config.json"
    if cfgp.exists():
        meta["collection"] = json.load(open(cfgp))
    scorer.submit(power_row, rd, sig, pw, meta, OUT / "rows" / "real" / f"{p_real_name(layer)}.json")


def do_null(scorer, layer):
    sig = p_sigma(layer)
    _, rd = train_sparse(PGROUP, P_MODEL, layer, WIDTH, K, P_NULL_M, SP2_L, SP2_RHO[layer])
    scale = json.load(open(rd / "train.json"))["input_scale"]
    pw = power_pass(rd, null_batches(sig, layer, scale))
    meta = {"step": "P", "layer": layer, "cond": "sparse2", "dataset": f"{SIGMA_MODEL}_L{layer}", "whitened": False,
            "model": "gemma-2-2b", "rho": SP2_RHO[layer], "setting": {"M": P_NULL_M, "L": SP2_L, "overflow": None},
            "power_tokens": f"1M fresh generator tokens, stream {POWER_SEED + 1000 * SP_DS}"}
    scorer.submit(power_row, rd, sig, pw, meta, OUT / "rows" / "sparse2" / f"{p_null_name(layer)}.json")


def hf_access():
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(HF_MODEL)
        return True
    except Exception as e:
        log(f"P: no access to {HF_MODEL} ({type(e).__name__}: {e}); real SAEs skipped on this run")
        return False


def step_p(scorer, layers):
    use_windows(GEMMA_WINDOWS)
    pend_null = [L for L in layers if not done(p_null_name(L))]
    pend_real = [L for L in layers if not done(p_real_name(L))]
    for L in layers:
        if L not in pend_real:
            drop_acts(L)
    if pend_real and not SMOKE and not hf_access():
        pend_real = []
    for L in layers:  # Sigma for the nulls before any collection rewrites stats.npz
        p_sigma(L)
    proc, coll, fails = None, [], 0
    while pend_real or pend_null:
        if proc is not None and proc.poll() is not None:
            missing = [L for L in coll if not (act_dir(P_MODEL, L) / "acts.npy").exists()]
            if proc.returncode != 0 or missing:
                fails += 1
                log(f"P: collection exited {proc.returncode}, layers without tokens {missing} (attempt {fails}); "
                    f"see robust_collect.log")
                if fails >= 2:
                    log("P: giving up on real SAEs for this run")
                    pend_real = []
            proc = None
        ready = [L for L in pend_real if (act_dir(P_MODEL, L) / "acts.npy").exists()]
        if ready:
            do_real(scorer, ready[0])
            pend_real.remove(ready[0])
            continue
        if pend_real and proc is None:
            todo = [L for L in pend_real if L not in ready]
            proc, coll = start_collect(todo)
            if proc is None:
                pend_real = []
            if fails and proc is not None:  # after a failed attempt (e.g. GPU memory) the collection runs alone
                proc.wait()
            continue
        if pend_null:
            L = pend_null.pop(0)
            try:
                do_null(scorer, L)
            except subprocess.CalledProcessError as e:
                if proc is None:
                    raise
                log(f"P: null at layer {L} failed while collecting ({e}); retried after the collection")
                pend_null.append(L)
                proc.wait()
            continue
        if proc is not None:
            log("P: waiting for the collection")
            while proc.poll() is None:
                time.sleep(20)
    if proc is not None:
        proc.wait()
    scorer.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_g", action="store_true")
    ap.add_argument("--skip_m", action="store_true")
    ap.add_argument("--skip_p", action="store_true")
    ap.add_argument("--g_layers", type=int, nargs="*", default=None, help="default 13 0 (add 21 on a later run)")
    ap.add_argument("--m_layers", type=int, nargs="*", default=None)
    ap.add_argument("--p_layers", type=int, nargs="*", default=None)
    ap.add_argument("--collect", type=int, nargs="*", default=None, help=argparse.SUPPRESS)  # background worker
    a = ap.parse_args()
    if a.smoke:
        smoke_config()
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # two processes share the card
    OUT.mkdir(parents=True, exist_ok=True)
    if a.collect is not None:
        collect_layers(a.collect)
        return
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, job sha256 {job_sha()[:12]}, out {OUT}, "
        f"{rc.disk_free_gb():.0f} GB free")
    scorer = rc.Scorer()
    if not a.skip_g:
        log("G: generator sensitivity")
        step_g(scorer, a.g_layers if a.g_layers is not None else G_LAYERS)
    if not a.skip_m:
        log("M: codable null on GPT-2 small")
        step_m(scorer, a.m_layers if a.m_layers is not None else M_LAYERS)
    if not a.skip_p:
        log("P: feature power")
        step_p(scorer, a.p_layers if a.p_layers is not None else P_LAYERS)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:  # rerun locally after the pull; never fail the job on it
        log(f"r2_summary failed ({e}); rerun locally")
    log("JOB DONE robust")


if __name__ == "__main__":
    main()
