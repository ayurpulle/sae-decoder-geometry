"""Round-2 GPU job 1 (pod 1, needs Gemma-2-2B access): Q2, Q1, Q4 on real Gemma-2-2B activations. Unattended and
restartable: a run whose row exists is skipped, a trained but unscored run is scored, activation caches are collected
only while a run still needs them. Launch with gpu/run_job.sh (watchdog, markers, pod stop).

PRE-REGISTRATION (fixed before any run; experiments/r2_summary.py implements it unchanged)
==========================================================================================
Common to every step
  Recipe (as the existing GPU queue's steps A and B, so results are comparable): Gemma-2-2B resid_post, 10M Pile
  tokens cached per layer by collect_acts.py (ctx 1024, bf16, seed 0, BOS dropped), last 500k held out; TopK SAE
  (train_sae.py) width 16,384, 3 epochs, batch 4096, AuxK 512, lr 2e-4, TF32 on CUDA.
  Windows: 0-based slices (3,50), (10,150), (30,450), (100,1000) of the descending eigenvalues = ranks 4-50, 11-150,
  31-450, 101-1000. Primary window: ranks 11-150. Window rule: a reading counts as supported (or contradicted) only
  if it is supported (contradicted) in the primary window and in at least two of the other three; otherwise "mixed".
  Sigma: the training set's own covariance (stats.npz over all cached tokens). For the Gaussian condition this is
  the covariance it samples from.
  Spectra: centred covariance of live decoder rows (fired at least once on the eval split), geometry._eigvals.
  Primary scoring: fixed live count, 8,192 random live latents (mean of 4 draws, rng seed k*100+seed as in
  control_analyze.py); if fewer than 8,192 are live, all are used and fixed_n_short is recorded. Full-live scoring
  is secondary. gamma = decoder slope / Sigma slope (per window); "raw" = decoder slope itself.
  Per-doubling slope s: least-squares slope of the statistic against log2 k, fitted per SAE seed; seed-level
  uncertainty is the sd over seeds (ddof 1) and se = sd / sqrt(n_seeds). Combined se of a difference of two means:
  sqrt(se_a^2 + se_b^2).

Q2  Layer 13, real vs Gaussian (N(mu, Sigma)), k in {8, 16, 24, 32, 48, 64, 128}, seeds 0, 1, 2 (42 SAEs).
  Question: does a Sigma-only SAE's gamma decline with k as steadily as real data's, or step once between k = 8
  and 16 and then flatten?
  Statement (i): "a Sigma-only (Gaussian) SAE's gamma declines with k as steadily as real data's over k = 16-128".
  Fit s over k in {16, 24, 32, 48, 64, 128} for real (s_R) and Gaussian (s_G), in gamma and raw units.
  Per window and unit:
    steady_G   := mean s_G has the sign of mean s_R (negative in gamma units, positive in raw units, where the
                  decoder slope flattens toward 0 as k rises), |mean s_G| > 2 se_G, and every leave-one-k-out fit
                  of s_G (6 refits, seed means) has that sign too. [Correction made before any job-1 run: the first
                  draft wrote "< 0" in both units, which reads every raw-unit decline as flat.]
    ratio      := mean s_G / mean s_R; LOO ratios from the same 6 refits of both.
  A window SUPPORTS (i) if, in both gamma and raw units: steady_G, ratio >= 0.75, min LOO ratio >= 0.5.
  A window CONTRADICTS (i) if, in both units: not steady_G (flat or LOO sign flip), or ratio < 0.5 with
  |mean s_R| - |mean s_G| > 2 combined se, signs as above (Gaussian significantly shallower).
  Otherwise the window is "partial". Window rule as above.
  Reported, not a verdict: s over k = 8-128, the k = 8 -> 16 step of each condition, full-live versions.

Q1  Layers 0, 3, 6, 9, 13, 17, 21, 24; real vs Gaussian at k = 40; seeds 0, 1, 2 (48 SAEs).
  Layer 13 k = 40 uses the same run names (and the same cache group) as Q2 would, so it is trained once.
  Question: is the mid-stack profile and the bulk flattening what real structure adds to a Sigma-only SAE?
  Statement (ii): "the real-minus-Gaussian difference in raw decoder slope has a mid-stack maximum in the top
  windows, and real is flatter than the null in the bulk windows at every layer".
  Per layer L and window: D(L) = mean over seeds of (raw fixed-n decoder slope, Gaussian) - (same, real), i.e. how
  much steeper real is than its null (slopes are negative; D > 0 means real steeper). se_D = combined se.
  Top windows (ranks 4-50, 11-150), sub-claim (a) "D has a mid-stack maximum":
    interior = {6, 9, 13, 17}, ends = {0, 3, 21, 24}; L* = argmax D over all 8 layers.
    SUPPORTED if L* is interior and D(L*) - D(e) > 2 sqrt(se_D(L*)^2 + se_D(e)^2) for every end layer e.
    CONTRADICTED if max over end layers of D >= max over interior of D (point estimates).
    Otherwise partial.
  Bulk windows (ranks 31-450, 101-1000), sub-claim (b) "real is flatter than the null at every layer":
    SUPPORTED if -D(L) > 2 se_D(L) at all 8 layers (real less steep than Gaussian beyond noise).
    CONTRADICTED if D(L) > 2 se_D(L) at any layer (real significantly steeper than its null there).
    Otherwise partial.
  Each window is scored on its own sub-claim, and the window rule (primary + two of the other three) decides (ii).
  Reported, not a verdict: the same in gamma units and with full-live scoring; real gamma's own layer profile.

Q4  Whitened baseline, layer 12 (collected here, 10M tokens): real data, k in {16, 40, 128}, seeds 0, 1, each
  trained twice on the same tokens: unwhitened (control) and whitened (12 SAEs).
  Whitening (train_sae.py --whiten_ridge 1e-3):
    x_w = (Sigma + eps I)^{-1/2} (x - mu),  eps = 1e-3 * mean eigenvalue of Sigma,
  mu, Sigma = mean and covariance of all 10M cached tokens (stats.npz), symmetric inverse square root via eigh.
  Decoder rows w (whitened space) are mapped back as u = w (Sigma + eps I)^{1/2} / ||w (Sigma + eps I)^{1/2}|| and
  scored exactly like every other SAE against Sigma (gamma_map). Also recorded: the whitened-space decoder slopes
  minus those of an isotropic Gaussian cloud of the same n and d (anisotropy A_w; 0 if isotropic), and an
  isotropic null: n_live directions drawn isotropic in whitened space, mapped and renormalised the same way, scored
  the same way (gamma_iso). If decoders were isotropic in whitened space, gamma_map would equal gamma_iso (expected
  near 1 because Cov(Sigma^{1/2} z) = Sigma; unit renormalisation moves it, so the job computes it per run rather
  than assuming 1).
  Question: is the decoder spectrum just Sigma reintroduced, i.e. are decoders trained on whitened data isotropic?
  Statement (iii): "whitening the input removes the decoder cloud's anisotropy in the original space beyond what
  Sigma^{1/2} mapping alone reintroduces".
  Per window, k and unit gamma (fixed-n primary): Delta(k) = mean over seeds of (gamma_map - gamma_iso),
  se = seed se of gamma_map.
    A window SUPPORTS (iii) if |Delta(k)| <= 0.05 at all three k.
    A window CONTRADICTS (iii) if |Delta(k)| > 0.05 and |Delta(k)| > 2 se at all three k.
    Otherwise partial. Window rule as above.
  Reported, not a verdict: A_w per window, gamma of the unwhitened control at the same k, basis_gamma of both.

Q5  Sparse covariance-matched null (added because the Gaussian null confounds "non-Gaussian
  structure" with "not sparsely codable": Gaussian-null SAEs reach eval FVU 0.53-0.60 against 0.18-0.29 for real).
  Fixed before any Q5 run (written into this docstring before the code that trains it).
  Data (train_sae.py --cond sparse): x = mu + s A, A = R Sigma^{1/2} (M x d), R = Q of the reduced QR of an M x d
  standard-normal matrix (R^T R = I_d), M = 65,536 sources; s i.i.d., nonzero with probability p = L/M, L = 64
  expected active sources per token, nonzero values N(0, 1)/sqrt(p) (a standard normal scaled by p^{-1/2}: zero mean,
  variance 1/p), so Cov(x) = Sigma exactly in expectation. mu, Sigma are the training-set statistics the Gaussian
  cond uses. Data seed 0 for the dictionary and the token streams (training epoch e: 7_000_000 + e, eval 9_000_000,
  1M-token check sample 8_000_000); SAE seeds 0, 1 set only the SAE initialisation, so their spread is init noise.
  Each row records the generator settings and the 1M-token check sample's covariance slopes in all four windows;
  the ground-truth dictionary (unit-normalised rows of A, all 65,536) is scored once per layer like a decoder.
  Runs: (a) layer 13, k in {8, 16, 24, 32, 48, 64, 128}, SAE seeds 0, 1 (14 SAEs); (b) each Q1 layer, k = 40, seeds
  0, 1 (16 SAEs; layer 13 k = 40 is new here). Width, tokens and epochs as the Gaussian runs beside them.
  Statement (v): "the signatures that separate real from Gaussian-null SAEs (flatter bulk; mid-stack profile of the
  top-window slope; steady decline of gamma with k over 16-128) also separate real from the sparse null".
  A signature is tested against the sparse null only where it separates real from the Gaussian null, judged by the
  same rules as (i) and (ii); f is the fraction of the real-vs-Gaussian gap that remains between real and sparse:
    S1 flatter bulk (windows ranks 31-450, 101-1000; raw fixed-n decoder slope at k = 40, 8 layers). Applies in a
      window if (ii)(b) supports for the Gaussian there. f = sum_L D_sp(L) / sum_L D_g(L), D_null(L) as in (ii).
      The window SUPPORTS (v) if f >= 0.75 and (ii)(b) also supports with the sparse null in place of the Gaussian;
      CONTRADICTS if f < 0.5; partial otherwise.
    S2 mid-stack top-window profile (windows ranks 4-50, 11-150). Applies if (ii)(a) supports for the Gaussian.
      Amplitude P(D) = max over interior layers {6,9,13,17} of D - max over end layers {0,3,21,24} of D;
      f = P(D_sp) / P(D_g). SUPPORTS if f >= 0.75 and (ii)(a) supports with the sparse null; CONTRADICTS if f < 0.5.
    S3 decline of gamma with k over k = 16-128 at layer 13 (all four windows). Applies in a window if the (i) rule
      for the Gaussian says "contradicts" there (the Gaussian does not decline as steadily as real). With s fitted per
      seed as in (i): f = (mean s_sp - mean s_R) / (mean s_g - mean s_R), in gamma and raw units. SUPPORTS if
      f >= 0.75 in both units and the (i) rule with the sparse null also says "contradicts"; CONTRADICTS if f < 0.5 in
      both units; partial otherwise.
  Window rule per group: the profile group takes S2's verdict in the top windows and S1's in the bulk windows; S3
  uses all four. A group is "moot" when the Gaussian null does not separate it ((ii) not "supports" for the profile
  group, (i) not "contradicts" for S3); otherwise its verdict needs the primary window plus two others, a
  non-applicable window counting as neither. (v) is SUPPORTED if every non-moot group supports, CONTRADICTED if any
  non-moot group contradicts, "moot" if both groups are moot, mixed otherwise.
  Also reported, not verdicts: (i) and (ii) rerun with the sparse null; eval FVU, L0 and live count of real,
  Gaussian and sparse side by side (per layer at k = 40, per k at layer 13); the ground-truth dictionary's gamma;
  the check sample's slopes against Sigma's.

Scoring for every SAE also records live count, eval FVU and eval L0, the basis_gamma matrix test against Sigma on the
four windows, the decoder covariance and live mask, and per dataset Sigma's eigenvalues and covariance (r2_common).

Order (priority): Q2 (layer 13), then Q1 and Q5 at layer 13 while its cache is there; layer 13's acts.npy is deleted
only after Q2, Q1 and Q5 are done with it, then Q1's other layers one at a time with Q5(b) interleaved (collect, train,
score, ground-truth dictionary, delete), then Q4. Sparse runs need only stats.npz, so no cache is needed twice.
Peak disk: one 46 GB cache at a time (plus ~5 GB of model weights, ~150 MB per kept W_dec).
Usage (pod): python experiments/r2_job_real.py     smoke test (Mac): python experiments/r2_job_real.py --smoke
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "real"
HF_MODEL, MODEL = "google/gemma-2-2b", "gemma-2-2b"
WIDTH, TOKENS, EVAL, EPOCHS = 16384, 10_000_000, 500_000, 3
CONDS = ["real", "gaussian"]
Q2_LAYER, Q2_K, Q2_SEEDS = 13, [8, 16, 24, 32, 48, 64, 128], [0, 1, 2]
Q1_LAYERS, Q1_K, Q1_SEEDS = [0, 3, 6, 9, 13, 17, 21, 24], 40, [0, 1, 2]
Q4_LAYER, Q4_K, Q4_SEEDS, RIDGE = 12, [16, 40, 128], [0, 1], 1e-3
SP_M, SP_L, SP_DS, Q5A_K, Q5_SEEDS = 65536, 64, 0, [8, 16, 24, 32, 48, 64, 128], [0, 1]
GROUP, OUT, N_FIX = "saes_r2", ROOT / "results" / "r2", rc.N_FIX
SMOKE_SRC = None


def smoke_config():
    """Tiny everything; activations are file symlinks to an existing local cache (never written or deleted)."""
    global MODEL, WIDTH, TOKENS, EVAL, EPOCHS, Q2_K, Q2_SEEDS, Q1_LAYERS, Q1_K, Q1_SEEDS, Q4_K, Q4_SEEDS, GROUP, OUT
    global N_FIX, SMOKE_SRC, SP_M, SP_L, Q5A_K
    SMOKE_SRC = Path(os.environ.get("SMOKE_SRC", ROOT / "cache" / "acts" / "pythia-70m" / "blocks.3.hook_resid_post"))
    MODEL, WIDTH, TOKENS, EVAL, EPOCHS = "smoke-real", 512, 300_000, 16_384, 1
    Q2_K, Q2_SEEDS = [8, 16, 24, 32], [0, 1]
    Q1_LAYERS, Q1_K, Q1_SEEDS = [0, 13], 16, [0, 1]  # k=16 is in Q2_K: tests that layer 13 is not trained twice
    Q4_K, Q4_SEEDS = [16], [0, 1]
    SP_M, SP_L, Q5A_K = 4096, 16, [8, 16, 24, 32]
    GROUP, OUT, N_FIX = "saes_r2_smoke", ROOT / "results" / "r2_smoke", 256


def hook(layer):
    return f"blocks.{layer}.hook_resid_post"


def act_dir(layer):
    return ROOT / "cache" / "acts" / MODEL / hook(layer)


def run_name(layer, cond, k, seed, white=False):
    return (f"{MODEL}__{hook(layer)}__{cond}__w{WIDTH}__k{k}__s{seed}" + ("__white" if white else "")
            + (f"__ds{SP_DS}" if cond == "sparse" else ""))  # train_sae.py's naming


def gt_name(layer):
    return f"{MODEL}__{hook(layer)}__groundtruth__M{SP_M}__L{SP_L:g}__ds{SP_DS}"


STATS_ONLY = ("gaussian", "sparse")  # conditions that train from stats.npz alone


def run_dir(*a, **kw):
    return ROOT / "cache" / GROUP / run_name(*a, **kw)


def done(*a, **kw):
    return rc.row_path(OUT, JOB, run_name(*a, **kw)).exists()


def ensure_cache(layer):
    d = act_dir(layer)
    if (d / "acts.npy").exists():
        return
    if SMOKE_SRC is not None:
        d.mkdir(parents=True, exist_ok=True)
        for f in ("acts.npy", "stats.npz"):
            if not (d / f).exists():
                os.symlink((SMOKE_SRC / f).resolve(), d / f)
        log(f"smoke: linked {d} -> {SMOKE_SRC}")
        return
    log(f"collect layer {layer} ({rc.disk_free_gb():.0f} GB free)")
    py(ROOT / "collect_acts.py", "--backend", "hf", "--model", HF_MODEL, "--hook", hook(layer), "--ctx", 1024,
       "--batch", 16, "--dtype", "bfloat16", "--n_tokens", TOKENS)


def drop_cache(layer):
    p = act_dir(layer) / "acts.npy"
    if p.exists() or p.is_symlink():
        if SMOKE_SRC is not None:
            assert p.is_symlink(), f"smoke mode only ever removes symlinks, {p} is not one"
        p.unlink()
        log(f"deleted {p} (stats.npz kept); {rc.disk_free_gb():.0f} GB free")


def sigma(layer):
    return rc.Sigma(act_dir(layer) / "stats.npz", f"{MODEL}_L{layer}", OUT)


def run(scorer, step, layer, cond, k, seed, white=False):
    if done(layer, cond, k, seed, white):
        return
    rd = run_dir(layer, cond, k, seed, white)
    if not (rd / "train.json").exists():
        if cond not in STATS_ONLY or not (act_dir(layer) / "stats.npz").exists():
            ensure_cache(layer)
        # With acts.npy present train_sae uses n_train = n_cached - EVAL (9.5M, as in the GPU queue). A Gaussian run
        # (or sparse) run restarted after its cache was deleted runs from stats.npz alone, where n_train = --tokens: pass 9.5M so
        # the step count and sample stream are identical.
        toks = TOKENS if (act_dir(layer) / "acts.npy").exists() else TOKENS - EVAL
        log(f"train {step} {rd.name}")
        py(ROOT / "train_sae.py", "--model", MODEL, "--hook", hook(layer), "--cond", cond, "--width", WIDTH, "--k", k,
           "--tokens", toks, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", seed, "--out", f"cache/{GROUP}",
           *(["--whiten_ridge", RIDGE] if white else []),
           *(["--sparse_M", SP_M, "--sparse_L", SP_L, "--data_seed", SP_DS] if cond == "sparse" else []))
    S = None
    sig = sigma(layer)
    if white:
        from train_sae import whiten_mats
        S = whiten_mats(sig.cov, RIDGE)[1]
    meta = {"step": step, "layer": layer, "cond": cond, "dataset": f"{MODEL}_L{layer}", "whitened": white}
    scorer.submit(rc.score, rd, sig, OUT, JOB, meta, n_fix=N_FIX, whiten_S=S)


def all_done(runs):
    return all(done(*r[:4], **({"white": r[4]} if len(r) > 4 else {})) for r in runs)


def needs_acts(runs):
    """True if some run still has to be trained and needs real tokens (Gaussian runs need only stats.npz)."""
    for r in runs:
        white = len(r) > 4 and r[4]
        if not done(*r[:4], white=white) and not (run_dir(*r[:4], white=white) / "train.json").exists():
            if r[1] not in STATS_ONLY or not (act_dir(r[0]) / "stats.npz").exists():
                return True
    return False


def ground_truth(scorer, layer):
    """Score the sparse null's ground-truth dictionary at this layer once (after its sparse runs)."""
    if rc.row_path(OUT, JOB, gt_name(layer)).exists():
        return
    import json
    from train_sae import sparse_dictionary
    sig = sigma(layer)
    tj = sorted((ROOT / "cache" / GROUP).glob(f"{MODEL}__{hook(layer)}__sparse__*__ds{SP_DS}/train.json"))
    settings = {"M": SP_M, "L": SP_L, "p": SP_L / SP_M, "data_seed": SP_DS, "values": "N(0,1)/sqrt(p)",
                "dict_seed": 5_000_000 + SP_DS}
    if tj:
        settings["check_1M"] = rc.sparse_meta(json.load(open(tj[0]))["sparse"], sig)["check_1M"]
    meta = {"step": "GT", "layer": layer, "cond": "sparse_dictionary", "dataset": f"{MODEL}_L{layer}", "whitened": False}
    scorer.submit(rc.score_dictionary, sparse_dictionary(sig.cov, SP_M, SP_DS), sig, OUT, JOB, gt_name(layer), meta,
                  settings, n_fix=N_FIX)


def layer_block(scorer, step, layer, runs, keep=False):
    """runs: tuples (layer, cond, k, seed[, white]); a sparse run is logged as step Q5 whatever block it is in."""
    if all_done(runs):
        log(f"{step} layer {layer}: all {len(runs)} runs done")
    else:
        if needs_acts(runs):
            ensure_cache(layer)
        sigma(layer)  # saves Sigma's eigenvalues and covariance before any delete
        for r in runs:
            run(scorer, "Q5" if r[1] == "sparse" else step, *r)
    if any(r[1] == "sparse" for r in runs):
        ground_truth(scorer, layer)
    scorer.finish()
    if not keep:
        drop_cache(layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        smoke_config()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, out {OUT}, {rc.disk_free_gb():.0f} GB free")
    if SMOKE_SRC is None:
        from transformers import AutoConfig
        try:
            AutoConfig.from_pretrained(HF_MODEL)
        except Exception as e:
            sys.exit(f"No access to {HF_MODEL} ({type(e).__name__}: {e}). Set HF_TOKEN or run gpu/hf_login.sh.")
    scorer = rc.Scorer()

    q2 = [(Q2_LAYER, c, k, s) for s in Q2_SEEDS for k in Q2_K for c in CONDS]
    q1_13 = [(Q2_LAYER, c, Q1_K, s) for s in Q1_SEEDS for c in CONDS]  # same names as Q2 would use: never twice
    q5a = [(Q2_LAYER, "sparse", k, s) for s in Q5_SEEDS for k in Q5A_K]
    q5b = lambda L: [(L, "sparse", Q1_K, s) for s in Q5_SEEDS if not (L == Q2_LAYER and Q1_K in Q5A_K)]  # noqa: E731
    log(f"Q2: layer {Q2_LAYER} k sweep ({len(q2)} SAEs)")
    layer_block(scorer, "Q2", Q2_LAYER, q2, keep=True)
    log(f"Q1 + Q5: layer {Q2_LAYER} at k={Q1_K}, sparse k sweep and k={Q1_K}, while its cache is there")
    layer_block(scorer, "Q1", Q2_LAYER, q1_13 + q5a + q5b(Q2_LAYER), keep=False)  # layer 13's last use
    for layer in [L for L in Q1_LAYERS if L != Q2_LAYER]:
        log(f"Q1 + Q5: layer {layer}")
        layer_block(scorer, "Q1", layer, [(layer, c, Q1_K, s) for s in Q1_SEEDS for c in CONDS] + q5b(layer))

    q4 = [(Q4_LAYER, "real", k, s, w) for s in Q4_SEEDS for k in Q4_K for w in (True, False)]
    log(f"Q4: whitened baseline at layer {Q4_LAYER} ({len(q4)} SAEs)")
    layer_block(scorer, "Q4", Q4_LAYER, q4)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:  # the summary is rerun locally after the pull; never fail the job on it
        log(f"r2_summary failed on the pod ({e}); rerun locally")
    log("JOB DONE real")


if __name__ == "__main__":
    main()
