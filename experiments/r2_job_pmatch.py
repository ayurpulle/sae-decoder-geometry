"""Round-2 GPU job 6 ("pmatch"): the power-matched sparse null. Does unequal feature power explain why real decoders are
flatter than the codable null in the bulk windows? Unattended and restartable like the other jobs: a run whose row
exists is skipped, a trained but unscored run is scored, calibration reads finished runs back, dictionaries are built
once and kept. Rows go to results/r2/runs_pmatch.jsonl, dictionary reports to results/r2/pmatch/, per-latent power of
the accepted null SAEs to results/r2/power_pmatch/.

PRE-REGISTRATION (written 2026-11-13, before any run of this job; notes/prereg_pmatch.md says the same in prose;
experiments/r2_summary.py, section "pmatch", implements it)
=====================================================================================================================
Windows as in every round-2 job: 0-based slices (3,50), (10,150), (30,450), (100,1000) = ranks 4-50, 11-150, 31-450,
101-1000; the bulk windows are the last two. gamma = decoder slope / Sigma slope, fixed-n unit scoring (8,192 live
latents, 4 draws, r2_common.score), Sigma = the layer's saved training covariance (results/r2/sigma).
Reference numbers (gamma_fixed, bulk windows 31-450 / 101-1000; FVU):
  real (mean of job 1's three k = 40 SAEs)        layer 13  .595 / .863  FVU .288     layer 0  .375 / .623  FVU .114
  V0 equal-power null, M 8192 L 16 (sparse2 s0)   layer 13  .930 / 1.156 FVU .275     layer 0  .906 / 1.030 FVU .112
  gap V0 - real                                   layer 13  .335 / .293               layer 0  .531 / .407
  V0 at layer 13, M 8192 L 8 (robust step G)      .758 / 1.035 (rho .43)    M 8192 L 32: 1.168 / 1.337 (rho .42)
(seed-to-seed spread at a fixed generator: real <= .005, null <= .011 in these windows.)

The null. x = mu + sqrt(1 - rho) (s - E s) A + sqrt(rho) g, g ~ N(0, Sigma), s with M sources, L active per token,
half-normal values with Var(s_j) = 1 (all as the codable null V0); only the dictionary A changes (train_sae.py
--sparse_dict). Every A satisfies A^T A = Sigma exactly (experiments/r2_pmatch_gen.py, proof in its docstring), so
Cov(x) = Sigma; checked twice (relative Frobenius error of A^T A in float64 and float32 at build time, the job stops if
the float64 error exceeds 1e-6; and train_sae.py's 1M-token check sample).
V1 (power). Target: the real k = 40, seed-0 SAE at the same layer (job robust, step P, results/r2/power/<real
run>.npz), its live latents' powers pi_i = E[a_i^2] ||d_i||^2 on 1M tokens. Atom j takes the power of the latent at
power rank floor((j + 0.5) n / M) (the quantiles of the measured distribution), scaled to sum to tr Sigma. A power
vector is achievable with A^T A = Sigma only if it is majorized by Sigma's eigenvalues (Schur-Horn); the head is
capped at 0.9 to 1 times the eigenvalue partial sums (factor 1 - 0.1 (1 - m/d) for the top m) and the excess spread
over the other atoms in proportion; the moved share is reported. Directions start as V0's (Sigma^{1/2} z, independent
of power); the iteration then applies, round after round, the least linear change that makes A^T A = Sigma (the
optimal-transport map) and rescales rows toward the target powers, the k-th largest target going to the atom with the
k-th largest power; it stops at a 3% maximum power error or after 240 s. No SAE decoder direction is used.
What V1 holds fixed relative to V0: Sigma, mu, M, L, the value law, the dense share's form, the calibration rule; it
changes only the atom powers and whatever alignment of atom directions the covariance constraint then forces.
V0 is not rerun: its rows are runs_sparse2.jsonl (S2a/S2b, seed 0) and, for other (M, L), runs_robust.jsonl step G.
Not run: the variant with the measured (power, Rayleigh quotient) pairs. Locally (layer-13 Sigma, Gemma Scope 65k power
and quotients as stand-ins) the iteration did not approach the pairs (power errors 2-8 in log, quotient errors
0.3-0.6 in log10 after 25 rounds), so a fixed point with those pairs and A^T A = Sigma does not seem to exist: the
measured latents are not a decomposition of the covariance. The generator keeps the code (V2, V3) for local use.

Settings, in this order (key PM|V1|L<layer>|M<M>_L<L>|ds<ds>); SAE seed 0, width 16,384, k = 40, 9.5M tokens x 3:
  1. layer 13, M 8192, L 16, dictionary seed 0        (rule)
  2. layer 0,  M 8192, L 16, dictionary seed 0        (rule)
  3. layer 13, M 8192, L 8,  dictionary seed 0        (generator setting; V0 comparator from step G)
  4. layer 13, M 8192, L 32, dictionary seed 0        (generator setting; V0 comparator from step G)
  5. layer 13, M 8192, L 16, dictionary seed 1        (new dictionary draw and token streams; replicate of 1)
Dictionary seed for (layer, ds): 6_000_000 + layer + 1000 ds; the dictionary does not depend on L, so 3 and 4 reuse 1's.
Calibration per setting: target T = real FVU above, tolerance |F - T| <= 0.03. Run 1 at V0's accepted rho for that
(layer, M, L) (13: .43 at L 16 and L 8, .42 at L 32; 0: .13); run 2, if needed, a Newton step with V0's own slope dF/drho
(13/L16 .593, 0/L16 .711, 13/L8 .598, 13/L32 .640, from V0's calibration runs); run 3, if needed, a secant step through
the two tried points. The first run within tolerance is used, else the closest, flagged unmatched. No SAE starts
after --budget_min minutes (default 90). Every tried run is scored (step "PM" accepted, "PMCAL" others).
Manipulation check (reported, not part of the verdict): on 1M fresh generator tokens, the accepted null SAE's latent
power distribution (top-10 share, top-1% share, effective-sample fraction) beside the real SAE's.

Decision rule, per setting with an FVU-matched accepted run. N = V1's gamma_fixed, R = real, Z = V0 at the same (layer,
M, L). Per bulk window, closed fraction f = (Z - N) / (Z - R) and a class:
  closed     |N - R| <= 0.10;   overshoot  N < R - 0.10;   partial  f >= 0.30 (not closed or overshoot);   open  else.
  EXPLAINS           both bulk windows closed.
  DOES NOT EXPLAIN   at least one bulk window open.
  PARTLY EXPLAINS    otherwise (every window closed, overshoot or partial, not both closed; "overshoots" noted when
                     both overshoot).
Headline (settings 1 and 2): the shared verdict when both layers agree; otherwise PARTLY EXPLAINS; insufficient
unless both are FVU-matched. Generator robustness (settings 3 and 4, layer 13): "same at other (M, L)" if both give
setting 1's verdict, "differs" otherwise, insufficient if either is missing or unmatched. Replicate (setting 5): the
largest |N(ds 1) - N(ds 0)| over the bulk windows; above 0.05 the margin of 0.10 is flagged as too narrow for the draw.
Prediction. Two inputs, both fixed before any SAE is trained. (a) The generator alone: with a Gemma-Scope-like power
distribution the alignment forced by A^T A = Sigma makes V1's own atoms much flatter than Sigma in ranks 4-450 but
hardly in 101-1000. Local stand-ins (atom-cloud gamma, 4 windows; V0's atoms .99-1.00 everywhere): layer 13 with Gemma
Scope 65k L12 powers .15/.22/.51/.96; layer 13 with the equal-power null SAE's own latent powers .46/.50/.69/1.02;
layer 0 with Gemma Scope L0 powers .17/.28/.48/.84. (b) The local pilot: one-parameter power reweightings fit the bulk
at one layer but not the window profile, and a dense share at matched rho erased flattening built into the atoms.
Quantitative rule, recorded per setting in the summary: N is predicted as atom(V1) + [Z - atom(V0)], the SAE shifting
the power-matched atoms as it shifts the equal-power ones; each dictionary report stores both atom clouds before
training. From the stand-ins: layer 13 about .44-.62 / 1.12-1.17 (31-450 closed or overshoot, 101-1000 open), layer
0 about .39 / .87 (closed, partial). Verdict predicted: layer 13 DOES NOT EXPLAIN, layer 0 PARTLY EXPLAINS, headline
PARTLY EXPLAINS (probabilities: partly .45, does not .45, explains .10; because of (b), any miss is more likely to
leave the SAE steeper than the additive rule says, i.e. towards "does not"). Settings 3 and 4: DOES NOT EXPLAIN, the
same as setting 1. Replicate: within 0.05.

Budget on one 32 GB card: one null SAE (width 16,384, 9.5M tokens x 3 epochs, 1M-token check sample) 7.5-8.1 min wall
(sparse2 and step G logs: train_seconds 368-474); power pass on 1M generator tokens about 1 min; scoring about 2 min on
the CPU, overlapped with the next SAE. Three dictionaries (layer 13, layer 0, layer 13 second draw) are built in a
background thread on the faster float64 device, at most 240 s each; only the first delays the GPU. Expected 1.5 SAEs
per setting: 5 settings, about 8 SAEs x 9 min = 72 min, plus 4 min, plus 2 min summary: about 1.3 h. Worst case is cut
by the budget: no SAE starts after 90 min, so the last one ends with its power pass and score by about 102 min; with
run_job.sh's 10-minute grace the pod stops by about 112 min, inside the 2.0 h cap. Disk: about 0.3 GB per SAE (W_dec,
W_enc), 75 MB per dictionary, about 20 MB of results per SAE; under 6 GB in all. No real activations are needed.

  bash gpu/run_job.sh experiments/r2_job_pmatch.py 2.0 10     (pod, after: bash gpu/push_r2.sh <host> <port> pmatch)
  python experiments/r2_job_pmatch.py --smoke                  (laptop: every step at toy scale)
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
import r2_pmatch_gen as pg  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "pmatch"
OUT, N_FIX = ROOT / "results" / "r2", rc.N_FIX
SIGMA_SRC = ROOT / "results" / "r2" / "sigma"
GROUP, DICT_DIR = "saes_pm", ROOT / "cache" / "pmatch_dicts"
EXTRA = []
SMOKE = None
T0 = time.time()

MODEL, SIGMA_MODEL = "gemma-2-2b-sig", "gemma-2-2b"  # own stats-only dir: always the saved training Sigma
REAL_POWER_MODEL, REAL_WIDTH = "gemma-2-2b", 16384
WIDTH, TOKENS, EVAL, EPOCHS, K = 16384, 10_000_000, 500_000, 3, 40
VALUES = "halfnormal"
TARGET = {0: 0.114, 13: 0.288}
# (layer, M, L): V0's accepted rho and dF/drho from V0's own calibration runs (sparse2, robust step G)
V0_CAL = {(13, 8192, 16): (0.43, 0.593), (0, 8192, 16): (0.13, 0.711),
          (13, 8192, 8): (0.43, 0.598), (13, 8192, 32): (0.42, 0.640)}
TOL, RHO_MAX, MAX_RUNS = 0.03, 0.95, 3
PLAN = [("V1", 13, 8192, 16, 0), ("V1", 0, 8192, 16, 0), ("V1", 13, 8192, 8, 0), ("V1", 13, 8192, 32, 0),
        ("V1", 13, 8192, 16, 1)]
DICT_SEED = 6_000_000
N_POWER, POWER_SEED = 1_000_000, 9_500_000
BUDGET_MIN = 90.0  # no SAE starts later than this many minutes into the job
DICT_SECONDS, DICT_TOL, DICT_RQ_TOL, DICT_MAX_ITER = 240, 0.03, 0.02, 400  # per-dictionary cap and stopping rule


def smoke_config():
    """Toy scale on a laptop: pythia-70m's Sigma (d = 512) stands in for both layers, M = 1024 for every setting, and a
    synthetic power file stands in for each real one when it is absent."""
    global SMOKE, MODEL, SIGMA_MODEL, WIDTH, TOKENS, EVAL, EPOCHS, GROUP, DICT_DIR, OUT, N_FIX, EXTRA, N_POWER
    global REAL_POWER_MODEL, REAL_WIDTH, PLAN, V0_CAL, DICT_SECONDS
    src = Path("/Users/ayurpulle/Desktop/SAE/cache/acts/pythia-70m/blocks.3.hook_resid_post/stats.npz")
    SMOKE = {"stats": Path(os.environ.get("SMOKE_STATS", src))}
    MODEL, SIGMA_MODEL, REAL_POWER_MODEL, REAL_WIDTH = "pmatch-smoke", "pmatch-smoke", "pmatch-smoke", 1024
    WIDTH, TOKENS, EVAL, EPOCHS = 1024, 200_000, 16_384, 1
    GROUP, DICT_DIR, OUT, N_FIX = "pmatch_smoke", ROOT / "cache" / "pmatch_smoke_dicts", ROOT / "cache" / "pmatch_smoke_out", 256
    N_POWER, DICT_SECONDS, EXTRA = 32_768, 30, ["--batch", 2048]
    PLAN = [(v, L, 1024, l, ds) for v, L, _, l, ds in PLAN]
    V0_CAL = {(L, 1024, l): x for (L, _, l), x in V0_CAL.items()}


def hook(layer):
    return f"blocks.{layer}.hook_resid_post"


def job_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def gen_sha():
    return hashlib.sha256((ROOT / "experiments" / "r2_pmatch_gen.py").read_bytes()).hexdigest()


def done(name):
    return rc.row_path(OUT, JOB, name).exists()


def minutes():
    return (time.time() - T0) / 60


def ensure_stats(layer):
    """stats.npz of this job's own stats-only directory, copied from the saved training Sigma (results/r2/sigma), so the
    nulls see exactly the covariance the codable null V0 was trained on."""
    d = ROOT / "cache" / "acts" / MODEL / hook(layer)
    p = d / "stats.npz"
    src = SMOKE["stats"] if SMOKE else SIGMA_SRC / f"{SIGMA_MODEL}_L{layer}.npz"
    assert src.exists(), f"{src} missing (push it with gpu/push_r2.sh ... pmatch)"
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True)
        z = np.load(src)
        tmp = d / "stats.tmp.npz"
        np.savez(tmp, mean=z["mean"], cov=z["cov"], n=z["n"])
        os.replace(tmp, p)
        json.dump({"source": str(src), "note": "stats copied from a saved training Sigma"}, open(d / "config.json", "w"))
        log(f"wrote {p} from {src}")
    assert np.array_equal(np.load(p)["cov"], np.load(src)["cov"]), f"{p} differs from {src}"
    return p


def sigma(layer):
    return rc.Sigma(ensure_stats(layer), f"{SIGMA_MODEL}_L{layer}", OUT)


def real_power_path(layer):
    return OUT / "power" / f"{REAL_POWER_MODEL}__{hook(layer)}__real__w{REAL_WIDTH}__k{K}__s0.npz"


def real_power(layer):
    """(pi, rq, path) of the live latents of the real seed-0 SAE at this layer. Smoke: a synthetic stand-in is written
    when the file is absent. Off smoke a missing file returns None (the layer is skipped and logged)."""
    p = real_power_path(layer)
    if not p.exists():
        if not SMOKE:
            return None
        pi, rq = pg.standin_power(REAL_WIDTH, 100 + layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, pi=pi, rayleigh=rq, live=np.ones(len(pi), bool), standin=True)
        log(f"smoke: wrote stand-in power file {p}")
    z = np.load(p)
    live = z["live"].astype(bool)
    return z["pi"][live], z["rayleigh"][live], p


def tag(variant, layer, M, ds):
    return f"pm{variant}L{layer}M{M}ds{ds}"


def setting_key(variant, layer, M, L, ds):
    return f"PM|{variant}|L{layer}|M{M}_L{L}|ds{ds}"


def build_device():
    """Device for the float64 dictionary iteration: the GPU unless one float64 eigendecomposition of a 2304 x 2304
    matrix is faster on the CPU (some GPUs run float64 slowly)."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    X = torch.randn(2304, 2304, dtype=torch.float64)
    X = X @ X.T
    t = {}
    for dev in ("cuda", "cpu"):
        Y = X.to(dev)
        torch.linalg.eigh(Y)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        torch.linalg.eigh(Y)
        if dev == "cuda":
            torch.cuda.synchronize()
        t[dev] = time.time() - t0
    dev = min(t, key=t.get)
    log(f"dictionary device {dev} (float64 eigh of 2304 x 2304: cuda {t['cuda']:.2f}s, cpu {t['cpu']:.2f}s)")
    return dev


class DictBuilder:
    """Builds the planned dictionaries in plan order in one background thread while SAEs train; wait(tag) blocks until
    that dictionary exists (its path), cannot be built for want of a real power file (None), or failed (raises)."""

    def __init__(self, plan, sigs):
        todo = []
        for v, L, M, _, ds in plan:
            if tag(v, L, M, ds) not in [tag(*x) for x in todo]:
                todo.append((v, L, M, ds))
        self.ev = {tag(*x): threading.Event() for x in todo}
        self.res, self.err = {}, {}
        self.dev = build_device()
        self.th = threading.Thread(target=self._run, args=(todo, sigs), daemon=True)
        self.th.start()

    def _run(self, todo, sigs):
        for v, L, M, ds in todo:
            t = tag(v, L, M, ds)
            try:
                self.res[t] = ensure_dict(v, L, M, ds, sigs[L], self.dev)
            except Exception as e:  # raised where the dictionary is needed
                self.err[t] = e
                log(f"{t}: dictionary build failed: {type(e).__name__}: {e}")
            self.ev[t].set()

    def wait(self, t):
        if not self.ev[t].is_set():
            log(f"waiting for dictionary {t}")
        self.ev[t].wait()
        if t in self.err:
            raise self.err[t]
        return self.res.get(t)


def ensure_dict(variant, layer, M, ds, sig, dev=None):
    """Build (once) a dictionary; returns its path, or None without a real power file."""
    t = tag(variant, layer, M, ds)
    path, rep = DICT_DIR / f"{t}.npz", OUT / "pmatch" / f"{t}.json"
    if path.exists() and rep.exists():
        return path
    rp = real_power(layer)
    if rp is None:
        log(f"{t}: no real power file {real_power_path(layer)}; its settings are skipped on this run")
        return None
    pi, rq, src = rp
    p_share, rq_t = pg.targets(pi, rq, M)
    A, info, p_ach, rq_ach = pg.build(sig.cov, variant, p_share, rq_t, DICT_SEED + layer + 1000 * ds,
                                      max_iter=DICT_MAX_ITER, tol=DICT_TOL, rq_tol=DICT_RQ_TOL, dev=dev, log=log,
                                      max_seconds=DICT_SECONDS)
    assert info["cov_rel_frob_err_float64"] < 1e-6, f"{t}: A^T A differs from Sigma ({info['cov_rel_frob_err_float64']:.2e})"
    real_lz = pg.lorenz(pi)
    W = (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)
    cs = rc._cloud_stats(W, np.ones(len(W), bool), 0, ds, N_FIX, sig.slopes)
    from train_sae import sparse_dictionary
    A0 = sparse_dictionary(sig.cov, M, ds).astype(np.float64)  # the equal-power dictionary V0 trains on
    cs0 = rc._cloud_stats((A0 / np.linalg.norm(A0, axis=1, keepdims=True)).astype(np.float32), np.ones(M, bool), 0, ds,
                          N_FIX, sig.slopes)
    del A0
    info |= {"v0_atom_cloud_gamma_fixed": cs0["gamma_fixed"], "tag": t, "layer": layer,
             "dict_seed": DICT_SEED + layer + 1000 * ds,
             "real_power_file": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
             "real_power_file_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "real_n_live": int(len(pi)),
             "real_power": real_lz, "real_rq_log10_quantiles": {str(q): float(np.quantile(np.log10(rq), q))
                                                                  for q in (.01, .1, .5, .9, .99)},
             "real_spearman_power_rq": pg.spearman(pi, rq),
             "atom_cloud_gamma_fixed": cs["gamma_fixed"], "atom_cloud_gamma_full": cs["gamma_full"],
             "generator_sha256": gen_sha(), "job_code_sha256": job_sha(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "pmatch").mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, A=A.astype(np.float32))
    os.replace(tmp, path)
    info["dict_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(OUT / "pmatch" / f"{t}_atoms.npz", power=p_ach, rq=rq_ach, target_share=p_share, target_rq=rq_t)
    tmp = rep.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=1, default=float))
    os.replace(tmp, rep)
    g = " / ".join("%.3f" % cs["gamma_fixed"][str(w)] for w in rc.WINDOWS)
    log(f"{t}: atom-cloud gamma {g}, top-10 power share "
        f"{info['power']['share_top10']:.3f} (real {real_lz['share_top10']:.3f}), top-1% {info['power']['share_top1pct']:.3f} "
        f"(real {real_lz['share_top1pct']:.3f}), moved {info['power_share_moved_by_projection']:.3f}")
    return path


def run_name(layer, M, L, ds, rho, t):  # train_sae.py's naming
    return (f"{MODEL}__{hook(layer)}__sparse__w{WIDTH}__k{K}__s0__ds{ds}__M{M}__L{L:g}__{VALUES}"
            f"__rho{rho:.4f}__D{t}")


def train(layer, M, L, ds, rho, dpath, t):
    rd = ROOT / "cache" / GROUP / run_name(layer, M, L, ds, rho, t)
    if not (rd / "train.json").exists():
        log(f"train {rd.name} ({minutes():.0f} min into the job)")
        py(ROOT / "train_sae.py", "--model", MODEL, "--hook", hook(layer), "--cond", "sparse", "--width", WIDTH,
           "--k", K, "--tokens", TOKENS - EVAL, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", 0,
           "--out", f"cache/{GROUP}", "--sparse_M", M, "--sparse_L", L, "--data_seed", ds,
           "--sparse_values", VALUES, "--sparse_rho", rho, "--save_params",
           "--sparse_dict", dpath.relative_to(ROOT), "--sparse_dict_tag", t, *EXTRA)
    return json.load(open(rd / "train.json"))["fvu_eval"], rd


def next_rho(pts, T):
    """Secant step through the bracketing pair of tried (rho, F) points, else the two nearest to T."""
    pts = sorted(pts.items())
    below = [p for p in pts if p[1] < T]
    above = [p for p in pts if p[1] >= T]
    if below and above:
        a, b = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
    else:
        a, b = sorted(pts, key=lambda p: abs(p[1] - T))[:2]
    if abs(b[1] - a[1]) < 1e-6:
        return None
    return round(min(max(a[0] + (T - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), 0.0), RHO_MAX), 2)


def calibrate(trainf, T, rho1, slope, can_start):
    """The rule in the docstring. trainf(rho) -> FVU; can_start() -> False once the time budget is spent.
    Returns (rho or None, {rho: FVU}, matched, complete)."""
    tried, rho = {}, rho1
    while len(tried) < MAX_RUNS and rho is not None and rho not in tried:
        if not can_start():
            log(f"budget: not starting rho {rho:.2f} ({minutes():.0f} min)")
            if not tried:
                return None, tried, False, False
            best = min(tried, key=lambda r: abs(tried[r] - T))
            return best, tried, abs(tried[best] - T) <= TOL, False
        tried[rho] = trainf(rho)
        if abs(tried[rho] - T) <= TOL:
            return rho, tried, True, True
        if len(tried) == 1:
            nxt = round(min(max(rho + (T - tried[rho]) / slope, 0.0), RHO_MAX), 2)
            rho = nxt if nxt != rho else round(min(max(rho + (0.01 if T > tried[rho] else -0.01), 0.0), RHO_MAX), 2)
        else:
            rho = next_rho(tried, T)
    best = min(tried, key=lambda r: abs(tried[r] - T))
    return best, tried, abs(tried[best] - T) <= TOL, True


def save_cal(key, entry):
    f = OUT / "pmatch_calibration.json"
    cal = json.load(open(f)) if f.exists() else {}
    cal[key] = entry
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=1))
    os.replace(tmp, f)


def null_batches(sig, M, L, ds, rho, dpath, scale):
    from r2_job_robust import device
    from train_sae import sparse_batches, sparse_setup
    dev = device()
    sp = sparse_setup(sig.cov, sig.mean, M, L, ds, dev, VALUES, rho, dict_path=dpath)
    return sparse_batches(sp, N_POWER, 8192, POWER_SEED + 1000 * ds, scale, dev)


def power_check(pw, sig, real_pi):
    """Manipulation check: the null SAE's latent power distribution beside the real SAE's (shares, ESS, rq)."""
    W = pw["W_dec"].astype(np.float64)
    norm = np.linalg.norm(W, axis=1)
    n = pw["n"]
    pi = pw["s2"] / n * norm ** 2
    live = pw["fires"] > 0
    D = W / norm[:, None]
    rq = ((D @ sig.cov) * D).sum(1) / (np.trace(sig.cov) / D.shape[1])
    lz, rl = pg.lorenz(pi[live]), pg.lorenz(real_pi)
    stats = {"null": lz, "real": rl, "power_fvu": pw["fvu"], "power_l0": pw["l0"], "n_live": int(live.sum()),
             "rq_median_log10": float(np.median(np.log10(rq[live]))), "spearman_pi_rq": pg.spearman(pi[live], rq[live]),
             "top1pct_share_diff": lz["share_top1pct"] - rl["share_top1pct"], "ess_ratio": lz["ess_frac"] / rl["ess_frac"]}
    return stats, {"pi": pi, "rayleigh": rq, "live": live, "freq": pw["fires"] / n, "n_tokens": n}


def score_one(rd, sig, meta, arrays=None):
    if arrays is not None:
        (OUT / "power_pmatch").mkdir(parents=True, exist_ok=True)
        tmp = OUT / "power_pmatch" / f"{rd.name}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, OUT / "power_pmatch" / f"{rd.name}.npz")
    return rc.score(rd, sig, OUT, JOB, meta, n_fix=N_FIX)


def do_setting(scorer, builder, sigs, setting, can_start):
    variant, layer, M, L, ds = setting
    key, t, sig = setting_key(*setting), tag(variant, layer, M, ds), sigs[layer]
    if not can_start():
        log(f"{key}: budget spent ({minutes():.0f} min); not started")
        return
    dpath = builder.wait(t)
    if dpath is None:
        return
    rep = json.loads((OUT / "pmatch" / f"{t}.json").read_text())
    runs = {}

    def trainf(rho):
        runs[rho] = train(layer, M, L, ds, rho, dpath, t)
        return runs[rho][0]
    rho1, slope = V0_CAL[(layer, M, L)]
    rho, tried, matched, complete = calibrate(trainf, TARGET[layer], rho1, slope, can_start)
    save_cal(key, {"variant": variant, "layer": layer, "M": M, "L": L, "ds": ds, "rho": rho, "target": TARGET[layer],
                   "matched": matched, "complete": complete, "v0_rho": rho1, "v0_slope": slope,
                   "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}})
    if rho is None:
        return
    log(f"{key}: rho {rho:.2f}, FVU {tried[rho]:.3f}, target {TARGET[layer]:.3f}, {'matched' if matched else 'UNMATCHED'}")
    base = {"layer": layer, "cond": "sparse_pmatch", "variant": variant, "variant_label": pg.VARIANTS[variant],
            "dataset": f"{SIGMA_MODEL}_L{layer}", "whitened": False, "model": "gemma-2-2b", "setting_key": key,
            "setting": {"M": M, "L": L, "values": VALUES, "dict_tag": t, "ds": ds}, "fvu_target": TARGET[layer],
            "matched": matched, "calibration_complete": complete, "dictionary": {
                k: rep.get(k) for k in ("power", "power_target", "rq_log10_quantiles", "spearman_power_rq",
                                        "power_share_moved_by_projection", "majorization_gap_before_projection",
                                        "cov_rel_frob_err_float64", "cov_rel_frob_err_float32", "atom_cloud_gamma_fixed",
                                        "real_power", "real_power_file_sha256", "dict_file_sha256", "iterations",
                                        "converged", "final_power_max_abs_log_err", "dict_seed",
                                        "v0_atom_cloud_gamma_fixed")},
            "job_code_sha256": job_sha(), "generator_sha256": gen_sha()}
    for r in sorted(tried):
        rd = runs[r][1]
        if done(rd.name):
            continue
        meta, arrays = {**base, "step": "PM" if r == rho else "PMCAL", "rho": r}, None
        if r == rho:
            from r2_job_robust import power_pass
            if not (rd / "W_enc.npy").exists():
                log(f"{rd.name}: no W_enc.npy, power check skipped")
            else:
                scale = json.load(open(rd / "train.json"))["input_scale"]
                pw = power_pass(rd, null_batches(sig, M, L, ds, r, dpath, scale))
                meta["power_check"], arrays = power_check(pw, sig, real_power(layer)[0])
                pc = meta["power_check"]
                log(f"{rd.name}: null SAE top-1% power share {pc['null']['share_top1pct']:.3f} (real "
                    f"{pc['real']['share_top1pct']:.3f}), ESS fraction {pc['null']['ess_frac']:.4f} (real "
                    f"{pc['real']['ess_frac']:.4f})")
        scorer.submit(score_one, rd, sig, meta, arrays)


def main():
    global BUDGET_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget_min", type=float, default=None, help="no SAE starts after this many minutes (default 90)")
    ap.add_argument("--settings", type=int, nargs="*", default=None, help="1-based setting numbers (default: all, in order)")
    a = ap.parse_args()
    if a.smoke:
        smoke_config()
    if a.budget_min is not None:
        BUDGET_MIN = a.budget_min
    plan = [PLAN[i - 1] for i in a.settings] if a.settings else PLAN
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, job sha256 {job_sha()[:12]}, generator "
        f"sha256 {gen_sha()[:12]}, out {OUT}, {rc.disk_free_gb():.0f} GB free, budget {BUDGET_MIN:.0f} min")
    layers = sorted({s[1] for s in plan})
    for L in layers:
        if real_power(L) is None:
            log(f"WARNING no real power file for layer {L}: {real_power_path(L)}")
    sigs = {L: sigma(L) for L in layers}  # before the builder thread, which reads them
    builder = DictBuilder(plan, sigs)
    scorer = rc.Scorer()
    for s in plan:
        log(f"setting {setting_key(*s)}")
        do_setting(scorer, builder, sigs, s, lambda: minutes() <= BUDGET_MIN)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:  # rerun locally after the pull; never fail the job on it
        log(f"r2_summary failed ({e}); rerun locally")
    log(f"JOB DONE pmatch ({minutes():.0f} min)")


if __name__ == "__main__":
    main()
