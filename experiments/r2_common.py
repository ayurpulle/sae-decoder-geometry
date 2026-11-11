"""Shared pieces of the round-2 GPU jobs (r2_job_real.py, r2_job_resynth.py): on-pod scoring of every SAE, per-dataset
Sigma files, and atomic per-run rows. Scoring follows control_analyze.py (fixed-n draws seeded k*100+seed) and
gpu_summarize.py (live = fired at least once on the eval split), on all four pre-registered windows.

Per SAE, written to <out>/:
  rows/<job>/<run>.json      the run's row (os.replace, so it is complete or absent); its presence marks the run done
  runs_<job>.jsonl           the same row appended (flush + fsync) as each run finishes; readers skip a torn last line
  dec/<run>.npz              decoder covariance over live latents (geometry.cloud_cov, float32) and the live mask;
                             whitened runs also store the whitened-space covariance (cov_whitened)
Per dataset: sigma/<label>.npz with Sigma's descending eigenvalues, stats.npz's cov (float64), mean and n.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import _eigvals, _gaussian_eigvals, basis_gamma, cloud_cov, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]  # 0-based [lo, hi): ranks 4-50, 11-150, 31-450, 101-1000
PRIMARY = (10, 150)
N_FIX, N_DRAWS = 8192, 4
CODE_FILES = ["train_sae.py", "collect_acts.py", "resynth.py", "geometry.py", "experiments/r2_common.py",
              "experiments/r2_job_real.py", "experiments/r2_job_resynth.py", "experiments/r2_job_models.py"]
_lock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def py(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=ROOT, env=os.environ)


def git_hash():
    f = ROOT / ".git_hash"  # written by gpu/push_r2.sh; the pod has no .git
    if f.exists():
        return f.read_text().strip()
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() or None


def code_sha256():
    """Hash of the code that ran (the git hash alone misses uncommitted edits)."""
    h = hashlib.sha256()
    for f in CODE_FILES:
        p = ROOT / f
        h.update(f.encode() + b"\0" + (p.read_bytes() if p.exists() else b"<missing>") + b"\0")
    return h.hexdigest()


def slopes(eig):
    return {str(w): slope_from_eigenvalues(eig, w)["slope"] for w in WINDOWS}


def ratio(a, b):
    return {w: a[w] / b[w] for w in a}


class Sigma:
    """Training Sigma of one dataset (stats.npz), saved once to <out>/sigma/<label>.npz."""

    def __init__(self, stats_path, label, out):
        st = np.load(stats_path)
        self.cov, self.mean, self.n = st["cov"], st["mean"], int(st["n"])
        self.eig = np.sort(np.linalg.eigvalsh(self.cov))[::-1]
        self.slopes = slopes(self.eig)
        self.label = label
        p = out / "sigma" / f"{label}.npz"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.stem + ".tmp.npz")
            np.savez_compressed(tmp, eigenvalues=self.eig, cov=self.cov, mean=self.mean, n=self.n)
            os.replace(tmp, p)


def _cloud_stats(W, live, k, seed, n_fix, sig_slopes):
    """Full-live and fixed-n spectra and slopes of the rows W[live] (fixed-n rng as in control_analyze.py)."""
    Wl = W[live]
    e_full = _eigvals(Wl, center=True)[0]
    rng = np.random.default_rng(k * 100 + seed)
    short = len(Wl) < n_fix
    draws = [e_full] if short else [_eigvals(Wl[rng.choice(len(Wl), n_fix, replace=False)], center=True)[0]
                                    for _ in range(N_DRAWS)]
    full, fixed = slopes(e_full), [slopes(e) for e in draws]
    gfix = [ratio(f, sig_slopes) for f in fixed]
    return {"n_fix_used": int(len(Wl) if short else n_fix), "n_draws_used": len(draws), "fixed_n_short": short,
            "dec_full": full, "gamma_full": ratio(full, sig_slopes),
            "dec_fixed": {w: float(np.mean([f[w] for f in fixed])) for w in full},
            "dec_fixed_sd": {w: float(np.std([f[w] for f in fixed])) for w in full},
            "gamma_fixed": {w: float(np.mean([g[w] for g in gfix])) for w in full},
            "gamma_fixed_sd": {w: float(np.std([g[w] for g in gfix])) for w in full},
            "eig_full": e_full.tolist(), "eig_fixed_mean": np.mean(draws, 0).tolist()}


def _iso_null(S, n, k, seed, n_fix, sig_slopes):
    """Whitening null: n isotropic Gaussian directions in whitened space mapped by S = (Sigma + eps I)^{1/2} and
    renormalised to unit rows, scored exactly like the mapped decoders (full n and fixed n)."""
    rng = np.random.default_rng(10_000 + k * 100 + seed)
    Z = rng.standard_normal((n, S.shape[0]))
    M = Z @ S
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    r = _cloud_stats(M.astype(np.float32), np.ones(n, bool), k, seed, n_fix, sig_slopes)
    return {key: r[key] for key in ("dec_full", "gamma_full", "dec_fixed", "dec_fixed_sd", "gamma_fixed", "gamma_fixed_sd",
                                    "n_fix_used")}


def score(run_dir, sig, out, job, meta, n_fix=N_FIX, whiten_S=None):
    """Score one trained SAE against its training Sigma, save its covariance and live mask, write its row."""
    t0 = time.time()
    info = json.load(open(run_dir / "train.json"))
    c = info["config"]
    fires = np.load(run_dir / "fires.npy")
    live = fires > 0
    W = np.load(run_dir / "W_dec.npy")
    extra, covs = {}, {}
    if whiten_S is not None:  # decoder rows live in whitened space: score them there, then map back
        Ww = W
        ew = _eigvals(Ww[live], center=True)[0]
        g = _gaussian_eigvals(int(live.sum()), W.shape[1], float(np.sqrt(ew.sum() / W.shape[1])), True, c["seed"])
        extra["whitened_space"] = {"dec_full": slopes(ew), "isotropic_full": slopes(g),
                                   "anisotropy_full": {w: slopes(ew)[w] - slopes(g)[w] for w in slopes(ew)},
                                   "eig_full": ew.tolist()}
        covs["cov_whitened"] = cloud_cov(Ww, mask=live).astype(np.float32)
        W = (Ww.astype(np.float64) @ whiten_S)
        W = (W / np.linalg.norm(W, axis=1, keepdims=True)).astype(np.float32)
        extra["iso_null"] = _iso_null(whiten_S, int(live.sum()), c["k"], c["seed"], n_fix, sig.slopes)
    cs = _cloud_stats(W, live, c["k"], c["seed"], n_fix, sig.slopes)
    C = cloud_cov(W, mask=live)
    bg = basis_gamma(C, sig.cov, rank_ranges=tuple(WINDOWS))
    bg["subspace_overlap"] = {str(m): v for m, v in bg["subspace_overlap"].items()}
    name = run_dir.name
    (out / "dec").mkdir(parents=True, exist_ok=True)
    tmp = out / "dec" / f"{name}.tmp.npz"
    np.savez_compressed(tmp, cov=C.astype(np.float32), live=live, n_live=int(live.sum()), **covs)
    os.replace(tmp, out / "dec" / f"{name}.npz")
    row = {"job": job, "run": name, **meta, "sigma_label": sig.label, "config": c,
           "git_hash": git_hash(), "train_git_hash": c.get("git_hash"), "code_sha256": code_sha256(),
           "n_live": int(live.sum()), "width": c["width"], "k": c["k"], "seed": c["seed"],
           "fvu_eval": info["fvu_eval"], "l0_eval": float(fires.sum() / info["n_eval_tokens"]),
           "n_train_tokens": info["n_train_tokens"], "n_eval_tokens": info["n_eval_tokens"],
           "input_scale": info["input_scale"], "train_seconds": info["seconds"], "device": info.get("device"),
           "whiten": info.get("whiten"), "sparse": sparse_meta(info.get("sparse"), sig),
           "final_train": info["train_log"][-1],
           "sigma_slope": sig.slopes, "sigma_n": sig.n, **cs, "basis_gamma": bg, **extra,
           "score_seconds": time.time() - t0, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_row(out, job, row)
    pw = str(PRIMARY)
    log(f"scored {name}: live {row['n_live']}, fvu {row['fvu_eval']:.3f}, L0 {row['l0_eval']:.1f}, primary window {pw} gamma "
        f"full {cs['gamma_full'][pw]:.3f} fixed-n {cs['gamma_fixed'][pw]:.3f}"
        + (f" (iso null {extra['iso_null']['gamma_fixed'][pw]:.3f})" if whiten_S is not None else "")
        + f" [{row['score_seconds']:.0f}s]")
    return row


def sparse_meta(sp, sig):
    """Generator settings of a --cond sparse run plus its 1M-token check sample's slopes (all windows) and gamma."""
    if not sp:
        return None
    sp = dict(sp)
    chk = dict(sp.pop("check"))
    e = np.asarray(chk.pop("eigenvalues"))
    chk["slopes"] = slopes(e)
    chk["slope_ratio_to_sigma"] = ratio(chk["slopes"], sig.slopes)
    chk["eigenvalues"] = e.tolist()
    return sp | {"check_1M": chk}


def score_dictionary(A, sig, out, job, name, meta, settings, n_fix=N_FIX):
    """Score a ground-truth dictionary (unit-normalised rows of A) like an SAE decoder: all rows are 'live'."""
    W = A / np.linalg.norm(A, axis=1, keepdims=True)
    live = np.ones(len(W), bool)
    cs = _cloud_stats(W, live, 0, settings.get("data_seed", 0), n_fix, sig.slopes)
    C = cloud_cov(W)
    bg = basis_gamma(C, sig.cov, rank_ranges=tuple(WINDOWS))
    bg["subspace_overlap"] = {str(m): v for m, v in bg["subspace_overlap"].items()}
    (out / "dec").mkdir(parents=True, exist_ok=True)
    tmp = out / "dec" / f"{name}.tmp.npz"
    np.savez_compressed(tmp, cov=C.astype(np.float32), live=live, n_live=len(W))
    os.replace(tmp, out / "dec" / f"{name}.npz")
    row = {"job": job, "run": name, **meta, "sigma_label": sig.label, "settings": settings, "git_hash": git_hash(),
           "code_sha256": code_sha256(), "n_live": len(W), "width": len(W), "k": None, "seed": None,
           "sigma_slope": sig.slopes, "sigma_n": sig.n, **cs, "basis_gamma": bg,
           "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_row(out, job, row)
    log(f"scored dictionary {name}: {len(W)} rows, primary window {PRIMARY} gamma full {cs['gamma_full'][str(PRIMARY)]:.3f}")
    return row


def row_path(out, job, name):
    return out / "rows" / job / f"{name}.json"


def write_row(out, job, row):
    p = row_path(out, job, row["run"])
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=float)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(line)
    os.replace(tmp, p)
    with _lock, open(out / f"runs_{job}.jsonl", "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


class Scorer:
    """Scores in one background thread while the next SAE trains on the GPU. At most one score is in flight; a
    failure surfaces at the next submit or at finish(). An unscored run keeps its train.json and W_dec.npy, so a
    restarted job scores it then."""

    def __init__(self):
        self.ex, self.fut = ThreadPoolExecutor(1), None

    def submit(self, fn, *a, **kw):
        self.finish()
        self.fut = self.ex.submit(fn, *a, **kw)

    def finish(self):
        if self.fut is not None:
            f, self.fut = self.fut, None
            f.result()


def disk_free_gb(path=ROOT):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def du_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file() and not f.is_symlink()) / 1e9
