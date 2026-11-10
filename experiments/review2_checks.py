"""Local checks requested by the second external review of the write-up (2026-11-10).

1. matrix: finite-sample benchmark for the matrix test. For each Gemma Scope L12 16k SAE, draw an exact
   Sigma^gamma cloud (n = 16,384 unit rows, gamma = that SAE's basis gamma in ranks 101-1000) and a null cloud with
   the decoder's own eigenvalues in a random basis, then run geometry.basis_gamma on both, next to the real decoder.
2. sigma: Sigma's eigen-slopes per window from each 1M-token activation cache still on disk (layer 12 only; the
   other layers kept just their 200k-token covariance): first 200k tokens, the two halves split by alternating
   documents (1,023 tokens each), and all 1M; plus the 200k Sigma used in the paper.
3. fixedn: the layer-12 gamma(L0) law for the 28 Gemma Scope SAEs, rescored on 16,384 random decoder rows
   (mean of 4 draws) so every width has the same number of points.
Writes results/review/review2_checks.json.  Usage: python experiments/review2_checks.py [matrix|matrix1m|sigma|fixedn ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import _eigvals, basis_gamma, cloud_cov, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
RANGES = ((1, 150), (10, 150), (100, 1000))
OUT = ROOT / "results" / "review" / "review2_checks.json"
WDEC = ROOT / "cache" / "gemmascope_wdec"
SIGMA_FILE = ROOT / "results" / "test1" / "sigma_gemma-2-2b.npz"
SIGMA_1M = ROOT / "results" / "test1" / "sigma_gemma-2-2b_1m.npz"  # matrix1m: the same benchmark with 1M-token Sigma
DOC = 1023  # tokens per document (context 1024, BOS dropped)
N_FIX, N_DRAWS = 16384, 4


def slopes(eig):
    return {str(w): slope_from_eigenvalues(eig, w)["slope"] for w in WINDOWS}


def unit_cloud(rng, U, var, n):
    """n unit rows from N(0, U diag(var) U^T)."""
    X = (rng.standard_normal((n, len(var))) * np.sqrt(var)) @ U.T
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def matrix_check(sigma_file=SIGMA_FILE):
    Sig = np.load(sigma_file, mmap_mode="r")["cov"][12].astype(np.float64)
    lam, U = np.linalg.eigh(Sig)
    lam, U = lam[::-1], U[:, ::-1]
    rng = np.random.default_rng(0)
    rows = []
    for sae in sorted((WDEC / "layer_12" / "width_16k").iterdir(), key=lambda p: int(p.name.split("_")[-1])):
        W = np.load(sae / "W_dec.npy")
        C = cloud_cov(W)
        real = basis_gamma(C, Sig, rank_ranges=RANGES)
        g_bulk = real["fits"][-1]["gamma"]
        exact = basis_gamma(cloud_cov(unit_cloud(rng, U, lam ** g_bulk, len(W))), Sig, rank_ranges=RANGES)
        ev, _ = np.linalg.eigh(C)
        Q, _ = np.linalg.qr(rng.standard_normal((len(lam), len(lam))))
        rand = basis_gamma(cloud_cov(unit_cloud(rng, Q, np.clip(ev[::-1], 0, None), len(W))), Sig, rank_ranges=RANGES)
        rows.append({"sae": f"layer_12/width_16k/{sae.name}", "gamma_used": g_bulk,
                     "real": real, "exact_sigma_gamma": exact, "random_basis": rand})
        f = lambda r: " ".join(f"{x['gamma']:+.2f}(R2 {x['r2']:.2f})" for x in r["fits"])  # noqa: E731
        print(f"{sae.name:<16} gamma {g_bulk:.2f}\n  real   diag {real['diag_share']:.2f} ov10 {real['subspace_overlap'][10]:.2f}  {f(real)}"
              f"\n  exact  diag {exact['diag_share']:.2f} ov10 {exact['subspace_overlap'][10]:.2f}  {f(exact)}"
              f"\n  random diag {rand['diag_share']:.2f} ov10 {rand['subspace_overlap'][10]:.2f}  {f(rand)}", flush=True)
    return rows


def sigma_check():
    paper = np.load(SIGMA_FILE, mmap_mode="r")["cov"]
    out = {}
    for d in sorted((ROOT / "cache" / "acts" / "gemma-2-2b").iterdir(), key=lambda p: int(p.name.split(".")[1])):
        if not (d / "acts.npy").exists():
            continue
        L = int(d.name.split(".")[1])
        X = np.load(d / "acts.npy", mmap_mode="r")
        n, dim = X.shape
        parts = {k: [np.zeros(dim), np.zeros((dim, dim)), 0] for k in ("first200k", "even_docs", "odd_docs")}
        t0 = time.time()
        for i in range(0, n, DOC * 20):
            c = np.asarray(X[i:i + DOC * 20], dtype=np.float32)
            docs = (i + np.arange(len(c))) // DOC
            for key, m in (("even_docs", docs % 2 == 0), ("odd_docs", docs % 2 == 1),
                           ("first200k", (i + np.arange(len(c))) < 200_000)):
                if m.any():
                    s = c[m]
                    parts[key][0] += s.sum(0, dtype=np.float64)
                    parts[key][1] += (s.T @ s).astype(np.float64)
                    parts[key][2] += len(s)
        res = {}
        for key, (s, G, k) in parts.items():
            res[key] = slopes(np.sort(np.linalg.eigvalsh((G - np.outer(s, s) / k) / (k - 1)))[::-1])
        s = parts["even_docs"][0] + parts["odd_docs"][0]
        G = parts["even_docs"][1] + parts["odd_docs"][1]
        k = parts["even_docs"][2] + parts["odd_docs"][2]
        res["all"] = slopes(np.sort(np.linalg.eigvalsh((G - np.outer(s, s) / k) / (k - 1)))[::-1])
        res["paper_200k"] = slopes(np.sort(np.linalg.eigvalsh(np.asarray(paper[L], dtype=np.float64)))[::-1])
        res["n_tokens"] = int(n)
        out[str(L)] = res
        print(f"layer {L:>2} ({time.time() - t0:.0f}s)  " + "  ".join(
            f"{w}: " + "/".join(f"{res[key][w]:.3f}" for key in ("paper_200k", "first200k", "even_docs", "odd_docs", "all"))
            for w in map(str, WINDOWS)), flush=True)
    return out


def fixedn_check():
    prev = json.load(open(ROOT / "results" / "test1" / "gamma_l0_L12.json"))
    sig = prev["sigma12_slopes"]
    rng = np.random.default_rng(0)
    rows = []
    for r in prev["rows"]:
        W = np.load(WDEC / r["sae_id"] / "W_dec.npy", mmap_mode="r")
        draws = []
        for _ in range(N_DRAWS if len(W) > N_FIX else 1):
            idx = np.sort(rng.choice(len(W), N_FIX, replace=False)) if len(W) > N_FIX else np.arange(len(W))
            draws.append(slopes(_eigvals(np.asarray(W[idx], dtype=np.float32), center=True)[0]))
        g = {w: float(np.mean([d[w] for d in draws])) / sig[w] for w in sig}
        rows.append({"sae_id": r["sae_id"], "n_latents": r["n_latents"], "l0": r["l0"], "gamma_full": r["gamma"],
                     "gamma_fixed": g})
        print(f"{r['sae_id']:<36} L0 {r['l0']:>6.1f}  ranks 11-150: full {r['gamma']['(10, 150)']:.3f} fixed {g['(10, 150)']:.3f}",
              flush=True)
    fits = {}
    l0 = np.log2([x["l0"] for x in rows])
    for w in sig:
        for m in ("gamma_full", "gamma_fixed"):
            lr = linregress(l0, [x[m][w] for x in rows])
            fits[f"{m}|{w}"] = {"a": lr.intercept, "b_per_doubling": lr.slope, "r2": lr.rvalue ** 2}
        print(f"{w}: full {fits[f'gamma_full|{w}']['a']:.3f} {fits[f'gamma_full|{w}']['b_per_doubling']:+.3f} "
              f"(R2 {fits[f'gamma_full|{w}']['r2']:.3f})   fixed-n {fits[f'gamma_fixed|{w}']['a']:.3f} "
              f"{fits[f'gamma_fixed|{w}']['b_per_doubling']:+.3f} (R2 {fits[f'gamma_fixed|{w}']['r2']:.3f})", flush=True)
    return {"rows": rows, "fits": fits, "n_fix": N_FIX, "n_draws": N_DRAWS}


def main():
    which = sys.argv[1:] or ["matrix", "sigma", "fixedn"]
    res = json.load(open(OUT)) if OUT.exists() else {}
    for w in which:
        print(f"== {w}", flush=True)
        res[w] = {"matrix": matrix_check, "matrix1m": lambda: matrix_check(SIGMA_1M), "sigma": sigma_check, "fixedn": fixedn_check}[w]()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
