"""Check, as far as local SAEs allow:
do Gaussian / scrambled-null TopK decoders differ from real ones by collapsing onto near-duplicate, high-variance
directions, and are they closer to a Sigma^gamma cloud than real decoders?

SAEs (all local TopK SAEs with real/null pairs or nulls on Gemma-2-2B / Pythia-70m):
  cache/saes: Pythia-70m blocks.3 resid_post, real / gaussian / scrambled x width 1024, 4096, 16384 x 3 seeds, k = 32;
  cache/saes_gemma_null: Gemma-2-2B k = 40, width 4096: real and gaussian at layer 12 (800k training tokens), and
    gaussian only at layers 0, 3, 7, 13, 17, 21, 24 (3M tokens; no local real partner).
Training covariance Sigma: stats.npz of the activation cache each SAE was trained from (the Gaussian null samples
N(mu, Sigma) from exactly this file). The scrambled null's data covariance is diag(Sigma).

Analysis choices (fixed before looking at results):
  Rows: live latents (fired on the eval split), as for all our TopK spectra; rows rescaled to unit norm for cosines.
  (1) FVU: fvu_eval from train.json, side by side with everything else.
  (2) Near-duplicates: each live row's maximum (signed) cosine to any other live row; quantiles 0.1/0.5/0.9/0.99 and
      the fraction above 0.3, 0.5, 0.7, 0.9.
  (3) Merge test (as test2_neardup.py): random-order greedy dedup at tau = 0.5, 0.7, 0.9 (seed 1); rows removed and
      change in slope (kept - live) in each window that fits.
  (4) Matrix test: geometry.basis_gamma(cloud_cov(live rows), Sigma) -> diag share, top-10/50 subspace overlap, and
      gamma, R^2 of log diag(U^T C U) on log lambda over each window; scrambled also against diag(Sigma).
      Benchmark: the same decoder covariance in a random orthonormal basis (one draw, seed 0), diag share only.
  Windows: Pythia d = 512 -> (3,50), (10,150), (30,450); Gemma d = 2304 -> also (100,1000).
  Aggregation: mean and sd over seeds per (model, layer, cond, width).
  Readings stated in advance: the "collapse" account of the null bulk predicts (i) null decoders have more
  near-duplicates than real (higher nn-cosine quantiles, larger fraction above 0.5) and (ii) merging moves the null
  slope more than the real one; "the null is a Sigma^gamma cloud" predicts a null diag share near 1 and above real.
Writes results/review/regime.json.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import _eigvals, basis_gamma, cloud_cov, slope_from_eigenvalues  # noqa: E402

MAIN = Path("/Users/ayurpulle/Desktop/SAE")  # read-only main cache (a worktree cache/ may hold unrelated runs)
CACHE = MAIN / "cache"
OUT = ROOT / "results" / "review" / "regime.json"
TAUS = [0.5, 0.7, 0.9]
NN_T = [0.3, 0.5, 0.7, 0.9]
BLK = 4096


def windows(d):
    return [(3, 50), (10, 150), (30, 450)] + ([(100, 1000)] if d >= 1000 else [])


def nn_and_pairs(U, tmin):
    n = len(U)
    nn = np.full(n, -1.0, np.float32)
    pairs = []
    for s in range(0, n, BLK):
        S = U[s:s + BLK] @ U.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -np.inf
        nn[s:s + BLK] = S.max(1)
        qi, bj = np.nonzero(S > tmin)
        pairs.append(np.stack([qi + s, bj, S[qi, bj]], 1))
    return nn, np.concatenate(pairs)


def greedy_keep(n, pairs, tau, rng):
    """Copied from test2_neardup.py: keep latent i unless an already-kept latent has cosine > tau with it."""
    p = pairs[pairs[:, 2] > tau]
    nbrs = [[] for _ in range(n)]
    for i, j, _ in p:
        nbrs[int(i)].append(int(j))
    keep = np.zeros(n, dtype=bool)
    for i in rng.permutation(n):
        if not any(keep[j] for j in nbrs[i]):
            keep[i] = True
    return keep


def wslopes(X, ws):
    e = _eigvals(X, center=True)[0]
    return {str(w): (slope_from_eigenvalues(e, w)["slope"] if w[1] <= min(len(X) - 1, X.shape[1]) else None) for w in ws}


def score(d, Sigma):
    info = json.load(open(d / "train.json"))
    c = info["config"]
    live = np.load(d / "fires.npy") > 0
    W = np.load(d / "W_dec.npy").astype(np.float32)[live]
    dm = W.shape[1]
    ws = windows(dm)
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    nn, pairs = nn_and_pairs(U, min(TAUS))
    base = wslopes(W.astype(np.float64), ws)
    merge = {}
    for tau in TAUS:
        keep = greedy_keep(len(W), pairs, tau, np.random.default_rng(1))
        s = wslopes(W[keep].astype(np.float64), ws)
        merge[str(tau)] = {"n_kept": int(keep.sum()), "frac_removed": float(1 - keep.mean()),
                           "dslope": {w: (s[w] - base[w] if s[w] is not None and base[w] is not None else None) for w in s}}
    C = cloud_cov(W)
    rr = tuple(ws)
    mt = {"vs_sigma": basis_gamma(C, Sigma, rank_ranges=rr)}
    if c["cond"] == "scrambled":
        mt["vs_diag_sigma"] = basis_gamma(C, np.diag(np.diag(Sigma)), rank_ranges=rr)
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((dm, dm)))
    Dr = Q.T @ C @ Q
    mt["random_basis_diag_share"] = float((np.diag(Dr) ** 2).sum() / (Dr ** 2).sum())
    return {"model": c["model"], "hook": c["hook"], "cond": c["cond"], "width": c["width"], "k": c["k"], "seed": c["seed"],
            "train_tokens": c["tokens"], "fvu_eval": info["fvu_eval"], "n_live": int(live.sum()),
            "slopes_live": base,
            "nn_cos_quantiles": {str(q): float(np.quantile(nn, q)) for q in (0.1, 0.5, 0.9, 0.99)},
            "frac_nn_above": {str(t): float((nn > t).mean()) for t in NN_T},
            "merge": merge, "matrix_test": mt}


def flat(r):
    """Scalar summary fields of one SAE record, for aggregation."""
    f = {"fvu_eval": r["fvu_eval"], "n_live": r["n_live"]}
    f.update({f"nn_q{q}": v for q, v in r["nn_cos_quantiles"].items()})
    f.update({f"nn_frac>{t}": v for t, v in r["frac_nn_above"].items()})
    for w, v in r["slopes_live"].items():
        f[f"slope{w}"] = v
    for tau, m in r["merge"].items():
        f[f"merge{tau}_frac_removed"] = m["frac_removed"]
        for w, v in m["dslope"].items():
            f[f"merge{tau}_dslope{w}"] = v
    for ref, bg in r["matrix_test"].items():
        if ref == "random_basis_diag_share":
            f[ref] = bg
            continue
        f[f"{ref}_diag_share"] = bg["diag_share"]
        f[f"{ref}_overlap10"] = bg["subspace_overlap"][10]
        f[f"{ref}_overlap50"] = bg["subspace_overlap"][50]
        for x in bg["fits"]:
            f[f"{ref}_gamma{tuple(x['rank_range'])}"] = x["gamma"]
            f[f"{ref}_r2{tuple(x['rank_range'])}"] = x["r2"]
    return f


def main():
    t0 = time.time()
    out = {"notes": __doc__.split("Writes")[0].strip(), "saes": {}}
    sig_cache = {}
    dirs = sorted((CACHE / "saes").iterdir()) + sorted((CACHE / "saes_gemma_null").iterdir())
    for d in dirs:
        if not (d / "fires.npy").exists():
            continue
        c = json.load(open(d / "train.json"))["config"]
        key = (c["model"], c["hook"])
        if key not in sig_cache:
            sig_cache[key] = np.load(CACHE / "acts" / c["model"] / c["hook"] / "stats.npz")["cov"].astype(np.float64)
        r = score(d, sig_cache[key])
        out["saes"][d.name] = r
        mt = r["matrix_test"]["vs_sigma"]
        print(f"{d.name:<70} fvu {r['fvu_eval']:.3f} live {r['n_live']:>5} nn q50 {r['nn_cos_quantiles']['0.5']:.3f}"
              f" q99 {r['nn_cos_quantiles']['0.99']:.3f} >0.5 {r['frac_nn_above']['0.5']:.4f}"
              f" merge0.5 rm {r['merge']['0.5']['frac_removed']:.4f} d(10,150) {r['merge']['0.5']['dslope']['(10, 150)']:+.4f}"
              f" | diag {mt['diag_share']:.3f} (rand {r['matrix_test']['random_basis_diag_share']:.3f})"
              f" gamma(10,150) {mt['fits'][1]['gamma']:.3f} R2 {mt['fits'][1]['r2']:.3f}  {time.time() - t0:.0f}s", flush=True)

    groups = defaultdict(list)
    for r in out["saes"].values():
        groups[f"{r['model']}|{r['hook']}|{r['cond']}|w{r['width']}|k{r['k']}|tok{r['train_tokens']}"].append(flat(r))
    agg = {}
    for g, rows in sorted(groups.items()):
        agg[g] = {"n": len(rows)}
        for k in rows[0]:
            v = [x[k] for x in rows if x.get(k) is not None]
            if v:
                agg[g][k] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    out["aggregate"] = agg
    print("\nAggregate (mean over seeds)")
    cols = ["fvu_eval", "n_live", "nn_q0.5", "nn_q0.99", "nn_frac>0.5", "merge0.5_frac_removed", "merge0.5_dslope(10, 150)",
            "merge0.5_dslope(30, 450)", "slope(10, 150)", "slope(30, 450)", "vs_sigma_diag_share", "random_basis_diag_share",
            "vs_sigma_overlap10", "vs_sigma_gamma(10, 150)", "vs_sigma_r2(10, 150)", "vs_sigma_gamma(30, 450)",
            "vs_diag_sigma_diag_share", "vs_diag_sigma_gamma(10, 150)"]
    for g, a in agg.items():
        print(f"  {g}  n={a['n']}")
        print("     " + "  ".join(f"{c}={a[c][0]:.4g}" for c in cols if c in a))
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
