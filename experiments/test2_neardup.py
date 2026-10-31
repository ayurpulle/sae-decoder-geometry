"""Test 2 (H2: feature splitting sets the slope): near-duplicate decoder latents vs L0, layer-12 Gemma Scope.

H2 predicts (a) the fraction of latents with a near-parallel neighbour rises as L0 falls, and (b) merging
near-duplicates (greedy: keep a latent only if no kept latent has cosine > tau with it) removes most of the
L0 dependence of the eigen slope.

Widths <= 131k: all latents are queries; thresholded pairs are extracted for the merge test (tau in TAUS).
262k-1m: nearest-neighbour statistics only, from N_QUERY random query latents against all latents.
CPU only (Phase 2 training is using the GPU); cosine = dot product since rows are unit norm.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402
from phase1_width_ladder import LADDER, LAYER, WINDOWS, fetch_wdec, git_hash  # noqa: E402

TAUS = [0.5, 0.7, 0.9]
NN_THRESH = [0.3, 0.5, 0.7, 0.9]
FULL_WIDTHS = {"16k", "32k", "65k", "131k"}
N_QUERY = 16384
BLOCK = 16384
Q_CHUNK = 4096
OUT = ROOT / "results" / "test2"


def scan(W, q_idx, taus):
    """One streaming pass over W's row blocks (memmap-friendly); queries (<= 1.2 GB) held in RAM and
    processed in chunks. Returns each query's nearest-neighbour cosine (self excluded) and, if taus,
    the (i, j, cos) pairs with cosine > min(taus), i != j."""
    Q = np.ascontiguousarray(W[q_idx], dtype=np.float32)
    nn = np.full(len(q_idx), -1.0, dtype=np.float32)
    pairs = []
    tmin = min(taus) if taus else None
    for s in range(0, W.shape[0], BLOCK):
        B = np.asarray(W[s:s + BLOCK], dtype=np.float32)
        for a in range(0, len(q_idx), Q_CHUNK):
            qa = q_idx[a:a + Q_CHUNK]
            S = Q[a:a + Q_CHUNK] @ B.T
            hit = (qa >= s) & (qa < s + len(B))
            S[np.nonzero(hit)[0], qa[hit] - s] = -np.inf
            np.maximum(nn[a:a + Q_CHUNK], S.max(1), out=nn[a:a + Q_CHUNK])
            if tmin is not None:
                qi, bj = np.nonzero(S > tmin)
                pairs.append(np.stack([qa[qi], bj + s, S[qi, bj]], 1))
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 3))
    return nn, pairs


def greedy_keep(n, pairs, tau, rng):
    """Random-order greedy dedup: keep latent i unless an already-kept latent has cosine > tau with it."""
    p = pairs[pairs[:, 2] > tau]
    nbrs = [[] for _ in range(n)]
    for i, j, _ in p:
        nbrs[int(i)].append(int(j))
    keep = np.zeros(n, dtype=bool)
    for i in rng.permutation(n):
        if not any(keep[j] for j in nbrs[i]):
            keep[i] = True
    return keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "neardup_L12.json"
    done = json.load(open(out_json))["results"] if out_json.exists() else {}
    for width, l0s in LADDER.items():
        for l0 in l0s:
            sae_id = f"layer_{LAYER}/width_{width}/average_l0_{l0}"
            if sae_id in done:
                continue
            t0 = time.time()
            W = np.load(fetch_wdec(sae_id), mmap_mode="r")
            n = W.shape[0]
            rng = np.random.default_rng(0)
            full = width in FULL_WIDTHS
            q_idx = np.arange(n) if full else np.sort(rng.choice(n, N_QUERY, replace=False))
            nn, pairs = scan(W, q_idx, TAUS if full else None)
            rec = {
                "width": width, "l0": l0, "n_latents": n, "n_query": len(q_idx), "full": full,
                "nn_cos_quantiles": {str(q): float(np.quantile(nn, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]},
                "frac_nn_above": {str(t): float((nn > t).mean()) for t in NN_THRESH},
            }
            if full:
                base = eigen_slope(W)
                rec["slope"] = base["slope"]
                rec["dedup"] = {}
                for tau in TAUS:
                    keep = greedy_keep(n, pairs, tau, np.random.default_rng(1))
                    r = eigen_slope(W, mask=keep)
                    rec["dedup"][str(tau)] = {
                        "n_kept": int(keep.sum()), "slope": r["slope"], "r2": r["r2"],
                        "windows": [slope_from_eigenvalues(r["eigenvalues"], w) for w in WINDOWS],
                    }
            done[sae_id] = rec
            fr = " ".join(f">{t}:{v:.3f}" for t, v in rec["frac_nn_above"].items())
            dd = " ".join(f"tau{t}:{v['slope']:+.3f}(n={v['n_kept']})" for t, v in rec.get("dedup", {}).items())
            print(f"{sae_id}: median nn {rec['nn_cos_quantiles']['0.5']:.3f} frac[{fr}] "
                  f"slope {rec.get('slope', float('nan')):+.3f} dedup[{dd}] {time.time() - t0:.0f}s", flush=True)
            config = {"layer": LAYER, "taus": TAUS, "nn_thresh": NN_THRESH, "n_query_large": N_QUERY,
                      "full_widths": sorted(FULL_WIDTHS), "eig_windows": [list(w) for w in WINDOWS],
                      "dedup": "random-order greedy, seed 1", "git_hash": git_hash()}
            json.dump({"config": config, "results": done}, open(out_json, "w"), indent=1)
            del W


if __name__ == "__main__":
    main()
