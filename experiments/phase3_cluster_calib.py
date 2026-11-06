"""Calibrate clustered planted directions (synth.py --n_clusters) against Gemma before any GPU run.

For d = 2304, the power-law-Sigma generator (alpha 0.5, beta 0, noise 0.05, --var_slope 1.0) and two densities, sweep
the cluster tightness rho (cosine of each feature to its centre) and the within-cluster dimension r. No tokens are
generated: for point features Sigma = sum_i Var_i u_i u_i^T + noise^2 I exactly, with Var_i from feature_scales.
Reports per setting:
  - Sigma's eigen-slopes in Gemma's windows (Gemma L13: -0.98 / -0.95 / -0.75 / -0.70; ranks 4-50 ... 101-1000),
  - the Phase 4 statistics of the feature cloud (unit u_i) at K = M/16 spherical k-means clusters, real minus a
    covariance-matched Gaussian cloud: excess between-cluster share and excess median cosine to centroid
    (Gemma L13 16k decoders at K = 1024: +0.054 and +0.079), and PR16.
Feature-level statistics are a proxy: what matters is the trained decoders, measured on the pod after training.
Writes results/phase3/cluster_calib.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import slope_from_eigenvalues  # noqa: E402
from phase4_clusters import cluster_shape, decompose, matched_cloud, skmeans  # noqa: E402
from synth import cluster_directions, directions, feature_probs, feature_scales  # noqa: E402

D = 2304
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
GEMMA_SIGMA = [-0.98, -0.95, -0.75, -0.70]


def stats(U):
    K = len(U) // 16
    out = {}
    for name, X in (("real", U.astype(np.float32)), ("matched", matched_cloud(U.astype(np.float32)))):
        lab = skmeans(X, K, 0)
        d, _ = decompose(X, lab)
        out[name] = {"share": d["trace_share_between"], **cluster_shape(X, lab)}
    return {"excess_share": out["real"]["share"] - out["matched"]["share"],
            "excess_cos": out["real"]["median_cos_to_centroid"] - out["matched"]["median_cos_to_centroid"],
            "pr16_real": out["real"]["median_pr16"], "pr16_matched": out["matched"]["median_pr16"]}


def main():
    res = []
    for a, b in ((4, 32), (8, 16)):
        M, active = a * D, D // b
        p = feature_probs(M, 0.5, active)
        sc = feature_scales(p, "normal", 1.0)
        var = sc ** 2 * (p - p ** 2 * 2 / np.pi)
        U0, _ = directions(M, D, 0.0, np.random.default_rng(0))
        for rho, r in [(0.0, 0), (0.3, 0), (0.45, 0), (0.6, 0), (0.45, 16), (0.6, 16)]:
            U = U0 if rho == 0 else cluster_directions(M, D, M // 16, rho, r, 0)[0]
            S = (U.T * var) @ U + 0.05 ** 2 * np.eye(D)
            e = np.sort(np.linalg.eigvalsh(S))[::-1]
            sig = [slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS]
            row = {"M": M, "active": active, "rho": rho, "cluster_dim": r, "sigma_slopes": sig, **stats(U)}
            res.append(row)
            print(f"M={M:<6} active={active:<4} rho={rho:<4} r={r:<3} Sigma " + " ".join(f"{x:.2f}" for x in sig)
                  + f"  excess share {row['excess_share']:+.3f} cos {row['excess_cos']:+.3f}"
                  f"  PR16 {row['pr16_real']:.2f}/{row['pr16_matched']:.2f}", flush=True)
    json.dump({"gemma_sigma": GEMMA_SIGMA, "gemma_decoder_excess_L13": {"share": 0.054, "cos": 0.079},
               "rows": res}, open(ROOT / "results" / "phase3" / "cluster_calib.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
