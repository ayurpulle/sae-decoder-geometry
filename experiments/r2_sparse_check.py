"""Local check of train_sae.py --cond sparse: does the sample covariance of n generated tokens match Sigma?
Uses a small-d Sigma (default: Pythia-70m layer 3, d = 512) with the job's generator settings (M = 65536, L = 64).
Reports the relative Frobenius error ||C - Sigma||_F / ||Sigma||_F and the eigen-slope of C vs Sigma in each window
(windows past d are truncated at d), next to a Gaussian N(mu, Sigma) sample of the same n as the sampling-error
reference. Writes the JSON given by --out.
Usage: python experiments/r2_sparse_check.py --stats <stats.npz> --n 2000000 --out <json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402
from train_sae import sparse_batches, sparse_dictionary  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]


def cov_of(gen, d):
    s, G, k = np.zeros(d), np.zeros((d, d)), 0
    for x in gen:
        x = x.cpu().double().numpy()
        s += x.sum(0)
        G += x.T @ x
        k += len(x)
    return (G - np.outer(s, s) / k) / (k - 1), k


def report(C, Sigma):
    e, es = np.sort(np.linalg.eigvalsh(C))[::-1], np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    return {"rel_frob_err": float(np.linalg.norm(C - Sigma) / np.linalg.norm(Sigma)),
            "slopes": {str(w): {"sample": slope_from_eigenvalues(e, w)["slope"], "sigma": slope_from_eigenvalues(es, w)["slope"]}
                       for w in WINDOWS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--M", type=int, default=65536)
    ap.add_argument("--L", type=float, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = np.load(a.stats)
    Sigma, mu = st["cov"], st["mean"]
    d, t0 = len(mu), time.time()
    sp = {"A": torch.tensor(sparse_dictionary(Sigma, a.M, 0)), "mu": torch.tensor(mu, dtype=torch.float32), "p": a.L / a.M}
    Cs, n = cov_of(sparse_batches(sp, a.n, 4096, 8_000_000, 1.0, "cpu"), d)
    t1 = time.time()
    gen = torch.Generator().manual_seed(1)
    Lt = torch.tensor(np.linalg.cholesky(Sigma).T, dtype=torch.float32)
    Cg, _ = cov_of((torch.randn(4096, d, generator=gen) @ Lt + torch.tensor(mu, dtype=torch.float32)
                    for _ in range(a.n // 4096)), d)
    out = {"d": d, "n": n, "M": a.M, "L": a.L, "sparse": report(Cs, Sigma), "gaussian_reference": report(Cg, Sigma),
           "sparse_seconds": t1 - t0}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"d={d} n={n:,} M={a.M} L={a.L}  ({t1 - t0:.0f}s)")
    for k in ("sparse", "gaussian_reference"):
        r = out[k]
        print(f"{k:<19} rel Frobenius error {r['rel_frob_err']:.4f}  slopes (sample vs Sigma): "
              + "  ".join(f"{w}: {v['sample']:+.4f} vs {v['sigma']:+.4f}" for w, v in r["slopes"].items()))


if __name__ == "__main__":
    main()
