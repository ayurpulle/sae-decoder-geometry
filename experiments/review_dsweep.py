"""Review test: does gamma follow 1 - ln k / ln d when d changes within one model?

Gemma-2-2B L12 activations (1M cached tokens) are projected onto their top d' principal components,
d' in D_PRIMES, and cached as datasets gemma-2-2b-pca<d'>. TopK SAEs at fixed width and k in KS are trained on
each (same token budget, seed 0). gamma is measured two ways: basis-matched (geometry.basis_gamma, ranks
1..50 and 1..d'/8) and the rank-window ratio on ranks 6–50. The law predicts gamma = 1 - ln k / ln d'.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import basis_gamma, cloud_cov, eigen_slope, slope_from_eigenvalues  # noqa: E402

SRC = ROOT / "cache/acts/gemma-2-2b/blocks.12.hook_resid_post"
D_PRIMES = [256, 512, 1024, 2304]
KS = [8, 32]
WIDTH, EPOCHS, EVAL = 8192, 4, 100_000
CHUNK = 100_000


def build(dp):
    out = ROOT / f"cache/acts/gemma-2-2b-pca{dp}/blocks.12.hook_resid_post"
    if (out / "acts.npy").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    acts = np.load(SRC / "acts.npy", mmap_mode="r")
    st = np.load(SRC / "stats.npz")
    lam, U = np.linalg.eigh(st["cov"])
    P = U[:, ::-1][:, :dp]  # top-d' principal directions
    n = len(acts)
    dst = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(n, dp))
    s, G = np.zeros(dp), np.zeros((dp, dp))
    for i in range(0, n, CHUNK):
        y = (np.asarray(acts[i:i + CHUNK], dtype=np.float32) - st["mean"].astype(np.float32)) @ P.astype(np.float32)
        dst[i:i + len(y)] = y.astype(np.float16)
        y64 = y.astype(np.float64)
        s += y64.sum(0)
        G += y64.T @ y64
    dst.flush()
    del dst
    (out / "acts.npy.part").rename(out / "acts.npy")
    mean = s / n
    np.savez(out / "stats.npz", mean=mean, cov=(G - n * np.outer(mean, mean)) / (n - 1), n=n)
    json.dump({"source": str(SRC.relative_to(ROOT)), "d_prime": dp, "projection": "top-d' PCs of the source covariance, mean removed"},
              open(out / "config.json", "w"), indent=1)


def main():
    for dp in D_PRIMES:
        build(dp)
        for k in KS:
            name = f"gemma-2-2b-pca{dp}__blocks.12.hook_resid_post__real__w{WIDTH}__k{k}__s0"
            if (ROOT / "cache/saes_dsweep" / name / "train.json").exists():
                continue
            subprocess.run([sys.executable, "-u", str(ROOT / "train_sae.py"), "--model", f"gemma-2-2b-pca{dp}",
                            "--hook", "blocks.12.hook_resid_post", "--cond", "real", "--width", str(WIDTH), "--k", str(k),
                            "--tokens", "1000000", "--eval_tokens", str(EVAL), "--epochs", str(EPOCHS), "--seed", "0",
                            "--out", "cache/saes_dsweep"], check=True, cwd=ROOT, env=os.environ | {"TRANSFORMERLENS_ALLOW_MPS": "1"})
    rows = []
    for dp in D_PRIMES:
        Sig = np.load(ROOT / f"cache/acts/gemma-2-2b-pca{dp}/blocks.12.hook_resid_post/stats.npz")["cov"]
        se = np.sort(np.linalg.eigvalsh(Sig))[::-1]
        for k in KS:
            d = ROOT / f"cache/saes_dsweep/gemma-2-2b-pca{dp}__blocks.12.hook_resid_post__real__w{WIDTH}__k{k}__s0"
            info = json.load(open(d / "train.json"))
            fires = np.load(d / "fires.npy")
            W = np.load(d / "W_dec.npy")
            alive = fires > 0
            bg = basis_gamma(cloud_cov(W, mask=alive), Sig, rank_ranges=((1, 50), (1, max(dp // 8, 20))))
            e = eigen_slope(W, mask=alive)["eigenvalues"]
            rank_g = slope_from_eigenvalues(e, (5, 50))["slope"] / slope_from_eigenvalues(se, (5, 50))["slope"]
            rows.append({"d": dp, "k": k, "law": 1 - np.log(k) / np.log(dp), "fvu": info["fvu_eval"], "alive": int(alive.sum()),
                         "basis_gamma": [f["gamma"] for f in bg["fits"]], "basis_r2": [f["r2"] for f in bg["fits"]],
                         "rank_gamma_6_50": rank_g, "overlap": bg["subspace_overlap"]})
            x = rows[-1]
            print(f"d'={dp:>5} k={k:>3}  law {x['law']:.3f}  basis γ(1-50) {x['basis_gamma'][0]:.3f}  basis γ(1-d'/8) {x['basis_gamma'][1]:.3f}"
                  f"  rank γ(6-50) {rank_g:.3f}  fvu {x['fvu']:.3f}  alive {x['alive']}  overlap10 {x['overlap'][10]:.2f}", flush=True)
    out = ROOT / "results/review"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "dsweep.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
