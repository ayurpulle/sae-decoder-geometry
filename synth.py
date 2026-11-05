"""Planted-feature activations with a known generating process, cached in collect_acts.py's format so
train_sae.py (real / gaussian / scrambled) runs on them unchanged.

Each token is a sparse sum of features plus optional isotropic noise:
    x = sum_i  a_i * m_i * (cos(theta_i) u_i + sin(theta_i) v_i)  +  noise * N(0, I_d)
  a_i ~ Bernoulli(p_i), p_i = c * i^-(1+alpha) capped at P_MAX, c chosen so E[sum_i a_i] = mean_active.
  m_i  magnitude: |N(0,1)| (tail="normal") or lognormal(0, 1) (tail="lognormal"), times scale_i: 1 by default;
       with --var_slope tau, scale_i is set so feature i's variance Var(a_i m_i scale_i) ∝ i^-tau exactly, whatever
       p_i (including features at the P_MAX cap), rescaled so the total feature variance is unchanged.
  point feature:    theta_i = 0, so the token moves along +u_i only.
  manifold feature: theta_i ~ U(0, 2 pi) per token, a circle in span(u_i, v_i) (frac_manifold of features).
  directions: u_i, v_i = normalize(Q (s * g)), g ~ N(0, I_d), s_j = j^(-beta/2), Q a random rotation, so
              feature directions favour a power-law set of axes (beta = 0: isotropic). v_i is orthogonalised to u_i.
Writes cache/acts/synth-<name>/blocks.0.hook_resid_post/{acts.npy (fp16), stats.npz, config.json, truth.npz}.
Usage: python synth.py --name base --n_tokens 6000000
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
P_MAX = 0.5
CHUNK = 32_768


def feature_probs(M, alpha, mean_active):
    raw = np.arange(1, M + 1, dtype=np.float64) ** -(1 + alpha)
    lo, hi = 0.0, mean_active / raw.min()
    for _ in range(200):  # bisection on c so that sum(min(c*raw, P_MAX)) = mean_active
        c = (lo + hi) / 2
        if np.minimum(c * raw, P_MAX).sum() > mean_active:
            hi = c
        else:
            lo = c
    return np.minimum(lo * raw, P_MAX)


def feature_scales(p, tail, var_slope):
    """Per-feature magnitude scale giving feature variances ∝ i^-var_slope (None: all ones)."""
    if var_slope is None:
        return np.ones_like(p)
    m1, m2 = (math.sqrt(2 / math.pi), 1.0) if tail == "normal" else (math.exp(0.5), math.exp(2.0))
    var = p * m2 - (p * m1) ** 2
    target = np.arange(1, len(p) + 1, dtype=np.float64) ** -var_slope
    target *= var.sum() / target.sum()
    return np.sqrt(target / var)


def directions(M, d, beta, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.arange(1, d + 1, dtype=np.float64) ** (-beta / 2)
    U = (rng.standard_normal((M, d)) * s) @ Q.T
    V = (rng.standard_normal((M, d)) * s) @ Q.T
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V -= (V * U).sum(1, keepdims=True) * U
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return U, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_tokens", type=int, default=6_000_000)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--M", type=int, default=4096, help="number of planted features")
    ap.add_argument("--alpha", type=float, default=0.5, help="p_i ∝ i^-(1+alpha)")
    ap.add_argument("--mean_active", type=float, default=32.0)
    ap.add_argument("--beta", type=float, default=1.0, help="anisotropy of feature directions (0 = isotropic)")
    ap.add_argument("--frac_manifold", type=float, default=0.0)
    ap.add_argument("--tail", choices=["normal", "lognormal"], default="normal")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--var_slope", type=float, default=None, help="feature variance ∝ i^-var_slope via magnitudes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p = feature_probs(args.M, args.alpha, args.mean_active)
    scale = feature_scales(p, args.tail, args.var_slope)
    U, V = directions(args.M, args.d, args.beta, rng)
    is_manifold = np.zeros(args.M, dtype=bool)
    is_manifold[rng.choice(args.M, int(round(args.frac_manifold * args.M)), replace=False)] = True

    out = ROOT / "cache" / "acts" / f"synth-{args.name}" / "blocks.0.hook_resid_post"
    out.mkdir(parents=True, exist_ok=True)
    acts = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(args.n_tokens, args.d))
    # On CUDA draw on the GPU (CPU draws dominate at large M); MPS/CPU keep the CPU stream, so earlier data is unchanged.
    gdev = "cuda" if dev == "cuda" else "cpu"
    gen = torch.Generator(device=gdev).manual_seed(args.seed)
    pt = torch.tensor(p, dtype=torch.float32, device=dev)
    sc = torch.tensor(scale, dtype=torch.float32, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)
    Vt = torch.tensor(V, dtype=torch.float32, device=dev)
    man = torch.tensor(is_manifold, device=dev)
    s, G, n, act_count = np.zeros(args.d), np.zeros((args.d, args.d)), 0, 0.0
    t0 = time.time()
    while n < args.n_tokens:
        b = min(CHUNK, args.n_tokens - n)
        a = (torch.rand(b, args.M, generator=gen, device=gdev).to(dev) < pt).float()
        z = torch.randn(b, args.M, generator=gen, device=gdev).to(dev)
        m = z.abs() if args.tail == "normal" else torch.exp(z)
        theta = torch.rand(b, args.M, generator=gen, device=gdev).to(dev) * (2 * math.pi) * man
        coef = a * m * sc
        x = (coef * torch.cos(theta)) @ Ut + (coef * torch.sin(theta)) @ Vt
        if args.noise:
            x = x + args.noise * torch.randn(b, args.d, generator=gen, device=gdev).to(dev)
        acts[n:n + b] = x.cpu().numpy().astype(np.float16)
        s += x.sum(0).cpu().double().numpy()
        G += (x.T @ x).cpu().double().numpy()
        act_count += a.sum().item()
        n += b
    acts.flush()
    del acts
    (out / "acts.npy.part").rename(out / "acts.npy")
    mean = s / n
    cov = (G - n * np.outer(mean, mean)) / (n - 1)
    np.savez(out / "stats.npz", mean=mean, cov=cov, n=n)
    np.savez(out / "truth.npz", U=U, V=V, p=p, scale=scale, is_manifold=is_manifold)
    json.dump(vars(args) | {"P_MAX": P_MAX, "measured_mean_active": act_count / n, "seconds": time.time() - t0,
                            "p_top5": p[:5].tolist(), "p_min": float(p.min())},
              open(out / "config.json", "w"), indent=1)
    print(f"done {out.parent.name}: {n:,} tokens, measured mean active {act_count / n:.2f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
