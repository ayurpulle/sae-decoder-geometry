"""Resynthesised Gemma activations: rebuild real activations from a Gemma Scope SAE's codes, then break one property
at a time. Output is cached in synth.py's format (cache/acts/synth-<name>/blocks.0.hook_resid_post/), so train_sae.py
and phase3_match_analyze.py run on it unchanged.

Codes: z = JumpReLU(x W_enc + b_enc) with the SAE's thresholds (Gemma Scope, no b_dec subtraction on input).
Variants (x_hat = z' W' + b_dec):
  A  z' = z, W' = W_dec                                   resynthesis as-is (keeps all the SAE captures)
  B  z' = independent draws, W' = W_dec                   each latent fires independently with its own frequency and
                                                          its own nonzero-value distribution: co-occurrence removed
  C  z' = z, W' = W_dec[perm]                             real codes, decoder rows randomly permuted across latents:
                                                          frequency/co-occurrence-to-direction link removed
  D  independent draws and permuted rows                  both removed
  E  independent draws, W' = random unit directions       Gemma Scope's per-latent marginals on a random dictionary
  F  uniform independent draws, W' = W_dec                Gemma Scope's dictionary with every live latent given the
                                                          same frequency (mean over live latents, so L0 is kept) and
                                                          the same value distribution (pooled over firings)
  G  independent draws, W' = Gaussian-matched directions  Gemma Scope's marginals on unit directions drawn from a
                                                          Gaussian with the mean and covariance of W_dec's rows (the
                                                          dictionary's second-order geometry only; compare with D)
  H  uniform independent draws, W' = Gaussian-matched     F's marginals on G's directions (compare with F)
Per-latent frequency and value quantiles (for B, D-H) come from the real training split (codestats.npz next to the real
cache). A and C read real tokens in order, so the last eval_tokens stay the held-out ones; B, D-H draw fresh tokens.
Usage: python resynth.py --kind A --src cache/acts/gemma-2-2b/blocks.12.hook_resid_post \
           --sae cache/gemmascope/L12/params.npz --name rs12-A --n_tokens 12000000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CHUNK = 4096
Q = 64  # value quantiles per latent
STATS_TOKENS = 4_000_000  # tokens for the code statistics (frequencies and value quantiles)


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_sae(path, dev):
    p = np.load(path)
    return {k: torch.tensor(p[k], dtype=torch.float32, device=dev) for k in p.files}


def encode(x, sae):
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def real_chunks(acts, start, stop, dev):
    for i in range(start, stop, CHUNK):
        yield torch.tensor(np.asarray(acts[i:min(i + CHUNK, stop)], dtype=np.float32), device=dev)


def code_stats(acts, n, sae, dev):
    """Per-latent firing frequency and Q quantiles of the nonzero values over the first n tokens."""
    M = sae["W_enc"].shape[1]
    cols, vals, seen = [], [], 0
    for x in real_chunks(acts, 0, n, dev):
        z = encode(x, sae)
        r, c = z.nonzero(as_tuple=True)
        cols.append(c.int().cpu())
        vals.append(z[r, c].cpu())
        seen += len(x)
    sdev = "cuda" if dev == "cuda" else "cpu"
    c, v = torch.cat(cols).to(sdev).long(), torch.cat(vals).to(sdev)
    o = torch.argsort(v)
    c, v = c[o], v[o]
    o = torch.sort(c, stable=True).indices
    c, v = c[o], v[o]
    cnt = torch.bincount(c, minlength=M)
    start = torch.cumsum(cnt, 0) - cnt
    t = torch.linspace(0, 1, Q, dtype=torch.float64, device=sdev)
    pos = start[:, None].double() + (cnt[:, None] - 1).clamp(min=0).double() * t[None]
    lo = pos.floor().long().clamp(max=len(v) - 1)
    hi = torch.minimum(lo + 1, (start + cnt - 1).clamp(min=0)[:, None])
    w = (pos - lo).float()
    q = v[lo] * (1 - w) + v[hi] * w
    q[cnt == 0] = 0
    cnt, q = cnt.cpu(), q.cpu()
    return {"freq": (cnt.double() / seen).numpy(), "quantiles": q.numpy(), "n_tokens": seen,
            "mean_l0": float(cnt.sum() / seen)}


def sample_codes(b, freq, quant, gen, gdev, dev):
    """b tokens of independent per-latent codes: fire with freq_i, value from latent i's quantile function."""
    a = torch.rand(b, len(freq), generator=gen, device=gdev).to(dev) < freq
    r, c = a.nonzero(as_tuple=True)
    pos = torch.rand(len(r), generator=gen, device=gdev).to(dev) * (Q - 1)
    lo = pos.floor().long()
    w = pos - lo
    v = quant[c, lo] * (1 - w) + quant[c, (lo + 1).clamp(max=Q - 1)] * w
    z = torch.zeros(b, len(freq), device=dev)
    z[r, c] = v
    return z


def uniform_marginals(stats, seed, n=4_000_000):
    """Every live latent gets the mean live frequency and the pooled (firing-weighted) value quantile function."""
    f, q = stats["freq"], stats["quantiles"]
    live = f > 0
    rng = np.random.default_rng(seed + 4000)
    lat = rng.choice(len(f), n, p=f / f.sum())
    pos = rng.random(n) * (Q - 1)
    lo = pos.astype(int)
    w = pos - lo
    v = q[lat, lo] * (1 - w) + q[lat, np.minimum(lo + 1, Q - 1)] * w
    pooled = np.quantile(v, np.linspace(0, 1, Q))
    return {"freq": np.where(live, f[live].mean(), 0.0), "quantiles": np.where(live[:, None], pooled[None], 0.0)}


def gaussian_dictionary(Wd, seed):
    """Unit directions drawn from a Gaussian with the mean and covariance (ddof 1) of Wd's rows, one per row."""
    W = Wd.cpu().double()
    mu, C = W.mean(0), torch.cov(W.T)
    lam, U = torch.linalg.eigh(C)
    Z = torch.randn(W.shape, generator=torch.Generator().manual_seed(seed + 5000), dtype=torch.float64)
    R = mu + Z @ (U * lam.clamp(min=0).sqrt()).T
    return (R / R.norm(dim=1, keepdim=True)).float()


def variant(kind, sae, acts, n_tokens, seed, stats, dev):
    """Yield (x_hat chunk, x_real chunk or None) for variant kind."""
    Wd = sae["W_dec"]
    if kind in "CD":
        perm = torch.randperm(len(Wd), generator=torch.Generator().manual_seed(seed + 2000)).to(dev)
        Wd = Wd[perm]
    if kind == "E":
        R = torch.randn(Wd.shape, generator=torch.Generator().manual_seed(seed + 3000))
        Wd = (R / R.norm(dim=1, keepdim=True)).to(dev)
    if kind in "GH":
        Wd = gaussian_dictionary(Wd, seed).to(dev)
    if kind in "FH":
        stats = uniform_marginals(stats, seed)
    if kind in "AC":
        for x in real_chunks(acts, 0, n_tokens, dev):
            yield encode(x, sae) @ Wd + sae["b_dec"], x
    else:
        gdev = "cuda" if dev == "cuda" else "cpu"
        gen = torch.Generator(device=gdev).manual_seed(seed)
        f = torch.tensor(stats["freq"], dtype=torch.float32, device=dev)
        q = torch.tensor(stats["quantiles"], dtype=torch.float32, device=dev)
        for i in range(0, n_tokens, CHUNK):
            yield sample_codes(min(CHUNK, n_tokens - i), f, q, gen, gdev, dev) @ Wd + sae["b_dec"], None


def get_stats(src, acts, eval_tokens, sae, dev):
    p = Path(src) / "codestats.npz"
    if not p.exists():
        np.savez(p, **code_stats(acts, min(len(acts) - eval_tokens, STATS_TOKENS), sae, dev))
    s = np.load(p)
    return {k: s[k] for k in s.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list("ABCDEFGH"), required=True)
    ap.add_argument("--src", required=True, help="real activation cache dir (acts.npy)")
    ap.add_argument("--sae", required=True, help="Gemma Scope params.npz")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_tokens", type=int, default=None, help="default: all real tokens")
    ap.add_argument("--eval_tokens", type=int, default=500_000, help="held out from the code statistics")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = device()
    sae = load_sae(args.sae, dev)
    need_real = args.kind in "AC" or not (Path(args.src) / "codestats.npz").exists()
    acts = np.load(Path(args.src) / "acts.npy", mmap_mode="r") if need_real else None
    n = args.n_tokens if acts is None else min(args.n_tokens or len(acts), len(acts))
    stats = get_stats(args.src, acts, args.eval_tokens, sae, dev) if args.kind in "BDEFGH" else None
    out = ROOT / "cache" / "acts" / f"synth-{args.name}" / "blocks.0.hook_resid_post"
    out.mkdir(parents=True, exist_ok=True)
    d = sae["W_dec"].shape[1]
    mm = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(n, d))
    s, G, i, err, tot, t0 = torch.zeros(d, dtype=torch.float64, device="cpu"), np.zeros((d, d)), 0, 0.0, 0.0, time.time()
    for xh, x in variant(args.kind, sae, acts, n, args.seed, stats, dev):
        b = len(xh)
        mm[i:i + b] = xh.cpu().numpy().astype(np.float16)
        s += xh.sum(0).cpu().double()
        G += (xh.T @ xh).cpu().double().numpy()
        if x is not None:
            err += float(((xh - x) ** 2).sum())
            tot += float(((x - x.mean(0)) ** 2).sum())
        i += b
    mm.flush()
    del mm
    (out / "acts.npy.part").rename(out / "acts.npy")
    mean = (s / n).numpy()
    np.savez(out / "stats.npz", mean=mean, cov=(G - n * np.outer(mean, mean)) / (n - 1), n=n)
    json.dump(vars(args) | {"n_tokens_written": n, "fvu_vs_real": err / tot if tot else None,
                            "seconds": time.time() - t0}, open(out / "config.json", "w"), indent=1)
    print(f"done {out.parent.name}: {n:,} tokens, fvu vs real {err / tot if tot else float('nan'):.3f}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
