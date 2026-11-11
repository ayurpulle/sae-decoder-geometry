"""TopK SAE (Gao et al. 2024) trained on real, Gaussian-null or scrambled-null activations.

Data conditions (identical architecture/optimiser/budget across them):
  real      cached activations, shuffled within 1M-token chunks (chunk order random).
  gaussian  N(mu, Sigma) with mu, Sigma from all cached activations (stats.npz).
  scrambled real chunk with every coordinate independently permuted across the chunk's tokens:
            keeps per-coordinate marginals, destroys joint structure.
Inputs are scaled by one constant c (from stats) so E||c x||^2 = d for every condition.
The last `eval_tokens` cached tokens are held out; eval uses held-out tokens for real/scrambled and
fresh samples for gaussian.

--epochs E passes over the cached tokens E times (fresh chunk order / shuffles / scrambles per epoch).
--drop_dim D --drop_frac f drops tokens whose |x_D| exceeds the (1-f) quantile of the first training chunk
(e.g. Gemma's massive-activation dimension 1170), in training and eval; real/scrambled only.
On CUDA, chunks move to the GPU and shuffling/scrambling run there; the CPU/MPS path is unchanged.
--whiten_ridge r (real data only; off by default, and with it off nothing below changes): train on whitened inputs
    x_w = (Sigma + eps I)^{-1/2} (x - mu),   eps = r * mean eigenvalue of Sigma,
with mu, Sigma from stats.npz (all cached tokens). The constant input scale is then computed from x_w's own mean (0)
and covariance, so E||c x_w||^2 = d as for every other run, and W_dec lives in whitened space. Map a decoder row w
back with w (Sigma + eps I)^{1/2} (whiten_mats). Eval FVU and firing counts are in whitened space.
--cond sparse (off unless chosen; other conditions are unchanged): a sparse covariance-matched null,
    x = mu + s A,   A = R Sigma^{1/2}  (M x d),
R (M x d) = Q from the reduced QR of an M x d standard-normal matrix (numpy default_rng(5_000_000 + data_seed)), so
R^T R = I_d; Sigma^{1/2} the symmetric PSD root of stats.npz's cov. s has i.i.d. entries: nonzero with probability
p = L / M, nonzero values N(0, 1) / sqrt(p) (zero mean, variance 1/p), so E[s] = 0, Cov(s) = I_M and
Cov(x) = Sigma^{1/2} R^T R Sigma^{1/2} = Sigma exactly in expectation; mu and Sigma are those the gaussian cond uses.
--sparse_M (65536 sources), --sparse_L (64 expected active per token), --data_seed (dictionary and token streams;
separate from --seed, which sets only the SAE init). Tokens are drawn per batch on the GPU (CPU generator off CUDA):
training epoch e uses seed 7_000_000 + 1000 * data_seed + e, eval 9_000_000 + 1000 * data_seed (held out), and a 1M-token
check sample 8_000_000 + 1000 * data_seed whose covariance eigenvalues and relative Frobenius error against Sigma
are written to train.json ("sparse"). Like gaussian, it runs from stats.npz alone.
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CHUNK = 1_000_000


class TopKSAE(torch.nn.Module):
    def __init__(self, d, width, k, gen):
        super().__init__()
        W = torch.randn(width, d, generator=gen)
        W /= W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.W_enc = torch.nn.Parameter(W.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(width))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.k = k

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def forward(self, x):
        """Returns reconstruction, (top-k values, indices), and pre-activations. Sparse decode."""
        pre = self.pre(x)
        v, i = pre.topk(self.k, dim=-1)
        v = torch.relu(v)
        return (v.unsqueeze(-1) * self.W_dec[i]).sum(1) + self.b_dec, (v, i), pre

    @torch.no_grad()
    def renorm(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)


def scramble(x, rng):
    """Independently permute each coordinate across tokens."""
    return np.stack([x[rng.permutation(len(x)), j] for j in range(x.shape[1])], axis=1)


def batches(cond, acts, lo, hi, stats, seed, d, batch, scale, dev, shuffle=True, drop=None):
    """Yield scaled float32 (batch, d) tensors on `dev` covering tokens [lo, hi) once.

    Real/scrambled data stay fp16 on the CPU one CHUNK at a time; only batches move to the device
    (on CUDA the whole chunk moves and shuffling/scrambling run on the GPU).
    Gaussian samples are drawn per batch (CPU generator for reproducibility, transform on device).
    drop=(dim, thr): skip tokens with |x[dim]| > thr (raw units).
    """
    rng = np.random.default_rng(seed)
    torch_path = dev == "cuda" or os.environ.get("SAE_TORCH_DATA") == "1"  # env switch lets the GPU path be tested on CPU/MPS
    if cond == "gaussian":
        Lt = torch.tensor(np.linalg.cholesky(stats["cov"]).T * scale, dtype=torch.float32, device=dev)
        mu = torch.tensor(stats["mean"] * scale, dtype=torch.float32, device=dev)
        if torch_path:
            gen = torch.Generator(device=dev).manual_seed(seed)
            for _ in range((hi - lo) // batch):
                yield torch.randn(batch, d, generator=gen, device=dev) @ Lt + mu
            return
        gen = torch.Generator().manual_seed(seed)
        for _ in range((hi - lo) // batch):
            yield torch.randn(batch, d, generator=gen).to(dev) @ Lt + mu
        return
    starts = np.arange(lo, hi, CHUNK)
    if torch_path:
        gen = torch.Generator(device=dev).manual_seed(seed)
        for c in (rng.permutation(starts) if shuffle else starts):
            x = torch.from_numpy(np.array(acts[c:min(c + CHUNK, hi)])).to(dev)
            if drop is not None:
                x = x[x[:, drop[0]].abs() <= drop[1]]
            if cond == "scrambled":
                for j in range(0, d, 256):  # column blocks bound the argsort's memory
                    blk = x[:, j:j + 256]
                    order = torch.argsort(torch.rand(blk.shape, device=dev, generator=gen), dim=0)
                    x[:, j:j + 256] = torch.gather(blk, 0, order)
            idx = torch.randperm(len(x), device=dev, generator=gen) if shuffle else torch.arange(len(x), device=dev)
            for b in range(0, len(x) - batch + 1, batch):
                yield x[idx[b:b + batch]].float() * scale
            del x
        return
    for c in (rng.permutation(starts) if shuffle else starts):
        x = np.array(acts[c:min(c + CHUNK, hi)])  # copy into RAM; fancy-indexing a memmap is slow
        if drop is not None:
            x = x[np.abs(x[:, drop[0]].astype(np.float32)) <= drop[1]]
        if cond == "scrambled":
            x = scramble(x, rng)
        idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
        for b in range(0, len(x) - batch + 1, batch):
            yield torch.from_numpy(x[idx[b:b + batch]].astype(np.float32)).to(dev) * scale
        del x


def whiten_mats(cov, ridge):
    """(Sigma + eps I)^{-1/2}, (Sigma + eps I)^{1/2} and eps = ridge * mean eigenvalue, all float64."""
    lam, U = np.linalg.eigh(cov)
    lam = np.clip(lam, 0, None)
    eps = ridge * lam.mean()
    return (U / np.sqrt(lam + eps)) @ U.T, (U * np.sqrt(lam + eps)) @ U.T, float(eps)


def sparse_dictionary(cov, M, data_seed):
    """A = R Sigma^{1/2} (float32, M x d) for --cond sparse; deterministic in (cov, M, data_seed), any device."""
    lam, U = np.linalg.eigh(cov)
    root = (U * np.sqrt(np.clip(lam, 0, None))) @ U.T
    R, _ = np.linalg.qr(np.random.default_rng(5_000_000 + data_seed).standard_normal((M, cov.shape[0])))
    return (R @ root).astype(np.float32)


def sparse_batches(sp, n, batch, seed, scale, dev):
    """n // batch batches of x = (mu + s A) * scale for --cond sparse (see docstring)."""
    gdev = "cuda" if dev == "cuda" else "cpu"
    gen = torch.Generator(device=gdev).manual_seed(seed)
    A, mu, p = sp["A"], sp["mu"], sp["p"]
    M, d = A.shape
    for _ in range(n // batch):
        r, c = (torch.rand(batch, M, generator=gen, device=gdev) < p).nonzero(as_tuple=True)
        v = torch.randn(len(r), generator=gen, device=gdev) / math.sqrt(p)
        x = torch.zeros(batch, d, device=dev).index_add_(0, r.to(dev), A[c.to(dev)] * v.to(dev)[:, None])
        yield (x + mu) * scale


def sparse_check(sp, cov, data_seed, dev, n=1_000_000, batch=4096):
    """Covariance of an n-token check sample (raw units, float64): eigenvalues and relative Frobenius error."""
    d = cov.shape[0]
    s, G, k = np.zeros(d), np.zeros((d, d)), 0
    for x in sparse_batches(sp, n, batch, 8_000_000 + 1000 * data_seed, 1.0, dev):
        x = x.double() if dev == "cuda" else x.cpu().double()
        s += x.sum(0).cpu().numpy()
        G += (x.T @ x).cpu().numpy()
        k += len(x)
    C = (G - np.outer(s, s) / k) / (k - 1)
    return {"n": k, "eigenvalues": np.sort(np.linalg.eigvalsh(C))[::-1].tolist(),
            "rel_frob_err": float(np.linalg.norm(C - cov) / np.linalg.norm(cov))}


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # no checkout (e.g. a pod synced by gpu/push.sh, which writes .git_hash instead)
        f = ROOT / ".git_hash"
        return f.read_text().strip() if f.exists() else None


def train(args):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 tensor cores; every GPU-queue run uses the same setting
    act_dir = ROOT / "cache" / "acts" / args.model / args.hook
    stats = dict(np.load(act_dir / "stats.npz"))
    if (act_dir / "acts.npy").exists():
        acts = np.load(act_dir / "acts.npy", mmap_mode="r")
        n_total, d = acts.shape
    else:  # stats-only directory: Gaussian null can run from (mean, cov) alone
        assert args.cond in ("gaussian", "sparse"), f"{act_dir} has no acts.npy; only gaussian/sparse run from stats"
        acts, d = None, len(stats["mean"])
        n_total = args.tokens + args.eval_tokens
    n_train = min(args.tokens, n_total - args.eval_tokens)
    wt = None
    if args.whiten_ridge is not None:  # whitened inputs (see docstring); wt stays None otherwise
        assert args.cond == "real" and args.drop_dim is None, "--whiten_ridge applies to real data without --drop_dim"
        P, _, eps = whiten_mats(stats["cov"], args.whiten_ridge)
        wt = {"mu": torch.tensor(stats["mean"], dtype=torch.float32, device=dev),
              "P": torch.tensor(P, dtype=torch.float32, device=dev), "eps": eps}
        stats = {"mean": np.zeros(d), "cov": P @ stats["cov"] @ P, "n": stats["n"]}
    scale = math.sqrt(d / (np.trace(stats["cov"]) + stats["mean"] @ stats["mean"]))
    bscale = scale if wt is None else 1.0  # whitening needs raw-unit batches; it applies `scale` itself

    def feed(gen):
        return gen if wt is None else (((x - wt["mu"]) @ wt["P"]) * scale for x in gen)

    sp, sp_info = None, {}
    if args.cond == "sparse":  # sparse covariance-matched null (see docstring); sp stays None otherwise
        p_act = args.sparse_L / args.sparse_M
        sp = {"A": torch.tensor(sparse_dictionary(stats["cov"], args.sparse_M, args.data_seed), device=dev),
              "mu": torch.tensor(stats["mean"], dtype=torch.float32, device=dev), "p": p_act}
        sp_info = {"sparse": {"M": args.sparse_M, "L": args.sparse_L, "p": p_act, "data_seed": args.data_seed,
                              "values": "N(0,1)/sqrt(p)", "dict_seed": 5_000_000 + args.data_seed,
                              "train_seed": f"7_000_000 + 1000*{args.data_seed} + epoch",
                              "eval_seed": 9_000_000 + 1000 * args.data_seed,
                              "check": sparse_check(sp, stats["cov"], args.data_seed, dev)}}
    drop = None
    if args.drop_dim is not None:
        assert args.cond != "gaussian", "--drop_dim applies to real/scrambled data"
        col = np.abs(np.asarray(acts[:min(CHUNK, n_train), args.drop_dim], dtype=np.float32))
        drop = (args.drop_dim, float(np.quantile(col, 1 - args.drop_frac)))

    gen = torch.Generator().manual_seed(args.seed)
    sae = TopKSAE(d, args.width, args.k, gen).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor(stats["mean"] * scale, dtype=torch.float32))
    lr = args.lr if args.lr else 2e-4 / math.sqrt(args.width / 16384)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    n_steps = args.epochs * (n_train // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (n_steps - s) / (0.2 * n_steps)))
    since_fired = torch.zeros(args.width, device=dev)
    dead_window = args.dead_window
    step, t0 = 0, time.time()
    log = []
    def stream():
        for epoch in range(args.epochs):
            if sp is not None:
                yield from sparse_batches(sp, n_train, args.batch, 7_000_000 + 1000 * args.data_seed + epoch, scale, dev)
                continue
            yield from feed(batches(args.cond, acts, 0, n_train, stats, args.seed + 1000 * epoch, d, args.batch,
                                    bscale, dev, drop=drop))

    for x in stream():
        if step >= n_steps:
            break
        recon, (v, i), pre = sae(x)
        err = x - recon
        mse = err.pow(2).sum(-1).mean()
        fvu = mse / (x - x.mean(0)).pow(2).sum(-1).mean()
        since_fired += args.batch
        fired = torch.zeros_like(since_fired).scatter_add_(0, i.flatten(), (v.flatten() > 0).float())
        since_fired.mul_((fired == 0).float())  # fixed shapes: MPS caches a graph per new shape
        dead = since_fired > dead_window
        loss = fvu
        if args.aux_k and dead.any():
            pre_dead = pre.masked_fill(~dead, -torch.inf)
            kk = min(args.aux_k, int(dead.sum()))
            va, ia = pre_dead.topk(kk, dim=-1)
            # Dense product: gathering W_dec[ia] would materialise batch x aux_k x d floats (19 GB at d=2304).
            aux_recon = torch.zeros_like(pre).scatter_(-1, ia, torch.relu(va)) @ sae.W_dec
            aux = (err.detach() - aux_recon).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean()
            loss = loss + aux / 32
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Remove the gradient component parallel to each decoder row (keeps rows on the sphere).
        with torch.no_grad():
            g = sae.W_dec.grad
            g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step()
        sched.step()
        sae.renorm()
        step += 1
        if dev == "mps" and step % 200 == 0:
            torch.mps.empty_cache()  # MPS's caching allocator otherwise grows without bound on some runs
        if step % 500 == 0 or step == n_steps or step == 50:
            log.append({"step": step, "fvu": fvu.item(), "dead_frac": dead.float().mean().item()})
            print(f"[{args.cond} w{args.width} s{args.seed}] step {step}/{n_steps} fvu {fvu.item():.4f} "
                  f"dead {dead.float().mean().item():.3f} {time.time() - t0:.0f}s", flush=True)

    log.append({"step": step, "fvu": fvu.item(), "dead_frac": dead.float().mean().item()})

    # Eval: FVU and per-latent firing counts on held-out / fresh data. Two passes over the same eval stream (seeded): mean first, then errors and firings.
    eval_args = (args.cond, acts, n_train, n_train + args.eval_tokens, stats, 10_000 + args.seed, d, 8192, bscale, dev)
    eval_kw = {"drop": drop}
    fires = torch.zeros(args.width, device=dev)
    se, sv, mu_e, n_e = 0.0, 0.0, torch.zeros(d, device=dev), 0
    with torch.no_grad():
        for x in (feed(batches(*eval_args, shuffle=False, **eval_kw)) if sp is None else
                  sparse_batches(sp, args.eval_tokens, 8192, 9_000_000 + 1000 * args.data_seed, scale, dev)):
            mu_e += x.sum(0)
            n_e += len(x)
        mu_e /= n_e
        for x in (feed(batches(*eval_args, shuffle=False, **eval_kw)) if sp is None else
                  sparse_batches(sp, args.eval_tokens, 8192, 9_000_000 + 1000 * args.data_seed, scale, dev)):
            recon, (v, i), _ = sae(x)
            fires.scatter_add_(0, i.flatten(), (v.flatten() > 0).float())
            se += (x - recon).pow(2).sum().item()
            sv += (x - mu_e).pow(2).sum().item()
    return sae, {"fvu_eval": se / sv, "fires": fires.cpu().numpy(), "train_log": log, "lr": lr,
                 "n_train_tokens": step * args.batch, "n_eval_tokens": n_e, "input_scale": scale, "seconds": time.time() - t0,
                 "device": dev, "drop_threshold": drop[1] if drop else None} | (
                 {"whiten": {"ridge": args.whiten_ridge, "eps": wt["eps"]}} if wt else {}) | sp_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-70m")
    ap.add_argument("--hook", default="blocks.3.hook_resid_post")
    ap.add_argument("--cond", choices=["real", "gaussian", "scrambled", "sparse"], default="real")
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=19_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--aux_k", type=int, default=512)
    ap.add_argument("--dead_window", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/saes")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--drop_dim", type=int, default=None)
    ap.add_argument("--drop_frac", type=float, default=0.001)
    ap.add_argument("--whiten_ridge", type=float, default=None, help="train on whitened inputs (see docstring)")
    ap.add_argument("--sparse_M", type=int, default=65536, help="--cond sparse: number of sources")
    ap.add_argument("--sparse_L", type=float, default=64, help="--cond sparse: expected active sources per token")
    ap.add_argument("--data_seed", type=int, default=0, help="--cond sparse: dictionary and token-stream seed")
    args = ap.parse_args()
    sae, info = train(args)
    name = f"{args.model}__{args.hook}__{args.cond}__w{args.width}__k{args.k}__s{args.seed}"
    if args.drop_dim is not None:
        name += f"__drop{args.drop_dim}q{args.drop_frac:g}"
    if args.whiten_ridge is not None:
        name += "__white"
    if args.cond == "sparse":
        name += f"__ds{args.data_seed}"
    out = ROOT / args.out / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "W_dec.npy", sae.W_dec.detach().cpu().numpy())
    np.save(out / "fires.npy", info.pop("fires"))
    cfg = {k: v for k, v in vars(args).items() if not (k == "whiten_ridge" and v is None)  # off: config as before
           and not (k in ("sparse_M", "sparse_L", "data_seed") and args.cond != "sparse")}
    json.dump({"config": cfg | {"git_hash": git_hash()}} | info, open(out / "train.json", "w"), indent=1)
    print("saved", out, "fvu_eval", info["fvu_eval"])


if __name__ == "__main__":
    main()
