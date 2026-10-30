"""TopK SAE (Gao et al. 2024) trained on real, Gaussian-null or scrambled-null activations.

Data conditions (identical architecture/optimiser/budget across them):
  real      cached activations, shuffled within 1M-token chunks (chunk order random).
  gaussian  N(mu, Sigma) with mu, Sigma from all cached activations (stats.npz).
  scrambled real chunk with every coordinate independently permuted across the chunk's tokens:
            keeps per-coordinate marginals, destroys joint structure.
Inputs are scaled by one constant c (from stats) so E||c x||^2 = d for every condition.
The last `eval_tokens` cached tokens are held out; eval uses held-out tokens for real/scrambled and
fresh samples for gaussian.
"""
import argparse
import json
import math
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


def chunks(cond, acts, n_train, stats, rng, d):
    """Yield float32 (CHUNK, d) training arrays for one pass over n_train tokens."""
    if cond == "gaussian":
        L = np.linalg.cholesky(stats["cov"])
        for _ in range(math.ceil(n_train / CHUNK)):
            yield (rng.standard_normal((CHUNK, d)) @ L.T + stats["mean"]).astype(np.float32)
        return
    for c in rng.permutation(math.ceil(n_train / CHUNK)):
        x = np.asarray(acts[c * CHUNK:min((c + 1) * CHUNK, n_train)], dtype=np.float32)
        if cond == "scrambled":
            x = scramble(x, rng)
        else:
            x = x[rng.permutation(len(x))]
        yield x


def eval_set(cond, acts, n_train, stats, rng, d, n_eval):
    if cond == "gaussian":
        L = np.linalg.cholesky(stats["cov"])
        return (rng.standard_normal((n_eval, d)) @ L.T + stats["mean"]).astype(np.float32)
    x = np.asarray(acts[n_train:n_train + n_eval], dtype=np.float32)
    if cond == "scrambled":
        x = scramble(x, rng)
    return x


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def train(args):
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = ROOT / "cache" / "acts" / args.model / args.hook
    acts = np.load(act_dir / "acts.npy", mmap_mode="r")
    stats = dict(np.load(act_dir / "stats.npz"))
    n_total, d = acts.shape
    n_train = min(args.tokens, n_total - args.eval_tokens)
    scale = math.sqrt(d / (np.trace(stats["cov"]) + stats["mean"] @ stats["mean"]))

    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    sae = TopKSAE(d, args.width, args.k, gen).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor(stats["mean"] * scale, dtype=torch.float32))
    lr = args.lr if args.lr else 2e-4 / math.sqrt(args.width / 16384)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    n_steps = n_train // args.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (n_steps - s) / (0.2 * n_steps)))
    since_fired = torch.zeros(args.width, device=dev)
    dead_window = args.dead_window
    step, t0 = 0, time.time()
    log = []
    for x_np in chunks(args.cond, acts, n_train, stats, rng, d):
        x_all = torch.from_numpy(x_np).to(dev) * scale
        for b in range(0, len(x_all) - args.batch + 1, args.batch):
            if step >= n_steps:
                break
            x = x_all[b:b + args.batch]
            recon, (v, i), pre = sae(x)
            err = x - recon
            mse = err.pow(2).sum(-1).mean()
            fvu = mse / (x - x.mean(0)).pow(2).sum(-1).mean()
            since_fired += args.batch
            since_fired[i[v > 0]] = 0
            dead = since_fired > dead_window
            loss = fvu
            if args.aux_k and dead.any():
                pre_dead = pre.masked_fill(~dead, -torch.inf)
                kk = min(args.aux_k, int(dead.sum()))
                va, ia = pre_dead.topk(kk, dim=-1)
                aux_recon = (torch.relu(va).unsqueeze(-1) * sae.W_dec[ia]).sum(1)
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
            if step % 500 == 0 or step == n_steps or step == 50:
                log.append({"step": step, "fvu": fvu.item(), "dead_frac": dead.float().mean().item()})
                print(f"[{args.cond} w{args.width} s{args.seed}] step {step}/{n_steps} fvu {fvu.item():.4f} "
                      f"dead {dead.float().mean().item():.3f} {time.time() - t0:.0f}s", flush=True)

    log.append({"step": step, "fvu": fvu.item(), "dead_frac": dead.float().mean().item()})

    # Eval: FVU and per-latent firing counts on held-out / fresh data.
    xe = eval_set(args.cond, acts, n_train, stats, np.random.default_rng(10_000 + args.seed), d, args.eval_tokens)
    fires = torch.zeros(args.width, device=dev)
    se, sv = 0.0, 0.0
    mu_e = torch.tensor(xe.mean(0) * scale, device=dev)
    with torch.no_grad():
        for b in range(0, len(xe), 8192):
            x = torch.from_numpy(xe[b:b + 8192]).to(dev) * scale
            recon, (v, i), _ = sae(x)
            fires.index_add_(0, i[v > 0], torch.ones_like(v[v > 0]))
            se += (x - recon).pow(2).sum().item()
            sv += (x - mu_e).pow(2).sum().item()
    return sae, {"fvu_eval": se / sv, "fires": fires.cpu().numpy(), "train_log": log, "lr": lr,
                 "n_train_tokens": n_steps * args.batch, "input_scale": scale, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-70m")
    ap.add_argument("--hook", default="blocks.3.hook_resid_post")
    ap.add_argument("--cond", choices=["real", "gaussian", "scrambled"], default="real")
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
    args = ap.parse_args()
    sae, info = train(args)
    name = f"{args.model}__{args.hook}__{args.cond}__w{args.width}__k{args.k}__s{args.seed}"
    out = ROOT / args.out / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "W_dec.npy", sae.W_dec.detach().cpu().numpy())
    np.save(out / "fires.npy", info.pop("fires"))
    json.dump({"config": vars(args) | {"git_hash": git_hash()}} | info, open(out / "train.json", "w"), indent=1)
    print("saved", out, "fvu_eval", info["fvu_eval"])


if __name__ == "__main__":
    main()
