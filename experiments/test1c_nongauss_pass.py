"""Test 1c (b): per-layer non-Gaussianity of Gemma-2-2B residual activations.

One fp16 pass (same setup as test1_sigma_layers.py). For each layer, activations are centred with Test 1's
mean and projected onto the top N_PC eigenvectors of Test 1's Sigma; per-PC moments give excess kurtosis
(0 for a Gaussian). Also: per-raw-coordinate kurtosis (massive-activation dims) and token-norm quantiles.
Output: results/test1/nongauss.npz with kurt_pc (L, N_PC), kurt_coord (L, d), norm_q (L, 4) at [.5,.9,.99,.999].
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_acts import token_batches  # noqa: E402

OUT = ROOT / "results" / "test1"
N_PC = 150


class Moments:
    def __init__(self, means, vecs, dev):
        self.mu = [torch.tensor(m, dtype=torch.float32, device=dev) for m in means]
        self.V = [torch.tensor(v, dtype=torch.float32, device=dev) for v in vecs]  # (d, N_PC)
        L, d = means.shape
        self.m_pc = torch.zeros(L, 4, N_PC, dtype=torch.float64)
        self.m_co = torch.zeros(L, 4, d, dtype=torch.float64)
        self.norms = [[] for _ in range(L)]
        self.n = 0

    def hook(self, layer):
        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            a = h[:, 1:].reshape(-1, h.shape[-1]).float() - self.mu[layer]
            z = a @ self.V[layer]
            for k in range(4):
                self.m_pc[layer, k] += z.pow(k + 1).sum(0).cpu().double()
                self.m_co[layer, k] += a.pow(k + 1).sum(0).cpu().double()
            self.norms[layer].append((a + self.mu[layer]).norm(dim=1).cpu())
            if layer == 0:
                self.n += a.shape[0]
        return f


def kurtosis(m, n):
    """Excess kurtosis from raw power sums m[0..3] (sums of x, x^2, x^3, x^4) over n samples."""
    m1, m2, m3, m4 = m / n
    var = m2 - m1 ** 2
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return mu4 / var ** 2 - 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=100_000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)  # different shard from Test 1's Sigma pass
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(OUT / "sigma_gemma-2-2b.npz")
    vecs = []
    for c in z["cov"]:
        w, v = np.linalg.eigh(c)
        vecs.append(v[:, ::-1][:, :N_PC].copy())
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    name = "google/gemma-2-2b"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model.lm_head = torch.nn.Identity()
    embed = model.model.embed_tokens.eval()
    model.model.embed_tokens = torch.nn.Embedding(1, model.config.hidden_size)
    model = model.to(torch.float16).to(dev).eval()
    mom = Moments(z["mean"], vecs, dev)
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(mom.hook(i))
    t0 = time.time()
    with torch.no_grad():
        for toks in token_batches(tok, args.ctx, args.batch, args.seed):
            model.model(inputs_embeds=embed(toks).to(torch.float16).to(dev))
            if mom.n // 25_000 != (mom.n - toks.shape[0] * (args.ctx - 1)) // 25_000:
                print(f"{mom.n:,} tokens  {time.time() - t0:.0f}s", flush=True)
            if mom.n >= args.n_tokens:
                break
    n = mom.n
    kurt_pc = np.stack([kurtosis(mom.m_pc[l].numpy(), n) for l in range(len(vecs))])
    kurt_co = np.stack([kurtosis(mom.m_co[l].numpy(), n) for l in range(len(vecs))])
    norm_q = np.stack([np.quantile(torch.cat(mom.norms[l]).numpy(), [0.5, 0.9, 0.99, 0.999]) for l in range(len(vecs))])
    np.savez(OUT / "nongauss.npz", kurt_pc=kurt_pc, kurt_coord=kurt_co, norm_q=norm_q, n=n, n_pc=N_PC)
    print("done", n, "tokens", f"{time.time() - t0:.0f}s")
    print("mean excess kurtosis of top-50 PCs by layer:", np.round(kurt_pc[:, :50].mean(1), 2).tolist())


if __name__ == "__main__":
    main()
