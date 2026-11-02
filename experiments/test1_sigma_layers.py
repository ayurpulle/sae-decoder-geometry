"""Test 1 (H1: covariance inheritance): activation-covariance spectra for every Gemma-2-2B layer.

One forward pass over N tokens with a hook on each decoder layer's resid_post, accumulating the
per-layer sum and Gram matrix (float32 Gram per batch on the device, float64 accumulation on CPU).
No activations are stored. BOS position dropped, as in collect_acts.py.

H1 predicts the layer profile of Sigma's eigen slope mirrors the decoder cloud's: a mid-stack peak in the
top rank windows and an early-layer peak in the bulk windows. Analysis: test1_sigma_plot.py.

Usage: python experiments/test1_sigma_layers.py --dtype float16 --n_tokens 200000
       python experiments/test1_sigma_layers.py --dtype float32 --n_tokens 8184 --max_layers 13 --tag val32
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
from collect_acts import token_batches  # noqa: E402

OUT = ROOT / "results" / "test1"


class SigmaAccumulator:
    def __init__(self, n_layers, d, dev):
        self.s = [torch.zeros(d, dtype=torch.float64) for _ in range(n_layers)]
        self.G = [torch.zeros(d, d, dtype=torch.float64) for _ in range(n_layers)]
        self.norm2 = [0.0] * n_layers
        self.n = 0
        self.dev = dev

    def hook(self, layer):
        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            a = h[:, 1:].reshape(-1, h.shape[-1]).float()  # drop BOS
            self.s[layer] += a.sum(0).cpu().double()
            self.G[layer] += (a.T @ a).cpu().double()
            self.norm2[layer] += a.pow(2).sum().item()
            if layer == 0:
                self.n += a.shape[0]
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--n_tokens", type=int, default=200_000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_layers", type=int, default=None, help="truncate the model (validation runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.max_layers:
        model.model.layers = model.model.layers[: args.max_layers]
    model.lm_head = torch.nn.Identity()
    embed = model.model.embed_tokens.eval()  # scaled embedding stays on CPU in bf16
    model.model.embed_tokens = torch.nn.Embedding(1, model.config.hidden_size)
    model = model.to(dtype).to(dev).eval()
    n_layers, d = len(model.model.layers), model.config.hidden_size
    acc = SigmaAccumulator(n_layers, d, dev)
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(acc.hook(i))

    t0 = time.time()
    with torch.no_grad():
        for toks in token_batches(tok, args.ctx, args.batch, args.seed):
            model.model(inputs_embeds=embed(toks).to(dtype).to(dev))
            if acc.n // 50_000 != (acc.n - toks.shape[0] * (args.ctx - 1)) // 50_000:
                print(f"{acc.n:,} tokens  {time.time() - t0:.0f}s", flush=True)
            if acc.n >= args.n_tokens:
                break
    n = acc.n
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"sigma_{args.model.split('/')[-1]}{('_' + args.tag) if args.tag else ''}"
    means = np.stack([s.numpy() / n for s in acc.s])
    covs = np.stack([(G.numpy() - n * np.outer(m, m)) / (n - 1) for G, m in zip(acc.G, means)])
    np.savez(OUT / f"{name}.npz", mean=means, cov=covs, n=n,
             mean_sq_norm=np.array(acc.norm2) / n)
    json.dump(vars(args) | {"n_tokens_seen": n, "n_layers": n_layers, "d_model": d, "seconds": time.time() - t0,
                            "dataset": "monology/pile-uncopyrighted", "bos_dropped": True, "centered": True},
              open(OUT / f"{name}.json", "w"), indent=1)
    print("done", n, "tokens", n_layers, "layers", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
