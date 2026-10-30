"""Cache residual-stream activations once per (model, layer).

Writes cache/acts/<model>/<hook>/acts.npy (fp16 memmap, shape (n_tokens, d)), and stats.npz with
float64 mean and covariance over all cached tokens (used for the Gaussian null). Position 0 (BOS) is
dropped because its residual norm is an outlier.
Usage: python collect_acts.py --model pythia-70m --hook blocks.3.hook_resid_post --n_tokens 20_000_000
       python collect_acts.py --backend hf --model google/gemma-2-2b --hook blocks.12.hook_resid_post \
           --ctx 1024 --batch 4 --dtype bfloat16 --n_tokens 1_000_000
The hf backend reads resid_post of block L via a forward hook on decoder layer L and stops the forward
pass there (no TransformerLens weight conversion, which doubles peak memory for Gemma on a 16 GB Mac).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent


def token_batches(tokenizer, ctx, batch, seed):
    """Yield (batch, ctx) token tensors from the Pile (uncopyrighted), BOS + ctx-1 packed tokens."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    buf, rows = [], []
    for ex in ds:
        buf.extend(tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
        while len(buf) >= ctx - 1:
            rows.append([tokenizer.bos_token_id] + buf[:ctx - 1])
            buf = buf[ctx - 1:]
            if len(rows) == batch:
                yield torch.tensor(rows)
                rows = []


class _Stop(Exception):
    pass


class HFResid:
    """resid_post of decoder layer `layer` from a HF causal LM, stopping the forward pass early."""

    def __init__(self, name, layer, dev, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
        self.d_model = self.model.config.hidden_size
        self.dev = dev
        self.model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._out = output[0] if isinstance(output, tuple) else output
        raise _Stop

    def __call__(self, toks):
        try:
            self.model.model(toks.to(self.dev))
        except _Stop:
            pass
        return self._out


class TLResid:
    def __init__(self, name, hook, dev, dtype):
        self.model = HookedTransformer.from_pretrained(name, device=dev, dtype=dtype)
        self.tokenizer, self.d_model, self.dev, self.hook = self.model.tokenizer, self.model.cfg.d_model, dev, hook
        self.layer = int(hook.split(".")[1])

    def __call__(self, toks):
        _, cache = self.model.run_with_cache(toks.to(self.dev), names_filter=self.hook, stop_at_layer=self.layer + 1)
        return cache[self.hook]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-70m")
    ap.add_argument("--hook", default="blocks.3.hook_resid_post")
    ap.add_argument("--n_tokens", type=int, default=20_000_000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--backend", choices=["tl", "hf"], default="tl")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    layer = int(args.hook.split(".")[1])
    if args.backend == "hf":
        resid = HFResid(args.model, layer, dev, getattr(torch, args.dtype))
    else:
        resid = TLResid(args.model, args.hook, dev, getattr(torch, args.dtype))
    out = ROOT / "cache" / "acts" / args.model.split("/")[-1] / args.hook
    out.mkdir(parents=True, exist_ok=True)
    d = resid.d_model
    acts = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(args.n_tokens, d))
    s, G, n = np.zeros(d), np.zeros((d, d)), 0
    t0 = time.time()
    with torch.no_grad():
        for toks in token_batches(resid.tokenizer, args.ctx, args.batch, args.seed):
            a = resid(toks)[:, 1:].reshape(-1, d).float().cpu().numpy().astype(np.float64)
            a = a[: args.n_tokens - n]
            acts[n:n + len(a)] = a.astype(np.float16)
            s += a.sum(0)
            G += a.T @ a
            n += len(a)
            if n // 1_000_000 != (n - len(a)) // 1_000_000:
                print(f"{n:,} tokens  {time.time() - t0:.0f}s", flush=True)
            if n >= args.n_tokens:
                break
    acts.flush()
    del acts
    (out / "acts.npy.part").rename(out / "acts.npy")
    mean = s / n
    cov = (G - n * np.outer(mean, mean)) / (n - 1)
    np.savez(out / "stats.npz", mean=mean, cov=cov, n=n)
    json.dump(vars(args) | {"d_model": d, "n_tokens_written": n, "dataset": "monology/pile-uncopyrighted",
                            "bos_dropped": True, "seconds": time.time() - t0},
              open(out / "config.json", "w"), indent=1)
    print("done", n, d, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
