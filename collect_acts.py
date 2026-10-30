"""Cache residual-stream activations once per (model, layer).

Writes cache/acts/<model>/<hook>/acts.npy (fp16 memmap, shape (n_tokens, d)), and stats.npz with
float64 mean and covariance over all cached tokens (used for the Gaussian null). Position 0 (BOS) is
dropped because its residual norm is an outlier.
Usage: python collect_acts.py --model pythia-70m --hook blocks.3.hook_resid_post --n_tokens 20_000_000
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
        buf.extend(tokenizer(ex["text"])["input_ids"] + [tokenizer.eos_token_id])
        while len(buf) >= ctx - 1:
            rows.append([tokenizer.bos_token_id] + buf[:ctx - 1])
            buf = buf[ctx - 1:]
            if len(rows) == batch:
                yield torch.tensor(rows)
                rows = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-70m")
    ap.add_argument("--hook", default="blocks.3.hook_resid_post")
    ap.add_argument("--n_tokens", type=int, default=20_000_000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model, device=dev, dtype=getattr(torch, args.dtype))
    layer = int(args.hook.split(".")[1])
    out = ROOT / "cache" / "acts" / args.model / args.hook
    out.mkdir(parents=True, exist_ok=True)
    d = model.cfg.d_model
    acts = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(args.n_tokens, d))
    s, G, n = np.zeros(d), np.zeros((d, d)), 0
    t0 = time.time()
    with torch.no_grad():
        for toks in token_batches(model.tokenizer, args.ctx, args.batch, args.seed):
            _, cache = model.run_with_cache(toks.to(dev), names_filter=args.hook, stop_at_layer=layer + 1)
            a = cache[args.hook][:, 1:].reshape(-1, d).float().cpu().numpy().astype(np.float64)
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
