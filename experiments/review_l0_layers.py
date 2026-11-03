"""Review check 3: measured L0 of every Gemma Scope 16k residual SAE, all 26 layers, on our tokens.

Phase 1b interpolated layer slopes to matched L0 using the L0 in each SAE's name. At layer 12 the name
matches our measurement within ~4% (phase1_freqs.py); if the ratio measured/named varies by layer, the
matched-L0 layer profile (and its mid-stack peak) could shift. This measures every 16k SAE on the same
32,736 tokens (32 Pile sequences, ctx 1024, BOS dropped, seed 0: the first rows of the layer-12 cache).

Stage 1: one full forward pass of Gemma-2-2b (fp32 on MPS), resid_post of every layer -> fp16 arrays.
         Layer 12 is checked against cache/acts/gemma-2-2b/blocks.12.hook_resid_post/acts.npy.
Stage 2: per SAE, range-read W_enc/b_enc/threshold, L0 = mean #(x @ W_enc + b_enc > threshold).
         W_enc (151 MB each, 20 GB total) is deleted after use; b_enc/threshold are kept.
Writes results/review/l0_layers.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from collect_acts import HFResid, token_batches  # noqa: E402
from phase1_freqs import count_fires  # noqa: E402
from phase1_width_ladder import fetch_array, git_hash  # noqa: E402

N_SEQ, CTX, SEED = 32, 1024, 0
ACTS = ROOT / "cache" / "acts" / "gemma-2-2b-all"
OUT = ROOT / "results" / "review" / "l0_layers.json"


def collect(dev):
    if (ACTS / "L25.npy").exists():
        return
    ACTS.mkdir(parents=True, exist_ok=True)
    resid = HFResid("google/gemma-2-2b", 25, dev, torch.float32)
    layers = resid.model.model.layers
    outs = {}
    for L in range(25):
        layers[L].register_forward_hook(lambda m, i, o, L=L: outs.__setitem__(L, o[0] if isinstance(o, tuple) else o))
    bufs = {L: [] for L in range(26)}
    t0 = time.time()
    with torch.no_grad():
        for j, toks in enumerate(token_batches(resid.tokenizer, CTX, 4, SEED)):
            out25 = resid(toks)
            for L in range(26):
                a = (out25 if L == 25 else outs[L])[:, 1:].reshape(-1, resid.d_model)
                bufs[L].append(a.float().cpu().numpy().astype(np.float16))
            print(f"batch {j + 1}/{N_SEQ // 4}  {time.time() - t0:.0f}s", flush=True)
            if (j + 1) * 4 >= N_SEQ:
                break
    for L in range(26):
        np.save(ACTS / f"L{L}.npy", np.concatenate(bufs[L]))
    del resid


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    collect(dev)
    a12 = np.load(ACTS / "L12.npy", mmap_mode="r")
    ref = np.load(ROOT / "cache" / "acts" / "gemma-2-2b" / "blocks.12.hook_resid_post" / "acts.npy", mmap_mode="r")
    x, y = np.asarray(a12, np.float32), np.asarray(ref[:len(a12)], np.float32)
    check = float(np.linalg.norm(x - y) / np.linalg.norm(y))
    print(f"layer-12 check vs cache: relative error {check:.2e}", flush=True)

    avail = json.load(open(ROOT / "results" / "available_saes.json"))["variants"]
    saes = sorted((v for v in avail if v["width"] == "16k" and v["layer"] != "embedding"),
                  key=lambda v: (v["layer"], v["l0"]))
    res = json.load(open(OUT))["results"] if OUT.exists() else {}
    for v in saes:
        sid = v["sae_id"]
        if sid in res:
            continue
        t0 = time.time()
        acts = np.load(ACTS / f"L{v['layer']}.npy", mmap_mode="r")
        p_enc = fetch_array(sid, "W_enc")
        W = np.load(p_enc)
        b, thr = (np.load(fetch_array(sid, n)) for n in ("b_enc", "threshold"))
        fires, l0 = count_fires(acts, W, b, thr, dev)
        p_enc.unlink()
        res[sid] = {"layer": v["layer"], "l0_name": v["l0"], "l0_measured": l0, "ratio": l0 / v["l0"],
                    "dead": int((fires == 0).sum())}
        print(f"{sid:40s} named {v['l0']:4d}  measured {l0:7.1f}  ratio {l0 / v['l0']:.3f}  "
              f"dead {res[sid]['dead']}  {time.time() - t0:.0f}s", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": {"n_tokens": len(acts), "n_seq": N_SEQ, "ctx": CTX, "seed": SEED, "width": "16k",
                              "layer12_check_rel_err": check, "git": git_hash()}, "results": res},
                  open(OUT.with_suffix(".tmp"), "w"), indent=1)
        OUT.with_suffix(".tmp").replace(OUT)


if __name__ == "__main__":
    main()
