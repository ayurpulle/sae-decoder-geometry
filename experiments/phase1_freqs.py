"""Phase 1 step 2: latent firing frequencies for layer-12 Gemma Scope SAEs on cached activations.

Encoder (Gemma Scope JumpReLU, sae_lens cfg: apply_b_dec_to_input=False):
    pre = x @ W_enc + b_enc;  fires = pre > threshold
on raw blocks.12.hook_resid_post activations (BOS dropped, ctx 1024; collect_acts.py --backend hf).
Validation: measured mean L0 should match the L0 in each SAE's name.

Per SAE: firing counts, dead latents (zero firings), rank-frequency exponent over explicit windows, and the
decoder eigen slope with dead latents excluded (all latents is in Phase 1a). Widths <= 131k: the 524k/1m
encoders do not fit next to a batch of pre-activations in 16 GB.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402
from phase1_width_ladder import LADDER, WINDOWS, fetch_array, git_hash  # noqa: E402

LAYER = 12
WIDTHS = ["16k", "32k", "65k", "131k"]
ACTS = ROOT / "cache" / "acts" / "gemma-2-2b" / f"blocks.{LAYER}.hook_resid_post" / "acts.npy"
OUT = ROOT / "results" / "phase1"
# Rank windows (0-based [start, end)) for log(freq) vs log(rank); clipped to the number of alive latents.
FREQ_WINDOWS = [(10, 100), (100, 1000), (1000, 10000), (10, None)]
BATCH = 2048


def count_fires(acts, W_enc, b_enc, thr, dev):
    W = torch.from_numpy(W_enc).to(dev)
    b = torch.from_numpy(b_enc).to(dev)
    t = torch.from_numpy(thr).to(dev)
    fires = torch.zeros(W.shape[1], device=dev)
    l0 = 0.0
    with torch.no_grad():
        for i in range(0, len(acts), BATCH):
            x = torch.from_numpy(np.asarray(acts[i:i + BATCH], dtype=np.float32)).to(dev)
            on = (x @ W + b) > t
            fires += on.sum(0)
            l0 += on.sum().item()
    return fires.cpu().numpy().astype(np.int64), l0 / len(acts)


def freq_fits(freq_sorted):
    out = []
    n = len(freq_sorted)
    for lo, hi in FREQ_WINDOWS:
        hi = n // 2 if hi is None else min(hi, n)
        if hi - lo < 10:
            continue
        x = np.log10(np.arange(lo + 1, hi + 1))
        r = linregress(x, np.log10(freq_sorted[lo:hi]))
        out.append({"rank_range": [lo, hi], "exponent": float(r.slope), "r2": float(r.rvalue ** 2)})
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    acts = np.load(ACTS, mmap_mode="r")
    n_tok = len(acts)
    out_json = OUT / "freqs_L12.json"
    done = json.load(open(out_json))["results"] if out_json.exists() else {}
    for width in WIDTHS:
        for l0_name in LADDER[width]:
            sae_id = f"layer_{LAYER}/width_{width}/average_l0_{l0_name}"
            if sae_id in done:
                continue
            t0 = time.time()
            arrs = {k: np.load(fetch_array(sae_id, k)) for k in ["W_enc", "b_enc", "threshold"]}
            fires, l0 = count_fires(acts, arrs["W_enc"], arrs["b_enc"], arrs["threshold"], dev)
            np.save(OUT / f"fires_{sae_id.replace('/', '__')}.npy", fires)
            alive = fires > 0
            freq = np.sort(fires[alive])[::-1] / n_tok
            r = eigen_slope(np.load(fetch_array(sae_id, "W_dec"), mmap_mode="r"), mask=alive)
            done[sae_id] = {
                "width": width, "l0_name": l0_name, "l0_measured": l0, "n_tokens": n_tok,
                "n_latents": len(fires), "n_dead": int((~alive).sum()),
                "n_rare": int(((fires > 0) & (fires < 10)).sum()),
                "freq_fits": freq_fits(freq),
                "slope_alive": r["slope"], "windows_alive": [slope_from_eigenvalues(r["eigenvalues"], w) for w in WINDOWS],
            }
            fits = " ".join(f"{f['rank_range']}:{f['exponent']:+.2f}" for f in done[sae_id]["freq_fits"])
            print(f"{sae_id}: L0 measured {l0:.1f} (name {l0_name}) dead {done[sae_id]['n_dead']} "
                  f"slope_alive {r['slope']:+.4f} freq[{fits}] {time.time() - t0:.0f}s", flush=True)
            config = {"model": "gemma-2-2b", "layer": LAYER, "acts": str(ACTS.relative_to(ROOT)), "n_tokens": n_tok,
                      "encoder": "pre = x @ W_enc + b_enc; fire = pre > threshold (no b_dec on input)",
                      "freq_windows": FREQ_WINDOWS, "eig_windows": WINDOWS, "git_hash": git_hash()}
            json.dump({"config": config, "results": done}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
