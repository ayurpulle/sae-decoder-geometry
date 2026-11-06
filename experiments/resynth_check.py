"""Local check of resynth.py's variants before any GPU run, on the local 1M-token Gemma-2-2B layer-12 cache.

Code statistics from the first N_STATS tokens; each variant's Sigma from the next N_EVAL tokens (A, C) or N_EVAL fresh
draws (B, D), against real Sigma on the same N_EVAL tokens. Reports the Gemma Scope SAE's L0 and FVU, and Sigma's
eigen-slope per window. gamma normalises by each dataset's own Sigma, so Sigma may move, but a variant whose Sigma is
far from power-law in a window makes gamma there a ratio over a different denominator.
Writes results/resynth/check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resynth  # noqa: E402
from geometry import slope_from_eigenvalues  # noqa: E402

SRC = ROOT / "cache" / "acts" / "gemma-2-2b" / "blocks.12.hook_resid_post"
SAE = ROOT / "cache" / "gemmascope" / "L12" / "params.npz"
N_STATS, N_EVAL = 300_000, 200_000
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]


class Window:  # acts view over [off, off + n)
    def __init__(self, a, off, n):
        self.a, self.off, self.n = a, off, n

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        return self.a[self.off + s.start:self.off + min(s.stop, self.n)]


def cov(chunks):
    s, G, n = 0, 0, 0
    for x in chunks:
        x = x.cpu().double()
        s, G, n = s + x.sum(0), G + x.T @ x, n + len(x)
    m = s / n
    return ((G - n * torch.outer(m, m)) / (n - 1)).numpy()


def slopes(C):
    e = np.sort(np.linalg.eigvalsh(C))[::-1]
    return [slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS], e


def main():
    dev = resynth.device()
    sae = resynth.load_sae(SAE, dev)
    acts = np.load(SRC / "acts.npy", mmap_mode="r")
    stats = resynth.code_stats(acts, N_STATS, sae, dev)
    ev = Window(acts, N_STATS, N_EVAL)
    out = {"mean_l0": stats["mean_l0"], "n_alive": int((stats["freq"] > 0).sum()), "variants": {}}
    print(f"SAE L0 {stats['mean_l0']:.1f}, alive {out['n_alive']}", flush=True)
    sl, e = slopes(cov(resynth.real_chunks(ev, 0, N_EVAL, dev)))
    out["variants"]["real"] = {"sigma_slopes": sl, "eig": e[:1000].tolist()}
    print("real ", " ".join(f"{x:.3f}" for x in sl), flush=True)
    for kind in "ABCD":
        err = tot = 0.0

        def gen():
            nonlocal err, tot
            for xh, x in resynth.variant(kind, sae, ev, N_EVAL, 0, stats, dev):
                if x is not None:
                    err += float(((xh - x) ** 2).sum())
                    tot += float(((x - x.mean(0)) ** 2).sum())
                yield xh
        sl, e = slopes(cov(gen()))
        out["variants"][kind] = {"sigma_slopes": sl, "fvu_vs_real": err / tot if tot else None, "eig": e[:1000].tolist()}
        print(f"{kind}     " + " ".join(f"{x:.3f}" for x in sl) + (f"   fvu {err / tot:.3f}" if tot else ""), flush=True)
    (ROOT / "results" / "resynth").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "resynth" / "check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
