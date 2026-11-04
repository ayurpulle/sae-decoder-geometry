"""Summarise the width sweep: gamma(k) fit per width and condition at fixed density (i-m2048-a16), and the k-scale
log2 k0 = intercept / -slope (where the fitted gamma would reach 0), against log2 width.

Reads results/phase3/{saes_p3i, saes_p3w}.json (phase3_analyze.py output). Primary window (ranks 6-50); runs need
>= MIN_ALIVE alive latents (lower than phase3_analyze's 1000 so that narrow SAEs count; 250 is 5x the window's top
rank). Gemma's TopK fit (16k wide, ranks 11-150, k = 1-128) has log2 k0 = 10.8. Writes results/phase3/width_summary.{json,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3"
DATA = "i-m2048-a16"
K = [4, 8, 16, 32, 64]
PRIMARY = "(5, 50)"
MIN_ALIVE = 250
GEMMA_LOG2_K0 = 1.016 / 0.094


def main():
    rows = []
    for g in ["saes_p3i", "saes_p3w"]:
        p = OUT / f"{g}.json"
        if p.exists():
            rows += [r for r in json.load(open(p))["rows"] if r["data"] == DATA]
    widths = sorted({r["width"] for r in rows})
    out = {}
    print(f"{'cond':<9} {'width':>6} " + " ".join(f"k={k:<5}" for k in K) + "  intercept  per-doubling  R2   log2 k0  alive(k=4)")
    for cond in ["real", "gaussian"]:
        for w in widths:
            xs = {r["k"]: r for r in rows if r["cond"] == cond and r["width"] == w and r["k"] in K}
            g = {k: xs[k]["gamma"].get(PRIMARY) for k in xs}
            ok = [k for k in K if k in xs and xs[k]["alive"] >= MIN_ALIVE and g[k] is not None]
            fit = linregress(np.log2(ok), [g[k] for k in ok]) if len(ok) >= 3 else None
            k0 = fit.intercept / -fit.slope if fit and fit.slope < 0 else None
            out[f"{cond}|{w}"] = {"gamma": g, "alive": {k: xs[k]["alive"] for k in xs}, "ks_fit": ok,
                                  "intercept": fit.intercept if fit else None, "per_doubling": fit.slope if fit else None,
                                  "r2": fit.rvalue ** 2 if fit else None, "log2_k0": k0}
            cells = " ".join(f"{g[k]:<7.3f}" if g.get(k) is not None else "   -   " for k in K)
            fs = f"{fit.intercept:9.3f}  {fit.slope:+12.3f}  {fit.rvalue ** 2:.2f}" if fit else " " * 30
            print(f"{cond:<9} {w:>6} {cells}  {fs}  {k0 if k0 is None else round(k0, 2):>7}  {xs.get(4, {}).get('alive')}")
    json.dump({"data": DATA, "K": K, "window": PRIMARY, "min_alive": MIN_ALIVE, "gemma_log2_k0": GEMMA_LOG2_K0,
               "fits": out}, open(OUT / "width_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for cond, ls in [("real", "-"), ("gaussian", "--")]:
        for i, w in enumerate(widths):
            f = out[f"{cond}|{w}"]
            ks = sorted(f["gamma"])
            ax[0].plot(np.log2(ks), [f["gamma"][k] for k in ks], ls, color=plt.cm.viridis(i / max(1, len(widths) - 1)),
                       marker="o", ms=3, label=f"{cond}, w{w}")
        pts = [(w, out[f"{cond}|{w}"]["log2_k0"]) for w in widths if out[f"{cond}|{w}"]["log2_k0"] is not None]
        if pts:
            ax[1].plot(np.log2([p[0] for p in pts]), [p[1] for p in pts], "o" + ls, label=cond)
    ax[1].axhline(GEMMA_LOG2_K0, color="k", ls=":", label="Gemma L13 TopK (w16k)")
    ax[0].set_xlabel("log₂ k")
    ax[0].set_ylabel("γ (ranks 6–50)")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xlabel("log₂ width")
    ax[1].set_ylabel("log₂ k₀ (fitted γ = 0)")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"Width sweep, planted {DATA}, d = 256")
    fig.tight_layout()
    fig.savefig(OUT / "width_summary.png", dpi=150)


if __name__ == "__main__":
    main()
