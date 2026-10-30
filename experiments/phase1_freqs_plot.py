"""Phase 1 step 2 plots: frequency exponent vs width/L0, and eigen slope vs frequency exponent (layer 12)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1"
WIDTH_N = {"16k": 2**14, "32k": 2**15, "65k": 2**16, "131k": 2**17}


def main():
    d = json.load(open(OUT / "freqs_L12.json"))["results"]
    widths = [w for w in WIDTH_N if any(r["width"] == w for r in d.values())]
    cmap = dict(zip(widths, plt.cm.viridis(np.linspace(0, 0.9, len(widths)))))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    print(f"{'sae':<36} {'L0':>6} {'dead':>5} {'eig':>7} " + " ".join(f"{str(f['rank_range']):>14}" for f in next(iter(d.values()))["freq_fits"]))
    for sae, r in sorted(d.items(), key=lambda kv: (WIDTH_N[kv[1]["width"]], kv[1]["l0_name"])):
        overall = r["freq_fits"][-1]["exponent"]
        c = cmap[r["width"]]
        axes[0].scatter(r["l0_measured"], overall, color=c)
        axes[1].scatter(r["l0_measured"], r["n_dead"] / r["n_latents"], color=c)
        axes[2].scatter(overall, r["slope_alive"], color=c)
        print(f"{sae:<36} {r['l0_measured']:>6.1f} {r['n_dead']:>5} {r['slope_alive']:>+7.3f} "
              + " ".join(f"{f['exponent']:>+14.2f}" for f in r["freq_fits"]))
    for w, c in cmap.items():
        axes[0].scatter([], [], color=c, label=w)
    axes[0].legend(title="width")
    axes[0].set(xscale="log", xlabel="measured L0", ylabel="rank-frequency exponent (ranks 11 to n_alive/2)")
    axes[1].set(xscale="log", xlabel="measured L0", ylabel="dead fraction (1M tokens)")
    axes[2].set(xlabel="rank-frequency exponent", ylabel="eigen slope, ranks 11–150 (alive)")
    fig.suptitle("Gemma-2-2B L12 Gemma Scope: firing frequencies (1M Pile tokens)")
    fig.tight_layout()
    fig.savefig(OUT / "freqs_L12.png", dpi=150)


if __name__ == "__main__":
    main()
