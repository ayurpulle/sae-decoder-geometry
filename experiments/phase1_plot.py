"""Phase 1a plots: layer-12 eigen slope vs L0 (per width) and vs width (L0 matched by interpolation)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1"
WIDTH_N = {"16k": 2**14, "32k": 2**15, "65k": 2**16, "131k": 2**17, "262k": 2**18, "524k": 2**19, "1m": 2**20}


def main():
    d = json.load(open(OUT / "width_ladder_L12.json"))
    res, wins = d["results"], d["config"]["windows"]
    by_w = {}
    for r in res.values():
        by_w.setdefault(r["width"], []).append(r)
    widths = sorted(by_w, key=WIDTH_N.get)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))

    fig, axes = plt.subplots(1, len(wins), figsize=(4 * len(wins), 3.8), sharey=True)
    for j, (ax, win) in enumerate(zip(axes, wins)):
        for w, c in zip(widths, cmap):
            rs = sorted(by_w[w], key=lambda r: r["l0"])
            ax.plot([r["l0"] for r in rs], [r["windows"][j]["slope"] for r in rs], "o-", color=c, label=w)
        ax.set_xscale("log")
        ax.set_xlabel("L0")
        ax.set_title(f"ranks {win[0] + 1}–{win[1]}")
    axes[0].set_ylabel("eigen slope")
    axes[0].legend(title="width", fontsize=8)
    fig.suptitle("Gemma-2-2B L12 Gemma Scope: decoder-direction eigen slope")
    fig.tight_layout()
    fig.savefig(OUT / "slope_vs_l0_L12.png", dpi=150)

    # Slope vs width at matched L0 (log-linear interpolation in L0 within each width's range).
    targets = [20, 40, 70]
    fig, axes = plt.subplots(1, len(wins), figsize=(4 * len(wins), 3.8), sharey=True)
    table = {}
    for j, (ax, win) in enumerate(zip(axes, wins)):
        for t in targets:
            xs, ys = [], []
            for w in widths:
                rs = sorted(by_w[w], key=lambda r: r["l0"])
                l0 = np.log([r["l0"] for r in rs])
                if not (l0[0] <= np.log(t) <= l0[-1]):
                    continue
                xs.append(WIDTH_N[w])
                ys.append(float(np.interp(np.log(t), l0, [r["windows"][j]["slope"] for r in rs])))
            ax.plot(xs, ys, "o-", label=f"L0≈{t}")
            table[f"ranks{win[0] + 1}-{win[1]}_L0_{t}"] = dict(zip(map(str, xs), ys))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("width")
        ax.set_title(f"ranks {win[0] + 1}–{win[1]}")
    axes[0].set_ylabel("eigen slope (interpolated to L0)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "slope_vs_width_L12.png", dpi=150)
    json.dump(table, open(OUT / "slope_vs_width_matched_l0.json", "w"), indent=1)
    for k, v in table.items():
        print(k, {WIDTH_N_INV.get(int(x), x): round(y, 3) for x, y in v.items()})


WIDTH_N_INV = {v: k for k, v in WIDTH_N.items()}

if __name__ == "__main__":
    main()
