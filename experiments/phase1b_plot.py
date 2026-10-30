"""Phase 1b plots: eigen slope vs layer at matched L0 (log-linear interpolation in L0 per layer/width)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1"
TARGETS = [20, 40, 70, 140]
# The paper's one-SAE-per-layer choice (65k), for overlay.
PAPER_L0 = {0: 73, 4: 89, 8: 111, 12: 72, 16: 128, 20: 114, 24: 124}


def interp(rs, t, j):
    rs = sorted(rs, key=lambda r: r["l0"])
    l0 = np.log([r["l0"] for r in rs])
    if not (l0[0] <= np.log(t) <= l0[-1]):
        return None
    return float(np.interp(np.log(t), l0, [r["windows"][j]["slope"] for r in rs]))


def main():
    d = json.load(open(OUT / "layers.json"))
    res, wins = d["results"], d["config"]["windows"]
    groups = {}
    for r in res.values():
        groups.setdefault((r["width"], r["layer"]), []).append(r)
    table = {}
    widths = [w for w in ["16k", "65k"] if any(k[0] == w for k in groups)]
    fig, axes = plt.subplots(len(widths), len(wins), figsize=(4 * len(wins), 3.6 * len(widths)), squeeze=False)
    for a, width in enumerate(widths):
        layers = sorted(L for w, L in groups if w == width)
        for j, win in enumerate(wins):
            ax = axes[a, j]
            for t in TARGETS:
                pts = [(L, interp(groups[(width, L)], t, j)) for L in layers]
                pts = [(L, s) for L, s in pts if s is not None]
                ax.plot(*zip(*pts), "o-", ms=3, label=f"L0={t}")
                table[f"{width}_ranks{win[0] + 1}-{win[1]}_L0_{t}"] = dict(pts)
            if width == "65k":
                pp = [(L, next(r["windows"][j]["slope"] for r in groups[(width, L)] if r["l0"] == l0))
                      for L, l0 in PAPER_L0.items() if (width, L) in groups]
                if pp:
                    ax.plot(*zip(*pp), "k*--", ms=8, label="paper's SAEs")
            ax.set_title(f"{width}, ranks {win[0] + 1}–{win[1]}")
            ax.set_xlabel("layer")
        axes[a, 0].set_ylabel("eigen slope (interp. to L0)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "slope_vs_layer_matched_l0.png", dpi=150)
    json.dump(table, open(OUT / "slope_vs_layer_matched_l0.json", "w"), indent=1)
    for k, v in table.items():
        if "ranks11-150" in k:
            print(k, " ".join(f"{L}:{s:+.2f}" for L, s in v.items()))


if __name__ == "__main__":
    main()
