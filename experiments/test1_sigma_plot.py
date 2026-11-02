"""Test 1 analysis: Sigma eigen slope vs layer, compared with the decoder-cloud slope at matched L0.

H1 (covariance inheritance) predicts, window by window, that the layer profile of Sigma's slope has the
same shape as the decoder profile from Phase 1b: mid-stack peak in the top windows, early peak in the bulk.
Reports per-window Pearson r across layers and the per-layer ratio gamma = decoder slope / Sigma slope
(if decoder eigenvalues ~ Sigma eigenvalues^gamma, gamma should be roughly constant across layers).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "test1"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
DEC = {("65k", 40): "C1", ("65k", 70): "C2", ("16k", 40): "C1", ("16k", 70): "C2"}


def main():
    z = np.load(OUT / "sigma_gemma-2-2b.npz")
    cov, n = z["cov"], int(z["n"])
    L = cov.shape[0]
    eig = np.stack([np.sort(np.linalg.eigvalsh(c))[::-1] for c in cov])
    sig = {w: np.array([slope_from_eigenvalues(e, w)["slope"] for e in eig]) for w in WINDOWS}
    dec = json.load(open(ROOT / "results" / "phase1" / "slope_vs_layer_matched_l0.json"))

    rows = {"n_tokens": n, "windows": [list(w) for w in WINDOWS],
            "sigma_slope": {str(w): sig[w].tolist() for w in WINDOWS},
            "top1_frac": (eig[:, 0] / eig.sum(1)).tolist(),
            "mean_sq_norm": z["mean_sq_norm"].tolist(), "correlations": {}}
    print(f"{'window':>12} " + " ".join(f"L{l:<5}" for l in range(0, L, 2)))
    for w in WINDOWS:
        print(f"{str(w):>12} " + " ".join(f"{sig[w][l]:+.2f} " for l in range(0, L, 2)))

    fig, axes = plt.subplots(3, len(WINDOWS), figsize=(4.2 * len(WINDOWS), 10.5))
    for j, w in enumerate(WINDOWS):
        key = f"ranks{w[0] + 1}-{w[1]}"
        ax = axes[0, j]
        ax.plot(range(L), sig[w], "ko-", ms=4, label="Σ (activations)")
        ax.set_title(f"ranks {w[0] + 1}–{w[1]}")
        ax.set_xlabel("layer")
        ax2 = ax.twinx()
        for (width, l0), c in DEC.items():
            dk = f"{width}_{key}_L0_{l0}"
            if dk not in dec:
                continue
            pts = sorted((int(k), v) for k, v in dec[dk].items())
            ls = "-" if width == "65k" else ":"
            ax2.plot(*zip(*pts), ls, color=c, ms=3, marker="o", label=f"decoder {width} L0={l0}")
        ax2.set_ylabel("decoder slope", color="C1")
        if j == 0:
            ax.set_ylabel("Σ eigen slope")
            ax.legend(loc="lower left", fontsize=7)
            ax2.legend(loc="lower right", fontsize=7)
        # Scatter + correlation, 65k L0=40/70 and 16k L0=40/70.
        ax = axes[1, j]
        for (width, l0), c in DEC.items():
            dk = f"{width}_{key}_L0_{l0}"
            if dk not in dec:
                continue
            layers = sorted(int(k) for k in dec[dk])
            x = sig[w][layers]
            y = np.array([dec[dk][str(l)] for l in layers])
            r = np.corrcoef(x, y)[0, 1]
            rows["correlations"][f"{width}_L0{l0}_{key}"] = {"pearson_r": float(r), "n_layers": len(layers),
                                                             "gamma_mean": float((y / x).mean()), "gamma_sd": float((y / x).std())}
            ax.scatter(x, y, color=c, marker="o" if width == "65k" else "x", label=f"{width} L0={l0}: r={r:+.2f}")
            for l, xi, yi in zip(layers, x, y):
                ax.annotate(str(l), (xi, yi), fontsize=6, alpha=0.6)
        ax.set_xlabel("Σ eigen slope")
        ax.set_ylabel("decoder eigen slope" if j == 0 else "")
        ax.legend(fontsize=7)
        # gamma vs layer
        ax = axes[2, j]
        for (width, l0), c in DEC.items():
            dk = f"{width}_{key}_L0_{l0}"
            if dk not in dec:
                continue
            layers = sorted(int(k) for k in dec[dk])
            g = np.array([dec[dk][str(l)] for l in layers]) / sig[w][layers]
            ax.plot(layers, g, "-" if width == "65k" else ":", color=c, marker="o", ms=3, label=f"{width} L0={l0}")
        ax.set_xlabel("layer")
        ax.set_ylabel("γ = decoder slope / Σ slope" if j == 0 else "")
        ax.axhline(0.5, color="0.7", lw=0.8)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"Test 1: activation covariance Σ vs decoder cloud, Gemma-2-2B ({n:,} Pile tokens)")
    fig.tight_layout()
    fig.savefig(OUT / "sigma_vs_decoder_layers.png", dpi=150)
    np.save(OUT / "sigma_eigenvalues.npy", eig)
    json.dump(rows, open(OUT / "sigma_slopes.json", "w"), indent=1)
    print("\ncorrelations across layers (decoder slope vs Σ slope):")
    for k, v in rows["correlations"].items():
        print(f"  {k:<28} r={v['pearson_r']:+.2f}  gamma={v['gamma_mean']:.2f}±{v['gamma_sd']:.2f}  (n={v['n_layers']})")


if __name__ == "__main__":
    main()
