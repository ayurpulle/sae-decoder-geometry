"""Test 1c: what explains the layer profile of gamma = decoder slope / Sigma slope?

Candidates, each correlated with gamma across the 26 layers (per window; decoder at matched L0 from Phase 1b):
  (a) Sigma's shape: top-1 and top-10 variance fraction, participation ratio (effective dimension), and
      |mu|^2 / tr(Sigma) (mean-direction dominance) — from Test 1's per-layer covariances.
  (b) Non-Gaussianity: mean excess kurtosis along the top PCs, and the heavy-tail of token norms — from
      test1c_nongauss_pass.py (results/test1/nongauss.npz), if present.
  (c) Direct test: gamma of Sigma-matched Gaussian-null SAEs trained per layer (results/test1/null_saes.json),
      if present. If their gamma reproduces the mid-stack hump, the profile is the SAE's response to Sigma's
      shape; if it is flat, the hump comes from non-Gaussian structure.
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
PRIMARY = (10, 150)
DEC_KEYS = [("65k", 40), ("65k", 70), ("16k", 40), ("16k", 70)]


def gamma_profile(eig, dec, w, width, l0):
    key = f"{width}_ranks{w[0] + 1}-{w[1]}_L0_{l0}"
    layers = sorted(int(k) for k in dec[key])
    sig = np.array([slope_from_eigenvalues(eig[l], w)["slope"] for l in layers])
    d = np.array([dec[key][str(l)] for l in layers])
    return np.array(layers), d / sig, d, sig


def main():
    z = np.load(OUT / "sigma_gemma-2-2b.npz")
    cov, mean = z["cov"], z["mean"]
    eig = np.load(OUT / "sigma_eigenvalues.npy")
    dec = json.load(open(ROOT / "results" / "phase1" / "slope_vs_layer_matched_l0.json"))
    L = eig.shape[0]
    tr = eig.sum(1)
    stats = {
        "top1_frac": eig[:, 0] / tr,
        "top10_frac": eig[:, :10].sum(1) / tr,
        "participation_ratio": tr ** 2 / (eig ** 2).sum(1),
        "mean2_over_trace": (mean ** 2).sum(1) / tr,
        "sigma_slope_primary": np.array([slope_from_eigenvalues(e, PRIMARY)["slope"] for e in eig]),
        "sigma_slope_bulk": np.array([slope_from_eigenvalues(e, (100, 1000))["slope"] for e in eig]),
    }
    ng_path = OUT / "nongauss.npz"
    if ng_path.exists():
        ng = np.load(ng_path)
        stats["kurt_top50_mean"] = ng["kurt_pc"][:, :50].mean(1)
        stats["kurt_top50_median"] = np.median(ng["kurt_pc"][:, :50], axis=1)
        stats["kurt_pc1"] = ng["kurt_pc"][:, 0]
        stats["norm_q99_over_median"] = ng["norm_q"][:, -1] / ng["norm_q"][:, 2]  # quantiles saved as [.5,.9,.99,.999]? see pass
        stats["kurt_coord_max"] = ng["kurt_coord"].max(1)
    null_path = OUT / "null_saes.json"
    null = json.load(open(null_path)) if null_path.exists() else None

    out = {"layers": list(range(L)), "stats": {k: v.tolist() for k, v in stats.items()}, "gamma": {}, "correlations": {}}
    print(f"{'layer':>5} " + " ".join(f"{k[:14]:>14}" for k in stats) + "   gamma(65k,L0=40,11-150)")
    layers, g_ref, _, _ = gamma_profile(eig, dec, PRIMARY, "65k", 40)
    for l in range(L):
        gl = g_ref[list(layers).index(l)] if l in layers else float("nan")
        print(f"{l:>5} " + " ".join(f"{stats[k][l]:>14.3f}" for k in stats) + f"   {gl:.3f}")
    print("\nPearson r of each statistic with gamma across layers:")
    print(f"{'statistic':<22} " + " ".join(f"{w[0] + 1}-{w[1]}/{wd}/L0{l0}".rjust(16) for w in WINDOWS for wd, l0 in DEC_KEYS[:2]))
    for k, v in stats.items():
        cells = []
        for w in WINDOWS:
            for wd, l0 in DEC_KEYS[:2]:
                layers, g, _, _ = gamma_profile(eig, dec, w, wd, l0)
                r = np.corrcoef(v[layers], g)[0, 1]
                out["correlations"][f"{k}|{w}|{wd}|L0{l0}"] = float(r)
                cells.append(f"{r:+.2f}".rjust(16))
        print(f"{k:<22} " + " ".join(cells))
    for w in WINDOWS:
        for wd, l0 in DEC_KEYS:
            layers, g, d, s = gamma_profile(eig, dec, w, wd, l0)
            out["gamma"][f"{wd}_L0{l0}_{w}"] = {"layers": layers.tolist(), "gamma": g.tolist(), "decoder": d.tolist(), "sigma": s.tolist()}

    # Figure: gamma profile (primary) with candidates overlaid, normalised per panel.
    n_panels = 2 + (1 if ng_path.exists() else 0) + (1 if null else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4))
    axes = np.atleast_1d(axes)
    layers, g, d, s = gamma_profile(eig, dec, PRIMARY, "65k", 40)
    ax = axes[0]
    ax.plot(layers, g, "ko-", ms=4, label="γ (65k, L0=40, ranks 11–150)")
    _, g70, _, _ = gamma_profile(eig, dec, PRIMARY, "65k", 70)
    ax.plot(layers, g70, "o-", ms=3, color="0.5", label="γ (65k, L0=70)")
    ax.set_xlabel("layer"); ax.set_ylabel("γ"); ax.legend(fontsize=7); ax.set_title("γ layer profile")
    ax = axes[1]
    for k, c in [("participation_ratio", "C0"), ("top1_frac", "C1"), ("mean2_over_trace", "C2"), ("sigma_slope_primary", "C3")]:
        v = stats[k]
        ax.plot(range(L), (v - v.min()) / (v.max() - v.min()), "o-", ms=3, color=c,
                label=f"{k} (r={np.corrcoef(v[layers], g)[0, 1]:+.2f})")
    ax.plot(layers, (g - g.min()) / (g.max() - g.min()), "k-", lw=2, alpha=0.4, label="γ (scaled)")
    ax.set_xlabel("layer"); ax.set_ylabel("min–max scaled"); ax.legend(fontsize=6.5); ax.set_title("(a) Σ shape statistics")
    p = 2
    if ng_path.exists():
        ax = axes[p]; p += 1
        for k, c in [("kurt_top50_mean", "C0"), ("kurt_pc1", "C1"), ("norm_q99_over_median", "C2"), ("kurt_coord_max", "C3")]:
            v = stats[k]
            ax.plot(range(L), (v - v.min()) / (v.max() - v.min()), "o-", ms=3, color=c,
                    label=f"{k} (r={np.corrcoef(v[layers], g)[0, 1]:+.2f})")
        ax.plot(layers, (g - g.min()) / (g.max() - g.min()), "k-", lw=2, alpha=0.4, label="γ (scaled)")
        ax.set_xlabel("layer"); ax.legend(fontsize=6.5); ax.set_title("(b) non-Gaussianity")
    if null:
        ax = axes[p]
        ax.plot(layers, g, "ko-", ms=4, label="γ real (Gemma Scope 65k, L0=40)")
        ls = sorted(int(k) for k in null["gaussian"])
        for key, c, lab in [("gaussian", "C3", "γ Σ-matched Gaussian null SAE (k=40)"), ("real", "C0", "γ our TopK on real L12 acts")]:
            if key in null and null[key]:
                xs = sorted(int(k) for k in null[key])
                ax.plot(xs, [null[key][str(x)]["gamma"][str(PRIMARY)] for x in xs], "s--" if key == "gaussian" else "D", ms=6, color=c, label=lab)
        ax.set_xlabel("layer"); ax.set_ylabel("γ"); ax.legend(fontsize=7); ax.set_title("(c) Gaussian-null SAEs per layer")
    fig.suptitle("Test 1c: explaining the layer profile of γ")
    fig.tight_layout()
    fig.savefig(OUT / "gamma_layers_explained.png", dpi=150)
    json.dump(out, open(OUT / "gamma_layers_explained.json", "w"), indent=1)


if __name__ == "__main__":
    main()
