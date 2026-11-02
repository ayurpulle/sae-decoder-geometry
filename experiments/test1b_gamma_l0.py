"""Test 1b: gamma(L0) at Gemma layer 12 — gamma = decoder eigen slope / Sigma eigen slope, per ladder SAE.

Inputs already on disk: decoder slopes for the 28 layer-12 SAEs (Phase 1a, 4 windows), Sigma_12 eigenvalues
(Test 1), measured L0 (Phase 1 step 2; name L0 for 262k-1m). Pythia-70m Phase 2 runs (k=32, d=512) are
added as a second model: Sigma from the cached activation stats, decoder slopes for real and Gaussian-null
SAEs, plotted against k/d.

Fits gamma against L0 two ways (gamma = a + b*log2 L0; gamma = c * L0^-beta) per window, reports R^2 and
the L0 at which the log-linear fit reaches gamma = 1.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "test1"
WIDTH_N = {"16k": 2**14, "32k": 2**15, "65k": 2**16, "131k": 2**17, "262k": 2**18, "524k": 2**19, "1m": 2**20}
PYTHIA_WINDOWS = [(2, 25), (5, 50), (10, 100)]
D_GEMMA, D_PYTHIA = 2304, 512


def main():
    ladder = json.load(open(ROOT / "results" / "phase1" / "width_ladder_L12.json"))
    windows = [tuple(w) for w in ladder["config"]["windows"]]
    freqs = json.load(open(ROOT / "results" / "phase1" / "freqs_L12.json"))["results"]
    sig_eig = np.load(OUT / "sigma_eigenvalues.npy")[12]
    sig = {w: slope_from_eigenvalues(sig_eig, w)["slope"] for w in windows}

    rows = []
    for sae_id, r in ladder["results"].items():
        l0 = freqs[sae_id]["l0_measured"] if sae_id in freqs else float(r["l0"])
        rows.append({"sae_id": sae_id, "width": r["width"], "n_latents": WIDTH_N[r["width"]], "l0": l0,
                     "l0_source": "measured" if sae_id in freqs else "name",
                     "gamma": {str(w): r["windows"][j]["slope"] / sig[w] for j, w in enumerate(windows)}})
    rows.sort(key=lambda x: (x["n_latents"], x["l0"]))

    fits = {}
    print(f"Sigma_12 slopes: " + "  ".join(f"{w}: {sig[w]:+.3f}" for w in windows))
    print(f"\n{'sae':<36} {'L0':>6} " + " ".join(f"g{str(w):>11}" for w in windows))
    for x in rows:
        print(f"{x['sae_id']:<36} {x['l0']:>6.1f} " + " ".join(f"{x['gamma'][str(w)]:>12.3f}" for w in windows))
    l0 = np.array([x["l0"] for x in rows])
    for w in windows:
        g = np.array([x["gamma"][str(w)] for x in rows])
        lin = linregress(np.log2(l0), g)
        pw = linregress(np.log(l0), np.log(g))
        fits[str(w)] = {"loglin_a": lin.intercept, "loglin_b_per_doubling": lin.slope, "loglin_r2": lin.rvalue ** 2,
                        "l0_at_gamma1": float(2 ** ((1 - lin.intercept) / lin.slope)),
                        "power_c": float(np.exp(pw.intercept)), "power_beta": -pw.slope, "power_r2": pw.rvalue ** 2}
        f = fits[str(w)]
        print(f"\nwindow {w}: gamma = {f['loglin_a']:.3f} {f['loglin_b_per_doubling']:+.3f}*log2(L0)  R2={f['loglin_r2']:.3f}"
              f"  -> gamma=1 at L0={f['l0_at_gamma1']:.1f}   |  gamma = {f['power_c']:.2f}*L0^-{f['power_beta']:.3f}  R2={f['power_r2']:.3f}")

    # Pythia-70m: Sigma from cached stats, decoder slopes from Phase 2 (mean over seeds).
    st = np.load(ROOT / "cache" / "acts" / "pythia-70m" / "blocks.3.hook_resid_post" / "stats.npz")
    p_eig = np.sort(np.linalg.eigvalsh(st["cov"]))[::-1]
    p_sig = {w: slope_from_eigenvalues(p_eig, w)["slope"] for w in PYTHIA_WINDOWS}
    p2 = json.load(open(ROOT / "results" / "phase2" / "phase2_small.json"))
    p2w = [tuple(w) for w in p2["config"]["windows"]]
    pythia = {}
    for cond in ["real", "gaussian", "scrambled"]:
        for width in sorted({r["width"] for r in p2["runs"]}):
            rs = [r for r in p2["runs"] if r["cond"] == cond and r["width"] == width]
            if not rs:
                continue
            pythia[f"{cond}_w{width}"] = {
                "k_over_d": 32 / D_PYTHIA, "n_seeds": len(rs),
                "gamma": {str(w): float(np.mean([r["windows"][p2w.index(w)]["slope"] for r in rs]) / p_sig[w])
                          for w in PYTHIA_WINDOWS}}
    print("\nPythia-70m L3 (k=32, k/d=6.25%), gamma by window", PYTHIA_WINDOWS)
    for k, v in pythia.items():
        print(f"  {k:<18} " + " ".join(f"{v['gamma'][str(w)]:.3f}" for w in PYTHIA_WINDOWS))

    json.dump({"sigma12_slopes": {str(w): sig[w] for w in windows}, "rows": rows, "fits": fits,
               "pythia": {"sigma_slopes": {str(w): p_sig[w] for w in PYTHIA_WINDOWS}, "runs": pythia}},
              open(OUT / "gamma_l0_L12.json", "w"), indent=1)

    fig, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 4), sharey=True)
    widths = [w for w in WIDTH_N if any(x["width"] == w for x in rows)]
    cmap = dict(zip(widths, plt.cm.viridis(np.linspace(0, 0.9, len(widths)))))
    for ax, w in zip(axes, windows):
        for width in widths:
            xs = [x for x in rows if x["width"] == width]
            ax.plot([x["l0"] for x in xs], [x["gamma"][str(w)] for x in xs], "o-", ms=4, color=cmap[width], label=width)
        f = fits[str(w)]
        grid = np.logspace(np.log10(4), np.log10(600), 50)
        ax.plot(grid, f["loglin_a"] + f["loglin_b_per_doubling"] * np.log2(grid), "k--", lw=1,
                label=f"log-linear R²={f['loglin_r2']:.2f}")
        ax.plot(grid, f["power_c"] * grid ** (-f["power_beta"]), "k:", lw=1, label=f"power β={f['power_beta']:.2f} R²={f['power_r2']:.2f}")
        # Pythia real/gaussian at k/d=6.25% -> equivalent Gemma L0 = 0.0625*2304 = 144, on the nearest Pythia window.
        pw_ = PYTHIA_WINDOWS[min(len(PYTHIA_WINDOWS) - 1, windows.index(w))]
        for cond, mk in [("real", "s"), ("gaussian", "^")]:
            ys = [v["gamma"][str(pw_)] for k, v in pythia.items() if k.startswith(cond)]
            ax.scatter([32 / D_PYTHIA * D_GEMMA] * len(ys), ys, marker=mk, color="C3" if cond == "gaussian" else "C0",
                       zorder=5, label=f"Pythia {cond} (k/d→L0 equiv., win {pw_})")
        ax.axhline(1, color="0.8", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("L0 (measured where available)")
        ax.set_title(f"ranks {w[0] + 1}–{w[1]}")
    axes[0].set_ylabel("γ = decoder slope / Σ slope")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Gemma-2-2B layer 12: γ vs L0 across the Gemma Scope width ladder")
    fig.tight_layout()
    fig.savefig(OUT / "gamma_l0_L12.png", dpi=150)


if __name__ == "__main__":
    main()
