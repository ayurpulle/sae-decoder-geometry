"""Phase 0: reproduce Li, Michaud et al. (2410.19750) galaxy-scale eigen slopes.

Exact SAEs from ejmichaud/feature-geometry galaxy notebook (cell 10): gemma-scope-2b-pt-res,
width 65k, one L0 per layer. Raw params.npz W_dec equals sae_lens' SAE.W_dec for residual
layers (the loader only rescales embedding SAEs), so we read the npz directly.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import PAPER_FIT_RANGE, eigen_slope, slope_from_eigenvalues  # noqa: E402

# (layer, L0) as in the original notebook, and the slopes it printed (cell 15/18 outputs).
PAPER = {0: (73, -0.2383), 12: (72, -0.4684), 24: (124, -0.2545)}
CACHE = ROOT / "cache" / "gemmascope"
OUT = ROOT / "results" / "phase0"
# Robustness windows (0-based [start, end) on rank-sorted eigenvalues).
ALT_RANGES = [(1, 15), (3, 50), (10, 150), (30, 450), (100, 1500), (10, 1500)]


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer, (l0, paper_slope) in PAPER.items():
        W = np.load(CACHE / f"L{layer}" / "params.npz")["W_dec"]
        variants = {
            "paper": eigen_slope(W),  # centred, raw rows
            "uncentered": eigen_slope(W, center=False),
            "unit_norm": eigen_slope(W, normalize=True),
        }
        p = variants["paper"]
        np.save(OUT / f"eig_L{layer}.npy", p["eigenvalues"])
        rel = (p["slope"] - paper_slope) / abs(paper_slope)
        rec = {
            "sae": f"gemma-scope-2b-pt-res/layer_{layer}/width_65k/average_l0_{l0}",
            "paper_slope": paper_slope,
            "rel_diff": rel,
            "variants": {k: {kk: vv for kk, vv in v.items() if kk not in ("eigenvalues",)}
                         | {"gaussian": {kk: vv for kk, vv in v["gaussian"].items() if kk != "eigenvalues"}}
                         for k, v in variants.items()},
            "fit_range_robustness": [slope_from_eigenvalues(p["eigenvalues"], r) for r in ALT_RANGES],
            "top_eigs": p["eigenvalues"][:5].tolist(),
        }
        results[layer] = rec
        print(f"L{layer:>2} slope={p['slope']:+.4f} paper={paper_slope:+.4f} rel={rel:+.2%} R2={p['r2']:.3f} "
              f"| uncent={variants['uncentered']['slope']:+.4f} unit={variants['unit_norm']['slope']:+.4f} "
              f"| gauss={p['gaussian']['slope']:+.4f} | norms {p['row_norm']}")
        for r in rec["fit_range_robustness"]:
            print(f"      range {r['fit_range']}: slope={r['slope']:+.4f} R2={r['r2']:.3f}")
        ranks = np.arange(1, len(p["eigenvalues"]) + 1)
        ax.loglog(ranks, p["eigenvalues"], label=f"L{layer} (slope {p['slope']:.3f})")
        if layer == 12:
            ax.loglog(ranks, p["gaussian"]["eigenvalues"], "k--", label=f"matched Gaussian ({p['gaussian']['slope']:.3f})")
        del W
    lo, hi = PAPER_FIT_RANGE
    ax.axvspan(lo + 1, hi, color="0.9", zorder=0, label="fit ranks 11–150")
    ax.set_xlabel("rank")
    ax.set_ylabel("eigenvalue of decoder-direction covariance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", dpi=150)

    config = {"model": "gemma-2-2b", "release": "gemma-scope-2b-pt-res", "width": "65k",
              "fit_range": list(PAPER_FIT_RANGE), "center": True, "normalize": False,
              "dead_latents_excluded": 0, "git_hash": git_hash(),
              "reference": "ejmichaud/feature-geometry galaxy/PointCloudStructure_Galaxy.ipynb cells 10,15,18"}
    json.dump({"config": config, "results": results}, open(OUT / "phase0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
