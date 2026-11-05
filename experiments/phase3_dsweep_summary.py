"""Summarise the d-sweep: gamma(k) per (d, width, condition) and the k-scale log2 k0 = intercept / -slope, against
log2 d, for the scaled-width (width = 16 d) and fixed-width (4096) arms.

Analysis is done here rather than by phase3_analyze.py, whose rank windows assume d = 256. gamma = decoder eigen slope
(alive latents) / slope of the training data's covariance, in two windows per run:
    fixed   ranks 6-50 (0-based (5, 50)), as in every other Phase 3 fit
    scaled  ranks d/43 .. d/5, which is ranks 6-51 at d = 256
Runs need >= MIN_ALIVE alive latents. Gemma's TopK fit (d = 2304, width 16k) has log2 k0 = 10.8.
Reads cache/saes_p3d plus the d = 256 runs (i-m2048-a16, width 4096) in cache/saes_p3i.
Writes results/phase3/dsweep_summary.{json,png}.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "phase3"
K = [4, 8, 16, 32, 64]
MIN_ALIVE = 250
GEMMA_LOG2_K0 = 1.016 / 0.094
ARMS = {"scaled": lambda d, w: w == 16 * d, "fixed": lambda d, w: w == 4096}


def windows(d):
    return {"fixed": (5, 50), "scaled": (max(1, round(d / 43)), round(d / 5))}


def main():
    dirs = sorted(glob.glob(str(ROOT / "cache" / "saes_p3d" / "*" / "train.json")))
    dirs += sorted(glob.glob(str(ROOT / "cache" / "saes_p3i" / "synth-i-m2048-a16__*" / "train.json")))
    sig, rows = {}, []
    for tj in dirs:
        rd = Path(tj).parent
        c = json.load(open(tj))["config"]
        if c["k"] not in K:
            continue
        act = ROOT / "cache" / "acts" / c["model"] / c["hook"]
        if c["model"] not in sig:
            sig[c["model"]] = np.sort(np.linalg.eigvalsh(np.load(act / "stats.npz")["cov"]))[::-1]
        e_sig = sig[c["model"]]
        d = len(e_sig)
        fires = np.load(rd / "fires.npy")
        alive = int((fires > 0).sum())
        row = {"model": c["model"], "d": d, "width": c["width"], "k": c["k"], "cond": c["cond"], "alive": alive, "gamma": {}}
        if alive >= MIN_ALIVE:
            e = eigen_slope(np.load(rd / "W_dec.npy"), mask=fires > 0)["eigenvalues"]
            for wn, w in windows(d).items():
                if w[1] <= d:
                    row["gamma"][wn] = slope_from_eigenvalues(e, w)["slope"] / slope_from_eigenvalues(e_sig, w)["slope"]
        rows.append(row)

    fits = {}
    configs = sorted({(r["d"], r["width"]) for r in rows})
    print(f"{'cond':<9} {'d':>5} {'width':>6} {'win':<7} " + " ".join(f"k={k:<5}" for k in K) + "  A        s       R2   log2 k0")
    for cond in ["real", "gaussian"]:
        for d, w in configs:
            xs = {r["k"]: r for r in rows if r["cond"] == cond and r["d"] == d and r["width"] == w}
            for wn in ["fixed", "scaled"]:
                g = {k: xs[k]["gamma"].get(wn) for k in xs}
                ok = [k for k in K if g.get(k) is not None]
                fit = linregress(np.log2(ok), [g[k] for k in ok]) if len(ok) >= 3 else None
                k0 = fit.intercept / -fit.slope if fit and fit.slope < 0 else None
                fits[f"{cond}|{d}|{w}|{wn}"] = {"gamma": g, "alive": {k: xs[k]["alive"] for k in xs}, "ks_fit": ok,
                                                "window": windows(d)[wn],
                                                "intercept": fit.intercept if fit else None,
                                                "per_doubling": fit.slope if fit else None,
                                                "r2": fit.rvalue ** 2 if fit else None, "log2_k0": k0}
                cells = " ".join(f"{g[k]:<7.3f}" if g.get(k) is not None else "   -   " for k in K)
                fs = f"{fit.intercept:6.3f}  {fit.slope:+.3f}  {fit.rvalue ** 2:.2f}" if fit else " " * 21
                print(f"{cond:<9} {d:>5} {w:>6} {wn:<7} {cells}  {fs}  {k0 if k0 is None else round(k0, 2)}")
    json.dump({"K": K, "min_alive": MIN_ALIVE, "gemma_log2_k0": GEMMA_LOG2_K0, "rows": rows, "fits": fits},
              open(OUT / "dsweep_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, wn in zip(ax, ["fixed", "scaled"]):
        for arm, mk in [("scaled", "o"), ("fixed", "s")]:
            for cond, ls in [("real", "-"), ("gaussian", "--")]:
                pts = [(d, fits[f"{cond}|{d}|{w}|{wn}"]["log2_k0"]) for d, w in configs if ARMS[arm](d, w)
                       and fits.get(f"{cond}|{d}|{w}|{wn}", {}).get("log2_k0") is not None]
                if pts:
                    a.plot(np.log2([p[0] for p in pts]), [p[1] for p in pts], mk + ls, label=f"{arm} width, {cond}")
        a.plot(np.log2(2304), GEMMA_LOG2_K0, "k*", ms=12, label="Gemma L13 TopK")
        a.set_xlabel("log₂ d")
        a.set_title(f"{wn} rank window")
    ax[0].set_ylabel("log₂ k₀ (fitted γ = 0)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "dsweep_summary.png", dpi=150)


if __name__ == "__main__":
    main()
