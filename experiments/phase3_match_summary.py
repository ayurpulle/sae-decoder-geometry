"""Summarise the Gemma-matched planted check (d = 2304, width 16384): planted gamma(k) per density, condition and
window against Gemma L13 TopK 16k (results/gpu/analysis.json, section B, 2 seeds).

Reads results/phase3/{--inp}.json (phase3_match_analyze.py on the pod; default gemma_match, or powerlaw_match with
--prefix p2304 for the power-law-Sigma generator). Seeds are averaged where there are
several (stage 2 density). Distance to Gemma = RMS of gamma differences over k in {4, 8, 16, 40, 128}, with the planted
gamma at k = 40 interpolated linearly in log2 k between k = 32 and 64. Also refits gamma(k) on the mostly-alive runs
(k >= 32, >= 10k of 16384 latents alive in every real run). Also records Sigma's eigen-slope per window: with
P_MAX = 0.5 the top 25-101 features saturate at d = 2304, Sigma is flat over the top ranks and gamma there is a ratio
over a near-zero denominator; the d-scaled window (ranks 55-461) is the clean one. Writes results/phase3/{--out}.{json,png} (default match_summary).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "phase3"
WINDOWS = {"(3, 50)": "ranks 4–50", "(10, 150)": "ranks 11–150", "(30, 450)": "ranks 31–450",
           "(54, 461)": "ranks 55–461 (d-scaled)"}
GEMMA_W = {"(54, 461)": "(30, 450)"}  # Gemma was not analysed in the d-scaled window; compare with its nearest
K = [4, 8, 16, 32, 64, 128]
GEMMA_K = [4, 8, 16, 40, 128]
SUFFIX = {"m4608-a36": "2d feat, d/64 active", "m9216-a72": "4d feat, d/32 active", "m18432-a144": "8d feat, d/16 active"}


def at(g, k):
    if k in g:
        return g[k]
    lo, hi = 32, 64  # only k = 40 is off-grid
    t = (np.log2(k) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
    return (1 - t) * g[lo] + t * g[hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="gemma_match")
    ap.add_argument("--out", default="match_summary")
    ap.add_argument("--prefix", default="g2304", help="g2304 (default generator) or p2304 (--var_slope 1)")
    ap.add_argument("--title", default="Planted point features at Gemma's d = 2304, width 16384 vs Gemma L13 TopK 16k")
    args = ap.parse_args()
    DENS = [f"{args.prefix}-{x}" for x in SUFFIX]
    LABEL = {f"{args.prefix}-{x}": v for x, v in SUFFIX.items()}
    m = json.load(open(OUT / f"{args.inp}.json"))
    gem = json.load(open(ROOT / "results" / "gpu" / "analysis.json"))["B"]
    out = {"gemma": {}, "planted": {}, "sigma_slope": {}}
    sig_g = json.load(open(ROOT / "results" / "gpu" / "sigma.json"))["blocks.13.hook_resid_post"]["eigenvalues"]
    for w in WINDOWS:
        wt = tuple(map(int, w.strip("()").split(",")))
        out["sigma_slope"][f"gemma|{w}"] = slope_from_eigenvalues(np.array(sig_g), wt)["slope"]
        for n, e in m["sigma_eigenvalues"].items():
            out["sigma_slope"][f"{n.removeprefix('synth-')}|{w}"] = slope_from_eigenvalues(np.array(e), wt)["slope"]
    for w in WINDOWS:
        if w in GEMMA_W:
            continue
        for cond in ["real", "gaussian"]:
            out["gemma"][f"{cond}|{w}"] = {int(k): v[0] for k, v in gem[f"{cond}|{w}"].items()}

    print(f"{'data':<18} {'cond':<9} {'window':<10} " + " ".join(f"k={k:<11}" for k in K) + " RMS-vs-Gemma  A(k>=32)  s(k>=32)")
    for w in WINDOWS:
        g_real = out["gemma"][f"real|{GEMMA_W.get(w, w)}"]
        for n in DENS:
            for cond in ["real", "gaussian"]:
                rs = [r for r in m["rows"] if r["data"] == n and r["cond"] == cond]
                if not rs:
                    continue
                gam, sd, alive = {}, {}, {}
                for k in K:
                    v = [r["gamma"][w] for r in rs if r["k"] == k and w in r["gamma"]]
                    alive[k] = [r["alive"] for r in rs if r["k"] == k]
                    if v:
                        gam[k], sd[k] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None
                ok = all(k in gam for k in (8, 16, 32, 64))  # interpolation and grid need these
                rms = float(np.sqrt(np.mean([(at(gam, k) - g_real[k]) ** 2 for k in GEMMA_K if k in g_real]))) \
                    if ok and 4 in gam and 128 in gam else None
                hi = [k for k in (32, 64, 128) if k in gam]
                lr = linregress(np.log2(hi), [gam[k] for k in hi]) if len(hi) == 3 else None
                out["planted"][f"{n}|{cond}|{w}"] = {
                    "gamma": gam, "sd": sd, "seeds": len({r["seed"] for r in rs}), "alive": alive,
                    "rms_vs_gemma_real": rms, "fit_k32plus": None if lr is None else
                    {"intercept": lr.intercept, "per_doubling": lr.slope}}
                cells = " ".join((f"{gam[k]:.3f}" + (f"±{sd[k]:.3f}" if sd.get(k) else "      ")).ljust(13)
                                 if k in gam else "-".ljust(13) for k in K)
                print(f"{n:<18} {cond:<9} {w:<10} {cells} {rms if rms is None else round(rms, 3)!s:<13} "
                      + (f"{lr.intercept:6.3f}  {lr.slope:+.3f}" if lr else ""))
        print(f"{'Gemma L13 TopK':<18} {'real':<9} {GEMMA_W.get(w, w):<10} "
              + " ".join(f"k={k}:{g_real[k]:.3f}" for k in GEMMA_K))
    json.dump(out, open(OUT / f"{args.out}.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    cols = dict(zip(DENS, plt.cm.viridis([0.1, 0.5, 0.85])))
    for a, (w, wl) in zip(ax, WINDOWS.items()):
        for n in DENS:
            for cond, ls in [("real", "-"), ("gaussian", "--")]:
                if f"{n}|{cond}|{w}" not in out["planted"]:
                    continue
                g = out["planted"][f"{n}|{cond}|{w}"]["gamma"]
                ks = sorted(g)
                a.plot(np.log2(ks), [g[k] for k in ks], ls, marker="o", ms=3, color=cols[n],
                       label=f"{LABEL[n]}, {cond}")
        for cond, mk, c in [("real", "*-", "k"), ("gaussian", "x--", "0.5")]:
            g = out["gemma"][f"{cond}|{GEMMA_W.get(w, w)}"]
            ks = sorted(g)
            a.plot(np.log2(ks), [g[k] for k in ks], mk, color=c, ms=8, lw=1.2, label=f"Gemma L13 TopK, {cond}")
        ss = ", ".join(f"{out['sigma_slope'][f'{n}|{w}']:.2f}" for n in DENS)
        a.set_title(f"{wl}\nΣ slope planted {ss}; Gemma {out['sigma_slope'][f'gemma|{GEMMA_W.get(w, w)}']:.2f}"
                    + (" (ranks 31–450)" if w in GEMMA_W else ""), fontsize=9)
        a.set_xlabel("log₂ k")
    ax[0].set_ylabel("γ = decoder slope / Σ slope")
    ax[0].legend(fontsize=6)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(OUT / f"{args.out}.png", dpi=150)


if __name__ == "__main__":
    main()
