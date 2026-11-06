"""Summarise the clustered-planted-directions check (phase3_clustered.py) against Gemma L13 TopK 16k and the rho = 0
control (power-law-Sigma run, results/phase3/powerlaw_match.json, same generator without clusters).

Per dataset (real SAEs, seeds averaged) and window: gamma(k), RMS vs Gemma's real gamma over k in {4, 8, 16, 40, 128}
(k = 40 interpolated in log2 k), and a straight-line fit of gamma on log2 k over all six k (Gemma, ranks 11-150:
1.016 - 0.094 log2 k). Also the trained decoders' own cluster excess over a covariance-matched cloud (K = width/16;
Gemma L13 16k: share +0.054, cos +0.079) and the Sigma slopes.
Pre-registered reading (phase3_clustered.py): clusters restore Gemma's law if real gamma in ranks 11-150 has RMS vs
Gemma < 0.15 and per-doubling slope within 0.04 of -0.094.
Writes results/phase3/clustered_summary.{json,png}.
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

OUT = ROOT / "results" / "phase3"
WINDOWS = {"(3, 50)": "ranks 4–50", "(10, 150)": "ranks 11–150", "(30, 450)": "ranks 31–450"}
K = [4, 8, 16, 32, 64, 128]
GEMMA_K = [4, 8, 16, 40, 128]
GEMMA_EXCESS = {"share": 0.054, "cos": 0.079}
DATA = {  # name -> (source file, density label, rho)
    "p2304-m9216-a72": ("powerlaw_match", "4d feat, d/32", 0.0),
    "c2304-m9216-a72-r40": ("clustered_match", "4d feat, d/32", 0.4),
    "c2304-m9216-a72-r60": ("clustered_match", "4d feat, d/32", 0.6),
    "p2304-m18432-a144": ("powerlaw_match", "8d feat, d/16", 0.0),
    "c2304-m18432-a144-r40": ("clustered_match", "8d feat, d/16", 0.4),
    "c2304-m18432-a144-r60": ("clustered_match", "8d feat, d/16", 0.6),
}


def at(g, k):
    if k in g:
        return g[k]
    t = (np.log2(k) - 5) / 1  # k = 40 between 32 and 64
    return (1 - t) * g[32] + t * g[64]


def main():
    src = {f: json.load(open(OUT / f"{f}.json")) for f in {v[0] for v in DATA.values()}}
    gem = json.load(open(ROOT / "results" / "gpu" / "analysis.json"))["B"]
    sig_g = np.array(json.load(open(ROOT / "results" / "gpu" / "sigma.json"))["blocks.13.hook_resid_post"]["eigenvalues"])
    out = {"gemma": {w: {int(k): v[0] for k, v in gem[f"real|{w}"].items()} for w in WINDOWS},
           "gemma_sigma_slope": {w: slope_from_eigenvalues(sig_g, tuple(map(int, w.strip("()").split(","))))["slope"]
                                 for w in WINDOWS},
           "gemma_decoder_excess": GEMMA_EXCESS, "data": {}}
    for w in WINDOWS:
        g = out["gemma"][w]
        lr = linregress(np.log2(GEMMA_K), [g[k] for k in GEMMA_K])
        out["gemma"][f"fit|{w}"] = {"intercept": lr.intercept, "per_doubling": lr.slope}
    for n, (f, lab, rho) in DATA.items():
        m = src[f]
        rows = [r for r in m["rows"] if r["data"] == n and r["cond"] == "real"]
        e = np.array(m["sigma_eigenvalues"][f"synth-{n}"])
        d = {"label": lab, "rho": rho, "seeds": sorted({r["seed"] for r in rows}), "windows": {},
             "sigma_slope": {w: slope_from_eigenvalues(e, tuple(map(int, w.strip("()").split(","))))["slope"]
                             for w in WINDOWS},
             "fvu": {k: float(np.mean([r["fvu"] for r in rows if r["k"] == k])) for k in K},
             "alive": {k: float(np.mean([r["alive"] for r in rows if r["k"] == k])) for k in K}}
        if all("clusters" in r for r in rows):
            d["excess"] = {k: {s: float(np.mean([r["clusters"][f"excess_{s}"] for r in rows if r["k"] == k]))
                               for s in ("share", "cos")} for k in K}
        for w in WINDOWS:
            per_seed = {s: {r["k"]: r["gamma"][w] for r in rows if r["seed"] == s} for s in d["seeds"]}
            gam = {k: float(np.mean([p[k] for p in per_seed.values()])) for k in K}
            lr = linregress(np.log2(K), [gam[k] for k in K])
            d["windows"][w] = {
                "gamma": gam, "per_seed": per_seed,
                "rms_vs_gemma": float(np.sqrt(np.mean([(at(gam, k) - out["gemma"][w][k]) ** 2 for k in GEMMA_K]))),
                "fit": {"intercept": lr.intercept, "per_doubling": lr.slope, "r2": lr.rvalue ** 2}}
            if len(per_seed) > 1:
                d["windows"][w]["seed_sd"] = {k: float(np.std([p[k] for p in per_seed.values()], ddof=1)) for k in K}
        w0 = d["windows"]["(10, 150)"]
        d["passes_preregistered"] = bool(w0["rms_vs_gemma"] < 0.15 and abs(w0["fit"]["per_doubling"] + 0.094) <= 0.04)
        out["data"][n] = d

    for w, wl in WINDOWS.items():
        gf = out["gemma"][f"fit|{w}"]
        print(f"\n== {wl}   Gemma: " + " ".join(f"k{k}:{out['gemma'][w][k]:.3f}" for k in GEMMA_K)
              + f"   fit {gf['intercept']:.3f} {gf['per_doubling']:+.3f}/doubling   Σ slope {out['gemma_sigma_slope'][w]:.2f}")
        for n, d in out["data"].items():
            x = d["windows"][w]
            print(f"{n:<22} rho {d['rho']:.1f} s{len(d['seeds'])} Σ {d['sigma_slope'][w]:.2f}  "
                  + " ".join(f"{x['gamma'][k]:.3f}" for k in K)
                  + f"  RMS {x['rms_vs_gemma']:.3f}  fit {x['fit']['intercept']:.3f} {x['fit']['per_doubling']:+.3f}"
                  + (f"  seed sd max {max(x['seed_sd'].values()):.3f}" if "seed_sd" in x else ""))
    print("\ndecoder cluster excess over matched cloud (share / cos), Gemma +0.054 / +0.079")
    for n, d in out["data"].items():
        if "excess" in d:
            print(f"{n:<22} " + " ".join(f"k{k}:{d['excess'][k]['share']:+.3f}/{d['excess'][k]['cos']:+.3f}" for k in K))
    print("\npasses pre-registered criterion:", {n: d["passes_preregistered"] for n, d in out["data"].items()})
    json.dump(out, open(OUT / "clustered_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    sty = {0.0: ":", 0.4: "--", 0.6: "-"}
    col = {"4d feat, d/32": "tab:blue", "8d feat, d/16": "tab:green"}
    lk = np.log2(K)
    for a, (w, wl) in zip(ax[:3], WINDOWS.items()):
        for n, d in out["data"].items():
            a.plot(lk, [d["windows"][w]["gamma"][k] for k in K], sty[d["rho"]], marker="o", ms=3,
                   color=col[d["label"]], label=f"{d['label']}, ρ={d['rho']}" + (f" ({len(d['seeds'])} seeds)" if len(d['seeds']) > 1 else ""))
        a.plot(np.log2(GEMMA_K), [out["gemma"][w][k] for k in GEMMA_K], "k*-", ms=9, label="Gemma L13 TopK 16k")
        a.set_title(f"{wl}: real γ(k)", fontsize=10)
        a.set_xlabel("log₂ k")
    ax[0].set_ylabel("γ = decoder slope / Σ slope")
    ax[0].legend(fontsize=6)
    a = ax[3]
    for n, d in out["data"].items():
        if "excess" in d:
            a.plot(lk, [d["excess"][k]["share"] for k in K], sty[d["rho"]], marker="s", ms=3, color=col[d["label"]],
                   label=f"{d['label']}, ρ={d['rho']}")
    a.axhline(GEMMA_EXCESS["share"], color="k", lw=1, label="Gemma L13 16k (+0.054)")
    a.axhline(0, color="0.6", lw=0.6)
    a.set_title("trained decoders: excess between-cluster share\nover covariance-matched cloud (K = 1024)", fontsize=10)
    a.set_xlabel("log₂ k")
    a.legend(fontsize=6)
    fig.suptitle("Clustered planted directions (power-law Σ, d = 2304, width 16384) vs Gemma; ρ = 0 is the no-cluster control")
    fig.tight_layout()
    fig.savefig(OUT / "clustered_summary.png", dpi=150)


if __name__ == "__main__":
    main()
