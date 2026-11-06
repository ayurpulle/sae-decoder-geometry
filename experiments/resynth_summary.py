"""Summarise the resynthesis dissection (resynth_job.py) with its pre-registered reading.

Reads results/phase3/resynth_match.json. Per variant (R real layer 12, A resynthesis, B independent firing, C permuted
directions, D both) and window: gamma(k) (seeds averaged), straight-line fit of gamma on log2 k (6 k values), RMS vs R.
Gate (ranks 11-150): s_R <= -0.05; RMS(A - R) <= 0.15 and |s_A - s_R| <= 0.04. Then retained = s_X / s_A for B, C, D:
< 0.5 removes the law, > 0.75 keeps it, partial between. Also the decoders' cluster excess over a covariance-matched
cloud, and each dataset's Sigma slopes. Gemma L13 (results/gpu/analysis.json B) shown for context.
Writes results/phase3/resynth_summary.{json,png}.
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

P3 = ROOT / "results" / "phase3"
WINDOWS = {"(3, 50)": "ranks 4–50", "(10, 150)": "ranks 11–150", "(30, 450)": "ranks 31–450"}
K = [4, 8, 16, 32, 64, 128]
NAMES = {"R": "gemma-2-2b", "A": "rs12-A", "B": "rs12-B", "C": "rs12-C", "D": "rs12-D"}
LABEL = {"R": "R real L12", "A": "A resynthesis", "B": "B independent firing", "C": "C permuted directions",
         "D": "D both"}
MAIN = "(10, 150)"


def main():
    m = json.load(open(P3 / "resynth_match.json"))
    gem = json.load(open(ROOT / "results" / "gpu" / "analysis.json"))["B"]
    out = {"variants": {}, "gemma_l13": {w: {int(k): v[0] for k, v in gem[f"real|{w}"].items()} for w in WINDOWS}}
    for v, n in NAMES.items():
        rows = [r for r in m["rows"] if r["data"] == n and r["cond"] == "real"]
        if not rows:
            continue
        sk = f"synth-{n}" if v != "R" else n
        e = np.array(m["sigma_eigenvalues"][sk])
        d = {"seeds": sorted({r["seed"] for r in rows}), "windows": {},
             "sigma_slope": {w: slope_from_eigenvalues(e, tuple(map(int, w.strip("()").split(","))))["slope"]
                             for w in WINDOWS},
             "fvu": {k: float(np.mean([r["fvu"] for r in rows if r["k"] == k])) for k in K},
             "alive": {k: float(np.mean([r["alive"] for r in rows if r["k"] == k])) for k in K}}
        if all("clusters" in r for r in rows):
            d["excess"] = {k: {s: float(np.mean([r["clusters"][f"excess_{s}"] for r in rows if r["k"] == k]))
                               for s in ("share", "cos")} for k in K}
        for w in WINDOWS:
            ps = {s: {r["k"]: r["gamma"][w] for r in rows if r["seed"] == s and w in r["gamma"]} for s in d["seeds"]}
            if not all(len(p) == len(K) for p in ps.values()):
                continue
            g = {k: float(np.mean([p[k] for p in ps.values()])) for k in K}
            lr = linregress(np.log2(K), [g[k] for k in K])
            fits = [linregress(np.log2(K), [p[k] for k in K]).slope for p in ps.values()]
            d["windows"][w] = {"gamma": g, "per_seed": ps, "fit": {"intercept": lr.intercept, "per_doubling": lr.slope,
                                                                   "r2": lr.rvalue ** 2, "per_seed_slopes": fits}}
        out["variants"][v] = d
    V = out["variants"]
    for v, d in V.items():
        for w, x in d["windows"].items():
            if "R" in V and w in V["R"]["windows"]:
                r = V["R"]["windows"][w]["gamma"]
                x["rms_vs_R"] = float(np.sqrt(np.mean([(x["gamma"][k] - r[k]) ** 2 for k in K])))
            if "A" in V and w in V["A"]["windows"]:
                x["retained_vs_A"] = x["fit"]["per_doubling"] / V["A"]["windows"][w]["fit"]["per_doubling"]
    verdict = {}
    if "R" in V and "A" in V:
        sR, sA = (V[v]["windows"][MAIN]["fit"]["per_doubling"] for v in "RA")
        verdict["law_in_R"] = sR <= -0.05
        verdict["A_reproduces_R"] = V["A"]["windows"][MAIN]["rms_vs_R"] <= 0.15 and abs(sA - sR) <= 0.04
        verdict["gate"] = verdict["law_in_R"] and verdict["A_reproduces_R"]
        for v in "BCD":
            if v in V and MAIN in V[v]["windows"]:
                r = V[v]["windows"][MAIN]["retained_vs_A"]
                verdict[v] = "removes" if r < 0.5 else ("keeps" if r > 0.75 else "partial")
    out["verdict"] = verdict

    for w, wl in WINDOWS.items():
        print(f"\n== {wl}   Gemma L13: " + " ".join(f"{k}:{g:.3f}" for k, g in out["gemma_l13"][w].items()))
        for v, d in V.items():
            if w not in d["windows"]:
                continue
            x = d["windows"][w]
            print(f"{LABEL[v]:<24} s{len(d['seeds'])} Σ {d['sigma_slope'][w]:.2f}  "
                  + " ".join(f"{x['gamma'][k]:.3f}" for k in K)
                  + f"  fit {x['fit']['intercept']:.3f} {x['fit']['per_doubling']:+.3f}"
                  + (f" (seeds {', '.join(f'{s:+.3f}' for s in x['fit']['per_seed_slopes'])})" if len(d['seeds']) > 1 else "")
                  + (f"  RMS vs R {x['rms_vs_R']:.3f}" if "rms_vs_R" in x else "")
                  + (f"  retained {x['retained_vs_A']:.2f}" if "retained_vs_A" in x and v != "A" else ""))
    print("\ndecoder cluster excess (share / cos); Gemma Scope L13 16k +0.054 / +0.079")
    for v, d in V.items():
        if "excess" in d:
            print(f"{LABEL[v]:<24} " + " ".join(f"k{k}:{d['excess'][k]['share']:+.3f}/{d['excess'][k]['cos']:+.3f}" for k in K))
    print("\nFVU", {v: {k: round(f, 3) for k, f in d["fvu"].items()} for v, d in V.items()})
    print("\nverdict:", verdict)
    json.dump(out, open(P3 / "resynth_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    col = {"R": "k", "A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
    for a, (w, wl) in zip(ax[:3], WINDOWS.items()):
        for v, d in V.items():
            if w in d["windows"]:
                g = d["windows"][w]["gamma"]
                a.plot(np.log2(K), [g[k] for k in K], "-o", ms=3, color=col[v], lw=2 if v in "RA" else 1.2,
                       label=LABEL[v] + (f" ({len(d['seeds'])} seeds)" if len(d["seeds"]) > 1 else ""))
        gl = out["gemma_l13"][w]
        a.plot(np.log2(list(gl)), list(gl.values()), "*:", color="0.5", ms=7, label="our TopK on real L13 (Phase 2)")
        a.set_title(f"{wl}: γ(k)", fontsize=10)
        a.set_xlabel("log₂ k")
    ax[0].set_ylabel("γ = decoder slope / Σ slope")
    ax[0].legend(fontsize=6)
    a = ax[3]
    for v, d in V.items():
        if "excess" in d:
            a.plot(np.log2(K), [d["excess"][k]["share"] for k in K], "-s", ms=3, color=col[v], label=LABEL[v])
    a.axhline(0, color="0.6", lw=0.6)
    a.set_title("decoders: excess between-cluster share\nover covariance-matched cloud", fontsize=10)
    a.set_xlabel("log₂ k")
    a.legend(fontsize=6)
    fig.suptitle("Resynthesis dissection, Gemma-2-2B layer 12 (Gemma Scope 65k codes), TopK width 16384")
    fig.tight_layout()
    fig.savefig(P3 / "resynth_summary.png", dpi=150)


if __name__ == "__main__":
    main()
