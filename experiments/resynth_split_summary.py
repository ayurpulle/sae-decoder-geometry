"""Summarise the split test (resynth_split_job.py) against the resynthesis references, with its pre-registered reading.

Reads results/phase3/resynth_match.json (R, A, B seed 0) and resynth_split.json (E, F, B seed 1). Per variant and window:
gamma(k) (seeds averaged), per-doubling slope over all k and over k 8-128 (primary), retained = s_X / s_A on the same k
set, per-seed slopes, Sigma slope, decoder cluster excess. Verdict per resynth_split_job.py's docstring.
Writes results/phase3/resynth_split_summary.{json,png}.
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
K8 = K[1:]
MAIN = "(10, 150)"
NAMES = {"R": "gemma-2-2b", "A": "rs12-A", "B": "rs12-B", "E": "rs12-E", "F": "rs12-F"}
LABEL = {"R": "R real L12", "A": "A resynthesis", "B": "B Gemma marginals + Gemma dirs",
         "E": "E Gemma marginals + random dirs", "F": "F uniform marginals + Gemma dirs"}


def fit(g, ks):
    return linregress(np.log2(ks), [g[k] for k in ks]).slope


def main():
    files = [json.load(open(P3 / f)) for f in ("resynth_match.json", "resynth_split.json") if (P3 / f).exists()]
    rows = [r for m in files for r in m["rows"] if r["cond"] == "real"]
    sig = {k: v for m in files for k, v in m["sigma_eigenvalues"].items()}
    out = {"variants": {}}
    for v, n in NAMES.items():
        rs = [r for r in rows if r["data"] == n]
        if not rs:
            continue
        e = np.array(sig[n if v == "R" else f"synth-{n}"])
        d = {"seeds": sorted({r["seed"] for r in rs}), "windows": {},
             "sigma_slope": {w: slope_from_eigenvalues(e, tuple(map(int, w.strip("()").split(","))))["slope"] for w in WINDOWS},
             "alive": {k: float(np.mean([r["alive"] for r in rs if r["k"] == k])) for k in K},
             "fvu": {k: float(np.mean([r["fvu"] for r in rs if r["k"] == k])) for k in K},
             "excess_share": {k: float(np.mean([r["clusters"]["excess_share"] for r in rs if r["k"] == k])) for k in K}}
        for w in WINDOWS:
            ps = {s: {r["k"]: r["gamma"][w] for r in rs if r["seed"] == s and w in r["gamma"]} for s in d["seeds"]}
            ps = {s: p for s, p in ps.items() if len(p) == len(K)}
            if not ps:
                continue
            g = {k: float(np.mean([p[k] for p in ps.values()])) for k in K}
            d["windows"][w] = {"gamma": g, "per_seed": ps, "s_all": fit(g, K), "s_k8": fit(g, K8),
                               "per_seed_s_all": [fit(p, K) for p in ps.values()],
                               "per_seed_s_k8": [fit(p, K8) for p in ps.values()]}
        out["variants"][v] = d
    V = out["variants"]
    for v, d in V.items():
        for w, x in d["windows"].items():
            a = V["A"]["windows"][w]
            x["retained_all"] = x["s_all"] / a["s_all"]
            x["retained_k8"] = x["s_k8"] / a["s_k8"]
    call = lambda r: "removes" if r < 0.5 else ("keeps" if r > 0.75 else "partial")  # noqa: E731
    verdict = {}
    for v in "BEF":
        if v in V and MAIN in V[v]["windows"]:
            x = V[v]["windows"][MAIN]
            verdict[v] = {"primary_k8": call(x["retained_k8"]), "all_k": call(x["retained_all"]),
                          "sigma_ill_conditioned": V[v]["sigma_slope"][MAIN] > -0.3}
    if "E" in verdict and "F" in verdict:
        e, f = verdict["E"]["primary_k8"] == "keeps", verdict["F"]["primary_k8"] == "keeps"
        er, fr = verdict["E"]["primary_k8"] == "removes", verdict["F"]["primary_k8"] == "removes"
        verdict["reading"] = ("marginals carry the law" if e and fr else "dictionary geometry carries the law" if er and f
                              else "either suffices" if e and f else "needs both" if er and fr else "mixed / partial")
    out["verdict"] = verdict

    for w, wl in WINDOWS.items():
        print(f"\n== {wl}")
        for v, d in V.items():
            if w not in d["windows"]:
                continue
            x = d["windows"][w]
            print(f"{LABEL[v]:<34} s{','.join(map(str, d['seeds']))} Σ {d['sigma_slope'][w]:+.2f}  "
                  + " ".join(f"{x['gamma'][k]:.3f}" for k in K)
                  + f"  s_all {x['s_all']:+.3f} ({x['retained_all']:.2f})  s_k8 {x['s_k8']:+.3f} ({x['retained_k8']:.2f})"
                  + (f"  seeds k8 {', '.join(f'{s:+.3f}' for s in x['per_seed_s_k8'])}" if len(x["per_seed_s_k8"]) > 1 else ""))
    print("\nalive", {v: [int(d["alive"][k]) for k in K] for v, d in V.items()})
    print("cluster excess share", {v: [round(d["excess_share"][k], 3) for k in K] for v, d in V.items()})
    print("\nverdict:", json.dumps(verdict, default=str))
    json.dump(out, open(P3 / "resynth_split_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    col = {"R": "k", "A": "tab:blue", "B": "tab:orange", "E": "tab:purple", "F": "tab:olive"}
    for a, (w, wl) in zip(ax, WINDOWS.items()):
        for v, d in V.items():
            if w in d["windows"]:
                g = d["windows"][w]["gamma"]
                a.plot(np.log2(K), [g[k] for k in K], "-o", ms=3, color=col[v], lw=2 if v in "RA" else 1.4,
                       label=f"{LABEL[v]} ({len(d['seeds'])} seed{'s' if len(d['seeds']) > 1 else ''}; Σ {d['sigma_slope'][w]:+.2f})")
        a.set_title(f"{wl}: γ(k)", fontsize=10)
        a.set_xlabel("log₂ k")
    ax[0].set_ylabel("γ = decoder slope / Σ slope")
    ax[1].legend(fontsize=7)
    fig.suptitle("Split test: Gemma Scope L12 marginals vs dictionary directions, TopK width 16384")
    fig.tight_layout()
    fig.savefig(P3 / "resynth_split_summary.png", dpi=150)


if __name__ == "__main__":
    main()
