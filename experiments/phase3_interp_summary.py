"""Summarise the dense-to-sparse interpolation: gamma(k) fit per dataset and condition on a common k set, placed in
the (intercept, per-doubling slope) plane next to Gemma's TopK fit at layer 13 (1.016, -0.094).

Reads results/phase3/{saes_p3, saes_p3s2, saes_p3i}.json (phase3_analyze.py output); fits gamma (primary window,
ranks 6-50) on log2 k over K, runs with >= 1000 alive latents. Writes results/phase3/interp_summary.{json,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3"
K = [4, 8, 16, 32, 64]
PRIMARY = "(5, 50)"
ORDER = ["s2-point", "i-m1024-a8", "i-m2048-a16", "i-m4096-a32", "base"]
GEMMA = (1.016, -0.094)


def main():
    rows = []
    for g in ["saes_p3", "saes_p3s2", "saes_p3i"]:
        p = OUT / f"{g}.json"
        if p.exists():
            rows += json.load(open(p))["rows"]
    out = {}
    print(f"{'data':<12} {'cond':<9} " + " ".join(f"k={k:<5}" for k in K) + "  intercept  per-doubling  R2   matched(k=8)")
    for data in ORDER:
        for cond in ["real", "gaussian"]:
            xs = {r["k"]: r for r in rows if r["data"] == data and r["cond"] == cond and r["width"] == 4096 and r["k"] in K}
            if not xs:
                continue
            g = {k: xs[k]["gamma"].get(PRIMARY) for k in K if k in xs}
            ok = [k for k in K if k in xs and xs[k]["alive"] >= 1000 and g[k] is not None]
            fit = linregress(np.log2(ok), [g[k] for k in ok]) if len(ok) >= 3 else None
            m8 = xs.get(8, {}).get("match", {}).get("frac_matched")
            out[f"{data}|{cond}"] = {"gamma": g, "ks_fit": ok, "intercept": fit.intercept if fit else None,
                                     "per_doubling": fit.slope if fit else None, "r2": fit.rvalue ** 2 if fit else None,
                                     "frac_matched_k8": m8, "fvu": {k: xs[k]["fvu"] for k in xs}}
            cells = " ".join(f"{g[k]:<7.3f}" if g.get(k) is not None else "   -   " for k in K)
            fs = f"{fit.intercept:9.3f}  {fit.slope:+12.3f}  {fit.rvalue ** 2:.2f}" if fit else " " * 30
            print(f"{data:<12} {cond:<9} {cells}  {fs}   {m8 if m8 is None else round(m8, 2)}")
    json.dump({"K": K, "window": PRIMARY, "gemma_topk_L13": GEMMA, "fits": out},
              open(OUT / "interp_summary.json", "w"), indent=1, default=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cond, mk in [("real", "o-"), ("gaussian", "s--")]:
        pts = [(d, out[f"{d}|{cond}"]) for d in ORDER[:4] if f"{d}|{cond}" in out and out[f"{d}|{cond}"]["intercept"] is not None]
        if pts:
            ax.plot([p["intercept"] for _, p in pts], [p["per_doubling"] for _, p in pts], mk, label=f"planted, β = 0, {cond}")
            for d, p in pts:
                ax.annotate(d.replace("i-", "").replace("s2-point", "m512-a4"), (p["intercept"], p["per_doubling"]), fontsize=7)
    for cond, mk in [("real", "^"), ("gaussian", "v")]:
        b = out.get(f"base|{cond}")
        if b and b["intercept"] is not None:
            ax.plot(b["intercept"], b["per_doubling"], mk, ms=8, label=f"stage 1 (β = 1), {cond}")
    ax.plot(*GEMMA, "k*", ms=14, label="Gemma L13, TopK")
    ax.set_xlabel("γ intercept (k = 1)")
    ax.set_ylabel("γ change per doubling of k")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "interp_summary.png", dpi=150)


if __name__ == "__main__":
    main()
