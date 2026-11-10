"""Summary and pre-registered verdict for the matched-covariance control (see control_job.py's docstring).
Reads results/phase3/control.json (and the earlier resynth_match.json / resynth_split.json for the reproduction check);
writes results/phase3/control_summary.{json,png}.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results" / "phase3"
W = "(10, 150)"
KS = [8, 16, 32, 64, 128]
KEEP, REMOVE, ILL = 0.75, 0.5, -0.3
PAIRS = [("G", "D"), ("H", "F")]
LABEL = {"D": "D: Gemma marginals, Gemma rows permuted", "G": "G: Gemma marginals, Gaussian-matched directions",
         "F": "F: uniform marginals, Gemma rows", "H": "H: uniform marginals, Gaussian-matched directions"}
EARLIER = {"D": ("resynth_match.json", "rs12-D|real|0|"), "F": ("resynth_split.json", "rs12-F|real|0|")}


def per_doubling(ys):
    return float(np.polyfit(np.log2(KS), ys, 1)[0])


def verdict(rg, rr):
    if rg >= KEEP and rr >= KEEP:
        return "keeps"
    if rg < REMOVE and rr < REMOVE:
        return "removes"
    return "partial"


def main():
    d = json.load(open(P3 / "control.json"))
    out = {"window": W, "k": KS, "n_fix": d["n_fix"], "variants": {}, "checks": {}, "verdict": {}}
    for v in "DGFH":
        rows = {r["k"]: r for r in d["rows"] if r["data"] == f"rs12-{v}" and r["seed"] == 0}
        if sorted(rows) != KS:
            print(f"{v}: missing k ({sorted(rows)})")
            continue
        sig = rows[KS[0]]["sigma_slope"][W]
        x = {"sigma_slope": sig, "alive": [rows[k]["alive"] for k in KS], "fvu": [rows[k]["fvu"] for k in KS]}
        for m in ("full", "fixed"):
            g = [rows[k][f"gamma_{m}"][W] for k in KS]
            raw = [rows[k][f"dec_{m}"][W] for k in KS]
            x[m] = {"gamma": g, "dec": raw, "s_gamma": per_doubling(g), "s_raw": per_doubling(raw)}
        x["fixed"]["dec_sd"] = [rows[k]["dec_fixed_sd"][W] for k in KS]
        x["ill_conditioned"] = sig > ILL
        out["variants"][v] = x
    V = out["variants"]
    for v, (fn, key) in EARLIER.items():
        if v in V and (P3 / fn).exists():
            prev = json.load(open(P3 / fn))["fits"].get(key + W)
            if prev:
                diffs = [abs(V[v]["full"]["gamma"][i] - prev["gamma"][str(k)]) for i, k in enumerate(KS)]
                out["checks"][f"{v}_reproduces_earlier"] = {"max_abs_diff": max(diffs), "pass": max(diffs) <= 0.02}
        if v in V:
            out["checks"][f"{v}_live_count_bias"] = {
                "s_gamma_fixed_minus_full": V[v]["fixed"]["s_gamma"] - V[v]["full"]["s_gamma"],
                "s_raw_fixed_minus_full": V[v]["fixed"]["s_raw"] - V[v]["full"]["s_raw"]}
    for new, ref in PAIRS:
        if new in V and ref in V:
            res = {}
            for m in ("fixed", "full"):
                rg = V[new][m]["s_gamma"] / V[ref][m]["s_gamma"]
                rr = V[new][m]["s_raw"] / V[ref][m]["s_raw"]
                res[m] = {"retained_gamma": rg, "retained_raw": rr, "verdict": verdict(rg, rr)}
            res["primary"] = res["fixed"]["verdict"]
            out["verdict"][f"{new}_vs_{ref}"] = res
    vg, vh = (out["verdict"].get(p, {}).get("primary") for p in ("G_vs_D", "H_vs_F"))
    if vg and vh:
        if vg == vh == "keeps":
            out["reading"] = "the dictionary's covariance suffices: Gemma Scope's specific directions are not needed"
        elif vg == vh == "removes":
            out["reading"] = "structure in Gemma Scope's directions beyond their covariance carries the law"
        elif vg != vh:
            out["reading"] = f"G and H disagree (G {vg}, H {vh}): no claim"
        else:
            out["reading"] = "partial in both: no claim"

    for v, x in V.items():
        print(f"{LABEL[v]:<52} Sigma {x['sigma_slope']:+.3f}  alive {x['alive']}")
        for m in ("fixed", "full"):
            print(f"   {m:<5} gamma " + " ".join(f"{g:.3f}" for g in x[m]["gamma"])
                  + f"   s_gamma {x[m]['s_gamma']:+.4f}   s_raw {x[m]['s_raw']:+.4f}")
    print(json.dumps({"checks": out["checks"], "verdict": out["verdict"], "reading": out.get("reading")}, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    col = {"D": "#1f77b4", "G": "#7fb7e6", "F": "#d62728", "H": "#f19c9c"}
    for v, x in V.items():
        ax[0].plot(KS, x["fixed"]["gamma"], "o-", color=col[v], label=LABEL[v])
        ax[1].plot(KS, x["fixed"]["dec"], "o-", color=col[v])
    for a, t in zip(ax, ["gamma (decoder / Sigma slope)", "raw decoder slope"]):
        a.set_xscale("log", base=2)
        a.set_xlabel("k")
        a.set_title(f"{t}, ranks 11-150, {d['n_fix']} live latents")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(P3 / "control_summary.png", dpi=130)
    json.dump(out, open(P3 / "control_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
