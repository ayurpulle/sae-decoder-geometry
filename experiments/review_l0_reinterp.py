"""Review check 3, analysis: redo the matched-L0 layer profiles (Phase 1b slope, Test 1c gamma) at measured L0.

Phase 1b interpolated each layer's 16k slopes (log-linear in L0) to target L0s using the L0 in the SAE name.
Here the same interpolation uses the L0 measured on our tokens (review_l0_layers.py), and the two profiles
are compared: per-layer shift, peak layer, and peak-minus-edges contrast. gamma = decoder slope / Sigma slope
per window, with Sigma's eigenvalues from Test 1. Writes results/review/l0_reinterp.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

TARGETS = [20, 40, 70, 140]
OUT = ROOT / "results" / "review" / "l0_reinterp.json"


def interp(pts, t):
    if len(pts) < 2:
        return None
    pts = sorted(pts)
    x = np.log([p[0] for p in pts])
    if not (x[0] <= np.log(t) <= x[-1]):
        return None
    return float(np.interp(np.log(t), x, [p[1] for p in pts]))


def main():
    lay = json.load(open(ROOT / "results" / "phase1" / "layers.json"))
    wins = [tuple(w) for w in lay["config"]["windows"]]
    meas = json.load(open(ROOT / "results" / "review" / "l0_layers.json"))["results"]
    eig = np.load(ROOT / "results" / "test1" / "sigma_eigenvalues.npy")
    rows = {sid: r for sid, r in lay["results"].items() if r["width"] == "16k" and sid in meas}

    ratio = {}
    for r in meas.values():
        ratio.setdefault(r["layer"], []).append(r["ratio"])
    out = {"ratio_by_layer": {L: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
                              for L, v in sorted(ratio.items())}, "profiles": {}}
    print("measured/named L0 by layer (median [min, max]):")
    print("  " + "  ".join(f"{L}:{np.median(v):.2f}[{min(v):.2f},{max(v):.2f}]" for L, v in sorted(ratio.items())))

    for j, w in enumerate(wins):
        sig = {L: slope_from_eigenvalues(eig[L], w)["slope"] for L in range(eig.shape[0])}
        for t in TARGETS:
            prof = {}
            for L in range(26):
                rs = [r for r in rows.values() if r["layer"] == L]
                named = interp([(r["l0"], r["windows"][j]["slope"]) for r in rs], t)
                measd = interp([(meas[s]["l0_measured"], r["windows"][j]["slope"]) for s, r in rows.items()
                                if r["layer"] == L], t)
                if named is not None and measd is not None:
                    prof[L] = {"slope_named": named, "slope_measured": measd,
                               "gamma_named": named / sig[L], "gamma_measured": measd / sig[L]}
            if not prof:
                continue
            key = f"ranks{w[0] + 1}-{w[1]}_L0_{t}"
            out["profiles"][key] = prof
            Ls = sorted(prof)
            gn = np.array([prof[L]["gamma_named"] for L in Ls])
            gm = np.array([prof[L]["gamma_measured"] for L in Ls])
            edge = [i for i, L in enumerate(Ls) if L <= 3 or L >= 22]
            print(f"{key:22s} gamma peak layer named {Ls[gn.argmax()]:2d} ({gn.max():.3f})  measured "
                  f"{Ls[gm.argmax()]:2d} ({gm.max():.3f})  peak-minus-edges {gn.max() - gn[edge].mean():.3f} -> "
                  f"{gm.max() - gm[edge].mean():.3f}  max |shift| {np.abs(gm - gn).max():.3f}  "
                  f"r(named, measured) {np.corrcoef(gn, gm)[0, 1]:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
