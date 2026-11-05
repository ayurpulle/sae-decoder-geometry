"""Analyse the Gemma-matched planted runs (cache/saes_p3g): gamma per run in Gemma's rank windows plus the d-scaled
one, and gamma(k) = A + s·log2 k fits per (data, cond, seed, window), with log2 k0 = A / -s.

Windows (0-based, as everywhere): (3, 50), (10, 150), (30, 450) = Gemma's GPU-queue windows, ranks 4-50, 11-150,
31-450; and (54, 461) = ranks d/43 .. d/5, the d-scaled window of the planted d-sweep. Stores decoder and Sigma
eigenvalues so the W_dec files never need copying off the pod. Writes results/phase3/gemma_match.json.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (54, 461)]
MIN_ALIVE = 1000


def main():
    sig, rows = {}, []
    for tj in sorted(glob.glob(str(ROOT / "cache" / "saes_p3g" / "*" / "train.json"))):
        rd = Path(tj).parent
        info = json.load(open(tj))
        c = info["config"]
        act = ROOT / "cache" / "acts" / c["model"] / c["hook"]
        if c["model"] not in sig:
            sig[c["model"]] = np.sort(np.linalg.eigvalsh(np.load(act / "stats.npz")["cov"]))[::-1]
        fires = np.load(rd / "fires.npy")
        alive = int((fires > 0).sum())
        row = {"data": c["model"].removeprefix("synth-"), "cond": c["cond"], "k": c["k"], "seed": c["seed"],
               "width": c["width"], "fvu": info["fvu_eval"], "alive": alive, "seconds": info["seconds"], "gamma": {}}
        if alive >= MIN_ALIVE:
            e = eigen_slope(np.load(rd / "W_dec.npy"), mask=fires > 0)["eigenvalues"]
            row["eigenvalues"] = e[:600].tolist()
            for w in WINDOWS:
                row["gamma"][str(w)] = slope_from_eigenvalues(e, w)["slope"] / slope_from_eigenvalues(sig[c["model"]], w)["slope"]
        rows.append(row)

    fits = {}
    for key in sorted({(r["data"], r["cond"], r["seed"]) for r in rows}):
        xs = sorted((r for r in rows if (r["data"], r["cond"], r["seed"]) == key), key=lambda r: r["k"])
        for w in map(str, WINDOWS):
            pts = [(r["k"], r["gamma"][w]) for r in xs if w in r["gamma"]]
            if len(pts) >= 3:
                lr = linregress(np.log2([p[0] for p in pts]), [p[1] for p in pts])
                fits[f"{key[0]}|{key[1]}|{key[2]}|{w}"] = {
                    "gamma": dict(pts), "intercept": lr.intercept, "per_doubling": lr.slope, "r2": lr.rvalue ** 2,
                    "log2_k0": lr.intercept / -lr.slope if lr.slope < 0 else None}
                f = fits[f"{key[0]}|{key[1]}|{key[2]}|{w}"]
                print(f"{key[0]:<18} {key[1]:<9} s{key[2]} {w:<10} A={lr.intercept:6.3f} s={lr.slope:+.3f} "
                      f"R2={lr.rvalue ** 2:.2f} log2k0={f['log2_k0'] if f['log2_k0'] is None else round(f['log2_k0'], 2)}  "
                      + " ".join(f"{k}:{g:.3f}" for k, g in pts), flush=True)
    gh = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not gh and (ROOT / ".git_hash").exists():  # the pod gets the code without .git (gpu/push.sh)
        gh = (ROOT / ".git_hash").read_text().strip()
    out = ROOT / "results" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"git": gh, "windows": WINDOWS, "min_alive": MIN_ALIVE,
               "sigma_eigenvalues": {m: e[:600].tolist() for m, e in sig.items()}, "rows": rows, "fits": fits},
              open(out / "gemma_match.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
