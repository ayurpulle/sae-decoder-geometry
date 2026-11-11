"""Live-count bias of rank-window gamma, with the real layer-13 Sigma spectrum,
plus regeneration of the other log-only numbers where the data is on this machine.

Simulation model (fixed in advance; the same as LOG 2026-11-10):
  Sigma_13 eigenvalues lam (results/gpu/sigma.json, 10M tokens, d = 2304). A cloud with true gamma g has
  covariance diag(lam^g) (rotation does not change eigenvalues); n points x ~ N(0, diag(lam^g)), each scaled to
  unit norm (decoder rows are unit norm), centred covariance, eigenvalues, slope in each window, divided by
  Sigma_13's slope in that window -> measured gamma.
  Grid: g in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, n in {672, 2000, 4786, 8192, 16384}, windows (3,50), (10,150),
  (30,450), (100,1000), 6 draws. A window whose upper rank exceeds n - 1 is undefined (reported null).
  Inversion: for the measured layer-13 k = 1 value (gamma = 1.034, ranks 11-150, 672 live latents; 2-seed means
  from runs.json), find the true g whose mean measured gamma at n = 672 equals it, by linear interpolation on a
  grid g = 0, 0.2, ..., 2.0 (2 draws; runtime limit). The same inversion is applied to every real and Gaussian k in the
  layer-13 k sweep (results/gpu/runs.json, group saes_gpu_k), each at its own mean live count, ranks 11-150 and
  ranks 4-50. Caveat stated with the result: the calibration assumes a Gaussian cloud whose covariance is a power
  of Sigma; real decoder clouds are not Gaussian.
Log-only numbers regenerated:
  - Pythia-70m L3 Sigma: top-eigenvalue share of the trace (cache stats.npz).
  - Variant A FVU vs real (results/resynth/check.json, local recomputation) and RMS vs R
    (results/phase3/resynth_summary.json).
  - Near-duplicate fraction (nearest-neighbour cosine > 0.5) of every SAE whose decoder is on this machine
    (results/review/checks.json basis_gamma rows: Pythia L3 SAEs and local Gemma nulls). The GPU-run decoders
    (results/gpu/runs.json) were not copied off the pod; their near-duplicate fraction cannot be regenerated.
Writes results/review/live_count.json.
"""
import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

MAIN = ROOT if (ROOT / "cache").exists() else ROOT.parents[2]
OUT = ROOT / "results" / "review" / "live_count.json"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
GAMMAS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
NS = [672, 2000, 4786, 8192, 16384]
DRAWS = 6


def measured(lam, sig_sl, g, n, rng):
    d = len(lam)
    x = rng.standard_normal((n, d)).astype(np.float32) * np.sqrt(lam ** g).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x -= x.mean(0)
    if n < d:  # same nonzero spectrum from the n x n Gram
        e = np.sort(np.linalg.eigvalsh((x @ x.T).astype(np.float64) / (n - 1)))[::-1]
    else:  # only ranks <= 1000 are used
        e = np.sort(eigh((x.T @ x).astype(np.float64) / (n - 1), eigvals_only=True,
                         subset_by_index=[d - 1001, d - 1]))[::-1]
    return {str(w): (slope_from_eigenvalues(e, w)["slope"] / sig_sl[str(w)] if w[1] <= n - 1 else None) for w in WINDOWS}


def main():
    t0 = time.time()
    sig = json.load(open(ROOT / "results" / "gpu" / "sigma.json"))["blocks.13.hook_resid_post"]
    lam = np.asarray(sig["eigenvalues"], float)
    lam = lam / lam[0]
    sig_sl = {str(w): slope_from_eigenvalues(lam, w)["slope"] for w in WINDOWS}
    out = {"sigma13_slopes": sig_sl, "grid": {}}
    rng = np.random.default_rng(0)
    print("measured gamma (mean ± sd over draws), ranks 11-150")
    for g in GAMMAS:
        for n in NS:
            ms = [measured(lam, sig_sl, g, n, rng) for _ in range(DRAWS)]
            cell = {}
            for w in map(str, WINDOWS):
                v = [m[w] for m in ms if m[w] is not None]
                cell[w] = [float(np.mean(v)), float(np.std(v, ddof=1))] if v else None
            out["grid"][f"{g}|{n}"] = cell
        print(f"  true {g:<5}" + "  ".join(f"n{n}: {out['grid'][f'{g}|{n}'][str((10, 150))][0]:.3f}±{out['grid'][f'{g}|{n}'][str((10, 150))][1]:.3f}" for n in NS), flush=True)

    # live counts in the layer-13 k sweep
    runs = [r for r in json.load(open(ROOT / "results" / "gpu" / "runs.json"))["runs"] if r["group"] == "saes_gpu_k"]
    width = 16384
    sweep = defaultdict(list)
    for r in runs:
        g = ast.literal_eval(r["gamma"]) if isinstance(r["gamma"], str) else r["gamma"]
        sweep[(r["cond"], int(r["k"]))].append((width - int(r["n_dead"]), g))
    fine = np.round(np.arange(0, 2.0001, 0.2), 2)
    cal_cache = {}

    def calib(n):
        if n not in cal_cache:
            r2 = np.random.default_rng(1000 + n)
            cal_cache[n] = {str(w): [] for w in WINDOWS[:2]}
            for g in fine:
                ms = [measured(lam, sig_sl, g, n, r2) for _ in range(2)]
                for w in WINDOWS[:2]:
                    cal_cache[n][str(w)].append(float(np.mean([m[str(w)] for m in ms])))
        return cal_cache[n]

    def invert(n, w, gm):
        c = np.asarray(calib(n)[str(w)])
        if not np.all(np.diff(c) > 0):
            # enforce monotone by running max for interpolation; record that it was needed
            c = np.maximum.accumulate(c)
        if gm > c[-1]:
            return None
        return float(np.interp(gm, c, fine))

    corr = {}
    print("bias-corrected gamma, layer-13 k sweep (live count, measured -> corrected)")
    for (cond, k), xs in sorted(sweep.items()):
        n = int(round(np.mean([x[0] for x in xs])))
        row = {"live_mean": n, "live_per_seed": [x[0] for x in xs]}
        for w in WINDOWS[:2]:
            gm = float(np.mean([x[1][str(w)] for x in xs]))
            row[str(w)] = {"measured": gm, "corrected": invert(n, w, gm)}
        corr[f"{cond}|{k}"] = row
        print(f"  {cond:<9} k{k:<4} live {n:>6}  11-150 {row[str((10, 150))]['measured']:.3f} -> {row[str((10, 150))]['corrected']}"
              f"   4-50 {row[str((3, 50))]['measured']:.3f} -> {row[str((3, 50))]['corrected']}", flush=True)
    out["k_sweep_corrected"] = corr
    out["calibration_fine"] = {str(n): {"gamma_true": fine.tolist(), **c} for n, c in cal_cache.items()}
    k1 = corr["real|1"]
    out["headline"] = {"measured_k1_ranks11_150": k1[str((10, 150))]["measured"], "live_k1": k1["live_mean"],
                       "corrected_k1_ranks11_150": k1[str((10, 150))]["corrected"],
                       "true1_measures_at_672": out["grid"]["1.0|672"][str((10, 150))],
                       "true0.5_measures": {n: out["grid"][f"0.5|{n}"][str((10, 150))] for n in NS}}
    # fit of corrected real gamma vs log2 k (k = 1..128), ranks 11-150
    ks = sorted(k for c, k in sweep if c == "real")
    yc = [corr[f"real|{k}"][str((10, 150))]["corrected"] for k in ks]
    ym = [corr[f"real|{k}"][str((10, 150))]["measured"] for k in ks]
    if all(v is not None for v in yc):
        from scipy.stats import linregress
        for lab, y in (("measured", ym), ("corrected", yc)):
            lr = linregress(np.log2(ks), y)
            out["headline"][f"fit_{lab}"] = {"intercept_k1": lr.intercept, "per_doubling": lr.slope, "r2": lr.rvalue ** 2}

    # ---- log-only numbers ----
    lo = {}
    st = np.load(MAIN / "cache" / "acts" / "pythia-70m" / "blocks.3.hook_resid_post" / "stats.npz")
    ev = np.sort(np.linalg.eigvalsh(st["cov"].astype(np.float64)))[::-1]
    lo["pythia_L3_sigma"] = {"top1_share": float(ev[0] / ev.sum()), "top2_share": float(ev[1] / ev.sum()),
                             "n_tokens": float(st["n"]) if "n" in st.files else None,
                             "slope_ranks6-50": slope_from_eigenvalues(ev, (5, 50))["slope"]}
    chk = json.load(open(ROOT / "results" / "resynth" / "check.json"))
    rs = json.load(open(ROOT / "results" / "phase3" / "resynth_summary.json"))["variants"]["A"]["windows"]
    lo["variant_A"] = {"fvu_vs_real_local": chk["variants"]["A"]["fvu_vs_real"],
                       "fvu_note": "local recomputation (resynth_check.py); the 0.215 quoted in the log is from the pod "
                                   "run's synthesized-activation config, which is not on this machine",
                       "rms_vs_R": {w: rs[w]["rms_vs_R"] for w in rs}}
    bg = json.load(open(ROOT / "results" / "review" / "checks.json"))["basis_gamma"]
    dups = [{"model": x["model"], "layer": x.get("layer"), "cond": x.get("cond"), "width": x.get("width"),
             "seed": x.get("seed"), "alive": x["alive"], "dup05": x["dup05"]} for x in bg if "dup05" in x]
    lo["near_duplicates"] = {"rows": dups, "max": max(d["dup05"] for d in dups),
                             "max_row": max(dups, key=lambda d: d["dup05"]),
                             "max_nulls": max(d["dup05"] for d in dups if d["cond"] != "real"),
                             "n_over_0.004": sum(d["dup05"] > 0.004 for d in dups),
                             "gpu_runs": "not computable: GPU-run decoders are not on this machine"}
    out["log_numbers"] = lo
    print("Pythia top1 share", lo["pythia_L3_sigma"]["top1_share"], " A fvu", lo["variant_A"]["fvu_vs_real_local"],
          " A rms(10,150)", lo["variant_A"]["rms_vs_R"]["(10, 150)"], " dup max", lo["near_duplicates"]["max"],
          lo["near_duplicates"]["max_row"])
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
