"""The causal layer comparison in raw decoder slope and in gamma, and diag(Sigma) slopes.

Data: results/gpu/runs.json, group saes_gpu_p2 (TopK k = 40, layers 3/13/23, widths 16384/65536, conditions
real / gaussian (N(mu, Sigma)) / scrambled (each coordinate permuted independently across tokens), 3 seeds);
results/gpu/sigma.json (Sigma eigenvalues, 10M tokens); results/gpu/stats/L{3,13,23}.npz (mean/cov from the pod,
gitignored, read from the main checkout if absent here).

Reported (choices fixed in advance):
  per (width, condition, layer, window): raw decoder slope and gamma, mean and sd (ddof = 1) over seeds.
  gamma: real and gaussian against Sigma's eigen-slope; scrambled both against Sigma (as stored in runs.json)
  and against diag(Sigma) sorted descending (its own covariance; as gpu_analyze.py reports).
  diag(Sigma) "slope": the sorted per-coordinate variances fitted in the same rank windows; also the share of the
  trace in the top 1 / 10 coordinates, and Sigma's own top-1 eigenvalue share, for comparison.
  Check: 16k, ranks 11-150, raw slopes real -0.29/-0.48/-0.29 and gaussian -0.42/-0.51/-0.51 (L3/13/23).
  Scrambled-null check: is the scrambled null's gamma ~1.07 at L23 (vs diag) a small-denominator effect, i.e. is
  diag(Sigma)'s slope at L23 flatter than at L3/L13 while the scrambled decoder slope is not?
Writes results/review/causal_layers.json.
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

MAIN = ROOT if (ROOT / "results" / "gpu" / "stats").exists() else ROOT.parents[2]
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
OUT = ROOT / "results" / "review" / "causal_layers.json"


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)]


def wkey(w):
    return f"ranks{w[0] + 1}-{w[1]}"


def main():
    runs = json.load(open(ROOT / "results" / "gpu" / "runs.json"))["runs"]
    sig = json.load(open(ROOT / "results" / "gpu" / "sigma.json"))
    out = {"sigma": {}, "diag": {}, "table": {}, "raw_slope_check": {}, "scrambled_l23": {}}
    diag_sl = {}
    for L in (3, 13, 23):
        st = np.load(MAIN / "results" / "gpu" / "stats" / f"L{L}.npz")
        cov = st["cov"].astype(np.float64)
        dg = np.sort(np.diag(cov))[::-1]
        ev = np.asarray(sig[f"blocks.{L}.hook_resid_post"]["eigenvalues"], float)
        diag_sl[L] = {str(w): slope_from_eigenvalues(dg, w)["slope"] for w in WINDOWS}
        out["sigma"][L] = {"slopes": sig[f"blocks.{L}.hook_resid_post"]["slopes"],
                           "n_tokens": sig[f"blocks.{L}.hook_resid_post"]["n_tokens"],
                           "top1_eig_share": float(ev[0] / ev.sum()), "top10_eig_share": float(ev[:10].sum() / ev.sum()),
                           "stats_n": float(st["n"]) if "n" in st.files else None}
        out["diag"][L] = {"slopes": diag_sl[L], "top1_share": float(dg[0] / dg.sum()),
                          "top10_share": float(dg[:10].sum() / dg.sum()),
                          "top_dims": [int(i) for i in np.argsort(np.diag(cov))[::-1][:5]],
                          "trace_check_rel": float(abs(dg.sum() - ev.sum()) / ev.sum())}
    groups = defaultdict(list)
    for r in runs:
        if r["group"] != "saes_gpu_p2":
            continue
        ds = ast.literal_eval(r["decoder_slope"]) if isinstance(r["decoder_slope"], str) else r["decoder_slope"]
        g = ast.literal_eval(r["gamma"]) if isinstance(r["gamma"], str) else r["gamma"]
        groups[(int(r["width"]), r["cond"], int(r["layer"]))].append((ds, g, int(r["seed"])))
    print("raw decoder slope (mean ± sd over seeds) | gamma")
    for width in (16384, 65536):
        for w in map(str, WINDOWS):
            for cond in ("real", "gaussian", "scrambled"):
                row = {}
                for L in (3, 13, 23):
                    rs = groups[(width, cond, L)]
                    ssl = [x[0][w] for x in rs]
                    rec = {"raw_slope": ms(ssl), "gamma_vs_sigma": ms([x[1][w] for x in rs]),
                           "seeds": sorted(x[2] for x in rs)}
                    # sanity: stored gamma = slope / sigma slope
                    rec["stored_gamma_check"] = float(np.max(np.abs(np.array([x[1][w] for x in rs]) -
                                                                    np.array(ssl) / out["sigma"][L]["slopes"][w])))
                    if cond == "scrambled":
                        rec["gamma_vs_diag"] = ms([s / diag_sl[L][w] for s in ssl])
                    row[L] = rec
                out["table"][f"{width}|{cond}|{w}"] = row
                gk = "gamma_vs_diag" if cond == "scrambled" else "gamma_vs_sigma"
                print(f"  w{width:<6}{wkey(eval(w)):<15}{cond:<10}" + "  ".join(
                    f"L{L}: {row[L]['raw_slope'][0]:+.3f}±{row[L]['raw_slope'][1]:.3f} | {row[L][gk][0]:.3f}±{row[L][gk][1]:.3f}"
                    for L in (3, 13, 23)))
    t = out["table"]
    out["raw_slope_check"] = {c: {L: round(t[f"16384|{c}|(10, 150)"][L]["raw_slope"][0], 3) for L in (3, 13, 23)}
                            for c in ("real", "gaussian", "scrambled")}
    print("check, 16k ranks 11-150 raw slopes:", out["raw_slope_check"])
    for L in (3, 13, 23):
        print(f"  L{L} Sigma slopes " + " ".join(f"{wkey(eval(w))}:{v:+.3f}" for w, v in out["sigma"][L]["slopes"].items())
              + f"  top1 eig share {out['sigma'][L]['top1_eig_share']:.3f}")
        print(f"      diag slopes  " + " ".join(f"{wkey(eval(w))}:{v:+.3f}" for w, v in diag_sl[L].items())
              + f"  top1 diag share {out['diag'][L]['top1_share']:.3f} top dims {out['diag'][L]['top_dims'][:3]}")
    # scrambled-null decomposition: log gamma_scr = log |slope_scr| - log |diag slope|
    for width in (16384, 65536):
        for w in map(str, WINDOWS):
            out["scrambled_l23"][f"{width}|{w}"] = {L: {"scr_raw_slope": t[f"{width}|scrambled|{w}"][L]["raw_slope"][0],
                                                   "diag_slope": diag_sl[L][w],
                                                   "gamma_vs_diag": t[f"{width}|scrambled|{w}"][L]["gamma_vs_diag"][0],
                                                   "gamma_vs_sigma": t[f"{width}|scrambled|{w}"][L]["gamma_vs_sigma"][0],
                                                   "real_gamma": t[f"{width}|real|{w}"][L]["gamma_vs_sigma"][0]}
                                               for L in (3, 13, 23)}
    # Follow-up added after the diag-slope result (stated here before computing it): scrambled data keeps each
    # coordinate's marginal, so a candidate cause of the steep scrambled decoder at L23 is that its high-variance
    # coordinates are heavy-tailed (sparse, feature-like) and attract many latents. Per-coordinate excess kurtosis
    # on the 32,736-token cache (cache/acts/gemma-2-2b-all, fp16), by variance rank: median over ranks 1-10,
    # 11-150, 151-1000, 1001-2304, and Spearman(variance, kurtosis) over the top 150 coordinates.
    from scipy.stats import kurtosis, spearmanr
    out["marginal_kurtosis"] = {}
    for L in (3, 13, 23):
        A = np.load(MAIN / "cache" / "acts" / "gemma-2-2b-all" / f"L{L}.npy").astype(np.float64)
        var = A.var(0)
        kt = kurtosis(A, axis=0, fisher=True)
        o = np.argsort(var)[::-1]
        out["marginal_kurtosis"][L] = {
            **{f"median_ranks{a + 1}-{b}": float(np.median(kt[o[a:b]])) for a, b in ((0, 10), (10, 150), (150, 1000), (1000, 2304))},
            "spearman_var_kurt_top150": float(spearmanr(var[o[:150]], kt[o[:150]])[0]),
            "n_tokens": len(A)}
        print(f"  L{L} kurtosis by variance rank: " + "  ".join(f"{k}:{v:.2f}" for k, v in out["marginal_kurtosis"][L].items()))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
