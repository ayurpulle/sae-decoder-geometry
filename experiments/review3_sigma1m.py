"""Redo every Gemma Scope result that used the 200k-token Sigma with a 1M-token Sigma (second review, 2026-11-10).

The second review's split-half check found the 200k-token Sigma too steep in the bulk windows at layer 12
(ranks 31-450: -0.81 against -0.73; ranks 101-1000: -0.84 against -0.72). This script, given
results/test1/sigma_gemma-2-2b_1m.npz (test1_sigma_layers.py --n_tokens 1000000 --tag 1m), reports:
  1. Sigma's slope per layer and window, 200k against 1M (and the layer-profile features the write-up quotes);
  2. the layer profile: decoder slope against Sigma slope across 26 layers (Pearson r, mean gamma), as in
     test1_sigma_plot.py, with each Sigma;
  3. Gemma Scope gamma by layer (65k and 16k, L0 = 40 and 70), the mid-stack numbers, with each Sigma;
  4. the layer-12 gamma(L0) law across 28 SAEs in every window (gamma = decoder slope / Sigma slope, so the
     decoder slopes are reused and only the denominator changes);
  5. the matrix test (basis gamma, diagonal share, top-10 overlap) for the layer-12 16k SAEs with the 1M Sigma.
Writes results/test1/sigma_1m_compare.json and results/test1/sigma_eigenvalues_1m.npy.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import basis_gamma, cloud_cov, slope_from_eigenvalues  # noqa: E402

T1 = ROOT / "results" / "test1"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
DEC = [("65k", 40), ("65k", 70), ("16k", 40), ("16k", 70)]
RANGES = ((1, 150), (10, 150), (100, 1000))


def key(w):
    return f"ranks{w[0] + 1}-{w[1]}"


def main():
    z2, z1 = np.load(T1 / "sigma_gemma-2-2b.npz"), np.load(T1 / "sigma_gemma-2-2b_1m.npz")
    cov2, cov1 = z2["cov"], z1["cov"]
    eig = {"200k": np.stack([np.sort(np.linalg.eigvalsh(c.astype(np.float64)))[::-1] for c in cov2]),
           "1m": np.stack([np.sort(np.linalg.eigvalsh(c.astype(np.float64)))[::-1] for c in cov1])}
    np.save(T1 / "sigma_eigenvalues_1m.npy", eig["1m"])
    L = len(cov1)
    sig = {s: {w: np.array([slope_from_eigenvalues(e, w)["slope"] for e in eig[s]]) for w in WINDOWS} for s in eig}
    out = {"n_tokens": {"200k": int(z2["n"]), "1m": int(z1["n"])},
           "sigma_slope": {s: {str(w): sig[s][w].tolist() for w in WINDOWS} for s in sig}}

    print("1. Sigma slope by layer (200k / 1M)")
    for w in WINDOWS:
        d = sig["1m"][w] - sig["200k"][w]
        print(f"  {key(w):<14} max |diff| {np.abs(d).max():.3f} (layer {int(np.abs(d).argmax())}), mean diff {d.mean():+.3f}")
    feats = {}
    for s in sig:
        top = sig[s][(10, 150)]
        bulk = sig[s][(30, 450)]
        feats[s] = {"top_layer0": float(top[0]), "top_min": float(top.min()), "top_argmin": int(top.argmin()),
                    "bulk_min": float(bulk.min()), "bulk_argmin": int(bulk.argmin()),
                    "bulk_layers2_6_mean": float(bulk[2:7].mean()), "bulk_rest_mean": float(np.r_[bulk[:2], bulk[7:]].mean()),
                    "deep_min": float(sig[s][(100, 1000)].min()), "deep_argmin": int(sig[s][(100, 1000)].argmin())}
        print(f"  {s}: ranks 11-150 layer 0 {feats[s]['top_layer0']:+.2f}, steepest {feats[s]['top_min']:+.2f} at layer "
              f"{feats[s]['top_argmin']}; ranks 31-450 steepest {feats[s]['bulk_min']:+.2f} at layer {feats[s]['bulk_argmin']}, "
              f"layers 2-6 mean {feats[s]['bulk_layers2_6_mean']:+.2f} vs rest {feats[s]['bulk_rest_mean']:+.2f}; "
              f"ranks 101-1000 steepest {feats[s]['deep_min']:+.2f} at layer {feats[s]['deep_argmin']}")
    out["profile_features"] = feats

    print("2. layer profile: decoder slope vs Sigma slope across layers")
    dec = json.load(open(ROOT / "results" / "phase1" / "slope_vs_layer_matched_l0.json"))
    prof = {}
    for w in WINDOWS:
        for width, l0 in DEC:
            dk = f"{width}_{key(w)}_L0_{l0}"
            if dk not in dec:
                continue
            layers = sorted(int(k) for k in dec[dk])
            y = np.array([dec[dk][str(l)] for l in layers])
            row = {}
            for s in sig:
                x = sig[s][w][layers]
                g = y / x
                row[s] = {"pearson_r": float(np.corrcoef(x, y)[0, 1]), "gamma_mean": float(g.mean()),
                          "gamma_by_layer": dict(zip(map(str, layers), g.tolist()))}
            prof[f"{width}_L0{l0}_{key(w)}"] = row
            print(f"  {width} L0={l0} {key(w):<14} r {row['200k']['pearson_r']:+.2f} -> {row['1m']['pearson_r']:+.2f}   "
                  f"mean gamma {row['200k']['gamma_mean']:.2f} -> {row['1m']['gamma_mean']:.2f}")
    out["profile"] = prof

    print("3. mid-stack gamma (Gemma Scope 65k, L0 = 40), layer groups 0-3 / 12-16 / 23-24")
    mid = {}
    for w in WINDOWS:
        r = prof.get(f"65k_L040_{key(w)}")
        if not r:
            continue
        mid[key(w)] = {}
        for s in sig:
            g = r[s]["gamma_by_layer"]
            grp = [np.mean([g[str(l)] for l in ls if str(l) in g]) for ls in (range(0, 4), range(12, 17), range(23, 25))]
            mid[key(w)][s] = grp
        print(f"  {key(w):<14} 200k {np.round(mid[key(w)]['200k'], 2)}  1M {np.round(mid[key(w)]['1m'], 2)}")
    out["midstack_65k_L040"] = mid

    print("4. layer-12 gamma(L0) law, 28 Gemma Scope SAEs")
    law = json.load(open(T1 / "gamma_l0_L12.json"))
    old_sig = law["sigma12_slopes"]
    fits = {}
    l0 = np.log2([r["l0"] for r in law["rows"]])
    for w in WINDOWS:
        g_old = np.array([r["gamma"][str(w)] for r in law["rows"]])
        g_new = g_old * old_sig[str(w)] / sig["1m"][w][12]
        for s, g in (("200k", g_old), ("1m", g_new)):
            lr = linregress(l0, g)
            fits[f"{s}|{w}"] = {"a": lr.intercept, "b_per_doubling": lr.slope, "r2": lr.rvalue ** 2}
        print(f"  {key(w):<14} 200k {fits[f'200k|{w}']['a']:.3f} {fits[f'200k|{w}']['b_per_doubling']:+.3f} (R2 {fits[f'200k|{w}']['r2']:.3f})"
              f"   1M {fits[f'1m|{w}']['a']:.3f} {fits[f'1m|{w}']['b_per_doubling']:+.3f} (R2 {fits[f'1m|{w}']['r2']:.3f})")
    out["l12_law"] = fits

    print("5. matrix test, layer-12 16k SAEs, with the 1M Sigma")
    S12 = cov1[12].astype(np.float64)
    mt = []
    wdir = ROOT / "cache" / "gemmascope_wdec" / "layer_12" / "width_16k"
    for sae in sorted(wdir.iterdir(), key=lambda p: int(p.name.split("_")[-1])):
        r = basis_gamma(cloud_cov(np.load(sae / "W_dec.npy")), S12, rank_ranges=RANGES)
        mt.append({"sae": sae.name, **r})
        print(f"  {sae.name:<16} diag {r['diag_share']:.2f} ov10 {r['subspace_overlap'][10]:.2f}  "
              + " ".join(f"{f['gamma']:+.2f}(R2 {f['r2']:.2f})" for f in r["fits"]))
    out["matrix_test_1m"] = mt
    json.dump(out, open(T1 / "sigma_1m_compare.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
