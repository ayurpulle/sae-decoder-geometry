"""Check: uncertainty and design dependence of
the 288-SAE decomposition, and the L0 share of Li et al.'s layer-12 vs layer-24 contrast.

Data: the 288 Gemma Scope SAEs of results/phase1/layers.json + width_ladder_L12.json, slopes over all rows
(as in paper/figures/make_figures_thesisB.py), named L0, the same design (intercept, log2 L0, 25 layer dummies,
log2 width). The point estimates must reproduce thesisB_numbers.json decomposition_named_L0 (checked).

Analysis choices (fixed before looking at results):
  (a) Drop-one-factor Delta R^2 for L0, layer, width (full minus model without the factor), plus full R^2,
      b_L0 per doubling, layer-effect range, b_width per doubling, R^2 with L0 x layer, in each of the four windows.
  (b) Layer-clustered bootstrap: resample the 26 layers with replacement (all SAEs of a drawn layer come along,
      including layer 12's width ladder); a layer drawn twice enters as two clusters with separate dummies.
      B = 2000, seed 0, percentile 95% intervals. Replicates in which a factor is degenerate (e.g. one width only)
      are kept; the count of such replicates is stored.
  (c) Both on all 288 SAEs and on SAEs with named L0 in [30, 150] (the band containing Li et al.'s 72-124).
  (d) Li et al. contrast = slope(L24) - slope(L12), 65k SAEs (their width). Reported: Li's own SAEs (L0 72 vs 124).
      Matched contrasts, each a defensible match point:
        - at L0 = 72 (L24 interpolated; the published thesisB number),
        - at L0 = 124 (L12 interpolated; the interpolated case),
        - at L0 = 94.4 (geometric mean of 72 and 124, both interpolated),
        - every L0 on a 41-point log grid from 72 to 124 (both interpolated; min-max reported),
        - per-layer log-linear OLS fit of slope on log2 L0 over each layer's 65k ladder, evaluated at 72 and 124
          (fit instead of interpolation between neighbouring rungs),
        - the pooled additive model: share = b_L0 * log2(124/72) / reported contrast (b from (a), with its
          bootstrap interval),
        - the 16k ladders interpolated at 72 and 124 (width robustness; Li's SAEs are 65k).
      Interpolation is linear in log L0 between the two bracketing rungs of the same layer and width.
      L0 share = 1 - matched / reported. The reported range is the min-max over these match points.
Writes results/review/regression_ci.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "review" / "regression_ci.json"
WINS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
B = 2000


def wslope(entry, w):
    a, b = (int(s) for s in w.strip("()").split(","))
    for e in entry["windows"]:
        if list(e["fit_range"]) == [a, b]:
            return e["slope"]
    raise KeyError(w)


def r2_of(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


def fit_all(ll0, lw, clus, y):
    """All quantities of (a) for one data set; clus = integer cluster (layer) id per SAE."""
    n = len(y)
    ids = np.unique(clus)
    D = (clus[:, None] == ids[None, :]).astype(float)[:, 1:]
    one = np.ones((n, 1))
    X = {"L0": ll0[:, None], "layer": D, "width": lw[:, None]}

    def des(parts):
        return np.hstack([one] + [X[p] for p in parts])

    beta, full = r2_of(des(["L0", "layer", "width"]), y)
    out = {"r2_full": full,
           "drop_L0": full - r2_of(des(["layer", "width"]), y)[1],
           "drop_layer": full - r2_of(des(["L0", "width"]), y)[1],
           "drop_width": full - r2_of(des(["L0", "layer"]), y)[1],
           "r2_L0_alone": r2_of(des(["L0"]), y)[1], "r2_layer_alone": r2_of(des(["layer"]), y)[1],
           "b_L0_per_doubling": beta[1], "b_width_per_doubling": beta[-1]}
    lay = np.r_[0.0, beta[2:2 + D.shape[1]]]
    out["layer_effect_range"] = lay.max() - lay.min()
    Xi = np.hstack([des(["L0", "layer", "width"]), D * ll0[:, None]])
    try:  # secondary quantity; a rank-deficient bootstrap design can fail to converge -> NaN (nanpercentile)
        out["r2_with_L0xlayer"] = r2_of(Xi, y)[1]
    except np.linalg.LinAlgError:
        out["r2_with_L0xlayer"] = np.nan
    out["width_degenerate"] = bool(np.ptp(lw) == 0)
    return out


def boot(ll0, lw, layer, y, rng):
    layers = np.unique(layer)
    reps = []
    for _ in range(B):
        draw = rng.choice(layers, size=len(layers), replace=True)
        idx, cl = [], []
        for c, L in enumerate(draw):
            m = np.nonzero(layer == L)[0]
            idx.append(m)
            cl.append(np.full(len(m), c))
        idx, cl = np.concatenate(idx), np.concatenate(cl)
        reps.append(fit_all(ll0[idx], lw[idx], cl, y[idx]))
    keys = [k for k in reps[0] if k != "width_degenerate"]
    ci = {k: [float(np.nanpercentile([r[k] for r in reps], 2.5)), float(np.nanpercentile([r[k] for r in reps], 97.5))]
          for k in keys}
    ci["n_interaction_fit_failed"] = int(sum(np.isnan(r["r2_with_L0xlayer"]) for r in reps))
    ci["n_width_degenerate"] = int(sum(r["width_degenerate"] for r in reps))
    ci["frac_drop_layer_gt_drop_L0"] = float(np.mean([r["drop_layer"] > r["drop_L0"] for r in reps]))
    return ci


def main():
    saes = {}
    for f in ("layers.json", "width_ladder_L12.json"):
        for k, v in json.load(open(RES / "phase1" / f))["results"].items():
            saes[k] = {"layer": v["layer"], "width": v["width"], "n": v["n_latents"], "l0": v["l0"],
                       **{w: wslope(v, w) for w in WINS}}
    keys = sorted(saes)
    assert len(keys) == 288
    layer = np.array([saes[k]["layer"] for k in keys])
    ll0 = np.log2([saes[k]["l0"] for k in keys])
    lw = np.log2([saes[k]["n"] for k in keys])
    l0 = 2 ** ll0
    ref = json.load(open(ROOT / "paper" / "figures" / "thesisB_numbers.json"))["decomposition_named_L0"]
    out = {"notes": __doc__.split("Writes")[0].strip(), "decomposition": {}}
    for band, sel in (("all_288", np.ones(len(keys), bool)), ("L0_30_150", (l0 >= 30) & (l0 <= 150))):
        rng = np.random.default_rng(0)
        idx = np.nonzero(sel)[0]
        out["decomposition"][band] = {"n": int(len(idx)), "n_layers": int(len(np.unique(layer[idx]))),
                                      "n_widths": int(len(np.unique(lw[idx]))), "windows": {}}
        print(f"\n== {band}: n = {len(idx)}")
        for w in WINS:
            y = np.array([saes[keys[i]][w] for i in idx])
            pt = fit_all(ll0[idx], lw[idx], layer[idx], y)
            if band == "all_288":
                for a, b in (("drop_L0", "drop_L0"), ("drop_layer", "drop_layer"), ("drop_width", "drop_width"),
                             ("r2_full", "r2_L0+layer+width"), ("b_L0_per_doubling", "b_L0_per_doubling")):
                    assert abs(round(pt[a], 3) - ref[w][b]) < 1.5e-3, (w, a, pt[a], ref[w][b])
            ci = boot(ll0[idx], lw[idx], layer[idx], y, rng)
            out["decomposition"][band]["windows"][w] = {"point": pt, "ci95": ci}
            print(f"  {w:<12} R2 {pt['r2_full']:.3f} [{ci['r2_full'][0]:.3f},{ci['r2_full'][1]:.3f}]"
                  f"  dropL0 {pt['drop_L0']:.3f} [{ci['drop_L0'][0]:.3f},{ci['drop_L0'][1]:.3f}]"
                  f"  dropLayer {pt['drop_layer']:.3f} [{ci['drop_layer'][0]:.3f},{ci['drop_layer'][1]:.3f}]"
                  f"  dropWidth {pt['drop_width']:.4f} [{ci['drop_width'][0]:.4f},{ci['drop_width'][1]:.4f}]"
                  f"  b {pt['b_L0_per_doubling']:.3f} [{ci['b_L0_per_doubling'][0]:.3f},{ci['b_L0_per_doubling'][1]:.3f}]"
                  f"  layRange {pt['layer_effect_range']:.3f} [{ci['layer_effect_range'][0]:.3f},{ci['layer_effect_range'][1]:.3f}]"
                  f"  P(dropLayer>dropL0) {ci['frac_drop_layer_gt_drop_L0']:.3f}", flush=True)

    # ---- (d) Li contrast ----
    def ladder(L, width, w):
        return sorted((saes[k]["l0"], saes[k][w]) for k in keys if saes[k]["layer"] == L and saes[k]["width"] == width)

    def interp(L, width, t, w):
        pts = ladder(L, width, w)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= t <= x1:
                a = (math.log(t) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return y0 + a * (y1 - y0)
        return None

    def olsfit(L, width, t, w):
        pts = np.array(ladder(L, width, w))
        b, a = np.polyfit(np.log2(pts[:, 0]), pts[:, 1], 1)
        return a + b * math.log2(t)

    li = {}
    print("\n== Li contrast L24 - L12 (65k); share of the reported contrast attributable to L0")
    for w in WINS:
        s12 = saes["layer_12/width_65k/average_l0_72"][w]
        s24 = saes["layer_24/width_65k/average_l0_124"][w]
        rep = s24 - s12
        m = {"L0_72_interp": interp(24, "65k", 72, w) - s12,
             "L0_124_interp": s24 - interp(12, "65k", 124, w),
             "L0_94.4_interp_both": interp(24, "65k", math.sqrt(72 * 124), w) - interp(12, "65k", math.sqrt(72 * 124), w),
             "L0_72_olsfit": olsfit(24, "65k", 72, w) - olsfit(12, "65k", 72, w),
             "L0_124_olsfit": olsfit(24, "65k", 124, w) - olsfit(12, "65k", 124, w)}
        grid = np.exp(np.linspace(math.log(72), math.log(124), 41))
        g = [interp(24, "65k", t, w) - interp(12, "65k", t, w) for t in grid]
        m["grid_72_124_min"], m["grid_72_124_max"] = float(min(g)), float(max(g))
        for t in (72, 124):
            a16, b16 = interp(24, "16k", t, w), interp(12, "16k", t, w)
            if a16 is not None and b16 is not None:
                m[f"16k_L0_{t}_interp"] = a16 - b16
        dec = out["decomposition"]["all_288"]["windows"][w]
        bL0, bci = dec["point"]["b_L0_per_doubling"], dec["ci95"]["b_L0_per_doubling"]
        dl = math.log2(124 / 72)
        share = {k: 1 - v / rep for k, v in m.items()}
        share["additive_model_pooled_b"] = bL0 * dl / rep
        share["additive_model_pooled_b_ci95"] = [bci[0] * dl / rep, bci[1] * dl / rep]
        vals = [v for k, v in share.items() if not k.endswith("ci95")]
        li[w] = {"reported": rep, "L12_at72": s12, "L24_at124": s24, "matched_contrast": m, "L0_share": share,
                 "L0_share_range": [float(min(vals)), float(max(vals))],
                 "L0_share_range_interp_only": [float(min(share[k] for k in m if "16k" not in k)),
                                                float(max(share[k] for k in m if "16k" not in k))],
                 "ladders_65k": {"L12": ladder(12, "65k", w), "L24": ladder(24, "65k", w)}}
        print(f"  {w:<12} reported {rep:+.3f}  " + "  ".join(f"{k} {v:+.3f}" for k, v in m.items()))
        print(f"  {'':<12} L0 share: " + "  ".join(f"{k} {v:.2f}" for k, v in share.items() if not k.endswith('ci95'))
              + f"  additive CI [{share['additive_model_pooled_b_ci95'][0]:.2f},{share['additive_model_pooled_b_ci95'][1]:.2f}]"
              + f"  -> range {li[w]['L0_share_range'][0]:.2f}-{li[w]['L0_share_range'][1]:.2f}")
    out["li_contrast"] = li
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
