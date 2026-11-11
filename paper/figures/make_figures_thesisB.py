"""Numbers and figures for paper/thesisB.tex, computed only from files under results/.

Run:  python paper/figures/make_figures_thesisB.py
Writes paper/figures/thesisB_numbers.json (every derived number the paper quotes, keyed by name) and
paper/figures/thesisB_fig*.pdf / .png.

Conventions
- Rank windows are 0-based [start, end) slices of the sorted spectrum: (3, 50) = ranks 4-50, (10, 150) = ranks
  11-150 (primary, Li et al.'s notebook window), (30, 450) = ranks 31-450, (100, 1000) = ranks 101-1000.
- "slope" is the least-squares slope of log10(eigenvalue) on log10(rank) in a window (negative; more negative =
  steeper). Nothing about the decoder slope depends on any Sigma estimate.
- gamma = decoder slope / activation-covariance slope in the same window. Gemma Scope gammas use the 1M-token
  Sigma (results/test1/sigma_1m_compare.json). Our own TopK SAEs use their 10M-token training Sigma.
- L0: for the 288-SAE decomposition we use the L0 in the SAE's release name (the only value available for all
  288). Measured L0 exists for the 135 16k SAEs (results/review/l0_layers.json) and is used as a sensitivity check.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
WINS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WLAB = {"(3, 50)": "ranks 4-50", "(10, 150)": "ranks 11-150", "(30, 450)": "ranks 31-450",
        "(100, 1000)": "ranks 101-1000"}
WKEY = {"(3, 50)": "ranks4-50", "(10, 150)": "ranks11-150", "(30, 450)": "ranks31-450",
        "(100, 1000)": "ranks101-1000"}
N = {}


def load(p):
    return json.load(open(RES / p))


def r(x, nd=4):
    return None if x is None else float(round(float(x), nd))


def wslope(entry, w):
    a, b = (int(s) for s in w.strip("()").split(","))
    for e in entry["windows"]:
        if list(e["fit_range"]) == [a, b]:
            return e["slope"]
    raise KeyError(w)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return beta, r2


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.c_[np.ones_like(x), x]
    beta, r2 = ols(X, y)
    return float(beta[0]), float(beta[1]), float(r2)


# --------------------------------------------------------------------------------------------------------------
# 1. Reproduction of Li et al. (phase0)
p0 = load("phase0/phase0.json")
rep = {}
for L, v in p0["results"].items():
    rep[L] = {"sae": v["sae"], "paper": v["paper_slope"], "ours": r(v["variants"]["paper"]["slope"], 4),
              "rel_diff_pct": r(100 * abs(v["rel_diff"]), 3), "r2": r(v["variants"]["paper"]["r2"], 3),
              "iso_gaussian": r(v["variants"]["paper"].get("gaussian_slope"), 4),
              "uncentred_delta": r(abs(v["variants"]["uncentered"]["slope"] - v["variants"]["paper"]["slope"]), 4),
              "unitnorm_delta": r(abs(v["variants"]["unit_norm"]["slope"] - v["variants"]["paper"]["slope"]), 4),
              "fit_range_robustness": [(fr["fit_range"], r(fr["slope"], 3)) for fr in v["fit_range_robustness"]]}
N["reproduction"] = rep
N["reproduction_max_rel_diff_pct"] = max(v["rel_diff_pct"] for v in rep.values())

# --------------------------------------------------------------------------------------------------------------
# 2. The 288 Gemma Scope SAEs
Ld = load("phase1/layers.json")["results"]
Wd = load("phase1/width_ladder_L12.json")["results"]
saes = {}
for src in (Ld, Wd):
    for k, v in src.items():
        saes[k] = {"layer": v["layer"], "width": v["width"], "n": v["n_latents"], "l0": v["l0"],
                   **{w: wslope(v, w) for w in WINS}}
keys = sorted(saes)
N["n_saes"] = len(keys)
N["n_saes_by_width"] = {w: sum(1 for k in keys if saes[k]["width"] == w) for w in
                        ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]}
layer = np.array([saes[k]["layer"] for k in keys])
lw = np.log2([saes[k]["n"] for k in keys])
ll0 = np.log2([saes[k]["l0"] for k in keys])
N["l0_range_named"] = [int(min(saes[k]["l0"] for k in keys)), int(max(saes[k]["l0"] for k in keys))]

Lay = np.zeros((len(keys), 26))
Lay[np.arange(len(keys)), layer] = 1.0
one = np.ones((len(keys), 1))


def design(parts):
    cols = [one]
    if "L0" in parts:
        cols.append(ll0[:, None])
    if "layer" in parts:
        cols.append(Lay[:, 1:])
    if "width" in parts:
        cols.append(lw[:, None])
    if "L0xlayer" in parts:
        cols.append(Lay[:, 1:] * ll0[:, None])
    return np.hstack(cols)


decomp = {}
for w in WINS:
    y = np.array([saes[k][w] for k in keys])
    out = {"slope_min": r(y.min(), 3), "slope_max": r(y.max(), 3), "slope_sd": r(y.std(ddof=1), 3)}
    for name, parts in {"L0": ["L0"], "layer": ["layer"], "width": ["width"], "L0+layer": ["L0", "layer"],
                        "L0+layer+width": ["L0", "layer", "width"],
                        "L0+layer+width+L0xlayer": ["L0", "layer", "width", "L0xlayer"]}.items():
        beta, r2 = ols(design(parts), y)
        out["r2_" + name] = r(r2, 3)
        if name == "L0+layer+width":
            out["b_L0_per_doubling"] = r(beta[1], 3)
            lay_eff = np.r_[0.0, beta[2:27]]
            out["layer_effect_range"] = r(lay_eff.max() - lay_eff.min(), 3)
            out["layer_effect_argmin"] = int(np.argmin(lay_eff))
            out["b_width_per_doubling"] = r(beta[27], 3)
            out["width_effect_6_doublings"] = r(6 * beta[27], 3)
            out["L0_effect_over_range"] = r(beta[1] * (ll0.max() - ll0.min()), 3)
            res = y - design(parts) @ beta
            out["resid_sd"] = r(res.std(ddof=design(parts).shape[1]), 3)
    full = out["r2_L0+layer+width"]
    out["drop_L0"] = r(full - ols(design(["layer", "width"]), y)[1], 3)
    out["drop_layer"] = r(full - ols(design(["L0", "width"]), y)[1], 3)
    out["drop_width"] = r(full - ols(design(["L0", "layer"]), y)[1], 3)
    decomp[w] = out
N["decomposition_named_L0"] = decomp
N["log2_L0_span"] = r(ll0.max() - ll0.min(), 2)

# Same decomposition restricted to the 135 16k SAEs with measured L0 (sensitivity to name-vs-measured L0).
l0m = load("review/l0_layers.json")["results"]
k16 = [k for k in keys if saes[k]["width"] == "16k"]
assert all(k in l0m for k in k16), "measured L0 missing"
sens = {}
for lab, l0vec in [("named", [saes[k]["l0"] for k in k16]), ("measured", [l0m[k]["l0_measured"] for k in k16])]:
    x = np.log2(l0vec)
    lay16 = np.array([saes[k]["layer"] for k in k16])
    LL = np.zeros((len(k16), 26))
    LL[np.arange(len(k16)), lay16] = 1
    for w in WINS:
        y = np.array([saes[k][w] for k in k16])
        X = np.c_[np.ones(len(k16)), x, LL[:, 1:]]
        beta, r2 = ols(X, y)
        r2L0 = ols(np.c_[np.ones(len(k16)), x], y)[1]
        r2lay = ols(np.c_[np.ones(len(k16)), LL[:, 1:]], y)[1]
        sens.setdefault(w, {})[lab] = {"b_L0": r(beta[1], 3), "r2_full": r(r2, 3), "r2_L0": r(r2L0, 3),
                                       "r2_layer": r(r2lay, 3)}
N["decomposition_16k_named_vs_measured"] = sens
ratios = [l0m[k]["ratio"] for k in k16]
N["l0_measured_over_named_range"] = [r(min(ratios), 3), r(max(ratios), 3)]
rbl = load("review/l0_reinterp.json")
N["l0_ratio_median_by_layer_range"] = [r(min(v["median"] for v in rbl["ratio_by_layer"].values()), 3),
                                       r(max(v["median"] for v in rbl["ratio_by_layer"].values()), 3)]
mx = 0.0
for cell, prof in rbl["profiles"].items():
    for L, v in prof.items():
        mx = max(mx, abs(v["slope_measured"] - v["slope_named"]))
N["l0_reinterp_max_abs_slope_change"] = r(mx, 3)

# --------------------------------------------------------------------------------------------------------------
# 3. Layer 12: slope against L0 across 7 widths (Sigma-free) and the gamma law
g12 = load("test1/gamma_l0_L12.json")
rows = g12["rows"]
s200 = g12["sigma12_slopes"]
s1m_all = load("test1/sigma_1m_compare.json")
s1m12 = {w: s1m_all["sigma_slope"]["1m"][w][12] for w in WINS}
N["sigma12_1m"] = {w: r(s1m12[w], 3) for w in WINS}
l12 = {}
for w in WINS:
    slopes = [row["gamma"][w] * s200[w] for row in rows]  # exact: gamma was stored against the 200k Sigma
    # cross-check the reconstructed slope against the width-ladder file
    for row, s in zip(rows, slopes):
        assert abs(s - wslope(Wd[row["sae_id"]], w)) < 1e-9
    l0s = [row["l0"] for row in rows]
    a, b, r2 = linfit(np.log2(l0s), slopes)
    gam = [s / s1m12[w] for s in slopes]
    ga, gb, gr2 = linfit(np.log2(l0s), gam)
    l12[w] = {"slope_a": r(a, 3), "slope_b_per_doubling": r(b, 3), "slope_r2": r(r2, 3),
              "gamma_a": r(ga, 3), "gamma_b": r(gb, 3), "gamma_r2": r(gr2, 4),
              "slope_min": r(min(slopes), 3), "slope_max": r(max(slopes), 3),
              "l0_at_min": r(l0s[int(np.argmin(slopes))], 1), "l0_at_max": r(l0s[int(np.argmax(slopes))], 1)}
    ref = s1m_all["l12_law"]["1m|" + w]
    assert abs(ga - ref["a"]) < 1e-6 and abs(gb - ref["b_per_doubling"]) < 1e-6
N["l12_law"] = l12
N["l12_n_saes"] = len(rows)
N["l12_n_measured_l0"] = sum(1 for row in rows if row["l0_source"] == "measured")

# fixed-n rescoring (16,384 random rows per SAE): stored as gamma against the 200k Sigma -> convert to slopes
fx = load("review/review2_checks.json")["fixedn"]
fixed = {}
for w in WINS:
    full = [row["gamma_full"][w] * s200[w] for row in fx["rows"]]
    fixd = [row["gamma_fixed"][w] * s200[w] for row in fx["rows"]]
    l0s = [row["l0"] for row in fx["rows"]]
    a1, b1, r21 = linfit(np.log2(l0s), full)
    a2, b2, r22 = linfit(np.log2(l0s), fixd)
    fixed[w] = {"full_b": r(b1, 3), "fixed_b": r(b2, 3), "full_r2": r(r21, 3), "fixed_r2": r(r22, 3),
                "max_abs_change": r(max(abs(p - q) for p, q in zip(full, fixd)), 3)}
N["l12_fixed_n"] = fixed

# width at matched L0
wm = load("phase1/slope_vs_width_matched_l0.json")
wspan = {}
for k, v in wm.items():
    vals = list(v.values())
    wspan[k] = {"n_widths": len(vals), "span": r(max(vals) - min(vals), 3)}
N["width_span_matched_l0"] = wspan

# subsampling 1m/524k/262k decoders to 16,384 rows
sub = load("review/checks.json")["subsample"]
subs = {}
for k, v in sub.items():
    subs[k] = {w: {"full": r(v["full"][i], 3), "sub16384": r(v["sub16384"][i], 3),
                   "shift": r(v["sub16384"][i] - v["full"][i], 3)} for i, w in enumerate(WINS)}
N["subsample_rows"] = subs
# fraction of the gap to the 16k SAE at similar L0 closed by subsampling (ranks 101-1000)
pairs = {"layer_12/width_1m/average_l0_19": "layer_12/width_16k/average_l0_22",
         "layer_12/width_1m/average_l0_73": "layer_12/width_16k/average_l0_82"}
frac = {}
for big, small in pairs.items():
    i = 3
    full, s16, tgt = sub[big]["full"][i], sub[big]["sub16384"][i], wslope(Wd[small], WINS[i])
    frac[big] = {"target_16k": r(tgt, 3), "frac_closed": r((s16 - full) / (tgt - full), 2)}
N["subsample_frac_closed_101_1000"] = frac

# --------------------------------------------------------------------------------------------------------------
# 4. Layer profile at matched L0 and Li et al.'s layer contrast
lm = load("phase1/slope_vs_layer_matched_l0.json")
prof = {}
for k, v in lm.items():
    vals = {int(L): s for L, s in v.items()}
    if len(vals) < 26:
        continue
    Lmin = min(vals, key=vals.get)
    Lmax = max(vals, key=vals.get)
    edge = max(vals[0], vals[23]) if 23 in vals else None
    prof[k] = {"steepest_layer": Lmin, "steepest": r(vals[Lmin], 3), "flattest_layer": Lmax,
               "flattest": r(vals[Lmax], 3), "range": r(vals[Lmax] - vals[Lmin], 3),
               "L0": r(vals[0], 3), "L12": r(vals[12], 3), "L13": r(vals[13], 3), "L23": r(vals[23], 3),
               "L24": r(vals[24], 3), "mid13_15_mean": r(np.mean([vals[13], vals[14], vals[15]]), 3)}
N["layer_profile_matched_l0"] = prof


def interp_l0(layer_, width, target, w):
    pts = sorted((saes[k]["l0"], saes[k][w]) for k in keys
                 if saes[k]["layer"] == layer_ and saes[k]["width"] == width)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target <= x1:
            t = (math.log(target) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0), (x0, x1)
    raise ValueError


li = {}
for w in WINS:
    s0 = saes["layer_0/width_65k/average_l0_73"][w]
    s12 = saes["layer_12/width_65k/average_l0_72"][w]
    s24 = saes["layer_24/width_65k/average_l0_124"][w]
    s24m, br = interp_l0(24, "65k", 72, w)
    li[w] = {"L0_at73": r(s0, 3), "L12_at72": r(s12, 3), "L24_at124": r(s24, 3), "L24_at72_interp": r(s24m, 3),
             "bracket": br, "contrast_12_24_reported": r(s24 - s12, 3), "contrast_12_24_matched": r(s24m - s12, 3),
             "contrast_shrink": r((s24 - s12) - (s24m - s12), 3),
             "contrast_shrink_frac": r(((s24 - s12) - (s24m - s12)) / (s24 - s12), 2)}
N["li_contrast"] = li
# what the L12 65k ladder spans vs what depth spans at fixed L0
lad = [saes[k]["(10, 150)"] for k in keys if saes[k]["layer"] == 12 and saes[k]["width"] == "65k"]
N["L12_65k_ladder_span_11_150"] = [r(min(lad), 3), r(max(lad), 3), r(max(lad) - min(lad), 3)]
lad_all = [saes[k]["(10, 150)"] for k in keys if saes[k]["layer"] == 12]
N["L12_all_widths_span_11_150"] = [r(min(lad_all), 3), r(max(lad_all), 3), r(max(lad_all) - min(lad_all), 3)]
# per-layer ladder span (16k and 65k), ranks 11-150
spans = []
for L_ in range(26):
    for wd in ["16k", "65k"]:
        v = [saes[k]["(10, 150)"] for k in keys if saes[k]["layer"] == L_ and saes[k]["width"] == wd]
        spans.append(max(v) - min(v))
N["per_layer_ladder_span_11_150"] = {"min": r(min(spans), 3), "median": r(float(np.median(spans)), 3),
                                     "max": r(max(spans), 3)}

# --------------------------------------------------------------------------------------------------------------
# 5. Our TopK k sweep at layer 13 (Sigma-free raw slopes)
runs = load("gpu/runs.json")["runs"]
ks = sorted({x["k"] for x in runs if x["group"] == "saes_gpu_k"})
ksw = {}
for cond in ["real", "gaussian"]:
    for k in ks:
        rr = [x for x in runs if x["group"] == "saes_gpu_k" and x["cond"] == cond and x["k"] == k]
        ksw.setdefault(cond, {})[k] = {
            "n_seeds": len(rr), "live": [x["width"] - x["n_dead"] for x in rr],
            **{w: {"slope": r(np.mean([x["decoder_slope"][w] for x in rr]), 3),
                   "slope_seed_spread": r(np.ptp([x["decoder_slope"][w] for x in rr]), 3),
                   "gamma": r(np.mean([x["gamma"][w] for x in rr]), 3)} for w in WINS},
            "fvu": r(np.mean([x["fvu_eval"] for x in rr]), 3)}
N["k_sweep_L13"] = ksw
kk = np.array(ks, float)
for w in WINS:
    s = [ksw["real"][k][w]["slope"] for k in ks]
    a, b, r2 = linfit(np.log2(kk), s)
    N.setdefault("k_sweep_real_slope_fit", {})[w] = {"a": r(a, 3), "b_per_doubling": r(b, 3), "r2": r(r2, 3)}
    sel = kk >= 8
    a, b, r2 = linfit(np.log2(kk[sel]), np.array(s)[sel])
    N.setdefault("k_sweep_real_slope_fit_k8plus", {})[w] = {"b_per_doubling": r(b, 3), "r2": r(r2, 3)}
N["k_sweep_training_tokens"] = int(next(x["n_train_tokens"] for x in runs if x["group"] == "saes_gpu_k"))

# --------------------------------------------------------------------------------------------------------------
# 6. Live-count bias (results/review/live_count.json)
lc = load("review/live_count.json")
grid = {}
for key, v in lc["grid"].items():
    g, n = key.split("|")
    grid.setdefault(g, {})[int(n)] = {w: (r(v[w][0], 3) if v[w] is not None else None) for w in WINS}
N["live_count_grid"] = grid
N["live_count_sd_max"] = r(max(v[w][1] for v in lc["grid"].values() for w in WINS if v[w] is not None), 4)
N["live_count_headline"] = {"measured_k1": r(lc["headline"]["measured_k1_ranks11_150"], 3),
                            "corrected_k1": r(lc["headline"]["corrected_k1_ranks11_150"], 2),
                            "true1_at_672": r(lc["headline"]["true1_measures_at_672"][0], 3)}
kc = {}
for key, v in lc["k_sweep_corrected"].items():
    if "|" in key:
        kc[key] = {"live": v.get("live_mean"), "measured": r(v["(10, 150)"]["measured"], 3),
                   "corrected": r(v["(10, 150)"]["corrected"], 2)}
N["live_count_k_corrected"] = kc
ksreal = [k for k in ks]
meas = [kc[f"real|{k}"]["measured"] for k in ksreal]
corr = [kc[f"real|{k}"]["corrected"] for k in ksreal]
N["live_count_fit_measured"] = dict(zip(["a", "b", "r2"], [r(x, 3) for x in linfit(np.log2(kk), meas)]))
N["live_count_fit_corrected"] = dict(zip(["a", "b", "r2"], [r(x, 3) for x in linfit(np.log2(kk), corr)]))

# --------------------------------------------------------------------------------------------------------------
# 7. Sigma estimation: token count and split halves
sig = load("review/review2_checks.json")["sigma"]["12"]
N["sigma12_split"] = {w: {"even": r(sig["even_docs"][w], 3), "odd": r(sig["odd_docs"][w], 3),
                          "all": r(sig["all"][w], 3), "first200k": r(sig["first200k"][w], 3),
                          "halves_diff": r(abs(sig["even_docs"][w] - sig["odd_docs"][w]), 4),
                          "first200k_minus_all": r(sig["first200k"][w] - sig["all"][w], 3)} for w in WINS}
d = {}
for w in WINS:
    a1 = np.array(s1m_all["sigma_slope"]["1m"][w])
    a2 = np.array(s1m_all["sigma_slope"]["200k"][w])
    d[w] = {"mean_1m_minus_200k": r((a1 - a2).mean(), 3), "max_abs_layer_diff": r(np.abs(a1 - a2).max(), 3),
            "argmax_layer": int(np.abs(a1 - a2).argmax())}
N["sigma_1m_vs_200k"] = d
N["sigma_n_tokens"] = s1m_all["n_tokens"]
prof_r = {}
for k, v in s1m_all["profile"].items():
    prof_r[k] = {"r_1m": r(v["1m"]["pearson_r"], 3), "r_200k": r(v["200k"]["pearson_r"], 3)}
N["profile_r_1m_vs_200k"] = prof_r
N["profile_r_1m_range"] = [min(v["r_1m"] for v in prof_r.values()), max(v["r_1m"] for v in prof_r.values())]
N["profile_r_200k_range"] = [min(v["r_200k"] for v in prof_r.values()), max(v["r_200k"] for v in prof_r.values())]
pf = s1m_all["profile_features"]["1m"]
N["sigma_profile_1m"] = {k: (r(v, 3) if isinstance(v, float) else v) for k, v in pf.items()}

# activation slope against number of tokens (Li protocol file)
lp = load("review/li_protocol.json")
tc = {}
for n_, v in lp["L12_token_count"].items():
    if "random" in v:
        tc[int(n_)] = {w: r(np.mean([d_["raw_centred"][w] for d_ in v["random"]
                                     if d_["raw_centred"][w] is not None]), 3)
                       if any(d_["raw_centred"][w] is not None for d_ in v["random"]) else None
                       for w in ["ranks1-100", "ranks11-150"]}
N["act_slope_vs_tokens_L12"] = tc

# --------------------------------------------------------------------------------------------------------------
# 8. Cross-layer correlation: effective sample size
lr = load("review/layer_r_ci.json")["profile"]
cells = {}
for k, v in lr.items():
    p = v["pearson"]
    if p["n"] != 26:
        continue
    cells[k] = {"r": r(p["r"], 3), "n_eff": r(p["n_eff"], 1), "p_neff": r(p["fisher_neff"]["p"], 3),
                "ci_neff": [r(x, 2) for x in p["fisher_neff"]["ci95"]],
                "p_phase": r(p.get("phase_surrogate", {}).get("p", float("nan")), 4) if isinstance(
                    p.get("phase_surrogate"), dict) else None}
N["layer_r_cells"] = cells
N["n_eff_range"] = [min(v["n_eff"] for v in cells.values()), max(v["n_eff"] for v in cells.values())]

# --------------------------------------------------------------------------------------------------------------
# 9. Activations vs decoder under Li-style protocols
L12s = lp["L12_1M"]["spectra"]
N["li_protocol_L12_1M"] = {p: {w: r(L12s[p]["slopes"][w], 3) for w in ["ranks1-100", "ranks11-150",
                                                                       "ranks31-450", "ranks101-1000"]}
                           for p in L12s}
N["li_decoder_L12_65k_L072"] = {w: r(v, 3) for w, v in lp["decoder_L12_65k_L072"]["centred"].items()}
N["li_codes"] = {p: {w: r(v, 3) for w, v in lp["L12_sae_codes"]["slopes"][p].items()}
                 for p in lp["L12_sae_codes"]["slopes"]}
summ = lp["layer_summary"]
cnt = {}
for row in summ:
    key = f'{row["act"]}|{row["window"]}'
    c = cnt.setdefault(key, {"n": 0, "act_steeper": 0, "gamma_min": 9, "gamma_max": -9})
    c["n"] += 1
    c["act_steeper"] += int(not row["act_flatter"])
    c["gamma_min"] = min(c["gamma_min"], row["gamma"])
    c["gamma_max"] = max(c["gamma_max"], row["gamma"])
for c in cnt.values():
    c["gamma_min"], c["gamma_max"] = r(c["gamma_min"], 2), r(c["gamma_max"], 2)
N["li_layer_summary_32k"] = cnt
# the 1M Sigma against the Gemma Scope 16k decoders at matched L0: steeper at every layer?
for cell in ["16k_L040", "65k_L040", "16k_L070", "65k_L070"]:
    for w in WINS:
        gl = s1m_all["profile"][f"{cell}_{WKEY[w]}"]["1m"]["gamma_by_layer"]
        vals = list(gl.values())
        N.setdefault("gamma_1m_by_cell", {})[f"{cell}_{WKEY[w]}"] = {"min": r(min(vals), 3), "max": r(max(vals), 3),
                                                                      "n_below_1": sum(v < 1 for v in vals)}
ms = s1m_all["midstack_65k_L040"]
N["midstack_65k_L040_gamma_1m"] = {k: [r(x, 2) for x in v["1m"]] for k, v in ms.items()}

# --------------------------------------------------------------------------------------------------------------
# 10. What the exponent does not reflect
nd = load("test2/neardup_L12.json")["results"]
dd = {"n": 0, "max_removed_frac": 0, "max_removed_sae": None, "max_abs_dslope": {w: 0 for w in WINS}}
nn05 = []
for k, v in nd.items():
    if "dedup" not in v or "0.5" not in v["dedup"]:
        continue
    dd["n"] += 1
    nkept = v["dedup"]["0.5"]["n_kept"]
    fr = 1 - nkept / v["n_query"]
    if fr > dd["max_removed_frac"]:
        dd["max_removed_frac"], dd["max_removed_sae"] = fr, k
    for i, w in enumerate(WINS):
        base = wslope(Wd[k], w)
        dd["max_abs_dslope"][w] = max(dd["max_abs_dslope"][w], abs(v["dedup"]["0.5"]["windows"][i]["slope"] - base))
    nn05.append(v["frac_nn_above"]["0.5"])
dd["max_removed_frac"] = r(dd["max_removed_frac"], 3)
dd["max_abs_dslope"] = {w: r(x, 4) for w, x in dd["max_abs_dslope"].items()}
dd["frac_nn_above_0.5_range"] = [r(min(nn05), 3), r(max(nn05), 3)]
N["near_duplicates"] = dd

ml = load("test1/massive_latents_L12.json")
mm = {"n": 0, "max_abs_dslope": 0, "max_removed_per_dim": 0, "max_abs_cos_range": [1, 0]}
for k, v in ml["results"].items():
    mm["n"] += 1
    for dim in ["massive_1170", "massive_334"]:
        e = v["dims"][dim]
        mm["max_abs_dslope"] = max(mm["max_abs_dslope"], abs(e["slope_rm_0.2"] - v["slope"]))
        mm["max_removed_per_dim"] = max(mm["max_removed_per_dim"], e["n_above_0.2"])
        mm["max_abs_cos_range"] = [min(mm["max_abs_cos_range"][0], e["max_abs_cos"]),
                                   max(mm["max_abs_cos_range"][1], e["max_abs_cos"])]
mm["max_abs_dslope"] = r(mm["max_abs_dslope"], 4)
mm["max_abs_cos_range"] = [r(x, 2) for x in mm["max_abs_cos_range"]]
mm["control_dim_max_abs_cos"] = r(max(v["dims"]["control_100"]["max_abs_cos"] for v in ml["results"].values()), 2)
N["massive_latents"] = mm

base = [x for x in runs if x["group"] == "saes_gpu_p2" and x["layer"] == 13 and x["cond"] == "real"
        and x["width"] == 16384]
dm = [x for x in runs if x["group"] == "saes_gpu_dm"]
mt = {}
for frac_ in [None, 0.001, 0.01]:
    rr = base if frac_ is None else [x for x in dm if x["drop_frac"] == frac_]
    mt[str(frac_)] = {"n": len(rr), **{w: r(np.mean([x["decoder_slope"][w] for x in rr]), 3) for w in WINS},
                      "fvu": r(np.mean([x["fvu_eval"] for x in rr]), 3)}
N["massive_tokens_L13"] = mt
N["massive_tokens_max_abs_dslope"] = r(max(abs(mt[f][w] - mt["None"][w]) for f in ["0.001", "0.01"] for w in WINS), 3)

fq = load("phase1/freqs_L12.json")["results"]
fe, es, fe_mid = [], [], []
for k, v in fq.items():
    fe.append(v["freq_fits"][3]["exponent"])
    fe_mid.append(v["freq_fits"][0]["exponent"])
    es.append(v["windows_alive"][1]["slope"])
N["freq"] = {"n": len(fe), "full_exponent_range": [r(min(fe), 2), r(max(fe), 2)],
             "ranks10_100_exponent_range": [r(min(fe_mid), 2), r(max(fe_mid), 2)],
             "eig_slope_range": [r(min(es), 3), r(max(es), 3)],
             "pearson_full_vs_slope": r(np.corrcoef(fe, es)[0, 1], 2),
             "pearson_10_100_vs_slope": r(np.corrcoef(fe_mid, es)[0, 1], 2)}
fa, fb, fr2 = linfit(np.log2([v["l0_measured"] for v in fq.values()]), fe)
N["freq"]["full_exponent_vs_log2L0"] = {"b": r(fb, 3), "r2": r(fr2, 2)}

# --------------------------------------------------------------------------------------------------------------
# 11. Secondary: Sigma-only Gaussian null (results/review/causal_layers.json)
cl = load("review/causal_layers.json")["table"]
null = {}
for wd in ["16384", "65536"]:
    for w in WINS:
        for cond in ["real", "gaussian"]:
            null.setdefault(f"{wd}|{w}", {})[cond] = {L: r(cl[f"{wd}|{cond}|{w}"][L]["raw_slope"][0], 3)
                                                      for L in ["3", "13", "23"]}
        null[f"{wd}|{w}"]["sd_max"] = r(max(cl[f"{wd}|{c}|{w}"][L]["raw_slope"][1] for c in ["real", "gaussian"]
                                           for L in ["3", "13", "23"]), 3)
N["sigma_only_null"] = null
close = []
for cell, v in null.items():
    for L in ["3", "13", "23"]:
        if abs(v["real"][L] - v["gaussian"][L]) <= 0.03:
            close.append(f"{cell}|L{L}|diff={abs(v['real'][L] - v['gaussian'][L]):.3f}")
N["sigma_only_null_cells_within_0.03"] = {"n_cells": 24, "n_close": len(close), "cells": close}
ratio = [v["gaussian"][L] / v["real"][L] for cell, v in null.items() if "(30, 450)" in cell or "(100, 1000)" in cell
         for L in ["3", "13", "23"]]
N["sigma_only_null_bulk_ratio_range"] = [r(min(ratio), 2), r(max(ratio), 2)]

# per-layer L0 dependence (16k ladders, ranks 11-150 and 31-450)
pl = {}
for w in WINS:
    bs = []
    for L_ in range(26):
        kk_ = [k for k in keys if saes[k]["layer"] == L_ and saes[k]["width"] == "16k"]
        a_, b_, r2_ = linfit(np.log2([saes[k]["l0"] for k in kk_]), [saes[k][w] for k in kk_])
        bs.append(b_)
    pl[w] = {"min": r(min(bs), 3), "max": r(max(bs), 3), "argmin": int(np.argmin(bs)), "argmax": int(np.argmax(bs)),
             "median": r(float(np.median(bs)), 3)}
N["per_layer_L0_slope_16k"] = pl
for wd in ["16k", "65k"]:
    kk_ = [k for k in keys if saes[k]["layer"] == 13 and saes[k]["width"] == wd]
    a_, b_, r2_ = linfit(np.log2([saes[k]["l0"] for k in kk_]), [saes[k]["(10, 150)"] for k in kk_])
    N.setdefault("L13_gemmascope_L0_slope_11_150", {})[wd] = {"b": r(b_, 3), "r2": r(r2_, 3), "n": len(kk_),
                                                              "l0": sorted(saes[k]["l0"] for k in kk_)}

# matrix test with the 1M Sigma (layer 12, 16k): appendix
mt1 = load("review/review2_checks.json")["matrix1m"]
N["matrix1m"] = [{"sae": e["sae"], "diag_real": r(e["real"]["diag_share"], 2),
                  "diag_exact": r(e["exact_sigma_gamma"]["diag_share"], 2),
                  "diag_random": r(e["random_basis"]["diag_share"], 2),
                  "overlap10_real": r(e["real"]["subspace_overlap"]["10"], 2),
                  "overlap10_exact": r(e["exact_sigma_gamma"]["subspace_overlap"]["10"], 2),
                  "r2_101_1000_real": r(e["real"]["fits"][2]["r2"], 3),
                  "r2_11_150_real": r(e["real"]["fits"][1]["r2"], 3),
                  "r2_random_max": r(max(f["r2"] for f in e["random_basis"]["fits"]), 3)} for e in mt1]

# covariance-matched dictionary control (layer 12, one seed, one draw): appendix
cs = load("phase3/control_summary.json")
kc_ = np.log2(cs["k"])
ctl = {}
for v in ["D", "G", "F", "H"]:
    dec = np.array(cs["variants"][v]["fixed"]["dec"])
    gam = np.array(cs["variants"][v]["fixed"]["gamma"])
    ctl[v] = {"sigma_slope": r(cs["variants"][v]["sigma_slope"], 3), "dec": [r(x, 3) for x in dec],
              "s_raw_8_128": r(linfit(kc_, dec)[1], 4), "s_raw_16_128": r(linfit(kc_[1:], dec[1:])[1], 4),
              "s_gamma_8_128": r(linfit(kc_, gam)[1], 4), "s_gamma_16_128": r(linfit(kc_[1:], gam[1:])[1], 4)}
N["control"] = ctl
N["control_k"] = cs["k"]
N["control_G_over_D"] = {"raw_8_128": r(ctl["G"]["s_raw_8_128"] / ctl["D"]["s_raw_8_128"], 3),
                         "raw_16_128": r(ctl["G"]["s_raw_16_128"] / ctl["D"]["s_raw_16_128"], 2),
                         "gamma_8_128": r(ctl["G"]["s_gamma_8_128"] / ctl["D"]["s_gamma_8_128"], 2),
                         "gamma_16_128": r(ctl["G"]["s_gamma_16_128"] / ctl["D"]["s_gamma_16_128"], 2)}
N["sigma_train_L3_13_23"] = {L: {w: r(v, 3) for w, v in load("review/causal_layers.json")["sigma"][L]["slopes"].items()}
                             for L in ["3", "13", "23"]}

# --------------------------------------------------------------------------------------------------------------
# 12. Window overlap
def ranks(w):
    a, b = (int(s) for s in w.strip("()").split(","))
    return set(range(a + 1, b + 1))


N["window_overlap"] = {f"{a} in {b}": len(ranks(a) & ranks(b)) for a in WINS for b in WINS if a != b}
N["window_size"] = {w: len(ranks(w)) for w in WINS}

json.dump(N, open(OUT / "thesisB_numbers.json", "w"), indent=1, default=str)
if __name__ == "__main__" and "--figs" not in sys.argv:
    print(json.dumps(N, indent=1, default=str)[:200])


# ==============================================================================================================
# Figures
# ==============================================================================================================
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "lines.markersize": 3.2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
WCOLW = {"(3, 50)": OI["sky"], "(10, 150)": OI["blue"], "(30, 450)": OI["orange"], "(100, 1000)": OI["verm"]}
L0COL = {20: OI["purple"], 40: OI["blue"], 70: OI["green"], 140: OI["orange"]}
WIDTHS = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
WCOL = dict(zip(WIDTHS, plt.get_cmap("viridis")(np.linspace(0.0, 0.88, len(WIDTHS)))))
WMARK = dict(zip(WIDTHS, ["o", "s", "^", "D", "v", "P", "X"]))
LI = ["layer_0/width_65k/average_l0_73", "layer_12/width_65k/average_l0_72", "layer_24/width_65k/average_l0_124"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


# ---- Figure 1: sparsity, layer, and a controlled k sweep (ranks 11-150) -------------------------------------
P = "(10, 150)"
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.15), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
ax = axs[0]
cmap = plt.get_cmap("viridis")
for k in keys:
    s = saes[k]
    mk = {"16k": "o", "65k": "s"}.get(s["width"], "D")
    ax.scatter(s["l0"], s[P], color=cmap(s["layer"] / 25), marker=mk, s=7, lw=0, alpha=0.9)
for k in LI:
    ax.scatter(saes[k]["l0"], saes[k][P], marker="*", s=70, facecolor="none", edgecolor="k", lw=0.8, zorder=5)
log2_axis(ax, [8, 16, 32, 64, 128, 256, 512])
ax.set_xlabel("L0 (named, log scale)")
ax.set_ylabel("decoder eigen-slope, ranks 11–150")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 25))
cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("layer", labelpad=1)
cb.ax.tick_params(labelsize=6)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="grey", ms=3, label="16k"),
                   Line2D([], [], marker="s", ls="", color="grey", ms=3, label="65k"),
                   Line2D([], [], marker="D", ls="", color="grey", ms=3, label="L12, 32k-1M"),
                   Line2D([], [], marker="*", ls="", mfc="none", mec="k", ms=7, label="Li et al.")],
          loc="lower right", handletextpad=0.2, borderaxespad=0.0, fontsize=5.8)
ax.set_ylim(-0.93, -0.08)
ax.set_title("a   288 Gemma Scope SAEs", loc="left")

ax = axs[1]
for L0v in [40, 70, 140]:
    key = f"65k_ranks11-150_L0_{L0v}"
    if key not in lm:
        continue
    v = {int(a): b for a, b in lm[key].items()}
    xs = sorted(v)
    ax.plot(xs, [v[x] for x in xs], "-o", color=L0COL[L0v], ms=2.2, label=f"L0 = {L0v}")
for k in LI:
    ax.scatter(saes[k]["layer"], saes[k][P], marker="*", s=70, color="k", zorder=5)
ax.scatter(24, li[P]["L24_at72_interp"], marker="*", s=70, facecolor="none", edgecolor="k", lw=0.8, zorder=5)
ax.annotate("", xy=(24, li[P]["L24_at72_interp"]), xytext=(24, li[P]["L24_at124"]),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="k"))
ax.set_xlabel("layer")
ax.set_ylabel("decoder eigen-slope, ranks 11–150")
ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.1)
ax.set_title("b   65k, interpolated to fixed L0", loc="left")

ax = axs[2]
for wd, mk in [("16k", "o"), ("65k", "s")]:
    pts = sorted((saes[k]["l0"], saes[k][P]) for k in keys if saes[k]["layer"] == 13 and saes[k]["width"] == wd)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=OI["grey"], ms=3, lw=0.8,
            mfc="white" if wd == "65k" else OI["grey"], label=f"Gemma Scope, {wd}")
xs, ms_ = [], []
for kx in ks:
    rr = [x for x in runs if x["group"] == "saes_gpu_k" and x["cond"] == "real" and x["k"] == kx]
    for x in rr:
        live = x["width"] - x["n_dead"]
        ax.scatter(kx, x["decoder_slope"][P], s=9, color=OI["blue"], marker="o",
                   facecolor="none" if live < 2000 else OI["blue"], lw=0.7, zorder=4)
    xs.append(kx)
    ms_.append(np.mean([x["decoder_slope"][P] for x in rr]))
ax.plot(xs, ms_, color=OI["blue"], lw=1.0, label="our TopK, 16k")
log2_axis(ax, [1, 4, 16, 64, 256])
ax.set_xlabel("L0 or k (log scale)")
ax.set_ylabel("decoder eigen-slope, ranks 11–150")
ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.1)
ax.set_ylim(-1.25, -0.2)
ax.set_title("c   layer 13: changing only k", loc="left")
fig.tight_layout(w_pad=0.6)
save(fig, "thesisB_fig1_sparsity_layer")

# ---- Figure 2: the window decides what the layer profile looks like; width barely matters in the top -------
fig, axs = plt.subplots(2, 4, figsize=(6.75, 3.5), sharex="row")
for j, w in enumerate(WINS):
    ax = axs[0, j]
    for L0v in [40, 70, 140]:
        v = {int(a): b for a, b in lm[f"16k_{WKEY[w]}_L0_{L0v}"].items()}
        xs = sorted(v)
        ax.plot(xs, [v[x] for x in xs], "-", color=L0COL[L0v], label=f"L0 = {L0v}")
    ax.set_title(WLAB[w].replace("ranks", "ranks"), loc="left")
    ax.set_xlabel("layer")
    if j == 0:
        ax.set_ylabel("slope, 16k, fixed L0")
        ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.1)
    ax.set_ylim(-0.82, -0.15)
    ax = axs[1, j]
    for wd in WIDTHS:
        pts = sorted((row["l0"], row["gamma"][w] * s200[w]) for row in rows if row["width"] == wd)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=WMARK[wd], color=WCOL[wd], ms=2.6, lw=0.7,
                    label=wd)
    xx = np.array([10, 500])
    ax.plot(xx, l12[w]["slope_a"] + l12[w]["slope_b_per_doubling"] * np.log2(xx), "k--", lw=0.6)
    log2_axis(ax, [16, 64, 256])
    ax.set_xlabel("L0 (measured, log scale)")
    ax.set_ylim(-0.82, -0.15)
    ax.text(0.97, 0.05, f"{l12[w]['slope_b_per_doubling']:+.3f}/doubling\n$R^2$ = {l12[w]['slope_r2']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    if j == 0:
        ax.set_ylabel("slope, layer 12")
        ax.legend(loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1, fontsize=5.6)
fig.tight_layout(h_pad=0.8, w_pad=0.5)
save(fig, "thesisB_fig2_windows")

# ---- Figure 3: estimator properties -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.95), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
ax = axs[0]
ns = sorted(grid["0"])
gcol = {"0": OI["grey"], "0.5": OI["green"], "1.0": OI["blue"], "1.5": OI["verm"]}
for g in ["0", "0.5", "1.0", "1.5"]:
    ax.plot(ns, [grid[g][n_][P] for n_ in ns], "-o", ms=2.2, color=gcol[g])
    ax.axhline(float(g), color=gcol[g], lw=0.5, ls=":")
    ax.text(20000, grid[g][16384][P], f"true {float(g):.1f}", color=gcol[g], fontsize=5.6, va="center")
ax.axvline(672, color="k", lw=0.5, ls="--")
ax.set_xscale("log")
ax.set_xticks([1000, 4000, 16000])
ax.set_xticklabels(["1k", "4k", "16k"])
ax.minorticks_off()
ax.set_xlabel("points in the cloud")
ax.set_ylabel(r"measured $\gamma$, ranks 11–150")
ax.set_ylim(-0.05, 1.6)
ax.set_xlim(500, 60000)
ax.set_title("a   live-count bias", loc="left")

ax = axs[1]
nt = sorted(tc)
ax.plot(nt, [tc[n_]["ranks1-100"] for n_ in nt], "-o", ms=2.2, color=OI["blue"], label="activations, ranks 1–100")
ax.axhline(N["li_decoder_L12_65k_L072"]["ranks1-100"], color=OI["verm"], lw=0.8, ls="--",
           label="decoder (65k, L0 72)")
ax.axhline(N["li_protocol_L12_1M"]["raw_centred"]["ranks1-100"], color=OI["blue"], lw=0.5, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks([256, 2048, 16384, 65536])
ax.set_xticklabels(["256", "2k", "16k", "64k"])
ax.minorticks_off()
ax.set_xlabel("tokens (random draws)")
ax.set_ylabel("eigen-slope, layer 12")
ax.set_ylim(-1.0, -0.4)
ax.legend(loc="center right", handlelength=1.4, borderaxespad=0.0, fontsize=5.6)
ax.set_title(r"b   $\Sigma$ slope vs tokens", loc="left")

ax = axs[2]
for w in WINS:
    a1 = np.array(s1m_all["sigma_slope"]["1m"][w])
    a2 = np.array(s1m_all["sigma_slope"]["200k"][w])
    ax.plot(range(26), a1 - a2, "-", color=WCOLW[w], label=WLAB[w].replace("ranks ", ""))
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("layer")
ax.set_ylabel(r"$\Sigma$ slope change, 1M $-$ 200k")
ax.set_ylim(-0.06, 0.32)
ax.legend(loc="upper center", handlelength=1.0, borderaxespad=0.0, fontsize=5.6, ncol=2, columnspacing=0.5)
ax.set_title(r"c   $\Sigma$ estimate", loc="left")

ax = axs[3]
order = [(wd, L0v, w) for L0v in [40, 70, 140] for wd in ["16k", "65k"] for w in WINS
         if f"{wd}_{WKEY[w]}_L0_{L0v}" in cells]
for i, (wd, L0v, w) in enumerate(order):
    c = cells[f"{wd}_{WKEY[w]}_L0_{L0v}"]
    y = -i
    ax.plot(c["ci_neff"], [y, y], color=WCOLW[w], lw=0.9)
    ax.plot(c["r"], y, "o" if wd == "16k" else "s", color=WCOLW[w], ms=2.6)
ax.axvline(0, color="k", lw=0.5)
ax.set_yticks([-i for i in range(len(order))])
ax.set_yticklabels([f"{wd} L0 {L0v}" if j_ % 4 == 0 else "" for j_, (wd, L0v, w) in enumerate(order)],
                   fontsize=5.4)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.7, 1.02)
ax.set_xlabel(r"Pearson $r$, 95% CI at $n_{\rm eff}$")
ax.set_title("d   decoder vs $\\Sigma$ across layers", loc="left")
fig.tight_layout(w_pad=0.4)
save(fig, "thesisB_fig3_estimator")
print("figures written")
