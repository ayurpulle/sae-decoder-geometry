"""Figures, tables and numbers for the round-2 null comparisons (Gemma-2-2B 8 layers, layer-13 k sweep, whitened
baseline at layer 12, the codable sparse null, GPT-2 small, Gemma Scope 9b).

Reads only results/r2/runs_real.jsonl, runs_models.jsonl, runs_sparse2.jsonl and runs_robust.jsonl (rows deduplicated
by `run`, the last row kept), results/r2/sparse2_calibration.json, robust_calibration.json, results/r2/dec/*.npz, results/r2/sigma/*.npz, results/test1/sigma_eigenvalues_1m.npy
and results/gpu/runs.json (the three-layer set). Every quantity is recomputed from the rows here, independently of
experiments/r2_summary.py; the pre-registered verdicts (i)-(iii), (v) with the codable sparse null, GPT-2's R1-R3
and Gemma-2-9b's (a)-(c) are recomputed with the rules in the docstrings of experiments/r2_job_real.py,
r2_job_sparse2.py and r2_job_models.py and checked against results/r2/summary.json.

Writes
  paper/figures/fig8_null_layers_r2.{pdf,png}   8-layer profile, real vs Gaussian null, k=40, 3 seeds
  paper/figures/fig9_k_sweep_r2.{pdf,png}       layer-13 k sweep, real and Gaussian 3 seeds, sparse null 2 seeds
  paper/figures/fig10_gpt2_r2.{pdf,png}         GPT-2 small, 12 layers (k=32) and the layer-6 k sweep
  paper/figures/tab_null8_r2.tex                8-layer table (raw slopes, FVU, live counts)
  paper/figures/tab_sparse2_r2.tex              8-layer gamma, real / sparse null / Gaussian null, rho and FVU
  paper/figures/tab_whiten_r2.tex               whitened baseline
  paper/figures/fig11_matched_null_r2.{pdf,png}  main-text figure: real vs matched sparse null (layers; k at layer 13)
  paper/figures/tab_matched_r2.tex              main-text table: real vs matched sparse null with the 11-150 interval
  paper/figures/tab_robust_r2.tex               main-text table: generator settings of the sparse null (layers 13, 0)
  paper/figures/fig12_gpt2_codable_r2.{pdf,png} GPT-2 small: codable sparse null minus real, 12 layers
  paper/figures/fig13_pmatch_r2.{pdf,png}       main-text figure: bulk gap, real / equal-power / power-matched null
  paper/figures/tab_pmatch_r2.tex               main-text table: the power-matched null (results/r2/runs_pmatch.jsonl)
  results/r2/paper_numbers_r2.json              every number the paper quotes from these runs
Run from anywhere:  python paper/figures/make_fig_null_r2.py

Conventions: windows are 0-based slices of the sorted eigenvalues; for GPT-2 (d=768) each Gemma window maps to the
scaled window of r2_job_models.py. "raw" = decoder eigen-slope; gamma = raw / Sigma slope (training-set Sigma).
Primary scoring is the fixed live count (8,192 random live latents, mean of 4 draws); "full" uses all live rows.
D(L) = null raw slope - real raw slope (slopes are negative, so D > 0 means real is steeper than its null).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import OI, RES, W2, WINDOWS, WIN_LABEL, log2_axis, panel_tag, plt, save  # noqa: E402
from geometry import slope_from_eigenvalues  # noqa: E402

LAYERS = [0, 3, 6, 9, 13, 17, 21, 24]
INTERIOR, ENDS = [6, 9, 13, 17], [0, 3, 21, 24]
TOP, BULK = WINDOWS[:2], WINDOWS[2:]
KS = [8, 16, 24, 32, 40, 48, 64, 128]
K_FIT = [16, 24, 32, 48, 64, 128]  # pre-registered fit range for (i); k=40 comes from Q1 and is not in the fit
GPT_WIN = {"(3, 50)": "(1, 17)", "(10, 150)": "(3, 50)", "(30, 450)": "(10, 150)", "(100, 1000)": "(33, 333)"}
GPT_LABEL = {"(3, 50)": "ranks 2–17", "(10, 150)": "ranks 4–50", "(30, 450)": "ranks 11–150",
             "(100, 1000)": "ranks 34–333"}
COL = {"real": OI["blue"], "gaussian": OI["verm"], "sparse2": OI["green"]}
LAB = {"real": "real activations", "gaussian": r"Gaussian $N(\mu,\Sigma)$ null",
       "sparse2": r"equal-power null (FVU matched at $k=40$)"}


def rows(name):
    d = {}
    for line in open(RES / "r2" / name):
        r = json.loads(line)
        d[r["run"]] = r  # last row per run wins
    return list(d.values())


REAL = rows("runs_real.jsonl")
MOD = rows("runs_models.jsonl")
N = {}


def msd(x):
    x = np.asarray(x, float)
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    return float(np.mean(x)), sd, sd / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def pick(rs, **kw):
    return [r for r in rs if all(r.get(k) == v for k, v in kw.items())]


def fit(x, y):
    return float(np.polyfit(np.log2(x), y, 1)[0])


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


# ----------------------------------------------------------------------------------------------- 8 layers at k=40
def layers8():
    out = {}
    for L in LAYERS:
        o = {}
        for c in ("real", "gaussian"):
            rs = [r for r in pick(REAL, step="Q1", layer=L, cond=c, k=40)]
            assert len(rs) == 3, (L, c, len(rs))
            o[c] = {"seeds": sorted(r["seed"] for r in rs),
                    "fvu": msd([r["fvu_eval"] for r in rs]), "n_live": [min(r["n_live"] for r in rs),
                                                                       max(r["n_live"] for r in rs)],
                    "raw": {w: msd([r["dec_fixed"][w] for r in rs]) for w in WINDOWS},
                    "raw_full": {w: msd([r["dec_full"][w] for r in rs]) for w in WINDOWS},
                    "gamma": {w: msd([r["gamma_fixed"][w] for r in rs]) for w in WINDOWS},
                    "sigma": {w: rs[0]["sigma_slope"][w] for w in WINDOWS},
                    "_per_seed_raw": {w: [r["dec_fixed"][w] for r in rs] for w in WINDOWS}}
        o["D"] = {}
        for w in WINDOWS:
            mR, _, seR = o["real"]["raw"][w]
            mG, _, seG = o["gaussian"]["raw"][w]
            o["D"][w] = (mG - mR, float(np.hypot(seR, seG)))
        o["ratio_gauss_over_real"] = {w: o["gaussian"]["raw"][w][0] / o["real"]["raw"][w][0] for w in WINDOWS}
        out[L] = o
    # (ii) pre-registered
    verd = {}
    for w in TOP:
        D = {L: out[L]["D"][w] for L in LAYERS}
        Ls = max(LAYERS, key=lambda L: D[L][0])
        if Ls in INTERIOR and all(D[Ls][0] - D[e][0] > 2 * np.hypot(D[Ls][1], D[e][1]) for e in ENDS):
            v = "supports"
        elif max(D[e][0] for e in ENDS) >= max(D[i][0] for i in INTERIOR):
            v = "contradicts"
        else:
            v = "partial"
        margin = min(D[Ls][0] - D[e][0] for e in ENDS)
        verd[w] = {"verdict": v, "argmax": Ls, "min_margin_over_ends": margin,
                   "margin_in_se": min((D[Ls][0] - D[e][0]) / np.hypot(D[Ls][1], D[e][1]) for e in ENDS)}
    for w in BULK:
        D = {L: out[L]["D"][w] for L in LAYERS}
        if all(-D[L][0] > 2 * D[L][1] for L in LAYERS):
            v = "supports"
        elif any(D[L][0] > 2 * D[L][1] for L in LAYERS):
            v = "contradicts"
        else:
            v = "partial"
        verd[w] = {"verdict": v}
    # post hoc, reported: correlation of each condition's raw slope with Sigma's slope across the 8 layers
    corr = {}
    for w in WINDOWS:
        s = [out[L]["real"]["sigma"][w] for L in LAYERS]
        corr[w] = {c: float(np.corrcoef(s, [out[L][c]["raw"][w][0] for L in LAYERS])[0, 1]) for c in ("real", "gaussian")}
        corr[w]["range"] = {c: float(np.ptp([out[L][c]["raw"][w][0] for L in LAYERS])) for c in ("real", "gaussian")}
        corr[w]["range"]["sigma"] = float(np.ptp(s))
    seed_sd_max = {c: max(out[L][c]["raw"][w][1] for L in LAYERS for w in WINDOWS) for c in ("real", "gaussian")}
    N["layers8"] = {"by_layer": {L: {k: v for k, v in o.items() if not k.startswith("_")} for L, o in out.items()},
                    "ii": verd, "sigma_corr_posthoc": corr, "seed_sd_max_raw": seed_sd_max,
                    "fvu_range": {c: [min(out[L][c]["fvu"][0] for L in LAYERS), max(out[L][c]["fvu"][0] for L in LAYERS)]
                                  for c in ("real", "gaussian")},
                    "bulk_ratio_range": {w: [min(out[L]["ratio_gauss_over_real"][w] for L in LAYERS),
                                             max(out[L]["ratio_gauss_over_real"][w] for L in LAYERS)] for w in WINDOWS},
                    "top_real_steeper_layers": {w: [L for L in LAYERS if out[L]["D"][w][0] > 0] for w in TOP},
                    "gamma_range": {c: {w: [min(out[L][c]["gamma"][w][0] for L in LAYERS),
                                            max(out[L][c]["gamma"][w][0] for L in LAYERS)] for w in WINDOWS}
                                    for c in ("real", "gaussian")},
                    # share of the decoder covariance on the diagonal of Sigma's eigenbasis, per SAE (all 48)
                    "diag_share_range": {c: [min(r["basis_gamma"]["diag_share"] for r in pick(REAL, step="Q1", cond=c)),
                                             max(r["basis_gamma"]["diag_share"] for r in pick(REAL, step="Q1", cond=c))]
                                         for c in ("real", "gaussian")}}
    return out


def fig_layers(out):
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for j, w in enumerate(WINDOWS):
        ax = axes[0, j]
        for c in ("real", "gaussian"):
            m = [out[L][c]["raw"][w][0] for L in LAYERS]
            sd = [out[L][c]["raw"][w][1] for L in LAYERS]
            for L in LAYERS:
                ax.plot([L] * 3, out[L][c]["_per_seed_raw"][w], ".", ms=1.6,
                        color=COL[c], alpha=0.5)
            ax.errorbar(LAYERS, m, yerr=sd, color=COL[c], marker="o", ms=2.6, capsize=1.2, elinewidth=0.6,
                        lw=1.0, label=LAB[c])
        ax.plot(LAYERS, [out[L]["real"]["sigma"][w] for L in LAYERS], "*--", color=OI["grey"], ms=3.5, lw=0.7,
                label=r"$\Sigma$ (10M tokens)")
        ax.set_title(WIN_LABEL[w], fontsize=7.5)
        if j == 0:
            ax.set_ylabel("decoder eigen-slope")
        panel_tag(ax, "abcd"[j], x=-0.08)
        ax2 = axes[1, j]
        D = [out[L]["D"][w][0] for L in LAYERS]
        se = [out[L]["D"][w][1] for L in LAYERS]
        ax2.axhline(0, color="k", lw=0.5)
        ax2.errorbar(LAYERS, D, yerr=[2 * s for s in se], color=OI["black"], marker="s", ms=2.4, capsize=1.2,
                     elinewidth=0.6, lw=0.9)
        ax2.axvspan(5, 18, color=OI["grey"], alpha=0.12, lw=0)
        ax2.set_xlabel("layer")
        ax2.set_xticks(LAYERS)
        ax2.set_xticklabels([str(L) for L in LAYERS], fontsize=5.8)
        if j == 0:
            ax2.set_ylabel(r"$D$ = null $-$ real")
        panel_tag(ax2, "efgh"[j], x=-0.08)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=6.8)
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    save(fig, "fig8_null_layers_r2")


def table_layers(out):
    lines = []
    for c, name in (("real", "real"), ("gaussian", "Gaussian")):
        lines.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{name}}}}} \\")
        for L in LAYERS:
            o = out[L][c]
            lv = o["n_live"]
            live = f"{lv[0]:,}" if lv[0] == lv[1] else f"{lv[0]:,}--{lv[1]:,}"
            cells = [str(L)] + [f"${o['raw'][w][0]:.3f}$" for w in WINDOWS] + \
                    [f"{max(o['raw'][w][1] for w in WINDOWS):.3f}", f"{o['fvu'][0]:.3f}", live]
            if c == "real":
                cells += [f"${o['sigma'][w]:.3f}$" for w in (WINDOWS[1], WINDOWS[3])]
            else:
                cells += ["", ""]
            lines.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{tabular}{r rrrr r r r rr}
\toprule
layer & 4--50 & 11--150 & 31--450 & 101--1000 & max sd & FVU & live & $\Sig$ 11--150 & $\Sig$ 101--1000 \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_null8_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_real.jsonl\n" + tex)


# ----------------------------------------------------------------------------------------------- k sweep, layer 13
def ksweep():
    cur = {}
    for c in ("real", "gaussian"):
        cur[c] = {}
        for k in KS:
            rs = pick(REAL, layer=13, cond=c, k=k, whitened=False)
            rs = [r for r in rs if r["step"] in ("Q1", "Q2")]
            assert len(rs) == 3, (c, k, len(rs))
            cur[c][k] = {r["seed"]: r for r in rs}
    res = {"curves": {}, "i": {}, "fvu": {}, "n_live": {}, "short": []}
    for c in cur:
        res["curves"][c] = {w: {k: msd([r["gamma_fixed"][w] for r in cur[c][k].values()]) for k in KS} for w in WINDOWS}
        res["fvu"][c] = {k: msd([r["fvu_eval"] for r in cur[c][k].values()])[0] for k in KS}
        res["n_live"][c] = {k: [min(r["n_live"] for r in cur[c][k].values()),
                                max(r["n_live"] for r in cur[c][k].values())] for k in KS}
        res["short"] += [r["run"] for k in KS for r in cur[c][k].values() if r["fixed_n_short"]]

    def seedfit(c, ks, unit, w):
        key = "gamma_fixed" if unit == "gamma" else "dec_fixed"
        return [fit(ks, [cur[c][k][s][key][w] for k in ks]) for s in (0, 1, 2)]

    verdicts = {}
    for w in WINDOWS:
        x = {}
        for unit in ("gamma", "raw"):
            a, b = seedfit("real", K_FIT, unit, w), seedfit("gaussian", K_FIT, unit, w)
            (mR, sdR, seR), (mG, sdG, seG) = msd(a), msd(b)
            loo = []
            for d in K_FIT:
                kk = [k for k in K_FIT if k != d]
                loo.append((float(np.mean(seedfit("real", kk, unit, w))), float(np.mean(seedfit("gaussian", kk, unit, w)))))
            sgn = 1.0 if mR >= 0 else -1.0
            steady = sgn * mG > 0 and abs(mG) > 2 * seG and all(sgn * g > 0 for _, g in loo)
            rat = mG / mR
            minloo = min(g / r for r, g in loo)
            sec = float(np.hypot(seR, seG))
            sup = steady and rat >= 0.75 and minloo >= 0.5
            con = (not steady) or (rat < 0.5 and sgn * (mR - mG) > 2 * sec)
            key = "gamma_fixed" if unit == "gamma" else "dec_fixed"
            step = {c: float(np.mean([cur[c][16][s][key][w] - cur[c][8][s][key][w] for s in (0, 1, 2)]))
                    for c in cur}
            s8 = {c: float(np.mean(seedfit(c, [8] + K_FIT, unit, w))) for c in cur}
            x[unit] = {"s_real": (mR, sdR), "s_gauss": (mG, sdG), "ratio": rat, "min_loo_ratio": minloo,
                       "loo_gauss": [g for _, g in loo], "steady_gauss": bool(steady), "supports": bool(sup),
                       "contradicts": bool(con), "step_8_16": step, "s_8_128": s8}
        v = ("supports" if x["gamma"]["supports"] and x["raw"]["supports"] else
             "contradicts" if x["gamma"]["contradicts"] and x["raw"]["contradicts"] else "partial")
        x["verdict"] = v
        verdicts[w] = v
        res["i"][w] = x
    res["i_window_verdicts"] = verdicts
    # real minus null gamma per k (seed means), reported
    res["real_minus_null_gamma"] = {w: {k: res["curves"]["real"][w][k][0] - res["curves"]["gaussian"][w][k][0]
                                        for k in KS} for w in WINDOWS}
    res["seed_sd_max_gamma"] = max(res["curves"][c][w][k][1] for c in cur for w in WINDOWS for k in KS)
    # post hoc: matched-FVU pairs (real k, Gaussian k) with |FVU difference| <= 0.03
    pairs = []
    for kr in KS:
        for kg in KS:
            if abs(res["fvu"]["real"][kr] - res["fvu"]["gaussian"][kg]) <= 0.03:
                pairs.append({"k_real": kr, "k_gauss": kg, "fvu_real": res["fvu"]["real"][kr],
                              "fvu_gauss": res["fvu"]["gaussian"][kg],
                              "gamma_real": {w: res["curves"]["real"][w][kr][0] for w in WINDOWS},
                              "gamma_gauss": {w: res["curves"]["gaussian"][w][kg][0] for w in WINDOWS}})
    res["matched_fvu_pairs_posthoc"] = pairs
    N["ksweep"] = res
    return cur, res


def fig_k(cur, res, cur_sp=None):
    fig, axes = plt.subplots(1, 5, figsize=(W2, 1.95), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]})
    for ax, w, tag in zip(axes, WINDOWS, "abcd"):
        if cur_sp is not None:  # codable sparse null, two seeds
            for k in KS:
                ax.plot([k] * len(cur_sp[k]), [r["gamma_fixed"][w] for r in cur_sp[k].values()], ".", ms=1.6,
                        color=COL["sparse2"], alpha=0.5)
            c = N["sparse2"]["k13"]["curves"][w]
            ax.errorbar(KS, [c[k][0] for k in KS], yerr=[c[k][1] for k in KS], color=COL["sparse2"], marker="^",
                        ms=2.4, capsize=1.0, elinewidth=0.6, lw=1.0, label=LAB["sparse2"])
        for c in ("real", "gaussian"):
            for k in KS:
                ax.plot([k] * 3, [r["gamma_fixed"][w] for r in cur[c][k].values()], ".", ms=1.6, color=COL[c],
                        alpha=0.5)
            m = [res["curves"][c][w][k][0] for k in KS]
            sd = [res["curves"][c][w][k][1] for k in KS]
            ax.errorbar(KS, m, yerr=sd, color=COL[c], marker="o", ms=2.4, capsize=1.0, elinewidth=0.6, lw=1.0,
                        label=LAB[c])
            if c == "real":  # real k=8 has fewer than 8,192 live latents: all live rows used
                ax.plot([8], [res["curves"][c][w][8][0]], "x", color="k", ms=4, mew=0.8)
        t = f"(i) Gauss. {res['i'][w]['verdict']}"
        if cur_sp is not None:
            t += f"\nsparse {N['sparse2']['k13']['i'][w]['verdict']}"
        ax.set_title(f"{WIN_LABEL[w]}\n{t}", fontsize=6.3)
        log2_axis(ax, [8, 16, 32, 64, 128])
        ax.set_xlabel("k")
        if tag == "a":
            ax.set_ylabel(r"$\gamma$, 8,192 live latents")
        panel_tag(ax, tag, x=-0.2, y=1.3)
    ax = axes[4]
    for c in ("real", "gaussian"):
        ax.plot(KS, [res["fvu"][c][k] for k in KS], "o-", color=COL[c], ms=2.4)
    if cur_sp is not None:
        ax.plot(KS, [N["sparse2"]["k13"]["fvu"][k] for k in KS], "^-", color=COL["sparse2"], ms=2.4)
    log2_axis(ax, [8, 16, 32, 64, 128])
    ax.set_xlabel("k")
    ax.set_ylabel("eval FVU")
    panel_tag(ax, "e", x=-0.2, y=1.3)
    h, l_ = axes[0].get_legend_handles_labels()
    order = [l_.index(x) for x in (LAB["real"], LAB["gaussian"], LAB["sparse2"]) if x in l_]
    fig.legend([h[i] for i in order], [l_[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, -0.1),
               ncol=3, fontsize=6.8)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig9_k_sweep_r2")


# ----------------------------------------------------------------------------------------------- whitened baseline
def whiten():
    out = {"by_k": {}, "verdict": {}}
    for k in (16, 40, 128):
        rw = pick(REAL, step="Q4", k=k, whitened=True)
        rc = pick(REAL, step="Q4", k=k, whitened=False)
        assert len(rw) == 2 and len(rc) == 2
        o = {"fvu_whitened": msd([r["fvu_eval"] for r in rw])[0], "fvu_control": msd([r["fvu_eval"] for r in rc])[0],
             "n_live_whitened": [r["n_live"] for r in rw]}
        for w in WINDOWS:
            gm = msd([r["gamma_fixed"][w] for r in rw])
            gi = float(np.mean([r["iso_null"]["gamma_fixed"][w] for r in rw]))
            o[w] = {"gamma_map": gm[0], "gamma_map_se": gm[2], "gamma_iso": gi, "delta": gm[0] - gi,
                    "A_w": float(np.mean([r["whitened_space"]["anisotropy_full"][w] for r in rw])),
                    "dec_whitened": float(np.mean([r["whitened_space"]["dec_full"][w] for r in rw])),
                    "iso_whitened": float(np.mean([r["whitened_space"]["isotropic_full"][w] for r in rw])),
                    "gamma_control": float(np.mean([r["gamma_fixed"][w] for r in rc]))}
        out["by_k"][k] = o
    for w in WINDOWS:
        d = [out["by_k"][k][w] for k in (16, 40, 128)]
        if all(abs(x["delta"]) <= 0.05 for x in d):
            v = "supports"
        elif all(abs(x["delta"]) > 0.05 and abs(x["delta"]) > 2 * x["gamma_map_se"] for x in d):
            v = "contradicts"
        else:
            v = "partial"
        out["verdict"][w] = v
    N["whiten"] = out
    lines = []
    for k in (16, 40, 128):
        o = out["by_k"][k]
        lines.append(" & ".join([str(k)] + [f"{o[w]['A_w']:+.3f}" for w in WINDOWS] +
                                [f"{o[w]['gamma_map']:.2f}/{o[w]['gamma_iso']:.2f}/{o[w]['gamma_control']:.2f}"
                                 for w in WINDOWS] + [f"{o['fvu_whitened']:.3f}", f"{o['fvu_control']:.3f}"]) + r" \\")
    tex = (r"""\begin{tabular}{r rrrr cccc rr}
\toprule
 & \multicolumn{4}{c}{whitened: slope $-$ isotropic slope} & \multicolumn{4}{c}{$\gamma$: mapped/isotropic mapped/unwhitened} & \multicolumn{2}{c}{FVU} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}
$k$ & 4--50 & 11--150 & 31--450 & 101--1000 & 4--50 & 11--150 & 31--450 & 101--1000 & whit. & unwh. \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_whiten_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_real.jsonl\n" + tex)
    return out


# ----------------------------------------------------------------------------------------------- GPT-2 small
def gpt2():
    m1 = [r for r in MOD if r.get("step") == "M1"]
    res = {"by_layer": {}, "k6": {}}
    for L in range(12):
        o = {}
        for c in ("real", "gaussian"):
            rs = pick(m1, layer=L, cond=c, k=32)
            o[c] = {"n_seeds": len(rs), "fvu": msd([r["fvu_eval"] for r in rs])[0],
                    "raw": {w: msd([r["dec_fixed"][GPT_WIN[w]] for r in rs])[0] for w in WINDOWS},
                    "raw_seeds": {w: [r["dec_fixed"][GPT_WIN[w]] for r in rs] for w in WINDOWS},
                    "sigma": {w: rs[0]["sigma_slope"][GPT_WIN[w]] for w in WINDOWS},
                    "n_live": [r["n_live"] for r in rs]}
        o["D"] = {w: o["gaussian"]["raw"][w] - o["real"]["raw"][w] for w in WINDOWS}
        o["ratio"] = {w: o["gaussian"]["raw"][w] / o["real"]["raw"][w] for w in WINDOWS}
        res["by_layer"][L] = o
    BL = res["by_layer"]
    res["R1_point"] = {w: {"real_flatter_layers": [L for L in range(12) if BL[L]["D"][w] < 0],
                           "real_steeper_layers": [L for L in range(12) if BL[L]["D"][w] > 0],
                           "ratio_range": [min(BL[L]["ratio"][w] for L in range(12)),
                                           max(BL[L]["ratio"][w] for L in range(12))]} for w in BULK}
    res["R2"] = {}
    for w in TOP:
        D = {L: BL[L]["D"][w] for L in range(12)}
        hi, lo = max(D, key=D.get), min(D, key=D.get)
        res["R2"][w] = {"range": D[hi] - D[lo], "argmax": hi, "argmin": lo, "D": D}
    res["top_real_steeper_layers"] = {w: [L for L in range(12) if BL[L]["D"][w] > 0] for w in TOP}
    res["fvu_range"] = {c: [min(BL[L][c]["fvu"] for L in range(12)), max(BL[L][c]["fvu"] for L in range(12))]
                        for c in ("real", "gaussian")}
    # k sweep at layer 6, (i) rule over k = 16, 32, 64, 128 (2 seeds)
    kk = [8, 16, 32, 64, 128]
    cur = {c: {k: {r["seed"]: r for r in pick(m1, layer=6, cond=c, k=k)} for k in kk} for c in ("real", "gaussian")}
    res["k6"]["curves"] = {c: {w: {k: msd([r["gamma_fixed"][GPT_WIN[w]] for r in cur[c][k].values()]) for k in kk}
                               for w in WINDOWS} for c in cur}
    res["k6"]["fvu"] = {c: {k: msd([r["fvu_eval"] for r in cur[c][k].values()])[0] for k in kk} for c in cur}
    kf = [16, 32, 64, 128]
    ver = {}
    for w in WINDOWS:
        x = {}
        for unit in ("gamma", "raw"):
            key = "gamma_fixed" if unit == "gamma" else "dec_fixed"

            def sf(c, ks):
                return [fit(ks, [cur[c][k][s][key][GPT_WIN[w]] for k in ks]) for s in (0, 1)]
            (mR, _, seR), (mG, _, seG) = msd(sf("real", kf)), msd(sf("gaussian", kf))
            loo = [(np.mean(sf("real", [k for k in kf if k != d])), np.mean(sf("gaussian", [k for k in kf if k != d])))
                   for d in kf]
            sgn = 1.0 if mR >= 0 else -1.0
            steady = sgn * mG > 0 and abs(mG) > 2 * seG and all(sgn * g > 0 for _, g in loo)
            rat = mG / mR
            minloo = min(g / r for r, g in loo)
            sup = steady and rat >= 0.75 and minloo >= 0.5
            con = (not steady) or (rat < 0.5 and sgn * (mR - mG) > 2 * np.hypot(seR, seG))
            x[unit] = {"s_real": mR, "s_gauss": mG, "ratio": rat, "min_loo_ratio": minloo, "supports": bool(sup),
                       "contradicts": bool(con)}
        x["verdict"] = ("supports" if x["gamma"]["supports"] and x["raw"]["supports"] else
                        "contradicts" if x["gamma"]["contradicts"] and x["raw"]["contradicts"] else "partial")
        ver[w] = x
    res["k6"]["i"] = ver
    N["gpt2"] = res
    return res, cur


def fig_gpt2(res, cur):
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.2))
    Ls = list(range(12))
    BL = res["by_layer"]
    for j, w in enumerate(WINDOWS):
        ax = axes[0, j]
        for c in ("real", "gaussian"):
            ax.plot(Ls, [BL[L][c]["raw"][w] for L in Ls], "o-", color=COL[c], ms=2.4, label=LAB[c])
            s6 = BL[6][c]["raw_seeds"][w]
            ax.plot([6, 6], [min(s6), max(s6)], "-", color=COL[c], lw=2.5, alpha=0.5)
        ax.plot(Ls, [BL[L]["real"]["sigma"][w] for L in Ls], "*--", color=OI["grey"], ms=3.5, lw=0.7,
                label=r"$\Sigma$")
        ax.set_title(GPT_LABEL[w], fontsize=7.5)
        ax.set_xlabel("layer")
        if j == 0:
            ax.set_ylabel("decoder eigen-slope, k=32")
        panel_tag(ax, "abcd"[j], x=-0.08)
        ax = axes[1, j]
        kk = [8, 16, 32, 64, 128]
        for c in ("real", "gaussian"):
            for k in kk:
                ax.plot([k] * len(cur[c][k]), [r["gamma_fixed"][GPT_WIN[w]] for r in cur[c][k].values()], ".",
                        ms=1.6, color=COL[c], alpha=0.6)
            ax.plot(kk, [res["k6"]["curves"][c][w][k][0] for k in kk], "o-", color=COL[c], ms=2.4)
        ax.set_title(f"layer 6: {res['k6']['i'][w]['verdict']}", fontsize=7)
        log2_axis(ax, [8, 16, 32, 64, 128])
        ax.set_xlabel("k")
        if j == 0:
            ax.set_ylabel(r"$\gamma$, 8,192 live latents")
        panel_tag(ax, "efgh"[j], x=-0.08)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=6.8)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save(fig, "fig10_gpt2_r2")


# ----------------------------------------------------------------------------------------------- Sigma convergence
def sigma_conv():
    eig1m = np.load(RES / "test1" / "sigma_eigenvalues_1m.npy")
    out = {}
    for L in LAYERS + [12]:
        z = np.load(RES / "r2" / "sigma" / f"gemma-2-2b_L{L}.npz")
        o = {}
        for w in WINDOWS:
            rng = tuple(int(x) for x in w.strip("()").split(","))
            a = slope_from_eigenvalues(z["eigenvalues"], rng)["slope"]
            b = slope_from_eigenvalues(eig1m[L], rng)["slope"]
            o[w] = {"10M": a, "1M": b, "diff_1M_minus_10M": b - a}
        o["n_10M"] = int(z["n"])
        out[L] = o
    out["max_abs_diff"] = {w: max(abs(out[L][w]["diff_1M_minus_10M"]) for L in LAYERS + [12]) for w in WINDOWS}
    d = [out[L][WINDOWS[3]]["diff_1M_minus_10M"] for L in LAYERS + [12]]
    out["bulk101_1M_steeper_at_all_layers"] = bool(all(x < 0 for x in d))
    out["bulk101_1M_minus_10M_range"] = [min(d), max(d)]
    N["sigma_1M_vs_10M"] = out
    # the Sigma slopes stored on the rows must equal the npz eigenvalues
    for L in LAYERS:
        r = pick(REAL, step="Q1", layer=L, cond="real", k=40)[0]
        for w in WINDOWS:
            assert abs(r["sigma_slope"][w] - out[L][w]["10M"]) < 1e-6, (L, w)


def earlier_set():
    """New full-live real and Gaussian slopes at layers 3 and 13 against the earlier set (results/gpu/runs.json,
    group saes_gpu_p2, 16k, k=40, seed means; all live rows)."""
    old = json.load(open(RES / "gpu" / "runs.json"))["runs"]
    out = {}
    for c in ("real", "gaussian"):
        for L in (3, 13):
            o = [r for r in old if r["group"] == "saes_gpu_p2" and r["layer"] == L and r["cond"] == c
                 and r["width"] == 16384 and r["k"] == 40]
            n = pick(REAL, step="Q1", layer=L, cond=c, k=40)
            assert len(o) == 3 and len(n) == 3, (c, L, len(o))
            out[f"{c}_L{L}"] = {w: float(np.mean([r["dec_full"][w] for r in n]) -
                                         np.mean([r["decoder_slope"][w] for r in o])) for w in WINDOWS}
    out["max_abs"] = {c: max(abs(v) for k, d in out.items() if k.startswith(c) for v in d.values())
                      for c in ("real", "gaussian")}
    N["earlier_set_diff_full_live"] = out
    # scrambled-to-real bulk slope ratio at width 16,384 only (the 65k SAEs of that set look undertrained)
    rat = {}
    for L in (3, 13, 23):
        sc = [r for r in old if r["group"] == "saes_gpu_p2" and r["layer"] == L and r["cond"] == "scrambled"
              and r["width"] == 16384 and r["k"] == 40]
        re = [r for r in old if r["group"] == "saes_gpu_p2" and r["layer"] == L and r["cond"] == "real"
              and r["width"] == 16384 and r["k"] == 40]
        assert len(sc) == 3 and len(re) == 3
        rat[L] = {w: float(np.mean([r["decoder_slope"][w] for r in sc]) / np.mean([r["decoder_slope"][w] for r in re]))
                  for w in BULK}
    N["earlier_set_scrambled_over_real_16k"] = {
        "by_layer": rat,
        "range_excl_L23_101_1000": [min(rat[L][w] for L in (3, 13, 23) for w in BULK if not (L == 23 and w == BULK[1])),
                                    max(rat[L][w] for L in (3, 13, 23) for w in BULK if not (L == 23 and w == BULK[1]))]}


def check_summary():
    p = RES / "r2" / "summary.json"
    if not p.exists():
        return
    s = json.load(open(p))
    i = s["i_gauss_decline_as_steady_as_real"]["window_verdicts"]
    assert i == N["ksweep"]["i_window_verdicts"], (i, N["ksweep"]["i_window_verdicts"])
    ii = s["ii_layer_profile_real_minus_gauss"]["window_verdicts"]
    assert ii == {w: N["layers8"]["ii"][w]["verdict"] for w in WINDOWS}, ii
    iii = s["iii_whitening_isotropic"]["window_verdicts"]
    assert iii == N["whiten"]["verdict"], iii
    for w in WINDOWS:
        for unit in ("gamma", "raw"):
            a = s["i_gauss_decline_as_steady_as_real"]["windows"][w]["fixed"][unit]
            assert abs(a["s_real"] - N["ksweep"]["i"][w][unit]["s_real"][0]) < 1e-9
            assert abs(a["s_gauss"] - N["ksweep"]["i"][w][unit]["s_gauss"][0]) < 1e-9
    r3 = s["models"]["M1_gpt2"]["R3_k_decline"]["gaussian"]["gpt2_detail"]
    assert r3 == {w: N["gpt2"]["k6"]["i"][w]["verdict"] for w in WINDOWS}, r3
    N["summary_json_agrees"] = True


# ----------------------------------------------------------------------------------------------- codable sparse null
SP2 = rows("runs_sparse2.jsonl") if (RES / "r2" / "runs_sparse2.jsonl").exists() else []
KEEP, REMOVE = 0.75, 0.5  # thresholds of statement (v), r2_job_real.py


def _sp2_final(L, k):
    rs = [r for r in SP2 if r.get("step") in ("S2a", "S2b") and r["layer"] == L and r["k"] == k]
    return rs


def _i_rule(curR, curN, w, unit):
    """(i) rule of r2_job_real.py at layer 13 over K_FIT with any number of seeds per condition."""
    key = "gamma_fixed" if unit == "gamma" else "dec_fixed"

    def sf(cur, ks):
        return [fit(ks, [cur[k][s][key][w] for k in ks]) for s in sorted(cur[ks[0]])]
    (mR, sdR, seR), (mN, sdN, seN) = msd(sf(curR, K_FIT)), msd(sf(curN, K_FIT))
    loo = [(float(np.mean(sf(curR, [k for k in K_FIT if k != d]))), float(np.mean(sf(curN, [k for k in K_FIT if k != d]))))
           for d in K_FIT]
    sgn = 1.0 if mR >= 0 else -1.0
    steady = sgn * mN > 0 and abs(mN) > 2 * seN and all(sgn * g > 0 for _, g in loo)
    rat, minloo = mN / mR, min(g / r for r, g in loo)
    sup = steady and rat >= 0.75 and minloo >= 0.5
    con = (not steady) or (rat < 0.5 and sgn * (mR - mN) > 2 * float(np.hypot(seR, seN)))
    return {"s_real": (mR, sdR), "s_null": (mN, sdN), "ratio": rat, "min_loo_ratio": minloo,
            "supports": bool(sup), "contradicts": bool(con)}


def sparse2(out8, kcur):
    """Codable sparse null (r2_job_sparse2.py). Real and Gaussian rows are job 1's (out8 = layers8(), kcur = ksweep()).
    Statement (v) is recomputed with the rules of r2_job_real.py: S1 (bulk) f = sum_L D_sp / sum_L D_g; S2 (top)
    f = P(D_sp) / P(D_g), P(D) = max interior D - max end D; S3 f = (s_sp - s_R) / (s_g - s_R) in gamma and raw."""
    res = {"by_layer": {}, "calibration": {}, "gt": {}, "k13": {}}
    cal = json.load(open(RES / "r2" / "sparse2_calibration.json"))
    for L in LAYERS:
        rs = _sp2_final(L, 40)
        assert len(rs) == 2, (L, len(rs))
        o = {"seeds": sorted(r["seed"] for r in rs), "rho": rs[0]["rho"], "fvu": msd([r["fvu_eval"] for r in rs]),
             "n_live": [min(r["n_live"] for r in rs), max(r["n_live"] for r in rs)],
             "raw": {w: msd([r["dec_fixed"][w] for r in rs]) for w in WINDOWS},
             "gamma": {w: msd([r["gamma_fixed"][w] for r in rs]) for w in WINDOWS}}
        assert all(r["rho"] == cal[str(L)]["rho"] for r in rs)
        o["D"] = {}
        for w in WINDOWS:
            mR, _, seR = out8[L]["real"]["raw"][w]
            o["D"][w] = (o["raw"][w][0] - mR, float(np.hypot(seR, o["raw"][w][2])))
        o["fvu_minus_real"] = o["fvu"][0] - out8[L]["real"]["fvu"][0]
        # gamma triplet real / sparse / Gaussian, and the sparse null's share of the real-to-Gaussian gamma gap
        o["gamma_triplet"] = {w: (out8[L]["real"]["gamma"][w][0], o["gamma"][w][0], out8[L]["gaussian"]["gamma"][w][0])
                              for w in WINDOWS}
        res["by_layer"][L] = o
        c = cal[str(L)]
        tried = {float(k): v for k, v in c["tried"].items()}
        res["calibration"][L] = {"rho": c["rho"], "target": c["target"], "gaussian": c["gaussian"],
                                 "matched": c["matched"], "fvu_rho0": tried[0.0], "tried": tried,
                                 "abs_err_seed0": abs(tried.get(c["rho"], o["fvu"][0]) - c["target"])}
        g = [r for r in SP2 if r.get("step") == "GT" and r["layer"] == L]
        assert len(g) == 1
        res["gt"][L] = {w: g[0]["gamma_fixed"][w] for w in WINDOWS}
    BL = res["by_layer"]
    res["n_final"] = sum(1 for r in SP2 if r.get("step") in ("S2a", "S2b"))
    res["n_calibration"] = sum(1 for r in SP2 if r.get("step") == "CAL")
    res["n_calibration_runs_per_layer"] = sorted({len(cal[str(L)]["tried"]) for L in LAYERS})
    # post hoc: does the sparse null's raw slope follow Sigma across layers (range and Pearson r over the 8 layers)
    res["raw_range_posthoc"] = {w: float(np.ptp([BL[L]["raw"][w][0] for L in LAYERS])) for w in WINDOWS}
    res["r_with_sigma_posthoc"] = {w: float(np.corrcoef([out8[L]["real"]["sigma"][w] for L in LAYERS],
                                                        [BL[L]["raw"][w][0] for L in LAYERS])[0, 1]) for w in WINDOWS}
    res["rho_range"] = [min(BL[L]["rho"] for L in LAYERS), max(BL[L]["rho"] for L in LAYERS)]
    res["fvu_range"] = [min(BL[L]["fvu"][0] for L in LAYERS), max(BL[L]["fvu"][0] for L in LAYERS)]
    res["max_abs_fvu_minus_real"] = max(abs(BL[L]["fvu_minus_real"]) for L in LAYERS)
    res["fvu_rho0_range"] = [min(res["calibration"][L]["fvu_rho0"] for L in LAYERS),
                             max(res["calibration"][L]["fvu_rho0"] for L in LAYERS)]
    res["gt_gamma_range"] = [min(v for L in LAYERS for v in res["gt"][L].values()),
                             max(v for L in LAYERS for v in res["gt"][L].values())]
    res["seed_sd_max_gamma"] = max(BL[L]["gamma"][w][1] for L in LAYERS for w in WINDOWS)
    res["seed_sd_max_raw"] = max(BL[L]["raw"][w][1] for L in LAYERS for w in WINDOWS)
    res["n_live_range"] = [min(BL[L]["n_live"][0] for L in LAYERS), max(BL[L]["n_live"][1] for L in LAYERS)]
    # descriptive: where real is steeper than the sparse null (gamma real > gamma sparse), and the gaps
    res["real_steeper_layers"] = {w: [L for L in LAYERS if BL[L]["gamma_triplet"][w][0] > BL[L]["gamma_triplet"][w][1]]
                                  for w in WINDOWS}
    res["gamma_gap_real_minus_sparse"] = {w: {L: BL[L]["gamma_triplet"][w][0] - BL[L]["gamma_triplet"][w][1]
                                              for L in LAYERS} for w in WINDOWS}
    res["D_over_2se"] = {w: {L: BL[L]["D"][w][0] / (2 * BL[L]["D"][w][1]) for L in LAYERS} for w in WINDOWS}
    res["sparse_gamma_range"] = {w: [min(BL[L]["gamma"][w][0] for L in LAYERS), max(BL[L]["gamma"][w][0] for L in LAYERS)]
                                 for w in WINDOWS}
    res["bulk_ratio_sparse_over_real_raw"] = {w: [min(BL[L]["raw"][w][0] / out8[L]["real"]["raw"][w][0] for L in LAYERS),
                                                  max(BL[L]["raw"][w][0] / out8[L]["real"]["raw"][w][0] for L in LAYERS)]
                                              for w in WINDOWS}
    # (ii) with the sparse null in place of the Gaussian
    ii = {}
    for w in WINDOWS:
        D = {L: BL[L]["D"][w] for L in LAYERS}
        if w in TOP:
            Ls = max(LAYERS, key=lambda L: D[L][0])
            sup = Ls in INTERIOR and all(D[Ls][0] - D[e][0] > 2 * np.hypot(D[Ls][1], D[e][1]) for e in ENDS)
            con = max(D[e][0] for e in ENDS) >= max(D[i][0] for i in INTERIOR)
            ii[w] = {"argmax": Ls, "min_margin_over_ends": min(D[Ls][0] - D[e][0] for e in ENDS)}
        else:
            sup = all(-D[L][0] > 2 * D[L][1] for L in LAYERS)
            con = any(D[L][0] > 2 * D[L][1] for L in LAYERS)
            ii[w] = {}
        ii[w]["verdict"] = "supports" if sup else "contradicts" if con else "partial"
    res["ii"] = ii
    # S1 / S2
    v = {}
    for w in WINDOWS:
        Dg = {L: out8[L]["D"][w][0] for L in LAYERS}
        Ds = {L: BL[L]["D"][w][0] for L in LAYERS}
        if w in BULK:
            f = sum(Ds.values()) / sum(Dg.values())
        else:
            amp = lambda D: max(D[L] for L in INTERIOR) - max(D[L] for L in ENDS)  # noqa: E731
            f = amp(Ds) / amp(Dg)
        gsep = N["layers8"]["ii"][w]["verdict"] == "supports"
        verdict = ("n/a" if not gsep else "supports" if f >= KEEP and ii[w]["verdict"] == "supports"
                   else "contradicts" if f < REMOVE else "partial")
        v[w] = {"profile_f": f, "profile_verdict": verdict}
    # k sweep at layer 13 and S3
    cur = {}
    for k in KS:
        rs = _sp2_final(13, k)
        assert len(rs) == 2, (k, len(rs))
        cur[k] = {r["seed"]: r for r in rs}
    res["k13"]["curves"] = {w: {k: msd([r["gamma_fixed"][w] for r in cur[k].values()]) for k in KS} for w in WINDOWS}
    res["k13"]["fvu"] = {k: msd([r["fvu_eval"] for r in cur[k].values()])[0] for k in KS}
    res["k13"]["n_live"] = {k: [min(r["n_live"] for r in cur[k].values()), max(r["n_live"] for r in cur[k].values())]
                            for k in KS}
    res["k13"]["seed_sd_max_gamma"] = max(res["k13"]["curves"][w][k][1] for w in WINDOWS for k in KS)
    res["k13"]["real_minus_sparse_gamma"] = {w: {k: N["ksweep"]["curves"]["real"][w][k][0] - res["k13"]["curves"][w][k][0]
                                                 for k in KS} for w in WINDOWS}
    res["k13"]["fvu_minus_real"] = {k: res["k13"]["fvu"][k] - N["ksweep"]["fvu"]["real"][k] for k in KS}
    iS = {}
    for w in WINDOWS:
        x = {u: _i_rule(kcur["real"], cur, w, u) for u in ("gamma", "raw")}
        x["verdict"] = ("supports" if x["gamma"]["supports"] and x["raw"]["supports"] else
                        "contradicts" if x["gamma"]["contradicts"] and x["raw"]["contradicts"] else "partial")
        iS[w] = x
        gi = N["ksweep"]["i"][w]
        fs = {}
        for u in ("gamma", "raw"):
            sR, sG, sS = gi[u]["s_real"][0], gi[u]["s_gauss"][0], x[u]["s_null"][0]
            fs[u] = (sS - sR) / (sG - sR)
        gsep = gi["verdict"] == "contradicts"
        v[w]["S3_f"] = fs
        v[w]["S3_verdict"] = ("n/a" if not gsep else "supports" if all(f >= KEEP for f in fs.values())
                              and x["verdict"] == "contradicts" else
                              "contradicts" if all(f < REMOVE for f in fs.values()) else "partial")
    res["k13"]["i"] = iS
    res["k13"]["i_window_verdicts"] = {w: iS[w]["verdict"] for w in WINDOWS}

    def wrule(vs, yes, no):
        others = [vs[w] for w in WINDOWS if w != "(10, 150)"]
        if vs["(10, 150)"] == yes and sum(o == yes for o in others) >= 2:
            return yes
        if vs["(10, 150)"] == no and sum(o == no for o in others) >= 2:
            return no
        return "mixed"
    prof = {w: v[w]["profile_verdict"] for w in WINDOWS}
    s3 = {w: v[w]["S3_verdict"] for w in WINDOWS}
    groups = {"profile": wrule(prof, "supports", "contradicts"), "S3_decline": wrule(s3, "supports", "contradicts")}
    res["v"] = {"windows": v, "groups": groups,
                "verdict": "contradicts" if "contradicts" in groups.values() else
                           "supports" if all(g == "supports" for g in groups.values()) else "mixed"}
    N["sparse2"] = res
    return cur


def table_sparse2():
    s, l8 = N["sparse2"], N["layers8"]["by_layer"]
    lines = []
    for L in LAYERS:
        o = s["by_layer"][L]
        cells = [str(L), f"{o['rho']:.2f}"]
        cells += ["/".join(f"{x:.2f}" for x in o["gamma_triplet"][w]) for w in WINDOWS]
        cells += ["/".join(f"{x:.2f}" for x in (l8[L]["real"]["fvu"][0], o["fvu"][0], l8[L]["gaussian"]["fvu"][0]))]
        lines.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{tabular}{r r cccc c}
\toprule
 & & \multicolumn{4}{c}{$\gamma$ at 8,192 live latents: real / sparse null / Gaussian null} & FVU \\
\cmidrule(lr){3-6}
layer & $\rho$ & 4--50 & 11--150 & 31--450 & 101--1000 & real / sparse / Gauss. \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_sparse2_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_real.jsonl and runs_sparse2.jsonl\n" + tex)


def old_sparse():
    """The first sparse null (cond "sparse", M = 65,536, L = 64): FVU and gamma against the Gaussian null, per cell."""
    cells = sorted({(r["layer"], r["k"]) for r in REAL if r["cond"] == "sparse" and not r.get("whitened")})
    dF, dG = [], {w: [] for w in WINDOWS}
    for L, k in cells:
        sp = [r for r in REAL if r["cond"] == "sparse" and r["layer"] == L and r["k"] == k and not r.get("whitened")]
        ga = [r for r in REAL if r["cond"] == "gaussian" and r["layer"] == L and r["k"] == k and not r.get("whitened")
              and r["step"] in ("Q1", "Q2")]
        if not ga:
            continue
        dF.append(abs(np.mean([r["fvu_eval"] for r in sp]) - np.mean([r["fvu_eval"] for r in ga])))
        for w in WINDOWS:
            dG[w].append(abs(np.mean([r["gamma_fixed"][w] for r in sp]) - np.mean([r["gamma_fixed"][w] for r in ga])))
    m1 = [r for r in MOD if r.get("step") == "M1"]
    gF = []
    for L in range(12):
        sp, ga = pick(m1, layer=L, cond="sparse", k=32), pick(m1, layer=L, cond="gaussian", k=32)
        if sp and ga:
            gF.append(abs(np.mean([r["fvu_eval"] for r in sp]) - np.mean([r["fvu_eval"] for r in ga])))
    N["sparse_first"] = {"n_cells": len(dF), "max_abs_fvu_minus_gauss": max(dF),
                         "max_abs_gamma_minus_gauss": {w: max(v) for w, v in dG.items()},
                         "gpt2_n_layers": len(gF), "gpt2_max_abs_fvu_minus_gauss": max(gF) if gF else None}


# ----------------------------------------------------------------------------------------------- Gemma-2-9b (M2)
PRED_A_2B = {"(3, 50)": 0.0937, "(10, 150)": 0.0909, "(30, 450)": 0.0561, "(100, 1000)": 0.0489}  # r2_job_models.py
NINE_WIN = {"(3, 50)": "(5, 78)", "(10, 150)": "(16, 233)", "(30, 450)": "(47, 700)", "(100, 1000)": "(156, 1556)"}
NINE_LABEL = {"(3, 50)": "6--78", "(10, 150)": "17--233", "(30, 450)": "48--700", "(100, 1000)": "157--1556"}


def gemma9b():
    """Gemma Scope 9b width-16k SAEs (step M2), predictions (a)-(c) of r2_job_models.py, recomputed from the rows."""
    m2 = [r for r in MOD if r.get("step") == "M2"]
    res = {"n_saes": len(m2), "layers": sorted({r["layer"] for r in m2}), "a": {}, "b": {}, "c": {}}
    nl = m2[0]["n_layers"]
    # (a) per-doubling slope of the decoder slope in L0 at layers 9, 20, 31
    for w in WINDOWS:
        w9 = NINE_WIN[w]
        per = {}
        for L in (9, 20, 31):
            rs = sorted(pick(m2, layer=L), key=lambda r: r["l0"])
            per[L] = fit([r["l0"] for r in rs], [r["dec_slope"][w9] for r in rs])
        mean, ref = float(np.mean(list(per.values()))), PRED_A_2B[w]
        if 0.5 * ref <= mean <= 1.5 * ref and all(v > 0 for v in per.values()):
            v = "agrees"
        elif not (0.25 * ref <= mean <= 2 * ref) or any(v <= 0 for v in per.values()):
            v = "disagrees"
        else:
            v = "partial"
        res["a"][w] = {"per_layer": per, "mean": mean, "interval": [0.5 * ref, 1.5 * ref], "verdict": v}
    res["a_n_l0"] = {L: sorted(r["l0"] for r in pick(m2, layer=L)) for L in (9, 20, 31)}
    # (b) mid-stack contrast at L0 nearest 40 and 70

    def nearest(L, t):
        rs = pick(m2, layer=L)
        return min(rs, key=lambda r: abs(np.log(r["l0"] / t)))
    for t in (40, 70):
        sel = {L: nearest(L, t) for L in res["layers"]}
        res["b"][t] = {"l0_range": [min(r["l0"] for r in sel.values()), max(r["l0"] for r in sel.values())]}
        for w in WINDOWS:
            w9 = NINE_WIN[w]
            y = {L: sel[L]["dec_slope"][w9] for L in sel}
            dep = {L: L / (nl - 1) for L in sel}
            mid = np.mean([y[L] for L in sel if 0.35 <= dep[L] <= 0.65])
            early = np.mean([y[L] for L in sel if dep[L] <= 0.12])
            late = np.mean([y[L] for L in sel if dep[L] >= 0.88])
            sig = {L: sel[L]["sigma_slope"][w9] for L in sel}
            Ls = sorted(sel)
            res["b"][t][w] = {"contrast": float(min(early, late) - mid), "steepest_layer": min(y, key=y.get),
                              "sigma_steepest_layer": min(sig, key=sig.get),
                              "r_with_sigma_posthoc": float(np.corrcoef([y[L] for L in Ls], [sig[L] for L in Ls])[0, 1]),
                              "gamma_range": [min(sel[L]["gamma"][w9] for L in Ls), max(sel[L]["gamma"][w9] for L in Ls)]}
    cs = [res["b"][t][w]["contrast"] for t in (40, 70) for w in TOP]
    res["b"]["verdict"] = "supports" if all(c >= 0.05 for c in cs) else "contradicts" if all(c <= 0 for c in cs) else "partial"
    res["b"]["top_contrast_range"] = [min(cs), max(cs)]
    res["b"]["bulk_contrast_range"] = [min(res["b"][t][w]["contrast"] for t in (40, 70) for w in BULK),
                                       max(res["b"][t][w]["contrast"] for t in (40, 70) for w in BULK)]
    # (c) gamma < 1 for every SAE with L0 >= 20
    c20 = [r for r in m2 if r["l0"] >= 20]
    for w in WINDOWS:
        g = [r["gamma"][NINE_WIN[w]] for r in c20]
        frac = float(np.mean([x >= 1 for x in g]))
        res["c"][w] = {"n": len(g), "n_ge_1": int(sum(x >= 1 for x in g)), "max_gamma": max(g),
                       "verdict": "holds" if frac == 0 else "fails" if frac > 0.1 else "partial"}
    res["c_n_l0_lt_20"] = len(m2) - len(c20)
    res["n_layers"] = len(res["layers"])
    res["n_at_9_20_31"] = {L: len(pick(m2, layer=L)) for L in (9, 20, 31)}
    res["l0_range"] = [min(r["l0"] for r in m2), max(r["l0"] for r in m2)]
    res["sigma_n_tokens"] = sorted({r["sigma_n_tokens"] for r in m2})
    N["gemma9b"] = res


def check_summary_new():
    p = RES / "r2" / "summary.json"
    s = json.load(open(p))
    v = s["v_sparse2_null_separates_like_gaussian"]
    me = N["sparse2"]
    assert v["verdict"] == me["v"]["verdict"], (v["verdict"], me["v"]["verdict"])
    assert v["groups"] == me["v"]["groups"], (v["groups"], me["v"]["groups"])
    for w in WINDOWS:
        assert abs(v["windows"][w]["profile"]["f"] - me["v"]["windows"][w]["profile_f"]) < 1e-9, w
        assert v["windows"][w]["profile"]["verdict"] == me["v"]["windows"][w]["profile_verdict"], w
        assert v["windows"][w]["S3_decline"]["verdict"] == me["v"]["windows"][w]["S3_verdict"], w
        for u in ("gamma", "raw"):
            assert abs(v["windows"][w]["S3_decline"]["f"][u] - me["v"]["windows"][w]["S3_f"][u]) < 1e-9, (w, u)
    assert v["i_sparse"]["window_verdicts"] == me["k13"]["i_window_verdicts"]
    assert v["ii_sparse"]["window_verdicts"] == {w: me["ii"][w]["verdict"] for w in WINDOWS}
    m = s["models"]["M2_gemma9b"]
    g = N["gemma9b"]
    assert m["a_l0_per_doubling"]["verdict"] == window_rule9({w: g["a"][w]["verdict"] for w in WINDOWS}, "agrees", "disagrees")
    for w in WINDOWS:
        assert m["a_l0_per_doubling"]["windows"][w]["verdict"] == g["a"][w]["verdict"], w
        assert abs(m["a_l0_per_doubling"]["windows"][w]["mean"] - g["a"][w]["mean"]) < 1e-9, w
        assert m["c_gamma_below_1"]["windows"][w]["verdict"] == g["c"][w]["verdict"], w
        for t in (40, 70):
            assert abs(m["b_midstack_steeper"]["cells"][f"L0~{t}|{w}"]["contrast"] - g["b"][t][w]["contrast"]) < 1e-9
    assert m["b_midstack_steeper"]["verdict"] == g["b"]["verdict"]
    g["verdicts"] = {"a": m["a_l0_per_doubling"]["verdict"], "b": g["b"]["verdict"], "c": m["c_gamma_below_1"]["verdict"]}
    N["summary_json_agrees_sparse2_m2"] = True


def window_rule9(vs, yes, no):
    others = [vs[w] for w in WINDOWS if w != "(10, 150)"]
    if vs["(10, 150)"] == yes and sum(o == yes for o in others) >= 2:
        return yes
    if vs["(10, 150)"] == no and sum(o == no for o in others) >= 2:
        return no
    return "partial"


# ----------------------------------------------------------------------------------------------- matched sparse null:
# equivalence in ranks 11-150, bulk at every k, the main-text figure and table
EQ_MARGIN = 0.06  # chosen after seeing the data; stated as such in the paper


def equivalence():
    """Per layer, real minus sparse-null gamma (fixed-n, k=40) in every window with a 90% Welch interval from the seed
    spread (3 real seeds, 2 sparse-null seeds). The interval inside +-EQ_MARGIN is the two one-sided test at 5%."""
    from scipy import stats
    l8, sp = N["layers8"]["by_layer"], N["sparse2"]["by_layer"]
    out = {"margin": EQ_MARGIN, "level": 0.90, "by_layer": {}}
    for L in LAYERS:
        o = {}
        for w in WINDOWS:
            mR, sdR, _ = l8[L]["real"]["gamma"][w]
            mS, sdS, _ = sp[L]["gamma"][w]
            vR, vS = sdR ** 2 / 3, sdS ** 2 / 2
            se = float(np.sqrt(vR + vS))
            df = (vR + vS) ** 2 / (vR ** 2 / 2 + vS ** 2 / 1)
            t = float(stats.t.ppf(0.95, df))
            o[w] = {"diff": mR - mS, "se": se, "df": df, "ci90": [mR - mS - t * se, mR - mS + t * se]}
        out["by_layer"][L] = o
    P = "(10, 150)"
    by = out["by_layer"]
    out["primary_inside_margin_layers"] = [L for L in LAYERS if -EQ_MARGIN < by[L][P]["ci90"][0] and
                                           by[L][P]["ci90"][1] < EQ_MARGIN]
    out["primary_point_inside_margin_layers"] = [L for L in LAYERS if abs(by[L][P]["diff"]) < EQ_MARGIN]
    out["primary_max_abs_diff_6_17"] = max(abs(by[L][P]["diff"]) for L in INTERIOR)
    out["primary_abs_diff_range_ends"] = [min(abs(by[L][P]["diff"]) for L in ENDS), max(abs(by[L][P]["diff"]) for L in ENDS)]
    out["primary_null_steeper_all_layers"] = all(by[L][P]["diff"] < 0 for L in LAYERS)
    out["primary_ci_excludes_zero_layers"] = [L for L in LAYERS if by[L][P]["ci90"][1] < 0]
    out["df_range"] = [min(by[L][P]["df"] for L in LAYERS), max(by[L][P]["df"] for L in LAYERS)]
    for w in BULK:
        out[f"bulk_diff_range {w}"] = [min(by[L][w]["diff"] for L in LAYERS), max(by[L][w]["diff"] for L in LAYERS)]
        out[f"bulk_ci_upper_max {w}"] = max(by[L][w]["ci90"][1] for L in LAYERS)
    k = N["sparse2"]["k13"]["real_minus_sparse_gamma"]
    out["k13"] = {
        "bulk_real_flatter_every_k": {w: all(k[w][kk] < 0 for kk in KS) for w in BULK},
        "bulk_diff_range_over_k": {w: [min(k[w].values()), max(k[w].values())] for w in BULK},
        "primary_max_abs_diff_k_ge_24": max(abs(k[P][kk]) for kk in KS if kk >= 24),
        "primary_max_abs_diff_k_ge_32": max(abs(k[P][kk]) for kk in KS if kk >= 32),
        "primary_diff_k8_k16": [k[P][8], k[P][16]],
        "real_null_gamma_k8_k16": {kk: [N["ksweep"]["curves"]["real"][P][kk][0], N["sparse2"]["k13"]["curves"][P][kk][0]]
                                   for kk in (8, 16)},
        "primary_per_doubling_ratio_null_over_real": {
            f"k{ks[0]}-128": fit(ks, [N["sparse2"]["k13"]["curves"][P][kk][0] for kk in ks]) /
            fit(ks, [N["ksweep"]["curves"]["real"][P][kk][0] for kk in ks]) for ks in ([16, 24, 32, 48, 64, 128],
                                                                                     [24, 32, 48, 64, 128])},
        "real_steeper_top_every_k": all(k["(3, 50)"][kk] > 0 for kk in KS),
        "top_diff_range_over_k": [min(k["(3, 50)"].values()), max(k["(3, 50)"].values())]}
    N["equiv"] = out


def fig_matched():
    """Main-text figure: real (3 seeds) against the matched sparse null (2 seeds), gamma at 8,192 live latents.
    Top row: eight layers at k=40. Bottom row: layer 13 over k (the null's FVU is matched at k=40 only)."""
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.15), gridspec_kw={"height_ratios": [1, 1]})
    l8, sp = N["layers8"]["by_layer"], N["sparse2"]["by_layer"]
    for j, w in enumerate(WINDOWS):
        ax = axes[0, j]
        for c, src in (("real", None), ("sparse2", None)):
            if c == "real":
                m = [l8[L]["real"]["gamma"][w][0] for L in LAYERS]
                for L in LAYERS:
                    ax.plot([L] * 3, [r["gamma_fixed"][w] for r in pick(REAL, step="Q1", layer=L, cond="real", k=40)],
                            ".", ms=1.6, color=COL[c], alpha=0.5)
            else:
                m = [sp[L]["gamma"][w][0] for L in LAYERS]
                for L in LAYERS:
                    ax.plot([L] * 2, [r["gamma_fixed"][w] for r in _sp2_final(L, 40)], ".", ms=1.6, color=COL[c],
                            alpha=0.6)
            ax.plot(LAYERS, m, marker="o" if c == "real" else "^", ms=2.6, lw=1.0, color=COL[c], label=LAB[c])
        ax.plot(LAYERS, [N["sparse2"]["gt"][L][w] for L in LAYERS], ":", color=OI["grey"], lw=0.9,
                label="null's own dictionary")
        if w == "(10, 150)":
            ax.axvspan(5, 18, color=OI["grey"], alpha=0.12, lw=0)
        ax.set_title(WIN_LABEL[w], fontsize=7.5)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([str(L) for L in LAYERS], fontsize=5.8)
        ax.set_ylim(0.1, 1.25)
        if j == 0:
            ax.set_ylabel(r"$\gamma$, $k=40$")
        panel_tag(ax, "abcd"[j], x=-0.08)
        ax = axes[1, j]
        for k in KS:
            ax.plot([k] * 3, [r["gamma_fixed"][w] for r in pick(REAL, layer=13, cond="real", k=k, whitened=False)
                              if r["step"] in ("Q1", "Q2")], ".", ms=1.6, color=COL["real"], alpha=0.5)
            ax.plot([k] * 2, [r["gamma_fixed"][w] for r in _sp2_final(13, k)], ".", ms=1.6, color=COL["sparse2"],
                    alpha=0.6)
        ax.plot(KS, [N["ksweep"]["curves"]["real"][w][k][0] for k in KS], "o-", ms=2.4, lw=1.0, color=COL["real"])
        ax.plot([8], [N["ksweep"]["curves"]["real"][w][8][0]], "x", color="k", ms=4, mew=0.8)
        ax.plot(KS, [N["sparse2"]["k13"]["curves"][w][k][0] for k in KS], "^-", ms=2.4, lw=1.0, color=COL["sparse2"])
        ax.axvline(40, color=OI["grey"], lw=0.5, ls="--")
        log2_axis(ax, [8, 16, 32, 64, 128])
        ax.set_xlabel("k (layer 13)", fontsize=7)
        if j == 0:
            ax.set_ylabel(r"$\gamma$, layer 13")
        panel_tag(ax, "efgh"[j], x=-0.08)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=6.8)
    fig.tight_layout(w_pad=0.8, h_pad=0.7)
    save(fig, "fig11_matched_null_r2")


def table_matched():
    """Main-text table: per layer rho, FVU, gamma real/null in four windows, and the ranks 11-150 difference with its
    90% interval."""
    s, l8, e = N["sparse2"]["by_layer"], N["layers8"]["by_layer"], N["equiv"]["by_layer"]
    lines = []
    for L in LAYERS:
        o = s[L]
        cells = [str(L), f"{o['rho']:.2f}", f"{l8[L]['real']['fvu'][0]:.2f}/{o['fvu'][0]:.2f}"]
        for w in WINDOWS:
            cells.append(f"{l8[L]['real']['gamma'][w][0]:.2f}/{o['gamma'][w][0]:.2f}")
            if w == "(10, 150)":
                d = e[L][w]
                cells.append(f"${d['diff']:+.3f}$ [${d['ci90'][0]:+.3f}$, ${d['ci90'][1]:+.3f}$]")
        lines.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{tabular}{r r c cc c cc}
\toprule
 & & FVU & \multicolumn{5}{c}{$\gamma$ at 8,192 live latents, real / sparse null} \\
\cmidrule(lr){4-8}
layer & $\rho$ & real / null & 4--50 & 11--150 & real $-$ null, 11--150 [90\%] & 31--450 & 101--1000 \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_matched_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_real.jsonl and runs_sparse2.jsonl\n" + tex)


# ----------------------------------------------------------------------------------------------- local analyses
PP = RES / "r2" / "pilot_power"


def local_analyses():
    """Numbers quoted from the local CPU analyses in results/r2/pilot_power (single seed; see each file)."""
    out = {}
    # (1) dense share rho: null gamma at rho = 0 / ~0.1 / matched (seed 0, k = 40) against real (3-seed mean)
    c = json.load(open(PP / "cal_rho_table.json"))
    lay = [str(L) for L in LAYERS]
    rho = {"by_layer": {L: {"real": c[L]["real_gamma_fixed"], "null": [(n["rho"], n["fvu"], n["gamma_fixed"])
                                                                         for n in c[L]["null"]]} for L in lay}}
    for i, w in enumerate(WINDOWS):
        rho[w] = {"null_rho0": [min(c[L]["null"][0]["gamma_fixed"][i] for L in lay), max(c[L]["null"][0]["gamma_fixed"][i] for L in lay)],
                  "null_rho_mid": [min(c[L]["null"][1]["gamma_fixed"][i] for L in lay), max(c[L]["null"][1]["gamma_fixed"][i] for L in lay)],
                  "null_matched": [min(c[L]["null"][2]["gamma_fixed"][i] for L in lay), max(c[L]["null"][2]["gamma_fixed"][i] for L in lay)],
                  "real": [min(c[L]["real_gamma_fixed"][i] for L in lay), max(c[L]["real_gamma_fixed"][i] for L in lay)]}
    rho["rho_mid_range"] = [min(c[L]["null"][1]["rho"] for L in lay), max(c[L]["null"][1]["rho"] for L in lay)]
    rho["fvu_rho_mid_range"] = [min(c[L]["null"][1]["fvu"] for L in lay), max(c[L]["null"][1]["fvu"] for L in lay)]
    rho["top_real_between_mid_and_matched_layers"] = [int(L) for L in lay if c[L]["null"][2]["gamma_fixed"][0] <=
                                                      c[L]["real_gamma_fixed"][0] <= c[L]["null"][1]["gamma_fixed"][0]]
    rho["top_real_between_rho0_and_matched_layers"] = [int(L) for L in lay if c[L]["null"][2]["gamma_fixed"][0] <=
                                                       c[L]["real_gamma_fixed"][0] <= c[L]["null"][0]["gamma_fixed"][0]]
    rho["top_real_below_all_layers"] = [int(L) for L in lay if all(c[L]["real_gamma_fixed"][0] < n["gamma_fixed"][0]
                                                                   for n in c[L]["null"])]
    rho["bulk_real_below_null_every_rho_every_layer"] = all(
        c[L]["real_gamma_fixed"][i] < n["gamma_fixed"][i] for L in lay for n in c[L]["null"] for i in (2, 3))
    rho["primary_null_decreases_with_rho_every_layer"] = all(
        c[L]["null"][0]["gamma_fixed"][1] > c[L]["null"][1]["gamma_fixed"][1] > c[L]["null"][2]["gamma_fixed"][1] for L in lay)
    out["rho"] = rho
    # (2) Sigma's eigenbasis: eigen-gamma vs sorted-diagonal gamma, coherence (n * mean squared correlation), far share
    b = json.load(open(PP / "basis_reweight_vs_rotation.json"))
    conds = ("real_s0", "sparse2_matched_s0", "sparse2_rho0_s0", "groundtruth")
    basis = {"by_layer": b}
    for w in BULK:
        basis[w] = {cd: {m: [min(b[L][cd][w][m] for L in lay), max(b[L][cd][w][m] for L in lay)]
                         for m in ("gamma_eig", "gamma_sdiag", "coh", "far")} for cd in conds}
        basis[w]["real_sdiag_minus_eig"] = [min(b[L]["real_s0"][w]["gamma_sdiag"] - b[L]["real_s0"][w]["gamma_eig"] for L in lay),
                                            max(b[L]["real_s0"][w]["gamma_sdiag"] - b[L]["real_s0"][w]["gamma_eig"] for L in lay)]
        basis[w]["null_abs_sdiag_minus_eig_max"] = max(abs(b[L][cd][w]["gamma_sdiag"] - b[L][cd][w]["gamma_eig"])
                                                       for L in lay for cd in conds[1:])
    out["basis"] = basis
    # (3) one-parameter power family fitted to the real bulk
    p = json.load(open(PP / "part1_fit.json"))
    fam = {}
    for f, v in p["fams"].items():
        per = v["per_layer"]
        over = [per[L]["gamma8"][w] - p["real"][L][w] for L in lay for w in TOP]
        fam[f] = {"shared_param": v["shared_param"],
                  "shared_bulk_within_005_layers": [int(L) for L in lay if v["shared"][L]["within05_bulk"]],
                  "shared_top_range": [min(v["shared"][L]["gamma8"][w] for L in lay for w in TOP),
                                       max(v["shared"][L]["gamma8"][w] for L in lay for w in TOP)],
                  "shared_top_over_real_min": min(v["shared"][L]["gamma8"][w] - p["real"][L][w] for L in lay for w in TOP),
                  "per_layer_param_range": [min(per[L]["param"] for L in lay), max(per[L]["param"] for L in lay)],
                  "per_layer_top_over_real_range": [min(over), max(over)],
                  "per_layer_top_over_real_n_positive": [int(sum(o > 0 for o in over)), len(over)],
                  "per_layer_bulk_rms_max": max(per[L]["bulk_rms"] for L in lay),
                  "per_param_vs_layer_r": v["per_param_vs_layer_r"],
                  "per_param_vs_sigma_slope31_r": v["per_param_vs_sigma_slope31_r"]}
    fam["real_top_range"] = [min(p["real"][L][w] for L in lay for w in TOP), max(p["real"][L][w] for L in lay for w in TOP)]
    fam["eq_gamma8_range"] = [min(p["eq"][L]["gamma8"][w] for L in lay for w in WINDOWS),
                              max(p["eq"][L]["gamma8"][w] for L in lay for w in WINDOWS)]
    out["family"] = fam
    # (4) Gemma Scope 65k JumpReLU feature power at layers 0, 12, 24 on 32,736 cached tokens
    g = json.load(open(PP / "part2_power_gemmascope65k.json"))
    t = json.load(open(PP / "part2_power_top.json"))
    ray = json.load(open(PP / "part2_rayleigh.json"))
    pw = {"by_layer": {}}
    for L in g:
        o = g[L]
        pw["by_layer"][L] = {k: o[k] for k in ("n_tokens", "dropped", "fvu", "l0", "n_live", "power_ess_frac_live",
                                                "power_top1pct_share", "pi_vs_rq_spearman", "pi_vs_rq_r2",
                                                "power_weighted_trace_ratio", "power_weighted_gamma",
                                                "unweighted_gamma_fixed1", "power_weighted_rms_log_err_4_1000")}
        pw["by_layer"][L]["top10"] = t[L]["top10"]
        pw["by_layer"][L]["rest"] = t[L]["rest"]
        pw["by_layer"][L]["top10_rq_ratio"] = 10 ** t[L]["top10"]["med_log10_rq"]
        pw["by_layer"][L]["rest_rq_ratio"] = 10 ** t[L]["rest"]["med_log10_rq"]
    Ls = list(g)
    pw["power_weighted_gamma_range"] = [min(min(g[L]["power_weighted_gamma"].values()) for L in Ls),
                                        max(max(g[L]["power_weighted_gamma"].values()) for L in Ls)]
    pw["unweighted_gamma_range"] = [min(min(g[L]["unweighted_gamma_fixed1"].values()) for L in Ls),
                                    max(max(g[L]["unweighted_gamma_fixed1"].values()) for L in Ls)]
    pw["top10_share_range"] = [min(t[L]["top10"]["power_share"] for L in Ls), max(t[L]["top10"]["power_share"] for L in Ls)]
    pw["top10_rq_ratio_range"] = [min(10 ** t[L]["top10"]["med_log10_rq"] for L in Ls),
                                  max(10 ** t[L]["top10"]["med_log10_rq"] for L in Ls)]
    pw["rest_rq_ratio_range"] = [min(10 ** t[L]["rest"]["med_log10_rq"] for L in Ls),
                                 max(10 ** t[L]["rest"]["med_log10_rq"] for L in Ls)]
    pw["rest_power_share_range"] = [min(t[L]["rest"]["power_share"] for L in Ls), max(t[L]["rest"]["power_share"] for L in Ls)]
    pw["ess_frac_range"] = [min(g[L]["power_ess_frac_live"] for L in Ls), max(g[L]["power_ess_frac_live"] for L in Ls)]
    pw["spearman_range"] = [min(g[L]["pi_vs_rq_spearman"] for L in Ls), max(g[L]["pi_vs_rq_spearman"] for L in Ls)]
    pw["trace_ratio_range"] = [min(g[L]["power_weighted_trace_ratio"] for L in Ls),
                               max(g[L]["power_weighted_trace_ratio"] for L in Ls)]
    pw["l0_range"] = [min(g[L]["l0"] for L in Ls), max(g[L]["l0"] for L in Ls)]
    pw["fvu_range"] = [min(g[L]["fvu"] for L in Ls), max(g[L]["fvu"] for L in Ls)]
    pw["n_tokens"] = max(g[L]["n_tokens"] for L in Ls)
    # Rayleigh quotients of the equal-power Sigma-shaped dictionary (EQ, M = 16,384) at the eight layers
    eq = [ray["models"][L]["EQ{}"]["q"]["0.5"] for L in ray["models"]]
    pw["eq_dictionary_rq_ratio_range"] = [10 ** min(eq), 10 ** max(eq)]
    gs = [ray["gemmascope16k"][L][s]["q"]["0.5"] for L in ray["gemmascope16k"] for s in ray["gemmascope16k"][L]]
    pw["gemmascope16k_rq_ratio_range_all_l0"] = [10 ** min(gs), 10 ** max(gs)]
    out["power"] = pw
    N["pilot"] = out


# ----------------------------------------------------------------------------------------------- operating point
def operating_point():
    """Gemma Scope 2b numbers of the operating-point section, recomputed from the files make_figures_thesisA.py uses
    (test1/gamma_l0_L12.json, phase1/eig_*.npy, phase1/slope_vs_layer_matched_l0.json, test1/sigma_1m_compare.json)
    and Li et al.'s ordering at 26 layers (review/li_ordering_1m.json, review/li_protocol.json)."""
    P = "(10, 150)"
    g = json.load(open(RES / "test1" / "gamma_l0_L12.json"))
    pts = []
    for r in g["rows"]:
        nom = r["sae_id"].split("average_l0_")[1]
        eig = np.load(RES / "phase1" / f"eig_layer_12__width_{r['width']}__average_l0_{nom}.npy")
        pts.append((r["width"], r["l0"], {w: slope_from_eigenvalues(eig, tuple(int(v) for v in w.strip("()").split(",")))["slope"]
                                          for w in WINDOWS}))
    x = np.log2([p[1] for p in pts])
    y = np.array([p[2][P] for p in pts])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
    lo, hi = min(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[1])
    li = [p for p in pts if p[0] == "65k" and 60 < p[1] < 90][0]
    out = {"n_saes_L12": len(pts), "per_doubling": float(b), "r2": float(r2),
           "lowest": [lo[1], lo[2][P]], "highest": [hi[1], hi[2][P]], "range": float(y.max() - y.min()),
           "l0_ratio": hi[1] / lo[1], "li_sae": {"l0_measured": li[1], "slopes": li[2]},
           "width_spread_matched_l0": [[lo_, hi_, max(v) - min(v)] for lo_, hi_ in [(19, 24), (36, 44), (64, 86)]
                                       for v in [[p[2][P] for p in pts if lo_ <= p[1] <= hi_]]]}
    m = json.load(open(RES / "phase1" / "slope_vs_layer_matched_l0.json"))
    amp = {}
    for l0 in (40, 70, 140):
        d = m[f"16k_ranks11-150_L0_{l0}"]
        v = {int(k): val for k, val in d.items()}
        amp[l0] = {"peak_below_layer0": float(v[0] - min(v.values())), "steepest_layer": min(v, key=v.get)}
    out["midstack_peak"] = amp
    s = json.load(open(RES / "test1" / "sigma_1m_compare.json"))["profile_features"]["1m"]
    out["sigma_1m_11_150"] = {"layer0": s["top_layer0"], "steepest": s["top_min"], "steepest_layer": s["top_argmin"]}
    prof = json.load(open(RES / "test1" / "sigma_1m_compare.json"))["profile"]
    out["sigma_decoder_r_26_layers"] = {k: prof[k]["1m"]["pearson_r"] for k in prof}
    lo_ = json.load(open(RES / "review" / "li_ordering_1m.json"))["summary"]
    out["li_raw_steeper_all_layers"] = {k: lo_[k]["li_ordering_holds"] == 0 and lo_[k]["n_layers"] == 26
                                        for k in lo_ if k.startswith("raw|")}
    out["li_raw_gamma_range_1_100_11_150"] = [min(lo_[k]["gamma_min"] for k in lo_ if k.startswith("raw|") and
                                                  ("ranks1-100" in k or "ranks11-150" in k)),
                                              max(lo_[k]["gamma_max"] for k in lo_ if k.startswith("raw|") and
                                                  ("ranks1-100" in k or "ranks11-150" in k))]
    out["li_corr_holds"] = {k: lo_[k]["layers_holding"] for k in lo_ if k.startswith("corr|") and
                            ("ranks1-100" in k or "ranks11-150" in k)}
    lp = json.load(open(RES / "review" / "li_protocol.json"))
    out["li_L12_1_100"] = {"decoder": lp["decoder_L12_65k_L072"]["centred"]["ranks1-100"],
                           "raw": lp["L12_1M"]["spectra"]["raw_centred"]["slopes"]["ranks1-100"]}
    N["operating_point"] = out
    # GPT-2 small, descriptive (real SAEs only): gamma at k = 32 over 12 layers, steepest layer, decline with k at L6
    BL = N["gpt2"]["by_layer"]
    gam = {L: {w: BL[L]["real"]["raw"][w] / BL[L]["real"]["sigma"][w] for w in WINDOWS} for L in range(12)}
    N["gpt2"]["descriptive"] = {
        "real_gamma_max": max(v for L in gam for v in gam[L].values()),
        "real_steepest_layer": {w: min(range(12), key=lambda L: BL[L]["real"]["raw"][w]) for w in WINDOWS},
        "real_k6_gamma_per_doubling": {w: N["gpt2"]["k6"]["i"][w]["gamma"]["s_real"] for w in WINDOWS}}


# ----------------------------------------------------------------------------------------------- robustness job
# experiments/r2_job_robust.py: G (generator settings of the codable null, Gemma-2-2b layers 13 and 0), M (codable null
# on GPT-2 small, 12 layers, k = 32), P (feature power; only the layer-13 null completed). Rules from that docstring.
ROB = rows("runs_robust.jsonl") if (RES / "r2" / "runs_robust.jsonl").exists() else []
G_ROBUST_GAP, G_DEPENDENT_GAP = 0.15, 0.05
M_TRACK = 0.06
GPT_TOP, GPT_PRIMARY, GPT_BULK = "(1, 17)", "(3, 50)", ["(10, 150)", "(33, 333)"]
GPT_INTERIOR, GPT_ENDS = [3, 4, 5, 6, 7, 8], [0, 1, 10, 11]


def _setting_label(r):
    s = r["setting"]
    lab = f"M={s['M']}, L={s['L']}"
    if s.get("overflow"):
        lab += f", overflow {s['overflow'][0]}x{s['overflow'][1]}"
    return lab


def _mean_rq(L, run):
    """Mean Rayleigh quotient of a stored decoder covariance (live latents, centred): tr(C Sigma) / tr(C) divided by
    tr(Sigma) / d, the value for a random direction. Same Sigma (10M-token training cache) for every SAE at a layer."""
    p = RES / "r2" / "dec" / f"{run}.npz"
    if not p.exists():
        return None
    S = np.load(RES / "r2" / "sigma" / f"gemma-2-2b_L{L}.npz")["cov"].astype(np.float64)
    C = np.load(p)["cov"].astype(np.float64)
    return float(np.sum(C * S) / np.trace(C) / (np.trace(S) / S.shape[0]))


def robust():
    if not ROB:
        return
    out = {}
    l8 = N["layers8"]["by_layer"]
    # ---- G
    cal = json.load(open(RES / "r2" / "robust_calibration.json"))
    G = {}
    for r in [r for r in ROB if r.get("step") == "G"]:
        L = r["layer"]
        key = f"G|L{L}|M{r['setting']['M']}_L{r['setting']['L']}" + (
            f"_OF{r['setting']['overflow'][0]}x{r['setting']['overflow'][1]}" if r["setting"].get("overflow") else "")
        c = cal[key]
        real = {w: l8[L]["real"]["gamma"][w][0] for w in WINDOWS}
        G[key] = {"layer": L, "setting": _setting_label(r), "rho": r["rho"], "fvu": r["fvu_eval"], "fvu_target": c["target"],
                  "matched": bool(c["matched"]), "n_live": r["n_live"],
                  "null_gamma": {w: r["gamma_fixed"][w] for w in WINDOWS},
                  "null_minus_real": {w: r["gamma_fixed"][w] - real[w] for w in WINDOWS},
                  "mean_rq": _mean_rq(L, r["run"])}
    ref = {}
    for L in (0, 13):
        r0 = [r for r in _sp2_final(L, 40) if r["seed"] == 0][0]
        ref[L] = {"setting": "M=8192, L=16 (reference)", "rho": r0["rho"], "fvu": r0["fvu_eval"],
                  "null_minus_real": {w: r0["gamma_fixed"][w] - l8[L]["real"]["gamma"][w][0] for w in WINDOWS}}
    mt = {k: v for k, v in G.items() if v["matched"]}
    gaps = [v["null_minus_real"][w] for v in mt.values() for w in BULK]
    verdict = ("generator-dependent" if any(g <= G_DEPENDENT_GAP for g in gaps) else
               "robust" if mt and all(g >= G_ROBUST_GAP for g in gaps) else "partial")
    P_, T_ = "(10, 150)", "(3, 50)"

    def rng(L, w):
        v = [x["null_minus_real"][w] for x in mt.values() if x["layer"] == L]
        return [min(v), max(v)]
    out["G"] = {"settings": G, "reference": ref, "unmatched": sorted(k for k, v in G.items() if not v["matched"]),
                "n_matched": len(mt), "verdict": verdict, "min_bulk_gap": min(gaps), "max_bulk_gap": max(gaps),
                "bulk_gap_range_by_layer": {L: [min(x["null_minus_real"][w] for x in mt.values() if x["layer"] == L
                                                    for w in BULK),
                                                max(x["null_minus_real"][w] for x in mt.values() if x["layer"] == L
                                                    for w in BULK)] for L in (0, 13)},
                "primary_range_by_layer": {L: rng(L, P_) for L in (0, 13)},
                "top_range_by_layer": {L: rng(L, T_) for L in (0, 13)},
                "L13_primary_within_0.06": sorted(k for k, v in mt.items() if v["layer"] == 13 and
                                                  abs(v["null_minus_real"][P_]) <= 0.06),
                "rho_range_matched": [min(v["rho"] for v in mt.values()), max(v["rho"] for v in mt.values())],
                "fvu_abs_diff_max_matched": max(abs(v["fvu"] - v["fvu_target"]) for v in mt.values()),
                "mean_rq_range_matched": [min(v["mean_rq"] for v in mt.values()), max(v["mean_rq"] for v in mt.values())]}
    # ---- M (GPT-2, k = 32); real = job 3's M1 real SAEs (mean over seeds; two at layer 6)
    m1 = [r for r in MOD if r.get("step") == "M1"]
    M = {}
    for r in [r for r in ROB if r.get("step") == "M"]:
        L = r["layer"]
        rr = pick(m1, layer=L, cond="real", k=32)
        real = {w: float(np.mean([x["gamma_fixed"][w] for x in rr])) for w in GPT_WIN.values()}
        M[L] = {"rho": r["rho"], "fvu": r["fvu_eval"], "fvu_target": cal[f"M|L{L}"]["target"],
                "matched": bool(cal[f"M|L{L}"]["matched"]), "real_gamma": real,
                "null_gamma": {w: r["gamma_fixed"][w] for w in GPT_WIN.values()},
                "null_minus_real": {w: r["gamma_fixed"][w] - real[w] for w in GPT_WIN.values()}}
    ml = sorted(L for L in M if M[L]["matched"])
    frac = {w: sum(M[L]["null_minus_real"][w] >= 0.15 for L in ml) / len(ml) for w in GPT_BULK}
    bulk_v = ("replicates" if all(f >= 0.9 for f in frac.values()) else "fails" if any(f < 0.5 for f in frac.values())
              else "partial")
    track = [L for L in ml if abs(M[L]["null_minus_real"][GPT_PRIMARY]) <= M_TRACK]
    it = sum(L in track for L in GPT_INTERIOR if L in ml) / sum(L in ml for L in GPT_INTERIOR)
    en = sum(L in track for L in GPT_ENDS if L in ml)
    prim_v = "replicates" if it >= 2 / 3 and en <= 1 else "fails" if it < 1 / 3 or en >= 3 else "partial"
    real_steeper = [L for L in ml if M[L]["null_minus_real"][GPT_TOP] < 0]
    real_flatter = [L for L in ml if M[L]["null_minus_real"][GPT_TOP] > 0]
    top_v = ("replicates" if len(real_steeper) >= 0.75 * len(ml) else "fails" if len(real_flatter) >= 0.5 * len(ml)
             else "partial")
    out["M"] = {"by_layer": M, "matched_layers": ml, "rho0_layers": [L for L in ml if M[L]["rho"] == 0],
                "rho_range": [min(M[L]["rho"] for L in ml), max(M[L]["rho"] for L in ml)],
                "fvu_abs_diff_max": max(abs(M[L]["fvu"] - M[L]["fvu_target"]) for L in ml),
                "bulk_fraction_ge_0.15": frac,
                "bulk_gap_range": {w: [min(M[L]["null_minus_real"][w] for L in ml), max(M[L]["null_minus_real"][w] for L in ml)]
                                   for w in GPT_BULK},
                "bulk_below_0.15_layers": {w: [L for L in ml if M[L]["null_minus_real"][w] < 0.15] for w in GPT_BULK},
                "primary_tracking_layers": track, "primary_interior_fraction": it, "primary_end_count": en,
                "primary_abs_diff_range_tracking": [min(abs(M[L]["null_minus_real"][GPT_PRIMARY]) for L in track),
                                                    max(abs(M[L]["null_minus_real"][GPT_PRIMARY]) for L in track)],
                "primary_diff_not_tracking": {L: M[L]["null_minus_real"][GPT_PRIMARY] for L in ml if L not in track},
                "top_real_steeper_layers": real_steeper, "top_real_flatter_layers": real_flatter,
                "top_diff_range": [min(M[L]["null_minus_real"][GPT_TOP] for L in ml),
                                   max(M[L]["null_minus_real"][GPT_TOP] for L in ml)],
                "verdict": {"bulk": bulk_v, "primary": prim_v, "top": top_v}}
    # ---- P (feature power); only rows that exist. Head/body rule recomputed from the row.
    P = {}
    for r in [r for r in ROB if r.get("step") == "P"]:
        pw, uw = r["power_weighted_gamma"], r["unweighted"]["gamma_fixed"]
        a = all(abs(pw[w] - 1) <= 0.25 and uw[w] <= pw[w] - 0.15 for w in BULK)
        b = r["rq_median_log10"]["rest_1pct"] < 0.5
        P[f"L{r['layer']}_{r['cond']}"] = {
            "power_weighted_gamma": pw, "unweighted_gamma_fixed": uw, "share_top10": r["power_share_top10"],
            "share_top1pct": r["power_share_top1pct"], "ess_frac_live": r["power_ess_frac_live"],
            "rq_median_log10": r["rq_median_log10"], "rq_ratio_top10": 10 ** r["rq_median_log10"]["top10"],
            "rq_ratio_rest_1pct": 10 ** r["rq_median_log10"]["rest_1pct"], "spearman_pi_rq": r["spearman_pi_rq"],
            "trace_ratio": r["sum_pi_over_total_var"], "power_n_tokens": r["power_n_tokens"], "fvu": r["fvu_eval"],
            "rule_a": bool(a), "rule_b": bool(b), "present": bool(a and b)}
    n_real = sum(k.endswith("_real") for k in P)
    out["P"] = {"by_run": P, "n_real_layers": n_real, "n_null_layers": sum(not k.endswith("_real") for k in P),
                "verdict": "insufficient" if n_real < 4 else "see summary.json"}
    # like-for-like Rayleigh comparison from stored decoder covariances, all eight layers, k = 40
    rq = {}
    for L in LAYERS:
        rq[L] = {"real": [_mean_rq(L, r["run"]) for r in pick(REAL, step="Q1", layer=L, cond="real", k=40)],
                 "sparse_null_sae": [_mean_rq(L, r["run"]) for r in _sp2_final(L, 40)],
                 "gaussian_null_sae": [_mean_rq(L, r["run"]) for r in pick(REAL, step="Q1", layer=L, cond="gaussian", k=40)],
                 "null_dictionary": _mean_rq(L, f"gemma-2-2b__blocks.{L}.hook_resid_post__groundtruth__M8192__L16__halfnormal__ds0")}
    out["mean_rq"] = {"by_layer": rq, "range": {c: [min(min(rq[L][c]) for L in LAYERS), max(max(rq[L][c]) for L in LAYERS)]
                                                for c in ("real", "sparse_null_sae", "gaussian_null_sae")}}
    out["mean_rq"]["range"]["null_dictionary"] = [min(rq[L]["null_dictionary"] for L in LAYERS),
                                                  max(rq[L]["null_dictionary"] for L in LAYERS)]
    out["mean_rq"]["real_ge_null_sae_layers"] = [L for L in LAYERS if np.mean(rq[L]["real"]) >= np.mean(rq[L]["sparse_null_sae"])]
    out["n_rows"] = {"total": len(ROB), "G_and_GCAL": sum(r.get("step") in ("G", "GCAL") for r in ROB),
                     "M_and_MCAL": sum(r.get("step") in ("M", "MCAL") for r in ROB), "P": len(P)}
    N["robust"] = out
    # ---- agreement with experiments/r2_summary.py
    s = json.load(open(RES / "r2" / "summary.json"))["robust"]
    assert s["G_generator"]["verdict"] == verdict, (s["G_generator"]["verdict"], verdict)
    assert abs(s["G_generator"]["min_bulk_gap"] - out["G"]["min_bulk_gap"]) < 1e-9
    for k, v in G.items():
        for w in WINDOWS:
            assert abs(s["G_generator"]["settings"][k]["null_minus_real"][w] - v["null_minus_real"][w]) < 1e-9, (k, w)
    sm = s["M_gpt2_codable_null"]
    assert sm["verdict"] == out["M"]["verdict"], (sm["verdict"], out["M"]["verdict"])
    for L in M:
        for w in GPT_WIN.values():
            assert abs(sm["null_minus_real"][str(L)][w] - M[L]["null_minus_real"][w]) < 1e-9, (L, w)
    for k, v in P.items():
        assert s["P_feature_power"]["layers"][k]["head_body"]["present"] == v["present"], k
    N["summary_json_agrees_robust"] = True


def table_robust():
    """Main-text table: real minus null gamma (sign as in Table 1) for every generator setting of step G, with the
    reference setting. * = reference (seed-0 run of the main table), o = overflow, dagger = not matched."""
    if "robust" not in N:
        return
    G, ref = N["robust"]["G"]["settings"], N["robust"]["G"]["reference"]

    def key(kv):
        k = kv[0]
        return (int(k.split("|M")[1].split("_")[0]), int(k.split("_L")[1].split("_")[0]), "OF" in k)
    lines = []
    for L in (13, 0):
        items = [("8192$^{*}$", "16", ref[L])]
        for k, v in sorted(((k, v) for k, v in G.items() if v["layer"] == L), key=key):
            Mv = k.split("|M")[1].split("_")[0] + ("" if v["matched"] else r"$^\dagger$")
            Lv = k.split("_L")[1].split("_")[0] + (r"$^{o}$" if "OF" in k else "")
            items.append((Mv, Lv, v))
        for j, (Mv, Lv, v) in enumerate(items):
            cells = [str(L) if j == 0 else "", Mv, Lv, f"{v['rho']:.2f}", f"{v['fvu']:.3f}"]
            cells += [f"${-v['null_minus_real'][w]:+.2f}$" for w in WINDOWS]  # printed as real - null, as Table 1
            lines.append(" & ".join(cells) + r" \\")
        if L == 13:
            lines.append(r"\midrule")
    tex = (r"""\begin{tabular}{r r r r r rrrr}
\toprule
 & & & & & \multicolumn{4}{c}{real $-$ null $\gamma$} \\
\cmidrule(lr){6-9}
layer & $M$ & $L$ & $\rho$ & FVU & 4--50 & 11--150 & 31--450 & 101--1000 \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_robust_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_robust.jsonl, runs_sparse2.jsonl and "
        "runs_real.jsonl\n" + tex)


def fig_robust_gpt2():
    """GPT-2 small: real minus codable-sparse-null gamma at k = 32 over the 12 layers, four scaled windows."""
    if "robust" not in N:
        return
    M = N["robust"]["M"]["by_layer"]
    Ls = sorted(M)
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 1.9))
    ax.axvspan(-0.5, 2.5, color=OI["grey"], alpha=0.12, lw=0)
    ax.axhspan(-M_TRACK, M_TRACK, color=OI["green"], alpha=0.10, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(-0.15, color=OI["grey"], lw=0.6, ls="--")
    sty = {"(3, 50)": ("o", OI["orange"]), "(10, 150)": ("s", OI["blue"]), "(30, 450)": ("^", OI["sky"]),
           "(100, 1000)": ("v", OI["verm"])}
    for w in WINDOWS:
        g = GPT_WIN[w]
        mk, c = sty[w]
        ax.plot(Ls, [-M[L]["null_minus_real"][g] for L in Ls], marker=mk, ms=2.6, lw=0.9, color=c, label=GPT_LABEL[w])
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls], fontsize=6)
    ax.set_xlabel("GPT-2 small layer", fontsize=7)
    ax.set_ylabel(r"real $-$ null $\gamma$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.6, ncol=2, loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig12_gpt2_codable_r2")


# ----------------------------------------------------------------------------------------------- power-matched null
# experiments/r2_job_pmatch.py (rules fixed in notes/prereg_pmatch.md, 2026-11-13, with the sha256 of the job, generator,
# trainer and summary code). N = power-matched null (V1), R = real (three-seed mean, k = 40), Z = equal-power null (V0)
# at the same (layer, M, L): seed-0 sparse2 run for M 8192 L 16, robust step G for L 8 and L 32.
PMR = rows("runs_pmatch.jsonl") if (RES / "r2" / "runs_pmatch.jsonl").exists() else []
PM_MARGIN, PM_FRAC = 0.10, 0.30
PM_ORDER = ["PM|V1|L13|M8192_L16|ds0", "PM|V1|L13|M8192_L16|ds1", "PM|V1|L13|M8192_L8|ds0", "PM|V1|L13|M8192_L32|ds0",
            "PM|V1|L0|M8192_L16|ds0"]
PM_LABEL = {"PM|V1|L13|M8192_L16|ds0": "13, ref.", "PM|V1|L13|M8192_L16|ds1": "13, dict. 2",
            "PM|V1|L13|M8192_L8|ds0": "13, $L$=8", "PM|V1|L13|M8192_L32|ds0": "13, $L$=32",
            "PM|V1|L0|M8192_L16|ds0": "0, ref."}
# predictions recorded in notes/prereg_pmatch.md before any run
PM_PREDICTION = {"L13": "DOES NOT EXPLAIN", "L0": "PARTLY EXPLAINS", "headline": "PARTLY EXPLAINS",
                 "generator_settings": "same at other (M, L)", "replicate_within_0.05": True}


def pmatch():
    if not PMR:
        return
    l8 = N["layers8"]["by_layer"]
    acc = {r["setting_key"]: r for r in PMR if r.get("step") == "PM"}
    out = {"settings": {}, "margin": PM_MARGIN, "min_closed_fraction": PM_FRAC, "n_rows": len(PMR)}
    for key in PM_ORDER:
        r = acc[key]
        L, Ls = r["layer"], r["setting"]["L"]
        R = {w: l8[L]["real"]["gamma"][w][0] for w in WINDOWS}
        if Ls == 16:
            z = [x for x in _sp2_final(L, 40) if x["seed"] == 0][0]
            Z = z["gamma_fixed"]
            Z2 = {w: N["sparse2"]["by_layer"][L]["gamma"][w][0] for w in WINDOWS}  # two-seed mean, for reference
            zrho = z["rho"]
        else:
            z = [x for x in ROB if x.get("step") == "G" and x["layer"] == L and x["setting"]["M"] == 8192
                 and x["setting"]["L"] == Ls and not x["setting"].get("overflow")][0]
            Z, Z2, zrho = z["gamma_fixed"], None, z["rho"]
        Nv = r["gamma_fixed"]
        win = {}
        for w in BULK:
            f = (Z[w] - Nv[w]) / (Z[w] - R[w])
            cls = ("closed" if abs(Nv[w] - R[w]) <= PM_MARGIN else "overshoot" if Nv[w] < R[w] - PM_MARGIN
                   else "partial" if f >= PM_FRAC else "open")
            win[w] = {"real": R[w], "equal_power": Z[w], "power_matched": Nv[w], "gap_equal_power": Z[w] - R[w],
                      "gap_power_matched": Nv[w] - R[w], "closed_fraction": f, "class": cls}
        cl = [win[w]["class"] for w in BULK]
        v = "EXPLAINS" if all(c == "closed" for c in cl) else "DOES NOT EXPLAIN" if "open" in cl else "PARTLY EXPLAINS"
        dic, pc = r["dictionary"], r["power_check"]
        a1, a0 = dic["atom_cloud_gamma_fixed"], dic["v0_atom_cloud_gamma_fixed"]
        out["settings"][key] = {
            "layer": L, "M": r["setting"]["M"], "L": Ls, "ds": r["setting"]["ds"], "rho": r["rho"], "rho_equal_power": zrho,
            "fvu": r["fvu_eval"], "fvu_target": r["fvu_target"], "matched": bool(r["matched"]),
            "n_calibration_runs": sum(x["setting_key"] == key for x in PMR), "n_live": r["n_live"],
            "windows": win, "verdict": v,
            "all_windows": {"real": R, "equal_power": Z, "equal_power_two_seed_mean": Z2, "power_matched": Nv},
            "top_primary_move_from_real": {w: abs(Nv[w] - R[w]) - abs(Z[w] - R[w]) for w in ("(3, 50)", "(10, 150)")},
            "atom_cloud_gamma": {"power_matched": a1, "equal_power": a0},
            "additive_prediction": {w: a1[w] + Z[w] - a0[w] for w in WINDOWS},
            "dictionary": {"share_top1pct": dic["power"]["share_top1pct"], "share_top10": dic["power"]["share_top10"],
                           "target_share_top1pct": dic["power_target"]["share_top1pct"],
                           "power_moved_by_projection": dic["power_share_moved_by_projection"],
                           "cov_rel_err_float64": f'{dic["cov_rel_frob_err_float64"]:.1e}',
                           "cov_rel_err_float32": f'{dic["cov_rel_frob_err_float32"]:.1e}',
                           "sample_cov_rel_err_1M": r["sparse"]["check_1M"]["rel_frob_err"],
                           "max_abs_log_power_err": dic["final_power_max_abs_log_err"], "converged": dic["converged"],
                           "iterations": dic["iterations"], "spearman_power_rq": dic["spearman_power_rq"],
                           "rq_median_log10": dic["rq_log10_quantiles"]["0.5"]},
            "manipulation_check": {"null_share_top1pct": pc["null"]["share_top1pct"],
                                   "real_share_top1pct": pc["real"]["share_top1pct"],
                                   "null_share_top10": pc["null"]["share_top10"], "real_share_top10": pc["real"]["share_top10"],
                                   "null_ess_frac": pc["null"]["ess_frac"], "real_ess_frac": pc["real"]["ess_frac"],
                                   "null_body_rq_median_log10": pc["rq_median_log10"]},
            "train_seconds": r["train_seconds"]}
    S = out["settings"]
    per = {"L13": S["PM|V1|L13|M8192_L16|ds0"]["verdict"], "L0": S["PM|V1|L0|M8192_L16|ds0"]["verdict"]}
    head = per["L13"] if per["L13"] == per["L0"] else "PARTLY EXPLAINS"
    gen = ("same at other (M, L)" if all(S[k]["verdict"] == per["L13"] for k in
                                         ("PM|V1|L13|M8192_L8|ds0", "PM|V1|L13|M8192_L32|ds0")) else "differs")
    rep = max(abs(S["PM|V1|L13|M8192_L16|ds1"]["windows"][w]["power_matched"] -
                  S["PM|V1|L13|M8192_L16|ds0"]["windows"][w]["power_matched"]) for w in BULK)
    cf = [S[k]["windows"][w]["closed_fraction"] for k in PM_ORDER for w in BULK]
    out.update({"per_layer": per, "headline": head, "generator_settings": gen, "replicate_max_abs_bulk_diff": rep,
                "replicate_flag": rep > 0.05, "closed_fraction_range_all": [min(cf), max(cf)],
                "prediction": PM_PREDICTION,
                "prediction_met": {"L13": per["L13"] == PM_PREDICTION["L13"], "L0": per["L0"] == PM_PREDICTION["L0"],
                                   "headline": head == PM_PREDICTION["headline"],
                                   "generator_settings": gen == PM_PREDICTION["generator_settings"],
                                   "replicate": rep <= 0.05},
                "gpu_minutes_total": sum(x["train_seconds"] + x.get("score_seconds", 0) for x in PMR) / 60})
    N["pmatch"] = out
    # ---- agreement with experiments/r2_summary.py
    s = json.load(open(RES / "r2" / "summary.json"))["pmatch"]
    assert s["headline"]["verdict"] == head, (s["headline"], head)
    assert s["headline"]["per_layer"] == per, (s["headline"]["per_layer"], per)
    assert s["generator_settings"]["verdict"] == gen
    assert abs(s["replicate"]["max_abs_bulk_diff"] - rep) < 1e-9
    for key, v in S.items():
        sv = s["settings"][key]
        assert sv["verdict"] == v["verdict"], (key, sv["verdict"], v["verdict"])
        for w in BULK:
            for a, b in (("null", "power_matched"), ("real", "real"), ("v0", "equal_power")):
                assert abs(sv["windows"][w][a] - v["windows"][w][b]) < 1e-9, (key, w, a)
            assert abs(sv["windows"][w]["closed_fraction"] - v["windows"][w]["closed_fraction"]) < 1e-9
            assert sv["windows"][w]["class"] == v["windows"][w]["class"]
        for w in WINDOWS:
            assert abs(sv["additive_prediction"][w] - v["additive_prediction"][w]) < 1e-9
    N["summary_json_agrees_pmatch"] = True
    # feature power on the real SAEs (robust step P), restated against summary.json
    sp = json.load(open(RES / "r2" / "summary.json"))["robust"]["P_feature_power"]
    P = N["robust"]["P"]["by_run"]
    n_real_present = sum(P[k]["present"] for k in P if k.endswith("_real"))
    assert sp["present_real"] == f"{n_real_present}/{sum(k.endswith('_real') for k in P)}"
    assert sp["verdict"] == "insufficient" == N["robust"]["P"]["verdict"]
    for k in P:
        for w in WINDOWS:
            assert abs(sp["layers"][k]["power_weighted_gamma"][w] - P[k]["power_weighted_gamma"][w]) < 1e-9
    N["feature_power"] = {"present_real": sp["present_real"], "present_null": sp["present_null"], "verdict": sp["verdict"],
                          "prediction": "supported"}


def fig_pmatch():
    """Main-text figure: the bulk gap decomposed. Per bulk window and setting: real, equal-power null, power-matched
    null, with the +-0.10 band of the pre-registered 'closed' class around real and the fraction of the gap closed."""
    if "pmatch" not in N:
        return
    S = N["pmatch"]["settings"]
    # Settings with three SAE seeds (results/r2_pm2/pmatch2_summary.json, written by experiments/r2_pmatch2_summary.py)
    # are drawn from their three-seed means: R three seeds, Z two seeds (three at 32,768 atoms), N three seeds with
    # its seed sd and seed dots. The other settings are the first run (one seed per null, Z seed 0). Nothing drawn
    # here is written to N, so paper_numbers_r2.json does not depend on this figure.
    p2f = RES / "r2_pm2" / "pmatch2_summary.json"
    P2 = json.load(open(p2f))["settings"] if p2f.exists() else {}
    order = PM_ORDER[:4] + (["PM|V1|L13|M32768_L16|ds0"] if "PM|V1|L13|M32768_L16|ds0" in P2 else []) + PM_ORDER[4:]
    label = {**PM_LABEL, "PM|V1|L13|M32768_L16|ds0": "13, $M$=32,768"}

    def vals(k, w):
        if k in P2:
            x = P2[k]["windows"][w]
            return {"real": x["R"], "equal_power": x["Z"], "power_matched": x["N"], "sd": x["N_sd"],
                    "seeds": list(x["N_per_seed"].values()), "closed_fraction": x["closed_fraction"], "three": True}
        o = S[k]["windows"][w]
        return {**o, "sd": None, "seeds": [], "three": False}

    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.45), sharey=False)
    xs = np.arange(len(order))
    sep = len(order) - 1.5  # between the layer-13 settings and layer 0
    for j, w in enumerate(BULK):
        ax = axes[j]
        for i, k in enumerate(order):
            o = vals(k, w)
            ax.add_patch(plt.Rectangle((i - 0.32, o["real"] - PM_MARGIN), 0.64, 2 * PM_MARGIN, color=OI["blue"],
                                       alpha=0.13, lw=0))
            ax.plot([i], [o["real"]], "o", color=OI["blue"], ms=3.4, label="real" if i == 0 else None)
            ax.plot([i], [o["equal_power"]], "^", mfc="white", mec=OI["green"], ms=3.8, mew=0.9,
                    label="equal-power null" if i == 0 else None)
            ax.plot([i], [o["power_matched"]], "D", color=OI["verm"], ms=3.2, label="power-matched null" if i == 0 else None)
            if o["three"]:
                ax.plot([i + 0.2] * len(o["seeds"]), o["seeds"], ".", color=OI["verm"], ms=1.6, alpha=0.7)
                ax.errorbar([i], [o["power_matched"]], yerr=[o["sd"]], color=OI["verm"], lw=0.6, capsize=1.2)
            ax.annotate("", xy=(i, o["power_matched"] + 0.012), xytext=(i, o["equal_power"] - 0.015),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=OI["grey"], mutation_scale=5))
            ax.text(i, o["equal_power"] + 0.03, f"{100 * o['closed_fraction']:.0f}%", ha="center", va="bottom",
                    fontsize=5.6, fontweight="bold" if o["three"] else "normal")
        ax.axvline(sep, color=OI["grey"], lw=0.5, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([label[k] + (r"$^{*}$" if k in P2 else "") for k in order], rotation=55, ha="right",
                           fontsize=6.2)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.set_title(WIN_LABEL[w], fontsize=7.5)
        ax.set_xlim(-0.6, len(order) - 0.4)
        lo = min(min(vals(k, w)["real"] for k in order) - 0.14, 0.3)
        hi = max(vals(k, w)["equal_power"] for k in order) + 0.15
        ax.set_ylim(lo, hi)
        if j == 0:
            ax.set_ylabel(r"$\gamma$ at 8,192 live latents", fontsize=7)
        panel_tag(ax, "ab"[j], x=-0.1)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=6.6, frameon=False,
               handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    save(fig, "fig13_pmatch_r2")


def table_pmatch():
    """Main-text table: real, equal-power and power-matched gamma per setting in all four windows, with the fraction of
    the bulk gap closed and the pre-registered verdict."""
    if "pmatch" not in N:
        return
    S = N["pmatch"]["settings"]
    short = {"EXPLAINS": "explains", "PARTLY EXPLAINS": "partly", "DOES NOT EXPLAIN": "does not"}
    lines = []
    for k in PM_ORDER:
        o = S[k]
        a = o["all_windows"]
        cells = [PM_LABEL[k], f"{o['rho']:.2f}", f"{o['fvu']:.3f}"]
        for w in ("(3, 50)", "(10, 150)"):
            cells.append(f"{a['real'][w]:.2f}/{a['equal_power'][w]:.2f}/{a['power_matched'][w]:.2f}")
        for w in BULK:
            x = o["windows"][w]
            cells.append(f"{x['real']:.2f}/{x['equal_power']:.2f}/{x['power_matched']:.2f}")
            cells.append(f"{100 * x['closed_fraction']:.0f}")
        cells.append(short[o["verdict"]])
        lines.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{tabular}{l r r cc cr cr l}
\toprule
 & & & \multicolumn{2}{c}{$\gamma$: real / equal / matched} & \multicolumn{4}{c}{bulk: $\gamma$ real / equal / matched, \% closed} & \\
\cmidrule(lr){4-5}\cmidrule(lr){6-9}
layer, setting & $\rho$ & FVU & 4--50 & 11--150 & 31--450 & \% & 101--1000 & \% & verdict \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_pmatch_r2.tex").write_text(
        "% generated by paper/figures/make_fig_null_r2.py from results/r2/runs_pmatch.jsonl, runs_sparse2.jsonl, "
        "runs_robust.jsonl and runs_real.jsonl\n" + tex)


def bulk_bias():
    """Finite-n bias of gamma in every window: unit-normalised Gaussian clouds with covariance Sigma_13^g (training-set
    Sigma), n points, centred, two draws; gamma = cloud slope / Sigma slope. Extends the ranks 11-150 calibration table
    of the appendix to the bulk windows at the scoring size n = 8,192."""
    S = np.load(RES / "r2" / "sigma" / "gemma-2-2b_L13.npz")["cov"].astype(np.float64)
    lam, U = np.linalg.eigh(S)
    lam, U = np.clip(lam[::-1], 1e-12, None), U[:, ::-1]
    s_sig = {w: slope_from_eigenvalues(lam, eval(w))["slope"] for w in WINDOWS}
    out = {}
    for g in (0.5, 1.0):
        for n in (8192,):
            vals = {w: [] for w in WINDOWS}
            for dr in range(2):
                rng = np.random.default_rng(1000 * dr + int(10 * g))
                X = (rng.standard_normal((n, len(lam))) * lam ** (g / 2)) @ U.T
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X -= X.mean(0)
                e = np.linalg.eigvalsh(X.T @ X / (n - 1))[::-1]
                for w in WINDOWS:
                    vals[w].append(slope_from_eigenvalues(e, eval(w))["slope"] / s_sig[w])
            out[f"g{g}_n{n}"] = {w: float(np.mean(v)) for w, v in vals.items()}
    N["bulk_bias"] = out


def closed_band():
    """Rough SAE-seed band on the closed share (Z - N)/(Z - R). Each null setting has one SAE seed, so its seed sd is
    assumed equal to the seed sd of the two-seed equal-power null at the same layer (reference setting); real is the
    three-seed mean with its own seed sd. Two-seed sds are noisy, so the null sd is the largest seed sd of the
    equal-power null over the eight layers and both bulk windows (conservative). Normal draws, 20,000 replicates,
    5th-95th percentile. Descriptive only."""
    if "pmatch" not in N:
        return
    rng = np.random.default_rng(0)
    out = {}
    sz = max(N["sparse2"]["by_layer"][L]["gamma"][w][1] for L in LAYERS for w in BULK)
    for key in PM_ORDER:
        o = N["pmatch"]["settings"][key]
        L = o["layer"]
        out[key] = {}
        for w in BULK:
            x = o["windows"][w]
            sr = N["layers8"]["by_layer"][L]["real"]["gamma"][w][1]
            R = x["real"] + rng.standard_normal(20000) * sr / np.sqrt(3)
            Z = x["equal_power"] + rng.standard_normal(20000) * sz
            Nv = x["power_matched"] + rng.standard_normal(20000) * sz
            f = (Z - Nv) / (Z - R)
            out[key][w] = {"sd_real": sr, "sd_null_assumed": sz,
                           "band90": [float(np.percentile(f, 5)), float(np.percentile(f, 95))]}
    N["pmatch_closed_band"] = {"by_setting": out, "sd_null_assumed": sz, "max_half_width": max(
        max(v[w]["band90"][1] - N["pmatch"]["settings"][k]["windows"][w]["closed_fraction"],
            N["pmatch"]["settings"][k]["windows"][w]["closed_fraction"] - v[w]["band90"][0])
        for k, v in out.items() for w in BULK)}


def trace_ratio():
    """Share of the variance carried by the latent powers, sum_i pi_i / total variance, for the step-P SAEs (real seed 0
    and the equal-power null at layers 13 and 0). 'row' is the job's own value, against the empirical variance of the
    1M tokens on which power was measured; 'sigma10M' recomputes it from power/*.npz against tr(Sigma) of the 10M-token
    Sigma that the nulls are built from, in the same input-scaled units (E||x||^2 = d)."""
    out = {}
    for L in (13, 0):
        s = np.load(RES / "r2" / "sigma" / f"gemma-2-2b_L{L}.npz")
        cov, mu = s["cov"].astype(np.float64), s["mean"].astype(np.float64)
        tr = float(np.trace(cov))
        tv = cov.shape[0] / (tr + float(mu @ mu)) * tr
        for c, cond in (("real", "real"), ("sparse", "equal_power_null")):
            f = sorted((RES / "r2" / "power").glob(f"*blocks.{L}.hook_resid_post__{c}__*.npz"))[0]
            z = np.load(f)
            row = [r for r in ROB if r.get("step") == "P" and r["layer"] == L and
                   (r["cond"] == "real") == (c == "real")][0]
            out[f"L{L}_{cond}"] = {"row": row["sum_pi_over_total_var"],
                                   "sigma10M": float(z["pi"][z["live"]].sum() / tv)}
    N["trace_ratio"] = out


GEN_ORDER = {13: [("G|L13|M4096_L16", "4096, 16"), ("ref", "8192, 16"), ("G|L13|M32768_L16", "32768, 16"),
                  ("G|L13|M8192_L8", "8192, 8"), ("G|L13|M8192_L32", "8192, 32"),
                  ("G|L13|M8192_L16_OF65536x64", "overflow")],
             0: [("G|L0|M4096_L16", "4096, 16"), ("ref", "8192, 16"), ("G|L0|M8192_L8", "8192, 8"),
                 ("G|L0|M8192_L32", "8192, 32")]}
GEN_PM = {(13, "ref"): "PM|V1|L13|M8192_L16|ds0", (13, "G|L13|M8192_L8"): "PM|V1|L13|M8192_L8|ds0",
          (13, "G|L13|M8192_L32"): "PM|V1|L13|M8192_L32|ds0", (0, "ref"): "PM|V1|L0|M8192_L16|ds0"}


def fig_generator():
    """Main-text figure: the equal-power bulk gap (null minus real gamma) at every matched generator setting at layers
    13 and 0, and, where the power-matched null was run, the part of it that power matching closes."""
    if "robust" not in N or "pmatch" not in N:
        return
    G, ref = N["robust"]["G"]["settings"], N["robust"]["G"]["reference"]
    S = N["pmatch"]["settings"]
    rec = []
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.5), sharey=True)
    for j, w in enumerate(BULK):
        ax = axes[j]
        x = 0
        ticks, labels = [], []
        for L in (13, 0):
            for key, lab in GEN_ORDER[L]:
                v = ref[L] if key == "ref" else G[key]
                if key != "ref" and not v["matched"]:
                    continue
                gap = v["null_minus_real"][w]
                pm = GEN_PM.get((L, key))
                if pm is None:
                    ax.bar(x, gap, 0.7, color=OI["green"], alpha=0.35, lw=0)
                    r = {"layer": L, "setting": lab, "window": w, "gap_equal_power": gap}
                else:
                    o = S[pm]["windows"][w]
                    rem = o["gap_power_matched"]
                    ax.bar(x, rem, 0.7, color=OI["verm"], lw=0)
                    ax.bar(x, o["gap_equal_power"] - rem, 0.7, bottom=rem, color=OI["green"], alpha=0.35, lw=0,
                           hatch="////", edgecolor=OI["verm"])
                    ax.text(x, o["gap_equal_power"] + 0.012, f"{100 * o['closed_fraction']:.0f}%", ha="center",
                            va="bottom", fontsize=5.6, rotation=90)
                    r = {"layer": L, "setting": lab, "window": w, "gap_equal_power": o["gap_equal_power"],
                         "gap_power_matched": rem, "closed_fraction": o["closed_fraction"]}
                rec.append(r)
                ticks.append(x)
                labels.append(lab)
                x += 1
            if L == 13:
                ax.axvline(x - 0.5 + 0.35, color=OI["grey"], lw=0.5, ls=":")
                x += 0.7
        ax.axhline(0.10, color=OI["grey"], lw=0.5, ls="--")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=5.6)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.set_title(WIN_LABEL[w], fontsize=7.5)
        ax.text(2.5, 0.72, "layer 13", ha="center", fontsize=6.2)
        ax.text(8.2, 0.72, "layer 0", ha="center", fontsize=6.2)
        ax.set_ylim(0, 0.78)
        if j == 0:
            ax.set_ylabel(r"null $-$ real $\gamma$", fontsize=7)
        panel_tag(ax, "ab"[j], x=-0.1)
    from matplotlib.patches import Patch
    hs = [Patch(color=OI["green"], alpha=0.35, lw=0, label="equal-power gap"),
          Patch(facecolor=OI["green"], alpha=0.35, hatch="////", edgecolor=OI["verm"], lw=0,
                label="closed by power matching"),
          Patch(color=OI["verm"], lw=0, label="remaining")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=5.8, frameon=False,
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.4)
    save(fig, "fig14_generator_r2")
    gaps = [r["gap_equal_power"] for r in rec]
    N["generator_figure"] = {"bars": rec, "gap_range": [min(gaps), max(gaps)]}


def main():
    out = layers8()
    fig_layers(out)
    table_layers(out)
    cur, res = ksweep()
    cur_sp = sparse2(out, cur) if SP2 else None
    fig_k(cur, res, cur_sp)
    whiten()
    g, gc = gpt2()
    fig_gpt2(g, gc)
    sigma_conv()
    earlier_set()
    check_summary()
    if SP2:
        table_sparse2()
    old_sparse()
    gemma9b()
    check_summary_new()
    equivalence()
    fig_matched()
    table_matched()
    local_analyses()
    operating_point()
    robust()
    table_robust()
    fig_robust_gpt2()
    pmatch()
    fig_pmatch()
    table_pmatch()
    closed_band()
    trace_ratio()
    fig_generator()
    bulk_bias()
    p = RES / "r2" / "paper_numbers_r2.json"
    p.write_text(json.dumps(rnd(N), indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
