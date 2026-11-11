"""Regenerate every paper figure from files under results/ (nothing typed in by hand).

Run from anywhere:  python paper/figures/make_figures.py
Writes paper/figures/fig{1..6}_*.pdf (vector) and .png (preview), and prints the numbers each caption quotes.

Conventions
- Rank windows are 0-based [start, end) slices of the sorted spectrum, as in geometry._fit: (3, 50) is ranks 4-50,
  (10, 150) ranks 11-150 (the pre-registered primary window), (30, 450) ranks 31-450, (100, 1000) ranks 101-1000.
  "Top" windows are 4-50 and 11-150; "bulk" windows are 31-450 and 101-1000.
- Eigen-slope = slope of log10(eigenvalue) on log10(rank) over the window (dimensionless). gamma = decoder slope /
  Sigma slope (dimensionless; gamma = 1 means the decoder spectrum falls as steeply as the activation spectrum).
- Every Gemma Scope gamma or Sigma slope uses the 1M-token Sigma (results/test1/sigma_1m_compare.json,
  sigma_eigenvalues_1m.npy). results/test1/gamma_l0_L12.json stores gammas against the 200k-token Sigma; we convert
  them exactly with gamma_1m = gamma_200k * s200k / s1m, where s are that file's sigma12_slopes and the 1M layer-12
  slope. The script checks that the converted gammas reproduce sigma_1m_compare.json's l12_law fit.
- Our own TopK SAEs (results/gpu/) use the Sigma of their own 10M training tokens (results/gpu/sigma.json).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

W1, W2 = 3.25, 6.75  # ICML single / double column, inches
WINDOWS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WIN_LABEL = {"(3, 50)": "ranks 4–50", "(10, 150)": "ranks 11–150", "(30, 450)": "ranks 31–450",
             "(100, 1000)": "ranks 101–1000"}
PRIMARY = "(10, 150)"

# Okabe-Ito (colour-blind safe)
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
COND = {"real": OI["blue"], "gaussian": OI["verm"], "scrambled": OI["green"]}
COND_LABEL = {"real": "real activations", "gaussian": r"Gaussian $N(\mu,\Sigma)$ null", "scrambled": "scrambled null"}
WIDTHS = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
WCOL = dict(zip(WIDTHS, plt.get_cmap("viridis")(np.linspace(0.0, 0.88, len(WIDTHS)))))
WMARK = dict(zip(WIDTHS, ["o", "s", "^", "D", "v", "P", "X"]))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
})

NUMBERS = {}  # caption numbers, printed at the end


def load(rel):
    with open(RES / rel) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def sigma1m():
    """1M-token Sigma slopes per window and layer, cross-checked against the raw eigenvalues."""
    c = load("test1/sigma_1m_compare.json")
    s = {w: np.array(c["sigma_slope"]["1m"][w]) for w in WINDOWS}
    eig = np.load(RES / "test1" / "sigma_eigenvalues_1m.npy")
    for w in WINDOWS:
        rng = tuple(int(x) for x in w.strip("()").split(","))
        recomputed = np.array([slope_from_eigenvalues(eig[L], rng)["slope"] for L in range(eig.shape[0])])
        assert np.allclose(recomputed, s[w], atol=1e-6), f"1M Sigma slope mismatch in {w}"
    return s, c


def gemma_l12_rows():
    """Layer-12 Gemma Scope SAEs: width, L0 and gamma against the 1M Sigma, plus the decoder slope."""
    g = load("test1/gamma_l0_L12.json")
    s1m, c = sigma1m()
    rows = []
    for r in g["rows"]:
        dec = {w: r["gamma"][w] * g["sigma12_slopes"][w] for w in WINDOWS}
        rows.append({"sae_id": r["sae_id"], "width": r["width"], "l0": r["l0"], "l0_source": r["l0_source"],
                     "dec": dec, "gamma": {w: dec[w] / s1m[w][12] for w in WINDOWS}})
    for w in WINDOWS:  # the converted gammas must reproduce the stored 1M law
        x = np.log2([r["l0"] for r in rows])
        y = np.array([r["gamma"][w] for r in rows])
        b, a = np.polyfit(x, y, 1)
        law = c["l12_law"][f"1m|{w}"]
        assert abs(a - law["a"]) < 1e-6 and abs(b - law["b_per_doubling"]) < 1e-6, w
    return rows, s1m, c


# --------------------------------------------------------------------------------------------- Figure 1
def fig1():
    """Decoder eigen-slope vs L0 at layer 12, by width; eigenvalues from results/phase1/eig_layer_12__*.npy."""
    rows, s1m, _ = gemma_l12_rows()
    lay = load("phase1/layers.json")["results"]
    pts = []
    for r in rows:
        nominal = r["sae_id"].split("average_l0_")[1]
        width = r["width"]
        eig = np.load(RES / "phase1" / f"eig_layer_12__width_{width}__average_l0_{nominal}.npy")
        sl = {}
        for w in WINDOWS:
            rng = tuple(int(x) for x in w.strip("()").split(","))
            sl[w] = slope_from_eigenvalues(eig, rng)["slope"]
            assert abs(sl[w] - r["dec"][w]) < 1e-6, (r["sae_id"], w)  # same spectra as gamma_l0_L12.json
        key = f"layer_12/width_{width}/average_l0_{nominal}"
        if key in lay:
            ref = {str(tuple(x["fit_range"])): x["slope"] for x in lay[key]["windows"]}
            assert all(abs(ref[w] - sl[w]) < 1e-6 for w in WINDOWS), key
        pts.append((width, r["l0"], sl))

    fig = plt.figure(figsize=(W2, 2.9))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.35, 1.35, 1.0], hspace=0.55, wspace=0.42)
    big = fig.add_subplot(gs[:, :2])
    small = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    order = [PRIMARY] + [w for w in WINDOWS if w != PRIMARY]
    for ax, w in zip([big] + small, order):
        for width in WIDTHS:
            p = sorted([(l0, sl[w]) for wd, l0, sl in pts if wd == width])
            if not p:
                continue
            x, y = zip(*p)
            ax.plot(x, y, "-", color=WCOL[width], marker=WMARK[width], lw=0.9 if ax is big else 0.7,
                    ms=4 if ax is big else 2.5, label=f"{width} ({len(p)})")
        ax.axhline(s1m[w][12], color=OI["grey"], ls="--", lw=0.8)
        log2_axis(ax, [16, 64, 256] if ax is not big else [16, 32, 64, 128, 256, 512])
        ax.set_xlim(10, 520)
    big.set_xlabel(r"$L_0$ (mean active latents per token)")
    big.set_ylabel(f"decoder eigen-slope, {WIN_LABEL[PRIMARY]}")
    big.text(11, s1m[PRIMARY][12] + 0.012, r"$\Sigma$ slope (1M tokens), $\gamma=1$", color="#555555", fontsize=7)
    big.legend(title="width (no. of SAEs)", title_fontsize=7, loc="lower right", ncol=2, handlelength=1.6)
    panel_tag(big, "a")
    for ax, w, t in zip(small, order[1:], "bcd"):
        ax.set_title(WIN_LABEL[w], fontsize=7, pad=2)
        ax.tick_params(labelsize=6.5)
        panel_tag(ax, t)
    small[-1].set_xlabel(r"$L_0$")
    small[1].set_ylabel("decoder eigen-slope")
    save(fig, "fig1_slope_vs_l0_L12")

    y = {w: [sl[w] for _, _, sl in pts] for w in WINDOWS}
    spread = []
    for lo, hi in [(19, 24), (36, 44), (64, 86)]:  # matched-L0 buckets across widths
        v = [sl[PRIMARY] for _, l0, sl in pts if lo <= l0 <= hi]
        spread.append((lo, hi, len(v), max(v) - min(v)))
    NUMBERS["fig1"] = {"n_saes": len(pts), "primary_slope_range": (min(y[PRIMARY]), max(y[PRIMARY])),
                       "sigma1m_L12": {w: float(s1m[w][12]) for w in WINDOWS},
                       "width_spread_primary_at_matched_L0": spread}


# --------------------------------------------------------------------------------------------- Figure 2
def runs_p2():
    rr = [r for r in load("gpu/runs.json")["runs"] if r["group"] == "saes_gpu_p2"]
    out = {}
    for r in rr:
        out.setdefault((r["cond"], r["width"], r["layer"]), []).append(r)
    return out


def fig2():
    s1m, c = sigma1m()
    m = load("phase1/slope_vs_layer_matched_l0.json")
    gsig = load("gpu/sigma.json")
    p2 = runs_p2()
    L = np.arange(26)
    fig, axes = plt.subplots(2, 2, figsize=(W2, 3.9), sharex=True)
    rvals = {}
    for ax, w, t in zip(axes.ravel(), WINDOWS, "abcd"):
        wl = WIN_LABEL[w].replace("–", "-").replace(" ", "")
        ax.plot(L, s1m[w], "-", color=OI["black"], lw=1.2, label=r"$\Sigma$ (1M tokens)")
        for width, ls in (("16k", "-"), ("65k", "--")):
            d = m[f"{width}_{wl}_L0_40"]
            ax.plot([int(k) for k in d], list(d.values()), ls, color=OI["grey"], lw=1.0,
                    label=f"Gemma Scope {width} decoder, $L_0$=40")
            rvals[(width, w)] = float(np.corrcoef([d[str(i)] for i in L], s1m[w])[0, 1])
        lays = [3, 13, 23]
        ax.plot(lays, [gsig[f"blocks.{l}.hook_resid_post"]["slopes"][w] for l in lays], "*", color=OI["black"],
                ms=6, mfc="white", mew=0.8, label=r"$\Sigma$ of our training tokens (10M)")
        for i, cond in enumerate(["real", "gaussian", "scrambled"]):
            for j, (width, mfc) in enumerate((("16k", None), ("65k", "white"))):
                off = (i - 1) * 0.55 + (j - 0.5) * 0.22
                mu, sd = [], []
                for l in lays:
                    v = [r["decoder_slope"][w] for r in p2[(cond, 16384 if width == "16k" else 65536, l)]]
                    mu.append(np.mean(v)); sd.append(np.std(v, ddof=1))
                ax.errorbar(np.array(lays) + off, mu, yerr=sd, fmt="o", ms=3, color=COND[cond], mfc=mfc or COND[cond],
                            mew=0.8, elinewidth=0.8, capsize=1.5)
        ax.set_ylabel(f"eigen-slope, {WIN_LABEL[w]}")
        panel_tag(ax, t)
        ax.set_xlim(-0.8, 25.8)
    for ax in axes[1]:
        ax.set_xlabel("layer")
        ax.set_xticks([0, 3, 6, 9, 13, 16, 19, 23, 25])
    axes[0, 0].text(0.0, 1.13, "top windows", transform=axes[0, 0].transAxes, fontsize=8, style="italic")
    axes[1, 0].text(0.0, 1.06, "bulk windows", transform=axes[1, 0].transAxes, fontsize=8, style="italic")
    handles = [Line2D([], [], color=OI["black"], lw=1.2, label=r"$\Sigma$, 1M tokens"),
               Line2D([], [], color=OI["grey"], lw=1.0, label=r"Gemma Scope 16k, $L_0$=40"),
               Line2D([], [], color=OI["grey"], lw=1.0, ls="--", label=r"Gemma Scope 65k, $L_0$=40"),
               Line2D([], [], color=OI["black"], marker="*", ls="", mfc="white", ms=6, label=r"$\Sigma$, our 10M tokens")]
    handles += [Line2D([], [], color=COND[cnd], marker="o", ls="", label=f"ours, {COND_LABEL[cnd]}")
                for cnd in ["real", "gaussian", "scrambled"]]
    handles += [Line2D([], [], color="k", marker="o", ls="", label="filled 16k, open 65k (mean ± sd, 3 seeds, k=40)",
                       mfc="none", ms=0)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.11), columnspacing=1.2,
               handlelength=1.8)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    save(fig, "fig2_layer_profile")

    ours = {}
    for (cond, width, l), rs in p2.items():
        for w in WINDOWS:
            ours[(cond, width, l, w)] = (float(np.mean([r["decoder_slope"][w] for r in rs])),
                                         float(np.std([r["decoder_slope"][w] for r in rs], ddof=1)),
                                         float(np.mean([r["gamma"][w] for r in rs])))
    rng = {}
    for cond in ["real", "gaussian", "scrambled"]:
        for grp, ws in (("top", WINDOWS[:2]), ("bulk", WINDOWS[2:])):
            g = [ours[(cond, wd, l, w)][2] for wd in (16384, 65536) for l in (3, 13, 23) for w in ws]
            rng[(cond, grp)] = (min(g), max(g))
    NUMBERS["fig2"] = {"pearson_r_decoder_vs_sigma1m_L0_40": {f"{k[0]} {k[1]}": v for k, v in rvals.items()},
                       "stored_profile_r": {k: c["profile"][k]["1m"]["pearson_r"] for k in c["profile"]
                                            if "L040" in k},
                       "gamma_range_ours_by_window_group": {f"{k[0]} {k[1]}": v for k, v in rng.items()}}


# --------------------------------------------------------------------------------------------- Figure 3
def fig3():
    rr = [r for r in load("gpu/runs.json")["runs"] if r["group"] == "saes_gpu_k"]
    ks = sorted({r["k"] for r in rr})
    fig = plt.figure(figsize=(W2, 3.9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.42)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])]
    axl = fig.add_subplot(gs[1, 1:])
    an = load("gpu/analysis.json")["B"]
    flagged = []
    for ax, w, t in zip(axs, WINDOWS, "abcd"):
        top = int(w.strip("()").split(",")[1])
        for i, cond in enumerate(["real", "gaussian"]):
            jit = 1.06 if i else 1 / 1.06
            mean = []
            for k in ks:
                sel = [r for r in rr if r["cond"] == cond and r["k"] == k]
                v = [r["gamma"][w] for r in sel]
                short = [16384 - r["n_dead"] < top for r in sel]
                for vv, sh in zip(v, short):
                    ax.plot([k * jit], [vv], "x" if sh else "o", ms=3.2 if sh else 2.6, color=COND[cond],
                            mfc="none", mew=0.8 if sh else 0.7)
                    if sh:
                        flagged.append((w, cond, k))
                mean.append(np.mean(v))
                assert abs(np.mean(v) - an[f"{cond}|{w}"][str(k)][0]) < 1e-9
            ax.plot(ks, mean, "-", color=COND[cond], lw=1.0)
        ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":")
        ax.set_yscale("log")
        yt = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
        lo, hi = ax.get_ylim()
        ax.set_yticks([y for y in yt if lo <= y <= hi])
        ax.set_yticklabels([f"{y:g}" for y in yt if lo <= y <= hi])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        log2_axis(ax, [1, 4, 16, 64])
        ax.set_xlim(0.75, 170)
        ax.set_ylabel(r"$\gamma$ (log scale)")
        ax.set_title(WIN_LABEL[w], fontsize=7, pad=2)
        ax.set_xlabel("k (active latents per token)")
        panel_tag(ax, t)
    NUMBERS["fig3_flagged_live_below_window"] = sorted(set(flagged))
    for cond in ["real", "gaussian"]:
        mean = []
        for k in ks:
            v = [16384 - r["n_dead"] for r in rr if r["cond"] == cond and r["k"] == k]
            axl.plot([k] * len(v), v, "o", ms=2.6, color=COND[cond], mfc="none", mew=0.7)
            mean.append(np.mean(v))
        axl.plot(ks, mean, "-", color=COND[cond], lw=1.0)
    axl.set_yscale("log")
    log2_axis(axl, ks)
    axl.set_xlim(0.75, 170)
    axl.axhline(16384, color=OI["grey"], lw=0.6, ls=":")
    axl.text(0.8, 16384 * 1.12, "width 16,384", fontsize=6.5, color="#555555")
    axl.set_ylim(60, 30000)
    axl.set_xlabel("k (active latents per token)")
    axl.set_ylabel("live latents (of 16,384)")
    panel_tag(axl, "e")
    handles = [Line2D([], [], color=COND[c_], marker="o", mfc="none", label=f"{COND_LABEL[c_]}, 2 seeds")
               for c_ in ["real", "gaussian"]]
    handles.append(Line2D([], [], color="#555555", marker="x", ls="", label="(a-d) live latents < window's top rank"))
    axl.legend(handles=handles, loc="lower right", fontsize=6.3, handlelength=1.5, borderaxespad=0.2)
    save(fig, "fig3_gamma_vs_k_L13")

    tab = {}
    for cond in ["real", "gaussian"]:
        for k in ks:
            tab[f"{cond} k={k}"] = {
                "gamma_primary": an[f"{cond}|{PRIMARY}"][str(k)][0],
                "live_mean": float(np.mean([16384 - r["n_dead"] for r in rr if r["cond"] == cond and r["k"] == k]))}
    NUMBERS["fig3"] = {"fits": {w: an[f"fit_real|{w}"] for w in WINDOWS}, "table": tab}


# --------------------------------------------------------------------------------------------- Figure 4
def fig4():
    rows, s1m, c = gemma_l12_rows()
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.95), sharey=True)
    xs = np.geomspace(10, 520, 50)
    for ax, w, t in zip(axes, WINDOWS, "abcd"):
        for width in WIDTHS:
            p = sorted([(r["l0"], r["gamma"][w]) for r in rows if r["width"] == width])
            x, y = zip(*p)
            ax.plot(x, y, "-", color=WCOL[width], marker=WMARK[width], lw=0.7, ms=2.8, label=width)
        law = c["l12_law"][f"1m|{w}"]
        ax.plot(xs, law["a"] + law["b_per_doubling"] * np.log2(xs), color=OI["black"], lw=0.8, ls="--")
        ax.text(0.97, 0.97, f"{law['b_per_doubling']:+.3f}/doubling\n$R^2$={law['r2']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        log2_axis(ax, [16, 64, 256])
        ax.set_xlim(10, 520)
        ax.set_title(WIN_LABEL[w], fontsize=7, pad=2)
        ax.set_xlabel(r"$L_0$ (active latents/token)")
        panel_tag(ax, t)
    axes[0].set_ylabel(r"$\gamma$ (1M-token $\Sigma$)")
    h, l_ = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=OI["black"], lw=0.8, ls="--"))
    l_.append(r"fit $a + b\log_2 L_0$ (all 28)")
    fig.tight_layout(w_pad=0.6)
    fig.legend(h, l_, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.2), handlelength=1.6, columnspacing=1.0,
               title="Gemma Scope width", title_fontsize=7)
    save(fig, "fig4_gamma_vs_l0_L12_1m")
    g = [r["gamma"][PRIMARY] for r in rows]
    NUMBERS["fig4"] = {"law_1m": {w: c["l12_law"][f"1m|{w}"] for w in WINDOWS},
                       "gamma_primary_range": (min(g), max(g), max(g) - min(g)),
                       "n": len(rows), "n_l0_from_name": sum(r["l0_source"] == "name" for r in rows)}


# --------------------------------------------------------------------------------------------- Figure 5
def fig5():
    mt = load("review/review2_checks.json")["matrix1m"]
    l0 = {r["sae_id"]: r["l0"] for r in load("test1/gamma_l0_L12.json")["rows"]}
    x = [l0[e["sae"]] for e in mt]
    conds = [("real", "Gemma Scope decoder", OI["blue"], "o"),
             ("exact_sigma_gamma", r"exact $\Sigma^\gamma$ cloud", OI["orange"], "s"),
             ("random_basis", "random-basis null", OI["grey"], "^")]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.62, 1.9))
    for key, lab, col, mk in conds:
        axes[0].plot(x, [e[key]["diag_share"] for e in mt], "-", marker=mk, color=col, label=lab)
        axes[1].plot(x, [e[key]["subspace_overlap"]["10"] for e in mt], "-", marker=mk, color=col, label=lab)
    axes[0].set_ylabel(r"diagonal share in $\Sigma$'s eigenbasis")
    axes[1].set_ylabel(r"top-10 subspace overlap")
    for ax, t in zip(axes, "ab"):
        log2_axis(ax, [16, 32, 64, 128, 256, 512])
        ax.set_xlim(16, 560)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(r"$L_0$ (active latents/token)")
        panel_tag(ax, t)
    fig.tight_layout(w_pad=1.2)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.56, 1.09), handlelength=1.8)
    save(fig, "fig5_matrix_test_1m")
    NUMBERS["fig5"] = {e["sae"]: {"l0": l0[e["sae"]], **{k: (round(e[k]["diag_share"], 3),
                                                            round(e[k]["subspace_overlap"]["10"], 3))
                                                        for k, *_ in conds}} for e in mt}


# --------------------------------------------------------------------------------------------- Figure 6
def fig6():
    rs = load("phase3/resynth_summary.json")["variants"]
    sp = load("phase3/resynth_split_summary.json")["variants"]
    ct = load("phase3/control_summary.json")
    # Prefer the split file (2 seeds for B); C and D exist only in resynth_summary (1 seed).
    src = {"R": sp, "A": sp, "B": sp, "C": rs, "D": rs, "E": sp, "F": sp}
    lab = {"R": "R real L12 activations", "A": "A faithful resynthesis", "B": "B independent firing",
           "C": "C permuted directions", "D": "D indep. firing + permuted",
           "E": "E Gemma marg., random dirs", "F": "F uniform marg., Gemma dirs"}
    col = {"R": OI["black"], "A": OI["grey"], "B": OI["sky"], "C": OI["yellow"], "D": OI["blue"],
           "E": OI["verm"], "F": OI["green"]}
    mk = {"R": "o", "A": "s", "B": "^", "C": "v", "D": "D", "E": "X", "F": "P"}
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.2), sharex=True)
    nums = {}
    for v in "RABCDEF":
        d = src[v][v]
        ss = d["sigma_slope"][PRIMARY]
        wd = d["windows"][PRIMARY]
        ks = sorted(int(k) for k in wd["gamma"])
        g = [wd["gamma"][str(k)] for k in ks]
        nseed = len(d["seeds"])
        for ax, scale in ((axes[0, 0], 1.0), (axes[1, 0], ss)):
            ax.plot(ks, np.array(g) * scale, "-", color=col[v], marker=mk[v], ms=3,
                    mec="#444444" if v == "C" else col[v], mew=0.5,
                    label=f"{lab[v]} ({nseed})")
            if nseed > 1:
                for s_, pg in wd["per_seed"].items():
                    ax.plot(ks, [pg[str(k)] * scale for k in ks], ".", color=col[v], ms=2.0, alpha=0.9)
        k8 = [k for k in ks if k >= 8]
        nums[v] = {"sigma_slope": ss, "per_doubling_gamma_k8": float(np.polyfit(np.log2(k8),
                                                                                 [wd["gamma"][str(k)] for k in k8], 1)[0]),
                   "per_doubling_raw_k8": float(np.polyfit(np.log2(k8),
                                                           [wd["gamma"][str(k)] * ss for k in k8], 1)[0]),
                   "seeds": nseed}
    ccol = {"D": OI["blue"], "G": OI["purple"], "F": OI["green"], "H": OI["orange"]}
    cmk = {"D": "D", "G": "o", "F": "P", "H": "s"}
    clab = {"D": "D Gemma marg., Gemma rows permuted", "G": "G Gemma marg., Gaussian-matched dirs",
            "F": "F uniform marg., Gemma rows", "H": "H uniform marg., Gaussian-matched dirs"}
    ks = ct["k"]
    for v in "DGFH":
        d = ct["variants"][v]
        fx = d["fixed"]
        ss = d["sigma_slope"]
        sd = np.array(fx["dec_sd"])
        axes[0, 1].errorbar(ks, fx["gamma"], yerr=sd / abs(ss), color=ccol[v], marker=cmk[v], ms=3, capsize=1.5,
                            elinewidth=0.7, label=clab[v])
        axes[1, 1].errorbar(ks, fx["dec"], yerr=sd, color=ccol[v], marker=cmk[v], ms=3, capsize=1.5, elinewidth=0.7,
                            label=clab[v])
        nums["control " + v] = {"sigma_slope": ss, "s_gamma": fx["s_gamma"], "s_raw": fx["s_raw"]}
    nums["control verdict"] = ct["verdict"]
    for ax in axes.ravel():
        log2_axis(ax, [4, 8, 16, 32, 64, 128])
        ax.set_xlim(3.4, 150)
    for ax in axes[1]:
        ax.set_xlabel("k (active latents per token)")
    axes[0, 0].set_ylabel(r"$\gamma$, ranks 11–150")
    axes[1, 0].set_ylabel("decoder eigen-slope, ranks 11–150")
    for ax, t in zip(axes.ravel(), "acbd"):
        panel_tag(ax, t)
    axes[0, 0].text(0.02, 1.07, "resynthesis variants", transform=axes[0, 0].transAxes, fontsize=7.5, style="italic")
    axes[0, 1].text(0.02, 1.07, "matched-covariance control", transform=axes[0, 1].transAxes, fontsize=7.5,
                    style="italic")
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], color="#555555", marker=".", ls="", ms=4))
    l_.append("single seeds")
    axes[1, 0].legend(h, l_, loc="upper left", bbox_to_anchor=(-0.12, -0.27), ncol=2, fontsize=6.3,
                      handlelength=1.4, columnspacing=0.6, title="variant (no. of seeds)", title_fontsize=6.3)
    h, l_ = axes[0, 1].get_legend_handles_labels()
    l_ = [x + "" for x in l_]
    axes[1, 1].legend(h, l_, loc="upper left", bbox_to_anchor=(0.08, -0.27), ncol=1, fontsize=6.3,
                      handlelength=1.4, title="seed 0; bars: sd, 4 draws of 8,192 live latents",
                      title_fontsize=6.3)
    save(fig, "fig6_resynthesis")
    NUMBERS["fig6"] = nums


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print(json.dumps(NUMBERS, indent=1, default=lambda o: o if not isinstance(o, np.floating) else float(o)))
