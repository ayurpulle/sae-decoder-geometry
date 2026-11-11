"""Figures for paper/thesisA.tex, redrawn from files under results/ only.

Run:  python paper/figures/make_figures_thesisA.py
Writes paper/figures/thesisA_fig1_setup.{pdf,png} and thesisA_fig2_null_layers.{pdf,png}, and prints, as JSON,
every number that thesisA.tex quotes from these figures and from the tables built on the same files.

Conventions (as in make_figures.py): a rank window "(a, b)" is a 0-based slice, so (10, 150) is ranks 11-150.
Top windows: ranks 4-50 and 11-150. Bulk windows: ranks 31-450 and 101-1000. Eigen-slope = least-squares slope of
log10 eigenvalue on log10 rank over the window. Gemma Scope Sigma slopes use the 1M-token Sigma; our own SAEs use
the Sigma of their own 10M training tokens (results/gpu/sigma.json).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
W2 = 6.75
WINDOWS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WLAB = {"(3, 50)": "ranks 4–50", "(10, 150)": "ranks 11–150", "(30, 450)": "ranks 31–450",
        "(100, 1000)": "ranks 101–1000"}
PRIMARY = "(10, 150)"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
COND = {"real": OI["blue"], "gaussian": OI["verm"], "scrambled": OI["green"]}
CLAB = {"real": "real activations", "gaussian": r"Gaussian $N(\mu,\Sigma)$", "scrambled": "scrambled"}
WIDTHS = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
WCOL = dict(zip(WIDTHS, plt.get_cmap("viridis")(np.linspace(0.0, 0.88, len(WIDTHS)))))
WMARK = dict(zip(WIDTHS, ["o", "s", "^", "D", "v", "P", "X"]))
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
})
NUM = {}


def load(rel):
    with open(RES / rel) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tag(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(a), float(b), float(r2)


# ---------------------------------------------------------------- Figure 1: slope vs L0 (L12) and matched-L0 profile
def fig1():
    s1m = {w: np.array(load("test1/sigma_1m_compare.json")["sigma_slope"]["1m"][w]) for w in WINDOWS}
    g = load("test1/gamma_l0_L12.json")
    pts = []
    for r in g["rows"]:
        nom = r["sae_id"].split("average_l0_")[1]
        eig = np.load(RES / "phase1" / f"eig_layer_12__width_{r['width']}__average_l0_{nom}.npy")
        pts.append((r["width"], r["l0"], {w: slope_from_eigenvalues(eig, tuple(int(v) for v in w.strip("()").split(",")))["slope"]
                                          for w in WINDOWS}))
    m = load("phase1/slope_vs_layer_matched_l0.json")

    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.25), gridspec_kw={"width_ratios": [1, 1.25]})
    for width in WIDTHS:
        p = sorted((l0, sl[PRIMARY]) for wd, l0, sl in pts if wd == width)
        if p:
            x, y = zip(*p)
            a.plot(x, y, "-", color=WCOL[width], marker=WMARK[width], lw=0.8, ms=3.2, label=f"{width} ({len(p)})")
    a.axhline(s1m[PRIMARY][12], color=OI["grey"], ls="--", lw=0.8)
    a.text(11, s1m[PRIMARY][12] + 0.02, r"$\Sigma$ slope, layer 12", fontsize=6.5, color="#555555")
    li = [(l0, sl[PRIMARY]) for wd, l0, sl in pts if wd == "65k" and 60 < l0 < 90][0]
    a.plot([li[0]], [li[1]], "o", ms=8, mfc="none", mec="k", mew=0.8)
    a.annotate("Li et al.'s SAE", xy=li, xytext=(17, -0.33), fontsize=6.5,
               arrowprops=dict(arrowstyle="-", lw=0.5))
    a.set_xscale("log", base=2)
    a.set_xticks([16, 32, 64, 128, 256]); a.set_xticklabels(["16", "32", "64", "128", "256"]); a.minorticks_off()
    a.set_xlim(10, 520)
    a.set_xlabel(r"$L_0$ (mean active latents per token)")
    a.set_ylabel("decoder eigen-slope, ranks 11–150")
    a.legend(title="width (no. of SAEs)", title_fontsize=6.5, loc="lower right", ncol=2, handlelength=1.4,
             bbox_to_anchor=(1.0, 0.06),
             columnspacing=0.8)
    tag(a, "a")

    cols = {20: OI["sky"], 40: OI["blue"], 70: OI["purple"], 140: OI["orange"]}
    amp = {}
    for l0 in (20, 40, 70, 140):
        d = m[f"16k_ranks11-150_L0_{l0}"]
        ls_ = sorted(int(k) for k in d)
        v = [d[str(k)] for k in ls_]
        full = np.full(26, np.nan)
        full[ls_] = v  # NaN breaks the line where no SAE brackets this L0
        b.plot(np.arange(26), full, "-o", color=cols[l0], ms=2.2, lw=0.9, label=fr"$L_0$ = {l0}")
        if "0" in d and "23" in d:
            amp[l0] = (float(d["0"] - min(v)), float(d["23"] - min(v)), int(ls_[int(np.argmin(v))]))
    b.set_xlabel("layer")
    b.set_ylabel("decoder eigen-slope, ranks 11–150")
    b.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    b.legend(title="Gemma Scope 16k, interpolated to", title_fontsize=6.5, loc="upper left", ncol=4,
             bbox_to_anchor=(0.0, 1.17), columnspacing=0.8, handlelength=1.4)
    tag(b, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "thesisA_fig1_setup")

    x = np.log2([p[1] for p in pts])
    out = {"n": len(pts), "fit_primary": fit(x, [p[2][PRIMARY] for p in pts])}
    out["lowest_l0"] = min(pts, key=lambda p: p[1])[:2] + (min(pts, key=lambda p: p[1])[2][PRIMARY],)
    out["highest_l0"] = max(pts, key=lambda p: p[1])[:2] + (max(pts, key=lambda p: p[1])[2][PRIMARY],)
    out["li_sae"] = li
    out["width_spread_matched_l0"] = []
    for lo, hi in [(19, 24), (36, 44), (64, 86)]:
        v = [p[2][PRIMARY] for p in pts if lo <= p[1] <= hi]
        out["width_spread_matched_l0"].append((lo, hi, len(v), max(v) - min(v)))
    out["peak_amplitude_vs_layer0_layer23_argmin"] = amp
    out["sigma1m_L12"] = {w: float(s1m[w][12]) for w in WINDOWS}
    NUM["fig1"] = out


# ---------------------------------------------------------------- Figure 2 / Table 1: real vs nulls at layers 3, 13, 23
def p2table():
    rr = [r for r in load("gpu/runs.json")["runs"] if r["group"] == "saes_gpu_p2"]
    t = {}
    for cond in ("real", "gaussian", "scrambled"):
        for wd in (16384, 65536):
            for w in WINDOWS:
                for lay in (3, 13, 23):
                    v = [r["decoder_slope"][w] for r in rr if r["cond"] == cond and r["width"] == wd and r["layer"] == lay]
                    assert len(v) == 3
                    t[(cond, wd, w, lay)] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    return t


def fig2():
    t = p2table()
    sig = load("gpu/sigma.json")
    lays = [3, 13, 23]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.2))
    for ax, w, s in zip(axes, WINDOWS, "abcd"):
        ax.plot(lays, [sig[f"blocks.{l}.hook_resid_post"]["slopes"][w] for l in lays], "--*", color="k", lw=0.7,
                ms=6, mfc="white", mew=0.7)
        for i, cond in enumerate(("real", "gaussian", "scrambled")):
            for wd, mfc, ls in ((16384, None, "-"), (65536, "white", ":")):
                off = (i - 1) * 0.9 + (0.35 if wd == 65536 else -0.35)
                mu = [t[(cond, wd, w, l)][0] for l in lays]
                sd = [t[(cond, wd, w, l)][1] for l in lays]
                ax.errorbar(np.array(lays) + off, mu, yerr=sd, fmt=ls, marker="o", ms=3, lw=0.8, color=COND[cond],
                            mfc=mfc or COND[cond], mew=0.8, elinewidth=0.7, capsize=1.2)
        ax.set_xticks(lays)
        ax.set_xlim(0, 26)
        ax.set_xlabel("layer")
        ax.set_title(WLAB[w], fontsize=7, pad=2)
        tag(ax, s)
    axes[0].set_ylabel("eigen-slope")
    axes[0].text(0.02, 1.16, "top windows", transform=axes[0].transAxes, fontsize=7.5, style="italic")
    axes[2].text(0.02, 1.16, "bulk windows", transform=axes[2].transAxes, fontsize=7.5, style="italic")
    h = [Line2D([], [], color=COND[c], marker="o", label=f"SAE on {CLAB[c]}") for c in ("real", "gaussian", "scrambled")]
    h += [Line2D([], [], color="k", ls="--", marker="*", mfc="white", ms=6, lw=0.7, label=r"$\Sigma$ (10M training tokens)"),
          Line2D([], [], color="#555555", marker="o", ls="-", label="width 16k (filled)"),
          Line2D([], [], color="#555555", marker="o", mfc="white", ls=":", label="width 65k (open)")]
    fig.tight_layout(w_pad=0.8)
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12), handlelength=1.8, columnspacing=1.0)
    save(fig, "thesisA_fig2_null_layers")

    out = {"table_raw_mean_sd": {f"{c}|{wd}|{w}|{l}": t[(c, wd, w, l)] for (c, wd, w, l) in t},
           "sigma10M": {l: sig[f"blocks.{l}.hook_resid_post"]["slopes"] for l in lays},
           "max_sd_all": max(v[1] for v in t.values())}
    top = {c: [t[(c, wd, w, l)][0] for wd in (16384, 65536) for w in WINDOWS[:2] for l in lays]
           for c in ("real", "gaussian")}
    out["top_window_range_and_mean"] = {c: (min(v), max(v), float(np.mean(v))) for c, v in top.items()}
    gap = {}
    for wd in (16384, 65536):
        for w in WINDOWS:
            d = [t[("real", wd, w, l)][0] - t[("gaussian", wd, w, l)][0] for l in lays]
            gap[f"{wd}|{w}"] = (d, lays[int(np.argmin(d))])
    out["real_minus_gaussian"] = gap
    ratio = {}
    for c in ("gaussian", "scrambled"):
        r_ = [t[(c, wd, w, l)][0] / t[("real", wd, w, l)][0] for wd in (16384, 65536) for w in WINDOWS[2:] for l in lays]
        ratio[c] = (min(r_), max(r_), sorted(r_)[:2])
    out["bulk_null_over_real_ratio"] = ratio
    mono = {}
    for wd in (16384, 65536):
        for w in WINDOWS[:2]:
            g_ = [t[("gaussian", wd, w, l)][0] for l in lays]
            r_ = [t[("real", wd, w, l)][0] for l in lays]
            mono[f"{wd}|{w}"] = {"gauss_monotone_steepening": bool(g_[0] > g_[1] > g_[2]),
                                 "real_steepest_at_13": bool(r_[1] < r_[0] and r_[1] < r_[2])}
    out["profile_shape"] = mono
    # changes between layers, 16k primary: decoder vs Sigma
    s = {l: sig[f"blocks.{l}.hook_resid_post"]["slopes"][PRIMARY] for l in lays}
    ch = {}
    for c in ("real", "gaussian"):
        for wd in (16384, 65536):
            v = {l: t[(c, wd, PRIMARY, l)][0] for l in lays}
            ch[f"{c}|{wd}"] = (v[13] - v[3], v[23] - v[13])
    ch["sigma"] = (s[13] - s[3], s[23] - s[13])
    out["primary_changes_3to13_13to23"] = ch
    gam = {}
    for c in ("real", "gaussian"):
        for grp, ws in (("top", WINDOWS[:2]), ("bulk", WINDOWS[2:])):
            v = [t[(c, wd, w, l)][0] / sig[f"blocks.{l}.hook_resid_post"]["slopes"][w]
                 for wd in (16384, 65536) for w in ws for l in lays]
            gam[f"{c}|{grp}"] = (min(v), max(v))
    out["gamma_ranges"] = gam
    NUM["fig2"] = out


# ---------------------------------------------------------------- numbers for the k sweep (Figure 3 is fig3_gamma_vs_k_L13)
def ksweep():
    an = load("gpu/analysis.json")["B"]
    rr = [r for r in load("gpu/runs.json")["runs"] if r["group"] == "saes_gpu_k"]
    ks = [1, 2, 4, 8, 16, 40, 128]
    out = {}
    for w in WINDOWS:
        for cond in ("real", "gaussian"):
            y = [an[f"{cond}|{w}"][str(k)][0] for k in ks]
            out[f"{cond}|{w}|means"] = y
            for lo in (8, 16):
                idx = [i for i, k in enumerate(ks) if k >= lo]
                out[f"{cond}|{w}|k>={lo}"] = fit(np.log2([ks[i] for i in idx]), [y[i] for i in idx])
        out[f"fit_real_all|{w}"] = an[f"fit_real|{w}"]
    for cond in ("real", "gaussian"):
        out[f"live|{cond}"] = {k: float(np.mean([16384 - r["n_dead"] for r in rr if r["cond"] == cond and r["k"] == k]))
                               for k in ks}
        out[f"perseed_k16|{cond}"] = [fit(np.log2([16, 40, 128]), [[r["gamma"][PRIMARY] for r in rr if r["cond"] == cond
                                                                     and r["k"] == k and r["seed"] == s][0]
                                                                    for k in (16, 40, 128)])[1]
                                      for s in sorted({r["seed"] for r in rr})]
    NUM["ksweep"] = out


def control():
    ct = load("phase3/control_summary.json")
    k = np.log2(ct["k"])
    out = {"k": ct["k"]}
    for v in "DGFH":
        d = ct["variants"][v]
        out[v] = {"sigma_slope": d["sigma_slope"], "gamma_fixed": d["fixed"]["gamma"],
                  "k8_128_gamma": fit(k, d["fixed"]["gamma"])[1], "k16_128_gamma": fit(k[1:], d["fixed"]["gamma"][1:])[1],
                  "k8_128_raw": fit(k, d["fixed"]["dec"])[1], "k16_128_raw": fit(k[1:], d["fixed"]["dec"][1:])[1]}
    for u in ("gamma", "raw"):
        for rng in ("k8_128", "k16_128"):
            out[f"G/D retained {u} {rng}"] = out["G"][f"{rng}_{u}"] / out["D"][f"{rng}_{u}"]
    NUM["control"] = out


if __name__ == "__main__":
    fig1(); fig2(); ksweep(); control()
    print(json.dumps(NUM, indent=1, default=float))
