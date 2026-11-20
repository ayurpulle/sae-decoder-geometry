"""Figure for the ladder (experiments/r2_job_ladder.py): real (R), equal-power null (Z) and power-matched null (N) gamma
in the two bulk windows for the four ladder datasets and their trained Gemma-2-2B counterparts, so the remainder N - R
is visible beside its trained reference.

Reads results/r2/ladder_summary.json (written by experiments/r2_ladder_summary.py; the values and verdicts are taken
from it unchanged), results/r2/ladder_extra.json (experiments/r2_ladder_extra.py) and results/r2/runs_ladder.jsonl (seed
dots of the ladder's real SAEs). Values above the axis (randL13, ranks 101-1000) are drawn as arrows at the top edge
with their values printed.

Writes paper/figures/fig15_ladder.{pdf,png}, paper/figures/tab_ladder_r2.tex (main-text table) and
paper/figures/tab_ladder_detail_r2.tex (appendix table).
Run from anywhere:  python paper/figures/make_fig_ladder.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import OI, RES, WIN_LABEL, panel_tag, plt, save  # noqa: E402

BULK = ["(30, 450)", "(100, 1000)"]
# pairs: trained counterpart, then the rung without trained weights
ORDER = [("embT", "Gemma"), ("embR", "random"), ("gemma_L0", "Gemma"), ("randL0", "random"),
         ("gemma_L13", "Gemma"), ("randL13", "random")]
GROUPS = ["embeddings", "block 0", "block 13"]
YMAX = 1.75


def main():
    S = json.load(open(RES / "r2" / "ladder_summary.json"))
    D = S["datasets"]
    lad = [json.loads(l) for l in open(RES / "r2" / "runs_ladder.jsonl") if l.strip()]
    seeds = {r: [x for x in lad if x["rung"] == r and x["step"] == "real" and x["k"] == 40]
             for r in ("embT", "embR", "randL0", "randL13")}
    xs = []
    x = 0.0
    for i in range(len(ORDER)):
        xs.append(x)
        x += 1.0
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
    for j, w in enumerate(BULK):
        ax = axes[j]
        for i, (key, _) in enumerate(ORDER):
            o = D[key]["windows"][w]
            xi = xs[i]
            R, Z, Nv = o["R"], o["Z"], o["N"]
            random_rung = i % 2 == 1
            # remainder N - R as a bar
            ax.add_patch(plt.Rectangle((xi - 0.17, R), 0.34, min(Nv, YMAX) - R, color=OI["verm"],
                                       alpha=0.22 if random_rung else 0.14, lw=0))
            ax.plot([xi], [R], "o", color=OI["blue"], ms=3.6, zorder=3, label="real" if i == 0 else None)
            for r in seeds.get(key, []):
                ax.plot([xi - 0.24], [r["gamma_fixed"][w]], ".", color=OI["blue"], ms=1.6, alpha=0.6)
            for v, mk, kw, lab in ((Z, "^", dict(mfc="white", mec=OI["green"], mew=0.9, ms=4.0), "equal-power null"),
                                   (Nv, "D", dict(color=OI["verm"], ms=3.4), "power-matched null")):
                if v <= YMAX:
                    ax.plot([xi], [v], mk, zorder=3, label=lab if i == 0 else None, **kw)
                else:
                    ax.annotate("", xy=(xi + (0.16 if mk == "^" else -0.16), YMAX), xytext=(xi + (0.16 if mk == "^" else -0.16), YMAX - 0.13),
                                arrowprops=dict(arrowstyle="-|>", lw=0.8, mutation_scale=6,
                                                color=OI["green"] if mk == "^" else OI["verm"]))
            off = [(n, v) for n, v in (("N", Nv), ("Z", Z)) if v > YMAX]
            if off:
                ax.text(xi + 0.32, YMAX - 0.02, "\n".join(f"{n} {v:.2f}" for n, v in off), ha="left", va="top",
                        fontsize=5.4, linespacing=1.1)
            ax.text(xi, R - 0.06, f"{Nv - R:.2f}", ha="center", va="top", fontsize=5.6, color=OI["verm"])
        for k in (1, 3):
            ax.axvline((xs[k] + xs[k + 1]) / 2, color=OI["grey"], lw=0.5, ls=":")
        ax.axvspan(xs[5] - 0.45, xs[5] + 0.45, color=OI["grey"], alpha=0.12, lw=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in ORDER], fontsize=5.8, rotation=35, ha="right", rotation_mode="anchor")
        for g, name in enumerate(GROUPS):
            ax.text((xs[2 * g] + xs[2 * g + 1]) / 2, -0.28, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6.4)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.7)
        ax.set_ylim(0.0, YMAX)
        ax.set_title(WIN_LABEL[w], fontsize=7.5)
        if j == 0:
            ax.set_ylabel(r"$\gamma$ at 8,192 live latents", fontsize=7)
        panel_tag(ax, "ab"[j], x=-0.06)
    from matplotlib.patches import Patch
    h, l_ = axes[0].get_legend_handles_labels()
    h.append(Patch(color=OI["verm"], alpha=0.2, lw=0))
    l_.append(r"remainder $N-R$ (printed)")
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=6.4, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 1), w_pad=0.8)
    save(fig, "fig15_ladder")
    print("wrote fig15_ladder")


NAME = {"embT": "embT (Gemma emb.)", "embR": "embR (random emb.)", "gemma_L0": "Gemma, block 0",
        "randL0": "randL0", "gemma_L13": "Gemma, block 13", "randL13": "randL13"}
PAIR = {"embR": "embR vs embT", "randL0": "randL0 vs gemma_L0", "randL13": "randL13 vs gemma_L13"}


def _gemma_power():
    """Top-1% latent-power shares at trained layers 0 and 13 from the first power-matched run (runs_pmatch.jsonl, the
    reference rows the ladder rule uses): real seed-0 SAE, the dictionary, and the power-matched null's SAE."""
    out = {}
    for l in open(RES / "r2" / "runs_pmatch.jsonl"):
        r = json.loads(l)
        for L in (0, 13):
            if r.get("step") == "PM" and r.get("setting_key") == f"PM|V1|L{L}|M8192_L16|ds0":
                out[f"gemma_L{L}"] = {"real_seed0_top1pct": r["power_check"]["real"]["share_top1pct"],
                                      "dictionary_top1pct": r["dictionary"]["power"]["share_top1pct"],
                                      "null_sae_top1pct": r["power_check"]["null"]["share_top1pct"]}
    return out


GEMMA_POWER = _gemma_power()


def tables():
    S = json.load(open(RES / "r2" / "ladder_summary.json"))
    X = json.load(open(RES / "r2" / "ladder_extra.json"))
    D = S["datasets"]
    lines = []
    for i, (key, _) in enumerate(ORDER):
        e = D[key]
        cells = [NAME[key], f"{e['fvu']['real_mean']:.3f}", f"{e['rho']['v0']:.2f}"]
        for w in BULK:
            o = e["windows"][w]
            cells += [f"{o[c]:.3f}" for c in ("R", "Z", "N", "G", "Rem")]
        if key in PAIR:
            c = S["comparisons"][PAIR[key]]
            r = c["rem_ratio"]
            cells.append(f"{c['verdict']}" + (r"$^\dagger$" if key == "randL13" else "")
                         + f" ({r[BULK[0]]:.2f}, {r[BULK[1]]:.2f})")
        else:
            cells.append("reference")
        lines.append(" & ".join(cells) + r" \\" + (r" \addlinespace" if i % 2 == 1 and i < 5 else ""))
    tex = (r"""\begin{tabular}{l rr rrrrr rrrrr l}
\toprule
 & & & \multicolumn{5}{c}{ranks 31--450} & \multicolumn{5}{c}{ranks 101--1000} & \\
\cmidrule(lr){4-8}\cmidrule(lr){9-13}
dataset & FVU & $\rho$ & $R$ & $Z$ & $N$ & $G$ & Rem & $R$ & $Z$ & $N$ & $G$ & Rem & verdict (Rem ratio) \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_ladder_r2.tex").write_text(
        "% generated by paper/figures/make_fig_ladder.py from results/r2/ladder_summary.json\n" + tex)

    lines = []
    for key, _ in ORDER:
        e, x = D[key], X["datasets"][key]
        sg = x["sigma"]
        cells = [NAME[key], f"{sg['top1_share']:.3f}", f"{sg['top10_share']:.3f}",
                 f"{sg['participation_ratio']:.0f}"]
        for w in BULK:
            cells.append(f"${sg['windows'][w]['slope']:.3f}$ ({sg['windows'][w]['r2']:.3f})")
        cells.append(f"{min(x['clouds'][r][w]['r2_min'] for r in 'RZN' for w in BULK[:1]):.3f}"
                     f"/{min(x['clouds'][r][w]['r2_min'] for r in 'RZN' for w in BULK[1:]):.3f}")
        lv = x["live"]
        cells.append(f"{min(lv['R']):,}/{min(lv['Z']):,}/{min(lv['N']):,}".replace(",", "{,}"))
        pw = x.get("power") or GEMMA_POWER.get(key)
        cells.append(f"{pw['real_seed0_top1pct']:.2f}/{pw['dictionary_top1pct']:.2f}/{pw['null_sae_top1pct']:.2f}"
                     if pw else "--")
        lines.append(" & ".join(cells) + r" \\")
    tex = (r"""\begin{tabular}{l rrr cc c c c}
\toprule
 & \multicolumn{3}{c}{$\Sigma$} & \multicolumn{2}{c}{$\Sigma$ slope ($R^2$)} & cloud fit & live latents & top-1\% power \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
dataset & top-1 & top-10 & PR & 31--450 & 101--1000 & min $R^2$ & $R$/$Z$/$N$ (min) & real/dict./null SAE \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    (Path(__file__).resolve().parent / "tab_ladder_detail_r2.tex").write_text(
        "% generated by paper/figures/make_fig_ladder.py from results/r2/ladder_summary.json and ladder_extra.json\n"
        + tex)
    print("wrote tab_ladder_r2.tex, tab_ladder_detail_r2.tex")


if __name__ == "__main__":
    main()
    tables()
