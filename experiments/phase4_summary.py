"""Phase 4 summary figure and table from results/phase4/{l12,layers,layers_null}.json -> results/phase4/summary.{json,png}.

Panel 1: per layer (16k, L0 ~ 70), eigen-slope in ranks 11-150 and 31-450 next to the excess cluster structure
over a covariance-matched Gaussian cloud (between-cluster trace share and median cosine to centroid, K = 1024).
Panel 2: layer 12, fraction of each window's top eigenvalues explained by between-cluster structure: Gemma,
covariance-matched cloud, isotropic cloud, our TopK SAEs on real activations and on their Gaussian twin.
Panel 3: allocation-law slope (log mean latent frequency on log cluster size) vs latents per cluster, per width,
with the frequency-permutation null (mean ± 2 sd).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "results" / "phase4"
W = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WL = ["4–50", "11–150", "31–450", "101–1000"]


def main():
    l12 = json.load(open(P4 / "l12.json"))
    g = json.load(open(P4 / "layers.json"))["gemma"]
    m = json.load(open(P4 / "layers_null.json"))["matched"]
    out = {"layers": {}, "spearman_vs_slope": {}}
    for K in (256, 1024):
        ex = {"share": [], "cos": [], "pr16": []}
        sl = {w: [] for w in W}
        msh = []
        for L in range(26):
            a, b = g[f"{L}|{K}"], m[f"{L}|{K}"]
            ex["share"].append(a["trace_share_between"] - b["trace_share_between"])
            ex["cos"].append(a["median_cos_to_centroid"] - b["median_cos_to_centroid"])
            ex["pr16"].append(a["median_pr16"] - b["median_pr16"])
            msh.append(b["trace_share_between"])
            for w in W:
                sl[w].append(a["windows"][w]["slope_C"])
        out["layers"][K] = {**{f"excess_{k}": v for k, v in ex.items()}, "matched_share": msh,
                            **{f"slope_{w}": v for w, v in sl.items()}}
        out["spearman_vs_slope"][K] = {
            f"{k}|{w}": float(spearmanr(sl[w], v)[0]) for k, v in {**{f"excess_{a}": b for a, b in ex.items()},
                                                                    "matched_share": msh}.items() for w in W}

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    L = np.arange(26)
    d = out["layers"][1024]
    a = ax[0]
    a.plot(L, -np.array(d["slope_(10, 150)"]), "k-o", ms=3, label="−slope, ranks 11–150")
    a.plot(L, -np.array(d["slope_(30, 450)"]), "-o", color="0.6", ms=3, label="−slope, ranks 31–450")
    a.set_xlabel("layer (16k, L0 ≈ 70)")
    a.set_ylabel("−eigen-slope")
    b = a.twinx()
    b.plot(L, d["excess_share"], "-s", color="tab:red", ms=3, label="excess between-share")
    b.plot(L, d["excess_cos"], "-^", color="tab:orange", ms=3, label="excess cos to centroid")
    b.set_ylabel("real − covariance-matched (K = 1024)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower center")
    rs = out["spearman_vs_slope"][1024]
    a.set_title(f"Cluster structure beyond anisotropy vs layer\nSpearman(slope 11–150, excess share) = "
                f"{rs['excess_share|(10, 150)']:+.2f}; (31–450) = {rs['excess_share|(30, 450)']:+.2f}", fontsize=9)

    a = ax[1]
    B = l12["B"]
    series = [("Gemma 16k L0 82", B["gemma_16k_L0_82|kmeans|1024"], "k"),
              ("covariance-matched", m["12|1024"], "tab:red"),
              ("isotropic", B["isotropic_16k|kmeans|1024"], "0.6"),
              ("TopK w4096 real (K=256)", B["topk_w4096_real|kmeans|256"], "tab:blue"),
              ("TopK w4096 Gaussian (K=256)", B["topk_w4096_gaussian|kmeans|256"], "tab:cyan")]
    x = np.arange(4)
    for i, (lab, r, c) in enumerate(series):
        a.bar(x + (i - 2) * 0.16, [r["windows"][w]["explained_by_between"] for w in W], 0.16, color=c, label=lab)
    a.set_xticks(x, [f"ranks {w}" for w in WL])
    a.set_ylabel("mean v_j·B·v_j / λ_j")
    a.set_title("Layer 12: share of top eigenvalues carried by between-cluster structure\n"
                "(16 latents per cluster, spherical k-means)", fontsize=9)
    a.legend(fontsize=7)

    a = ax[2]
    C = l12["C"]
    for width, c in zip(["16k", "32k", "65k", "131k"], plt.cm.viridis([0.1, 0.4, 0.65, 0.9])):
        rows = sorted((v for k, v in C.items() if k.startswith(width + "|")), key=lambda v: -v["K"])
        per = [v["n_alive"] / v["K"] for v in rows]
        a.plot(per, [v["slope_logf_on_logn"] for v in rows], "-o", color=c, label=width)
        a.fill_between(per, [v["null_mean"] - 2 * v["null_sd"] for v in rows],
                       [v["null_mean"] + 2 * v["null_sd"] for v in rows], color=c, alpha=0.15)
    a.axhline(0, color="k", lw=0.6)
    a.set_xscale("log", base=2)
    a.set_xlabel("latents per cluster")
    a.set_ylabel("d log(mean latent freq) / d log n_c")
    a.set_title("Allocation law per cluster (layer 12, L0 ≈ 70)\ntheory: +b > 0; band = permutation null ± 2 sd",
                fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(P4 / "summary.png", dpi=150)
    json.dump(out, open(P4 / "summary.json", "w"), indent=1, default=float)
    for K in (256, 1024):
        print(K, {k: round(v, 2) for k, v in out["spearman_vs_slope"][K].items()})


if __name__ == "__main__":
    main()
