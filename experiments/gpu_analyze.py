"""Analyse the GPU queue (results/gpu/runs.json): A = real/gaussian/scrambled at L3/13/23, B = small-k sweep
at L13, C = de-massive at L13. gamma = decoder slope / covariance slope, per pre-registered window.

For scrambled runs gamma is reported two ways: against the real Sigma (as gpu_summarize stores it) and against
the scrambled data's own covariance, diag(Sigma) (scrambling destroys cross-coordinate correlation), which is
the like-for-like reference. Needs results/gpu/stats/L{3,13,23}.npz (per-layer mean/cov from the pod).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

RES = ROOT / "results" / "gpu"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
PRIMARY = str((10, 150))


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt(v):
    m, s, n = ms(v)
    return f"{m:.3f}±{s:.3f}" + ("" if n == 3 or n == 2 else f"(n{n})")


def main():
    runs = json.load(open(RES / "runs.json"))["runs"]
    diag_slope = {}
    for L in [3, 13, 23]:
        st = np.load(RES / "stats" / f"L{L}.npz")
        e = np.sort(np.diag(st["cov"]))[::-1]
        diag_slope[L] = {str(w): slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS}
    for r in runs:
        r["gamma_own"] = dict(r["gamma"])
        if r["cond"] == "scrambled":
            r["gamma_own"] = {w: r["decoder_slope"][w] / diag_slope[r["layer"]][w] for w in r["decoder_slope"]}
    out = {}

    # ---- A: estimator nulls at Gemma scale ----
    A = defaultdict(list)
    for r in runs:
        if r["group"] == "saes_gpu_p2":
            A[(r["width"], r["cond"], r["layer"])].append(r)
    print("A. gamma (mean ± sd over seeds), TopK k=40, 3 epochs over 9.5M tokens")
    out["A"] = {}
    for w in [str(x) for x in WINDOWS]:
        print(f"\n  window {w}")
        for width in [16384, 65536]:
            for cond in ["real", "gaussian", "scrambled"]:
                cells, rec = [], {}
                for L in [3, 13, 23]:
                    rs = A[(width, cond, L)]
                    key = "gamma_own" if cond == "scrambled" else "gamma"
                    v = [r[key][w] for r in rs]
                    rec[L] = ms(v)
                    cells.append(f"L{L}: {fmt(v)}")
                out["A"][f"{width}|{cond}|{w}"] = {L: list(x) for L, x in rec.items()}
                print(f"    w{width:<6} {cond:<9} " + "   ".join(cells))
    print("\n  eval FVU (16k / 65k), mean over seeds:")
    for cond in ["real", "gaussian", "scrambled"]:
        print(f"    {cond:<9} " + "   ".join(
            f"L{L}: {np.mean([r['fvu_eval'] for r in A[(16384, cond, L)]]):.3f} / {np.mean([r['fvu_eval'] for r in A[(65536, cond, L)]]):.3f}"
            for L in [3, 13, 23]))

    # ---- B: small-k sweep at L13 ----
    B = defaultdict(list)
    for r in runs:
        if r["group"] == "saes_gpu_k":
            B[(r["cond"], r["k"])].append(r)
    ks = sorted({k for _, k in B})
    print("\nB. small-k sweep, L13, width 16384: gamma (mean ± sd over 2 seeds)")
    out["B"] = {}
    for w in [str(x) for x in WINDOWS]:
        print(f"  window {w}")
        for cond in ["real", "gaussian"]:
            vals = [ms([r["gamma"][w] for r in B[(cond, k)]]) for k in ks]
            out["B"][f"{cond}|{w}"] = {k: list(v) for k, v in zip(ks, vals)}
            print(f"    {cond:<9} " + "  ".join(f"k{k}:{m:.3f}±{s:.3f}" for k, (m, s, _) in zip(ks, vals)))
        lr = linregress(np.log2(ks), [ms([r["gamma"][w] for r in B[("real", k)]])[0] for k in ks])
        out["B"][f"fit_real|{w}"] = {"intercept_k1": lr.intercept, "per_doubling": lr.slope, "r2": lr.rvalue ** 2}
        print(f"    real fit: gamma = {lr.intercept:.3f} {lr.slope:+.3f}*log2(k)   R2={lr.rvalue ** 2:.3f}")
    print("    measured L0 on eval (real): " + "  ".join(f"k{k}:{np.mean([r['l0_eval'] for r in B[('real', k)]]):.2f}" for k in ks))
    print("    dead latents (real): " + "  ".join(f"k{k}:{np.mean([r['n_dead'] for r in B[('real', k)]]):.0f}" for k in ks))

    # ---- C: de-massive at L13 ----
    C = defaultdict(list)
    for r in runs:
        if r["group"] == "saes_gpu_dm":
            C[r["drop_frac"]].append(r)
    base = A[(16384, "real", 13)]
    print("\nC. de-massive, L13, width 16384, k=40: gamma (mean ± sd over 3 seeds); baseline = A real L13")
    out["C"] = {}
    for w in [str(x) for x in WINDOWS]:
        row = {"none": ms([r["gamma"][w] for r in base])}
        for f in sorted(C):
            row[f] = ms([r["gamma"][w] for r in C[f]])
        out["C"][w] = {str(k): list(v) for k, v in row.items()}
        print(f"  window {w}: " + "   ".join(f"drop {k}: {m:.3f}±{s:.3f}" for k, (m, s, _) in row.items()))
    for f in sorted(C):
        print(f"  drop {f}: |x_1170| threshold {np.mean([r['drop_threshold'] for r in C[f]]):.1f}, "
              f"FVU {np.mean([r['fvu_eval'] for r in C[f]]):.3f} (baseline {np.mean([r['fvu_eval'] for r in base]):.3f})")
    json.dump(out, open(RES / "analysis.json", "w"), indent=1, default=float)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    col = {"real": "C0", "gaussian": "C3", "scrambled": "C2"}
    ax = axes[0]
    for cond in ["real", "gaussian", "scrambled"]:
        for width, ls in [(16384, "-"), (65536, "--")]:
            m = [out["A"][f"{width}|{cond}|{PRIMARY}"][L] for L in [3, 13, 23]]
            ax.errorbar([3, 13, 23], [x[0] for x in m], yerr=[x[1] for x in m], fmt="o" + ls, color=col[cond], capsize=3,
                        label=f"{cond} {'16k' if width == 16384 else '65k'}" + (" (vs own cov)" if cond == "scrambled" else ""))
    ax.set(xlabel="layer", ylabel="γ, ranks 11–150", title="A. Real vs nulls, equal budget (3 seeds)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for cond in ["real", "gaussian"]:
        m = [out["B"][f"{cond}|{PRIMARY}"][k] for k in ks]
        ax.errorbar(ks, [x[0] for x in m], yerr=[x[1] for x in m], fmt="o-", color=col[cond], capsize=3, label=cond)
    f = out["B"][f"fit_real|{PRIMARY}"]
    ax.plot(ks, [f["intercept_k1"] + f["per_doubling"] * np.log2(k) for k in ks], "k:", lw=1,
            label=f"real fit: {f['intercept_k1']:.2f} {f['per_doubling']:+.3f}·log₂k")
    ax.plot(ks, [1.245 - 0.114 * np.log2(k) for k in ks], color="0.6", lw=1, label="Gemma Scope L12 law (Test 1b)")
    ax.axhline(1, color="0.85", lw=0.8)
    ax.set(xscale="log", xlabel="k (= L0)", ylabel="γ, ranks 11–150", title="B. Small-k sweep, L13 (2 seeds)")
    ax.legend(fontsize=7)
    ax = axes[2]
    labels = ["none"] + [str(k) for k in sorted(C)]
    for j, w in enumerate([str(x) for x in WINDOWS]):
        m = [out["C"][w][lab] for lab in labels]
        ax.errorbar(np.arange(len(labels)) + 0.08 * j, [x[0] for x in m], yerr=[x[1] for x in m], fmt="o-", capsize=3,
                    label=f"ranks {eval(w)[0] + 1}–{eval(w)[1]}")
    ax.set_xticks(range(len(labels)), ["all tokens", "drop top 0.1%", "drop top 1%"])
    ax.set(ylabel="γ", title="C. Remove massive-activation tokens (dim 1170), L13")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "gpu_results.png", dpi=150)


if __name__ == "__main__":
    main()
