"""Phase 2 analysis: eigen slope of our TopK SAEs (real vs Gaussian vs scrambled) across widths.

Pre-registered windows for d=512 (0-based [start, end), kept < ~20% of d to stay off the bulk edge):
primary (5, 50); alternates (2, 25), (10, 100); and the paper's (10, 150) for reference only.
Slopes are computed with all latents and with dead latents (zero eval firings) excluded.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

PRIMARY = (5, 50)
WINDOWS = [(2, 25), PRIMARY, (10, 100), (10, 150)]
OUT = ROOT / "results" / "phase2"
PAT = re.compile(r"(?P<model>[^_]+)__(?P<hook>.+)__(?P<cond>real|gaussian|scrambled)__w(?P<width>\d+)__k(?P<k>\d+)__s(?P<seed>\d+)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted((ROOT / "cache" / "saes").glob("*/train.json")):
        m = PAT.fullmatch(run.parent.name)
        info = json.load(open(run))
        W = np.load(run.parent / "W_dec.npy")
        fires = np.load(run.parent / "fires.npy")
        alive = fires > 0
        r_all = eigen_slope(W, fit_range=PRIMARY)
        r_alive = eigen_slope(W, fit_range=PRIMARY, mask=alive) if alive.sum() > W.shape[1] else None
        freq = np.sort(fires[alive])[::-1] / info["n_eval_tokens"]
        rows.append({
            "run": run.parent.name, "cond": m["cond"], "width": int(m["width"]), "k": int(m["k"]),
            "seed": int(m["seed"]), "fvu_eval": info["fvu_eval"], "n_dead": int((~alive).sum()),
            "slope": r_all["slope"], "r2": r_all["r2"], "gaussian_slope": r_all["gaussian"]["slope"],
            "slope_alive": r_alive["slope"] if r_alive else None,
            "windows": [slope_from_eigenvalues(r_all["eigenvalues"], w) for w in WINDOWS],
            "windows_alive": [slope_from_eigenvalues(r_alive["eigenvalues"], w) for w in WINDOWS] if r_alive else None,
            "eigenvalues": r_all["eigenvalues"].tolist(),
            "freq_sorted_top": freq[:2000].tolist(),
            "git_hash": info["config"].get("git_hash"), "n_train_tokens": info["n_train_tokens"],
        })
        print(f"{m['cond']:>9} w{m['width']:>5} s{m['seed']} fvu={info['fvu_eval']:.3f} dead={rows[-1]['n_dead']:>5} "
              f"slope={r_all['slope']:+.3f} alive={rows[-1]['slope_alive'] if r_alive else float('nan'):+.3f} "
              + " ".join(f"{w['slope']:+.3f}" for w in rows[-1]["windows"]))
    json.dump({"config": {"primary_window": PRIMARY, "windows": WINDOWS, "center": True}, "runs": rows},
              open(OUT / "phase2_small.json", "w"))

    # Summary: mean ± sd over seeds, per (cond, width), for each window.
    agg = defaultdict(list)
    for r in rows:
        agg[(r["cond"], r["width"])].append(r)
    print("\ncond       width  n  " + "  ".join(f"{str(w):>16}" for w in WINDOWS) + "   dead  fvu")
    for (cond, width), rs in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        cols = []
        for j in range(len(WINDOWS)):
            s = np.array([r["windows"][j]["slope"] for r in rs])
            cols.append(f"{s.mean():+.3f}±{s.std(ddof=1) if len(s) > 1 else 0:.3f}".rjust(16))
        print(f"{cond:<9} {width:>6} {len(rs):>2}  " + "  ".join(cols)
              + f"  {np.mean([r['n_dead'] for r in rs]):>5.0f}  {np.mean([r['fvu_eval'] for r in rs]):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"real": "C0", "gaussian": "C3", "scrambled": "C2"}
    for cond in colors:
        ws = sorted({w for c, w in agg if c == cond})
        if not ws:
            continue
        mu = [np.mean([r["slope"] for r in agg[(cond, w)]]) for w in ws]
        sd = [np.std([r["slope"] for r in agg[(cond, w)]]) for w in ws]
        axes[0].errorbar(ws, mu, yerr=sd, marker="o", capsize=3, color=colors[cond], label=cond)
        r0 = [r for r in rows if r["cond"] == cond and r["seed"] == 0]
        for r in r0:
            e = np.array(r["eigenvalues"])
            axes[1].loglog(np.arange(1, len(e) + 1), e / e.sum(), color=colors[cond],
                           alpha=0.4 + 0.2 * WIDTHS_IDX.get(r["width"], 0), lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("SAE width")
    axes[0].set_ylabel(f"eigen slope, ranks {PRIMARY[0] + 1}–{PRIMARY[1]}")
    axes[0].legend()
    axes[0].set_title("Pythia-70m L3 resid, TopK k=32 (mean ± sd over seeds)")
    axes[1].axvspan(PRIMARY[0] + 1, PRIMARY[1], color="0.9", zorder=0)
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("eigenvalue / trace")
    axes[1].set_title("spectra, seed 0 (darker = wider)")
    fig.tight_layout()
    fig.savefig(OUT / "phase2_small.png", dpi=150)


WIDTHS_IDX = {1024: 0, 4096: 1, 16384: 2}

if __name__ == "__main__":
    main()
