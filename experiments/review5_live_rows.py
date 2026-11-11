"""Check:
which rows enter each decoder spectrum, and how much the choice moves the slope.

Code audit (read before running; recorded in the output as CONVENTION):
  - Gemma Scope spectra (phase1b_layers.py, phase1_width_ladder.py -> results/phase1/layers.json,
    width_ladder_L12.json, eig_*.npy; every 288-SAE number, layer profile, Li contrast, gamma of Gemma Scope):
    ALL rows (eigen_slope without mask; files record dead_latents_excluded = 0).
  - Our TopK SAEs (gpu_summarize.py -> results/gpu/runs.json; control_analyze.py; test1c_null_saes.py;
    r2_common.py; phase3_analyze.py; review_dsweep.py; gpu_peek.py): LIVE rows only (mask = fires > 0, fired at
    least once on the held-out eval split).
  - phase2_analyze.py (Pythia-70m L3): computes both (slope = all rows, slope_alive = live rows).
  - phase1_freqs.py (22 layer-12 Gemma Scope SAEs, 1M tokens): stores a live-row slope as a side result.

Analysis choices (fixed before looking at results):
  Slopes with all rows vs live rows (centred covariance, rows as stored, geometry.eigen_slope), four windows
  (3,50), (10,150), (30,450), (100,1000) where the window fits inside min(n_rows, d); difference = live - all.
  (a) The 22 layer-12 Gemma Scope SAEs with firing counts on 1M tokens (results/phase1/fires_layer_12_*.npy):
      live = fired at least once on those tokens.
  (b) Every local TopK SAE on Gemma-2-2B or Pythia-70m activations with fires.npy: cache/saes (27 Pythia L3),
      cache/saes_gemma_null (9), cache/saes_dsweep (8, Gemma L12 PCA-projected, d = 256-2304).
  (c) The GPU runs (decoders not on this machine): dead counts per cell from results/gpu/runs.json, to say where
      the convention can matter; no slope difference can be computed for them.
  Not computed: the isotropic-baseline slope (eigen_slope's matched Gaussian) is ignored.
Writes results/review/live_rows.json.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import _eigvals, slope_from_eigenvalues  # noqa: E402

MAIN = Path("/Users/ayurpulle/Desktop/SAE")  # read-only main cache (a worktree cache/ may hold unrelated runs)
CACHE = MAIN / "cache"
OUT = ROOT / "results" / "review" / "live_rows.json"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
CONVENTION = {
    "gemma_scope": "all rows (phase1b_layers.py, phase1_width_ladder.py; dead_latents_excluded = 0)",
    "our_topk": "live rows only, live = fired >= 1 time on held-out eval tokens (gpu_summarize.py, control_analyze.py, "
                "test1c_null_saes.py, r2_common.py, phase3_analyze.py, review_dsweep.py)",
    "pythia_phase2": "both stored (phase2_analyze.py: slope = all rows, slope_alive = live rows)",
}


def slopes(eig, n_rows):
    d = len(eig)
    out = {}
    for w in WINDOWS:
        out[str(w)] = slope_from_eigenvalues(eig, w)["slope"] if w[1] <= min(n_rows - 1, d) else None
    return out


def compare(W, live):
    e_all = _eigvals(W, center=True)[0]
    e_live = _eigvals(W, center=True, mask=live)[0]
    s_all, s_live = slopes(e_all, W.shape[0]), slopes(e_live, int(live.sum()))
    diff = {w: (s_live[w] - s_all[w] if s_live[w] is not None and s_all[w] is not None else None) for w in s_all}
    return {"n_rows": int(W.shape[0]), "n_live": int(live.sum()), "n_dead": int((~live).sum()),
            "slope_all": s_all, "slope_live": s_live, "live_minus_all": diff}


def main():
    t0 = time.time()
    out = {"notes": __doc__.split("Writes")[0].strip(), "convention": CONVENTION, "gemma_scope_L12": {}, "topk_local": {}}

    print("(a) Gemma Scope layer 12, 22 SAEs with firing counts on 1M tokens")
    pat = re.compile(r"fires_layer_12__width_(\w+)__average_l0_(\d+)\.npy")
    for p in sorted((ROOT / "results" / "phase1").iterdir()):
        m = pat.fullmatch(p.name)
        if not m:
            continue
        width, l0 = m.groups()
        fires = np.load(p)
        W = np.load(CACHE / "gemmascope_wdec" / "layer_12" / f"width_{width}" / f"average_l0_{l0}" / "W_dec.npy", mmap_mode="r")
        r = compare(W, fires > 0)
        # all-row spectrum must match the stored phase1 eigenvalues
        e_ref = np.load(ROOT / "results" / "phase1" / f"eig_layer_12__width_{width}__average_l0_{l0}.npy")
        r["check_all_vs_phase1_11_150"] = slope_from_eigenvalues(e_ref, (10, 150))["slope"]
        out["gemma_scope_L12"][f"width_{width}/average_l0_{l0}"] = r
        print(f"  {width:>5} L0 {l0:>4} dead {r['n_dead']:>6}/{r['n_rows']:<7} " +
              "  ".join(f"{w}: {r['live_minus_all'][w]:+.4f}" for w in r['live_minus_all'] if r['live_minus_all'][w] is not None)
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    print("(b) local TopK SAEs")
    for grp in ("saes", "saes_gemma_null", "saes_dsweep"):
        for d in sorted((CACHE / grp).iterdir()):
            if not ((d / "fires.npy").exists() and (d / "W_dec.npy").exists()):
                continue
            info = json.load(open(d / "train.json"))
            c = info["config"]
            fires = np.load(d / "fires.npy")
            r = compare(np.load(d / "W_dec.npy"), fires > 0)
            r.update({"group": grp, "model": c["model"], "cond": c["cond"], "width": c["width"], "k": c["k"],
                      "seed": c["seed"], "fvu_eval": info["fvu_eval"]})
            out["topk_local"][f"{grp}/{d.name}"] = r
            if r["n_dead"]:
                print(f"  {grp}/{d.name}: dead {r['n_dead']}/{r['n_rows']} " +
                      "  ".join(f"{w}: {v:+.4f}" for w, v in r['live_minus_all'].items() if v is not None), flush=True)

    def maxabs(rows):
        res = {}
        for w in map(str, WINDOWS):
            v = [abs(r["live_minus_all"][w]) for r in rows if r["live_minus_all"][w] is not None]
            res[w] = float(max(v)) if v else None
        return res

    gs = list(out["gemma_scope_L12"].values())
    tk = list(out["topk_local"].values())
    out["summary"] = {
        "gemma_scope_L12": {"n": len(gs), "dead_range": [min(r["n_dead"] for r in gs), max(r["n_dead"] for r in gs)],
                            "dead_frac_max": max(r["n_dead"] / r["n_rows"] for r in gs), "max_abs_live_minus_all": maxabs(gs)},
        "topk_local_all": {"n": len(tk), "n_with_dead": sum(r["n_dead"] > 0 for r in tk), "max_abs_live_minus_all": maxabs(tk)},
        "topk_local_with_dead": {grp: {"n": len(v), "dead": [r["n_dead"] for r in v], "n_rows": [r["n_rows"] for r in v],
                                       "max_abs_live_minus_all": maxabs(v)}
                                 for grp in ("saes", "saes_dsweep") if (v := [r for r in tk if r["group"] == grp and r["n_dead"]])}}

    print("(c) GPU runs: dead latents per cell (decoders not local)")
    runs = json.load(open(ROOT / "results" / "gpu" / "runs.json"))["runs"]
    cells = defaultdict(list)
    for x in runs:
        cells[f"{x['group']}|{x['cond']}|L{x['layer']}|w{x['width']}|k{x['k']}"].append(x["n_dead"])
    out["gpu_dead_counts"] = {k: {"n_dead": v, "dead_frac_max": max(v) / int(k.split('|w')[1].split('|')[0])}
                              for k, v in sorted(cells.items())}
    for k, v in sorted(cells.items()):
        if max(v) > 0:
            print(f"  {k:<45} {v}")
    print("summary", json.dumps(out["summary"], indent=1))
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
