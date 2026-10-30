"""Phase 1b: eigen slope vs layer at matched L0 (Gemma Scope res, 16k and 65k, all layers; paper layers first).

The paper compared layers using one SAE per layer with different L0s; Phase 1a showed slope moves
~0.10 per doubling of L0, so here every available L0 is measured and slopes are interpolated to common L0s
in experiments/phase1b_plot.py. Reuses fetch_wdec/WINDOWS from phase1_width_ladder.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402
from phase1_width_ladder import WINDOWS, fetch_wdec, git_hash  # noqa: E402

OUT = ROOT / "results" / "phase1"
PAPER_LAYERS = [0, 4, 8, 12, 16, 20, 24]


def plan():
    avail = json.load(open(ROOT / "results" / "available_saes.json"))["variants"]
    by = {(v["layer"], v["width"]): sorted({v["l0"] for v in avail if (v["layer"], v["width"]) == (L, w)})
          for v in avail for L, w in [(v["layer"], v["width"])] if L != "embedding"}
    jobs = [(L, "16k", l0) for L in range(26) for l0 in by[(L, "16k")]]
    jobs += [(L, "65k", l0) for L in PAPER_LAYERS for l0 in by[(L, "65k")]]
    jobs += [(L, "65k", l0) for L in range(26) if L not in PAPER_LAYERS for l0 in by[(L, "65k")]]
    return jobs


def main():
    out_json = OUT / "layers.json"
    done = json.load(open(out_json))["results"] if out_json.exists() else {}
    ladder = json.load(open(OUT / "width_ladder_L12.json"))["results"]
    for layer, width, l0 in plan():
        sae_id = f"layer_{layer}/width_{width}/average_l0_{l0}"
        if sae_id in done:
            continue
        if sae_id in ladder:  # layer 12 already measured in Phase 1a with the same function and windows
            done[sae_id] = ladder[sae_id]
            continue
        t0 = time.time()
        W = np.load(fetch_wdec(sae_id), mmap_mode="r")
        t1 = time.time()
        r = eigen_slope(W)
        done[sae_id] = {
            "layer": layer, "width": width, "n_latents": r["n_latents"], "l0": l0,
            "slope": r["slope"], "r2": r["r2"], "gaussian_slope": r["gaussian"]["slope"],
            "row_norm": r["row_norm"],
            "windows": [slope_from_eigenvalues(r["eigenvalues"], w) for w in WINDOWS],
            "top_eig": float(r["eigenvalues"][0]),
        }
        np.save(OUT / f"eig_{sae_id.replace('/', '__')}.npy", r["eigenvalues"])
        ws = " ".join(f"{w['slope']:+.3f}" for w in done[sae_id]["windows"])
        print(f"{sae_id}: slope={r['slope']:+.4f} windows[{ws}] fetch={t1 - t0:.0f}s", flush=True)
        config = {"model": "gemma-2-2b", "release": "gemma-scope-2b-pt-res", "windows": [list(w) for w in WINDOWS],
                  "center": True, "normalize": False, "dead_latents_excluded": 0, "git_hash": git_hash()}
        json.dump({"config": config, "results": done}, open(out_json, "w"), indent=1)
        del W


if __name__ == "__main__":
    main()
