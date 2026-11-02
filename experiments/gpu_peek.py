"""Quick look at GPU-queue runs while the queue is still going: gamma per window for finished runs.

Usage (on the pod): .venv/bin/python experiments/gpu_peek.py [width]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]


def main():
    width = sys.argv[1] if len(sys.argv) > 1 else "*"
    sig = {}
    for tj in sorted(glob.glob(str(ROOT / f"cache/saes_gpu_*/*__w{width}__*/train.json"))):
        d = Path(tj).parent
        info = json.load(open(tj))
        c = info["config"]
        if c["hook"] not in sig:
            st = np.load(ROOT / "cache" / "acts" / c["model"] / c["hook"] / "stats.npz")
            e = np.sort(np.linalg.eigvalsh(st["cov"]))[::-1]
            sig[c["hook"]] = {w: slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS}
        f = np.load(d / "fires.npy")
        r = eigen_slope(np.load(d / "W_dec.npy"), mask=f > 0)
        g = [slope_from_eigenvalues(r["eigenvalues"], w)["slope"] / sig[c["hook"]][w] for w in WINDOWS]
        tag = c["cond"] + (f"-drop{c['drop_frac']:g}" if c.get("drop_dim") is not None else "")
        print(f"{d.parent.name:<15} L{c['hook'].split('.')[1]:>2} {tag:<16} w{c['width']:<6} k{c['k']:<4} s{c['seed']} "
              f"fvu={info['fvu_eval']:.3f}  " + "  ".join(f"γ{w}={x:.3f}" for w, x in zip(WINDOWS, g)), flush=True)


if __name__ == "__main__":
    main()
