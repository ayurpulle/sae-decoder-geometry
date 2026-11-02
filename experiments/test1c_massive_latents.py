"""Test 1c (d): do Gemma Scope L12 decoders respond to the massive-activation dimension (1170)?

For each layer-12 ladder SAE (16k, 65k): the distribution of |W_dec[:, 1170]| (unit-norm rows, so this is
the cosine with that axis), the count of latents with |cos| > tau, and the eigen slope after removing them
(compared with removing the same number of random latents). Also the same for dim 334 (heavy-tailed at
layers 10–13) and a control dimension with ordinary kurtosis.
If the mid-stack amplification of gamma comes from latents the SAE devotes to the massive dimension,
removing them should flatten the slope noticeably more than removing random latents.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402
from phase1_width_ladder import LADDER, WINDOWS, fetch_wdec  # noqa: E402

DIMS = {"massive_1170": 1170, "massive_334": 334, "control_100": 100}
TAUS = [0.2, 0.5]
OUT = ROOT / "results" / "test1"


def main():
    res = {}
    rng = np.random.default_rng(0)
    print(f"{'sae':<36} {'dim':<13} {'max|cos|':>8} {'n>0.2':>6} {'n>0.5':>6} {'slope':>7} {'rm>0.2':>7} {'rm_rand':>8} {'rm>0.5':>7}")
    for width in ["16k", "65k"]:
        for l0 in LADDER[width]:
            sae_id = f"layer_12/width_{width}/average_l0_{l0}"
            W = np.load(fetch_wdec(sae_id), mmap_mode="r")
            base = eigen_slope(W)["slope"]
            rec = {"slope": base, "dims": {}}
            for name, dim in DIMS.items():
                c = np.abs(np.asarray(W[:, dim], dtype=np.float32))
                row = {"max_abs_cos": float(c.max()), "mean_abs_cos": float(c.mean())}
                for tau in TAUS:
                    keep = c <= tau
                    n_rm = int((~keep).sum())
                    row[f"n_above_{tau}"] = n_rm
                    row[f"slope_rm_{tau}"] = eigen_slope(W, mask=keep)["slope"] if n_rm else base
                    if tau == 0.2 and n_rm:
                        rmask = np.ones(len(c), dtype=bool)
                        rmask[rng.choice(len(c), n_rm, replace=False)] = False
                        row["slope_rm_random_same_n"] = eigen_slope(W, mask=rmask)["slope"]
                    elif tau == 0.2:
                        row["slope_rm_random_same_n"] = base
                rec["dims"][name] = row
                print(f"{sae_id:<36} {name:<13} {row['max_abs_cos']:>8.3f} {row['n_above_0.2']:>6} {row['n_above_0.5']:>6} "
                      f"{base:>+7.3f} {row['slope_rm_0.2']:>+7.3f} {row['slope_rm_random_same_n']:>+8.3f} {row['slope_rm_0.5']:>+7.3f}")
            res[sae_id] = rec
            del W
    json.dump({"dims": DIMS, "taus": TAUS, "window": list(WINDOWS[1]), "results": res},
              open(OUT / "massive_latents_L12.json", "w"), indent=1)


if __name__ == "__main__":
    main()
