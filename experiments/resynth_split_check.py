"""Local check of resynth.py variants (default E and F, with B for reference) before a GPU run: Sigma's eigen-slope per
window from N_EVAL sampled tokens, using the code statistics measured on the pod (results/resynth/codestats_L12_pod.npz).
gamma divides by Sigma's slope, so a variant whose Sigma is near-flat in a window gives an ill-conditioned gamma there.
Writes results/resynth/split_check.json, or <name>.json with: python resynth_split_check.py KINDS NAME.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import resynth  # noqa: E402
from resynth_check import cov, slopes  # noqa: E402

SAE = ROOT / "cache" / "gemmascope" / "L12" / "params.npz"
STATS = ROOT / "results" / "resynth" / "codestats_L12_pod.npz"
N_EVAL = 200_000


def main():
    dev = resynth.device()
    sae = resynth.load_sae(SAE, dev)
    s = np.load(STATS)
    stats = {k: s[k] for k in s.files}
    out = {}
    kinds, name = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("BEF", "split_check")
    for kind in kinds:
        sl, e = slopes(cov(xh for xh, _ in resynth.variant(kind, sae, None, N_EVAL, 0, stats, dev)))
        out[kind] = {"sigma_slopes": sl, "eig": e[:1000].tolist()}
        print(f"{kind}  " + " ".join(f"{x:.3f}" for x in sl) + f"   top eig {e[0]:.2f}, eig 150 {e[149]:.4f}", flush=True)
    json.dump(out, open(ROOT / "results" / "resynth" / f"{name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
