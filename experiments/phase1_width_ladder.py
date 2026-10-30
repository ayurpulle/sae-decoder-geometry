"""Phase 1a: decoder-direction eigen slope across the layer-12 Gemma Scope width ladder.

Geometry only (no firing frequencies yet), so no dead-latent exclusion: dead_latents_excluded=0.
Only W_dec.npy is range-read out of each remote params.npz (stored uncompressed), halving downloads.
W_dec is cached as cache/gemmascope_wdec/<sae_id>/W_dec.npy and loaded memmapped.
"""
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import PAPER_FIT_RANGE, eigen_slope, slope_from_eigenvalues  # noqa: E402

REPO = "google/gemma-scope-2b-pt-res"
LAYER = 12
# All L0s for widths <= 131k; two (one ~20, one ~70) for the large widths to bound download size.
LADDER = {
    "16k": [22, 41, 82, 176, 445],
    "32k": [12, 22, 40, 76, 155, 360],
    "65k": [21, 38, 72, 141, 297],
    "131k": [12, 20, 36, 67, 129, 264],
    "262k": [21, 67],
    "524k": [22, 65],
    "1m": [19, 73],
}
# Pre-registered fit windows (0-based [start, end)); see notes/LOG.md. Primary = paper's.
WINDOWS = [(3, 50), PAPER_FIT_RANGE, (30, 450), (100, 1000)]
CACHE = ROOT / "cache" / "gemmascope_wdec"
OUT = ROOT / "results" / "phase1"


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def fetch_array(sae_id: str, name: str = "W_dec") -> Path:
    """Range-read one array (e.g. W_dec, W_enc, b_enc, threshold) out of a remote params.npz; cached."""
    dst = CACHE / sae_id / f"{name}.npy"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with HfFileSystem().open(f"{REPO}/{sae_id}/params.npz", "rb", block_size=2**26) as f:
        with zipfile.ZipFile(f).open(f"{name}.npy") as m, open(tmp, "wb") as o:
            shutil.copyfileobj(m, o, length=2**24)
    tmp.rename(dst)
    return dst


def fetch_wdec(sae_id: str) -> Path:
    return fetch_array(sae_id, "W_dec")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "width_ladder_L12.json"
    done = json.load(open(out_json))["results"] if out_json.exists() else {}
    for width, l0s in LADDER.items():
        for l0 in l0s:
            sae_id = f"layer_{LAYER}/width_{width}/average_l0_{l0}"
            if sae_id in done:
                continue
            t0 = time.time()
            W = np.load(fetch_wdec(sae_id), mmap_mode="r")
            t1 = time.time()
            r = eigen_slope(W)
            tag = sae_id.replace("/", "__")
            np.save(OUT / f"eig_{tag}.npy", r["eigenvalues"])
            done[sae_id] = {
                "layer": LAYER, "width": width, "n_latents": r["n_latents"], "l0": l0,
                "slope": r["slope"], "r2": r["r2"], "gaussian_slope": r["gaussian"]["slope"],
                "row_norm": r["row_norm"],
                "windows": [slope_from_eigenvalues(r["eigenvalues"], w) for w in WINDOWS],
                "gaussian_windows": [slope_from_eigenvalues(r["gaussian"]["eigenvalues"], w) for w in WINDOWS],
                "top_eig": float(r["eigenvalues"][0]),
            }
            ws = " ".join(f"{w['slope']:+.3f}" for w in done[sae_id]["windows"])
            print(f"{sae_id}: slope={r['slope']:+.4f} R2={r['r2']:.3f} windows[{ws}] "
                  f"gauss={r['gaussian']['slope']:+.4f} fetch={t1 - t0:.0f}s eig={time.time() - t1:.0f}s", flush=True)
            config = {"model": "gemma-2-2b", "release": "gemma-scope-2b-pt-res", "layer": LAYER,
                      "primary_fit_range": list(PAPER_FIT_RANGE), "windows": [list(w) for w in WINDOWS],
                      "center": True, "normalize": False, "dead_latents_excluded": 0, "git_hash": git_hash()}
            json.dump({"config": config, "results": done}, open(out_json, "w"), indent=1)
            del W


if __name__ == "__main__":
    main()
