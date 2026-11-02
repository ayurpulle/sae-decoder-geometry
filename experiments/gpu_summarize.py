"""Summarize GPU-queue SAEs into small JSON (so only results/gpu/ needs copying back, not W_dec files).

For every cache/saes_gpu_*/<run>/train.json: decoder eigen spectrum (dead latents excluded) using
geometry.eigen_slope, slopes on the pre-registered Gemma windows, the layer's Sigma slopes from the run's own
activation cache, gamma = decoder slope / Sigma slope, eval FVU, dead count, and mean L0 on eval.
Writes results/gpu/runs.json and results/gpu/sigma.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
OUT = ROOT / "results" / "gpu"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sigma, runs = {}, []
    for tj in sorted(ROOT.glob("cache/saes_gpu_*/*/train.json")):
        info = json.load(open(tj))
        cfg = info["config"]
        hook = cfg["hook"]
        if hook not in sigma:
            st = np.load(ROOT / "cache" / "acts" / cfg["model"] / hook / "stats.npz")
            e = np.sort(np.linalg.eigvalsh(st["cov"]))[::-1]
            sigma[hook] = {"eigenvalues": e.tolist(), "slopes": {str(w): slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS},
                           "n_tokens": int(st["n"])}
        fires = np.load(tj.parent / "fires.npy")
        r = eigen_slope(np.load(tj.parent / "W_dec.npy"), mask=fires > 0)
        dec = {str(w): slope_from_eigenvalues(r["eigenvalues"], w)["slope"] for w in WINDOWS}
        runs.append({
            "group": tj.parent.parent.name, "run": tj.parent.name, "hook": hook, "layer": int(hook.split(".")[1]),
            "cond": cfg["cond"], "width": cfg["width"], "k": cfg["k"], "seed": cfg["seed"],
            "drop_dim": cfg.get("drop_dim"), "drop_frac": cfg.get("drop_frac") if cfg.get("drop_dim") is not None else None,
            "drop_threshold": info.get("drop_threshold"), "epochs": cfg.get("epochs", 1),
            "fvu_eval": info["fvu_eval"], "n_dead": int((fires == 0).sum()), "l0_eval": float(fires.sum() / info["n_eval_tokens"]),
            "n_train_tokens": info["n_train_tokens"], "seconds": info["seconds"], "device": info.get("device"),
            "git_hash": cfg.get("git_hash"),
            "decoder_slope": dec, "gamma": {w: dec[w] / sigma[hook]["slopes"][w] for w in dec},
            "eigenvalues": r["eigenvalues"].tolist(),
        })
        print(f"{runs[-1]['group']:<15} {tj.parent.name:<70} fvu={info['fvu_eval']:.3f} dead={runs[-1]['n_dead']:>5} "
              f"γ{WINDOWS[1]}={runs[-1]['gamma'][str(WINDOWS[1])]:.3f}", flush=True)
    json.dump({"windows": [list(w) for w in WINDOWS], "runs": runs}, open(OUT / "runs.json", "w"))
    json.dump(sigma, open(OUT / "sigma.json", "w"))
    print(f"wrote {len(runs)} runs to {OUT / 'runs.json'}")


if __name__ == "__main__":
    main()
