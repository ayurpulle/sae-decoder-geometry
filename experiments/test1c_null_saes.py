"""Test 1c (c): Sigma-matched Gaussian-null TopK SAEs at several Gemma-2-2B layers, plus real-vs-null at L12.

Each layer's N(mu, Sigma) comes from Test 1 (stats.npz written into cache/acts/gemma-2-2b/blocks.L.hook_resid_post/).
L12 also has 1M real cached tokens: a real run and a token-matched Gaussian run are trained there too.
k = 40 to match the L0=40 gamma curve. Width 4096, 3M tokens (Phase 1a/1b: width barely moves the top-window
slope). Skips finished runs. Then measures gamma = decoder slope / Sigma_L slope -> results/test1/null_saes.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

LAYERS = [0, 3, 7, 13, 17, 21, 24]
K, WIDTH, TOKENS, EVAL = 40, 4096, 3_000_000, 200_000
OUT_DIR = "cache/saes_gemma_null"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
RES = ROOT / "results" / "test1"


def run(layer, cond, tokens, seed=0):
    name = f"gemma-2-2b__blocks.{layer}.hook_resid_post__{cond}__w{WIDTH}__k{K}__s{seed}"
    if (ROOT / OUT_DIR / name / "train.json").exists():
        return name
    subprocess.run([sys.executable, "-u", str(ROOT / "train_sae.py"), "--model", "gemma-2-2b",
                    "--hook", f"blocks.{layer}.hook_resid_post", "--cond", cond, "--width", str(WIDTH), "--k", str(K),
                    "--tokens", str(tokens), "--eval_tokens", str(EVAL), "--seed", str(seed), "--out", OUT_DIR],
                   check=True, env=os.environ | {"TRANSFORMERLENS_ALLOW_MPS": "1"})
    return name


def main():
    jobs = [(12, "real", 800_000), (12, "gaussian", 800_000)] + [(L, "gaussian", TOKENS) for L in LAYERS]
    names = {}
    for layer, cond, tokens in jobs:
        names[(layer, cond, tokens)] = run(layer, cond, tokens)
        print("finished", names[(layer, cond, tokens)], flush=True)

    sig_eig = np.load(RES / "sigma_eigenvalues.npy")
    out = {"config": {"k": K, "width": WIDTH, "tokens": TOKENS, "eval_tokens": EVAL, "windows": [list(w) for w in WINDOWS]},
           "gaussian": {}, "real": {}, "gaussian_800k": {}}
    for (layer, cond, tokens), name in names.items():
        d = ROOT / OUT_DIR / name
        W = np.load(d / "W_dec.npy")
        fires = np.load(d / "fires.npy")
        info = json.load(open(d / "train.json"))
        r = eigen_slope(W, mask=fires > 0)
        sig = {w: slope_from_eigenvalues(sig_eig[layer], w)["slope"] for w in WINDOWS}
        dec = {w: slope_from_eigenvalues(r["eigenvalues"], w)["slope"] for w in WINDOWS}
        bucket = "gaussian_800k" if (cond == "gaussian" and tokens == 800_000) else cond
        out[bucket][str(layer)] = {"run": name, "fvu_eval": info["fvu_eval"], "n_dead": int((fires == 0).sum()),
                                   "decoder_slope": {str(w): dec[w] for w in WINDOWS},
                                   "sigma_slope": {str(w): sig[w] for w in WINDOWS},
                                   "gamma": {str(w): dec[w] / sig[w] for w in WINDOWS}}
        print(f"L{layer:>2} {bucket:<13} fvu={info['fvu_eval']:.3f} dead={int((fires == 0).sum()):>4} "
              + " ".join(f"γ{w}={dec[w] / sig[w]:.3f}" for w in WINDOWS), flush=True)
    json.dump(out, open(RES / "null_saes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
