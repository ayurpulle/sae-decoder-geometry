"""Phase 2 (small model): TopK SAEs on Pythia-70m layer-3 resid: real vs Gaussian vs scrambled.

Runs train_sae.py for every (seed, width, cond), seed-major so each completed seed is a full
comparison. Skips runs whose train.json already exists. Analysis: experiments/phase2_analyze.py.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL, HOOK, K, TOKENS = "pythia-70m", "blocks.3.hook_resid_post", 32, 15_000_000
WIDTHS = [1024, 4096, 16384]
CONDS = ["real", "gaussian", "scrambled"]
SEEDS = [0, 1, 2]

for seed in SEEDS:
    for width in WIDTHS:
        for cond in CONDS:
            name = f"{MODEL}__{HOOK}__{cond}__w{width}__k{K}__s{seed}"
            if (ROOT / "cache" / "saes" / name / "train.json").exists():
                continue
            subprocess.run([sys.executable, "-u", str(ROOT / "train_sae.py"), "--model", MODEL, "--hook", HOOK,
                            "--cond", cond, "--width", str(width), "--k", str(K), "--tokens", str(TOKENS),
                            "--seed", str(seed)], check=True, env=os.environ | {"TRANSFORMERLENS_ALLOW_MPS": "1"})
