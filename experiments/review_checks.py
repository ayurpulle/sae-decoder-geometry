"""Checks requested by the external review (notes/LOG.md, 2026-11-03), all from data already on disk.

1. The zero-parameter law gamma = 1 - ln k / ln d against every gamma measured so far.
2. Alive counts of the GPU-queue nulls (could "real spreads decoders out" just be "more latents survive"?).
3. Subsample the 1m / 524k / 262k Gemma Scope decoders to 16k and 65k rows and refit: estimation (rows per
   dimension) vs geometry as the source of the deep-window width drift.
4. Basis-matched gamma and principal-angle overlaps (geometry.basis_gamma) for Gemma Scope L12, our Pythia
   SAEs (real / gaussian / scrambled) and the local Gemma Gaussian nulls; near-duplicate fractions for the nulls.
Writes results/review/checks.json.
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import basis_gamma, cloud_cov, eigen_slope, slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "review"
GEMMA_WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
GEMMA_RANGES = ((1, 150), (10, 150), (100, 1000))
PYTHIA_RANGES = ((1, 50), (5, 50), (10, 100))


def law(k, d):
    return 1 - np.log(k) / np.log(d)


def nn_dup_frac(W, thr=0.5, block=4096):
    W = np.asarray(W, dtype=np.float32)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    best = np.full(len(W), -1.0, dtype=np.float32)
    for i in range(0, len(W), block):
        S = W[i:i + block] @ W.T
        S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -1
        best[i:i + block] = S.max(1)
    return float((best > thr).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # ---- 1. law vs measurements ----
    print("1. gamma vs 1 - ln k / ln d")
    runs = json.load(open(ROOT / "results/gpu/runs.json"))["runs"]
    k_real = defaultdict(list)
    for r in runs:
        if r["group"] == "saes_gpu_k" and r["cond"] == "real":
            k_real[r["k"]].append(r["gamma"][str((10, 150))])
    rows = [{"source": "our TopK, Gemma L13, ranks 11-150", "k": k, "d": 2304, "measured": float(np.mean(v)), "law": law(k, 2304)}
            for k, v in sorted(k_real.items())]
    gl = json.load(open(ROOT / "results/test1/gamma_l0_L12.json"))
    for x in gl["rows"]:
        rows.append({"source": f"Gemma Scope L12 {x['width']}, ranks 11-150", "k": x["l0"], "d": 2304,
                     "measured": x["gamma"][str((10, 150))], "law": law(x["l0"], 2304)})
    for key, v in gl["pythia"]["runs"].items():
        if key.startswith("real"):
            for w in ["(2, 25)", "(5, 50)", "(10, 100)"]:
                rows.append({"source": f"our TopK, Pythia L3 {key}, ranks {w}", "k": 32, "d": 512, "measured": v["gamma"][w], "law": law(32, 512)})
    for x in rows:
        x["gap"] = x["measured"] - x["law"]
    res["law"] = rows
    for x in rows:
        if "Gemma Scope" not in x["source"] or x["source"].endswith("16k, ranks 11-150") or "1m" in x["source"]:
            print(f"   {x['source']:<46} k={x['k']:>6.1f} d={x['d']}  measured {x['measured']:.3f}  law {x['law']:.3f}  gap {x['gap']:+.3f}")

    # ---- 2. alive counts in the GPU nulls ----
    print("\n2. dead latents in GPU-queue runs (group A, mean over seeds)")
    dead = defaultdict(list)
    for r in runs:
        if r["group"] == "saes_gpu_p2":
            dead[(r["width"], r["cond"], r["layer"])].append(r["n_dead"])
    res["dead_A"] = {f"{w}|{c}|L{L}": float(np.mean(v)) for (w, c, L), v in sorted(dead.items())}
    for key, v in res["dead_A"].items():
        print(f"   {key:<22} dead {v:.0f}")

    # ---- 3. subsampling the widest Gemma Scope SAEs ----
    print("\n3. subsample wide Gemma Scope decoders (L12) to 16k / 65k rows, 3 draws each; slope in 4 windows")
    rng = np.random.default_rng(0)
    sub = {}
    for sae in ["layer_12/width_1m/average_l0_19", "layer_12/width_1m/average_l0_73",
                "layer_12/width_524k/average_l0_22", "layer_12/width_262k/average_l0_67"]:
        W = np.load(ROOT / "cache/gemmascope_wdec" / sae / "W_dec.npy", mmap_mode="r")
        full = eigen_slope(W)["eigenvalues"]
        rec = {"full": [slope_from_eigenvalues(full, w)["slope"] for w in GEMMA_WINDOWS]}
        for n in [16384, 65536]:
            draws = []
            for _ in range(3):
                idx = np.sort(rng.choice(W.shape[0], n, replace=False))
                e = eigen_slope(np.asarray(W[idx]))["eigenvalues"]
                draws.append([slope_from_eigenvalues(e, w)["slope"] for w in GEMMA_WINDOWS])
            rec[f"sub{n}"] = np.mean(draws, 0).tolist()
        sub[sae] = rec
        print(f"   {sae:<36} full " + " ".join(f"{s:+.3f}" for s in rec["full"])
              + " | sub16k " + " ".join(f"{s:+.3f}" for s in rec["sub16384"])
              + " | sub65k " + " ".join(f"{s:+.3f}" for s in rec["sub65536"]))
    ladder = json.load(open(ROOT / "results/phase1/width_ladder_L12.json"))["results"]
    for sae in ["layer_12/width_16k/average_l0_22", "layer_12/width_16k/average_l0_82", "layer_12/width_65k/average_l0_21", "layer_12/width_65k/average_l0_72"]:
        print(f"   reference {sae:<26} " + " ".join(f"{w['slope']:+.3f}" for w in ladder[sae]["windows"]))
    res["subsample"] = sub

    # ---- 4. basis-matched gamma + subspace overlap ----
    print("\n4. basis-matched gamma (regress log diag(U^T C U) on log lam) and top-m subspace overlap")
    bg = []
    Sig12 = np.load(ROOT / "results/test1/sigma_gemma-2-2b.npz")["cov"][12]
    freqs = json.load(open(ROOT / "results/phase1/freqs_L12.json"))["results"]
    for sae_id in sorted(ladder, key=lambda s: (int(re.search(r"width_(\d+)", s.replace("k", "000").replace("1m", "1000000")).group(1)), ladder[s]["l0"])):
        W = np.load(ROOT / "cache/gemmascope_wdec" / sae_id / "W_dec.npy", mmap_mode="r")
        r = basis_gamma(cloud_cov(W), Sig12, rank_ranges=GEMMA_RANGES)
        l0 = freqs[sae_id]["l0_measured"] if sae_id in freqs else float(ladder[sae_id]["l0"])
        bg.append({"model": "gemma-scope-L12", "sae": sae_id, "k": l0, "d": 2304, **r,
                   "rank_gamma": ladder[sae_id]["windows"][1]["slope"] / slope_from_eigenvalues(np.sort(np.linalg.eigvalsh(Sig12))[::-1], (10, 150))["slope"]})
        f = r["fits"]
        print(f"   {sae_id:<36} L0={l0:>6.1f}  basis γ " + " ".join(f"{x['gamma']:.3f}(R²{x['r2']:.2f})" for x in f)
              + f"  rank γ {bg[-1]['rank_gamma']:.3f}  law {law(l0, 2304):.3f}  overlap10 {r['subspace_overlap'][10]:.2f} overlap50 {r['subspace_overlap'][50]:.2f}  diag {r['diag_share']:.2f}")
    st = np.load(ROOT / "cache/acts/pythia-70m/blocks.3.hook_resid_post/stats.npz")
    SigP, SigPd = st["cov"], np.diag(np.diag(st["cov"]))
    for tj in sorted(glob.glob(str(ROOT / "cache/saes/*/train.json"))):
        d = Path(tj).parent
        m = re.search(r"__(real|gaussian|scrambled)__w(\d+)__k(\d+)__s(\d)", d.name)
        fires = np.load(d / "fires.npy")
        W = np.load(d / "W_dec.npy")
        r = basis_gamma(cloud_cov(W, mask=fires > 0), SigPd if m[1] == "scrambled" else SigP, rank_ranges=PYTHIA_RANGES)
        bg.append({"model": "pythia-L3", "cond": m[1], "width": int(m[2]), "k": int(m[3]), "seed": int(m[4]), "d": 512,
                   "alive": int((fires > 0).sum()), "dup05": nn_dup_frac(W[fires > 0]), **r})
    for tj in sorted(glob.glob(str(ROOT / "cache/saes_gemma_null/*/train.json"))):
        d = Path(tj).parent
        m = re.search(r"blocks\.(\d+)\.hook_resid_post__(real|gaussian)__w(\d+)__k(\d+)", d.name)
        L = int(m[1])
        Sig = np.load(ROOT / "results/test1/sigma_gemma-2-2b.npz")["cov"][L]
        fires = np.load(d / "fires.npy")
        W = np.load(d / "W_dec.npy")
        r = basis_gamma(cloud_cov(W, mask=fires > 0), Sig, rank_ranges=GEMMA_RANGES)
        bg.append({"model": "gemma-local-null", "layer": L, "cond": m[2], "width": int(m[3]), "k": int(m[4]), "d": 2304,
                   "alive": int((fires > 0).sum()), "dup05": nn_dup_frac(W[fires > 0]), **r})
    res["basis_gamma"] = bg
    print("\n   Pythia L3 (k=32, d=512; law 0.444), mean over seeds:")
    agg = defaultdict(list)
    for x in bg:
        if x["model"] == "pythia-L3":
            agg[(x["cond"], x["width"])].append(x)
    for (c, w), xs in sorted(agg.items()):
        g = np.mean([[f["gamma"] for f in x["fits"]] for x in xs], 0)
        print(f"   {c:<9} w{w:<6} basis γ " + " ".join(f"{v:.3f}" for v in g)
              + f"  overlap10 {np.mean([x['subspace_overlap'][10] for x in xs]):.2f} overlap50 {np.mean([x['subspace_overlap'][50] for x in xs]):.2f}"
              + f"  alive {np.mean([x['alive'] for x in xs]):.0f}  dup>0.5 {np.mean([x['dup05'] for x in xs]):.3f}")
    print("\n   local Gemma SAEs (k=40, width 4096, 0.8–3M tokens; law 0.524):")
    for x in bg:
        if x["model"] == "gemma-local-null":
            print(f"   L{x['layer']:>2} {x['cond']:<9} basis γ " + " ".join(f"{f['gamma']:.3f}" for f in x["fits"])
                  + f"  overlap10 {x['subspace_overlap'][10]:.2f} overlap50 {x['subspace_overlap'][50]:.2f}  alive {x['alive']}  dup>0.5 {x['dup05']:.3f}")
    json.dump(res, open(OUT / "checks.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
