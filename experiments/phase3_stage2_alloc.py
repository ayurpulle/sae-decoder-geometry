"""Phase 3 stage 2: is gamma the allocation exponent? (real runs in cache/saes_p3s2)

With near-orthogonal planted directions, a decoder cloud made of n_i latents near each u_i has eigenvalues ∝ n_i,
so gamma (decoder slope / Sigma slope) should equal c in n_i ∝ p_i^c. Per run:
  gamma_all      all alive latents (as phase3_analyze.py)
  gamma_matched  only latents matched to a feature (score > 0.7)
  c              slope of log n_i on log p_i over covered features (n_i = matched latents per feature),
                 and over the top 50 / top 100 features (the ranks the gamma windows see)
  junk           alive latents with no feature above 0.3: count and median firing count vs matched
The unmatched latents in the pilot fire ~20x less than matched ones and carry little feature energy (noise fits),
so gamma_all mixes allocation with an isotropic floor. Writes results/phase3/stage2_alloc.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(2, 25), (5, 50), (10, 100)]
COS_MATCH, COS_JUNK = 0.7, 0.3


def main():
    rows = []
    for tj in sorted(glob.glob(str(ROOT / "cache" / "saes_p3s2" / "*__real__*" / "train.json"))):
        d = Path(tj).parent
        c = json.load(open(tj))["config"]
        a = ROOT / "cache" / "acts" / c["model"] / c["hook"]
        t, st = np.load(a / "truth.npz"), np.load(a / "stats.npz")
        sig = np.sort(np.linalg.eigvalsh(st["cov"]))[::-1]
        W, fires = np.load(d / "W_dec.npy"), np.load(d / "fires.npy")
        W, f = W[fires > 0], fires[fires > 0]
        U, V, man, p = t["U"], t["V"], t["is_manifold"], t["p"]
        sc = np.where(man[None], np.hypot(W @ U.T, W @ V.T), W @ U.T)
        best, s = sc.argmax(1), sc.max(1)
        ok, junk = s > COS_MATCH, s < COS_JUNK
        n_i = np.bincount(best[ok], minlength=len(p))

        def gam(mask):
            if mask.sum() < 120:
                return {}
            e = eigen_slope(W, mask=mask)["eigenvalues"]
            return {str(w): slope_from_eigenvalues(e, w)["slope"] / slope_from_eigenvalues(sig, w)["slope"] for w in WINDOWS}

        alloc = {}
        for name, top in [("covered", len(p)), ("top50", 50), ("top100", 100)]:
            have = n_i[:top] > 0  # features are indexed by frequency rank
            if have.sum() > 10:
                lr = linregress(np.log(p[:top][have]), np.log(n_i[:top][have]))
                alloc[name] = {"c": lr.slope, "r2": lr.rvalue ** 2, "n_features": int(have.sum())}
        rows.append({"data": c["model"].removeprefix("synth-"), "k": c["k"], "alive": int(len(W)),
                     "matched": int(ok.sum()), "junk": int(junk.sum()),
                     "fires_matched_median": float(np.median(f[ok])) if ok.any() else None,
                     "fires_junk_median": float(np.median(f[junk])) if junk.any() else None,
                     "n_top5": n_i[:5].tolist(), "gamma_all": gam(np.ones(len(W), bool)), "gamma_matched": gam(ok),
                     "alloc": alloc})
    rows.sort(key=lambda r: (r["data"], r["k"]))
    print(f"{'data':<10} {'k':>3} {'alive':>5} {'match':>5} {'junk':>5}  {'γ_all(5,50)':>11} {'γ_match(5,50)':>13} "
          f"{'c_cov':>6} {'c_top100':>8} {'c_top50':>7}  n_top5")
    for r in rows:
        g = lambda x: f"{x.get('(5, 50)', float('nan')):.3f}"
        al = lambda n: f"{r['alloc'][n]['c']:.3f}" if n in r["alloc"] else "  -  "
        print(f"{r['data']:<10} {r['k']:>3} {r['alive']:>5} {r['matched']:>5} {r['junk']:>5}  {g(r['gamma_all']):>11} "
              f"{g(r['gamma_matched']):>13} {al('covered'):>6} {al('top100'):>8} {al('top50'):>7}  {r['n_top5']}")
    out = ROOT / "results" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"windows": WINDOWS, "cos_match": COS_MATCH, "cos_junk": COS_JUNK, "rows": rows},
              open(out / "stage2_alloc.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
