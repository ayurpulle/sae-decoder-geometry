"""Phase 3 analysis: gamma per run on planted-feature data, plus (with --match) latent-to-feature matching.

gamma = decoder eigen slope (alive latents) / slope of the training data's covariance, on the d=256 windows.
For scrambled runs the reference covariance is diag(Sigma). --match assigns each alive latent to the planted
feature whose direction (u_i, or the (u_i, v_i) plane for a circle) it is closest to, and reports latents per
feature n_i against p_i (the allocation law n_i ∝ p_i^(1/(1+b)) is a straight line in log-log).
Usage: python experiments/phase3_analyze.py [group=saes_p3] [--match]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import eigen_slope, slope_from_eigenvalues  # noqa: E402

WINDOWS = [(2, 25), (5, 50), (10, 100)]
PRIMARY = str((5, 50))
COS_MATCH = 0.7  # a latent counts as allocated to a feature only above this cosine (or in-plane norm for circles)


def match(W, truth):
    """Best planted feature per latent: cosine with u_i for points, norm of the projection on (u_i, v_i) for circles."""
    U, V, man = truth["U"], truth["V"], truth["is_manifold"]
    cu, cv = W @ U.T, W @ V.T
    score = np.where(man[None, :], np.sqrt(cu ** 2 + cv ** 2), cu)
    best = score.argmax(1)
    return best, score[np.arange(len(W)), best]


def main():
    group = next((a for a in sys.argv[1:] if not a.startswith("--")), "saes_p3")
    do_match = "--match" in sys.argv
    sig, rows = {}, []
    for tj in sorted(glob.glob(str(ROOT / "cache" / group / "*" / "train.json"))):
        d = Path(tj).parent
        info = json.load(open(tj))
        c = info["config"]
        act_dir = ROOT / "cache" / "acts" / c["model"] / c["hook"]
        if c["model"] not in sig:
            st = np.load(act_dir / "stats.npz")
            e = np.sort(np.linalg.eigvalsh(st["cov"]))[::-1]
            dg = np.sort(np.diag(st["cov"]))[::-1]
            sig[c["model"]] = {"full": {str(w): slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS},
                               "diag": {str(w): slope_from_eigenvalues(dg, w)["slope"] for w in WINDOWS}}
        W = np.load(d / "W_dec.npy")
        fires = np.load(d / "fires.npy")
        alive = fires > 0
        r = eigen_slope(W, mask=alive) if alive.sum() > 120 else None
        ref = sig[c["model"]]["diag" if c["cond"] == "scrambled" else "full"]
        dec = {str(w): slope_from_eigenvalues(r["eigenvalues"], w)["slope"] for w in WINDOWS} if r else {}
        row = {"run": d.name, "data": c["model"].removeprefix("synth-"), "cond": c["cond"], "width": c["width"], "k": c["k"],
               "seed": c["seed"], "fvu": info["fvu_eval"], "alive": int(alive.sum()),
               "decoder_slope": dec, "gamma": {w: dec[w] / ref[w] for w in dec}, "sigma_slope": ref}
        if do_match and c["cond"] == "real" and (act_dir / "truth.npz").exists():
            truth = dict(np.load(act_dir / "truth.npz"))
            best, score = match(W[alive], truth)
            ok = score > COS_MATCH
            n_i = np.bincount(best[ok], minlength=len(truth["p"]))
            p = truth["p"]
            have = n_i > 0
            fit = linregress(np.log(p[have]), np.log(n_i[have])) if have.sum() > 10 else None
            row["match"] = {"frac_matched": float(ok.mean()), "features_covered": int(have.sum()),
                            "n_features": len(p), "slope_logn_vs_logp": fit.slope if fit else None,
                            "r2": fit.rvalue ** 2 if fit else None,
                            "n_manifold_mean": float(n_i[truth["is_manifold"] & have].mean()) if (truth["is_manifold"] & have).any() else None,
                            "n_point_mean": float(n_i[~truth["is_manifold"] & have].mean()) if (~truth["is_manifold"] & have).any() else None}
        rows.append(row)
    rows.sort(key=lambda x: (x["data"], x["cond"], x["width"], x["k"], x["seed"]))
    print(f"{'data':<12} {'cond':<9} {'width':>6} {'k':>4} s {'alive':>6} {'fvu':>6}  " + "  ".join(f"γ{w}".rjust(12) for w in WINDOWS))
    for x in rows:
        g = "  ".join(f"{x['gamma'].get(str(w), float('nan')):>12.3f}" for w in WINDOWS)
        line = f"{x['data']:<12} {x['cond']:<9} {x['width']:>6} {x['k']:>4} {x['seed']} {x['alive']:>6} {x['fvu']:>6.3f}  {g}"
        if "match" in x:
            mt = x["match"]
            line += (f"   matched {mt['frac_matched']:.2f}, covers {mt['features_covered']}/{mt['n_features']}, "
                     f"log n ~ {mt['slope_logn_vs_logp'] if mt['slope_logn_vs_logp'] is None else round(mt['slope_logn_vs_logp'], 3)}·log p")
        print(line)
    # gamma(k) fits for real/gaussian per dataset/width, primary window, runs with >= 1000 alive latents
    fits = {}
    groups = defaultdict(list)
    for x in rows:
        if PRIMARY in x["gamma"] and x["alive"] >= 1000:
            groups[(x["data"], x["cond"], x["width"])].append(x)
    for key, xs in sorted(groups.items()):
        ks = sorted({x["k"] for x in xs})
        if len(ks) < 3:
            continue
        g = [np.mean([x["gamma"][PRIMARY] for x in xs if x["k"] == k]) for k in ks]
        lr = linregress(np.log2(ks), g)
        fits["|".join(map(str, key))] = {"ks": ks, "gamma": g, "intercept": lr.intercept, "per_doubling": lr.slope, "r2": lr.rvalue ** 2}
        print(f"fit {key}: gamma = {lr.intercept:.3f} {lr.slope:+.3f}*log2(k)  R2={lr.rvalue ** 2:.3f}  (k with >=1000 alive: {ks})")
    out = ROOT / "results" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"windows": WINDOWS, "rows": rows, "fits": fits}, open(out / f"{group}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
