"""Descriptive checks on the ladder job (experiments/r2_job_ladder.py), beside the fixed rule of
experiments/r2_ladder_summary.py. Nothing here changes a verdict; it reports how far the inputs to the rule can be
trusted. Reads results/r2/runs_ladder.jsonl, results/r2/ladder_summary.json, results/r2/ladder_status.json,
results/r2/sigma/*.npz, the Gemma reference rows and results/r2_pm2/pmatch2_summary.json; writes
results/r2/ladder_extra.json.

Per dataset (four ladder rungs and trained Gemma-2-2B layers 0 and 13):
  sigma        top-1 / top-10 eigenvalue shares, participation ratio, slope and R^2 of the log-log fit in each bulk
               window, and the slopes of the lower and upper halves of each window (split at the log-midpoint rank)
  clouds       for R (real k = 40, three seeds), Z (accepted V0 seeds), N (accepted V1): decoder slope, fit R^2 and
               half-window slopes of the fixed-n mean spectrum (eig_fixed_mean); gamma = slope / Sigma's slope
  z_vs_rho     gamma of every V0 run (calibration and accepted) against its rho
  live         live latents of every run, and whether any run was scored on fewer than 8,192
  power        top-1% latent-power share of the real seed-0 SAE, the V1 dictionary, and the V1 SAE
Also: the trained remainder with the three-seed power-matched mean of pmatch2 in place of the seed-0 run, and the
ratio rem_random / rem_trained under it (descriptive only; the rule uses the seed-0 run).
  python experiments/r2_ladder_extra.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from geometry import slope_from_eigenvalues  # noqa: E402
from r2_summary import load  # noqa: E402

R2 = ROOT / "results" / "r2"
BULK = [(30, 450), (100, 1000)]
SIGMA_FILES = {"embT": "ladder-embT", "embR": "ladder-embR", "randL0": "ladder-rand_L0",
               "randL13": "ladder-rand_L13", "gemma_L0": "gemma-2-2b_L0", "gemma_L13": "gemma-2-2b_L13"}
PAIRS = [("embR", "embT"), ("randL0", "gemma_L0"), ("randL13", "gemma_L13")]


def halves(w):
    lo, hi = w
    mid = int(round(np.sqrt((lo + 1) * hi)))
    return (lo, mid), (mid, hi)


def fit(eig, w):
    f = slope_from_eigenvalues(np.asarray(eig), w)
    a, b = halves(w)
    return {"slope": f["slope"], "r2": f["r2"],
            "slope_lower_half": slope_from_eigenvalues(np.asarray(eig), a)["slope"],
            "slope_upper_half": slope_from_eigenvalues(np.asarray(eig), b)["slope"],
            "halves": [list(a), list(b)]}


def sigma_stats(label):
    e = np.load(R2 / "sigma" / f"{label}.npz")["eigenvalues"]
    e = np.sort(e)[::-1]
    return e, {"top1_share": float(e[0] / e.sum()), "top10_share": float(e[:10].sum() / e.sum()),
               "participation_ratio": float(e.sum() ** 2 / (e ** 2).sum()),
               "windows": {str(w): fit(e, w) for w in BULK}}


def cloud(rows, sig_e):
    out = {}
    for w in BULK:
        sw = slope_from_eigenvalues(sig_e, w)["slope"]
        fs = [fit(r["eig_fixed_mean"], w) for r in rows]
        out[str(w)] = {"slope": float(np.mean([f["slope"] for f in fs])),
                       "r2_min": float(min(f["r2"] for f in fs)),
                       "gamma_of_mean_spectrum": float(np.mean([f["slope"] for f in fs]) / sw),
                       "gamma_lower_half": float(np.mean([f["slope_lower_half"] for f in fs])
                                                 / slope_from_eigenvalues(sig_e, halves(w)[0])["slope"]),
                       "gamma_upper_half": float(np.mean([f["slope_upper_half"] for f in fs])
                                                 / slope_from_eigenvalues(sig_e, halves(w)[1])["slope"]),
                       "gamma_fixed_rows": [r["gamma_fixed"][str(w)] for r in rows]}
    return out


def main():
    lad = [json.loads(l) for l in open(R2 / "runs_ladder.jsonl") if l.strip()]
    ref = [r for r in load(R2) if r["job"] != "ladder"]
    summ = json.load(open(R2 / "ladder_summary.json"))
    status = json.load(open(R2 / "ladder_status.json"))
    pm2 = json.load(open(ROOT / "results" / "r2_pm2" / "pmatch2_summary.json"))

    groups = {}
    for rung in ("embT", "embR", "randL0", "randL13"):
        rs = [r for r in lad if r["rung"] == rung and "eig_fixed_mean" in r]
        groups[rung] = {"R": [r for r in rs if r["step"] == "real" and r["k"] == 40],
                        "Z": [r for r in rs if r["step"] == "v0"], "N": [r for r in rs if r["step"] == "v1"],
                        "Z_cal": [r for r in rs if r["step"] == "v0_cal"],
                        "extra_k": [r for r in rs if r["step"] == "real_k"]}
    for L in (0, 13):
        groups[f"gemma_L{L}"] = {
            "R": [r for r in ref if r["job"] == "real" and r.get("step") == "Q1" and r.get("cond") == "real"
                  and r.get("layer") == L and r.get("k") == 40 and not r.get("whitened")],
            "Z": [r for r in ref if r["job"] == "sparse2" and r.get("step") in ("S2a", "S2b") and r.get("layer") == L
                  and r.get("k") == 40],
            "N": [r for r in ref if r["job"] == "pmatch" and r.get("step") == "PM"
                  and r.get("setting_key") == f"PM|V1|L{L}|M8192_L16|ds0"],
            "Z_cal": [], "extra_k": []}

    out = {"datasets": {}, "pairs": {}}
    for name, g in groups.items():
        sig_e, sig = sigma_stats(SIGMA_FILES[name])
        d = {"sigma": sig, "clouds": {}, "live": {}, "fixed_n_short_runs": []}
        for role in ("R", "Z", "N"):
            rows = [r for r in g[role] if "eig_fixed_mean" in r]
            if rows:
                d["clouds"][role] = cloud(rows, sig_e)
            d["live"][role] = [r["n_live"] for r in g[role]]
        for r in g["R"] + g["Z"] + g["N"] + g["extra_k"]:
            if r.get("fixed_n_short"):
                d["fixed_n_short_runs"].append({"run": r["run"], "n_live": r["n_live"], "step": r.get("step")})
        if name in status:
            zr = [(r["sparse"]["rho"], r["fvu_eval"], {str(w): r["gamma_fixed"][str(w)] for w in BULK})
                  for r in g["Z"] + g["Z_cal"]]
            d["z_vs_rho"] = sorted(zr, key=lambda t: t[0])
            d["extra_k"] = {f"k{r['k']}": {"fvu": r["fvu_eval"], "n_live": r["n_live"],
                                           "fixed_n_short": r.get("fixed_n_short"),
                                           "gamma_fixed": {str(w): r["gamma_fixed"][str(w)] for w in BULK},
                                           "dec_fixed": {str(w): r["dec_fixed"][str(w)] for w in BULK}}
                            for r in g["extra_k"]}
            d["dec_fixed_R_mean"] = {str(w): float(np.mean([r["dec_fixed"][str(w)] for r in g["R"]])) for w in BULK}
            v1 = g["N"][0]
            dic = json.load(open(R2 / "ladder_pmatch" / f"{v1['sparse']['dictionary_file']['tag']}.json"))
            d["power"] = {"real_seed0_top1pct": v1["power_check"]["real"]["share_top1pct"],
                          "dictionary_top1pct": dic["power"]["share_top1pct"],
                          "dictionary_target_top1pct": dic["power_target"]["share_top1pct"],
                          "null_sae_top1pct": v1["power_check"]["null"]["share_top1pct"],
                          "dictionary_spearman_power_rq": dic["spearman_power_rq"],
                          "dictionary_rounds": dic["iterations"],
                          "dictionary_max_abs_log_power_err": dic["final_power_max_abs_log_err"],
                          "dictionary_cov_rel_frob_err_float64": dic["cov_rel_frob_err_float64"],
                          "sample_cov_rel_frob_err_1M": v1["sparse"]["check_1M"]["rel_frob_err"]}
            d["init"] = next(r["collection"]["weights"] for r in lad if r["rung"] == name and "collection" in r)
        out["datasets"][name] = d

    # trained remainder with the three-seed power-matched mean (pmatch2) beside the rule's seed-0 run
    for rnd, trn in PAIRS:
        c = summ["comparisons"][f"{rnd} vs {trn}"]
        p = {"rule_verdict": c["verdict"], "rem_random": c["rem_random"], "rem_trained_rule": c["rem_trained"],
             "ratio_rule": c["rem_ratio"]}
        if trn.startswith("gemma"):
            L = trn.split("L")[-1]
            s = pm2["settings"][f"PM|V1|L{L}|M8192_L16|ds0"]["windows"]
            R = summ["datasets"][trn]["windows"]
            p["rem_trained_three_seed_N"] = {str(w): s[str(w)]["N"] - R[str(w)]["R"] for w in BULK}
            p["ratio_three_seed_N"] = {str(w): c["rem_random"][str(w)] / p["rem_trained_three_seed_N"][str(w)]
                                       for w in BULK}
        out["pairs"][f"{rnd} vs {trn}"] = p
    json.dump(out, open(R2 / "ladder_extra.json", "w"), indent=1, default=float)

    for name, d in out["datasets"].items():
        s = d["sigma"]
        print(f"{name:<10} Sigma top1 {s['top1_share']:.3f} top10 {s['top10_share']:.3f} PR {s['participation_ratio']:.1f}  "
              + "  ".join(f"{w}: slope {s['windows'][w]['slope']:.3f} r2 {s['windows'][w]['r2']:.4f} "
                          f"halves {s['windows'][w]['slope_lower_half']:.3f}/{s['windows'][w]['slope_upper_half']:.3f}"
                          for w in s["windows"]))
        for role, c in d["clouds"].items():
            print(f"    {role} " + "  ".join(f"{w}: slope {c[w]['slope']:.3f} r2min {c[w]['r2_min']:.4f} "
                                             f"gamma halves {c[w]['gamma_lower_half']:.3f}/{c[w]['gamma_upper_half']:.3f}"
                                             for w in c) + f"  live {d['live'][role]}")
        if "z_vs_rho" in d:
            print("    Z vs rho", [(round(a, 2), round(b, 3), {k: round(v, 3) for k, v in g.items()}) for a, b, g in d["z_vs_rho"]])
            print("    power", {k: round(v, 3) if isinstance(v, float) else v for k, v in d["power"].items()})
        if d["fixed_n_short_runs"]:
            print("    short", d["fixed_n_short_runs"])
    for k, p in out["pairs"].items():
        print(k, json.dumps({a: b for a, b in p.items()}, default=lambda x: round(x, 3)))
    print(f"-> {R2 / 'ladder_extra.json'}")


if __name__ == "__main__":
    main()
