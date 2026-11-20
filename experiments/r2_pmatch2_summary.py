"""Readings of job "pmatch2" (experiments/r2_job_pmatch2.py) under the rules of notes/prereg_ladder.md, Parts 1 and 2.

Three power-matched settings, each with three SAE seeds of the power-matched null N:
  PM|V1|L13|M8192_L16|ds0   (reference, layer 13; seed 0 from results/r2/runs_pmatch.jsonl, seeds 1-2 from pmatch2)
  PM|V1|L0|M8192_L16|ds0    (reference, layer 0; same)
  PM|V1|L13|M32768_L16|ds0  (layer 13, 32,768 atoms; all three seeds from pmatch2)
R is the real k = 40 SAE (three-seed mean, results/r2/runs_real.jsonl, selected as in r2_summary.py's _real_mean).
Z is the equal-power null at the same (layer, M, L): sparse2 S2a/S2b seeds 0-1 for M 8192 (r2_summary.py uses seed 0
only; both are reported), robust step G seed 0 plus pmatch2 V0SEED seeds 1-2 for M 32,768.

Rule (notes/prereg_pmatch.md, unchanged by prereg_ladder.md), applied to the seed means, per bulk window:
closed if |N - R| <= 0.10, overshoot if N < R - 0.10, partial if (Z - N)/(Z - R) >= 0.30, open otherwise;
EXPLAINS if both closed, DOES NOT EXPLAIN if either open, PARTLY EXPLAINS otherwise.

Writes results/r2_pm2/pmatch2_summary.json and paper/figures/tab_pmatch2_r2.tex.
Usage: python experiments/r2_pmatch2_summary.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R2, PM2 = ROOT / "results" / "r2", ROOT / "results" / "r2_pm2"
BULK = ["(30, 450)", "(100, 1000)"]
WINDOWS = ["(3, 50)", "(10, 150)"] + BULK
MARGIN, FRAC = 0.10, 0.30
KEYS = ["PM|V1|L13|M8192_L16|ds0", "PM|V1|L0|M8192_L16|ds0", "PM|V1|L13|M32768_L16|ds0"]
LABEL = {KEYS[0]: "13, ref.", KEYS[1]: "0, ref.", KEYS[2]: "13, $M$=32,768"}
FIRST_RUN = {KEYS[0]: "PARTLY EXPLAINS", KEYS[1]: "DOES NOT EXPLAIN"}  # verdicts of the first run (seed 0)


def rows(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # torn last line
            continue
    return out


def msd(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), (float(xs.std(ddof=1)) if len(xs) > 1 else float("nan"))


def window_class(N, R, Z):
    f = (Z - N) / (Z - R)
    return ("closed" if abs(N - R) <= MARGIN else "overshoot" if N < R - MARGIN
            else "partial" if f >= FRAC else "open"), f


def verdict(classes):
    return ("EXPLAINS" if all(c == "closed" for c in classes) else "DOES NOT EXPLAIN" if "open" in classes
            else "PARTLY EXPLAINS")


def _close(a, b, tol=1e-9):
    """Equal up to rounding (relative 1e-9) for numbers, recursively for dicts; exact otherwise."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def by_seed(rs):
    d = {}
    for r in rs:
        assert r["seed"] not in d, ("duplicate seed", r["run"])
        d[r["seed"]] = r
    return dict(sorted(d.items()))


def dic_tag(key):
    """pmatch dictionary tag of a setting key, e.g. PM|V1|L13|M8192_L16|ds0 -> pmV1L13M8192ds0."""
    _, v, L, ML, ds = key.split("|")
    return f"pm{v}{L}{ML.split('_')[0]}{ds}"


def main():
    real = rows(R2 / "runs_real.jsonl")
    sp2 = rows(R2 / "runs_sparse2.jsonl")
    rob = rows(R2 / "runs_robust.jsonl")
    pm1 = rows(R2 / "runs_pmatch.jsonl")
    pm2 = rows(PM2 / "runs_pmatch2.jsonl")

    out = {"sources": {"pmatch2": "results/r2_pm2/runs_pmatch2.jsonl", "pmatch": "results/r2/runs_pmatch.jsonl",
                       "real": "results/r2/runs_real.jsonl", "equal_power": "results/r2/runs_sparse2.jsonl, "
                       "results/r2/runs_robust.jsonl (step G), results/r2_pm2/runs_pmatch2.jsonl (step V0SEED)"},
           "rule": {"margin": MARGIN, "min_closed_fraction": FRAC, "note": "notes/prereg_pmatch.md"},
           "pmatch2_job_sha256": sorted({r["job_code_sha256"] for r in pm2}), "settings": {}}

    for key in KEYS:
        layer = 13 if "|L13|" in key else 0
        M = 32768 if "M32768" in key else 8192
        R_rows = [r for r in real if r["job"] == "real" and r.get("cond") == "real" and r.get("k") == 40
                  and not r.get("whitened") and r["layer"] == layer]
        if M == 8192:
            Z_rows = [r for r in sp2 if r.get("k") == 40 and r["layer"] == layer and r.get("step") in ("S2a", "S2b")]
            N_rows = ([r for r in pm1 if r["setting_key"] == key and r.get("step") == "PM"] +
                      [r for r in pm2 if r.get("setting_key") == key and r.get("step") == "PMSEED"])
            calib = []
        else:
            Z_rows = ([r for r in rob if r.get("step") == "G" and r["layer"] == layer and r["setting"]["M"] == M
                       and r["setting"]["L"] == 16 and not r["setting"].get("overflow")] +
                      [r for r in pm2 if r.get("step") == "V0SEED" and r["layer"] == layer and r["setting"]["M"] == M])
            N_rows = [r for r in pm2 if r.get("setting_key") == key and r.get("step") in ("PM", "PMSEED")]
            calib = [r for r in pm2 if r.get("setting_key") == key and r.get("step") == "PMCAL"]
        Rs, Zs, Ns = by_seed(R_rows), by_seed(Z_rows), by_seed(N_rows)
        assert len(Rs) == 3 and len(Ns) == 3 and len(Zs) >= 2, (key, list(Rs), list(Zs), list(Ns))
        assert all(r["matched"] for r in Ns.values()) and all(r.get("matched", True) for r in Zs.values()), key
        rhos = sorted({r["rho"] for r in Ns.values()})
        assert len(rhos) == 1, (key, rhos)

        s = {"layer": layer, "M": M, "L": 16, "rho": rhos[0], "seeds_real": list(Rs), "seeds_equal_power": list(Zs),
             "seeds_power_matched": list(Ns),
             "fvu": {"power_matched": [r["fvu_eval"] for r in Ns.values()],
                     "equal_power": [r["fvu_eval"] for r in Zs.values()],
                     "target": Ns[min(Ns)]["fvu_target"], "real_mean": msd([r["fvu_eval"] for r in Rs.values()])[0]},
             "n_live_power_matched": [r["n_live"] for r in Ns.values()], "windows": {}, "top_windows": {}}
        classes = []
        for w in BULK:
            R, sdR = msd([r["gamma_fixed"][w] for r in Rs.values()])
            Z, sdZ = msd([r["gamma_fixed"][w] for r in Zs.values()])
            N, sdN = msd([r["gamma_fixed"][w] for r in Ns.values()])
            cls, f = window_class(N, R, Z)
            classes.append(cls)
            f_seed = {sd: (Z - r["gamma_fixed"][w]) / (Z - R) for sd, r in Ns.items()}  # each N seed, Z and R means
            f_all = [(z["gamma_fixed"][w] - r["gamma_fixed"][w]) / (z["gamma_fixed"][w] - R)
                     for z in Zs.values() for r in Ns.values()]  # every (Z seed, N seed) pair
            x = {"R": R, "R_sd": sdR, "Z": Z, "Z_sd": sdZ, "Z_per_seed": {sd: r["gamma_fixed"][w] for sd, r in Zs.items()},
                 "N": N, "N_sd": sdN, "N_per_seed": {sd: r["gamma_fixed"][w] for sd, r in Ns.items()},
                 "Z_minus_R": Z - R, "N_minus_R": N - R, "gap_over_margin": (Z - R) / MARGIN,
                 "closed_fraction": f, "closed_fraction_per_N_seed": f_seed,
                 "closed_fraction_sd_over_N_seeds": msd(list(f_seed.values()))[1],
                 "closed_fraction_range_all_seed_pairs": [min(f_all), max(f_all)], "class": cls}
            if M == 8192:  # the first run's numbers: N seed 0 against Z seed 0, as reported before
                n0, z0 = Ns[0]["gamma_fixed"][w], Zs[0]["gamma_fixed"][w]
                x["first_run"] = {"N_seed0": n0, "Z_seed0": z0, "closed_fraction": (z0 - n0) / (z0 - R),
                                  "class": window_class(n0, R, z0)[0]}
                x["closed_fraction_move_points"] = 100 * (f - x["first_run"]["closed_fraction"])
                x["closed_fraction_with_Z_seed0"] = (z0 - N) / (z0 - R)
            s["windows"][w] = x
        for w in ("(3, 50)", "(10, 150)"):
            s["top_windows"][w] = {"R": msd([r["gamma_fixed"][w] for r in Rs.values()])[0],
                                   "Z": msd([r["gamma_fixed"][w] for r in Zs.values()])[0],
                                   "N": msd([r["gamma_fixed"][w] for r in Ns.values()])[0],
                                   "N_sd": msd([r["gamma_fixed"][w] for r in Ns.values()])[1]}
        s["classes"] = dict(zip(BULK, classes))
        s["verdict"] = verdict(classes)
        if M == 8192:
            s["first_run_verdict"] = verdict([s["windows"][w]["first_run"]["class"] for w in BULK])
            assert s["first_run_verdict"] == FIRST_RUN[key], (key, s["first_run_verdict"])
            s["confirmed"] = s["verdict"] == FIRST_RUN[key]
        top = {sd: r["power_check"]["null"]["share_top1pct"] for sd, r in Ns.items()}
        s["manipulation_check"] = {"null_top1pct_per_seed": top, "null_top1pct_mean": msd(list(top.values()))[0],
                                   "real_top1pct": Ns[min(Ns)]["power_check"]["real"]["share_top1pct"],
                                   "null_top10_per_seed": {sd: r["power_check"]["null"]["share_top10"]
                                                           for sd, r in Ns.items()}}
        if calib:
            c = calib[0]
            s["rho0_run"] = {"rho": c["rho"], "fvu": c["fvu_eval"], "matched": c["matched"],
                             "gamma_fixed": {w: c["gamma_fixed"][w] for w in WINDOWS},
                             "null_top1pct": c["power_check"]["null"]["share_top1pct"],
                             "null_top10": c["power_check"]["null"]["share_top10"]}
            s["calibration_tried"] = json.load(open(PM2 / "pmatch_calibration.json"))[key]["tried"]
        dic = Ns[max(Ns)]["dictionary"]
        s["dictionary"] = {"atom_cloud_gamma_fixed": dic["atom_cloud_gamma_fixed"],
                           "equal_power_atom_cloud_gamma_fixed": dic["v0_atom_cloud_gamma_fixed"],
                           "share_top1pct": dic["power"]["share_top1pct"],
                           "target_share_top1pct": dic["power_target"]["share_top1pct"],
                           "power_moved_by_projection": dic["power_share_moved_by_projection"],
                           "iterations": dic["iterations"], "converged": dic["converged"],
                           "spearman_power_rq": json.load(open(PM2 / "pmatch" / f"{dic_tag(key)}.json"))["spearman_power_rq"],
                           "max_abs_log_power_err": dic["final_power_max_abs_log_err"]}
        out["settings"][key] = s

    # rebuilt reference dictionaries against the first run's (Part 2)
    reb = {}
    for tag in ("pmV1L13M8192ds0", "pmV1L0M8192ds0"):
        a, b = json.load(open(R2 / "pmatch" / f"{tag}.json")), json.load(open(PM2 / "pmatch" / f"{tag}.json"))
        na, nb = np.load(R2 / "pmatch" / f"{tag}_atoms.npz"), np.load(PM2 / "pmatch" / f"{tag}_atoms.npz")
        same = ["seed", "iterations", "converged", "final_power_max_abs_log_err", "power_share_moved_by_projection",
                "spearman_power_rq", "atom_cloud_gamma_fixed", "v0_atom_cloud_gamma_fixed", "power",
                "real_power_file_sha256", "generator_sha256"]
        reb[tag] = {"report_fields_equal_to_1e-9": {k: _close(a[k], b[k]) for k in same},
                    "max_abs_diff_atom_cloud_gamma": max(abs(a["atom_cloud_gamma_fixed"][w] - b["atom_cloud_gamma_fixed"][w])
                                                         for w in WINDOWS),
                    "rel_diff_atom_power": float(np.linalg.norm(na["power"] - nb["power"]) / np.linalg.norm(na["power"])),
                    "rel_diff_atom_rq": float(np.linalg.norm(na["rq"] - nb["rq"]) / np.linalg.norm(na["rq"])),
                    "dict_file_sha256_first": a["dict_file_sha256"], "dict_file_sha256_rebuilt": b["dict_file_sha256"],
                    "file_identical": a["dict_file_sha256"] == b["dict_file_sha256"]}
    out["rebuilt_dictionaries"] = reb

    # predictions of notes/prereg_ladder.md
    A, S13, S0 = out["settings"][KEYS[2]], out["settings"][KEYS[0]], out["settings"][KEYS[1]]
    cfA = [A["windows"][w]["closed_fraction"] for w in BULK]
    sds = [S["windows"][w]["N_sd"] for S in (S13, S0) for w in BULK]
    moves = [abs(S["windows"][w]["closed_fraction_move_points"]) for S in (S13, S0) for w in BULK]
    out["predictions"] = {
        "part1_verdict": {"predicted": "EXPLAINS (p .55; PARTLY .35; DOES NOT .10)", "observed": A["verdict"],
                          "right": A["verdict"] == "EXPLAINS"},
        "part1_closed_fraction_40_80": {"predicted": "40-80% in both windows", "observed": cfA,
                                        "right": all(0.40 <= f <= 0.80 for f in cfA)},
        "part1_top1pct_below_0.45": {"predicted": "null SAE top-1% share < .45",
                                     "observed": A["manipulation_check"]["null_top1pct_per_seed"],
                                     "right": all(v < 0.45 for v in A["manipulation_check"]["null_top1pct_per_seed"].values())},
        "part1_design_note": "rho = 0 gave FVU %.3f against target %.3f, so calibration raised rho to %.2f; the "
                             "matched null is not dense-share-free" % (A["rho0_run"]["fvu"], A["fvu"]["target"], A["rho"]),
        "part2_seed_sd_le_0.015": {"predicted": "N seed sd <= 0.015, both bulk windows, both layers", "observed": sds,
                                   "right": all(x <= 0.015 for x in sds)},
        "part2_verdicts_confirmed": {"predicted": "L13 PARTLY, L0 DOES NOT", "observed": [S13["verdict"], S0["verdict"]],
                                     "right": S13["confirmed"] and S0["confirmed"]},
        "part2_closed_fraction_moves_lt_5pts": {"predicted": "< 5 points", "observed_points": moves,
                                                "right": all(m < 5 for m in moves)},
    }
    p = PM2 / "pmatch2_summary.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)

    # small table for the paper
    short = {"EXPLAINS": "explains", "PARTLY EXPLAINS": "partly", "DOES NOT EXPLAIN": "does not"}
    lines = []
    for key in KEYS:
        s = out["settings"][key]
        cells = [LABEL[key], f"{s['rho']:.2f}"]
        for w in BULK:
            x = s["windows"][w]
            cells += [f"{x['R']:.3f}", f"{x['Z']:.3f}", f"{x['N']:.3f}$\\pm${x['N_sd']:.3f}", f"{x['N_minus_R']:.3f}",
                      f"{100 * x['closed_fraction']:.0f}$\\pm${100 * x['closed_fraction_sd_over_N_seeds']:.1f}"]
        cells.append(short[s["verdict"]])
        lines.append(" & ".join(cells) + r" \\")
    tex = ("% generated by experiments/r2_pmatch2_summary.py from results/r2_pm2/runs_pmatch2.jsonl and results/r2/"
           "runs_pmatch.jsonl, runs_sparse2.jsonl, runs_robust.jsonl, runs_real.jsonl\n"
           r"""\begin{tabular}{l r rrrrr rrrrr l}
\toprule
 & & \multicolumn{5}{c}{ranks 31--450} & \multicolumn{5}{c}{ranks 101--1000} & \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}
layer, setting & $\rho$ & $R$ & $Z$ & $N$ & $N-R$ & \% & $R$ & $Z$ & $N$ & $N-R$ & \% & verdict \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    t = ROOT / "paper" / "figures" / "tab_pmatch2_r2.tex"
    t.write_text(tex)
    print("wrote", t)
    for key in KEYS:
        s = out["settings"][key]
        print(key, s["verdict"], {w: {k: round(s["windows"][w][k], 4) for k in
                                      ("R", "Z", "N", "N_sd", "N_minus_R", "closed_fraction",
                                       "closed_fraction_sd_over_N_seeds")} for w in BULK},
              "top1%", {k: round(v, 3) for k, v in s["manipulation_check"]["null_top1pct_per_seed"].items()})
    print(json.dumps({k: v["right"] for k, v in out["predictions"].items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
