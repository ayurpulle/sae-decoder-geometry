"""Independent recomputation of the round-2 resynthesis control (Q3, reading iv): does G (Gaussian directions with
the covariance of Gemma Scope's layer-12 decoder rows) retain D's (permuted Gemma Scope rows) decline of gamma with k?
Reads only results/r2/runs_resynth.jsonl (84 rows; 2 variants x 3 data seeds x 2 SAE seeds x 7 k). It does not import
experiments/r2_summary.py; it compares its numbers with results/r2/summary.json["iv_G_retains_D_decline"] at the end.

ANALYSIS CHOICES (fixed before looking at the numbers)
- Per run (variant X, data seed j, SAE seed s): OLS slope of the value against log2 k ("per doubling"), in gamma units
  (gamma_*) and raw decoder-slope units (dec_*), both scoring modes: fixed-n (8,192 live latents, mean of 4 draws;
  the pre-registered primary) and full live count. All four windows (ranks 4-50, 11-150 primary, 31-450, 101-1000).
- k ranges: 8-128 (all seven k; pre-registered) and 16-128 (post hoc, chosen after round 1 because of the k=8 step).
  A leave-one-k-out range over 8-128 is also reported, post hoc.
- Replicates: the 6 (j, s) pairs. Reported: mean, sd (ddof=1), min, max of s_D and s_G over the 6; per-draw values
  s_X(j) = mean over s, and their sd over the 3 draws (the pre-registered draw-level spread).
- Retention (pre-registered estimator): s_G / s_D with s_X the mean over all 6 replicates (equal to the mean over
  draws of the per-draw means, since the design is balanced). Per-draw retention s_G(j)/s_D(j), paired by data seed
  (D and G share code draws at a given j); per-replicate retention paired by (j, s).
- Interval: two-level paired bootstrap, 20,000 resamples (rng seed 0): resample the 3 data seeds with replacement,
  then within each the 2 SAE seeds with replacement, carrying D and G of the same (j, s) together; percentile 95%
  interval of the ratio of means. With 3 draws this interval is coarse, so the min-max of the per-draw retention is
  reported beside it as the seed-spread interval.
- Verdict (pre-registered): keeps if retention >= 0.75 in both units, removes if < 0.5 in both, partial otherwise;
  "firm" if all three per-draw verdicts equal the pooled one. Window rule: supported if keeps in the primary window and
  at least two of the other three, contradicted if removes there, otherwise mixed. Decided on fixed-n; full-live is
  reported only.
- Also reported: mean +- sd over the 6 replicates of gamma and raw slope at each k (for the figure), FVU per k, live
  counts, each dataset's Sigma slope (ill-conditioned flag if shallower than -0.3 in ranks 11-150), and the data-seed-0
  SAE-seed-0 reproduction check against results/phase3/control.json (fixed-n gamma, primary window, within 0.02).
Usage: python experiments/r2_resynth_analysis.py   -> results/r2/resynth_analysis.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "r2" / "runs_resynth.jsonl"
OUT = ROOT / "results" / "r2" / "resynth_analysis.json"
SUMMARY = ROOT / "results" / "r2" / "summary.json"
CONTROL = ROOT / "results" / "phase3" / "control.json"
WINDOWS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WNAME = {"(3, 50)": "4-50", "(10, 150)": "11-150", "(30, 450)": "31-450", "(100, 1000)": "101-1000"}
PRIMARY = "(10, 150)"
KS = [8, 16, 24, 32, 48, 64, 128]
RANGES = {"k8_128": KS, "k16_128_posthoc": [k for k in KS if k >= 16]}
UNITS = {"gamma": "gamma", "raw": "dec"}
MODES = ["fixed", "full"]
KEEP, REMOVE, ILL = 0.75, 0.5, -0.3
NBOOT = 20_000


def load():
    rows = {}
    for line in open(SRC):
        r = json.loads(line)
        rows[r["run"]] = r
    rows = list(rows.values())
    assert len(rows) == 84, len(rows)
    cube = {}
    for r in rows:
        assert r["width"] == 16384 and r["n_fix_used"] == 8192 and not r["fixed_n_short"]
        cube[r["kind"], r["data_seed"], r["seed"], r["k"]] = r
    assert len(cube) == 84
    return rows, cube


DS, SS = [0, 1, 2], [0, 1]
REPS = [(j, s) for j in DS for s in SS]


def v(r, unit, mode, w):
    return r[f"{UNITS[unit]}_{mode}"][w]


def slope(ks, ys):
    return float(np.polyfit(np.log2(ks), ys, 1)[0])


def verdict(rg, rr):
    if rg >= KEEP and rr >= KEEP:
        return "keeps"
    if rg < REMOVE and rr < REMOVE:
        return "removes"
    return "partial"


def window_rule(vs):
    others = [vs[w] for w in WINDOWS if w != PRIMARY]
    for lab, out in (("keeps", "supported"), ("removes", "contradicted")):
        if vs[PRIMARY] == lab and sum(o == lab for o in others) >= 2:
            return out
    return "mixed"


def stats(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max())}


def boot_ratio(sD, sG, rng):
    """sD, sG: arrays [3 data seeds, 2 SAE seeds]. Two-level paired bootstrap of mean(sG)/mean(sD)."""
    jj = rng.integers(0, 3, size=(NBOOT, 3))
    ss = rng.integers(0, 2, size=(NBOOT, 3, 2))
    d = sD[jj[:, :, None], ss].mean(axis=(1, 2))
    g = sG[jj[:, :, None], ss].mean(axis=(1, 2))
    r = g / d
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def analyse_range(cube, ks, rng, with_boot=True):
    out = {}
    for w in WINDOWS:
        res = {}
        for mode in MODES:
            m = {}
            S = {}
            for u in UNITS:
                for X in "DG":
                    S[X, u] = np.array([[slope(ks, [v(cube[X, j, s, k], u, mode, w) for k in ks]) for s in SS]
                                        for j in DS])
            for u in UNITS:
                sD, sG = S["D", u], S["G", u]
                per_draw = {str(j): float(sG[i].mean() / sD[i].mean()) for i, j in enumerate(DS)}
                per_rep = {f"ds{j}_s{s}": float(sG[i, s] / sD[i, s]) for i, j in enumerate(DS) for s in SS}
                m[u] = {
                    "s_D_per_replicate": {f"ds{j}_s{s}": float(sD[i, s]) for i, j in enumerate(DS) for s in SS},
                    "s_G_per_replicate": {f"ds{j}_s{s}": float(sG[i, s]) for i, j in enumerate(DS) for s in SS},
                    "s_D": stats(sD.ravel()), "s_G": stats(sG.ravel()),
                    "s_D_per_draw": {str(j): float(sD[i].mean()) for i, j in enumerate(DS)},
                    "s_G_per_draw": {str(j): float(sG[i].mean()) for i, j in enumerate(DS)},
                    "s_D_draw_sd": float(sD.mean(1).std(ddof=1)), "s_G_draw_sd": float(sG.mean(1).std(ddof=1)),
                    "retained": float(sG.mean() / sD.mean()),
                    "retained_per_draw": per_draw,
                    "retained_per_draw_range": [min(per_draw.values()), max(per_draw.values())],
                    "retained_per_replicate": per_rep,
                    "retained_per_replicate_range": [min(per_rep.values()), max(per_rep.values())],
                }
                if with_boot:
                    m[u]["retained_boot95"] = boot_ratio(sD, sG, rng)
            vd = verdict(m["gamma"]["retained"], m["raw"]["retained"])
            pdv = {j: verdict(m["gamma"]["retained_per_draw"][j], m["raw"]["retained_per_draw"][j])
                   for j in m["gamma"]["retained_per_draw"]}
            m["verdict"] = vd
            m["per_draw_verdicts"] = pdv
            m["firm"] = all(x == vd for x in pdv.values())
            res[mode] = m
        out[w] = res
    vs = {w: out[w]["fixed"]["verdict"] for w in WINDOWS}
    return {"k": ks, "windows": out, "window_verdicts_fixed": vs, "window_rule_fixed": window_rule(vs),
            "window_verdicts_full": {w: out[w]["full"]["verdict"] for w in WINDOWS},
            "window_rule_full": window_rule({w: out[w]["full"]["verdict"] for w in WINDOWS})}


def curves(cube):
    out = {}
    for X in "DG":
        o = {}
        for w in WINDOWS:
            o[w] = {}
            for mode in MODES:
                for u in UNITS:
                    vals = [[v(cube[X, j, s, k], u, mode, w) for (j, s) in REPS] for k in KS]
                    o[w][f"{u}_{mode}"] = {"mean": [float(np.mean(x)) for x in vals],
                                           "sd": [float(np.std(x, ddof=1)) for x in vals],
                                           "per_replicate": {f"ds{j}_s{s}": [float(vals[i][n]) for i in range(len(KS))]
                                                             for n, (j, s) in enumerate(REPS)}}
        o["fvu"] = {"mean": [float(np.mean([cube[X, j, s, k]["fvu_eval"] for j, s in REPS])) for k in KS],
                    "sd": [float(np.std([cube[X, j, s, k]["fvu_eval"] for j, s in REPS], ddof=1)) for k in KS]}
        o["n_live"] = {"min": [int(min(cube[X, j, s, k]["n_live"] for j, s in REPS)) for k in KS],
                       "max": [int(max(cube[X, j, s, k]["n_live"] for j, s in REPS)) for k in KS]}
        o["sigma_slope"] = {str(j): cube[X, j, 0, 8]["sigma_slope"] for j in DS}
        out[X] = o
    return out


def leave_one_k_out(cube):
    out = {}
    for w in WINDOWS:
        o = {}
        for drop in KS:
            ks = [k for k in KS if k != drop]
            r = {}
            for u in UNITS:
                sD = np.mean([slope(ks, [v(cube["D", j, s, k], u, "fixed", w) for k in ks]) for j, s in REPS])
                sG = np.mean([slope(ks, [v(cube["G", j, s, k], u, "fixed", w) for k in ks]) for j, s in REPS])
                r[u] = float(sG / sD)
            o[str(drop)] = r
        allg = [x["gamma"] for x in o.values()]
        allr = [x["raw"] for x in o.values()]
        out[w] = {"by_dropped_k": o, "gamma_range": [min(allg), max(allg)], "raw_range": [min(allr), max(allr)]}
    return out


def checks(cube):
    c = {}
    ill = sorted({f"{X}_ds{j}" for X in "DG" for j in DS if cube[X, j, 0, 8]["sigma_slope"][PRIMARY] > ILL})
    c["ill_conditioned"] = ill
    for X in "DG":  # Sigma is per dataset: every row of a dataset must carry the same slope
        for j in DS:
            vals = {cube[X, j, s, k]["sigma_slope"][PRIMARY] for s in SS for k in KS}
            assert len(vals) == 1, (X, j, vals)
    if CONTROL.exists():
        prev = json.load(open(CONTROL))["rows"]
        for X in "DG":
            old = {r["k"]: r["gamma_fixed"][PRIMARY] for r in prev
                   if r["data"] == f"rs12-{X}" and r["seed"] == 0 and r.get("gamma_fixed")}
            diffs = {str(k): cube[X, 0, 0, k]["gamma_fixed"][PRIMARY] - old[k] for k in sorted(old) if k in KS}
            c[f"{X}_ds0_s0_vs_control_json"] = {"diffs": diffs, "max_abs": max(map(abs, diffs.values())),
                                                "pass_0.02": max(map(abs, diffs.values())) <= 0.02}
    return c


def compare_summary(res):
    """Compare with r2_summary.py's reading iv (k 8-128)."""
    if not SUMMARY.exists():
        return {"note": "summary.json missing"}
    iv = json.load(open(SUMMARY))["iv_G_retains_D_decline"]
    a = res["k8_128"]["windows"]
    diffs = []
    for w in WINDOWS:
        for mode in MODES:
            sm = iv["windows"][w][mode]
            mine = a[w][mode]
            for u in UNITS:
                diffs.append(abs(sm[f"retained_{u}"] - mine[u]["retained"]))
                for X in "DG":
                    diffs.append(abs(sm["pooled"][X][u] - mine[u][f"s_{X}"]["mean"]))
                    diffs.append(abs(sm["draw_sd"][X][u] - mine[u][f"s_{X}_draw_sd"]))
                    for j in DS:
                        diffs.append(abs(sm["s"][f"{X}_ds{j}"][u] - mine[u][f"s_{X}_per_draw"][str(j)]))
                for j in DS:
                    diffs.append(abs(sm["per_draw"][str(j)][f"retained_{u}"] - mine[u]["retained_per_draw"][str(j)]))
            assert sm["verdict"] == mine["verdict"], (w, mode, sm["verdict"], mine["verdict"])
            assert sm["firm"] == mine["firm"], (w, mode)
    rule = {"supported": "supports", "contradicted": "contradicts"}.get(res["k8_128"]["window_rule_fixed"], "mixed")
    return {"n_values_compared": len(diffs), "max_abs_diff": float(max(diffs)),
            "verdicts_agree": True, "window_rule_agree": rule == iv["verdict"],
            "summary_verdict": iv["verdict"], "summary_window_verdicts": iv["window_verdicts"],
            "summary_checks": iv.get("checks")}


def main():
    rows, cube = load()
    rng = np.random.default_rng(0)
    res = {name: analyse_range(cube, ks, rng) for name, ks in RANGES.items()}
    out = {"source": str(SRC.relative_to(ROOT)), "n_rows": len(rows), "replicates": [f"ds{j}_s{s}" for j, s in REPS],
           "k": KS, "windows": {w: WNAME[w] for w in WINDOWS}, "primary": PRIMARY,
           "thresholds": {"keeps": KEEP, "removes": REMOVE}, "n_boot": NBOOT, **res,
           "leave_one_k_out_posthoc": leave_one_k_out(cube), "curves": curves(cube), "checks": checks(cube)}
    out["agreement_with_summary_json"] = compare_summary(res)
    json.dump(out, open(OUT, "w"), indent=1)
    for name in RANGES:
        print(f"== {name}  rule(fixed) {res[name]['window_rule_fixed']}  rule(full) {res[name]['window_rule_full']}")
        for w in WINDOWS:
            for mode in MODES:
                m = res[name]["windows"][w][mode]
                print(f"  {WNAME[w]:>8} {mode:5} " + "  ".join(
                    f"{u}: D {m[u]['s_D']['mean']:+.4f}±{m[u]['s_D']['sd']:.4f} G {m[u]['s_G']['mean']:+.4f}"
                    f"±{m[u]['s_G']['sd']:.4f} ret {m[u]['retained']:.3f} [{m[u]['retained_boot95'][0]:.2f},"
                    f"{m[u]['retained_boot95'][1]:.2f}] draws {m[u]['retained_per_draw_range'][0]:.2f}-"
                    f"{m[u]['retained_per_draw_range'][1]:.2f}" for u in UNITS) + f"  {m['verdict']} firm={m['firm']}")
    print("loko", {WNAME[w]: [round(x, 2) for x in o["gamma_range"]] for w, o in out["leave_one_k_out_posthoc"].items()})
    print("checks", json.dumps(out["checks"]))
    print("agreement", json.dumps(out["agreement_with_summary_json"]))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
