"""Readings of the ladder job (experiments/r2_job_ladder.py): per dataset and bulk window the real SAEs' gamma R, the
equal-power null's Z (V0) and the power-matched null's N (V1), and the rule below, which compares each random rung with
its trained counterpart. Works on partial results (a missing piece makes a comparison "uninformative"). Writes
<dir>/ladder_summary.json.

Values (gamma_fixed: fixed-n unit scoring, r2_common.score; bulk windows ranks 31-450 and 101-1000):
  R    mean over the three real k = 40 SAEs (seeds 0, 1, 2)
  Z    mean over the accepted V0 SAEs (seeds 0 and 1 at the accepted rho)
  N    the accepted V1 SAE (seed 0)
  G = Z - R (gap),  Rem = N - R (remainder),  closed fraction (Z - N) / (Z - R).
Trained counterparts (Gemma-2-2B, from the reference directory): R = job 1's step Q1 real k = 40 rows
(runs_real.jsonl), Z = sparse2's S2a/S2b k = 40 rows (runs_sparse2.jsonl; mean of seeds 0 and 1, seed 0 alone also
reported, which is the comparator r2_summary.py's pmatch section uses), N = pmatch's PM row of setting
PM|V1|L<layer>|M8192_L16|ds0 (runs_pmatch.jsonl). Pairs: embR vs embT, randL0 vs Gemma layer 0, randL13 vs Gemma layer 13.
Rule per pair, fixed before any ladder run:
  uninformative  the trained counterpart's Rem is below 0.10 in either bulk window, or a needed null (V0 or V1 of
                 either side) is unmatched or missing, or a real mean is missing;
  absent         otherwise, the random rung's Rem < 0.10 in both bulk windows;
  persists       the random rung's Rem >= 0.70 x the trained Rem in both bulk windows;
  partial        otherwise.
Reported separately: "no gap" when the random rung's G is below 0.10 in both bulk windows.
  python experiments/r2_ladder_summary.py [--dir results/r2] [--ref_dir results/r2]
  python experiments/r2_ladder_summary.py --dir cache/ladder_smoke_out      (smoke output; reference from results/r2)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from r2_summary import load  # noqa: E402  (runs_*.jsonl plus rows/*/*.json, torn lines skipped)

WINDOWS = [str(w) for w in [(3, 50), (10, 150), (30, 450), (100, 1000)]]
BULK = WINDOWS[2:]
REM_MIN, PERSIST_FRAC, GAP_MIN = 0.10, 0.70, 0.10
PAIRS = [("embR", "embT"), ("randL0", "gemma_L0"), ("randL13", "gemma_L13")]
RUNGS = ["embT", "embR", "randL0", "randL13"]


def mean_g(rs):
    return {w: float(np.mean([r["gamma_fixed"][w] for r in rs])) for w in WINDOWS} if rs else None


def entry(real, v0, v1, extra=None):
    """One dataset: values per window, FVUs, matched flags, live counts, and what is missing."""
    R, Z = mean_g(real), mean_g(v0)
    N = v1["gamma_fixed"] if v1 else None
    miss = [k for k, v in (("real (3 seeds)", len(real) == 3), ("V0", bool(v0)), ("V1", v1 is not None)) if not v]
    win = {}
    for w in WINDOWS:
        o = {"R": R[w] if R else None, "Z": Z[w] if Z else None, "N": N[w] if N else None}
        o["G"] = o["Z"] - o["R"] if R and Z else None
        o["Rem"] = o["N"] - o["R"] if R and N else None
        o["closed_fraction"] = ((o["Z"] - o["N"]) / (o["Z"] - o["R"]) if R and Z and N and o["Z"] != o["R"]
                                else None)
        win[w] = o
    allr = list(real) + list(v0) + ([v1] if v1 else [])
    return {"windows": win, "missing": miss,
            "fvu": {"real_mean": float(np.mean([r["fvu_eval"] for r in real])) if real else None,
                    "real": [r["fvu_eval"] for r in real], "v0": [r["fvu_eval"] for r in v0],
                    "v1": v1["fvu_eval"] if v1 else None,
                    "target_v0": v0[0].get("fvu_target") if v0 else None,
                    "target_v1": v1.get("fvu_target") if v1 else None},
            "matched": {"v0": bool(v0) and all(bool(r.get("matched")) for r in v0),
                        "v1": bool(v1) and bool(v1.get("matched"))},
            "rho": {"v0": v0[0].get("rho") if v0 else None, "v1": v1.get("rho") if v1 else None},
            "seeds": {"real": sorted(r["seed"] for r in real), "v0": sorted(r["seed"] for r in v0)},
            "n_live": {"real": [r["n_live"] for r in real], "v0": [r["n_live"] for r in v0],
                       "v1": v1["n_live"] if v1 else None},
            "fixed_n_short": any(r.get("fixed_n_short") for r in allr),
            "fixed_n_short_runs": [r["run"] for r in allr if r.get("fixed_n_short")],
            **({"extra": extra} if extra is not None else {})}


def ladder_entries(rows):
    lad = [r for r in rows if r["job"] == "ladder"]
    out = {}
    for rung in RUNGS:
        rs = [r for r in lad if r.get("rung") == rung]
        if not rs:
            continue
        real = sorted((r for r in rs if r.get("step") == "real" and r.get("k") == 40), key=lambda r: r["seed"])
        v0 = sorted((r for r in rs if r.get("step") == "v0"), key=lambda r: r["seed"])
        v1s = [r for r in rs if r.get("step") == "v1"]
        extra = {f"k{r['k']}": {"gamma_fixed": r["gamma_fixed"], "fvu": r["fvu_eval"], "n_live": r["n_live"],
                                "l0": r["l0_eval"], "fixed_n_short": r.get("fixed_n_short")}
                 for r in rs if r.get("step") == "real_k"}
        e = entry(real, v0, v1s[0] if v1s else None, extra or None)
        e |= {"dataset": rs[0].get("dataset"), "description": rs[0].get("dataset_description"),
              "n_rows": len(rs), "calibration_rows": {
                  "v0_cal": sorted((r["rho"], r["fvu_eval"]) for r in rs if r.get("step") == "v0_cal"),
                  "v1_cal": sorted((r["rho"], r["fvu_eval"]) for r in rs if r.get("step") == "v1_cal")}}
        pw = [r for r in rs if r.get("step") == "power"]
        if pw:
            e["real_seed0_power"] = {k: pw[0].get(k) for k in ("power", "n_live", "power_fvu", "rq_median_log10")}
        if v1s and v1s[0].get("power_check"):
            pc = v1s[0]["power_check"]
            e["v1_power_check"] = {"null_share_top1pct": pc["null"]["share_top1pct"],
                                   "real_share_top1pct": pc["real"]["share_top1pct"],
                                   "null_ess_frac": pc["null"]["ess_frac"], "real_ess_frac": pc["real"]["ess_frac"]}
        out[rung] = e
    return out


def gemma_entries(rows):
    out = {}
    for L in (0, 13):
        real = sorted((r for r in rows if r["job"] == "real" and r.get("step") == "Q1" and r.get("cond") == "real"
                       and r.get("layer") == L and r.get("k") == 40 and not r.get("whitened")), key=lambda r: r["seed"])
        v0 = sorted((r for r in rows if r["job"] == "sparse2" and r.get("step") in ("S2a", "S2b")
                     and r.get("layer") == L and r.get("k") == 40), key=lambda r: r["seed"])
        v1 = [r for r in rows if r["job"] == "pmatch" and r.get("step") == "PM"
              and r.get("setting_key") == f"PM|V1|L{L}|M8192_L16|ds0"]
        if not (real or v0 or v1):
            continue
        e = entry(real, v0, v1[0] if v1 else None)
        z0 = [r for r in v0 if r["seed"] == 0]
        e["v0_seed0_gamma_fixed"] = z0[0]["gamma_fixed"] if z0 else None
        e["dataset"], e["description"] = f"gemma-2-2b_L{L}", f"trained Gemma-2-2B resid_post, block {L}"
        out[f"gemma_L{L}"] = e
    return out


def compare(rand, trained):
    """The rule in the docstring for one (random rung, trained counterpart) pair."""
    def rem(e, w):
        return e["windows"][w]["Rem"] if e else None
    why = []
    for nm, e in (("random", rand), ("trained", trained)):
        if e is None:
            why.append(f"{nm} rung missing")
            continue
        why += [f"{nm}: {m} missing" for m in e["missing"]]
        why += [f"{nm}: {k} unmatched" for k in ("v0", "v1") if not e["matched"][k] and k.upper() not in e["missing"]]
    tr = {w: rem(trained, w) for w in BULK}
    rr = {w: rem(rand, w) for w in BULK}
    if not why and any(tr[w] is None or tr[w] < REM_MIN for w in BULK):
        why.append(f"trained Rem below {REM_MIN} in a bulk window")
    g = {w: rand["windows"][w]["G"] if rand else None for w in BULK}
    no_gap = all(g[w] is not None and g[w] < GAP_MIN for w in BULK) if rand else None
    if why:
        v = "uninformative"
    elif all(rr[w] < REM_MIN for w in BULK):
        v = "absent"
    elif all(rr[w] >= PERSIST_FRAC * tr[w] for w in BULK):
        v = "persists"
    else:
        v = "partial"
    return {"verdict": v, "reasons_uninformative": why, "rem_random": rr, "rem_trained": tr,
            "rem_ratio": {w: rr[w] / tr[w] if rr[w] is not None and tr[w] else None for w in BULK},
            "gap_random": g, "gap_trained": {w: trained["windows"][w]["G"] if trained else None for w in BULK},
            "no_gap": no_gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "r2"), help="directory with runs_ladder.jsonl")
    ap.add_argument("--ref_dir", default=str(ROOT / "results" / "r2"), help="directory with the Gemma reference rows")
    a = ap.parse_args()
    d, ref = Path(a.dir), Path(a.ref_dir)
    lad = ladder_entries(load(d))
    gem = gemma_entries(load(ref) if ref.resolve() != d.resolve() else [r for r in load(d) if r["job"] != "ladder"])
    ents = {**lad, **gem}
    comp = {f"{x} vs {y}": compare(ents.get(x), ents.get(y)) for x, y in PAIRS}
    out = {"rule": {"rem_min": REM_MIN, "persist_frac": PERSIST_FRAC, "gap_min": GAP_MIN, "bulk_windows": BULK,
                    "value": "gamma_fixed"},
           "dir": str(d), "ref_dir": str(ref), "datasets": ents, "comparisons": comp,
           "verdicts": {k: v["verdict"] for k, v in comp.items()},
           "no_gap": {k: v["no_gap"] for k, v in comp.items()}}
    status = d / "ladder_status.json"
    if status.exists():
        out["job_status"] = {k: {"ok": v.get("ok"), "steps": v.get("steps"), "notes": v.get("notes"),
                                 "score_errors": v.get("score_errors")} for k, v in json.load(open(status)).items()}
    for k, e in ents.items():
        b = " / ".join("R %.3f Z %.3f N %.3f" % (e["windows"][w]["R"] or np.nan, e["windows"][w]["Z"] or np.nan,
                                                 e["windows"][w]["N"] or np.nan) for w in BULK)
        print(f"{k:<10} {b}  matched {e['matched']}  missing {e['missing']}")
    for k, v in comp.items():
        print(f"{k:<22} {v['verdict']:<14} no gap {v['no_gap']}  {'; '.join(v['reasons_uninformative'])}")
    json.dump(out, open(d / "ladder_summary.json", "w"), indent=1, default=float)
    print(f"-> {d / 'ladder_summary.json'}")


if __name__ == "__main__":
    main()
