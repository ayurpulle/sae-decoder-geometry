"""Pre-registered readings (i)-(v) of the round-2 GPU jobs, from results/r2/runs_real.jsonl, runs_resynth.jsonl and
runs_sparse2.jsonl. The rules are fixed in the docstrings of r2_job_real.py ((i)-(iii), (v)), r2_job_resynth.py ((iv))
and r2_job_sparse2.py ((v) with the codable sparse null in place of the first one); the "robust" section implements
the rules of r2_job_robust.py (runs_robust.jsonl) and the "pmatch" section those of r2_job_pmatch.py (runs_pmatch.jsonl);
this file only implements them. Works on partial results: every reading lists the runs it used and is "insufficient" when a fit
cannot be made. Writes <dir>/summary.json.
Usage: python experiments/r2_summary.py [--dir results/r2]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [str(w) for w in [(3, 50), (10, 150), (30, 450), (100, 1000)]]
PRIMARY = str((10, 150))
TOP, BULK = WINDOWS[:2], WINDOWS[2:]
K_I = [16, 24, 32, 48, 64, 128]          # (i): fit range k = 16-128
K_IV = [8, 16, 24, 32, 48, 64, 128]      # (iv): fit range k = 8-128
Q1_K, Q1_INTERIOR, Q1_ENDS = 40, {6, 9, 13, 17}, {0, 3, 21, 24}
Q4_K, DELTA_TOL = [16, 40, 128], 0.05
KEEP, REMOVE, ILL = 0.75, 0.5, -0.3
UNITS = {"gamma": "gamma", "raw": "dec"}


def load(d):
    rows = {}
    for f in sorted(Path(d).glob("runs_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # torn last line of a job that is still writing
                continue
            rows[(r["job"], r["run"])] = r
    for f in sorted(Path(d).glob("rows/*/*.json")):  # rows written but not yet appended (or a lost jsonl)
        r = json.loads(f.read_text())
        rows.setdefault((r["job"], r["run"]), r)
    return list(rows.values())


def val(r, unit, mode, w):
    return r[f"{UNITS[unit]}_{mode}"][w]


def fit(ks, ys):
    return float(np.polyfit(np.log2(ks), ys, 1)[0])


def msd(xs):
    xs = np.asarray(xs, float)
    sd = float(xs.std(ddof=1)) if len(xs) > 1 else float("nan")
    return float(xs.mean()), sd, sd / np.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def window_rule(v, yes, no):
    """yes/no if the primary window and at least two of the other three say so, else mixed."""
    others = [v[w] for w in WINDOWS if w != PRIMARY and w in v]
    for lab in (yes, no):
        if v.get(PRIMARY) == lab and sum(o == lab for o in others) >= 2:
            return lab
    return "mixed"


def by_seed(rows, ks):
    """{seed: {k: row}} keeping only seeds that have every k in ks."""
    d = defaultdict(dict)
    for r in rows:
        d[r["seed"]][r["k"]] = r
    return {s: m for s, m in sorted(d.items()) if all(k in m for k in ks)}


def kset(rows, wanted):
    have = {r["k"] for r in rows}
    return [k for k in wanted if k in have]


def seed_slopes(seeds, ks, unit, mode, w):
    return [fit(ks, [val(m[k], unit, mode, w) for k in ks]) for m in seeds.values()]


# (i) -------------------------------------------------------------------------------------------------------------
def reading_i(rows, null="gaussian", job="real", layer=13, kfit=K_I):
    q = [r for r in rows if r["job"] == job and r["layer"] == layer and not r["whitened"] and r["k"] in kfit + [8]]
    R, G = [r for r in q if r["cond"] == "real"], [r for r in q if r["cond"] == null]
    ks = [k for k in kset(R, kfit) if k in kset(G, kfit)]
    sR, sG = by_seed(R, ks), by_seed(G, ks)
    out = {"k_fit": ks, "seeds_real": list(sR), "seeds_gaussian": list(sG), "windows": {}}
    if len(ks) < 3 or not sR or not sG:
        out["verdict"] = "insufficient"
        return out
    verdicts = {}
    for w in WINDOWS:
        res = {}
        for mode in ("fixed", "full"):
            res[mode] = {}
            for unit in UNITS:
                a, b = seed_slopes(sR, ks, unit, mode, w), seed_slopes(sG, ks, unit, mode, w)
                (mR, sdR, seR), (mG, sdG, seG) = msd(a), msd(b)
                loo = []
                for drop in ks:
                    kk = [k for k in ks if k != drop]
                    lr = np.mean(seed_slopes(sR, kk, unit, mode, w))
                    lg = np.mean(seed_slopes(sG, kk, unit, mode, w))
                    loo.append({"drop_k": drop, "s_real": float(lr), "s_gauss": float(lg), "ratio": float(lg / lr)})
                se_c = float(np.hypot(seR, seG))
                # Signs are taken relative to real's slope: gamma falls with k (s < 0) while the raw decoder slope
                # rises toward 0 (s > 0); a fixed "< 0" test would misread every raw-unit decline as flat.
                sgn = 1.0 if mR >= 0 else -1.0
                steady = bool(sgn * mG > 0 and abs(mG) > 2 * seG and all(sgn * x["s_gauss"] > 0 for x in loo))
                rat = mG / mR
                x = {"s_real": mR, "s_real_sd": sdR, "s_gauss": mG, "s_gauss_sd": sdG, "se_combined": se_c,
                     "ratio": rat, "loo": loo, "min_loo_ratio": min(x["ratio"] for x in loo), "steady_gauss": steady,
                     "per_seed_real": a, "per_seed_gauss": b}
                x["supports"] = bool(steady and rat >= KEEP and x["min_loo_ratio"] >= REMOVE)
                x["contradicts"] = bool((not steady) or (rat < REMOVE and sgn * (mR - mG) > 2 * se_c))
                for c, s in (("real", sR), ("gauss", sG)):  # reported: the k=8->16 step and s over 8-128
                    if all(8 in m for m in s.values()):
                        x[f"step_8_16_{c}"] = float(np.mean([val(m[16], unit, mode, w) - val(m[8], unit, mode, w)
                                                             for m in s.values()]))
                        x[f"s_8_128_{c}"] = float(np.mean(seed_slopes(s, [8] + ks, unit, mode, w)))
                res[mode][unit] = x
        f = res["fixed"]
        res["verdict"] = ("supports" if f["gamma"]["supports"] and f["raw"]["supports"] else
                          "contradicts" if f["gamma"]["contradicts"] and f["raw"]["contradicts"] else "partial")
        verdicts[w] = res["verdict"]
        out["windows"][w] = res
    out["window_verdicts"] = verdicts
    out["verdict"] = window_rule(verdicts, "supports", "contradicts")
    return out


# (ii) ------------------------------------------------------------------------------------------------------------
def reading_ii(rows, null="gaussian", job="real", K=None, interior_set=Q1_INTERIOR, ends_set=Q1_ENDS,
               steps=("Q1", "Q2", "Q5"), skip_layers=(12,)):
    if K is None:
        q1k = sorted({r["k"] for r in rows if r["job"] == job and r["step"] == "Q1"})
        K = q1k[0] if len(q1k) == 1 else Q1_K  # 40 on the pod; the smoke test uses another k
    q = [r for r in rows if r["job"] == job and r["k"] == K and not r["whitened"]
         and r["step"] in steps and r["layer"] not in skip_layers]  # layer 13 at k = 40 may carry step Q2 if Q2 ever trains it: same run either way
    layers = sorted({r["layer"] for r in q})
    out = {"k": K, "layers": layers, "windows": {}}
    per = {}
    for L in layers:
        R = [r for r in q if r["layer"] == L and r["cond"] == "real"]
        G = [r for r in q if r["layer"] == L and r["cond"] == null]
        if R and G:
            per[L] = (R, G)
    interior, ends = [L for L in per if L in interior_set], [L for L in per if L in ends_set]
    if not interior or not ends:
        out["verdict"] = "insufficient"
        out["layers_with_both"] = list(per)
        return out
    verdicts = {}
    for w in WINDOWS:
        res = {}
        for mode in ("fixed", "full"):
            res[mode] = {}
            for unit in UNITS:
                tab = {}
                for L, (R, G) in per.items():
                    (mR, _, seR), (mG, _, seG) = msd([val(r, unit, mode, w) for r in R]), msd([val(r, unit, mode, w) for r in G])
                    tab[L] = {"real": mR, "gauss": mG, "D": mG - mR, "se_D": float(np.hypot(seR, seG)),
                              "n_real": len(R), "n_gauss": len(G)}
                res[mode][unit] = {"by_layer": {str(L): v for L, v in tab.items()}}
                if unit == "raw":
                    D = {L: v["D"] for L, v in tab.items()}
                    se = {L: v["se_D"] for L, v in tab.items()}
                    if w in TOP:
                        Ls = max(D, key=D.get)
                        sup = Ls in interior and all(D[Ls] - D[e] > 2 * np.hypot(se[Ls], se[e]) for e in ends)
                        con = max(D[e] for e in ends) >= max(D[i] for i in interior)
                        res[mode][unit]["sub_claim"] = "a: mid-stack maximum of D"
                        res[mode][unit]["argmax_layer"] = Ls
                    else:
                        sup = all(-D[L] > 2 * se[L] for L in D)
                        con = any(D[L] > 2 * se[L] for L in D)
                        res[mode][unit]["sub_claim"] = "b: real flatter than null at every layer"
                    res[mode][unit]["verdict"] = "supports" if sup else "contradicts" if con else "partial"
        res["verdict"] = res["fixed"]["raw"]["verdict"]
        verdicts[w] = res["verdict"]
        out["windows"][w] = res
    out["window_verdicts"] = verdicts
    out["verdict"] = window_rule(verdicts, "supports", "contradicts")
    out["complete"] = sorted(per) == sorted(interior_set | ends_set)
    return out


# (v) -------------------------------------------------------------------------------------------------------------
def _D(ii, w):
    t = ii["windows"][w]["fixed"]["raw"]["by_layer"]
    return {int(L): v["D"] for L, v in t.items()}


def reading_v(rows, null="sparse"):
    """Statement (v), rules in r2_job_real.py's docstring (Q5); null="sparse2" applies them to the codable null."""
    ig, isp = reading_i(rows, "gaussian"), reading_i(rows, null)
    iig, iisp = reading_ii(rows, "gaussian"), reading_ii(rows, null)
    out = {"i_sparse": isp, "ii_sparse": iisp, "windows": {}}
    prof, s3 = {}, {}
    if "windows" in iig and "windows" in iisp and iig["windows"] and iisp["windows"]:
        for w in WINDOWS:
            Dg, Ds = _D(iig, w), _D(iisp, w)
            Ls = [L for L in Dg if L in Ds]
            gsep = iig["windows"][w]["verdict"] == "supports"
            ssep = iisp["windows"][w]["verdict"] == "supports"
            if w in BULK:
                f = sum(Ds[L] for L in Ls) / sum(Dg[L] for L in Ls)
                sig = "S1 flatter bulk"
            else:
                interior, ends = [L for L in Ls if L in Q1_INTERIOR], [L for L in Ls if L in Q1_ENDS]
                amp = lambda D: max(D[L] for L in interior) - max(D[L] for L in ends)  # noqa: E731
                f = amp(Ds) / amp(Dg) if interior and ends else float("nan")
                sig = "S2 mid-stack top profile"
            v = "n/a" if not gsep else "supports" if (f >= KEEP and ssep) else "contradicts" if f < REMOVE else "partial"
            prof[w] = v
            out["windows"].setdefault(w, {})["profile"] = {"signature": sig, "gauss_separates": gsep,
                                                           "sparse_rule_separates": ssep, "f": f, "verdict": v}
    if "windows" in ig and "windows" in isp and ig.get("windows") and isp.get("windows"):
        for w in WINDOWS:
            gsep = ig["windows"][w]["verdict"] == "contradicts"
            ssep = isp["windows"][w]["verdict"] == "contradicts"
            fs = {}
            for u in UNITS:
                a, b = ig["windows"][w]["fixed"][u], isp["windows"][w]["fixed"][u]
                fs[u] = (b["s_gauss"] - b["s_real"]) / (a["s_gauss"] - a["s_real"])
            v = ("n/a" if not gsep else "supports" if all(x >= KEEP for x in fs.values()) and ssep
                 else "contradicts" if all(x < REMOVE for x in fs.values()) else "partial")
            s3[w] = v
            out["windows"].setdefault(w, {})["S3_decline"] = {"gauss_separates": gsep, "sparse_rule_separates": ssep,
                                                              "f": fs, "verdict": v}
    groups = {}
    groups["profile"] = ("moot" if iig.get("verdict") != "supports" else
                         window_rule(prof, "supports", "contradicts") if prof else "insufficient")
    groups["S3_decline"] = ("moot" if ig.get("verdict") != "contradicts" else
                            window_rule(s3, "supports", "contradicts") if s3 else "insufficient")
    out["groups"] = groups
    live = [g for g in groups.values() if g != "moot"]
    out["verdict"] = ("moot" if not live else "contradicts" if "contradicts" in live
                      else "supports" if all(g == "supports" for g in live) else "mixed")
    out["gaussian_verdicts"] = {"i": ig.get("verdict"), "ii": iig.get("verdict")}
    return out


def sparse2_rows(rows):
    """Final sparse2 runs (not calibration runs) relabelled into job 1's namespace (job "real", step "Q5") so that
    readings (i), (ii) and (v) compare them with job 1's real and Gaussian rows; job 1's own rows are untouched."""
    return [dict(r, job="real", step="Q5") for r in rows if r["job"] == "sparse2" and r.get("step") in ("S2a", "S2b")]


def reading_v_sparse2(rows):
    """Statement (v) with the codable sparse null (r2_job_sparse2.py); same rules and thresholds as reading_v."""
    s2 = sparse2_rows(rows)
    if not s2:
        return {"verdict": "insufficient", "n_sparse2_rows": 0}
    out = reading_v(rows + s2, "sparse2")
    out["n_sparse2_rows"] = len(s2)
    out["fvu_k40"] = side_by_side(rows + s2)
    out["calibration"] = [{k: r.get(k) for k in ("layer", "fvu_eval", "rho", "n_train_tokens", "run")}
                          for r in rows if r["job"] == "sparse2" and r.get("step") == "CAL"]
    return out


def side_by_side(rows):
    """Eval FVU, L0 and live count of real / gaussian / sparse (k = Q1's k per layer; per k at layer 13)."""
    q = [r for r in rows if r["job"] == "real" and r.get("k") is not None and not r["whitened"] and r["layer"] != 12]
    tab = defaultdict(lambda: defaultdict(list))
    for r in q:
        tab[(r["layer"], r["k"])][r["cond"]].append(r)
    out = {}
    for (L, k), conds in sorted(tab.items()):
        out[f"L{L}_k{k}"] = {c: {"fvu": msd([r["fvu_eval"] for r in rs])[0], "l0": msd([r["l0_eval"] for r in rs])[0],
                                 "n_live": msd([r["n_live"] for r in rs])[0], "n_seeds": len(rs),
                                 "gamma_fixed_primary": msd([r["gamma_fixed"][PRIMARY] for r in rs])[0]}
                             for c, rs in sorted(conds.items())}
    return out


def ground_truth(rows):
    out = {}
    for r in rows:
        if r.get("step") == "GT":
            out[f"L{r['layer']}"] = {"gamma_full": r["gamma_full"], "gamma_fixed": r["gamma_fixed"],
                                     "dec_full": r["dec_full"], "sigma_slope": r["sigma_slope"],
                                     "check_1M": {k: v for k, v in r["settings"].get("check_1M", {}).items()
                                                  if k != "eigenvalues"}}
    return out


# models (job 3) ----------------------------------------------------------------------------------------------------
G_INTERIOR, G_ENDS = set(range(3, 9)), {0, 1, 10, 11}  # GPT-2 layer groups; only used by the (ii) rule's top-window
#                                                       part, which R2 replaces with its range rule
PRED_A_2B = {"(3, 50)": 0.0937, "(10, 150)": 0.0909, "(30, 450)": 0.0561, "(100, 1000)": 0.0489}  # 2b L12 16k
PRED_B_MIN, PRED_C_FAIL = 0.05, 0.10


def remap(rows):
    """Rename a model's window keys to the Gemma-2-2b window they scale from, so the job-1 rules apply unchanged."""
    out = []
    for r in rows:
        m = {str(tuple(w)): g for w, g in zip(r["windows"], WINDOWS)}
        r = dict(r)
        for f, v in list(r.items()):
            if isinstance(v, dict) and v and all(k in m for k in v):
                r[f] = {m[k]: x for k, x in v.items()}
        out.append(r)
    return out


def reading_models(rows, real_rows):
    out = {"windows_note": "keys are the Gemma-2-2b windows each model window scales from (see r2_job_models.py)"}
    m1 = remap([r for r in rows if r["job"] == "models" and r["step"] in ("M1", "M1GT")])
    m2 = remap([r for r in rows if r["job"] == "models" and r["step"] == "M2"])
    if m1:
        o, nulls = {}, ("gaussian", "sparse")
        ii = {n: reading_ii(m1, n, job="models", K=32, interior_set=G_INTERIOR, ends_set=G_ENDS, steps=("M1",),
                            skip_layers=()) for n in nulls}
        r1, r2 = {}, {}
        for n in nulls:
            if "windows" not in ii[n] or not ii[n]["windows"]:
                r1[n] = r2[n] = "insufficient"
                continue
            r1[n] = {w: ii[n]["windows"][w]["fixed"]["raw"]["verdict"] for w in BULK}
            prof = {}
            for w in TOP:
                t = {int(L): v for L, v in ii[n]["windows"][w]["fixed"]["raw"]["by_layer"].items()}
                hi, lo = max(t, key=lambda L: t[L]["D"]), min(t, key=lambda L: t[L]["D"])
                rng, se = t[hi]["D"] - t[lo]["D"], float(np.hypot(t[hi]["se_D"], t[lo]["se_D"]))
                se = se if np.isfinite(se) else 0.0
                v = "shows" if rng > max(0.05, 2 * se) else "none" if (rng < 0.02 or rng <= 2 * se) else "partial"
                prof[w] = {"range": rng, "se": se, "argmax_layer": hi, "argmin_layer": lo,
                           "argmax_depth": hi / 11, "argmin_depth": lo / 11, "verdict": v}
            r2[n] = prof
        flat = [v for n in nulls if isinstance(r1[n], dict) for v in r1[n].values()]
        o["R1_bulk_flatter"] = {"per_null": r1, "verdict": "insufficient" if len(flat) < 4 else
                                "replicates" if all(v == "supports" for v in flat) else
                                "fails" if "contradicts" in flat else "partial"}
        pv = [x["verdict"] for n in nulls if isinstance(r2[n], dict) for x in r2[n].values()]
        o["R2_top_profile"] = {"per_null": r2, "verdict": "insufficient" if len(pv) < 4 else
                               "replicates" if all(v == "shows" for v in pv) else "fails" if "none" in pv else "partial"}
        gem = {"gaussian": reading_i(real_rows, "gaussian").get("verdict"),
               "sparse": reading_i(real_rows, "sparse").get("verdict")}
        r3 = {}
        for n in nulls:
            g2 = reading_i(m1, n, job="models", layer=6, kfit=[16, 32, 64, 128])
            gv = g2.get("verdict")
            if gem[n] in (None, "insufficient", "error"):
                v = "pending"
            elif gv == gem[n]:
                v = "replicates"
            elif {gv, gem[n]} == {"supports", "contradicts"}:
                v = "fails"
            else:
                v = "partial"
            r3[n] = {"gpt2": gv, "gemma": gem[n], "verdict": v, "gpt2_detail": g2.get("window_verdicts")}
        o["R3_k_decline"] = r3
        o["fvu"] = {f"L{L}_k{k}": {c: float(np.mean([r["fvu_eval"] for r in m1 if r["layer"] == L and r["k"] == k
                                                    and r["cond"] == c])) for c in ("real", "gaussian", "sparse")
                                   if any(r["layer"] == L and r["k"] == k and r["cond"] == c for r in m1)}
                    for L, k in sorted({(r["layer"], r["k"]) for r in m1 if r["k"] is not None})}
        out["M1_gpt2"] = o
    if m2:
        o = {"n_saes": len(m2), "layers": sorted({r["layer"] for r in m2})}
        slope = lambda r, w: r["dec_slope"][w]  # noqa: E731
        va = {}
        for w in WINDOWS:
            per = {}
            for L in sorted({r["layer"] for r in m2 if "all_l0" in r["reasons"]}):
                rs = [r for r in m2 if r["layer"] == L and "all_l0" in r["reasons"]]
                if len(rs) >= 3:
                    per[L] = fit([r["l0"] for r in rs], [slope(r, w) for r in rs])
            if not per:
                continue
            mean, ref = float(np.mean(list(per.values()))), PRED_A_2B[w]
            v = ("agrees" if 0.5 * ref <= mean <= 1.5 * ref and all(b > 0 for b in per.values()) else
                 "disagrees" if not (0.25 * ref <= mean <= 2 * ref) or any(b <= 0 for b in per.values()) else "partial")
            va[w] = {"per_layer": per, "mean": mean, "interval": [0.5 * ref, 1.5 * ref], "verdict": v}
        o["a_l0_per_doubling"] = {"windows": va, "verdict": window_rule({w: x["verdict"] for w, x in va.items()},
                                                                       "agrees", "disagrees") if va else "insufficient"}
        vb = {}
        for t in (40, 70):
            for w in WINDOWS:
                pick = {}
                for r in m2:
                    if f"nearest_{t}" in r["reasons"]:
                        pick[r["layer"]] = r
                if not pick:
                    continue
                mid = [slope(r, w) for r in pick.values() if 0.35 <= r["depth"] <= 0.65]
                early = [slope(r, w) for r in pick.values() if r["depth"] <= 0.12]
                late = [slope(r, w) for r in pick.values() if r["depth"] >= 0.88]
                c = (min(np.mean(early), np.mean(late)) - np.mean(mid)) if mid and early and late else float("nan")
                vb[f"L0~{t}|{w}"] = {"contrast": float(c), "mid": float(np.mean(mid)) if mid else None,
                                     "early": float(np.mean(early)) if early else None,
                                     "late": float(np.mean(late)) if late else None, "top_window": w in TOP}
        tops = [x["contrast"] for x in vb.values() if x["top_window"]]
        o["b_midstack_steeper"] = {"cells": vb, "verdict": "insufficient" if len(tops) < 4 or not all(np.isfinite(tops)) else
                                   "supports" if all(c >= PRED_B_MIN for c in tops) else
                                   "contradicts" if all(c <= 0 for c in tops) else "partial"}
        vc = {}
        for w in WINDOWS:
            g = [r["gamma"][w] for r in m2 if r["l0"] >= 20]
            if g:
                bad = float(np.mean([x >= 1 for x in g]))
                vc[w] = {"n": len(g), "frac_gamma_ge_1": bad, "max_gamma": float(max(g)),
                         "verdict": "holds" if bad == 0 else "fails" if bad > PRED_C_FAIL else "partial"}
        o["c_gamma_below_1"] = {"windows": vc, "verdict": window_rule({w: x["verdict"] for w, x in vc.items()},
                                                                     "holds", "fails") if vc else "insufficient"}
        out["M2_gemma9b"] = o
    return out


# (iii) -----------------------------------------------------------------------------------------------------------
def reading_iii(rows):
    q = [r for r in rows if r["job"] == "real" and r["layer"] == 12 and r["step"] == "Q4"]
    W, C = [r for r in q if r["whitened"]], [r for r in q if not r["whitened"]]
    ks = sorted({r["k"] for r in W})
    out = {"k": ks, "windows": {}}
    if not ks:
        out["verdict"] = "insufficient"
        return out
    verdicts = {}
    for w in WINDOWS:
        res = {"by_k": {}}
        sup, con = [], []
        for k in ks:
            wk, ck = [r for r in W if r["k"] == k], [r for r in C if r["k"] == k]
            gm = [r["gamma_fixed"][w] for r in wk]
            gi = [r["iso_null"]["gamma_fixed"][w] for r in wk]
            m, _, se = msd(gm)
            delta = float(np.mean(np.subtract(gm, gi)))
            x = {"gamma_map": m, "gamma_map_se": se, "gamma_iso": float(np.mean(gi)), "delta": delta,
                 "gamma_map_full": float(np.mean([r["gamma_full"][w] for r in wk])),
                 "gamma_iso_full": float(np.mean([r["iso_null"]["gamma_full"][w] for r in wk])),
                 "anisotropy_whitened": float(np.mean([r["whitened_space"]["anisotropy_full"][w] for r in wk])),
                 "gamma_control": msd([r["gamma_fixed"][w] for r in ck])[0] if ck else None,
                 "basis_gamma_map": float(np.mean([r["basis_gamma"]["fits"][WINDOWS.index(w)]["gamma"] for r in wk])),
                 "basis_gamma_control": float(np.mean([r["basis_gamma"]["fits"][WINDOWS.index(w)]["gamma"] for r in ck]))
                 if ck else None, "n_seeds": len(wk)}
            res["by_k"][str(k)] = x
            sup.append(abs(delta) <= DELTA_TOL)
            con.append(abs(delta) > DELTA_TOL and (abs(delta) > 2 * se if np.isfinite(se) else False))
        res["verdict"] = "supports" if all(sup) else "contradicts" if all(con) else "partial"
        verdicts[w] = res["verdict"]
        out["windows"][w] = res
    out["window_verdicts"] = verdicts
    out["verdict"] = window_rule(verdicts, "supports", "contradicts")
    out["complete"] = ks == Q4_K
    return out


# (iv) ------------------------------------------------------------------------------------------------------------
def keep_verdict(rg, rr):
    if rg >= KEEP and rr >= KEEP:
        return "keeps"
    if rg < REMOVE and rr < REMOVE:
        return "removes"
    return "partial"


def reading_iv(rows):
    q = [r for r in rows if r["job"] == "resynth" and r["step"] == "Q3"]
    ks = [k for k in kset([r for r in q if r["kind"] == "D"], K_IV) if k in kset([r for r in q if r["kind"] == "G"], K_IV)]
    draws = sorted({r["data_seed"] for r in q})
    out = {"k_fit": ks, "windows": {}, "checks": {}}
    S = {}
    for kind in "DG":
        for j in draws:
            sd = by_seed([r for r in q if r["kind"] == kind and r["data_seed"] == j], ks)
            if sd:
                S[kind, j] = sd
    both = [j for j in draws if ("D", j) in S and ("G", j) in S]
    out["draws_used"] = both
    if len(ks) < 2 or not both:
        out["verdict"] = "insufficient"
        return out
    verdicts = {}
    for w in WINDOWS:
        res = {}
        for mode in ("fixed", "full"):
            s = {(kind, j): {u: float(np.mean(seed_slopes(S[kind, j], ks, u, mode, w))) for u in UNITS}
                 for kind in "DG" for j in both}
            pooled = {kind: {u: float(np.mean([s[kind, j][u] for j in both])) for u in UNITS} for kind in "DG"}
            sd = {kind: {u: msd([s[kind, j][u] for j in both])[1] for u in UNITS} for kind in "DG"}
            ret = {u: pooled["G"][u] / pooled["D"][u] for u in UNITS}
            per_draw = {str(j): {"retained_gamma": s["G", j]["gamma"] / s["D", j]["gamma"],
                                 "retained_raw": s["G", j]["raw"] / s["D", j]["raw"]} for j in both}
            for x in per_draw.values():
                x["verdict"] = keep_verdict(x["retained_gamma"], x["retained_raw"])
            v = keep_verdict(ret["gamma"], ret["raw"])
            firm = all(x["verdict"] == v for x in per_draw.values()) and len(both) > 1
            res[mode] = {"s": {f"{k}_ds{j}": x for (k, j), x in s.items()}, "pooled": pooled, "draw_sd": sd,
                         "retained_gamma": ret["gamma"], "retained_raw": ret["raw"], "per_draw": per_draw,
                         "verdict": v, "firm": firm, "label": v if firm else f"{v}, draw-dependent"}
        res["verdict"] = res["fixed"]["verdict"]
        verdicts[w] = res["verdict"]
        out["windows"][w] = res
    out["window_verdicts"] = verdicts
    out["verdict"] = {"keeps": "supports", "removes": "contradicts"}.get(window_rule(verdicts, "keeps", "removes"), "mixed")
    out["primary_label"] = out["windows"][PRIMARY]["fixed"]["label"]
    # checks
    for r in q:
        if r["sigma_slope"][PRIMARY] > ILL:
            out["checks"].setdefault("ill_conditioned", []).append(r["dataset"])
    ctl = ROOT / "results" / "phase3" / "control.json"
    if ctl.exists():
        prev = json.load(open(ctl))["rows"]
        for kind in "DG":
            old = {r["k"]: r["gamma_fixed"][PRIMARY] for r in prev if r["data"] == f"rs12-{kind}" and r["seed"] == 0
                   and r.get("gamma_fixed")}
            new = {r["k"]: r["gamma_fixed"][PRIMARY] for r in q if r["kind"] == kind and r["data_seed"] == 0
                   and r["seed"] == 0 and r["width"] == 16384}
            common = sorted(set(old) & set(new))
            if common:
                diffs = {str(k): new[k] - old[k] for k in common}
                out["checks"][f"{kind}_ds0_reproduces_control"] = {
                    "diffs": diffs, "max_abs_diff": max(map(abs, diffs.values())),
                    "pass": max(map(abs, diffs.values())) <= 0.02}
    out["checks"]["ill_conditioned"] = sorted(set(out["checks"].get("ill_conditioned", [])))
    return out


# robust (job 5) ----------------------------------------------------------------------------------------------------
G_BULK_GAP, G_NEAR, M_TRACK = 0.15, 0.05, 0.06
M_TOP, M_PRIMARY, M_BULK = "(1, 17)", "(3, 50)", ["(10, 150)", "(33, 333)"]


def _real_mean(rows, job, k, steps=None):
    """{layer: {window: mean gamma_fixed}} and {layer: mean FVU} of the real SAEs of a job at one k."""
    g, f = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in rows:
        if (r["job"] == job and r.get("cond") == "real" and r.get("k") == k and not r.get("whitened")
                and (steps is None or r.get("step") in steps)):
            for w, v in r["gamma_fixed"].items():
                g[r["layer"]][w].append(v)
            f[r["layer"]].append(r["fvu_eval"])
    return ({L: {w: float(np.mean(v)) for w, v in ws.items()} for L, ws in g.items()},
            {L: float(np.mean(v)) for L, v in f.items()})


def robust_g(rows, rb):
    """Step G of r2_job_robust.py: generator sensitivity of the bulk result."""
    real, real_fvu = _real_mean(rows, "real", 40)
    out, gaps = {"settings": {}}, []
    base = [r for r in rows if r["job"] == "sparse2" and r.get("k") == 40 and r.get("seed") == 0
            and r.get("step") in ("S2a", "S2b")]
    for r in sorted(base, key=lambda r: r["layer"]):
        out["settings"][f"G|L{r['layer']}|M8192_L16 (sparse2, reference)"] = {
            "gamma_fixed": r["gamma_fixed"], "fvu": r["fvu_eval"], "rho": r.get("rho"), "in_rule": False}
    for r in sorted((r for r in rb if r.get("step") == "G"), key=lambda r: r["setting_key"]):
        L = r["layer"]
        if L not in real:
            continue
        d = {w: r["gamma_fixed"][w] - real[L][w] for w in WINDOWS}
        out["settings"][r["setting_key"]] = {
            "matched": r["matched"], "codable": r.get("codable"), "rho": r["rho"], "fvu": r["fvu_eval"],
            "fvu_target": r["fvu_target"], "n_live": r["n_live"], "gamma_fixed": r["gamma_fixed"],
            "real_gamma_fixed": real[L], "null_minus_real": d, "in_rule": bool(r["matched"])}
        if r["matched"]:
            gaps.append(min(d[w] for w in BULK))
    out["unmatched"] = sorted(k for k, v in out["settings"].items() if v["in_rule"] is False and "reference" not in k)
    out["n_matched"] = len(gaps)
    out["min_bulk_gap"] = min(gaps) if gaps else None
    out["verdict"] = ("insufficient" if not gaps else "generator-dependent" if min(gaps) <= G_NEAR
                      else "robust to the generator" if min(gaps) >= G_BULK_GAP else "partial")
    return out


def robust_m(rows, rb):
    """Step M of r2_job_robust.py: the codable null on GPT-2 small, three statements against Gemma's outcome."""
    real, _ = _real_mean(rows, "models", 32, steps=("M1",))
    null = {r["layer"]: r for r in rb if r.get("step") == "M"}
    Ls = sorted(L for L, r in null.items() if r["matched"] and L in real)
    if not Ls:
        return {"verdict": "insufficient", "n_layers": 0}
    d = {L: {w: null[L]["gamma_fixed"][w] - real[L][w] for w in real[L]} for L in Ls}
    out = {"layers_matched": Ls, "unmatched": sorted(L for L, r in null.items() if not r["matched"]),
           "fvu": {L: {"null": null[L]["fvu_eval"], "target": null[L]["fvu_target"], "rho": null[L]["rho"]}
                   for L in sorted(null)},
           "null_minus_real": d, "windows_note": "GPT-2 windows (1,17), (3,50) primary, (10,150), (33,333)"}
    frac = {w: sum(d[L][w] >= G_BULK_GAP for L in Ls) / len(Ls) for w in M_BULK}
    out["bulk"] = {"flatter_by_0.15_fraction": frac, "gemma": "8/8 in both bulk windows",
                   "verdict": ("replicates" if all(f >= 0.9 for f in frac.values())
                               else "fails" if any(f < 0.5 for f in frac.values()) else "partial")}
    track = [L for L in Ls if abs(d[L][M_PRIMARY]) <= M_TRACK]
    inner, ends = [L for L in Ls if L in G_INTERIOR], [L for L in Ls if L in G_ENDS]
    ti, te = sum(L in track for L in inner), sum(L in track for L in ends)
    out["primary"] = {"tracking_layers": track, "interior_tracking": f"{ti}/{len(inner)}",
                      "end_tracking": f"{te}/{len(ends)}", "gemma": "tracks at 6, 9, 13, 17; not at 0, 3, 21, 24",
                      "verdict": ("insufficient" if not inner else
                                  "replicates" if ti >= 2 / 3 * len(inner) and te <= 1 else
                                  "fails" if ti < len(inner) / 3 or te >= 3 else "partial")}
    steeper = sum(d[L][M_TOP] < 0 for L in Ls)  # real gamma above the null's: real steeper
    flatter = sum(d[L][M_TOP] > 0 for L in Ls)
    out["top"] = {"real_steeper_layers": steeper, "real_flatter_layers": flatter, "n": len(Ls),
                  "gemma": "real steeper at 7/8 layers",
                  "verdict": ("replicates" if steeper >= 0.75 * len(Ls) else "fails" if flatter >= 0.5 * len(Ls)
                              else "partial")}
    out["verdict"] = {k: out[k]["verdict"] for k in ("bulk", "primary", "top")}
    return out


def robust_p(rows, rb):
    """Step P of r2_job_robust.py: feature power and the head/body rule, real against the equal-power null."""
    ref = {(r["job"], r["run"]): r for r in rows if r["job"] in ("real", "sparse2")}
    out = {"layers": {}}
    pres = {"real": {}, "sparse2": {}}
    for r in sorted((r for r in rb if r.get("step") == "P"), key=lambda r: (r["layer"], r["cond"])):
        o = ref.get(("real" if r["cond"] == "real" else "sparse2", r["run"]))
        rep = None
        if o is not None:
            rep = {"fvu_new": r["fvu_eval"], "fvu_ref": o["fvu_eval"], "n_live_new": r["n_live_eval"],
                   "n_live_ref": o["n_live"], "max_abs_gamma_fixed_diff": max(
                       abs(r["reproduction"]["gamma_fixed_eval_live"][w] - o["gamma_fixed"][w]) for w in WINDOWS)}
        pres[r["cond"]][r["layer"]] = r["head_body"]["present"]
        out["layers"][f"L{r['layer']}_{r['cond']}"] = {
            "head_body": r["head_body"], "power_weighted_gamma": r["power_weighted_gamma"],
            "unweighted_gamma_fixed": r["unweighted"]["gamma_fixed"], "share_top10": r["power_share_top10"],
            "share_top1pct": r["power_share_top1pct"], "ess_frac_live": r["power_ess_frac_live"],
            "rq_median_log10": r["rq_median_log10"], "spearman_pi_rq": r["spearman_pi_rq"],
            "sum_pi_over_total_var": r["sum_pi_over_total_var"], "fvu": r["fvu_eval"], "reproduction": rep}
    nr, nn = len(pres["real"]), len(pres["sparse2"])
    fr = sum(pres["real"].values()) / nr if nr else float("nan")
    fn = sum(pres["sparse2"].values()) / nn if nn else float("nan")
    out["present_real"] = f"{sum(pres['real'].values())}/{nr}"
    out["present_null"] = f"{sum(pres['sparse2'].values())}/{nn}"
    out["verdict"] = ("insufficient" if nr < 4 else "not specific to real" if nn and fn >= 0.5
                      else "supported" if fr >= 0.75 and (not nn or fn <= 0.25) else "absent" if fr <= 0.25
                      else "partial") + ("" if nn >= 4 or nr < 4 else " (fewer than 4 null layers)")
    return out


def reading_robust(rows):
    """The three pre-registered verdicts of r2_job_robust.py (rules in its docstring)."""
    rb = [r for r in rows if r["job"] == "robust"]
    if not rb:
        return {"verdict": "insufficient", "n_rows": 0}
    out = {"n_rows": len(rb)}
    for name, fn in (("G_generator", robust_g), ("M_gpt2_codable_null", robust_m), ("P_feature_power", robust_p)):
        try:
            out[name] = fn(rows, rb)
        except Exception as e:  # partial results must not hide the other two
            out[name] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
    out["verdict"] = {k: out[k].get("verdict") for k in ("G_generator", "M_gpt2_codable_null", "P_feature_power")}
    return out


# pmatch (job 6) ----------------------------------------------------------------------------------------------------
PM_MARGIN, PM_FRAC, PM_LAYERS = 0.10, 0.30, (13, 0)


def _pm_window(N, R, Z):
    """One bulk window: closed (|N - R| <= margin), overshoot (N < R - margin), partial (f >= 0.30) or open."""
    f = (Z - N) / (Z - R)
    cls = ("closed" if abs(N - R) <= PM_MARGIN else "overshoot" if N < R - PM_MARGIN
           else "partial" if f >= PM_FRAC else "open")
    return {"null": N, "real": R, "v0": Z, "closed_fraction": f, "class": cls}


def _pm_layer_verdict(r, R, Z):
    w = {b: _pm_window(r["gamma_fixed"][b], R[b], Z[b]) for b in BULK}
    cls = [w[b]["class"] for b in BULK]
    v = "EXPLAINS" if all(c == "closed" for c in cls) else "DOES NOT EXPLAIN" if "open" in cls else "PARTLY EXPLAINS"
    if not r["matched"]:
        v = "insufficient (FVU not matched)"
    pc = r.get("power_check") or {}
    dic = r.get("dictionary") or {}
    a1, a0 = dic.get("atom_cloud_gamma_fixed"), dic.get("v0_atom_cloud_gamma_fixed")
    pred = {b: a1[b] + Z[b] - a0[b] for b in WINDOWS} if a1 and a0 else None  # registered additive prediction
    return {"verdict": v, "overshoots": all(c == "overshoot" for c in cls), "windows": w, "rho": r["rho"],
            "additive_prediction": pred, "additive_prediction_error": {b: r["gamma_fixed"][b] - pred[b] for b in BULK}
            if pred else None,
            "fvu": r["fvu_eval"], "fvu_target": r["fvu_target"], "matched": r["matched"],
            "calibration_complete": r.get("calibration_complete"), "n_live": r["n_live"],
            "gamma_fixed_all_windows": r["gamma_fixed"], "real_all_windows": R, "v0_all_windows": Z,
            "atom_cloud_gamma_fixed": dic.get("atom_cloud_gamma_fixed"), "dictionary_power": dic.get("power"),
            "dictionary_converged": dic.get("converged"),
            "power_moved_by_projection": dic.get("power_share_moved_by_projection"),
            "manipulation_check": {k: pc.get(k) for k in ("null", "real", "top1pct_share_diff", "ess_ratio",
                                                          "rq_median_log10", "spearman_pi_rq")} if pc else None}


def reading_pmatch(rows):
    """Rules of r2_job_pmatch.py: the power-matched null (V1) against real (mean of job 1's three k = 40 SAEs) and the
    equal-power null V0 at the same (layer, M, L): sparse2 S2a/S2b seed 0 for M 8192 L 16, robust step G otherwise."""
    pm = [r for r in rows if r["job"] == "pmatch"]
    if not pm:
        return {"verdict": "insufficient", "n_rows": 0}
    real, _ = _real_mean(rows, "real", 40)
    v0 = {(r["layer"], 16): r["gamma_fixed"] for r in rows if r["job"] == "sparse2" and r.get("k") == 40
          and r.get("seed") == 0 and r.get("step") in ("S2a", "S2b")}
    for r in rows:
        if r["job"] == "robust" and r.get("step") == "G" and r.get("matched") and (r.get("setting") or {}).get("M") == 8192 \
                and not (r.get("setting") or {}).get("overflow"):
            v0.setdefault((r["layer"], r["setting"]["L"]), r["gamma_fixed"])
    out = {"n_rows": len(pm), "margin": PM_MARGIN, "min_closed_fraction": PM_FRAC, "settings": {}}
    acc = {r["setting_key"]: r for r in pm if r.get("step") == "PM"}
    for key in sorted({r["setting_key"] for r in pm}):
        r = acc.get(key)
        tried = sorted((x["rho"], x["fvu_eval"]) for x in pm if x["setting_key"] == key)
        if r is None:
            out["settings"][key] = {"verdict": "insufficient", "tried": tried}
            continue
        L, Ls = r["layer"], r["setting"]["L"]
        if L not in real or (L, Ls) not in v0:
            out["settings"][key] = {"verdict": "insufficient (no real or V0 reference)", "tried": tried}
            continue
        out["settings"][key] = _pm_layer_verdict(r, real[L], v0[(L, Ls)]) | {"tried": tried, "setting": r["setting"]}

    def find(layer, L, ds):
        for k, v in out["settings"].items():
            if k.startswith(f"PM|V1|L{layer}|") and k.endswith(f"_L{L}|ds{ds}"):
                return k, v
        return None, None
    core = {Lyr: find(Lyr, 16, 0)[1] for Lyr in PM_LAYERS}
    vs = [c["verdict"] if c else "insufficient" for c in core.values()]
    head = ("insufficient" if any(v.startswith("insufficient") for v in vs) else vs[0] if vs[0] == vs[1]
            else "PARTLY EXPLAINS")
    out["headline"] = {"question": "does unequal feature power (V1) explain the bulk gap?", "verdict": head,
                       "per_layer": dict(zip([f"L{x}" for x in PM_LAYERS], vs))}
    gen = [find(13, Ls, 0)[1] for Ls in (8, 32)]
    gv = [g["verdict"] if g else "insufficient" for g in gen]
    out["generator_settings"] = {"L8": gv[0], "L32": gv[1], "L16_layer13": vs[0],
                                 "verdict": ("insufficient" if any(v.startswith("insufficient") for v in gv + vs[:1])
                                             else "same at other (M, L)" if all(v == vs[0] for v in gv) else "differs")}
    k0, s0 = find(13, 16, 0)
    k1, s1 = find(13, 16, 1)
    if s0 and s1 and "windows" in s0 and "windows" in s1:
        dmax = max(abs(s1["windows"][b]["null"] - s0["windows"][b]["null"]) for b in BULK)
        out["replicate"] = {"max_abs_bulk_diff": dmax, "flag_margin_too_narrow": dmax > 0.05}
    else:
        out["replicate"] = {"verdict": "insufficient"}
    out["verdict"] = {"headline": head, "generator_settings": out["generator_settings"]["verdict"],
                      "replicate_max_abs_bulk_diff": out["replicate"].get("max_abs_bulk_diff")}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "r2"))
    d = Path(ap.parse_args().dir)
    rows = load(d)
    out = {"n_rows": len(rows), "n_rows_by_job": {j: sum(r["job"] == j for r in rows) for j in ("real", "resynth", "models", "sparse2", "robust",
                                                                             "pmatch")},
           "git_hashes": sorted({str(r.get("git_hash")) for r in rows}),
           "code_sha256": sorted({r.get("code_sha256", "")[:12] for r in rows}),
           "fixed_n_short_runs": sorted(r["run"] for r in rows if r.get("fixed_n_short")),
           "window_rule": "primary ranks 11-150 plus at least two of the other three windows"}
    for name, fn in (("i_gauss_decline_as_steady_as_real", reading_i), ("ii_layer_profile_real_minus_gauss", reading_ii),
                     ("iii_whitening_isotropic", reading_iii), ("iv_G_retains_D_decline", reading_iv),
                     ("v_sparse_null_separates_like_gaussian", reading_v),
                     ("v_sparse2_null_separates_like_gaussian", reading_v_sparse2)):
        try:
            out[name] = fn(rows)
        except Exception as e:  # a partial pull must never stop the other readings
            out[name] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
        print(f"{name:<38} {out[name].get('verdict')}  {out[name].get('window_verdicts', '')}")
    try:
        out["models"] = reading_models(rows, rows)
    except Exception as e:
        out["models"] = {"error": f"{type(e).__name__}: {e}"}
    print("models", json.dumps({k: {kk: (vv.get("verdict") if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                                for k, v in out["models"].items() if isinstance(v, dict)}, default=str)[:600])
    try:
        out["robust"] = reading_robust(rows)
    except Exception as e:
        out["robust"] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
    print("robust", json.dumps(out["robust"].get("verdict"), default=str))
    try:
        out["pmatch"] = reading_pmatch(rows)
    except Exception as e:
        out["pmatch"] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
    print("pmatch", json.dumps(out["pmatch"].get("verdict"), default=str))
    for name, fn in (("fvu_real_gauss_sparse", side_by_side), ("sparse_ground_truth_dictionary", ground_truth)):
        try:
            out[name] = fn(rows)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, open(d / "summary.json", "w"), indent=1, default=float)
    print(f"{len(rows)} rows -> {d / 'summary.json'}")


if __name__ == "__main__":
    main()
