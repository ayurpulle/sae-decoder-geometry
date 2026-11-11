"""Pre-registered readings (i)-(iv) of the round-2 GPU jobs, from results/r2/runs_real.jsonl and runs_resynth.jsonl.
The rules are fixed in the docstrings of r2_job_real.py ((i)-(iii)) and r2_job_resynth.py ((iv)); this file only
implements them. Works on partial results: every reading lists the runs it used and is "insufficient" when a fit
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


def reading_v(rows):
    """Statement (v), rules in r2_job_real.py's docstring (Q5)."""
    ig, isp = reading_i(rows, "gaussian"), reading_i(rows, "sparse")
    iig, iisp = reading_ii(rows, "gaussian"), reading_ii(rows, "sparse")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "r2"))
    d = Path(ap.parse_args().dir)
    rows = load(d)
    out = {"n_rows": len(rows), "n_rows_by_job": {j: sum(r["job"] == j for r in rows) for j in ("real", "resynth", "models")},
           "git_hashes": sorted({str(r.get("git_hash")) for r in rows}),
           "code_sha256": sorted({r.get("code_sha256", "")[:12] for r in rows}),
           "fixed_n_short_runs": sorted(r["run"] for r in rows if r.get("fixed_n_short")),
           "window_rule": "primary ranks 11-150 plus at least two of the other three windows"}
    for name, fn in (("i_gauss_decline_as_steady_as_real", reading_i), ("ii_layer_profile_real_minus_gauss", reading_ii),
                     ("iii_whitening_isotropic", reading_iii), ("iv_G_retains_D_decline", reading_iv),
                     ("v_sparse_null_separates_like_gaussian", reading_v)):
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
    for name, fn in (("fvu_real_gauss_sparse", side_by_side), ("sparse_ground_truth_dictionary", ground_truth)):
        try:
            out[name] = fn(rows)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, open(d / "summary.json", "w"), indent=1, default=float)
    print(f"{len(rows)} rows -> {d / 'summary.json'}")


if __name__ == "__main__":
    main()
