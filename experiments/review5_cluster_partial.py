"""Check: cluster-excess vs decoder-slope correlations across the
26 layers (results/phase4/summary.json; 16k Gemma Scope SAE nearest L0 70 per layer), with every surrogate p-value,
partial correlations controlling for Sigma's slope, and the bulk-window values at 256 clusters.

Analysis choices (fixed before looking at results):
  Correlations: Spearman rho between the decoder slope (slope_C in the window) and each cluster statistic
    (excess_share, excess_cos, excess_pr16, matched_share), K = 256 and 1024, four windows: 32 cells.
  p-values for the raw rho: review4_layer_r_ci.analyse (Bartlett n_eff Fisher-z; 10,000 Ebisuzaki phase surrogates
    of the statistic series; all 25 circular shifts, floor 1/26 = 0.038).
  Partial Spearman controlling for z: rank-transform x (decoder slope), y (statistic), z; regress rank(x) and rank(y)
    on [1, rank(z)] by OLS; partial rho = Pearson r of the two residual series. Its p-values: n_eff Fisher-z with
    n_eff computed on the residual series and reduced by 1 for the conditioning variable; 10,000 phase surrogates of
    the y residuals; all 25 circular shifts of the y residuals (floor 1/26).
  Controls: (primary) z = 1M-token Sigma slope in the same window (results/test1/sigma_1m_compare.json,
    sigma_slope/1m); (secondary, for the excess statistics only) z = matched_share at the same K (the
    covariance-matched cloud's own between-cluster share, a proxy for that cloud's anisotropy);
    (tertiary) both together. Also stored: Spearman rho of each statistic with Sigma's slope directly.
  Reading rule (stated in advance): a correlation "survives the shared-profile control" if its partial rho keeps the
  sign of the raw rho, |partial rho| >= 0.5, and its phase-surrogate p < 0.05.
Writes results/review/cluster_partial.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from review4_layer_r_ci import analyse, fisher, n_eff, phase_surrogates, corr_rows  # noqa: E402

OUT = ROOT / "results" / "review" / "cluster_partial.json"
WINDOWS = ["(3, 50)", "(10, 150)", "(30, 450)", "(100, 1000)"]
WLAB = {"(3, 50)": "ranks4-50", "(10, 150)": "ranks11-150", "(30, 450)": "ranks31-450", "(100, 1000)": "ranks101-1000"}
STATS = ["excess_share", "excess_cos", "excess_pr16", "matched_share"]
NSUR = 10000


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def resid(v, Z):
    X = np.c_[np.ones(len(v)), Z]
    b, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ b


def partial(x, y, zs, seed):
    rx, ry = rank(x), rank(y)
    Z = np.column_stack([rank(z) for z in zs])
    ex, ey = resid(rx, Z), resid(ry, Z)
    r = float(np.corrcoef(ex, ey)[0, 1])
    ne, _, _ = n_eff(ex, ey)
    ne = max(ne - len(zs), 3.0)
    p_f, ci = fisher(r, ne)
    rng = np.random.default_rng(seed)
    rs = corr_rows(ex, phase_surrogates(ey, NSUR, rng), "pearson")
    shifts = np.stack([np.roll(ey, s) for s in range(1, len(ey))])
    rc = corr_rows(ex, shifts, "pearson")
    return {"partial_rho": r, "n_eff": ne, "p_neff": p_f, "ci95_neff": ci,
            "p_phase": float((np.abs(rs) >= abs(r)).mean()),
            "p_shift": float(((np.abs(rc) >= abs(r)).sum() + 1) / len(ey))}


def main():
    ph = json.load(open(ROOT / "results" / "phase4" / "summary.json"))
    sig = json.load(open(ROOT / "results" / "test1" / "sigma_1m_compare.json"))["sigma_slope"]["1m"]
    out = {"notes": __doc__.split("Writes")[0].strip(), "cells": {}}
    print(f"{'cell':<38}{'rho':>7}{'p_neff':>8}{'p_ph':>8}{'p_sh':>7} | {'rho|Sig':>8}{'p_ph':>8}{'p_sh':>7}"
          f" | {'rho|msh':>8}{'p_ph':>8} | {'rho|both':>8}{'p_ph':>8} | rho(stat,Sig)")
    for K in ("256", "1024"):
        d = ph["layers"][K]
        for stat in STATS:
            for w in WINDOWS:
                x, y, z = np.array(d[f"slope_{w}"]), np.array(d[stat]), np.array(sig[w])
                base = analyse(x, y, "spearman", 3)
                rec = {"rho": base["r"], "stored_rho": ph["spearman_vs_slope"][K][f"{stat}|{w}"],
                       "p_neff": base["fisher_neff"]["p"], "n_eff": base["n_eff"],
                       "p_phase": base["phase_surrogate"]["p"], "p_shift": base["circular_shift"]["p"],
                       "rho_stat_vs_sigma_slope": float(spearmanr(y, z)[0]),
                       "rho_dec_vs_sigma_slope": float(spearmanr(x, z)[0]),
                       "partial_sigma": partial(x, y, [z], 11)}
                if stat != "matched_share":
                    msh = np.array(d["matched_share"])
                    rec["partial_matched_share"] = partial(x, y, [msh], 12)
                    rec["partial_both"] = partial(x, y, [z, msh], 13)
                ps = rec["partial_sigma"]
                rec["survives_sigma_control"] = bool(np.sign(ps["partial_rho"]) == np.sign(rec["rho"])
                                                     and abs(ps["partial_rho"]) >= 0.5 and ps["p_phase"] < 0.05)
                out["cells"][f"K{K}|{stat}|{WLAB[w]}"] = rec
                pm = rec.get("partial_matched_share", {})
                pb = rec.get("partial_both", {})
                print(f"K{K:<5}{stat:<14}{WLAB[w]:<14}{rec['rho']:>+7.3f}{rec['p_neff']:>8.4f}{rec['p_phase']:>8.4f}"
                      f"{rec['p_shift']:>7.3f} | {ps['partial_rho']:>+8.3f}{ps['p_phase']:>8.4f}{ps['p_shift']:>7.3f}"
                      f" | {pm.get('partial_rho', np.nan):>+8.3f}{pm.get('p_phase', np.nan):>8.4f}"
                      f" | {pb.get('partial_rho', np.nan):>+8.3f}{pb.get('p_phase', np.nan):>8.4f}"
                      f" | {rec['rho_stat_vs_sigma_slope']:+.3f}  {'SURVIVES' if rec['survives_sigma_control'] else ''}")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
