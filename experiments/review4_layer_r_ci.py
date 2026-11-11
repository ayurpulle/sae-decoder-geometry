"""autocorrelation-aware uncertainty for the cross-layer correlations.

(1) Layer profile: Pearson r (and Spearman rho) across layers between the Gemma Scope decoder slope at matched L0
    (results/phase1/slope_vs_layer_matched_l0.json) and the 1M-token Sigma slope
    (results/test1/sigma_1m_compare.json, sigma_slope/1m), for EVERY width x L0 x window cell in the decoder file
    (16k, 65k; L0 20, 40, 70, 140; four windows). Cells with fewer than 26 layers are reported and flagged.
(2) Cluster excess: Spearman rho across 26 layers between the decoder slope and the excess cluster statistics
    (results/phase4/summary.json, layers/{256,1024}), every statistic x window x K the summary reports.

Methods (fixed before looking at results):
  a. Effective sample size (Bartlett 1935 / Bretherton et al. 1999 form):
     n_eff = n / (1 + 2 sum_{k=1}^{K} r_x(k) r_y(k)), lags k = 1..K with K = floor(n/4), clipped to [3, n].
     Fisher-z p-value and 95% CI using n_eff - 3 in place of n - 3.
  b. Phase-randomised surrogates (Ebisuzaki 1997): randomise the Fourier phases of the decoder-slope series,
     keeping its power spectrum, 10,000 surrogates; two-sided p = P(|r_surr| >= |r_obs|).
     A mean-removed, linearly detrended version is not used (a pre-registered choice: the hump is the signal).
  c. Circular-shift surrogates: all n - 1 non-trivial circular shifts of the decoder series against Sigma's
     (p has resolution 1/n; a coarse but assumption-light test).
  d. Circular moving-block bootstrap of the paired layers, block length 3 and 5, 10,000 resamples;
     percentile 95% CI of r (and of rho).
Writes results/review/layer_r_ci.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review" / "layer_r_ci.json"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
NSUR, NBOOT = 10000, 10000


def wkey(w):
    return f"ranks{w[0] + 1}-{w[1]}"


def acf(x, k):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


def n_eff(x, y):
    n = len(x)
    K = n // 4
    s = sum(acf(x, k) * acf(y, k) for k in range(1, K + 1))
    return float(np.clip(n / (1 + 2 * s), 3, n)), [acf(x, k) for k in (1, 2, 3)], [acf(y, k) for k in (1, 2, 3)]


def fisher(r, ne):
    z, se = np.arctanh(r), 1 / np.sqrt(max(ne - 3, 1e-9))
    p = 2 * norm.sf(abs(z) / se)
    return float(p), [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def phase_surrogates(y, n_sur, rng):
    n = len(y)
    Y = np.fft.rfft(y - y.mean())
    m = len(Y)
    ph = rng.uniform(0, 2 * np.pi, size=(n_sur, m))
    ph[:, 0] = 0
    if n % 2 == 0:
        ph[:, -1] = 0  # Nyquist stays real
    return np.fft.irfft(np.abs(Y)[None, :] * np.exp(1j * ph), n=n, axis=1) + y.mean()


def corr_rows(x, Ys, kind):
    if kind == "spearman":
        x = np.argsort(np.argsort(x)).astype(float)
        Ys = np.argsort(np.argsort(Ys, axis=1), axis=1).astype(float)
    xc = x - x.mean()
    Yc = Ys - Ys.mean(1, keepdims=True)
    return (Yc @ xc) / (np.linalg.norm(Yc, axis=1) * np.linalg.norm(xc))


def block_boot(x, y, b, kind, rng):
    n = len(x)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(NBOOT, nb))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(NBOOT, -1)[:, :n]
    X, Y = x[idx], y[idx]
    if kind == "spearman":
        X = np.argsort(np.argsort(X, axis=1), axis=1).astype(float)
        Y = np.argsort(np.argsort(Y, axis=1), axis=1).astype(float)
    Xc, Yc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(Yc, axis=1)
    r = (Xc * Yc).sum(1) / np.where(den > 0, den, np.nan)
    r = r[np.isfinite(r)]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def analyse(x, y, kind, seed):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    r = float(pearsonr(x, y)[0] if kind == "pearson" else spearmanr(x, y)[0])
    ne, ax, ay = n_eff(x, y)
    p_f, ci_f = fisher(r, ne)
    p_naive, ci_naive = fisher(r, len(x))
    rs = corr_rows(x, phase_surrogates(y, NSUR, rng), kind)
    shifts = np.stack([np.roll(y, s) for s in range(1, len(y))])
    rc = corr_rows(x, shifts, kind)
    return {"n": len(x), "r": r, "n_eff": ne, "acf_x_lag123": ax, "acf_y_lag123": ay,
            "fisher_naive": {"p": p_naive, "ci95": ci_naive},
            "fisher_neff": {"p": p_f, "ci95": ci_f},
            "phase_surrogate": {"p": float((np.abs(rs) >= abs(r)).mean()),
                                "r_null_95pct_abs": float(np.percentile(np.abs(rs), 95))},
            "circular_shift": {"p": float(((np.abs(rc) >= abs(r)).sum() + 1) / len(y)),
                               "max_abs_r_shift": float(np.abs(rc).max())},
            "block_boot_ci95": {"b3": block_boot(x, y, 3, kind, rng), "b5": block_boot(x, y, 5, kind, rng)}}


def main():
    sig = json.load(open(ROOT / "results" / "test1" / "sigma_1m_compare.json"))["sigma_slope"]["1m"]
    dec = json.load(open(ROOT / "results" / "phase1" / "slope_vs_layer_matched_l0.json"))
    out = {"method": __doc__.split("Methods")[1].split("Writes")[0].strip(), "profile": {}, "cluster": {}}
    print("1. layer profile: decoder slope vs 1M Sigma slope")
    print(f"  {'cell':<30}{'n':>3}{'r':>7}{'n_eff':>7}{'p_neff':>8}{'CI(neff)':>17}{'p_phase':>9}{'p_shift':>8}{'boot b3':>17}{'boot b5':>17}")
    for width in ("16k", "65k"):
        for l0 in (20, 40, 70, 140):
            for w in WINDOWS:
                k = f"{width}_{wkey(w)}_L0_{l0}"
                if k not in dec:
                    continue
                layers = sorted(int(L) for L in dec[k])
                y = [dec[k][str(L)] for L in layers]
                x = [sig[str(w)][L] for L in layers]
                res = {"layers": layers, "pearson": analyse(x, y, "pearson", 1), "spearman": analyse(x, y, "spearman", 2)}
                res["contiguous_26"] = len(layers) == 26
                out["profile"][k] = res
                a = res["pearson"]
                print(f"  {k:<30}{a['n']:>3}{a['r']:>+7.3f}{a['n_eff']:>7.1f}{a['fisher_neff']['p']:>8.4f}"
                      f"  [{a['fisher_neff']['ci95'][0]:+.2f},{a['fisher_neff']['ci95'][1]:+.2f}]{a['phase_surrogate']['p']:>9.4f}"
                      f"{a['circular_shift']['p']:>8.3f}  [{a['block_boot_ci95']['b3'][0]:+.2f},{a['block_boot_ci95']['b3'][1]:+.2f}]"
                      f"  [{a['block_boot_ci95']['b5'][0]:+.2f},{a['block_boot_ci95']['b5'][1]:+.2f}]")

    print("2. cluster excess vs decoder slope (Spearman, 26 layers, 16k L0 ~ 70)")
    ph = json.load(open(ROOT / "results" / "phase4" / "summary.json"))
    for K in ("256", "1024"):
        d = ph["layers"][K]
        for stat in ("excess_share", "excess_cos", "excess_pr16", "matched_share"):
            for w in WINDOWS:
                y, x = d[stat], d[f"slope_{w}"]
                res = analyse(x, y, "spearman", 3)
                res["stored_rho"] = ph["spearman_vs_slope"][K][f"{stat}|{w}"]
                out["cluster"][f"K{K}|{stat}|{wkey(w)}"] = res
                print(f"  K{K:<5}{stat:<14}{wkey(w):<16} rho {res['r']:+.3f} (stored {res['stored_rho']:+.3f}) n_eff {res['n_eff']:5.1f}"
                      f"  p_neff {res['fisher_neff']['p']:.4f}  p_phase {res['phase_surrogate']['p']:.4f}  p_shift {res['circular_shift']['p']:.3f}"
                      f"  boot5 [{res['block_boot_ci95']['b5'][0]:+.2f},{res['block_boot_ci95']['b5'][1]:+.2f}]")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
