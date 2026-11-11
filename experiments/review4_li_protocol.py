"""reconcile with Li et al. (arXiv:2410.19750, Sec. 5.1, Fig. 7), who report that activation spectra
"show a much slower decay" than SAE-feature spectra, while this project finds Sigma steeper than the decoder
cloud (gamma = decoder slope / Sigma slope < 1).

What Li et al. specify (paper text; their released notebook ejmichaud/feature-geometry,
galaxy/PointCloudStructure_Galaxy.ipynb, read 2026-11-10):
  - text: keep "the 100 largest principal components", fit "against the 100 largest eigenvalues" (ranks 1-100);
  - notebook: the decoder cloud is the raw Gemma Scope W_dec (65k, L0 = 72 at layer 12), centred covariance
    (np.cov), fit over 0-based [10, 150) = ranks 11-150 of the full 2304-d spectrum, no PC truncation;
  - the notebook contains NO activation-spectrum code. How Fig. 7's activations were sampled, centred or
    normalised, and from which layer, is not stated anywhere we could read. We therefore grid over the plausible
    choices rather than reproduce one protocol.
PCA to 100 components followed by the covariance eigenvalues returns exactly the top-100 eigenvalues of the
full covariance, so "100 PCs, top-100 fit" = window ranks 1-100 of the full spectrum (checked numerically below).

Analysis choices (fixed before looking at results):
  Decoder: Gemma Scope layer 12, 65k, L0 = 72 (Li's SAE); centred and uncentred; every window below.
  Activations, layer 12, our 1M cached Pile tokens (BOS dropped), five versions from one pass:
    raw_centred (= Sigma), raw_uncentred (second moment), corr (per-dimension standardised, centred),
    unit_centred and unit_uncentred (each token scaled to unit norm first).
  Token count: random subsets N in {256, 512, 1024, 2048, 4096, 16384, 65536} (3 draws) and the first N
    contiguous tokens, raw_centred and unit_centred.
  "Activations" = SAE latent activations: JumpReLU codes of the same SAE on 8,192 random tokens, centred
    covariance via the N x N Gram matrix, raw codes and binary firing indicators.
  Layer: all 26 layers from the 32,736-token cache (gemma-2-2b-all, fp16), raw_centred / unit_centred / corr,
    against Li's notebook SAE per layer (65k, their L0 list) from results/phase1/layers.json, plus the 1M Sigma.
  Windows: ranks 1-100 (Li text), plus our four 0-based windows (3,50), (10,150), (30,450), (100,1000).
  Reported quantity per protocol: activation slope, decoder slope, gamma = dec / act, and the sign of
  "activation flatter than decoder" (Li's ordering) i.e. |act slope| < |dec slope|.
Writes results/review/li_protocol.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

MAIN = ROOT if (ROOT / "cache").exists() else ROOT.parents[2]
CACHE = MAIN / "cache"
OUT = ROOT / "results" / "review" / "li_protocol.json"
WINDOWS = [(0, 100), (3, 50), (10, 150), (30, 450), (100, 1000)]
LI_MAP = {0: 73, 1: 121, 2: 77, 3: 89, 4: 89, 5: 105, 6: 107, 7: 107, 8: 111, 9: 118, 10: 128, 11: 70, 12: 72,
          13: 75, 14: 73, 15: 127, 16: 128, 17: 125, 18: 116, 19: 115, 20: 114, 21: 111, 22: 116, 23: 123,
          24: 124, 25: 93}
CH = 32768


def wkey(w):
    return f"ranks{w[0] + 1}-{w[1]}"


def slopes(eig):
    eig = np.asarray(eig, float)
    out = {}
    for w in WINDOWS:
        if w[1] <= len(eig) and np.all(eig[w[0]:w[1]] > 0):
            out[wkey(w)] = slope_from_eigenvalues(eig, w)["slope"]
        else:
            out[wkey(w)] = None
    return out


def eig_desc(C):
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def cov_spectra(X, kinds=("raw_centred", "unit_centred")):
    """Spectra of an in-memory (n, d) block under several preprocessings (float64)."""
    X = np.asarray(X, np.float64)
    res = {}
    for k in kinds:
        Y = X / np.linalg.norm(X, axis=1, keepdims=True) if k.startswith("unit") else X
        if k == "corr":
            C = np.corrcoef(Y, rowvar=False)
        elif k.endswith("uncentred"):
            C = Y.T @ Y / (len(Y) - 1)
        else:
            C = np.cov(Y, rowvar=False)
        e = eig_desc(C)
        res[k] = np.where(e > 1e-12 * e[0], e, np.nan)
    return res


def full_pass(acts):
    """One pass over all tokens: shifted sums and Grams for raw and unit-norm tokens."""
    n, d = acts.shape
    m0 = np.asarray(acts[:CH], np.float64).mean(0)
    x0 = np.asarray(acts[:CH], np.float64)
    u0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).mean(0)
    acc = {"raw": [np.zeros(d), np.zeros((d, d)), m0], "unit": [np.zeros(d), np.zeros((d, d)), u0]}
    norms = []
    t0 = time.time()
    for i in range(0, n, CH):
        x = np.asarray(acts[i:i + CH], np.float32)
        nr = np.linalg.norm(x, axis=1)
        norms.append(nr)
        for key, y in (("raw", x), ("unit", x / nr[:, None])):
            s, G, m = acc[key]
            yc = y - m.astype(np.float32)
            s += yc.sum(0, dtype=np.float64)
            G += (yc.T @ yc).astype(np.float64)
        if i // CH % 8 == 0:
            print(f"  pass {i + len(x):>8}/{n}  {time.time() - t0:.0f}s", flush=True)
    out = {}
    for key in acc:
        s, G, m = acc[key]
        mean = m + s / n
        cen = (G - np.outer(s, s) / n) / (n - 1)
        unc = (G + np.outer(s, m) + np.outer(m, s) + n * np.outer(m, m)) / (n - 1)
        out[key] = (mean, cen, unc)
    return out, np.concatenate(norms)


def jumprelu_codes(X, W_enc, b_enc, thr):
    pre = X @ W_enc + b_enc
    return np.where(pre > thr, pre, 0.0).astype(np.float32)


def gram_spectrum(F, top=1100):
    """Top eigenvalues of the centred covariance of F (n x m, m large) via the n x n Gram."""
    n = len(F)
    K = (F @ F.T).astype(np.float64)  # float32 Gram (n x n), then double-centring in float64
    r = K.mean(0)
    K = (K - r[None, :] - r[:, None] + r.mean()) / (n - 1)
    e = eig_desc(K)[:top]
    return np.where(e > 1e-12 * e[0], e, np.nan)


def main():
    t0 = time.time()
    out = {"notes": __doc__.split("Analysis choices")[0].strip()}
    acts = np.load(CACHE / "acts" / "gemma-2-2b" / "blocks.12.hook_resid_post" / "acts.npy", mmap_mode="r")
    n, d = acts.shape
    wdir = CACHE / "gemmascope_wdec" / "layer_12" / "width_65k" / "average_l0_72"
    W = np.load(wdir / "W_dec.npy").astype(np.float64)

    # ---- decoder cloud, Li's SAE ----
    dec = {"centred": eig_desc(np.cov(W, rowvar=False)), "uncentred": eig_desc(W.T @ W / (len(W) - 1))}
    dec_sl = {k: slopes(v) for k, v in dec.items()}
    out["decoder_L12_65k_L072"] = dec_sl
    print("decoder slopes", {k: {w: round(s, 3) for w, s in v.items()} for k, v in dec_sl.items()})

    # PCA(100) equivalence check on a 4096-token random subset
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, 4096, replace=False))
    Xs = np.asarray(acts[idx], np.float64)
    pca = PCA(n_components=100).fit(Xs)
    e_full = eig_desc(np.cov(Xs, rowvar=False))[:100]
    out["pca100_equivalence_max_rel_diff"] = float(np.max(np.abs(pca.explained_variance_ - e_full) / e_full))
    print("PCA(100) vs top-100 eig max rel diff", out["pca100_equivalence_max_rel_diff"])

    # ---- full 1M pass, five preprocessings ----
    print("full pass over", n, "tokens")
    fp, norms = full_pass(acts)
    mean_r, cen_r, unc_r = fp["raw"]
    _, cen_u, unc_u = fp["unit"]
    sd = np.sqrt(np.diag(cen_r))
    spec = {"raw_centred": eig_desc(cen_r), "raw_uncentred": eig_desc(unc_r),
            "corr": eig_desc(cen_r / np.outer(sd, sd)), "unit_centred": eig_desc(cen_u),
            "unit_uncentred": eig_desc(unc_u)}
    np.save(ROOT / "results" / "review" / "li_protocol_L12_spectra.npy", np.stack([spec[k] for k in spec]))
    full = {}
    for k, e in spec.items():
        s = slopes(e)
        full[k] = {"slopes": s, "top1_share": float(e[0] / e.sum()), "top10_share": float(e[:10].sum() / e.sum()),
                   "gamma_vs_dec_centred": {w: (dec_sl["centred"][w] / s[w] if s[w] else None) for w in s},
                   "act_flatter_than_dec": {w: (abs(s[w]) < abs(dec_sl["centred"][w]) if s[w] else None) for w in s}}
        print(f"  {k:<15} " + "  ".join(f"{w}:{v:+.3f}" for w, v in s.items() if v is not None)
              + f"  top1 {full[k]['top1_share']:.3f}")
    out["L12_1M"] = {"n_tokens": int(n), "token_norm": {"mean": float(norms.mean()), "median": float(np.median(norms)),
                                                       "p99": float(np.percentile(norms, 99)), "max": float(norms.max())},
                     "mean_norm_over_rms_norm": float(np.linalg.norm(mean_r) / np.sqrt((norms ** 2).mean())),
                     "spectra": full}
    top_dims = np.argsort(np.diag(cen_r))[::-1][:5]
    out["L12_1M"]["top_variance_dims"] = {int(i): float(np.diag(cen_r)[i] / np.trace(cen_r)) for i in top_dims}

    # ---- token count ----
    print("token-count subsets")
    tc = {}
    for N in [256, 512, 1024, 2048, 4096, 16384, 65536]:
        rows = {"random": [], "contiguous": None}
        for draw in range(3):
            idx = np.sort(np.random.default_rng(100 + draw).choice(n, N, replace=False))
            sp = cov_spectra(acts[idx])
            rows["random"].append({k: slopes(v) for k, v in sp.items()})
        sp = cov_spectra(acts[:N])
        rows["contiguous"] = {k: slopes(v) for k, v in sp.items()}
        agg = {}
        for k in ("raw_centred", "unit_centred"):
            agg[k] = {}
            for w in map(wkey, WINDOWS):
                v = [r[k][w] for r in rows["random"] if r[k][w] is not None]
                agg[k][w] = [float(np.mean(v)), float(np.std(v, ddof=1))] if len(v) > 1 else None
        rows["random_mean_sd"] = agg
        tc[N] = rows
        print(f"  N={N:<6} raw ranks1-100 {agg['raw_centred']['ranks1-100']}  11-150 {agg['raw_centred']['ranks11-150']}"
              f"  unit 1-100 {agg['unit_centred']['ranks1-100']}  contiguous raw 1-100 {rows['contiguous']['raw_centred']['ranks1-100']}")
    out["L12_token_count"] = tc

    # ---- SAE latent activations (codes) ----
    print("JumpReLU codes")
    W_enc = np.load(wdir / "W_enc.npy").astype(np.float32)
    b_enc, thr = np.load(wdir / "b_enc.npy").astype(np.float32), np.load(wdir / "threshold.npy").astype(np.float32)
    idx = np.sort(np.random.default_rng(7).choice(n, 8192, replace=False))
    X = np.asarray(acts[idx], np.float32)
    F = np.concatenate([jumprelu_codes(X[i:i + 1024], W_enc, b_enc, thr) for i in range(0, len(X), 1024)])
    del W_enc
    l0 = float((F > 0).sum(1).mean())
    codes = {"raw_codes": gram_spectrum(F), "firing_indicators": gram_spectrum((F > 0).astype(np.float32))}
    del F
    out["L12_sae_codes"] = {"n_tokens": 8192, "measured_l0": l0,
                            "slopes": {k: slopes(v) for k, v in codes.items()},
                            "gamma_vs_dec_centred": {k: {w: (dec_sl["centred"][w] / s if s else None)
                                                         for w, s in slopes(v).items()} for k, v in codes.items()}}
    print("  L0", l0, {k: {w: (round(s, 3) if s else None) for w, s in v.items()} for k, v in out["L12_sae_codes"]["slopes"].items()})

    # ---- layers (32,736-token cache) ----
    print("layers")
    lay = json.load(open(ROOT / "results" / "phase1" / "layers.json"))["results"]
    s1m = np.load(ROOT / "results" / "test1" / "sigma_eigenvalues_1m.npy")
    per = {}
    for L in range(26):
        A = np.load(CACHE / "acts" / "gemma-2-2b-all" / f"L{L}.npy")
        sp = cov_spectra(A, kinds=("raw_centred", "unit_centred", "corr"))
        r = lay.get(f"layer_{L}/width_65k/average_l0_{LI_MAP[L]}")
        dsl = None
        if r is not None:
            dsl = {wkey(tuple(x["fit_range"])): x["slope"] for x in r["windows"]}
            # ranks 1-100 for the decoder needs the spectrum; use the stored eigenvalues if present
            e = ROOT / "results" / "phase1" / f"eig_layer_{L}__width_65k__average_l0_{LI_MAP[L]}.npy"
            if e.exists():
                dsl["ranks1-100"] = slope_from_eigenvalues(np.load(e), (0, 100))["slope"]
        per[L] = {"act_32k": {k: slopes(v) for k, v in sp.items()}, "sigma_1m": slopes(s1m[L]),
                  "decoder_li_sae": dsl, "li_l0": LI_MAP[L]}
        a = per[L]["act_32k"]["raw_centred"]
        print(f"  L{L:<2} act raw 1-100 {a['ranks1-100']:+.3f} 11-150 {a['ranks11-150']:+.3f}  unit 1-100 "
              f"{per[L]['act_32k']['unit_centred']['ranks1-100']:+.3f}  corr 1-100 {per[L]['act_32k']['corr']['ranks1-100']:+.3f}"
              f"  dec {None if dsl is None else {k: round(v, 3) for k, v in dsl.items() if k in ('ranks1-100', 'ranks11-150')}}")
    out["layers"] = per

    # ---- summary: when is activation flatter than decoder? ----
    summ = []
    for L, r in per.items():
        if r["decoder_li_sae"] is None:
            continue
        for k, s in list(r["act_32k"].items()) + [("sigma_1m", r["sigma_1m"])]:
            for w in ("ranks1-100", "ranks11-150"):
                dv = r["decoder_li_sae"].get(w)
                if dv is None or s[w] is None:
                    continue
                summ.append({"layer": L, "act": k, "window": w, "act_slope": s[w], "dec_slope": dv,
                             "gamma": dv / s[w], "act_flatter": abs(s[w]) < abs(dv)})
    out["layer_summary"] = summ
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
