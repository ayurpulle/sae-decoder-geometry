"""Check: Li et al.'s ordering ("activation spectra decay
much slower than SAE-feature spectra") at all 26 layers on the 1M-token covariance, replacing the 32,736-token
26-layer sweep of review4_li_protocol.py.

Analysis choices (fixed before looking at results):
  Activations: results/test1/sigma_gemma-2-2b_1m.npz (key 'cov', 26 x 2304 x 2304, centred, 1M Pile tokens).
    - raw: eigenvalues of cov (= Sigma), the primary activation spectrum;
    - corr: per-dimension standardised = cov / outer(sd, sd) (the correlation matrix);
    - raw_uncentred (extra, not primary): cov + outer(mean, mean).
  Decoders (all rows, centred, eigenvalues stored by phase1b_layers.py in results/phase1/eig_*.npy):
    - 65k nearest Li et al.'s L0 per layer (their notebook's per-layer list LI_MAP; nearest named L0 if absent);
    - 16k whose named L0 is nearest 40 at that layer (ties -> the lower L0).
  Windows: ranks 1-100 and 11-150 are primary (Li's text and Li's notebook); 4-50, 31-450, 101-1000 also stored.
  Li's ordering holds at a layer when |activation slope| < |decoder slope| (activation flatter).
  Reported per (activation kind x decoder x window): number of layers (of 26) where Li's ordering holds,
  gamma = decoder slope / activation slope (min, max), and the layers where it holds.
  Check: L12 raw and corr must reproduce review4_li_protocol.py's 1M values (-0.923 and -0.446, ranks 1-100).
  Inconsistency record (thesisA lines 62/66/82): Li's notebook fits the decoder over the 0-based slice [10:150)
  (ranks 11-150); their Sec. 5.1 text fits the "100 largest eigenvalues" (ranks 1-100) when comparing activations
  with features. The SAE's release name is average_l0_72; its L0 measured on our tokens is stored by
  review4_li_protocol.py (L12_sae_codes.measured_l0). Both are copied into the output with their sources.
Writes results/review/li_ordering_1m.json.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "review" / "li_ordering_1m.json"
SIG = Path("/Users/ayurpulle/Desktop/SAE/results/test1/sigma_gemma-2-2b_1m.npz")
if not SIG.exists():
    SIG = ROOT / "results" / "test1" / "sigma_gemma-2-2b_1m.npz"
WINDOWS = [(0, 100), (3, 50), (10, 150), (30, 450), (100, 1000)]
PRIMARY = ["ranks1-100", "ranks11-150"]
LI_MAP = {0: 73, 1: 121, 2: 77, 3: 89, 4: 89, 5: 105, 6: 107, 7: 107, 8: 111, 9: 118, 10: 128, 11: 70, 12: 72,
          13: 75, 14: 73, 15: 127, 16: 128, 17: 125, 18: 116, 19: 115, 20: 114, 21: 111, 22: 116, 23: 123,
          24: 124, 25: 93}


def wkey(w):
    return f"ranks{w[0] + 1}-{w[1]}"


def slopes(e):
    return {wkey(w): slope_from_eigenvalues(e, w)["slope"] for w in WINDOWS}


def eig_desc(C):
    return np.sort(np.linalg.eigvalsh(C))[::-1]


def available(layer, width):
    pat = re.compile(rf"eig_layer_{layer}__width_{width}__average_l0_(\d+)\.npy")
    return sorted(int(m.group(1)) for p in (ROOT / "results" / "phase1").iterdir() if (m := pat.fullmatch(p.name)))


def nearest(l0s, target):
    return min(l0s, key=lambda x: (abs(np.log(x) - np.log(target)), x))


def main():
    t0 = time.time()
    z = np.load(SIG)
    cov_all, mean_all = z["cov"], z["mean"]
    out = {"notes": __doc__.split("Writes")[0].strip(), "sigma_file": str(SIG), "n_tokens": int(z["n"]),
           "layers": {}}
    for L in range(26):
        C = np.asarray(cov_all[L], np.float64)
        sd = np.sqrt(np.diag(C))
        acts = {"raw": eig_desc(C), "corr": eig_desc(C / np.outer(sd, sd)),
                "raw_uncentred": eig_desc(C + np.outer(mean_all[L], mean_all[L]))}
        a65 = available(L, "65k")
        a16 = available(L, "16k")
        d65, d16 = nearest(a65, LI_MAP[L]), nearest(a16, 40)
        decs = {"65k_li": (d65, np.load(ROOT / "results" / "phase1" / f"eig_layer_{L}__width_65k__average_l0_{d65}.npy")),
                "16k_L040": (d16, np.load(ROOT / "results" / "phase1" / f"eig_layer_{L}__width_16k__average_l0_{d16}.npy"))}
        rec = {"act": {k: slopes(v) for k, v in acts.items()},
               "act_top1_share": {k: float(v[0] / v.sum()) for k, v in acts.items()},
               "dec": {k: {"l0_named": l0, "li_l0": LI_MAP[L] if k == "65k_li" else 40, "slopes": slopes(e)}
                       for k, (l0, e) in decs.items()}}
        out["layers"][L] = rec
        print(f"L{L:<2} raw 1-100 {rec['act']['raw']['ranks1-100']:+.3f} corr 1-100 {rec['act']['corr']['ranks1-100']:+.3f}"
              f" | dec65k(L0 {d65}) {rec['dec']['65k_li']['slopes']['ranks1-100']:+.3f}"
              f" dec16k(L0 {d16}) {rec['dec']['16k_L040']['slopes']['ranks1-100']:+.3f}"
              f" | 11-150 raw {rec['act']['raw']['ranks11-150']:+.3f} corr {rec['act']['corr']['ranks11-150']:+.3f}"
              f" dec65k {rec['dec']['65k_li']['slopes']['ranks11-150']:+.3f} dec16k {rec['dec']['16k_L040']['slopes']['ranks11-150']:+.3f}",
              flush=True)

    summ = {}
    for ak in ("raw", "corr", "raw_uncentred"):
        for dk in ("65k_li", "16k_L040"):
            for w in map(wkey, WINDOWS):
                hold, gam = [], []
                for L, rec in out["layers"].items():
                    a, d = rec["act"][ak][w], rec["dec"][dk]["slopes"][w]
                    gam.append(d / a)
                    if abs(a) < abs(d):
                        hold.append(L)
                summ[f"{ak}|{dk}|{w}"] = {"n_layers": 26, "li_ordering_holds": len(hold), "layers_holding": hold,
                                         "gamma_min": float(min(gam)), "gamma_max": float(max(gam))}
    out["summary"] = summ
    print("\nLi's ordering (activation flatter than decoder): layers of 26")
    for k, v in summ.items():
        if k.split("|")[2] in PRIMARY or k.split("|")[0] == "corr":
            print(f"  {k:<40} {v['li_ordering_holds']:>2}/26  gamma {v['gamma_min']:.2f}-{v['gamma_max']:.2f}  {v['layers_holding']}")

    lp = json.load(open(ROOT / "results" / "review" / "li_protocol.json"))
    out["check_L12_vs_li_protocol"] = {
        "raw_1-100": [out["layers"][12]["act"]["raw"]["ranks1-100"], lp["L12_1M"]["spectra"]["raw_centred"]["slopes"]["ranks1-100"]],
        "corr_1-100": [out["layers"][12]["act"]["corr"]["ranks1-100"], lp["L12_1M"]["spectra"]["corr"]["slopes"]["ranks1-100"]],
        "dec_65k_L072_1-100": [out["layers"][12]["dec"]["65k_li"]["slopes"]["ranks1-100"], lp["decoder_L12_65k_L072"]["centred"]["ranks1-100"]]}
    out["inconsistency_record"] = {
        "li_decoder_fit_window": "notebook galaxy/PointCloudStructure_Galaxy.ipynb: linregress over 0-based [10:150] = ranks 11-150 (review4_li_protocol.py docstring)",
        "li_activation_vs_feature_window": "paper Sec. 5.1 text: 100 largest PCs / eigenvalues = ranks 1-100 (review4_li_protocol.py docstring)",
        "li_sae_release_name_l0": 72,
        "li_sae_measured_l0_ours": lp["L12_sae_codes"]["measured_l0"],
        "li_sae_measured_l0_source": "results/review/li_protocol.json L12_sae_codes.measured_l0 (JumpReLU on 8,192 random cached Pile tokens)"}
    print("check", out["check_L12_vs_li_protocol"], "measured L0", lp["L12_sae_codes"]["measured_l0"])
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
