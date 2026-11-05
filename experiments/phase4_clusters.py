"""Phase 4: cluster structure of real decoder-direction clouds (Gemma Scope, residual stream).

Phase 3 found that planted point features (random directions) give decoders much flatter than Sigma, so Gemma's
gamma(k) needs structure in the decoder cloud itself. Phase 4 measures that structure directly.

Stage `l12` (Gemma Scope layer 12):
  A. Cluster the unit decoder directions of the 16k, L0 = 82 SAE with four methods (spherical k-means, Ward on
     unit vectors, average-linkage cosine, spectral on a symmetric 20-NN cosine graph), K in {64, 256, 1024}
     (spectral at 64 and 256 only). Method and seed sensitivity: adjusted Rand index (ARI) and normalised mutual
     information (NMI) between labelings.
  B. Cluster -> eigenvalue map. With cluster means m_c, the cloud covariance splits exactly into between-cluster
     B = sum_c n_c (m_c - mu)(m_c - mu)^T / (N - 1) and within-cluster W = C - B. Report the eigen-slopes of C, B
     and W in the pre-registered windows, the trace share tr(B)/tr(C), and per window the mean fraction of C's
     top eigenvalues that B explains (v_j^T B v_j / lambda_j). References: random unit vectors (isotropic), our own
     TopK SAEs at layer 12 (w4096, k40) on real activations and on their Gaussian twin.
  C. Allocation law in the wild. Latent firing frequencies from 1M tokens (phase1_freqs.py) for the 16k-131k SAEs
     closest to L0 = 70. Per k-means cluster: n_c latents and summed frequency P_c. Theory: n_c ∝ P_c^(1/(1+b)),
     i.e. the mean latent frequency f_c = P_c / n_c rises as n_c^b. Fit log f_c on log n_c (clusters with
     n_c >= 2), against a null that permutes frequencies across latents (keeps cluster sizes; null slope ~0).

Stage `layers` (all 26 layers, 16k SAE closest to L0 = 70):
  D. Per layer, spherical k-means at K = width/16 and width/64: between-cluster share per window, cluster-size
     Zipf slope, median within-cluster cosine to centroid, and the median participation ratio of clusters
     subsampled to 16 latents (PR16, matched n so cluster size cannot drive it). Compared with the layer profile
     of the eigen-slope. Our Gaussian-null TopK SAEs (w4096, layers 0, 3, 7, 12, 13, 17, 21, 24) are the reference:
     if they show the same layer profile, it comes from Sigma.

Stage `layers_null`:
  E. The same per-layer statistics on a covariance-matched Gaussian cloud (N(mu, C) of the layer's decoder cloud,
     rows unit-normalised, same N). A more anisotropic cloud makes any clustering look tighter (higher between-share,
     higher cosine, lower PR), so only the excess real - matched measures cluster structure beyond anisotropy.

Writes results/phase4/{l12,layers,layers_null}.json. Usage: python experiments/phase4_clusters.py --stage l12|layers|layers_null
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import slope_from_eigenvalues  # noqa: E402

OUT = ROOT / "results" / "phase4"
WDEC = ROOT / "cache" / "gemmascope_wdec"
NULL = ROOT / "cache" / "saes_gemma_null"
FIRES = ROOT / "results" / "phase1"
WINDOWS = [(3, 50), (10, 150), (30, 450), (100, 1000)]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def git_hash():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def unit(W):
    W = np.asarray(W, dtype=np.float32)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def gemma_wdec(layer, width, l0):
    return unit(np.load(WDEC / f"layer_{layer}" / f"width_{width}" / f"average_l0_{l0}" / "W_dec.npy"))


def own_sae(layer, cond):
    d = NULL / f"gemma-2-2b__blocks.{layer}.hook_resid_post__{cond}__w4096__k40__s0"
    fires = np.load(d / "fires.npy")
    return unit(np.load(d / "W_dec.npy")[fires > 0]), int((fires == 0).sum())


# ---------------------------------------------------------------- clustering

def skmeans(X, K, seed, iters=100):
    """Spherical k-means (cosine), k-means++ init on 1 - cos. Returns labels (n,)."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X).to(DEV)
    n = len(X)
    idx = [int(torch.randint(n, (1,), generator=g))]
    dmin = (1 - Xt @ Xt[idx[0]]).clamp(min=0)
    for _ in range(K - 1):
        p = (dmin ** 2).cpu().double()
        j = int(torch.multinomial(p / p.sum(), 1, generator=g))
        idx.append(j)
        dmin = torch.minimum(dmin, (1 - Xt @ Xt[j]).clamp(min=0))
    C = Xt[idx].clone()
    lab = None
    for _ in range(iters):
        sim = Xt @ C.T
        new = sim.argmax(1)
        if lab is not None and bool((new == lab).all()):
            break
        lab = new
        cnt = torch.bincount(lab, minlength=K)
        S = torch.zeros_like(C).index_add_(0, lab, Xt)
        empty = (cnt == 0).nonzero().flatten()
        if len(empty):  # reseed empty clusters at the worst-fit points
            worst = sim.max(1).values.argsort()[:len(empty)]
            S[empty] = Xt[worst]
        C = S / S.norm(dim=1, keepdim=True)
    return lab.cpu().numpy()


def knn_graph(X, k=20):
    import scipy.sparse as sp
    Xt = torch.from_numpy(X).to(DEV)
    rows, cols, vals = [], [], []
    for i in range(0, len(X), 4096):
        s = Xt[i:i + 4096] @ Xt.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -2
        v, j = s.topk(k, dim=1)
        rows.append(np.repeat(np.arange(i, i + len(s)), k))
        cols.append(j.cpu().numpy().ravel())
        vals.append(v.clamp(min=0).cpu().numpy().ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(X),) * 2)
    return A.maximum(A.T)


def cluster(X, method, K, seed=0):
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    if method == "kmeans":
        return skmeans(X, K, seed)
    if method == "ward":
        return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
    if method == "average":
        return AgglomerativeClustering(n_clusters=K, linkage="average", metric="cosine").fit_predict(X)
    if method == "spectral":
        return SpectralClustering(n_clusters=K, affinity="precomputed", assign_labels="cluster_qr",
                                  random_state=seed).fit_predict(knn_graph(X))
    raise ValueError(method)


# ---------------------------------------------------------------- statistics

def size_stats(lab):
    n = np.sort(np.bincount(lab))[::-1]
    n = n[n > 0]
    r = np.arange(1, len(n) + 1)
    top = slice(0, max(3, len(n) // 2))
    zipf = float(np.polyfit(np.log(r[top]), np.log(n[top]), 1)[0])
    return {"K_nonempty": int(len(n)), "largest": int(n[0]), "median": float(np.median(n)),
            "singletons": int((n == 1).sum()), "frac_in_largest": float(n[0] / n.sum()), "zipf_top_half": zipf}


def decompose(X, lab):
    """Exact between/within split of the centred covariance (ddof = 1), eigen-slopes and explained fractions."""
    Xd = torch.from_numpy(X).double()  # float64 on CPU (MPS has no float64)
    N = len(X)
    mu = Xd.mean(0)
    C = ((Xd - mu).T @ (Xd - mu)) / (N - 1)
    K = int(lab.max()) + 1
    labt = torch.from_numpy(lab).long()
    S = torch.zeros(K, X.shape[1], dtype=torch.float64).index_add_(0, labt, Xd)
    cnt = torch.bincount(labt, minlength=K).double()
    keep = cnt > 0
    M = S[keep] / cnt[keep, None] - mu
    B = (M.T * cnt[keep]) @ M / (N - 1)
    W = C - B
    ec, V = torch.linalg.eigh(C)
    ec, V = ec.flip(0), V.flip(1)
    eb = torch.linalg.eigvalsh(B).flip(0)
    ew = torch.linalg.eigvalsh(W).flip(0)
    expl = ((V.T @ B) * V.T).sum(1) / ec  # v_j^T B v_j / lambda_j
    out = {"trace_share_between": float(B.trace() / C.trace()), "windows": {}}
    for w in WINDOWS:
        out["windows"][str(w)] = {
            "slope_C": slope_from_eigenvalues(ec.numpy(), w)["slope"],
            "slope_B": slope_from_eigenvalues(eb.clamp(min=1e-12).numpy(), w)["slope"] if K > w[1] else None,
            "slope_W": slope_from_eigenvalues(ew.clamp(min=1e-12).numpy(), w)["slope"],
            "explained_by_between": float(expl[w[0]:w[1]].mean())}
    return out, ec.numpy()


def cluster_shape(X, lab, n_sub=16, seed=0):
    """Median within-cluster cosine to the (normalised) centroid, and median PR of clusters subsampled to n_sub."""
    rng = np.random.default_rng(seed)
    cos, pr = [], []
    for c in np.unique(lab):
        Z = X[lab == c]
        m = Z.mean(0)
        cos.extend((Z @ (m / np.linalg.norm(m))).tolist())
        if len(Z) >= n_sub:
            Zs = Z[rng.choice(len(Z), n_sub, replace=False)].astype(np.float64)
            Zs -= Zs.mean(0)
            e = np.linalg.eigvalsh(Zs @ Zs.T)  # same nonzero spectrum as the d x d covariance (up to scale)
            e = np.clip(e, 0, None)
            pr.append(float(e.sum() ** 2 / (e ** 2).sum()))
    return {"median_cos_to_centroid": float(np.median(cos)), "median_pr16": float(np.median(pr)) if pr else None,
            "clusters_for_pr": len(pr)}


def allocation(lab, freq, n_perm=1000, seed=0):
    """Slope of log mean-frequency on log cluster size (n_c >= 2), with a frequency-permutation null."""
    def fit(f):
        K = lab.max() + 1
        n = np.bincount(lab, minlength=K)
        P = np.bincount(lab, weights=f, minlength=K)
        ok = (n >= 2) & (P > 0)
        x, y = np.log(n[ok]), np.log(P[ok] / n[ok])
        s, a = np.polyfit(x, y, 1)
        return float(s), float(a), int(ok.sum()), float(np.corrcoef(x, y)[0, 1])
    s, a, m, r = fit(freq)
    rng = np.random.default_rng(seed)
    null = np.array([fit(rng.permutation(freq))[0] for _ in range(n_perm)])
    return {"slope_logf_on_logn": s, "intercept": a, "clusters": m, "r": r,
            "implied_beta_n_on_P": 1 / (1 + s), "implied_b": s,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "z": float((s - null.mean()) / null.std())}


# ---------------------------------------------------------------- stages

def stage_l12():
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
    res = {"git": git_hash(), "windows": WINDOWS, "A": {}, "B": {}, "C": {}}
    X = gemma_wdec(12, "16k", 82)
    print(f"L12 16k L0=82: {X.shape}", flush=True)

    # A: methods and seeds
    labs = {}
    for K in (64, 256, 1024):
        for m in ("kmeans", "ward", "average", "spectral"):
            if m == "spectral" and K > 256:
                continue
            t0 = time.time()
            labs[(m, K)] = cluster(X, m, K)
            res["A"][f"{m}|{K}"] = {**size_stats(labs[(m, K)]), "seconds": round(time.time() - t0, 1)}
            print(f"A {m:<9} K={K:<5} {res['A'][f'{m}|{K}']}", flush=True)
        for s in (1, 2):
            labs[(f"kmeans_s{s}", K)] = skmeans(X, K, s)
        ms = [m for (m, k) in labs if k == K]
        res["A"][f"agreement|{K}"] = {f"{a}~{b}": {"ari": float(ari(labs[(a, K)], labs[(b, K)])),
                                                   "nmi": float(nmi(labs[(a, K)], labs[(b, K)]))}
                                      for i, a in enumerate(ms) for b in ms[i + 1:]}
        print(f"A agreement K={K}: " + ", ".join(f"{p} {v['ari']:.2f}/{v['nmi']:.2f}"
                                                 for p, v in res["A"][f"agreement|{K}"].items()), flush=True)

    # B: cluster -> eigenvalue map, Gemma 16k and references
    rng = np.random.default_rng(0)
    clouds = {"gemma_16k_L0_82": X,
              "isotropic_16k": unit(rng.standard_normal(X.shape).astype(np.float32))}
    for cond in ("real", "gaussian"):
        clouds[f"topk_w4096_{cond}"], dead = own_sae(12, cond)
        res["B"][f"topk_w4096_{cond}|dead"] = dead
    for name, Y in clouds.items():
        for frac in (64, 16):
            K = len(Y) // frac
            lab = labs[("kmeans", K)] if name == "gemma_16k_L0_82" and ("kmeans", K) in labs else skmeans(Y, K, 0)
            d, _ = decompose(Y, lab)
            res["B"][f"{name}|kmeans|{K}"] = {**d, **cluster_shape(Y, lab), **size_stats(lab)}
            print(f"B {name:<18} K={K:<5} share {d['trace_share_between']:.3f} " + " ".join(
                f"{w}: C {v['slope_C']:.3f} B {v['slope_B'] if v['slope_B'] is None else round(v['slope_B'], 3)} "
                f"W {v['slope_W']:.3f} expl {v['explained_by_between']:.2f}" for w, v in d["windows"].items()),
                flush=True)
    for m in ("ward", "average", "spectral"):  # method sensitivity of the map on Gemma
        K = 256
        d, _ = decompose(X, labs[(m, K)])
        res["B"][f"gemma_16k_L0_82|{m}|{K}"] = d
        print(f"B gemma {m:<8} K={K} share {d['trace_share_between']:.3f} " + " ".join(
            f"{w}: expl {v['explained_by_between']:.2f}" for w, v in d["windows"].items()), flush=True)

    # C: allocation law across widths
    for width, l0 in (("16k", 82), ("32k", 76), ("65k", 72), ("131k", 67)):
        fires = np.load(FIRES / f"fires_layer_12__width_{width}__average_l0_{l0}.npy")
        alive = fires > 0
        Y = gemma_wdec(12, width, l0)[alive]
        freq = fires[alive] / 1_000_000
        for frac in (256, 64, 16):
            K = len(Y) // frac
            lab = skmeans(Y, K, 0)
            res["C"][f"{width}|L0_{l0}|K{K}"] = {"n_alive": int(alive.sum()), "K": K, **allocation(lab, freq)}
            v = res["C"][f"{width}|L0_{l0}|K{K}"]
            print(f"C {width:<5} L0={l0:<3} K={K:<5} slope {v['slope_logf_on_logn']:+.3f} (null {v['null_mean']:+.3f}"
                  f"±{v['null_sd']:.3f}, z {v['z']:.1f}) r {v['r']:+.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "l12.json", "w"), indent=1, default=float)


def matched_cloud(X, seed=0):
    """Unit-normalised Gaussian cloud with the same centred covariance and mean as X (same N): anisotropy-matched null."""
    Xd = X.astype(np.float64)
    mu = Xd.mean(0)
    C = np.cov(Xd, rowvar=False)
    e, V = np.linalg.eigh(C)
    R = V * np.sqrt(np.clip(e, 0, None))
    Z = np.random.default_rng(seed).standard_normal(X.shape) @ R.T + mu
    return unit(Z)


def stage_layers_null():
    """E: per layer, the same clustering on a covariance-matched Gaussian cloud; excess = real - matched."""
    lay = json.load(open(FIRES / "layers.json"))["results"]
    res = {"git": git_hash(), "windows": WINDOWS, "matched": {}}
    for L in range(26):
        l0 = min((abs(math.log(v["l0"] / 70)), v["l0"]) for v in lay.values()
                 if v["layer"] == L and v["width"] == "16k")[1]
        Y = matched_cloud(gemma_wdec(L, "16k", l0))
        for frac in (64, 16):
            K = len(Y) // frac
            lab = skmeans(Y, K, 0)
            d, _ = decompose(Y, lab)
            res["matched"][f"{L}|{K}"] = {"layer": L, "l0": l0, "K": K, **d, **cluster_shape(Y, lab), **size_stats(lab)}
            v = res["matched"][f"{L}|{K}"]
            print(f"matched L{L:<2} K={K:<5} share {v['trace_share_between']:.3f} cos {v['median_cos_to_centroid']:.3f} "
                  f"PR16 {v['median_pr16']:.2f} " + " ".join(
                      f"{w}: C {x['slope_C']:.3f} expl {x['explained_by_between']:.2f}" for w, x in v["windows"].items()),
                  flush=True)
    json.dump(res, open(OUT / "layers_null.json", "w"), indent=1, default=float)


def stage_layers():
    lay = json.load(open(FIRES / "layers.json"))["results"]
    res = {"git": git_hash(), "windows": WINDOWS, "gemma": {}, "null": {}}
    for L in range(26):
        cands = [(abs(math.log(v["l0"] / 70)), v["l0"]) for k, v in lay.items()
                 if v["layer"] == L and v["width"] == "16k"]
        l0 = min(cands)[1]
        X = gemma_wdec(L, "16k", l0)
        for frac in (64, 16):
            K = len(X) // frac
            lab = skmeans(X, K, 0)
            d, _ = decompose(X, lab)
            res["gemma"][f"{L}|{K}"] = {"layer": L, "l0": l0, "K": K, **d, **cluster_shape(X, lab), **size_stats(lab)}
            v = res["gemma"][f"{L}|{K}"]
            print(f"L{L:<2} L0={l0:<3} K={K:<5} share {v['trace_share_between']:.3f} cos {v['median_cos_to_centroid']:.3f} "
                  f"PR16 {v['median_pr16']:.2f} zipf {v['zipf_top_half']:+.2f} " + " ".join(
                      f"{w}: C {x['slope_C']:.3f} expl {x['explained_by_between']:.2f}" for w, x in v["windows"].items()),
                  flush=True)
    for L in (0, 3, 7, 12, 13, 17, 21, 24):
        for cond in ("gaussian", "real"):
            if not (NULL / f"gemma-2-2b__blocks.{L}.hook_resid_post__{cond}__w4096__k40__s0").exists():
                continue
            X, dead = own_sae(L, cond)
            for frac in (64, 16):
                K = len(X) // frac
                lab = skmeans(X, K, 0)
                d, _ = decompose(X, lab)
                res["null"][f"{cond}|{L}|{K}"] = {"layer": L, "cond": cond, "dead": dead, "K": K, **d,
                                                  **cluster_shape(X, lab), **size_stats(lab)}
                v = res["null"][f"{cond}|{L}|{K}"]
                print(f"{cond} L{L:<2} K={K:<4} share {v['trace_share_between']:.3f} cos {v['median_cos_to_centroid']:.3f} "
                      f"PR16 {v['median_pr16']:.2f} " + " ".join(
                          f"{w}: C {x['slope_C']:.3f} expl {x['explained_by_between']:.2f}"
                          for w, x in v["windows"].items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "layers.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l12", "layers", "layers_null"], required=True)
    a = ap.parse_args()
    {"l12": stage_l12, "layers": stage_layers, "layers_null": stage_layers_null}[a.stage]()
