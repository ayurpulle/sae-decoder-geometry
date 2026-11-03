"""Eigenspectrum geometry of SAE decoder-direction point clouds.

Phase 0 matches the computation in ejmichaud/feature-geometry
(galaxy/PointCloudStructure_Galaxy.ipynb, cells 15/18):
    eig = sort_desc(eigvalsh(np.cov(W_dec, rowvar=False)))   # centred, ddof=1
    slope = linregress(log10(rank)[10:150], log10(eig)[10:150]).slope
i.e. rank indices 10..149 (0-based), which is ranks 11..150 (1-based).
Gemma Scope W_dec rows are not renormalised there; we keep that as the default
(`normalize=False`) and report row-norm stats so it is visible whether it matters.

All later phases must call `eigen_slope` rather than reimplementing this.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import linregress

PAPER_FIT_RANGE = (10, 150)  # 0-based [start, end) slice into rank-sorted eigenvalues


CHUNK = 65536


def _eigvals(X, center: bool, mask=None, normalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Descending covariance eigenvalues (ddof=1, float64) and the kept rows' norms.

    Accumulates sum and Gram matrix over row chunks so 1M-row float32 (or memmapped)
    W_dec never needs a float64 copy. `mask` keeps rows; `normalize` rescales kept rows to unit norm.
    """
    d = X.shape[1]
    s = np.zeros(d)
    G = np.zeros((d, d))
    norms = []
    for i in range(0, X.shape[0], CHUNK):
        c = np.asarray(X[i:i + CHUNK], dtype=np.float64)
        if mask is not None:
            c = c[mask[i:i + CHUNK]]
        nrm = np.linalg.norm(c, axis=1)
        norms.append(nrm)
        if normalize:
            c = c / nrm[:, None]
        s += c.sum(0)
        G += c.T @ c
    n = sum(len(x) for x in norms)
    if center:
        G -= np.outer(s, s) / n
    eig = np.sort(np.linalg.eigvalsh(G / (n - 1)))[::-1]
    return eig, np.concatenate(norms)


def _gaussian_eigvals(n: int, d: int, sigma: float, center: bool, seed: int) -> np.ndarray:
    """Eigenvalues of an isotropic N(0, sigma^2 I) cloud of shape (n, d), generated in chunks."""
    rng = np.random.default_rng(seed)
    gen = (rng.standard_normal((min(CHUNK, n - i), d), dtype=np.float32) * np.float32(sigma)
           for i in range(0, n, CHUNK))
    return _eigvals(_Chunks(gen, n, d), center)[0]


class _Chunks:
    """Minimal sliceable wrapper so _eigvals can consume a chunk generator in order."""

    def __init__(self, gen, n, d):
        self._gen, self.shape = gen, (n, d)

    def __getitem__(self, _):
        return next(self._gen)


def _fit(eig: np.ndarray, fit_range: tuple[int, int]) -> tuple[float, float, float]:
    """Slope, intercept and R^2 of log10(eig) vs log10(rank) over a 0-based [start, end) slice."""
    lo, hi = fit_range
    x = np.log10(np.arange(1, len(eig) + 1))[lo:hi]
    y = np.log10(eig[lo:hi])
    r = linregress(x, y)
    return float(r.slope), float(r.intercept), float(r.rvalue ** 2)


def eigen_slope(
    W_dec,
    fit_range: tuple[int, int] = PAPER_FIT_RANGE,
    center: bool = True,
    normalize: bool = False,
    mask: np.ndarray | None = None,
    baseline_seed: int = 0,
) -> dict:
    """Power-law slope of the decoder-direction covariance eigenspectrum.

    W_dec: (n_latents, d_model) array, memmap or torch tensor, one row per latent.
    fit_range: 0-based [start, end) indices into the descending eigenvalues.
    center: centre the cloud before the covariance (paper: True).
    normalize: rescale rows to unit norm first (paper: False; Gemma Scope rows are ~unit already).
    mask: optional boolean (n_latents,) of latents to KEEP (e.g. exclude dead latents).
    baseline_seed: seed for the matched isotropic Gaussian (same n, d, and total variance).
    """
    if hasattr(W_dec, "detach"):
        W_dec = W_dec.detach().cpu().float().numpy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    d = W_dec.shape[1]

    eig, norms = _eigvals(W_dec, center, mask=mask, normalize=normalize)
    n = len(norms)
    slope, intercept, r2 = _fit(eig, fit_range)

    # Matched isotropic Gaussian: same (n, d), per-coordinate variance = trace / d.
    sigma = np.sqrt(eig.sum() / d)
    g_eig = _gaussian_eigvals(n, d, sigma, center, baseline_seed)
    g_slope, g_intercept, g_r2 = _fit(g_eig, fit_range)

    return {
        "n_latents": n,
        "d_model": d,
        "fit_range": list(fit_range),
        "center": center,
        "normalize": normalize,
        "row_norm": {"mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max())},
        "eigenvalues": eig,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "gaussian": {"eigenvalues": g_eig, "slope": g_slope, "intercept": g_intercept, "r2": g_r2, "sigma": float(sigma)},
    }


def slope_from_eigenvalues(eig: np.ndarray, fit_range: tuple[int, int]) -> dict:
    """Refit an already-computed spectrum on another range (for fit-range robustness checks)."""
    slope, intercept, r2 = _fit(np.asarray(eig), fit_range)
    return {"fit_range": list(fit_range), "slope": slope, "intercept": intercept, "r2": r2}


def cloud_cov(W, mask=None, normalize: bool = False) -> np.ndarray:
    """Centred covariance (ddof=1, float64) of the decoder rows, accumulated in chunks like _eigvals."""
    d = W.shape[1]
    s, G, n = np.zeros(d), np.zeros((d, d)), 0
    for i in range(0, W.shape[0], CHUNK):
        c = np.asarray(W[i:i + CHUNK], dtype=np.float64)
        if mask is not None:
            c = c[mask[i:i + CHUNK]]
        if normalize:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
        s += c.sum(0)
        G += c.T @ c
        n += len(c)
    return (G - np.outer(s, s) / n) / (n - 1)


def basis_gamma(C_dec: np.ndarray, Sigma: np.ndarray, rank_ranges=((1, 150),), subspace_dims=(10, 50)) -> dict:
    """Test C_dec ∝ Sigma^gamma as a matrix statement, without rank windows on two separately sorted spectra.

    Rotates C_dec into Sigma's eigenbasis U (descending eigenvalues lam) and regresses
    log diag(U^T C_dec U) on log lam over each 0-based [lo, hi) rank range -> gamma, R^2.
    Alignment diagnostics: share of ||U^T C_dec U||_F^2 on the diagonal (1 = perfectly co-diagonal; a random
    basis gives about 1/d), and, for each m in subspace_dims, the mean squared cosine of the principal angles
    between the top-m eigenvector subspaces of C_dec and Sigma (1 = same subspace; random ≈ m/d).
    """
    lam, U = np.linalg.eigh(Sigma)
    lam, U = lam[::-1], U[:, ::-1]
    D = U.T @ C_dec @ U
    dg = np.diag(D)
    _, Vd = np.linalg.eigh(C_dec)
    Vd = Vd[:, ::-1]
    out = {"diag_share": float((dg ** 2).sum() / (D ** 2).sum()),
           "subspace_overlap": {m: float((np.linalg.norm(U[:, :m].T @ Vd[:, :m]) ** 2) / m) for m in subspace_dims},
           "fits": []}
    for lo, hi in rank_ranges:
        ok = (dg[lo:hi] > 0) & (lam[lo:hi] > 0)
        r = linregress(np.log(lam[lo:hi][ok]), np.log(dg[lo:hi][ok]))
        out["fits"].append({"rank_range": [lo, hi], "gamma": float(r.slope), "r2": float(r.rvalue ** 2)})
    return out
