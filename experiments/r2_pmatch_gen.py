"""Dictionaries for the power-matched sparse null (experiments/r2_job_pmatch.py). Pure numpy/torch, no training.

The codable null of r2_job_sparse2.py draws x = mu + sqrt(1 - rho) (s - E s) A + sqrt(rho) g with Cov(s) = I_M, so
Cov(x) = (1 - rho) A^T A + rho Sigma, and A^T A = Sigma makes Cov(x) = Sigma exactly in expectation. Its atoms are
A = R Sigma^{1/2} (R with orthonormal columns): every atom has nearly the same power ||a_j||^2 ~ tr(Sigma) / M.

Here the atoms are any M x d matrix A with A^T A = Sigma (so the covariance is still exact; the generator, M, L, the
value law and the dense share are those of the codable null). Write P = A A^T (M x M). Its nonzero eigenvalues are
those of A^T A = Sigma and its diagonal is the atom powers p_j = ||a_j||^2. By the Schur-Horn theorem a power vector p
is achievable with A^T A = Sigma exactly when p is majorized by the eigenvalues of Sigma padded with zeros:
    sum of the m largest p_j <= sum of the m largest eigenvalues, for every m, with equality at m = M.
So the head of a power distribution is limited by Sigma's top eigenvalues (one atom cannot carry more than lambda_1,
ten atoms not more than lambda_1 + ... + lambda_10), and an atom near that limit must point along the top eigenvectors.

Construction (all variants). Work in Sigma's eigenbasis (Sigma = U diag(lam) U^T).
  1. Target powers t_j (and for V2/V3 target Rayleigh quotients r_j) per atom, from the measured latents (targets()).
     The powers are scaled so sum t_j = tr Sigma and projected into the feasible set (project_majorized: the top-m
     sums are capped at (1 - 0.1 (1 - m/d)) times the eigenvalue sums, a little inside the boundary, where the head
     atoms would have to be exact eigenvectors; the excess is spread over the other atoms in proportion).
  2. Initial atoms: rows sqrt(t_j) u_j with unit directions u_j = normalise(lam^{beta_j / 2} * z_j), z_j ~ N(0, I_d)
     i.i.d. (rng seeded); beta_j = 1 for V1 (the equal-power null's Sigma^{1/2}-shaped directions), beta_j solved per
     atom so that u_j^T Sigma u_j / (tr Sigma / d) = r_j for V2/V3 (the Rayleigh quotient is increasing in beta).
  3. Covariance correction: A = A~ T with T the symmetric positive-definite solution of T S T = Sigma, S = A~^T A~:
         T = S^{-1/2} (S^{1/2} Sigma S^{1/2})^{1/2} S^{-1/2}.
     Then A^T A = T S T = Sigma exactly. T is the optimal-transport map from N(0, S) to N(0, Sigma), i.e. the smallest
     linear change of the atoms that makes the covariance exact; it acts on all atoms alike and knows nothing about
     any SAE.
  4. The correction moves the powers (and quotients); the targets are then re-assigned by rank (the k-th largest
     target power to the atom that now has the k-th largest power: only the distribution is a target, not which atom
     carries what), each row is rescaled to its target power (V2/V3: and re-tilted toward its quotient), and step 3
     is repeated. Stop when every power is within tol of its target (V2/V3: and the median quotient error is below
     rq_tol), after max_iter rounds, or after max_seconds. Every returned A is the output of step 3, so A^T A = Sigma
     holds to rounding whatever the targets; only the targets can be missed, and the achieved values are reported.
     On the layer-13 Gemma-2-2b Sigma with a Gemma Scope power distribution, the largest per-atom power error falls
     roughly as 1/sqrt(rounds): 0.10 after 40 rounds, 0.05 after 60 at layer 0.
Proof of exactness: (T S T) with T = S^{-1/2} M^{1/2} S^{-1/2}, M = S^{1/2} Sigma S^{1/2}:
  T S T = S^{-1/2} M^{1/2} S^{-1/2} S S^{-1/2} M^{1/2} S^{-1/2} = S^{-1/2} M S^{-1/2} = Sigma.   (S full rank: M >= d.)

Variants (what each holds fixed relative to the equal-power null V0 = r2_job_sparse2.py):
  V1  power      powers follow the measured latent power distribution (quantiles); directions drawn exactly as V0's
                 (Sigma^{1/2} z), independently of power; the correction is the only coupling of power and direction,
                 and it is the least one the covariance constraint forces.
  V2  power+rq   (power, Rayleigh quotient) pairs taken together from the same latents: the power marginal of V1 plus
                 the measured quotient distribution and its relation to power; directions random given the quotient.
  V3  rq only    the quotients of V2 with equal powers (separates the quotient marginal from the power coupling).
V2 and V3 are kept for local use only: with Gemma Scope pairs on the layer-13 Sigma the iteration did not approach
the targets (power errors 2-8 in log, quotient errors 0.3-0.6 in log10 after 25 rounds, under either assignment rule).
A^T A = Sigma fixes the power-weighted second moment of the directions, and the measured latents, whose powers add up
to well under the variance, are not a decomposition of the covariance, so the pairs need not be reachable.
No decoder direction of any SAE is used: per latent, only its power and (V2/V3) one scalar, its Rayleigh quotient.
"""
import math
import time

import numpy as np
import torch

VARIANTS = {"V1": "power", "V2": "power+rq", "V3": "rq"}


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"  # float64: MPS has none


def targets(pi, rq, M):
    """Per-atom (power, rq) targets from measured live latents: the latents at M evenly spaced power ranks
    (rank floor((j + 0.5) n / M), j = 0..M-1, descending power), so the targets keep the power quantiles and, at each
    power rank, the measured quotient. Powers are returned as shares (sum 1)."""
    pi, rq = np.asarray(pi, np.float64), np.asarray(rq, np.float64)
    order = np.argsort(-pi, kind="stable")
    n = len(pi)
    idx = order[np.minimum(((np.arange(M) + 0.5) * n / M).astype(int), n - 1)]
    t = pi[idx]
    return t / t.sum(), rq[idx]


def project_majorized(t, lam, iters=200, margin=0.9):
    """Nearest-in-spirit feasible powers: cap the head so every top-m partial sum is at most the top-m eigenvalue sum,
    and spread the removed mass over the uncapped atoms in proportion to their power (keeps the body's shape). t and
    lam are scaled to equal totals first. margin < 1 caps the top-m sum at (1 - (1 - margin)(1 - m/d)) x the eigenvalue sum (m < d), a point
    inside the feasible set: on its boundary the head atoms would have to be exact eigenvectors, which the iteration of
    build() reaches only slowly. Returns (p, moved share of total power)."""
    t = np.asarray(t, np.float64)
    lam = np.sort(np.asarray(lam, np.float64))[::-1]
    tot = lam.sum()
    p = t / t.sum() * tot
    order = np.argsort(-p, kind="stable")
    Lc = np.concatenate([np.cumsum(lam), np.full(max(len(p) - len(lam), 0), tot)])[:len(p)]
    m_ = np.arange(1, len(Lc) + 1)
    Lc = Lc * np.where(m_ < len(lam), 1 - (1 - margin) * (1 - m_ / len(lam)), 1.0)
    capped = np.zeros(len(p), bool)
    for _ in range(iters):
        ps = p[order]
        cs = np.cumsum(ps)
        bad = cs > Lc * (1 + 1e-12)
        if not bad.any():
            break
        # cap: the sorted vector becomes the eigenvalue increments wherever the partial sums overshoot
        m = np.flatnonzero(bad).max() + 1
        new = np.diff(np.concatenate([[0.0], np.minimum(cs[:m], Lc[:m])]))
        new = np.minimum(new, ps[:m])
        cap_idx = order[:m]
        excess = ps[:m].sum() - new.sum()
        p[cap_idx] = new
        capped[cap_idx] = True
        rest = ~capped
        p[rest] += excess * p[rest] / p[rest].sum()
        order = np.argsort(-p, kind="stable")
    moved = float(np.abs(p - t / t.sum() * tot).sum() / 2 / tot)
    return p, moved


def majorization_gap(p, lam):
    """max over m of (top-m power sum - top-m eigenvalue sum) / tr Sigma (<= 0 when feasible)."""
    p = np.sort(np.asarray(p, np.float64))[::-1]
    lam = np.sort(np.asarray(lam, np.float64))[::-1]
    tot = lam.sum()
    Lc = np.concatenate([np.cumsum(lam), np.full(max(len(p) - len(lam), 0), tot)])[:len(p)]
    return float(((np.cumsum(p / p.sum() * tot) - Lc) / tot).max())


def _sym_fn(S, f):
    w, V = torch.linalg.eigh(S)
    return (V * f(w.clamp_min(0))) @ V.T


def _ot_correct(At, lam):
    """A = A~ T, T the SPD solution of T S T = diag(lam) (eigenbasis), S = A~^T A~."""
    S = At.T @ At
    Sh = _sym_fn(S, torch.sqrt)
    Shi = _sym_fn(S, lambda w: 1 / torch.sqrt(w))
    Mh = _sym_fn(Sh * lam[None, :] @ Sh, torch.sqrt)  # (S^1/2 diag(lam) S^1/2)^1/2
    T = Shi @ Mh @ Shi
    return At @ T


def _rq(A, lam):
    """Rayleigh quotients of the rows (eigenbasis coordinates) relative to tr Sigma / d."""
    return (A.pow(2) * lam).sum(1) / A.pow(2).sum(1) / (lam.sum() / len(lam))


def _tilt_to(A, lam, target, lo=-6.0, hi=6.0, steps=40):
    """Per row, the exponent b with rq(row * lam^{b/2}) = target (bisection; rq is increasing in b). Returns rows
    multiplied by lam^{b/2} (unnormalised)."""
    ll = torch.log(lam)
    c2 = A.pow(2)
    a = torch.full((len(A),), lo, dtype=A.dtype, device=A.device)
    b = torch.full((len(A),), hi, dtype=A.dtype, device=A.device)
    ll0 = ll - ll.max()
    for _ in range(steps):
        m = (a + b) / 2
        w = c2 * torch.exp(m[:, None] * ll0[None, :])
        r = (w * lam).sum(1) / w.sum(1) / lam.mean()
        up = r < target
        a = torch.where(up, m, a)
        b = torch.where(up, b, m)
    m = (a + b) / 2
    return A * torch.exp(m[:, None] * ll0[None, :] / 2)


def build(cov, variant, p_share, rq_t, seed, max_iter=300, tol=0.01, rq_tol=0.01, dev=None, log=print, verbose=False,
          relax=1.0, margin=0.9, assign=None, max_seconds=None):
    """A (M x d, float64 numpy, raw units) with A^T A = cov, for variant V1/V2/V3 (see module docstring).
    p_share: target power shares (sum 1, length M; ignored for V3, which uses equal powers); rq_t: target Rayleigh
    quotients (V2/V3). assign: which current ranking the sorted targets follow each round, "power" (V1 default), "rq"
    (V2/V3 default: pairs follow the atom's current quotient rank) or "none" (fixed assignment). max_seconds: stop
    early (the covariance is exact at every round). Returns (A, info, achieved powers, achieved quotients)."""
    assert variant in VARIANTS
    t0 = time.time()
    dev = dev or _dev()
    cov = np.asarray(cov, np.float64)
    d = cov.shape[0]
    lam_np, U_np = np.linalg.eigh(cov)
    lam_np = np.clip(lam_np, lam_np.max() * 1e-10, None)
    M = len(p_share)
    assert M >= d, "M >= d is needed for A^T A = Sigma"
    p_t = np.full(M, 1.0 / M) if variant == "V3" else np.asarray(p_share, np.float64) / np.sum(p_share)
    gap0 = majorization_gap(p_t, lam_np)
    p_t, moved = project_majorized(p_t, lam_np, margin=margin)  # now sums to tr Sigma
    lam = torch.tensor(lam_np, dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    z = torch.tensor(rng.standard_normal((M, d)), dtype=torch.float64, device=dev)
    use_rq = variant in ("V2", "V3")
    rq_target = None
    if use_rq:
        rq_target = torch.tensor(np.clip(rq_t, lam_np.min() / lam_np.mean() * 1.01, lam_np.max() / lam_np.mean() * 0.99),
                                 dtype=torch.float64, device=dev)
        u = _tilt_to(z, lam, rq_target)
    else:
        u = z * lam.sqrt()
    u = u / u.norm(dim=1, keepdim=True)
    pt = torch.tensor(p_t, dtype=torch.float64, device=dev)
    assign = assign or ("power" if variant == "V1" else "rq")
    # targets sorted (descending) by the ranking key: power (V1), or quotient (V2: each quotient with its paired power;
    # V3: the quotients, all powers equal)
    key = pt if assign == "power" or not use_rq else rq_target
    order0 = torch.argsort(key, descending=True)
    t_sorted = pt[order0].clone()
    r_sorted = rq_target[order0].clone() if use_rq else None
    At = u * pt.sqrt()[:, None]
    hist = []
    for it in range(max_iter):
        A = _ot_correct(At, lam)
        p = A.pow(2).sum(1)
        cur = _rq(A, lam) if use_rq else None
        if assign != "none":
            # quantile matching: the k-th target (by the key) goes to the atom now k-th by the same key; only the
            # distribution (V2: the joint distribution of the pairs) is a target, not which atom carries what
            rank = torch.argsort(p if assign == "power" or not use_rq else cur, descending=True)
            pt = torch.empty_like(pt)
            pt[rank] = t_sorted
            if use_rq:
                rq_target = torch.empty_like(rq_target)
                rq_target[rank] = r_sorted
        perr = float((p / pt).log().abs().max())
        rerr = float(torch.median((cur / rq_target).log10().abs())) if use_rq else None
        hist.append((it, perr, rerr))
        if verbose:
            log(f"  round {it}: power max |log err| {perr:.2e}, rq median |log10 err| {rerr}, {time.time() - t0:.1f}s")
        if perr < tol and (rerr is None or rerr < rq_tol):
            break
        if max_seconds is not None and time.time() - t0 > max_seconds:
            break
        if use_rq:
            # re-tilt each row toward its target quotient, overshooting the remaining error by half (clipped)
            aim = rq_target * (rq_target / cur).clamp(0.5, 2.0) ** 0.5
            A = _tilt_to(A, lam, aim.clamp(float(lam.min() / lam.mean()) * 1.01, float(lam.max() / lam.mean()) * 0.99))
        At = A * ((pt / A.pow(2).sum(1)) ** (relax / 2))[:, None]  # rescaling toward the target power
    A_np = A.cpu().numpy() @ U_np.T  # back to the original coordinates
    G = A_np.T @ A_np
    err64 = float(np.linalg.norm(G - cov) / np.linalg.norm(cov))
    A32 = A_np.astype(np.float32).astype(np.float64)
    err32 = float(np.linalg.norm(A32.T @ A32 - cov) / np.linalg.norm(cov))
    p_ach = (A_np ** 2).sum(1)
    rq_ach = ((A_np @ cov) * A_np).sum(1) / p_ach / (np.trace(cov) / d)
    info = {"variant": variant, "label": VARIANTS[variant], "M": M, "d": d, "seed": seed, "iterations": len(hist),
            "assign": assign, "relax": relax, "margin": margin, "tol": tol, "rq_tol": rq_tol,
            "converged": bool(hist[-1][1] < tol and (hist[-1][2] is None or hist[-1][2] < rq_tol)),
            "final_power_max_abs_log_err": hist[-1][1], "final_rq_median_abs_log10_err": hist[-1][2],
            "majorization_gap_before_projection": gap0, "power_share_moved_by_projection": moved,
            "cov_rel_frob_err_float64": err64, "cov_rel_frob_err_float32": err32,
            "seconds": time.time() - t0, "device": dev,
            "power": lorenz(p_ach), "power_target": lorenz(p_t),
            "rq_log10_quantiles": {str(q): float(np.quantile(np.log10(rq_ach), q)) for q in (.01, .1, .5, .9, .99)},
            "spearman_power_rq": spearman(p_ach, rq_ach)}
    if use_rq:
        info["rq_target_log10_quantiles"] = {str(q): float(np.quantile(np.log10(rq_t), q)) for q in (.01, .1, .5, .9, .99)}
        info["spearman_rq_achieved_target"] = spearman(rq_ach, rq_target.cpu().numpy())
    log(f"dictionary {variant}: {len(hist)} rounds, power err {hist[-1][1]:.1e}, rq err {hist[-1][2]}, "
        f"cov err {err64:.1e} (float32 {err32:.1e}), moved {moved:.3f}, {time.time() - t0:.0f}s")
    return A_np, info, p_ach, rq_ach


def lorenz(p):
    p = np.sort(np.asarray(p, np.float64))[::-1]
    s, n = p.sum(), len(p)
    return {"n": int(n), "share_top1": float(p[0] / s), "share_top10": float(p[:10].sum() / s),
            "share_top1pct": float(p[:max(n // 100, 1)].sum() / s), "share_top10pct": float(p[:max(n // 10, 1)].sum() / s),
            "ess_frac": float(s ** 2 / (p ** 2).sum() / n), "median_over_mean": float(np.median(p) / p.mean()),
            "gini": float(1 - 2 * np.sum(np.cumsum(p[::-1]) / s) / n + 1 / n)}


def spearman(a, b):
    ra = np.empty(len(a))
    ra[np.argsort(a, kind="stable")] = np.arange(len(a))
    rb = np.empty(len(b))
    rb[np.argsort(b, kind="stable")] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def standin_power(n, seed):
    """Smoke-test stand-in for a real SAE's per-latent power and Rayleigh quotient (no real file): a lognormal body
    (sd 1.2 in ln) with a 10-latent head carrying ~20% of the power at high quotients, body quotients near isotropic."""
    rng = np.random.default_rng(seed)
    pi = np.exp(rng.normal(0, 1.2, n))
    pi[:10] = pi.sum() * 0.025
    lrq = rng.normal(0.25, 0.3, n)
    lrq[:10] = rng.normal(1.6, 0.2, 10)
    return pi, 10 ** lrq
