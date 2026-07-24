"""Gaussian-family MWGraD and A-MWGraD flows (Theorems 3.1, 3.6).

When objectives are KL divergences to zero-mean Gaussian targets, the
infinite-dimensional Wasserstein gradient flow reduces to a matrix ODE.

MWGraD flow (Theorem 3.1, eq 11):
    dSigma/dt = 2(S Sigma + Sigma S)
    S = -sum_k w_k grad_Sigma F_k(Sigma)

A-MWGraD flow (Theorem 3.6, eq 15):
    dSigma/dt = 2(S Sigma + Sigma S)
    dS/dt + alpha_t S + 2 S^2 + sum_k w_k grad_Sigma F_k(Sigma) = 0
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def kl_gaussian(Sigma: np.ndarray, Sigma_star: np.ndarray) -> float:
    """KL(N(0,Sigma) || N(0,Sigma_star)) = 1/2[tr(Sigma Sigma_star^{-1}) - log det(Sigma Sigma_star^{-1}) - d]."""
    d = Sigma.shape[0]
    SiS = np.linalg.solve(Sigma_star, Sigma)
    return 0.5 * (np.trace(SiS) - np.linalg.slogdet(SiS)[1] - d)


def grad_kl(Sigma: np.ndarray, Sigma_star_inv: np.ndarray) -> np.ndarray:
    """Gradient of F_k(Sigma) = KL(N(0,Sigma)||N(0,Sigma_k*)) w.r.t. Sigma.

    grad_Sigma F_k = 1/2 (Sigma_k*^{-1} - Sigma^{-1})
    """
    return 0.5 * (Sigma_star_inv - np.linalg.inv(Sigma))


def solve_weights(Sigma: np.ndarray, Sigma_star_invs: list[np.ndarray]) -> np.ndarray:
    """Solve w = argmin_{w in simplex} sum_{k,j} w_k w_j Q_{kj}

    where Q_{kj} = tr(A_k A_j Sigma), A_k = Sigma_k*^{-1} - Sigma^{-1}.

    This is eq (8) / (22) for the Gaussian family.
    Uses closed-form for K<=2, scipy for larger K.
    """
    K = len(Sigma_star_invs)
    Sigma_inv = np.linalg.inv(Sigma)
    A = [Si - Sigma_inv for Si in Sigma_star_invs]

    if K == 1:
        return np.array([1.0])

    if K == 2:
        Q11 = np.trace(A[0] @ A[0] @ Sigma)
        Q12 = np.trace(A[0] @ A[1] @ Sigma)
        Q22 = np.trace(A[1] @ A[1] @ Sigma)
        denom = Q11 - 2 * Q12 + Q22
        if denom < 1e-15:
            return np.array([0.5, 0.5])
        w1 = (Q22 - Q12) / denom
        w1 = np.clip(w1, 0.0, 1.0)
        return np.array([w1, 1.0 - w1])

    Q = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            Q[k, j] = np.trace(A[k] @ A[j] @ Sigma)

    def obj(w):
        return float(w @ Q @ w)
    def grad(w):
        return 2.0 * Q @ w
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(K)}]
    bounds = [(0, None)] * K
    w0 = np.ones(K) / K
    res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP")
    w = np.clip(res.x, 0, None)
    w /= w.sum()
    return w


def make_mwgrad_rhs(Sigma_star_invs: list[np.ndarray]):
    """RHS for the MWGraD flow ODE (Theorem 3.1).

    State: flatten(Sigma) (symmetric d x d -> d*(d+1)/2 params for lower triangle).
    """
    d = Sigma_star_invs[0].shape[0]
    triu_idx = np.triu_indices(d)
    n_params = len(triu_idx[0])

    def unpack(y):
        Sigma = np.zeros((d, d))
        Sigma[triu_idx] = y
        Sigma = Sigma + Sigma.T
        Sigma -= np.diag(np.diag(Sigma)) / 2  # avoid double-counting diagonal
        return Sigma

    def pack(Sigma):
        return Sigma[triu_idx].copy()

    def rhs(t, y):
        Sigma = unpack(y)
        # Ensure symmetry and PD
        Sigma = (Sigma + Sigma.T) / 2
        w = solve_weights(Sigma, Sigma_star_invs)
        g = sum(w[k] * grad_kl(Sigma, Sigma_star_invs[k]) for k in range(len(w)))
        S = -g
        dSigma = 2.0 * (S @ Sigma + Sigma @ S)
        return pack(dSigma)

    return rhs, pack, unpack, d, n_params


def make_amwgrad_rhs(Sigma_star_invs: list[np.ndarray], alpha_type: str = "convex", beta: float = 0.0):
    """RHS for the A-MWGraD flow ODE (Theorem 3.6).

    State: flatten(Sigma) ++ flatten(S).
    alpha_type: 'convex' -> alpha_t = 3/t, 'strong' -> alpha_t = 2*sqrt(beta).
    """
    d = Sigma_star_invs[0].shape[0]
    triu_idx = np.triu_indices(d)
    n_params = len(triu_idx[0])

    def unpack(y):
        Sigma = np.zeros((d, d))
        Sigma[triu_idx] = y[:n_params]
        Sigma = Sigma + Sigma.T
        Sigma -= np.diag(np.diag(Sigma)) / 2
        S = np.zeros((d, d))
        S[triu_idx] = y[n_params:]
        S = S + S.T
        S -= np.diag(np.diag(S)) / 2
        return Sigma, S

    def pack(Sigma, S):
        return np.concatenate([Sigma[triu_idx], S[triu_idx]])

    def rhs(t, y):
        Sigma, S = unpack(y)
        Sigma = (Sigma + Sigma.T) / 2
        if alpha_type == "convex":
            alpha_t = 3.0 / max(t, 1e-10)
        else:
            alpha_t = 2.0 * np.sqrt(beta)

        w = solve_weights(Sigma, Sigma_star_invs)
        g = sum(w[k] * grad_kl(Sigma, Sigma_star_invs[k]) for k in range(len(w)))
        dSigma = 2.0 * (S @ Sigma + Sigma @ S)
        dS = -alpha_t * S - 2.0 * (S @ S) - g
        return pack(dSigma, dS)

    return rhs, pack, unpack, d, n_params


def integrate_mwgrad(Sigma0: np.ndarray, Sigma_stars: list[np.ndarray],
                     t_span: tuple, t_eval: np.ndarray, **kw):
    """Integrate the MWGraD flow in Gaussian family."""
    Sigma_star_invs = [np.linalg.inv(Ss) for Ss in Sigma_stars]
    rhs, pack, unpack, d, n = make_mwgrad_rhs(Sigma_star_invs)
    y0 = pack(Sigma0)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="DOP853",
                    rtol=1e-10, atol=1e-12, **kw)
    Sigmas = []
    for i in range(sol.y.shape[1]):
        Sigmas.append(unpack(sol.y[:, i]))
    return np.array(sol.t), Sigmas


def integrate_amwgrad(Sigma0: np.ndarray, Sigma_stars: list[np.ndarray],
                      t_span: tuple, t_eval: np.ndarray,
                      alpha_type: str = "convex", beta: float = 0.0, **kw):
    """Integrate the A-MWGraD flow in Gaussian family."""
    Sigma_star_invs = [np.linalg.inv(Ss) for Ss in Sigma_stars]
    rhs, pack, unpack, d, n = make_amwgrad_rhs(Sigma_star_invs, alpha_type, beta)
    S0 = np.zeros((d, d))
    y0 = pack(Sigma0, S0)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="DOP853",
                    rtol=1e-10, atol=1e-12, **kw)
    results = []
    for i in range(sol.y.shape[1]):
        Sig, Ss = unpack(sol.y[:, i])
        results.append((Sig, Ss))
    return np.array(sol.t), results


def merit_function(Sigma: np.ndarray, Sigma_stars: list[np.ndarray]) -> float:
    """Compute M(rho) = sup_q min_k {F_k(rho) - F_k(q)} for Gaussian rho.

    For K=1: M = KL(rho||pi_1).
    For K=2: uses Lagrangian approach (fast closed-form + 1D root finding).
    For K>=3: numerical optimization.
    """
    K = len(Sigma_stars)
    F_vals = np.array([kl_gaussian(Sigma, Ss) for Ss in Sigma_stars])

    if K == 1:
        return float(F_vals[0])

    if K == 2:
        # Check if q=pi_k for some k gives a good value
        best_from_targets = 0.0
        for k in range(K):
            diffs = np.array([F_vals[j] - kl_gaussian(Sigma_stars[k], Sigma_stars[j])
                              for j in range(K)])
            val = np.min(diffs)
            best_from_targets = max(best_from_targets, val)

        # Interior solution: both constraints active
        # Sigma_q^{-1} = (1+lam) S1inv - lam S2inv
        S1inv = np.linalg.inv(Sigma_stars[0])
        S2inv = np.linalg.inv(Sigma_stars[1])
        delta = F_vals[0] - F_vals[1]

        def constraint(lam):
            Sq_inv = (1 + lam) * S1inv - lam * S2inv
            try:
                Sq = np.linalg.inv(Sq_inv)
                eigs = np.linalg.eigvalsh(Sq)
                if np.any(eigs <= 0):
                    return 1e10
                return kl_gaussian(Sq, Sigma_stars[0]) - kl_gaussian(Sq, Sigma_stars[1]) - delta
            except np.linalg.LinAlgError:
                return 1e10

        from scipy.optimize import brentq
        best_interior = 0.0
        for lam_range in [(-5, 5), (-20, 20), (-100, 100)]:
            try:
                cl = constraint(lam_range[0])
                cr = constraint(lam_range[1])
                if cl * cr < 0:
                    lam_star = brentq(constraint, lam_range[0], lam_range[1], xtol=1e-12)
                    Sq_inv = (1 + lam_star) * S1inv - lam_star * S2inv
                    Sq = np.linalg.inv(Sq_inv)
                    val = min(F_vals[0] - kl_gaussian(Sq, Sigma_stars[0]),
                              F_vals[1] - kl_gaussian(Sq, Sigma_stars[1]))
                    best_interior = max(best_interior, val)
                    break
            except Exception:
                continue

        return float(max(best_from_targets, best_interior, 0.0))

    # K >= 3: numerical optimization
    d = Sigma.shape[0]

    def neg_merit(params):
        L = np.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = np.exp(params[idx])
                else:
                    L[i, j] = params[idx]
                idx += 1
        Sigma_q = L @ L.T
        diffs = np.array([F_vals[k] - kl_gaussian(Sigma_q, Sigma_stars[k])
                          for k in range(K)])
        return -np.min(diffs)

    best_val = -np.inf
    for x0 in [np.zeros(d * (d + 1) // 2),
               np.random.randn(d * (d + 1) // 2) * 0.3]:
        try:
            res = minimize(neg_merit, x0, method="Nelder-Mead",
                           options={"maxiter": 2000})
            val = -res.fun
            if val > best_val:
                best_val = val
        except Exception:
            pass

    for k in range(K):
        diffs = np.array([F_vals[j] - kl_gaussian(Sigma_stars[k], Sigma_stars[j])
                          for j in range(K)])
        val = np.min(diffs)
        if val > best_val:
            best_val = val

    return float(max(best_val, 0.0))
