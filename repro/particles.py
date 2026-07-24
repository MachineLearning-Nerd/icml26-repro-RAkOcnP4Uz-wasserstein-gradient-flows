"""SVGD and Blob particle discretizations of the Wasserstein gradient (Eq 20, 21)
and MWGraD / A-MWGraD particle algorithms (Algorithm 1, 2).

For multi-target sampling with KL divergence objectives F_k(rho) = KL(rho||pi_k)
where pi_k ~ exp(-f_k), the Wasserstein gradient is Delta_k = grad f_k + grad log rho.
SVGD and Blob approximate this using kernel methods on a particle ensemble.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def rbf_kernel(X: np.ndarray, Y: np.ndarray, h: float = 1.0):
    """RBF kernel k(x,y) = exp(-||x-y||^2 / (2 h^2)) and its gradient w.r.t. first arg.

    Returns: (K, gradK) where K[i,j] = k(X_i, Y_j), gradK[i,j] = grad_x k(X_i, Y_j).
    """
    sq_dist = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * X @ Y.T
    K = np.exp(-sq_dist / (2.0 * h**2))
    # Vectorized: gradK[i,j,:] = -(X[i]-Y[j])/h^2 * K[i,j]
    # diff[i,j] = X[i] - Y[j]  via broadcasting
    diff = X[:, None, :] - Y[None, :, :]  # (m, n, d)
    gradK = -diff / h**2 * K[:, :, None]
    return K, gradK


def svgd_gradient(particles: np.ndarray, grad_f_k: callable, h: float = 1.0) -> np.ndarray:
    """SVGD approximation of Delta_k at each particle (Eq 20).

    Delta_k(x_i) = (1/m) sum_j [K(x_i,x_j) grad f_k(x_j) - grad_{x_j} K(x_i,x_j)]

    Note: grad_{x_j} K(x_i,x_j) = -grad_{x_i} K(x_i,x_j) for symmetric K.
    We compute using grad w.r.t. first arg (x_i) and negate.
    """
    m, d = particles.shape
    gf = grad_f_k(particles)  # (m, d)
    K, gradK_x = rbf_kernel(particles, particles, h)  # gradK_x[i,j] = grad_{x_i} k(x_i,x_j)
    # SVGD: (1/m) sum_j [K(x_i,x_j) grad f(x_j)] + (1/m) sum_j [grad_{x_i} K(x_i,x_j)]
    # (since grad_{x_j} k(x_i,x_j) = -grad_{x_i} k(x_i,x_j), and paper uses -grad_{x_j})
    phi = (K @ gf) / m + np.sum(gradK_x, axis=1) / m
    return phi


def blob_gradient(particles: np.ndarray, grad_f_k: callable, h: float = 1.0) -> np.ndarray:
    """Blob approximation of Delta_k at each particle (Eq 21, Carrillo et al. 2019).

    Delta_k(x_i) = grad f_k(x_i)
        - sum_j [grad_{x_j} K(x_i,x_j) / sum_l K(x_j,x_l)]
        - [sum_j grad_{x_j} K(x_i,x_j)] / [sum_l K(x_i,x_l)]

    Note: grad_{x_j} K(x_i,x_j) = (x_i - x_j)/h^2 * K(x_i,x_j) (grad w.r.t. 2nd arg).
    The first subtracted term normalizes per-particle by density at x_j;
    the second normalizes the sum by density at x_i.
    """
    m, d = particles.shape
    gf = grad_f_k(particles)  # (m, d)
    K, gradK_first = rbf_kernel(particles, particles, h)  # gradK_first[i,j] = grad_{x_i} k
    # grad w.r.t. second arg: grad_{x_j} K(x_i,x_j) = -grad_{x_i} K(x_i,x_j)
    gradK_second = -gradK_first  # (m, m, d): gradK_second[i,j] = grad_{x_j} K(x_i,x_j)

    rho_h = K.sum(axis=1)  # (m,) KDE at each particle

    # Term 2: sum_j [grad_{x_j} K(x_i,x_j) / rho_h(x_j)]
    term2 = np.zeros((m, d))
    for j in range(m):
        term2 += gradK_second[:, j, :] / rho_h[j]  # (m, d) contribution from particle j

    # Term 3: [sum_j grad_{x_j} K(x_i,x_j)] / rho_h(x_i)
    sum_gradK = np.sum(gradK_second, axis=1)  # (m, d)
    term3 = sum_gradK / rho_h[:, None]

    return gf - term2 - term3


def solve_particle_weights(deltas: np.ndarray) -> np.ndarray:
    """Solve w = argmin_{w in simplex} (1/m) sum_i ||sum_k w_k Delta_k(x_i)||^2 (Eq 22).

    deltas: (K, m, d) array of Delta_k at each particle.
    Returns: weights (K,).
    """
    K, m, d = deltas.shape
    # Q_{kj} = (1/m) sum_i <Delta_k(x_i), Delta_j(x_i)>
    Q = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            Q[k, j] = np.mean(np.sum(deltas[k] * deltas[j], axis=1))

    if K == 1:
        return np.array([1.0])

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


def compute_deltas(particles: np.ndarray, grad_f_ks: list[callable],
                   method: str = "svgd", h: float = 1.0) -> np.ndarray:
    """Compute Delta_k for all objectives at all particles.

    Returns: (K, m, d) array.
    """
    K = len(grad_f_ks)
    m, d = particles.shape
    deltas = np.zeros((K, m, d))
    grad_fn = svgd_gradient if method == "svgd" else blob_gradient
    for k in range(K):
        deltas[k] = grad_fn(particles, grad_f_ks[k], h)
    return deltas


def gradnorm(particles: np.ndarray, grad_f_ks: list[callable],
             method: str = "svgd", h: float = 1.0) -> float:
    """Compute GradNorm = (1/m) sum_i ||sum_k w_k Delta_k(x_i)||^2 (Eq 23)."""
    deltas = compute_deltas(particles, grad_f_ks, method, h)
    w = solve_particle_weights(deltas)
    combined = np.einsum("k,kmd->md", w, deltas)
    return float(np.mean(np.sum(combined**2, axis=1)))


def mwgrad_step(particles: np.ndarray, grad_f_ks: list[callable],
                method: str = "svgd", h: float = 1.0, eta: float = 0.01) -> np.ndarray:
    """MWGraD particle update (Algorithm 1): x_{n+1} = x_n - eta * sum_k w_k Delta_k(x_n)."""
    deltas = compute_deltas(particles, grad_f_ks, method, h)
    w = solve_particle_weights(deltas)
    combined = np.einsum("k,kmd->md", w, deltas)
    return particles - eta * combined


def amwgrad_step(particles: np.ndarray, velocities: np.ndarray, n: int,
                 grad_f_ks: list[callable], method: str = "svgd", h: float = 1.0,
                 eta: float = 0.01, beta: float = 0.0) -> tuple:
    """A-MWGraD particle update (Algorithm 2).

    x_{n+1} = x_n + sqrt(eta) * v_n
    v_{n+1} = alpha_n * v_n - sqrt(eta) * sum_k w_k Delta_k(x_n)

    alpha_n = (n-1)/(n+2) for geodesically convex (or beta unknown)
    alpha_n = (1 - sqrt(beta*eta))/(1 + sqrt(beta*eta)) for strongly convex
    """
    deltas = compute_deltas(particles, grad_f_ks, method, h)
    w = solve_particle_weights(deltas)
    combined = np.einsum("k,kmd->md", w, deltas)

    if beta > 0:
        alpha_n = (1.0 - np.sqrt(beta * eta)) / (1.0 + np.sqrt(beta * eta))
    else:
        alpha_n = (n - 1) / (n + 2)

    sq_eta = np.sqrt(eta)
    new_particles = particles + sq_eta * velocities
    new_velocities = alpha_n * velocities - sq_eta * combined
    return new_particles, new_velocities


def run_mwgrad(grad_f_ks: list[callable], x0: np.ndarray, T: int,
               method: str = "svgd", h: float = 1.0, eta: float = 0.01) -> tuple:
    """Run MWGraD for T iterations. Returns (particles, gradnorm_trajectory)."""
    particles = x0.copy()
    gn_traj = []
    for n in range(T):
        deltas = compute_deltas(particles, grad_f_ks, method, h)
        w = solve_particle_weights(deltas)
        combined = np.einsum("k,kmd->md", w, deltas)
        gn_traj.append(float(np.mean(np.sum(combined**2, axis=1))))
        particles = particles - eta * combined
    return particles, np.array(gn_traj)


def run_amwgrad(grad_f_ks: list[callable], x0: np.ndarray, T: int,
                method: str = "svgd", h: float = 1.0, eta: float = 0.01,
                beta: float = 0.0) -> tuple:
    """Run A-MWGraD for T iterations. Returns (particles, gradnorm_trajectory)."""
    particles = x0.copy()
    velocities = np.zeros_like(x0)
    gn_traj = []
    for n in range(T):
        deltas = compute_deltas(particles, grad_f_ks, method, h)
        w = solve_particle_weights(deltas)
        combined = np.einsum("k,kmd->md", w, deltas)
        gn_traj.append(float(np.mean(np.sum(combined**2, axis=1))))

        if beta > 0:
            alpha_n = (1.0 - np.sqrt(beta * eta)) / (1.0 + np.sqrt(beta * eta))
        else:
            alpha_n = (n) / (n + 3)  # (n-1)/(n+2) with 0-indexed n

        sq_eta = np.sqrt(eta)
        new_particles = particles + sq_eta * velocities
        new_velocities = alpha_n * velocities - sq_eta * combined
        particles = new_particles
        velocities = new_velocities
    return particles, np.array(gn_traj)
