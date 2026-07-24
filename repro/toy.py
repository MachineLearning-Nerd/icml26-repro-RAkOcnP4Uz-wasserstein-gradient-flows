"""Toy multi-target sampling experiment (Section 4.1).

Four mixture-of-two-Gaussians targets in 2D. Particles updated by MWGraD and
A-MWGraD with SVGD and Blob discretizations. Metric: GradNorm (Eq 23).
"""
from __future__ import annotations

import numpy as np


def make_toy_targets():
    """Create the 4 mixture-of-Gaussians targets from Section 4.1.

    Each target pi_k = 0.7 * N(mu_k1, I) + 0.3 * N(mu_k2, I) in 2D.
    Returns list of (means, weights) for each target.
    """
    targets = []
    mus = [
        ([4, -4], [0.1, 0.2]),
        ([-4, 4], [-0.1, 0.3]),
        ([-4, -4], [0.4, -0.4]),
        ([4, 4], [-0.2, 0.3]),
    ]
    for mu1, mu2 in mus:
        means = np.array([mu1, mu2], dtype=float)
        weights = np.array([0.7, 0.3])
        targets.append((means, weights))
    return targets


def make_negative_log_density_targets():
    """Create f_k(x) = -log pi_k(x) and grad f_k for each target.

    For mixture of Gaussians: pi_k(x) = sum_j gamma_kj N(x|mu_kj, Sigma)
    f_k(x) = -log(sum_j gamma_kj (2pi)^{-d/2} |Sigma|^{-1/2} exp(-0.5||x-mu_kj||^2))
    """
    targets = make_toy_targets()
    Sigma = np.eye(2)
    d = 2
    norm_const = (2 * np.pi)**(-d / 2) * np.linalg.det(Sigma)**(-0.5)

    def make_f_and_grad(means, weights):
        def f(x):
            # x: (m, d) or (d,)
            x = np.atleast_2d(x)
            comps = np.zeros((x.shape[0], len(weights)))
            for j in range(len(weights)):
                diff = x - means[j]
                comps[:, j] = weights[j] * norm_const * np.exp(-0.5 * np.sum(diff**2, axis=1))
            density = np.sum(comps, axis=1)
            density = np.maximum(density, 1e-300)
            return -np.log(density)

        def grad_f(x):
            x = np.atleast_2d(x)
            comps = np.zeros((x.shape[0], len(weights)))
            for j in range(len(weights)):
                diff = x - means[j]
                comps[:, j] = weights[j] * norm_const * np.exp(-0.5 * np.sum(diff**2, axis=1))
            density = np.sum(comps, axis=1, keepdims=True)
            density = np.maximum(density, 1e-300)
            # grad f_k(x) = -sum_j gamma_kj N(x|mu_kj) (x - mu_kj) / pi_k(x)
            # = sum_j [comp_j / density] * (mu_kj - x)
            grad = np.zeros_like(x)
            for j in range(len(weights)):
                diff = x - means[j]
                grad += (comps[:, j:j+1] / density) * (-diff)
            return grad

        return f, grad_f

    grad_f_ks = []
    fs = []
    for means, weights in targets:
        f, gf = make_f_and_grad(means, weights)
        fs.append(f)
        grad_f_ks.append(gf)
    return fs, grad_f_ks


def make_gaussian_targets():
    """Create Gaussian targets for convergence rate experiments (Appendix G.1).

    KL(ρ||π_k) is geodesically convex for Gaussian targets.
    For strong convexity: use targets with small covariance.
    """
    # Two targets with moderate covariance
    Sigma_stars = [
        np.array([[2.0, 0.5], [0.5, 1.0]]),
        np.array([[1.0, -0.3], [-0.3, 2.0]]),
    ]
    return Sigma_stars


def make_strongly_convex_targets(beta: float = 1.0):
    """Create strongly geodesically convex targets.

    KL(ρ||π_k) with π_k = N(0, (1/beta) I) is beta-strongly geodesically convex.
    """
    d = 2
    Sigma_stars = [np.eye(d) / beta, np.eye(d) / (beta * 1.5)]
    return Sigma_stars
