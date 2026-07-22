"""Clean-room MWGraD / A-MWGraD from "Accelerated Multiple Wasserstein Gradient Flows
for Multi-objective Optimization" (arXiv 2601.19220). numpy, CPU.

Multi-objective optimization: minimize F(theta) = sum_k w_k f_k(theta) over multiple objectives.
MWGraD = gradient descent on F (O(1/t) for convex). A-MWGraD = Nesterov-accelerated (O(1/t^2)).
"""
from __future__ import annotations
import numpy as np


def multi_objective(theta, objectives):
    """Sum of weighted objectives."""
    return sum(f(theta) for f, fg in objectives)


def multi_grad(theta, objectives):
    """Sum of gradients."""
    return sum(f_grad(theta) for f, f_grad in objectives)


def mwgrad(objectives, x0, T, lr=0.01):
    """MWGraD: gradient descent on sum of objectives. Returns gap trajectory."""
    x = x0.copy(); gaps = []
    for t in range(T):
        g = multi_grad(x, objectives)
        x = x - lr / np.sqrt(t + 1) * g
        gaps.append(float(multi_objective(x, objectives)))
    return x, gaps


def a_mwgrad(objectives, x0, T, lr=0.01, beta=0.9):
    """A-MWGraD: Nesterov-accelerated gradient on sum of objectives. Returns gap trajectory."""
    x = x0.copy(); y = x0.copy(); v = np.zeros_like(x0); gaps = []
    for t in range(T):
        g = multi_grad(y, objectives)
        v_new = beta * v - lr / (t + 1) * g
        x_new = y + v_new
        y = x_new + beta * (x_new - x)
        x = x_new; v = v_new
        gaps.append(float(multi_objective(x, objectives)))
    return x, gaps


def make_convex_objectives(d, n_obj=3, seed=0):
    """Generate n_obj convex quadratic objectives."""
    rng = np.random.default_rng(seed)
    objs = []
    for _ in range(n_obj):
        A = rng.standard_normal((d, d)); A = A @ A.T / d
        b = rng.standard_normal(d)
        f = lambda x, A=A, b=b: 0.5 * x @ A @ x + b @ x
        fg = lambda x, A=A, b=b: A @ x + b
        objs.append((f, fg))
    return objs
