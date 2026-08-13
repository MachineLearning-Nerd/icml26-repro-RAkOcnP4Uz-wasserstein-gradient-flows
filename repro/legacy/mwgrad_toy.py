"""Legacy finite-dimensional MWGraD / A-MWGraD toy implementation.

This file is retained for provenance only. It is not the paper-level
implementation used by the root verify_all.py suite.
"""
from __future__ import annotations

import numpy as np


def multi_objective(theta, objectives):
    return sum(f(theta) for f, fg in objectives)


def multi_grad(theta, objectives):
    return sum(f_grad(theta) for f, f_grad in objectives)


def mwgrad(objectives, x0, T, lr=0.01):
    x = x0.copy()
    gaps = []
    for t in range(T):
        x = x - lr / np.sqrt(t + 1) * multi_grad(x, objectives)
        gaps.append(float(multi_objective(x, objectives)))
    return x, gaps


def a_mwgrad(objectives, x0, T, lr=0.01, beta=0.9):
    x = x0.copy()
    y = x0.copy()
    v = np.zeros_like(x0)
    gaps = []
    for t in range(T):
        g = multi_grad(y, objectives)
        v_new = beta * v - lr / (t + 1) * g
        x_new = y + v_new
        y = x_new + beta * (x_new - x)
        x = x_new
        v = v_new
        gaps.append(float(multi_objective(x, objectives)))
    return x, gaps


def make_convex_objectives(d, n_obj=3, seed=0):
    rng = np.random.default_rng(seed)
    objectives = []
    for _ in range(n_obj):
        A = rng.standard_normal((d, d))
        A = A @ A.T / d
        b = rng.standard_normal(d)
        f = lambda x, A=A, b=b: 0.5 * x @ A @ x + b @ x
        fg = lambda x, A=A, b=b: A @ x + b
        objectives.append((f, fg))
    return objectives
