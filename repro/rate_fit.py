"""Rate fitting utilities for convergence rate analysis.

Fits M(rho_t) to power-law (M ~ C/t^alpha) and exponential (M ~ C*exp(-lambda*t))
models using log-log and semi-log regression.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def fit_power_law(t: np.ndarray, M: np.ndarray, t_min: float = 0.0):
    """Fit M(t) ~ C * t^{-alpha} via log-log linear regression.

    Returns dict with alpha, intercept, R^2, and the fitted values.
    """
    mask = (t > t_min) & (M > 0)
    log_t = np.log(t[mask])
    log_M = np.log(M[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_M)
    return {
        "alpha": -slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "std_err": std_err,
        "fitted_log_M": slope * log_t + intercept,
        "log_t": log_t,
        "log_M": log_M,
    }


def fit_exponential(t: np.ndarray, M: np.ndarray, t_min: float = 0.0):
    """Fit M(t) ~ C * exp(-lambda * t) via semi-log linear regression.

    Returns dict with lambda, intercept, R^2.
    """
    mask = (t > t_min) & (M > 0)
    t_fit = t[mask]
    log_M = np.log(M[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_fit, log_M)
    return {
        "lambda": -slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "std_err": std_err,
        "fitted_log_M": slope * t_fit + intercept,
        "t_fit": t_fit,
        "log_M": log_M,
    }


def compare_rates(t: np.ndarray, M: np.ndarray, t_min: float = 0.0):
    """Fit both power-law and exponential, return which fits better."""
    pl = fit_power_law(t, M, t_min)
    ex = fit_exponential(t, M, t_min)
    return {
        "power_law": pl,
        "exponential": ex,
        "better": "power_law" if pl["r_squared"] > ex["r_squared"] else "exponential",
    }
