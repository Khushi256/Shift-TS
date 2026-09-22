"""
metrics.py
==========
Core 5 — Evaluation metrics for SHIFT-TS.

Regression metrics (standard):
    mae, rmse

Uncertainty / calibration metrics (regression-appropriate):
    prediction_intervals    — lower/upper bounds from mean ± z*std
    coverage                — fraction of targets inside the interval
    mean_interval_width     — average interval width (narrower = more informative)
    error_vs_uncertainty    — Spearman ρ between |error| and std
    risk_coverage_curve     — selective prediction: trade MAE for coverage
    nll_gaussian            — NLL under Gaussian predictive distribution
                              (only meaningful if std is a calibrated estimate)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Standard regression
# ---------------------------------------------------------------------------

def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.abs(preds - targets).mean())


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(((preds - targets) ** 2).mean()))


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

def prediction_intervals(
    means: np.ndarray,
    stds:  np.ndarray,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric prediction intervals.

    PI = [mean - z * std,  mean + z * std]
    where z is the (1+level)/2 quantile of the standard normal.

    Parameters
    ----------
    means : (N,) predicted means
    stds  : (N,) predicted standard deviations (uncertainty)
    level : coverage level (default 0.95 → 95% PI)

    Returns
    -------
    lower, upper : (N,) arrays
    """
    z = float(stats.norm.ppf((1 + level) / 2))
    lower = means - z * stds
    upper = means + z * stds
    return lower, upper


def coverage(
    targets: np.ndarray,
    lower:   np.ndarray,
    upper:   np.ndarray,
) -> float:
    """
    Fraction of targets that fall within [lower, upper].

    For a well-calibrated 95% PI, this should be ≈ 0.95.
    Coverage < level → overconfident (intervals too narrow).
    Coverage > level → underconfident (intervals too wide).
    """
    inside = (targets >= lower) & (targets <= upper)
    return float(inside.mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Average width of prediction intervals.

    Narrower = more informative, provided coverage is maintained.
    """
    return float((upper - lower).mean())


# ---------------------------------------------------------------------------
# Error–uncertainty correlation
# ---------------------------------------------------------------------------

def error_vs_uncertainty(
    preds:   np.ndarray,
    targets: np.ndarray,
    stds:    np.ndarray,
) -> dict[str, float]:
    """
    Compute Spearman rank correlation between |error| and std.

    A positive Spearman ρ indicates that higher uncertainty corresponds to
    larger prediction errors — meaning uncertainty is informative.

    Returns
    -------
    dict: {"spearman_rho": float, "p_value": float}
    """
    abs_errors = np.abs(preds - targets)
    rho, pval  = stats.spearmanr(abs_errors, stds)
    return {"spearman_rho": float(rho), "p_value": float(pval)}


# ---------------------------------------------------------------------------
# Risk–coverage curve
# ---------------------------------------------------------------------------

def risk_coverage_curve(
    preds:   np.ndarray,
    targets: np.ndarray,
    stds:    np.ndarray,
    n_thresholds: int = 100,
) -> dict[str, np.ndarray]:
    """
    Selective prediction: reject the most uncertain predictions and measure MAE
    on the retained predictions.

    For each threshold τ:
        - Retain predictions where std ≤ τ (low uncertainty)
        - Compute MAE on retained predictions
        - Record coverage = fraction retained

    Returns
    -------
    dict:
        "coverage"       : (n_thresholds,) fraction of retained predictions
        "mae_at_coverage": (n_thresholds,) MAE on retained predictions
        "thresholds"     : (n_thresholds,) std thresholds used

    Usage
    -----
    Plot coverage (x) vs mae_at_coverage (y) to see how selectively
    rejecting uncertain predictions improves MAE.
    """
    thresholds = np.percentile(stds, np.linspace(0, 100, n_thresholds))
    coverages, maes_at = [], []

    for tau in thresholds:
        mask = stds <= tau
        cov  = mask.mean()
        if mask.sum() == 0:
            maes_at.append(float("nan"))
        else:
            maes_at.append(float(np.abs(preds[mask] - targets[mask]).mean()))
        coverages.append(float(cov))

    return {
        "coverage":        np.array(coverages),
        "mae_at_coverage": np.array(maes_at),
        "thresholds":      thresholds,
    }


# ---------------------------------------------------------------------------
# NLL under Gaussian assumption
# ---------------------------------------------------------------------------

def nll_gaussian(
    preds:   np.ndarray,
    targets: np.ndarray,
    stds:    np.ndarray,
    eps:     float = 1e-6,
) -> float:
    """
    Negative log-likelihood under a Gaussian predictive distribution:

        NLL = 0.5 * log(2π σ²) + (y - μ)² / (2σ²)

    NOTE: This is only meaningful if `stds` are calibrated estimates of the
    predictive standard deviation.  MC Dropout stds are heuristic proxies;
    treat this number as a rough indicator, not a ground-truth calibration score.

    Parameters
    ----------
    preds   : (N,) predicted means
    targets : (N,) true values
    stds    : (N,) predicted standard deviations
    eps     : small value to prevent log(0)

    Returns
    -------
    Average NLL per sample.
    """
    sigma2 = np.clip(stds ** 2, eps, None)
    nll    = 0.5 * (np.log(2 * np.pi * sigma2) + ((targets - preds) ** 2) / sigma2)
    return float(nll.mean())


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    means:   np.ndarray,
    stds:    np.ndarray,
    targets: np.ndarray,
    pi_level: float = 0.95,
) -> dict:
    """
    Compute the full uncertainty evaluation suite and return a summary dict.

    Parameters
    ----------
    means    : (N,) predicted means (from MC Dropout)
    stds     : (N,) predicted stds  (from MC Dropout)
    targets  : (N,) true RUL values
    pi_level : prediction interval coverage level

    Returns
    -------
    Flat dict of all metrics (for printing or logging).
    """
    lower, upper = prediction_intervals(means, stds, level=pi_level)
    cov          = coverage(targets, lower, upper)
    width        = mean_interval_width(lower, upper)
    corr         = error_vs_uncertainty(means, targets, stds)
    rc           = risk_coverage_curve(means, targets, stds)
    nll          = nll_gaussian(means, targets, stds)

    return {
        "mae":                    mae(means, targets),
        "rmse":                   rmse(means, targets),
        f"coverage_{int(pi_level*100)}pct": cov,
        "mean_interval_width":    width,
        "spearman_rho":           corr["spearman_rho"],
        "spearman_p":             corr["p_value"],
        "nll_gaussian":           nll,
        "risk_coverage":          rc,        # dict of arrays — log separately
    }
