"""
preprocessing.py
================
Feature normalisation for the SHIFT-TS pipeline.

Rules
-----
* The MinMaxScaler is FIT on SOURCE-TRAIN ENGINES ONLY (never on val/target).
* The same fitted scaler is APPLIED (transform only) to val and target data.
* This prevents any leakage from val/target distributions into the scaler.
* The scaler is serialised to disk at a caller-specified path so different
  scenarios (datasets, seeds) never overwrite each other's scalers.

Zero-range guard
----------------
If any feature column has zero range in the training data (constant sensor),
its min/max are perturbed by epsilon before fitting so that the transform
never produces NaN or inf.  The column remains in the feature set (it will
output a constant 0.0 after scaling) but does not crash downstream code.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .constants import FEATURE_COLS

# Small epsilon used to guard zero-range sensors
_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------

def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training-engine data.

    Zero-range guard: if a column is constant in the training data its range
    is artificially widened by ±epsilon so that downstream transforms never
    produce NaN / inf values.

    Parameters
    ----------
    df_train     : DataFrame containing only TRAIN engines (raw, un-scaled)
    feature_cols : columns to scale (default: FEATURE_COLS from constants)

    Returns
    -------
    Fitted MinMaxScaler instance.
    """
    feature_cols = feature_cols or FEATURE_COLS
    data = df_train[feature_cols].values.astype(np.float64)

    # Zero-range guard: widen any constant column by ±epsilon
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    zero_range = col_max - col_min < _EPSILON
    if zero_range.any():
        # Clone to avoid mutating the caller's data
        data = data.copy()
        for j in np.where(zero_range)[0]:
            data[:, j] = col_min[j]                 # ensure min row exists
            # Append a virtual row with col_min[j] + epsilon so sklearn sees range
            # Instead: directly set scaler data_min / data_max via a 2-row fit.
            # We achieve this by stacking a synthetic min/max row pair.
            pass

        # Build a 2-row array [min_row, max_row] with epsilon applied where needed
        synthetic_min = col_min.copy()
        synthetic_max = col_max.copy()
        synthetic_max[zero_range] = synthetic_min[zero_range] + _EPSILON
        guard_data = np.vstack([synthetic_min, synthetic_max, data])
    else:
        guard_data = data

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(guard_data)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply a **pre-fitted** scaler to any split (train / val / target / test).

    Parameters
    ----------
    df           : DataFrame to transform
    scaler       : already-fitted MinMaxScaler (call fit_scaler on train first)
    feature_cols : columns to scale (must match those used during fit)

    Returns
    -------
    Copy of df with feature columns replaced by scaled values.
    """
    feature_cols = feature_cols or FEATURE_COLS
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_scaler(scaler: MinMaxScaler, path: str | Path) -> None:
    """Serialise a fitted scaler to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str | Path) -> MinMaxScaler:
    """Load a previously saved scaler."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Variance audit (diagnostic utility)
# ---------------------------------------------------------------------------

def sensor_variance_report(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame showing mean and std for each sensor column.
    Useful for identifying near-constant sensors to drop.

    Parameters
    ----------
    df          : raw (un-scaled) training DataFrame
    sensor_cols : list of sensor column names to inspect

    Returns
    -------
    DataFrame with columns [sensor, mean, std, is_constant]
    sorted by std ascending.
    """
    stats = []
    for col in sensor_cols:
        vals = df[col].dropna().values
        stats.append({
            "sensor":      col,
            "mean":        float(np.mean(vals)),
            "std":         float(np.std(vals)),
            "is_constant": float(np.std(vals)) < 0.01,
        })
    report = pd.DataFrame(stats).sort_values("std").reset_index(drop=True)
    return report
