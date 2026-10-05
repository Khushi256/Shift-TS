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

from .constants import FEATURE_COLS, FIXED_OP_RANGES, DATASET_CONFIG

# Small epsilon used to guard zero-range sensors
_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------

def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: list[str] | None = None,
    dataset_id: str | None = None,
    use_fixed_op_ranges: bool | None = None,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training-engine data.

    Operating-Condition Scaling Rules (Step 3)
    -------------------------------------------
    - For sources with n_op == 1 (FD001 / FD003, or when use_fixed_op_ranges=True),
      the op columns are scaled using fixed global ranges:
          op1: 0 to 42   (k-ft, NOT 0-42000)
          op2: 0 to 0.84 (Mach)
          op3: 20 to 100 (TRA)
      This prevents scaling narrow single-condition operating noise into [0, 1].
    - For sources with n_op > 1 (FD002 / FD004), min and max are fit directly
      from the training engines.
    - Zero-range guard: if any sensor column is constant in training data, its range
      is artificially widened by ±epsilon so that downstream transforms never
      produce NaN / inf values.

    Parameters
    ----------
    df_train            : DataFrame containing only TRAIN engines (raw, un-scaled)
    feature_cols        : columns to scale (default: FEATURE_COLS from constants)
    dataset_id          : optional dataset ID ("FD001".."FD004")
    use_fixed_op_ranges : explicitly force or disable fixed op scaling

    Returns
    -------
    Fitted MinMaxScaler instance with ``fixed_op_scaling`` boolean attribute.
    """
    feature_cols = feature_cols or FEATURE_COLS

    if use_fixed_op_ranges is not None:
        apply_fixed_op = use_fixed_op_ranges
    elif dataset_id is not None:
        apply_fixed_op = DATASET_CONFIG.get(dataset_id, {}).get("n_op_conditions", 6) == 1
    else:
        # Auto-detect: if op1 range in training data is near-zero (< 1.0), it is a single-condition source
        if "op1" in df_train.columns:
            op1_span = float(df_train["op1"].max() - df_train["op1"].min())
            apply_fixed_op = op1_span < 1.0
        else:
            apply_fixed_op = False

    min_vals = []
    max_vals = []
    for col in feature_cols:
        if apply_fixed_op and col in FIXED_OP_RANGES:
            c_min, c_max = FIXED_OP_RANGES[col]
        else:
            c_min = float(df_train[col].min())
            c_max = float(df_train[col].max())
            if c_max - c_min < _EPSILON:
                c_max = c_min + _EPSILON
        min_vals.append(c_min)
        max_vals.append(c_max)

    synthetic_data = np.vstack([
        np.array(min_vals, dtype=np.float64),
        np.array(max_vals, dtype=np.float64),
    ])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(synthetic_data)
    scaler.fixed_op_scaling = bool(apply_fixed_op)
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
