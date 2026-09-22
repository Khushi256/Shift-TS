"""
preprocessing.py
================
Feature normalisation for the SHIFT-TS pipeline.

Rules
-----
* The MinMaxScaler is FIT on TRAIN ENGINES ONLY.
* The same fitted scaler is APPLIED (transform only) to val and target data.
* This prevents any leakage from val/target distributions into the scaler.
* The scaler is serialised to disk so it can be reloaded for inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .constants import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------

def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training-engine data.

    Parameters
    ----------
    df_train     : DataFrame containing only TRAIN engines
    feature_cols : columns to scale (default: FEATURE_COLS from constants)

    Returns
    -------
    Fitted MinMaxScaler instance.
    """
    feature_cols = feature_cols or FEATURE_COLS
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[feature_cols].values)
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
            "sensor":       col,
            "mean":         float(np.mean(vals)),
            "std":          float(np.std(vals)),
            "is_constant":  float(np.std(vals)) < 0.01,
        })
    report = pd.DataFrame(stats).sort_values("std").reset_index(drop=True)
    return report
