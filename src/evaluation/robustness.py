"""
robustness.py
=============
Core 6 — Input perturbation utilities for robustness experiments.

Perturbations are applied to normalised sensor windows at inference time.
They simulate real-world deployment degradations:

    1. gaussian_noise    — random measurement noise on all sensors
    2. missing_values    — random timesteps set to zero (sensor dropout)
    3. sensor_dropout    — entire sensor channel zeroed out
    4. op_cond_shift     — op-setting channels replaced with an alternative
                           condition (simulates a changed operating regime)

All functions take a numpy array (N, W, F) or torch Tensor and return the
same type.  They are stateless and reproducible via seed.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

def gaussian_noise(
    x:      np.ndarray | torch.Tensor,
    sigma:  float = 0.05,
    seed:   int   = 0,
) -> np.ndarray | torch.Tensor:
    """
    Add i.i.d. Gaussian noise to all features.

    Parameters
    ----------
    x     : (N, W, F) input array (normalised to [0,1])
    sigma : noise std-dev (relative to the [0,1] feature range)
    seed  : random seed

    Returns
    -------
    Noisy array, clipped to [0, 1].
    """
    is_tensor = isinstance(x, torch.Tensor)
    arr = x.numpy() if is_tensor else np.array(x)

    rng   = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=arr.shape).astype(arr.dtype)
    arr   = np.clip(arr + noise, 0.0, 1.0)

    return torch.from_numpy(arr) if is_tensor else arr


# ---------------------------------------------------------------------------
# Random timestep masking (missing values)
# ---------------------------------------------------------------------------

def missing_values(
    x:           np.ndarray | torch.Tensor,
    mask_ratio:  float = 0.10,
    fill_value:  float = 0.0,
    seed:        int   = 0,
) -> np.ndarray | torch.Tensor:
    """
    Randomly zero-out ``mask_ratio`` fraction of timesteps (all features).

    Parameters
    ----------
    x          : (N, W, F) input array
    mask_ratio : fraction of timesteps to zero out per sample
    fill_value : value to fill masked timesteps (default 0.0)
    seed       : random seed

    Returns
    -------
    Array with masked timesteps.
    """
    is_tensor = isinstance(x, torch.Tensor)
    arr = x.numpy().copy() if is_tensor else np.array(x)

    rng = np.random.default_rng(seed)
    N, W, F = arr.shape
    n_mask = max(1, int(W * mask_ratio))

    for i in range(N):
        mask_idx = rng.choice(W, size=n_mask, replace=False)
        arr[i, mask_idx, :] = fill_value

    return torch.from_numpy(arr) if is_tensor else arr


# ---------------------------------------------------------------------------
# Sensor dropout (whole channel dropout)
# ---------------------------------------------------------------------------

def sensor_dropout(
    x:            np.ndarray | torch.Tensor,
    n_sensors:    int   = 2,
    fill_value:   float = 0.0,
    seed:         int   = 0,
    sensor_range: tuple[int, int] | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Zero out entire sensor channels (same channels for all samples in batch).

    Parameters
    ----------
    x            : (N, W, F) input array
    n_sensors    : number of sensor channels to drop
    fill_value   : replacement value for dropped channels
    seed         : random seed
    sensor_range : (start, end) indices of droppable sensors
                   (default: first F-3 columns, i.e. sensor columns only)

    Returns
    -------
    Array with dropped sensor channels.
    """
    is_tensor = isinstance(x, torch.Tensor)
    arr = x.numpy().copy() if is_tensor else np.array(x)

    N, W, F = arr.shape
    rng = np.random.default_rng(seed)

    # By default, drop from sensor columns only (not op-setting columns)
    lo, hi = sensor_range if sensor_range else (0, F - 3)
    drop_cols = rng.choice(hi - lo, size=min(n_sensors, hi - lo), replace=False) + lo

    arr[:, :, drop_cols] = fill_value

    return torch.from_numpy(arr) if is_tensor else arr


# ---------------------------------------------------------------------------
# Operating condition shift
# ---------------------------------------------------------------------------

def op_cond_shift(
    x:          np.ndarray | torch.Tensor,
    new_values: np.ndarray | list,
    op_col_start: int = -3,
) -> np.ndarray | torch.Tensor:
    """
    Replace the op-setting columns with fixed ``new_values``.

    Simulates a change in operating regime (e.g., deploying at a different
    altitude/Mach number than seen during training).

    Parameters
    ----------
    x            : (N, W, F) input array — last 3 columns are op-settings
    new_values   : (3,) array of replacement op-setting values
                   (already normalised to match the scaler range)
    op_col_start : index of first op-setting column (default -3 → last 3)

    Returns
    -------
    Array with replaced op-setting columns.
    """
    is_tensor = isinstance(x, torch.Tensor)
    arr = x.numpy().copy() if is_tensor else np.array(x)

    new_vals = np.array(new_values, dtype=arr.dtype)
    arr[:, :, op_col_start:] = new_vals[np.newaxis, np.newaxis, :]

    return torch.from_numpy(arr) if is_tensor else arr


# ---------------------------------------------------------------------------
# Perturbation registry (for experiment loops)
# ---------------------------------------------------------------------------

PERTURBATIONS = {
    "clean":           lambda x, seed: x,
    "noise_low":       lambda x, seed: gaussian_noise(x, sigma=0.02, seed=seed),
    "noise_med":       lambda x, seed: gaussian_noise(x, sigma=0.05, seed=seed),
    "noise_high":      lambda x, seed: gaussian_noise(x, sigma=0.10, seed=seed),
    "missing_10pct":   lambda x, seed: missing_values(x, mask_ratio=0.10, seed=seed),
    "missing_30pct":   lambda x, seed: missing_values(x, mask_ratio=0.30, seed=seed),
    "sensor_drop_1":   lambda x, seed: sensor_dropout(x, n_sensors=1, seed=seed),
    "sensor_drop_3":   lambda x, seed: sensor_dropout(x, n_sensors=3, seed=seed),
}
