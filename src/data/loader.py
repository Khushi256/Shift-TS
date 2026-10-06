"""
loader.py
=========
Raw data loading, RUL labelling, operating-condition characterisation, and
**engine-level** split for C-MAPSS datasets (FD001–FD004).

Dataset-agnostic design
-----------------------
All public functions accept a ``dataset_id`` parameter (default: "FD002") that
selects the appropriate file names and operating-condition count from
DATASET_CONFIG.  The feature columns, RUL cap, window size, and split fractions
are identical across all datasets.

Operating conditions
--------------------
* FD002 / FD004 — 6 discrete operating conditions characterised by K-Means
  (k=6) on the three op-setting columns.
* FD001 / FD003 — 1 operating condition; K-Means is skipped and every row
  receives op_condition = 0.

Engine-level split design
--------------------------
FD002's six operating conditions are interleaved within every engine's run.
Investigating the data shows that ~99% of engines have condition 5 (op1≈42)
as their most frequent condition — meaning the "dominant condition" approach
cannot produce a meaningful per-engine environment assignment.

Instead we use a **random engine-level split** (70 / 15 / 15):

    train  (~70%) → SSL pretraining + supervised training
    val    (~15%) → hyperparameter selection / early stopping
    target (~15%) → unseen cohort for few-shot adaptation

The split is done at the engine level (before windowing) so that no engine
appears in more than one partition, eliminating window-level leakage.

``split_seed`` is now a parameter (not hard-coded 42) so different scenarios
can use different seeds without touching constants.  The default stays 42 to
preserve FD002 backward-compatibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .constants import (
    COLUMNS,
    DATASET_CONFIG,
    FEATURE_COLS,
    FIXED_REGIME_CENTERS,
    OP_COLS,
    RUL_CAP,
    SPLIT_SEED,
    TARGET_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    VALID_DATASET_IDS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cfg(dataset_id: str) -> dict:
    """Validate dataset_id and return its config dict."""
    if dataset_id not in VALID_DATASET_IDS:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. "
            f"Valid options: {VALID_DATASET_IDS}"
        )
    return DATASET_CONFIG[dataset_id]


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_raw(
    data_dir: str | Path,
    split: str = "train",
    dataset_id: str = "FD002",
) -> pd.DataFrame:
    """
    Load a C-MAPSS text file into a tidy DataFrame.

    Parameters
    ----------
    data_dir   : path to the CMAPSSData folder
    split      : "train" or "test"
    dataset_id : one of "FD001", "FD002", "FD003", "FD004"

    Returns
    -------
    DataFrame with columns defined in constants.COLUMNS (26 cols).
    The trailing blank column is silently dropped.
    """
    cfg = _get_cfg(dataset_id)
    fname = cfg["train_file"] if split == "train" else cfg["test_file"]
    path = Path(data_dir) / fname
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        na_values=["nan"],
    )
    # Drop any extra all-NaN columns (trailing whitespace artifact)
    df = df.dropna(axis=1, how="all")
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"]     = df["cycle"].astype(int)
    return df


def load_test_rul(
    data_dir: str | Path,
    dataset_id: str = "FD002",
) -> pd.Series:
    """
    Load ground-truth RUL for the CMAPSS test set.

    Each row in ``RUL_{dataset_id}.txt`` is the RUL at the last observed
    cycle for one test engine.

    Returns
    -------
    pd.Series  index = engine_id (1-indexed), values = true RUL
    """
    cfg  = _get_cfg(dataset_id)
    path = Path(data_dir) / cfg["rul_file"]
    rul  = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul.index = rul.index + 1       # 1-indexed engine_id
    rul.index.name = "engine_id"
    return rul["rul"]


# ---------------------------------------------------------------------------
# RUL labelling
# ---------------------------------------------------------------------------

def compute_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Add a ``rul`` column using a piecewise-linear (capped) scheme.

    RUL = min(max_cycle − cycle, cap)

    Parameters
    ----------
    df  : DataFrame with engine_id + cycle columns (from load_raw)
    cap : maximum RUL value (default 125 cycles)

    Returns
    -------
    Copy of df with ``rul`` column added.
    """
    df = df.copy()
    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
    )
    df = df.join(max_cycles, on="engine_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Operating-condition characterisation (Fixed FD002 Centers)
# ---------------------------------------------------------------------------

class FixedRegimeAssigner:
    """
    Operating regime assigner based on fixed FD002 reference centroids.
    Avoids refitting K-Means per dataset, ensuring global cross-dataset regime consistency.
    """
    def __init__(self, centers: np.ndarray = FIXED_REGIME_CENTERS):
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        self.n_clusters = len(self.cluster_centers_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        # Euclidean distance to each fixed center: shape (N, n_clusters)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


def assign_op_conditions(
    df: pd.DataFrame,
    kmeans: Any | None = None,
    label_map: np.ndarray | None = None,
    n_clusters: int = 6,
    dataset_id: str | None = None,
) -> tuple[pd.DataFrame, Any | None, np.ndarray | None]:
    """
    Assign each cycle to an operating condition regime.

    Operating Regime Assignment (Step 3)
    ------------------------------------
    - FD001 and FD003 (n_clusters == 1) have exactly 1 flight condition (sea-level);
      they are assigned op_condition = 0, skipping clustering entirely.
    - All other datasets (FD002, FD004) are assigned regime IDs 0..5 by proximity
      to the fixed FD002 K-Means reference centers (FIXED_REGIME_CENTERS).
    - We do NOT refit K-Means per dataset, guaranteeing that regime IDs correspond
      to the exact same physical flight conditions everywhere.

    Parameters
    ----------
    df         : DataFrame with op1, op2, op3 columns
    kmeans     : pre-fitted assigner or KMeans model (optional)
    label_map  : permutation array (optional)
    n_clusters : number of operating conditions (1 or 6)
    dataset_id : optional dataset ID ("FD001", "FD002", etc.)

    Returns
    -------
    (df_with_condition, assigner_or_None, label_map_or_None)
    """
    df = df.copy()

    # --- Single-condition shortcut (FD001 / FD003) ---
    if n_clusters == 1 or dataset_id in ("FD001", "FD003"):
        df["op_condition"] = 0
        return df, None, None

    # --- Multi-condition path (FD002 / FD004): Fixed FD002 centers ---
    assigner = kmeans if (kmeans is not None and hasattr(kmeans, "predict")) else FixedRegimeAssigner()
    op_data = df[OP_COLS].values.astype(np.float64)
    df["op_condition"] = assigner.predict(op_data)

    label_map = label_map if label_map is not None else np.arange(assigner.n_clusters)
    return df, assigner, label_map


# ---------------------------------------------------------------------------
# Engine-level split  (random, by engine ID)
# ---------------------------------------------------------------------------

def split_engines(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    seed:       int   = SPLIT_SEED,
) -> dict[str, list[int]]:
    """
    Randomly partition all engine IDs into train / val / target sets.

    Splits are MUTUALLY EXCLUSIVE at the engine level — no engine appears in
    more than one partition.  The target fraction is the remainder after
    train + val are allocated.

    Parameters
    ----------
    df         : DataFrame with ``engine_id`` column
    train_frac : fraction of engines assigned to train (default 0.70)
    val_frac   : fraction of engines assigned to val   (default 0.15)
    seed       : random seed for reproducibility       (default 42)

    Returns
    -------
    dict with keys "train", "val", "target"; values are sorted lists of
    engine_ids.
    """
    rng = np.random.default_rng(seed)

    all_engines = sorted(df["engine_id"].unique())
    n = len(all_engines)

    # Shuffle
    shuffled = np.array(all_engines)
    rng.shuffle(shuffled)

    n_train  = int(np.floor(n * train_frac))
    n_val    = int(np.floor(n * val_frac))
    # target gets the remainder — avoids rounding loss
    n_target = n - n_train - n_val

    train_engines  = sorted(shuffled[:n_train].tolist())
    val_engines    = sorted(shuffled[n_train : n_train + n_val].tolist())
    target_engines = sorted(shuffled[n_train + n_val :].tolist())

    # Integrity checks
    assert len(target_engines) == n_target
    assert set(train_engines) & set(val_engines)    == set(), "Train/val overlap!"
    assert set(train_engines) & set(target_engines) == set(), "Train/target overlap!"
    assert set(val_engines)   & set(target_engines) == set(), "Val/target overlap!"
    assert len(train_engines) + len(val_engines) + len(target_engines) == n

    return {
        "train":  train_engines,
        "val":    val_engines,
        "target": target_engines,
    }


def assert_no_target_engine_leak(
    train_engines: set[int] | list[int] | Any,
    target_engines: set[int] | list[int] | Any,
    stage_name: str = "fitting",
) -> None:
    """
    Ensure strict data isolation between training and target engines.

    Raises AssertionError if any target engine is present in training data
    during scaler fitting, SSL pretraining, baseline training, or fine-tuning.
    """
    train_set = set(train_engines)
    target_set = set(target_engines)
    overlap = train_set & target_set
    if overlap:
        raise AssertionError(
            f"DATA LEAK DETECTED at stage '{stage_name}': target engines "
            f"{sorted(overlap)} overlap with training engines!"
        )


# ---------------------------------------------------------------------------
# Operating condition mix per split (diagnostic / paper table)
# ---------------------------------------------------------------------------

def condition_mix_per_split(
    df: pd.DataFrame,
    splits: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Return a DataFrame showing the fraction of cycles under each operating
    condition for each split.  Used to verify that all conditions are
    represented in train / val / target.

    Requires ``op_condition`` column (from assign_op_conditions).
    """
    rows = []
    for role, eids in splits.items():
        sub = df[df["engine_id"].isin(eids)]
        total = len(sub)
        for cond in sorted(df["op_condition"].unique()):
            frac = (sub["op_condition"] == cond).sum() / total
            rows.append({"split": role, "condition": cond, "fraction": frac})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-dataset target loading
# ---------------------------------------------------------------------------

def load_target_engines(
    data_dir: str | Path,
    dataset_id: str,
    cap: int = RUL_CAP,
) -> pd.DataFrame:
    """
    Load ALL engines from the training file of any C-MAPSS dataset with full
    labeled trajectories.  Intended for use as a **cross-dataset target**
    (e.g. train on FD002, adapt/evaluate on FD001 or FD004 engines).

    This returns the entire training set (no further splitting), so callers
    can decide how to sample from it for few-shot adaptation.

    Parameters
    ----------
    data_dir   : path to the CMAPSSData folder
    dataset_id : one of "FD001", "FD002", "FD003", "FD004"
    cap        : RUL cap (default 125)

    Returns
    -------
    DataFrame with columns: engine_id, cycle, op1–op3, s1–s21, rul,
    op_condition.  Sensors are raw (un-scaled).
    """
    cfg = _get_cfg(dataset_id)
    df  = load_raw(data_dir, split="train", dataset_id=dataset_id)
    df  = compute_rul(df, cap=cap)
    df, _, _ = assign_op_conditions(
        df, n_clusters=cfg["n_op_conditions"], dataset_id=dataset_id
    )
    return df


# ---------------------------------------------------------------------------
# Scaler out-of-range verification table
# ---------------------------------------------------------------------------

def scaler_range_check(
    scaler,                          # fitted MinMaxScaler
    dfs: dict[str, pd.DataFrame],   # {label: raw_df} — each df must have FEATURE_COLS
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each DataFrame in ``dfs``, apply the scaler and compute the fraction
    of values outside [0, 1] separately for sensor columns and op-setting
    columns.

    Parameters
    ----------
    scaler       : fitted MinMaxScaler (from fit_scaler)
    dfs          : mapping of label → raw DataFrame (un-scaled)
    feature_cols : columns that were used to fit the scaler (default FEATURE_COLS)

    Returns
    -------
    DataFrame with columns:
        dataset | n_engines | n_rows | frac_out_sensors | frac_out_op
    """
    from .constants import SENSOR_COLS, OP_COLS as _OP_COLS
    feature_cols = feature_cols or FEATURE_COLS

    sensor_idx = [feature_cols.index(c) for c in SENSOR_COLS if c in feature_cols]
    op_idx     = [feature_cols.index(c) for c in _OP_COLS    if c in feature_cols]

    rows = []
    for label, df in dfs.items():
        vals = scaler.transform(df[feature_cols].values)     # (N, F)
        out  = (vals < 0) | (vals > 1)

        frac_s  = out[:, sensor_idx].mean() if sensor_idx else float("nan")
        frac_op = out[:, op_idx].mean()     if op_idx     else float("nan")

        rows.append({
            "dataset":          label,
            "n_engines":        int(df["engine_id"].nunique()),
            "n_rows":           len(df),
            "frac_out_sensors": round(float(frac_s),  4),
            "frac_out_op":      round(float(frac_op), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------

def build_engine_splits(
    data_dir: str | Path,
    dataset_id: str = "FD002",
    split_seed: int = SPLIT_SEED,
    kmeans_save_path: str | Path | None = None,
) -> dict:
    """
    End-to-end data pipeline for one C-MAPSS dataset.

    Steps
    -----
    1. Load raw training file for ``dataset_id``
    2. Compute piecewise-linear RUL (capped at RUL_CAP)
    3. Assign operating conditions (K-Means k=n_op, or trivial k=1)
    4. Split engines randomly: 70% train / 15% val / 15% target
    5. Partition DataFrame by split

    Parameters
    ----------
    data_dir         : path to CMAPSSData folder
    dataset_id       : one of "FD001", "FD002", "FD003", "FD004"
    split_seed       : random seed for engine-level split (default 42 → FD002 compat)
    kmeans_save_path : if given, saves {kmeans, label_map} here for reuse
                       on the test set (only meaningful for k>1 datasets)

    Returns
    -------
    dict with keys:
        "df_train"   : DataFrame (train engines, all cycles, with rul + op_condition)
        "df_val"     : DataFrame (val engines)
        "df_target"  : DataFrame (target/unseen engines)
        "splits"     : {"train": [...], "val": [...], "target": [...]}
        "kmeans"     : fitted KMeans model (None for single-condition datasets)
        "label_map"  : centroid sort permutation (None for single-condition datasets)
        "summary"    : engine-count summary dict
        "dataset_id" : the dataset identifier used
        "n_op_conditions": number of operating conditions for this dataset
    """
    data_dir = Path(data_dir)
    cfg      = _get_cfg(dataset_id)
    n_op     = cfg["n_op_conditions"]

    # 1. Load raw
    df = load_raw(data_dir, split="train", dataset_id=dataset_id)

    # 2. Compute RUL
    df = compute_rul(df)

    # 3. Assign operating conditions
    df, kmeans, label_map = assign_op_conditions(df, n_clusters=n_op, dataset_id=dataset_id)

    # 4. Split engines (random, engine-level)
    splits = split_engines(df, seed=split_seed)

    # 5. Partition DataFrame
    df_train  = df[df["engine_id"].isin(splits["train"])].copy()
    df_val    = df[df["engine_id"].isin(splits["val"])].copy()
    df_target = df[df["engine_id"].isin(splits["target"])].copy()

    summary = {
        "n_train_engines":  len(splits["train"]),
        "n_val_engines":    len(splits["val"]),
        "n_target_engines": len(splits["target"]),
        "n_train_rows":     len(df_train),
        "n_val_rows":       len(df_val),
        "n_target_rows":    len(df_target),
        "total_engines":    len(splits["train"]) + len(splits["val"]) + len(splits["target"]),
    }

    # 6. Optionally persist KMeans for test-set condition assignment
    if kmeans_save_path is not None and kmeans is not None:
        save_path = Path(kmeans_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"kmeans": kmeans, "label_map": label_map}, f)

    return {
        "df_train":         df_train,
        "df_val":           df_val,
        "df_target":        df_target,
        "splits":           splits,
        "kmeans":           kmeans,
        "label_map":        label_map,
        "summary":          summary,
        "dataset_id":       dataset_id,
        "n_op_conditions":  n_op,
    }
