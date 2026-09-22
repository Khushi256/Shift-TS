"""
loader.py
=========
Raw data loading, RUL labelling, operating-condition characterisation, and
**engine-level** split for the C-MAPSS FD002 dataset.

Design notes — engine-level split
----------------------------------
FD002's six operating conditions are interleaved within every engine's run.
Investigating the data shows that ~99% of engines have condition 5 (op1≈42)
as their most frequent condition — meaning the "dominant condition" approach
cannot produce a meaningful per-engine environment assignment.

Instead we use a **random engine-level split** (70 / 15 / 15):

    train  (~182 engines) → SSL pretraining + supervised training
    val    ( ~39 engines) → hyperparameter selection / early stopping
    target ( ~39 engines) → unseen cohort for few-shot adaptation

The split is done at the engine level (before windowing) so that no engine
appears in more than one partition, eliminating window-level leakage.

Operating conditions are still characterised via K-Means and attached to every
row (``op_condition`` column, 0–5), which lets downstream analysis verify that
all 6 conditions appear in each split and quantify any condition-frequency
differences across the train / val / target cohorts.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .constants import (
    COLUMNS,
    N_OP_CONDITIONS,
    OP_COLS,
    RUL_CAP,
    SPLIT_SEED,
    TARGET_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_raw(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """
    Load ``{split}_FD002.txt`` into a tidy DataFrame.

    Parameters
    ----------
    data_dir : path to the CMAPSSData folder
    split    : "train" or "test"

    Returns
    -------
    DataFrame with columns defined in constants.COLUMNS (26 cols).
    The trailing blank column is silently dropped.
    """
    path = Path(data_dir) / f"{split}_FD002.txt"
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


def load_test_rul(data_dir: str | Path) -> pd.Series:
    """
    Load ground-truth RUL for the test set.

    Each row in ``RUL_FD002.txt`` is the RUL at the last observed cycle for
    one test engine.  The file may have 259 or 260 rows depending on the
    version — the index is set to match whatever is present.

    Returns
    -------
    pd.Series  index = engine_id (1-indexed), values = true RUL
    """
    path = Path(data_dir) / "RUL_FD002.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
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
# Operating-condition characterisation (for analysis, not for splitting)
# ---------------------------------------------------------------------------

def _sort_kmeans_by_centroid(kmeans: KMeans) -> np.ndarray:
    """
    Return a permutation that sorts K-Means cluster indices by the first
    centroid coordinate (op1 ≈ altitude) ascending, making the label mapping
    deterministic regardless of random init order.
    """
    return np.argsort(kmeans.cluster_centers_[:, 0])


def assign_op_conditions(
    df: pd.DataFrame,
    kmeans: KMeans | None = None,
    label_map: np.ndarray | None = None,
    n_clusters: int = N_OP_CONDITIONS,
) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    """
    Assign each cycle to one of ``n_clusters`` operating conditions via
    K-Means on the three op-setting columns.

    This is used for *characterisation and analysis* — not for splitting.
    In FD002, every engine cycles through all 6 conditions, so the condition
    label cannot be used as an engine-level environment assignment.

    Parameters
    ----------
    df        : DataFrame with op1, op2, op3 columns
    kmeans    : pre-fitted KMeans model (pass when processing test data)
    label_map : sorting permutation from a previous call (pass with kmeans)
    n_clusters: number of operating conditions (default 6)

    Returns
    -------
    (df_with_condition, fitted_kmeans, label_map)
    ``op_condition`` column is an integer in [0, n_clusters).
    Cluster 0 ≡ lowest op1 centroid, cluster 5 ≡ highest.
    """
    df = df.copy()
    op_data = df[OP_COLS].values.astype(np.float64)

    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(op_data)
        label_map = _sort_kmeans_by_centroid(kmeans)

    raw_labels = kmeans.predict(op_data)

    # Re-index so cluster 0 always corresponds to the lowest-op1 regime
    inverse_map = np.zeros(n_clusters, dtype=int)
    for new_idx, old_idx in enumerate(label_map):
        inverse_map[old_idx] = new_idx

    df["op_condition"] = inverse_map[raw_labels]
    return df, kmeans, label_map


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

    # Integrity check
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


# ---------------------------------------------------------------------------
# Operating condition mix per split (diagnostic / paper table)
# ---------------------------------------------------------------------------

def condition_mix_per_split(
    df: pd.DataFrame,
    splits: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Return a DataFrame showing the fraction of cycles under each operating
    condition for each split.  Used to verify that all 6 conditions are
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
# High-level pipeline helper
# ---------------------------------------------------------------------------

def build_engine_splits(
    data_dir: str | Path,
    kmeans_save_path: str | Path | None = None,
) -> dict:
    """
    End-to-end data pipeline for Core 1.

    Steps
    -----
    1. Load raw FD002 training file
    2. Compute piecewise-linear RUL (capped at RUL_CAP)
    3. Fit K-Means (k=6) on op-settings to label operating conditions
    4. Split engines randomly: 70% train / 15% val / 15% target
    5. Partition DataFrame by split

    Parameters
    ----------
    data_dir         : path to CMAPSSData folder
    kmeans_save_path : if given, saves {kmeans, label_map} here for reuse
                       on the test set

    Returns
    -------
    dict with keys:
        "df_train"   : DataFrame (train engines, all cycles, with rul + op_condition)
        "df_val"     : DataFrame (val engines)
        "df_target"  : DataFrame (target/unseen engines)
        "splits"     : {"train": [...], "val": [...], "target": [...]}
        "kmeans"     : fitted KMeans model
        "label_map"  : centroid sort permutation (needed for test-set reuse)
        "summary"    : engine-count summary dict
    """
    data_dir = Path(data_dir)

    # 1. Load raw
    df = load_raw(data_dir, split="train")

    # 2. Compute RUL
    df = compute_rul(df)

    # 3. Assign operating conditions (for analysis — not for splitting)
    df, kmeans, label_map = assign_op_conditions(df)

    # 4. Split engines (random, engine-level)
    splits = split_engines(df)

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
    if kmeans_save_path is not None:
        save_path = Path(kmeans_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"kmeans": kmeans, "label_map": label_map}, f)

    return {
        "df_train":  df_train,
        "df_val":    df_val,
        "df_target": df_target,
        "splits":    splits,
        "kmeans":    kmeans,
        "label_map": label_map,
        "summary":   summary,
    }
