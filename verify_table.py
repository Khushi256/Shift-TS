"""
verify_table.py
===============
Generates the dataset verification table:
  - Engine counts, n_op_conditions per dataset
  - Fraction of feature values outside [0,1] when a FD002-fitted scaler
    is applied to each other dataset (sensors vs op cols)

Run from project root:
    python verify_table.py
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import pandas as pd
from src.data.loader import build_engine_splits, load_raw, scaler_range_check
from src.data.preprocessing import fit_scaler
from src.data.constants import DATASET_CONFIG, FEATURE_COLS

DATA_DIR = Path("CMAPSSData")

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

print("=" * 70)
print("SHIFT-TS  —  Dataset Verification Table")
print("=" * 70)

# ── Part 1: engine counts & op conditions ─────────────────────────────────
print("\n[1] Dataset overview\n")
rows = []
for dsid in ALL_DATASETS:
    out = build_engine_splits(DATA_DIR, dataset_id=dsid)
    s   = out["summary"]
    rows.append({
        "Dataset":    dsid,
        "Description": DATASET_CONFIG[dsid]["description"],
        "N_engines":  s["total_engines"],
        "N_train":    s["n_train_engines"],
        "N_val":      s["n_val_engines"],
        "N_target":   s["n_target_engines"],
        "N_op":       out["n_op_conditions"],
        "KMeans_fitted": out["kmeans"] is not None,
    })

overview = pd.DataFrame(rows)
print(overview.to_string(index=False))

# ── Part 2: scaler range check ────────────────────────────────────────────
print("\n[2] Fraction of values outside [0,1] when using FD002-fitted scaler\n")
fd002_out  = build_engine_splits(DATA_DIR, dataset_id="FD002")
fd002_scaler = fit_scaler(fd002_out["df_train"])

raw_dfs = {}
for dsid in ALL_DATASETS:
    raw = load_raw(DATA_DIR, "train", dataset_id=dsid)
    raw_dfs[dsid] = raw

table = scaler_range_check(fd002_scaler, raw_dfs)
print(table.to_string(index=False))

print("\n" + "=" * 70)
print("Note: FD001/FD003 op-cols are constant (1 condition) -> op frac_out")
print("      may be 0 or 1 depending on whether the single condition")
print("      falls within FD002's op range. Sensor columns should be")
print("      mostly within [0,1] since the 14 sensors are shared.")
print("=" * 70)
