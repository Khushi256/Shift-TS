"""
verify_table.py
===============
Generates the dataset & scenario verification table:
  - Engine counts, n_op_conditions per dataset
  - Transfer & Domain-Shift Scenarios (Step 3 taxonomy) with:
      * "regimes seen in source_train"
      * "regimes unseen in target"
      * "fixed_op_scaling"
  - Fraction of feature values outside [0,1] when a FD002-fitted scaler
    is applied to each other dataset (sensors vs op cols)

Run from project root:
    python verify_table.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import pandas as pd
from src.data.loader import build_engine_splits, load_raw, scaler_range_check
from src.data.preprocessing import fit_scaler
from src.data.constants import DATASET_CONFIG, FEATURE_COLS, SCENARIOS

DATA_DIR = Path("CMAPSSData")
ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

print("=" * 80)
print("SHIFT-TS  —  Dataset & Scenario Verification Tables (Step 3)")
print("=" * 80)

# ── Part 1: engine counts & op conditions ─────────────────────────────────
print("\n[1] Dataset Overview\n")
rows = []
for dsid in ALL_DATASETS:
    out = build_engine_splits(DATA_DIR, dataset_id=dsid)
    s   = out["summary"]
    rows.append({
        "Dataset":        dsid,
        "Description":    DATASET_CONFIG[dsid]["description"],
        "N_engines":      s["total_engines"],
        "N_train":        s["n_train_engines"],
        "N_val":          s["n_val_engines"],
        "N_target":       s["n_target_engines"],
        "N_op":           out["n_op_conditions"],
        "Assigner":       "FixedRegimeAssigner" if out["kmeans"] is not None else "None (Single-op)",
    })

overview = pd.DataFrame(rows)
print(overview.to_string(index=False))

# ── Part 2: Step 3 Transfer & Domain-Shift Scenarios ──────────────────────
print("\n[2] Transfer & Domain-Shift Scenarios (Step 3 Taxonomy)\n")
scenario_rows = []
for sname, scfg in SCENARIOS.items():
    scenario_rows.append({
        "Scenario":                      sname,
        "Source":                        scfg["source"],
        "Target":                        scfg["target"],
        "Shift Type":                    scfg["shift_type"],
        "regimes seen in source_train":  scfg["regimes_seen"],
        "regimes unseen in target":      scfg["regimes_unseen"],
        "fixed_op_scaling":              scfg["fixed_op_scaling"],
        "Description":                   scfg["description"],
    })

scenario_table = pd.DataFrame(scenario_rows)
print(scenario_table.to_string(index=False))

# ── Part 3: scaler range check ────────────────────────────────────────────
print("\n[3] Scaler Range Check (FD002-fitted Scaler Applied Cross-Dataset)\n")
fd002_out    = build_engine_splits(DATA_DIR, dataset_id="FD002")
fd002_scaler = fit_scaler(fd002_out["df_train"])

raw_dfs = {}
for dsid in ALL_DATASETS:
    raw = load_raw(DATA_DIR, "train", dataset_id=dsid)
    raw_dfs[dsid] = raw

table = scaler_range_check(fd002_scaler, raw_dfs)
print(table.to_string(index=False))

print("\n" + "=" * 80)
print("Step 3 Configuration Summary:")
print("  - Fixed regime centers: 6 FD002 reference centers (sorted by altitude).")
print("  - Operating regime assignment: nearest center for FD002/FD004; regime 0 for FD001/FD003.")
print("  - Fixed op scaling for single-op sources (FD001/FD003): op1 [0, 42], op2 [0, 0.84], op3 [20, 100].")
print("  - Evaluated scenarios: ID_FD002, ID_FD004, FAULT_1, FAULT_2, OPCOND_1, OPCOND_2, COMBINED.")
print("=" * 80)
