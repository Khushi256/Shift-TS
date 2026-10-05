"""
constants.py
============
All fixed configuration for the SHIFT-TS / C-MAPSS pipeline.
Change values here — not scattered across modules.

Dataset-agnostic design
-----------------------
Feature columns (sensors + op-settings) are GLOBAL across all four FD00x
datasets.  The same 14 informative sensors and 3 op-setting columns are used
everywhere.  Per-dataset differences that affect runtime behaviour are encoded
in DATASET_CONFIG below.

Operating conditions note
--------------------------
FD002 / FD004 have 6 discrete operating conditions.
FD001 / FD003 have exactly 1 operating condition (sea-level).
DATASET_CONFIG encodes n_op_conditions per dataset; when n_op_conditions == 1
the K-Means clustering step is skipped and op_condition is set to 0 for every row.

We use a **random engine-level split** (70 / 15 / 15) inside each dataset's
training file, and the environment nomenclature refers to the split role:

    Train  (Env A/B)  — ~70% of engines   → SSL + supervised training
    Val    (Env C)    — ~15% of engines    → hyperparameter selection
    Target (Env D)    — ~15% of engines    → few-shot adaptation + evaluation

All operating conditions appear in every split for FD002/FD004 (since each
engine sees all conditions), and trivially for FD001/FD003 (single condition).
The "distribution shift" is therefore an engine-cohort shift, not a
condition-exclusion shift.  This is documented explicitly in the paper.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Raw data column layout
# ---------------------------------------------------------------------------
# All FD00x files share the same 26 space-delimited columns (last is always
# blank/NaN and gets dropped during loading).
COLUMNS = [
    "engine_id", "cycle",
    "op1", "op2", "op3",
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

OP_COLS = ["op1", "op2", "op3"]

ALL_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

# ---------------------------------------------------------------------------
# Informative sensors  (GLOBAL — applied to ALL datasets)
# ---------------------------------------------------------------------------
# Sensors with near-zero variance in FD002 are uninformative and dropped.
# We keep this list fixed across datasets for a single consistent input space.
# Known constant / near-constant sensors in FD002:
#   s1, s5, s6, s10, s16, s18, s19
DROPPED_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}

SENSOR_COLS = [s for s in ALL_SENSOR_COLS if s not in DROPPED_SENSORS]
# → s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21  (14 sensors)

# Feature columns fed to the model: sensors + op-settings
FEATURE_COLS = SENSOR_COLS + OP_COLS   # 14 + 3 = 17 features
N_FEATURES   = len(FEATURE_COLS)       # 17

# ---------------------------------------------------------------------------
# Temporal windowing
# ---------------------------------------------------------------------------
WINDOW_SIZE = 30    # sliding window length (cycles)
STRIDE      = 1     # stride between windows during training

# ---------------------------------------------------------------------------
# RUL target
# ---------------------------------------------------------------------------
RUL_CAP = 125   # piecewise-linear cap: engines far from failure are treated
                # as having RUL = 125 to focus learning on near-failure signal
                # Kept at 125 for ALL datasets (consistent with CMAPSS literature)

# ---------------------------------------------------------------------------
# Engine-level split fractions
# ---------------------------------------------------------------------------
# Splits are done RANDOMLY at the engine level.
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
TARGET_FRAC = 0.15
SPLIT_SEED  = 42     # default seed for FD002 backward-compatibility

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
# Only fields that genuinely differ between datasets live here.
# Feature columns, RUL_CAP, window size, stride, and split fractions are
# IDENTICAL across all four datasets.
#
#   n_op_conditions : number of K-Means clusters for operating conditions.
#                     Set to 1 for FD001/FD003 (single flight condition).
#                     When == 1, clustering is skipped; op_condition = 0.
DATASET_CONFIG: dict[str, dict] = {
    "FD001": {
        "train_file":      "train_FD001.txt",
        "test_file":       "test_FD001.txt",
        "rul_file":        "RUL_FD001.txt",
        "n_op_conditions": 1,
        "description":     "Single operating condition, one fault mode",
    },
    "FD002": {
        "train_file":      "train_FD002.txt",
        "test_file":       "test_FD002.txt",
        "rul_file":        "RUL_FD002.txt",
        "n_op_conditions": 6,
        "description":     "Six operating conditions, one fault mode",
    },
    "FD003": {
        "train_file":      "train_FD003.txt",
        "test_file":       "test_FD003.txt",
        "rul_file":        "RUL_FD003.txt",
        "n_op_conditions": 1,
        "description":     "Single operating condition, two fault modes",
    },
    "FD004": {
        "train_file":      "train_FD004.txt",
        "test_file":       "test_FD004.txt",
        "rul_file":        "RUL_FD004.txt",
        "n_op_conditions": 6,
        "description":     "Six operating conditions, two fault modes",
    },
}

VALID_DATASET_IDS = list(DATASET_CONFIG.keys())

# ---------------------------------------------------------------------------
# Operating-condition cluster count for backward-compat (FD002)
# ---------------------------------------------------------------------------
N_OP_CONDITIONS = DATASET_CONFIG["FD002"]["n_op_conditions"]   # = 6

# ---------------------------------------------------------------------------
# Fixed FD002 Operating Regime Centers (sorted by op1 / altitude ascending)
# ---------------------------------------------------------------------------
# All datasets use these 6 fixed reference centroids for regime assignment.
# We do NOT refit K-Means per dataset to ensure cross-dataset regime consistency.
# FD001 and FD003 map to regime 0 (sea-level).
FIXED_REGIME_CENTERS = np.array([
    [0.00150451, 0.00049434, 100.0],  # Regime 0: Sea level / static ground
    [10.00297127, 0.25049503, 100.0],  # Regime 1: 10k ft, 0.25 Mach, TRA 100
    [20.00299895, 0.70051519, 100.0],  # Regime 2: 20k ft, 0.70 Mach, TRA 100
    [25.00303803, 0.62050187, 60.0],   # Regime 3: 25k ft, 0.62 Mach, TRA 60
    [35.00304897, 0.84050058, 100.0],  # Regime 4: 35k ft, 0.84 Mach, TRA 100
    [42.00297633, 0.84048518, 100.0],  # Regime 5: 42k ft, 0.84 Mach, TRA 100
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Fixed Operating-Condition Scaling Ranges (for sources with n_op == 1)
# ---------------------------------------------------------------------------
# When training on single-condition sources (FD001 / FD003), fitting min-max on
# narrow operating ranges would artificially magnify noise into [0, 1].
# Instead, we scale op columns using the fixed global operating envelop:
#   op1 (altitude): 0 to 42 (k-ft, NOT 0-42000)
#   op2 (Mach)    : 0 to 0.84
#   op3 (TRA)     : 20 to 100
FIXED_OP_RANGES: dict[str, tuple[float, float]] = {
    "op1": (0.0, 42.0),
    "op2": (0.0, 0.84),
    "op3": (20.0, 100.0),
}

# ---------------------------------------------------------------------------
# Transfer & Domain-Shift Scenarios (Step 3 taxonomy)
# ---------------------------------------------------------------------------
# Evaluates in-domain, fault-mode shift, operating-condition shift, and combined shift.
# Note: Leave-one-regime-out is dropped because >96% of 30-cycle sliding windows
# in FD002/FD004 cover all 6 regimes, making within-engine regime masking infeasible.
SCENARIOS: dict[str, dict] = {
    "ID_FD002": {
        "source": "FD002",
        "target": "FD002",
        "shift_type": "in_domain",
        "description": "In-domain FD002 (engine cohort shift)",
        "fixed_op_scaling": False,
        "op_ranges": "minmax_fit",
        "regimes_seen": "0, 1, 2, 3, 4, 5",
        "regimes_unseen": "none",
    },
    "ID_FD004": {
        "source": "FD004",
        "target": "FD004",
        "shift_type": "in_domain",
        "description": "In-domain FD004 (engine cohort shift)",
        "fixed_op_scaling": False,
        "op_ranges": "minmax_fit",
        "regimes_seen": "0, 1, 2, 3, 4, 5",
        "regimes_unseen": "none",
    },
    "FAULT_1": {
        "source": "FD001",
        "target": "FD003",
        "shift_type": "fault_mode",
        "description": "FD001 -> FD003 (1 fault -> 2 faults, single op condition)",
        "fixed_op_scaling": True,
        "op_ranges": FIXED_OP_RANGES,
        "regimes_seen": "0",
        "regimes_unseen": "none",
    },
    "FAULT_2": {
        "source": "FD002",
        "target": "FD004",
        "shift_type": "fault_mode",
        "description": "FD002 -> FD004 (1 fault -> 2 faults, 6 op conditions)",
        "fixed_op_scaling": False,
        "op_ranges": "minmax_fit",
        "regimes_seen": "0, 1, 2, 3, 4, 5",
        "regimes_unseen": "none",
    },
    "OPCOND_1": {
        "source": "FD001",
        "target": "FD002",
        "shift_type": "operating_condition",
        "description": "FD001 -> FD002 (1 condition -> 6 operating conditions)",
        "fixed_op_scaling": True,
        "op_ranges": FIXED_OP_RANGES,
        "regimes_seen": "0",
        "regimes_unseen": "1, 2, 3, 4, 5",
    },
    "OPCOND_2": {
        "source": "FD003",
        "target": "FD004",
        "shift_type": "operating_condition",
        "description": "FD003 -> FD004 (1 condition -> 6 operating conditions)",
        "fixed_op_scaling": True,
        "op_ranges": FIXED_OP_RANGES,
        "regimes_seen": "0",
        "regimes_unseen": "1, 2, 3, 4, 5",
    },
    "COMBINED": {
        "source": "FD001",
        "target": "FD004",
        "shift_type": "combined",
        "description": "FD001 -> FD004 (1 cond, 1 fault -> 6 conds, 2 faults)",
        "fixed_op_scaling": True,
        "op_ranges": FIXED_OP_RANGES,
        "regimes_seen": "0",
        "regimes_unseen": "1, 2, 3, 4, 5",
    },
}

# ---------------------------------------------------------------------------
# Environment role labels (for documentation and logging)
# ---------------------------------------------------------------------------
SPLIT_TO_ENV = {
    "train":  "A/B (train source)",
    "val":    "C (validation)",
    "target": "D (unseen target)",
}
