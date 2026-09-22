"""
constants.py
============
All fixed configuration for the SHIFT-TS / C-MAPSS FD002 pipeline.
Change values here — not scattered across modules.

Operating conditions note
--------------------------
FD002 has 6 discrete operating conditions (recovered by K-Means on op-settings).
In FD002, *every engine cycles through all 6 conditions* within its run.
Therefore, "environment" is not a per-engine property — it describes the
multi-condition distribution seen during a given experiment split.

We use a **random engine-level split** (stratified to preserve the condition
mix across splits), and the environment nomenclature refers to the split role:

    Train  (Env A/B)  — ~70% of engines   → SSL + supervised training
    Val    (Env C)    — ~15% of engines    → hyperparameter selection
    Target (Env D)    — ~15% of engines    → few-shot adaptation + evaluation

All 6 operating conditions appear in every split (since each engine sees all 6).
The "distribution shift" is therefore an engine-cohort shift, not a
condition-exclusion shift.  This is documented explicitly in the paper.
"""

# ---------------------------------------------------------------------------
# Raw data column layout
# ---------------------------------------------------------------------------
# FD002 has 26 space-delimited columns (the last is always blank/NaN).
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
# Informative sensors
# ---------------------------------------------------------------------------
# Sensors with near-zero variance in FD002 are uninformative and dropped.
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

# ---------------------------------------------------------------------------
# Operating-condition clustering
# ---------------------------------------------------------------------------
N_OP_CONDITIONS = 6   # K-Means k; recovers the 6 known C-MAPSS FD002 regimes

# ---------------------------------------------------------------------------
# Engine-level split fractions
# ---------------------------------------------------------------------------
# Splits are done RANDOMLY at the engine level (not by condition, since every
# engine sees all 6 conditions in FD002).
TRAIN_FRAC  = 0.70   # ~182 engines
VAL_FRAC    = 0.15   # ~39 engines
TARGET_FRAC = 0.15   # ~39 engines
SPLIT_SEED  = 42     # fixed seed for reproducibility

# ---------------------------------------------------------------------------
# Environment role labels (for documentation and logging)
# ---------------------------------------------------------------------------
SPLIT_TO_ENV = {
    "train":  "A/B (train source)",
    "val":    "C (validation)",
    "target": "D (unseen target)",
}
