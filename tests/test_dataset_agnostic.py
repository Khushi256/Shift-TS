"""
test_dataset_agnostic.py
========================
Smoke tests for the dataset-agnostic data pipeline.

Tests
-----
1. FD002 backward-compatibility: engine counts and split IDs match the
   previous hard-coded run (seed=42, 260 engines → 182/39/39).
2. No engine appears in two splits for every dataset.
3. Feature dimension is 17 for all datasets (global feature list).
4. Single-condition bypass: FD001/FD003 get op_condition=0 everywhere,
   no KMeans is run.
5. Zero-range guard: fit_scaler produces finite transforms on all datasets.
6. load_target_engines: returns all engines with rul column.
7. scaler_range_check produces a table with correct shape.

Run with:
    pytest tests/test_dataset_agnostic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.constants import (
    DATASET_CONFIG,
    FEATURE_COLS,
    N_FEATURES,
    SPLIT_SEED,
)
from src.data.loader import (
    build_engine_splits,
    load_raw,
    load_target_engines,
    scaler_range_check,
)
from src.data.preprocessing import fit_scaler, apply_scaler

DATA_DIR = Path(__file__).parent.parent / "CMAPSSData"

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd002_out():
    """Canonical FD002 pipeline output (seed=42 — must match old run)."""
    return build_engine_splits(DATA_DIR, dataset_id="FD002", split_seed=42)


@pytest.fixture(scope="module", params=ALL_DATASETS)
def any_dataset_out(request):
    return request.param, build_engine_splits(DATA_DIR, dataset_id=request.param)


# ---------------------------------------------------------------------------
# 1. FD002 backward-compatibility (smoke — numbers must match previous run)
# ---------------------------------------------------------------------------

class TestFD002BackwardCompat:

    def test_total_engines_260(self, fd002_out):
        """FD002 training file must have exactly 260 engines."""
        assert fd002_out["summary"]["total_engines"] == 260, (
            f"Expected 260 engines, got {fd002_out['summary']['total_engines']}"
        )

    def test_split_counts_match_previous(self, fd002_out):
        """70/15/15 split of 260 engines → 182 / 39 / 39."""
        s = fd002_out["summary"]
        assert s["n_train_engines"]  == 182, f"Train: {s['n_train_engines']}"
        assert s["n_val_engines"]    == 39,  f"Val  : {s['n_val_engines']}"
        assert s["n_target_engines"] == 39,  f"Target: {s['n_target_engines']}"

    def test_split_ids_reproducible(self):
        """Two calls with same seed must produce identical engine ID lists."""
        out1 = build_engine_splits(DATA_DIR, dataset_id="FD002", split_seed=42)
        out2 = build_engine_splits(DATA_DIR, dataset_id="FD002", split_seed=42)
        assert out1["splits"]["train"]  == out2["splits"]["train"]
        assert out1["splits"]["val"]    == out2["splits"]["val"]
        assert out1["splits"]["target"] == out2["splits"]["target"]

    def test_n_op_conditions_is_6(self, fd002_out):
        assert fd002_out["n_op_conditions"] == 6
        assert fd002_out["kmeans"] is not None
        df = fd002_out["df_train"]
        assert "op_condition" in df.columns
        assert sorted(df["op_condition"].unique()) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 2. No engine appears in two splits — for ALL datasets
# ---------------------------------------------------------------------------

class TestNoEngineLeakAllDatasets:

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_no_overlap(self, dataset_id):
        out = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        splits = out["splits"]
        train  = set(splits["train"])
        val    = set(splits["val"])
        target = set(splits["target"])

        assert train  & val    == set(), f"[{dataset_id}] Train/val overlap"
        assert train  & target == set(), f"[{dataset_id}] Train/target overlap"
        assert val    & target == set(), f"[{dataset_id}] Val/target overlap"

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_all_engines_covered(self, dataset_id):
        out = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        splits = out["splits"]
        raw = load_raw(DATA_DIR, "train", dataset_id=dataset_id)
        all_engines = set(raw["engine_id"].unique())
        split_engines = (
            set(splits["train"]) | set(splits["val"]) | set(splits["target"])
        )
        assert split_engines == all_engines, (
            f"[{dataset_id}] Missing engines: {all_engines - split_engines}"
        )


# ---------------------------------------------------------------------------
# 3. Feature dimension is 17 for ALL datasets
# ---------------------------------------------------------------------------

class TestFeatureDimension:

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_n_features_is_17(self, dataset_id):
        """Global feature list gives 17 columns for every dataset."""
        from src.data.dataset import CMAPSSDataset
        out    = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        scaler = fit_scaler(out["df_train"])
        df_s   = apply_scaler(out["df_train"], scaler)
        ds     = CMAPSSDataset(df_s)

        x, _, _ = ds[0]
        assert x.shape[-1] == N_FEATURES == 17, (
            f"[{dataset_id}] Expected 17 features, got {x.shape[-1]}"
        )
        assert ds.n_features == 17

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_feature_cols_unchanged(self, dataset_id):
        """FEATURE_COLS must be the same list regardless of dataset."""
        assert len(FEATURE_COLS) == 17


# ---------------------------------------------------------------------------
# 4. Single-condition bypass for FD001 / FD003
# ---------------------------------------------------------------------------

class TestSingleConditionBypass:

    @pytest.mark.parametrize("dataset_id", ["FD001", "FD003"])
    def test_op_condition_all_zero(self, dataset_id):
        out = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        for split_df in [out["df_train"], out["df_val"], out["df_target"]]:
            assert (split_df["op_condition"] == 0).all(), (
                f"[{dataset_id}] op_condition is not uniformly 0"
            )

    @pytest.mark.parametrize("dataset_id", ["FD001", "FD003"])
    def test_kmeans_is_none(self, dataset_id):
        out = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        assert out["kmeans"]    is None, f"[{dataset_id}] KMeans should be None"
        assert out["label_map"] is None, f"[{dataset_id}] label_map should be None"

    @pytest.mark.parametrize("dataset_id", ["FD002", "FD004"])
    def test_kmeans_fitted_for_multi_condition(self, dataset_id):
        out = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        assert out["kmeans"]    is not None
        assert out["label_map"] is not None
        assert out["n_op_conditions"] == 6


# ---------------------------------------------------------------------------
# 5. Zero-range guard: fit_scaler produces finite transforms on all datasets
# ---------------------------------------------------------------------------

class TestScalerFinite:

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_transform_is_finite(self, dataset_id):
        """apply_scaler must never produce NaN or inf for any dataset."""
        out    = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        scaler = fit_scaler(out["df_train"])
        for split_name, df in [
            ("train",  out["df_train"]),
            ("val",    out["df_val"]),
            ("target", out["df_target"]),
        ]:
            df_s = apply_scaler(df, scaler)
            vals = df_s[FEATURE_COLS].values
            assert np.isfinite(vals).all(), (
                f"[{dataset_id}/{split_name}] Non-finite values after scaling"
            )

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_train_in_unit_range(self, dataset_id):
        """Train data scaled by its own scaler must lie in [0-eps, 1+eps]."""
        out    = build_engine_splits(DATA_DIR, dataset_id=dataset_id)
        scaler = fit_scaler(out["df_train"])
        df_s   = apply_scaler(out["df_train"], scaler)
        vals   = df_s[FEATURE_COLS].values
        assert vals.min() >= -1e-5, f"[{dataset_id}] Scaled train values < 0"
        assert vals.max() <= 1 + 1e-5, f"[{dataset_id}] Scaled train values > 1"


# ---------------------------------------------------------------------------
# 6. load_target_engines
# ---------------------------------------------------------------------------

class TestLoadTargetEngines:

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_has_rul_column(self, dataset_id):
        df = load_target_engines(DATA_DIR, dataset_id=dataset_id)
        assert "rul" in df.columns, f"[{dataset_id}] Missing 'rul' column"
        assert "op_condition" in df.columns

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_rul_nonnegative_and_capped(self, dataset_id):
        from src.data.constants import RUL_CAP
        df = load_target_engines(DATA_DIR, dataset_id=dataset_id)
        assert df["rul"].min() >= 0
        assert df["rul"].max() <= RUL_CAP

    @pytest.mark.parametrize("dataset_id", ALL_DATASETS)
    def test_returns_all_engines(self, dataset_id):
        """load_target_engines must return ALL engines from the training file."""
        raw = load_raw(DATA_DIR, "train", dataset_id=dataset_id)
        df  = load_target_engines(DATA_DIR, dataset_id=dataset_id)
        assert set(df["engine_id"].unique()) == set(raw["engine_id"].unique()), (
            f"[{dataset_id}] Engine sets differ"
        )


# ---------------------------------------------------------------------------
# 7. scaler_range_check table
# ---------------------------------------------------------------------------

class TestScalerRangeCheck:

    def test_table_shape(self):
        """Table must have one row per dataset label passed in."""
        fd002_out = build_engine_splits(DATA_DIR, dataset_id="FD002")
        scaler    = fit_scaler(fd002_out["df_train"])

        dfs = {}
        for dsid in ALL_DATASETS:
            raw = load_raw(DATA_DIR, "train", dataset_id=dsid)
            dfs[dsid] = raw

        table = scaler_range_check(scaler, dfs)
        assert len(table) == len(ALL_DATASETS), (
            f"Expected {len(ALL_DATASETS)} rows, got {len(table)}"
        )
        for col in ["dataset", "n_engines", "n_rows", "frac_out_sensors", "frac_out_op"]:
            assert col in table.columns, f"Missing column: {col}"

    def test_fd002_train_zero_out_of_range(self):
        """When fitting AND evaluating on FD002 train, frac_out should be ~0."""
        fd002_out = build_engine_splits(DATA_DIR, dataset_id="FD002")
        scaler    = fit_scaler(fd002_out["df_train"])
        table     = scaler_range_check(scaler, {"FD002_train": fd002_out["df_train"]})
        row = table[table["dataset"] == "FD002_train"].iloc[0]
        assert row["frac_out_sensors"] < 0.01, (
            f"FD002 train sensors out of range: {row['frac_out_sensors']:.4f}"
        )
