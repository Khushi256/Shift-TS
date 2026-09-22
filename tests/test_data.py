"""
test_data.py
============
Unit tests for the Core 1 data pipeline.

Run with:
    pytest tests/test_data.py -v

All tests use the real CMAPSSData/FD002 files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Make src importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.constants import (
    FEATURE_COLS,
    N_FEATURES,
    RUL_CAP,
    WINDOW_SIZE,
)
from src.data.loader import (
    assign_op_conditions,
    build_engine_splits,
    compute_rul,
    load_raw,
    load_test_rul,
    split_engines,
    condition_mix_per_split,
)
from src.data.preprocessing import (
    apply_scaler,
    fit_scaler,
    load_scaler,
    save_scaler,
)
from src.data.dataset import CMAPSSDataset, FewShotDataset, SSLDataset

DATA_DIR = Path(__file__).parent.parent / "CMAPSSData"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_train():
    return load_raw(DATA_DIR, "train")


@pytest.fixture(scope="module")
def pipeline_output():
    return build_engine_splits(DATA_DIR)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:

    def test_raw_train_shape(self, raw_train):
        """Should load rows and the expected columns."""
        assert len(raw_train) > 0
        for col in ["engine_id", "cycle", "op1", "op2", "op3"]:
            assert col in raw_train.columns, f"Missing column: {col}"

    def test_raw_train_n_engines(self, raw_train):
        """FD002 training set has exactly 260 engines."""
        n = raw_train["engine_id"].nunique()
        assert n == 260, f"Expected 260 engines, got {n}"

    def test_compute_rul_cap(self, raw_train):
        """RUL values must not exceed RUL_CAP."""
        df = compute_rul(raw_train)
        assert "rul" in df.columns
        assert df["rul"].max() <= RUL_CAP
        assert df["rul"].min() == 0  # last cycle always has RUL=0

    def test_compute_rul_monotone(self, raw_train):
        """Within each engine, RUL should be non-increasing cycle-over-cycle."""
        df = compute_rul(raw_train)
        for _, group in df.groupby("engine_id"):
            g = group.sort_values("cycle")
            diffs = g["rul"].diff().dropna()
            # RUL decreases or stays flat (plateau at cap); never increases
            assert (diffs <= 0.01).all(), "RUL increased within an engine!"

    def test_assign_op_conditions(self, raw_train):
        """Each cycle should be assigned to one of 6 clusters."""
        df, kmeans, label_map = assign_op_conditions(raw_train)
        assert "op_condition" in df.columns
        unique_conds = sorted(df["op_condition"].unique())
        assert len(unique_conds) == 6
        assert min(unique_conds) == 0
        assert max(unique_conds) == 5

    def test_op_conditions_stable_with_refitting(self, raw_train):
        """Fitting twice with same seed must give identical cluster labels."""
        df1, km1, lm1 = assign_op_conditions(raw_train)
        df2, km2, lm2 = assign_op_conditions(raw_train)
        np.testing.assert_array_equal(
            df1["op_condition"].values,
            df2["op_condition"].values,
        )

    def test_load_test_rul_shape(self):
        """RUL_FD002.txt must have at least 1 value per test engine."""
        rul = load_test_rul(DATA_DIR)
        # FD002 test set has 259 engines in some versions, 260 in others
        assert len(rul) >= 259
        assert rul.index.name == "engine_id"
        assert rul.min() >= 0


# ---------------------------------------------------------------------------
# Engine-level split integrity
# ---------------------------------------------------------------------------

class TestEngineSplit:

    def test_no_overlap_between_splits(self, pipeline_output):
        """Core invariant: no engine_id appears in more than one split."""
        splits = pipeline_output["splits"]
        train  = set(splits["train"])
        val    = set(splits["val"])
        target = set(splits["target"])

        assert train  & val    == set(), "Train/val engine overlap!"
        assert train  & target == set(), "Train/target engine overlap!"
        assert val    & target == set(), "Val/target engine overlap!"

    def test_all_engines_accounted_for(self, pipeline_output):
        """Every engine in the training file belongs to exactly one split."""
        splits = pipeline_output["splits"]
        all_split_engines = (
            set(splits["train"]) | set(splits["val"]) | set(splits["target"])
        )
        all_engines = set(
            load_raw(DATA_DIR, "train")["engine_id"].unique()
        )
        assert all_split_engines == all_engines

    def test_splits_non_empty(self, pipeline_output):
        splits = pipeline_output["splits"]
        assert len(splits["train"])  > 0, "Train split is empty"
        assert len(splits["val"])    > 0, "Val split is empty"
        assert len(splits["target"]) > 0, "Target split is empty"

    def test_split_engine_counts(self, pipeline_output):
        """All splits have a reasonable number of engines."""
        s = pipeline_output["summary"]
        print(f"\n  Train engines : {s['n_train_engines']}")
        print(f"  Val engines   : {s['n_val_engines']}")
        print(f"  Target engines: {s['n_target_engines']}")
        assert s["n_train_engines"]  >= 10
        assert s["n_val_engines"]    >= 5
        assert s["n_target_engines"] >= 5
        assert s["total_engines"] == 260

    def test_all_conditions_in_each_split(self, pipeline_output):
        """
        All 6 operating conditions should appear in each split.
        This verifies the random split doesn't accidentally exclude a condition.
        """
        df_full = load_raw(DATA_DIR, "train")
        df_full = compute_rul(df_full)
        df_full, _, _ = assign_op_conditions(df_full)

        splits = pipeline_output["splits"]
        mix = condition_mix_per_split(df_full, splits)

        for split_name in ["train", "val", "target"]:
            fracs = mix[mix["split"] == split_name]["fraction"].values
            assert (fracs > 0).all(), \
                f"Some condition missing entirely from {split_name} split"

    def test_split_reproducibility(self):
        """Same seed must always produce the same split."""
        out1 = build_engine_splits(DATA_DIR)
        out2 = build_engine_splits(DATA_DIR)
        assert out1["splits"]["train"]  == out2["splits"]["train"]
        assert out1["splits"]["val"]    == out2["splits"]["val"]
        assert out1["splits"]["target"] == out2["splits"]["target"]


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------

class TestScaler:

    def test_scaler_fits_on_train_only(self, pipeline_output):
        """Scaler fitted on train must produce [0,1]-bounded train features."""
        df_train = pipeline_output["df_train"]
        df_val   = pipeline_output["df_val"]

        scaler = fit_scaler(df_train)
        df_train_scaled = apply_scaler(df_train, scaler)
        df_val_scaled   = apply_scaler(df_val, scaler)

        # Train data must be in [0, 1]
        for col in FEATURE_COLS:
            vals = df_train_scaled[col].values
            assert vals.min() >= -1e-6,    f"Train col {col} below 0"
            assert vals.max() <= 1 + 1e-6, f"Train col {col} above 1"

        # Val data may be outside [0,1] due to distribution shift — must be finite
        for col in FEATURE_COLS:
            assert np.isfinite(df_val_scaled[col].values).all(), \
                f"Non-finite value in val col {col}"

    def test_scaler_save_load(self, tmp_path, pipeline_output):
        """Saved and reloaded scaler must produce identical transforms."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)

        path = tmp_path / "test_scaler.pkl"
        save_scaler(scaler, path)
        scaler2 = load_scaler(path)

        sample = df_train[FEATURE_COLS].values[:50]
        np.testing.assert_array_almost_equal(
            scaler.transform(sample),
            scaler2.transform(sample),
        )


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestCMAPSSDataset:

    def test_dataset_length_positive(self, pipeline_output):
        """Dataset must have at least one window."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)
        df_scaled = apply_scaler(df_train, scaler)
        ds = CMAPSSDataset(df_scaled)
        assert len(ds) > 0

    def test_item_shapes(self, pipeline_output):
        """Each item must be (window_tensor, rul_scalar, engine_id)."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)
        df_scaled = apply_scaler(df_train, scaler)
        ds = CMAPSSDataset(df_scaled)

        x, y, eid = ds[0]
        assert x.shape == (WINDOW_SIZE, N_FEATURES), \
            f"Expected ({WINDOW_SIZE}, {N_FEATURES}), got {x.shape}"
        assert y.ndim == 0,   "RUL must be a scalar tensor"
        assert isinstance(eid, (int, np.integer))

    def test_no_cross_engine_windows(self, pipeline_output):
        """Windows within a Dataset must not span multiple engines."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)
        df_scaled = apply_scaler(df_train, scaler)
        ds = CMAPSSDataset(df_scaled)

        for eid, start in ds._index:
            features, ruls = ds._data[eid]
            end = start + ds.window_size
            assert end <= len(features), \
                f"Window ({eid}, {start}:{end}) overruns engine data"

    def test_rul_is_nonnegative(self, pipeline_output):
        """All RUL targets must be ≥ 0."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)
        df_scaled = apply_scaler(df_train, scaler)
        ds = CMAPSSDataset(df_scaled)

        indices = np.random.default_rng(0).choice(len(ds), min(500, len(ds)), replace=False)
        for i in indices:
            _, y, _ = ds[int(i)]
            assert float(y) >= 0.0, f"Negative RUL at index {i}: {float(y)}"

    def test_engine_isolation(self, pipeline_output):
        """
        No engine in the val or target Dataset should appear in the train Dataset.
        This is the core anti-leakage check.
        """
        df_train  = pipeline_output["df_train"]
        df_val    = pipeline_output["df_val"]
        df_target = pipeline_output["df_target"]
        scaler    = fit_scaler(df_train)

        ds_train  = CMAPSSDataset(apply_scaler(df_train, scaler))
        ds_val    = CMAPSSDataset(apply_scaler(df_val, scaler))
        ds_target = CMAPSSDataset(apply_scaler(df_target, scaler))

        train_eids  = set(ds_train.engine_ids)
        val_eids    = set(ds_val.engine_ids)
        target_eids = set(ds_target.engine_ids)

        assert train_eids & val_eids    == set(), "Engine leak: train ∩ val"
        assert train_eids & target_eids == set(), "Engine leak: train ∩ target"
        assert val_eids   & target_eids == set(), "Engine leak: val ∩ target"


class TestFewShotDataset:

    def test_fraction_1pct(self, pipeline_output):
        """1% labelling fraction produces the right number of items."""
        df_target = pipeline_output["df_target"]
        scaler    = fit_scaler(pipeline_output["df_train"])
        df_scaled = apply_scaler(df_target, scaler)
        base      = CMAPSSDataset(df_scaled)
        fs        = FewShotDataset(base, label_fraction=0.01)
        expected  = max(1, int(len(base) * 0.01))
        assert len(fs) == expected

    def test_fraction_5pct(self, pipeline_output):
        df_target = pipeline_output["df_target"]
        scaler    = fit_scaler(pipeline_output["df_train"])
        base      = CMAPSSDataset(apply_scaler(df_target, scaler))
        fs        = FewShotDataset(base, label_fraction=0.05)
        expected  = max(1, int(len(base) * 0.05))
        assert len(fs) == expected

    def test_reproducible_sampling(self, pipeline_output):
        """Two FewShotDatasets with the same seed must return identical items."""
        df_target = pipeline_output["df_target"]
        scaler    = fit_scaler(pipeline_output["df_train"])
        base      = CMAPSSDataset(apply_scaler(df_target, scaler))

        fs1 = FewShotDataset(base, label_fraction=0.05, seed=42)
        fs2 = FewShotDataset(base, label_fraction=0.05, seed=42)
        np.testing.assert_array_equal(fs1._indices, fs2._indices)

    def test_different_seeds_differ(self, pipeline_output):
        """Different seeds should produce different subsets."""
        df_target = pipeline_output["df_target"]
        scaler    = fit_scaler(pipeline_output["df_train"])
        base      = CMAPSSDataset(apply_scaler(df_target, scaler))

        fs1 = FewShotDataset(base, label_fraction=0.10, seed=1)
        fs2 = FewShotDataset(base, label_fraction=0.10, seed=2)
        assert not np.array_equal(fs1._indices, fs2._indices), \
            "Different seeds produced identical subsets"


class TestSSLDataset:

    def test_ssl_item_shapes(self, pipeline_output):
        """SSLDataset items must have correct shapes."""
        df_train  = pipeline_output["df_train"]
        scaler    = fit_scaler(df_train)
        ds = SSLDataset(apply_scaler(df_train, scaler))

        v1, v2, mask, y, eid = ds[0]
        assert v1.shape == (WINDOW_SIZE, N_FEATURES)
        assert v2.shape == (WINDOW_SIZE, N_FEATURES)
        assert mask.shape == (WINDOW_SIZE,)
        assert mask.dtype == torch.bool

    def test_two_views_differ(self, pipeline_output):
        """The two augmented views of the same window must differ."""
        df_train  = pipeline_output["df_train"]
        scaler    = fit_scaler(df_train)
        ds = SSLDataset(apply_scaler(df_train, scaler), jitter_sigma=0.05)

        v1, v2, *_ = ds[0]
        assert not torch.allclose(v1, v2), "Views are identical — augmentation not applied"

    def test_mask_fraction(self, pipeline_output):
        """Mask should cover approximately mask_ratio fraction of timesteps."""
        df_train = pipeline_output["df_train"]
        scaler   = fit_scaler(df_train)
        mask_ratio = 0.15
        ds = SSLDataset(apply_scaler(df_train, scaler), mask_ratio=mask_ratio)

        # Average over 50 items
        fracs = []
        for i in range(50):
            _, _, mask, _, _ = ds[i]
            fracs.append(mask.float().mean().item())
        avg = np.mean(fracs)
        # Allow ±5% tolerance
        assert abs(avg - mask_ratio) < 0.05, \
            f"Avg mask fraction {avg:.3f} far from target {mask_ratio}"
