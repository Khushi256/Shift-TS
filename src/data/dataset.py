"""
dataset.py
==========
PyTorch Dataset classes for the SHIFT-TS pipeline.

Key rule: Windows are extracted AFTER the engine-level split.
No engine appears in more than one Dataset instance.

Classes
-------
CMAPSSDataset  — sliding-window dataset for train / val / target engines
FewShotDataset — randomly samples ``label_fraction`` windows from a
                 CMAPSSDataset for few-shot adaptation experiments
SSLDataset     — returns pairs of augmented views of the same window
                 (for contrastive pre-training)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import FEATURE_COLS, WINDOW_SIZE, STRIDE


# ---------------------------------------------------------------------------
# CMAPSSDataset — standard supervised sliding-window dataset
# ---------------------------------------------------------------------------

class CMAPSSDataset(Dataset):
    """
    Extracts sliding windows from a set of engine time-series.

    Each item is a (window, rul, engine_id) triple:
        window     : float32 tensor of shape (window_size, n_features)
        rul        : float32 scalar — RUL at the END of the window
        engine_id  : int (for grouping predictions at evaluation time)

    Parameters
    ----------
    df           : pre-split, pre-scaled DataFrame; must contain columns in
                   ``feature_cols`` plus ``rul`` and ``engine_id``
    window_size  : number of consecutive cycles per window
    feature_cols : input feature columns (default: constants.FEATURE_COLS)
    stride       : step between window start positions (default: 1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        feature_cols: list[str] | None = None,
        stride: int = STRIDE,
    ) -> None:
        self.window_size  = window_size
        self.feature_cols = feature_cols or FEATURE_COLS
        self.stride       = stride

        # Build index: list of (engine_id, window_start_index)
        self._index: list[tuple[int, int]] = []
        # Engine data cache: engine_id → (feature_array, rul_array)
        self._data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for engine_id, group in df.groupby("engine_id"):
            group = group.sort_values("cycle").reset_index(drop=True)

            features = group[self.feature_cols].values.astype(np.float32)
            ruls     = group["rul"].values.astype(np.float32)
            self._data[engine_id] = (features, ruls)

            n_cycles = len(group)
            for start in range(0, n_cycles - window_size + 1, stride):
                self._index.append((engine_id, start))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        engine_id, start = self._index[idx]
        features, ruls   = self._data[engine_id]

        x = torch.from_numpy(features[start : start + self.window_size])     # (W, F)
        y = torch.tensor(ruls[start + self.window_size - 1], dtype=torch.float32)
        return x, y, engine_id

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def engine_ids(self) -> list[int]:
        return sorted(self._data.keys())

    def n_windows_per_engine(self) -> dict[int, int]:
        """Return {engine_id: window_count} for debugging."""
        counts: dict[int, int] = {}
        for eid, _ in self._index:
            counts[eid] = counts.get(eid, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# FewShotDataset — sparse label subset for adaptation experiments
# ---------------------------------------------------------------------------

class FewShotDataset(Dataset):
    """
    Wraps a CMAPSSDataset and exposes only a random fraction of its windows.

    Use this for the 1% / 5% / 20% few-shot adaptation experiments.
    The sampled indices are fixed by ``seed`` for reproducibility.

    Parameters
    ----------
    base_dataset   : a CMAPSSDataset instance (target engines)
    label_fraction : fraction of windows to expose (e.g. 0.01, 0.05, 0.20)
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        base_dataset: CMAPSSDataset,
        label_fraction: float = 0.05,
        seed: int = 42,
    ) -> None:
        if not (0.0 < label_fraction <= 1.0):
            raise ValueError(f"label_fraction must be in (0, 1], got {label_fraction}")

        rng = np.random.default_rng(seed)
        n_total  = len(base_dataset)
        n_sample = max(1, int(n_total * label_fraction))

        self._indices = rng.choice(n_total, size=n_sample, replace=False)
        self._base    = base_dataset

        self.label_fraction = label_fraction
        self.n_labeled      = n_sample
        self.n_total        = n_total

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._base[int(self._indices[idx])]


# ---------------------------------------------------------------------------
# SSLDataset — augmented pairs for contrastive pre-training
# ---------------------------------------------------------------------------

class SSLDataset(Dataset):
    """
    Returns two independently augmented views of the same sliding window.

    Used for contrastive pre-training (NT-Xent) in Core 3.

    Augmentations applied independently to each view:
        - Gaussian jitter  (σ = jitter_sigma)
        - Random temporal crop + resize (if crop_ratio < 1.0)

    Parameters
    ----------
    df           : pre-split, pre-scaled DataFrame (train engines only)
    window_size  : window length in cycles
    feature_cols : input feature columns
    stride       : stride between window positions
    jitter_sigma : std-dev of Gaussian noise augmentation
    crop_ratio   : fraction of the window to randomly crop (e.g. 0.9)
    mask_ratio   : fraction of timesteps to zero-mask (for masked recon head)
    seed         : base seed (per-item seeds are deterministic from this)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        feature_cols: list[str] | None = None,
        stride: int = STRIDE,
        jitter_sigma: float = 0.02,
        crop_ratio: float = 0.9,
        mask_ratio: float = 0.15,
        seed: int = 0,
    ) -> None:
        self._base        = CMAPSSDataset(df, window_size, feature_cols, stride)
        self.jitter_sigma = jitter_sigma
        self.crop_ratio   = crop_ratio
        self.mask_ratio   = mask_ratio
        self._rng         = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._base)

    def _augment(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Apply jitter augmentation to a (W, F) window tensor."""
        x = x.clone()
        if self.jitter_sigma > 0:
            noise = torch.from_numpy(
                rng.normal(0, self.jitter_sigma, size=x.shape).astype(np.float32)
            )
            x = (x + noise).clamp(0.0, 1.0)   # keep in [0, 1] after MinMax scaling
        return x

    def _make_mask(self, window_size: int, rng: np.random.Generator) -> torch.Tensor:
        """
        Create a boolean mask tensor of shape (window_size,).
        True = timestep is masked (to be reconstructed).
        """
        n_masked = max(1, int(window_size * self.mask_ratio))
        mask_idx = rng.choice(window_size, size=n_masked, replace=False)
        mask = torch.zeros(window_size, dtype=torch.bool)
        mask[mask_idx] = True
        return mask

    def __getitem__(self, idx: int):
        x, y, engine_id = self._base[idx]

        # Independent RNG streams for the two views
        seed_v1 = int(self._rng.integers(0, 2**31))
        seed_v2 = int(self._rng.integers(0, 2**31))
        rng1 = np.random.default_rng(seed_v1)
        rng2 = np.random.default_rng(seed_v2)

        view1 = self._augment(x, rng1)
        view2 = self._augment(x, rng2)

        # Mask for masked-reconstruction head
        mask = self._make_mask(x.shape[0], rng1)

        return view1, view2, mask, y, engine_id
