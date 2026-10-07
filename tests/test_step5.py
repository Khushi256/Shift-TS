import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_step5 import get_fixed_masks, compute_stage_bins

def test_fixed_masks_determinism_and_ratio():
    masks1 = get_fixed_masks(seq_len=30, missing_rate=0.15, K=5)
    masks2 = get_fixed_masks(seq_len=30, missing_rate=0.15, K=5)
    
    assert len(masks1) == 5
    for m1, m2 in zip(masks1, masks2):
        # Determinism
        assert (m1 == m2).all()
        # Ratio
        assert m1.sum().item() == int(30 * 0.15)  # exactly 4 True values

def test_stage_bins_only_use_val_bounds():
    val_preds = np.array([10.0, 50.0, 100.0])
    val_errs = np.array([2.0, 4.0, 6.0])
    
    bins, bin_means = compute_stage_bins(val_preds, val_errs)
    
    assert len(bins) == 11
    assert bins[0] == -np.inf
    assert bins[-1] == np.inf
    
    # 10.0 to 100.0 means range is 90. 10 bins -> width 9.
    # We just check the bounds and bin_means calculation
    assert np.isclose(bin_means.mean(), 4.0)
    
    # test mapping
    test_preds = np.array([-5.0, 15.0, 150.0])
    b_idx = np.digitize(test_preds, bins) - 1
    b_idx = np.clip(b_idx, 0, 9)
    
    assert b_idx[0] == 0  # mapped to first bin because -5.0 < 10.0
    assert b_idx[2] == 9  # mapped to last bin because 150.0 > 100.0
