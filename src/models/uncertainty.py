"""
uncertainty.py
==============
Core 5 — MC Dropout uncertainty estimation.

Approach
--------
MC Dropout keeps dropout layers ACTIVE at inference time.
By running T stochastic forward passes, we obtain a distribution of
predictions for each input window, from which we derive:

    mean_pred  : point estimate
    std_pred   : aleatoric + epistemic uncertainty proxy

These are then used to compute:
    - Prediction intervals  [mean ± z * std]
    - Coverage (95% PI)
    - Interval width
    - Error vs uncertainty (Spearman ρ)
    - Risk-coverage curve

Note: dropout uncertainty is a proxy for uncertainty, not a calibrated
Bayesian posterior.  NLL with an explicit Gaussian head would be required
for proper probabilistic calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class MCDropoutWrapper(nn.Module):
    """
    Wraps any model to enable MC Dropout at inference.

    During inference, we:
      1. Set encoder dropout layers to train() mode (so dropout fires)
      2. Keep BatchNorm etc. in eval() mode if present
      3. Run T stochastic forward passes
      4. Aggregate mean + std across passes

    Parameters
    ----------
    model : a GRUBaseline (or any encoder+head model with dropout)
    n_passes : number of stochastic forward passes T
    """

    def __init__(self, model: nn.Module, n_passes: int = 50) -> None:
        super().__init__()
        self.model    = model
        self.n_passes = n_passes

    def _enable_dropout(self) -> None:
        """Set only Dropout modules to train mode; leave everything else eval."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def mc_predict(
        self,
        x: torch.Tensor,   # (B, W, F)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run T stochastic forward passes.

        Returns
        -------
        mean : (B,)  — mean prediction across passes
        std  : (B,)  — std across passes (uncertainty)
        """
        self._enable_dropout()
        all_preds = []
        for _ in range(self.n_passes):
            pred = self.model(x)     # (B,)
            all_preds.append(pred)

        stacked = torch.stack(all_preds, dim=0)   # (T, B)
        mean    = stacked.mean(dim=0)             # (B,)
        std     = stacked.std(dim=0)              # (B,)
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard deterministic forward (model in eval mode)."""
        self.model.eval()
        return self.model(x)


# ---------------------------------------------------------------------------
# Batch inference with uncertainty
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_uncertainty(
    wrapper: MCDropoutWrapper,
    loader:  DataLoader,
    device:  str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Run MC Dropout inference over a DataLoader.

    Returns
    -------
    dict with arrays:
        means      : (N,) mean predictions
        stds       : (N,) uncertainty (std across passes)
        targets    : (N,) true RUL
        engine_ids : (N,) engine id per window
    """
    means_all, stds_all, targets_all, eids_all = [], [], [], []
    dev = torch.device(device)
    wrapper.model.to(dev)

    for batch in loader:
        x, y, eid = batch
        x = x.to(dev)

        mean, std = wrapper.mc_predict(x)
        means_all.append(mean.cpu().numpy())
        stds_all.append(std.cpu().numpy())
        targets_all.append(y.numpy())
        eids_all.append(np.array(eid) if not isinstance(eid, np.ndarray) else eid)

    return {
        "means":      np.concatenate(means_all),
        "stds":       np.concatenate(stds_all),
        "targets":    np.concatenate(targets_all),
        "engine_ids": np.concatenate(eids_all),
    }
