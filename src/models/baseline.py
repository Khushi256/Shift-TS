"""
baseline.py
===========
GRU-based supervised RUL predictor (Core 2 baseline).

Architecture
------------
    Input (B, W, F)
        ↓
    GRUEncoder  →  last_hidden (B, H)
        ↓
    RegressionHead (Linear → ReLU → Linear)
        ↓
    RUL prediction (B,)

The ReLU before the output ensures predictions are non-negative, which is a
sensible inductive bias for RUL regression (RUL ≥ 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import GRUEncoder


class RegressionHead(nn.Module):
    """
    Two-layer MLP regression head.

    Parameters
    ----------
    input_dim  : dimension of the encoder output (encoder.output_dim)
    hidden_dim : intermediate projection size
    dropout    : dropout in the head (for MC Dropout)
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int   = 32,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),   # RUL ≥ 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        rul : (batch,)  non-negative RUL prediction
        """
        return self.net(x).squeeze(-1)


class GRUBaseline(nn.Module):
    """
    End-to-end GRU baseline for RUL regression.

    Parameters
    ----------
    input_dim      : number of input features (default: 17)
    hidden_dim     : GRU hidden size
    num_layers     : number of GRU layers
    head_hidden    : hidden size of the regression head
    dropout        : dropout rate (applied in encoder + head)
    bidirectional  : whether to use bidirectional GRU
    """

    def __init__(
        self,
        input_dim:    int   = 17,
        hidden_dim:   int   = 64,
        num_layers:   int   = 2,
        head_hidden:  int   = 32,
        dropout:      float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = GRUEncoder(
            input_dim     = input_dim,
            hidden_dim    = hidden_dim,
            num_layers    = num_layers,
            dropout       = dropout,
            bidirectional = bidirectional,
        )

        self.head = RegressionHead(
            input_dim  = self.encoder.output_dim,
            hidden_dim = head_hidden,
            dropout    = dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window_size, input_dim) float32

        Returns
        -------
        rul : (batch,) float32 — predicted RUL (non-negative)
        """
        _, last_hidden = self.encoder(x)
        return self.head(last_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return last hidden state (for downstream tasks)."""
        _, last_hidden = self.encoder(x)
        return last_hidden
