"""
encoder.py
==========
Shared GRU encoder backbone for all SHIFT-TS models.

The encoder maps a (batch, window, features) tensor to:
    - hidden_states : (batch, window, hidden_dim) — all timestep representations
    - last_hidden   : (batch, hidden_dim)         — final timestep representation

It is designed to be shared across:
    - GRUBaseline (Core 2) — supervised regression
    - SSL pretraining heads (Core 3) — masked recon + contrastive
    - Few-shot adaptation (Core 4) — encoder frozen, head fine-tuned
    - MC Dropout inference (Core 5) — dropout kept active at test time
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Multi-layer GRU encoder with dropout.

    Parameters
    ----------
    input_dim   : number of input features per timestep (default: 17)
    hidden_dim  : GRU hidden state dimension
    num_layers  : number of stacked GRU layers
    dropout     : dropout probability applied between GRU layers (and during
                  MC Dropout inference if dropout > 0 and model in train mode
                  or MCDropoutWrapper is used)
    bidirectional: if True, use bidirectional GRU; output_dim = 2 * hidden_dim
    """

    def __init__(
        self,
        input_dim:     int   = 17,
        hidden_dim:    int   = 64,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        bidirectional: bool  = False,
    ) -> None:
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        # Additional dropout applied after the GRU (for MC Dropout)
        self.output_dropout = nn.Dropout(p=dropout)

        # Output dimension depends on bidirectionality
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(
        self,
        x: torch.Tensor,   # (batch, window_size, input_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, window_size, input_dim) float32

        Returns
        -------
        hidden_states : (batch, window_size, output_dim)
        last_hidden   : (batch, output_dim)
        """
        hidden_states, _ = self.gru(x)          # (B, W, output_dim)
        hidden_states     = self.output_dropout(hidden_states)
        last_hidden       = hidden_states[:, -1, :]   # last timestep
        return hidden_states, last_hidden
