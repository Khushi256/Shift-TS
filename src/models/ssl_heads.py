"""
ssl_heads.py
============
Self-supervised learning heads for Core 3.

Two objectives are implemented:

1. MaskedReconHead
   ─────────────────
   Takes the full sequence of encoder hidden states + a boolean mask,
   reconstructs the original sensor values at masked timesteps.
   Loss: MSE between reconstructed and original sensor values at masked positions.

2. ContrastiveHead (NT-Xent)
   ─────────────────────────
   Maps the final encoder representation (last_hidden) of two augmented views
   of the same window to a projection space and computes NT-Xent (InfoNCE) loss.
   Positive pairs = same window, different augmentations.
   Negative pairs = different windows in the batch.

3. SSLModel (combined)
   ─────────────────────
   Wraps GRUEncoder with both heads; loss = α * masked_recon + β * contrastive.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GRUEncoder


# ---------------------------------------------------------------------------
# Masked reconstruction head
# ---------------------------------------------------------------------------

class MaskedReconHead(nn.Module):
    """
    Decodes encoder hidden states at MASKED timesteps back to sensor space.

    Architecture: Linear → ReLU → Linear → sensor predictions at masked positions.

    Parameters
    ----------
    hidden_dim  : encoder output dimension
    output_dim  : number of sensors/features to reconstruct (= n_features)
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, W, hidden_dim)
        mask:          torch.Tensor,  # (B, W) bool — True = masked
    ) -> torch.Tensor:
        """
        Returns
        -------
        recon : (n_masked_total, output_dim) — reconstructed values at masked positions
        """
        masked_hs = hidden_states[mask]          # (n_masked, hidden_dim)
        return self.decoder(masked_hs)           # (n_masked, output_dim)

    @staticmethod
    def loss(
        recon:    torch.Tensor,   # (n_masked, F)
        original: torch.Tensor,   # (B, W, F)
        mask:     torch.Tensor,   # (B, W) bool
    ) -> torch.Tensor:
        """MSE between reconstructed and original at masked positions."""
        target = original[mask]                  # (n_masked, F)
        return F.mse_loss(recon, target)


# ---------------------------------------------------------------------------
# Contrastive head (NT-Xent)
# ---------------------------------------------------------------------------

class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive (NT-Xent) learning.

    Maps last_hidden → projection space; NT-Xent computed in that space.

    Parameters
    ----------
    hidden_dim  : encoder output_dim
    proj_dim    : projection space dimension (e.g. 64)
    temperature : NT-Xent temperature τ
    """

    def __init__(
        self,
        hidden_dim:  int   = 64,
        proj_dim:    int   = 64,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, hidden_dim)

        Returns
        -------
        p : (B, proj_dim) — L2-normalised projection
        """
        p = self.projector(z)
        return F.normalize(p, dim=-1)

    def nt_xent_loss(
        self,
        z1: torch.Tensor,   # (B, hidden_dim)
        z2: torch.Tensor,   # (B, hidden_dim)
    ) -> torch.Tensor:
        """
        NT-Xent loss for a batch of positive pairs (z1[i], z2[i]).

        Parameters
        ----------
        z1, z2 : encoder representations of the two augmented views

        Returns
        -------
        scalar loss
        """
        B = z1.shape[0]
        p1 = self.forward(z1)   # (B, proj_dim)
        p2 = self.forward(z2)   # (B, proj_dim)

        # Concatenate: [p1; p2] → (2B, proj_dim)
        z  = torch.cat([p1, p2], dim=0)

        # Similarity matrix (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices:  i→(i+B), (i+B)→i
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# Combined SSL model
# ---------------------------------------------------------------------------

class SSLModel(nn.Module):
    """
    GRU encoder + masked reconstruction head + contrastive head.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dim  : GRU hidden size
    num_layers  : GRU layers
    proj_dim    : contrastive projection dimension
    temperature : NT-Xent temperature
    dropout     : dropout in encoder
    alpha       : weight of masked reconstruction loss
    beta        : weight of contrastive loss
    """

    def __init__(
        self,
        input_dim:   int   = 17,
        hidden_dim:  int   = 64,
        num_layers:  int   = 2,
        proj_dim:    int   = 64,
        temperature: float = 0.5,
        dropout:     float = 0.2,
        alpha:       float = 1.0,
        beta:        float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        self.encoder = GRUEncoder(
            input_dim  = input_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            dropout    = dropout,
        )
        self.recon_head = MaskedReconHead(
            hidden_dim = self.encoder.output_dim,
            output_dim = input_dim,
        )
        self.contrast_head = ContrastiveHead(
            hidden_dim  = self.encoder.output_dim,
            proj_dim    = proj_dim,
            temperature = temperature,
        )

    def forward(
        self,
        view1: torch.Tensor,   # (B, W, F)  — augmented view 1
        view2: torch.Tensor,   # (B, W, F)  — augmented view 2
        mask:  torch.Tensor,   # (B, W) bool — True = masked
    ) -> dict[str, torch.Tensor]:
        """
        Compute masked reconstruction + contrastive losses.

        Parameters
        ----------
        view1, view2 : two independently augmented versions of the same windows
        mask         : which timesteps to reconstruct in view1

        Returns
        -------
        dict with keys: "recon_loss", "contrastive_loss", "total_loss"
        """
        # Encode both views
        hs1, lh1 = self.encoder(view1)   # hs1: (B, W, H), lh1: (B, H)
        _,   lh2 = self.encoder(view2)

        # Masked reconstruction on view1
        recon = self.recon_head(hs1, mask)             # (n_masked, F)
        recon_loss = MaskedReconHead.loss(recon, view1, mask)

        # Contrastive loss between the two representations
        contrast_loss = self.contrast_head.nt_xent_loss(lh1, lh2)

        total = self.alpha * recon_loss + self.beta * contrast_loss

        return {
            "recon_loss":       recon_loss,
            "contrastive_loss": contrast_loss,
            "total_loss":       total,
        }

    def get_encoder(self) -> GRUEncoder:
        """Return the pretrained encoder for downstream use."""
        return self.encoder
