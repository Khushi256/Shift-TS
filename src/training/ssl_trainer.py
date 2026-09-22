"""
ssl_trainer.py
==============
Core 3 — Self-supervised pretraining loop.

Trains the SSLModel on TRAIN ENGINE data only (no labels used).
After pretraining, the encoder weights are saved for:
  - Initialising GRUBaseline in Core 4 (few-shot adaptation)
  - Comparing SSL-pretrained vs. from-scratch baselines
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.ssl_heads import SSLModel


@dataclass
class SSLTrainerConfig:
    run_name:      str   = "ssl_pretrain"
    checkpoint_dir: str  = "models"
    log_dir:        str  = "runs"
    epochs:        int   = 50
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    batch_size:    int   = 256
    grad_clip:     float = 1.0
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"


class SSLTrainer:
    """
    Self-supervised pretraining loop for SSLModel.

    Parameters
    ----------
    model  : SSLModel instance
    config : SSLTrainerConfig
    """

    def __init__(self, model: SSLModel, config: SSLTrainerConfig | None = None) -> None:
        self.config = config or SSLTrainerConfig()
        self.device = torch.device(self.config.device)
        self.model  = model.to(self.device)

        self.optimizer = Adam(
            model.parameters(),
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay,
        )

        # TensorBoard
        log_path = Path(self.config.log_dir) / self.config.run_name
        self.writer = SummaryWriter(str(log_path))

        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def fit(self, train_loader: DataLoader) -> dict[str, list[float]]:
        """
        Pretrain for ``config.epochs`` epochs.

        Returns
        -------
        history: {"total_loss", "recon_loss", "contrast_loss"}
        """
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = self.config.epochs,
            eta_min = 1e-5,
        )

        history: dict[str, list[float]] = {
            "total_loss": [], "recon_loss": [], "contrast_loss": []
        }

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            self.model.train()

            ep_total, ep_recon, ep_contrast = 0.0, 0.0, 0.0
            n_batches = 0

            for batch in train_loader:
                # SSLDataset returns: view1, view2, mask, y, engine_id
                v1, v2, mask, _, _ = batch
                v1   = v1.to(self.device)
                v2   = v2.to(self.device)
                mask = mask.to(self.device)

                losses = self.model(v1, v2, mask)

                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.optimizer.step()

                ep_total    += losses["total_loss"].item()
                ep_recon    += losses["recon_loss"].item()
                ep_contrast += losses["contrastive_loss"].item()
                n_batches   += 1

            scheduler.step()

            avg_total    = ep_total    / n_batches
            avg_recon    = ep_recon    / n_batches
            avg_contrast = ep_contrast / n_batches

            self.writer.add_scalar("ssl/total",      avg_total,    epoch)
            self.writer.add_scalar("ssl/recon",      avg_recon,    epoch)
            self.writer.add_scalar("ssl/contrastive", avg_contrast, epoch)

            history["total_loss"].append(avg_total)
            history["recon_loss"].append(avg_recon)
            history["contrast_loss"].append(avg_contrast)

            elapsed = time.time() - t0
            print(
                f"SSL Epoch {epoch:3d}/{self.config.epochs}  |  "
                f"Total {avg_total:.4f}  Recon {avg_recon:.4f}  "
                f"Contrast {avg_contrast:.4f}  |  {elapsed:.1f}s"
            )

            # Save best
            if avg_total < self.best_loss:
                self.best_loss = avg_total
                self.save_encoder(str(
                    self.ckpt_dir / f"{self.config.run_name}_encoder_best.pt"
                ))
                print(f"  → New best SSL loss: {avg_total:.4f}  (encoder saved)")

        self.writer.close()
        return history

    def save_encoder(self, path: str) -> None:
        """Save only the encoder weights (for downstream initialisation)."""
        torch.save(self.model.encoder.state_dict(), path)
        print(f"  Encoder saved to {path}")

    def save_full(self, path: str) -> None:
        """Save full SSL model state."""
        torch.save({
            "model_state": self.model.state_dict(),
            "best_loss":   self.best_loss,
        }, path)
