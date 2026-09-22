"""
trainer.py
==========
Generic supervised training loop for SHIFT-TS.

Features
--------
* Epoch-level train + validation loop
* MSE loss with gradient clipping
* TensorBoard scalar logging (loss, MAE, RMSE per epoch)
* Best-model checkpointing (by val RMSE)
* Configurable via a simple TrainerConfig dataclass
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # Paths
    checkpoint_dir: str = "models"
    log_dir:        str = "runs"
    run_name:       str = "baseline"

    # Training
    epochs:       int   = 60
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    batch_size:   int   = 256
    grad_clip:    float = 1.0

    # Scheduler
    lr_patience:  int   = 5
    lr_factor:    float = 0.5
    min_lr:       float = 1e-5

    # Early stopping
    early_stop_patience: int = 15

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(((pred - target) ** 2).mean().item())


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Supervised trainer for any nn.Module that takes (B, W, F) → (B,).

    Parameters
    ----------
    model      : PyTorch model
    config     : TrainerConfig
    """

    def __init__(self, model: nn.Module, config: TrainerConfig | None = None) -> None:
        self.config  = config or TrainerConfig()
        self.device  = torch.device(self.config.device)
        self.model   = model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            model.parameters(),
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode      = "min",
            patience  = self.config.lr_patience,
            factor    = self.config.lr_factor,
            min_lr    = self.config.min_lr,
        )

        # TensorBoard
        log_path = Path(self.config.log_dir) / self.config.run_name
        self.writer = SummaryWriter(str(log_path))

        # Checkpointing
        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = self.ckpt_dir / f"{self.config.run_name}_best.pt"

        self.best_val_rmse = float("inf")
        self._no_improve   = 0

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
    ) -> tuple[float, float, float]:
        """
        Run one epoch.

        Returns
        -------
        (loss, mae, rmse)
        """
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        all_preds:   list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                x, y, _ = batch         # ignore engine_id here
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)    # (B,)
                loss = self.criterion(pred, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
                all_preds.append(pred.detach().cpu())
                all_targets.append(y.detach().cpu())

        all_preds_t   = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)
        avg_loss = total_loss / len(all_targets_t)
        mae      = _mae(all_preds_t, all_targets_t)
        rmse     = _rmse(all_preds_t, all_targets_t)
        return avg_loss, mae, rmse

    # ------------------------------------------------------------------
    # Full training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> dict[str, list[float]]:
        """
        Train for up to ``config.epochs`` epochs with early stopping.

        Returns
        -------
        history dict: {"train_loss", "val_loss", "train_mae", "val_mae",
                       "train_rmse", "val_rmse"}
        """
        history: dict[str, list[float]] = {
            k: [] for k in
            ["train_loss", "val_loss", "train_mae", "val_mae",
             "train_rmse", "val_rmse"]
        }

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            tr_loss, tr_mae, tr_rmse = self._run_epoch(train_loader, train=True)
            va_loss, va_mae, va_rmse = self._run_epoch(val_loader,   train=False)

            self.scheduler.step(va_rmse)

            # Logging
            for tag, val in [
                ("loss/train", tr_loss), ("loss/val", va_loss),
                ("mae/train",  tr_mae),  ("mae/val",  va_mae),
                ("rmse/train", tr_rmse), ("rmse/val", va_rmse),
                ("lr", self.optimizer.param_groups[0]["lr"]),
            ]:
                self.writer.add_scalar(tag, val, epoch)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_mae"].append(tr_mae)
            history["val_mae"].append(va_mae)
            history["train_rmse"].append(tr_rmse)
            history["val_rmse"].append(va_rmse)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train RMSE {tr_rmse:.2f}  MAE {tr_mae:.2f} | "
                f"Val RMSE {va_rmse:.2f}  MAE {va_mae:.2f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpointing
            if va_rmse < self.best_val_rmse:
                self.best_val_rmse = va_rmse
                self._no_improve   = 0
                self.save(str(self.best_ckpt))
                print(f"  → New best val RMSE: {va_rmse:.2f}  (saved)")
            else:
                self._no_improve += 1
                if self._no_improve >= self.config.early_stop_patience:
                    print(f"  Early stopping after {epoch} epochs.")
                    break

        self.writer.close()
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a DataLoader.

        Returns
        -------
        (predictions, targets) as numpy arrays
        """
        self.model.eval()
        preds, targets = [], []
        for batch in loader:
            x, y, _ = batch
            x = x.to(self.device)
            pred = self.model(x).cpu()
            preds.append(pred.numpy())
            targets.append(y.numpy())
        return np.concatenate(preds), np.concatenate(targets)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_rmse":   self.best_val_rmse,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_rmse = ckpt["best_val_rmse"]
        print(f"Loaded checkpoint from {path}  (best val RMSE: {self.best_val_rmse:.2f})")
