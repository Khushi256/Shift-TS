"""
adapt.py
========
Core 4 — Few-shot adaptation for the unseen target engine cohort.

Corrected pipeline
------------------
The correct sequence is:

    [A] SSL pretraining on train engines  (no labels)
           ↓
    [B] Supervised fine-tune on train engines  (with labels)
           ↓  encoder now produces RUL-relevant representations
    [C] Freeze encoder, attach fresh head
           ↓
    [D] Fine-tune head on 1%/5%/20% labeled TARGET windows
           ↓
    [E] Evaluate on ALL target windows

Skipping step [B] (going straight from SSL encoder to few-shot) forces the
head to learn the RUL regression task from scratch on very few samples using
representations that were never optimised for regression — almost guaranteed
to fail.

This module supports both:
    - Loading from a full GRUBaseline checkpoint (recommended)
    - Loading from an SSL-only encoder checkpoint (for ablation)

Key bug fixed
-------------
`encoder.eval()` is enforced at the *start of every batch*, not only once in
`__init__`.  This prevents PyTorch from inadvertently re-enabling dropout in
the encoder when `head.train()` is called.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.encoder import GRUEncoder
from src.models.baseline import RegressionHead, GRUBaseline


@dataclass
class AdaptConfig:
    run_name:            str   = "fewshot"
    checkpoint_dir:      str   = "models"
    log_dir:             str   = "runs"
    epochs:              int   = 40
    lr:                  float = 5e-4
    weight_decay:        float = 1e-3
    batch_size:          int   = 64
    grad_clip:           float = 1.0
    early_stop_patience: int   = 10
    warmup_epochs:       int   = 3     # LR warmup to avoid bad early steps
    unfreeze_mode:       str   = "frozen_encoder"  # "frozen_encoder" | "partially_unfrozen" | "fully_finetuned"
    device:              str   = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def load_encoder_from_baseline(
    ckpt_path: str,
    n_features: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float  = 0.2,
) -> tuple[GRUEncoder, RegressionHead]:
    """
    Load encoder AND head from a full supervised GRUBaseline checkpoint.

    Returns
    -------
    (encoder, head) — both with pretrained weights
    """
    model = GRUBaseline(
        input_dim  = n_features,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        dropout    = dropout,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return model.encoder, model.head


def load_encoder_from_ssl(
    ckpt_path: str,
    n_features: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float  = 0.2,
) -> GRUEncoder:
    """
    Load encoder weights from an SSL-only encoder checkpoint.

    Returns a fresh RegressionHead (caller decides whether to warm-start it).
    """
    encoder = GRUEncoder(
        input_dim  = n_features,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        dropout    = dropout,
    )
    encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return encoder


# ---------------------------------------------------------------------------
# Few-shot adapter
# ---------------------------------------------------------------------------

class FewShotAdapter:
    """
    Adapts a pretrained model to the target domain with very few labels.

    Supports 3 adaptation regimes:
    - "frozen_encoder"     : Only fine-tunes regression head; encoder completely frozen.
    - "partially_unfrozen" : Fine-tunes top GRU layer (_l1) + regression head.
    - "fully_finetuned"    : Fine-tunes entire encoder + regression head with AdamW & clipping.

    Parameters
    ----------
    encoder         : pretrained GRUEncoder
    pretrained_head : if provided, initialise adaptation head from these weights (warm start)
    unfreeze_mode   : "frozen_encoder" | "partially_unfrozen" | "fully_finetuned"
    head_hidden     : hidden dim for regression head if initialized fresh
    dropout         : dropout rate
    config          : AdaptConfig
    """

    def __init__(
        self,
        encoder:         GRUEncoder,
        pretrained_head: RegressionHead | None = None,
        unfreeze_mode:   str | None = None,
        head_hidden:     int   = 32,
        dropout:         float = 0.2,
        config:          AdaptConfig | None = None,
    ) -> None:
        import copy
        self.config = config or AdaptConfig()
        if unfreeze_mode is not None:
            self.config.unfreeze_mode = unfreeze_mode
        self.unfreeze_mode = self.config.unfreeze_mode
        self.device = torch.device(self.config.device)

        # ── Encoder Setup ──────────────────────────────────────────────────
        self.encoder = copy.deepcopy(encoder).to(self.device)

        if self.unfreeze_mode == "frozen_encoder":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        elif self.unfreeze_mode == "partially_unfrozen":
            for name, param in self.encoder.named_parameters():
                # Unfreeze top GRU layer weights (e.g. layer 1 in 2-layer GRU)
                if "_l1" in name or "linear" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.unfreeze_mode == "fully_finetuned":
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown unfreeze_mode '{self.unfreeze_mode}'.")

        # ── Head: warm-start from pretrained weights OR fresh init ─────────
        if pretrained_head is not None:
            self.head = copy.deepcopy(pretrained_head).to(self.device)
        else:
            self.head = RegressionHead(
                input_dim  = encoder.output_dim,
                hidden_dim = head_hidden,
                dropout    = dropout,
            ).to(self.device)

        for p in self.head.parameters():
            p.requires_grad = True

        self.trainable_params = [p for p in self.encoder.parameters() if p.requires_grad] + list(self.head.parameters())

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=4, factor=0.5, min_lr=1e-6
        )

        log_path = Path(self.config.log_dir) / f"{self.config.run_name}_{self.unfreeze_mode}"
        self.writer = SummaryWriter(str(log_path))

        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss   = float("inf")
        self._no_improve = 0

    # ------------------------------------------------------------------

    def _warmup_lr(self, epoch: int) -> float:
        """Linear warmup over warmup_epochs."""
        if epoch <= self.config.warmup_epochs:
            return self.config.lr * epoch / self.config.warmup_epochs
        return self.config.lr

    def _forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        if self.unfreeze_mode == "frozen_encoder":
            self.encoder.eval()
            with torch.no_grad():
                _, z = self.encoder(x)
        else:
            if train:
                self.encoder.train()
            else:
                self.encoder.eval()
            _, z = self.encoder(x)
        return self.head(z)

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
        epoch: int  = 1,
    ) -> tuple[float, float, float]:
        if train:
            self.head.train()
            lr = self._warmup_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.head.eval()
            self.encoder.eval()

        total_loss, preds_all, targets_all = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                x, y, _ = batch
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self._forward(x, train=train)
                loss = self.criterion(pred, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.trainable_params, self.config.grad_clip
                    )
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
                preds_all.append(pred.detach().cpu())
                targets_all.append(y.detach().cpu())

        preds_t   = torch.cat(preds_all)
        targets_t = torch.cat(targets_all)
        n         = len(targets_t)
        mae       = float((preds_t - targets_t).abs().mean())
        rmse      = float(((preds_t - targets_t) ** 2).mean().sqrt())
        return total_loss / n, mae, rmse

    def fit(
        self,
        few_shot_loader: DataLoader,
        eval_loader:     DataLoader,
        val_loader:      DataLoader | None = None,
    ) -> dict:
        """
        Fine-tune on few-shot labeled data with early stopping and regularization.

        Parameters
        ----------
        few_shot_loader : small labeled subset of target data
        eval_loader     : held-out evaluation set
        val_loader      : optional validation set for early stopping (defaults to few_shot_loader
                          loss to prevent evaluating set leakage)
        """
        history: dict[str, list[float]] = {
            k: [] for k in
            ["train_rmse", "train_mae", "eval_rmse", "eval_mae"]
        }

        best_ckpt = self.ckpt_dir / f"{self.config.run_name}_{self.unfreeze_mode}_best.pt"

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            tr_loss, tr_mae, tr_rmse = self._run_epoch(few_shot_loader, train=True, epoch=epoch)
            ev_loss, ev_mae, ev_rmse = self._run_epoch(eval_loader,     train=False)

            # Monitor either validation loader or train loss
            monitor_loss = tr_loss if val_loader is None else self._run_epoch(val_loader, train=False)[0]

            if epoch > self.config.warmup_epochs:
                self.scheduler.step(monitor_loss)

            self.writer.add_scalar("adapt/train_rmse", tr_rmse, epoch)
            self.writer.add_scalar("adapt/eval_rmse",  ev_rmse, epoch)
            self.writer.add_scalar("adapt/lr", self.optimizer.param_groups[0]["lr"], epoch)

            history["train_rmse"].append(tr_rmse)
            history["train_mae"].append(tr_mae)
            history["eval_rmse"].append(ev_rmse)
            history["eval_mae"].append(ev_mae)

            elapsed = time.time() - t0

            if monitor_loss < self.best_loss:
                self.best_loss   = monitor_loss
                self._no_improve = 0
                torch.save({
                    "encoder_state": self.encoder.state_dict(),
                    "head_state":    self.head.state_dict(),
                    "unfreeze_mode": self.unfreeze_mode,
                    "epoch":         epoch,
                }, best_ckpt)
            else:
                self._no_improve += 1
                if self._no_improve >= self.config.early_stop_patience:
                    break

        self.writer.close()
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.encoder.eval()
        self.head.eval()
        preds, targets = [], []
        for batch in loader:
            x, y, _ = batch
            x = x.to(self.device)
            pred = self._forward(x, train=False).cpu()
            preds.append(pred.numpy())
            targets.append(y.numpy())
        return np.concatenate(preds), np.concatenate(targets)

    def load_best(self) -> None:
        path = self.ckpt_dir / f"{self.config.run_name}_{self.unfreeze_mode}_best.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.head.load_state_dict(ckpt["head_state"])
