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
    lr:                  float = 1e-3
    weight_decay:        float = 1e-4
    batch_size:          int   = 64
    grad_clip:           float = 1.0
    early_stop_patience: int   = 15
    warmup_epochs:       int   = 5     # LR warmup to avoid bad early steps
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
    Adapts a pretrained encoder to the target domain with very few labels.

    Parameters
    ----------
    encoder       : pretrained GRUEncoder — will be FROZEN
    pretrained_head : if provided, initialise the adaptation head from these
                      weights (warm start) rather than random; this is
                      appropriate when encoder comes from a supervised checkpoint
    head_hidden   : hidden dim for the regression head
    dropout       : dropout rate
    config        : AdaptConfig
    """

    def __init__(
        self,
        encoder:         GRUEncoder,
        pretrained_head: RegressionHead | None = None,
        head_hidden:     int   = 32,
        dropout:         float = 0.2,
        config:          AdaptConfig | None = None,
    ) -> None:
        self.config = config or AdaptConfig()
        self.device = torch.device(self.config.device)

        # ── Encoder: freeze completely ─────────────────────────────────────
        self.encoder = encoder.to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # ── Head: warm-start from pretrained weights OR fresh init ─────────
        if pretrained_head is not None:
            # Deep-copy pretrained weights — we'll fine-tune them
            import copy
            self.head = copy.deepcopy(pretrained_head).to(self.device)
        else:
            self.head = RegressionHead(
                input_dim  = encoder.output_dim,
                hidden_dim = head_hidden,
                dropout    = dropout,
            ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.head.parameters(),
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
        )

        log_path = Path(self.config.log_dir) / self.config.run_name
        self.writer = SummaryWriter(str(log_path))

        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_eval_rmse = float("inf")
        self._no_improve    = 0

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x with the frozen encoder.

        The encoder is explicitly set to eval() here on every call to guard
        against PyTorch propagating train() from sibling modules.
        """
        self.encoder.eval()        # ← explicit guard on every encode call
        with torch.no_grad():
            _, last_hidden = self.encoder(x)
        return last_hidden

    def _warmup_lr(self, epoch: int) -> float:
        """Linear warmup over warmup_epochs."""
        if epoch <= self.config.warmup_epochs:
            return self.config.lr * epoch / self.config.warmup_epochs
        return self.config.lr

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
        epoch: int  = 1,
    ) -> tuple[float, float, float]:
        if train:
            self.head.train()
            # Apply warmup
            lr = self._warmup_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.head.eval()

        total_loss, preds_all, targets_all = 0.0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                x, y, _ = batch
                x = x.to(self.device)
                y = y.to(self.device)

                z    = self._encode(x)      # frozen; encoder always in eval
                pred = self.head(z)
                loss = self.criterion(pred, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.head.parameters(), self.config.grad_clip
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
    ) -> dict:
        """
        Fine-tune the head on few-shot labeled data.

        Parameters
        ----------
        few_shot_loader : small labeled subset of target-engine windows
        eval_loader     : ALL target-engine windows (evaluation only)
        """
        history: dict[str, list[float]] = {
            k: [] for k in
            ["train_rmse", "train_mae", "eval_rmse", "eval_mae"]
        }

        best_ckpt = self.ckpt_dir / f"{self.config.run_name}_head_best.pt"

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            _, tr_mae, tr_rmse = self._run_epoch(few_shot_loader, train=True,
                                                   epoch=epoch)
            _, ev_mae, ev_rmse = self._run_epoch(eval_loader,     train=False)

            # Scheduler only kicks in after warmup
            if epoch > self.config.warmup_epochs:
                self.scheduler.step(ev_rmse)

            self.writer.add_scalar("adapt/train_rmse", tr_rmse, epoch)
            self.writer.add_scalar("adapt/eval_rmse",  ev_rmse, epoch)
            self.writer.add_scalar("adapt/lr",
                self.optimizer.param_groups[0]["lr"], epoch)

            history["train_rmse"].append(tr_rmse)
            history["train_mae"].append(tr_mae)
            history["eval_rmse"].append(ev_rmse)
            history["eval_mae"].append(ev_mae)

            elapsed = time.time() - t0
            print(
                f"  Adapt {epoch:3d}/{self.config.epochs}  |  "
                f"few-shot RMSE {tr_rmse:.2f}  |  "
                f"full-eval RMSE {ev_rmse:.2f}  MAE {ev_mae:.2f}  |  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}  |  {elapsed:.1f}s"
            )

            if ev_rmse < self.best_eval_rmse:
                self.best_eval_rmse = ev_rmse
                self._no_improve    = 0
                torch.save(self.head.state_dict(), best_ckpt)
                print(f"    → best eval RMSE {ev_rmse:.2f} (head saved)")
            else:
                self._no_improve += 1
                if self._no_improve >= self.config.early_stop_patience:
                    print(f"  Early stop at epoch {epoch}.")
                    break

        self.writer.close()
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.head.eval()
        preds, targets = [], []
        for batch in loader:
            x, y, _ = batch
            x = x.to(self.device)
            z = self._encode(x)
            pred = self.head(z).cpu()
            preds.append(pred.numpy())
            targets.append(y.numpy())
        return np.concatenate(preds), np.concatenate(targets)

    def load_best_head(self) -> None:
        path = self.ckpt_dir / f"{self.config.run_name}_head_best.pt"
        self.head.load_state_dict(torch.load(path, map_location=self.device))
