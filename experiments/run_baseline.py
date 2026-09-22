"""
run_baseline.py
===============
Core 2 — GRU Baseline experiment.

Usage
-----
    python experiments/run_baseline.py [--epochs 60] [--hidden 64] [--layers 2]
                                       [--dropout 0.2] [--batch 256] [--lr 1e-3]
                                       [--data CMAPSSData] [--run-name baseline]

What this does
--------------
1. Loads and preprocesses FD002 (engine-level split)
2. Builds train / val PyTorch DataLoaders
3. Trains GRUBaseline with MSE loss
4. Evaluates best model on val set and target (unseen) set
5. Prints MAE / RMSE for both
6. Saves predictions to models/{run_name}_predictions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    build_engine_splits,
    fit_scaler,
    apply_scaler,
    save_scaler,
    CMAPSSDataset,
)
from src.models.baseline import GRUBaseline
from src.training.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS GRU Baseline")
    p.add_argument("--data",     default="CMAPSSData", help="CMAPSSData folder path")
    p.add_argument("--run-name", default="baseline",   help="Experiment name")
    p.add_argument("--epochs",   type=int,   default=60)
    p.add_argument("--hidden",   type=int,   default=64,   help="GRU hidden size")
    p.add_argument("--layers",   type=int,   default=2,    help="GRU layers")
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--workers",  type=int,   default=0,    help="DataLoader workers")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = Path(args.data)

    print("=" * 60)
    print("SHIFT-TS — Core 2: GRU Baseline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Loading and splitting data …")
    splits_out = build_engine_splits(data_dir)

    df_train  = splits_out["df_train"]
    df_val    = splits_out["df_val"]
    df_target = splits_out["df_target"]
    summary   = splits_out["summary"]

    print(f"  Train engines : {summary['n_train_engines']}  "
          f"({summary['n_train_rows']:,} rows)")
    print(f"  Val engines   : {summary['n_val_engines']}  "
          f"({summary['n_val_rows']:,} rows)")
    print(f"  Target engines: {summary['n_target_engines']}  "
          f"({summary['n_target_rows']:,} rows)")

    # Fit scaler on TRAIN ONLY
    scaler = fit_scaler(df_train)
    save_scaler(scaler, "data/scaler.pkl")

    df_train_s  = apply_scaler(df_train,  scaler)
    df_val_s    = apply_scaler(df_val,    scaler)
    df_target_s = apply_scaler(df_target, scaler)

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    print("\n[2/4] Building DataLoaders …")
    ds_train  = CMAPSSDataset(df_train_s)
    ds_val    = CMAPSSDataset(df_val_s)
    ds_target = CMAPSSDataset(df_target_s)

    print(f"  Train windows : {len(ds_train):,}")
    print(f"  Val windows   : {len(ds_val):,}")
    print(f"  Target windows: {len(ds_target):,}")

    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=False)
    train_loader  = DataLoader(ds_train,  shuffle=True,  **loader_kw)
    val_loader    = DataLoader(ds_val,    shuffle=False, **loader_kw)
    target_loader = DataLoader(ds_target, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # 3. Model + training
    # ------------------------------------------------------------------
    print("\n[3/4] Training GRU Baseline …")
    model = GRUBaseline(
        input_dim  = ds_train.n_features,
        hidden_dim = args.hidden,
        num_layers = args.layers,
        dropout    = args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    config = TrainerConfig(
        run_name  = args.run_name,
        epochs    = args.epochs,
        lr        = args.lr,
        batch_size = args.batch,
    )
    trainer = Trainer(model, config)
    trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluating best model …")
    trainer.load(str(Path("models") / f"{args.run_name}_best.pt"))

    def metrics(preds, targets):
        mae  = float(np.abs(preds - targets).mean())
        rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
        return mae, rmse

    val_preds,    val_targets    = trainer.predict(val_loader)
    target_preds, target_targets = trainer.predict(target_loader)

    val_mae,    val_rmse    = metrics(val_preds,    val_targets)
    target_mae, target_rmse = metrics(target_preds, target_targets)

    print(f"\n  Validation  — MAE: {val_mae:.2f}  RMSE: {val_rmse:.2f}")
    print(f"  Target      — MAE: {target_mae:.2f}  RMSE: {target_rmse:.2f}")
    print("\n  (Target = unseen engine cohort — measures zero-shot generalisation)")

    # Save predictions
    out_path = Path("models") / f"{args.run_name}_predictions.npz"
    np.savez(
        out_path,
        val_preds=val_preds,       val_targets=val_targets,
        target_preds=target_preds, target_targets=target_targets,
    )
    print(f"\n  Predictions saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
