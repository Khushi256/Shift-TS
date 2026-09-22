"""
run_uncertainty.py
==================
Core 5 — MC Dropout uncertainty evaluation.

Usage
-----
    python experiments/run_uncertainty.py --model models/baseline_best.pt
                                          [--passes 50] [--data CMAPSSData]

Pipeline
--------
MC Dropout
    ↓
50 stochastic forward passes
    ↓
Mean prediction + std (uncertainty)
    ↓
Prediction intervals  [mean ± z·std]
    ↓
Coverage + interval width
    ↓
Error vs uncertainty  (Spearman ρ)
    ↓
Risk-coverage curve
    ↓
NLL (Gaussian, for reference)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (build_engine_splits, fit_scaler, apply_scaler,
                       CMAPSSDataset)
from src.models.baseline import GRUBaseline
from src.models.uncertainty import MCDropoutWrapper, predict_with_uncertainty
from src.evaluation.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS Uncertainty Evaluation")
    p.add_argument("--data",    default="CMAPSSData")
    p.add_argument("--model",   required=True, help="Path to trained model checkpoint")
    p.add_argument("--hidden",  type=int,   default=64)
    p.add_argument("--layers",  type=int,   default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--passes",  type=int,   default=50, help="MC Dropout passes")
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--workers", type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SHIFT-TS — Core 5: Uncertainty Estimation")
    print(f"  MC Dropout passes: {args.passes}")
    print("=" * 60)

    # Data
    out    = build_engine_splits(Path(args.data))
    scaler = fit_scaler(out["df_train"])

    # Evaluate on BOTH val and target
    for split_name, df in [("val", out["df_val"]), ("target", out["df_target"])]:
        print(f"\n{'─'*50}")
        print(f"  Split: {split_name}")

        df_s    = apply_scaler(df, scaler)
        ds      = CMAPSSDataset(df_s)
        loader  = DataLoader(ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers)

        # Load model
        model = GRUBaseline(
            input_dim  = ds.n_features,
            hidden_dim = args.hidden,
            num_layers = args.layers,
            dropout    = args.dropout,
        )
        ckpt = torch.load(args.model, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        # MC Dropout
        wrapper = MCDropoutWrapper(model, n_passes=args.passes)
        result  = predict_with_uncertainty(wrapper, loader)

        means   = result["means"]
        stds    = result["stds"]
        targets = result["targets"]

        # Metrics
        metrics = compute_all_metrics(means, stds, targets, pi_level=0.95)

        print(f"  MAE              : {metrics['mae']:.3f}")
        print(f"  RMSE             : {metrics['rmse']:.3f}")
        print(f"  95% PI Coverage  : {metrics['coverage_95pct']:.3f}  "
              f"(ideal ≈ 0.950)")
        print(f"  Mean PI Width    : {metrics['mean_interval_width']:.3f}")
        print(f"  Spearman ρ       : {metrics['spearman_rho']:.3f}  "
              f"(p={metrics['spearman_p']:.4f})")
        print(f"  NLL (Gaussian)   : {metrics['nll_gaussian']:.3f}  "
              f"[informational only]")

        # Save results
        out_path = Path("models") / f"uncertainty_{split_name}.npz"
        rc = metrics["risk_coverage"]
        np.savez(
            out_path,
            means=means, stds=stds, targets=targets,
            coverage=rc["coverage"],
            mae_at_coverage=rc["mae_at_coverage"],
            thresholds=rc["thresholds"],
        )
        print(f"  Results saved → {out_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
