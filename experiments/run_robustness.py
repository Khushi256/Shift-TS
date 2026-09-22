"""
run_robustness.py
=================
Core 6 — Robustness evaluation under input perturbations.

Usage
-----
    python experiments/run_robustness.py --model models/baseline_best.pt
                                         [--data CMAPSSData] [--passes 50]

What this does
--------------
For each perturbation in PERTURBATIONS registry:
    1. Apply perturbation to target-split windows
    2. Run MC Dropout inference (mean prediction)
    3. Compute MAE + RMSE vs. clean baseline
    4. Print comparison table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (build_engine_splits, fit_scaler, apply_scaler,
                       CMAPSSDataset)
from src.models.baseline import GRUBaseline
from src.models.uncertainty import MCDropoutWrapper
from src.evaluation.robustness import PERTURBATIONS
from src.evaluation.metrics import mae, rmse


def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS Robustness Experiments")
    p.add_argument("--data",    default="CMAPSSData")
    p.add_argument("--model",   required=True)
    p.add_argument("--hidden",  type=int,   default=64)
    p.add_argument("--layers",  type=int,   default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--passes",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=256)
    return p.parse_args()


def eval_on_perturbed(wrapper, x_all, y_all, batch_size, device):
    """Run inference on a pre-built numpy array of perturbed windows."""
    wrapper.model.to(device)
    all_means = []

    for i in range(0, len(x_all), batch_size):
        xb = torch.from_numpy(x_all[i:i+batch_size]).to(device)
        mean, _ = wrapper.mc_predict(xb)
        all_means.append(mean.cpu().numpy())

    return np.concatenate(all_means)


def main():
    args = parse_args()

    print("=" * 60)
    print("SHIFT-TS — Core 6: Robustness Experiments")
    print("=" * 60)

    # Data
    out    = build_engine_splits(Path(args.data))
    scaler = fit_scaler(out["df_train"])
    df_s   = apply_scaler(out["df_target"], scaler)
    ds     = CMAPSSDataset(df_s)

    # Build full numpy array for efficient perturbation
    all_x, all_y = [], []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        all_x.append(x.numpy())
        all_y.append(float(y))
    x_clean = np.stack(all_x)    # (N, W, F)
    y_arr   = np.array(all_y)    # (N,)

    print(f"\nTarget windows: {len(ds):,}")

    # Load model
    model = GRUBaseline(
        input_dim=ds.n_features, hidden_dim=args.hidden,
        num_layers=args.layers,  dropout=args.dropout,
    )
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    wrapper = MCDropoutWrapper(model, n_passes=args.passes)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate each perturbation
    print(f"\n  {'Perturbation':25}  {'MAE':>8}  {'RMSE':>8}  {'ΔMAE':>8}")
    print(f"  {'-'*55}")

    baseline_mae = None
    results = {}

    for name, perturb_fn in PERTURBATIONS.items():
        x_pert = perturb_fn(x_clean, seed=0)
        if isinstance(x_pert, torch.Tensor):
            x_pert = x_pert.numpy()

        preds = eval_on_perturbed(wrapper, x_pert, y_arr, args.batch, device)
        m   = mae(preds, y_arr)
        r   = rmse(preds, y_arr)
        results[name] = {"mae": m, "rmse": r}

        if name == "clean":
            baseline_mae = m

        delta = (m - baseline_mae) if baseline_mae is not None else 0.0
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"  {name:25}  {m:>8.2f}  {r:>8.2f}  {delta_str:>8}")

    # Save
    out_path = Path("models") / "robustness_results.npz"
    np.savez(out_path, **{
        f"{k}_{metric}": v[metric]
        for k, v in results.items()
        for metric in ("mae", "rmse")
    })
    print(f"\n  Results saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
