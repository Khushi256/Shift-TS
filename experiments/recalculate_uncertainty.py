"""
recalculate_uncertainty.py
==========================
Verify MC Dropout with T=20, active dropout, and re-compute:
- MAE, RMSE
- Nominal 95% interval coverage
- Spearman rank correlation (rho)
- Risk-coverage curve from raw predictions

Usage
-----
    python experiments/recalculate_uncertainty.py [--dataset FD002]
                                                  [--data CMAPSSData]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

sys.path.insert(0, ".")

from src.data import build_engine_splits, fit_scaler, apply_scaler, CMAPSSDataset
from src.models.baseline import GRUBaseline
from src.models.uncertainty import MCDropoutWrapper


def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS Recalculate Uncertainty")
    p.add_argument("--data",    default="CMAPSSData")
    p.add_argument("--dataset", default="FD002",
                   choices=["FD001", "FD002", "FD003", "FD004"],
                   help="C-MAPSS dataset to use")
    return p.parse_args()


args = parse_args()

print("Loading data...")
out = build_engine_splits(Path(args.data), dataset_id=args.dataset)
scaler = fit_scaler(out["df_train"])
df_target_s = apply_scaler(out["df_target"], scaler)
ds_target = CMAPSSDataset(df_target_s)
loader = DataLoader(ds_target, batch_size=256, shuffle=False)

print(f"Target windows: {len(ds_target):,}")

# Load model
model = GRUBaseline(input_dim=ds_target.n_features, hidden_dim=64, num_layers=2, dropout=0.2)
ckpt = torch.load("models/baseline_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])

# Set T=20
T = 20
wrapper = MCDropoutWrapper(model, n_passes=T)

# Ensure dropout is active across GRU and head
wrapper._enable_dropout()
# Also activate gru layer dropout
wrapper.model.encoder.gru.train()

print(f"Running MC Dropout with T={T} stochastic passes...")
all_means, all_stds, all_targets = [], [], []

with torch.no_grad():
    for x, y, _ in loader:
        passes = []
        for _ in range(T):
            pred = wrapper.model(x)
            passes.append(pred)
        stacked = torch.stack(passes, dim=0) # (T, B)
        mean = stacked.mean(dim=0).cpu().numpy()
        std  = stacked.std(dim=0).cpu().numpy()
        all_means.append(mean)
        all_stds.append(std)
        all_targets.append(y.numpy())

means = np.concatenate(all_means)
stds = np.concatenate(all_stds)
targets = np.concatenate(all_targets)
errors = np.abs(means - targets)

mae = float(errors.mean())
rmse = float(np.sqrt(((means - targets)**2).mean()))

# 95% nominal predictive interval: mu +- 1.96 * sigma
lower = means - 1.96 * stds
upper = means + 1.96 * stds
in_interval = (targets >= lower) & (targets <= upper)
coverage_95 = float(in_interval.mean() * 100)

# Spearman rho
rho, p_val = spearmanr(stds, errors)
rho = float(rho)

# Risk-coverage curve
sorted_idx = np.argsort(stds)
risk_coverage = []
for cov_pct in np.linspace(5, 100, 20):
    k = max(1, int(round(len(means) * cov_pct / 100.0)))
    sub_idx = sorted_idx[:k]
    sub_mae = float(errors[sub_idx].mean())
    risk_coverage.append({"coverage": round(cov_pct, 1), "mae": round(sub_mae, 2)})

print("\n" + "=" * 60)
print(f"RECALCULATED GROUND TRUTH METRICS  [{args.dataset}]  (Target Cohort, Zero-Shot, T=20)")
print("=" * 60)
print(f"MAE:              {mae:.2f}")
print(f"RMSE:             {rmse:.2f}")
print(f"Coverage (95% PI):{coverage_95:.1f}%")
print(f"Spearman rho:     {rho:.3f} (p-value: {p_val:.2e})")
print(f"Average std (sigma): {stds.mean():.2f}")
print("\nRisk-Coverage Curve sample:")
for pt in risk_coverage[::4]:
    print(f"  Coverage: {pt['coverage']:5.1f}% -> MAE: {pt['mae']:.2f}")
