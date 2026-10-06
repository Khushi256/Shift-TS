
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from src.data.loader import build_engine_splits, load_target_engines
from src.data.preprocessing import fit_scaler, apply_scaler
from src.data.dataset import CMAPSSDataset
from src.models.baseline import GRUBaseline

DATA_DIR = Path("CMAPSSData")

# 1. Constant predictor on FD001 seed 0
out = build_engine_splits(DATA_DIR, dataset_id="FD001", split_seed=0)
train_rul = out["df_train"]["rul"].values
mean_source_train_rul = float(train_rul.mean())
print(f"Mean source_train RUL (FD001, seed 0): {mean_source_train_rul:.4f}")

# Target datasets
scaler = fit_scaler(out["df_train"], dataset_id="FD001")

# Held-out
df_heldout = out["df_target"]
ds_heldout = CMAPSSDataset(apply_scaler(df_heldout, scaler))
heldout_targets = np.array([ds_heldout[i][1].item() for i in range(len(ds_heldout))])
heldout_const_pred = np.full_like(heldout_targets, mean_source_train_rul)
heldout_mae = float(np.abs(heldout_const_pred - heldout_targets).mean())
heldout_rmse = float(np.sqrt(((heldout_const_pred - heldout_targets)**2).mean()))
print(f"Heldout (control) — MAE: {heldout_mae:.3f}, RMSE: {heldout_rmse:.3f}")

targets = [
    ("FAULT_1", "FD003"),
    ("OPCOND_1", "FD002"),
    ("COMBINED", "FD004"),
]

constant_rows = [{
    "scenario": "in_domain_heldout",
    "source": "FD001",
    "target": "FD001",
    "seed": 0,
    "model": "constant",
    "split": "source_heldout",
    "n_windows": len(heldout_targets),
    "MAE": round(heldout_mae, 3),
    "RMSE": round(heldout_rmse, 3),
    "degenerate": False,
}]

target_loaders = {}
for sname, tgt_ds in targets:
    df_tgt = load_target_engines(DATA_DIR, dataset_id=tgt_ds)
    ds_tgt = CMAPSSDataset(apply_scaler(df_tgt, scaler))
    loader = DataLoader(ds_tgt, batch_size=256, shuffle=False)
    target_loaders[sname] = loader
    
    tgt_targets = np.array([ds_tgt[i][1].item() for i in range(len(ds_tgt))])
    const_pred = np.full_like(tgt_targets, mean_source_train_rul)
    t_mae = float(np.abs(const_pred - tgt_targets).mean())
    t_rmse = float(np.sqrt(((const_pred - tgt_targets)**2).mean()))
    is_deg = bool(t_mae > 3.0 * heldout_mae)
    print(f"Target {sname} ({tgt_ds}) — MAE: {t_mae:.3f}, RMSE: {t_rmse:.3f}, Degenerate: {is_deg}")
    constant_rows.append({
        "scenario": sname,
        "source": "FD001",
        "target": tgt_ds,
        "seed": 0,
        "model": "constant",
        "split": "target",
        "n_windows": len(tgt_targets),
        "MAE": round(t_mae, 3),
        "RMSE": round(t_rmse, 3),
        "degenerate": is_deg,
    })

# Append to results/step4_metrics.csv
csv_path = Path("results/step4_metrics.csv")
curr_df = pd.read_csv(csv_path)
# Remove any existing constant model rows for FD001 seed 0
mask = (curr_df["source"] == "FD001") & (curr_df["seed"] == 0) & (curr_df["model"] == "constant")
curr_df = curr_df[~mask]
new_df = pd.concat([curr_df, pd.DataFrame(constant_rows)], ignore_index=True)
new_df.to_csv(csv_path, index=False)
print("Updated results/step4_metrics.csv successfully.")

# 2. OPCOND_1 prediction distribution for baseline and ssl_finetuned
print("\n" + "="*60)
print("OPCOND_1 Prediction Distribution Check (Target: FD002, 46,219 windows)")
print("="*60)

save_dir = Path("models/step4/FD001/seed0")
device = torch.device("cpu")

baseline = GRUBaseline(input_dim=17, hidden_dim=64, num_layers=2, dropout=0.2)
baseline.load_state_dict(torch.load(save_dir / "baseline.pt", map_location=device, weights_only=True))
baseline.eval()

ssl_ft = GRUBaseline(input_dim=17, hidden_dim=64, num_layers=2, dropout=0.2)
ssl_ft.load_state_dict(torch.load(save_dir / "ssl_finetuned.pt", map_location=device, weights_only=True))
ssl_ft.eval()

opcond1_loader = target_loaders["OPCOND_1"]

def get_preds(m, l):
    preds = []
    with torch.no_grad():
        for batch in l:
            x = batch[0].to(device)
            p = m(x).cpu().numpy().reshape(-1)
            preds.append(p)
    return np.concatenate(preds)

base_preds = get_preds(baseline, opcond1_loader)
ssl_preds = get_preds(ssl_ft, opcond1_loader)

for name, p in [("baseline", base_preds), ("ssl_finetuned", ssl_preds)]:
    p_min = float(p.min())
    p_mean = float(p.mean())
    p_max = float(p.max())
    pct_zero = float((p == 0.0).mean() * 100.0)
    pct_near_zero = float((p < 1e-3).mean() * 100.0)
    print(f"\nModel: {name}")
    print(f"  Min:                     {p_min:.4f}")
    print(f"  Mean:                    {p_mean:.4f}")
    print(f"  Max:                     {p_max:.4f}")
    print(f"  % exactly equal to 0.0:  {pct_zero:.2f}%")
    print(f"  % < 0.001:               {pct_near_zero:.2f}%")
