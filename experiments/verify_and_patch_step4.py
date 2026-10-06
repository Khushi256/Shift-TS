import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constants import SCENARIOS, VALID_DATASET_IDS
from src.data.dataset import CMAPSSDataset
from src.data.loader import build_engine_splits, load_target_engines
from src.data.preprocessing import apply_scaler, load_scaler
from src.models.baseline import GRUBaseline
from src.models.ssl_heads import SSLModel

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_CSV = RESULTS_DIR / "step4_metrics.csv"
DISTS_CSV = RESULTS_DIR / "step4_pred_distributions.csv"
SUMMARY_CSV = RESULTS_DIR / "step4_summary.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "step4"


def compute_distributions():
    """Load checkpoints and compute prediction distributions."""
    print("Computing prediction distributions...")
    dist_rows = []
    
    # We only process what's available
    for source in VALID_DATASET_IDS:
        for seed in [0, 1, 2]:
            seed_dir = MODELS_DIR / source / f"seed{seed}"
            if not seed_dir.exists():
                continue
                
            scaler_path = seed_dir / "scaler.pkl"
            if not scaler_path.exists():
                continue
                
            try:
                scaler = load_scaler(scaler_path)
            except FileNotFoundError:
                continue
                
            models = {}
            if (seed_dir / "baseline.pt").exists():
                b_model = GRUBaseline(input_dim=17, hidden_dim=64, num_layers=2, dropout=0.2)
                b_model.load_state_dict(torch.load(seed_dir / "baseline.pt", map_location="cpu"))
                b_model.eval()
                models["baseline"] = b_model
                
            if (seed_dir / "ssl_finetuned.pt").exists():
                s_model = GRUBaseline(input_dim=17, hidden_dim=64, num_layers=2, dropout=0.2)
                s_model.load_state_dict(torch.load(seed_dir / "ssl_finetuned.pt", map_location="cpu"))
                s_model.eval()
                models["ssl_finetuned"] = s_model
                
            if not models:
                continue

            splits = build_engine_splits(PROJECT_ROOT / "CMAPSSData", source, seed)
            loaders = {}
            
            # source val
            loaders[("none", "val")] = DataLoader(CMAPSSDataset(apply_scaler(splits["df_val"], scaler)), batch_size=512)
            # source heldout
            ho_scen = f"ID_{source}" if f"ID_{source}" in SCENARIOS else "in_domain_heldout"
            loaders[(ho_scen, "source_heldout")] = DataLoader(CMAPSSDataset(apply_scaler(splits["df_target"], scaler)), batch_size=512)
            
            # targets
            for sname, scfg in SCENARIOS.items():
                if scfg["source"] == source and scfg["shift_type"] != "in_domain":
                    tgt = scfg["target"]
                    tgt_df = apply_scaler(load_target_engines(PROJECT_ROOT / "CMAPSSData", tgt), scaler)
                    loaders[(sname, "target")] = DataLoader(CMAPSSDataset(tgt_df), batch_size=512)
            
            for (scen, split), loader in loaders.items():
                for mname, m in models.items():
                    preds = []
                    with torch.no_grad():
                        for batch in loader:
                            x = batch[0]
                            p = m(x).squeeze(-1).numpy()
                            preds.append(p)
                    if preds:
                        preds = np.concatenate(preds)
                        dist_rows.append({
                            "scenario": scen,
                            "source": source,
                            "seed": seed,
                            "model": mname,
                            "split": split,
                            "pred_mean": float(preds.mean()),
                            "pred_std": float(preds.std()),
                            "pred_min": float(preds.min()),
                            "pred_max": float(preds.max()),
                            "frac_zeros": float((preds < 0.001).mean())
                        })
    
    if dist_rows:
        pd.DataFrame(dist_rows).to_csv(DISTS_CSV, index=False)
        print(f"  [SUCCESS] Saved {DISTS_CSV}")


def main():
    if not METRICS_CSV.exists():
        print(f"Error: {METRICS_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)
    
    # 1. Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["scenario", "source", "target", "seed", "model", "split"])
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate rows.")
    
    # 2. Add worse_than_constant and constant_MAE
    df["constant_MAE"] = np.nan
    df["worse_than_constant"] = False
    
    for _, row in df.iterrows():
        if row["model"] != "constant" and row["split"] == "target":
            const_row = df[(df["scenario"] == row["scenario"]) & 
                           (df["seed"] == row["seed"]) & 
                           (df["model"] == "constant") & 
                           (df["split"] == "target")]
            if not const_row.empty:
                c_mae = const_row.iloc[0]["MAE"]
                df.loc[row.name, "constant_MAE"] = c_mae
                df.loc[row.name, "worse_than_constant"] = row["MAE"] > c_mae

    df.to_csv(METRICS_CSV, index=False)
    
    # 3. Create distributions
    compute_distributions()
    
    # 4. Create summary CSV
    print("Generating summary...")
    summary = df[df["model"] != "constant"].groupby(["scenario", "model"]).agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        seeds=("seed", "nunique")
    ).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  [SUCCESS] Saved {SUMMARY_CSV}")
    print("\nSummary:")
    print(summary.to_string())
    
    # 5. Degeneracy report
    print("\nDegenerate / Worse than Constant report:")
    if "degenerate" not in df.columns:
        df["degenerate"] = False
    bad_runs = df[(df["degenerate"] == True) | (df["worse_than_constant"] == True)]
    if bad_runs.empty:
        print("  All runs look healthy.")
    else:
        print(bad_runs[["scenario", "source", "target", "seed", "model", "MAE", "constant_MAE", "degenerate", "worse_than_constant"]].to_string())
    
    # 6. Coverage grid
    print("\nChecking Coverage...")
    missing = []
    
    models = ["baseline", "ssl_finetuned", "constant"]
    seeds = [0, 1, 2]
    
    # Expected scenarios per source
    expected = []
    for src in VALID_DATASET_IDS:
        scens = [s for s, cfg in SCENARIOS.items() if cfg["source"] == src]
        for s in seeds:
            for m in models:
                # source val
                if m != "constant":
                    expected.append({"source": src, "seed": s, "model": m, "split": "val", "scenario": "none"})
                
                # source heldout
                ho_scen = f"ID_{src}" if f"ID_{src}" in SCENARIOS else "in_domain_heldout"
                expected.append({"source": src, "seed": s, "model": m, "split": "source_heldout", "scenario": ho_scen})
                
                # targets
                for sc in scens:
                    if SCENARIOS[sc]["shift_type"] != "in_domain":
                        expected.append({"source": src, "seed": s, "model": m, "split": "target", "scenario": sc})
    
    expected_df = pd.DataFrame(expected)
    
    for _, exp in expected_df.iterrows():
        mask = (df["source"] == exp["source"]) & \
               (df["seed"] == exp["seed"]) & \
               (df["model"] == exp["model"]) & \
               (df["split"] == exp["split"])
        if "scenario" in exp and pd.notna(exp["scenario"]):
            mask &= (df["scenario"] == exp["scenario"])
            
        if not df[mask].shape[0]:
            missing.append(exp.to_dict())
            
    if missing:
        print("\nMISSING RUNS:")
        print(pd.DataFrame(missing).to_string())
        sys.exit(1)
    else:
        print("\nAll expected rows are present!")
        sys.exit(0)

if __name__ == "__main__":
    main()
