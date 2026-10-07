import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import build_engine_splits, load_target_engines, assert_no_target_engine_leak
from src.data.preprocessing import load_scaler, apply_scaler
from src.data.dataset import CMAPSSDataset
from torch.utils.data import DataLoader
from src.data.constants import VALID_DATASET_IDS, SCENARIOS

def compute_s_input(seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    step5_csv_path = PROJECT_ROOT / "results" / f"step5_scores_seed{seed}.csv"
    df5 = pd.read_csv(step5_csv_path)
    
    # We will build a dictionary to store s_input per row.
    # Key: (source, split, scenario, engine_id, window_idx)
    s_input_dict = {}
    
    sources = df5['source'].unique()
    
    for source in sources:
        seed_dir = PROJECT_ROOT / "models" / "step4" / source / f"seed{seed}"
        scaler = load_scaler(seed_dir / "scaler.pkl")
        
        splits = build_engine_splits(PROJECT_ROOT / "CMAPSSData", source, seed)
        train_eids = splits["splits"]["train"]
        target_eids = splits["splits"]["target"]
        
        assert_no_target_engine_leak(train_eids, target_eids, stage_name=f"s_input_train_{source}")
        
        train_df = apply_scaler(splits["df_train"], scaler)
        train_loader = DataLoader(CMAPSSDataset(train_df), batch_size=1024, shuffle=False)
        
        # 1. Fit Mahalanobis on feature-mean of source_train
        train_means = []
        for b_idx, batch in enumerate(train_loader):
            x = batch[0] # (B, seq_len, num_features)
            # feature-mean vector
            x_mean = x.mean(dim=1).numpy()
            train_means.append(x_mean)
        train_means = np.concatenate(train_means, axis=0)
        
        # exclude zero-variance columns
        variances = np.var(train_means, axis=0)
        valid_cols = variances > 1e-6
        train_means_valid = train_means[:, valid_cols]
        
        lw = LedoitWolf()
        lw.fit(train_means_valid)
        maha_mean = lw.location_
        maha_cov = lw.covariance_
        maha_inv_cov = np.linalg.inv(maha_cov)
        
        def process_loader(loader, scenario_name, split_name, tgt_ds):
            window_idx = 0
            for b_idx, batch in enumerate(loader):
                x, _, eids = batch
                x_mean = x.mean(dim=1).numpy()
                x_mean_valid = x_mean[:, valid_cols]
                
                diff = x_mean_valid - maha_mean
                # s_input is mahalanobis distance
                # shape: (B,)
                m_dist = np.sum(np.dot(diff, maha_inv_cov) * diff, axis=1)
                
                for i in range(len(x)):
                    key = (source, split_name, scenario_name, int(eids[i]), window_idx)
                    s_input_dict[key] = float(m_dist[i])
                    window_idx += 1

        # source_val
        val_df = apply_scaler(splits["df_val"], scaler)
        val_loader = DataLoader(CMAPSSDataset(val_df), batch_size=1024, shuffle=False)
        process_loader(val_loader, "none", "val", source)
        
        # source_heldout
        ho_df = apply_scaler(splits["df_target"], scaler)
        ho_loader = DataLoader(CMAPSSDataset(ho_df), batch_size=1024, shuffle=False)
        ho_scen = f"ID_{source}" if f"ID_{source}" in SCENARIOS else "in_domain_heldout"
        process_loader(ho_loader, ho_scen, "source_heldout", source)
        
        # targets
        for sname, scfg in SCENARIOS.items():
            if scfg["source"] == source and scfg["shift_type"] != "in_domain":
                tgt = scfg["target"]
                tgt_df = apply_scaler(load_target_engines(PROJECT_ROOT / "CMAPSSData", tgt), scaler)
                tgt_loader = DataLoader(CMAPSSDataset(tgt_df), batch_size=1024, shuffle=False)
                process_loader(tgt_loader, sname, "target", tgt)
                
    # Now merge s_input into df5
    def get_s_input(row):
        key = (row['source'], row['split'], row['scenario'], int(row['engine_id']), int(row['window_idx']))
        return s_input_dict.get(key, np.nan)
        
    df5['s_input'] = df5.apply(get_s_input, axis=1)
    
    # Assert row counts match and no NaNs in s_input for evaluated rows
    if df5['s_input'].isna().any():
        print(f"Failed: {df5['s_input'].isna().sum()} NaNs in s_input!")
        
    # Re-verify row counts
    df4 = pd.read_csv(PROJECT_ROOT / 'results' / 'step4_metrics.csv')
    df5_counts_baseline = df5[df5['predictor'] == 'baseline'].groupby(['source', 'split', 'scenario']).size().reset_index(name='n_windows_5')
    df4_baseline = df4[(df4['model'] == 'baseline') & (df4['seed'] == seed)][['source', 'split', 'scenario', 'n_windows']]
    merged = pd.merge(df5_counts_baseline, df4_baseline, on=['source', 'split', 'scenario'], how='inner')
    mismatches = merged[merged['n_windows_5'] != merged['n_windows']]
    
    if len(mismatches) > 0:
        print("Row count mismatches found against step4!")
        print(mismatches)
    else:
        print("Row counts successfully matched against step4 metrics.")
        
    df5.to_csv(step5_csv_path, index=False)
    print(f"Added s_input to {step5_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    compute_s_input(seed=args.seed)
