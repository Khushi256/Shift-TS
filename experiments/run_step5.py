import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constants import SCENARIOS, VALID_DATASET_IDS
from src.data.dataset import CMAPSSDataset
from src.data.loader import assert_no_target_engine_leak, build_engine_splits, load_target_engines
from src.data.preprocessing import apply_scaler, load_scaler
from src.models.baseline import GRUBaseline
from src.models.ssl_heads import SSLModel

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models" / "step4"


def enable_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def get_fixed_masks(seq_len=30, missing_rate=0.15, K=5):
    masks = []
    n_missing = int(seq_len * missing_rate)
    for k in range(K):
        rng = np.random.RandomState(1000 + k)
        m = np.zeros(seq_len, dtype=bool)
        idx = rng.choice(seq_len, size=n_missing, replace=False)
        m[idx] = True
        masks.append(torch.from_numpy(m))
    return masks


def compute_stage_bins(val_preds, val_abs_errs):
    min_val, max_val = val_preds.min(), val_preds.max()
    # 10 equal bins
    bins = np.linspace(min_val, max_val, 11)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bin_idx = np.digitize(val_preds, bins) - 1  # 0 to 9
    
    bin_means = np.zeros(10)
    global_mean = val_abs_errs.mean()
    for i in range(10):
        mask = (bin_idx == i)
        if np.any(mask):
            bin_means[i] = val_abs_errs[mask].mean()
        else:
            bin_means[i] = global_mean
            
    return bins, bin_means


def run_step5(seed=0, source_filter=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_csv = RESULTS_DIR / f"step5_scores_seed{seed}.csv"
    out_meta = RESULTS_DIR / f"step5_fit_meta_seed{seed}.json"
    
    metrics_csv_path = RESULTS_DIR / "step4_metrics.csv"
    if not metrics_csv_path.exists():
        raise RuntimeError("step4_metrics.csv not found")
    step4_df = pd.read_csv(metrics_csv_path)
    
    fixed_masks = get_fixed_masks()
    
    all_rows = []
    meta_info = {
        "K": 5,
        "mask_seeds": [1000 + k for k in range(5)],
        "missing_rate": 0.15,
        "MC_T": 20,
        "maha_fits": {},
        "stage_fits": {}
    }
    
    sources = [source_filter] if source_filter else VALID_DATASET_IDS
    
    for source in sources:
        seed_dir = MODELS_DIR / source / f"seed{seed}"
        if not seed_dir.exists():
            continue
            
        scaler = load_scaler(seed_dir / "scaler.pkl")
        
        baseline = GRUBaseline(input_dim=17, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        baseline.load_state_dict(torch.load(seed_dir / "baseline.pt", map_location=device))
        baseline.eval()
        
        ssl_model = SSLModel(input_dim=17, hidden_dim=64, num_layers=2, proj_dim=64, dropout=0.2).to(device)
        ssl_model.load_state_dict(torch.load(seed_dir / "ssl_pretrained_full.pt", map_location=device))
        ssl_model.eval()
        
        splits = build_engine_splits(PROJECT_ROOT / "CMAPSSData", source, seed)
        train_eids = splits["splits"]["train"]
        val_eids = splits["splits"]["val"]
        target_eids = splits["splits"]["target"]
        
        assert_no_target_engine_leak(train_eids, target_eids, stage_name="step5_maha_train")
        assert_no_target_engine_leak(val_eids, target_eids, stage_name="step5_stage_val")
        
        train_df = apply_scaler(splits["df_train"], scaler)
        train_loader = DataLoader(CMAPSSDataset(train_df), batch_size=1024, shuffle=False)
        
        # 1. Fit Mahalanobis on source_train
        train_embs = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch[0].to(device)
                train_embs.append(baseline.encode(x).cpu().numpy())
        train_embs = np.concatenate(train_embs)
        
        lw = LedoitWolf()
        lw.fit(train_embs)
        maha_mean = lw.location_
        maha_cov = lw.covariance_
        maha_inv_cov = np.linalg.inv(maha_cov)
        
        meta_info["maha_fits"][source] = {
            "mean": maha_mean.tolist(),
            "cov": maha_cov.tolist()
        }
        
        # 2. Fit Stage bins on source_val
        val_df = apply_scaler(splits["df_val"], scaler)
        val_loader = DataLoader(CMAPSSDataset(val_df), batch_size=1024, shuffle=False)
        val_preds, val_errs = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = batch
                x = x.to(device)
                p = baseline(x).cpu().numpy()
                t = y.numpy()
                val_preds.append(p)
                val_errs.append(np.abs(p - t))
        val_preds = np.concatenate(val_preds)
        val_errs = np.concatenate(val_errs)
        
        bins, bin_means = compute_stage_bins(val_preds, val_errs)
        meta_info["stage_fits"][source] = {
            "bins": bins.tolist(),
            "bin_means": bin_means.tolist()
        }
        
        def process_loader(loader, scenario_name, split_name, tgt_ds):
            window_idx = 0
            for b_idx, batch in enumerate(loader):
                x, y, eids = batch
                x = x.to(device)
                y = y.cpu().numpy()
                B = x.shape[0]
                eids = eids.numpy()
                
                # Pred & error
                with torch.no_grad():
                    baseline.eval()
                    p = baseline(x).cpu().numpy()
                abs_err = np.abs(p - y)
                
                # s_maha
                with torch.no_grad():
                    baseline.eval()
                    emb = baseline.encode(x).cpu().numpy()
                diff = emb - maha_mean
                s_maha = np.sqrt(np.sum(np.dot(diff, maha_inv_cov) * diff, axis=1))
                
                # s_stage
                b_idx_stage = np.digitize(p, bins) - 1
                b_idx_stage = np.clip(b_idx_stage, 0, 9)
                s_stage = bin_means[b_idx_stage]
                
                # s_mc
                preds_mc = []
                torch.manual_seed(10000 + b_idx)
                for _ in range(20):
                    baseline.eval()
                    baseline.apply(enable_dropout)
                    with torch.no_grad():
                        preds_mc.append(baseline(x).cpu().numpy())
                preds_mc = np.stack(preds_mc)
                s_mc = np.var(preds_mc, axis=0)
                
                # s_rec
                s_recs = []
                ssl_model.eval()
                with torch.no_grad():
                    hs, _ = ssl_model.encoder(x)
                    for m in fixed_masks:
                        mask_b = m.unsqueeze(0).expand(B, -1).to(device)
                        recon = ssl_model.recon_head(hs, mask_b) # (n_masked, F)
                        target = x[mask_b]
                        
                        mse = (recon - target)**2 # (n_masked, F)
                        mse_f = mse.mean(dim=-1) # (n_masked)
                        
                        # Reshape back to (B, num_masked_per_seq)
                        mse_b = mse_f.view(B, -1)
                        s_recs.append(mse_b.mean(dim=-1).cpu().numpy())
                s_rec = np.stack(s_recs).mean(axis=0)
                
                for i in range(B):
                    all_rows.append({
                        "scenario": scenario_name,
                        "source": source,
                        "target": tgt_ds,
                        "seed": seed,
                        "split": split_name,
                        "predictor": "baseline",
                        "engine_id": int(eids[i]),
                        "window_idx": window_idx,
                        "true_rul": float(y[i]),
                        "pred_rul": float(p[i]),
                        "abs_err": float(abs_err[i]),
                        "s_rec": float(s_rec[i]),
                        "s_mc": float(s_mc[i]),
                        "s_maha": float(s_maha[i]),
                        "s_stage": float(s_stage[i])
                    })
                    window_idx += 1

        # source_val
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

    df = pd.DataFrame(all_rows)
    
    # Checks
    print("\n[CHECKS]")
    # b. No NaN or inf
    nan_counts = df.isna().sum()
    print(f"b. NaN counts per column:\n{nan_counts[nan_counts > 0]}")
    if nan_counts.sum() > 0:
        raise ValueError("NaNs found in scores")
        
    inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()
    if inf_counts.sum() > 0:
        raise ValueError("Infs found in scores")
    print("✓ Check b passed: No NaNs or Infs.")
    
    # c. Row counts and d. MAE
    check_c_passed = True
    check_d_passed = True
    
    for (src, split, scen), group in df.groupby(["source", "split", "scenario"]):
        n_windows = len(group)
        mae = group["abs_err"].mean()
        
        mask = (step4_df["source"] == src) & (step4_df["seed"] == seed) & \
               (step4_df["model"] == "baseline") & (step4_df["split"] == split)
        if split == "target":
            mask &= (step4_df["scenario"] == scen)
            
        s4_row = step4_df[mask]
        if s4_row.empty:
            continue
            
        s4_n = s4_row.iloc[0]["n_windows"]
        s4_mae = s4_row.iloc[0]["MAE"]
        
        if n_windows != s4_n:
            print(f"FAILED Row Count: {src} {split} {scen}. Expected {s4_n}, got {n_windows}")
            check_c_passed = False
            
        if abs(mae - s4_mae) > 0.01:
            print(f"FAILED MAE match: {src} {split} {scen}. Expected {s4_mae:.4f}, got {mae:.4f}")
            check_d_passed = False
            
    if check_c_passed:
        print("✓ Check c passed: Row counts match step4_metrics.")
    if check_d_passed:
        print("✓ Check d passed: MAE recomputed matches step4_metrics.")
        
    # f. Sanity on shift
    print("\nf. Sanity on shift (mean score source_heldout vs target):")
    for scen in [s for s, cfg in SCENARIOS.items() if cfg["shift_type"] != "in_domain"]:
        tgt_rows = df[df["scenario"] == scen]
        if tgt_rows.empty: continue
        src = tgt_rows.iloc[0]["source"]
        ho_scen = f"ID_{src}" if f"ID_{src}" in SCENARIOS else "in_domain_heldout"
        ho_rows = df[(df["source"] == src) & (df["scenario"] == ho_scen) & (df["split"] == "source_heldout")]
        
        print(f"Scenario: {scen} ({src} -> {tgt_rows.iloc[0]['target']})")
        for s in ["s_rec", "s_mc", "s_maha", "s_stage"]:
            print(f"  {s}: HO={ho_rows[s].mean():.4f}  TGT={tgt_rows[s].mean():.4f}")
            
    # g. Spearman on FD002 source_heldout
    print("\ng. Spearman correlation matrix on FD002 source_heldout:")
    fd2_ho = df[(df["source"] == "FD002") & (df["split"] == "source_heldout")]
    if not fd2_ho.empty:
        corr = fd2_ho[["s_rec", "s_mc", "s_maha", "s_stage"]].corr(method="spearman")
        print(corr.to_string())
        
    df.to_csv(out_csv, index=False)
    with open(out_meta, "w") as f:
        json.dump(meta_info, f)
        
    size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"\nSaved {out_csv.name} ({size_mb:.1f} MB)")
    
    if not (check_c_passed and check_d_passed):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()
    
    run_step5(args.seed, args.source)
