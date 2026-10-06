"""
run_step4.py
============
STEP 4: Model Training and Evaluation Pipeline.

4.0 Pre-checks:
    a) Step 3 Shift Table (mean standardized feature shift + % outside min/max
       for target and source_heldout control).
    b) Target engine leak assertion check.
    c) Confirmation that splits depend only on (source, seed).

4.1 Training per (source dataset, seed):
    - Sources: FD001, FD002, FD003, FD004
    - Seeds: 0, 1, 2
    - Models:
        1. baseline: GRUBaseline, early stopping on source_val RMSE.
        2. ssl_pretrained: SSLModel pretrained on source_train windows only (frozen).
        3. ssl_finetuned: Pretrained encoder + RegressionHead, fine-tuned end-to-end.

4.2 Saved Artifacts under models/step4/{source}/seed{seed}/:
    - scaler.pkl
    - baseline.pt
    - ssl_pretrained_full.pt
    - ssl_finetuned.pt
    - config.json
    - train.log

4.3 Evaluation & Metrics saved to results/step4_metrics.csv.

4.4 Run Order:
    - First: source FD001, seed 0 only.
    - Waits for user approval before remaining (source, seed) combinations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Use all available CPU cores for PyTorch intra-op parallelism
import os as _os
_n_cpu = _os.cpu_count() or 1
import torch as _torch
_torch.set_num_threads(_n_cpu)
print(f"[Speed] Using {_n_cpu} CPU threads for torch intra-op parallelism.")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.constants import (
    DATASET_CONFIG,
    FEATURE_COLS,
    FIXED_OP_RANGES,
    OP_COLS,
    SCENARIOS,
    SENSOR_COLS,
    WINDOW_SIZE,
)
from src.data.dataset import CMAPSSDataset, SSLDataset
from src.data.loader import (
    assert_no_target_engine_leak,
    build_engine_splits,
    load_raw,
    load_target_engines,
)
from src.data.preprocessing import apply_scaler, fit_scaler, save_scaler
from src.models.baseline import GRUBaseline
from src.models.ssl_heads import SSLModel
from src.training.ssl_trainer import SSLTrainer, SSLTrainerConfig
from src.training.trainer import Trainer, TrainerConfig

DATA_DIR = PROJECT_ROOT / "CMAPSSData"
MODELS_STEP4_DIR = PROJECT_ROOT / "models" / "step4"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_CSV_PATH = RESULTS_DIR / "step4_metrics.csv"


# ---------------------------------------------------------------------------
# Seeding Helper
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Set random seeds for python, numpy, torch, and cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Return current git commit hash, or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 4.0 Pre-checks
# ---------------------------------------------------------------------------

def run_prechecks(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Execute all 4.0 Pre-checks:
      a) Step 3 Shift Table
      b) Assertion helper validation
      c) Split determinism validation
    Returns shift table DataFrame.
    """
    print("\n" + "=" * 80)
    print("  4.0 PRE-CHECKS")
    print("=" * 80)

    # ── b) Test Assertion Helper ───────────────────────────────────────────
    print("\n[Pre-check b] Validating target engine leak assertion helper...")
    try:
        assert_no_target_engine_leak([1, 2, 3], [4, 5, 6], stage_name="precheck_pass")
        caught = False
        try:
            assert_no_target_engine_leak([1, 2, 3], [3, 4, 5], stage_name="precheck_fail")
        except AssertionError:
            caught = True
        if not caught:
            raise RuntimeError("Assertion helper failed to raise on deliberate leak!")
        print("  ✓ Assertion helper raises on overlap and passes on disjoint engines.")
    except Exception as e:
        print(f"  FAILED: {e}")
        raise

    # ── c) Confirm Split Determinism ──────────────────────────────────────
    print("\n[Pre-check c] Confirming splits depend only on (source, seed)...")
    for src in ["FD001", "FD002", "FD003", "FD004"]:
        for s in [0, 1, 2]:
            split1 = build_engine_splits(data_dir, dataset_id=src, split_seed=s)
            split2 = build_engine_splits(data_dir, dataset_id=src, split_seed=s)
            assert split1["splits"]["train"] == split2["splits"]["train"]
            assert split1["splits"]["val"] == split2["splits"]["val"]
            assert split1["splits"]["target"] == split2["splits"]["target"]
    print("  ✓ Engine splits depend strictly on (source dataset, seed).")
    print("    Scenarios sharing a source reuse identical engine splits.")

    # ── a) Step 3 Shift Table ─────────────────────────────────────────────
    print("\n[Pre-check a] Computing Step 3 Feature Shift Table across all scenarios...")
    print("    (Excluding zero-variance features when standardizing; reporting sensors & op cols separately)")
    rows = []
    for sname, scfg in SCENARIOS.items():
        src = scfg["source"]
        tgt = scfg["target"]
        shift_type = scfg["shift_type"]

        # Build source splits at reference seed 0
        src_splits = build_engine_splits(data_dir, dataset_id=src, split_seed=0)
        df_src_train = src_splits["df_train"]
        df_src_heldout = src_splits["df_target"]
        if shift_type == "in_domain":
            df_tgt = df_src_heldout
        else:
            df_tgt = load_target_engines(data_dir, dataset_id=tgt)

        def compute_group_shift(cols: list[str]) -> tuple[float, float, float, float]:
            tr = df_src_train[cols].values.astype(np.float64)
            ho = df_src_heldout[cols].values.astype(np.float64)
            tg = df_tgt[cols].values.astype(np.float64)

            mu_tr = tr.mean(axis=0)
            std_tr = tr.std(axis=0)
            min_tr = tr.min(axis=0)
            max_tr = tr.max(axis=0)

            # Exclude zero-variance columns when standardizing
            valid_std = std_tr > 1e-6

            if np.any(valid_std):
                ctrl_shift = float(np.mean(np.abs(ho.mean(axis=0)[valid_std] - mu_tr[valid_std]) / std_tr[valid_std]))
                tgt_shift = float(np.mean(np.abs(tg.mean(axis=0)[valid_std] - mu_tr[valid_std]) / std_tr[valid_std]))
            else:
                ctrl_shift = 0.0
                tgt_shift = 0.0

            ctrl_out = float(((ho < min_tr) | (ho > max_tr)).mean() * 100.0) if len(cols) > 0 else 0.0
            tgt_out = float(((tg < min_tr) | (tg > max_tr)).mean() * 100.0) if len(cols) > 0 else 0.0

            return ctrl_shift, ctrl_out, tgt_shift, tgt_out

        s_c_shift, s_c_out, s_t_shift, s_t_out = compute_group_shift(SENSOR_COLS)
        op_c_shift, op_c_out, op_t_shift, op_t_out = compute_group_shift(OP_COLS)

        rows.append({
            "Scenario": sname,
            "Source": src,
            "Target": tgt,
            "Shift Type": shift_type,
            "Ctrl Sens Shift": round(s_c_shift, 4),
            "Ctrl Sens % Out": round(s_c_out, 2),
            "Tgt Sens Shift": round(s_t_shift, 4),
            "Tgt Sens % Out": round(s_t_out, 2),
            "Ctrl Op Shift": round(op_c_shift, 4),
            "Ctrl Op % Out": round(op_c_out, 2),
            "Tgt Op Shift": round(op_t_shift, 4),
            "Tgt Op % Out": round(op_t_out, 2),
        })

    shift_df = pd.DataFrame(rows)
    print("\n" + shift_df.to_string(index=False))

    # Control assertions
    max_ctrl_s_shift = shift_df["Ctrl Sens Shift"].max()
    max_ctrl_s_out = shift_df["Ctrl Sens % Out"].max()
    max_ctrl_op_shift = shift_df["Ctrl Op Shift"].max()
    max_ctrl_op_out = shift_df["Ctrl Op % Out"].max()

    assert max_ctrl_s_shift < 0.15, f"Control sensor standardized shift too high: {max_ctrl_s_shift}"
    assert max_ctrl_s_out < 1.0, f"Control sensor % outside min/max too high: {max_ctrl_s_out}%"
    assert max_ctrl_op_shift < 0.15, f"Control op standardized shift too high: {max_ctrl_op_shift}"
    assert max_ctrl_op_out < 1.0, f"Control op % outside min/max too high: {max_ctrl_op_out}%"
    print(f"\n  ✓ Control shifts are near zero (sensor max shift={max_ctrl_s_shift:.4f}, out={max_ctrl_s_out:.2f}%; op max shift={max_ctrl_op_shift:.4f}, out={max_ctrl_op_out:.2f}%).")
    print("=" * 80)
    print("  ALL 4.0 PRE-CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)
    return shift_df


# ---------------------------------------------------------------------------
# Training & Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Compute MAE and RMSE."""
    mae = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    return round(mae, 3), round(rmse, 3)


def evaluate_model_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Inference over a DataLoader, returning numpy arrays of (preds, targets)."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            pred = model(x).cpu().numpy().reshape(-1)
            all_preds.append(pred)
            all_targets.append(y.numpy().reshape(-1))
    return np.concatenate(all_preds), np.concatenate(all_targets)


def train_and_eval_source_seed(
    source: str,
    seed: int,
    data_dir: Path = DATA_DIR,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Train and evaluate all models for a specific (source dataset, seed).

    Artifacts saved to models/step4/{source}/seed{seed}/:
      - scaler.pkl
      - baseline.pt
      - ssl_pretrained_full.pt
      - ssl_finetuned.pt
      - config.json
      - train.log

    Returns:
      (metrics_df, runtimes_dict)
    """
    save_dir = MODELS_STEP4_DIR / source / f"seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "train.log"

    def log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"\n{'='*72}\n  STARTING PIPELINE: Source={source}, Seed={seed}\n{'='*72}")

    # Set seeds
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    # Check if existing run can be skipped
    needed_files = [
        save_dir / "scaler.pkl",
        save_dir / "baseline.pt",
        save_dir / "ssl_pretrained_full.pt",
        save_dir / "ssl_finetuned.pt",
        save_dir / "config.json",
    ]
    all_exist = all(p.exists() for p in needed_files)

    # 1. Build splits & fit scaler
    splits_out = build_engine_splits(data_dir, dataset_id=source, split_seed=seed)
    df_train = splits_out["df_train"]
    df_val = splits_out["df_val"]
    df_heldout = splits_out["df_target"]
    train_eids = splits_out["splits"]["train"]
    val_eids = splits_out["splits"]["val"]
    target_eids = splits_out["splits"]["target"]

    # Strict leak assertions
    assert_no_target_engine_leak(train_eids, target_eids, stage_name=f"{source}_s{seed}_scaler")
    assert_no_target_engine_leak(train_eids, val_eids, stage_name=f"{source}_s{seed}_val_check")

    scaler_path = save_dir / "scaler.pkl"
    if not scaler_path.exists() or force:
        scaler = fit_scaler(df_train, dataset_id=source)
        save_scaler(scaler, scaler_path)
        log(f"  [1/4] Scaler fit on {len(train_eids)} source_train engines and saved to {scaler_path.name}")
    else:
        scaler = fit_scaler(df_train, dataset_id=source)  # re-fit in memory
        log(f"  [1/4] Loaded existing scaler from {scaler_path.name}")

    # Scale source splits
    df_train_s = apply_scaler(df_train, scaler)
    df_val_s = apply_scaler(df_val, scaler)
    df_heldout_s = apply_scaler(df_heldout, scaler)

    ds_train = CMAPSSDataset(df_train_s)
    ds_val = CMAPSSDataset(df_val_s)
    ds_heldout = CMAPSSDataset(df_heldout_s)

    loader_kw = dict(batch_size=512, shuffle=False, num_workers=0)
    train_loader_shuffled = DataLoader(ds_train, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, **loader_kw)
    heldout_loader = DataLoader(ds_heldout, **loader_kw)

    log(f"  Windows: train={len(ds_train):,}, val={len(ds_val):,}, heldout={len(ds_heldout):,}")

    runtimes = {}

    # -----------------------------------------------------------------------
    # 2. Train Baseline GRU
    # -----------------------------------------------------------------------
    baseline_pt_path = save_dir / "baseline.pt"
    baseline = GRUBaseline(
        input_dim=ds_train.n_features,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    if not baseline_pt_path.exists() or force:
        assert_no_target_engine_leak(train_eids, target_eids, stage_name=f"{source}_s{seed}_baseline")
        log("\n  [2/4] Training Baseline GRU (supervised on source_train, early stopping on val RMSE)...")
        b_cfg = TrainerConfig(
            run_name=f"{source}_s{seed}_baseline",
            checkpoint_dir=str(save_dir),
            epochs=60,
            lr=1e-3,
            batch_size=256,
            early_stop_patience=15,
            device=str(device),
        )
        b_trainer = Trainer(baseline, b_cfg)
        t0 = time.time()
        b_history = b_trainer.fit(train_loader_shuffled, val_loader)
        t_base = time.time() - t0
        runtimes["baseline"] = round(t_base, 2)

        # Load best checkpoint and save as canonical baseline.pt
        best_ckpt = save_dir / f"{b_cfg.run_name}_best.pt"
        if best_ckpt.exists():
            b_trainer.load(str(best_ckpt))
            torch.save(baseline.state_dict(), baseline_pt_path)
            try:
                best_ckpt.unlink()
            except Exception:
                pass
        else:
            torch.save(baseline.state_dict(), baseline_pt_path)
        log(f"  ✓ Baseline saved to baseline.pt (Runtime: {runtimes['baseline']}s, Best Val RMSE: {b_trainer.best_val_rmse:.3f})")
    else:
        baseline.load_state_dict(torch.load(baseline_pt_path, map_location=device, weights_only=True))
        log("  ✓ Loaded existing baseline.pt")
        runtimes["baseline"] = 0.0

    # -----------------------------------------------------------------------
    # 3. Pretrain SSL Model (Frozen representation)
    # -----------------------------------------------------------------------
    ssl_full_pt_path = save_dir / "ssl_pretrained_full.pt"
    ssl_model = SSLModel(
        input_dim=ds_train.n_features,
        hidden_dim=64,
        num_layers=2,
        proj_dim=64,
        alpha=1.0,
        beta=1.0,
        dropout=0.2,
    ).to(device)

    if not ssl_full_pt_path.exists() or force:
        assert_no_target_engine_leak(train_eids, target_eids, stage_name=f"{source}_s{seed}_ssl_pretrain")
        # Use fewer epochs for larger datasets (FD002/FD004) that converge faster
        ssl_epochs = 30 if len(ds_train) > 25_000 else 50
        log(f"\n  [3/4] Pretraining SSLModel on source_train windows only (mask=0.15, jitter=0.02, α=β=1.0, epochs={ssl_epochs})...")
        ds_ssl = SSLDataset(df_train_s, jitter_sigma=0.02, mask_ratio=0.15, seed=seed)
        ssl_loader = DataLoader(ds_ssl, batch_size=512, shuffle=True, num_workers=0)
        ssl_cfg = SSLTrainerConfig(
            run_name=f"{source}_s{seed}_ssl",
            checkpoint_dir=str(save_dir),
            epochs=ssl_epochs,
            lr=1e-3,
            batch_size=512,
            device=str(device),
        )
        ssl_trainer = SSLTrainer(ssl_model, ssl_cfg)
        t0 = time.time()
        ssl_history = ssl_trainer.fit(ssl_loader)
        t_ssl = time.time() - t0
        runtimes["ssl_pretrained"] = round(t_ssl, 2)

        # Save as frozen full checkpoint (never overwritten or fine-tuned)
        torch.save(ssl_model.state_dict(), ssl_full_pt_path)
        log(f"  ✓ SSL pretrained full model saved to ssl_pretrained_full.pt (Runtime: {runtimes['ssl_pretrained']}s)")
    else:
        ssl_model.load_state_dict(torch.load(ssl_full_pt_path, map_location=device, weights_only=True))
        log("  ✓ Loaded existing ssl_pretrained_full.pt")
        runtimes["ssl_pretrained"] = 0.0

    # -----------------------------------------------------------------------
    # 4. Fine-tune SSL Encoder + RegressionHead
    # -----------------------------------------------------------------------
    ssl_ft_pt_path = save_dir / "ssl_finetuned.pt"
    ssl_finetuned = GRUBaseline(
        input_dim=ds_train.n_features,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    if not ssl_ft_pt_path.exists() or force:
        assert_no_target_engine_leak(train_eids, target_eids, stage_name=f"{source}_s{seed}_ssl_finetune")
        log("\n  [4/4] Fine-tuning copy of SSL encoder + RegressionHead end-to-end on source_train labels...")
        # Inject pretrained encoder weights
        ssl_finetuned.encoder.load_state_dict(ssl_model.encoder.state_dict())

        ft_cfg = TrainerConfig(
            run_name=f"{source}_s{seed}_ssl_finetuned",
            checkpoint_dir=str(save_dir),
            epochs=60,
            lr=1e-3,
            batch_size=512,
            early_stop_patience=15,
            device=str(device),
        )
        ft_trainer = Trainer(ssl_finetuned, ft_cfg)
        t0 = time.time()
        ft_history = ft_trainer.fit(train_loader_shuffled, val_loader)
        t_ft = time.time() - t0
        runtimes["ssl_finetuned"] = round(t_ft, 2)

        best_ckpt = save_dir / f"{ft_cfg.run_name}_best.pt"
        if best_ckpt.exists():
            ft_trainer.load(str(best_ckpt))
            torch.save(ssl_finetuned.state_dict(), ssl_ft_pt_path)
            try:
                best_ckpt.unlink()
            except Exception:
                pass
        else:
            torch.save(ssl_finetuned.state_dict(), ssl_ft_pt_path)
        log(f"  ✓ SSL fine-tuned model saved to ssl_finetuned.pt (Runtime: {runtimes['ssl_finetuned']}s, Best Val RMSE: {ft_trainer.best_val_rmse:.3f})")
    else:
        ssl_finetuned.load_state_dict(torch.load(ssl_ft_pt_path, map_location=device, weights_only=True))
        log("  ✓ Loaded existing ssl_finetuned.pt")
        runtimes["ssl_finetuned"] = 0.0

    # -----------------------------------------------------------------------
    # 5. Save Config & Metadata
    # -----------------------------------------------------------------------
    if log_file.exists():
        import re
        log_content = log_file.read_text(encoding="utf-8")
        m_b = re.search(r"Baseline saved to baseline\.pt \(Runtime: ([\d\.]+)s", log_content)
        m_s = re.search(r"SSL pretrained full model saved to ssl_pretrained_full\.pt \(Runtime: ([\d\.]+)s", log_content)
        m_f = re.search(r"SSL fine-tuned model saved to ssl_finetuned\.pt \(Runtime: ([\d\.]+)s", log_content)
        if m_b and runtimes.get("baseline", 0.0) == 0.0:
            runtimes["baseline"] = float(m_b.group(1))
        if m_s and runtimes.get("ssl_pretrained", 0.0) == 0.0:
            runtimes["ssl_pretrained"] = float(m_s.group(1))
        if m_f and runtimes.get("ssl_finetuned", 0.0) == 0.0:
            runtimes["ssl_finetuned"] = float(m_f.group(1))

    config_dict = {
        "source": source,
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "runtimes_seconds": runtimes,
        "hyperparameters": {
            "window_size": WINDOW_SIZE,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 256,
            "baseline": {"epochs": 60, "lr": 1e-3, "early_stop_patience": 15},
            "ssl_pretrain": {"epochs": 50, "lr": 1e-3, "proj_dim": 64, "alpha": 1.0, "beta": 1.0, "jitter": 0.02, "mask": 0.15},
            "ssl_finetune": {"epochs": 60, "lr": 1e-3, "early_stop_patience": 15},
        },
        "engine_counts": {
            "train": len(train_eids),
            "val": len(val_eids),
            "heldout": len(target_eids),
        },
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # -----------------------------------------------------------------------
    # 6. Evaluation across val, heldout, and matching scenario targets
    # -----------------------------------------------------------------------
    log("\n  [Evaluation] Evaluating models on source_val, source_heldout, and matching targets...")
    eval_rows = []

    models_to_eval = [
        ("baseline", baseline),
        ("ssl_finetuned", ssl_finetuned),
    ]

    # Pre-load and cache target datasets required by scenarios where scfg["source"] == source
    matching_scenarios = {
        sname: scfg for sname, scfg in SCENARIOS.items() if scfg["source"] == source
    }
    target_loaders: dict[str, DataLoader] = {}
    for sname, scfg in matching_scenarios.items():
        tgt_ds = scfg["target"]
        if scfg["shift_type"] == "in_domain":
            continue
        if tgt_ds not in target_loaders:
            df_tgt_raw = load_target_engines(data_dir, dataset_id=tgt_ds)
            df_tgt_scaled = apply_scaler(df_tgt_raw, scaler)
            ds_tgt = CMAPSSDataset(df_tgt_scaled)
            target_loaders[tgt_ds] = DataLoader(ds_tgt, batch_size=512, shuffle=False, num_workers=0)

    # Evaluate each model
    last_ho_t: np.ndarray | None = None
    last_tgt_t: dict[str, np.ndarray] = {}

    for model_name, model in models_to_eval:
        # a) source_val
        val_p, val_t = evaluate_model_on_loader(model, val_loader, device)
        v_mae, v_rmse = compute_regression_metrics(val_p, val_t)
        eval_rows.append({
            "scenario": "none",
            "source": source,
            "target": source,
            "seed": seed,
            "model": model_name,
            "split": "val",
            "n_windows": len(val_p),
            "MAE": v_mae,
            "RMSE": v_rmse,
            "degenerate": False,
        })

        # b) source_heldout (control)
        ho_p, ho_t = evaluate_model_on_loader(model, heldout_loader, device)
        last_ho_t = ho_t
        ho_mae, ho_rmse = compute_regression_metrics(ho_p, ho_t)
        heldout_scenario_name = f"ID_{source}" if f"ID_{source}" in matching_scenarios else "in_domain_heldout"
        eval_rows.append({
            "scenario": heldout_scenario_name,
            "source": source,
            "target": source,
            "seed": seed,
            "model": model_name,
            "split": "source_heldout",
            "n_windows": len(ho_p),
            "MAE": ho_mae,
            "RMSE": ho_rmse,
            "degenerate": False,
        })

        # c) scenario targets
        for sname, scfg in matching_scenarios.items():
            tgt_ds = scfg["target"]
            if scfg["shift_type"] == "in_domain":
                continue

            tgt_loader = target_loaders[tgt_ds]
            t_p, t_t = evaluate_model_on_loader(model, tgt_loader, device)
            last_tgt_t[tgt_ds] = t_t
            t_mae, t_rmse = compute_regression_metrics(t_p, t_t)
            is_degenerate = bool(t_mae > 3.0 * ho_mae)

            eval_rows.append({
                "scenario": sname,
                "source": source,
                "target": tgt_ds,
                "seed": seed,
                "model": model_name,
                "split": "target",
                "n_windows": len(t_p),
                "MAE": t_mae,
                "RMSE": t_rmse,
                "degenerate": is_degenerate,
            })

    # d) Constant predictor: always predict mean source_train RUL
    if last_ho_t is not None:
        mean_source_train_rul = float(df_train["rul"].mean())
        ho_const_p = np.full_like(last_ho_t, mean_source_train_rul)
        ho_c_mae, ho_c_rmse = compute_regression_metrics(ho_const_p, last_ho_t)
        heldout_scenario_name = f"ID_{source}" if f"ID_{source}" in matching_scenarios else "in_domain_heldout"
        eval_rows.append({
            "scenario": heldout_scenario_name,
            "source": source,
            "target": source,
            "seed": seed,
            "model": "constant",
            "split": "source_heldout",
            "n_windows": len(last_ho_t),
            "MAE": ho_c_mae,
            "RMSE": ho_c_rmse,
            "degenerate": False,
        })

        for sname, scfg in matching_scenarios.items():
            tgt_ds = scfg["target"]
            if scfg["shift_type"] == "in_domain":
                continue
            if tgt_ds in last_tgt_t:
                t_t = last_tgt_t[tgt_ds]
                tgt_const_p = np.full_like(t_t, mean_source_train_rul)
                t_c_mae, t_c_rmse = compute_regression_metrics(tgt_const_p, t_t)
                is_c_deg = bool(t_c_mae > 3.0 * ho_c_mae)
                eval_rows.append({
                    "scenario": sname,
                    "source": source,
                    "target": tgt_ds,
                    "seed": seed,
                    "model": "constant",
                    "split": "target",
                    "n_windows": len(t_t),
                    "MAE": t_c_mae,
                    "RMSE": t_c_rmse,
                    "degenerate": is_c_deg,
                })

    metrics_df = pd.DataFrame(eval_rows)
    log("\n  Evaluation Results Summary:\n" + metrics_df.to_string(index=False))

    # Append to / update results/step4_metrics.csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_CSV_PATH.exists():
        old_df = pd.read_csv(METRICS_CSV_PATH)
        # Drop any existing rows matching this source and seed to allow clean overwrite
        mask = (old_df["source"] == source) & (old_df["seed"] == seed)
        merged_df = pd.concat([old_df[~mask], metrics_df], ignore_index=True)
        merged_df.to_csv(METRICS_CSV_PATH, index=False)
    else:
        metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    log(f"  ✓ Metrics saved/updated in {METRICS_CSV_PATH}")
    log(f"{'='*72}\n  PIPELINE FINISHED: Source={source}, Seed={seed}\n{'='*72}")
    return metrics_df, runtimes


# ---------------------------------------------------------------------------
# CLI & Execution Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SHIFT-TS Step 4 Training & Evaluation")
    parser.add_argument("--prechecks-only", action="store_true", help="Run 4.0 pre-checks only and exit")
    parser.add_argument("--skip-prechecks", action="store_true", help="Skip pre-checks if already passed")
    parser.add_argument("--source", default="FD001", choices=["FD001", "FD002", "FD003", "FD004", "all"])
    parser.add_argument("--seed", default="0", help="Seed number (0, 1, 2) or 'all'")
    parser.add_argument("--force", action="store_true", help="Force rerun even if checkpoints exist")
    args = parser.parse_args()

    # Pre-checks
    if not args.skip_prechecks:
        run_prechecks(DATA_DIR)
        if args.prechecks_only:
            return

    sources = ["FD001", "FD002", "FD003", "FD004"] if args.source == "all" else [args.source]
    seeds = [0, 1, 2] if args.seed == "all" else [int(args.seed)]

    for src in sources:
        for s in seeds:
            train_and_eval_source_seed(source=src, seed=s, data_dir=DATA_DIR, force=args.force)


if __name__ == "__main__":
    main()
