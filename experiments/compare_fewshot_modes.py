"""
compare_fewshot_modes.py
========================
Systematic comparison of three few-shot adaptation regimes on target engines:
1. Frozen encoder + head adaptation
2. Partially unfrozen encoder (top GRU layer) + head
3. Fully fine-tuned model (both GRU layers + head)

Samples are selected BY ENGINE (not random scattered windows).
Evaluation is performed on the remaining unseen target engines.
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")

from src.data import build_engine_splits, fit_scaler, apply_scaler, CMAPSSDataset
from src.models.baseline import GRUBaseline, RegressionHead
from src.models.encoder import GRUEncoder


def get_engine_fewshot_split(ds_target: CMAPSSDataset, fraction: float, seed: int = 42):
    """
    Select few-shot samples strictly by engine.
    Returns:
        adapt_indices: indices belonging to the selected adaptation engines
        eval_indices:  indices belonging to the held-out target engines
        selected_eids: list of engine IDs used for adaptation
        eval_eids:     list of engine IDs used for evaluation
    """
    rng = np.random.default_rng(seed)
    all_eids = np.array(ds_target.engine_ids)
    rng.shuffle(all_eids)

    n_engines = max(1, int(round(len(all_eids) * fraction)))
    selected_eids = sorted(all_eids[:n_engines].tolist())
    eval_eids = sorted(all_eids[n_engines:].tolist())

    selected_set = set(selected_eids)
    adapt_indices = [i for i, (eid, _) in enumerate(ds_target._index) if eid in selected_set]
    eval_indices = [i for i, (eid, _) in enumerate(ds_target._index) if eid not in selected_set]

    return adapt_indices, eval_indices, selected_eids, eval_eids


def run_adaptation(
    checkpoint_path: str,
    ds_target: CMAPSSDataset,
    ds_val: CMAPSSDataset,
    fraction: float,
    mode: str,  # 'frozen_encoder', 'partially_unfrozen', 'fully_finetuned'
    epochs: int = 25,
    seed: int = 42,
    device: str = "cpu",
):
    adapt_idx, eval_idx, sel_eids, ev_eids = get_engine_fewshot_split(ds_target, fraction, seed=seed)

    ds_adapt = Subset(ds_target, adapt_idx)
    ds_eval = Subset(ds_target, eval_idx)

    train_loader = DataLoader(ds_adapt, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=256, shuffle=False)
    eval_loader = DataLoader(ds_eval, batch_size=256, shuffle=False)

    # Load baseline model
    model = GRUBaseline(input_dim=ds_target.n_features, hidden_dim=64, num_layers=2, dropout=0.2)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Initial zero-shot evaluation on the held-out evaluation engines
    model.eval()
    init_preds, init_targets = [], []
    with torch.no_grad():
        for x, y, _ in eval_loader:
            init_preds.append(model(x.to(device)).cpu().numpy())
            init_targets.append(y.numpy())
    p0 = np.concatenate(init_preds)
    t0 = np.concatenate(init_targets)
    zero_shot_mae = float(np.abs(p0 - t0).mean())
    zero_shot_rmse = float(np.sqrt(((p0 - t0)**2).mean()))

    # Configure unfreezing and parameters
    if mode == "frozen_encoder":
        for p in model.encoder.parameters():
            p.requires_grad = False
        param_groups = [
            {"params": model.head.parameters(), "lr": 5e-4, "weight_decay": 1e-4}
        ]
    elif mode == "partially_unfrozen":
        # Freeze GRU layer 0, unfreeze GRU layer 1
        for name, p in model.encoder.gru.named_parameters():
            if "l0" in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        param_groups = [
            {"params": [p for n, p in model.encoder.gru.named_parameters() if "l1" in n], "lr": 5e-5, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
        ]
    elif mode == "fully_finetuned":
        # Unfreeze all layers with lower learning rate on encoder
        for p in model.encoder.parameters():
            p.requires_grad = True
        param_groups = [
            {"params": model.encoder.parameters(), "lr": 5e-5, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
        ]
    else:
        raise ValueError(f"Unknown mode {mode}")

    optimizer = torch.optim.AdamW(param_groups)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        if mode == "frozen_encoder":
            model.encoder.eval()

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation check for early stopping (on validation cohort, leak-free)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())
        v_loss = np.mean(val_losses)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on held-out target engines
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for x, y, _ in eval_loader:
            final_preds.append(model(x.to(device)).cpu().numpy())
            final_targets.append(y.numpy())
    pf = np.concatenate(final_preds)
    tf = np.concatenate(final_targets)
    final_mae = float(np.abs(pf - tf).mean())
    final_rmse = float(np.sqrt(((pf - tf)**2).mean()))

    return {
        "fraction": fraction,
        "n_adapt_engines": len(sel_eids),
        "n_eval_engines": len(ev_eids),
        "n_adapt_windows": len(ds_adapt),
        "n_eval_windows": len(ds_eval),
        "mode": mode,
        "zero_shot_mae": zero_shot_mae,
        "zero_shot_rmse": zero_shot_rmse,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "delta_mae": final_mae - zero_shot_mae,
        "delta_rmse": final_rmse - zero_shot_rmse,
    }


def main():
    print("=" * 70)
    print("SHIFT-TS: Few-Shot Adaptation Benchmark across 3 Unfreezing Regimes")
    print("=" * 70)

    out = build_engine_splits(Path("CMAPSSData"))
    scaler = fit_scaler(out["df_train"])
    df_val_s = apply_scaler(out["df_val"], scaler)
    df_target_s = apply_scaler(out["df_target"], scaler)

    ds_val = CMAPSSDataset(df_val_s)
    ds_target = CMAPSSDataset(df_target_s)

    fractions = [0.01, 0.05, 0.20]
    modes = ["frozen_encoder", "partially_unfrozen", "fully_finetuned"]

    results = []
    ckpt_path = "models/baseline_best.pt"

    sys.stdout.reconfigure(encoding='utf-8')
    for frac in fractions:
        for mode in modes:
            print(f"\nRunning Fraction: {frac*100:4.1f}% | Mode: {mode:18s} ...")
            res = run_adaptation(ckpt_path, ds_target, ds_val, frac, mode, epochs=20, device="cpu")
            results.append(res)
            print(f"  -> ZS MAE: {res['zero_shot_mae']:.2f} | Adapt MAE: {res['final_mae']:.2f} (dMAE {res['delta_mae']:+.2f}) | Adapt RMSE: {res['final_rmse']:.2f}")

    print("\n" + "=" * 80)
    print(f"{'Fraction':<10} {'Mode':<20} {'Adapt Eng':<10} {'Adapt Win':<10} {'ZS MAE':<10} {'Adapt MAE':<12} {'Delta MAE':<10}")
    print("=" * 80)
    for r in results:
        pct = f"{r['fraction']*100:.0f}%"
        print(f"{pct:<10} {r['mode']:<20} {r['n_adapt_engines']:<10} {r['n_adapt_windows']:<10} {r['zero_shot_mae']:<10.2f} {r['final_mae']:<12.2f} {r['delta_mae']:<+10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
