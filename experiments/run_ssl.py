"""
run_ssl.py
==========
Core 3 — Self-supervised pretraining experiment.

Usage
-----
    python experiments/run_ssl.py [--epochs 50] [--hidden 64] [--alpha 1.0]
                                  [--beta 1.0] [--data CMAPSSData]

What this does
--------------
1. Loads FD002 train engines (no labels used in pretraining)
2. Builds SSLDataset (augmented view pairs + masks)
3. Trains SSLModel (masked recon + contrastive) on train engines
4. Saves the encoder weights for Core 4 initialisation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_engine_splits, fit_scaler, apply_scaler
from src.data.dataset import SSLDataset
from src.models.ssl_heads import SSLModel
from src.training.ssl_trainer import SSLTrainer, SSLTrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS SSL Pretraining")
    p.add_argument("--data",     default="CMAPSSData")
    p.add_argument("--run-name", default="ssl_pretrain")
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--hidden",   type=int,   default=64)
    p.add_argument("--layers",   type=int,   default=2)
    p.add_argument("--proj-dim", type=int,   default=64)
    p.add_argument("--alpha",    type=float, default=1.0, help="Weight for recon loss")
    p.add_argument("--beta",     type=float, default=1.0, help="Weight for contrastive loss")
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--batch",    type=int,   default=256)
    p.add_argument("--workers",  type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SHIFT-TS — Core 3: Self-Supervised Pretraining")
    print("=" * 60)

    # 1. Data
    print("\n[1/3] Loading train engines …")
    out    = build_engine_splits(Path(args.data))
    scaler = fit_scaler(out["df_train"])
    df_train_s = apply_scaler(out["df_train"], scaler)

    ds_ssl = SSLDataset(df_train_s, jitter_sigma=0.02, mask_ratio=0.15)
    loader = DataLoader(ds_ssl, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers)
    print(f"  SSL windows: {len(ds_ssl):,}")

    # 2. Model
    print("\n[2/3] Building SSLModel …")
    model = SSLModel(
        input_dim   = ds_ssl._base.n_features,
        hidden_dim  = args.hidden,
        num_layers  = args.layers,
        proj_dim    = args.proj_dim,
        alpha       = args.alpha,
        beta        = args.beta,
        dropout     = args.dropout,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n:,}")

    # 3. Pretrain
    print("\n[3/3] Pretraining …")
    config  = SSLTrainerConfig(run_name=args.run_name, epochs=args.epochs,
                                batch_size=args.batch)
    trainer = SSLTrainer(model, config)
    trainer.fit(loader)

    # Save full model too
    trainer.save_full(f"models/{args.run_name}_full.pt")
    print(f"\nSSL pretraining complete. Encoder at models/{args.run_name}_encoder_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
