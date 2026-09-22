"""
run_fewshot.py
==============
Core 4 — Few-shot adaptation to the unseen target engine cohort.

Corrected pipeline
------------------
The key insight (and previous bug) is that few-shot adaptation REQUIRES a
warm encoder — one whose representations are already aligned with RUL
regression.  The three initialization modes reflect this:

    --init baseline   ← RECOMMENDED
        Load a supervised GRUBaseline checkpoint trained on train engines.
        Freeze encoder. Warm-start head from the pretrained head weights.
        Fine-tune head on 1%/5%/20% target labels.
        → Tests: "Can domain adaptation with few target labels beat zero-shot?"

    --init ssl-then-supervised   ← SSL ABLATION (correct SSL pipeline)
        Load an SSL-pretrained encoder, then fine-tune it (+ fresh head)
        on ALL train engine labels (supervised).  Then freeze and adapt head.
        → Tests: "Does SSL pretraining improve downstream supervised learning?"
        → Requires --ssl-encoder and --train-data

    --init ssl-only   ← FOR ABLATION / DEBUGGING ONLY
        Load an SSL encoder directly with a fresh head.
        Expected to perform poorly — SSL representations have no RUL structure.
        → Tests: "How much does the supervised step matter?"

Usage
-----
    # Correct pipeline (baseline encoder)
    python experiments/run_fewshot.py \\
        --init baseline \\
        --checkpoint models/baseline_best.pt \\
        --fractions 0.01 0.05 0.20

    # SSL ablation (full correct SSL pipeline)
    python experiments/run_fewshot.py \\
        --init ssl-then-supervised \\
        --ssl-encoder models/ssl_pretrain_encoder_best.pt \\
        --fractions 0.01 0.05 0.20

    # Debug only (known to perform poorly)
    python experiments/run_fewshot.py \\
        --init ssl-only \\
        --ssl-encoder models/ssl_pretrain_encoder_best.pt \\
        --fractions 0.01 0.05 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    build_engine_splits, fit_scaler, apply_scaler,
    CMAPSSDataset, FewShotDataset,
)
from src.models.baseline import GRUBaseline, RegressionHead
from src.models.encoder import GRUEncoder
from src.training.adapt import (
    FewShotAdapter, AdaptConfig,
    load_encoder_from_baseline, load_encoder_from_ssl,
)
from src.training.trainer import Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SHIFT-TS Few-Shot Adaptation")
    p.add_argument("--data",        default="CMAPSSData")
    p.add_argument("--init",        choices=["baseline", "ssl-then-supervised", "ssl-only"],
                   default="baseline",
                   help="Encoder initialisation strategy")
    p.add_argument("--checkpoint",  default="models/baseline_best.pt",
                   help="[baseline] Supervised GRUBaseline checkpoint path")
    p.add_argument("--ssl-encoder", default=None,
                   help="[ssl*] SSL encoder checkpoint path")
    p.add_argument("--fractions",   nargs="+", type=float, default=[0.01, 0.05, 0.20])
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--layers",      type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--workers",     type=int,   default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Supervised fine-tune helper (for ssl-then-supervised mode)
# ---------------------------------------------------------------------------

def supervised_finetune(
    encoder: GRUEncoder,
    df_train_s,
    df_val_s,
    n_features: int,
    hidden: int,
    dropout: float,
    batch: int,
    workers: int,
    run_name: str = "ssl_supervised",
) -> GRUBaseline:
    """
    After SSL pretraining, fine-tune encoder + fresh head on all train-engine
    labels.  This gives the encoder RUL-relevant representations before we
    freeze it for few-shot adaptation.
    """
    print("  [ssl-then-supervised] Supervised fine-tuning on train engines …")

    # Build a GRUBaseline and inject the SSL encoder weights
    model = GRUBaseline(
        input_dim  = n_features,
        hidden_dim = hidden,
        num_layers = encoder.num_layers,
        dropout    = dropout,
    )
    model.encoder.load_state_dict(encoder.state_dict())

    ds_train = CMAPSSDataset(df_train_s)
    ds_val   = CMAPSSDataset(df_val_s)
    tr_loader = DataLoader(ds_train, batch_size=batch, shuffle=True,
                            num_workers=workers)
    va_loader = DataLoader(ds_val,   batch_size=batch, shuffle=False,
                            num_workers=workers)

    config  = TrainerConfig(run_name=run_name, epochs=30, lr=5e-4)
    trainer = Trainer(model, config)
    trainer.fit(tr_loader, va_loader)

    # Load best checkpoint
    trainer.load(f"models/{run_name}_best.pt")
    print("  [ssl-then-supervised] Done.")
    return trainer.model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 65)
    print("SHIFT-TS — Core 4: Few-Shot Adaptation")
    print(f"  Initialisation mode : {args.init}")
    print(f"  Label fractions     : {[f'{f*100:.0f}%' for f in args.fractions]}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data...")
    out     = build_engine_splits(Path(args.data))
    scaler  = fit_scaler(out["df_train"])

    df_train_s  = apply_scaler(out["df_train"],  scaler)
    df_val_s    = apply_scaler(out["df_val"],    scaler)
    df_target_s = apply_scaler(out["df_target"], scaler)

    ds_target   = CMAPSSDataset(df_target_s)
    eval_loader = DataLoader(ds_target, batch_size=256, shuffle=False,
                              num_workers=args.workers)
    n_features  = ds_target.n_features

    print(f"  Target engines : {len(out['splits']['target'])},  "
          f"total windows : {len(ds_target):,}")

    # Zero-shot baseline (for reference)
    def zero_shot_rmse(model_path: str) -> float:
        model = GRUBaseline(input_dim=n_features, hidden_dim=args.hidden,
                             num_layers=args.layers, dropout=args.dropout)
        ckpt  = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y, _ in eval_loader:
                preds.append(model(x).numpy())
                targets.append(y.numpy())
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        return float(np.sqrt(((p - t)**2).mean()))

    # ------------------------------------------------------------------
    # Encoder initialisation
    # ------------------------------------------------------------------
    print("\n[2] Loading encoder (%s)..." % args.init)
    pretrained_head = None

    if args.init == "baseline":
        if not Path(args.checkpoint).exists():
            print(f"  ERROR: checkpoint not found: {args.checkpoint}")
            print("  Run: python experiments/run_baseline.py --epochs 60 first.")
            sys.exit(1)

        encoder, pretrained_head = load_encoder_from_baseline(
            args.checkpoint, n_features, args.hidden, args.layers, args.dropout
        )
        zs_rmse = zero_shot_rmse(args.checkpoint)
        print(f"  Loaded supervised encoder. Zero-shot target RMSE = {zs_rmse:.2f}")
        print("  Head will be WARM-STARTED from pretrained baseline head.")

    elif args.init == "ssl-then-supervised":
        if not args.ssl_encoder or not Path(args.ssl_encoder).exists():
            print("  ERROR: provide --ssl-encoder <path>")
            sys.exit(1)

        ssl_enc = load_encoder_from_ssl(
            args.ssl_encoder, n_features, args.hidden, args.layers, args.dropout
        )
        # Fine-tune on train engines with labels
        finetuned_model = supervised_finetune(
            ssl_enc, df_train_s, df_val_s,
            n_features, args.hidden, args.dropout,
            args.batch, args.workers,
            run_name="ssl_supervised",
        )
        encoder        = finetuned_model.encoder
        pretrained_head = finetuned_model.head
        # Zero-shot RMSE of the fine-tuned model
        finetuned_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y, _ in eval_loader:
                preds.append(finetuned_model(x).numpy())
                targets.append(y.numpy())
        zs_rmse = float(np.sqrt(((np.concatenate(preds) -
                                   np.concatenate(targets))**2).mean()))
        print("  SSL->supervised fine-tuned. Zero-shot target RMSE = %.2f" % zs_rmse)

    elif args.init == "ssl-only":
        if not args.ssl_encoder or not Path(args.ssl_encoder).exists():
            print("  ERROR: provide --ssl-encoder <path>")
            sys.exit(1)

        print("  WARNING: ssl-only mode uses SSL encoder with a FRESH head.")
        print("  This is expected to perform poorly. Use for ablation only.")
        encoder        = load_encoder_from_ssl(
            args.ssl_encoder, n_features, args.hidden, args.layers, args.dropout
        )
        pretrained_head = None
        zs_rmse        = float("nan")

    else:
        raise ValueError(f"Unknown --init mode: {args.init}")

    # ------------------------------------------------------------------
    # Few-shot adaptation for each label fraction
    # ------------------------------------------------------------------
    results = {}
    print("\n[3] Running few-shot adaptation...")
    print(f"    Zero-shot baseline RMSE = {zs_rmse:.2f}")
    print()

    for frac in args.fractions:
        pct = int(frac * 100)
        print("-" * 65)
        print(f"  Label fraction: {pct}%")

        ds_few    = FewShotDataset(ds_target, label_fraction=frac, seed=42)
        fs_loader = DataLoader(ds_few, batch_size=args.batch, shuffle=True,
                                num_workers=args.workers)
        print(f"  Few-shot windows: {len(ds_few)} / {len(ds_target):,} total")

        # Always reload encoder from the same checkpoint so each fraction
        # starts from identical weights (no contamination across fractions)
        if args.init == "baseline":
            encoder_i, head_i = load_encoder_from_baseline(
                args.checkpoint, n_features, args.hidden, args.layers, args.dropout
            )
        elif args.init == "ssl-then-supervised":
            import copy
            encoder_i = copy.deepcopy(encoder)
            head_i    = copy.deepcopy(pretrained_head)
        else:
            encoder_i = load_encoder_from_ssl(
                args.ssl_encoder, n_features, args.hidden, args.layers, args.dropout
            )
            head_i = None

        config = AdaptConfig(
            run_name  = f"fewshot_{args.init}_{pct}pct",
            epochs    = args.epochs,
            lr        = 1e-3,
        )
        adapter = FewShotAdapter(
            encoder         = encoder_i,
            pretrained_head = head_i,
            config          = config,
        )
        adapter.fit(fs_loader, eval_loader)
        adapter.load_best_head()

        preds, targets = adapter.predict(eval_loader)
        final_mae  = float(np.abs(preds - targets).mean())
        final_rmse = float(np.sqrt(((preds - targets)**2).mean()))

        results[frac] = {"mae": final_mae, "rmse": final_rmse}
        delta_rmse = final_rmse - zs_rmse
        sign = "+" if delta_rmse >= 0 else ""
        print("  -> %d%% labels  MAE %.2f  RMSE %.2f  (delta zero-shot: %s%.2f)" % (
              pct, final_mae, final_rmse, sign, delta_rmse))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Few-Shot Adaptation Summary  [%s]" % args.init)
    print("  Zero-shot baseline RMSE = %.2f" % zs_rmse)
    print("  %10s  %8s  %8s  %12s" % ("Fraction", "MAE", "RMSE", "Delta ZS"))
    for frac, m in results.items():
        d = m['rmse'] - zs_rmse
        sign2 = "+" if d >= 0 else ""
        print("  %9.0f%%  %8.2f  %8.2f    %s%.2f" % (frac*100, m['mae'], m['rmse'], sign2, d))
    print("=" * 65)
    print()
    print("Interpretation:")
    print("  Delta < 0  -> adaptation improved over zero-shot (good)")
    print("  Delta > 0  -> adaptation hurt performance (check encoder quality)")


if __name__ == "__main__":
    main()
