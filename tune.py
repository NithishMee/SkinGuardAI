"""Bayesian Hyperparameter Tuning for SkinGuardAI (MedFusionNet) using Optuna."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch import nn
from torch.optim import AdamW

from ham10000_data import create_dataloaders, load_split_config
from medfusionnet import MedFusionNet
from train import build_lr_scheduler, run_epoch, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--n-trials", type=int, default=15, help="Number of tuning trials to run.")
    parser.add_argument("--epochs-per-trial", type=int, default=8, help="Epochs to train each trial.")
    parser.add_argument("--vit-type", choices=("swin", "compact"), default="swin")
    parser.add_argument("--batch-size", type=int, default=8, help="Base batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tuning"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    device: torch.device,
    config: dict,
) -> float:
    # 1. Hyperparameter Search Space
    learning_rate = trial.suggest_float("learning_rate", 4e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.20, step=0.05)
    dropout = trial.suggest_float("dropout", 0.20, 0.50, step=0.10)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 3)

    print(f"\n{'='*78}")
    print(f"🚀 [Trial {trial.number + 1}/{args.n_trials}] Parameters:")
    print(
        f"   lr: {learning_rate:.2e} | wd: {weight_decay:.2e} | "
        f"label_smoothing: {label_smoothing:.2f} | dropout: {dropout:.2f} | warmup: {warmup_epochs}"
    )
    print(f"{'='*78}")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance="none",
        augmentation="advanced",
    )

    model = MedFusionNet(
        num_classes=len(config["classes"]),
        image_size=224,
        vit_type=args.vit_type,
        dropout=dropout,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    param_groups = model.get_parameter_groups(
        backbone_lr=learning_rate * 0.1,
        head_lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer = AdamW(param_groups)
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        total_epochs=args.epochs_per_trial,
        warmup_epochs=warmup_epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_trial_acc = 0.0

    for epoch in range(1, args.epochs_per_trial + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=device.type == "cuda",
            accumulation_steps=args.accumulation_steps,
        )
        val_loss, val_acc, val_f1, _ = run_epoch(
            model, val_loader, criterion, device, amp_enabled=device.type == "cuda"
        )
        scheduler.step()
        epoch_time = time.time() - epoch_start

        best_trial_acc = max(best_trial_acc, val_acc)

        print(
            f"  Epoch {epoch:02d}/{args.epochs_per_trial:02d} ({int(epoch_time)}s) | "
            f"Train Acc: {train_acc * 100:.1f}%, Loss: {train_loss:.3f} | "
            f"Val Acc: {val_acc * 100:.2f}%, F1: {val_f1:.4f}"
        )

        # Report to Optuna for automated median trial pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"  ✂️  Trial {trial.number + 1} pruned by Optuna MedianPruner at epoch {epoch}.")
            raise optuna.TrialPruned()

    return best_trial_acc


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    config = load_split_config(args.splits_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = f"sqlite:///{(args.output_dir / 'optuna_study.db').as_posix()}"

    print("\n" + "=" * 78)
    print("  🔬 SkinGuardAI: Optuna Bayesian Hyperparameter Optimization")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Trials: {args.n_trials} | Epochs per trial: {args.epochs_per_trial}")
    print(f"  Search Algorithm: TPESampler (Bayesian) + MedianPruner")
    print("=" * 78 + "\n")

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)

    study = optuna.create_study(
        study_name="skinguardai_tuning",
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args, device, config),
            n_trials=args.n_trials,
        )
    except KeyboardInterrupt:
        print("\n[PAUSED] Hyperparameter tuning interrupted by user.")

    print("\n" + "=" * 78)
    print("🎉 HYPERPARAMETER TUNING COMPLETE!")
    print(f"  Total Trials Evaluated: {len(study.trials)}")
    print(f"  🏆 Best Validation Accuracy: {study.best_value * 100:.2f}%")
    print("\n  Top Hyperparameters Found:")
    for param_name, param_val in study.best_params.items():
        if isinstance(param_val, float):
            print(f"   - {param_name:18s}: {param_val:.6f}")
        else:
            print(f"   - {param_name:18s}: {param_val}")
    print("=" * 78)

    # Save best parameters to JSON
    best_config_path = args.output_dir / "best_hyperparameters.json"
    with best_config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_val_accuracy": study.best_value,
                "best_parameters": study.best_params,
                "completed_trials": len(study.trials),
            },
            handle,
            indent=2,
        )
    print(f"\n📁 Saved best config to: {best_config_path.resolve()}")

    # Save complete study trials dataframe to CSV
    trials_df = study.trials_dataframe()
    trials_csv_path = args.output_dir / "tuning_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"📁 Saved trials history to: {trials_csv_path.resolve()}")

    # Print the ready-to-run final command
    bp = study.best_params
    print("\n" + "#" * 78)
    print("🚀 RUN FINAL FULL 50-EPOCH TRAINING WITH WINNING HYPERPARAMETERS:")
    print(f"python train.py \\\n"
          f"  --epochs 50 \\\n"
          f"  --batch-size {args.batch_size} \\\n"
          f"  --accumulation-steps {args.accumulation_steps} \\\n"
          f"  --vit-type {args.vit_type} \\\n"
          f"  --learning-rate {bp.get('learning_rate', 1e-4):.2e} \\\n"
          f"  --weight-decay {bp.get('weight_decay', 1e-2):.2e} \\\n"
          f"  --label-smoothing {bp.get('label_smoothing', 0.10):.2f} \\\n"
          f"  --warmup-epochs {bp.get('warmup_epochs', 3)} \\\n"
          f"  --balance none \\\n"
          f"  --patience 10 \\\n"
          f"  --no-auto-resume")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
