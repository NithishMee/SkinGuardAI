"""Train state-of-the-art MedFusionNet (ConvNeXt-Tiny + Swin Transformer) on HAM10000."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ham10000_data import (
    compute_effective_class_weights,
    create_dataloaders,
    get_class_weight_tensor,
    load_split_config,
)
from medfusionnet import MedFusionNet, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384, help="Input resolution (384x384 is optimal for dermoscopic fine features on GPU).")
    parser.add_argument(
        "--vit-type",
        choices=("swin", "compact"),
        default="swin",
        help="Swin-Transformer Tiny (pretrained, recommended) or compact 2-layer ViT from scratch.",
    )
    parser.add_argument("--vit-depth", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Classification head learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=2e-5, help="Backbone fine-tuning learning rate (preserves ImageNet features).")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--scheduler", choices=("cosine", "inverse"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lr-decay", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("focal", "ce"), default="focal", help="Multi-class Focal Loss focuses on hard borderline lesions.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--balance",
        choices=("balanced", "smooth", "none", "weighted_loss", "weighted_sampler"),
        default="balanced",
        help="Weighted sampling increases minority exposure; it does not guarantee every class in every batch.",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.20,
        help="Mixup alpha parameter to regularize decision boundaries between borderline classes (0 to disable).",
    )
    parser.add_argument(
        "--augmentation",
        choices=("advanced", "standard"),
        default="advanced",
        help="Use advanced dermatological augmentations (RandomErasing, Affine, Jitter) to prevent overfitting.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-view Test-Time Augmentation (TTA) during validation and testing.",
    )
    parser.add_argument(
        "--best-metric",
        choices=("accuracy", "macro_f1"),
        default="accuracy",
        help="Criterion to select the best checkpoint.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement before triggering early stopping (0 to disable).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum change in monitored score to qualify as an improvement.",
    )
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from checkpoints/medfusionnet_last.pt if present.",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/medfusionnet_best.pt"))
    parser.add_argument("--last-checkpoint", type=Path, default=Path("checkpoints/medfusionnet_last.pt"))
    parser.add_argument("--history", type=Path, default=Path("outputs/training_history.csv"))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_epochs: int,
    warmup_epochs: int = 3,
    lr_decay: float = 1e-3,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    if scheduler_type == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + float(np.cos(np.pi * progress)))
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == "inverse":
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + lr_decay * epoch))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with class weights and label smoothing."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    mixup: float = 0.0,
    use_tta: bool = False,
) -> tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    if training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if training else torch.inference_mode()
    desc = "  [Training]  " if training else "  [Validation]"
    pbar = tqdm(
        loader,
        desc=desc,
        leave=False,
        ncols=78,
        bar_format="{desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    with context:
        for batch_index, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_mixup = training and mixup > 0.0 and random.random() < 0.6
            if use_mixup:
                lam = float(np.random.beta(mixup, mixup))
                lam = max(lam, 1.0 - lam)
                index = torch.randperm(images.size(0), device=device)
                mixed_images = lam * images + (1.0 - lam) * images[index]
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    logits = model(mixed_images)
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[index])
            else:
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)

            if training:
                if scaler is None:
                    raise RuntimeError("A gradient scaler is required during training.")
                scaler.scale(loss / accumulation_steps).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.cpu().tolist())

            if use_tta and not training:
                # 4-view Test-Time Augmentation
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    p1 = torch.softmax(logits, dim=1)
                    p2 = torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)
                    p3 = torch.softmax(model(torch.flip(images, dims=[-2])), dim=1)
                    p4 = torch.softmax(model(torch.flip(images, dims=[-2, -1])), dim=1)
                    prob = (p1 + p2 + p3 + p4) / 4.0
                    batch_preds = prob.argmax(dim=1).cpu().tolist()
            else:
                batch_preds = logits.argmax(dim=1).cpu().tolist()

            predictions.extend(batch_preds)

    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(targets, predictions)

    return (
        total_loss / len(loader.dataset),
        acc,
        f1,
        bacc,
    )


HISTORY_FIELDNAMES = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_balanced_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
]


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    classes: list[str],
    epoch: int,
    best_score: float,
    history: list[dict[str, float]],
    parameter_count: int,
    patience_counter: int = 0,
    best_epoch: int = 1,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "classes": classes,
        "image_size": args.image_size,
        "vit_type": getattr(args, "vit_type", "swin"),
        "vit_depth": args.vit_depth,
        "balance": args.balance,
        "parameter_count": parameter_count,
        "epoch": epoch,
        "best_val_score": best_score,
        "best_val_macro_f1": best_score,
        "patience_counter": patience_counter,
        "best_epoch": best_epoch,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps must be at least 1.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    config = load_split_config(args.splits_dir)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        balance=args.balance,
        augmentation=args.augmentation,
    )

    resume_path = args.resume
    if resume_path is None and args.auto_resume and args.last_checkpoint.is_file():
        resume_path = args.last_checkpoint

    model = MedFusionNet(
        num_classes=len(config["classes"]),
        image_size=args.image_size,
        vit_type=args.vit_type,
        vit_depth=args.vit_depth,
        pretrained=not args.no_pretrained and resume_path is None,
    ).to(device)

    if args.balance in ("balanced", "weighted_loss"):
        class_weights = compute_effective_class_weights(args.splits_dir, beta=0.999, device=device)
    else:
        class_weights = None

    if args.loss == "focal":
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    param_groups = model.get_parameter_groups(
        backbone_lr=args.backbone_lr,
        head_lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.optimizer == "adamw":
        optimizer = AdamW(param_groups)
    else:
        optimizer = Adam(param_groups)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr_decay=args.lr_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    parameter_count = count_parameters(model)
    start_epoch = 1
    best_score = -1.0
    best_epoch = 1
    patience_counter = 0
    history: list[dict[str, float]] = []

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        if resume_checkpoint["classes"] != config["classes"]:
            raise ValueError("Checkpoint classes do not match the prepared dataset.")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_score = float(resume_checkpoint.get("best_val_score", resume_checkpoint.get("best_val_macro_f1", -1.0)))
        patience_counter = int(resume_checkpoint.get("patience_counter", 0))
        best_epoch = int(resume_checkpoint.get("best_epoch", max(1, start_epoch - 1)))
        history = list(resume_checkpoint.get("history", []))
        print(f"🔄 Resumed from epoch {start_epoch - 1}: {resume_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  SkinGuardAI High-Performance Training Pipeline (MedFusionNet)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Architecture: ConvNeXt-Tiny + {args.vit_type.upper()} ({parameter_count / 1e6:.2f}M parameters)")
    print(f"  Image Resolution: {args.image_size}x{args.image_size} (Aspect-Ratio Preserved)")
    print(f"  Optimizer: {args.optimizer.upper()} (Head LR={args.learning_rate:.2e}, Backbone LR={args.backbone_lr:.2e}, WD={args.weight_decay})")
    print(f"  Scheduler: {args.scheduler.upper()} (warmup={args.warmup_epochs} epochs)")
    print(f"  Loss: {args.loss.upper()} (gamma={args.focal_gamma if args.loss == 'focal' else 'N/A'}, smoothing={args.label_smoothing}, balance={args.balance})")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulation_steps})")
    print(f"  Evaluation TTA: {'Enabled (4 views)' if args.tta else 'Disabled'}")
    print(f"  Early Stopping: {'Disabled' if args.patience == 0 else f'Patience = {args.patience} epochs'}")
    print(f"  Best model selection metric: {args.best_metric.upper()}")
    print("=" * 78)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc, train_f1, train_bacc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                accumulation_steps=args.accumulation_steps,
                grad_clip=args.grad_clip,
                mixup=args.mixup,
                use_tta=False,
            )
            val_loss, val_acc, val_f1, val_bacc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                mixup=0.0,
                use_tta=args.tta,
            )
            scheduler.step()
            epoch_time = time.time() - epoch_start
            time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"

            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "train_balanced_accuracy": train_bacc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_balanced_accuracy": val_bacc,
            }
            history.append(row)
            write_history(args.history, history)

            current_score = val_acc if args.best_metric == "accuracy" else val_f1
            improved = current_score > (best_score + args.min_delta)
            if improved:
                best_score = current_score
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                config["classes"],
                epoch,
                best_score,
                history,
                parameter_count,
                patience_counter=patience_counter,
                best_epoch=best_epoch,
            )
            torch.save(payload, args.last_checkpoint)

            print("─" * 78)
            print(
                f"Epoch {epoch:03d}/{args.epochs}  |  LR: {optimizer.param_groups[0]['lr']:.2e}  |  Duration: {time_str}"
            )
            print(
                f"  Train -> Loss: {train_loss:.4f}  |  Accuracy: {train_acc * 100:.2f}%  |  Macro-F1: {train_f1:.4f}  |  Bal-Acc: {train_bacc * 100:.2f}%"
            )
            print(
                f"  Val   -> Loss: {val_loss:.4f}  |  Accuracy: {val_acc * 100:.2f}%  |  Macro-F1: {val_f1:.4f}  |  Bal-Acc: {val_bacc * 100:.2f}%"
            )
            metric_name = "Val Accuracy" if args.best_metric == "accuracy" else "Val Macro-F1"
            if improved:
                torch.save(payload, args.output)
                metric_val = f"{val_acc * 100:.2f}%" if args.best_metric == "accuracy" else f"{val_f1:.4f}"
                print(f"  ⭐ [NEW BEST MODEL] Saved checkpoint to {args.output.name} ({metric_name}: {metric_val})")
            else:
                if args.patience > 0:
                    best_val_str = f"{best_score * 100:.2f}%" if args.best_metric == "accuracy" else f"{best_score:.4f}"
                    print(f"  ⏳ No improvement for {patience_counter}/{args.patience} epochs (Best {metric_name}: {best_val_str} at Epoch {best_epoch:03d})")
            print("─" * 78)

            if args.patience > 0 and patience_counter >= args.patience:
                print("\n" + "=" * 78)
                print(f"  🛑 [EARLY STOPPING TRIGGERED]")
                print(f"  Validation {metric_name} did not improve for {args.patience} consecutive epochs.")
                print(f"  Best model was obtained at Epoch {best_epoch:03d} ({metric_name}: {best_score * 100 if args.best_metric == 'accuracy' else best_score:.2f}%).")
                print("=" * 78 + "\n")
                break

    except KeyboardInterrupt:
        print("\n" + "=" * 65)
        print("⏸️  [PAUSED] Training interrupted by user (Ctrl+C).")
        if args.last_checkpoint.is_file():
            print(f"📁 Checkpoint saved in: {args.last_checkpoint.resolve()}")
            print("🚀 To resume where you left off, simply run:")
            print("   python train.py")
        else:
            print("No complete epoch was saved yet.")
        print("=" * 65 + "\n")
        return

    if not args.output.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {args.output}")
    best_checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print("\n" + "=" * 78)
    print("  Evaluating Best Model on the Unseen Test Set...")
    test_loss, test_acc, test_f1, test_bacc = run_epoch(
        model, test_loader, criterion, device, amp_enabled=amp_enabled, mixup=0.0, use_tta=args.tta
    )
    best_checkpoint["test_loss"] = test_loss
    best_checkpoint["test_accuracy"] = test_acc
    best_checkpoint["test_macro_f1"] = test_f1
    best_checkpoint["test_balanced_accuracy"] = test_bacc
    torch.save(best_checkpoint, args.output)
    print(
        f"  Test Loss: {test_loss:.4f}  |  Accuracy: {test_acc * 100:.2f}%  |  "
        f"Macro-F1: {test_f1:.4f}  |  Balanced Accuracy: {test_bacc * 100:.2f}%"
    )
    print(f"  Saved final model to: {args.output.resolve()}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
