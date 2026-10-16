"""Evaluate MedFusionNet and create confusion-matrix and ROC plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from ham10000_data import create_dataloaders
from medfusionnet import MedFusionNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/medfusionnet_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="Use 4-view Test-Time Augmentation.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold below which sample is categorized as 'Unknown Category'.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = MedFusionNet(
        num_classes=len(classes),
        image_size=checkpoint["image_size"],
        vit_type=checkpoint.get("vit_type", "swin"),
        vit_depth=checkpoint.get("vit_depth", 2),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        image_size=checkpoint["image_size"],
    )
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            if args.tta:
                p1 = torch.softmax(model(images), dim=1)
                p2 = torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)
                p3 = torch.softmax(model(torch.flip(images, dims=[-2])), dim=1)
                p4 = torch.softmax(model(torch.flip(images, dims=[-2, -1])), dim=1)
                probs = ((p1 + p2 + p3 + p4) / 4.0).cpu().numpy()
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            targets.extend(labels.tolist())
            predictions.extend(probs.argmax(axis=1).tolist())
            probabilities.extend(probs)

    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    y_binary = label_binarize(y_true, classes=list(range(len(classes))))
    max_confidences = y_prob.max(axis=1)
    confident_mask = max_confidences >= args.threshold
    confident_accuracy = (
        float(accuracy_score(y_true[confident_mask], y_pred[confident_mask]))
        if confident_mask.sum() > 0
        else 0.0
    )

    metrics = {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "threshold": args.threshold,
        "high_confidence_accuracy": confident_accuracy,
        "high_confidence_sample_count": int(confident_mask.sum()),
        "unknown_category_sample_count": int((~confident_mask).sum()),
        "unknown_category_rate": float((~confident_mask).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_auc_ovr": roc_auc_score(y_binary, y_prob, average="macro", multi_class="ovr"),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, cmap="Blues", ax=axis, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    for index, class_name in enumerate(classes):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_binary[:, index], y_prob[:, index])
        class_auc = roc_auc_score(y_binary[:, index], y_prob[:, index])
        axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} ({class_auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="HAM10000 one-vs-rest ROC curves")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "roc_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
