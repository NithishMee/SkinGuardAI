"""PyTorch Dataset, transforms, and DataLoaders for prepared HAM10000 splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 288,
    augmentation: Literal["standard", "advanced"] = "advanced",
) -> tuple[transforms.Compose, transforms.Compose]:
    # Scale shorter edge to 1.15 * image_size so aspect ratio is preserved without squashing circular lesions
    scale_size = int(image_size * 1.15)
    if augmentation == "advanced":
        train_transform = transforms.Compose(
            [
                transforms.Resize(scale_size, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomAffine(
                    degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)
                ),
                transforms.ColorJitter(
                    brightness=0.20, contrast=0.20, saturation=0.20, hue=0.015
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), value="random"),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(scale_size, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.10, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(scale_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: Path, data_dir: Path, transform: transforms.Compose) -> None:
        self.samples = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.samples.iloc[index]
        image_path = self.data_dir / row["image_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, int(row["target"])


def load_split_config(splits_dir: Path) -> dict:
    with (Path(splits_dir) / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def get_class_weight_tensor(splits_dir: Path, device: torch.device | str) -> torch.Tensor:
    config = load_split_config(splits_dir)
    weights = [config["class_weights"][name] for name in config["classes"]]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_effective_class_weights(
    splits_dir: Path, beta: float = 0.999, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Compute Cui et al. Class-Balanced weights to heavily penalize errors on minority lesions."""
    df = pd.read_csv(Path(splits_dir) / "train.csv")
    counts = df["target"].value_counts().sort_index().to_numpy()
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_dataloaders(
    data_dir: Path = Path("dataset"),
    splits_dir: Path = Path("splits"),
    batch_size: int = 32,
    image_size: int = 288,
    num_workers: int = 0,
    balance: Literal["none", "smooth", "balanced", "weighted_loss", "weighted_sampler"] = "balanced",
    augmentation: Literal["standard", "advanced"] = "advanced",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders. Weighted sampling increases minority exposure but cannot guarantee batch composition."""
    train_transform, evaluation_transform = build_transforms(image_size, augmentation=augmentation)
    train_dataset = HAM10000Dataset(splits_dir / "train.csv", data_dir, train_transform)
    val_dataset = HAM10000Dataset(splits_dir / "val.csv", data_dir, evaluation_transform)
    test_dataset = HAM10000Dataset(splits_dir / "test.csv", data_dir, evaluation_transform)

    sampler = None
    shuffle = True
    if balance == "weighted_sampler":
        class_counts = train_dataset.samples["target"].value_counts()
        sample_weights = train_dataset.samples["target"].map(
            lambda target: 1.0 / class_counts[int(target)]
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    elif balance == "balanced":
        class_counts = train_dataset.samples["target"].value_counts()
        sample_weights = train_dataset.samples["target"].map(
            lambda target: (1.0 / class_counts[int(target)]) ** 0.65
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    elif balance == "smooth":
        class_counts = train_dataset.samples["target"].value_counts()
        sample_weights = train_dataset.samples["target"].map(
            lambda target: (1.0 / class_counts[int(target)]) ** 0.35
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    elif balance not in ("none", "weighted_loss"):
        raise ValueError("balance must be 'none', 'smooth', 'balanced', 'weighted_loss', or 'weighted_sampler'.")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.device_count() > 0,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader
