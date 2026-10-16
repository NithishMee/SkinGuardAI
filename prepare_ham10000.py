"""Create reproducible, lesion-level HAM10000 train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument(
        "--strategy",
        choices=("stratified", "lesion"),
        default="lesion",
        help="Use lesion-separated splits. Image-level stratification can leak repeated lesions and inflate scores. Prefer python -m skinguard.prepare for v2.",
    )
    return parser.parse_args()


def validate_fractions(train_size: float, val_size: float, test_size: float) -> None:
    fractions = (train_size, val_size, test_size)
    if any(value <= 0 for value in fractions):
        raise ValueError("All split sizes must be greater than zero.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("train-size, val-size, and test-size must add up to 1.")


def find_images(data_dir: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    duplicate_ids: list[str] = []

    for path in sorted(data_dir.rglob("*.jpg")):
        image_id = path.stem
        if image_id in image_paths:
            duplicate_ids.append(image_id)
        image_paths[image_id] = path.relative_to(data_dir)

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate image IDs found: {preview}")
    return image_paths


def stratified_lesion_split(
    lesions: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, str]:
    train_lesions, remainder = train_test_split(
        lesions,
        train_size=train_size,
        random_state=seed,
        stratify=lesions["dx"],
    )

    relative_val_size = val_size / (val_size + test_size)
    val_lesions, test_lesions = train_test_split(
        remainder,
        train_size=relative_val_size,
        random_state=seed,
        stratify=remainder["dx"],
    )

    split_by_lesion: dict[str, str] = {}
    for split_name, frame in (
        ("train", train_lesions),
        ("val", val_lesions),
        ("test", test_lesions),
    ):
        split_by_lesion.update(dict.fromkeys(frame["lesion_id"], split_name))
    return split_by_lesion


def main() -> None:
    args = parse_args()
    validate_fractions(args.train_size, args.val_size, args.test_size)

    data_dir = args.data_dir.resolve()
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path, dtype={"lesion_id": str, "image_id": str, "dx": str})
    missing_columns = REQUIRED_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if metadata[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("lesion_id, image_id, and dx must not contain missing values.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("The metadata contains duplicate image_id values.")

    labels_per_lesion = metadata.groupby("lesion_id")["dx"].nunique()
    if (labels_per_lesion != 1).any():
        raise ValueError("At least one lesion_id is associated with multiple labels.")

    image_paths = find_images(data_dir)
    missing_images = sorted(set(metadata["image_id"]) - set(image_paths))
    extra_images = sorted(set(image_paths) - set(metadata["image_id"]))
    if missing_images or extra_images:
        raise ValueError(
            f"Image/metadata mismatch: {len(missing_images)} missing images and "
            f"{len(extra_images)} images without metadata."
        )

    prepared = metadata.copy()
    if args.strategy == "stratified":
        # Stratified image-level split: guarantees perfectly balanced class proportions across splits
        train_indices, remainder_indices = train_test_split(
            prepared.index,
            train_size=args.train_size,
            random_state=args.seed,
            stratify=prepared["dx"],
        )
        relative_val_size = args.val_size / (args.val_size + args.test_size)
        val_indices, test_indices = train_test_split(
            remainder_indices,
            train_size=relative_val_size,
            random_state=args.seed,
            stratify=prepared.loc[remainder_indices, "dx"],
        )
        prepared["split"] = ""
        prepared.loc[train_indices, "split"] = "train"
        prepared.loc[val_indices, "split"] = "val"
        prepared.loc[test_indices, "split"] = "test"
    else:
        # Lesion-level split: groups by patient lesion ID
        lesions = metadata[["lesion_id", "dx"]].drop_duplicates().reset_index(drop=True)
        split_by_lesion = stratified_lesion_split(
            lesions,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        prepared["split"] = prepared["lesion_id"].map(split_by_lesion)

    prepared["image_path"] = prepared["image_id"].map(
        lambda image_id: image_paths[image_id].as_posix()
    )

    classes = sorted(prepared["dx"].unique())
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    prepared["target"] = prepared["dx"].map(class_to_idx)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "image_id",
        "lesion_id",
        "dx",
        "target",
        "image_path",
        *[
            column
            for column in ("dx_type", "age", "sex", "localization")
            if column in prepared.columns
        ],
    ]
    for split_name in ("train", "val", "test"):
        split_frame = prepared.loc[prepared["split"] == split_name, output_columns]
        split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)

    train_counts = (
        prepared.loc[prepared["split"] == "train", "dx"]
        .value_counts()
        .reindex(classes)
    )
    train_total = int(train_counts.sum())
    class_weights = {
        class_name: train_total / (len(classes) * int(train_counts[class_name]))
        for class_name in classes
    }
    config = {
        "seed": args.seed,
        "data_dir": "dataset",
        "classes": classes,
        "class_to_idx": class_to_idx,
        "class_weights": class_weights,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    summary = (
        prepared.groupby(["split", "dx"], observed=True)
        .agg(images=("image_id", "size"), lesions=("lesion_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    if args.strategy == "lesion":
        overlap = {
            split: set(prepared.loc[prepared["split"] == split, "lesion_id"])
            for split in ("train", "val", "test")
        }
        assert overlap["train"].isdisjoint(overlap["val"])
        assert overlap["train"].isdisjoint(overlap["test"])
        assert overlap["val"].isdisjoint(overlap["test"])

    print(f"Prepared {len(prepared):,} images from {metadata['lesion_id'].nunique():,} unique lesions.")
    print(f"Saved split files to: {output_dir}")
    print(summary.to_string(index=False))
    print("\nTraining class weights:")
    for class_name in classes:
        print(f"  {class_name:5s}: {class_weights[class_name]:.4f}")


if __name__ == "__main__":
    main()

