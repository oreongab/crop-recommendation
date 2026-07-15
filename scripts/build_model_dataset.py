#!/usr/bin/env python3
"""Create a frozen, leakage-safe split for the approved six-feature prototype."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "processed_dataset" / "cleaned_by_source" / "crop_recommendation.csv"
OUTPUT_DIR = ROOT / "processed_dataset" / "model_ready" / "six_feature_prototype_v1"
RANDOM_STATE = 42
FEATURES = ["n", "p", "k", "ph", "temperature", "rainfall"]
TARGET = "crop_label"
PROVENANCE = ["source_file", "source_row_id", "feature_fingerprint"]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE_PATH, low_memory=False)

    eligible = frame.loc[
        frame["is_specific_crop_label"]
        & ~frame["has_conflicting_crop_labels_for_features"]
        & frame[FEATURES].notna().all(axis=1)
        & ~frame["has_negative_required_numeric"]
        & ~frame["ph_outside_physical_range"],
        [*PROVENANCE, *FEATURES, TARGET],
    ].copy()

    if len(eligible) != 2200:
        raise RuntimeError(f"Expected 2200 eligible rows, found {len(eligible)}")
    if eligible[TARGET].nunique() != 22:
        raise RuntimeError("Expected 22 crop classes")
    if not eligible["feature_fingerprint"].is_unique:
        raise RuntimeError("Feature fingerprints must be unique for this prototype source")

    train, remaining = train_test_split(
        eligible,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=eligible[TARGET],
    )
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remaining[TARGET],
    )

    split_frames = {
        "train": train.copy(),
        "validation": validation.copy(),
        "test": test.copy(),
    }
    ordered_columns = [*PROVENANCE, *FEATURES, TARGET, "split"]
    for split_name, split_frame in split_frames.items():
        split_frame["split"] = split_name
        split_frame = split_frame.sort_values([TARGET, "source_row_id"]).reset_index(drop=True)
        split_frames[split_name] = split_frame[ordered_columns]
        split_frames[split_name].to_csv(OUTPUT_DIR / f"{split_name}.csv", index=False)

    combined = pd.concat(split_frames.values(), ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_splits.csv", index=False)
    combined[[*PROVENANCE, TARGET, "split"]].to_csv(
        OUTPUT_DIR / "split_manifest.csv", index=False
    )

    split_sets = {
        name: set(part["feature_fingerprint"]) for name, part in split_frames.items()
    }
    overlap = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    class_counts = (
        combined.groupby([TARGET, "split"])
        .size()
        .rename("row_count")
        .reset_index()
    )
    class_counts.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    feature_ranges = []
    for feature in FEATURES:
        for split_name, split_frame in split_frames.items():
            feature_ranges.append(
                {
                    "feature": feature,
                    "split": split_name,
                    "minimum": split_frame[feature].min(),
                    "maximum": split_frame[feature].max(),
                    "mean": split_frame[feature].mean(),
                    "median": split_frame[feature].median(),
                }
            )
    pd.DataFrame(feature_ranges).to_csv(OUTPUT_DIR / "feature_ranges.csv", index=False)

    integrity = {
        "dataset_version": "six_feature_prototype_v1",
        "source_file": "Crop_Recommendation.csv",
        "features": FEATURES,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "row_counts": {name: len(part) for name, part in split_frames.items()},
        "class_counts_per_split": {
            name: int(part[TARGET].nunique()) for name, part in split_frames.items()
        },
        "fingerprint_overlap": overlap,
        "conflicting_label_rows": 0,
        "missing_feature_values": int(combined[FEATURES].isna().sum().sum()),
        "full_eight_feature_model": False,
        "soil_type_used_by_model": False,
        "soil_moisture_used_by_model": False
    }

    if any(overlap.values()):
        raise RuntimeError(f"Feature fingerprint leakage detected: {overlap}")
    if integrity["missing_feature_values"]:
        raise RuntimeError("Missing feature values detected")
    if set(integrity["row_counts"].values()) != {1540, 330}:
        raise RuntimeError(f"Unexpected split sizes: {integrity['row_counts']}")

    file_hashes = {}
    for filename in [
        "train.csv",
        "validation.csv",
        "test.csv",
        "all_splits.csv",
        "split_manifest.csv",
        "class_distribution.csv",
        "feature_ranges.csv",
    ]:
        file_hashes[filename] = hash_file(OUTPUT_DIR / filename)
    integrity["file_sha256"] = file_hashes

    (OUTPUT_DIR / "split_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
