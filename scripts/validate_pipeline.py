#!/usr/bin/env python3
"""Validate data provenance, the frozen split, and model notebook safety."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_dataset" / "model_ready" / "six_feature_prototype_v1"
METADATA_DIR = ROOT / "processed_dataset" / "metadata"
MODEL_FEATURES = ["n", "p", "k", "ph", "temperature", "rainfall"]
MODEL_NOTEBOOKS = [
    "02_random_forest.ipynb",
    "03_xgboost.ipynb",
    "04_neural_network.ipynb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def notebook_source(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=str(path))
            sources.append(source)
    return payload, "\n".join(sources)


def main() -> None:
    checks: dict[str, object] = {}

    manifest = pd.read_csv(ROOT / "processed_dataset" / "manifests" / "raw_file_manifest.csv")
    raw_hash_failures = []
    for row in manifest.itertuples(index=False):
        path = ROOT / "dataset" / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            raw_hash_failures.append(row.source_file)
    check(not raw_hash_failures, f"Raw file hash mismatch: {raw_hash_failures}")
    checks["raw_source_files"] = int(len(manifest))
    checks["raw_hashes_match_manifest"] = True

    cleaning = pd.read_csv(METADATA_DIR / "cleaning_report.csv")
    check(cleaning["row_reconciliation_passed"].all(), "A cleaning reconciliation failed")
    check(not cleaning["value_imputation_applied"].any(), "Unexpected imputation detected")
    check(not cleaning["unit_conversion_applied"].any(), "Unexpected unit conversion detected")
    check(not cleaning["outlier_deletion_applied"].any(), "Unexpected outlier deletion detected")
    check(int(cleaning["rows"].sum()) == 300054, "Unexpected raw row total")
    check(int(cleaning["cleaned_rows"].sum()) == 296006, "Unexpected cleaned row total")
    checks["exact_duplicates_removed_from_cleaned_copies"] = int(
        (cleaning["rows"] - cleaning["cleaned_rows"]).sum()
    )
    checks["cleaning_reconciliation_passed"] = True

    gate = pd.read_csv(METADATA_DIR / "data_quality_gate.csv")
    selected_gate = gate.loc[gate["source_file"] == "Crop_Recommendation.csv"].iloc[0]
    check(bool(selected_gate["approved_for_six_feature_prototype"]), "Training source not approved")
    check(not bool(selected_gate["approved_for_full_eight_feature_training"]), "Eight-feature gate changed")

    units = pd.read_csv(METADATA_DIR / "unit_review.csv")
    selected_units = units.loc[
        (units["source_file"] == "Crop_Recommendation.csv")
        & units["canonical_feature"].isin(MODEL_FEATURES)
    ]
    check(set(selected_units["canonical_feature"]) == set(MODEL_FEATURES), "Unit rows are incomplete")
    check(
        selected_units["unit_status"].eq("verified_from_dataset_card").all(),
        "A model feature unit is not verified",
    )
    check(not selected_units["conversion_applied"].any(), "Unexpected model unit conversion")
    checks["verified_model_feature_units"] = sorted(selected_units["canonical_feature"].tolist())
    checks["soil_type_model_status"] = "not_available_not_used"
    checks["soil_moisture_model_status"] = "not_available_not_used"

    mappings = pd.read_csv(METADATA_DIR / "crop_label_mapping.csv")
    selected_mappings = mappings.loc[mappings["source_file"] == "Crop_Recommendation.csv"]
    check(selected_mappings["crop_mapping_status"].eq("approved").all(), "Unapproved selected-source mapping")
    check(selected_mappings["is_specific_crop_label"].all(), "Non-specific selected-source label")
    check(
        set(selected_mappings["crop_mapping_rule"]).issubset({"identity", "spacing_variant"}),
        "Unexpected selected-source crop mapping rule",
    )
    checks["selected_source_crop_mapping_rules"] = sorted(
        selected_mappings["crop_mapping_rule"].unique().tolist()
    )

    integrity = json.loads((DATA_DIR / "split_integrity.json").read_text(encoding="utf-8"))
    for filename, expected_hash in integrity["file_sha256"].items():
        check(sha256(DATA_DIR / filename) == expected_hash, f"Frozen artifact changed: {filename}")

    splits = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ["train", "validation", "test"]}
    expected_rows = {"train": 1540, "validation": 330, "test": 330}
    expected_class_rows = {"train": 70, "validation": 15, "test": 15}
    fingerprints = {}
    for name, frame in splits.items():
        check(len(frame) == expected_rows[name], f"Unexpected {name} row count")
        check(frame[MODEL_FEATURES].notna().all().all(), f"Missing {name} feature")
        check(frame["crop_label"].nunique() == 22, f"Missing crop class in {name}")
        check(frame["feature_fingerprint"].is_unique, f"Duplicate fingerprint in {name}")
        check(
            frame.groupby("crop_label").size().eq(expected_class_rows[name]).all(),
            f"Unexpected class support in {name}",
        )
        fingerprints[name] = set(frame["feature_fingerprint"])
    check(not fingerprints["train"] & fingerprints["validation"], "Train Validation leakage")
    check(not fingerprints["train"] & fingerprints["test"], "Train Test leakage")
    check(not fingerprints["validation"] & fingerprints["test"], "Validation Test leakage")
    checks["frozen_split_rows"] = expected_rows
    checks["crop_classes_per_split"] = 22
    checks["feature_fingerprint_overlap"] = 0
    checks["test_sha256"] = integrity["file_sha256"]["test.csv"]

    test1, _ = notebook_source(ROOT / "test1.ipynb")
    test1_code = [cell for cell in test1["cells"] if cell["cell_type"] == "code"]
    test1_errors = [
        output
        for cell in test1_code
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    check(all(cell.get("execution_count") is not None for cell in test1_code), "test1 is not fully executed")
    check(not test1_errors, "test1 contains execution errors")
    checks["test1"] = {
        "code_cells": len(test1_code),
        "executed_code_cells": len(test1_code),
        "execution_errors": 0,
        "png_visualizations": sum(
            "image/png" in output.get("data", {})
            for cell in test1_code
            for output in cell.get("outputs", [])
        ),
    }

    notebook_checks = {}
    for filename in MODEL_NOTEBOOKS:
        payload, source = notebook_source(ROOT / filename)
        check("load_training_validation_splits" in source, f"{filename} does not use sealed loader")
        check("tqdm" in source, f"{filename} has no progress bar")
        check("training_seconds" in source, f"{filename} has no elapsed time")
        check("plot_evaluation_dashboard" in source, f"{filename} has no shared dashboard")
        check("X_test" not in source and "y_test" not in source, f"{filename} exposes Test arrays")
        check("load_frozen_splits" not in source, f"{filename} loads Test directly")
        notebook_checks[filename] = {
            "code_cells_syntax_valid": sum(c["cell_type"] == "code" for c in payload["cells"]),
            "progress_bar": True,
            "elapsed_time": True,
            "shared_validation_dashboard": True,
            "test_arrays_exposed": False,
        }
    checks["model_notebooks"] = notebook_checks
    checks["overall_status"] = "pass_for_six_feature_prototype_model_experiments"
    checks["full_eight_feature_production_status"] = "blocked_pending_trusted_soil_data"

    output_path = METADATA_DIR / "final_pipeline_audit.json"
    output_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"Audit report written to {output_path}")


if __name__ == "__main__":
    main()
