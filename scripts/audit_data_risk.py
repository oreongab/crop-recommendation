#!/usr/bin/env python3
"""Run non-production diagnostics for dataset shift, weak signal, and leakage risk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "processed_dataset" / "cleaned_by_source"
METADATA_DIR = ROOT / "processed_dataset" / "metadata"
RANDOM_STATE = 42

REQUIRED_FEATURES = [
    "soil_type",
    "ph",
    "n",
    "p",
    "k",
    "soil_moisture",
    "temperature",
    "rainfall",
]


def build_model(features: list[str], frame: pd.DataFrame) -> object:
    categorical = [feature for feature in features if frame[feature].dtype == "object"]
    numeric = [feature for feature in features if feature not in categorical]
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return make_pipeline(
        transformer,
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )


def stratified_sample(frame: pd.DataFrame, target: str, maximum_rows: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame
    fraction = maximum_rows / len(frame)
    parts = []
    for _, group in frame.groupby(target):
        sample_size = max(1, int(round(len(group) * fraction)))
        parts.append(group.sample(min(sample_size, len(group)), random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True)


def signal_audit(source_file: Path) -> dict:
    frame = pd.read_csv(source_file, low_memory=False)
    source_name = str(frame["source_file"].iloc[0])
    features = [feature for feature in REQUIRED_FEATURES if frame[feature].notna().any()]
    usable = frame.loc[
        frame["is_specific_crop_label"] & frame[features].notna().all(axis=1),
        [
            "crop_label",
            "feature_fingerprint",
            "has_conflicting_crop_labels_for_features",
            *features,
        ],
    ].copy()

    conflicting_rows = int(usable["has_conflicting_crop_labels_for_features"].sum())
    conflicting_groups = int(
        usable.loc[
            usable["has_conflicting_crop_labels_for_features"], "feature_fingerprint"
        ].nunique()
    )
    # Conflicting labels for identical inputs cannot be resolved safely in a signal diagnostic.
    usable = usable.loc[~usable["has_conflicting_crop_labels_for_features"]]
    usable = usable.drop_duplicates("feature_fingerprint")
    class_counts = usable["crop_label"].value_counts()
    allowed_classes = class_counts[class_counts >= 5].index
    usable = usable[usable["crop_label"].isin(allowed_classes)]
    usable = stratified_sample(usable, "crop_label", maximum_rows=80000)

    result = {
        "source_file": source_name,
        "features_used": "|".join(features),
        "unique_fingerprint_rows": len(usable),
        "classes_evaluated": int(usable["crop_label"].nunique()),
        "conflicting_input_rows_excluded": conflicting_rows,
        "conflicting_input_groups_excluded": conflicting_groups,
    }
    if len(usable) < 100 or usable["crop_label"].nunique() < 2:
        return {
            **result,
            "majority_baseline": np.nan,
            "top_three_frequency_baseline": np.nan,
            "top_one_accuracy": np.nan,
            "top_three_accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "signal_decision": "insufficient_rows_for_diagnostic",
        }

    train, test = train_test_split(
        usable,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=usable["crop_label"],
    )
    model = build_model(features, usable)
    model.fit(train[features], train["crop_label"])
    prediction = model.predict(test[features])
    probability = model.predict_proba(test[features])
    classes = model.named_steps["randomforestclassifier"].classes_
    test_frequency = test["crop_label"].value_counts(normalize=True)
    majority = float(test_frequency.iloc[0])
    top_three_baseline = float(test_frequency.head(min(3, len(test_frequency))).sum())
    top_one = float(accuracy_score(test["crop_label"], prediction))
    top_three = float(
        top_k_accuracy_score(
            test["crop_label"],
            probability,
            k=min(3, len(classes)),
            labels=classes,
        )
    )
    balanced = float(balanced_accuracy_score(test["crop_label"], prediction))

    if top_one <= majority + 0.02:
        decision = "insufficient_crop_signal"
    elif top_one >= 0.95:
        decision = "possible_synthetic_or_target_encoded_pattern"
    else:
        decision = "signal_present_requires_external_validation"

    return {
        **result,
        "majority_baseline": majority,
        "top_three_frequency_baseline": top_three_baseline,
        "top_one_accuracy": top_one,
        "top_three_accuracy": top_three,
        "balanced_accuracy": balanced,
        "signal_decision": decision,
    }


def source_shift_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    shared = ["n", "p", "k", "ph", "temperature", "rainfall"]
    parts = []
    for source, frame in frames.items():
        if not frame[shared].notna().all(axis=1).any():
            continue
        usable = frame.loc[frame[shared].notna().all(axis=1), shared].drop_duplicates()
        if len(usable) > 3000:
            usable = usable.sample(3000, random_state=RANDOM_STATE)
        usable["source_file"] = source
        parts.append(usable)

    combined = pd.concat(parts, ignore_index=True)
    train, test = train_test_split(
        combined,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=combined["source_file"],
    )
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[shared], train["source_file"])
    prediction = model.predict(test[shared])
    majority = float(test["source_file"].value_counts(normalize=True).iloc[0])
    top_one = float(accuracy_score(test["source_file"], prediction))
    balanced = float(balanced_accuracy_score(test["source_file"], prediction))
    return pd.DataFrame(
        [
            {
                "features_used": "|".join(shared),
                "rows_evaluated": len(combined),
                "sources_evaluated": combined["source_file"].nunique(),
                "majority_baseline": majority,
                "source_top_one_accuracy": top_one,
                "source_balanced_accuracy": balanced,
                "dataset_shift_detected": bool(balanced >= 0.60),
                "decision": "do_not_naively_merge_sources" if balanced >= 0.60 else "review_shift_manually",
            }
        ]
    )


def discretized_nmi(feature: pd.Series, target: pd.Series) -> float:
    if feature.dtype == "object":
        values = feature.astype(str)
    else:
        unique = feature.nunique(dropna=True)
        if unique <= 1:
            return 0.0
        bins = min(20, unique)
        values = pd.qcut(feature, q=bins, duplicates="drop").astype(str)
    return float(normalized_mutual_info_score(target.astype(str), values))


def leakage_report(
    frames: dict[str, pd.DataFrame], signal_results: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    signal_lookup = signal_results.set_index("source_file")["signal_decision"].to_dict()
    for source, frame in frames.items():
        specific = frame[frame["is_specific_crop_label"]].copy()
        features = [feature for feature in REQUIRED_FEATURES if specific[feature].notna().any()]
        nmi_values = {}
        for feature in features:
            valid = specific[[feature, "crop_label"]].dropna()
            nmi_values[feature] = discretized_nmi(valid[feature], valid["crop_label"])
        max_feature = max(nmi_values, key=nmi_values.get) if nmi_values else ""
        max_nmi = nmi_values.get(max_feature, np.nan)
        duplicate_count = int(frame["feature_fingerprint"].duplicated().sum())
        conflict_rows = int(frame["has_conflicting_crop_labels_for_features"].sum())
        conflict_groups = int(
            frame.loc[
                frame["has_conflicting_crop_labels_for_features"], "feature_fingerprint"
            ].nunique()
        )
        rows.append(
            {
                "source_file": source,
                "cleaned_rows": len(frame),
                "specific_crop_rows": len(specific),
                "model_fingerprint_duplicates_after_first": duplicate_count,
                "model_fingerprint_duplicate_rate": duplicate_count / len(frame),
                "conflicting_input_label_rows": conflict_rows,
                "conflicting_input_label_groups": conflict_groups,
                "maximum_single_feature_nmi": max_nmi,
                "maximum_nmi_feature": max_feature,
                "signal_decision": signal_lookup.get(source, "not_evaluated"),
                "group_split_required": bool(duplicate_count > 0),
                "conflicting_labels_require_exclusion_or_resolution": bool(conflict_groups > 0),
                "single_feature_leakage_review_required": bool(max_nmi >= 0.90),
                "approved_for_training_after_audit": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frames = {}
    for path in sorted(CLEANED_DIR.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        frames[str(frame["source_file"].iloc[0])] = frame

    signal_results = pd.DataFrame([signal_audit(path) for path in sorted(CLEANED_DIR.glob("*.csv"))])
    signal_results.to_csv(METADATA_DIR / "model_signal_audit.csv", index=False)

    shift = source_shift_audit(frames)
    shift.to_csv(METADATA_DIR / "source_shift_audit.csv", index=False)

    leakage = leakage_report(frames, signal_results)
    leakage.to_csv(METADATA_DIR / "leakage_risk_report.csv", index=False)

    summary = {
        "sources_audited": len(frames),
        "naive_source_merge_approved": False,
        "dataset_shift_detected": bool(shift["dataset_shift_detected"].iloc[0]),
        "training_source_approved": False,
    }
    (METADATA_DIR / "risk_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
