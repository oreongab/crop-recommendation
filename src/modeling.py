"""Shared data loading, evaluation, visualization, and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_dataset" / "model_ready" / "six_feature_prototype_v1"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURES = ["n", "p", "k", "ph", "temperature", "rainfall"]
TARGET = "crop_label"
RANDOM_STATE = 42


def load_frozen_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    integrity = json.loads((DATA_DIR / "split_integrity.json").read_text(encoding="utf-8"))
    validate_frozen_splits(train, validation, test, integrity)
    return train, validation, test, integrity


def load_training_validation_splits() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Validate all splits but expose only Train and Validation to experiment notebooks."""
    train, validation, test, integrity = load_frozen_splits()
    sealed_test_summary = {
        "rows": len(test),
        "crop_classes": int(test[TARGET].nunique()),
        "file_sha256": integrity["file_sha256"]["test.csv"],
        "status": "sealed_not_exposed_to_model_notebook",
    }
    return train, validation, sealed_test_summary, integrity


def validate_frozen_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    integrity: dict,
) -> None:
    required = {"feature_fingerprint", TARGET, *FEATURES}
    for name, frame, expected_rows in [
        ("train", train, 1540),
        ("validation", validation, 330),
        ("test", test, 330),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{name} is missing columns: {sorted(missing)}")
        if len(frame) != expected_rows:
            raise RuntimeError(f"{name} row count changed: {len(frame)}")
        if frame[FEATURES].isna().any().any():
            raise RuntimeError(f"{name} contains missing features")
        if not frame["feature_fingerprint"].is_unique:
            raise RuntimeError(f"{name} contains duplicate feature fingerprints")
        if frame[TARGET].nunique() != 22:
            raise RuntimeError(f"{name} must contain all 22 crop classes")

    groups = {
        "train": set(train["feature_fingerprint"]),
        "validation": set(validation["feature_fingerprint"]),
        "test": set(test["feature_fingerprint"]),
    }
    if groups["train"] & groups["validation"]:
        raise RuntimeError("Train and validation fingerprint leakage detected")
    if groups["train"] & groups["test"]:
        raise RuntimeError("Train and test fingerprint leakage detected")
    if groups["validation"] & groups["test"]:
        raise RuntimeError("Validation and test fingerprint leakage detected")
    if any(integrity["fingerprint_overlap"].values()):
        raise RuntimeError("Frozen split integrity file reports leakage")


def encode_targets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[TARGET])
    y_validation = encoder.transform(validation[TARGET])
    y_test = encoder.transform(test[TARGET])
    if len(encoder.classes_) != 22:
        raise RuntimeError("Expected 22 encoded crop classes")
    return encoder, y_train, y_validation, y_test


def encode_training_validation_targets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[TARGET])
    y_validation = encoder.transform(validation[TARGET])
    if len(encoder.classes_) != 22:
        raise RuntimeError("Expected 22 encoded crop classes")
    return encoder, y_train, y_validation


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[float, pd.DataFrame]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(bin_accuracy - bin_confidence)
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "sample_count": int(mask.sum()),
            }
        )
    return float(ece), pd.DataFrame(rows)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, pd.DataFrame]:
    labels = np.arange(len(encoder.classes_))
    prediction = probabilities.argmax(axis=1)
    ece, calibration = expected_calibration_error(y_true, probabilities)
    metrics = {
        "top_1_accuracy": float(accuracy_score(y_true, prediction)),
        "top_3_accuracy": float(
            top_k_accuracy_score(y_true, probabilities, k=3, labels=labels)
        ),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "multiclass_log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "expected_calibration_error": ece,
    }
    report = pd.DataFrame(
        classification_report(
            y_true,
            prediction,
            labels=labels,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).T
    per_class = report.loc[encoder.classes_].reset_index().rename(columns={"index": "crop_label"})
    matrix = confusion_matrix(y_true, prediction, labels=labels, normalize="true")
    return metrics, per_class, matrix, calibration


def prediction_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    top_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    rows = []
    for index, true_index in enumerate(y_true):
        row = {
            "true_label": encoder.classes_[true_index],
            "predicted_label": encoder.classes_[top_indices[index, 0]],
            "top_1_probability": probabilities[index, top_indices[index, 0]],
        }
        for rank in range(3):
            row[f"top_{rank + 1}_label"] = encoder.classes_[top_indices[index, rank]]
            row[f"top_{rank + 1}_probability"] = probabilities[index, top_indices[index, rank]]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_evaluation_dashboard(
    model_name: str,
    metrics: dict[str, float],
    per_class: pd.DataFrame,
    matrix: np.ndarray,
    calibration: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    encoder: LabelEncoder,
    history: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    if not history.empty and "validation_top_3" in history:
        axes[0, 0].plot(
            history["iteration"], history["validation_top_3"], color="#2f7d32"
        )
        axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("Validation Top 3 Progress")
    axes[0, 0].set_xlabel("Training Iteration")
    axes[0, 0].set_ylabel("Top 3 Accuracy")

    metric_names = ["top_1_accuracy", "top_3_accuracy", "macro_f1", "balanced_accuracy"]
    metric_values = [metrics[name] for name in metric_names]
    sns.barplot(x=metric_values, y=metric_names, color="#2f7d32", ax=axes[0, 1])
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_title("Validation Metrics")
    axes[0, 1].set_xlabel("Score")
    axes[0, 1].set_ylabel("Metric")

    sns.heatmap(
        matrix,
        cmap="YlGn",
        vmin=0,
        vmax=1,
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        cbar_kws={"label": "Normalized Rate"},
        ax=axes[0, 2],
    )
    axes[0, 2].set_title("Normalized Confusion Matrix")
    axes[0, 2].set_xlabel("Predicted Crop")
    axes[0, 2].set_ylabel("True Crop")
    axes[0, 2].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0, 2].tick_params(axis="y", labelsize=7)

    ordered = per_class.sort_values("f1-score")
    sns.barplot(data=ordered, x="f1-score", y="crop_label", color="#4b8bbe", ax=axes[1, 0])
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Per Class F1")
    axes[1, 0].set_xlabel("F1 Score")
    axes[1, 0].set_ylabel("Crop Label")

    axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    if not calibration.empty:
        axes[1, 1].plot(
            calibration["mean_confidence"],
            calibration["accuracy"],
            marker="o",
            color="#c65d21",
            label=model_name,
        )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Confidence Calibration")
    axes[1, 1].set_xlabel("Mean Confidence")
    axes[1, 1].set_ylabel("Observed Accuracy")
    axes[1, 1].legend()

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    confidence_frame = pd.DataFrame(
        {
            "confidence": confidence,
            "prediction_result": np.where(correct, "Correct", "Incorrect"),
        }
    )
    sns.histplot(
        data=confidence_frame,
        x="confidence",
        hue="prediction_result",
        bins=15,
        multiple="stack",
        ax=axes[1, 2],
    )
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_title("Prediction Confidence")
    axes[1, 2].set_xlabel("Maximum Probability")
    axes[1, 2].set_ylabel("Validation Row Count")

    fig.suptitle(model_name + " Validation Evaluation", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def save_evaluation_outputs(
    model_name: str,
    metrics: dict[str, float],
    per_class: pd.DataFrame,
    calibration: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    training_seconds: float,
) -> Path:
    output_dir = ARTIFACT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **metrics,
        "training_seconds": float(training_seconds),
        "evaluation_split": "validation",
        "test_split_used": False,
        "features": FEATURES,
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    per_class.to_csv(output_dir / "validation_per_class.csv", index=False)
    calibration.to_csv(output_dir / "validation_calibration.csv", index=False)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    return output_dir


def split_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "crop_classes": frame[TARGET].nunique(),
                "missing_feature_values": int(frame[FEATURES].isna().sum().sum()),
                "unique_feature_fingerprints": frame["feature_fingerprint"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def training_validation_summary(
    train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name, frame in [("train", train), ("validation", validation)]:
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "crop_classes": frame[TARGET].nunique(),
                "missing_feature_values": int(frame[FEATURES].isna().sum().sum()),
                "unique_feature_fingerprints": frame["feature_fingerprint"].nunique(),
            }
        )
    return pd.DataFrame(rows)
