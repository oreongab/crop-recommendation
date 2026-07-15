#!/usr/bin/env python3
"""Build the data preparation and EDA notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "test1.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


cells = [
    markdown(
        """
# Crop Recommendation Data Preparation

This notebook audits the available crop datasets before model training.

The required model inputs are soil type, pH, nitrogen, phosphorus, potassium, soil moisture, temperature, and rainfall. The target is crop label.

Data safety rules used in this notebook:

- Raw files remain unchanged.
- Missing values are not imputed.
- Units are not converted unless the source unit is verified.
- Atmospheric humidity is not treated as soil moisture.
- Soil color is not treated as soil type.
- Precipitation near atmospheric pressure values is not treated as rainfall.
- Only approved crop spelling variants and established synonyms are mapped.
- Exact raw duplicates are removed only from cleaned copies and recorded in a manifest.
- No train validation test split is created in this phase.
"""
    ),
    markdown("## 1 Setup"),
    code(
        """
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "processed_dataset").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

PROCESSED_DIR = PROJECT_ROOT / "processed_dataset"
BY_SOURCE_DIR = PROCESSED_DIR / "by_source"
CLEANED_BY_SOURCE_DIR = PROCESSED_DIR / "cleaned_by_source"
METADATA_DIR = PROCESSED_DIR / "metadata"

REQUIRED_FEATURES = [
    "soil_type", "ph", "n", "p", "k",
    "soil_moisture", "temperature", "rainfall"
]

sns.set_theme(style="whitegrid", context="notebook")
CHART_COLOR = "#2f7d32"
PALETTE = "YlGn"
"""
    ),
    markdown("## 2 Dataset Inventory"),
    code(
        """
inventory = pd.read_csv(METADATA_DIR / "dataset_inventory.csv")
inventory[
    [
        "source_file", "rows", "cleaned_rows", "columns_original", "crop_classes",
        "raw_duplicate_rows_after_first", "complete_required_rows",
        "quality_tier", "training_status"
    ]
].sort_values("rows", ascending=False)
"""
    ),
    code(
        """
plot_data = inventory.sort_values("rows", ascending=True)
plt.figure(figsize=(11, 6))
sns.barplot(data=plot_data, x="rows", y="source_file", color=CHART_COLOR)
plt.title("Row Count by Dataset")
plt.xlabel("Row Count")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 3 Required Feature Coverage"),
    code(
        """
coverage = pd.read_csv(METADATA_DIR / "required_feature_coverage.csv")
coverage_matrix = coverage.pivot(
    index="source_file", columns="feature", values="coverage_rate"
).reindex(columns=REQUIRED_FEATURES)

plt.figure(figsize=(12, 6))
sns.heatmap(
    coverage_matrix,
    annot=True,
    fmt=".2f",
    cmap=PALETTE,
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Coverage Rate"},
)
plt.title("Required Feature Coverage by Dataset")
plt.xlabel("Required Feature")
plt.ylabel("Dataset")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
missing_matrix = (1 - coverage_matrix) * 100
plt.figure(figsize=(12, 6))
sns.heatmap(
    missing_matrix,
    annot=True,
    fmt=".1f",
    cmap="OrRd",
    vmin=0,
    vmax=100,
    linewidths=0.5,
    cbar_kws={"label": "Missing Rate Percent"},
)
plt.title("Required Feature Missing Rate by Dataset")
plt.xlabel("Required Feature")
plt.ylabel("Dataset")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 4 Duplicate Row Analysis"),
    code(
        """
duplicate_plot = inventory.sort_values("raw_duplicate_rows_after_first", ascending=True)
plt.figure(figsize=(11, 6))
sns.barplot(
    data=duplicate_plot,
    x="raw_duplicate_rows_after_first",
    y="source_file",
    color="#c65d21",
)
plt.title("Duplicate Rows after First Occurrence")
plt.xlabel("Duplicate Row Count")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
inventory[
    [
        "source_file", "rows", "raw_duplicate_rows_after_first",
        "raw_duplicate_rows_in_groups", "canonical_duplicate_rows_in_groups"
    ]
].sort_values("raw_duplicate_rows_after_first", ascending=False)
"""
    ),
    markdown("## 5 Crop Class Balance"),
    code(
        """
class_balance = inventory.copy()
class_balance["imbalance_ratio"] = (
    class_balance["maximum_class_count"] / class_balance["minimum_class_count"]
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(
    data=class_balance.sort_values("crop_classes"),
    x="crop_classes",
    y="source_file",
    color=CHART_COLOR,
    ax=axes[0],
)
axes[0].set_title("Crop Class Count by Dataset")
axes[0].set_xlabel("Crop Class Count")
axes[0].set_ylabel("Dataset")

sns.barplot(
    data=class_balance.sort_values("imbalance_ratio"),
    x="imbalance_ratio",
    y="source_file",
    color="#d49a28",
    ax=axes[1],
)
axes[1].set_title("Class Imbalance Ratio by Dataset")
axes[1].set_xlabel("Maximum Count Divided by Minimum Count")
axes[1].set_ylabel("Dataset")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
crop_mapping = pd.read_csv(METADATA_DIR / "crop_label_mapping.csv")
top_crops = (
    crop_mapping.groupby(["source_file", "crop_label"], as_index=False)["row_count"]
    .sum()
    .sort_values(["source_file", "row_count"], ascending=[True, False])
    .groupby("source_file")
    .head(10)
)

sources = top_crops["source_file"].unique()
fig, axes = plt.subplots(len(sources), 1, figsize=(12, 4 * len(sources)))
if len(sources) == 1:
    axes = [axes]

for axis, source in zip(axes, sources):
    subset = top_crops[top_crops["source_file"] == source].sort_values("row_count")
    sns.barplot(data=subset, x="row_count", y="crop_label", color=CHART_COLOR, ax=axis)
    axis.set_title("Top Crop Classes in " + source)
    axis.set_xlabel("Row Count")
    axis.set_ylabel("Crop Label")

plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 6 Crop Label Overlap"),
    code(
        """
crop_sets = {
    source: set(group["crop_label"].dropna())
    for source, group in crop_mapping.groupby("source_file")
}
source_names = list(crop_sets)
overlap = pd.DataFrame(index=source_names, columns=source_names, dtype=float)

for left in source_names:
    for right in source_names:
        union = crop_sets[left] | crop_sets[right]
        overlap.loc[left, right] = (
            len(crop_sets[left] & crop_sets[right]) / len(union) if union else 0
        )

plt.figure(figsize=(11, 9))
sns.heatmap(
    overlap,
    annot=True,
    fmt=".2f",
    cmap=PALETTE,
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Jaccard Similarity"},
)
plt.title("Crop Label Overlap between Datasets")
plt.xlabel("Dataset")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 7 Load Canonical Source Tables"),
    code(
        """
processed_files = sorted(CLEANED_BY_SOURCE_DIR.glob("*.csv"))
source_frames = {}

for path in processed_files:
    frame = pd.read_csv(path, low_memory=False)
    source_name = frame["source_file"].iloc[0]
    source_frames[source_name] = frame

print("Loaded cleaned source tables", len(source_frames))
print("Loaded cleaned rows", sum(len(frame) for frame in source_frames.values()))
"""
    ),
    markdown(
        """
## 8 NPK Distribution Review

The source units are not verified. These plots compare the stored values but do not claim that the values are directly compatible.
"""
    ),
    code(
        """
npk_parts = []
for source, frame in source_frames.items():
    sample = frame[["n", "p", "k"]].dropna(how="all")
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    sample = sample.assign(source_file=source)
    npk_parts.append(sample)

npk_long = pd.concat(npk_parts, ignore_index=True).melt(
    id_vars="source_file",
    value_vars=["n", "p", "k"],
    var_name="nutrient",
    value_name="value",
).dropna()

g = sns.catplot(
    data=npk_long,
    x="value",
    y="source_file",
    col="nutrient",
    kind="box",
    showfliers=False,
    sharex=False,
    height=6,
    aspect=0.75,
    color=CHART_COLOR,
)
g.set_axis_labels("Stored Value", "Dataset")
g.set_titles("Nutrient {col_name}")
g.fig.suptitle("NPK Distribution by Dataset", y=1.04)
plt.show()
"""
    ),
    markdown("## 9 Temperature Rainfall and Soil Moisture Review"),
    code(
        """
def sampled_feature(feature, sample_size=5000):
    parts = []
    for source, frame in source_frames.items():
        values = frame[[feature]].dropna()
        if len(values) > sample_size:
            values = values.sample(sample_size, random_state=42)
        if not values.empty:
            values = values.assign(source_file=source)
            parts.append(values)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

fig, axes = plt.subplots(3, 1, figsize=(13, 18))
for axis, feature, title in zip(
    axes,
    ["temperature", "rainfall", "soil_moisture"],
    [
        "Temperature Distribution by Dataset",
        "Rainfall Distribution by Dataset",
        "Soil Moisture Distribution by Dataset",
    ],
):
    values = sampled_feature(feature)
    sns.boxplot(
        data=values,
        x=feature,
        y="source_file",
        showfliers=False,
        color=CHART_COLOR,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Stored Value")
    axis.set_ylabel("Dataset")

plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 10 Soil Type Review"),
    code(
        """
soil_parts = []
for source, frame in source_frames.items():
    values = frame[["soil_type"]].dropna()
    if not values.empty:
        values = values.assign(source_file=source)
        soil_parts.append(values)

soil_counts = (
    pd.concat(soil_parts, ignore_index=True)
    .groupby(["source_file", "soil_type"])
    .size()
    .rename("row_count")
    .reset_index()
)

sources = soil_counts["source_file"].unique()
fig, axes = plt.subplots(len(sources), 1, figsize=(12, 4 * len(sources)))
if len(sources) == 1:
    axes = [axes]

for axis, source in zip(axes, sources):
    subset = soil_counts[soil_counts["source_file"] == source].sort_values("row_count")
    sns.barplot(data=subset, x="row_count", y="soil_type", color=CHART_COLOR, ax=axis)
    axis.set_title("Soil Type Distribution in " + source)
    axis.set_xlabel("Row Count")
    axis.set_ylabel("Soil Type")

plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 11 Suspicious Field Review"),
    code(
        """
raw_large = pd.read_csv(
    PROJECT_ROOT / "dataset" / "crop yield prediction dataset.csv",
    usecols=["precipitation", "humidity", "production"],
    low_memory=False,
)

suspicious_summary = pd.DataFrame(
    {
        "field": ["precipitation", "humidity", "production"],
        "minimum": [
            pd.to_numeric(raw_large["precipitation"], errors="coerce").min(),
            pd.to_numeric(raw_large["humidity"], errors="coerce").min(),
            pd.to_numeric(raw_large["production"], errors="coerce").min(),
        ],
        "median": [
            pd.to_numeric(raw_large["precipitation"], errors="coerce").median(),
            pd.to_numeric(raw_large["humidity"], errors="coerce").median(),
            pd.to_numeric(raw_large["production"], errors="coerce").median(),
        ],
        "maximum": [
            pd.to_numeric(raw_large["precipitation"], errors="coerce").max(),
            pd.to_numeric(raw_large["humidity"], errors="coerce").max(),
            pd.to_numeric(raw_large["production"], errors="coerce").max(),
        ],
        "non_numeric_count": [
            pd.to_numeric(raw_large["precipitation"], errors="coerce").isna().sum(),
            pd.to_numeric(raw_large["humidity"], errors="coerce").isna().sum(),
            pd.to_numeric(raw_large["production"], errors="coerce").isna().sum(),
        ],
        "decision": [
            "Do not map to rainfall",
            "Do not map to soil moisture",
            "Exclude from recommendation inputs",
        ],
    }
)
suspicious_summary
"""
    ),
    markdown("## 12 Feature Dictionary and Data Decisions"),
    code(
        """
feature_dictionary = pd.read_csv(METADATA_DIR / "source_feature_dictionary.csv")
feature_dictionary.sort_values(["canonical_feature", "source_file"])
"""
    ),
    code(
        """
inventory[
    ["source_file", "quality_tier", "training_status", "review_note"]
].sort_values(["training_status", "source_file"])
"""
    ),
    markdown("## 13 Cleaning Integrity"),
    code(
        """
cleaning_report = pd.read_csv(METADATA_DIR / "cleaning_report.csv")
cleaning_report.sort_values("raw_duplicate_rows_after_first", ascending=False)
"""
    ),
    code(
        """
cleaning_long = cleaning_report.melt(
    id_vars="source_file",
    value_vars=["rows", "cleaned_rows"],
    var_name="data_stage",
    value_name="row_count",
)

plt.figure(figsize=(12, 7))
sns.barplot(
    data=cleaning_long,
    x="row_count",
    y="source_file",
    hue="data_stage",
    palette=["#8c8c8c", CHART_COLOR],
)
plt.xscale("log")
plt.title("Raw and Cleaned Row Count on Log Scale")
plt.xlabel("Row Count")
plt.ylabel("Dataset")
plt.legend(title="Data Stage")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 14 Unit Review"),
    code(
        """
unit_review = pd.read_csv(METADATA_DIR / "unit_review.csv")
unit_review[
    [
        "source_file", "canonical_feature", "stored_unit", "unit_status",
        "conversion_applied", "approved_for_cross_source_merge", "review_note"
    ]
].sort_values(["canonical_feature", "source_file"])
"""
    ),
    code(
        """
unit_status_counts = (
    unit_review.groupby(["source_file", "unit_status"])
    .size()
    .rename("feature_count")
    .reset_index()
)

plt.figure(figsize=(13, 7))
sns.barplot(
    data=unit_status_counts,
    x="feature_count",
    y="source_file",
    hue="unit_status",
)
plt.title("Unit Verification Status by Dataset")
plt.xlabel("Feature Count")
plt.ylabel("Dataset")
plt.legend(title="Unit Status", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 15 Approved Crop Label Mapping"),
    code(
        """
mapping_changes = crop_mapping[
    crop_mapping["crop_mapping_rule"].ne("identity")
].copy()
mapping_changes[
    [
        "source_file", "crop_label_raw", "crop_label_normalized", "crop_label",
        "crop_mapping_rule", "crop_mapping_status", "is_specific_crop_label", "row_count"
    ]
].sort_values(["crop_mapping_status", "crop_label", "source_file"])
"""
    ),
    code(
        """
mapping_summary = (
    crop_mapping.groupby(["crop_mapping_status", "is_specific_crop_label"])["row_count"]
    .sum()
    .rename("row_count")
    .reset_index()
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=mapping_summary,
    x="crop_mapping_status",
    y="row_count",
    hue="is_specific_crop_label",
)
plt.title("Crop Mapping Approval Summary")
plt.xlabel("Mapping Status")
plt.ylabel("Row Count")
plt.legend(title="Specific Crop Label")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 16 Duplicate Leakage Risk"),
    code(
        """
fingerprint_rows = []
for source, frame in source_frames.items():
    fingerprint_rows.append(
        {
            "source_file": source,
            "cleaned_rows": len(frame),
            "unique_model_fingerprints": frame["feature_fingerprint"].nunique(),
            "model_fingerprint_duplicates_after_first": frame["feature_fingerprint"].duplicated().sum(),
            "conflicting_input_label_rows": frame["has_conflicting_crop_labels_for_features"].sum(),
            "conflicting_input_label_groups": frame.loc[
                frame["has_conflicting_crop_labels_for_features"], "feature_fingerprint"
            ].nunique(),
            "rows_in_cross_source_fingerprint_groups": frame[
                "global_feature_fingerprint_group_size"
            ].gt(1).sum(),
        }
    )

fingerprint_summary = pd.DataFrame(fingerprint_rows)
fingerprint_summary.sort_values("model_fingerprint_duplicates_after_first", ascending=False)
"""
    ),
    code(
        """
plt.figure(figsize=(12, 7))
sns.barplot(
    data=fingerprint_summary.sort_values("model_fingerprint_duplicates_after_first"),
    x="model_fingerprint_duplicates_after_first",
    y="source_file",
    color="#c65d21",
)
plt.title("Model Fingerprint Duplicates after Cleaning")
plt.xlabel("Duplicate Fingerprint Count")
plt.ylabel("Dataset")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 17 Data Quality Gate"),
    code(
        """
quality_gate = pd.read_csv(METADATA_DIR / "data_quality_gate.csv")
quality_gate.sort_values(["approved_for_six_feature_prototype", "source_file"], ascending=[False, True])
"""
    ),
    markdown("## 18 Model Risk Audit"),
    code(
        """
signal_audit = pd.read_csv(METADATA_DIR / "model_signal_audit.csv")
signal_audit[
    [
        "source_file", "features_used", "unique_fingerprint_rows", "classes_evaluated",
        "conflicting_input_rows_excluded", "conflicting_input_groups_excluded",
        "majority_baseline", "top_one_accuracy", "top_three_accuracy",
        "balanced_accuracy", "signal_decision"
    ]
].sort_values("top_one_accuracy", ascending=False)
"""
    ),
    code(
        """
signal_plot = signal_audit.melt(
    id_vars="source_file",
    value_vars=["majority_baseline", "top_one_accuracy"],
    var_name="metric",
    value_name="score",
)

plt.figure(figsize=(13, 7))
sns.barplot(data=signal_plot, x="score", y="source_file", hue="metric")
plt.title("Crop Signal Diagnostic after Fingerprint Deduplication")
plt.xlabel("Score")
plt.ylabel("Dataset")
plt.xlim(0, 1)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
source_shift = pd.read_csv(METADATA_DIR / "source_shift_audit.csv")
leakage_risk = pd.read_csv(METADATA_DIR / "leakage_risk_report.csv")

display(source_shift)
leakage_risk.sort_values(
    ["model_fingerprint_duplicate_rate", "maximum_single_feature_nmi"],
    ascending=False,
)
"""
    ),
    markdown(
        """
## 19 Preparation Decision

The raw files, canonical audit files, and cleaned source files are separated. Exact duplicates are removed only from cleaned copies and every removed row is recorded in a manifest. Approved crop aliases are mapped, while aggregate and ambiguous targets remain flagged for exclusion.

The unit review verifies the six core fields in Crop Recommendation where the dataset card provides evidence. Other sources remain unconverted because their NPK or soil moisture units are not documented well enough. No source is approved for a full eight feature model.

Required next modeling decisions:

1. Decide whether to use the verified six feature prototype source.
2. Keep soil type and soil moisture as compatibility inputs until trusted complete data is available.
3. Create duplicate and location aware split groups.
4. Create one frozen train validation test manifest for all model experiments.
5. Use the completed source shift and leakage audit when approving the split strategy.

No model training should start before these decisions are approved.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
