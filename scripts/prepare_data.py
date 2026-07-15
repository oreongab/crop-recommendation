#!/usr/bin/env python3
"""Prepare provenance-safe crop datasets without imputing or changing units."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "processed_dataset"
BY_SOURCE_DIR = OUTPUT_DIR / "by_source"
CLEANED_BY_SOURCE_DIR = OUTPUT_DIR / "cleaned_by_source"
METADATA_DIR = OUTPUT_DIR / "metadata"
MANIFEST_DIR = OUTPUT_DIR / "manifests"

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

CANONICAL_COLUMNS = [
    "source_file",
    "source_row_id",
    "raw_row_hash",
    "crop_label_raw",
    "crop_label_normalized",
    "crop_label",
    "crop_mapping_rule",
    "crop_mapping_status",
    "is_specific_crop_label",
    *REQUIRED_FEATURES,
    "location",
    "year",
    "season",
    "row_fingerprint",
    "feature_fingerprint",
    "is_raw_duplicate_in_source",
    "is_raw_duplicate_after_first",
    "duplicate_keep_source_row_id",
    "is_canonical_duplicate_in_source",
    "canonical_duplicate_group_size",
    "global_fingerprint_group_size",
    "feature_fingerprint_group_size",
    "global_feature_fingerprint_group_size",
    "feature_fingerprint_crop_label_count",
    "has_conflicting_crop_labels_for_features",
    "complete_required_features",
    "has_negative_required_numeric",
    "ph_outside_physical_range",
    "row_quality_status",
    "quality_tier",
    "training_status",
]


APPROVED_CROP_ALIASES = {
    "arhar/tur": ("pigeon pea", "common_name_synonym"),
    "bajra": ("pearl millet", "common_name_synonym"),
    "bhindi": ("okra", "common_name_synonym"),
    "bittergourd": ("bitter gourd", "spacing_variant"),
    "blackgram": ("black gram", "spacing_variant"),
    "blackpepper": ("black pepper", "spacing_variant"),
    "bottlegourd": ("bottle gourd", "spacing_variant"),
    "cotton(lint)": ("cotton", "harvest_form_parent_crop"),
    "cowpea(lobia)": ("cowpea", "parenthetical_synonym"),
    "drum stick": ("moringa", "common_name_synonym"),
    "drumstick": ("moringa", "common_name_synonym"),
    "finger millet (ragi)": ("finger millet", "parenthetical_synonym"),
    "gram": ("chickpea", "context_specific_common_name"),
    "ground nuts": ("groundnut", "spacing_variant"),
    "guinea corn": ("sorghum", "common_name_synonym"),
    "horse-gram": ("horse gram", "punctuation_variant"),
    "horsegram": ("horse gram", "spacing_variant"),
    "jack fruit": ("jackfruit", "spacing_variant"),
    "jowar": ("sorghum", "common_name_synonym"),
    "kapas": ("cotton", "common_name_synonym"),
    "kidneybeans": ("kidney bean", "spacing_variant"),
    "ladyfinger": ("okra", "common_name_synonym"),
    "masoor": ("lentil", "common_name_synonym"),
    "moong": ("mung bean", "common_name_synonym"),
    "moong(green gram)": ("mung bean", "common_name_synonym"),
    "mothbeans": ("moth bean", "spacing_variant"),
    "mungbean": ("mung bean", "spacing_variant"),
    "paddy": ("rice", "common_name_synonym"),
    "pearl millet (bajra)": ("pearl millet", "parenthetical_synonym"),
    "pigeonpeas": ("pigeon pea", "spacing_variant"),
    "pome granet": ("pomegranate", "spelling_correction"),
    "pomegranate (bhagwa variety)": ("pomegranate", "variety_parent_crop"),
    "pomegranates": ("pomegranate", "plural_variant"),
    "pump kin": ("pumpkin", "spacing_correction"),
    "ragi": ("finger millet", "common_name_synonym"),
    "sesamum": ("sesame", "common_name_synonym"),
    "sorghum (jowar)": ("sorghum", "parenthetical_synonym"),
    "soyabean": ("soybean", "spelling_variant"),
    "sweetpotato": ("sweet potato", "spacing_variant"),
    "tomatoes": ("tomato", "plural_variant"),
    "tur": ("pigeon pea", "common_name_synonym"),
    "urad": ("black gram", "common_name_synonym"),
    "water melon": ("watermelon", "spacing_variant"),
}


NON_SPECIFIC_CROP_LABELS = {
    "arcanut (processed)",
    "atcanut (raw)",
    "beans & mutter(vegetable)",
    "cashewnut processed",
    "cashewnut raw",
    "cond-spcs other",
    "fallow",
    "jute & mesta",
    "jobster",
    "millets",
    "oil seeds",
    "oilseeds total",
    "other cereals",
    "other cereals & millets",
    "other citrus fruit",
    "other dry fruit",
    "other fibres",
    "other fresh fruits",
    "other kharif pulses",
    "other misc. pulses",
    "other oilseeds",
    "other rabi pulses",
    "other summer pulses",
    "other vegetables",
    "peas & beans (pulses)",
    "pulses",
    "pulses total",
    "redish",
    "ribed guard",
    "small millets",
    "total foodgrain",
}


UNIT_EVIDENCE_URLS = {
    "Crop Recommendation using Soil Properties and Weather Prediction.csv":
        "https://data.mendeley.com/datasets/8v757rr4st/1",
    "Crop and fertilizer dataset.csv":
        "https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra",
    "Crop_Recommendation.csv":
        "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data",
}


SOURCE_CONFIG = {
    "Crop Recommendation using Soil Properties and Weather Prediction.csv": {
        "output_name": "crop_recommendation_soil_weather.csv",
        "label": "label",
        "direct": {
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "soil_moisture": "soil_moisture",
        },
        "quality_tier": "needs_semantic_verification",
        "training_status": "hold",
        "note": (
            "Seasonal temperature and precipitation are not converted to annual "
            "temperature or rainfall. Soil color is not mapped to soil type. NPK "
            "and soil moisture units require source verification."
        ),
    },
    "Crop and fertilizer dataset.csv": {
        "output_name": "crop_fertilizer.csv",
        "label": "label",
        "direct": {
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "location": ["district_name"],
        "quality_tier": "reference_only",
        "training_status": "hold",
        "note": (
            "Soil color is not mapped to soil type. Soil moisture is unavailable. "
            "NPK may represent fertilizer requirements rather than soil measurements."
        ),
    },
    "Crop_Recommendation.csv": {
        "output_name": "crop_recommendation.csv",
        "label": "label",
        "direct": {
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "quality_tier": "prototype_candidate",
        "training_status": "candidate_after_validation",
        "note": (
            "Atmospheric humidity is excluded because it is not soil moisture. "
            "Soil type and soil moisture are unavailable."
        ),
    },
    "crop yield prediction dataset.csv": {
        "output_name": "crop_yield_prediction.csv",
        "label": "crop_names",
        "direct": {
            "soil_type": "soil_type",
            "n": "N",
            "p": "P",
            "k": "K",
            "temperature": "temperature",
        },
        "location": ["state_names", "district_names"],
        "year": "crop_year",
        "season": "season_names",
        "quality_tier": "quarantine",
        "training_status": "exclude_v1",
        "note": (
            "Precipitation values near 1009 to 1023 are not mapped to rainfall. "
            "Atmospheric humidity is excluded. PH and soil moisture are unavailable."
        ),
    },
    "crop-yield.csv": {
        "output_name": "crop_yield_complete.csv",
        "label": "label",
        "direct": {
            "soil_type": "soil_type",
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "soil_moisture": "soil_moisture",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "location": ["region"],
        "season": "season",
        "quality_tier": "complete_but_low_signal",
        "training_status": "exclude_until_signal_review",
        "note": (
            "All required fields are present, but the initial crop classification "
            "diagnostic was near the majority baseline. Treat as a pipeline smoke test only."
        ),
    },
    "crop_data.csv": {
        "output_name": "crop_data.csv",
        "label": "label",
        "direct": {
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "quality_tier": "reference_only",
        "training_status": "hold",
        "note": (
            "Soil type and soil moisture are unavailable. The source has many crop "
            "classes with very small support and contains duplicate rows."
        ),
    },
    "crop_yield.csv": {
        "output_name": "crop_yield_historical.csv",
        "label": "label",
        "direct": {
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "location": ["state"],
        "year": "year",
        "season": "season",
        "quality_tier": "yield_reference",
        "training_status": "exclude_v1",
        "note": (
            "Atmospheric humidity is excluded. Soil type and soil moisture are unavailable. "
            "Yield and production fields are not recommendation inputs."
        ),
    },
    "data_core.csv": {
        "output_name": "data_core.csv",
        "label": "label",
        "direct": {
            "soil_type": "soil_type",
            "n": "n",
            "p": "p",
            "k": "k",
            "soil_moisture": "soil_moisture",
            "temperature": "temperature",
        },
        "quality_tier": "reference_only",
        "training_status": "hold",
        "note": (
            "PH and rainfall are unavailable. The initial crop classification "
            "diagnostic was near the majority baseline."
        ),
    },
    "original_dataset.csv": {
        "output_name": "original_dataset.csv",
        "label": "label",
        "direct": {
            "soil_type": "soil_type",
            "ph": "ph",
            "n": "n",
            "p": "p",
            "k": "k",
            "temperature": "temperature",
            "rainfall": "rainfall",
        },
        "quality_tier": "quarantine",
        "training_status": "exclude_v1",
        "note": (
            "Soil moisture is unavailable. The source contains extensive duplicate rows, "
            "and soil type is deterministic for each crop label."
        ),
    },
}


def normalize_text(series: pd.Series) -> pd.Series:
    """Apply syntax-only normalization without merging crop synonyms."""
    result = series.astype("string").str.strip().str.lower()
    result = result.str.replace("_", " ", regex=False)
    result = result.str.replace(r"\s+", " ", regex=True)
    return result.replace("", pd.NA)


def canonicalize_crop_label(value: object) -> tuple[object, str, str, bool]:
    """Map only approved spelling variants and well-established crop synonyms."""
    if pd.isna(value):
        return pd.NA, "missing_label", "excluded", False
    normalized = str(value)
    if normalized in NON_SPECIFIC_CROP_LABELS:
        return normalized, "non_specific_or_ambiguous_target", "excluded", False
    if normalized in APPROVED_CROP_ALIASES:
        canonical, rule = APPROVED_CROP_ALIASES[normalized]
        return canonical, rule, "approved", True
    return normalized, "identity", "approved", True


def hash_raw_rows(df: pd.DataFrame) -> pd.Series:
    """Hash every original column so exact duplicate removal is auditable."""
    text = df.copy()
    for column in text.columns:
        text[column] = text[column].map(
            lambda value: "<missing>" if pd.isna(value) else str(value).strip()
        )
    joined = text.astype(str).agg("||".join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_location(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = [normalize_text(df[column]).fillna("") for column in columns]
    combined = values[0]
    for value in values[1:]:
        combined = combined + " | " + value
    combined = combined.str.strip(" |")
    return combined.replace("", pd.NA).astype("string")


def fingerprint_rows(df: pd.DataFrame) -> pd.Series:
    return fingerprint_fields(df, ["crop_label", *REQUIRED_FEATURES])


def feature_fingerprint_rows(df: pd.DataFrame) -> pd.Series:
    """Hash model inputs without target so conflicting labels cannot cross splits."""
    return fingerprint_fields(df, REQUIRED_FEATURES)


def fingerprint_fields(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    text = df[fields].copy()
    for column in fields:
        text[column] = text[column].map(
            lambda value: "<missing>" if pd.isna(value) else str(value).strip().lower()
        )
    joined = text.astype(str).agg("||".join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def build_feature_dictionary() -> pd.DataFrame:
    rows = []
    for source_file, config in SOURCE_CONFIG.items():
        direct = config.get("direct", {})
        for feature in REQUIRED_FEATURES:
            original = direct.get(feature, "")
            semantic_status = "direct_name_match" if original else "not_available"
            if source_file == "crop yield prediction dataset.csv" and feature == "rainfall":
                semantic_status = "excluded_precipitation_not_verified_as_rainfall"
            elif source_file in {
                "Crop Recommendation using Soil Properties and Weather Prediction.csv",
                "Crop and fertilizer dataset.csv",
            } and feature == "soil_type":
                semantic_status = "excluded_soil_color_is_not_soil_type"
            elif feature == "soil_moisture" and not original:
                semantic_status = "excluded_atmospheric_humidity_is_not_soil_moisture"

            rows.append(
                {
                    "source_file": source_file,
                    "canonical_feature": feature,
                    "original_column": original,
                    "available": bool(original),
                    "transformation": "numeric_coercion_only" if original and feature != "soil_type" else (
                        "syntax_normalization_only" if original else "none"
                    ),
                    "unit_status": "unverified" if original and feature not in {"soil_type"} else "not_applicable",
                    "semantic_status": semantic_status,
                    "approved_for_training": False,
                }
            )
    return pd.DataFrame(rows)


def build_unit_review(feature_dictionary: pd.DataFrame) -> pd.DataFrame:
    """Record only source-supported units and leave uncertain values unconverted."""
    rows = []
    for record in feature_dictionary.to_dict("records"):
        source = record["source_file"]
        feature = record["canonical_feature"]
        available = bool(record["available"])
        unit = "not_available"
        status = "not_available"
        evidence = UNIT_EVIDENCE_URLS.get(source, "")
        note = "Feature is not present in the canonical source mapping."

        if available:
            unit = "unknown"
            status = "unverified_no_conversion"
            note = "The local file does not provide a trustworthy unit definition."

        if feature == "soil_type" and available:
            unit = "category"
            status = "semantic_category_verified"
            note = "Text category normalized for case and whitespace only."
        elif feature == "ph" and available:
            unit = "ph_scale"
            status = "physical_scale_verified"
            note = "PH is dimensionless and retained without conversion."

        if source == "Crop_Recommendation.csv" and available:
            verified = {
                "n": ("soil_nutrient_ratio", "Dataset card defines N as nitrogen ratio in soil."),
                "p": ("soil_nutrient_ratio", "Dataset card defines P as phosphorus ratio in soil."),
                "k": ("soil_nutrient_ratio", "Dataset card defines K as potassium ratio in soil."),
                "temperature": ("degree_celsius", "Dataset card defines temperature in degree Celsius."),
                "rainfall": ("millimeter", "Dataset card defines rainfall in millimeter."),
                "ph": ("ph_scale", "Dataset card defines soil PH."),
            }
            if feature in verified:
                unit, note = verified[feature]
                status = "verified_from_dataset_card"

        if source == "Crop and fertilizer dataset.csv" and available and feature in {"n", "p", "k", "temperature", "rainfall"}:
            unit = "rate_unspecified"
            status = "unverified_no_conversion"
            note = "Dataset card calls the field a rate but does not publish a unit."

        if source == "Crop Recommendation using Soil Properties and Weather Prediction.csv" and available:
            if feature in {"n", "p", "k", "soil_moisture"}:
                unit = "unknown"
                status = "unverified_no_conversion"
                note = "Dataset description confirms the feature domain but does not publish its unit on the data card."
            elif feature == "ph":
                unit = "ph_scale"
                status = "verified_from_dataset_description"
                note = "Dataset description identifies this field as soil PH."

        if source in {"crop-yield.csv", "data_core.csv"} and feature == "soil_moisture" and available:
            unit = "percent_like_scale"
            status = "inferred_not_approved"
            note = "Values resemble percent soil moisture, but no source documentation was found."

        rows.append(
            {
                "source_file": source,
                "canonical_feature": feature,
                "original_column": record["original_column"],
                "stored_unit": unit,
                "unit_status": status,
                "conversion_applied": False,
                "approved_for_cross_source_merge": False,
                "evidence_url": evidence,
                "review_note": note,
            }
        )
    return pd.DataFrame(rows)


def prepare_source(path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    prepared = pd.DataFrame(index=raw.index)
    prepared["source_file"] = path.name
    prepared["source_row_id"] = np.arange(1, len(raw) + 1, dtype=np.int64)
    prepared["raw_row_hash"] = hash_raw_rows(raw)
    prepared["crop_label_raw"] = raw[config["label"]].astype("string")
    prepared["crop_label_normalized"] = normalize_text(raw[config["label"]])
    mapped = prepared["crop_label_normalized"].map(canonicalize_crop_label)
    prepared["crop_label"] = mapped.map(lambda value: value[0]).astype("string")
    prepared["crop_mapping_rule"] = mapped.map(lambda value: value[1]).astype("string")
    prepared["crop_mapping_status"] = mapped.map(lambda value: value[2]).astype("string")
    prepared["is_specific_crop_label"] = mapped.map(lambda value: value[3]).astype(bool)

    for feature in REQUIRED_FEATURES:
        original = config.get("direct", {}).get(feature)
        if not original:
            prepared[feature] = pd.NA if feature == "soil_type" else np.nan
        elif feature == "soil_type":
            prepared[feature] = normalize_text(raw[original])
        else:
            prepared[feature] = pd.to_numeric(raw[original], errors="coerce")

    location_columns = config.get("location", [])
    prepared["location"] = (
        combine_location(raw, location_columns)
        if location_columns
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    year_column = config.get("year")
    prepared["year"] = raw[year_column] if year_column else pd.NA
    season_column = config.get("season")
    prepared["season"] = normalize_text(raw[season_column]) if season_column else pd.NA

    prepared["row_fingerprint"] = fingerprint_rows(prepared)
    prepared["feature_fingerprint"] = feature_fingerprint_rows(prepared)
    prepared["is_raw_duplicate_in_source"] = raw.duplicated(keep=False)
    prepared["is_raw_duplicate_after_first"] = raw.duplicated(keep="first")
    first_source_row = prepared.groupby("raw_row_hash")["source_row_id"].transform("min")
    prepared["duplicate_keep_source_row_id"] = first_source_row.astype(np.int64)
    canonical_group_size = prepared.groupby("row_fingerprint")["row_fingerprint"].transform("size")
    prepared["is_canonical_duplicate_in_source"] = canonical_group_size.gt(1)
    prepared["canonical_duplicate_group_size"] = canonical_group_size.astype(np.int64)
    feature_group_size = prepared.groupby("feature_fingerprint")["feature_fingerprint"].transform("size")
    prepared["feature_fingerprint_group_size"] = feature_group_size.astype(np.int64)
    feature_label_count = prepared.groupby("feature_fingerprint")["crop_label"].transform("nunique")
    prepared["feature_fingerprint_crop_label_count"] = feature_label_count.astype(np.int64)
    prepared["has_conflicting_crop_labels_for_features"] = feature_label_count.gt(1)
    prepared["complete_required_features"] = prepared[REQUIRED_FEATURES].notna().all(axis=1)
    numeric_features = [feature for feature in REQUIRED_FEATURES if feature != "soil_type"]
    prepared["has_negative_required_numeric"] = prepared[numeric_features].lt(0).any(axis=1)
    prepared["ph_outside_physical_range"] = prepared["ph"].notna() & ~prepared["ph"].between(0, 14)
    prepared["row_quality_status"] = "reviewable"
    prepared.loc[~prepared["is_specific_crop_label"], "row_quality_status"] = "exclude_non_specific_crop"
    prepared.loc[
        prepared["has_negative_required_numeric"] | prepared["ph_outside_physical_range"],
        "row_quality_status",
    ] = "exclude_physical_range_error"
    prepared.loc[
        prepared["has_conflicting_crop_labels_for_features"],
        "row_quality_status",
    ] = "review_conflicting_crop_labels"
    prepared.loc[prepared["is_raw_duplicate_after_first"], "row_quality_status"] = "remove_exact_duplicate"
    prepared["quality_tier"] = config["quality_tier"]
    prepared["training_status"] = config["training_status"]

    class_counts = prepared["crop_label"].value_counts(dropna=False)
    inventory = {
        "source_file": path.name,
        "processed_file": config["output_name"],
        "rows": int(len(raw)),
        "columns_original": int(raw.shape[1]),
        "crop_classes": int(prepared["crop_label"].nunique(dropna=True)),
        "minimum_class_count": int(class_counts.min()),
        "median_class_count": float(class_counts.median()),
        "maximum_class_count": int(class_counts.max()),
        "raw_duplicate_rows_after_first": int(raw.duplicated().sum()),
        "raw_duplicate_rows_in_groups": int(raw.duplicated(keep=False).sum()),
        "canonical_duplicate_rows_in_groups": int(prepared["is_canonical_duplicate_in_source"].sum()),
        "complete_required_rows": int(prepared["complete_required_features"].sum()),
        "non_specific_crop_rows": int((~prepared["is_specific_crop_label"]).sum()),
        "physical_range_error_rows": int(
            (prepared["has_negative_required_numeric"] | prepared["ph_outside_physical_range"]).sum()
        ),
        "conflicting_feature_label_rows": int(prepared["has_conflicting_crop_labels_for_features"].sum()),
        "conflicting_feature_groups": int(
            prepared.loc[
                prepared["has_conflicting_crop_labels_for_features"], "feature_fingerprint"
            ].nunique()
        ),
        "quality_tier": config["quality_tier"],
        "training_status": config["training_status"],
        "review_note": config["note"],
    }
    return prepared, inventory


def main() -> None:
    BY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_BY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    actual_files = {path.name for path in RAW_DIR.glob("*.csv")}
    configured_files = set(SOURCE_CONFIG)
    if actual_files != configured_files:
        missing = sorted(configured_files - actual_files)
        unexpected = sorted(actual_files - configured_files)
        raise RuntimeError(f"Dataset inventory changed. Missing={missing}, unexpected={unexpected}")

    raw_manifest_rows = []
    for source_file in sorted(actual_files):
        path = RAW_DIR / source_file
        frame = pd.read_csv(path, low_memory=False)
        raw_manifest_rows.append(
            {
                "source_file": source_file,
                "file_size_bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "rows": len(frame),
                "columns": frame.shape[1],
            }
        )

    prepared_sources = []
    inventory_rows = []
    for source_file, config in SOURCE_CONFIG.items():
        prepared, inventory = prepare_source(RAW_DIR / source_file, config)
        prepared_sources.append((config["output_name"], prepared))
        inventory_rows.append(inventory)

    audit_global_fingerprints = pd.concat(
        [frame[["row_fingerprint"]] for _, frame in prepared_sources],
        ignore_index=True,
    )["row_fingerprint"].value_counts()
    audit_global_feature_fingerprints = pd.concat(
        [frame[["feature_fingerprint"]] for _, frame in prepared_sources],
        ignore_index=True,
    )["feature_fingerprint"].value_counts()

    cleaned_sources = []
    removed_duplicate_parts = []
    for output_name, prepared in prepared_sources:
        cleaned = prepared.loc[~prepared["is_raw_duplicate_after_first"]].copy()
        canonical_group_size = cleaned.groupby("row_fingerprint")["row_fingerprint"].transform("size")
        cleaned["is_canonical_duplicate_in_source"] = canonical_group_size.gt(1)
        cleaned["canonical_duplicate_group_size"] = canonical_group_size.astype(np.int64)
        feature_group_size = cleaned.groupby("feature_fingerprint")["feature_fingerprint"].transform("size")
        cleaned["feature_fingerprint_group_size"] = feature_group_size.astype(np.int64)
        feature_label_count = cleaned.groupby("feature_fingerprint")["crop_label"].transform("nunique")
        cleaned["feature_fingerprint_crop_label_count"] = feature_label_count.astype(np.int64)
        cleaned["has_conflicting_crop_labels_for_features"] = feature_label_count.gt(1)
        cleaned_sources.append((output_name, cleaned))

        removed = prepared.loc[
            prepared["is_raw_duplicate_after_first"],
            [
                "source_file",
                "source_row_id",
                "duplicate_keep_source_row_id",
                "raw_row_hash",
                "crop_label_raw",
                "crop_label",
            ],
        ].copy()
        removed["removal_reason"] = "exact_raw_duplicate_after_first_occurrence"
        removed_duplicate_parts.append(removed)

    cleaned_global_fingerprints = pd.concat(
        [frame[["row_fingerprint"]] for _, frame in cleaned_sources],
        ignore_index=True,
    )["row_fingerprint"].value_counts()
    cleaned_global_feature_fingerprints = pd.concat(
        [frame[["feature_fingerprint"]] for _, frame in cleaned_sources],
        ignore_index=True,
    )["feature_fingerprint"].value_counts()

    coverage_rows = []
    label_rows = []
    for output_name, prepared in prepared_sources:
        prepared["global_fingerprint_group_size"] = (
            prepared["row_fingerprint"].map(audit_global_fingerprints).astype(np.int64)
        )
        prepared["global_feature_fingerprint_group_size"] = (
            prepared["feature_fingerprint"].map(audit_global_feature_fingerprints).astype(np.int64)
        )
        prepared = prepared[CANONICAL_COLUMNS]
        prepared.to_csv(BY_SOURCE_DIR / output_name, index=False)

    for output_name, cleaned in cleaned_sources:
        cleaned["global_fingerprint_group_size"] = (
            cleaned["row_fingerprint"].map(cleaned_global_fingerprints).astype(np.int64)
        )
        cleaned["global_feature_fingerprint_group_size"] = (
            cleaned["feature_fingerprint"].map(cleaned_global_feature_fingerprints).astype(np.int64)
        )
        cleaned = cleaned[CANONICAL_COLUMNS]
        cleaned.to_csv(CLEANED_BY_SOURCE_DIR / output_name, index=False)

        source_file = str(cleaned["source_file"].iloc[0])
        for feature in REQUIRED_FEATURES:
            observed = int(cleaned[feature].notna().sum())
            coverage_rows.append(
                {
                    "source_file": source_file,
                    "feature": feature,
                    "observed_rows": observed,
                    "missing_rows": int(len(cleaned) - observed),
                    "coverage_rate": observed / len(cleaned),
                }
            )

        counts = (
            cleaned.groupby(
                [
                    "crop_label_raw",
                    "crop_label_normalized",
                    "crop_label",
                    "crop_mapping_rule",
                    "crop_mapping_status",
                    "is_specific_crop_label",
                ],
                dropna=False,
            )
            .size()
            .rename("row_count")
            .reset_index()
        )
        counts.insert(0, "source_file", source_file)
        label_rows.append(counts)

    inventory = pd.DataFrame(inventory_rows).sort_values("source_file")
    cleaned_count_map = {
        str(frame["source_file"].iloc[0]): len(frame) for _, frame in cleaned_sources
    }
    cleaned_conflict_row_map = {
        str(frame["source_file"].iloc[0]): int(frame["has_conflicting_crop_labels_for_features"].sum())
        for _, frame in cleaned_sources
    }
    cleaned_conflict_group_map = {
        str(frame["source_file"].iloc[0]): int(
            frame.loc[frame["has_conflicting_crop_labels_for_features"], "feature_fingerprint"].nunique()
        )
        for _, frame in cleaned_sources
    }
    inventory["cleaned_rows"] = inventory["source_file"].map(cleaned_count_map).astype(int)
    inventory["cleaned_conflicting_feature_label_rows"] = (
        inventory["source_file"].map(cleaned_conflict_row_map).astype(int)
    )
    inventory["cleaned_conflicting_feature_groups"] = (
        inventory["source_file"].map(cleaned_conflict_group_map).astype(int)
    )
    inventory["row_reconciliation_passed"] = (
        inventory["rows"]
        == inventory["cleaned_rows"] + inventory["raw_duplicate_rows_after_first"]
    )
    inventory.to_csv(METADATA_DIR / "dataset_inventory.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(METADATA_DIR / "required_feature_coverage.csv", index=False)
    feature_dictionary = build_feature_dictionary()
    feature_dictionary.to_csv(METADATA_DIR / "source_feature_dictionary.csv", index=False)
    unit_review = build_unit_review(feature_dictionary)
    unit_review.to_csv(METADATA_DIR / "unit_review.csv", index=False)
    pd.concat(label_rows, ignore_index=True).to_csv(METADATA_DIR / "crop_label_mapping.csv", index=False)
    pd.DataFrame(raw_manifest_rows).to_csv(MANIFEST_DIR / "raw_file_manifest.csv", index=False)
    removed_duplicates = pd.concat(removed_duplicate_parts, ignore_index=True)
    removed_duplicates.to_csv(MANIFEST_DIR / "removed_exact_duplicates.csv", index=False)

    cleaning_report = inventory[
        [
            "source_file",
            "rows",
            "raw_duplicate_rows_after_first",
            "cleaned_rows",
            "non_specific_crop_rows",
            "physical_range_error_rows",
            "cleaned_conflicting_feature_label_rows",
            "cleaned_conflicting_feature_groups",
            "row_reconciliation_passed",
        ]
    ].copy()
    cleaning_report["value_imputation_applied"] = False
    cleaning_report["unit_conversion_applied"] = False
    cleaning_report["outlier_deletion_applied"] = False
    cleaning_report.to_csv(METADATA_DIR / "cleaning_report.csv", index=False)

    quality_gate_rows = []
    for record in inventory.to_dict("records"):
        source = record["source_file"]
        source_units = unit_review[unit_review["source_file"] == source]
        verified_units = int(source_units["unit_status"].str.startswith("verified").sum())
        full_features = record["complete_required_rows"] > 0
        six_feature_prototype = source == "Crop_Recommendation.csv"
        quality_gate_rows.append(
            {
                "source_file": source,
                "cleaning_reconciliation_passed": bool(record["row_reconciliation_passed"]),
                "exact_duplicates_removed_from_cleaned_copy": int(record["raw_duplicate_rows_after_first"]),
                "physical_range_error_rows": int(record["physical_range_error_rows"]),
                "non_specific_crop_rows": int(record["non_specific_crop_rows"]),
                "conflicting_feature_label_rows": int(
                    record["cleaned_conflicting_feature_label_rows"]
                ),
                "conflicting_feature_groups": int(
                    record["cleaned_conflicting_feature_groups"]
                ),
                "verified_required_unit_count": verified_units,
                "has_complete_eight_feature_rows": bool(full_features),
                "approved_for_full_eight_feature_training": False,
                "approved_for_six_feature_prototype": bool(six_feature_prototype),
                "leakage_control_required": True,
                "gate_decision": (
                    "prototype_six_features_only"
                    if six_feature_prototype
                    else "hold_or_exclude_pending_modeling_strategy"
                ),
                "gate_reason": record["review_note"],
            }
        )
    pd.DataFrame(quality_gate_rows).to_csv(METADATA_DIR / "data_quality_gate.csv", index=False)

    summary = {
        "raw_directory": str(RAW_DIR.relative_to(ROOT)),
        "processed_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        "source_files": len(prepared_sources),
        "total_rows": int(inventory["rows"].sum()),
        "cleaned_rows": int(inventory["cleaned_rows"].sum()),
        "exact_duplicate_rows_removed_from_cleaned_copies": int(
            inventory["raw_duplicate_rows_after_first"].sum()
        ),
        "complete_required_rows": int(inventory["complete_required_rows"].sum()),
        "imputation_applied": False,
        "unit_conversion_applied": False,
        "approved_crop_label_mapping_applied": True,
        "outlier_deletion_applied": False,
        "train_validation_test_split_created": False,
        "raw_files_modified": False,
        "all_row_reconciliations_passed": bool(inventory["row_reconciliation_passed"].all()),
        "full_eight_feature_training_source_approved": False,
    }
    (METADATA_DIR / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
