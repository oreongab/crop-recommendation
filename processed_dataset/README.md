# Processed Dataset Audit Layer

This directory contains provenance-safe audit and cleaned source tables.
It is not a model-ready merged dataset because no source has passed the full eight-feature quality gate.

## Safety guarantees

- Raw files in `dataset/` are unchanged.
- No missing value imputation is applied.
- No unit conversion is applied.
- Approved spelling variants and established crop synonyms are mapped with an explicit rule.
- Exact duplicate rows are removed only from cleaned copies and recorded in a manifest.
- `feature_fingerprint` hashes model inputs without the crop target so identical inputs and label conflicts can be kept out of different splits.
- Atmospheric humidity is not mapped to soil moisture.
- Soil color is not mapped to soil type.
- Unverified precipitation is not mapped to rainfall.
- No train, validation, or test split is created.

## Contents

- `by_source/`: one canonical audit table for each raw CSV.
- `cleaned_by_source/`: source tables after exact duplicate removal and approved crop mapping.
- `manifests/raw_file_manifest.csv`: raw file hashes, sizes, row counts, and column counts.
- `manifests/removed_exact_duplicates.csv`: every removed row and the retained source row.
- `metadata/dataset_inventory.csv`: row counts, duplicates, class balance, and source decisions.
- `metadata/cleaning_report.csv`: raw-to-cleaned row reconciliation and destructive-operation checks.
- `metadata/required_feature_coverage.csv`: observed and missing counts for required features.
- `metadata/source_feature_dictionary.csv`: source-to-canonical mappings and semantic warnings.
- `metadata/unit_review.csv`: verified, inferred, and unknown units with evidence links.
- `metadata/crop_label_mapping.csv`: approved canonical crop labels and excluded ambiguous targets.
- `metadata/data_quality_gate.csv`: source approval decisions.
- `metadata/model_signal_audit.csv`: fingerprint-deduplicated crop signal diagnostic.
- `metadata/source_shift_audit.csv`: source separability diagnostic.
- `metadata/leakage_risk_report.csv`: duplicate and single-feature leakage indicators.
- `metadata/preparation_summary.json`: preparation guarantees and total counts.

## Rebuild

Run from the repository root:

```bash
python3 -B scripts/prepare_data.py
python3 -B scripts/audit_data_risk.py
python3 -B scripts/build_test1_notebook.py
```

Review `test1.ipynb` before approving any source for model training.
