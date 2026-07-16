# Crop Recommendation

Machine learning project for crop recommendation experiments and data preparation.

## Repository setup

This repository uses Git LFS for large dataset artifacts.

### Prerequisites

- Git
- Git LFS
- Python 3.11+

### First-time clone

```bash
git clone https://github.com/oreongab/crop-recommendation.git
cd crop-recommendation
git lfs install
git lfs pull
```

### If you already cloned before installing Git LFS

```bash
git lfs install
git lfs pull
```

## Python environment (optional quick start)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-modeling.txt
```

## Large files tracked with LFS

- processed_dataset/by_source/crop_yield_prediction.csv
- processed_dataset/cleaned_by_source/crop_yield_prediction.csv

## Notes

- Data processing and model workflows are documented in PROJECT_PLAN.md and MODEL_RUN_ORDER.md.
- Additional processed dataset details are in processed_dataset/README.md.
