# Model Notebook Run Order

## Current model scope

The prepared model is a six-feature prototype. It uses N, P, K, pH, temperature in degree Celsius, and rainfall in millimeters. Soil type and soil moisture are not model features because compatible, source-verified training data is not available.

The frozen split contains 1,540 Train rows, 330 Validation rows, and 330 sealed Test rows across the same 22 crop classes. The three notebooks report Validation results only. Do not use Test to choose a model.

## Before running

Open a terminal in the project directory and run:

```bash
python3 -m pip install -r requirements-modeling.txt
python3 scripts/validate_pipeline.py
```

On macOS, if XGBoost reports a missing OpenMP runtime, install it once:

```bash
brew install libomp
```

The current prepared files already exist, so restarting the computer does not require rebuilding them. If the raw CSV files change or this project is copied without `processed_dataset`, rebuild in this order:

```bash
python3 scripts/prepare_data.py
python3 scripts/audit_data_risk.py
python3 scripts/build_model_dataset.py
python3 scripts/build_model_notebooks.py
python3 scripts/validate_pipeline.py
```

After that, open `test1.ipynb` and use Restart Kernel and Run All to regenerate its exploration outputs.

## Run the notebooks

Use Restart Kernel and Run All for each notebook in this order:

1. `02_random_forest.ipynb`
2. `03_xgboost.ipynb`
3. `04_neural_network.ipynb`

Each notebook displays a progress bar, elapsed training time, Validation metrics, a normalized confusion matrix, per-class F1, calibration, confidence distribution, and model-specific training visuals. Each also saves its model and Validation artifacts under `artifacts/<model_name>/`.

## After all three finish

Compare Top 3 Accuracy, Macro F1, Balanced Accuracy, Log Loss, calibration error, training time, and weak crop classes. Select and lock one model and its settings before opening the sealed Test split once for final evaluation.

N, P, and K must be entered on the same ratio scale used by the training dataset. They must not be presented as kg per hectare unless a documented conversion and a newly validated training contract are added.
