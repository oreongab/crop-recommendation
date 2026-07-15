#!/usr/bin/env python3
"""Generate three consistent model experiment notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


COMMON_SETUP = """
from pathlib import Path
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from tqdm.auto import tqdm

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (
    ARTIFACT_DIR,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    encode_training_validation_targets,
    evaluate_probabilities,
    load_training_validation_splits,
    plot_evaluation_dashboard,
    prediction_table,
    save_evaluation_outputs,
    training_validation_summary,
)

np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")
print("Python", platform.python_version())
print("Project root", PROJECT_ROOT)
"""


COMMON_DATA = """
train, validation, sealed_test_summary, split_integrity = load_training_validation_splits()
label_encoder, y_train, y_validation = encode_training_validation_targets(train, validation)

X_train = train[FEATURES].copy()
X_validation = validation[FEATURES].copy()

display(training_validation_summary(train, validation))
print("Features", FEATURES)
print("Crop classes", list(label_encoder.classes_))
print("Sealed Test metadata", sealed_test_summary)
"""


COMMON_SPLIT_PLOT = """
distribution = pd.concat(
    [
        train.assign(data_split="Train"),
        validation.assign(data_split="Validation"),
    ],
    ignore_index=True,
)
class_counts = (
    distribution.groupby([TARGET, "data_split"])
    .size()
    .rename("row_count")
    .reset_index()
)

plt.figure(figsize=(14, 7))
sns.barplot(data=class_counts, x=TARGET, y="row_count", hue="data_split")
plt.title("Frozen Split Class Distribution")
plt.xlabel("Crop Label")
plt.ylabel("Row Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
"""


COMMON_EVALUATION = """
validation_metrics, validation_per_class, validation_matrix, validation_calibration = (
    evaluate_probabilities(y_validation, validation_probabilities, label_encoder)
)
validation_predictions = prediction_table(
    y_validation, validation_probabilities, label_encoder
)

metrics_table = pd.DataFrame(
    {"metric": list(validation_metrics), "value": list(validation_metrics.values())}
)
display(metrics_table)
display(validation_per_class.sort_values("f1-score").reset_index(drop=True))

plot_evaluation_dashboard(
    MODEL_NAME,
    validation_metrics,
    validation_per_class,
    validation_matrix,
    validation_calibration,
    y_validation,
    validation_probabilities,
    label_encoder,
    training_history,
)
"""


COMMON_SAVE = """
output_dir = save_evaluation_outputs(
    MODEL_NAME,
    validation_metrics,
    validation_per_class,
    validation_calibration,
    validation_predictions,
    training_history,
    training_seconds,
)
print("Saved evaluation artifacts to", output_dir)
print("Training time seconds", round(training_seconds, 3))
print("Test split used", False)
"""


def notebook(cells: list[dict]) -> dict:
    return {
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


rf_cells = [
    markdown(
        """
# Random Forest Crop Recommendation Prototype

This notebook trains a six-feature prototype using N, P, K, pH, temperature, and rainfall.

Soil type and soil moisture are not used by this model because their source units and training signal are not approved. The Test split remains sealed. All reported results use Validation.
"""
    ),
    markdown("## 1 Setup"),
    code(COMMON_SETUP),
    markdown("## 2 Frozen Data Split"),
    code(COMMON_DATA),
    code(COMMON_SPLIT_PLOT),
    markdown("## 3 Train Random Forest"),
    code(
        """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, top_k_accuracy_score

MODEL_NAME = "random_forest"
TOTAL_TREES = 400
TREE_STEP = 20

model = RandomForestClassifier(
    n_estimators=0,
    warm_start=True,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

history_rows = []
training_start = time.perf_counter()
progress = tqdm(
    range(TREE_STEP, TOTAL_TREES + 1, TREE_STEP),
    desc="Random Forest trees",
    unit="stage",
)
for tree_count in progress:
    model.set_params(n_estimators=tree_count)
    model.fit(X_train, y_train)
    stage_probabilities = model.predict_proba(X_validation)
    stage_loss = log_loss(
        y_validation, stage_probabilities, labels=np.arange(len(label_encoder.classes_))
    )
    stage_top_3 = top_k_accuracy_score(
        y_validation,
        stage_probabilities,
        k=3,
        labels=np.arange(len(label_encoder.classes_)),
    )
    elapsed = time.perf_counter() - training_start
    history_rows.append(
        {
            "iteration": tree_count,
            "training_loss": np.nan,
            "validation_loss": stage_loss,
            "validation_top_3": stage_top_3,
            "elapsed_seconds": elapsed,
        }
    )
    progress.set_postfix(top_3=f"{stage_top_3:.3f}", loss=f"{stage_loss:.3f}")

training_seconds = time.perf_counter() - training_start
training_history = pd.DataFrame(history_rows)
validation_probabilities = model.predict_proba(X_validation)
print("Random Forest training completed in", round(training_seconds, 3), "seconds")
display(training_history.tail())
"""
    ),
    markdown("## 4 Validation Evaluation"),
    code(COMMON_EVALUATION),
    markdown("## 5 Feature Importance"),
    code(
        """
importance = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
).sort_values("importance", ascending=True)

plt.figure(figsize=(9, 5))
sns.barplot(data=importance, x="importance", y="feature", color="#2f7d32")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 6 Save Validation Artifacts"),
    code(COMMON_SAVE),
    code(
        """
joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "data_contract_version": "prototype-six-feature-v1",
    },
    output_dir / "model.joblib",
)
print("Saved model", output_dir / "model.joblib")
"""
    ),
]


xgb_cells = [
    markdown(
        """
# XGBoost Crop Recommendation Prototype

This notebook trains a six-feature prototype using N, P, K, pH, temperature, and rainfall.

Soil type and soil moisture are not used by this model because their source units and training signal are not approved. The Test split remains sealed. All reported results use Validation.
"""
    ),
    markdown("## 1 Setup"),
    code(COMMON_SETUP),
    code(
        """
try:
    import xgboost as xgb
except Exception as error:
    raise RuntimeError(
        "XGBoost is unavailable. Install xgboost and the macOS libomp runtime before running."
    ) from error
print("XGBoost", xgb.__version__)
"""
    ),
    markdown("## 2 Frozen Data Split"),
    code(COMMON_DATA),
    code(COMMON_SPLIT_PLOT),
    markdown("## 3 Train XGBoost"),
    code(
        """
from sklearn.metrics import top_k_accuracy_score

MODEL_NAME = "xgboost"
MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 60

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
dvalidation = xgb.DMatrix(
    X_validation, label=y_validation, feature_names=FEATURES
)

class TqdmTrainingCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.progress = None

    def before_training(self, model):
        self.progress = tqdm(total=self.total_rounds, desc="XGBoost rounds", unit="round")
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.progress.update(1)
        validation_loss = evals_log["validation"]["mlogloss"][-1]
        self.progress.set_postfix(loss=f"{validation_loss:.4f}")
        return False

    def after_training(self, model):
        self.progress.close()
        return model

parameters = {
    "objective": "multi:softprob",
    "num_class": len(label_encoder.classes_),
    "eval_metric": "mlogloss",
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "lambda": 1.0,
    "alpha": 0.0,
    "seed": RANDOM_STATE,
    "nthread": -1,
}
evaluation_results = {}
training_start = time.perf_counter()
model = xgb.train(
    parameters,
    dtrain,
    num_boost_round=MAX_ROUNDS,
    evals=[(dtrain, "train"), (dvalidation, "validation")],
    evals_result=evaluation_results,
    callbacks=[
        TqdmTrainingCallback(MAX_ROUNDS),
        xgb.callback.EarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS,
            metric_name="mlogloss",
            data_name="validation",
            save_best=True,
        ),
    ],
    verbose_eval=False,
)
training_seconds = time.perf_counter() - training_start
best_round = int(getattr(model, "best_iteration", len(evaluation_results["validation"]["mlogloss"]) - 1)) + 1
validation_probabilities = model.predict(
    dvalidation, iteration_range=(0, best_round)
)

history_rows = []
history_progress = tqdm(
    range(10, best_round + 1, 10),
    desc="XGBoost history",
    unit="checkpoint",
)
for round_count in history_progress:
    checkpoint_probabilities = model.predict(
        dvalidation, iteration_range=(0, round_count)
    )
    checkpoint_top_3 = top_k_accuracy_score(
        y_validation,
        checkpoint_probabilities,
        k=3,
        labels=np.arange(len(label_encoder.classes_)),
    )
    history_rows.append(
        {
            "iteration": round_count,
            "training_loss": evaluation_results["train"]["mlogloss"][round_count - 1],
            "validation_loss": evaluation_results["validation"]["mlogloss"][round_count - 1],
            "validation_top_3": checkpoint_top_3,
            "elapsed_seconds": np.nan,
        }
    )
if not history_rows or history_rows[-1]["iteration"] != best_round:
    final_top_3 = top_k_accuracy_score(
        y_validation,
        validation_probabilities,
        k=3,
        labels=np.arange(len(label_encoder.classes_)),
    )
    history_rows.append(
        {
            "iteration": best_round,
            "training_loss": evaluation_results["train"]["mlogloss"][best_round - 1],
            "validation_loss": evaluation_results["validation"]["mlogloss"][best_round - 1],
            "validation_top_3": final_top_3,
            "elapsed_seconds": training_seconds,
        }
    )
training_history = pd.DataFrame(history_rows)
print("Best boosting round", best_round)
print("XGBoost training completed in", round(training_seconds, 3), "seconds")
display(training_history.tail())
"""
    ),
    markdown("## 4 Validation Evaluation"),
    code(COMMON_EVALUATION),
    markdown("## 5 Feature Importance"),
    code(
        """
gain = model.get_score(importance_type="gain")
importance = pd.DataFrame(
    {"feature": FEATURES, "importance": [gain.get(feature, 0.0) for feature in FEATURES]}
).sort_values("importance", ascending=True)

plt.figure(figsize=(9, 5))
sns.barplot(data=importance, x="importance", y="feature", color="#2f7d32")
plt.title("XGBoost Feature Importance by Gain")
plt.xlabel("Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 6 Save Validation Artifacts"),
    code(COMMON_SAVE),
    code(
        """
model.save_model(output_dir / "model.json")
joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
print("Saved model", output_dir / "model.json")
"""
    ),
]


nn_cells = [
    markdown(
        """
# Neural Network Crop Recommendation Prototype

This notebook trains a PyTorch multilayer perceptron using N, P, K, pH, temperature, and rainfall.

Soil type and soil moisture are not used by this model because their source units and training signal are not approved. The Test split remains sealed. All reported results use Validation.
"""
    ),
    markdown("## 1 Setup"),
    code(COMMON_SETUP),
    code(
        """
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import top_k_accuracy_score

torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(1)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("PyTorch", torch.__version__)
print("Device", DEVICE)
print("PyTorch CPU threads", torch.get_num_threads())
"""
    ),
    markdown("## 2 Frozen Data Split"),
    code(COMMON_DATA),
    code(COMMON_SPLIT_PLOT),
    markdown("## 3 Scale Numeric Features"),
    code(
        """
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_validation_scaled = scaler.transform(X_validation).astype(np.float32)

train_dataset = TensorDataset(
    torch.tensor(X_train_scaled), torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_STATE),
)
X_validation_tensor = torch.tensor(X_validation_scaled, device=DEVICE)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.long, device=DEVICE)
print("Scaler fitted on Train only")
"""
    ),
    markdown("## 4 Train Neural Network"),
    code(
        """
MODEL_NAME = "neural_network"
MAX_EPOCHS = 250
PATIENCE = 30

class CropMLP(nn.Module):
    def __init__(self, input_size, class_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, class_count),
        )

    def forward(self, inputs):
        return self.network(inputs)

model = CropMLP(len(FEATURES), len(label_encoder.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

best_state = None
best_validation_loss = float("inf")
epochs_without_improvement = 0
history_rows = []
training_start = time.perf_counter()

progress = tqdm(range(1, MAX_EPOCHS + 1), desc="Neural Network epochs", unit="epoch")
for epoch in progress:
    model.train()
    training_loss_sum = 0.0
    training_rows = 0
    for batch_features, batch_targets in train_loader:
        batch_features = batch_features.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()
        training_loss_sum += loss.item() * len(batch_features)
        training_rows += len(batch_features)

    model.eval()
    with torch.no_grad():
        validation_logits = model(X_validation_tensor)
        validation_loss = criterion(validation_logits, y_validation_tensor).item()
        stage_probabilities = torch.softmax(validation_logits, dim=1).cpu().numpy()
    validation_top_3 = top_k_accuracy_score(
        y_validation,
        stage_probabilities,
        k=3,
        labels=np.arange(len(label_encoder.classes_)),
    )
    training_loss = training_loss_sum / training_rows
    elapsed = time.perf_counter() - training_start
    history_rows.append(
        {
            "iteration": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_top_3": validation_top_3,
            "elapsed_seconds": elapsed,
        }
    )
    progress.set_postfix(
        top_3=f"{validation_top_3:.3f}", validation_loss=f"{validation_loss:.3f}"
    )

    if validation_loss < best_validation_loss - 1e-4:
        best_validation_loss = validation_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= PATIENCE:
        print("Early stopping at epoch", epoch)
        break

training_seconds = time.perf_counter() - training_start
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    validation_probabilities = torch.softmax(model(X_validation_tensor), dim=1).cpu().numpy()
training_history = pd.DataFrame(history_rows)
print("Neural Network training completed in", round(training_seconds, 3), "seconds")
display(training_history.tail())
"""
    ),
    markdown("## 5 Training Curves"),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(training_history["iteration"], training_history["training_loss"], label="Train")
axes[0].plot(training_history["iteration"], training_history["validation_loss"], label="Validation")
axes[0].set_title("Neural Network Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross Entropy Loss")
axes[0].legend()

axes[1].plot(
    training_history["iteration"],
    training_history["validation_top_3"],
    color="#2f7d32",
)
axes[1].set_ylim(0, 1.02)
axes[1].set_title("Neural Network Validation Top 3")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Top 3 Accuracy")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 6 Validation Evaluation"),
    code(COMMON_EVALUATION),
    markdown("## 7 Save Validation Artifacts"),
    code(COMMON_SAVE),
    code(
        """
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
        "classes": list(label_encoder.classes_),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "data_contract_version": "prototype-six-feature-v1",
    },
    output_dir / "model.pt",
)
joblib.dump(scaler, output_dir / "scaler.joblib")
joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
print("Saved model", output_dir / "model.pt")
"""
    ),
]


NOTEBOOKS = {
    "02_random_forest.ipynb": notebook(rf_cells),
    "03_xgboost.ipynb": notebook(xgb_cells),
    "04_neural_network.ipynb": notebook(nn_cells),
}


for filename, content in NOTEBOOKS.items():
    path = ROOT / filename
    path.write_text(json.dumps(content, indent=1), encoding="utf-8")
    print("Wrote", path)
