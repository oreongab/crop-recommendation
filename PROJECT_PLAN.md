# Crop Recommendation System Project Plan

## Current readiness

The raw, audit, and cleaned data layers are complete and traceable. Exact duplicates are removed only from cleaned copies. Units are not converted without evidence. Crop labels are mapped with explicit rules. Feature-only fingerprints expose duplicate inputs and conflicting crop labels.

No current source is approved for a production model using all eight requested inputs. `Crop_Recommendation.csv` is approved only as a six-feature prototype source using N, P, K, pH, temperature, and rainfall.

The six-feature data contract, frozen Train Validation Test split, shared evaluation code, and three model notebooks are now prepared. The Test split remains sealed while the three models are compared on Validation.

## Phase 1 Data contract decision

Define the exact website contract before splitting data:

- N, P, and K measurement meaning and unit.
- Soil moisture measurement depth, method, and unit.
- Rainfall time window and unit.
- Temperature time window and unit.
- Supported soil type vocabulary.
- Supported crop target vocabulary.
- Required behavior for missing or out-of-range inputs.

Deliverables:

- `config/data_contract.json`
- `docs/data_card.md`
- Input validation rules shared by notebooks, API, and website.

Approval gate: do not create a split until the input units match the selected training source.

## Phase 2 Modeling strategy decision

Recommended V1:

- Train a six-feature prototype model from `Crop_Recommendation.csv`.
- Accept soil type and soil moisture in the website but use them only through a trusted compatibility layer.
- Do not invent weighted scores for soil type or moisture.
- Keep the full eight-feature model blocked until measured, source-documented data is available.

Alternative production path:

- Collect or acquire a complete eight-feature dataset with location, time, units, and measurement provenance.
- Re-run the same cleaning and leakage gates before training.

Approval gate: choose prototype hybrid or wait for complete measured data.

## Phase 3 Frozen split design

- Exclude non-specific crop targets.
- Exclude unresolved identical-input conflicting-label groups.
- Group by `feature_fingerprint` so identical inputs cannot cross splits.
- Use stratified group-aware Train 70, Validation 15, and Test 15 allocation.
- Check every crop has enough independent fingerprint groups for all splits.
- Fit transformations only on Train.
- Freeze one split manifest for every model.

Deliverables:

- `processed_dataset/model_ready/six_feature_prototype_v1/split_manifest.csv`
- `processed_dataset/model_ready/six_feature_prototype_v1/train.csv`
- `processed_dataset/model_ready/six_feature_prototype_v1/validation.csv`
- `processed_dataset/model_ready/six_feature_prototype_v1/test.csv`
- Split integrity report.

Approval gate: verify zero fingerprint overlap, zero label conflict, and acceptable class support.

## Phase 4 Shared preprocessing pipeline

- Validate numeric ranges without silently clipping values.
- Require complete prototype inputs instead of default imputation.
- Encode crop labels once and save the class order.
- Encode soil type only when a trusted full model becomes available.
- Scale numeric fields for Neural Network only.
- Keep Random Forest and XGBoost preprocessing logically equivalent.

Deliverables:

- `config/data_contract.json`
- `src/modeling.py`
- Saved label encoder and feature schema.
- Unit tests for transformations.

## Phase 5 Baseline and three model experiments

Use separate notebooks but the same split and evaluation code:

- `02_random_forest.ipynb`
- `03_xgboost.ipynb`
- `04_neural_network.ipynb`
- `src/modeling.py`

Shared metrics:

- Top 1 Accuracy
- Top 3 Accuracy
- Macro F1
- Weighted F1
- Balanced Accuracy
- Multiclass Log Loss
- Calibration Error
- Per Class Precision Recall F1
- Normalized Confusion Matrix
- Learning Curve
- Confidence Distribution

Random Forest and XGBoost use probability calibration on Validation when needed. Neural Network uses early stopping based only on Validation.

Approval gate: select the model using Top 3 performance, class balance, calibration, and robustness rather than accuracy alone.

## Phase 6 Final test and model card

- Lock the selected hyperparameters.
- Run Test once after model selection.
- Compare against frequency and simple-rule baselines.
- Document unsupported crops, input ranges, and known dataset limitations.
- Reject deployment if performance comes mainly from synthetic or source-specific patterns.

Deliverables:

- `artifacts/model/`
- `artifacts/metrics/final_test_metrics.json`
- `artifacts/figures/`
- `docs/model_card.md`

## Phase 7 Top three hybrid recommendation logic

- Generate calibrated crop probabilities from the selected model.
- Retrieve more than three model candidates internally.
- Apply trusted soil type and soil moisture compatibility tiers.
- Preserve model order within the same compatibility tier.
- Return the first three compatible crops.
- Return an insufficient-confidence message when three defensible crops are unavailable.
- Attach reason codes and input-range warnings.

Deliverables:

- `src/recommender.py`
- Compatibility reference table with provenance.
- Unit tests for ranking and edge cases.

## Phase 8 API

Recommended backend: FastAPI.

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /recommend`

The API validates the same data contract as training and returns:

- Top three crops
- Calibrated probabilities
- Compatibility status
- Reason codes
- Model version
- Input warnings

Deliverables:

- `app/api.py`
- API schema and tests
- Docker configuration

## Phase 9 Website

One-page user flow:

- Soil type selector
- pH input
- N, P, and K inputs with explicit units
- Soil moisture input with explicit unit and measurement note
- Temperature input
- Rainfall input with time-window note
- Top three recommendation cards
- Confidence and suitability explanation
- Out-of-distribution warning
- Clear disclaimer for prototype limitations

Approval gate: verify every form label and unit matches `data_contract.yaml`.

## Phase 10 Testing and validation

- Data reconciliation tests
- Raw hash tests
- Split leakage tests
- Preprocessing parity tests
- Model artifact loading tests
- API contract tests
- Recommendation ranking tests
- Website end-to-end tests
- Invalid and out-of-range input tests
- Reproducibility test from raw data to final prediction

## Phase 11 Deployment and monitoring

- Containerize API and website.
- Version model, schema, crop mapping, and compatibility table together.
- Log input range, prediction confidence, model version, and warnings.
- Do not log personal or precise farm-location data unless explicitly required.
- Monitor input drift, confidence drift, crop distribution, API failures, and latency.
- Define rollback and retraining procedures.

## Phase 12 Documentation and handoff

- Project README with exact run order.
- Data card and source limitations.
- Model card and evaluation results.
- API examples.
- Deployment instructions.
- Known limitations and future data collection plan.

## Required execution order

1. Approve data contract and modeling strategy.
2. Build and verify the frozen split.
3. Build shared preprocessing and evaluation code.
4. Train Random Forest, XGBoost, and Neural Network separately.
5. Select and test one model.
6. Build the compatibility and Top 3 recommendation layer.
7. Build API and website.
8. Run full testing and deployment checks.
9. Deploy with monitoring and documentation.

Each phase requires analysis and approval before implementation.
