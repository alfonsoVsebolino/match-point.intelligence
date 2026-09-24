---
id: LGBM-002
title: LightGBM Classifier Configuration and Categorical Handling
blocked_by: [LGBM-001]
status: ready-for-agent
---

# LGBM-002: LightGBM Classifier Configuration and Categorical Handling

## Objective
Instantiate and configure `LGBMClassifier` with categorical feature support and constrained hyperparameters for tennis match outcome prediction.

## Requirements
- Set objective: `binary`
- Set metric: `binary_logloss`
- Set categoricals: `categorical_feature=['surface_encoded', 'series_encoded', 'is_grand_slam']`
- Initial baseline hyperparameters:
  - `learning_rate`: 0.03
  - `num_leaves`: 31
  - `min_child_samples`: 30
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `n_estimators`: 1000
- Validate smoke training on single fold without errors on NaNs.
