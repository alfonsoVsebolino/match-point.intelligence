---
id: LR-002
title: ColumnTransformer Preprocessing Pipeline
blocked_by: [LR-001]
status: completed
---

# LR-002: ColumnTransformer Preprocessing Pipeline

## Objective
Build a scikit-learn `Pipeline` integrating `ColumnTransformer` to handle imputation, robust scaling, and categorical encoding for Logistic Regression.

## Requirements
- Numeric deltas (28 features): `SimpleImputer(strategy='median')` followed by `RobustScaler()`
- Categorical features (`surface_encoded`, `series_encoded`): `OneHotEncoder(drop='first', handle_unknown='ignore')`
- Binary flag (`is_grand_slam`): `passthrough`
- Estimator: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)`
- Test fit-transform pipeline on dummy slice to verify output dimensions and finiteness.
