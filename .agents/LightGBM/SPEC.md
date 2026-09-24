# Spec: LightGBM Match Outcome Prediction

## Problem Statement

Tennis match outcomes depend on complex, non-linear interactions across player differences (e.g., fatigue impact on older vs younger players, surface suitability divergence, form momentum). A high-capacity non-linear model is needed to capture these higher-order relationships natively without manual feature interaction engineering, while maintaining probabilistic calibration and zero future data leakage.

## Solution

A gradient-boosted decision tree pipeline using `lightgbm.LGBMClassifier` trained directly on fundamental player deltas. The model handles missing feature values natively without imputation, preserves numeric delta scales without normalization, utilizes native categorical split algorithms, and minimizes binary log-loss with early stopping over an expanding-window chronological validation harness.

## User Stories

1. As a machine learning practitioner, I want to ingest the 31 fundamental delta features with missing values preserved as native NaNs, so that LightGBM can optimize split directions for missingness patterns directly.
2. As a machine learning practitioner, I want metadata columns (`Date`, `Tournament`, `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`) and target dropped from $X$, so that no identity or label leakage occurs.
3. As a machine learning practitioner, I want bookmaker odds (`odds_diff`, `implied_prob_diff`) excluded from training features, so that tree splits reflect underlying tennis dynamics rather than betting market consensus.
4. As a machine learning practitioner, I want market implied probabilities recorded as an external benchmark array, so that model performance can be compared against bookmaker Brier scores.
5. As a machine learning practitioner, I want `surface_encoded`, `series_encoded`, and `is_grand_slam` designated as native categorical features, so that optimal categorical subsets are evaluated during tree construction.
6. As a machine learning practitioner, I want an expanding-window chronological CV setup matching the linear baseline, so that performance metrics are strictly comparable across identical validation folds.
7. As a machine learning practitioner, I want early stopping configured against validation fold log-loss (`early_stopping_rounds=50`), so that boosting halts before overfitting noisy match outcomes.
8. As a machine learning practitioner, I want leaf complexity and sample size constrained (`num_leaves=15-63`, `min_child_samples=20-100`), so that trees do not memorize small sample match idiosyncrasies.
9. As a machine learning practitioner, I want out-of-fold probability predictions evaluated via Log-Loss, Brier Score, and ROC-AUC, so that probabilistic quality is comprehensively tracked.
10. As a machine learning practitioner, I want calibration curves plotted across probability bins, so that probability compression or overconfidence in tail regions can be diagnosed.
11. As a machine learning practitioner, I want final holdout evaluation executed against `test_features.csv` (2024–2026), so that out-of-sample performance on modern seasons is quantified.
12. As a machine learning practitioner, I want feature importance metrics (gain and split count) extracted, so that the dominant non-linear drivers of match outcomes can be inspected.

## Implementation Decisions

- **Feature Partitioning**:
  - Excluded from $X$: Metadata columns, `target`, and betting odds (`odds_diff`, `implied_prob_diff`).
  - Categoricals explicitly flagged: `['surface_encoded', 'series_encoded', 'is_grand_slam']`.
  - Continuous features passed in raw form (no imputation, no scaling).
- **Model Architecture**:
  - Core estimator: `lightgbm.LGBMClassifier` (or `lightgbm.train`).
  - Objective: `binary`.
  - Metric: `binary_logloss`.
  - Regularization & Tree structure: `learning_rate` $\in [0.01, 0.05]$, `subsample` $\in [0.7, 0.9]$, `colsample_bytree` $\in [0.7, 0.9]$.
- **Validation Scheme**:
  - Identical 4 walk-forward expanding window folds over 2000–2023.
  - Evaluation fold used strictly for early stopping and out-of-fold scoring.
- **Evaluation Seam**:
  - Probability vector $P(y=1)$ computed via `predict_proba[:, 1]`.
  - Metrics: Log-Loss, Brier Score, ROC-AUC.

## Testing Decisions

- **Seam**: Top-level training and cross-validation orchestrator (`train_and_evaluate_lgbm(cv_splits, X, y)`).
- **Good Test Criteria**:
  - Probabilistic bounds test: All predicted probabilities fall strictly within $[0, 1]$.
  - Native NaN resilience test: Synthetic batch containing NaNs computes valid probabilities without error.
  - Categorical parameter test: Model verifies categorical indices match target columns.
  - Convergence test: Log-loss monotonically decreases on training set during initial boosting rounds.

## Out of Scope

- Pre-training data scaling or imputation.
- Market odds inclusion inside $X$.
- Post-hoc probability calibration (Platt/Isotonic) prior to baseline evaluation.
- Stacking or ensembling with Logistic Regression.

## Further Notes

- Primary champion model expected to surpass the Logistic Regression baseline if non-linear delta interactions provide predictive edge.
