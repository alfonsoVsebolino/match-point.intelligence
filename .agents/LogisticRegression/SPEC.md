# Spec: Regularized Logistic Regression Match Outcome Prediction

## Problem Statement

To assess whether complex non-linear models add genuine predictive value in tennis match prediction, an empirical, probabilistic linear baseline is required. The model must ingest player delta features without data leakage, handle categorical surface/tournament encodings, address missing values, and output well-calibrated win probabilities across historical time periods.

## Solution

A production-grade scikit-learn pipeline wrapping `LogisticRegression` with L2 regularization, executed across an expanding-window chronological cross-validation framework. Continuous delta features are median-imputed and scaled using `RobustScaler`, categorical variables are one-hot encoded, and predictions are evaluated using Log-Loss and Brier Score benchmarked against bookmaker implied odds.

## User Stories

1. As a machine learning practitioner, I want to partition match features into 31 fundamental predictors while excluding metadata and target columns, so that the training matrix is clean and devoid of label leakage.
2. As a machine learning practitioner, I want bookmaker odds (`odds_diff`, `implied_prob_diff`) excluded from the feature matrix $X$, so that the model learns fundamental tennis signals rather than consensus odds reconstruction.
3. As a machine learning practitioner, I want market implied probabilities preserved as an external benchmark array, so that I can evaluate the model's Brier score relative to market accuracy.
4. As a machine learning practitioner, I want continuous delta features imputed and scaled via `ColumnTransformer`, so that large-magnitude deltas do not disproportionately penalize regularized coefficients.
5. As a machine learning practitioner, I want `surface_encoded` and `series_encoded` one-hot encoded, so that the linear model does not assume an arbitrary ordinal relationship between court surfaces or tournament tiers.
6. As a machine learning practitioner, I want `is_grand_slam` passed through as a binary indicator, so that tournament format differences are directly factored into linear log-odds.
7. As a machine learning practitioner, I want an expanding-window walk-forward validation harness across multi-season folds, so that model performance is evaluated strictly on future temporal data without chronological leakage.
8. As a machine learning practitioner, I want to tune the regularization strength parameter $C$ over validation folds, so that coefficients do not overfit to collinear rolling-window features.
9. As a machine learning practitioner, I want out-of-fold probability predictions evaluated via Log-Loss and Brier Score, so that calibration quality is prioritized over uncalibrated binary accuracy.
10. As a machine learning practitioner, I want calibration curve diagnostics generated for out-of-fold predictions, so that probability distortion across confidence deciles can be visualized.
11. As a machine learning practitioner, I want final holdout evaluation executed against `test_features.csv` (2024–2026), so that true generalization on unseen future seasons is measured.

## Implementation Decisions

- **Feature Partitioning**:
  - Metadata dropped: `Date`, `Tournament`, `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`.
  - Target isolated: `target`.
  - External benchmark isolated: `implied_prob_diff`.
- **Pipeline Architecture**:
  - Encapsulated within `sklearn.pipeline.Pipeline`.
  - Preprocessor implemented via `sklearn.compose.ColumnTransformer`.
  - Numeric branch: `SimpleImputer(strategy='median')` followed by `RobustScaler()`.
  - Categorical branch: `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to `surface_encoded` and `series_encoded`.
  - Binary passthrough: `is_grand_slam`.
  - Estimator: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)`.
- **Validation Scheme**:
  - 4 walk-forward chronological splits over 2000–2023 train set.
  - Preprocessor fitted strictly on training folds and applied to validation/test folds.
- **Evaluation Seam**:
  - Output format: Matrix of predicted probabilities $P(y=1)$.
  - Primary evaluation functions: `log_loss` and `brier_score_loss`.

## Testing Decisions

- **Seam**: High-level execution seam at pipeline evaluation function level (`evaluate_pipeline(pipeline, cv_splits, X, y)`).
- **Good Test Criteria**:
  - Input-output contract tests: Pipeline accepts DataFrame and returns 1D array of probabilities bounded strictly in $[0, 1]$.
  - Data leakage tests: Imputer and scaler parameters computed exclusively on train indices.
  - Determinism tests: Fixed random seeds yield identical fold scores.
  - Monotonicity test: Positive delta in player skill/points corresponds to higher predicted win probability.

## Out of Scope

- Tree-based boosting or non-linear feature interaction modeling.
- In-training bookmaker odds usage.
- Post-hoc non-linear probability calibration (Platt/Isotonic scaling) prior to baseline diagnosis.

## Further Notes

- Serves as the primary linear reference benchmark against which the LightGBM model will be compared.
