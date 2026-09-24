# Model Development Plan: LightGBM vs. Regularized Logistic Regression

A structured blueprint for manual implementation, model tuning, and evaluation against temporal holdouts.

---

## 1. Architectural Stack & Imports

Key packages required for the pipeline:

* **Data Handling**: `pandas`, `numpy`
* **Linear Pipeline**:
  * `sklearn.linear_model.LogisticRegression`
  * `sklearn.preprocessing.RobustScaler`, `sklearn.preprocessing.OneHotEncoder`
  * `sklearn.impute.SimpleImputer`
  * `sklearn.compose.ColumnTransformer`
  * `sklearn.pipeline.Pipeline`
* **Gradient Boosting**:
  * `lightgbm.LGBMClassifier` (or native `lightgbm.train`)
* **Time-Series Validation**:
  * Custom chronological season mask or `sklearn.model_selection.TimeSeriesSplit`
* **Calibration & Metrics**:
  * `sklearn.metrics.log_loss`, `sklearn.metrics.brier_score_loss`, `sklearn.metrics.roc_auc_score`, `sklearn.metrics.accuracy_score`
  * `sklearn.calibration.calibration_curve`
* **Plotting / Diagnostics**: `matplotlib.pyplot`

---

## 2. Feature & Target Partitioning

### Dropped from Feature Matrix $X$
* **Metadata**: `Date`, `Tournament`, `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`
* **Target ($y$)**: `target`
* **External Market Baseline**: `odds_diff`, `implied_prob_diff` (held out strictly to benchmark bookmaker Brier/loss)

### Ingested Features (31 Fundamental Predictors)
* **Continuous Numeric Deltas (28 features)**: Rank, points, rolling win rates (short/med/career), comeback rates, bagel/breadstick rates, games/set, retirement freq, surface win rate diff, fatigue counters, form divergence, H2H stats.
* **Categorical / Flags (3 features)**: `surface_encoded`, `series_encoded`, `is_grand_slam`.

---

## 3. Validation Framework (Expanding Window CV)

Structure walk-forward time splits on `train_features.csv` (2000–2023) using the `Date` column:

```text
Fold 1: Train [2000–2016]  →  Val [2017–2018]
Fold 2: Train [2000–2018]  →  Val [2019–2020]
Fold 3: Train [2000–2020]  →  Val [2021–2022]
Fold 4: Train [2000–2022]  →  Val [2023]
Final Test: Train [2000–2023]  →  Test [2024–2026] (test_features.csv)
```

* Zero future-to-past leakage.
* Strictly chronological (never shuffle rows).

---

## 4. Dual-Track Modeling Pipeline

### Track 1: Regularized Logistic Regression (Linear Baseline)
* **Preprocessing (`ColumnTransformer`)**:
  * Numeric deltas: `SimpleImputer(strategy='median')` (or 0.0) → `RobustScaler()`.
  * Categoricals (`surface_encoded`, `series_encoded`): `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  * Flags (`is_grand_slam`): Passthrough.
* **Estimator**: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)`.
* **Tuning Grid**: Regularization parameter $C \in [10^{-4}, 10^2]$.

### Track 2: LightGBM (Non-Linear Champion)
* **Preprocessing**: None (pass raw $X$ with native NaNs).
* **Categorical Handling**: Specify `categorical_feature=['surface_encoded', 'series_encoded', 'is_grand_slam']`.
* **Tuning Parameters**:
  * `n_estimators`: 100–1500 (with `early_stopping_rounds=50` on validation fold).
  * `learning_rate`: 0.01–0.05.
  * `num_leaves`: 15–63.
  * `min_child_samples`: 20–100.
  * `colsample_bytree` / `subsample`: 0.7–0.9.
* **Objective**: `binary`, eval metric `binary_logloss`.

---

## 5. Evaluation & Diagnostics Protocol

For each validation fold and the final test set:

1. **Probabilistic Calibration**:
   * **Log-Loss**: `log_loss(y_val, y_prob[:, 1])`
   * **Brier Score**: `brier_score_loss(y_val, y_prob[:, 1])`
2. **Discrimination Reference**:
   * ROC-AUC and Accuracy at 0.5 threshold.
3. **Market Benchmark Comparison**:
   * Brier score of bookmaker implied probability: $\text{Brier}_{\text{odds}} = \frac{1}{N} \sum (p_{\text{implied}} - y)^2$.
4. **Reliability Curves**:
   * Plot `calibration_curve(y_val, y_prob[:, 1], n_bins=10)` against diagonal ideal line.

---

## 6. Expected Outcomes & Benchmarks

* **Non-linear Gain**: LightGBM Log-Loss improvement $\Delta \approx -0.01$ to $-0.03$ over Logistic Regression.
* **Brier Target**: Competitive tennis models typically hit `0.185 – 0.205`.
* **Symmetry Property**: P1 vs P2 inversion should yield consistent probabilities ($P(P_1) \approx 1 - P(P_2)$).
