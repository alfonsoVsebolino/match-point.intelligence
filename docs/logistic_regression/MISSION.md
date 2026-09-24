# Learning Mission: Probabilistic Linear Baselines in Sports Prediction

## Objective
To master the inner workings, mathematical derivations, parameter dynamics, optimization routines, and empirical diagnostics of regularized logistic regression when applied to high-dimensional sports prediction tasks.

## Why This Matters
1. **Linear Baseline Integrity**: Before accepting non-linear models (e.g. LightGBM), practitioners must evaluate whether complex tree interactions provide authentic predictive lift over a properly scaled, calibrated linear log-odds model.
2. **Probabilistic Calibration**: In match forecasting and market benchmarking, binary accuracy is secondary to probability calibration and Brier score.
3. **Data Leakage & Scaling**: Understanding how preprocessing (`ColumnTransformer`, `RobustScaler`, temporal fold partitioning) interacts with linear loss surfaces.

## Scope of Mastery
- [x] Logit link function & sigmoid probability derivation.
- [x] Maximum Likelihood Estimation & Log-loss vs MSE.
- [x] L2 regularization mechanics and the $C$ hyperparameter trade-off.
- [x] L-BFGS quasi-Newton solver mechanics.
- [x] Robust scaling and outlier mitigation for collinear delta predictors.
- [x] Expanding-window cross-validation and calibration diagnostics.
