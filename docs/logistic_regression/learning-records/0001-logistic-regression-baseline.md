# Learning Record 0001: Regularized Logistic Regression Baseline

**Date**: 2026-09-12  
**Context**: Establishing a linear baseline for ATP tennis match prediction (2000–2026).

## Key Non-Obvious Insights

### 1. Residuals Are NOT Squared in Logistic Regression
- **Misconception**: Residuals of feature differences are squared.
- **Reality**: Binary outcomes follow a Bernoulli distribution. The loss function is Binary Cross-Entropy (Negative Log-Likelihood) derived via MLE.
- **What is Squared**: The coefficient weights $\|w\|_2^2 = \sum w_j^2$ in the L2 penalty, not the prediction residuals.

### 2. The $C$ Hyperparameter Inversion
- In Scikit-learn, $C = \frac{1}{\lambda}$.
- Smaller $C \implies$ heavier penalty / more shrinkage $\implies$ simpler model (lower variance, higher bias).
- Optimal $C=0.01$ prevented exploding coefficients among collinear rolling-window tennis differentials (e.g., `short_winrate_diff`, `med_winrate_diff`, `career_winrate_diff`).

### 3. Scaling Is Non-Negotiable for Regularized Linear Models
- Tree-based models (LightGBM) are scale-invariant because splits are monotonic.
- Linear models with L2 regularization compute isotropic penalties $\sum w_j^2$. Unscaled features with large absolute ranges (e.g. ATP points $[-5000, 5000]$ vs win rate $[-1, 1]$) force the optimizer to shrink the wrong coefficients.
- `RobustScaler` (median + IQR) is superior to `StandardScaler` (mean + std) because ranking differentials contain extreme outlier mismatches (e.g., World No. 1 vs Wildcard No. 800).

### 4. Calibration & Log-Loss Outrank Raw Accuracy
- A model with 65% accuracy can have poor betting/risk utility if its probabilities are distorted (e.g. predicting 90% when true win rate is 65%).
- Out-of-fold calibration diagnostics and Brier score benchmark against bookmaker consensus ($0.2156$ vs $0.2021$) define real predictive utility.
