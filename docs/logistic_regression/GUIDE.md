# Educational Reference Guide: Regularized Logistic Regression for Tennis Match Prediction

An educational deep-dive into the probabilistic linear classifier implemented in `eda_and_feature_engineering.ipynb`. This document details the mathematical theory, parameter mechanics, optimization dynamics, feature transformations, and empirical validation scheme.

---

## 1. Core Concept & Probabilistic Formulation

### 1.1 From Linear Regression to Logistic Regression
In standard ordinary least squares (OLS) linear regression, a linear combination of input features $x \in \mathbb{R}^d$ models a continuous response variable:
$$\hat{y} = w^T x + b = \sum_{j=1}^d w_j x_j + b$$
Because $w^T x + b \in (-\infty, +\infty)$, this formulation cannot directly represent a probability $P(y=1|x) \in [0, 1]$.

Logistic Regression bridges this using the **logit link function**. It models the **log-odds** of the binary event $y = 1$ (Player 1 win) as a linear function:
$$\ln \left( \frac{p}{1 - p} \right) = z = w^T x + b$$
Where:
- $p = P(y=1|x)$: Probability that Player 1 wins the match.
- $\frac{p}{1-p}$: Odds ratio (probability of winning divided by probability of losing).
- $\ln\left(\frac{p}{1-p}\right)$: Log-odds or **logit**.

Solving for $p$ yields the **sigmoid / standard logistic activation function** $\sigma(z)$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(w^T x + b)}}$$

### 1.2 Properties of the Sigmoid Function
- $\lim_{z \to +\infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma(0) = 0.5$ (decision threshold for balanced classes)
- Derivative property: $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$

---

## 2. Objective Function: Why Residuals Are NOT Squared

### 2.1 Derivation from Maximum Likelihood Estimation (MLE)
For binary match outcomes $y_i \in \{0, 1\}$, each outcome is modeled as an independent Bernoulli random variable:
$$P(y_i | x_i) = p_i^{y_i} (1 - p_i)^{1 - y_i}$$
The joint likelihood $\mathcal{L}(w, b)$ over a training dataset of $n$ matches is:
$$\mathcal{L}(w, b) = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}$$
Taking the natural logarithm yields the **log-likelihood**:
$$\ln \mathcal{L}(w, b) = \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### 2.2 Binary Cross-Entropy (Log-Loss)
To frame this as a minimization problem, we negate the log-likelihood and normalize by sample size $n$:
$$J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### 2.3 Why Not Squared Residuals $(y_i - p_i)^2$?
1. **Non-Convexity**: Plugging $p_i = \sigma(w^T x_i + b)$ into Mean Squared Error $\frac{1}{n}\sum (y_i - p_i)^2$ creates a non-convex loss surface with many local minima and plateaus.
2. **Vanishing Gradients**: If the model makes a confident wrong prediction (e.g., $y=1, p=0.001$), the derivative of MSE includes $\sigma'(z) = p(1-p) \approx 0.001$, stalling gradient updates.
3. **Log-Loss Steepness**: With log-loss, the gradient with respect to logit $z_i$ simplifies cleanly to:
   $$\frac{\partial J}{\partial z_i} = p_i - y_i$$
   A severe misprediction ($y_i=1, p_i \approx 0 \implies p_i - y_i \approx -1$) yields a maximal restoring gradient.

---

## 3. L2 Regularization & The $C$ Hyperparameter

### 3.1 The Regularized Objective
Collinear rolling statistics (e.g., short-term vs medium-term vs career win rates) can cause linear coefficients to destabilize and grow arbitrarily large in opposite directions. To enforce numerical stability and generalizability, **L2 regularization (Ridge penalty)** is added:

$$\min_{w, b} \left\{ \sum_{i=1}^n \left[ -y_i \ln(p_i) - (1 - y_i)\ln(1 - p_i) \right] + \frac{1}{2C} \sum_{j=1}^d w_j^2 \right\}$$

*(Note: Scikit-learn minimizes the sum of losses rather than the mean, parameterized by $C$. The intercept $b$ is unregularized.)*

### 3.2 What the $C$ Parameter Controls
In Scikit-learn, $C$ is the **inverse regularization strength**:
$$C = \frac{1}{\lambda}$$

| Setting | Regularization Strength | Penalty Weight $\frac{1}{2C}$ | Coefficient Magnitudes $\|w\|_2$ | Model Bias | Model Variance | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small $C$** ($10^{-4}, 10^{-3}$)| Heavy | High | Strongly shrunk $\to 0$ | High | Low | Underfitting |
| **Optimal $C$** ($10^{-2} = 0.01$)| Balanced | Moderate | Penalized collinearity | Optimal | Optimal | Best validation log-loss |
| **Large $C$** ($10^1, 10^2$)| Negligible | Near 0 | Unconstrained | Low | High | Overfitting to sample noise |

### 3.3 Geometric Interpretation
The penalty $\sum_{j=1}^d w_j^2 \le r^2$ restricts the solution vector $w$ to an $L_2$ Euclidean hypersphere. The optimal regularized weight vector occurs at the tangency point between the log-loss elliptical contours and the spherical constraint boundary.

---

## 4. Parameters vs Hyperparameters

Understanding the distinction is essential for configuring and auditing the model:

### 4.1 Hyperparameters (Configured before training)
Configured by the engineer and tuned via validation:
- `C = 0.01`: Regularization trade-off weight.
- `penalty = 'l2'`: Norm used for weight shrinkage.
- `solver = 'lbfgs'`: Optimization algorithm.
- `max_iter = 1000`: Maximum iterations for optimizer convergence.
- `imputer.strategy = 'median'`: Imputation policy for missing feature deltas.
- `scaler = RobustScaler()`: Scaling strategy centering on median and scaling by IQR.
- `ohe.drop = 'first'`: Dropping baseline categorical dummy to avoid collinearity.

### 4.2 Learned Parameters (Optimized during training)
Determined mathematically by the optimizer from training data:
- **Weights ($w \in \mathbb{R}^{38}$)**: One scalar coefficient per transformed feature column.
  - A positive weight $w_j > 0$ indicates that as feature $x_j$ increases, Player 1's odds of winning increase:
    $$\Delta \text{Log-Odds} = w_j \cdot \Delta x_j$$
    $$\text{Odds Multiplier} = e^{w_j}$$
- **Intercept ($b \in \mathbb{R}$)**: The baseline log-odds of Player 1 winning when all standardized features equal zero (empirically close to 0 in balanced head-to-head match framing).

---

## 5. Solver Mechanics: L-BFGS

The parameter `solver='lbfgs'` stands for **Limited-memory Broyden–Fletcher–Goldfarb–Shanno**:
1. **Quasi-Newton Method**: Unlike standard gradient descent (which only uses first-order derivatives $\nabla J$), Newton's method uses the curvature (Hessian matrix $H = \nabla^2 J$) to take quadratic steps toward the minimum.
2. **Limited Memory**: Storing an exact $38 \times 38$ Hessian over 60,000 samples is computationally demanding. L-BFGS approximates the inverse Hessian using only the gradient vectors from the previous $m$ iterations (default $m=10$).
3. **Convergence**: Converges significantly faster than vanilla SGD or coordinate descent on smooth convex objectives like L2-regularized logistic regression.

---

## 6. Preprocessing Architecture: `ColumnTransformer`

Because regularized logistic regression computes penalties directly on coefficient magnitudes $w_j^2$, **unscaled features will distort the model**. If one feature is measured in ATP points ($[-5000, +5000]$) and another in win rates ($[-1.0, +1.0]$), the model would disproportionately shrink the win-rate coefficient.

```
Input Features (31 Columns)
 ├── 28 Continuous Deltas ──> SimpleImputer(median) ──> RobustScaler() ──> [28 scaled features]
 ├── 2 Categorical Encodings ──────────────────────────> OneHotEncoder(drop='first') ──> [9 dummy features]
 └── 1 Binary Flag (is_grand_slam) ────────────────────> Passthrough ───> [1 binary feature]
                                                                                │
                                                                                ▼
                                                              Concatenated Matrix (38 Features)
                                                                                │
                                                                                ▼
                                                                 LogisticRegression(C=0.01)
```

### 6.1 Transformations by Feature Group
1. **Continuous Deltas (28 features)**:
   - Imputation: Median replaces NaNs (e.g. players without previous tiebreaks).
   - Scaling: `RobustScaler`:
     $$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)} = \frac{x - Q_2}{Q_3 - Q_1}$$
     *Why RobustScaler over StandardScaler?* Tennis stats exhibit extreme outliers (e.g., massive ranking differentials between top-5 seeds and wildcards). RobustScaler prevents extreme outliers from compressing the interquartile range.
2. **Categorical Features (2 features: `surface_encoded`, `series_encoded`)**:
   - `OneHotEncoder(drop='first', handle_unknown='ignore')`:
     - Converts ordinal integers into distinct binary flags.
     - `drop='first'` prevents the **dummy variable trap** (perfect linear dependence among one-hot columns).
3. **Binary Passthrough (1 feature: `is_grand_slam`)**:
   - Already binary $\{0, 1\}$; requires neither scaling nor encoding.

---

## 7. Cross-Validation & Validation Harness

### 7.1 Temporal Expanding-Window Scheme
Random k-fold CV causes **lookahead leakage** (future match statistics informing predictions on past matches). To maintain strict chronological causality:
- **Fold 1**: Train 2000–2016 $\to$ Validate 2017–2018
- **Fold 2**: Train 2000–2018 $\to$ Validate 2019–2020
- **Fold 3**: Train 2000–2020 $\to$ Validate 2021–2022
- **Fold 4**: Train 2000–2022 $\to$ Validate 2023
- **Final Holdout Test**: Evaluate on completely unseen future seasons 2024–2026 (`test_features.csv`).

---

## 8. Diagnostic Metrics & Evaluation Theory

### 8.1 Log-Loss (Negative Log-Likelihood)
$$-\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right]$$
- Penalizes heavily when high confidence is assigned to the incorrect player.
- Primary metric for selecting $C$.

### 8.2 Brier Score
$$\text{Brier} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2$$
- Mean squared error of probabilities bounded in $[0, 1]$.
- A score of $0.25$ corresponds to random 50/50 guessing.
- Allows direct benchmarking against Bookmaker implied probabilities $p_{\text{market}} = \frac{\text{implied\_prob\_diff} + 1}{2}$.

### 8.3 Calibration Curve (Reliability Diagram)
- Bins predicted probabilities into deciles ($0.0-0.1, 0.1-0.2, \dots, 0.9-1.0$).
- Compares mean predicted probability in each bin against the empirical win rate of matches in that bin.
- Proximity to the $45^\circ$ diagonal demonstrates whether predicted 70% win probabilities actually win 70% of the time.

### 8.4 ROC Curve & AUC
- Evaluates classification ranking across all decision thresholds $\tau \in [0, 1]$.
- **True Positive Rate (Sensitivity)**: $\frac{TP}{TP + FN}$
- **False Positive Rate (1 - Specificity)**: $\frac{FP}{FP + TN}$
- **AUC (Area Under Curve)**: Probability that the model ranks a randomly chosen winner higher than a randomly chosen loser.
  - $\text{AUC} = 0.50$: Random coin toss.
  - $\text{AUC} = 0.7120$: Model performance on unseen 2024–2026 holdout.

---

## 9. Key Empirical Results Summary

| Model / Benchmark | OOF Log-Loss | OOF Brier Score | Test Log-Loss (2024–26) | Test Brier Score | Test ROC-AUC | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regularized Logistic Regression ($C=0.01$)** | **0.6195** | **0.2156** | **0.6205** | **0.2159** | **0.7120** | **64.95%** |
| **Bookmaker Implied Odds** | — | **0.2021** | — | **0.2016** | — | ~68.0% |
| **LightGBM Champion** | 0.6178 | 0.2148 | 0.6190 | 0.2152 | 0.7150 | 65.20% |
