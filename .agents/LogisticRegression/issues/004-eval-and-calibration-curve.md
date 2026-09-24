---
id: LR-004
title: Evaluation, Calibration Diagnostics & Holdout Scoring
blocked_by: [LR-003]
status: completed
---

# LR-004: Evaluation, Calibration Diagnostics & Holdout Scoring

## Objective
Compute out-of-fold calibration diagnostics, benchmark against bookmaker implied probabilities, and evaluate the final tuned model on the 2024–2026 test set.

## Requirements
- Plot calibration curve (`calibration_curve(y_val, y_prob, n_bins=10)`) vs ideal diagonal
- Compute benchmark comparison:
  - Model Log-Loss & Brier Score vs Bookmaker Implied Probability Brier Score
- Retrain pipeline with optimal $C$ on complete historical dataset (2000–2023)
- Generate predictions on `test_features.csv` (2024–2026) and log final holdout metrics.
