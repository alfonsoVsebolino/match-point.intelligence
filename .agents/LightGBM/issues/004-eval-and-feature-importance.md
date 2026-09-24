---
id: LGBM-004
title: Evaluation, Calibration Curves & Feature Importance
blocked_by: [LGBM-003]
status: ready-for-agent
---

# LGBM-004: Evaluation, Calibration Curves & Feature Importance

## Objective
Generate calibration curves and feature importance rankings on validation predictions, then evaluate the final trained model on `test_features.csv` (2024–2026).

## Requirements
- Calibration diagnostics:
  - Generate reliability curve (`calibration_curve(y_val, y_prob, n_bins=10)`)
  - Calculate Brier Score and Log-Loss vs bookmaker baseline
- Interpretability:
  - Extract and plot top 15 features by `gain` importance
- Final holdout:
  - Retrain on full historical dataset (2000–2023) using mean optimal iterations
  - Score on `test_features.csv` (2024–2026) and log comparison with Logistic Regression.
