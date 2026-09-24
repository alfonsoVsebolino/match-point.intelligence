---
id: LGBM-003
title: Cross-Validation with Early Stopping & Tuning
blocked_by: [LGBM-002]
status: ready-for-agent
---

# LGBM-003: Cross-Validation with Early Stopping & Tuning

## Objective
Train LightGBM across the 4 expanding-window folds using validation sets for early stopping to optimize out-of-fold log-loss.

## Requirements
- Configure early stopping callbacks: `early_stopping_rounds=50` monitoring validation fold `binary_logloss`
- Run walk-forward loop across all 4 folds:
  - Record optimal boosting round (`best_iteration_`) per fold
  - Collect out-of-fold probability predictions
- Perform targeted grid or Optuna search over `num_leaves` (15, 31, 63) and `min_child_samples` (20, 50, 100)
- Select configuration with lowest mean CV Log-Loss.
