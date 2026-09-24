---
id: LR-003
title: Cross-Validation & Regularization Parameter Tuning
blocked_by: [LR-002]
status: completed
---

# LR-003: Cross-Validation & Regularization Parameter Tuning

## Objective
Execute expanding-window cross-validation to evaluate and tune the regularization parameter $C$ for Logistic Regression against validation fold log-loss.

## Requirements
- Grid of $C$ values: $[10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100]$
- Fit pipeline on each training fold; evaluate out-of-fold predictions on corresponding validation fold
- Select optimal $C$ minimizing mean cross-validated `log_loss`
- Log validation Brier score and accuracy per fold.
