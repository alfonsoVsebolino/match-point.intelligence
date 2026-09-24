---
id: LGBM-001
title: Data Partition and Expanding Window CV Setup
blocked_by: []
status: ready-for-agent
---

# LGBM-001: Data Partition and Expanding Window CV Setup

## Objective
Load `train_features.csv`, drop metadata and market odds from feature matrix $X$, isolate target vector $y$, preserve native NaNs, and construct identical expanding-window fold masks.

## Requirements
- Exclude metadata: `Date`, `Tournament`, `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`
- Exclude market odds: `odds_diff`, `implied_prob_diff` (hold out as benchmark series)
- Isolate `target` as $y$
- Ingest all 31 features into $X$ with native NaNs retained (no scaling, no imputation)
- Build identical 4 expanding-window temporal folds on `Date` (2000–2023).
