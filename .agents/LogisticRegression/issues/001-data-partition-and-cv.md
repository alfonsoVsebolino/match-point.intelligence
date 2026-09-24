---
id: LR-001
title: Data Partition and Expanding Window CV Setup
blocked_by: []
status: completed
---

# LR-001: Data Partition and Expanding Window CV Setup

## Objective
Load `train_features.csv`, drop metadata columns, isolate target vector $y$, hold out market implied probabilities as benchmark, and construct chronological expanding-window fold masks.

## Requirements
- Drop: `Date`, `Tournament`, `Series`, `Court`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`
- Isolate `target` as $y$
- Isolate `implied_prob_diff` as external benchmark series
- Build 4 expanding-window temporal folds on `Date` (2000–2023):
  - Fold 1: Train 2000–2016, Val 2017–2018
  - Fold 2: Train 2000–2018, Val 2019–2020
  - Fold 3: Train 2000–2020, Val 2021–2022
  - Fold 4: Train 2000–2022, Val 2023
- Ensure no row shuffling and zero future leakage.
