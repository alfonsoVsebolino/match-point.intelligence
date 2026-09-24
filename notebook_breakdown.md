# Notebook Breakdown: `eda_and_feature_engineering.ipynb`

## Phase 1: Data Loading & Cleaning
- **Source**: `kagglehub` (`dissfya/atp-tennis-2000-2023daily-pull/versions/1146`)
- **Cleaning Actions**:
  - Parsed `Date` → `datetime` & sorted chronologically.
  - Replaced sentinel values (`-1`, `<= 0`) in numeric columns with `NaN`.
  - Audited missing values & duplicate matches.
  - Filtered invalid odds (`<= 1.0`).
- **Target Mapping**: `target = (Winner == Player_1)`

## Phase 2: Distributions & Baseline
- **Visuals**: Match volume/year, surface distribution, series tier counts, rank diff distribution.
- **Benchmarks**:
  - **Favorite Win Rate**: Theoretical accuracy if exclusively betting on the lower-ranked player.
  - **Implied Odds Baseline**: Bookmaker Brier score & implied probability calculation.

## Phase 3: Score Parsing
- **Engine**: `parse_score_robust` (Regex-based text parser).
- **Extracted Metrics (Per Player)**:
  - Sets (Won/Lost/Total).
  - Tiebreaks (Played/Won).
  - Comebacks (Won match post 1st set loss).
  - Bagels/Breadsticks (6-0, 6-1 sets).
  - Total Games (Won/Lost).
- **Edge Cases**: Implicit/Explicit retirements (`Ret.`) and walkovers (`w/o`).

## Phase 4: Feature Engineering (Zero Future Leakage)
- **Direct Diffs**: `rank_diff`, `pts_diff`, `odds_diff`, `implied_prob_diff`.
- **Rolling Stats (3 Tiers)**:
  - *Short*: Last 10 matches.
  - *Medium*: Last 52 weeks (365 days).
  - *Career*: Expanding historical window.
  - *Metrics Computed*: Win rate, TB win rate, Comeback rate, Bagel/BB rate, Avg games/set, Retirement freq.
- **Surface-Specific**: Career surface win rates.
- **H2H (Head-to-Head)**: Prior match win rate/counts (Global & Surface specific, Cold Start = `0.5`).
- **Fatigue**: `days_since_last`, `matches_14d`, `matches_30d`.
- **Contextual**: Surface/Series encoding, Grand Slam flag, `form_divergence` (Short - Career win rate).
- **Formatting**: Computed entirely as Deltas (`Feature_P1 - Feature_P2`).

## Phase 5: EDA on Engineered Features
- **Correlation Map**: Heatmap of engineered deltas vs target.
- **Bivariate Analysis**:
  - Rank diff vs Win Probability.
  - Odds Implied Prob vs Actual Win Rate.
  - Form Divergence & Fatigue vs Win Probability.
- **Feature Importance**: Mutual Information (`mutual_info_classif`) ranking.

## Phase 6: Feature Selection
- **Strategy**: Multicollinearity reduction.
- **Execution**: Dropped features with absolute Pearson correlation `|r| > 0.90`.
- **Result**: Filtered subset of independent predictive variables.

## Phase 7: Train/Test Split & Export
- **Temporal Strategy**: Fixed temporal cut (No random shuffle).
- **Cutoff**: Train (`2000-2023`), Test (`2024-2026`).
- **Output**: `train_features.csv`, `test_features.csv`.
