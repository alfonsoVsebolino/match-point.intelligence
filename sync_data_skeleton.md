# High-Overview Migration Plan: `sync_data_skeleton.py`

## 1. Architectural Gist & Dataflow
```mermaid
flowchart TD
    A[Kaggle API / kagglehub] -->|Latest Head| B[Raw atp_tennis.csv]
    B --> C[Phase 1: Ingestion & Sentinels]
    C --> D[Phase 3: Score Parser]
    D -->|Export| M[match_scores.csv]
    D --> E[Phase 4: Player Long-Table & Rolling Stats]
    E --> F[Phase 4: Symmetrical Deltas P1 - P2]
    F --> G[Phase 6: Multicollinearity Filter |r| > 0.90]
    G --> H[Phase 7: Temporal Split Cutoff 2023]
    H -->|Export| T1[train_features.csv 2000-2023]
    H -->|Export| T2[test_features.csv 2024-2026]
```

---

## 2. Key Method Calls & Libraries
- `kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")`: Auto-fetches latest version cache path without hardcoded `/versions/xxx`.
- `pd.to_datetime(df["Date"])`: Parse & sort chronologically to prevent temporal lookahead.
- `np.where(...)`: Implied probability and sentinel handling (`<= 0`, `-1` $\to$ `NaN`).
- `pd.concat([p1_table, p2_table])`: Stack match observations into long-format player timeline.
- `df.groupby("player")[col].cumsum() - df[col]`: Prior-only expanding window metrics (strictly zero future leakage).
- `rolling(window="365D", on="Date")` or `rolling(10)`: Past window aggregation.
- `corr_matrix.where(np.triu(..., k=1))`: Identify redundant $|r| > 0.90$ features.

---

## 3. Pseudocode: `sync_data_skeleton.py`

```python
"""
sync_data_skeleton.py
Standalone ETL pipeline extracting Phases 1-7 from eda_and_feature_engineering.ipynb.
"""

import os
import re
import numpy as np
import pandas as pd
import kagglehub

DATASET_HANDLE = "dissfya/atp-tennis-2000-2023daily-pull"
CUTOFF_YEAR = 2023
OUTPUT_DIR = "."


def fetch_latest_dataset(handle: str = DATASET_HANDLE) -> str:
    """Fetches latest version directory from KaggleHub and returns path to atp_tennis.csv."""
    download_dir = kagglehub.dataset_download(handle)
    return os.path.join(download_dir, "atp_tennis.csv")


def clean_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Phase 1:
    - Load raw CSV.
    - Parse Date to datetime and sort ascending.
    - Replace sentinel numbers (-1, <= 0 in ranks/odds) with np.nan.
    - Set target = (Winner == Player_1).astype(int).
    - Assign sequential match_id.
    """
    pass


def parse_scores_and_export_matches(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Phase 3:
    - Export ['Date', 'Player_1', 'Player_2', 'Score'] to match_scores.csv.
    - Apply regex parse_score_robust(score, winner, player_1):
        - sets won/lost, tiebreaks played/won, comebacks, bagels/breadsticks, total games.
    - Return df enriched with match-level parsed columns.
    """
    pass


def build_long_table_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4:
    - Build p1_table and p2_table stacked into long_df sorted by (Date, match_id).
    - Compute strictly past-only player stats:
        - Career: cumsum() - current_match
        - Short: rolling 10 matches (shift(1))
        - Medium: rolling 365D (closed='left')
    - Metrics: winrate, tb_winrate, comeback_rate, bb_rate, avg_games_per_set, fatigue (days_since_last, matches_14d, matches_30d).
    - Re-merge p1 and p2 stats back onto original df by match_id.
    - Compute symmetrical deltas: Delta = Feature_P1 - Feature_P2.
    """
    pass


def select_features(df: pd.DataFrame, diff_cols: list, threshold: float = 0.90) -> list:
    """
    Phase 6:
    - Compute correlation matrix across diff_cols.
    - Drop upper-triangle features where |r| > threshold.
    - Return pruned feature list.
    """
    pass


def export_splits(df: pd.DataFrame, selected_features: list, out_dir: str = OUTPUT_DIR) -> None:
    """
    Phase 7:
    - Keep meta_cols: ['Date', 'Tournament', 'Series', 'Court', 'Surface', 'Round', 'Player_1', 'Player_2', 'Winner', 'target'].
    - Export train_features.csv (Date.dt.year <= CUTOFF_YEAR).
    - Export test_features.csv (Date.dt.year > CUTOFF_YEAR).
    """
    pass


def main():
    # 1. Fetch
    csv_path = fetch_latest_dataset()
    
    # 2. Ingest & Clean
    df = clean_raw_data(csv_path)
    
    # 3. Parse Scores & Save match_scores.csv
    df = parse_scores_and_export_matches(df)
    
    # 4. Feature Engineering (Deltas & Rolling)
    df, diff_cols = build_long_table_features(df)
    
    # 5. Multicollinearity Filtering
    chosen_features = select_features(df, diff_cols, threshold=0.90)
    
    # 6. Export train_features.csv & test_features.csv
    export_splits(df, chosen_features)
    print("ETL complete. Datasets successfully synced.")


if __name__ == "__main__":
    main()
```
