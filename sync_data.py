# This is for the data synchronization and ETL pipeline for the kaggle dataset

import os
import re
import numpy as np
import pandas as pd
import kagglehub

DATASET_HANDLE = "dissfya/atp-tennis-2000-2023daily-pull"
CUTOFF_YEAR = 2023
OUTPUT_DIR = "."


# function to fetch the latest rows in the dataset
def fetch_latest_dataset(handle: str = DATASET_HANDLE) -> str:
    """Fetches latest version dir. from KaggleHub and returns path"""
    download_dir = kagglehub.dataset_download(handle)
    return os.path.join(download_dir, "atp_tennis.csv")

def clean_raw_data(csv_path: str) -> pd.DataFrame:
    # Phase 1
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)  
    df['match_id'] = np.arange(len(df))

    sentinel_columns = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]
    for col in sentinel_columns:
        df[col] = df[col].apply(lambda x: np.nan if pd.isna(x) or x <= 0 else float(x))

    df['target'] = (df['Winner'] == df['Player_1']).astype(int)

    return df

def parse_scores_and_export_matches(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    # Phase 3, Phase 2 is a visualization check so it's omitted from this decoupled ETL pipeline
    export_cols = ['Date', 'PLayer_1', 'Player_2', 'Score']
    match_df = df[export_cols].copy()

    match_df.to_csv(out_dir)

    #apply regex parse_score_robust
    