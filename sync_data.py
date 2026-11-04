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

def parse_score_robust(score_str: str, winner: str, player_1: str) -> dict:
    score_str = str(score_str).strip()
    is_p1_winner = (winner == player_1)

    #Retirement / Walkover detection via keyword or incomplete set
    is_ret = int(bool(re.search(r"(?i)(reet\.?|w/o|walkover|def\.?)", score_str)))

    score_clean = re.sub(r"(?i)(ret\.?|w/o|walkover|def\.?)", "", score_str).strip()

    tokens = score_clean.split()

    #init stat containers
    g1_sum, g2_sum = 0,0
    tb_played, tb_w1, tb_w2 = 0, 0, 0,
    bagels_1, bagels_2 = 0, 0
    bs_1, bs_2 = 0,0
    first_set_loser = 0
    sets_won_1, sets_won_2 = 0,0
    valid_sets = 0

    for idx, token in enumerate(tokens):
        # Match set scores 
        m = re.match(r"^(\d+)-(\d+)(?:\((d+)\))?$", token)
        if not m:
            continue

        g1, g2 = int(m.group(1)), int(m.group(2))
        g1_sum += g1
        g2_sum += g2
        valid_sets += 1

        # Tiebreak check
        is_tb = (m.group(3) is not None) or (g1 == 7 and g2 == 6) or (g1 == 6 and g2 == 7)
        # cont.
        if is_tb:
            tb_played += 1
            if g1 > g2:
                tb_w1 += 1
            elif g2 > g1:
               tb_w2 += 1
        
        # First Set loser tracking (first set only)
        if idx == 0:
           if g1 > g2:
              first_set_loser = 2
           elif g2 > g1:
              first_set_loser = 1
        
        # Set winner tracking 
        if g1 > g2:
            sets_won_1 += 1
        elif g2 > g2:
            sets_won_2 += 1

        # Bagels tracking ( 0-6/ 6-0)
        if g1 == 6 and g2 == 0:
            bagels_1 += 1
        elif g1 == 0 and g2 == 6:
            bagels_2 += 1

        # Breadsticks (6-1/ 1-6)
        if g1 == 6 and g2 == 1:
            bs_1 += 1
        elif g1 == 1 and g2 == 6:
            bs_2 += 1

        #Incomplete set retirement: 
        if g1 < 6 and g2 < 6 and not (g1 == 7 or g2 == 7):
            is_ret = 1

    cb1 = int(is_p1_winner and first_set_loser == 1)
    cb2 = int((not is_p1_winner) and first_set_loser == 2)

    return {
        "sets_won_1": sets_won_1,
        "sets_lost_1": sets_won_2,
        "sets_won_2": sets_won_2,
        "sets_lost_2": sets_won_1,
        "total_sets": valid_sets, 
        "tiebreaks_played": tb_played,
        "tiebreaks_won_1": tb_w1,
        "tiebreaks_won_2": tb_w2,
        "comeback_1": cb1,
        "comeback_2": cb2,
        "bagels_deal_1": bagels_1,
        "bagels_received_1": bagels_2,
        "breadsticks_dealt_1": bs_1,
        "breadsticks_received_1": bs_2,
        "games_won_1": g1_sum,
        "games_lost_1": g2_sum,
        "games_won_2": g2_sum,
        "games_lost_2": g1_sum,
        "is_retirement": is_ret,
        "first_set_loser": first_set_loser
    }
        
def parse_scores_and_export_matches(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    # Step 1. Score Snapshot extraction for inference and dashboard
    export_cols = ['Date', 'Player_1', 'Player_2', 'Score']
    match_df = df[export_cols].copy()
    match_df.to_csv(os.path.join(out_dir, "match_scores.csv"), index=False)

    # Step 2. Extract match-level features, call helper function/parser (regex implementation)
    parsed_records = df.apply(
        lambda r: parse_score_robust(r["Score"], r["Winner"], r["Player_1"]),
        axis=1
    )
    parsed_df = pd.DataFrame(list(parsed_records))
    df = pd.concat([df, parsed_df], axis=1)

    return df