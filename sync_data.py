# This is for the data synchronization and ETL pipeline for the kaggle dataset

import os
import re

import kagglehub
import numpy as np
import pandas as pd

DATASET_HANDLE = "dissfya/atp-tennis-2000-2023daily-pull"
CUTOFF_YEAR = 2023
OUTPUT_DIR = "."
SHORT_WINDOW = 10
H2H_COLD_START = 0.5


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
    is_ret = int(bool(re.search(r"(?i)(ret\.?|w/o|walkover|def\.?)", score_str)))

    score_clean = re.sub(r"(?i)(ret\.?|w/o|walkover|def\.?)", "", score_str).strip()

    tokens = score_clean.split()

    #init stat containers
    g1_sum, g2_sum = 0, 0
    tb_played, tb_w1, tb_w2 = 0, 0, 0
    bagels_1, bagels_2 = 0, 0
    bs_1, bs_2 = 0, 0
    first_set_loser = 0
    sets_won_1, sets_won_2 = 0, 0
    valid_sets = 0

    for idx, token in enumerate(tokens):
        # Match set scores 
        m = re.match(r"^(\d+)-(\d+)(?:\((\d+)\))?$", token)
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
        elif g2 > g1:
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
        "bagels_dealt_1": bagels_1,
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

def build_long_table_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # 1. Direct Differences
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
    df["odds_diff"] = df["Odd_1"] - df["Odd_2"]

    # Implied probabilities
    valid_odds_cond = (df["Odd_1"] > 1.0) & (df["Odd_2"] > 1.0)
    ip1 = np.where(valid_odds_cond, (1.0 / df["Odd_1"]) / ((1.0 / df["Odd_1"]) + (1.0 / df["Odd_2"])), np.nan)
    ip2 = np.where(valid_odds_cond, (1.0 / df["Odd_2"]) / ((1.0 / df["Odd_1"]) + (1.0 / df["Odd_2"])), np.nan)

    df["implied_prob_diff"] = ip1 - ip2

    # 2. Build p1_table + p2_table -> long_df
    p1_table = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "player": df["Player_1"],
        "opponent": df["Player_2"],
        "Surface": df["Surface"],
        "won": (df["Winner"] == df["Player_1"]).astype(int),
        "sets_total": df["total_sets"],
        "tb_played": df["tiebreaks_played"], 
        "tb_won": df["tiebreaks_won_1"],
        "lost_1st_set": (df["first_set_loser"] == 1).astype(int),
        "comeback_win": df["comeback_1"],
        "bb_dealt": df["bagels_dealt_1"] + df["breadsticks_dealt_1"],
        "games_won": df["games_won_1"],
        "is_retirement": df["is_retirement"],
        "p_num": 1,
    })

    p2_table = pd.DataFrame({
        "match_id": df["match_id"],
        "Date": df["Date"],
        "player": df["Player_2"],
        "opponent": df["Player_1"],
        "Surface": df["Surface"],
        "won": (df["Winner"] == df["Player_2"]).astype(int),
        "sets_total": df["total_sets"],
        "tb_played": df["tiebreaks_played"], 
        "tb_won": df["tiebreaks_won_2"],
        "lost_1st_set": (df["first_set_loser"] == 2).astype(int),
        "comeback_win": df["comeback_2"],
        "bb_dealt": df["bagels_received_1"] + df["breadsticks_received_1"],
        "games_won": df["games_won_2"],
        "is_retirement": df["is_retirement"],
        "p_num": 2,
    })

    long_df = pd.concat([p1_table, p2_table], ignore_index=True)
    long_df = long_df.sort_values(["player", "Date", "match_id"]).reset_index(drop=True)
    # 3. Compute rolling stats on long_df (career, short 10m, med 365d, fatigue)
    gb_p = long_df.groupby("player")

    # 3.a. Career Expanding Features (strictly past matches: cumsum() - current)
    long_df["prev_matches"] = gb_p.cumcount()
    long_df["career_wins"] = gb_p["won"].cumsum() - long_df["won"]
    long_df["career_winrate"] = np.where(
        long_df["prev_matches"] > 0,
        long_df["career_wins"] / long_df["prev_matches"], 
        0.5,
    )

    long_df["career_tb_played"] = (
        gb_p["tb_played"].cumsum() - long_df["tb_played"]
    )
    long_df["career_tb_won"] = gb_p["tb_won"].cumsum() - long_df["tb_won"]
    long_df["career_tb_winrate"] = np.where(
        long_df["career_tb_played"] > 0,
        long_df["career_tb_won"] / long_df["career_tb_played"],
        0.5,
    )

    long_df["career_lost_1st"] = (
        gb_p["lost_1st_set"].cumsum() - long_df["lost_1st_set"]
    )

    long_df["career_comebacks"] = (
        gb_p["comeback_win"].cumsum() - long_df["comeback_win"]
    )

    long_df["career_comeback_rate"] = np.where(
        long_df["career_lost_1st"] > 0,
        long_df["career_comebacks"] / long_df["career_lost_1st"],
        0.0,
    )

    long_df["career_sets_total"] = (
        gb_p["sets_total"].cumsum() - long_df["sets_total"]
    )

    long_df["career_bb_dealt"] = (
        gb_p["bb_dealt"].cumsum() - long_df["bb_dealt"]
    )

    long_df["career_bb_rate"] = np.where(
        long_df["career_sets_total"] > 0,
        long_df["career_bb_dealt"] / long_df["career_sets_total"],
        0.0,
    )

    long_df["career_games_won"] = (
        gb_p["games_won"].cumsum() - long_df["games_won"]
    )

    long_df["career_avg_games_per_set"] = np.where(
        long_df["career_sets_total"] > 0,
        long_df["career_games_won"] / long_df["career_sets_total"],
        4.9,
    )

    long_df["career_retirements"] = (
        gb_p["is_retirement"].cumsum() - long_df["is_retirement"]
    )

    long_df["career_ret_freq"] = np.where(
        long_df["prev_matches"] > 0,
        long_df["career_retirements"] / long_df["prev_matches"], 
        0.0,
    )

    # Surface Specific winrate
    gb_ps = long_df.groupby(["player", "Surface"])
    long_df["surf_prev_matches"] = gb_ps.cumcount()
    long_df["surf_wins"] = gb_ps["won"].cumsum() - long_df["won"]
    long_df["surface_winrate"] = np.where(
        long_df["surf_prev_matches"] > 0,
        long_df["surf_wins"] / long_df["surf_prev_matches"],
        0.5,
    )

    # Short Window Stats (last 10 matches: shift(1) + rolling(10))
    shift_cols = [
        "won",
        "sets_total",
        "tb_played",
        "tb_won",
        "lost_1st_set",
        "comeback_win",
        "bb_dealt",
        "games_won",
    ]
    for col in shift_cols:
        long_df[f"{col}_s"] = gb_p[col].shift(1)

    r10 = long_df.groupby("player")[[f"{c}_s" for c in shift_cols]].rolling(SHORT_WINDOW, min_periods=1)
    r10_sum = r10.sum().reset_index(level=0, drop=True)
    r10_cnt = r10.count().reset_index(level=0, drop=True)

    long_df["short_winrate"] = np.where(
        r10_cnt["won_s"] > 0,
        r10_sum["won_s"] / r10_cnt["won_s"],
        0.5,
    )
    long_df["short_tb_winrate"] = np.where(
        r10_sum["tb_played_s"] > 0,
        r10_sum["tb_won_s"] / r10_sum["tb_played_s"],
        0.5,
    )

    long_df["short_comeback_rate"] = np.where(
        r10_sum["lost_1st_set_s"] > 0,
        r10_sum["comeback_win_s"] / r10_sum["lost_1st_set_s"],
        0.0,
    )
    
    
    # 4. Re-merge back onto df and compute (p1 - p2) deltas
    # 5. Compute H2H features
    # 6. Encode categorical/context features (surface, series, grand_slam)
    diff_features: list[str] = [
        "rank_diff",
        "pts_diff",
        "odds_diff",
        "implied_prob_diff",
    ]

    return df, diff_features