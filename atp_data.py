import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_features.csv")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train_features.csv")

# Load datasets
test_df = pd.read_csv(TEST_CSV_PATH) if os.path.exists(TEST_CSV_PATH) else pd.DataFrame()
train_df = pd.read_csv(TRAIN_CSV_PATH) if os.path.exists(TRAIN_CSV_PATH) else pd.DataFrame()

# Create unified chronological dataset
if not test_df.empty and not train_df.empty:
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    if "Date" in full_df.columns:
        full_df["Date"] = pd.to_datetime(full_df["Date"])
        full_df = full_df.sort_values("Date").reset_index(drop=True)
elif not test_df.empty:
    full_df = test_df.copy()
else:
    full_df = pd.DataFrame()

SPLIT_CHOICES = ["Holdout Test (2024–2026)", "Full Dataset (2000–2026)"]

def get_df_for_split(split_name: str) -> pd.DataFrame:
    if "Holdout" in split_name:
        return test_df
    return full_df

def get_tournaments_for_split(split_name: str) -> list:
    df = get_df_for_split(split_name)
    if df.empty or "Tournament" not in df.columns:
        return []
    return sorted(df["Tournament"].dropna().unique().tolist())

def get_matches_for_tournament(arg1: str, arg2: str = None) -> list:
    if arg2 is None:
        split_name = "Holdout Test (2024–2026)"
        tourn_name = arg1
    else:
        split_name = arg1
        tourn_name = arg2
    df = get_df_for_split(split_name)
    if df.empty or not tourn_name or "Tournament" not in df.columns:
        return []
    m_sub = df[df["Tournament"] == tourn_name]
    matches = []
    for idx, row in m_sub.iterrows():
        d_str = str(row.get("Date", ""))[:10]
        round_name = row.get("Round", "")
        p1 = row.get("Player_1", "")
        p2 = row.get("Player_2", "")
        winner = row.get("Winner", "")
        label = f"[{d_str} | {round_name}] {p1} vs {p2} (Winner: {winner})"
        matches.append((label, str(idx)))
    return matches
