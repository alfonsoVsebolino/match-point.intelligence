import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_features.csv")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train_features.csv")
SCORES_CSV_PATH = os.path.join(BASE_DIR, "match_scores.csv")
SYNC_STATE_PATH = os.path.join(DATA_DIR, "sync_state.json")
DAILY_PREDICTIONS_JSON_PATH = os.path.join(DATA_DIR, "daily_predictions_latest.json")

SPLIT_CHOICES = ["Holdout Test (2024–2026)", "Full Dataset (2000–2026)"]

test_df = pd.DataFrame()
train_df = pd.DataFrame()
full_df = pd.DataFrame()


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and unified chronological match datasets from disk."""
    t_df = pd.read_csv(TEST_CSV_PATH) if os.path.exists(TEST_CSV_PATH) else pd.DataFrame()
    tr_df = pd.read_csv(TRAIN_CSV_PATH) if os.path.exists(TRAIN_CSV_PATH) else pd.DataFrame()

    if os.path.exists(SCORES_CSV_PATH):
        scores_df = pd.read_csv(SCORES_CSV_PATH)
        if not t_df.empty:
            t_df = t_df.drop(columns=["Score"], errors="ignore").merge(
                scores_df, on=["Date", "Player_1", "Player_2"], how="left"
            )
        if not tr_df.empty:
            tr_df = tr_df.drop(columns=["Score"], errors="ignore").merge(
                scores_df, on=["Date", "Player_1", "Player_2"], how="left"
            )

    if not t_df.empty and not tr_df.empty:
        f_df = pd.concat([tr_df, t_df], ignore_index=True)
        if "Date" in f_df.columns:
            f_df["Date"] = pd.to_datetime(f_df["Date"])
            f_df = f_df.sort_values("Date").reset_index(drop=True)
    elif not t_df.empty:
        f_df = t_df.copy()
    else:
        f_df = pd.DataFrame()

    return tr_df, t_df, f_df


def reload_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reload module-level DataFrames in place after a daily sync."""
    global train_df, test_df, full_df
    train_df, test_df, full_df = load_datasets()
    return train_df, test_df, full_df


train_df, test_df, full_df = load_datasets()


def get_sync_status() -> dict:
    """Return latest Kaggle daily sync metadata if available."""
    if not os.path.exists(SYNC_STATE_PATH):
        max_date = str(test_df["Date"].max())[:10] if not test_df.empty and "Date" in test_df.columns else "unknown"
        return {
            "status": "static",
            "last_synced_at": None,
            "latest_match_date": max_date,
            "test_rows": int(len(test_df)),
            "train_rows": int(len(train_df)),
        }
    try:
        with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"status": "unknown"}


def get_daily_predictions() -> dict:
    """Return latest generated daily schedule predictions payload."""
    if not os.path.exists(DAILY_PREDICTIONS_JSON_PATH):
        return {"generated_at": None, "target_date": None, "fixtures": []}
    try:
        with open(DAILY_PREDICTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"generated_at": None, "target_date": None, "fixtures": []}


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

