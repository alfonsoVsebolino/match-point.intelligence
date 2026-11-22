"""Reusable ATP feature engineering and player state computation pipeline.

Extracted from eda_and_feature_engineering.ipynb (Phases 1, 3, 4, 7) so the
daily Kaggle sync pipeline can refresh test_features.csv, match_scores.csv,
and latest_player_state without future-to-past leakage.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CUTOFF_YEAR = 2023
SHORT_WINDOW = 10
MEDIUM_WINDOW_DAYS = 365
H2H_COLD_START = 0.5

META_COLS = [
    "Date",
    "Tournament",
    "Series",
    "Court",
    "Surface",
    "Round",
    "Player_1",
    "Player_2",
    "Winner",
    "target",
]

SELECTED_FEATURES = [
    "rank_diff",
    "pts_diff",
    "odds_diff",
    "implied_prob_diff",
    "short_winrate_diff",
    "short_tb_winrate_diff",
    "short_comeback_rate_diff",
    "short_bb_rate_diff",
    "short_avg_games_per_set_diff",
    "med_winrate_diff",
    "med_tb_winrate_diff",
    "med_comeback_rate_diff",
    "med_bb_rate_diff",
    "med_avg_games_per_set_diff",
    "med_ret_freq_diff",
    "career_winrate_diff",
    "career_tb_winrate_diff",
    "career_comeback_rate_diff",
    "career_bb_rate_diff",
    "career_avg_games_per_set_diff",
    "career_ret_freq_diff",
    "surface_winrate_diff",
    "days_since_last_diff",
    "matches_14d_diff",
    "matches_30d_diff",
    "form_divergence_diff",
    "h2h_winrate",
    "h2h_match_count",
    "h2h_surface_winrate",
    "h2h_surface_count",
    "surface_encoded",
    "series_encoded",
    "is_grand_slam",
]

PLAYER_STATE_FEATURES = [
    "short_winrate",
    "short_tb_winrate",
    "short_comeback_rate",
    "short_bb_rate",
    "short_avg_games_per_set",
    "med_winrate",
    "med_tb_winrate",
    "med_comeback_rate",
    "med_bb_rate",
    "med_avg_games_per_set",
    "med_ret_freq",
    "career_winrate",
    "career_tb_winrate",
    "career_comeback_rate",
    "career_bb_rate",
    "career_avg_games_per_set",
    "career_ret_freq",
    "surface_winrate",
    "days_since_last",
    "matches_14d",
    "matches_30d",
    "form_divergence",
]

FALLBACK_SURF_MAP = {"Carpet": 0, "Clay": 1, "Grass": 2, "Hard": 3}
FALLBACK_SERIES_MAP = {
    "ATP250": 0,
    "ATP500": 1,
    "Grand Slam": 2,
    "International": 3,
    "International Gold": 4,
    "Masters": 5,
    "Masters 1000": 6,
    "Masters Cup": 7,
}


def parse_score_robust(score_str: Any, winner: Any, player_1: Any) -> dict[str, int]:
    """Parse ATP score strings into granular set, tiebreak, comeback, and dominance stats."""
    score_text = "" if pd.isna(score_str) else str(score_str).strip()
    is_p1_winner = winner == player_1

    is_ret = int(bool(re.search(r"(?i)(ret\.?|w/o|walkover|def\.?)", score_text)))
    score_clean = re.sub(r"(?i)(ret\.?|w/o|walkover|def\.?)", "", score_text).strip()
    tokens = score_clean.split()

    g1_sum, g2_sum = 0, 0
    tb_played, tb_w1, tb_w2 = 0, 0, 0
    bagels_1, bagels_2 = 0, 0
    bs_1, bs_2 = 0, 0
    first_set_loser = 0
    sets_won_1, sets_won_2 = 0, 0
    valid_sets = 0

    for idx, token in enumerate(tokens):
        m = re.match(r"^(\d+)-(\d+)(?:\((\d+)\))?$", token)
        if not m:
            continue

        g1, g2 = int(m.group(1)), int(m.group(2))
        g1_sum += g1
        g2_sum += g2
        valid_sets += 1

        is_tb = (m.group(3) is not None) or (g1 == 7 and g2 == 6) or (g1 == 6 and g2 == 7)
        if is_tb:
            tb_played += 1
            if g1 > g2:
                tb_w1 += 1
            elif g2 > g1:
                tb_w2 += 1

        if idx == 0:
            if g1 > g2:
                first_set_loser = 2
            elif g2 > g1:
                first_set_loser = 1

        if g1 > g2:
            sets_won_1 += 1
        elif g2 > g1:
            sets_won_2 += 1

        if g1 == 6 and g2 == 0:
            bagels_1 += 1
        elif g1 == 0 and g2 == 6:
            bagels_2 += 1

        if g1 == 6 and g2 == 1:
            bs_1 += 1
        elif g1 == 1 and g2 == 6:
            bs_2 += 1

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
        "first_set_loser": first_set_loser,
    }


def _to_positive_float(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    try:
        num = float(val)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if num <= 0 else num


def clean_raw_matches(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw Kaggle ATP dataset and split completed matches from pending/scheduled rows.

    Returns:
        (completed_df, pending_df)
    """
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player_1", "Player_2"]).copy()
    df["Player_1"] = df["Player_1"].astype(str).str.strip()
    df["Player_2"] = df["Player_2"].astype(str).str.strip()

    for col in ["Tournament", "Series", "Court", "Surface", "Round"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    if "Best of" not in df.columns:
        df["Best of"] = np.where(df["Series"] == "Grand Slam", 5, 3)
    if "Score" not in df.columns:
        df["Score"] = ""

    for col in ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]:
        if col in df.columns:
            df[col] = df[col].map(_to_positive_float)
        else:
            df[col] = np.nan

    df.loc[df["Odd_1"] <= 1.0, "Odd_1"] = np.nan
    df.loc[df["Odd_2"] <= 1.0, "Odd_2"] = np.nan

    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date", "Player_1", "Player_2"], keep="first").reset_index(drop=True)

    # Identify completed matches vs unplayed/pending fixtures
    winner_series = df["Winner"].fillna("").astype(str).str.strip() if "Winner" in df.columns else pd.Series("", index=df.index)
    is_completed = (winner_series != "") & (~winner_series.str.lower().isin(["nan", "none", "pending", "tbd", "scheduled"]))

    completed_df = df[is_completed].copy().reset_index(drop=True)
    pending_df = df[~is_completed].copy().reset_index(drop=True)

    if not completed_df.empty:
        completed_df["Winner"] = completed_df["Winner"].astype(str).str.strip()
        completed_df["match_id"] = np.arange(len(completed_df))
        completed_df["target"] = (completed_df["Winner"] == completed_df["Player_1"]).astype(int)

    return completed_df, pending_df


def _encode_category(series_vals: pd.Series, encoder: Optional[LabelEncoder], fallback_map: dict[str, int], default_code: int) -> np.ndarray:
    vals = series_vals.astype(str)
    if encoder is not None and hasattr(encoder, "classes_"):
        known = {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}
        return np.array([known.get(v, fallback_map.get(v, default_code)) for v in vals], dtype=int)
    return np.array([fallback_map.get(v, default_code) for v in vals], dtype=int)


def _compute_post_match_player_states(
    completed_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], float]]:
    """Compute up-to-date post-match player states across all completed matches.

    Unlike per-row training features (which exclude the current match to avoid leakage),
    future match inference needs each player's state *after* their most recent completed match.
    """
    if completed_df.empty or long_df.empty:
        return {}, {}

    # Latest Rank, Pts, and Date per player across P1 and P2 appearances
    p1_meta = completed_df[["Date", "match_id", "Player_1", "Rank_1", "Pts_1"]].rename(
        columns={"Player_1": "player", "Rank_1": "Rank", "Pts_1": "Pts"}
    )
    p2_meta = completed_df[["Date", "match_id", "Player_2", "Rank_2", "Pts_2"]].rename(
        columns={"Player_2": "player", "Rank_2": "Rank", "Pts_2": "Pts"}
    )
    meta_all = (
        pd.concat([p1_meta, p2_meta], ignore_index=True)
        .sort_values(["player", "Date", "match_id"], kind="mergesort")
        .groupby("player", sort=False)
        .last()
    )

    latest_state: dict[str, dict[str, Any]] = {}
    surface_wr_map: dict[tuple[str, str], float] = {}

    # Surface career win rates (including most recent match)
    surf_grp = long_df.groupby(["player", "Surface"], sort=False)["won"].agg(["sum", "count"])
    for (player, surf), row in surf_grp.iterrows():
        surface_wr_map[(str(player), str(surf))] = float(row["sum"] / row["count"]) if row["count"] > 0 else 0.5

    # Per-player rolling & expanding metrics inclusive of most recent completed match
    for player, grp in long_df.groupby("player", sort=False):
        n_total = len(grp)
        wins_total = float(grp["won"].sum())
        tb_played_total = float(grp["tb_played"].sum())
        tb_won_total = float(grp["tb_won"].sum())
        lost_1st_total = float(grp["lost_1st_set"].sum())
        cb_total = float(grp["comeback_win"].sum())
        sets_total = float(grp["sets_total"].sum())
        bb_total = float(grp["bb_dealt"].sum())
        games_total = float(grp["games_won"].sum())
        ret_total = float(grp["is_retirement"].sum())

        career_winrate = wins_total / n_total if n_total > 0 else 0.5
        career_tb_winrate = tb_won_total / tb_played_total if tb_played_total > 0 else 0.5
        career_comeback_rate = cb_total / lost_1st_total if lost_1st_total > 0 else 0.0
        career_bb_rate = bb_total / sets_total if sets_total > 0 else 0.0
        career_avg_gps = games_total / sets_total if sets_total > 0 else 4.9
        career_ret_freq = ret_total / n_total if n_total > 0 else 0.0

        short_grp = grp.tail(SHORT_WINDOW)
        s_n = len(short_grp)
        s_tb_p = float(short_grp["tb_played"].sum())
        s_lost_1st = float(short_grp["lost_1st_set"].sum())
        s_sets = float(short_grp["sets_total"].sum())

        short_winrate = float(short_grp["won"].sum()) / s_n if s_n > 0 else 0.5
        short_tb_winrate = float(short_grp["tb_won"].sum()) / s_tb_p if s_tb_p > 0 else 0.5
        short_comeback_rate = float(short_grp["comeback_win"].sum()) / max(s_lost_1st, 1.0)
        short_bb_rate = float(short_grp["bb_dealt"].sum()) / max(s_sets, 1.0)
        short_avg_gps = float(short_grp["games_won"].sum()) / max(s_sets, 1.0) if s_sets > 0 else 4.9

        last_dt = grp["Date"].iloc[-1]
        med_cutoff = last_dt - pd.Timedelta(days=MEDIUM_WINDOW_DAYS)
        med_grp = grp[grp["Date"] > med_cutoff]
        m_n = len(med_grp)
        m_tb_p = float(med_grp["tb_played"].sum())
        m_lost_1st = float(med_grp["lost_1st_set"].sum())
        m_sets = float(med_grp["sets_total"].sum())

        med_winrate = float(med_grp["won"].sum()) / m_n if m_n > 0 else 0.5
        med_tb_winrate = float(med_grp["tb_won"].sum()) / m_tb_p if m_tb_p > 0 else 0.5
        med_comeback_rate = float(med_grp["comeback_win"].sum()) / max(m_lost_1st, 1.0)
        med_bb_rate = float(med_grp["bb_dealt"].sum()) / max(m_sets, 1.0)
        med_avg_gps = float(med_grp["games_won"].sum()) / max(m_sets, 1.0) if m_sets > 0 else 4.9
        med_ret_freq = float(med_grp["is_retirement"].sum()) / max(m_n, 1)

        prev_dt = grp["Date"].iloc[-2] if n_total >= 2 else None
        days_since_last = float((last_dt - prev_dt).days) if prev_dt is not None else 60.0
        matches_14d = float((grp["Date"] > (last_dt - pd.Timedelta(days=14))).sum())
        matches_30d = float((grp["Date"] > (last_dt - pd.Timedelta(days=30))).sum())

        last_surf = str(grp["Surface"].iloc[-1])
        meta_row = meta_all.loc[player]
        rank_val = float(meta_row["Rank"]) if pd.notna(meta_row["Rank"]) else 500.0
        pts_val = float(meta_row["Pts"]) if pd.notna(meta_row["Pts"]) else 0.0

        latest_state[str(player)] = {
            "Date": last_dt,
            "Rank": rank_val,
            "Pts": pts_val,
            "short_winrate": short_winrate,
            "short_tb_winrate": short_tb_winrate,
            "short_comeback_rate": short_comeback_rate,
            "short_bb_rate": short_bb_rate,
            "short_avg_games_per_set": short_avg_gps,
            "med_winrate": med_winrate,
            "med_tb_winrate": med_tb_winrate,
            "med_comeback_rate": med_comeback_rate,
            "med_bb_rate": med_bb_rate,
            "med_avg_games_per_set": med_avg_gps,
            "med_ret_freq": med_ret_freq,
            "career_winrate": career_winrate,
            "career_tb_winrate": career_tb_winrate,
            "career_comeback_rate": career_comeback_rate,
            "career_bb_rate": career_bb_rate,
            "career_avg_games_per_set": career_avg_gps,
            "career_ret_freq": career_ret_freq,
            "surface_winrate": surface_wr_map.get((str(player), last_surf), career_winrate),
            "days_since_last": days_since_last,
            "matches_14d": matches_14d,
            "matches_30d": matches_30d,
            "form_divergence": short_winrate - career_winrate,
            "match_dates": [pd.Timestamp(d) for d in grp["Date"].tail(60).tolist()],
        }

    return latest_state, surface_wr_map


def compute_match_features(
    completed_df: pd.DataFrame,
    le_surf: Optional[LabelEncoder] = None,
    le_series: Optional[LabelEncoder] = None,
    cutoff_year: int = CUTOFF_YEAR,
) -> dict[str, Any]:
    """Run full Phase 3 + Phase 4 + Phase 7 feature engineering on cleaned completed matches."""
    if completed_df.empty:
        empty_cols = META_COLS + SELECTED_FEATURES
        empty_df = pd.DataFrame(columns=empty_cols)
        return {
            "full_features_df": empty_df,
            "train_df": empty_df,
            "test_df": empty_df,
            "scores_df": pd.DataFrame(columns=["Date", "Player_1", "Player_2", "Score"]),
            "latest_player_state": {},
            "player_surface_wr": {},
        }

    df = completed_df.sort_values(["Date", "match_id"], kind="mergesort").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))

    # Phase 3: Score parsing
    parsed_records = [
        parse_score_robust(s, w, p1)
        for s, w, p1 in zip(df["Score"], df["Winner"], df["Player_1"])
    ]
    parsed_df = pd.DataFrame(parsed_records)
    df = pd.concat([df, parsed_df], axis=1)

    # Phase 4a: Direct diffs
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
    df["odds_diff"] = df["Odd_1"] - df["Odd_2"]

    valid_odds_cond = (df["Odd_1"] > 1.0) & (df["Odd_2"] > 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv1 = np.where(valid_odds_cond, 1.0 / df["Odd_1"], np.nan)
        inv2 = np.where(valid_odds_cond, 1.0 / df["Odd_2"], np.nan)
        denom = inv1 + inv2
        ip1 = np.where(valid_odds_cond, inv1 / denom, np.nan)
        ip2 = np.where(valid_odds_cond, inv2 / denom, np.nan)
    df["implied_prob_diff"] = ip1 - ip2

    # Phase 4b, 4c, 4e: Long player-match table
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
    long_df = long_df.sort_values(["player", "Date", "match_id"], kind="mergesort").reset_index(drop=True)

    gb_p = long_df.groupby("player", sort=False)

    # Career expanding (strictly past)
    long_df["prev_matches"] = gb_p.cumcount()
    long_df["career_wins"] = gb_p["won"].cumsum() - long_df["won"]
    long_df["career_winrate"] = np.where(long_df["prev_matches"] > 0, long_df["career_wins"] / long_df["prev_matches"], 0.5)

    long_df["career_tb_played"] = gb_p["tb_played"].cumsum() - long_df["tb_played"]
    long_df["career_tb_won"] = gb_p["tb_won"].cumsum() - long_df["tb_won"]
    long_df["career_tb_winrate"] = np.where(long_df["career_tb_played"] > 0, long_df["career_tb_won"] / long_df["career_tb_played"], 0.5)

    long_df["career_lost_1st"] = gb_p["lost_1st_set"].cumsum() - long_df["lost_1st_set"]
    long_df["career_comebacks"] = gb_p["comeback_win"].cumsum() - long_df["comeback_win"]
    long_df["career_comeback_rate"] = np.where(long_df["career_lost_1st"] > 0, long_df["career_comebacks"] / long_df["career_lost_1st"], 0.0)

    long_df["career_sets_total"] = gb_p["sets_total"].cumsum() - long_df["sets_total"]
    long_df["career_bb_dealt"] = gb_p["bb_dealt"].cumsum() - long_df["bb_dealt"]
    long_df["career_bb_rate"] = np.where(long_df["career_sets_total"] > 0, long_df["career_bb_dealt"] / long_df["career_sets_total"], 0.0)

    long_df["career_games_won"] = gb_p["games_won"].cumsum() - long_df["games_won"]
    long_df["career_avg_games_per_set"] = np.where(long_df["career_sets_total"] > 0, long_df["career_games_won"] / long_df["career_sets_total"], 4.9)

    long_df["career_retirements"] = gb_p["is_retirement"].cumsum() - long_df["is_retirement"]
    long_df["career_ret_freq"] = np.where(long_df["prev_matches"] > 0, long_df["career_retirements"] / long_df["prev_matches"], 0.0)

    # Surface-specific career win rate
    gb_ps = long_df.groupby(["player", "Surface"], sort=False)
    long_df["surf_prev_matches"] = gb_ps.cumcount()
    long_df["surf_wins"] = gb_ps["won"].cumsum() - long_df["won"]
    long_df["surface_winrate"] = np.where(long_df["surf_prev_matches"] > 0, long_df["surf_wins"] / long_df["surf_prev_matches"], 0.5)

    # Short window (last 10 matches)
    shift_cols = ["won", "sets_total", "tb_played", "tb_won", "lost_1st_set", "comeback_win", "bb_dealt", "games_won"]
    for col in shift_cols:
        long_df[f"{col}_s"] = gb_p[col].shift(1)

    r10 = long_df.groupby("player", sort=False)[[f"{c}_s" for c in shift_cols]].rolling(SHORT_WINDOW, min_periods=1)
    r10_sum = r10.sum().reset_index(level=0, drop=True)
    r10_cnt = r10.count().reset_index(level=0, drop=True)

    long_df["short_winrate"] = (r10_sum["won_s"] / r10_cnt["won_s"]).fillna(0.5)
    long_df["short_tb_winrate"] = (r10_sum["tb_won_s"] / r10_sum["tb_played_s"]).fillna(0.5)
    long_df["short_comeback_rate"] = (r10_sum["comeback_win_s"] / np.maximum(r10_sum["lost_1st_set_s"], 1)).fillna(0.0)
    long_df["short_bb_rate"] = (r10_sum["bb_dealt_s"] / np.maximum(r10_sum["sets_total_s"], 1)).fillna(0.0)
    long_df["short_avg_games_per_set"] = (r10_sum["games_won_s"] / np.maximum(r10_sum["sets_total_s"], 1)).fillna(4.9)

    # Medium window (365 days)
    r365 = gb_p.rolling(f"{MEDIUM_WINDOW_DAYS}D", on="Date", closed="left")
    med_cols = ["won", "sets_total", "tb_played", "tb_won", "lost_1st_set", "comeback_win", "bb_dealt", "games_won", "is_retirement"]
    r365_sum = r365[med_cols].sum().reset_index(level=0, drop=True)
    r365_cnt = r365["won"].count().reset_index(level=0, drop=True)

    long_df["med_winrate"] = (r365_sum["won"] / r365_cnt.values).fillna(0.5).values
    long_df["med_tb_winrate"] = (r365_sum["tb_won"] / r365_sum["tb_played"]).fillna(0.5).values
    long_df["med_comeback_rate"] = (r365_sum["comeback_win"] / np.maximum(r365_sum["lost_1st_set"], 1)).fillna(0.0).values
    long_df["med_bb_rate"] = (r365_sum["bb_dealt"] / np.maximum(r365_sum["sets_total"], 1)).fillna(0.0).values
    long_df["med_avg_games_per_set"] = (r365_sum["games_won"] / np.maximum(r365_sum["sets_total"], 1)).fillna(4.9).values
    long_df["med_ret_freq"] = (r365_sum["is_retirement"] / np.maximum(r365_cnt.values, 1)).fillna(0.0).values

    # Fatigue features
    prev_dates = gb_p["Date"].shift(1)
    long_df["days_since_last"] = (long_df["Date"] - prev_dates).dt.days.fillna(60)
    m14 = gb_p.rolling("14D", on="Date", closed="left")["won"].count().reset_index(level=0, drop=True)
    m30 = gb_p.rolling("30D", on="Date", closed="left")["won"].count().reset_index(level=0, drop=True)
    long_df["matches_14d"] = m14.fillna(0).values
    long_df["matches_30d"] = m30.fillna(0).values
    long_df["form_divergence"] = long_df["short_winrate"] - long_df["career_winrate"]

    # Merge back into match rows
    p1_feats = long_df[long_df["p_num"] == 1].sort_values("match_id").reset_index(drop=True)
    p2_feats = long_df[long_df["p_num"] == 2].sort_values("match_id").reset_index(drop=True)

    for f in PLAYER_STATE_FEATURES:
        df[f"{f}_diff"] = p1_feats[f].values - p2_feats[f].values

    # Phase 4d: Head-to-Head features
    h2h_winrate = np.full(len(df), H2H_COLD_START)
    h2h_match_count = np.zeros(len(df), dtype=int)
    h2h_surface_winrate = np.full(len(df), H2H_COLD_START)
    h2h_surface_count = np.zeros(len(df), dtype=int)

    h2h_history: dict[tuple[str, str], int] = {}
    h2h_surf_history: dict[tuple[str, str, str], int] = {}

    for i, (p1, p2, winner, surf) in enumerate(
        zip(df["Player_1"].values, df["Player_2"].values, df["Winner"].values, df["Surface"].values)
    ):
        w1 = h2h_history.get((p1, p2), 0)
        w2 = h2h_history.get((p2, p1), 0)
        tot = w1 + w2
        h2h_match_count[i] = tot
        if tot > 0:
            h2h_winrate[i] = w1 / tot

        sw1 = h2h_surf_history.get((p1, p2, surf), 0)
        sw2 = h2h_surf_history.get((p2, p1, surf), 0)
        stot = sw1 + sw2
        h2h_surface_count[i] = stot
        if stot > 0:
            h2h_surface_winrate[i] = sw1 / stot

        if winner == p1:
            h2h_history[(p1, p2)] = w1 + 1
            h2h_surf_history[(p1, p2, surf)] = sw1 + 1
        elif winner == p2:
            h2h_history[(p2, p1)] = w2 + 1
            h2h_surf_history[(p2, p1, surf)] = sw2 + 1

    df["h2h_winrate"] = h2h_winrate
    df["h2h_match_count"] = h2h_match_count
    df["h2h_surface_winrate"] = h2h_surface_winrate
    df["h2h_surface_count"] = h2h_surface_count

    # Phase 4f: Contextual features
    df["surface_encoded"] = _encode_category(df["Surface"], le_surf, FALLBACK_SURF_MAP, default_code=3)
    df["series_encoded"] = _encode_category(df["Series"], le_series, FALLBACK_SERIES_MAP, default_code=0)
    df["is_grand_slam"] = ((df["Best of"] == 5) | (df["Series"] == "Grand Slam")).astype(int)

    export_cols = META_COLS + SELECTED_FEATURES
    full_features_df = df[export_cols].copy()
    full_features_df["Date"] = pd.to_datetime(full_features_df["Date"]).dt.strftime("%Y-%m-%d")

    train_mask = df["Date"].dt.year <= cutoff_year
    train_df = full_features_df[train_mask].copy().reset_index(drop=True)
    test_df = full_features_df[~train_mask].copy().reset_index(drop=True)

    scores_df = df[["Date", "Player_1", "Player_2", "Score"]].copy()
    scores_df["Date"] = pd.to_datetime(scores_df["Date"]).dt.strftime("%Y-%m-%d")

    latest_player_state, player_surface_wr = _compute_post_match_player_states(df, long_df)

    return {
        "full_features_df": full_features_df,
        "train_df": train_df,
        "test_df": test_df,
        "scores_df": scores_df,
        "latest_player_state": latest_player_state,
        "player_surface_wr": player_surface_wr,
    }
