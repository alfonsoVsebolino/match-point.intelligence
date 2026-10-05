import os
import joblib
import pandas as pd
import numpy as np

# Unpickling compatibility shims
class _RemainderColsList(list):
    pass

import __main__
__main__._RemainderColsList = _RemainderColsList
import sys
sys.modules["__main__"]._RemainderColsList = _RemainderColsList
import builtins
builtins._RemainderColsList = _RemainderColsList
import sklearn.compose._column_transformer
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

from atp_data import full_df, get_df_for_split
from atp_cards import render_diagnostic_card

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "atp_inference_bundle.joblib")
bundle = joblib.load(BUNDLE_PATH)

champion_lgb = bundle["champion_lgb"]
champion_lr = bundle["champion_lr"]
model_features = bundle["model_features"]
latest_player_state = bundle["latest_player_state"]
player_surface_wr = bundle["player_surface_wr"]
le_surf = bundle["le_surf"]
le_series = bundle.get("le_series", None)

# Fix ColumnTransformer imputer dtype across sklearn versions
pre = champion_lr.named_steps.get("preprocessor", None)
if pre and hasattr(pre, "transformers_"):
    for _, trans, _ in pre.transformers_:
        if hasattr(trans, "named_steps"):
            for _, step in trans.named_steps.items():
                if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
                    step._fill_dtype = step._fit_dtype

# Extract player pools
active_players = sorted([
    p for p, s in latest_player_state.items()
    if pd.to_datetime(s.get("Date", "2000-01-01")).year >= 2023
])
all_players = sorted(list(latest_player_state.keys()))

DEFAULT_P1 = "Alcaraz C." if "Alcaraz C." in active_players else active_players[0]
DEFAULT_P2 = "Sinner J." if "Sinner J." in active_players else (active_players[1] if len(active_players) > 1 else active_players[0])

def get_h2h_stats(p1: str, p2: str, surface: str, cutoff_date=None) -> dict:
    # Use full historical dataset (2000–2026) for complete H2H coverage
    if full_df.empty:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}
    
    h2h_sub = full_df[
        ((full_df["Player_1"] == p1) & (full_df["Player_2"] == p2)) |
        ((full_df["Player_1"] == p2) & (full_df["Player_2"] == p1))
    ]
    if cutoff_date is not None and "Date" in h2h_sub.columns:
        cutoff_dt = pd.to_datetime(cutoff_date)
        h2h_sub = h2h_sub[pd.to_datetime(h2h_sub["Date"]) < cutoff_dt]
        
    if len(h2h_sub) == 0:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}
        
    p1_wins = (h2h_sub["Winner"] == p1).sum()
    total = len(h2h_sub)
    surf_sub = h2h_sub[h2h_sub["Surface"] == surface]
    surf_total = len(surf_sub)
    surf_wins = (surf_sub["Winner"] == p1).sum() if surf_total > 0 else 0
    return {
        "h2h_winrate": float(p1_wins / total),
        "h2h_match_count": int(total),
        "h2h_surface_winrate": float(surf_wins / surf_total) if surf_total > 0 else 0.5,
        "h2h_surface_count": int(surf_total),
    }

def build_match_features(p1: str, p2: str, surface: str, series: str, target_date=None) -> pd.DataFrame:
    s1 = latest_player_state.get(p1, {
        "Rank": 500, "Pts": 0, "form_divergence": 0.0, "matches_14d": 0, "matches_30d": 0,
        "days_since_last": 14, "career_winrate": 0.5
    })
    s2 = latest_player_state.get(p2, {
        "Rank": 500, "Pts": 0, "form_divergence": 0.0, "matches_14d": 0, "matches_30d": 0,
        "days_since_last": 14, "career_winrate": 0.5
    })
    
    p1_swr = player_surface_wr.get((p1, surface), s1.get("career_winrate", 0.5))
    p2_swr = player_surface_wr.get((p2, surface), s2.get("career_winrate", 0.5))
    h2h = get_h2h_stats(p1, p2, surface, cutoff_date=target_date)
    
    surf_code = int(le_surf.transform([surface])[0]) if hasattr(le_surf, "classes_") and surface in le_surf.classes_ else {"Carpet": 0, "Clay": 1, "Grass": 2, "Hard": 3}.get(surface, 3)
    ser_code = int(le_series.transform([series])[0]) if hasattr(le_series, "classes_") and series in le_series.classes_ else {"ATP250": 0, "ATP500": 1, "Grand Slam": 2, "International": 3, "International Gold": 4, "Masters": 5, "Masters 1000": 6, "Masters Cup": 7}.get(series, 2)
    
    row = {
        "rank_diff": float(s1.get("Rank", 500) - s2.get("Rank", 500)),
        "pts_diff": float(s1.get("Pts", 0) - s2.get("Pts", 0)),
        "short_winrate_diff": float(s1.get("short_winrate", 0.5) - s2.get("short_winrate", 0.5)),
        "short_tb_winrate_diff": float(s1.get("short_tb_winrate", 0.5) - s2.get("short_tb_winrate", 0.5)),
        "short_comeback_rate_diff": float(s1.get("short_comeback_rate", 0.0) - s2.get("short_comeback_rate", 0.0)),
        "short_bb_rate_diff": float(s1.get("short_bb_rate", 0.0) - s2.get("short_bb_rate", 0.0)),
        "short_avg_games_per_set_diff": float(s1.get("short_avg_games_per_set", 4.9) - s2.get("short_avg_games_per_set", 4.9)),
        "med_winrate_diff": float(s1.get("med_winrate", 0.5) - s2.get("med_winrate", 0.5)),
        "med_tb_winrate_diff": float(s1.get("med_tb_winrate", 0.5) - s2.get("med_tb_winrate", 0.5)),
        "med_comeback_rate_diff": float(s1.get("med_comeback_rate", 0.0) - s2.get("med_comeback_rate", 0.0)),
        "med_bb_rate_diff": float(s1.get("med_bb_rate", 0.0) - s2.get("med_bb_rate", 0.0)),
        "med_avg_games_per_set_diff": float(s1.get("med_avg_games_per_set", 4.9) - s2.get("med_avg_games_per_set", 4.9)),
        "med_ret_freq_diff": float(s1.get("med_ret_freq", 0.0) - s2.get("med_ret_freq", 0.0)),
        "career_winrate_diff": float(s1.get("career_winrate", 0.5) - s2.get("career_winrate", 0.5)),
        "career_tb_winrate_diff": float(s1.get("career_tb_winrate", 0.5) - s2.get("career_tb_winrate", 0.5)),
        "career_comeback_rate_diff": float(s1.get("career_comeback_rate", 0.0) - s2.get("career_comeback_rate", 0.0)),
        "career_bb_rate_diff": float(s1.get("career_bb_rate", 0.0) - s2.get("career_bb_rate", 0.0)),
        "career_avg_games_per_set_diff": float(s1.get("career_avg_games_per_set", 4.9) - s2.get("career_avg_games_per_set", 4.9)),
        "career_ret_freq_diff": float(s1.get("career_ret_freq", 0.0) - s2.get("career_ret_freq", 0.0)),
        "surface_winrate_diff": float(p1_swr - p2_swr),
        "days_since_last_diff": float(s1.get("days_since_last", 14) - s2.get("days_since_last", 14)),
        "matches_14d_diff": float(s1.get("matches_14d", 0) - s2.get("matches_14d", 0)),
        "matches_30d_diff": float(s1.get("matches_30d", 0) - s2.get("matches_30d", 0)),
        "form_divergence_diff": float(s1.get("form_divergence", 0.0) - s2.get("form_divergence", 0.0)),
        "h2h_winrate": float(h2h["h2h_winrate"]),
        "h2h_match_count": float(h2h["h2h_match_count"]),
        "h2h_surface_winrate": float(h2h["h2h_surface_winrate"]),
        "h2h_surface_count": float(h2h["h2h_surface_count"]),
        "surface_encoded": surf_code,
        "series_encoded": ser_code,
        "is_grand_slam": 1 if series == "Grand Slam" else 0
    }
    return pd.DataFrame([row])[model_features]

def predict_matchup(p1: str, p2: str, surface: str, series: str) -> str:
    if not p1 or not p2:
        return "<div class='text-rose-400 p-4'>Please select both Player 1 and Player 2.</div>"
    if p1 == p2:
        return "<div class='text-amber-400 p-4'>Player 1 and Player 2 must be different athletes.</div>"
        
    X_fwd = build_match_features(p1, p2, surface, series)
    X_inv = build_match_features(p2, p1, surface, series)
    
    raw_lgb_p1 = float(champion_lgb.predict_proba(X_fwd)[0, 1])
    raw_lgb_p2 = float(champion_lgb.predict_proba(X_inv)[0, 1])
    raw_lr_p1 = float(champion_lr.predict_proba(X_fwd)[0, 1])
    raw_lr_p2 = float(champion_lr.predict_proba(X_inv)[0, 1])
    
    # Real raw model asymmetry before symmetrization
    raw_asym_lgb = abs((raw_lgb_p1 + raw_lgb_p2) - 1.0)
    
    # Symmetrized probabilities: P(P1) = [M(X) + (1 - M(-X))] / 2
    prob_lgb = (raw_lgb_p1 + (1.0 - raw_lgb_p2)) / 2.0
    prob_lr = (raw_lr_p1 + (1.0 - raw_lr_p2)) / 2.0
    
    # Symmetrization guarantee error
    sym_score = abs((prob_lgb + (1.0 - prob_lgb)) - 1.0)
    
    # Divergence check
    div_delta = abs(prob_lgb - prob_lr)
    is_divergent = div_delta > 0.15
    
    # Key stats
    h2h_data = get_h2h_stats(p1, p2, surface)
    h2h_p1_wins = int(round(h2h_data["h2h_winrate"] * h2h_data["h2h_match_count"]))
    h2h_p2_wins = int(h2h_data["h2h_match_count"] - h2h_p1_wins)
    
    key_stats = {
        "rank_diff": float(X_fwd["rank_diff"].iloc[0]),
        "surface_winrate_diff": float(X_fwd["surface_winrate_diff"].iloc[0]),
        "form_divergence_diff": float(X_fwd["form_divergence_diff"].iloc[0]),
        "matches_14d_diff": float(X_fwd["matches_14d_diff"].iloc[0]),
        "h2h_p1_wins": h2h_p1_wins,
        "h2h_p2_wins": h2h_p2_wins,
    }
    
    return render_diagnostic_card(
        p1=p1, p2=p2, surface=surface, series=series,
        prob_lgb=prob_lgb, prob_lr=prob_lr,
        key_stats=key_stats, sym_score=sym_score,
        raw_asym=raw_asym_lgb, is_divergent=is_divergent,
        div_delta=div_delta, is_historical=False
    )

def inspect_historical_match(split: str, tournament: str, match_idx_str: str) -> str:
    if not match_idx_str or match_idx_str == "None":
        return "<div class='text-slate-400 p-4'>Select a valid tournament and matchup to inspect.</div>"
    
    try:
        match_idx = int(match_idx_str)
    except (ValueError, TypeError):
        return "<div class='text-slate-400 p-4'>Invalid match identifier.</div>"
        
    df = get_df_for_split(split)
    if df.empty or match_idx not in df.index:
        return "<div class='text-rose-400 p-4'>Selected fixture not found in active dataset.</div>"
        
    row = df.loc[match_idx]
    p1 = row["Player_1"]
    p2 = row["Player_2"]
    surface = row.get("Surface", "Hard")
    series = row.get("Series", "Grand Slam")
    winner = row.get("Winner", "Unknown")
    score_str = row.get("Score", None)
    round_name = row.get("Round", "")
    date_str = str(row.get("Date", ""))[:10]
    
    # Ingest precomputed feature vector
    X_vec = df.loc[[match_idx], model_features].apply(pd.to_numeric)
    prob_lgb = float(champion_lgb.predict_proba(X_vec)[0, 1])
    prob_lr = float(champion_lr.predict_proba(X_vec)[0, 1])
    
    # Implied probability baseline
    bm_prob = None
    if "implied_prob_diff" in row and pd.notnull(row["implied_prob_diff"]):
        bm_prob = float((row["implied_prob_diff"] + 1.0) / 2.0)
        
    h2h_data = get_h2h_stats(p1, p2, surface, cutoff_date=row.get("Date", None))
    h2h_p1_wins = int(round(h2h_data["h2h_winrate"] * h2h_data["h2h_match_count"]))
    h2h_p2_wins = int(h2h_data["h2h_match_count"] - h2h_p1_wins)
    
    key_stats = {
        "rank_diff": float(row.get("rank_diff", 0.0)),
        "surface_winrate_diff": float(row.get("surface_winrate_diff", 0.0)),
        "form_divergence_diff": float(row.get("form_divergence_diff", 0.0)),
        "matches_14d_diff": float(row.get("matches_14d_diff", 0.0)),
        "h2h_p1_wins": h2h_p1_wins,
        "h2h_p2_wins": h2h_p2_wins,
    }
    
    return render_diagnostic_card(
        p1=p1, p2=p2, surface=surface, series=series,
        prob_lgb=prob_lgb, prob_lr=prob_lr,
        key_stats=key_stats, sym_score=0.0,
        is_divergent=abs(prob_lgb - prob_lr) > 0.15,
        div_delta=abs(prob_lgb - prob_lr),
        bm_prob=bm_prob, actual_winner=winner,
        score_str=score_str, date_str=date_str,
        round_name=round_name, tournament=tournament,
        is_historical=True
    )
