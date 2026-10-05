import warnings
warnings.filterwarnings("ignore")

import os
import sys
import joblib
import pandas as pd
import numpy as np
import gradio as gr

# ---------------------------------------------------------------------------
# Unpickling Compatibility Shim (sklearn cross-version deserialization)
# ---------------------------------------------------------------------------
class _RemainderColsList(list):
    pass

import __main__
__main__._RemainderColsList = _RemainderColsList
sys.modules["__main__"]._RemainderColsList = _RemainderColsList

import builtins
builtins._RemainderColsList = _RemainderColsList

import sklearn.compose._column_transformer
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ---------------------------------------------------------------------------
# Load Model Bundle & Static Artifacts
# ---------------------------------------------------------------------------
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "atp_inference_bundle.joblib")
TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "test_features.csv")

if not os.path.exists(BUNDLE_PATH):
    raise FileNotFoundError(f"Inference bundle missing at {BUNDLE_PATH}")

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

# Load holdout test fixtures for backtracker
if os.path.exists(TEST_CSV_PATH):
    test_df = pd.read_csv(TEST_CSV_PATH)
else:
    test_df = pd.DataFrame()

# Extract player pools
active_players = sorted([
    p for p, s in latest_player_state.items()
    if pd.to_datetime(s.get("Date", "2000-01-01")).year >= 2023
])
all_players = sorted(list(latest_player_state.keys()))

DEFAULT_P1 = "Alcaraz C." if "Alcaraz C." in active_players else active_players[0]
DEFAULT_P2 = "Sinner J." if "Sinner J." in active_players else (active_players[1] if len(active_players) > 1 else active_players[0])

# ---------------------------------------------------------------------------
# Feature Computation & Serving Logic
# ---------------------------------------------------------------------------
def get_h2h_stats(p1: str, p2: str, surface: str, cutoff_date=None) -> dict:
    if test_df.empty:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}
    
    h2h_sub = test_df[
        ((test_df["Player_1"] == p1) & (test_df["Player_2"] == p2)) |
        ((test_df["Player_1"] == p2) & (test_df["Player_2"] == p1))
    ]
    if cutoff_date is not None and "Date" in h2h_sub.columns:
        h2h_sub = h2h_sub[h2h_sub["Date"] < cutoff_date]
        
    if len(h2h_sub) == 0:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}
        
    p1_wins = (h2h_sub["Winner"] == p1).sum()
    total = len(h2h_sub)
    surf_sub = h2h_sub[h2h_sub["Surface"] == surface]
    surf_total = len(surf_sub)
    surf_wins = (surf_sub["Winner"] == p1).sum() if surf_total > 0 else 0
    return {
        "h2h_winrate": p1_wins / total,
        "h2h_match_count": total,
        "h2h_surface_winrate": (surf_wins / surf_total) if surf_total > 0 else 0.5,
        "h2h_surface_count": surf_total,
    }

def build_match_features(p1: str, p2: str, surface: str, series: str, target_date=None) -> pd.DataFrame:
    s1 = latest_player_state.get(p1, {
        "Rank": 500, "Pts": 0, "form_divergence": 0.0, "matches_14d": 0, "matches_30d": 0,
        "days_since_last": 30, "career_winrate": 0.5
    })
    s2 = latest_player_state.get(p2, {
        "Rank": 500, "Pts": 0, "form_divergence": 0.0, "matches_14d": 0, "matches_30d": 0,
        "days_since_last": 30, "career_winrate": 0.5
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

def predict_matchup(p1: str, p2: str, surface: str, series: str):
    if not p1 or not p2:
        return "<div class='text-rose-400 p-4'>Please select both Player 1 and Player 2.</div>"
    if p1 == p2:
        return "<div class='text-amber-400 p-4'>Player 1 and Player 2 must be different athletes.</div>"
        
    # Forward & inverse pass to guarantee inversion symmetry
    X_fwd = build_match_features(p1, p2, surface, series)
    X_inv = build_match_features(p2, p1, surface, series)
    
    raw_lgb_p1 = float(champion_lgb.predict_proba(X_fwd)[0, 1])
    raw_lgb_p2 = float(champion_lgb.predict_proba(X_inv)[0, 1])
    raw_lr_p1 = float(champion_lr.predict_proba(X_fwd)[0, 1])
    raw_lr_p2 = float(champion_lr.predict_proba(X_inv)[0, 1])
    
    # Raw tree asymmetry before dual-pass symmetrization
    raw_asym = abs(raw_lgb_p1 + raw_lgb_p2 - 1.0)
    
    # Symmetrized probabilities: P(P1) = [M(X) + (1 - M(-X))] / 2
    prob_lgb = (raw_lgb_p1 + (1.0 - raw_lgb_p2)) / 2.0
    prob_lr = (raw_lr_p1 + (1.0 - raw_lr_p2)) / 2.0
    
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
        key_stats=key_stats, raw_asym=raw_asym,
        is_divergent=is_divergent, div_delta=div_delta
    )

def inspect_historical_match(split: str, tournament: str, match_idx_str: str):
    if not match_idx_str or match_idx_str == "None":
        return "<div class='text-slate-400 p-4'>Select a valid tournament and matchup to inspect.</div>"
    
    try:
        match_idx = int(match_idx_str)
    except (ValueError, TypeError):
        return "<div class='text-slate-400 p-4'>Invalid match identifier.</div>"
        
    if test_df.empty or match_idx not in test_df.index:
        return "<div class='text-rose-400 p-4'>Selected fixture not found in test features store.</div>"
        
    row = test_df.loc[match_idx]
    p1 = row["Player_1"]
    p2 = row["Player_2"]
    surface = row.get("Surface", "Hard")
    series = row.get("Series", "Grand Slam")
    winner = row.get("Winner", "Unknown")
    round_name = row.get("Round", "")
    date_str = str(row.get("Date", ""))[:10]
    
    # Ingest precomputed feature vector
    X_vec = test_df.loc[[match_idx], model_features].apply(pd.to_numeric)
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
        key_stats=key_stats, raw_asym=None,
        is_divergent=abs(prob_lgb - prob_lr) > 0.15,
        div_delta=abs(prob_lgb - prob_lr),
        bm_prob=bm_prob, actual_winner=winner,
        date_str=date_str, round_name=round_name,
        tournament=tournament
    )

# ---------------------------------------------------------------------------
# HTML Card Renderer (MatchPoint.intelligence Visual Tokens)
# ---------------------------------------------------------------------------
def render_diagnostic_card(
    p1: str, p2: str, surface: str, series: str,
    prob_lgb: float, prob_lr: float, key_stats: dict,
    raw_asym: float = None, is_divergent: bool = False,
    div_delta: float = 0.0, bm_prob: float = None,
    actual_winner: str = None, date_str: str = None,
    round_name: str = None, tournament: str = None
) -> str:
    p2_prob_lgb = 1.0 - prob_lgb
    p2_prob_lr = 1.0 - prob_lr
    
    # Surface & Series colors
    surf_colors = {
        "Hard": "#38bdf8", "Clay": "#fb923c",
        "Grass": "#4ade80", "Carpet": "#a78bfa"
    }
    surf_col = surf_colors.get(surface, "#2dd4bf")
    
    # Winner verification
    winner_badge_html = ""
    if actual_winner:
        model_favors_p1 = prob_lgb >= 0.5
        actual_p1_won = actual_winner == p1
        correct = (model_favors_p1 == actual_p1_won)
        
        badge_bg = "rgba(16, 185, 129, 0.15)" if correct else "rgba(244, 63, 94, 0.15)"
        badge_border = "#10b981" if correct else "#f43f5e"
        badge_text = "#10b981" if correct else "#f43f5e"
        badge_icon = "✓" if correct else "✗"
        
        winner_badge_html = f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:{badge_bg}; border:1px solid {badge_border}; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:{badge_text}; font-weight:800; font-size:16px;">{badge_icon}</span>
                <span style="color:#e2e8f0; font-size:13px; font-weight:600;">Actual Winner: <strong style="color:#ffffff;">{actual_winner}</strong></span>
            </div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; padding:2px 8px; border-radius:4px; background:{badge_border}; color:#0b0f19; font-weight:700;">
                {'PREDICTION ACCURATE' if correct else 'UPSET / DIVERGENT'}
            </span>
        </div>
        """
        
    # Divergence alert
    divergence_banner_html = ""
    if is_divergent:
        divergence_banner_html = f"""
        <div style="display:flex; align-items:center; gap:8px; background:rgba(245, 158, 11, 0.12); border:1px solid #f59e0b; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
            <span style="color:#f59e0b; font-size:16px;">⚠️</span>
            <div style="font-size:12px; color:#fde68a; font-family:'Plus Jakarta Sans', sans-serif;">
                <strong>Non-linear divergence detected (|ΔP| = {div_delta*100:.1f}%)</strong>: LightGBM captures interactions unmodeled by regularized Logistic Regression.
            </div>
        </div>
        """
        
    # Implied Bookmaker Prob Bar (Historical mode)
    bm_html = ""
    if bm_prob is not None:
        bm_p2_prob = 1.0 - bm_prob
        bm_html = f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#94a3b8;">Market Implied (Bookmaker Odds)</span>
                <span style="color:#cbd5e1; font-weight:600;">{bm_prob*100:.1f}% vs {bm_p2_prob*100:.1f}%</span>
            </div>
            <div style="height:6px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{bm_prob*100:.1f}%; background:#94a3b8; transition:width 0.4s ease;"></div>
                <div style="width:{bm_p2_prob*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>
        """
        
    # Subtitle with Date / Tournament info
    sub_title = f"{series} • {surface} Court"
    if tournament and date_str:
        sub_title = f"{date_str} • {tournament} ({round_name}) • {surface}"

    # Symmetry badge (evaluated in Mode A dual-pass, omitted in Mode B historical replay)
    if raw_asym is not None:
        sym_chip_html = f"""
        <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); color:#10b981; font-weight:600;" title="Raw tree asymmetry before dual-pass symmetrization: {raw_asym:.4f}">
            Symmetry: Enforced (Raw Δ: {raw_asym*100:.2f}%)
        </span>
        """
    else:
        sym_chip_html = """
        <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:#1e293b; color:#94a3b8; font-weight:600;">
            Holdout Replay
        </span>
        """

    card_html = f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:20px; font-family:'Plus Jakarta Sans', sans-serif; color:#f8fafc; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.4);">
        <!-- Match Header -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; border-bottom:1px solid #1e293b; padding-bottom:12px;">
            <div>
                <div style="font-size:18px; font-weight:800; tracking-tight:tight; color:#ffffff;">
                    {p1} <span style="color:#64748b; font-weight:400;">vs</span> {p2}
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                    {sub_title}
                </div>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
                <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:rgba(20, 184, 166, 0.1); border:1px solid rgba(20, 184, 166, 0.3); color:{surf_col}; font-weight:700;">
                    {surface}
                </span>
                <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:#1e293b; color:#cbd5e1; font-weight:600;">
                    {series}
                </span>
                {sym_chip_html}
            </div>
        </div>

        {winner_badge_html}
        {divergence_banner_html}

        <!-- Dual Model Probability Bars -->
        <!-- LightGBM Champion -->
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#2dd4bf; font-weight:700;">★ LightGBM Champion</span>
                <span>
                    <strong style="color:#2dd4bf;">{prob_lgb*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p1})</span>
                    <span style="color:#64748b; margin:0 4px;">|</span>
                    <strong style="color:#94a3b8;">{p2_prob_lgb*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p2})</span>
                </span>
            </div>
            <div style="height:12px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{prob_lgb*100:.1f}%; background:linear-gradient(90deg, #0d9488, #2dd4bf); transition:width 0.4s ease;"></div>
                <div style="width:{p2_prob_lgb*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>

        <!-- Logistic Regression Baseline -->
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#60a5fa; font-weight:600;">Logistic Regression Baseline</span>
                <span>
                    <strong style="color:#60a5fa;">{prob_lr*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p1})</span>
                    <span style="color:#64748b; margin:0 4px;">|</span>
                    <strong style="color:#94a3b8;">{p2_prob_lr*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p2})</span>
                </span>
            </div>
            <div style="height:8px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{prob_lr*100:.1f}%; background:linear-gradient(90deg, #2563eb, #60a5fa); transition:width 0.4s ease;"></div>
                <div style="width:{p2_prob_lr*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>

        {bm_html}

        <!-- 5 Key Delta Summary Tiles -->
        <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-top:16px; background:#0b0f19; padding:12px; border-radius:8px; border:1px solid #1e293b; text-align:center;">
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Rank Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['rank_diff'] <= 0 else '#ef4444'};">
                    {key_stats['rank_diff']:+.0f}
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Surface WR Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['surface_winrate_diff'] >= 0 else '#ef4444'};">
                    {key_stats['surface_winrate_diff']*100:+.1f}%
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Form Div Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['form_divergence_diff'] >= 0 else '#ef4444'};">
                    {key_stats['form_divergence_diff']:+.2f}
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Fatigue 14d Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:#e2e8f0;">
                    {key_stats['matches_14d_diff']:+.0f} m
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">H2H Record</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:#38bdf8;">
                    {key_stats['h2h_p1_wins']} - {key_stats['h2h_p2_wins']}
                </div>
            </div>
        </div>
    </div>
    """
    return card_html

# ---------------------------------------------------------------------------
# UI Construction (Gradio Blocks)
# ---------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

.font-mono, pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glassmorphism Card Panels */
.glass-panel {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
}

/* Primary Brand Buttons */
.btn-primary {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(20, 184, 166, 0.35) !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
    transform: translateY(-1px) !important;
}

/* Swap Button */
.btn-swap {
    background: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.btn-swap:hover {
    background: #334155 !important;
}

/* Segmented Mode Radio Switch */
.segmented-mode-switch .wrap {
    gap: 8px !important;
    background: #0f172a !important;
    padding: 6px !important;
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#0f172a",
    block_border_color="#1e293b",
    input_background_fill="#0b0f19",
    input_border_color="#334155",
    button_primary_background_fill="#0d9488",
    button_primary_background_fill_hover="#14b8a6",
    button_primary_text_color="#ffffff",
)

# Precompute initial tournaments & matches for Mode B
initial_tournaments = sorted(test_df["Tournament"].dropna().unique().tolist()) if not test_df.empty else []
default_tournament = initial_tournaments[0] if initial_tournaments else ""

def get_matches_for_tournament(tourn_name: str):
    if test_df.empty or not tourn_name:
        return []
    m_sub = test_df[test_df["Tournament"] == tourn_name]
    matches = []
    for idx, row in m_sub.iterrows():
        d_str = str(row.get("Date", ""))[:10]
        label = f"[{d_str} | {row.get('Round', '')}] {row.get('Player_1', '')} vs {row.get('Player_2', '')} (Winner: {row.get('Winner', '')})"
        matches.append((label, str(idx)))
    return matches

initial_matches = get_matches_for_tournament(default_tournament)
default_match_id = initial_matches[0][1] if initial_matches else None

with gr.Blocks(title="MatchPoint.intelligence | ATP Predictor", theme=theme, css=custom_css) as demo:
    # Top Navbar & Telemetry
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML("""
            <div style="display:flex; align-items:center; gap:12px; padding:10px 0;">
                <div style="width:42px; height:42px; border-radius:10px; background:linear-gradient(135deg, #14b8a6, #3b82f6); display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 0 15px rgba(20, 184, 166, 0.4);">
                    🎾
                </div>
                <div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px; font-weight:800; color:#ffffff; tracking-tight:tight;">MatchPoint.intelligence</span>
                        <span style="font-size:10px; font-family:'JetBrains Mono', monospace; text-transform:uppercase; background:rgba(20, 184, 166, 0.15); border:1px solid rgba(20, 184, 166, 0.3); color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;">
                            Gradio Serving
                        </span>
                    </div>
                    <div style="font-size:12px; color:#94a3b8; font-family:'JetBrains Mono', monospace;">
                        Dual-Track ML Engine (LightGBM Champion vs LogReg Baseline) • 31 Fundamental Deltas
                    </div>
                </div>
            </div>
            """)
        with gr.Column(scale=4):
            gr.HTML("""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px; height:100%; font-family:'JetBrains Mono', monospace; font-size:11px;">
                <div style="background:#0f172a; padding:6px 12px; border-radius:6px; border:1px solid #1e293b; display:flex; align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                    <span style="color:#94a3b8;">Bundle:</span>
                    <span style="color:#e2e8f0; font-weight:600;">Loaded (31 Deltas)</span>
                </div>
                <div style="background:#0f172a; padding:6px 12px; border-radius:6px; border:1px solid #1e293b; display:flex; align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; background:#38bdf8;"></span>
                    <span style="color:#94a3b8;">Holdout:</span>
                    <span style="color:#e2e8f0; font-weight:600;">2024–2026 Test</span>
                </div>
            </div>
            """)

    gr.HTML("<div style='height:1px; background:#1e293b; margin:8px 0 16px 0;'></div>")

    # Mode Selector Toggle (Segmented Radio)
    with gr.Row():
        mode_radio = gr.Radio(
            choices=[
                "⚡ Upcoming Match Predictor",
                "🔍 Historical Match Backtracker"
            ],
            value="⚡ Upcoming Match Predictor",
            show_label=False,
            container=False,
            elem_classes=["segmented-mode-switch"]
        )

    # =========================================================================
    # Mode A: Upcoming Match Predictor
    # =========================================================================
    with gr.Column(visible=True) as mode_a_container:
        with gr.Row():
            roster_scope_cb = gr.Checkbox(
                value=False,
                label="Unlock All-Time Historical Roster (2000–2026)",
                info="Default is active modern roster (2023–2026)"
            )

        with gr.Row():
            p1_dropdown = gr.Dropdown(
                choices=active_players,
                value=DEFAULT_P1,
                label="Player 1",
                filterable=True,
                scale=5
            )
            swap_btn = gr.Button(
                "⇄ Swap",
                elem_classes=["btn-swap"],
                scale=1,
                min_width=70
            )
            p2_dropdown = gr.Dropdown(
                choices=active_players,
                value=DEFAULT_P2,
                label="Player 2",
                filterable=True,
                scale=5
            )

        with gr.Row():
            surface_dropdown = gr.Dropdown(
                choices=["Hard", "Clay", "Grass", "Carpet"],
                value="Hard",
                label="Court Surface",
                scale=4
            )
            series_dropdown = gr.Dropdown(
                choices=["Grand Slam", "Masters 1000", "ATP500", "ATP250"],
                value="Grand Slam",
                label="Tournament Series",
                scale=4
            )
            predict_btn = gr.Button(
                "⚡ Predict Matchup",
                elem_classes=["btn-primary"],
                scale=4
            )

        mode_a_output = gr.HTML(
            value=predict_matchup(DEFAULT_P1, DEFAULT_P2, "Hard", "Grand Slam"),
            label="Inversion Symmetry & Model Diagnostics"
        )

    # =========================================================================
    # Mode B: Historical Match Backtracker
    # =========================================================================
    with gr.Column(visible=False) as mode_b_container:
        with gr.Row():
            split_dropdown = gr.Dropdown(
                choices=["Holdout Test (2024–2026)"],
                value="Holdout Test (2024–2026)",
                label="Dataset Split",
                scale=4
            )
            tourn_dropdown = gr.Dropdown(
                choices=initial_tournaments,
                value=default_tournament,
                label="Tournament",
                filterable=True,
                scale=8
            )

        with gr.Row():
            match_dropdown = gr.Dropdown(
                choices=initial_matches,
                value=default_match_id,
                label="Select Matchup",
                filterable=True,
                scale=9
            )
            inspect_btn = gr.Button(
                "🔍 Inspect & Backtrack",
                elem_classes=["btn-primary"],
                scale=3
            )

        mode_b_output = gr.HTML(
            value=inspect_historical_match("Holdout Test (2024–2026)", default_tournament, default_match_id),
            label="Historical Match Evaluation"
        )

    # =========================================================================
    # Event Wiring & Callbacks
    # =========================================================================
    # 1. Mode Switch Toggle
    def on_mode_change(selected_mode):
        is_mode_a = (selected_mode == "⚡ Upcoming Match Predictor")
        return gr.update(visible=is_mode_a), gr.update(visible=not is_mode_a)

    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=[mode_a_container, mode_b_container]
    )

    # 2. Swap Players
    def on_swap_players(cur_p1, cur_p2):
        return cur_p2, cur_p1

    swap_btn.click(
        fn=on_swap_players,
        inputs=[p1_dropdown, p2_dropdown],
        outputs=[p1_dropdown, p2_dropdown]
    )

    # 3. Roster Scope Toggle
    def on_roster_toggle(unlock_all, cur_p1, cur_p2):
        pool = all_players if unlock_all else active_players
        new_p1 = cur_p1 if cur_p1 in pool else pool[0]
        new_p2 = cur_p2 if cur_p2 in pool else (pool[1] if len(pool) > 1 else pool[0])
        return gr.update(choices=pool, value=new_p1), gr.update(choices=pool, value=new_p2)

    roster_scope_cb.change(
        fn=on_roster_toggle,
        inputs=[roster_scope_cb, p1_dropdown, p2_dropdown],
        outputs=[p1_dropdown, p2_dropdown]
    )

    # 4. Predict Upcoming Matchup
    predict_btn.click(
        fn=predict_matchup,
        inputs=[p1_dropdown, p2_dropdown, surface_dropdown, series_dropdown],
        outputs=[mode_a_output]
    )

    # 5. Cascading Tournaments & Matchups in Mode B
    def on_tournament_change(tourn_name):
        matches = get_matches_for_tournament(tourn_name)
        new_val = matches[0][1] if matches else None
        return gr.update(choices=matches, value=new_val)

    tourn_dropdown.change(
        fn=on_tournament_change,
        inputs=[tourn_dropdown],
        outputs=[match_dropdown]
    )

    # 6. Inspect Historical Fixture
    inspect_btn.click(
        fn=inspect_historical_match,
        inputs=[split_dropdown, tourn_dropdown, match_dropdown],
        outputs=[mode_b_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_error=True)
