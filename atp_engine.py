from bisect import bisect_left, bisect_right
import math
import os
import threading
import joblib
import pandas as pd
import numpy as np
from scipy.optimize import brentq

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

import atp_data
from atp_data import get_df_for_split
from atp_cards import render_diagnostic_card

BASE_DIR = os.path.dirname(__file__)
BUNDLE_PATH = os.path.join(BASE_DIR, "atp_inference_bundle.joblib")
PLAYER_STATE_CACHE_PATH = os.path.join(BASE_DIR, "data", "player_state_cache.joblib")
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

# Precompute LogisticRegression coefficient lookup from ColumnTransformer output names
_lr_clf = champion_lr.named_steps.get("classifier", champion_lr[-1])
_lr_coefs = _lr_clf.coef_[0]
if pre is not None and hasattr(pre, "get_feature_names_out"):
    _lr_out_names = list(pre.get_feature_names_out())
    _lr_coef_lookup: dict[str, float] = {
        str(name): float(val) for name, val in zip(_lr_out_names, _lr_coefs)
    }
else:
    _lr_coef_lookup = {}

_FAST_LGB_SOURCE = champion_lgb
_FAST_LR_SOURCE = champion_lr
_FAST_LR_READY = False
_LR_IMP_STATS = np.zeros(28, dtype=np.float64)
_LR_SCALER_CENTER = np.zeros(28, dtype=np.float64)
_LR_SCALER_SCALE = np.ones(28, dtype=np.float64)
_LR_NUM_COEF = np.zeros(28, dtype=np.float64)
_LR_SURF_COEFS: dict[int, float] = {}
_LR_SER_COEFS: dict[int, float] = {}
_LR_BIN_COEF = 0.0
_LR_INTERCEPT = 0.0

try:
    if pre is not None and hasattr(pre, "named_transformers_") and "num" in pre.named_transformers_:
        _num_pipe = pre.named_transformers_["num"]
        _cat_ohe = pre.named_transformers_["cat"]
        _imp = _num_pipe.named_steps["imputer"]
        _scl = _num_pipe.named_steps["scaler"]
        _LR_IMP_STATS = np.asarray(_imp.statistics_, dtype=np.float64)
        _LR_SCALER_CENTER = np.asarray(_scl.center_, dtype=np.float64)
        _LR_SCALER_SCALE = np.asarray(_scl.scale_, dtype=np.float64)
        _coef_arr = np.asarray(_lr_coefs, dtype=np.float64)
        _LR_NUM_COEF = _coef_arr[:28]
        _LR_INTERCEPT = float(_lr_clf.intercept_[0])
        _LR_BIN_COEF = float(_lr_coef_lookup.get("bin__is_grand_slam", _coef_arr[-1]))
        _LR_SURF_COEFS = {
            int(c): float(_lr_coef_lookup.get(f"cat__surface_encoded_{int(c)}", 0.0))
            for c in _cat_ohe.categories_[0]
        }
        _LR_SER_COEFS = {
            int(c): float(_lr_coef_lookup.get(f"cat__series_encoded_{int(c)}", 0.0))
            for c in _cat_ohe.categories_[1]
        }
        _FAST_LR_READY = (len(_LR_IMP_STATS) == 28 and len(_LR_NUM_COEF) == 28)
except Exception:
    _FAST_LR_READY = False

_FEAT_INDEX: dict[str, int] = {name: i for i, name in enumerate(model_features)}
_LR_NUM_COEFS: tuple[float, ...] = tuple(
    float(_lr_coef_lookup.get(f"num__{name}", 0.0)) if math.isfinite(float(_lr_coef_lookup.get(f"num__{name}", 0.0))) else 0.0
    for name in model_features[:28]
)

_SURF_FALLBACK = {"Carpet": 0, "Clay": 1, "Grass": 2, "Hard": 3}
_SER_FALLBACK = {
    "ATP250": 0, "ATP500": 1, "Grand Slam": 2, "International": 3,
    "International Gold": 4, "Masters": 5, "Masters 1000": 6, "Masters Cup": 7,
}
_SURF_CODE_MAP: dict[str, int] = (
    {str(cls): int(i) for i, cls in enumerate(le_surf.classes_)}
    if hasattr(le_surf, "classes_")
    else _SURF_FALLBACK
)
_SER_CODE_MAP: dict[str, int] = (
    {str(cls): int(i) for i, cls in enumerate(le_series.classes_)}
    if le_series is not None and hasattr(le_series, "classes_")
    else _SER_FALLBACK
)

FEATURE_META: dict[str, dict[str, str]] = {
    "rank_diff": {"label": "ATP Rank Δ", "category": "Ranking"},
    "pts_diff": {"label": "ATP Ranking Points Δ", "category": "Ranking"},
    "short_winrate_diff": {"label": "10-Match Win Rate Δ", "category": "Short Form (10M)"},
    "short_tb_winrate_diff": {"label": "10-Match Tiebreak Win Rate Δ", "category": "Short Form (10M)"},
    "short_comeback_rate_diff": {"label": "10-Match Comeback Rate Δ", "category": "Short Form (10M)"},
    "short_bb_rate_diff": {"label": "10-Match Bagel/Breadstick Rate Δ", "category": "Short Form (10M)"},
    "short_avg_games_per_set_diff": {"label": "10-Match Avg Games/Set Δ", "category": "Short Form (10M)"},
    "med_winrate_diff": {"label": "365d Win Rate Δ", "category": "Med Form (365d)"},
    "med_tb_winrate_diff": {"label": "365d Tiebreak Win Rate Δ", "category": "Med Form (365d)"},
    "med_comeback_rate_diff": {"label": "365d Comeback Rate Δ", "category": "Med Form (365d)"},
    "med_bb_rate_diff": {"label": "365d Bagel/Breadstick Rate Δ", "category": "Med Form (365d)"},
    "med_avg_games_per_set_diff": {"label": "365d Avg Games/Set Δ", "category": "Med Form (365d)"},
    "med_ret_freq_diff": {"label": "365d Retirement Rate Δ", "category": "Med Form (365d)"},
    "career_winrate_diff": {"label": "Career Win Rate Δ", "category": "Career"},
    "career_tb_winrate_diff": {"label": "Career Tiebreak Win Rate Δ", "category": "Career"},
    "career_comeback_rate_diff": {"label": "Career Comeback Rate Δ", "category": "Career"},
    "career_bb_rate_diff": {"label": "Career Bagel/Breadstick Rate Δ", "category": "Career"},
    "career_avg_games_per_set_diff": {"label": "Career Avg Games/Set Δ", "category": "Career"},
    "career_ret_freq_diff": {"label": "Career Retirement Rate Δ", "category": "Career"},
    "surface_winrate_diff": {"label": "Surface Win Rate Δ", "category": "Context & Fatigue"},
    "days_since_last_diff": {"label": "Rest Days Δ", "category": "Context & Fatigue"},
    "matches_14d_diff": {"label": "14-Day Match Load Δ", "category": "Context & Fatigue"},
    "matches_30d_diff": {"label": "30-Day Match Load Δ", "category": "Context & Fatigue"},
    "form_divergence_diff": {"label": "Form Divergence (Short - Career) Δ", "category": "Context & Fatigue"},
    "h2h_winrate": {"label": "H2H Overall Win Rate", "category": "Head-to-Head"},
    "h2h_match_count": {"label": "H2H Total Matches", "category": "Head-to-Head"},
    "h2h_surface_winrate": {"label": "H2H Surface Win Rate", "category": "Head-to-Head"},
    "h2h_surface_count": {"label": "H2H Surface Matches", "category": "Head-to-Head"},
    "surface_encoded": {"label": "Court Surface", "category": "Match Context"},
    "series_encoded": {"label": "Tournament Series", "category": "Match Context"},
    "is_grand_slam": {"label": "Grand Slam Format (Bo5)", "category": "Match Context"},
}

_FEATURE_META_LIST = tuple(
    (
        name,
        FEATURE_META.get(name, {}).get("label", name),
        FEATURE_META.get(name, {}).get("category", "Other"),
    )
    for name in model_features
)

active_players: list[str] = []
all_players: list[str] = []
DEFAULT_P1 = "Alcaraz C."
DEFAULT_P2 = "Sinner J."

# In-memory H2H pair index: (min(p1,p2), max(p1,p2)) -> (sorted_ns_timestamps, surfaces, winners)
_NAT_NS = int(np.datetime64("NaT", "ns").astype("int64"))
_h2h_index: dict[tuple[str, str], tuple[list[int], list[str], list[str]]] = {}
_h2h_has_dates: bool = True
_h2h_source_sig: tuple[int, int] | None = None
_h2h_lock = threading.RLock()


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _get_lr_coef(feat_name: str, surf_code: int, ser_code: int) -> float:
    idx = _FEAT_INDEX.get(feat_name)
    if idx is not None and idx < 28:
        return _LR_NUM_COEFS[idx]
    if feat_name == "surface_encoded":
        return _safe_float(_LR_SURF_COEFS.get(int(surf_code), _lr_coef_lookup.get(f"cat__surface_encoded_{int(surf_code)}", 0.0)))
    if feat_name == "series_encoded":
        return _safe_float(_LR_SER_COEFS.get(int(ser_code), _lr_coef_lookup.get(f"cat__series_encoded_{int(ser_code)}", 0.0)))
    if feat_name == "is_grand_slam":
        return _safe_float(_LR_BIN_COEF if _FAST_LR_READY else _lr_coef_lookup.get("bin__is_grand_slam", 0.0))
    return _safe_float(_lr_coef_lookup.get(f"num__{feat_name}", 0.0))


def _build_h2h_index(df: pd.DataFrame | None) -> None:
    global _h2h_index, _h2h_has_dates, _h2h_source_sig
    with _h2h_lock:
        if df is None or df.empty:
            _h2h_index = {}
            _h2h_has_dates = False
            _h2h_source_sig = (id(df), 0)
            return

        p1s = df["Player_1"].to_numpy()
        p2s = df["Player_2"].to_numpy()
        surfs = df["Surface"].to_numpy() if "Surface" in df.columns else np.full(len(df), "", dtype=object)
        wins = df["Winner"].to_numpy() if "Winner" in df.columns else np.full(len(df), "", dtype=object)
        has_dates = "Date" in df.columns
        if has_dates:
            ts_arr = pd.to_datetime(df["Date"], errors="coerce").astype("datetime64[ns]").astype("int64").to_numpy()
        else:
            ts_arr = np.zeros(len(df), dtype=np.int64)

        temp: dict[tuple[str, str], tuple[list[int], list[str], list[str]]] = {}
        for a, b, t, s, w in zip(p1s, p2s, ts_arr, surfs, wins):
            pa = str(a)
            pb = str(b)
            key = (pa, pb) if pa <= pb else (pb, pa)
            t_int = int(t)
            s_str = "" if pd.isna(s) else str(s)
            w_str = "" if pd.isna(w) else str(w)
            entry = temp.get(key)
            if entry is None:
                temp[key] = ([t_int], [s_str], [w_str])
            else:
                entry[0].append(t_int)
                entry[1].append(s_str)
                entry[2].append(w_str)

        # Guarantee monotonic timestamp order per pair for bisect_left
        for key, (ts_list, surf_list, win_list) in temp.items():
            if len(ts_list) > 1 and any(ts_list[i] > ts_list[i + 1] for i in range(len(ts_list) - 1)):
                order = sorted(range(len(ts_list)), key=ts_list.__getitem__)
                temp[key] = (
                    [ts_list[i] for i in order],
                    [surf_list[i] for i in order],
                    [win_list[i] for i in order],
                )

        _h2h_index = temp
        _h2h_has_dates = has_dates
        _h2h_source_sig = (id(df), len(df))


def _ensure_h2h_index() -> None:
    df = atp_data.full_df
    sig = (id(df), len(df) if df is not None else 0)
    if _h2h_source_sig != sig:
        with _h2h_lock:
            df = atp_data.full_df
            sig = (id(df), len(df) if df is not None else 0)
            if _h2h_source_sig != sig:
                _build_h2h_index(df)


def _refresh_player_pools() -> None:
    global active_players, all_players, DEFAULT_P1, DEFAULT_P2
    active_players = sorted([
        p for p, s in latest_player_state.items()
        if pd.to_datetime(s.get("Date", "2000-01-01")).year >= 2023
    ])
    all_players = sorted(list(latest_player_state.keys()))
    DEFAULT_P1 = "Alcaraz C." if "Alcaraz C." in active_players else (active_players[0] if active_players else "")
    DEFAULT_P2 = "Sinner J." if "Sinner J." in active_players else (active_players[1] if len(active_players) > 1 else DEFAULT_P1)


def reload_state(cache_path: str | None = None) -> dict:
    """Reload refreshed player states (and dataset splits) from player_state_cache.joblib if present."""
    global latest_player_state, player_surface_wr
    atp_data.reload_data()
    target_cache = cache_path or PLAYER_STATE_CACHE_PATH
    if os.path.exists(target_cache):
        cached = joblib.load(target_cache)
        latest_player_state = cached.get("latest_player_state", bundle["latest_player_state"])
        player_surface_wr = cached.get("player_surface_wr", bundle["player_surface_wr"])
    else:
        latest_player_state = bundle["latest_player_state"]
        player_surface_wr = bundle["player_surface_wr"]
    _refresh_player_pools()
    _build_h2h_index(atp_data.full_df)
    return {
        "players_total": len(all_players),
        "players_active": len(active_players),
        "cache_loaded": os.path.exists(target_cache),
    }


if os.path.exists(PLAYER_STATE_CACHE_PATH):
    cached_state = joblib.load(PLAYER_STATE_CACHE_PATH)
    latest_player_state = cached_state.get("latest_player_state", latest_player_state)
    player_surface_wr = cached_state.get("player_surface_wr", player_surface_wr)

_refresh_player_pools()
_build_h2h_index(atp_data.full_df)


def get_h2h_stats(p1: str, p2: str, surface: str, cutoff_date=None) -> dict:
    _ensure_h2h_index()
    if not _h2h_index:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}

    key = (p1, p2) if p1 <= p2 else (p2, p1)
    entry = _h2h_index.get(key)
    if entry is None:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}

    ts_list, surf_list, win_list = entry
    if cutoff_date is not None and _h2h_has_dates:
        cutoff_dt = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff_dt):
            return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}
        cutoff_ns = pd.Timestamp(cutoff_dt).as_unit("ns").value
        start = bisect_right(ts_list, _NAT_NS) if ts_list[0] == _NAT_NS else 0
        end = bisect_left(ts_list, cutoff_ns, lo=start)
    else:
        start = 0
        end = len(ts_list)

    total = end - start
    if total <= 0:
        return {"h2h_winrate": 0.5, "h2h_match_count": 0, "h2h_surface_winrate": 0.5, "h2h_surface_count": 0}

    p1_wins = 0
    surf_total = 0
    surf_wins = 0
    for i in range(start, end):
        w = win_list[i]
        if w == p1:
            p1_wins += 1
        if surf_list[i] == surface:
            surf_total += 1
            if w == p1:
                surf_wins += 1

    return {
        "h2h_winrate": float(p1_wins / total),
        "h2h_match_count": int(total),
        "h2h_surface_winrate": float(surf_wins / surf_total) if surf_total > 0 else 0.5,
        "h2h_surface_count": int(surf_total),
    }


def _resolve_fatigue_counters(state: dict, target_date=None) -> tuple[float, float, float]:
    """Return (days_since_last, matches_14d, matches_30d) relative to target_date when provided."""
    if target_date is None or "Date" not in state:
        return (
            _safe_float(state.get("days_since_last", 14), 14.0),
            _safe_float(state.get("matches_14d", 0), 0.0),
            _safe_float(state.get("matches_30d", 0), 0.0),
        )

    target_dt = pd.to_datetime(target_date)
    last_dt = pd.to_datetime(state["Date"])
    days_since = float(max(0, (target_dt - last_dt).days))

    match_dates = state.get("match_dates")
    if match_dates:
        dt_series = pd.to_datetime(pd.Series(match_dates))
        prior = dt_series[dt_series < target_dt]
        m14 = float((prior >= (target_dt - pd.Timedelta(days=14))).sum())
        m30 = float((prior >= (target_dt - pd.Timedelta(days=30))).sum())
        return days_since, m14, m30

    return (
        days_since,
        _safe_float(state.get("matches_14d", 0), 0.0),
        _safe_float(state.get("matches_30d", 0), 0.0),
    )


def _build_player_card(player_name: str, surface: str) -> dict:
    """Build deterministic Arcade Player Card & 6-Axis Radar profile from latest_player_state."""
    is_known = player_name in latest_player_state
    s = latest_player_state.get(player_name, {})

    raw_rank = _safe_float(s.get("Rank"), 500.0)
    rank = int(round(raw_rank)) if raw_rank >= 1.0 else 500
    pts = max(0, int(round(_safe_float(s.get("Pts"), 0.0))))

    career_wr = min(1.0, max(0.0, _safe_float(s.get("career_winrate"), 0.5)))
    short_wr = min(1.0, max(0.0, _safe_float(s.get("short_winrate"), career_wr)))
    med_wr = min(1.0, max(0.0, _safe_float(s.get("med_winrate"), career_wr)))
    surface_wr = min(
        1.0,
        max(0.0, _safe_float(player_surface_wr.get((player_name, surface)), s.get("surface_winrate", career_wr))),
    )

    short_tb = min(1.0, max(0.0, _safe_float(s.get("short_tb_winrate"), 0.5)))
    med_tb = min(1.0, max(0.0, _safe_float(s.get("med_tb_winrate"), 0.5)))
    career_tb = min(1.0, max(0.0, _safe_float(s.get("career_tb_winrate"), 0.5)))
    comeback_rate = min(
        1.0, max(0.0, _safe_float(s.get("med_comeback_rate"), _safe_float(s.get("career_comeback_rate"), 0.0)))
    )
    bb_rate = min(1.0, max(0.0, _safe_float(s.get("med_bb_rate"), _safe_float(s.get("career_bb_rate"), 0.0))))
    ret_freq = max(0.0, _safe_float(s.get("med_ret_freq"), _safe_float(s.get("career_ret_freq"), 0.0)))

    days_since = max(0, int(round(_safe_float(s.get("days_since_last"), 14.0))))
    matches_14d = max(0, int(round(_safe_float(s.get("matches_14d"), 0.0))))
    matches_30d = max(0, int(round(_safe_float(s.get("matches_30d"), 0.0))))

    short_form = round(min(100.0, max(0.0, short_wr * 100.0)), 1)
    med_form = round(min(100.0, max(0.0, med_wr * 100.0)), 1)
    surface_mastery = round(min(100.0, max(0.0, surface_wr * 100.0)), 1)

    rank_score = 100.0 * (1.0 - math.log10(max(1, rank)) / math.log10(1000.0))
    pts_score = 100.0 * min(1.0, math.log1p(pts) / math.log1p(14000.0))
    rank_power = round(min(100.0, max(0.0, 0.55 * rank_score + 0.45 * pts_score)), 1)

    clutch_tb = round(min(100.0, max(0.0, (0.5 * med_tb + 0.3 * career_tb + 0.2 * short_tb) * 100.0)), 1)

    load_penalty = max(0, matches_14d - 3) * 6.0
    rust_penalty = min(35.0, max(0, days_since - 21) * 0.4)
    ret_penalty = min(30.0, ret_freq * 200.0)
    stamina_freshness = round(min(100.0, max(15.0, 92.0 - load_penalty - rust_penalty - ret_penalty)), 1)

    raw_ovr = (
        0.28 * rank_power
        + 0.22 * med_form
        + 0.18 * short_form
        + 0.17 * surface_mastery
        + 0.10 * clutch_tb
        + 0.05 * stamina_freshness
        + (5.0 if rank <= 10 else 0.0)
    )
    ovr = int(min(99, max(40, round(raw_ovr))))

    traits: list[str] = []
    if is_known:
        if rank <= 10:
            traits.append("TOP 10 ELITE")
        if short_wr >= 0.70:
            traits.append("STREAK FIRE")
        if surface_wr >= 0.72:
            traits.append("SURFACE SPECIALIST")
        if (0.5 * med_tb + 0.5 * career_tb) >= 0.58:
            traits.append("CLUTCH TIEBREAK")
        if comeback_rate >= 0.45:
            traits.append("COMEBACK KING")
        if bb_rate >= 0.12:
            traits.append("BAGEL HUNTER")
        if matches_14d >= 4 or matches_30d >= 8:
            traits.append("IRONMAN")
        if 2 <= days_since <= 10 and matches_14d <= 3:
            traits.append("FRESH LEGS")
    else:
        traits.append("COLD START")

    return {
        "name": player_name,
        "ovr": ovr,
        "rank": rank,
        "pts": pts,
        "surface_winrate": round(surface_wr, 4),
        "short_winrate": round(short_wr, 4),
        "career_winrate": round(career_wr, 4),
        "days_since_last": days_since,
        "matches_14d": matches_14d,
        "traits": traits,
        "radar": {
            "short_form": short_form,
            "med_form": med_form,
            "surface_mastery": surface_mastery,
            "rank_power": rank_power,
            "clutch_tb": clutch_tb,
            "stamina_freshness": stamina_freshness,
        },
    }


def compute_edge_and_kelly(prob_lgb: float, odd_1: float | None, odd_2: float | None) -> dict | None:
    """Power-Method de-vigging, Model Edge %, Signal Classification, and Quarter-Kelly stake."""
    if odd_1 is None or odd_2 is None:
        return None
    try:
        o1 = float(odd_1)
        o2 = float(odd_2)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(o1) and math.isfinite(o2)) or o1 <= 1.0 or o2 <= 1.0:
        return None

    q1 = 1.0 / o1
    q2 = 1.0 / o2
    overround = q1 + q2

    if abs(overround - 1.0) < 1e-12:
        p = 1.0
    else:
        def _pow_obj(x: float) -> float:
            return (q1 ** x) + (q2 ** x) - 1.0

        lo = 1e-6
        hi = 100.0
        while _pow_obj(hi) > 0.0 and hi < 1e9:
            hi *= 2.0
        while _pow_obj(lo) < 0.0 and lo > 1e-12:
            lo *= 0.5
        p = float(brentq(_pow_obj, lo, hi))

    power_k = float(1.0 / p)
    implied_prob_p1 = float(q1 ** p)
    implied_prob_p2 = float(1.0 - implied_prob_p1)

    p1_prob = float(prob_lgb)
    p2_prob = 1.0 - p1_prob
    edge_p1 = p1_prob - implied_prob_p1
    edge_p2 = p2_prob - implied_prob_p2

    if edge_p1 >= 0.03:
        signal = "VALUE_P1"
    elif edge_p2 >= 0.03:
        signal = "VALUE_P2"
    elif -0.02 < edge_p1 < 0.02:
        signal = "ALIGNMENT"
    else:
        signal = "NEGATIVE"

    f1 = max(0.0, ((o1 - 1.0) * p1_prob - (1.0 - p1_prob)) / (o1 - 1.0))
    f2 = max(0.0, ((o2 - 1.0) * p2_prob - (1.0 - p2_prob)) / (o2 - 1.0))
    kelly_p1_pct = round(min(2.5, 0.25 * f1 * 100.0), 4)
    kelly_p2_pct = round(min(2.5, 0.25 * f2 * 100.0), 4)

    return {
        "odd_1": o1,
        "odd_2": o2,
        "implied_prob_p1": round(implied_prob_p1, 6),
        "implied_prob_p2": round(implied_prob_p2, 6),
        "overround": round(overround, 6),
        "power_k": power_k,
        "edge_p1": round(edge_p1, 6),
        "edge_p2": round(edge_p2, 6),
        "signal": signal,
        "kelly_p1_pct": kelly_p1_pct,
        "kelly_p2_pct": kelly_p2_pct,
    }


def _fast_lr_predict_proba(X_np: np.ndarray) -> np.ndarray:
    """Pure-numpy LogReg probability evaluator matching champion_lr.predict_proba[:, 1] to <1e-12."""
    X_num = np.where(np.isnan(X_np[:, :28]), _LR_IMP_STATS, X_np[:, :28])
    X_scaled = (X_num - _LR_SCALER_CENTER) / _LR_SCALER_SCALE
    logits = _LR_INTERCEPT + (X_scaled @ _LR_NUM_COEF)
    out = np.empty(X_np.shape[0], dtype=np.float64)
    for r in range(X_np.shape[0]):
        s_code = int(X_np[r, 28])
        ser_code = int(X_np[r, 29])
        is_gs = float(X_np[r, 30])
        z = float(logits[r]) + _LR_SURF_COEFS.get(s_code, 0.0) + _LR_SER_COEFS.get(ser_code, 0.0) + is_gs * _LR_BIN_COEF
        out[r] = 1.0 / (1.0 + math.exp(-z))
    return out


def _build_explainability_payload(
    X_vec: pd.DataFrame | np.ndarray,
    shap_values: list[float],
    bias: float,
    surf_code: int,
    ser_code: int,
) -> tuple[dict, list[dict]]:
    vec = X_vec if isinstance(X_vec, np.ndarray) else X_vec.to_numpy(dtype=np.float64)[0]
    surf_coef = _safe_float(_LR_SURF_COEFS.get(int(surf_code), 0.0))
    ser_coef = _safe_float(_LR_SER_COEFS.get(int(ser_code), 0.0))
    bin_coef = _safe_float(_LR_BIN_COEF)
    features_31: list[dict] = []
    for i, (feat_name, label, category) in enumerate(_FEATURE_META_LIST):
        val = _safe_float(vec[i], 0.0)
        shap_val = _safe_float(shap_values[i], 0.0)
        favors = "p1" if shap_val > 1e-9 else ("p2" if shap_val < -1e-9 else "neutral")
        if i < 28:
            lr_c = _LR_NUM_COEFS[i]
        elif feat_name == "surface_encoded":
            lr_c = surf_coef
        elif feat_name == "series_encoded":
            lr_c = ser_coef
        elif feat_name == "is_grand_slam":
            lr_c = bin_coef
        else:
            lr_c = _get_lr_coef(feat_name, surf_code, ser_code)
        features_31.append({
            "name": feat_name,
            "label": label,
            "category": category,
            "value": val,
            "shap": shap_val,
            "lr_coef": lr_c,
            "favors": favors,
        })

    contributions = sorted(
        [
            {
                "feature": f["name"],
                "label": f["label"],
                "category": f["category"],
                "raw_delta": f["value"],
                "shap": f["shap"],
                "favors": f["favors"],
            }
            for f in features_31
        ],
        key=lambda item: abs(item["shap"]),
        reverse=True,
    )
    treeshap = {
        "bias": _safe_float(bias, 0.0),
        "p1_total_contrib": _safe_float(sum(shap_values), 0.0),
        "contributions": contributions,
    }
    return treeshap, features_31


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

    p1_days, p1_m14, p1_m30 = _resolve_fatigue_counters(s1, target_date=target_date)
    p2_days, p2_m14, p2_m30 = _resolve_fatigue_counters(s2, target_date=target_date)

    surf_code = _SURF_CODE_MAP.get(surface, _SURF_FALLBACK.get(surface, 3))
    ser_code = _SER_CODE_MAP.get(series, _SER_FALLBACK.get(series, 2))

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
        "days_since_last_diff": float(p1_days - p2_days),
        "matches_14d_diff": float(p1_m14 - p2_m14),
        "matches_30d_diff": float(p1_m30 - p2_m30),
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


_ORIG_BUILD_MATCH_FEATURES = build_match_features
_COLD_START_STATE = {
    "Rank": 500, "Pts": 0, "form_divergence": 0.0, "matches_14d": 0, "matches_30d": 0,
    "days_since_last": 14, "career_winrate": 0.5,
}


def _build_match_feature_pair_np(
    p1: str,
    p2: str,
    surface: str,
    series: str,
    target_date=None,
) -> tuple[np.ndarray, dict, bool, bool]:
    """Build (2, 31) float64 feature matrix [forward(p1,p2), inverse(p2,p1)] without DataFrames."""
    p1_known = p1 in latest_player_state
    p2_known = p2 in latest_player_state
    s1 = latest_player_state.get(p1, _COLD_START_STATE)
    s2 = latest_player_state.get(p2, _COLD_START_STATE)

    p1_swr = player_surface_wr.get((p1, surface), s1.get("career_winrate", 0.5))
    p2_swr = player_surface_wr.get((p2, surface), s2.get("career_winrate", 0.5))
    h2h = get_h2h_stats(p1, p2, surface, cutoff_date=target_date)

    p1_days, p1_m14, p1_m30 = _resolve_fatigue_counters(s1, target_date=target_date)
    p2_days, p2_m14, p2_m30 = _resolve_fatigue_counters(s2, target_date=target_date)

    surf_code = float(_SURF_CODE_MAP.get(surface, _SURF_FALLBACK.get(surface, 3)))
    ser_code = float(_SER_CODE_MAP.get(series, _SER_FALLBACK.get(series, 2)))
    is_gs = 1.0 if series == "Grand Slam" else 0.0

    X_pair = np.empty((2, 31), dtype=np.float64)
    # 0..23: directional delta features
    X_pair[0, 0] = float(s1.get("Rank", 500) - s2.get("Rank", 500))
    X_pair[0, 1] = float(s1.get("Pts", 0) - s2.get("Pts", 0))
    X_pair[0, 2] = float(s1.get("short_winrate", 0.5) - s2.get("short_winrate", 0.5))
    X_pair[0, 3] = float(s1.get("short_tb_winrate", 0.5) - s2.get("short_tb_winrate", 0.5))
    X_pair[0, 4] = float(s1.get("short_comeback_rate", 0.0) - s2.get("short_comeback_rate", 0.0))
    X_pair[0, 5] = float(s1.get("short_bb_rate", 0.0) - s2.get("short_bb_rate", 0.0))
    X_pair[0, 6] = float(s1.get("short_avg_games_per_set", 4.9) - s2.get("short_avg_games_per_set", 4.9))
    X_pair[0, 7] = float(s1.get("med_winrate", 0.5) - s2.get("med_winrate", 0.5))
    X_pair[0, 8] = float(s1.get("med_tb_winrate", 0.5) - s2.get("med_tb_winrate", 0.5))
    X_pair[0, 9] = float(s1.get("med_comeback_rate", 0.0) - s2.get("med_comeback_rate", 0.0))
    X_pair[0, 10] = float(s1.get("med_bb_rate", 0.0) - s2.get("med_bb_rate", 0.0))
    X_pair[0, 11] = float(s1.get("med_avg_games_per_set", 4.9) - s2.get("med_avg_games_per_set", 4.9))
    X_pair[0, 12] = float(s1.get("med_ret_freq", 0.0) - s2.get("med_ret_freq", 0.0))
    X_pair[0, 13] = float(s1.get("career_winrate", 0.5) - s2.get("career_winrate", 0.5))
    X_pair[0, 14] = float(s1.get("career_tb_winrate", 0.5) - s2.get("career_tb_winrate", 0.5))
    X_pair[0, 15] = float(s1.get("career_comeback_rate", 0.0) - s2.get("career_comeback_rate", 0.0))
    X_pair[0, 16] = float(s1.get("career_bb_rate", 0.0) - s2.get("career_bb_rate", 0.0))
    X_pair[0, 17] = float(s1.get("career_avg_games_per_set", 4.9) - s2.get("career_avg_games_per_set", 4.9))
    X_pair[0, 18] = float(s1.get("career_ret_freq", 0.0) - s2.get("career_ret_freq", 0.0))
    X_pair[0, 19] = float(p1_swr - p2_swr)
    X_pair[0, 20] = float(p1_days - p2_days)
    X_pair[0, 21] = float(p1_m14 - p2_m14)
    X_pair[0, 22] = float(p1_m30 - p2_m30)
    X_pair[0, 23] = float(s1.get("form_divergence", 0.0) - s2.get("form_divergence", 0.0))
    X_pair[1, :24] = -X_pair[0, :24]

    # 24..27: H2H features
    h2h_cnt = float(h2h["h2h_match_count"])
    h2h_wr = float(h2h["h2h_winrate"])
    h2h_scnt = float(h2h["h2h_surface_count"])
    h2h_swr = float(h2h["h2h_surface_winrate"])

    X_pair[0, 24] = h2h_wr
    X_pair[0, 25] = h2h_cnt
    X_pair[0, 26] = h2h_swr
    X_pair[0, 27] = h2h_scnt
    X_pair[1, 24] = (1.0 - h2h_wr) if h2h["h2h_match_count"] > 0 else 0.5
    X_pair[1, 25] = h2h_cnt
    X_pair[1, 26] = (1.0 - h2h_swr) if h2h["h2h_surface_count"] > 0 else 0.5
    X_pair[1, 27] = h2h_scnt

    # 28..30: match context
    X_pair[:, 28] = surf_code
    X_pair[:, 29] = ser_code
    X_pair[:, 30] = is_gs

    return X_pair, h2h, p1_known, p2_known


def predict_match_structured(
    p1: str,
    p2: str,
    surface: str,
    series: str,
    target_date=None,
    odd_1: float | None = None,
    odd_2: float | None = None,
) -> dict:
    """Compute dual-pass symmetrized predictions, TreeSHAP, Arcade cards, and optional Edge/Kelly."""
    if not p1 or not p2:
        return {"error": "Please select both Player 1 and Player 2."}
    if p1 == p2:
        return {"error": "Player 1 and Player 2 must be different athletes."}

    if build_match_features is not _ORIG_BUILD_MATCH_FEATURES:
        X_fwd = build_match_features(p1, p2, surface, series, target_date=target_date)
        X_inv = build_match_features(p2, p1, surface, series, target_date=target_date)
        fwd_vals = X_fwd.to_numpy(dtype=np.float64)[0]
        inv_vals = X_inv.to_numpy(dtype=np.float64)[0]
        X_batch_np = np.vstack([fwd_vals, inv_vals])
        p1_known = p1 in latest_player_state
        p2_known = p2 in latest_player_state
    else:
        X_batch_np, _, p1_known, p2_known = _build_match_feature_pair_np(
            p1, p2, surface, series, target_date=target_date
        )
        fwd_vals = X_batch_np[0]
        X_fwd = None
        X_inv = None

    n_feat = len(model_features)
    if champion_lgb is _FAST_LGB_SOURCE:
        contribs = champion_lgb.booster_.predict(X_batch_np, pred_contrib=True, num_threads=1)
        contrib_fwd = contribs[0]
        contrib_inv = contribs[1]
        raw_lgb_p1 = 1.0 / (1.0 + math.exp(-float(contrib_fwd.sum())))
        raw_lgb_p2 = 1.0 / (1.0 + math.exp(-float(contrib_inv.sum())))
    else:
        X_batch_df = (
            pd.concat([X_fwd, X_inv], ignore_index=True)
            if X_fwd is not None
            else pd.DataFrame(X_batch_np, columns=model_features)
        )
        lgb_probs = champion_lgb.predict_proba(X_batch_df)
        raw_lgb_p1 = float(lgb_probs[0, 1])
        raw_lgb_p2 = float(lgb_probs[1, 1])
        contribs = champion_lgb.booster_.predict(X_batch_df, pred_contrib=True, num_threads=1)
        contrib_fwd = contribs[0]
        contrib_inv = contribs[1]

    if _FAST_LR_READY and champion_lr is _FAST_LR_SOURCE:
        lr_probs_1d = _fast_lr_predict_proba(X_batch_np)
        raw_lr_p1 = float(lr_probs_1d[0])
        raw_lr_p2 = float(lr_probs_1d[1])
    else:
        X_batch_df = (
            pd.concat([X_fwd, X_inv], ignore_index=True)
            if X_fwd is not None
            else pd.DataFrame(X_batch_np, columns=model_features)
        )
        lr_probs = champion_lr.predict_proba(X_batch_df)
        raw_lr_p1 = float(lr_probs[0, 1])
        raw_lr_p2 = float(lr_probs[1, 1])

    raw_asym_lgb = abs((raw_lgb_p1 + raw_lgb_p2) - 1.0)
    prob_lgb = (raw_lgb_p1 + (1.0 - raw_lgb_p2)) / 2.0
    prob_lr = (raw_lr_p1 + (1.0 - raw_lr_p2)) / 2.0
    sym_score = abs((prob_lgb + (1.0 - prob_lgb)) - 1.0)

    div_delta = abs(prob_lgb - prob_lr)
    is_divergent = div_delta > 0.15

    h2h_total = int(round(_safe_float(fwd_vals[_FEAT_INDEX["h2h_match_count"]], 0.0)))
    h2h_wr = _safe_float(fwd_vals[_FEAT_INDEX["h2h_winrate"]], 0.5)
    h2h_p1_wins = int(round(h2h_wr * h2h_total))
    h2h_p2_wins = int(h2h_total - h2h_p1_wins)

    key_stats = {
        "rank_diff": _safe_float(fwd_vals[_FEAT_INDEX["rank_diff"]], 0.0),
        "surface_winrate_diff": _safe_float(fwd_vals[_FEAT_INDEX["surface_winrate_diff"]], 0.0),
        "form_divergence_diff": _safe_float(fwd_vals[_FEAT_INDEX["form_divergence_diff"]], 0.0),
        "matches_14d_diff": _safe_float(fwd_vals[_FEAT_INDEX["matches_14d_diff"]], 0.0),
        "h2h_p1_wins": h2h_p1_wins,
        "h2h_p2_wins": h2h_p2_wins,
    }

    # Symmetrized LightGBM TreeSHAP contributions
    shap_sym = [_safe_float((contrib_fwd[i] - contrib_inv[i]) / 2.0) for i in range(n_feat)]
    bias = _safe_float((contrib_fwd[n_feat] - contrib_inv[n_feat]) / 2.0)
    surf_code = int(_safe_float(fwd_vals[_FEAT_INDEX["surface_encoded"]], 3.0))
    ser_code = int(_safe_float(fwd_vals[_FEAT_INDEX["series_encoded"]], 2.0))
    treeshap, features_31 = _build_explainability_payload(fwd_vals, shap_sym, bias, surf_code, ser_code)

    player_cards = {
        "p1": _build_player_card(p1, surface),
        "p2": _build_player_card(p2, surface),
    }

    edge = compute_edge_and_kelly(prob_lgb, odd_1, odd_2)

    return {
        "p1": p1,
        "p2": p2,
        "surface": surface,
        "series": series,
        "prob_lgb": prob_lgb,
        "prob_lr": prob_lr,
        "key_stats": key_stats,
        "sym_score": sym_score,
        "raw_asym": raw_asym_lgb,
        "is_divergent": is_divergent,
        "div_delta": div_delta,
        "is_historical": False,
        "p1_known": p1 in latest_player_state,
        "p2_known": p2 in latest_player_state,
        "player_cards": player_cards,
        "treeshap": treeshap,
        "features_31": features_31,
        "edge": edge,
    }


def predict_matchup(p1: str, p2: str, surface: str, series: str) -> str:
    if not p1 or not p2:
        return "<div class='text-rose-400 p-4'>Please select both Player 1 and Player 2.</div>"
    if p1 == p2:
        return "<div class='text-amber-400 p-4'>Player 1 and Player 2 must be different athletes.</div>"

    res = predict_match_structured(p1, p2, surface, series)
    if "error" in res:
        return f"<div class='text-rose-400 p-4'>{res['error']}</div>"

    return render_diagnostic_card(
        p1=res["p1"], p2=res["p2"], surface=res["surface"], series=res["series"],
        prob_lgb=res["prob_lgb"], prob_lr=res["prob_lr"],
        key_stats=res["key_stats"], sym_score=res["sym_score"],
        raw_asym=res["raw_asym"], is_divergent=res["is_divergent"],
        div_delta=res["div_delta"], is_historical=False
    )


def inspect_historical_structured(split: str, tournament: str, match_idx_str: str) -> dict:
    """Compute structured historical match inspection payload with single-pass TreeSHAP & player cards."""
    if not match_idx_str or match_idx_str == "None":
        return {"error": "Select a valid tournament and matchup to inspect."}

    try:
        match_idx = int(match_idx_str)
    except (ValueError, TypeError):
        return {"error": "Invalid match identifier."}

    df = get_df_for_split(split)
    if df.empty or match_idx not in df.index:
        return {"error": "Selected fixture not found in active dataset."}

    row = df.loc[match_idx]
    p1 = str(row["Player_1"])
    p2 = str(row["Player_2"])
    surface = str(row.get("Surface", "Hard"))
    series = str(row.get("Series", "Grand Slam"))
    winner = str(row.get("Winner", "Unknown"))
    raw_score = row.get("Score", None)
    score_str = None if pd.isna(raw_score) else str(raw_score)
    round_name = str(row.get("Round", ""))
    date_str = str(row.get("Date", ""))[:10]
    tourn_name = tournament or str(row.get("Tournament", ""))

    # Ingest precomputed feature vector
    X_vec = df.loc[[match_idx], model_features].apply(pd.to_numeric)
    prob_lgb = float(champion_lgb.predict_proba(X_vec)[0, 1])
    prob_lr = float(champion_lr.predict_proba(X_vec)[0, 1])

    # Implied probability baseline
    bm_prob = None
    if "implied_prob_diff" in row and pd.notnull(row["implied_prob_diff"]):
        bm_prob = _safe_float((row["implied_prob_diff"] + 1.0) / 2.0, 0.5)

    h2h_data = get_h2h_stats(p1, p2, surface, cutoff_date=row.get("Date", None))
    h2h_p1_wins = int(round(h2h_data["h2h_winrate"] * h2h_data["h2h_match_count"]))
    h2h_p2_wins = int(h2h_data["h2h_match_count"] - h2h_p1_wins)

    key_stats = {
        "rank_diff": _safe_float(row.get("rank_diff", 0.0), 0.0),
        "surface_winrate_diff": _safe_float(row.get("surface_winrate_diff", 0.0), 0.0),
        "form_divergence_diff": _safe_float(row.get("form_divergence_diff", 0.0), 0.0),
        "matches_14d_diff": _safe_float(row.get("matches_14d_diff", 0.0), 0.0),
        "h2h_p1_wins": h2h_p1_wins,
        "h2h_p2_wins": h2h_p2_wins,
    }

    n_feat = len(model_features)
    contrib = champion_lgb.booster_.predict(X_vec, pred_contrib=True, num_threads=1)[0]
    shap_vals = [_safe_float(contrib[i]) for i in range(n_feat)]
    bias = _safe_float(contrib[n_feat])
    surf_code = int(_safe_float(X_vec["surface_encoded"].iloc[0], 3.0))
    ser_code = int(_safe_float(X_vec["series_encoded"].iloc[0], 2.0))
    treeshap, features_31 = _build_explainability_payload(X_vec, shap_vals, bias, surf_code, ser_code)

    player_cards = {
        "p1": _build_player_card(p1, surface),
        "p2": _build_player_card(p2, surface),
    }

    div_delta = abs(prob_lgb - prob_lr)
    is_divergent = div_delta > 0.15

    return {
        "p1": p1,
        "p2": p2,
        "surface": surface,
        "series": series,
        "prob_lgb": prob_lgb,
        "prob_lr": prob_lr,
        "key_stats": key_stats,
        "sym_score": 0.0,
        "raw_asym": 0.0,
        "is_divergent": is_divergent,
        "div_delta": div_delta,
        "bm_prob": bm_prob,
        "actual_winner": winner,
        "score_str": score_str,
        "date_str": date_str,
        "round_name": round_name,
        "tournament": tourn_name,
        "is_historical": True,
        "p1_known": p1 in latest_player_state,
        "p2_known": p2 in latest_player_state,
        "player_cards": player_cards,
        "treeshap": treeshap,
        "features_31": features_31,
        "edge": None,
        "prediction_accurate": bool((prob_lgb >= 0.5) == (winner == p1)),
    }


def inspect_historical_match(split: str, tournament: str, match_idx_str: str) -> str:
    res = inspect_historical_structured(split, tournament, match_idx_str)
    if "error" in res:
        err = res["error"]
        cls = "text-rose-400" if "not found" in err.lower() else "text-slate-400"
        return f"<div class='{cls} p-4'>{err}</div>"

    return render_diagnostic_card(
        p1=res["p1"], p2=res["p2"], surface=res["surface"], series=res["series"],
        prob_lgb=res["prob_lgb"], prob_lr=res["prob_lr"],
        key_stats=res["key_stats"], sym_score=res["sym_score"],
        is_divergent=res["is_divergent"],
        div_delta=res["div_delta"],
        bm_prob=res["bm_prob"], actual_winner=res["actual_winner"],
        score_str=res["score_str"], date_str=res["date_str"],
        round_name=res["round_name"], tournament=res["tournament"],
        is_historical=True
    )
