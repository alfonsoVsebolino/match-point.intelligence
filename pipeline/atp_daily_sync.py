"""Daily Kaggle dataset sync, published ATP schedule ingestion, and automated dual-model prediction.

Replaces static version-pinned Kaggle loading with an automated daily pipeline:
1. Pulls the latest unversioned `dissfya/atp-tennis-2000-2023daily-pull` dataset via `kagglehub`.
2. Recomputes leakage-free rolling features (`test_features.csv`, `match_scores.csv`)
   and post-match player states (`data/player_state_cache.joblib`) while keeping trained
   `champion_lgb` and `champion_lr` weights intact.
3. Fetches newly published daily ATP schedules (ESPN public ATP scoreboard, local CSV/JSON
   schedule drop-in, or unplayed rows from Kaggle), resolves player names, and generates
   symmetrized dual-model win probabilities.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import atp_engine
from pipeline import atp_data, atp_features
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_DIR = os.path.join(DATA_DIR, "predictions_history")

DEFAULT_KAGGLE_HANDLE = "dissfya/atp-tennis-2000-2023daily-pull"
DEFAULT_KAGGLE_FILE = "atp_tennis.csv"
ESPN_ATP_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"

DEFAULT_SCHEDULE_CSV = os.path.join(DATA_DIR, "upcoming_schedule.csv")
PENDING_KAGGLE_CSV = os.path.join(DATA_DIR, "pending_kaggle_fixtures.csv")
PLAYER_STATE_CACHE_PATH = os.path.join(DATA_DIR, "player_state_cache.joblib")
SYNC_STATE_PATH = os.path.join(DATA_DIR, "sync_state.json")
DAILY_PREDICTIONS_JSON_PATH = os.path.join(DATA_DIR, "daily_predictions_latest.json")
DAILY_PREDICTIONS_CSV_PATH = os.path.join(DATA_DIR, "daily_predictions_latest.csv")

VALID_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}
VALID_SERIES = {
    "ATP250",
    "ATP500",
    "Grand Slam",
    "International",
    "International Gold",
    "Masters",
    "Masters 1000",
    "Masters Cup",
}

# Canonical tournament context hints for public schedule feeds lacking explicit surface/series tags
TOURNAMENT_HINTS: list[tuple[tuple[str, ...], str, str]] = [
    (("australian open",), "Hard", "Grand Slam"),
    (("french open", "roland garros"), "Clay", "Grand Slam"),
    (("wimbledon",), "Grass", "Grand Slam"),
    (("us open",), "Hard", "Grand Slam"),
    (("atp finals", "nitto atp finals", "masters cup"), "Hard", "Masters Cup"),
    (("indian wells", "bnp paribas open"), "Hard", "Masters 1000"),
    (("miami",), "Hard", "Masters 1000"),
    (("monte carlo", "monte-carlo"), "Clay", "Masters 1000"),
    (("madrid",), "Clay", "Masters 1000"),
    (("rome", "internazionali", "italian open"), "Clay", "Masters 1000"),
    (("canada", "montreal", "toronto", "rogers cup", "national bank open"), "Hard", "Masters 1000"),
    (("cincinnati",), "Hard", "Masters 1000"),
    (("shanghai",), "Hard", "Masters 1000"),
    (("paris", "rolex paris"), "Hard", "Masters 1000"),
    (("rotterdam",), "Hard", "ATP500"),
    (("doha", "qatar"), "Hard", "ATP500"),
    (("rio",), "Clay", "ATP500"),
    (("dubai",), "Hard", "ATP500"),
    (("acapulco", "mexican open"), "Hard", "ATP500"),
    (("dallas",), "Hard", "ATP500"),
    (("barcelona",), "Clay", "ATP500"),
    (("munich", "bavarian"), "Clay", "ATP500"),
    (("hamburg",), "Clay", "ATP500"),
    (("queen", "london"), "Grass", "ATP500"),
    (("halle",), "Grass", "ATP500"),
    (("washington", "citi open"), "Hard", "ATP500"),
    (("tokyo", "japan open"), "Hard", "ATP500"),
    (("beijing", "china open"), "Hard", "ATP500"),
    (("vienna", "erste bank"), "Hard", "ATP500"),
    (("basel", "swiss indoors"), "Hard", "ATP500"),
]


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_write_joblib(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(obj, tmp_path, compress=3)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize_token(text: str) -> str:
    cleaned = _strip_accents(str(text or "")).replace("'", " ").replace("’", " ").replace("`", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def infer_tournament_context(
    tournament_name: str,
    raw_surface: Optional[str] = None,
    raw_series: Optional[str] = None,
) -> tuple[str, str]:
    """Infer valid (Surface, Series) tags from explicit metadata or tournament name."""
    surf = str(raw_surface or "").strip().title()
    if surf not in VALID_SURFACES:
        surf = ""

    series = str(raw_series or "").strip()
    if series not in VALID_SERIES:
        series_lower = series.lower()
        if "grand slam" in series_lower:
            series = "Grand Slam"
        elif "1000" in series_lower or "masters" in series_lower:
            series = "Masters 1000"
        elif "500" in series_lower:
            series = "ATP500"
        elif "250" in series_lower:
            series = "ATP250"
        else:
            series = ""

    t_lower = _normalize_token(tournament_name)
    hinted_surf, hinted_series = "", ""
    for keywords, s_val, ser_val in TOURNAMENT_HINTS:
        if any(kw in t_lower for kw in keywords):
            hinted_surf, hinted_series = s_val, ser_val
            break

    final_surf = surf or hinted_surf or "Hard"
    final_series = series or hinted_series or "ATP250"
    return final_surf, final_series


class PlayerEntityResolver:
    """Resolve full athlete names (e.g. 'Carlos Alcaraz') to Kaggle canonical format ('Alcaraz C.')."""

    def __init__(self, known_players: list[str], active_players: Optional[list[str]] = None):
        self.known_players = list(known_players)
        self.active_set = set(active_players or [])
        self.exact_map: dict[str, str] = {}
        self.norm_map: dict[str, str] = {}
        self.surname_initial_map: dict[tuple[str, str], list[str]] = {}

        # Sort so active players win deterministic tie-breaks when iterating
        ordered = sorted(self.known_players, key=lambda p: (0 if p in self.active_set else 1, p))
        for p in ordered:
            self.exact_map[p] = p
            norm = _normalize_token(p)
            self.norm_map.setdefault(norm, p)

            # Parse canonical "Surname [Surname2] F.[M.]"
            m = re.match(r"^(.+?)\s+([a-z](?:\.[a-z])*\.?)$", norm)
            if m:
                surname = m.group(1).strip()
                initials = m.group(2).replace(".", "")
                if initials:
                    self.surname_initial_map.setdefault((surname, initials[0]), []).append(p)
                    self.surname_initial_map.setdefault((surname, initials), []).append(p)

        self.norm_keys = list(self.norm_map.keys())

    def resolve(self, raw_name: str) -> tuple[str, float, str]:
        """Return (canonical_name, confidence, method)."""
        name = str(raw_name or "").strip()
        if not name:
            return "", 0.0, "empty"

        if name in self.exact_map:
            return self.exact_map[name], 1.0, "exact"

        norm = _normalize_token(name)
        if norm in self.norm_map:
            return self.norm_map[norm], 1.0, "normalized"

        # Generate candidate canonical forms from "First [Middle...] Last"
        tokens = [t for t in re.split(r"\s+", norm) if t]
        candidates: list[str] = []

        if len(tokens) >= 2:
            # 1) Single first name + remaining surname (handles "Alex de Minaur" -> "de minaur a.")
            first = tokens[0]
            first_parts = [p for p in first.replace(".", " ").replace("-", " ").split() if p]
            surname_rest = " ".join(tokens[1:])
            if first_parts:
                candidates.append(f"{surname_rest} {first_parts[0][0]}.")
                if len(first_parts) > 1:
                    joined_init = ".".join(p[0] for p in first_parts) + "."
                    candidates.append(f"{surname_rest} {joined_init}")

            # 2) Multi-word given name + single-token surname (handles "Juan Manuel Cerundolo" -> "cerundolo j.m.")
            if len(tokens) >= 3:
                last_one = tokens[-1]
                given_initials = ".".join(t[0] for t in tokens[:-1] if t) + "."
                candidates.append(f"{last_one} {given_initials}")
                candidates.append(f"{last_one} {tokens[0][0]}.")
                last_two = " ".join(tokens[-2:])
                candidates.append(f"{last_two} {tokens[0][0]}.")

            # 3) Check surname + initial index
            for split_idx in range(1, len(tokens)):
                surn = " ".join(tokens[split_idx:])
                givens = tokens[:split_idx]
                init_full = "".join(g[0] for g in givens if g)
                init_first = givens[0][0] if givens and givens[0] else ""
                for init_key in (init_full, init_first):
                    if init_key and (surn, init_key) in self.surname_initial_map:
                        hits = self.surname_initial_map[(surn, init_key)]
                        return hits[0], 0.95, "surname_initial"

        for cand in candidates:
            if cand in self.norm_map:
                return self.norm_map[cand], 0.95, "rule_canonical"

        # Fuzzy fallback over generated candidates or normalized name
        for query in candidates + [norm]:
            close = difflib.get_close_matches(query, self.norm_keys, n=1, cutoff=0.84)
            if close:
                return self.norm_map[close[0]], 0.85, "fuzzy"

        return name, 0.0, "cold_start"


def fetch_kaggle_raw_df(
    dataset_handle: str = DEFAULT_KAGGLE_HANDLE,
    filename: str = DEFAULT_KAGGLE_FILE,
    force_download: bool = True,
    local_raw_path: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str]:
    """Download or load raw ATP Kaggle CSV without pinned version locks.

    Returns:
        (raw_df, resolved_csv_path, sha256_hex)
    """
    if local_raw_path:
        if not os.path.exists(local_raw_path):
            raise FileNotFoundError(f"Local raw CSV not found: {local_raw_path}")
        csv_path = local_raw_path
    else:
        import kagglehub

        downloaded = kagglehub.dataset_download(
            dataset_handle,
            path=filename,
            force_download=force_download,
        )
        csv_path = downloaded if os.path.isfile(downloaded) else os.path.join(downloaded, filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Downloaded Kaggle file not found at {csv_path}")

    sha = _file_sha256(csv_path)
    raw_df = pd.read_csv(csv_path)
    return raw_df, csv_path, sha


def sync_kaggle_dataset(
    dataset_handle: str = DEFAULT_KAGGLE_HANDLE,
    filename: str = DEFAULT_KAGGLE_FILE,
    force: bool = False,
    local_raw_path: Optional[str] = None,
    update_train_csv: bool = False,
) -> dict[str, Any]:
    """Fetch latest daily Kaggle dataset and refresh local feature splits & player states."""
    _ensure_dirs()

    raw_df, source_path, raw_sha = fetch_kaggle_raw_df(
        dataset_handle=dataset_handle,
        filename=filename,
        force_download=True,
        local_raw_path=local_raw_path,
    )

    prev_state = {}
    if os.path.exists(SYNC_STATE_PATH):
        try:
            with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
                prev_state = json.load(f)
        except Exception:
            prev_state = {}

    if not force and prev_state.get("raw_sha256") == raw_sha and os.path.exists(PLAYER_STATE_CACHE_PATH):
        return {
            "status": "up_to_date",
            "raw_rows": int(len(raw_df)),
            "raw_sha256": raw_sha,
            "latest_match_date": prev_state.get("latest_match_date"),
            "new_matches": 0,
        }

    completed_df, pending_df = atp_features.clean_raw_matches(raw_df)

    # Preserve exact categorical encoders from the trained inference bundle
    engineered = atp_features.compute_match_features(
        completed_df,
        le_surf=atp_engine.le_surf,
        le_series=atp_engine.le_series,
        cutoff_year=atp_features.CUTOFF_YEAR,
    )

    test_df = engineered["test_df"]
    train_df = engineered["train_df"]
    scores_df = engineered["scores_df"]
    latest_player_state = engineered["latest_player_state"]
    player_surface_wr = engineered["player_surface_wr"]

    prev_completed = int(prev_state.get("completed_rows", len(atp_data.full_df) if not atp_data.full_df.empty else 0))
    new_matches = max(0, len(completed_df) - prev_completed)

    _atomic_write_csv(test_df, atp_data.TEST_CSV_PATH)
    _atomic_write_csv(scores_df, atp_data.SCORES_CSV_PATH)
    if update_train_csv or not os.path.exists(atp_data.TRAIN_CSV_PATH):
        _atomic_write_csv(train_df, atp_data.TRAIN_CSV_PATH)

    if not pending_df.empty:
        pending_out = pending_df.copy()
        pending_out["Date"] = pd.to_datetime(pending_out["Date"]).dt.strftime("%Y-%m-%d")
        _atomic_write_csv(pending_out, PENDING_KAGGLE_CSV)

    latest_date = str(test_df["Date"].max())[:10] if not test_df.empty else (
        str(train_df["Date"].max())[:10] if not train_df.empty else None
    )
    synced_at = datetime.now(timezone.utc).isoformat()

    cache_payload = {
        "latest_player_state": latest_player_state,
        "player_surface_wr": player_surface_wr,
        "synced_at": synced_at,
        "latest_match_date": latest_date,
        "completed_rows": int(len(completed_df)),
    }
    _atomic_write_joblib(cache_payload, PLAYER_STATE_CACHE_PATH)

    sync_meta = {
        "status": "synced",
        "dataset_handle": dataset_handle if not local_raw_path else f"local:{os.path.basename(source_path)}",
        "synced_at": synced_at,
        "raw_sha256": raw_sha,
        "raw_rows": int(len(raw_df)),
        "completed_rows": int(len(completed_df)),
        "pending_kaggle_rows": int(len(pending_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "new_matches": int(new_matches),
        "players_tracked": int(len(latest_player_state)),
        "latest_match_date": latest_date,
    }
    _atomic_write_text(SYNC_STATE_PATH, json.dumps(sync_meta, indent=2))

    # Refresh in-memory dataframes & engine state
    atp_engine.reload_state(PLAYER_STATE_CACHE_PATH)

    return sync_meta


def _parse_espn_competitions(
    competitions: list[dict],
    tournament_name: str,
    target_date_str: str,
    default_surface: str,
    default_series: str,
) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for comp in competitions:
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Filter out doubles matches ("/" in athlete displayName)
        p1_raw = (competitors[0].get("athlete") or {}).get("displayName") or competitors[0].get("displayName", "")
        p2_raw = (competitors[1].get("athlete") or {}).get("displayName") or competitors[1].get("displayName", "")
        if not p1_raw or not p2_raw or "/" in p1_raw or "/" in p2_raw:
            continue
        if "tbd" in p1_raw.lower() or "tbd" in p2_raw.lower():
            continue

        comp_date = str(comp.get("date", ""))[:10] or target_date_str
        round_info = comp.get("round") or {}
        round_name = round_info.get("displayName") or round_info.get("name") or "Main Draw"
        status_type = (comp.get("status") or {}).get("type") or {}
        status_name = str(status_type.get("name", "STATUS_SCHEDULED"))
        status_desc = str(status_type.get("description") or status_type.get("shortDetail") or "Scheduled")

        if "FINAL" in status_name.upper():
            lifecycle = "Completed"
        elif "IN_PROGRESS" in status_name.upper() or "LIVE" in status_name.upper():
            lifecycle = "LiveInPlay"
        elif "CANCEL" in status_name.upper() or "POSTPONE" in status_name.upper():
            lifecycle = "Cancelled"
        else:
            lifecycle = "LineupsConfirmed"

        fixtures.append({
            "fixture_id": str(comp.get("id") or f"{comp_date}_{tournament_name}_{p1_raw}_{p2_raw}"),
            "Date": comp_date,
            "Scheduled_Time": comp.get("date", ""),
            "Tournament": tournament_name,
            "Surface": default_surface,
            "Series": default_series,
            "Round": round_name,
            "Player_1": p1_raw.strip(),
            "Player_2": p2_raw.strip(),
            "Status": lifecycle,
            "Status_Detail": status_desc,
            "Source": "espn_atp",
        })
    return fixtures


def fetch_espn_atp_schedule(
    target_date: Optional[str] = None,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Fetch published ATP singles schedule from ESPN's public scoreboard JSON API."""
    date_dt = pd.to_datetime(target_date) if target_date else pd.Timestamp.now(tz="UTC")
    date_param = date_dt.strftime("%Y%m%d")
    date_iso = date_dt.strftime("%Y-%m-%d")

    resp = requests.get(
        ESPN_ATP_SCOREBOARD_URL,
        params={"dates": date_param},
        headers={"User-Agent": "MatchPointIntelligence/1.0"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()

    fixtures: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        tourn_name = event.get("name") or event.get("shortName") or "ATP Tour"
        venue = event.get("venue") or {}
        raw_surface = venue.get("surface") or event.get("surface")
        raw_series = (event.get("season") or {}).get("displayName") or event.get("category")
        surf, series = infer_tournament_context(tourn_name, raw_surface=raw_surface, raw_series=raw_series)

        # ESPN tennis events nest singles matches inside groupings -> competitions
        groupings = event.get("groupings", [])
        if groupings:
            for grp in groupings:
                grp_name = str((grp.get("grouping") or {}).get("displayName") or "").lower()
                if grp_name and "singles" not in grp_name:
                    continue
                comps = grp.get("competitions", [])
                fixtures.extend(_parse_espn_competitions(comps, tourn_name, date_iso, surf, series))
        elif event.get("competitions"):
            fixtures.extend(
                _parse_espn_competitions(event["competitions"], tourn_name, date_iso, surf, series)
            )

    return fixtures


def load_schedule_from_file(path: str) -> list[dict[str, Any]]:
    """Load published ATP fixtures from a CSV or JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schedule file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("fixtures", data) if isinstance(data, dict) else data
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    if df.empty:
        return []

    fixtures: list[dict[str, Any]] = []
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for idx, row in df.iterrows():
        p1 = str(row.get("Player_1", "")).strip()
        p2 = str(row.get("Player_2", "")).strip()
        if not p1 or not p2 or p1.lower() == "nan" or p2.lower() == "nan":
            continue
        tourn = str(row.get("Tournament", "ATP Tour")).strip()
        date_str = str(row.get("Date", today_str))[:10]
        if not date_str or date_str.lower() == "nan":
            date_str = today_str
        surf, series = infer_tournament_context(
            tourn,
            raw_surface=row.get("Surface"),
            raw_series=row.get("Series"),
        )
        odd1 = atp_features._to_positive_float(row.get("Odd_1"))
        odd2 = atp_features._to_positive_float(row.get("Odd_2"))
        fixtures.append({
            "fixture_id": str(row.get("fixture_id") or f"{date_str}_{tourn}_{p1}_{p2}_{idx}"),
            "Date": date_str,
            "Scheduled_Time": str(row.get("Scheduled_Time", "") or ""),
            "Tournament": tourn,
            "Surface": surf,
            "Series": series,
            "Round": str(row.get("Round", "Main Draw") or "Main Draw").strip(),
            "Player_1": p1,
            "Player_2": p2,
            "Odd_1": None if np.isnan(odd1) else float(odd1),
            "Odd_2": None if np.isnan(odd2) else float(odd2),
            "Status": str(row.get("Status", "LineupsConfirmed") or "LineupsConfirmed").strip(),
            "Status_Detail": str(row.get("Status_Detail", "Published Schedule") or "Published Schedule"),
            "Source": str(row.get("Source", f"file:{os.path.basename(path)}")),
        })
    return fixtures


def fetch_published_schedule(
    target_date: Optional[str] = None,
    schedule_file: Optional[str] = None,
    allow_espn: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch upcoming published ATP schedule using priority fallback chain:
    1) Explicit `--schedule-file` (CSV/JSON)
    2) Public ESPN ATP Scoreboard API (`allow_espn=True`)
    3) Default local drop-in `data/upcoming_schedule.csv`
    4) Unplayed/pending rows found in the Kaggle daily-pull CSV
    """
    if schedule_file:
        return load_schedule_from_file(schedule_file), f"file:{schedule_file}"

    espn_error: Optional[str] = None
    if allow_espn:
        try:
            espn_fixtures = fetch_espn_atp_schedule(target_date=target_date)
            if espn_fixtures:
                return espn_fixtures, "espn_atp"
        except Exception as exc:
            espn_error = str(exc)

    if os.path.exists(DEFAULT_SCHEDULE_CSV):
        file_fixtures = load_schedule_from_file(DEFAULT_SCHEDULE_CSV)
        if file_fixtures:
            return file_fixtures, f"file:{DEFAULT_SCHEDULE_CSV}"

    if os.path.exists(PENDING_KAGGLE_CSV):
        kaggle_pending = load_schedule_from_file(PENDING_KAGGLE_CSV)
        if kaggle_pending:
            return kaggle_pending, "kaggle_pending"

    source_note = "espn_empty" if espn_error is None else f"espn_unavailable ({espn_error})"
    return [], source_note


def _solve_power_method_devig(
    odd1: Optional[float], odd2: Optional[float]
) -> Optional[tuple[float, float, float, float]]:
    """Solve Power-Method de-vigging q1^(1/k) + q2^(1/k) = 1.0 with dynamic bracketing.

    Returns (implied_p1, implied_p2, overround, power_k) or None if odds are invalid.
    """
    if odd1 is None or odd2 is None or isinstance(odd1, bool) or isinstance(odd2, bool):
        return None
    try:
        o1 = float(odd1)
        o2 = float(odd2)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(o1) and math.isfinite(o2)) or o1 <= 1.0 or o2 <= 1.0:
        return None

    q1 = 1.0 / o1
    q2 = 1.0 / o2
    overround = q1 + q2
    if abs(overround - 1.0) < 1e-12:
        inv_k = 1.0
    else:
        def _pow_obj(x: float) -> float:
            return (q1 ** x) + (q2 ** x) - 1.0

        lo = 1e-6
        hi = 100.0
        while _pow_obj(hi) > 0.0 and hi < 1e6:
            hi *= 10.0
        while _pow_obj(lo) < 0.0 and lo > 1e-12:
            lo /= 10.0
        inv_k = float(brentq(_pow_obj, lo, hi))

    implied_p1 = float(q1 ** inv_k)
    implied_p2 = float(1.0 - implied_p1)
    power_k = float(1.0 / inv_k)
    return implied_p1, implied_p2, float(overround), power_k


def _compute_bookmaker_implied_prob(odd1: Optional[float], odd2: Optional[float]) -> Optional[float]:
    """Compute Power-Method de-vigged P1 implied probability solving q1^(1/k) + q2^(1/k) = 1.0."""
    solved = _solve_power_method_devig(odd1, odd2)
    return None if solved is None else solved[0]


def _compute_edge_and_kelly_safe(
    prob_lgb: float, odd1: Optional[float], odd2: Optional[float]
) -> Optional[dict[str, Any]]:
    """Delegate to atp_engine.compute_edge_and_kelly with dynamic-bracket fallback on ValueError."""
    if odd1 is None or odd2 is None or isinstance(odd1, bool) or isinstance(odd2, bool):
        return None
    try:
        return atp_engine.compute_edge_and_kelly(prob_lgb, odd1, odd2)
    except ValueError:
        solved = _solve_power_method_devig(odd1, odd2)
        if solved is None:
            return None
        implied_p1, implied_p2, overround, power_k = solved
        o1 = float(odd1)
        o2 = float(odd2)
        p1_prob = float(prob_lgb)
        p2_prob = 1.0 - p1_prob
        edge_p1 = p1_prob - implied_p1
        edge_p2 = p2_prob - implied_p2
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
        return {
            "odd_1": o1,
            "odd_2": o2,
            "implied_prob_p1": round(implied_p1, 6),
            "implied_prob_p2": round(implied_p2, 6),
            "overround": round(overround, 6),
            "power_k": power_k,
            "edge_p1": round(edge_p1, 6),
            "edge_p2": round(edge_p2, 6),
            "signal": signal,
            "kelly_p1_pct": round(min(2.5, 0.25 * f1 * 100.0), 4),
            "kelly_p2_pct": round(min(2.5, 0.25 * f2 * 100.0), 4),
        }


def _safe_optional_float(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _is_within_freeze_window(scheduled_time: str, now_utc: datetime) -> bool:
    if not scheduled_time:
        return False
    try:
        dt = pd.to_datetime(scheduled_time, utc=True)
        if pd.isna(dt):
            return False
        minutes_to_start = (dt.to_pydatetime() - now_utc).total_seconds() / 60.0
        return minutes_to_start <= 30.0
    except Exception:
        return False


def predict_published_schedule(
    fixtures: list[dict[str, Any]],
    source_label: str = "manual",
    target_date: Optional[str] = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Resolve player names and generate dual-model predictions for every published fixture."""
    _ensure_dirs()
    now_utc = datetime.now(timezone.utc)
    date_label = target_date or now_utc.strftime("%Y-%m-%d")

    resolver = PlayerEntityResolver(atp_engine.all_players, atp_engine.active_players)

    # Preserve previously frozen T-30m snapshots if re-run on the same day
    frozen_by_id: dict[str, dict[str, Any]] = {}
    if os.path.exists(DAILY_PREDICTIONS_JSON_PATH):
        try:
            with open(DAILY_PREDICTIONS_JSON_PATH, "r", encoding="utf-8") as f:
                prev_payload = json.load(f)
            for item in prev_payload.get("fixtures", []):
                if item.get("snapshot_locked") and item.get("fixture_id"):
                    frozen_by_id[item["fixture_id"]] = item
        except Exception:
            pass

    predicted_fixtures: list[dict[str, Any]] = []
    for fix in fixtures:
        fid = str(fix.get("fixture_id") or "")
        if fid in frozen_by_id:
            locked_copy = dict(frozen_by_id[fid])
            locked_copy["Status"] = fix.get("Status", locked_copy.get("Status"))
            locked_copy["Status_Detail"] = fix.get("Status_Detail", locked_copy.get("Status_Detail"))
            predicted_fixtures.append(locked_copy)
            continue

        p1_raw = str(fix.get("Player_1", "")).strip()
        p2_raw = str(fix.get("Player_2", "")).strip()
        p1_canon, p1_conf, p1_method = resolver.resolve(p1_raw)
        p2_canon, p2_conf, p2_method = resolver.resolve(p2_raw)

        surf, series = infer_tournament_context(
            str(fix.get("Tournament", "")),
            raw_surface=fix.get("Surface"),
            raw_series=fix.get("Series"),
        )
        match_date = str(fix.get("Date") or date_label)[:10]

        res = atp_engine.predict_match_structured(
            p1_canon,
            p2_canon,
            surf,
            series,
            target_date=match_date,
        )
        if "error" in res:
            continue

        odd_1 = _safe_optional_float(fix.get("Odd_1"))
        odd_2 = _safe_optional_float(fix.get("Odd_2"))
        edge_info = _compute_edge_and_kelly_safe(res["prob_lgb"], odd_1, odd_2)
        if edge_info is not None:
            bm_prob = edge_info["implied_prob_p1"]
            bm_prob_p2 = edge_info["implied_prob_p2"]
            overround = edge_info["overround"]
            power_k = edge_info["power_k"]
            edge_lgb = edge_info["edge_p1"]
            edge_signal = edge_info["signal"]
            kelly_p1_pct = edge_info["kelly_p1_pct"]
            kelly_p2_pct = edge_info["kelly_p2_pct"]
        else:
            bm_prob = None
            bm_prob_p2 = None
            overround = None
            power_k = None
            edge_lgb = None
            edge_signal = None
            kelly_p1_pct = None
            kelly_p2_pct = None

        favored = p1_canon if res["prob_lgb"] >= 0.5 else p2_canon
        status = str(fix.get("Status", "LineupsConfirmed"))
        locked = status in {"LiveInPlay", "Completed", "PreMatchFeaturesLocked"} or _is_within_freeze_window(
            str(fix.get("Scheduled_Time", "")), now_utc
        )

        predicted_fixtures.append({
            "fixture_id": fid,
            "Date": match_date,
            "Scheduled_Time": fix.get("Scheduled_Time", ""),
            "Tournament": fix.get("Tournament", "ATP Tour"),
            "Surface": surf,
            "Series": series,
            "Round": fix.get("Round", "Main Draw"),
            "Status": "PreMatchFeaturesLocked" if locked and status == "LineupsConfirmed" else status,
            "Status_Detail": fix.get("Status_Detail", "Scheduled"),
            "Source": fix.get("Source", source_label),
            "Player_1_Raw": p1_raw,
            "Player_2_Raw": p2_raw,
            "Player_1": p1_canon,
            "Player_2": p2_canon,
            "p1_resolution": {"confidence": p1_conf, "method": p1_method, "in_state_store": res["p1_known"]},
            "p2_resolution": {"confidence": p2_conf, "method": p2_method, "in_state_store": res["p2_known"]},
            "prob_lgb_p1": round(res["prob_lgb"], 6),
            "prob_lgb_p2": round(1.0 - res["prob_lgb"], 6),
            "prob_lr_p1": round(res["prob_lr"], 6),
            "prob_lr_p2": round(1.0 - res["prob_lr"], 6),
            "predicted_winner": favored,
            "is_divergent": bool(res["is_divergent"]),
            "div_delta": round(res["div_delta"], 6),
            "sym_score": res["sym_score"],
            "Odd_1": odd_1,
            "Odd_2": odd_2,
            "bm_implied_p1": bm_prob,
            "bm_implied_p2": bm_prob_p2,
            "overround": overround,
            "power_k": power_k,
            "edge_lgb_p1": edge_lgb,
            "edge_signal": edge_signal,
            "kelly_p1_pct": kelly_p1_pct,
            "kelly_p2_pct": kelly_p2_pct,
            "key_stats": res["key_stats"],
            "snapshot_locked": locked,
            "predicted_at": now_utc.isoformat(),
        })

    payload = {
        "generated_at": now_utc.isoformat(),
        "target_date": date_label,
        "source": source_label,
        "fixture_count": len(predicted_fixtures),
        "sync_state": atp_data.get_sync_status(),
        "fixtures": predicted_fixtures,
    }

    if save_outputs:
        _atomic_write_text(DAILY_PREDICTIONS_JSON_PATH, json.dumps(payload, indent=2))
        flat_rows = [
            {
                "Date": item["Date"],
                "Tournament": item["Tournament"],
                "Surface": item["Surface"],
                "Series": item["Series"],
                "Round": item["Round"],
                "Status": item["Status"],
                "Player_1": item["Player_1"],
                "Player_2": item["Player_2"],
                "Prob_LGB_P1": item["prob_lgb_p1"],
                "Prob_LR_P1": item["prob_lr_p1"],
                "Predicted_Winner": item["predicted_winner"],
                "Is_Divergent": item["is_divergent"],
                "Div_Delta": item["div_delta"],
                "Rank_Diff": item["key_stats"]["rank_diff"],
                "H2H_P1_Wins": item["key_stats"]["h2h_p1_wins"],
                "H2H_P2_Wins": item["key_stats"]["h2h_p2_wins"],
                "Odd_1": item.get("Odd_1"),
                "Odd_2": item.get("Odd_2"),
                "BM_Implied_P1": item.get("bm_implied_p1"),
                "Edge_LGB_P1": item.get("edge_lgb_p1"),
                "Overround": item.get("overround"),
                "Power_K": item.get("power_k"),
                "Kelly_P1_Pct": item.get("kelly_p1_pct"),
                "Kelly_P2_Pct": item.get("kelly_p2_pct"),
                "Edge_Signal": item.get("edge_signal"),
                "Snapshot_Locked": item["snapshot_locked"],
                "Source": item["Source"],
            }
            for item in predicted_fixtures
        ]
        flat_df = pd.DataFrame(flat_rows)
        _atomic_write_csv(flat_df, DAILY_PREDICTIONS_CSV_PATH)
        history_path = os.path.join(HISTORY_DIR, f"predictions_{date_label}.csv")
        _atomic_write_csv(flat_df, history_path)

    return payload


def run_daily_pipeline(
    target_date: Optional[str] = None,
    schedule_file: Optional[str] = None,
    local_raw_path: Optional[str] = None,
    skip_kaggle: bool = False,
    force_kaggle: bool = False,
    allow_espn: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end daily pipeline: Kaggle sync -> schedule fetch -> dual-model predictions."""
    sync_result: dict[str, Any]
    if skip_kaggle:
        sync_result = {"status": "skipped", **atp_data.get_sync_status()}
    else:
        sync_result = sync_kaggle_dataset(force=force_kaggle, local_raw_path=local_raw_path)

    fixtures, source_label = fetch_published_schedule(
        target_date=target_date,
        schedule_file=schedule_file,
        allow_espn=allow_espn,
    )
    predictions = predict_published_schedule(
        fixtures=fixtures,
        source_label=source_label,
        target_date=target_date,
        save_outputs=True,
    )
    return {
        "sync": sync_result,
        "predictions": predictions,
    }


def render_daily_schedule_html(payload: Optional[dict[str, Any]] = None) -> str:
    """Render published schedule predictions and sync telemetry as a dark-theme HTML feed."""
    data = payload if payload is not None else atp_data.get_daily_predictions()
    sync = data.get("sync_state") or atp_data.get_sync_status()
    fixtures = data.get("fixtures", [])

    last_sync = sync.get("last_synced_at") or sync.get("synced_at") or "Static bundled CSV"
    latest_match = sync.get("latest_match_date") or "unknown"
    test_rows = sync.get("test_rows", len(atp_data.test_df))
    gen_at = data.get("generated_at") or "Not yet run"
    src = data.get("source") or "none"

    header_html = f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px 16px; margin-bottom:14px; display:flex; flex-wrap:wrap; justify-content:space-between; gap:10px; font-family:'JetBrains Mono', monospace; font-size:12px;">
        <div><span style="color:#94a3b8;">Latest Kaggle Match:</span> <strong style="color:#2dd4bf;">{latest_match}</strong> <span style="color:#64748b;">({test_rows} holdout rows)</span></div>
        <div><span style="color:#94a3b8;">Last Sync:</span> <strong style="color:#e2e8f0;">{str(last_sync)[:19]}</strong></div>
        <div><span style="color:#94a3b8;">Schedule Source:</span> <strong style="color:#38bdf8;">{src}</strong></div>
        <div><span style="color:#94a3b8;">Generated:</span> <strong style="color:#e2e8f0;">{str(gen_at)[:19]}</strong></div>
    </div>
    """

    if not fixtures:
        return header_html + """
        <div style="background:#0f172a; border:1px dashed #334155; border-radius:12px; padding:24px; text-align:center; color:#94a3b8;">
            <div style="font-size:15px; font-weight:700; color:#f8fafc; margin-bottom:6px;">No Published ATP Fixtures Loaded</div>
            <div style="font-size:13px;">Run <code>python pipeline/atp_daily_sync.py run-daily</code> or click <strong>Sync Kaggle &amp; Fetch Schedule</strong> to pull today's ATP order of play.</div>
        </div>
        """

    cards: list[str] = []
    for fx in fixtures:
        p1 = fx["Player_1"]
        p2 = fx["Player_2"]
        lgb_p1 = float(fx["prob_lgb_p1"]) * 100.0
        lgb_p2 = 100.0 - lgb_p1
        lr_p1 = float(fx["prob_lr_p1"]) * 100.0
        lr_p2 = 100.0 - lr_p1
        surf = fx.get("Surface", "Hard")
        surf_col = {"Hard": "#38bdf8", "Clay": "#fb923c", "Grass": "#4ade80", "Carpet": "#a78bfa"}.get(surf, "#2dd4bf")
        div_badge = (
            f"<span style='background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#fde68a; font-size:11px; padding:2px 8px; border-radius:4px;'>⚠️ Divergent ({fx['div_delta']*100:.1f}%)</span>"
            if fx.get("is_divergent")
            else ""
        )
        edge_badge = ""
        if fx.get("edge_lgb_p1") is not None:
            edge_pct = float(fx["edge_lgb_p1"]) * 100.0
            edge_col = "#10b981" if edge_pct >= 3.0 else ("#f43f5e" if edge_pct <= -3.0 else "#94a3b8")
            edge_badge = f"<span style='color:{edge_col}; font-size:11px; font-family:\"JetBrains Mono\", monospace;'>Market Edge P1: {edge_pct:+.1f}%</span>"

        kelly_badge = ""
        kelly_val = _safe_optional_float(fx.get("kelly_p1_pct"))
        if kelly_val is not None and kelly_val > 0.0:
            kelly_badge = f"<span style='background:rgba(16,185,129,0.15); border:1px solid #10b981; color:#6ee7b7; font-size:11px; padding:2px 8px; border-radius:4px; font-family:\"JetBrains Mono\", monospace;'>1/4 Kelly P1: {kelly_val:.2f}%</span>"

        cold_warn = ""
        if not fx.get("p1_resolution", {}).get("in_state_store", True) or not fx.get("p2_resolution", {}).get("in_state_store", True):
            cold_warn = "<span style='color:#fbbf24; font-size:11px; margin-left:8px;'>[Cold-Start Player Fallback]</span>"

        cards.append(f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-left:4px solid {surf_col}; border-radius:10px; padding:14px 16px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
                <div style="font-size:15px; font-weight:700; color:#ffffff;">
                    {p1} <span style="color:#64748b; font-weight:400;">vs</span> {p2} {cold_warn}
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                    {div_badge}
                    {edge_badge}
                    {kelly_badge}
                    <span style="font-family:'JetBrains Mono', monospace; font-size:11px; padding:2px 8px; border-radius:4px; background:#1e293b; color:{surf_col};">{surf} • {fx.get('Series')}</span>
                </div>
            </div>
            <div style="font-size:12px; color:#94a3b8; font-family:'JetBrains Mono', monospace; margin-bottom:10px;">
                {fx.get('Date')} • {fx.get('Tournament')} ({fx.get('Round')}) • Status: <strong style="color:#e2e8f0;">{fx.get('Status')}</strong>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-family:'JetBrains Mono', monospace; font-size:12px;">
                <div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span style="color:#2dd4bf;">★ LightGBM</span>
                        <span style="color:#f8fafc;">{lgb_p1:.1f}% vs {lgb_p2:.1f}%</span>
                    </div>
                    <div style="height:6px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                        <div style="width:{lgb_p1:.1f}%; background:#14b8a6;"></div>
                        <div style="width:{lgb_p2:.1f}%; background:#334155;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <span style="color:#38bdf8;">LogReg Baseline</span>
                        <span style="color:#f8fafc;">{lr_p1:.1f}% vs {lr_p2:.1f}%</span>
                    </div>
                    <div style="height:6px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                        <div style="width:{lr_p1:.1f}%; background:#38bdf8;"></div>
                        <div style="width:{lr_p2:.1f}%; background:#334155;"></div>
                    </div>
                </div>
            </div>
        </div>
        """)

    return header_html + "".join(cards)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MatchPoint.intelligence Daily Kaggle Sync & Schedule Predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Fetch latest Kaggle dataset and refresh feature splits & player states")
    p_sync.add_argument("--raw-csv", default=None, help="Optional local CSV path instead of downloading from Kaggle")
    p_sync.add_argument("--force", action="store_true", help="Recompute features even if raw file SHA256 is unchanged")
    p_sync.add_argument("--update-train", action="store_true", help="Also overwrite train_features.csv (2000-2023)")

    p_pred = sub.add_parser("predict-schedule", help="Fetch published ATP schedule and run dual-model predictions")
    p_pred.add_argument("--date", default=None, help="Target schedule date (YYYY-MM-DD, default today UTC)")
    p_pred.add_argument("--schedule-file", default=None, help="Optional local CSV/JSON schedule file")
    p_pred.add_argument("--no-espn", action="store_true", help="Skip ESPN public API and use local files only")

    p_daily = sub.add_parser("run-daily", help="Run full daily Kaggle sync followed by schedule prediction")
    p_daily.add_argument("--date", default=None, help="Target schedule date (YYYY-MM-DD, default today UTC)")
    p_daily.add_argument("--schedule-file", default=None, help="Optional local CSV/JSON schedule file")
    p_daily.add_argument("--raw-csv", default=None, help="Optional local raw Kaggle CSV path")
    p_daily.add_argument("--no-kaggle", action="store_true", help="Skip Kaggle download and predict using current state")
    p_daily.add_argument("--no-espn", action="store_true", help="Skip ESPN public API and use local files only")
    p_daily.add_argument("--force", action="store_true", help="Force feature recompute even if unchanged")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.command == "sync":
        res = sync_kaggle_dataset(
            force=args.force,
            local_raw_path=args.raw_csv,
            update_train_csv=args.update_train,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "predict-schedule":
        fixtures, source_label = fetch_published_schedule(
            target_date=args.date,
            schedule_file=args.schedule_file,
            allow_espn=not args.no_espn,
        )
        res = predict_published_schedule(
            fixtures=fixtures,
            source_label=source_label,
            target_date=args.date,
            save_outputs=True,
        )
        print(json.dumps({k: v for k, v in res.items() if k != "fixtures"}, indent=2))
        print(f"Predicted {res['fixture_count']} fixtures -> {DAILY_PREDICTIONS_CSV_PATH}")
        return 0

    if args.command == "run-daily":
        res = run_daily_pipeline(
            target_date=args.date,
            schedule_file=args.schedule_file,
            local_raw_path=args.raw_csv,
            skip_kaggle=args.no_kaggle,
            force_kaggle=args.force,
            allow_espn=not args.no_espn,
        )
        summary = {
            "sync": res["sync"],
            "fixture_count": res["predictions"]["fixture_count"],
            "source": res["predictions"]["source"],
            "target_date": res["predictions"]["target_date"],
            "output_csv": DAILY_PREDICTIONS_CSV_PATH,
            "output_json": DAILY_PREDICTIONS_JSON_PATH,
        }
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
