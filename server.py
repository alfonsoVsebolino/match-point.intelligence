"""FastAPI + Uvicorn web server for MatchPoint.intelligence."""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import atp_daily_sync, atp_data
import atp_engine
import atp_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

SURFACES_ORDER = ["Hard", "Clay", "Grass", "Carpet"]
SERIES_ORDER = [
    "Grand Slam",
    "Masters 1000",
    "ATP500",
    "ATP250",
    "Masters Cup",
    "Masters",
    "International Gold",
    "International",
]

VALID_SURFACES = set(
    list(atp_engine.le_surf.classes_) if hasattr(atp_engine.le_surf, "classes_") else SURFACES_ORDER
)
VALID_SERIES = set(
    list(atp_engine.le_series.classes_) if atp_engine.le_series is not None and hasattr(atp_engine.le_series, "classes_") else SERIES_ORDER
)

app = FastAPI(
    title="MatchPoint.intelligence API",
    version="1.0.0",
    description="Dual-model ATP match inference, TreeSHAP explainability, Arcade scouting cards, and live schedule telemetry.",
)

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class PredictRequest(BaseModel):
    p1: str
    p2: str
    surface: str
    series: str
    odd_1: float | None = None
    odd_2: float | None = None


def _normalize_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _normalize_schedule_payload(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    sync_status = atp_data.get_sync_status()
    raw_fixtures = raw.get("fixtures") or []

    fixtures: list[dict[str, Any]] = []
    live_count = 0
    value_edges_count = 0
    divergent_count = 0

    for idx, item in enumerate(raw_fixtures):
        p1 = str(item.get("Player_1") or item.get("p1") or "")
        p2 = str(item.get("Player_2") or item.get("p2") or "")
        prob_lgb = _normalize_float(item.get("prob_lgb_p1", item.get("prob_lgb")))
        if prob_lgb is None:
            prob_lgb = 0.5
        prob_lr = _normalize_float(item.get("prob_lr_p1", item.get("prob_lr")))
        if prob_lr is None:
            prob_lr = 0.5

        div_delta = _normalize_float(item.get("div_delta"))
        if div_delta is None:
            div_delta = abs(prob_lgb - prob_lr)
        is_divergent = bool(item.get("is_divergent", div_delta > 0.15))

        odd_1 = _normalize_float(item.get("Odd_1", item.get("odd_1")))
        odd_2 = _normalize_float(item.get("Odd_2", item.get("odd_2")))

        edge_info = atp_engine.compute_edge_and_kelly(prob_lgb, odd_1, odd_2)
        if edge_info is not None:
            bm_prob = edge_info["implied_prob_p1"]
            edge_lgb_p1 = edge_info["edge_p1"]
            kelly_p1_pct = edge_info["kelly_p1_pct"]
            if edge_info["signal"] in ("VALUE_P1", "VALUE_P2"):
                value_edges_count += 1
        else:
            bm_prob = _normalize_float(item.get("bm_implied_p1", item.get("bm_prob")))
            edge_lgb_p1 = _normalize_float(item.get("edge_lgb_p1"))
            kelly_p1_pct = _normalize_float(item.get("kelly_p1_pct"))
            if edge_lgb_p1 is not None and abs(edge_lgb_p1) >= 0.03:
                value_edges_count += 1

        status = str(item.get("Status") or item.get("status") or "LineupsConfirmed")
        status_detail = str(item.get("Status_Detail") or item.get("status_detail") or "")
        if "live" in status.lower():
            live_count += 1
        if is_divergent:
            divergent_count += 1

        p1_res = item.get("p1_resolution") or {}
        p2_res = item.get("p2_resolution") or {}
        p1_known = p1_res.get("in_state_store", p1 in atp_engine.latest_player_state)
        p2_known = p2_res.get("in_state_store", p2 in atp_engine.latest_player_state)
        cold_start = bool(item.get("cold_start", (not p1_known) or (not p2_known)))

        fixtures.append({
            "fixture_id": str(item.get("fixture_id") or f"fix_{idx}"),
            "tournament": str(item.get("Tournament") or item.get("tournament") or "ATP Tour"),
            "round_name": str(item.get("Round") or item.get("round_name") or "Main Draw"),
            "surface": str(item.get("Surface") or item.get("surface") or "Hard"),
            "series": str(item.get("Series") or item.get("series") or "ATP250"),
            "status": status,
            "status_detail": status_detail,
            "p1": p1,
            "p2": p2,
            "prob_lgb": round(prob_lgb, 6),
            "prob_lr": round(prob_lr, 6),
            "is_divergent": is_divergent,
            "div_delta": round(div_delta, 6),
            "odd_1": odd_1,
            "odd_2": odd_2,
            "bm_prob": bm_prob,
            "edge_lgb_p1": edge_lgb_p1,
            "kelly_p1_pct": kelly_p1_pct,
            "snapshot_locked": bool(item.get("snapshot_locked", False)),
            "cold_start": cold_start,
        })

    return {
        "generated_at": raw.get("generated_at"),
        "target_date": raw.get("target_date"),
        "source": raw.get("source") or "none",
        "sync_status": sync_status,
        "kpis": {
            "total_fixtures": len(fixtures),
            "live_count": live_count,
            "value_edges_count": value_edges_count,
            "divergent_count": divergent_count,
        },
        "fixtures": fixtures,
    }


def _refresh_schedule_sync() -> tuple[dict[str, Any], bool, str | None]:
    existing = atp_service.get_daily_schedule()
    try:
        fixtures, source_label = atp_daily_sync.fetch_published_schedule(allow_espn=True)
    except Exception as exc:
        return existing, False, str(exc)

    if not fixtures:
        if existing.get("fixtures"):
            return existing, False, source_label or "No fixtures returned; preserved cached schedule."
        if "unavailable" in str(source_label).lower() or "error" in str(source_label).lower():
            return existing, False, source_label
        return existing, True, None

    try:
        updated = atp_daily_sync.predict_published_schedule(
            fixtures=fixtures,
            source_label=source_label,
            save_outputs=True,
        )
        return updated, True, None
    except Exception as exc:
        return existing, False, str(exc)


@app.get("/")
def serve_root():
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "MatchPoint.intelligence API ready; web/ not built yet"})


@app.get("/healthz")
def healthz():
    sched = atp_service.get_daily_schedule()
    return {
        "status": "ok",
        "models_loaded": bool(atp_engine.champion_lgb is not None and atp_engine.champion_lr is not None),
        "players_total": len(atp_engine.all_players),
        "players_active": len(atp_engine.active_players),
        "sync_status": atp_data.get_sync_status(),
        "schedule_fixtures_count": len(sched.get("fixtures") or []),
    }


@app.get("/api/v1/meta")
def get_meta(all_time: bool = Query(default=False)):
    _ = all_time
    trees = int(atp_engine.champion_lgb.booster_.num_trees())
    return {
        "active_players": atp_engine.active_players,
        "all_players": atp_engine.all_players,
        "surfaces": [s for s in SURFACES_ORDER if s in VALID_SURFACES] or sorted(VALID_SURFACES),
        "series": [s for s in SERIES_ORDER if s in VALID_SERIES] or sorted(VALID_SERIES),
        "splits": list(atp_data.SPLIT_CHOICES),
        "default_p1": atp_engine.DEFAULT_P1,
        "default_p2": atp_engine.DEFAULT_P2,
        "sync_status": atp_data.get_sync_status(),
        "model_metrics": {
            "lgbm_logloss": 0.6149,
            "lgbm_brier": 0.2139,
            "lgbm_auc": 0.7171,
            "trees": trees,
            "features_count": len(atp_engine.model_features),
        },
    }


@app.post("/api/v1/predict")
def post_predict(req: PredictRequest):
    if not req.p1.strip() or not req.p2.strip():
        raise HTTPException(status_code=400, detail="Please select both Player 1 and Player 2.")
    p1 = req.p1 if req.p1 in atp_engine.latest_player_state else req.p1.strip()
    p2 = req.p2 if req.p2 in atp_engine.latest_player_state else req.p2.strip()
    if p1 == p2:
        raise HTTPException(status_code=400, detail="Player 1 and Player 2 must be different athletes.")
    if req.surface not in VALID_SURFACES:
        raise HTTPException(status_code=400, detail=f"Invalid surface '{req.surface}'. Must be one of {sorted(VALID_SURFACES)}.")
    if req.series not in VALID_SERIES:
        raise HTTPException(status_code=400, detail=f"Invalid series '{req.series}'. Must be one of {sorted(VALID_SERIES)}.")

    res = atp_service.predict(
        p1,
        p2,
        req.surface,
        req.series,
        odd_1=req.odd_1,
        odd_2=req.odd_2,
    )
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.get("/api/v1/fixtures/historical")
def get_historical_fixtures(
    split: str = Query(default="Holdout Test (2024–2026)"),
    tournament: str | None = Query(default=None),
):
    if split not in atp_data.SPLIT_CHOICES:
        raise HTTPException(status_code=400, detail=f"Invalid split '{split}'. Must be one of {atp_data.SPLIT_CHOICES}.")

    tournaments = atp_data.get_tournaments_for_split(split)
    if tournament is None:
        selected = tournaments[0] if tournaments else None
    else:
        selected = tournament

    raw_matches = atp_data.get_matches_for_tournament(split, selected) if selected else []
    matches = [{"label": label, "match_idx": str(mid)} for label, mid in raw_matches]
    return {
        "split": split,
        "tournaments": tournaments,
        "selected_tournament": selected,
        "matches": matches,
    }


@app.get("/api/v1/fixtures/historical/inspect")
def inspect_historical_fixture(
    split: str = Query(default=""),
    tournament: str = Query(default=""),
    match_idx: str = Query(default=""),
):
    if not split or split not in atp_data.SPLIT_CHOICES:
        raise HTTPException(status_code=400, detail=f"Invalid split '{split}'.")
    if not match_idx or match_idx == "None":
        raise HTTPException(status_code=400, detail="Select a valid tournament and matchup to inspect.")
    try:
        int(match_idx)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid match identifier.")

    if not tournament or tournament not in atp_data.get_tournaments_for_split(split):
        raise HTTPException(status_code=404, detail="Selected fixture not found in specified tournament.")
    valid_ids = {str(mid) for _, mid in atp_data.get_matches_for_tournament(split, tournament)}
    if str(match_idx) not in valid_ids:
        raise HTTPException(status_code=404, detail="Selected fixture not found in specified tournament.")

    res = atp_service.inspect(split, tournament, match_idx)
    if "error" in res:
        err = res["error"]
        if "not found" in err.lower():
            raise HTTPException(status_code=404, detail=err)
        raise HTTPException(status_code=400, detail=err)
    return res


@app.get("/api/v1/schedule/live")
def get_live_schedule():
    raw = atp_service.get_daily_schedule()
    return _normalize_schedule_payload(raw)


@app.post("/api/v1/schedule/refresh")
async def refresh_live_schedule():
    raw, refreshed, refresh_error = await asyncio.to_thread(_refresh_schedule_sync)
    payload = _normalize_schedule_payload(raw)
    payload["refreshed"] = refreshed
    payload["refresh_error"] = refresh_error
    return payload


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
