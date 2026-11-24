"""Tier 2 Opaque-Box Boundary & Error-Handling Tests (F1–F9) for MatchPoint.intelligence.

Covers >= 5 distinct boundary, corner, concurrency, and failure-isolation cases per
feature F1–F9 without mocking the application under test or mutating repo state.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import re

import httpx
import pandas as pd
import pytest
import requests

from pipeline import atp_daily_sync, atp_data
from src import atp_engine

from tests.e2e.conftest import (
    RADAR_AXES,
    REPO_ROOT,
    assert_valid_features_31,
    assert_valid_player_card,
    create_inprocess_client,
    free_port,
    run_cdp_scenario,
    solve_power_devig,
    start_ephemeral_server,
)

pytestmark = pytest.mark.tier2


def _isolate_full_sync_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    data_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(atp_data, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_data, "TEST_CSV_PATH", str(tmp_path / "test_features.csv"))
    monkeypatch.setattr(atp_data, "TRAIN_CSV_PATH", str(tmp_path / "train_features.csv"))
    monkeypatch.setattr(atp_data, "SCORES_CSV_PATH", str(tmp_path / "match_scores.csv"))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(
        atp_data,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(data_dir / "daily_predictions_latest.json"),
    )

    monkeypatch.setattr(
        atp_engine, "PLAYER_STATE_CACHE_PATH", str(data_dir / "player_state_cache.joblib")
    )

    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(
        atp_daily_sync, "DEFAULT_SCHEDULE_CSV", str(data_dir / "upcoming_schedule.csv")
    )
    monkeypatch.setattr(
        atp_daily_sync, "PENDING_KAGGLE_CSV", str(data_dir / "pending_kaggle_fixtures.csv")
    )
    monkeypatch.setattr(
        atp_daily_sync, "PLAYER_STATE_CACHE_PATH", str(data_dir / "player_state_cache.joblib")
    )
    monkeypatch.setattr(atp_daily_sync, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(
        atp_daily_sync,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(data_dir / "daily_predictions_latest.json"),
    )
    monkeypatch.setattr(
        atp_daily_sync,
        "DAILY_PREDICTIONS_CSV_PATH",
        str(data_dir / "daily_predictions_latest.csv"),
    )
    return data_dir


def _sample_kaggle_raw_with_pending() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Tournament": "Australian Open",
            "Date": "2023-01-20",
            "Series": "Grand Slam",
            "Court": "Outdoor",
            "Surface": "Hard",
            "Round": "3rd Round",
            "Best of": 5,
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Winner": "Alcaraz C.",
            "Rank_1": 1,
            "Rank_2": 15,
            "Pts_1": 6800,
            "Pts_2": 2400,
            "Odd_1": 1.65,
            "Odd_2": 2.25,
            "Score": "6-4 4-6 7-6(4) 6-3",
        },
        {
            "Tournament": "Wimbledon",
            "Date": "2024-07-10",
            "Series": "Grand Slam",
            "Court": "Outdoor",
            "Surface": "Grass",
            "Round": "Semifinals",
            "Best of": 5,
            "Player_1": "Sinner J.",
            "Player_2": "Alcaraz C.",
            "Winner": "Sinner J.",
            "Rank_1": 1,
            "Rank_2": 3,
            "Pts_1": 9500,
            "Pts_2": 8100,
            "Odd_1": 1.80,
            "Odd_2": 2.05,
            "Score": "6-7(5) 6-4 6-3 6-4",
        },
        {
            "Tournament": "ATP Finals",
            "Date": "2026-11-21",
            "Series": "Masters Cup",
            "Court": "Indoor",
            "Surface": "Hard",
            "Round": "Semifinals",
            "Best of": 3,
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Winner": "",
            "Rank_1": 2,
            "Rank_2": 1,
            "Pts_1": 11000,
            "Pts_2": 11500,
            "Odd_1": 1.95,
            "Odd_2": 1.85,
            "Score": "",
        },
    ])


# ===========================================================================
# F1 Boundaries: Server Entrypoint, Env & HTTP Protocol (5 tests)
# ===========================================================================


def test_t2_f1_custom_port_env_binding():
    custom_port = free_port()
    with start_ephemeral_server(
        port=custom_port,
        env_overrides={"PORT": str(custom_port)},
        use_python_entrypoint=True,
    ) as srv:
        assert srv["port"] == custom_port
        with httpx.Client(base_url=srv["base_url"], trust_env=False, timeout=10.0) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


def test_t2_f1_missing_data_dir_startup_resilient(tmp_path, monkeypatch):
    missing_dir = tmp_path / "nonexistent_data_dir"
    monkeypatch.setattr(atp_data, "DATA_DIR", str(missing_dir))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(missing_dir / "sync_state.json"))
    monkeypatch.setattr(
        atp_data,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(missing_dir / "daily_predictions_latest.json"),
    )
    monkeypatch.setattr(
        atp_engine,
        "PLAYER_STATE_CACHE_PATH",
        str(missing_dir / "player_state_cache.joblib"),
    )
    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(missing_dir))
    monkeypatch.setattr(atp_daily_sync, "SYNC_STATE_PATH", str(missing_dir / "sync_state.json"))
    monkeypatch.setattr(
        atp_daily_sync,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(missing_dir / "daily_predictions_latest.json"),
    )
    monkeypatch.setattr(
        atp_daily_sync,
        "PLAYER_STATE_CACHE_PATH",
        str(missing_dir / "player_state_cache.joblib"),
    )
    atp_engine.reload_state()
    try:
        client = create_inprocess_client(monkeypatch)
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["models_loaded"] is True
        assert health.json()["schedule_fixtures_count"] == 0

        sched = client.get("/api/v1/schedule/live")
        assert sched.status_code == 200
        assert sched.json()["fixtures"] == []
        assert sched.json()["kpis"]["total_fixtures"] == 0
    finally:
        monkeypatch.undo()
        atp_engine.reload_state()


def test_t2_f1_unknown_route_returns_404(http_client):
    resp = http_client.get("/api/v1/does_not_exist_xyz")
    assert resp.status_code == 404


def test_t2_f1_malformed_json_body_returns_422(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_t2_f1_disallowed_http_method_returns_405(http_client):
    resp = http_client.delete("/healthz")
    assert resp.status_code == 405


# ===========================================================================
# F2 Boundaries: Validation, Cold-Start & Concurrency (6 tests)
# ===========================================================================


def test_t2_f2_identical_players_rejected(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"},
    )
    assert resp.status_code in (400, 422)


def test_t2_f2_empty_player_rejected(http_client):
    for bad_p1 in ("", "   "):
        resp = http_client.post(
            "/api/v1/predict",
            json={"p1": bad_p1, "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
        )
        assert resp.status_code in (400, 422)


def test_t2_f2_cold_start_unknown_rookie_predicts_cleanly(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Rookie Z.", "p2": "Alcaraz C.", "surface": "Hard", "series": "ATP250"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["p1_known"] is False
    assert data["p2_known"] is True
    assert 0.0 <= data["prob_lgb"] <= 1.0
    assert 0.0 <= data["prob_lr"] <= 1.0


def test_t2_f2_invalid_surface_rejected(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Wood", "series": "Grand Slam"},
    )
    assert resp.status_code in (400, 422)


def test_t2_f2_invalid_series_rejected(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Exhibition"},
    )
    assert resp.status_code in (400, 422)


def test_t2_f2_concurrent_predict_burst_thread_safe(live_server):
    base_url = live_server["base_url"]
    payloads = [
        {"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
        {"p1": "Nadal R.", "p2": "Federer R.", "surface": "Clay", "series": "Grand Slam"},
    ]

    def _call(idx: int) -> tuple[int, float, float]:
        payload = payloads[idx % 2]
        with httpx.Client(base_url=base_url, trust_env=False, timeout=15.0) as client:
            r = client.post("/api/v1/predict", json=payload)
            assert r.status_code == 200
            d = r.json()
            return idx % 2, d["prob_lgb"], d["prob_lr"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_call, range(16)))

    even_probs = {(lgb, lr) for kind, lgb, lr in results if kind == 0}
    odd_probs = {(lgb, lr) for kind, lgb, lr in results if kind == 1}
    assert len(even_probs) == 1
    assert len(odd_probs) == 1


# ===========================================================================
# F3 Boundaries: Edge Encodings, Finite 31 Features & Zero-SHAP (5 tests)
# ===========================================================================


def test_t2_f3_cold_start_vs_cold_start_31_finite_floats(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Rookie Alpha", "p2": "Rookie Beta", "surface": "Hard", "series": "ATP250"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert_valid_features_31(data["features_31"])
    for f in data["features_31"]:
        assert math.isfinite(f["value"])
        assert math.isfinite(f["shap"])
        assert math.isfinite(f["lr_coef"])


def test_t2_f3_zero_h2h_history_neutral_encoding(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Rookie Alpha", "p2": "Rookie Beta", "surface": "Hard", "series": "ATP250"},
    ).json()
    by_name = {f["name"]: f for f in data["features_31"]}
    assert by_name["h2h_match_count"]["value"] == 0.0
    assert by_name["h2h_winrate"]["value"] == pytest.approx(0.5, abs=1e-6)
    assert by_name["h2h_surface_count"]["value"] == 0.0
    assert by_name["h2h_surface_winrate"]["value"] == pytest.approx(0.5, abs=1e-6)


def test_t2_f3_carpet_masters_cup_encoding(http_client):
    carpet_mc = http_client.post(
        "/api/v1/predict",
        json={"p1": "Agassi A.", "p2": "Sampras P.", "surface": "Carpet", "series": "Masters Cup"},
    )
    assert carpet_mc.status_code == 200
    f_cmc = {f["name"]: f for f in carpet_mc.json()["features_31"]}

    hard_250 = http_client.post(
        "/api/v1/predict",
        json={"p1": "Agassi A.", "p2": "Sampras P.", "surface": "Hard", "series": "ATP250"},
    ).json()
    f_h250 = {f["name"]: f for f in hard_250["features_31"]}

    assert math.isfinite(f_cmc["surface_encoded"]["value"])
    assert math.isfinite(f_cmc["series_encoded"]["value"])
    assert f_cmc["surface_encoded"]["value"] != f_h250["surface_encoded"]["value"]
    assert f_cmc["series_encoded"]["value"] != f_h250["series_encoded"]["value"]


def test_t2_f3_is_grand_slam_flag_encoding(http_client):
    gs = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    a250 = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "ATP250"},
    ).json()
    gs_map = {f["name"]: f["value"] for f in gs["features_31"]}
    a250_map = {f["name"]: f["value"] for f in a250["features_31"]}
    assert gs_map["is_grand_slam"] == 1.0
    assert a250_map["is_grand_slam"] == 0.0


def test_t2_f3_zero_shap_favors_neutral(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Rookie Alpha", "p2": "Rookie Beta", "surface": "Hard", "series": "ATP250"},
    ).json()
    zero_items = [f for f in data["features_31"] if abs(f["shap"]) <= 1e-9]
    assert len(zero_items) > 0
    for item in zero_items:
        assert item["favors"] in ("neutral", "even")
    for c in data["treeshap"]["contributions"]:
        if abs(c["shap"]) <= 1e-9:
            assert c["favors"] in ("neutral", "even")


# ===========================================================================
# F4 Boundaries: Cold-Start Cards, Retired Legends & Extremes (5 tests)
# ===========================================================================


def test_t2_f4_cold_start_card_clamping(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Rookie Z.", "p2": "Alcaraz C.", "surface": "Hard", "series": "ATP250"},
    ).json()
    rookie = data["player_cards"]["p1"]
    elite = data["player_cards"]["p2"]
    assert_valid_player_card(rookie, expected_name="Rookie Z.")
    assert_valid_player_card(elite, expected_name="Alcaraz C.")
    assert 0 <= rookie["ovr"] <= 99
    assert rookie["rank"] >= 100
    assert rookie["pts"] == 0
    assert rookie["ovr"] < elite["ovr"]
    for axis in RADAR_AXES:
        assert 0.0 <= rookie["radar"][axis] <= 100.0


def test_t2_f4_legacy_retired_players_cards(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Agassi A.", "p2": "Sampras P.", "surface": "Hard", "series": "Grand Slam"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert_valid_player_card(data["player_cards"]["p1"], expected_name="Agassi A.")
    assert_valid_player_card(data["player_cards"]["p2"], expected_name="Sampras P.")


def test_t2_f4_zero_recent_14d_matches_valid_bounds(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Agassi A.", "p2": "Sampras P.", "surface": "Grass", "series": "Grand Slam"},
    ).json()
    for slot in ("p1", "p2"):
        card = data["player_cards"][slot]
        assert card["matches_14d"] >= 0
        assert card["days_since_last"] >= 0
        assert 0.0 <= card["radar"]["stamina_freshness"] <= 100.0


def test_t2_f4_extreme_rank_disparity(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Rookie Z.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    elite = data["player_cards"]["p1"]
    rookie = data["player_cards"]["p2"]
    assert elite["ovr"] > rookie["ovr"]
    assert elite["rank"] < rookie["rank"]
    assert elite["radar"]["rank_power"] > rookie["radar"]["rank_power"]


def test_t2_f4_card_determinism_across_slots(http_client):
    a = http_client.post(
        "/api/v1/predict",
        json={"p1": "Nadal R.", "p2": "Federer R.", "surface": "Clay", "series": "Grand Slam"},
    ).json()
    b = http_client.post(
        "/api/v1/predict",
        json={"p1": "Federer R.", "p2": "Nadal R.", "surface": "Clay", "series": "Grand Slam"},
    ).json()
    assert a["player_cards"]["p1"] == b["player_cards"]["p2"]
    assert a["player_cards"]["p2"] == b["player_cards"]["p1"]


# ===========================================================================
# F5 Boundaries: Odds Validation, Extreme Convergence & Kelly Caps (6 tests)
# ===========================================================================


def test_t2_f5_partial_odds_returns_null_edge(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.85,
            "odd_2": None,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["edge"] is None


def test_t2_f5_invalid_odds_lte_one_never_500(http_client):
    for bad_odd_1 in (1.0, 0.9, -2.0):
        resp = http_client.post(
            "/api/v1/predict",
            json={
                "p1": "Alcaraz C.",
                "p2": "Sinner J.",
                "surface": "Hard",
                "series": "Grand Slam",
                "odd_1": bad_odd_1,
                "odd_2": 2.0,
            },
        )
        assert resp.status_code in (200, 400, 422)
        if resp.status_code == 200:
            assert resp.json()["edge"] is None


def test_t2_f5_extreme_odds_solver_convergence(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.01,
            "odd_2": 100.0,
        },
    )
    assert resp.status_code == 200
    edge = resp.json()["edge"]
    assert edge is not None
    exp_p1, exp_p2, exp_overround, _ = solve_power_devig(1.01, 100.0)
    assert edge["implied_prob_p1"] == pytest.approx(exp_p1, abs=1e-3)
    assert edge["implied_prob_p2"] == pytest.approx(exp_p2, abs=1e-3)
    assert edge["implied_prob_p1"] + edge["implied_prob_p2"] == pytest.approx(1.0, abs=1e-5)
    assert edge["overround"] == pytest.approx(exp_overround, abs=1e-4)


def test_t2_f5_equal_even_odds_exact_half(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 2.0,
            "odd_2": 2.0,
        },
    )
    assert resp.status_code == 200
    edge = resp.json()["edge"]
    assert edge["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)
    assert edge["implied_prob_p2"] == pytest.approx(0.5, abs=1e-6)
    assert edge["overround"] == pytest.approx(1.0, abs=1e-6)


def test_t2_f5_huge_positive_edge_caps_kelly_at_2_5_pct(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 50.0,
            "odd_2": 1.02,
        },
    )
    assert resp.status_code == 200
    edge = resp.json()["edge"]
    assert edge["edge_p1"] > 0.4
    assert 0.0 < edge["kelly_p1_pct"] <= 2.5
    assert edge["kelly_p1_pct"] == pytest.approx(2.5, abs=1e-4)


def test_t2_f5_negative_edge_gives_zero_kelly(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.05,
            "odd_2": 10.0,
        },
    )
    assert resp.status_code == 200
    edge = resp.json()["edge"]
    assert edge["edge_p1"] < 0.0
    assert edge["kelly_p1_pct"] == 0.0


# ===========================================================================
# F6 Boundaries: Historical Lookup Errors, Missing Odds & Special Scores (5 tests)
# ===========================================================================


def test_t2_f6_nonexistent_match_idx_returns_404_or_400(http_client):
    resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": "Rio Open",
            "match_idx": "999999999",
        },
    )
    assert resp.status_code in (400, 404)


def test_t2_f6_non_numeric_and_none_match_idx_rejected(http_client):
    for bad_idx in ("abc", "None"):
        resp = http_client.get(
            "/api/v1/fixtures/historical/inspect",
            params={
                "split": "Holdout Test (2024–2026)",
                "tournament": "Rio Open",
                "match_idx": bad_idx,
            },
        )
        assert resp.status_code in (400, 404)


def test_t2_f6_unknown_tournament_returns_empty_matches(http_client):
    resp = http_client.get(
        "/api/v1/fixtures/historical",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": "Nonexistent Tournament XYZ",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["matches"] == []


def test_t2_f6_historical_row_with_missing_bm_prob(http_client):
    nan_rows = atp_data.test_df[atp_data.test_df["implied_prob_diff"].isna()]
    assert not nan_rows.empty, "Expected at least one holdout row with NaN implied_prob_diff"
    target_idx = str(nan_rows.index[0])
    target_tourn = str(nan_rows.iloc[0]["Tournament"])
    resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": target_tourn,
            "match_idx": target_idx,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bm_prob"] is None
    assert 0.0 <= data["prob_lgb"] <= 1.0


def test_t2_f6_bo5_and_retirement_score_string_handled_cleanly(http_client, monkeypatch):
    # Real Bo5 match discovered dynamically from Holdout Test
    split = "Holdout Test (2024–2026)"
    hist_base = http_client.get("/api/v1/fixtures/historical", params={"split": split})
    assert hist_base.status_code == 200
    tournaments = hist_base.json()["tournaments"]
    chosen_tourn = "Australian Open" if "Australian Open" in tournaments else tournaments[0]
    t_resp = http_client.get(
        "/api/v1/fixtures/historical",
        params={"split": split, "tournament": chosen_tourn},
    )
    assert t_resp.status_code == 200
    matches = t_resp.json()["matches"]
    assert len(matches) > 0
    resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": split,
            "tournament": chosen_tourn,
            "match_idx": matches[0]["match_idx"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["score_str"]

    # Isolated retirement/special score string row in Holdout DataFrame
    patched_df = atp_data.test_df.copy()
    patched_df.loc[patched_df.index[0], "Score"] = "6-4 2-1 Ret."
    monkeypatch.setattr(atp_data, "test_df", patched_df)

    client = create_inprocess_client(monkeypatch)
    ret_resp = client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": str(patched_df.loc[patched_df.index[0], "Tournament"]),
            "match_idx": str(patched_df.index[0]),
        },
    )
    assert ret_resp.status_code == 200
    assert ret_resp.json()["score_str"] == "6-4 2-1 Ret."


# ===========================================================================
# F7 Boundaries: Schedule Corruption, ESPN Failures, Filtering & Cold-Start (5 tests)
# ===========================================================================


def test_t2_f7_missing_data_dir_schedule_live_safe(tmp_path, monkeypatch):
    missing = tmp_path / "absent_data"
    monkeypatch.setattr(atp_data, "DATA_DIR", str(missing))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(missing / "sync_state.json"))
    monkeypatch.setattr(
        atp_data,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(missing / "daily_predictions_latest.json"),
    )
    client = create_inprocess_client(monkeypatch)
    resp = client.get("/api/v1/schedule/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fixtures"] == []
    assert data["kpis"]["total_fixtures"] == 0


def test_t2_f7_espn_http_500_and_timeout_soft_fallback(tmp_path, monkeypatch):
    _isolate_full_sync_paths(monkeypatch, tmp_path)

    def _timeout(*args, **kwargs):
        raise requests.Timeout("ESPN timed out")

    monkeypatch.setattr(atp_daily_sync.requests, "get", _timeout)
    client = create_inprocess_client(monkeypatch)
    r1 = client.post("/api/v1/schedule/refresh")
    assert r1.status_code == 200
    assert r1.json()["refreshed"] is False
    assert r1.json()["refresh_error"]

    class _Resp500:
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(atp_daily_sync.requests, "get", lambda *a, **kw: _Resp500())
    r2 = client.post("/api/v1/schedule/refresh")
    assert r2.status_code == 200
    assert r2.json()["refreshed"] is False
    assert r2.json()["refresh_error"]


def test_t2_f7_corrupted_daily_predictions_json_safe(tmp_path, monkeypatch):
    data_dir = _isolate_full_sync_paths(monkeypatch, tmp_path)
    (data_dir / "daily_predictions_latest.json").write_text("{corrupted json!!!", encoding="utf-8")
    client = create_inprocess_client(monkeypatch)
    resp = client.get("/api/v1/schedule/live")
    assert resp.status_code == 200
    assert resp.json()["fixtures"] == []


def test_t2_f7_espn_filters_doubles_and_tbd(monkeypatch):
    espn_payload = {
        "events": [
            {
                "name": "Wimbledon",
                "groupings": [
                    {
                        "grouping": {"displayName": "Men's Singles"},
                        "competitions": [
                            {
                                "id": "doubles-1",
                                "date": "2026-07-05T12:00Z",
                                "competitors": [
                                    {"athlete": {"displayName": "Player A / Player B"}},
                                    {"athlete": {"displayName": "Player C / Player D"}},
                                ],
                            },
                            {
                                "id": "tbd-1",
                                "date": "2026-07-05T13:00Z",
                                "competitors": [
                                    {"athlete": {"displayName": "Carlos Alcaraz"}},
                                    {"athlete": {"displayName": "TBD"}},
                                ],
                            },
                            {
                                "id": "singles-ok",
                                "date": "2026-07-05T14:00Z",
                                "competitors": [
                                    {"athlete": {"displayName": "Carlos Alcaraz"}},
                                    {"athlete": {"displayName": "Jannik Sinner"}},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return espn_payload

    monkeypatch.setattr(atp_daily_sync.requests, "get", lambda *a, **kw: _FakeResp())
    fixtures = atp_daily_sync.fetch_espn_atp_schedule(target_date="2026-07-05")
    assert len(fixtures) == 1
    assert fixtures[0]["fixture_id"] == "singles-ok"


def test_t2_f7_cold_start_espn_player_flagged(tmp_path, monkeypatch):
    _isolate_full_sync_paths(monkeypatch, tmp_path)
    espn_payload = {
        "events": [
            {
                "name": "Miami Open",
                "groupings": [
                    {
                        "grouping": {"displayName": "Men's Singles"},
                        "competitions": [
                            {
                                "id": "cold-1",
                                "date": "2026-11-21T16:00Z",
                                "competitors": [
                                    {"athlete": {"displayName": "Unknown Rookie Z."}},
                                    {"athlete": {"displayName": "Carlos Alcaraz"}},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return espn_payload

    monkeypatch.setattr(atp_daily_sync.requests, "get", lambda *a, **kw: _FakeResp())
    client = create_inprocess_client(monkeypatch)
    resp = client.post("/api/v1/schedule/refresh")
    assert resp.status_code == 200
    fx = resp.json()["fixtures"][0]
    assert fx["cold_start"] is True
    assert 0.0 <= fx["prob_lgb"] <= 1.0


# ===========================================================================
# F8 Boundaries: Airgapped Assets, UI Edge Cases & Mobile CDP (5 tests)
# ===========================================================================


def test_t2_f8_zero_external_urls_in_served_html_css_js(http_client):
    root = http_client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers.get("content-type", "").lower()
    html = root.text
    allowed_cdn_prefixes = (
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
    )
    raw_attr_urls = re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    for raw in raw_attr_urls:
        if raw.startswith(("http://", "https://", "//")):
            assert raw.startswith(allowed_cdn_prefixes), f"Disallowed external URL in HTML: {raw}"

    hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    local_assets = [
        raw if raw.startswith("/") else f"/{raw.lstrip('./')}"
        for raw in hrefs + srcs
        if not raw.startswith(("http://", "https://", "//", "data:"))
    ]
    assert len(local_assets) >= 4, f"Expected CSS/JS assets referenced by index.html, got {local_assets}"

    url_pattern = re.compile(r"https?://[^\s'\"<>)]+")
    allowed_prefixes = (
        "http://127.0.0.1",
        "http://localhost",
        "http://www.w3.org/2000/svg",
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
    )

    def _assert_no_external_http_urls(content: str, label: str) -> None:
        for found in url_pattern.findall(content):
            assert found.startswith(allowed_prefixes), (
                f"Disallowed external URL {found!r} found in {label}"
            )
        lowered = content.lower()
        for cdn_hint in ("unpkg.com", "jsdelivr.net", "cdnjs.", "cdn.tailwindcss.com"):
            assert cdn_hint not in lowered, f"External CDN {cdn_hint} found in {label}"

    _assert_no_external_http_urls(html, "/")
    for asset_url in local_assets:
        r = http_client.get(asset_url)
        assert r.status_code == 200, f"GET {asset_url} returned {r.status_code}"
        _assert_no_external_http_urls(r.text, asset_url)


def test_t2_f8_daily_schedule_board_and_filters_cdp(live_server):
    res = run_cdp_scenario(live_server["base_url"], "daily_schedule_filters_and_ticker")
    assert res["ok"] is True


def test_t2_f8_swap_and_31_matrix_cdp(live_server):
    res = run_cdp_scenario(live_server["base_url"], "h2h_odds_swap_matrix")
    assert res["ok"] is True
    assert res["data"]["featureRows"] >= 31
    b0 = res["data"]["beforePair"][0]
    a0 = res["data"]["afterPair"][0]
    assert abs((b0 + a0) - 100.0) <= 1.2


def test_t2_f8_invalid_odds_ui_no_nan_cdp(live_server):
    res = run_cdp_scenario(live_server["base_url"], "invalid_odds_ui")
    assert res["ok"] is True
    assert res["data"]["checked"] is True


def test_t2_f8_mobile_viewport_no_horizontal_overflow_cdp(live_server):
    res = run_cdp_scenario(
        live_server["base_url"],
        "mobile_viewport_layout",
        viewport=(375, 812),
    )
    assert res["ok"] is True
    assert res["data"]["scrollWidth"] <= res["data"]["innerWidth"] + 2
    assert res["data"]["bodyHeight"] >= 200


# ===========================================================================
# F9 Boundaries: Hydrogen Sync Edge Cases & Container Config (5 tests)
# ===========================================================================


def test_t2_f9_daily_sync_invalid_odds_no_crash(tmp_path, monkeypatch):
    _isolate_full_sync_paths(monkeypatch, tmp_path)
    fixtures = [
        {
            "fixture_id": "bad-odds-1",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Final",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": 1.0,
            "Odd_2": float("nan"),
        },
        {
            "fixture_id": "bad-odds-2",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Final",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": -1.0,
            "Odd_2": 2.0,
        },
    ]
    payload = atp_daily_sync.predict_published_schedule(
        fixtures=fixtures, source_label="file:test", save_outputs=True
    )
    assert len(payload["fixtures"]) == 2
    for fx in payload["fixtures"]:
        assert fx["bm_implied_p1"] is None
        assert fx["edge_lgb_p1"] is None


def test_t2_f9_unchanged_kaggle_sha256_skips_recompute(tmp_path, monkeypatch):
    _isolate_full_sync_paths(monkeypatch, tmp_path)
    raw_csv = tmp_path / "raw_kaggle.csv"
    _sample_kaggle_raw_with_pending().to_csv(raw_csv, index=False)

    try:
        r1 = atp_daily_sync.sync_kaggle_dataset(local_raw_path=str(raw_csv))
        assert r1["status"] == "synced"
        r2 = atp_daily_sync.sync_kaggle_dataset(local_raw_path=str(raw_csv))
        assert r2["status"] == "up_to_date"
        assert r2["new_matches"] == 0
    finally:
        monkeypatch.undo()
        atp_engine.reload_state()


def test_t2_f9_pending_kaggle_rows_routed_to_pending_csv(tmp_path, monkeypatch):
    data_dir = _isolate_full_sync_paths(monkeypatch, tmp_path)
    raw_csv = tmp_path / "raw_kaggle.csv"
    _sample_kaggle_raw_with_pending().to_csv(raw_csv, index=False)

    try:
        res = atp_daily_sync.sync_kaggle_dataset(local_raw_path=str(raw_csv))
        assert res["pending_kaggle_rows"] == 1
        pending_path = data_dir / "pending_kaggle_fixtures.csv"
        assert pending_path.is_file()
        pending_df = pd.read_csv(pending_path)
        assert len(pending_df) == 1
        assert pending_df.loc[0, "Tournament"] == "ATP Finals"
    finally:
        monkeypatch.undo()
        atp_engine.reload_state()


def test_t2_f9_dockerfile_honors_port_env_without_hardcoding():
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Missing Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    cmd_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*(CMD|ENTRYPOINT)\b", line)
    ]
    assert cmd_lines, "Dockerfile must define CMD or ENTRYPOINT"
    launch_stmt = " ".join(cmd_lines)
    assert re.search(r"\$\{?PORT\b", launch_stmt), (
        f"Dockerfile CMD/ENTRYPOINT must bind $PORT dynamically: {launch_stmt}"
    )
    assert not re.search(r"--port\s+8080\b", launch_stmt), (
        f"Dockerfile CMD/ENTRYPOINT must not hardcode --port 8080: {launch_stmt}"
    )
    assert "0.0.0.0" in launch_stmt


def test_t2_f9_workflow_cron_and_no_web_server():
    wf = REPO_ROOT / ".github" / "workflows" / "daily_atp_sync.yml"
    assert wf.is_file(), "Missing .github/workflows/daily_atp_sync.yml"
    text = wf.read_text(encoding="utf-8")
    assert re.search(r'cron:\s*["\']', text)
    assert "atp_daily_sync.py" in text
    assert "uvicorn server:app" not in text
