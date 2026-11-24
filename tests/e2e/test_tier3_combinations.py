"""Tier 3 pairwise cross-feature combination tests (opaque-box contracts)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import (
    assert_valid_features_31,
    assert_valid_player_card,
    assert_valid_treeshap,
    compute_quarter_kelly_pct,
    create_inprocess_client,
    run_cdp_scenario,
    sigmoid,
    solve_power_devig,
)

pytestmark = pytest.mark.tier3


def _isolate_schedule_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect schedule/sync disk paths to tmp_path so repo data/ is never mutated."""
    from pipeline import atp_daily_sync, atp_data

    history_dir = tmp_path / "predictions_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(atp_data, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(tmp_path / "sync_state.json"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(tmp_path / "daily_predictions_latest.json"))

    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(history_dir))
    monkeypatch.setattr(atp_daily_sync, "DEFAULT_SCHEDULE_CSV", str(tmp_path / "upcoming_schedule.csv"))
    monkeypatch.setattr(atp_daily_sync, "PENDING_KAGGLE_CSV", str(tmp_path / "pending_kaggle_fixtures.csv"))
    monkeypatch.setattr(atp_daily_sync, "PLAYER_STATE_CACHE_PATH", str(tmp_path / "player_state_cache.joblib"))
    monkeypatch.setattr(atp_daily_sync, "SYNC_STATE_PATH", str(tmp_path / "sync_state.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(tmp_path / "daily_predictions_latest.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(tmp_path / "daily_predictions_latest.csv"))
    return tmp_path


def test_f2_f5_swap_symmetry_with_odds(http_client: httpx.Client) -> None:
    """1. F2 x F5: Swapping P1<->P2 and odd_1<->odd_2 mirrors Power-Method edges & Quarter-Kelly."""
    fwd_resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.85,
            "odd_2": 2.05,
        },
    )
    rev_resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Sinner J.",
            "p2": "Alcaraz C.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 2.05,
            "odd_2": 1.85,
        },
    )
    assert fwd_resp.status_code == 200, fwd_resp.text
    assert rev_resp.status_code == 200, rev_resp.text

    fwd = fwd_resp.json()
    rev = rev_resp.json()

    assert fwd["prob_lgb"] + rev["prob_lgb"] == pytest.approx(1.0, abs=1e-5)
    assert fwd["prob_lr"] + rev["prob_lr"] == pytest.approx(1.0, abs=1e-5)

    fwd_edge = fwd["edge"]
    rev_edge = rev["edge"]
    assert isinstance(fwd_edge, dict) and isinstance(rev_edge, dict)

    assert fwd_edge["edge_p1"] == pytest.approx(rev_edge["edge_p2"], abs=1e-4)
    assert fwd_edge["edge_p2"] == pytest.approx(rev_edge["edge_p1"], abs=1e-4)
    assert fwd_edge["kelly_p1_pct"] == pytest.approx(rev_edge["kelly_p2_pct"], abs=1e-2)
    assert fwd_edge["kelly_p2_pct"] == pytest.approx(rev_edge["kelly_p1_pct"], abs=1e-2)

    imp1, imp2, overround, power_k = solve_power_devig(1.85, 2.05)
    assert fwd_edge["implied_prob_p1"] == pytest.approx(imp1, abs=1e-4)
    assert fwd_edge["implied_prob_p2"] == pytest.approx(imp2, abs=1e-4)
    assert fwd_edge["overround"] == pytest.approx(overround, abs=1e-4)
    assert fwd_edge["power_k"] == pytest.approx(power_k, abs=1e-3)
    assert fwd_edge["kelly_p1_pct"] == pytest.approx(compute_quarter_kelly_pct(fwd["prob_lgb"], 1.85), abs=1e-2)


def test_f2_f3_swap_flips_treeshap_and_preserves_sigmoid(http_client: httpx.Client) -> None:
    """2. F2 x F3: Swapping P1<->P2 flips anti-symmetric delta SHAP signs & preserves sigmoid."""
    fwd_resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    )
    rev_resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"},
    )
    assert fwd_resp.status_code == 200, fwd_resp.text
    assert rev_resp.status_code == 200, rev_resp.text
    fwd = fwd_resp.json()
    rev = rev_resp.json()

    for payload in (fwd, rev):
        assert_valid_treeshap(payload["treeshap"])
        assert_valid_features_31(payload["features_31"])
        sum_shap = sum(float(f["shap"]) for f in payload["features_31"])
        bias = float(payload["treeshap"]["bias"])
        raw_asym = float(payload["raw_asym"])
        assert 0.0 <= raw_asym < 0.35, f"Unexpected raw_asym={raw_asym}"
        assert payload["treeshap"]["p1_total_contrib"] == pytest.approx(sum_shap, abs=1e-3)
        reconstructed_prob = sigmoid(bias + sum_shap)
        assert abs(reconstructed_prob - float(payload["prob_lgb"])) <= min(0.35, raw_asym + 1e-4)

    fwd_by_name = {f["name"]: f for f in fwd["features_31"]}
    rev_by_name = {f["name"]: f for f in rev["features_31"]}

    for feat_name in ("rank_diff", "pts_diff", "surface_winrate_diff", "short_winrate_diff"):
        assert feat_name in fwd_by_name and feat_name in rev_by_name
        assert fwd_by_name[feat_name]["value"] == pytest.approx(-rev_by_name[feat_name]["value"], abs=1e-5)
        assert fwd_by_name[feat_name]["shap"] == pytest.approx(-rev_by_name[feat_name]["shap"], abs=1e-3)


def test_f2_f4_swap_symmetry_arcade_cards_and_key_stats(http_client: httpx.Client) -> None:
    """3. F2 x F4: Swapping P1<->P2 swaps player_cards identically & negates antisymmetric key_stats."""
    fwd_resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    )
    rev_resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"},
    )
    assert fwd_resp.status_code == 200, fwd_resp.text
    assert rev_resp.status_code == 200, rev_resp.text
    fwd = fwd_resp.json()
    rev = rev_resp.json()

    assert_valid_player_card(fwd["player_cards"]["p1"], expected_name="Alcaraz C.")
    assert_valid_player_card(fwd["player_cards"]["p2"], expected_name="Sinner J.")
    assert_valid_player_card(rev["player_cards"]["p1"], expected_name="Sinner J.")
    assert_valid_player_card(rev["player_cards"]["p2"], expected_name="Alcaraz C.")

    for a_key, b_key in (("p1", "p2"), ("p2", "p1")):
        card_a = fwd["player_cards"][a_key]
        card_b = rev["player_cards"][b_key]
        assert card_a["name"] == card_b["name"]
        assert card_a["ovr"] == card_b["ovr"]
        assert card_a["traits"] == card_b["traits"]
        assert card_a["radar"] == card_b["radar"]

    fwd_ks = fwd["key_stats"]
    rev_ks = rev["key_stats"]
    for stat_key in ("rank_diff", "surface_winrate_diff", "form_divergence_diff", "matches_14d_diff"):
        assert fwd_ks[stat_key] == pytest.approx(-rev_ks[stat_key], abs=1e-5)

    assert fwd_ks["h2h_p1_wins"] == rev_ks["h2h_p2_wins"]
    assert fwd_ks["h2h_p2_wins"] == rev_ks["h2h_p1_wins"]


def test_f3_f6_historical_backtracker_holdout_inspect(http_client: httpx.Client) -> None:
    """4. F3 x F6: Holdout historical inspect (including upset) returns TreeSHAP, features_31, bm_prob & winner."""
    split = "Holdout Test (2024–2026)"
    list_resp = http_client.get("/api/v1/fixtures/historical", params={"split": split})
    assert list_resp.status_code == 200, list_resp.text
    tournaments = list_resp.json()["tournaments"]
    assert len(tournaments) > 0

    upset_payload: dict[str, Any] | None = None
    verified_payload: dict[str, Any] | None = None

    for tourn in tournaments[:10]:
        t_resp = http_client.get(
            "/api/v1/fixtures/historical",
            params={"split": split, "tournament": tourn},
        )
        assert t_resp.status_code == 200, t_resp.text
        matches = t_resp.json()["matches"]
        for m in matches[:20]:
            insp_resp = http_client.get(
                "/api/v1/fixtures/historical/inspect",
                params={"split": split, "tournament": tourn, "match_idx": m["match_idx"]},
            )
            assert insp_resp.status_code == 200, insp_resp.text
            payload = insp_resp.json()
            if verified_payload is None and payload.get("bm_prob") is not None:
                verified_payload = payload
            if payload.get("prediction_accurate") is False and payload.get("bm_prob") is not None:
                upset_payload = payload
                break
        if upset_payload is not None:
            break

    target = upset_payload or verified_payload
    assert target is not None, "Expected at least one inspectable holdout match with bookmaker probability"
    assert upset_payload is not None, "Expected to find at least one holdout upset match (prediction_accurate is False)"

    assert target["is_historical"] is True
    assert_valid_treeshap(target["treeshap"])
    assert_valid_features_31(target["features_31"])
    sum_shap = sum(float(f["shap"]) for f in target["features_31"])
    assert target["treeshap"]["p1_total_contrib"] == pytest.approx(sum_shap, abs=1e-3)
    assert sigmoid(float(target["treeshap"]["bias"]) + sum_shap) == pytest.approx(float(target["prob_lgb"]), abs=1e-4)
    assert isinstance(target["bm_prob"], float) and 0.0 < target["bm_prob"] < 1.0
    assert target["actual_winner"] in (target["p1"], target["p2"])
    assert target["prediction_accurate"] is False


def test_f4_f6_historical_legacy_full_dataset_inspect(http_client: httpx.Client) -> None:
    """5. F4 x F6: Legacy Full Dataset (2000-2026) inspect returns Arcade cards, 6-axis radar & score/date."""
    split = "Full Dataset (2000–2026)"
    list_resp = http_client.get("/api/v1/fixtures/historical", params={"split": split})
    assert list_resp.status_code == 200, list_resp.text
    list_data = list_resp.json()
    assert len(list_data["tournaments"]) > 0
    assert len(list_data["matches"]) > 0

    tourn = list_data["selected_tournament"]
    match_idx = list_data["matches"][0]["match_idx"]
    insp_resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={"split": split, "tournament": tourn, "match_idx": match_idx},
    )
    assert insp_resp.status_code == 200, insp_resp.text
    data = insp_resp.json()

    assert data["is_historical"] is True
    assert_valid_player_card(data["player_cards"]["p1"], expected_name=data["p1"])
    assert_valid_player_card(data["player_cards"]["p2"], expected_name=data["p2"])
    assert isinstance(data["score_str"], str) and data["score_str"].strip()
    assert isinstance(data["date_str"], str) and len(data["date_str"]) >= 10


def test_f5_f7_populated_schedule_power_devig_and_value_edge_kpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6. F5 x F7: Populated schedule computes Power-Method bm_prob, edge_lgb_p1, kelly_p1_pct & value_edges_count."""
    _isolate_schedule_storage(tmp_path, monkeypatch)
    from pipeline import atp_daily_sync

    fixtures = [
        {
            "fixture_id": "fix_value_1",
            "Date": "2026-11-21",
            "Tournament": "US Open",
            "Surface": "Hard",
            "Series": "Grand Slam",
            "Round": "Final",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": 1.50,
            "Odd_2": 2.75,
            "Status": "LineupsConfirmed",
            "Status_Detail": "Scheduled",
        },
        {
            "fixture_id": "fix_no_odds",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Round Robin",
            "Player_1": "Zverev A.",
            "Player_2": "Medvedev D.",
            "Odd_1": None,
            "Odd_2": None,
            "Status": "LineupsConfirmed",
            "Status_Detail": "Scheduled",
        },
    ]
    atp_daily_sync.predict_published_schedule(fixtures=fixtures, source_label="test_fixture", save_outputs=True)

    with create_inprocess_client(monkeypatch) as client:
        resp = client.get("/api/v1/schedule/live")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["kpis"]["total_fixtures"] == 2
        fix1 = next(f for f in data["fixtures"] if f["fixture_id"] == "fix_value_1")

        imp1, _, _, _ = solve_power_devig(1.50, 2.75)
        assert fix1["bm_prob"] == pytest.approx(imp1, abs=1e-4)
        assert fix1["edge_lgb_p1"] == pytest.approx(fix1["prob_lgb"] - imp1, abs=1e-4)
        assert fix1["kelly_p1_pct"] == pytest.approx(compute_quarter_kelly_pct(fix1["prob_lgb"], 1.50), abs=1e-2)

        expected_value_count = sum(
            1 for f in data["fixtures"]
            if f["edge_lgb_p1"] is not None and abs(f["edge_lgb_p1"]) >= 0.03
        )
        assert data["kpis"]["value_edges_count"] == expected_value_count


def test_f2_f7_schedule_divergent_fixture_kpi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7. F2 x F7: Divergent fixture in daily schedule sets is_divergent=True & increments divergent_count."""
    _isolate_schedule_storage(tmp_path, monkeypatch)
    from pipeline import atp_daily_sync

    with create_inprocess_client(monkeypatch) as client:
        pred_resp = client.post(
            "/api/v1/predict",
            json={"p1": "Sinner J.", "p2": "Djokovic N.", "surface": "Hard", "series": "Grand Slam"},
        )
        assert pred_resp.status_code == 200, pred_resp.text
        pred = pred_resp.json()
        assert pred["is_divergent"] is True and pred["div_delta"] > 0.15

        fixtures = [
            {
                "fixture_id": "fix_div_sinner_djoko",
                "Date": "2026-11-21",
                "Tournament": "Australian Open",
                "Surface": "Hard",
                "Series": "Grand Slam",
                "Round": "Semifinals",
                "Player_1": "Sinner J.",
                "Player_2": "Djokovic N.",
                "Status": "LineupsConfirmed",
            },
            {
                "fixture_id": "fix_nondiv_alc_sinner",
                "Date": "2026-11-21",
                "Tournament": "US Open",
                "Surface": "Hard",
                "Series": "Grand Slam",
                "Round": "Final",
                "Player_1": "Alcaraz C.",
                "Player_2": "Sinner J.",
                "Status": "LineupsConfirmed",
            },
        ]
        atp_daily_sync.predict_published_schedule(fixtures=fixtures, source_label="test_fixture", save_outputs=True)

        resp = client.get("/api/v1/schedule/live")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        div_fix = next(f for f in data["fixtures"] if f["fixture_id"] == "fix_div_sinner_djoko")
        assert div_fix["is_divergent"] is True
        assert div_fix["div_delta"] > 0.15
        expected_div_count = sum(1 for f in data["fixtures"] if f["is_divergent"])
        assert data["kpis"]["divergent_count"] == expected_div_count
        assert data["kpis"]["divergent_count"] >= 1


def test_f1_f7_schedule_refresh_updates_healthz_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """8. F1 x F7: POST /api/v1/schedule/refresh updates schedule_fixtures_count in GET /healthz."""
    _isolate_schedule_storage(tmp_path, monkeypatch)
    from pipeline import atp_daily_sync

    class _FakeEspnResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "events": [
                    {
                        "name": "US Open",
                        "venue": {"surface": "Hard"},
                        "season": {"displayName": "Grand Slam"},
                        "groupings": [
                            {
                                "grouping": {"displayName": "Men's Singles"},
                                "competitions": [
                                    {
                                        "id": "espn_101",
                                        "date": "2026-11-21T18:00Z",
                                        "round": {"displayName": "Final"},
                                        "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                                        "competitors": [
                                            {"athlete": {"displayName": "Carlos Alcaraz"}},
                                            {"athlete": {"displayName": "Jannik Sinner"}},
                                        ],
                                    },
                                    {
                                        "id": "espn_102",
                                        "date": "2026-11-21T20:00Z",
                                        "round": {"displayName": "Semifinal"},
                                        "status": {"type": {"name": "STATUS_IN_PROGRESS", "description": "Set 2"}},
                                        "competitors": [
                                            {"athlete": {"displayName": "Alexander Zverev"}},
                                            {"athlete": {"displayName": "Daniil Medvedev"}},
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }

    monkeypatch.setattr(atp_daily_sync.requests, "get", lambda *args, **kwargs: _FakeEspnResp())

    with create_inprocess_client(monkeypatch) as client:
        initial_health_resp = client.get("/healthz")
        assert initial_health_resp.status_code == 200, initial_health_resp.text
        assert initial_health_resp.json()["schedule_fixtures_count"] == 0

        refresh_resp = client.post("/api/v1/schedule/refresh")
        assert refresh_resp.status_code == 200, refresh_resp.text
        refresh_data = refresh_resp.json()
        assert refresh_data["refreshed"] is True
        assert refresh_data["refresh_error"] is None
        assert refresh_data["kpis"]["total_fixtures"] == 2
        assert refresh_data["kpis"]["live_count"] == 1

        updated_health_resp = client.get("/healthz")
        assert updated_health_resp.status_code == 200, updated_health_resp.text
        assert updated_health_resp.json()["schedule_fixtures_count"] == 2


def test_f1_f2_meta_defaults_feed_predict(http_client: httpx.Client) -> None:
    """9. F1 x F2: GET /api/v1/meta defaults feed directly into POST /api/v1/predict."""
    meta_resp = http_client.get("/api/v1/meta")
    assert meta_resp.status_code == 200, meta_resp.text
    meta = meta_resp.json()

    pred_resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": meta["default_p1"],
            "p2": meta["default_p2"],
            "surface": meta["surfaces"][0],
            "series": meta["series"][0],
        },
    )
    assert pred_resp.status_code == 200, pred_resp.text
    pred = pred_resp.json()

    assert 0.0 < pred["prob_lgb"] < 1.0
    assert 0.0 < pred["prob_lr"] < 1.0
    assert_valid_player_card(pred["player_cards"]["p1"], expected_name=meta["default_p1"])
    assert_valid_player_card(pred["player_cards"]["p2"], expected_name=meta["default_p2"])
    assert_valid_treeshap(pred["treeshap"])
    assert_valid_features_31(pred["features_31"])


def test_f5_f9_daily_sync_power_method_devig_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10. F5 x F9: atp_daily_sync Power-Method de-vig matches POST /api/v1/predict (< 1e-4)."""
    _isolate_schedule_storage(tmp_path, monkeypatch)
    from pipeline import atp_daily_sync

    with create_inprocess_client(monkeypatch) as client:
        pred_resp = client.post(
            "/api/v1/predict",
            json={
                "p1": "Alcaraz C.",
                "p2": "Sinner J.",
                "surface": "Hard",
                "series": "Grand Slam",
                "odd_1": 1.85,
                "odd_2": 2.05,
            },
        )
        assert pred_resp.status_code == 200, pred_resp.text
        pred = pred_resp.json()

    sync_payload = atp_daily_sync.predict_published_schedule(
        fixtures=[
            {
                "fixture_id": "parity_1",
                "Date": "2026-11-21",
                "Tournament": "US Open",
                "Surface": "Hard",
                "Series": "Grand Slam",
                "Round": "Final",
                "Player_1": "Alcaraz C.",
                "Player_2": "Sinner J.",
                "Odd_1": 1.85,
                "Odd_2": 2.05,
                "Status": "LineupsConfirmed",
            }
        ],
        source_label="parity_test",
        save_outputs=False,
    )
    assert len(sync_payload["fixtures"]) == 1
    sync_fix = sync_payload["fixtures"][0]

    assert sync_fix["bm_implied_p1"] == pytest.approx(pred["edge"]["implied_prob_p1"], abs=1e-4)
    assert sync_fix["bm_implied_p2"] == pytest.approx(pred["edge"]["implied_prob_p2"], abs=1e-4)
    assert sync_fix["overround"] == pytest.approx(pred["edge"]["overround"], abs=1e-4)
    assert sync_fix["power_k"] == pytest.approx(pred["edge"]["power_k"], abs=1e-3)
    assert sync_fix["edge_lgb_p1"] == pytest.approx(sync_fix["prob_lgb_p1"] - sync_fix["bm_implied_p1"], abs=1e-4)
    assert sync_fix["kelly_p1_pct"] == pytest.approx(compute_quarter_kelly_pct(sync_fix["prob_lgb_p1"], 1.85), abs=1e-2)


def test_f2_service_and_http_concurrent_parity(live_server: dict[str, Any]) -> None:
    """11. F2 x F2-Service: atp_service.predict & POST /api/v1/predict match under concurrent calls."""
    from src import atp_service

    matchups = [
        ("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam"),
        ("Sinner J.", "Alcaraz C.", "Hard", "Grand Slam"),
        ("Sinner J.", "Djokovic N.", "Hard", "Grand Slam"),
        ("Zverev A.", "Medvedev D.", "Clay", "Masters 1000"),
    ] * 2

    base_url = live_server["base_url"]

    def _run_pair(args: tuple[str, str, str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        p1, p2, surf, ser = args
        svc_res = atp_service.predict(p1, p2, surf, ser)
        with httpx.Client(base_url=base_url, trust_env=False, timeout=20.0) as cli:
            http_resp = cli.post(
                "/api/v1/predict",
                json={"p1": p1, "p2": p2, "surface": surf, "series": ser},
            )
            assert http_resp.status_code == 200, http_resp.text
            return svc_res, http_resp.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_run_pair, matchups))

    for svc_res, http_res in results:
        assert "error" not in svc_res
        assert svc_res["prob_lgb"] == pytest.approx(http_res["prob_lgb"], abs=1e-6)
        assert svc_res["prob_lr"] == pytest.approx(http_res["prob_lr"], abs=1e-6)
        assert svc_res["sym_score"] == pytest.approx(http_res["sym_score"], abs=1e-6)
        assert svc_res["div_delta"] == pytest.approx(http_res["div_delta"], abs=1e-6)


def test_f2_f8_ui_divergence_and_legacy_chromium(live_server: dict[str, Any]) -> None:
    """12. F2 x F8: Chromium verifies All-Time roster, Federer vs Nadal Clay H2H & Sinner vs Djokovic divergence."""
    res = run_cdp_scenario(live_server["base_url"], "alltime_legacy_and_divergence")
    assert res["ok"] is True
    assert res["errors"] == []
    assert res["data"]["expandedCount"] > 1000
    assert res["data"]["divergenceFound"] is True
