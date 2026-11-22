"""Tier 1 Opaque-Box Feature Tests (F1–F9) for MatchPoint.intelligence.

All tests are derived strictly from PROJECT.md, ORIGINAL_REQUEST.md, and public
HTTP/Python interface contracts. No application internals are mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pytest
import requests

from pipeline import atp_daily_sync, atp_data
import atp_engine
import atp_service

from tests.e2e.conftest import (
    RADAR_AXES,
    REPO_ROOT,
    assert_valid_features_31,
    assert_valid_player_card,
    assert_valid_treeshap,
    compute_quarter_kelly_pct,
    create_inprocess_client,
    run_cdp_scenario,
    sigmoid,
    solve_power_devig,
)

pytestmark = pytest.mark.tier1


def _extract_local_asset_urls(html: str) -> list[str]:
    """Extract local stylesheet hrefs and script srcs from HTML."""
    hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    urls: list[str] = []
    for raw in hrefs + srcs:
        if raw.startswith(("http://", "https://", "//", "data:")):
            continue
        urls.append(raw if raw.startswith("/") else f"/{raw.lstrip('./')}")
    return urls


def _find_css_asset_url(html: str) -> str:
    for url in _extract_local_asset_urls(html):
        if "styles.css" in url:
            return url
    return "/static/styles.css"


def _isolate_schedule_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    data_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(atp_data, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(
        atp_data,
        "DAILY_PREDICTIONS_JSON_PATH",
        str(data_dir / "daily_predictions_latest.json"),
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


# ===========================================================================
# F1: Health, Meta & Static Serving (5 tests)
# ===========================================================================


def test_f1_healthz_status_and_models(http_client):
    resp = http_client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert isinstance(body["schedule_fixtures_count"], int)
    assert body["schedule_fixtures_count"] >= 0
    sync_status = body.get("sync_status")
    assert isinstance(sync_status, dict)
    assert isinstance(sync_status.get("status"), str) and sync_status["status"].strip()


def test_f1_healthz_player_counts(http_client):
    resp = http_client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["players_total"] >= body["players_active"] > 0
    assert body["players_total"] >= 1823
    assert body["players_active"] >= 488


def test_f1_root_serves_neon_espn_html(http_client):
    resp = http_client.get("/")
    assert resp.status_code == 200
    ctype = resp.headers.get("content-type", "").lower()
    assert "text/html" in ctype, f"Expected text/html from GET /, got {ctype!r}"
    html = resp.text
    assert "<html" in html.lower()
    assert "matchpoint" in html.lower()


def test_f1_static_assets_served_non_empty(http_client):
    root_resp = http_client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "").lower()
    asset_urls = _extract_local_asset_urls(root_resp.text)

    required_assets = ("styles.css", "app.js", "charts.js", "arcade_cards.js")
    for asset_name in required_assets:
        matched = [u for u in asset_urls if asset_name in u]
        assert matched, f"Expected index.html to reference {asset_name}, found {asset_urls}"
        for url in matched:
            asset_resp = http_client.get(url)
            assert asset_resp.status_code == 200, f"GET {url} returned {asset_resp.status_code}"
            assert len(asset_resp.text.strip()) > 0, f"Asset {url} returned empty body"


def test_f1_meta_endpoint_contract(http_client):
    resp = http_client.get("/api/v1/meta")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["surfaces"]) == 4
    assert set(data["surfaces"]) == {"Hard", "Clay", "Grass", "Carpet"}
    assert len(data["series"]) == 8
    assert len(data["splits"]) == 2
    assert "Holdout Test (2024–2026)" in data["splits"]
    assert "Full Dataset (2000–2026)" in data["splits"]
    assert data["default_p1"] == "Alcaraz C."
    assert data["default_p2"] == "Sinner J."
    assert len(data["active_players"]) >= 488
    assert len(data["all_players"]) >= len(data["active_players"])
    metrics = data["model_metrics"]
    assert metrics["trees"] == 184
    assert metrics["features_count"] == 31
    assert 0.4 < metrics["lgbm_logloss"] < 0.75
    assert 0.65 < metrics["lgbm_auc"] < 0.85
    assert 0.15 < metrics["lgbm_brier"] < 0.28


# ===========================================================================
# F2: Dual-Model Symmetrized Inference, H2H & Backward Compat (6 tests)
# ===========================================================================


def test_f2_predict_alcaraz_sinner_probabilities(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["p1"] == "Alcaraz C."
    assert data["p2"] == "Sinner J."
    assert 0.0 <= data["prob_lgb"] <= 1.0
    assert 0.0 <= data["prob_lr"] <= 1.0
    assert data["p1_known"] is True
    assert data["p2_known"] is True


def test_f2_predict_swap_invariance(http_client):
    fwd = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    inv = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert abs((fwd["prob_lgb"] + inv["prob_lgb"]) - 1.0) < 1e-5
    assert abs((fwd["prob_lr"] + inv["prob_lr"]) - 1.0) < 1e-5


def test_f2_sym_score_and_raw_asymmetry(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert data["sym_score"] < 1e-5
    assert 0.0 <= data["raw_asym"] < 0.35


def test_f2_divergence_delta_and_threshold(http_client):
    non_div = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert non_div["div_delta"] == pytest.approx(abs(non_div["prob_lgb"] - non_div["prob_lr"]), abs=1e-6)
    assert non_div["is_divergent"] is (non_div["div_delta"] > 0.15)
    assert non_div["is_divergent"] is False

    div = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Djokovic N.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert div["div_delta"] == pytest.approx(abs(div["prob_lgb"] - div["prob_lr"]), abs=1e-6)
    assert div["is_divergent"] is (div["div_delta"] > 0.15)
    assert div["is_divergent"] is True


def test_f2_h2h_nadal_federer_and_antisymmetric_stats(http_client):
    fwd = http_client.post(
        "/api/v1/predict",
        json={"p1": "Nadal R.", "p2": "Federer R.", "surface": "Clay", "series": "Grand Slam"},
    ).json()
    ks = fwd["key_stats"]
    assert ks["h2h_p1_wins"] + ks["h2h_p2_wins"] == 40
    assert ks["h2h_p1_wins"] == 24
    assert ks["h2h_p2_wins"] == 16

    inv = http_client.post(
        "/api/v1/predict",
        json={"p1": "Federer R.", "p2": "Nadal R.", "surface": "Clay", "series": "Grand Slam"},
    ).json()
    ks_inv = inv["key_stats"]
    assert ks_inv["h2h_p1_wins"] == 16
    assert ks_inv["h2h_p2_wins"] == 24
    assert ks_inv["rank_diff"] == pytest.approx(-ks["rank_diff"], abs=1e-6)
    assert ks_inv["surface_winrate_diff"] == pytest.approx(-ks["surface_winrate_diff"], abs=1e-6)
    assert ks_inv["form_divergence_diff"] == pytest.approx(-ks["form_divergence_diff"], abs=1e-6)
    assert ks_inv["matches_14d_diff"] == pytest.approx(-ks["matches_14d_diff"], abs=1e-6)


def test_f2_legacy_python_modules_match_api(http_client):
    api_res = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    svc_res = atp_service.predict("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam")
    assert svc_res["prob_lgb"] == pytest.approx(api_res["prob_lgb"], abs=1e-6)
    assert svc_res["prob_lr"] == pytest.approx(api_res["prob_lr"], abs=1e-6)
    assert svc_res["sym_score"] < 1e-5
    assert svc_res["is_historical"] is False

    html = atp_engine.predict_matchup("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam")
    assert isinstance(html, str) and "<div" in html.lower()
    assert "Alcaraz C." in html
    assert "Sinner J." in html


# ===========================================================================
# F3: LightGBM TreeSHAP & 31-Feature Payload (5 tests)
# ===========================================================================


def test_f3_features_31_length_and_no_odds_leakage(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert_valid_features_31(data["features_31"])
    names = {f["name"] for f in data["features_31"]}
    assert "odds_diff" not in names
    assert "implied_prob_diff" not in names


def test_f3_treeshap_schema(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert_valid_treeshap(data["treeshap"])


def test_f3_treeshap_sum_and_sigmoid_consistency(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Djokovic N.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    ts = data["treeshap"]
    sum_shap = sum(f["shap"] for f in data["features_31"])
    assert sum_shap == pytest.approx(ts["p1_total_contrib"], abs=1e-4)
    reconstructed_prob = sigmoid(ts["bias"] + sum_shap)
    assert abs(reconstructed_prob - data["prob_lgb"]) <= min(0.35, data["raw_asym"] + 1e-4)


def test_f3_contribution_favors_sign_alignment(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Nadal R.", "p2": "Federer R.", "surface": "Clay", "series": "Grand Slam"},
    ).json()
    for item in data["treeshap"]["contributions"] + data["features_31"]:
        shap = item["shap"]
        if shap > 1e-9:
            assert item["favors"] == "p1"
        elif shap < -1e-9:
            assert item["favors"] == "p2"
        else:
            assert item["favors"] in ("neutral", "even")


def test_f3_features_31_finite_values_and_categories(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert_valid_features_31(data["features_31"])
    categories = {f["category"] for f in data["features_31"]}
    assert {"Ranking", "Head-to-Head", "Match Context"}.issubset(categories)


# ===========================================================================
# F4: Arcade Player Profile Engine (5 tests)
# ===========================================================================


def test_f4_player_cards_valid_for_elite_matchup(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    cards = data["player_cards"]
    assert_valid_player_card(cards["p1"], expected_name="Alcaraz C.")
    assert_valid_player_card(cards["p2"], expected_name="Sinner J.")
    assert len(cards["p1"]["traits"]) >= 1
    assert len(cards["p2"]["traits"]) >= 1
    assert cards["p1"]["radar"] != cards["p2"]["radar"]


def test_f4_ovr_bounds_and_elite_beats_rookie(http_client):
    elite_vs_rookie = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Unknown Rookie Z.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    p1_card = elite_vs_rookie["player_cards"]["p1"]
    p2_card = elite_vs_rookie["player_cards"]["p2"]
    assert 0 <= p1_card["ovr"] <= 99
    assert 0 <= p2_card["ovr"] <= 99
    assert p1_card["ovr"] > p2_card["ovr"]
    assert p1_card["radar"]["rank_power"] > p2_card["radar"]["rank_power"]
    assert p1_card["radar"] != p2_card["radar"]


def test_f4_radar_has_all_six_axes_in_bounds(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Clay", "series": "Masters 1000"},
    ).json()
    for slot in ("p1", "p2"):
        radar = data["player_cards"][slot]["radar"]
        assert set(RADAR_AXES).issubset(radar.keys())
        for axis in RADAR_AXES:
            assert 0.0 <= radar[axis] <= 100.0
    assert data["player_cards"]["p1"]["radar"] != data["player_cards"]["p2"]["radar"]


def test_f4_traits_non_empty_uppercase_readable(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    for slot in ("p1", "p2"):
        traits = data["player_cards"][slot]["traits"]
        assert isinstance(traits, list) and len(traits) >= 1
        for trait in traits:
            assert isinstance(trait, str) and len(trait.strip()) >= 2
            assert trait == trait.upper()


def test_f4_cards_deterministic_and_slot_invariant(http_client):
    r1 = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    r2 = http_client.post(
        "/api/v1/predict",
        json={"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"},
    ).json()
    assert r1["player_cards"]["p1"] == r2["player_cards"]["p2"]
    assert r1["player_cards"]["p2"] == r2["player_cards"]["p1"]


# ===========================================================================
# F5: Power-Method De-Vig, Edge & Quarter-Kelly (6 tests)
# ===========================================================================


def test_f5_symmetric_odds_devig(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.90,
            "odd_2": 1.90,
        },
    ).json()
    edge = data["edge"]
    assert edge is not None
    assert edge["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)
    assert edge["implied_prob_p2"] == pytest.approx(0.5, abs=1e-6)
    assert edge["overround"] == pytest.approx(2.0 / 1.90, abs=1e-6)


def test_f5_asymmetric_odds_power_devig(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.50,
            "odd_2": 2.75,
        },
    ).json()
    edge = data["edge"]
    exp_p1, exp_p2, exp_overround, exp_k = solve_power_devig(1.50, 2.75)
    assert edge["implied_prob_p1"] == pytest.approx(exp_p1, abs=1e-5)
    assert edge["implied_prob_p2"] == pytest.approx(exp_p2, abs=1e-5)
    assert edge["overround"] == pytest.approx(exp_overround, abs=1e-5)
    assert edge["power_k"] == pytest.approx(exp_k, abs=1e-4)
    assert edge["implied_prob_p1"] + edge["implied_prob_p2"] == pytest.approx(1.0, abs=1e-6)
    k = edge["power_k"]
    assert ((1.0 / 1.50) ** (1.0 / k)) + ((1.0 / 2.75) ** (1.0 / k)) == pytest.approx(1.0, abs=1e-5)


def test_f5_edge_p1_p2_antisymmetry(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.85,
            "odd_2": 2.05,
        },
    ).json()
    edge = data["edge"]
    assert edge["edge_p1"] == pytest.approx(data["prob_lgb"] - edge["implied_prob_p1"], abs=1e-5)
    assert edge["edge_p2"] == pytest.approx(-edge["edge_p1"], abs=1e-5)


def test_f5_quarter_kelly_matches_oracle(http_client):
    data = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Sinner J.",
            "p2": "Djokovic N.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.85,
            "odd_2": 2.15,
        },
    ).json()
    edge = data["edge"]
    exp_k1 = compute_quarter_kelly_pct(data["prob_lgb"], 1.85)
    exp_k2 = compute_quarter_kelly_pct(1.0 - data["prob_lgb"], 2.15)
    assert edge["kelly_p1_pct"] == pytest.approx(exp_k1, abs=0.02)
    assert edge["kelly_p2_pct"] == pytest.approx(exp_k2, abs=0.02)


def test_f5_omitted_odds_returns_null_edge(http_client):
    resp = http_client.post(
        "/api/v1/predict",
        json={"p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"},
    )
    assert resp.status_code == 200
    assert resp.json()["edge"] is None


def test_f5_edge_signal_positive_vs_negative_value(http_client):
    pos = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 3.00,
            "odd_2": 1.40,
        },
    ).json()
    assert pos["edge"]["edge_p1"] >= 0.03
    assert pos["edge"]["signal"] == "VALUE_P1"

    neg = http_client.post(
        "/api/v1/predict",
        json={
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "odd_1": 1.20,
            "odd_2": 4.50,
        },
    ).json()
    assert neg["edge"]["edge_p1"] <= -0.03
    assert neg["edge"]["signal"] == "VALUE_P2"


# ===========================================================================
# F6: Historical Match Backtracker 2000–2026 (5 tests)
# ===========================================================================


def test_f6_historical_defaults_holdout_split(http_client):
    resp = http_client.get("/api/v1/fixtures/historical")
    assert resp.status_code == 200
    data = resp.json()
    assert "Holdout" in data["split"]
    assert isinstance(data["tournaments"], list) and len(data["tournaments"]) > 0
    assert isinstance(data["matches"], list) and len(data["matches"]) > 0


def test_f6_full_dataset_has_more_tournaments_than_holdout(http_client):
    holdout = http_client.get(
        "/api/v1/fixtures/historical",
        params={"split": "Holdout Test (2024–2026)"},
    ).json()
    full = http_client.get(
        "/api/v1/fixtures/historical",
        params={"split": "Full Dataset (2000–2026)"},
    ).json()
    assert len(full["tournaments"]) > 200
    assert len(full["tournaments"]) > len(holdout["tournaments"])


def _discover_holdout_match(http_client, preferred_tournament: str = "Rio Open") -> tuple[str, str]:
    """Discover a real holdout tournament and match_idx via GET /api/v1/fixtures/historical."""
    split = "Holdout Test (2024–2026)"
    base_resp = http_client.get("/api/v1/fixtures/historical", params={"split": split})
    assert base_resp.status_code == 200
    base_data = base_resp.json()
    tournaments = base_data["tournaments"]
    assert len(tournaments) > 0
    chosen_tourn = preferred_tournament if preferred_tournament in tournaments else tournaments[0]

    t_resp = http_client.get(
        "/api/v1/fixtures/historical",
        params={"split": split, "tournament": chosen_tourn},
    )
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["selected_tournament"] == chosen_tourn
    assert len(t_data["matches"]) > 0
    return chosen_tourn, t_data["matches"][0]["match_idx"]


def test_f6_holdout_tournament_matches_discoverable(http_client):
    split = "Holdout Test (2024–2026)"
    base = http_client.get("/api/v1/fixtures/historical", params={"split": split}).json()
    chosen_tourn = "Rio Open" if "Rio Open" in base["tournaments"] else base["tournaments"][0]
    resp = http_client.get(
        "/api/v1/fixtures/historical",
        params={"split": split, "tournament": chosen_tourn},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_tournament"] == chosen_tourn
    assert len(data["matches"]) > 0
    for m in data["matches"]:
        assert isinstance(m["label"], str) and m["label"].strip()
        assert isinstance(m["match_idx"], str) and m["match_idx"].strip()


def test_f6_historical_inspect_metadata(http_client):
    chosen_tourn, match_idx = _discover_holdout_match(http_client, preferred_tournament="Rio Open")
    resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": chosen_tourn,
            "match_idx": match_idx,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_historical"] is True
    assert isinstance(data["actual_winner"], str) and data["actual_winner"].strip()
    assert isinstance(data["score_str"], str) and data["score_str"].strip()
    assert isinstance(data["date_str"], str) and data["date_str"].strip()
    assert isinstance(data["round_name"], str) and data["round_name"].strip()
    assert data["tournament"] == chosen_tourn
    assert isinstance(data["prediction_accurate"], bool)


def test_f6_historical_inspect_arcade_payload_and_accuracy(http_client):
    chosen_tourn, match_idx = _discover_holdout_match(http_client, preferred_tournament="Rio Open")
    resp = http_client.get(
        "/api/v1/fixtures/historical/inspect",
        params={
            "split": "Holdout Test (2024–2026)",
            "tournament": chosen_tourn,
            "match_idx": match_idx,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert_valid_player_card(data["player_cards"]["p1"], expected_name=data["p1"])
    assert_valid_player_card(data["player_cards"]["p2"], expected_name=data["p2"])
    assert_valid_treeshap(data["treeshap"])
    assert_valid_features_31(data["features_31"])
    assert data["bm_prob"] is None or (0.0 <= data["bm_prob"] <= 1.0)
    expected_acc = (data["prob_lgb"] >= 0.5) == (data["actual_winner"] == data["p1"])
    assert data["prediction_accurate"] is expected_acc


# ===========================================================================
# F7: Daily/Live Schedule & Refresh (5 tests)
# ===========================================================================


def test_f7_schedule_live_endpoint_contract(http_client):
    resp = http_client.get("/api/v1/schedule/live")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["kpis"], dict)
    for k in ("total_fixtures", "live_count", "value_edges_count", "divergent_count"):
        assert k in data["kpis"] and isinstance(data["kpis"][k], int)
    assert isinstance(data["fixtures"], list)
    assert isinstance(data["sync_status"], dict)
    assert isinstance(data["source"], str)


def test_f7_isolated_empty_data_returns_empty_schedule(tmp_path, monkeypatch):
    _isolate_schedule_paths(monkeypatch, tmp_path)
    client = create_inprocess_client(monkeypatch)
    resp = client.get("/api/v1/schedule/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fixtures"] == []
    assert data["kpis"]["total_fixtures"] == 0
    assert data["source"] == "none"


def test_f7_isolated_refresh_offline_espn_soft_fallback(tmp_path, monkeypatch):
    _isolate_schedule_paths(monkeypatch, tmp_path)

    def _offline_get(*args, **kwargs):
        raise requests.ConnectionError("Simulated offline network")

    monkeypatch.setattr(atp_daily_sync.requests, "get", _offline_get)
    client = create_inprocess_client(monkeypatch)
    resp = client.post("/api/v1/schedule/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["refreshed"], bool)
    assert data["refreshed"] is False
    assert isinstance(data["refresh_error"], str) and data["refresh_error"].strip()


def test_f7_isolated_refresh_with_mocked_espn_json(tmp_path, monkeypatch):
    _isolate_schedule_paths(monkeypatch, tmp_path)
    espn_payload = {
        "events": [
            {
                "name": "Mutua Madrid Open",
                "groupings": [
                    {
                        "grouping": {"displayName": "Men's Singles"},
                        "competitions": [
                            {
                                "id": "espn-t1-01",
                                "date": "2026-11-21T15:00Z",
                                "round": {"displayName": "Semifinals"},
                                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                                "competitors": [
                                    {"athlete": {"displayName": "Carlos Alcaraz"}},
                                    {"athlete": {"displayName": "Jannik Sinner"}},
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
    data = resp.json()
    assert data["refreshed"] is True
    assert data["refresh_error"] is None
    assert data["kpis"]["total_fixtures"] >= 1
    fx = data["fixtures"][0]
    assert fx["p1"] == "Alcaraz C."
    assert fx["p2"] == "Sinner J."
    assert 0.0 < fx["prob_lgb"] < 1.0
    assert 0.0 < fx["prob_lr"] < 1.0


def test_f7_isolated_t30m_snapshot_lock_preserves_predictions(tmp_path, monkeypatch):
    _isolate_schedule_paths(monkeypatch, tmp_path)
    t_soon = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    initial_fixtures = [
        {
            "fixture_id": "lock-test-001",
            "Date": "2026-11-21",
            "Scheduled_Time": t_soon,
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Final",
            "Player_1": "Carlos Alcaraz",
            "Player_2": "Jannik Sinner",
            "Status": "LineupsConfirmed",
            "Status_Detail": "15m to start",
        }
    ]
    first = atp_daily_sync.predict_published_schedule(
        fixtures=initial_fixtures, source_label="espn_atp", save_outputs=True
    )
    fx1 = first["fixtures"][0]
    assert fx1["snapshot_locked"] is True

    second_fixtures = [
        {
            **initial_fixtures[0],
            "Status": "LiveInPlay",
            "Status_Detail": "Set 1 4-3",
        }
    ]
    second = atp_daily_sync.predict_published_schedule(
        fixtures=second_fixtures, source_label="espn_atp", save_outputs=True
    )
    fx2 = second["fixtures"][0]
    assert fx2["snapshot_locked"] is True
    assert fx2["prob_lgb_p1"] == fx1["prob_lgb_p1"]
    assert fx2["prob_lr_p1"] == fx1["prob_lr_p1"]
    assert fx2["predicted_at"] == fx1["predicted_at"]
    assert fx2["Status"] == "LiveInPlay"


# ===========================================================================
# F8: Neon ESPN UI Assets & Markup Contracts (5 tests)
# ===========================================================================


def test_f8_styles_css_defines_obsidian_and_neon_palette(http_client):
    html_resp = http_client.get("/")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "").lower()
    css_url = _find_css_asset_url(html_resp.text)
    css_resp = http_client.get(css_url)
    assert css_resp.status_code == 200
    css = css_resp.text.lower()
    for hex_token in ("#060913", "#00f5ff", "#ccff00", "#ff2bd6"):
        assert hex_token in css, f"Expected neon token {hex_token} in {css_url}"


def test_f8_html_contains_zero_external_cdn_links(http_client):
    html_resp = http_client.get("/")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "").lower()
    html = html_resp.text
    external_tags = re.findall(
        r'<(?:script|link)[^>]+(?:src|href)=["\'](?:https?:)?//[^"\']+["\']',
        html,
        flags=re.IGNORECASE,
    )
    assert external_tags == [], f"Unexpected external CDN tags in index.html: {external_tags}"


def test_f8_html_contains_ticker_and_three_mode_controls(http_client):
    html_resp = http_client.get("/")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "").lower()
    html = html_resp.text.lower()
    assert "ticker" in html
    assert "schedule" in html
    assert "arena" in html or "h2h" in html
    assert "historical" in html or "backtrack" in html


def test_f8_ui_references_full_arcade_suite(http_client):
    html_resp = http_client.get("/")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "").lower()
    parts = [html_resp.text]
    for url in _extract_local_asset_urls(html_resp.text):
        r = http_client.get(url)
        assert r.status_code == 200, f"GET {url} returned {r.status_code}"
        parts.append(r.text)
    bundle_text = "\n".join(parts).lower()
    for concept in ("ovr", "radar", "treeshap", "divergen"):
        assert concept in bundle_text, f"Expected Arcade concept {concept!r} in frontend assets"
    assert "31" in bundle_text


def test_f8_live_chromium_boot_and_theme(live_server):
    result = run_cdp_scenario(live_server["base_url"], "boot_and_theme")
    assert result["ok"] is True
    assert result["data"]["bodyBg"] == "rgb(6, 9, 19)"
    assert result["data"]["hasCyan"] is True
    assert result["data"]["hasVolt"] is True
    assert result["data"]["hasMagenta"] is True


# ===========================================================================
# F9: Cloud Run Dockerization & Hydrogen Worker Offload (5 tests)
# ===========================================================================


def test_f9_dockerfile_cloud_run_contract():
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Missing Dockerfile for Cloud Run deployment"
    text = dockerfile.read_text(encoding="utf-8")
    assert "requirements.txt" in text
    cmd_lines = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*(CMD|ENTRYPOINT)\b", line)
    ]
    assert cmd_lines, "Dockerfile must define CMD or ENTRYPOINT"
    launch_stmt = " ".join(cmd_lines)
    assert "0.0.0.0" in launch_stmt
    assert re.search(r"\$\{?PORT\b", launch_stmt), (
        f"Dockerfile CMD/ENTRYPOINT must reference $PORT dynamically: {launch_stmt}"
    )
    assert not re.search(r"--port\s+8080\b", launch_stmt), (
        f"Dockerfile CMD/ENTRYPOINT must not hardcode --port 8080: {launch_stmt}"
    )
    assert "uvicorn" in launch_stmt
    assert "server:app" in launch_stmt


def test_f9_dockerignore_excludes_dev_artifacts_not_runtime():
    dockerignore = REPO_ROOT / ".dockerignore"
    assert dockerignore.is_file(), "Missing .dockerignore"
    lines = [
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(lines)
    for required_ignore in (".venv", ".git", "__pycache__"):
        assert required_ignore in joined, f".dockerignore should exclude {required_ignore}"
    forbidden_patterns = {
        "*",
        "*.py",
        "*.joblib",
        "*.csv",
        "web",
        "web/",
        "web/*",
        "server.py",
        "atp_engine.py",
        "atp_service.py",
        "atp_data.py",
        "atp_features.py",
        "atp_cards.py",
        "atp_daily_sync.py",
        "atp_inference_bundle.joblib",
        "train_features.csv",
        "test_features.csv",
        "match_scores.csv",
        "requirements.txt",
    }
    for line in lines:
        assert line not in forbidden_patterns, f".dockerignore must NOT exclude {line!r}"


def test_f9_requirements_declares_fastapi_and_uvicorn():
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("fastapi", "uvicorn", "lightgbm", "scikit-learn", "joblib", "pandas", "scipy"):
        assert re.search(rf"^\s*{re.escape(pkg)}\b", req, flags=re.MULTILINE), (
            f"requirements.txt missing required package {pkg!r}"
        )


def test_f9_daily_sync_predict_schedule_uses_power_devig(tmp_path, monkeypatch):
    _isolate_schedule_paths(monkeypatch, tmp_path)
    fixtures = [
        {
            "fixture_id": "power-devig-001",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Semifinals",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": 1.50,
            "Odd_2": 2.75,
        }
    ]
    payload = atp_daily_sync.predict_published_schedule(
        fixtures=fixtures,
        source_label="file:test",
        target_date="2026-11-21",
        save_outputs=True,
    )
    fx = payload["fixtures"][0]
    exp_p1, exp_p2, _, exp_k = solve_power_devig(1.50, 2.75)
    assert fx["bm_implied_p1"] == pytest.approx(exp_p1, abs=1e-4)
    assert ((1.0 / 1.50) ** (1.0 / exp_k)) + ((1.0 / 2.75) ** (1.0 / exp_k)) == pytest.approx(1.0, abs=1e-6)
    assert fx["edge_lgb_p1"] == pytest.approx(fx["prob_lgb_p1"] - exp_p1, abs=1e-4)


def test_f9_github_workflow_offloads_daily_sync():
    wf = REPO_ROOT / ".github" / "workflows" / "daily_atp_sync.yml"
    assert wf.is_file(), "Missing .github/workflows/daily_atp_sync.yml"
    text = wf.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert "cron:" in text
    assert "atp_daily_sync.py" in text
