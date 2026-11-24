import json
import os
import pandas as pd
import pytest

from pipeline import atp_daily_sync, atp_data, atp_features
from src import atp_engine, atp_service


def _sample_raw_matches() -> pd.DataFrame:
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
            "Tournament": "US Open",
            "Date": "2026-09-01",
            "Series": "Grand Slam",
            "Court": "Outdoor",
            "Surface": "Hard",
            "Round": "Quarterfinals",
            "Best of": 5,
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Winner": "Alcaraz C.",
            "Rank_1": 2,
            "Rank_2": 1,
            "Pts_1": 11000,
            "Pts_2": 11500,
            "Odd_1": -1,
            "Odd_2": -1,
            "Score": "6-0 6-4 6-2",
        },
        # Duplicate of previous match to test deduplication
        {
            "Tournament": "US Open",
            "Date": "2026-09-01",
            "Series": "Grand Slam",
            "Court": "Outdoor",
            "Surface": "Hard",
            "Round": "Quarterfinals",
            "Best of": 5,
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Winner": "Alcaraz C.",
            "Rank_1": 2,
            "Rank_2": 1,
            "Pts_1": 11000,
            "Pts_2": 11500,
            "Odd_1": 1.90,
            "Odd_2": 1.90,
            "Score": "6-0 6-4 6-2",
        },
        # Unplayed / pending row with empty Winner
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


def test_clean_and_engineer_features_schema_and_post_state():
    raw_df = _sample_raw_matches()
    completed, pending = atp_features.clean_raw_matches(raw_df)

    assert len(completed) == 3
    assert len(pending) == 1
    assert pd.isna(completed.loc[2, "Odd_1"])

    out = atp_features.compute_match_features(
        completed,
        le_surf=atp_engine.le_surf,
        le_series=atp_engine.le_series,
    )
    assert len(out["train_df"]) == 1
    assert len(out["test_df"]) == 2
    assert set(atp_engine.model_features) <= set(out["test_df"].columns)

    # First match in history has cold-start 0.5 career winrate diff (0.0)
    assert out["train_df"].loc[0, "career_winrate_diff"] == pytest.approx(0.0)

    # Post-match state reflects all 3 completed matches (Alcaraz 2-1, Sinner 1-2)
    alc_state = out["latest_player_state"]["Alcaraz C."]
    sin_state = out["latest_player_state"]["Sinner J."]
    assert alc_state["career_winrate"] == pytest.approx(2.0 / 3.0)
    assert sin_state["career_winrate"] == pytest.approx(1.0 / 3.0)
    assert out["player_surface_wr"][("Alcaraz C.", "Hard")] == pytest.approx(1.0)


def test_player_entity_resolver():
    known = [
        "Alcaraz C.",
        "Sinner J.",
        "De Minaur A.",
        "O Connell C.",
        "Van De Zandschulp B.",
        "Struff J.L.",
        "Etcheverry T.",
        "Etcheverry T. M.",
        "Agassi A.",
    ]
    resolver = atp_daily_sync.PlayerEntityResolver(known, active_players=["Alcaraz C.", "Sinner J.", "Etcheverry T."])

    assert resolver.resolve("Carlos Alcaraz")[0] == "Alcaraz C."
    assert resolver.resolve("Jannik Sinner")[0] == "Sinner J."
    assert resolver.resolve("Alex de Minaur")[0] == "De Minaur A."
    assert resolver.resolve("Christopher O'Connell")[0] == "O Connell C."
    assert resolver.resolve("Botic van de Zandschulp")[0] == "Van De Zandschulp B."
    assert resolver.resolve("Jan-Lennard Struff")[0] == "Struff J.L."
    assert resolver.resolve("Tomas Martin Etcheverry")[0] == "Etcheverry T."

    unknown_name, conf, method = resolver.resolve("Completely Fictional Player")
    assert unknown_name == "Completely Fictional Player"
    assert conf == 0.0
    assert method == "cold_start"


def test_infer_tournament_context():
    assert atp_daily_sync.infer_tournament_context("Roland Garros") == ("Clay", "Grand Slam")
    assert atp_daily_sync.infer_tournament_context("Wimbledon") == ("Grass", "Grand Slam")
    assert atp_daily_sync.infer_tournament_context("Internazionali BNL d'Italia") == ("Clay", "Masters 1000")
    assert atp_daily_sync.infer_tournament_context("Unknown Open", raw_surface="Grass", raw_series="ATP500") == ("Grass", "ATP500")


def test_fetch_espn_atp_schedule_parsing(monkeypatch):
    sample_json = {
        "events": [
            {
                "name": "Mutua Madrid Open",
                "groupings": [
                    {
                        "grouping": {"displayName": "Men's Singles"},
                        "competitions": [
                            {
                                "id": "espn-101",
                                "date": "2026-11-21T14:00Z",
                                "round": {"displayName": "Quarterfinals"},
                                "status": {"type": {"name": "STATUS_SCHEDULED", "description": "Scheduled"}},
                                "competitors": [
                                    {"athlete": {"displayName": "Carlos Alcaraz"}},
                                    {"athlete": {"displayName": "Jannik Sinner"}},
                                ],
                            },
                            {
                                "id": "espn-doubles",
                                "date": "2026-11-21T16:00Z",
                                "competitors": [
                                    {"athlete": {"displayName": "Player A / Player B"}},
                                    {"athlete": {"displayName": "Player C / Player D"}},
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return sample_json

    monkeypatch.setattr(atp_daily_sync.requests, "get", lambda *a, **kw: FakeResp())
    fixtures = atp_daily_sync.fetch_espn_atp_schedule(target_date="2026-11-21")
    assert len(fixtures) == 1
    assert fixtures[0]["Tournament"] == "Mutua Madrid Open"
    assert fixtures[0]["Surface"] == "Clay"
    assert fixtures[0]["Series"] == "Masters 1000"
    assert fixtures[0]["Player_1"] == "Carlos Alcaraz"
    assert fixtures[0]["Player_2"] == "Jannik Sinner"


def test_sync_and_predict_end_to_end_isolated(tmp_path, monkeypatch):
    # Isolate all output paths in tmp_path so repo files are never mutated
    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    raw_csv = tmp_path / "raw_kaggle.csv"
    sched_csv = tmp_path / "today_schedule.csv"

    _sample_raw_matches().to_csv(raw_csv, index=False)
    pd.DataFrame([
        {
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Semifinals",
            "Player_1": "Carlos Alcaraz",
            "Player_2": "Jannik Sinner",
            "Odd_1": 1.85,
            "Odd_2": 2.00,
        }
    ]).to_csv(sched_csv, index=False)

    monkeypatch.setattr(atp_data, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_data, "TEST_CSV_PATH", str(tmp_path / "test_features.csv"))
    monkeypatch.setattr(atp_data, "TRAIN_CSV_PATH", str(tmp_path / "train_features.csv"))
    monkeypatch.setattr(atp_data, "SCORES_CSV_PATH", str(tmp_path / "match_scores.csv"))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))

    monkeypatch.setattr(atp_engine, "PLAYER_STATE_CACHE_PATH", str(data_dir / "player_state_cache.joblib"))

    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(atp_daily_sync, "PENDING_KAGGLE_CSV", str(data_dir / "pending_kaggle_fixtures.csv"))
    monkeypatch.setattr(atp_daily_sync, "PLAYER_STATE_CACHE_PATH", str(data_dir / "player_state_cache.joblib"))
    monkeypatch.setattr(atp_daily_sync, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(data_dir / "daily_predictions_latest.csv"))

    try:
        res = atp_daily_sync.run_daily_pipeline(
            target_date="2026-11-21",
            schedule_file=str(sched_csv),
            local_raw_path=str(raw_csv),
            allow_espn=False,
        )
        assert res["sync"]["status"] == "synced"
        assert res["sync"]["completed_rows"] == 3
        assert res["sync"]["pending_kaggle_rows"] == 1
        assert res["predictions"]["fixture_count"] == 1

        fx = res["predictions"]["fixtures"][0]
        assert fx["Player_1"] == "Alcaraz C."
        assert fx["Player_2"] == "Sinner J."
        assert 0.0 < fx["prob_lgb_p1"] < 1.0
        assert fx["prob_lgb_p1"] + fx["prob_lgb_p2"] == pytest.approx(1.0, abs=1e-5)
        assert fx["bm_implied_p1"] is not None
        assert fx["edge_lgb_p1"] is not None

        # Second sync with unchanged raw file should skip recompute cleanly
        res2 = atp_daily_sync.sync_kaggle_dataset(local_raw_path=str(raw_csv))
        assert res2["status"] == "up_to_date"

        # HTML renderer works with generated payload
        html = atp_daily_sync.render_daily_schedule_html(res["predictions"])
        assert "Alcaraz C." in html
        assert "Sinner J." in html
        assert "ATP Finals" in html

        # Service helpers read isolated state cleanly
        assert atp_service.get_daily_schedule()["fixture_count"] == 1
    finally:
        # Restore original repo data in memory for subsequent tests
        monkeypatch.undo()
        atp_engine.reload_state()


def test_power_method_devig_math_and_symmetry():
    # Fair market (2.0 vs 2.0 -> zero overround, power_k == 1.0, implied_prob == 0.5)
    fair_p1 = atp_daily_sync._compute_bookmaker_implied_prob(2.0, 2.0)
    assert fair_p1 == pytest.approx(0.5, abs=1e-12)

    fair_edge = atp_engine.compute_edge_and_kelly(0.5, 2.0, 2.0)
    assert fair_edge is not None
    assert fair_edge["overround"] == pytest.approx(1.0, abs=1e-6)
    assert fair_edge["power_k"] == pytest.approx(1.0, abs=1e-6)
    assert fair_edge["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)
    assert fair_edge["implied_prob_p2"] == pytest.approx(0.5, abs=1e-6)

    # Skewed market (1.15 vs 6.00) demonstrates favorite-longshot bias separation vs proportional
    o1, o2 = 1.15, 6.00
    q1, q2 = 1.0 / o1, 1.0 / o2
    prop_p1 = q1 / (q1 + q2)
    pow_p1 = atp_daily_sync._compute_bookmaker_implied_prob(o1, o2)
    assert pow_p1 is not None
    assert pow_p1 != pytest.approx(prop_p1, abs=1e-4)
    assert pow_p1 > prop_p1  # Power method assigns more true prob to heavy favorite

    skew_edge = atp_engine.compute_edge_and_kelly(0.85, o1, o2)
    assert skew_edge is not None
    assert skew_edge["power_k"] < 1.0
    # Verify exact Power-Method equation q1**(1/k) + q2**(1/k) == 1.0 within 1e-9 using unrounded solve
    pow_p2 = atp_daily_sync._compute_bookmaker_implied_prob(o2, o1)
    assert pow_p2 is not None
    assert pow_p1 + pow_p2 == pytest.approx(1.0, abs=1e-9)
    # Derive exact k from pow_p1 = q1**(1/k) and verify q1**(1/k) + q2**(1/k) == 1.0 within 1e-9
    import math
    inv_k = math.log(pow_p1) / math.log(q1)
    assert (q1 ** inv_k) + (q2 ** inv_k) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "odd1, odd2",
    [
        (None, 2.0),
        (2.0, None),
        (1.0, 2.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (-1.5, 2.0),
        (float("nan"), 2.0),
        (2.0, float("inf")),
        ("bad", 2.0),
    ],
)
def test_power_method_invalid_odds_return_none(odd1, odd2):
    assert atp_daily_sync._compute_bookmaker_implied_prob(odd1, odd2) is None
    assert atp_engine.compute_edge_and_kelly(0.60, odd1, odd2) is None


def test_quarter_kelly_non_negative_and_capped():
    # Negative edge on P1 -> kelly_p1_pct == 0.0
    neg_edge = atp_engine.compute_edge_and_kelly(0.40, 1.80, 2.05)
    assert neg_edge is not None
    assert neg_edge["kelly_p1_pct"] == 0.0
    assert neg_edge["edge_p1"] < 0.0

    # Massive edge -> Quarter-Kelly capped strictly at 2.5%
    huge_edge = atp_engine.compute_edge_and_kelly(0.90, 2.50, 1.60)
    assert huge_edge is not None
    assert huge_edge["signal"] == "VALUE_P1"
    assert huge_edge["kelly_p1_pct"] == pytest.approx(2.5, abs=1e-6)
    assert 0.0 <= huge_edge["kelly_p2_pct"] <= 2.5

    # Moderate positive edge -> exact Quarter-Kelly formula 0.25 * ((b*p - (1-p)) / b) * 100
    mod_edge = atp_engine.compute_edge_and_kelly(0.56, 1.95, 1.95)
    assert mod_edge is not None
    b = 1.95 - 1.0
    expected_f = max(0.0, (b * 0.56 - 0.44) / b)
    expected_pct = round(min(2.5, 0.25 * expected_f * 100.0), 4)
    assert mod_edge["kelly_p1_pct"] == pytest.approx(expected_pct, abs=1e-6)
    assert 0.0 < mod_edge["kelly_p1_pct"] <= 2.5


def test_predict_published_schedule_exposes_devig_and_kelly_json_csv(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(data_dir / "daily_predictions_latest.csv"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))

    fixtures = [
        {
            "fixture_id": "fix_valid_odds",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Final",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": 3.50,  # Underdog odds so positive edge & Kelly > 0 are exercised
            "Odd_2": 1.35,
            "Status": "LineupsConfirmed",
        },
        {
            "fixture_id": "fix_missing_odds",
            "Date": "2026-11-21",
            "Tournament": "ATP Finals",
            "Surface": "Hard",
            "Series": "Masters Cup",
            "Round": "Semifinals",
            "Player_1": "Alcaraz C.",
            "Player_2": "Sinner J.",
            "Odd_1": None,
            "Odd_2": 1.90,
            "Status": "LineupsConfirmed",
        },
    ]

    payload = atp_daily_sync.predict_published_schedule(
        fixtures=fixtures,
        source_label="unit_test",
        target_date="2026-11-21",
        save_outputs=True,
    )
    assert payload["fixture_count"] == 2

    fx_valid = payload["fixtures"][0]
    assert fx_valid["bm_implied_p1"] == pytest.approx(
        atp_daily_sync._compute_bookmaker_implied_prob(3.50, 1.35), abs=1e-5
    )
    assert fx_valid["bm_implied_p1"] + fx_valid["bm_implied_p2"] == pytest.approx(1.0, abs=1e-5)
    assert fx_valid["overround"] > 1.0
    assert 0.0 < fx_valid["power_k"] < 1.0
    assert fx_valid["edge_lgb_p1"] == pytest.approx(
        fx_valid["prob_lgb_p1"] - fx_valid["bm_implied_p1"], abs=1e-5
    )
    assert fx_valid["edge_signal"] in {"VALUE_P1", "VALUE_P2", "ALIGNMENT", "NEGATIVE"}
    assert 0.0 <= fx_valid["kelly_p1_pct"] <= 2.5
    assert 0.0 <= fx_valid["kelly_p2_pct"] <= 2.5

    fx_missing = payload["fixtures"][1]
    assert fx_missing["bm_implied_p1"] is None
    assert fx_missing["bm_implied_p2"] is None
    assert fx_missing["overround"] is None
    assert fx_missing["power_k"] is None
    assert fx_missing["edge_lgb_p1"] is None
    assert fx_missing["edge_signal"] is None
    assert fx_missing["kelly_p1_pct"] is None
    assert fx_missing["kelly_p2_pct"] is None

    # Verify JSON and CSV files written to isolated tmp_path
    with open(data_dir / "daily_predictions_latest.json", "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert len(saved_json["fixtures"]) == 2

    csv_df = pd.read_csv(data_dir / "daily_predictions_latest.csv")
    hist_df = pd.read_csv(hist_dir / "predictions_2026-11-21.csv")
    for col in ("Overround", "Power_K", "Kelly_P1_Pct", "Kelly_P2_Pct", "Edge_Signal", "BM_Implied_P1", "Edge_LGB_P1"):
        assert col in csv_df.columns
        assert col in hist_df.columns
    assert 0.0 <= float(csv_df.loc[0, "Kelly_P1_Pct"]) <= 2.5
    assert pd.isna(csv_df.loc[1, "Kelly_P1_Pct"])

    html = atp_daily_sync.render_daily_schedule_html(payload)
    assert "1/4 Kelly P1:" in html or "Market Edge P1:" in html


@pytest.mark.parametrize("o1, o2", [(1.005, 1.005), (1.001, 1.001)])
def test_power_method_extreme_high_vig_near_one_odds(o1, o2, tmp_path, monkeypatch):
    # Dynamic bracketing handles 1/k > 100 cleanly (1.005 -> ~138.98, 1.001 -> ~693.5)
    p1 = atp_daily_sync._compute_bookmaker_implied_prob(o1, o2)
    assert p1 == pytest.approx(0.5, abs=1e-9)

    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(data_dir / "daily_predictions_latest.csv"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))

    payload = atp_daily_sync.predict_published_schedule(
        fixtures=[
            {
                "fixture_id": f"fix_extreme_{o1}",
                "Date": "2026-11-21",
                "Tournament": "ATP Finals",
                "Surface": "Hard",
                "Series": "Masters Cup",
                "Round": "Final",
                "Player_1": "Alcaraz C.",
                "Player_2": "Sinner J.",
                "Odd_1": o1,
                "Odd_2": o2,
                "Status": "LineupsConfirmed",
            }
        ],
        source_label="unit_test",
        target_date="2026-11-21",
        save_outputs=False,
    )
    fx = payload["fixtures"][0]
    assert fx["bm_implied_p1"] == pytest.approx(0.5, abs=1e-6)
    assert fx["bm_implied_p2"] == pytest.approx(0.5, abs=1e-6)
    assert fx["power_k"] is not None and 0.0 < fx["power_k"] < 0.01


def test_power_method_and_schedule_reject_booleans(tmp_path, monkeypatch):
    assert atp_daily_sync._compute_bookmaker_implied_prob(True, 2.0) is None
    assert atp_daily_sync._compute_bookmaker_implied_prob(2.0, False) is None
    assert atp_daily_sync._safe_optional_float(True) is None
    assert atp_daily_sync._safe_optional_float(False) is None

    # Also verify fallback inside _compute_edge_and_kelly_safe when atp_engine raises ValueError
    def _raise_val_err(*_args, **_kwargs):
        raise ValueError("f(a) and f(b) must have different signs")

    monkeypatch.setattr(atp_engine, "compute_edge_and_kelly", _raise_val_err)
    fallback_edge = atp_daily_sync._compute_edge_and_kelly_safe(0.55, 1.005, 1.005)
    assert fallback_edge is not None
    assert fallback_edge["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)
    assert fallback_edge["signal"] == "VALUE_P1"
    monkeypatch.undo()

    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(data_dir / "daily_predictions_latest.csv"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(data_dir / "daily_predictions_latest.json"))

    payload = atp_daily_sync.predict_published_schedule(
        fixtures=[
            {
                "fixture_id": "fix_bool_odds",
                "Date": "2026-11-21",
                "Tournament": "ATP Finals",
                "Surface": "Hard",
                "Series": "Masters Cup",
                "Round": "Final",
                "Player_1": "Alcaraz C.",
                "Player_2": "Sinner J.",
                "Odd_1": True,
                "Odd_2": 2.0,
                "Status": "LineupsConfirmed",
            }
        ],
        source_label="unit_test",
        target_date="2026-11-21",
        save_outputs=False,
    )
    fx = payload["fixtures"][0]
    assert fx["Odd_1"] is None
    assert fx["Odd_2"] == 2.0
    assert fx["bm_implied_p1"] is None
    assert fx["edge_lgb_p1"] is None
    assert fx["kelly_p1_pct"] is None


