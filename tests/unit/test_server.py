import json
import math
import time
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline import atp_daily_sync, atp_data
from src import atp_engine, server

client = TestClient(server.app)


def test_root_and_healthz():
    r_root = client.get("/")
    assert r_root.status_code == 200

    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert data["players_total"] == len(atp_engine.all_players)
    assert data["players_active"] == len(atp_engine.active_players)
    assert isinstance(data["sync_status"], dict)
    assert isinstance(data["schedule_fixtures_count"], int)


def test_meta_endpoint():
    r = client.get("/api/v1/meta")
    assert r.status_code == 200
    data = r.json()
    assert data["default_p1"] == atp_engine.DEFAULT_P1
    assert data["default_p2"] == atp_engine.DEFAULT_P2
    assert set(data["surfaces"]) == {"Hard", "Clay", "Grass", "Carpet"}
    assert "Grand Slam" in data["series"]
    assert data["splits"] == atp_data.SPLIT_CHOICES
    assert len(data["active_players"]) <= len(data["all_players"])
    metrics = data["model_metrics"]
    assert metrics["lgbm_logloss"] == pytest.approx(0.6149)
    assert metrics["lgbm_brier"] == pytest.approx(0.2139)
    assert metrics["lgbm_auc"] == pytest.approx(0.7171)
    assert metrics["trees"] == int(atp_engine.champion_lgb.booster_.num_trees())
    assert metrics["features_count"] == 31

    r_all = client.get("/api/v1/meta?all_time=true")
    assert r_all.status_code == 200


def test_predict_endpoint_and_treeshap_symmetry():
    r1 = client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.",
        "p2": "Sinner J.",
        "surface": "Hard",
        "series": "Grand Slam",
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["edge"] is None
    assert d1["sym_score"] < 1e-5
    assert d1["treeshap"]["bias"] == pytest.approx(0.0, abs=1e-12)
    assert len(d1["treeshap"]["contributions"]) == 31
    assert len(d1["features_31"]) == 31
    assert "p1" in d1["player_cards"] and "p2" in d1["player_cards"]

    r2 = client.post("/api/v1/predict", json={
        "p1": "Sinner J.",
        "p2": "Alcaraz C.",
        "surface": "Hard",
        "series": "Grand Slam",
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d1["prob_lgb"] + d2["prob_lgb"] == pytest.approx(1.0, abs=1e-6)
    assert d1["prob_lr"] + d2["prob_lr"] == pytest.approx(1.0, abs=1e-6)
    for f1, f2 in zip(d1["features_31"], d2["features_31"]):
        assert f1["shap"] == pytest.approx(-f2["shap"], abs=1e-9)


def test_predict_validation_errors():
    assert client.post("/api/v1/predict", json={
        "p1": "", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam"
    }).status_code in (400, 422)

    assert client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.", "p2": "Alcaraz C.", "surface": "Hard", "series": "Grand Slam"
    }).status_code in (400, 422)

    assert client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Wood", "series": "Grand Slam"
    }).status_code in (400, 422)

    assert client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "InvalidSeries"
    }).status_code in (400, 422)


def test_power_method_devig_and_kelly_math():
    # Zero-vig fair book: p=1.0, power_k=1.0
    fair = atp_engine.compute_edge_and_kelly(0.55, 2.0, 2.0)
    assert fair is not None
    assert fair["overround"] == pytest.approx(1.0, abs=1e-6)
    assert fair["power_k"] == pytest.approx(1.0, abs=1e-6)
    assert fair["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)
    assert fair["edge_p1"] == pytest.approx(0.05, abs=1e-6)
    assert fair["signal"] == "VALUE_P1"
    # f* = ((2-1)*0.55 - 0.45)/(2-1) = 0.10 -> 0.25*10 = 2.5%
    assert fair["kelly_p1_pct"] == pytest.approx(2.5, abs=1e-4)
    assert fair["kelly_p2_pct"] == 0.0

    # Value on P2
    val_p2 = atp_engine.compute_edge_and_kelly(0.40, 1.85, 2.05)
    assert val_p2["signal"] == "VALUE_P2"
    assert val_p2["implied_prob_p1"] + val_p2["implied_prob_p2"] == pytest.approx(1.0, abs=1e-6)
    q1, q2 = 1.0 / 1.85, 1.0 / 2.05
    p_exp = 1.0 / val_p2["power_k"]
    assert (q1 ** p_exp) + (q2 ** p_exp) == pytest.approx(1.0, abs=1e-5)

    # Alignment signal (-0.02 < edge_p1 < 0.02)
    align = atp_engine.compute_edge_and_kelly(0.505, 2.0, 2.0)
    assert align["signal"] == "ALIGNMENT"

    # Negative signal (edge_p1 <= -0.02 and edge_p2 < 0.03)
    neg = atp_engine.compute_edge_and_kelly(0.475, 2.0, 2.0)
    assert neg["signal"] == "NEGATIVE"

    # HTTP integration with odds and invalid odds
    r_ok = client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam",
        "odd_1": 1.85, "odd_2": 2.05,
    })
    assert r_ok.status_code == 200
    assert r_ok.json()["edge"] is not None

    r_bad_odds = client.post("/api/v1/predict", json={
        "p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam",
        "odd_1": 0.95, "odd_2": 2.05,
    })
    assert r_bad_odds.status_code == 200
    assert r_bad_odds.json()["edge"] is None

    # Heavy-vig extreme odds (1.001, 1.001) and (1.005, 1.005) + full-precision power_k
    for odd_val in (1.001, 1.005):
        r_ext = client.post("/api/v1/predict", json={
            "p1": "Alcaraz C.", "p2": "Sinner J.", "surface": "Hard", "series": "Grand Slam",
            "odd_1": odd_val, "odd_2": odd_val,
        })
        assert r_ext.status_code == 200
        edge_ext = r_ext.json()["edge"]
        assert edge_ext is not None
        pk = edge_ext["power_k"]
        assert abs((1.0 / odd_val) ** (1.0 / pk) + (1.0 / odd_val) ** (1.0 / pk) - 1.0) < 1e-10
        assert edge_ext["implied_prob_p1"] == pytest.approx(0.5, abs=1e-6)


def test_predict_preserves_exact_trailing_space_player_keys():
    if "Nadal R. " not in atp_engine.latest_player_state:
        pytest.skip("'Nadal R. ' not present in latest_player_state")

    r_space = client.post("/api/v1/predict", json={
        "p1": "Nadal R. ",
        "p2": "Sinner J.",
        "surface": "Clay",
        "series": "Grand Slam",
    })
    assert r_space.status_code == 200
    d_space = r_space.json()
    assert d_space["p1"] == "Nadal R. "
    assert d_space["player_cards"]["p1"]["rank"] == int(round(atp_engine.latest_player_state["Nadal R. "]["Rank"]))

    r_clean = client.post("/api/v1/predict", json={
        "p1": "Nadal R.",
        "p2": "Sinner J.",
        "surface": "Clay",
        "series": "Grand Slam",
    })
    assert r_clean.status_code == 200
    assert r_clean.json()["p1"] == "Nadal R."
    assert r_clean.json()["player_cards"]["p1"]["rank"] != d_space["player_cards"]["p1"]["rank"]


def test_historical_fixtures_and_inspect_endpoints():
    r_bad_split = client.get("/api/v1/fixtures/historical?split=BogusSplit")
    assert r_bad_split.status_code == 400

    split = atp_data.SPLIT_CHOICES[0]
    r_fix = client.get("/api/v1/fixtures/historical", params={"split": split})
    assert r_fix.status_code == 200
    fix_data = r_fix.json()
    assert fix_data["split"] == split
    assert len(fix_data["tournaments"]) > 0
    assert fix_data["selected_tournament"] in fix_data["tournaments"]
    assert len(fix_data["matches"]) > 0

    first_match = fix_data["matches"][0]
    r_insp = client.get("/api/v1/fixtures/historical/inspect", params={
        "split": split,
        "tournament": fix_data["selected_tournament"],
        "match_idx": first_match["match_idx"],
    })
    assert r_insp.status_code == 200
    insp = r_insp.json()
    assert insp["is_historical"] is True
    assert insp["tournament"] == fix_data["selected_tournament"]
    assert isinstance(insp["prediction_accurate"], bool)
    assert len(insp["features_31"]) == 31
    assert len(insp["treeshap"]["contributions"]) == 31

    # Missing / invalid / not found / mismatched tournament
    assert client.get("/api/v1/fixtures/historical/inspect", params={
        "split": split, "tournament": fix_data["selected_tournament"], "match_idx": "999999999"
    }).status_code == 404
    assert client.get("/api/v1/fixtures/historical/inspect", params={
        "split": split, "tournament": "Nonexistent Open 9999", "match_idx": first_match["match_idx"]
    }).status_code == 404
    if len(fix_data["tournaments"]) > 1:
        other_tourn = fix_data["tournaments"][1]
        assert client.get("/api/v1/fixtures/historical/inspect", params={
            "split": split, "tournament": other_tourn, "match_idx": first_match["match_idx"]
        }).status_code == 404
    assert client.get("/api/v1/fixtures/historical/inspect", params={
        "split": split, "tournament": fix_data["selected_tournament"], "match_idx": "abc"
    }).status_code == 400


def test_h2h_index_parity_and_monkeypatch_rebuild(monkeypatch):
    # Classic rivalry parity
    h2h = atp_engine.get_h2h_stats("Nadal R.", "Federer R.", "Clay")
    assert h2h["h2h_match_count"] == 40
    assert h2h["h2h_surface_count"] == 16
    assert h2h["h2h_winrate"] == pytest.approx(0.6, abs=1e-6)

    # Strict cutoff_date parity
    h2h_cut = atp_engine.get_h2h_stats("Nadal R.", "Federer R.", "Clay", cutoff_date="2012-01-01")
    df = atp_data.full_df
    sub = df[
        (((df["Player_1"] == "Nadal R.") & (df["Player_2"] == "Federer R.")) |
         ((df["Player_1"] == "Federer R.") & (df["Player_2"] == "Nadal R."))) &
        (pd.to_datetime(df["Date"]) < pd.to_datetime("2012-01-01"))
    ]
    assert h2h_cut["h2h_match_count"] == len(sub)
    assert h2h_cut["h2h_surface_count"] == int((sub["Surface"] == "Clay").sum())

    # Monkeypatching atp_data.full_df auto-rebuilds _h2h_index
    toy_df = pd.DataFrame([
        {"Date": "2020-05-01", "Player_1": "Alpha A.", "Player_2": "Beta B.", "Surface": "Clay", "Winner": "Alpha A."},
        {"Date": "2021-05-01", "Player_1": "Beta B.", "Player_2": "Alpha A.", "Surface": "Hard", "Winner": "Beta B."},
    ])
    monkeypatch.setattr(atp_data, "full_df", toy_df)
    toy_h2h = atp_engine.get_h2h_stats("Alpha A.", "Beta B.", "Clay", cutoff_date="2021-01-01")
    assert toy_h2h["h2h_match_count"] == 1
    assert toy_h2h["h2h_winrate"] == pytest.approx(1.0)
    monkeypatch.undo()
    # Force rebuild back to original full_df
    assert atp_engine.get_h2h_stats("Nadal R.", "Federer R.", "Clay")["h2h_match_count"] == 40


def test_warm_prediction_latency_under_15ms():
    # Warm-up
    res_fast = atp_engine.predict_match_structured("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam")

    # Verify fast evaluators match sklearn/LightGBM predict_proba to <1e-12
    X_fwd = atp_engine.build_match_features("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam")
    X_inv = atp_engine.build_match_features("Sinner J.", "Alcaraz C.", "Hard", "Grand Slam")
    ref_lgb = (
        float(atp_engine.champion_lgb.predict_proba(X_fwd)[0, 1])
        + (1.0 - float(atp_engine.champion_lgb.predict_proba(X_inv)[0, 1]))
    ) / 2.0
    ref_lr = (
        float(atp_engine.champion_lr.predict_proba(X_fwd)[0, 1])
        + (1.0 - float(atp_engine.champion_lr.predict_proba(X_inv)[0, 1]))
    ) / 2.0
    assert abs(res_fast["prob_lgb"] - ref_lgb) < 1e-12
    assert abs(res_fast["prob_lr"] - ref_lr) < 1e-12

    times_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        atp_engine.predict_match_structured("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam")
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times_ms)
    p95_ms = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    assert p95_ms < 15.0, f"Warm predict_match_structured p95 latency {p95_ms:.2f}ms >= 15ms"


def test_build_match_feature_pair_np_parity():
    cases = [
        ("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam", None),
        ("Nadal R.", "Federer R.", "Clay", "Grand Slam", "2012-01-01"),
        ("Unknown Challenger X.", "Alcaraz C.", "Grass", "ATP500", None),
        ("Unknown Challenger X.", "Unknown Challenger Y.", "Carpet", "ATP250", "2025-05-01"),
    ]
    for p1, p2, surf, ser, dt in cases:
        X_fwd = atp_engine.build_match_features(p1, p2, surf, ser, target_date=dt)
        X_inv = atp_engine.build_match_features(p2, p1, surf, ser, target_date=dt)
        ref = np.vstack([X_fwd.to_numpy(dtype=np.float64), X_inv.to_numpy(dtype=np.float64)])
        fast, _, p1_k, p2_k = atp_engine._build_match_feature_pair_np(p1, p2, surf, ser, target_date=dt)
        assert p1_k == (p1 in atp_engine.latest_player_state)
        assert p2_k == (p2 in atp_engine.latest_player_state)
        assert np.array_equal(np.isnan(ref), np.isnan(fast))
        valid_mask = ~np.isnan(ref)
        if np.any(valid_mask):
            assert np.max(np.abs(ref[valid_mask] - fast[valid_mask])) < 1e-14


def test_live_schedule_empty_data_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(atp_data, "DATA_DIR", str(tmp_path / "missing_data"))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(tmp_path / "missing_data" / "sync_state.json"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(tmp_path / "missing_data" / "daily_predictions_latest.json"))

    r_sched = client.get("/api/v1/schedule/live")
    assert r_sched.status_code == 200
    data = r_sched.json()
    assert data["fixtures"] == []
    assert data["kpis"] == {"total_fixtures": 0, "live_count": 0, "value_edges_count": 0, "divergent_count": 0}
    assert data["source"] == "none"

    r_hz = client.get("/healthz")
    assert r_hz.status_code == 200
    assert r_hz.json()["schedule_fixtures_count"] == 0


def test_schedule_refresh_offline_resilience_and_success(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    hist_dir = data_dir / "predictions_history"
    data_dir.mkdir(parents=True, exist_ok=True)
    sched_json = data_dir / "daily_predictions_latest.json"

    monkeypatch.setattr(atp_data, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_data, "SYNC_STATE_PATH", str(data_dir / "sync_state.json"))
    monkeypatch.setattr(atp_data, "DAILY_PREDICTIONS_JSON_PATH", str(sched_json))
    monkeypatch.setattr(atp_daily_sync, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(atp_daily_sync, "HISTORY_DIR", str(hist_dir))
    monkeypatch.setattr(atp_daily_sync, "DEFAULT_SCHEDULE_CSV", str(data_dir / "missing_upcoming.csv"))
    monkeypatch.setattr(atp_daily_sync, "PENDING_KAGGLE_CSV", str(data_dir / "missing_pending.csv"))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_JSON_PATH", str(sched_json))
    monkeypatch.setattr(atp_daily_sync, "DAILY_PREDICTIONS_CSV_PATH", str(data_dir / "daily_predictions_latest.csv"))

    # 1) Offline ESPN with no cache -> 200 OK empty + refresh_error
    def fail_espn(*a, **kw):
        raise RuntimeError("ESPN offline")

    monkeypatch.setattr(atp_daily_sync, "fetch_espn_atp_schedule", fail_espn)
    r_empty = client.post("/api/v1/schedule/refresh")
    assert r_empty.status_code == 200
    d_empty = r_empty.json()
    assert d_empty["refreshed"] is False
    assert d_empty["refresh_error"]
    assert d_empty["fixtures"] == []

    # 2) Seed cached schedule, then simulate offline ESPN -> MUST NOT wipe cache!
    cached_payload = {
        "generated_at": "2026-11-21T08:00:00+00:00",
        "target_date": "2026-11-21",
        "source": "espn_atp",
        "fixtures": [
            {
                "fixture_id": "cached-1",
                "Tournament": "ATP Finals",
                "Round": "Final",
                "Surface": "Hard",
                "Series": "Masters Cup",
                "Status": "LiveInPlay",
                "Status_Detail": "Set 2",
                "Player_1": "Alcaraz C.",
                "Player_2": "Sinner J.",
                "prob_lgb_p1": 0.56,
                "prob_lr_p1": 0.54,
                "Odd_1": 1.80,
                "Odd_2": 2.10,
                "snapshot_locked": True,
            }
        ],
    }
    sched_json.write_text(json.dumps(cached_payload), encoding="utf-8")

    r_preserved = client.post("/api/v1/schedule/refresh")
    assert r_preserved.status_code == 200
    d_pres = r_preserved.json()
    assert d_pres["refreshed"] is False
    assert d_pres["refresh_error"]
    assert len(d_pres["fixtures"]) == 1
    assert d_pres["fixtures"][0]["fixture_id"] == "cached-1"
    assert d_pres["kpis"]["live_count"] == 1
    assert d_pres["fixtures"][0]["kelly_p1_pct"] is not None

    # 3) Online ESPN returning fixtures -> refreshes and computes Power-Method fields
    def ok_espn(*a, **kw):
        return [
            {
                "fixture_id": "live-2",
                "Date": "2026-11-21",
                "Scheduled_Time": "2026-11-21T15:00Z",
                "Tournament": "ATP Finals",
                "Surface": "Hard",
                "Series": "Masters Cup",
                "Round": "Semifinals",
                "Player_1": "Carlos Alcaraz",
                "Player_2": "Jannik Sinner",
                "Odd_1": 1.85,
                "Odd_2": 2.05,
                "Status": "LineupsConfirmed",
                "Status_Detail": "Scheduled",
                "Source": "espn_atp",
            }
        ]

    monkeypatch.setattr(atp_daily_sync, "fetch_espn_atp_schedule", ok_espn)
    r_ok = client.post("/api/v1/schedule/refresh")
    assert r_ok.status_code == 200
    d_ok = r_ok.json()
    assert d_ok["refreshed"] is True
    assert d_ok["refresh_error"] is None
    assert len(d_ok["fixtures"]) == 1
    fx = d_ok["fixtures"][0]
    assert fx["fixture_id"] == "live-2"
    assert fx["p1"] == "Alcaraz C."
    assert fx["p2"] == "Sinner J."
    assert fx["bm_prob"] is not None
    assert fx["kelly_p1_pct"] is not None
