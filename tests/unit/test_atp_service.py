from concurrent.futures import ThreadPoolExecutor
import math
import pytest
from pipeline import atp_data
import atp_engine
import atp_service

P1, P2, SURF, SER = "Alcaraz C.", "Sinner J.", "Hard", "Grand Slam"
SPLIT = atp_data.SPLIT_CHOICES[0]
TOURN = atp_data.get_tournaments_for_split(SPLIT)[0]
MATCH_IDX = atp_data.get_matches_for_tournament(SPLIT, TOURN)[0][1]


def test_predict_parity_with_engine_html():
    html = atp_engine.predict_matchup(P1, P2, SURF, SER)
    res = atp_service.predict(P1, P2, SURF, SER)
    assert "error" not in res
    assert f"{res['prob_lgb'] * 100:.1f}%" in html
    assert f"{res['prob_lr'] * 100:.1f}%" in html
    assert (res["p1"], res["p2"], res["surface"], res["series"]) == (P1, P2, SURF, SER)
    assert {"rank_diff", "h2h_p1_wins", "h2h_p2_wins"} <= res["key_stats"].keys()
    assert {"sym_score", "raw_asym", "is_divergent", "div_delta"} <= res.keys()


def test_inspect_historical_extras():
    res = atp_service.inspect(SPLIT, TOURN, MATCH_IDX)
    assert "error" not in res
    assert res["is_historical"] is True and res["tournament"] == TOURN
    for k in ("actual_winner", "score_str", "round_name", "date_str", "bm_prob"):
        assert k in res


def test_predict_error_paths():
    assert atp_service.predict("", P2, SURF, SER) == {"error": "Please select both Player 1 and Player 2."}
    assert atp_service.predict(P1, P1, SURF, SER) == {"error": "Player 1 and Player 2 must be different athletes."}


def test_inspect_error_paths():
    assert atp_service.inspect(SPLIT, TOURN, None) == {"error": "Select a valid tournament and matchup to inspect."}
    bad = atp_service.inspect(SPLIT, TOURN, "999999999")
    assert "error" in bad and "<" not in bad["error"]


def test_engine_function_restored_after_call():
    orig = atp_engine.render_diagnostic_card
    atp_service.predict(P1, P2, SURF, SER)
    atp_service.inspect(SPLIT, TOURN, "999999999")
    assert atp_engine.render_diagnostic_card is orig


def test_engine_function_restored_after_exception(monkeypatch):
    orig = atp_engine.render_diagnostic_card
    def boom(*a, **k): raise RuntimeError("fail")
    monkeypatch.setattr(atp_engine, "build_match_features", boom)
    with pytest.raises(RuntimeError, match="fail"):
        atp_service.predict(P1, P2, SURF, SER)
    assert atp_engine.render_diagnostic_card is orig


def test_concurrent_thread_safety_no_monkeypatch():
    pairs = [
        ("Alcaraz C.", "Sinner J.", "Hard", "Grand Slam"),
        ("Djokovic N.", "Zverev A.", "Clay", "Masters 1000"),
        ("Nadal R.", "Federer R.", "Clay", "Grand Slam"),
        ("Medvedev D.", "Tsitsipas S.", "Hard", "ATP500"),
    ]
    orig_card_fn = atp_engine.render_diagnostic_card

    def worker(args):
        p1, p2, s, ser = args
        res = atp_service.predict(p1, p2, s, ser)
        assert atp_engine.render_diagnostic_card is orig_card_fn
        return res

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, pairs * 6))

    assert len(results) == 24
    for r in results:
        assert "error" not in r
        assert r["sym_score"] < 1e-5
        assert len(r["features_31"]) == 31


def test_predict_treeshap_features_31_and_player_cards():
    fwd = atp_service.predict(P1, P2, SURF, SER)
    inv = atp_service.predict(P2, P1, SURF, SER)

    # TreeSHAP structure and exact swap negation symmetry
    assert fwd["treeshap"]["bias"] == pytest.approx(0.0, abs=1e-12)
    assert inv["treeshap"]["bias"] == pytest.approx(0.0, abs=1e-12)
    assert fwd["treeshap"]["p1_total_contrib"] == pytest.approx(-inv["treeshap"]["p1_total_contrib"], abs=1e-9)

    contribs = fwd["treeshap"]["contributions"]
    assert len(contribs) == 31
    abs_shaps = [abs(c["shap"]) for c in contribs]
    assert abs_shaps == sorted(abs_shaps, reverse=True)

    # features_31 canonical order and LR coefficient mapping
    f31_fwd = fwd["features_31"]
    f31_inv = inv["features_31"]
    assert [f["name"] for f in f31_fwd] == list(atp_engine.model_features)
    for a, b in zip(f31_fwd, f31_inv):
        assert a["shap"] == pytest.approx(-b["shap"], abs=1e-9)
        assert math.isfinite(a["value"]) and math.isfinite(a["shap"]) and math.isfinite(a["lr_coef"])
        assert a["category"] in {
            "Ranking", "Short Form (10M)", "Med Form (365d)", "Career",
            "Context & Fatigue", "Head-to-Head", "Match Context",
        }
    # Hard=3 and Grand Slam=2 are non-reference categories, so lr_coef != 0
    f_map = {f["name"]: f for f in f31_fwd}
    assert f_map["surface_encoded"]["lr_coef"] != 0.0
    assert f_map["series_encoded"]["lr_coef"] != 0.0
    assert f_map["is_grand_slam"]["lr_coef"] != 0.0

    # Carpet=0 and ATP250=0 are dropped reference categories -> lr_coef == 0.0
    ref_res = atp_service.predict(P1, P2, "Carpet", "ATP250")
    ref_map = {f["name"]: f for f in ref_res["features_31"]}
    assert ref_map["surface_encoded"]["lr_coef"] == 0.0
    assert ref_map["series_encoded"]["lr_coef"] == 0.0

    # Arcade player cards
    for side in ("p1", "p2"):
        card = fwd["player_cards"][side]
        assert 40 <= card["ovr"] <= 99
        assert isinstance(card["traits"], list)
        assert set(card["radar"].keys()) == {
            "short_form", "med_form", "surface_mastery",
            "rank_power", "clutch_tb", "stamina_freshness",
        }
        for v in card["radar"].values():
            assert 0.0 <= v <= 100.0

    # Cold-start player card safety
    cold = atp_service.predict("Unknown Challenger X.", P2, SURF, SER)
    assert cold["p1_known"] is False
    assert cold["player_cards"]["p1"]["traits"] == ["COLD START"]
    assert 40 <= cold["player_cards"]["p1"]["ovr"] <= 99


def test_predict_power_method_edge_and_kelly():
    no_odds = atp_service.predict(P1, P2, SURF, SER)
    assert no_odds["edge"] is None

    bad_odds = atp_service.predict(P1, P2, SURF, SER, odd_1=1.0, odd_2=2.0)
    assert bad_odds["edge"] is None

    with_odds = atp_service.predict(P1, P2, SURF, SER, odd_1=1.85, odd_2=2.05)
    edge = with_odds["edge"]
    assert edge is not None
    assert edge["implied_prob_p1"] + edge["implied_prob_p2"] == pytest.approx(1.0, abs=1e-6)
    assert edge["overround"] == pytest.approx(1.0 / 1.85 + 1.0 / 2.05, abs=1e-5)
    assert edge["edge_p1"] == pytest.approx(with_odds["prob_lgb"] - edge["implied_prob_p1"], abs=1e-5)
    assert edge["signal"] in {"VALUE_P1", "VALUE_P2", "ALIGNMENT", "NEGATIVE"}
    assert 0.0 <= edge["kelly_p1_pct"] <= 2.5
    assert 0.0 <= edge["kelly_p2_pct"] <= 2.5


def test_inspect_historical_structured_payload():
    res = atp_service.inspect(SPLIT, TOURN, MATCH_IDX)
    assert "error" not in res
    assert res["is_historical"] is True
    assert res["raw_asym"] == 0.0
    assert isinstance(res["prediction_accurate"], bool)
    assert len(res["features_31"]) == 31
    assert len(res["treeshap"]["contributions"]) == 31
    assert "p1" in res["player_cards"] and "p2" in res["player_cards"]
    assert res["edge"] is None
