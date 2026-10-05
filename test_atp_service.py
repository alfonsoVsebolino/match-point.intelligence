import pytest
import atp_data
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
