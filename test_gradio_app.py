import pytest
import pandas as pd
import gradio_app

def test_predict_matchup_symmetry_and_diagnostics():
    # Test valid matchup
    p1 = "Alcaraz C."
    p2 = "Sinner J."
    html_fwd = gradio_app.predict_matchup(p1, p2, "Hard", "Grand Slam")
    assert "Alcaraz C." in html_fwd
    assert "Sinner J." in html_fwd
    assert "LightGBM Champion" in html_fwd
    assert "Logistic Regression Baseline" in html_fwd
    assert "Rank Δ" in html_fwd
    assert "Surface WR Δ" in html_fwd
    assert "Form Div Δ" in html_fwd
    assert "Fatigue 14d Δ" in html_fwd
    assert "H2H Record" in html_fwd
    assert "Symmetry: Enforced" in html_fwd

    # Test swap identity
    html_inv = gradio_app.predict_matchup(p2, p1, "Hard", "Grand Slam")
    assert "Sinner J." in html_inv
    assert "Alcaraz C." in html_inv

def test_predict_matchup_identical_players():
    res = gradio_app.predict_matchup("Alcaraz C.", "Alcaraz C.", "Hard", "Grand Slam")
    assert "must be different" in res

def test_predict_matchup_missing_players():
    res = gradio_app.predict_matchup("", "Alcaraz C.", "Hard", "Grand Slam")
    assert "Please select both" in res

def test_historical_match_backtracker():
    tournaments = gradio_app.initial_tournaments
    assert len(tournaments) > 0, "Initial tournaments should not be empty"
    first_tourn = tournaments[0]
    matches = gradio_app.get_matches_for_tournament(first_tourn)
    assert len(matches) > 0, f"Tournament {first_tourn} should have matches"
    
    first_match_id = matches[0][1]
    card_html = gradio_app.inspect_historical_match("Holdout Test (2024–2026)", first_tourn, first_match_id)
    assert "Actual Winner:" in card_html
    assert "★ LightGBM Champion" in card_html
    assert "Rank Δ" in card_html
    assert "H2H Record" in card_html
    assert "Holdout Replay" in card_html
    assert "Symmetry: Enforced" not in card_html

def test_swap_players():
    p1_out, p2_out = gradio_app.on_swap_players("Alcaraz C.", "Sinner J.")
    assert p1_out == "Sinner J."
    assert p2_out == "Alcaraz C."

def test_roster_scope_toggle():
    # Modern active roster
    active_pool = gradio_app.active_players
    all_pool = gradio_app.all_players
    assert len(all_pool) >= len(active_pool)

    # Test toggle to all
    p1_upd, p2_upd = gradio_app.on_roster_toggle(True, "Alcaraz C.", "Sinner J.")
    assert len(p1_upd["choices"]) == len(all_pool)
    assert len(p2_upd["choices"]) == len(all_pool)

    # Test toggle to active
    p1_upd_act, p2_upd_act = gradio_app.on_roster_toggle(False, "Alcaraz C.", "Sinner J.")
    assert len(p1_upd_act["choices"]) == len(active_pool)
    assert len(p2_upd_act["choices"]) == len(active_pool)
