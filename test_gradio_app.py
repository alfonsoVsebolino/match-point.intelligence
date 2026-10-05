import pytest
import pandas as pd
import gradio_app
import atp_data
import atp_engine
import atp_cards

def test_predict_matchup_symmetry_and_diagnostics():
    p1 = "Alcaraz C."
    p2 = "Sinner J."
    html_fwd = atp_engine.predict_matchup(p1, p2, "Hard", "Grand Slam")
    assert "Alcaraz C." in html_fwd
    assert "Sinner J." in html_fwd
    assert "LightGBM Champion" in html_fwd
    assert "Logistic Regression Baseline" in html_fwd
    assert "Rank Δ" in html_fwd
    assert "Surface WR Δ" in html_fwd
    assert "Form Div Δ" in html_fwd
    assert "Fatigue 14d Δ" in html_fwd
    assert "H2H Record" in html_fwd
    assert "Symmetry: Δ &lt; 1e-5" in html_fwd

    # Test swap identity
    html_inv = atp_engine.predict_matchup(p2, p1, "Hard", "Grand Slam")
    assert "Sinner J." in html_inv
    assert "Alcaraz C." in html_inv

def test_predict_matchup_identical_players():
    res = atp_engine.predict_matchup("Alcaraz C.", "Alcaraz C.", "Hard", "Grand Slam")
    assert "must be different" in res

def test_predict_matchup_missing_players():
    res = atp_engine.predict_matchup("", "Alcaraz C.", "Hard", "Grand Slam")
    assert "Please select both" in res

def test_historical_match_backtracker():
    tournaments = atp_data.get_tournaments_for_split("Holdout Test (2024–2026)")
    assert len(tournaments) > 0, "Initial tournaments should not be empty"
    first_tourn = tournaments[0]
    matches = atp_data.get_matches_for_tournament("Holdout Test (2024–2026)", first_tourn)
    assert len(matches) > 0, f"Tournament {first_tourn} should have matches"
    
    first_match_id = matches[0][1]
    card_html = atp_engine.inspect_historical_match("Holdout Test (2024–2026)", first_tourn, first_match_id)
    assert "Actual Winner:" in card_html
    assert "★ LightGBM Champion" in card_html
    assert "Rank Δ" in card_html
    assert "H2H Record" in card_html
    # Mode B should not display the Mode A symmetry chip
    assert "Symmetry: Δ &lt; 1e-5" not in card_html

def test_swap_players():
    p1_out, p2_out = gradio_app.on_swap_players("Alcaraz C.", "Sinner J.")
    assert p1_out == "Sinner J."
    assert p2_out == "Alcaraz C."

def test_roster_scope_toggle():
    active_pool = atp_engine.active_players
    all_pool = atp_engine.all_players
    assert len(all_pool) >= len(active_pool)

    # Test toggle to all
    p1_upd, p2_upd = gradio_app.on_roster_toggle(True, "Alcaraz C.", "Sinner J.")
    assert len(p1_upd["choices"]) == len(all_pool)
    assert len(p2_upd["choices"]) == len(all_pool)

    # Test toggle to active
    p1_upd_act, p2_upd_act = gradio_app.on_roster_toggle(False, "Alcaraz C.", "Sinner J.")
    assert len(p1_upd_act["choices"]) == len(active_pool)
    assert len(p2_upd_act["choices"]) == len(active_pool)

def test_full_dataset_h2h_coverage():
    # Legendary rivalry across 2000-2023 (Nadal vs Federer)
    h2h = atp_engine.get_h2h_stats("Nadal R.", "Federer R.", "Clay")
    assert h2h["h2h_match_count"] == 40
    assert h2h["h2h_surface_count"] == 16
    assert abs(h2h["h2h_winrate"] - 0.6) < 1e-3

def test_split_switching_and_cascading():
    # Test Holdout split
    holdout_tourns = atp_data.get_tournaments_for_split("Holdout Test (2024–2026)")
    assert len(holdout_tourns) > 0
    # Test Full Dataset split
    full_tourns = atp_data.get_tournaments_for_split("Full Dataset (2000–2026)")
    assert len(full_tourns) > len(holdout_tourns)

    # Test cascading callback in Gradio app
    tourn_upd, match_upd = gradio_app.on_split_change("Full Dataset (2000–2026)")
    assert len(tourn_upd["choices"]) == len(full_tourns)
    assert len(match_upd["choices"]) > 0
