import subprocess
import sys
import pytest
import atp_engine
from atp_engine import DEFAULT_P1, DEFAULT_P2, active_players, all_players

def test_no_gradio_imported():
    cmd = [
        sys.executable,
        "-c",
        "import tk_app, sys; assert 'gradio' not in sys.modules, 'gradio was unexpectedly imported'"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Import check failed: {res.stderr}"

def test_constants_and_defaults():
    import tk_app
    assert tk_app.DEFAULT_P1 == DEFAULT_P1
    assert tk_app.DEFAULT_P2 == DEFAULT_P2
    assert tk_app.DEFAULT_SURFACE == "Hard"
    assert tk_app.DEFAULT_SERIES == "Grand Slam"
    assert tk_app.SURFACE_CHOICES == ["Hard", "Clay", "Grass", "Carpet"]
    assert tk_app.SERIES_CHOICES == ["Grand Slam", "Masters 1000", "ATP500", "ATP250"]

def test_resolve_swap():
    from tk_app import resolve_swap
    p1, p2 = resolve_swap("Alcaraz C.", "Sinner J.")
    assert p1 == "Sinner J."
    assert p2 == "Alcaraz C."

def test_resolve_roster_toggle_unlock_all():
    from tk_app import resolve_roster_toggle
    active_subset = ["Alcaraz C.", "Sinner J."]
    all_subset = ["Agassi A.", "Alcaraz C.", "Sampras P.", "Sinner J."]

    pool, new_p1, new_p2 = resolve_roster_toggle(
        unlock_all=True,
        cur_p1="Alcaraz C.",
        cur_p2="Sinner J.",
        active_pool=active_subset,
        all_pool=all_subset,
    )
    assert pool == all_subset
    assert new_p1 == "Alcaraz C."
    assert new_p2 == "Sinner J."

def test_resolve_roster_toggle_lock_active_replaces_invalid():
    from tk_app import resolve_roster_toggle
    active_subset = ["Alcaraz C.", "Sinner J."]
    all_subset = ["Agassi A.", "Alcaraz C.", "Sampras P.", "Sinner J."]

    pool, new_p1, new_p2 = resolve_roster_toggle(
        unlock_all=False,
        cur_p1="Agassi A.",
        cur_p2="Sampras P.",
        active_pool=active_subset,
        all_pool=all_subset,
    )
    assert pool == active_subset
    assert new_p1 == "Alcaraz C."
    assert new_p2 == "Sinner J."

def test_resolve_roster_toggle_defaults_to_engine_pools():
    from tk_app import resolve_roster_toggle
    pool_all, p1_all, p2_all = resolve_roster_toggle(True, DEFAULT_P1, DEFAULT_P2)
    assert pool_all == all_players
    assert p1_all == DEFAULT_P1
    assert p2_all == DEFAULT_P2

    pool_act, p1_act, p2_act = resolve_roster_toggle(False, DEFAULT_P1, DEFAULT_P2)
    assert pool_act == active_players
    assert p1_act == DEFAULT_P1
    assert p2_act == DEFAULT_P2

def test_format_error_text():
    from tk_app import format_error_text
    assert format_error_text({"error": "Please select both Player 1 and Player 2."}) == "Please select both Player 1 and Player 2."
    assert format_error_text("Player 1 and Player 2 must be different athletes.") == "Player 1 and Player 2 must be different athletes."
    assert format_error_text({}) == ""
    assert format_error_text({"prob_lgb": 0.5}) == ""

def test_compute_display_probabilities():
    from tk_app import compute_display_probabilities
    probs = compute_display_probabilities(0.6123, 0.4567)
    assert probs["lgb_p1"] == pytest.approx(0.6123)
    assert probs["lgb_p2"] == pytest.approx(0.3877)
    assert probs["lgb_p1_pct"] == "61.2%"
    assert probs["lgb_p2_pct"] == "38.8%"
    assert probs["lr_p1"] == pytest.approx(0.4567)
    assert probs["lr_p2"] == pytest.approx(0.5433)
    assert probs["lr_p1_pct"] == "45.7%"
    assert probs["lr_p2_pct"] == "54.3%"

def test_gui_smoke_and_acceptance_criteria():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        from tk_app import App, DEFAULT_P1, DEFAULT_P2
        app = App(root)

        # AC 1: Window starts with Gradio's default players, surface and series
        assert app.p1_var.get() == DEFAULT_P1
        assert app.p2_var.get() == DEFAULT_P2
        assert app.surface_var.get() == "Hard"
        assert app.series_var.get() == "Grand Slam"
        assert app.roster_var.get() is False
        assert len(app.notebook.tabs()) == 1
        assert "Upcoming" in app.notebook.tab(0, "text")

        # AC 2: Swap exchanges players
        app.on_swap()
        assert app.p1_var.get() == DEFAULT_P2
        assert app.p2_var.get() == DEFAULT_P1

        # AC 2: Roster toggle switches active/all-time pools and keeps valid selections
        app.roster_var.set(True)
        app.on_roster_toggle()
        assert list(app.p1_combo["values"]) == all_players
        assert app.p1_var.get() == DEFAULT_P2
        assert app.p2_var.get() == DEFAULT_P1

        app.roster_var.set(False)
        app.on_roster_toggle()
        assert list(app.p1_combo["values"]) == active_players
        assert app.p1_var.get() == DEFAULT_P2
        assert app.p2_var.get() == DEFAULT_P1

        # AC 3: Predict shows both models' probabilities for both players
        app.on_predict()
        assert app.error_var.get() == ""
        assert app.last_prediction is not None
        assert "prob_lgb" in app.last_prediction
        assert "prob_lr" in app.last_prediction
        assert app.prob_display.has_data is True
        assert app.prob_display.prob_lgb == app.last_prediction["prob_lgb"]
        assert app.prob_display.prob_lr == app.last_prediction["prob_lr"]

        # AC 4: Same-player error
        app.p2_var.set(app.p1_var.get())
        app.on_predict()
        assert "different athletes" in app.error_var.get().lower()

        # AC 4: Empty selection error
        app.p1_var.set("")
        app.on_predict()
        assert "select both" in app.error_var.get().lower()

    finally:
        root.destroy()
