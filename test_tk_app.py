import subprocess
import sys
import threading
import time
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

def test_resolve_surface_color():
    from tk_app import resolve_surface_color, PALETTE
    assert resolve_surface_color("Hard") == "#38bdf8"
    assert resolve_surface_color("Clay") == "#fb923c"
    assert resolve_surface_color("Grass") == "#4ade80"
    assert resolve_surface_color("Carpet") == "#a78bfa"
    assert resolve_surface_color("Unknown") == PALETTE["accent"]

def test_format_symmetry_text():
    from tk_app import format_symmetry_text
    sym_zero = format_symmetry_text(0.0123, 0.0)
    assert "Δ < 1e-5" in sym_zero
    assert "Raw Δ: 1.2%" in sym_zero

    sym_nonzero = format_symmetry_text(0.0456, 0.00025)
    assert "Δ 2.5e-04" in sym_nonzero
    assert "Raw Δ: 4.6%" in sym_nonzero

def test_resolve_divergence_status():
    from tk_app import resolve_divergence_status
    # Hidden when delta <= 0.15 and not is_divergent
    vis, msg = resolve_divergence_status(False, 0.12)
    assert vis is False
    assert "12.0%" in msg

    vis15, msg15 = resolve_divergence_status(False, 0.15)
    assert vis15 is False

    # Visible when delta > 0.15
    vis_high, msg_high = resolve_divergence_status(False, 0.185)
    assert vis_high is True
    assert "18.5%" in msg_high

    # Visible when is_divergent is True even if delta <= 0.15
    vis_flag, msg_flag = resolve_divergence_status(True, 0.10)
    assert vis_flag is True
    assert "10.0%" in msg_flag

def test_format_key_stats():
    from tk_app import format_key_stats
    stats_p1_fav = {
        "rank_diff": -2.0,
        "surface_winrate_diff": 0.145,
        "form_divergence_diff": 0.35,
        "matches_14d_diff": 1.0,
        "h2h_p1_wins": 5,
        "h2h_p2_wins": 2,
    }
    res_p1 = {item["id"]: item for item in format_key_stats(stats_p1_fav)}
    assert res_p1["rank"]["val"] == "-2"
    assert res_p1["rank"]["color"] == "#10b981"
    assert res_p1["swr"]["val"] == "+14.5%"
    assert res_p1["swr"]["color"] == "#10b981"
    assert res_p1["form"]["val"] == "+0.35"
    assert res_p1["form"]["color"] == "#10b981"
    assert res_p1["fatigue"]["val"] == "+1 m"
    assert res_p1["fatigue"]["color"] == "#e2e8f0"
    assert res_p1["h2h"]["val"] == "5 - 2"
    assert res_p1["h2h"]["color"] == "#10b981"

    stats_p2_fav = {
        "rank_diff": 8.0,
        "surface_winrate_diff": -0.224,
        "form_divergence_diff": -0.40,
        "matches_14d_diff": -2.0,
        "h2h_p1_wins": 1,
        "h2h_p2_wins": 4,
    }
    res_p2 = {item["id"]: item for item in format_key_stats(stats_p2_fav)}
    assert res_p2["rank"]["val"] == "+8"
    assert res_p2["rank"]["color"] == "#ef4444"
    assert res_p2["swr"]["val"] == "-22.4%"
    assert res_p2["swr"]["color"] == "#ef4444"
    assert res_p2["form"]["val"] == "-0.40"
    assert res_p2["form"]["color"] == "#ef4444"
    assert res_p2["fatigue"]["val"] == "-2 m"
    assert res_p2["h2h"]["val"] == "1 - 4"
    assert res_p2["h2h"]["color"] == "#ef4444"

    stats_tied = {"h2h_p1_wins": 3, "h2h_p2_wins": 3}
    res_tied = {item["id"]: item for item in format_key_stats(stats_tied)}
    assert res_tied["h2h"]["val"] == "3 - 3"
    assert res_tied["h2h"]["color"] == "#38bdf8"

def test_resolve_split_change():
    from atp_data import SPLIT_CHOICES
    from tk_app import resolve_split_change
    tourns, first_tourn, matches, first_match_id = resolve_split_change(SPLIT_CHOICES[0])
    assert isinstance(tourns, list) and len(tourns) > 0
    assert first_tourn == tourns[0]
    assert isinstance(matches, list) and len(matches) > 0
    assert matches[0][1] == first_match_id

    # Nonexistent split returns empty lists and strings
    t_empty, ft_empty, m_empty, mid_empty = resolve_split_change("Nonexistent Split")
    assert t_empty == []
    assert ft_empty == ""
    assert m_empty == []
    assert mid_empty == ""

def test_resolve_tournament_change():
    from atp_data import SPLIT_CHOICES, get_tournaments_for_split
    from tk_app import resolve_tournament_change
    tourns = get_tournaments_for_split(SPLIT_CHOICES[0])
    matches, first_match_id = resolve_tournament_change(SPLIT_CHOICES[0], tourns[0])
    assert isinstance(matches, list) and len(matches) > 0
    assert matches[0][1] == first_match_id

    # Empty tournament returns empty list and string
    m_empty, mid_empty = resolve_tournament_change(SPLIT_CHOICES[0], "")
    assert m_empty == []
    assert mid_empty == ""

def test_evaluate_historical_outcome():
    from tk_app import evaluate_historical_outcome
    # 1. P1 favored, P1 wins -> Accurate
    res1 = evaluate_historical_outcome("Alcaraz C.", "Alcaraz C.", 0.68)
    assert res1["correct"] is True
    assert res1["badge_text"] == "PREDICTION ACCURATE"
    assert res1["icon"] == "✓"
    assert res1["badge_border"] == "#10b981"
    assert res1["badge_text_color"] == "#10b981"

    # 2. P1 favored, P2 wins -> Upset
    res2 = evaluate_historical_outcome("Alcaraz C.", "Sinner J.", 0.68)
    assert res2["correct"] is False
    assert res2["badge_text"] == "UPSET / DIVERGENT"
    assert res2["icon"] == "✗"
    assert res2["badge_border"] == "#f43f5e"
    assert res2["badge_text_color"] == "#f43f5e"

    # 3. P2 favored, P2 wins -> Accurate
    res3 = evaluate_historical_outcome("Alcaraz C.", "Sinner J.", 0.32)
    assert res3["correct"] is True
    assert res3["badge_text"] == "PREDICTION ACCURATE"
    assert res3["icon"] == "✓"

    # 4. P2 favored, P1 wins -> Upset
    res4 = evaluate_historical_outcome("Alcaraz C.", "Alcaraz C.", 0.32)
    assert res4["correct"] is False
    assert res4["badge_text"] == "UPSET / DIVERGENT"
    assert res4["icon"] == "✗"

def test_format_bm_odds():
    from tk_app import format_bm_odds
    assert format_bm_odds(None) is None
    res = format_bm_odds(0.625)
    assert res == {"bm_p1_pct": "62.5%", "bm_p2_pct": "37.5%"}
    res2 = format_bm_odds(0.50)
    assert res2 == {"bm_p1_pct": "50.0%", "bm_p2_pct": "50.0%"}

def test_gui_smoke_and_acceptance_criteria():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        from tk_app import App, DEFAULT_P1, DEFAULT_P2, SURFACE_COLORS
        app = App(root)

        # Baseline checks
        assert app.p1_var.get() == DEFAULT_P1
        assert app.p2_var.get() == DEFAULT_P2
        assert app.surface_var.get() == "Hard"
        assert app.series_var.get() == "Grand Slam"
        assert app.roster_var.get() is False
        assert len(app.notebook.tabs()) == 2
        assert "Upcoming" in app.notebook.tab(0, "text")
        assert "Historical" in app.notebook.tab(1, "text")

        # Initial prediction verified on startup
        app.wait_for_prediction()
        assert app.last_prediction is not None
        assert app.prob_display.has_data is True

        # Swap exchanges players
        app.on_swap()
        assert app.p1_var.get() == DEFAULT_P2
        assert app.p2_var.get() == DEFAULT_P1

        # Roster toggle switches active/all-time pools
        app.roster_var.set(True)
        app.on_roster_toggle()
        assert list(app.p1_combo["values"]) == all_players

        app.roster_var.set(False)
        app.on_roster_toggle()
        assert list(app.p1_combo["values"]) == active_players

        # Prediction execution with real engine
        app.on_predict()
        app.wait_for_prediction()
        assert app.error_var.get() == ""
        assert app.last_prediction is not None
        assert "prob_lgb" in app.last_prediction
        assert "prob_lr" in app.last_prediction
        assert app.prob_display.has_data is True

        # AC 1: All 5 key stats tiles populated with orientation vs Player 1
        for key in ["rank", "swr", "form", "fatigue", "h2h"]:
            assert key in app.stats_tiles
            val_txt = app.stats_tiles[key]["val"].cget("text")
            assert len(val_txt) > 0

        # AC 2: Symmetry and raw-asymmetry values displayed in symmetry badge
        sym_txt = app.symmetry_badge.cget("text")
        assert "Symmetry:" in sym_txt
        assert "Raw Δ:" in sym_txt

        # AC 3: Divergence warning alert behavior
        # Test divergent state
        divergent_mock = {
            "p1": "Alcaraz C.",
            "p2": "Sinner J.",
            "surface": "Hard",
            "series": "Grand Slam",
            "prob_lgb": 0.70,
            "prob_lr": 0.50,
            "div_delta": 0.20,
            "is_divergent": True,
            "sym_score": 0.0,
            "raw_asym": 0.02,
            "key_stats": {
                "rank_diff": -1.0,
                "surface_winrate_diff": 0.05,
                "form_divergence_diff": 0.12,
                "matches_14d_diff": 0.0,
                "h2h_p1_wins": 4,
                "h2h_p2_wins": 4,
            },
        }
        app.show_prediction(divergent_mock)
        assert app.is_divergent_visible is True
        assert app.divergence_frame.winfo_manager() != ""
        assert "20.0%" in app.divergence_label.cget("text")

        # Test non-divergent state (delta <= 0.15)
        non_divergent_mock = dict(divergent_mock, div_delta=0.08, is_divergent=False, prob_lr=0.65)
        app.show_prediction(non_divergent_mock)
        assert app.is_divergent_visible is False
        assert app.divergence_frame.winfo_manager() == ""

        # AC 4: Accent colour follows selected surface across all surfaces
        for surf, expected_hex in SURFACE_COLORS.items():
            mock_res = dict(non_divergent_mock, surface=surf)
            app.show_prediction(mock_res)
            assert app.surface_badge.cget("fg") == expected_hex
            assert app.surface_accent_strip.cget("bg") == expected_hex

        # Error path hides diagnostic card elements
        app.show_error("Test Error")
        assert app.divergence_frame.winfo_manager() == ""
        assert app.stats_frame.winfo_manager() == ""
        assert app.surface_accent_strip.winfo_manager() == ""
        assert app.header_frame.winfo_manager() == ""

        # Same-player and empty validation errors
        app.p2_var.set(app.p1_var.get())
        app.on_predict()
        app.wait_for_prediction()
        assert "different athletes" in app.error_var.get().lower()

        app.p1_var.set("")
        app.on_predict()
        app.wait_for_prediction()
        assert "select both" in app.error_var.get().lower()

        # -------------------------------------------------------------------
        # Historical Tab Smoke & Acceptance Criteria
        # -------------------------------------------------------------------
        from atp_data import SPLIT_CHOICES

        # Baseline checks for historical tab
        assert app.split_var.get() == SPLIT_CHOICES[0]
        assert len(app.tourn_combo["values"]) > 0
        assert len(app.match_combo["values"]) > 0

        # AC 1: Split change refreshes tournaments and matches
        app.split_var.set(SPLIT_CHOICES[1])
        app.on_historical_split_change()
        assert len(app.tourn_combo["values"]) > 0
        assert app.tourn_var.get() == app.tourn_combo["values"][0]
        assert len(app.match_combo["values"]) > 0

        # Tournament change refreshes matches
        tourns = list(app.tourn_combo["values"])
        if len(tourns) > 1:
            app.tourn_var.set(tourns[1])
            app.on_historical_tournament_change()
            assert len(app.match_combo["values"]) > 0

        # Reset back to split 0 for predictable testing
        app.split_var.set(SPLIT_CHOICES[0])
        app.on_historical_split_change()

        # AC 2: Inspect shows model probabilities for stored fixture
        app.on_inspect()
        app.wait_for_inspect()
        assert app.historical_error_var.get() == ""
        assert app.last_historical_prediction is not None
        assert "prob_lgb" in app.last_historical_prediction
        assert "prob_lr" in app.last_historical_prediction
        assert app.hist_canvas_bars.has_data is True

        # AC 3: Winner badge reflects LGBM favourite vs actual winner
        assert app.hist_winner_badge_frame.winfo_manager() != ""
        assert "Actual Winner:" in app.hist_winner_text.cget("text")
        status_pill_text = app.hist_status_pill.cget("text")
        assert ("PREDICTION ACCURATE" in status_pill_text or "UPSET / DIVERGENT" in status_pill_text)

        # 5 key stats delta tiles populated in historical tab
        for key in ["rank", "swr", "form", "fatigue", "h2h"]:
            assert key in app.hist_stats_tiles
            assert len(app.hist_stats_tiles[key]["val"].cget("text")) > 0

        # AC 4: Bookmaker probability shown when available, hidden otherwise
        mock_with_bm = dict(app.last_historical_prediction, bm_prob=0.62)
        app.show_historical_prediction(mock_with_bm)
        assert app.hist_bm_frame.winfo_manager() != ""
        assert "62.0%" in app.hist_bm_val_label.cget("text")

        mock_without_bm = dict(app.last_historical_prediction, bm_prob=None)
        app.show_historical_prediction(mock_without_bm)
        assert app.hist_bm_frame.winfo_manager() == ""

        # Winner badge status verification: accurate vs upset
        mock_accurate = dict(
            app.last_historical_prediction,
            p1="Player A",
            actual_winner="Player A",
            prob_lgb=0.65,
            score_str="6-4 6-3"
        )
        app.show_historical_prediction(mock_accurate)
        assert "PREDICTION ACCURATE" in app.hist_status_pill.cget("text")
        assert app.hist_winner_icon.cget("text") == "✓"
        assert app.hist_score_text.winfo_manager() != ""
        assert "6-4 6-3" in app.hist_score_text.cget("text")

        mock_upset = dict(
            app.last_historical_prediction,
            p1="Player A",
            actual_winner="Player B",
            prob_lgb=0.65,
            score_str=None
        )
        app.show_historical_prediction(mock_upset)
        assert "UPSET / DIVERGENT" in app.hist_status_pill.cget("text")
        assert app.hist_winner_icon.cget("text") == "✗"
        assert app.hist_score_text.winfo_manager() == ""

        # Divergence banner in historical tab
        mock_hist_div = dict(app.last_historical_prediction, is_divergent=True, div_delta=0.20)
        app.show_historical_prediction(mock_hist_div)
        assert app.hist_is_divergent_visible is True
        assert app.hist_divergence_frame.winfo_manager() != ""
        assert "20.0%" in app.hist_divergence_label.cget("text")

        mock_hist_nodiv = dict(app.last_historical_prediction, is_divergent=False, div_delta=0.05)
        app.show_historical_prediction(mock_hist_nodiv)
        assert app.hist_is_divergent_visible is False
        assert app.hist_divergence_frame.winfo_manager() == ""

        # AC 5: Empty/invalid selection shows clear message
        app.match_var.set("")
        app.on_inspect()
        app.wait_for_inspect()
        assert "valid tournament and matchup" in app.historical_error_var.get().lower()
        assert app.hist_error_frame.winfo_manager() != ""
        assert app.hist_winner_badge_frame.winfo_manager() == ""
        assert app.hist_canvas_bars.winfo_manager() == ""

        # show_historical_error hides diagnostic card elements
        app.show_historical_error("Manual test error message")
        assert "Manual test error message" in app.historical_error_var.get()
        assert app.hist_error_frame.winfo_manager() != ""
        assert app.hist_stats_frame.winfo_manager() == ""
        assert app.hist_surface_accent_strip.winfo_manager() == ""
        assert app.hist_header_frame.winfo_manager() == ""


    finally:
        root.destroy()


def test_readme_launch_docs():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "python tk_app.py" in content
    assert "python3-tk" in content


def test_initial_prediction_on_startup():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        from tk_app import App, DEFAULT_P1, DEFAULT_P2, DEFAULT_SURFACE, DEFAULT_SERIES
        app = App(root, init_predict=True)
        # Wait for the startup prediction scheduled via root.after(50, ...)
        app.wait_for_prediction()

        assert app.last_prediction is not None
        assert app.last_prediction["p1"] == DEFAULT_P1
        assert app.last_prediction["p2"] == DEFAULT_P2
        assert app.last_prediction["surface"] == DEFAULT_SURFACE
        assert app.last_prediction["series"] == DEFAULT_SERIES
        assert app.prob_display.has_data is True
        assert app.placeholder_frame.winfo_manager() == ""
        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"
    finally:
        root.destroy()


def test_async_predict_buttons_state_and_completion(monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        import atp_service
        from tk_app import App
        app = App(root, init_predict=False)

        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"

        started_evt = threading.Event()
        resume_evt = threading.Event()

        def slow_predict(*args, **kwargs):
            started_evt.set()
            resume_evt.wait(timeout=2.0)
            return {
                "p1": "Alcaraz C.",
                "p2": "Sinner J.",
                "surface": "Hard",
                "series": "Grand Slam",
                "prob_lgb": 0.6,
                "prob_lr": 0.5,
                "key_stats": {},
            }

        monkeypatch.setattr(atp_service, "predict", slow_predict)

        thread = app.on_predict()
        started_evt.wait(timeout=1.0)
        root.update()

        # Both buttons disabled during prediction run
        assert str(app.predict_btn["state"]) == "disabled"
        assert str(app.hist_inspect_btn["state"]) == "disabled"

        # Resume worker
        resume_evt.set()
        app.wait_for_prediction()

        # Both buttons re-enabled after run
        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"
        assert app.last_prediction is not None
    finally:
        root.destroy()


def test_async_inspect_buttons_state_and_completion(monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        import atp_service
        from tk_app import App
        app = App(root, init_predict=False)

        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"

        started_evt = threading.Event()
        resume_evt = threading.Event()

        def slow_inspect(*args, **kwargs):
            started_evt.set()
            resume_evt.wait(timeout=2.0)
            return {
                "p1": "Alcaraz C.",
                "p2": "Sinner J.",
                "surface": "Hard",
                "series": "Grand Slam",
                "prob_lgb": 0.6,
                "prob_lr": 0.5,
                "actual_winner": "Alcaraz C.",
                "key_stats": {},
            }

        monkeypatch.setattr(atp_service, "inspect", slow_inspect)

        thread = app.on_inspect()
        started_evt.wait(timeout=1.0)
        root.update()

        assert str(app.predict_btn["state"]) == "disabled"
        assert str(app.hist_inspect_btn["state"]) == "disabled"

        resume_evt.set()
        app.wait_for_inspect()

        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"
        assert app.last_historical_prediction is not None
    finally:
        root.destroy()


def test_prediction_exception_surfaces_as_banner_without_crash(monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        import atp_service
        from tk_app import App
        app = App(root, init_predict=False)

        def failing_predict(*args, **kwargs):
            raise RuntimeError("Engine model load failure: simulated network timeout")

        monkeypatch.setattr(atp_service, "predict", failing_predict)

        app.on_predict()
        app.wait_for_prediction()

        assert "Engine model load failure: simulated network timeout" in app.error_var.get()
        assert app.error_frame.winfo_manager() != ""
        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"
    finally:
        root.destroy()


def test_inspect_exception_surfaces_as_banner_without_crash(monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available for Tk smoke test")

    try:
        import atp_service
        from tk_app import App
        app = App(root, init_predict=False)

        def failing_inspect(*args, **kwargs):
            raise RuntimeError("Corrupt match data partition")

        monkeypatch.setattr(atp_service, "inspect", failing_inspect)

        app.on_inspect()
        app.wait_for_inspect()

        assert "Corrupt match data partition" in app.historical_error_var.get()
        assert app.hist_error_frame.winfo_manager() != ""
        assert str(app.predict_btn["state"]) == "normal"
        assert str(app.hist_inspect_btn["state"]) == "normal"
    finally:
        root.destroy()
