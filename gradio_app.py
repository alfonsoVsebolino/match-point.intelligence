import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from atp_data import (
    SPLIT_CHOICES,
    get_tournaments_for_split,
    get_matches_for_tournament,
    test_df
)
from atp_engine import (
    active_players,
    all_players,
    DEFAULT_P1,
    DEFAULT_P2,
    predict_matchup,
    inspect_historical_match
)

# ---------------------------------------------------------------------------
# UI Theme & Styling
# ---------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #e2e8f0 !important;
}

.font-mono, pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Primary Brand Buttons */
.btn-primary {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(20, 184, 166, 0.35) !important;
    transition: all 0.2s ease !important;
}
.btn-primary:hover {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
    transform: translateY(-1px) !important;
}

/* Swap Button */
.btn-swap {
    background: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.btn-swap:hover {
    background: #334155 !important;
}

/* Segmented Mode Radio Switch */
.segmented-mode-switch .wrap {
    gap: 8px !important;
    background: #0f172a !important;
    padding: 6px !important;
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#0f172a",
    block_border_color="#1e293b",
    input_background_fill="#0b0f19",
    input_border_color="#334155",
    button_primary_background_fill="#0d9488",
    button_primary_background_fill_hover="#14b8a6",
    button_primary_text_color="#ffffff",
)

default_split = SPLIT_CHOICES[0]
initial_tournaments = get_tournaments_for_split(default_split)
default_tournament = initial_tournaments[0] if initial_tournaments else ""
initial_matches = get_matches_for_tournament(default_split, default_tournament)
default_match_id = initial_matches[0][1] if initial_matches else None

with gr.Blocks(title="MatchPoint.intelligence | ATP Predictor", theme=theme, css=custom_css) as demo:
    # Top Navbar & Telemetry
    with gr.Row():
        with gr.Column(scale=8):
            gr.HTML("""
            <div style="display:flex; align-items:center; gap:12px; padding:10px 0;">
                <div style="width:42px; height:42px; border-radius:10px; background:linear-gradient(135deg, #14b8a6, #3b82f6); display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 0 15px rgba(20, 184, 166, 0.4);">
                    🎾
                </div>
                <div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:20px; font-weight:800; color:#ffffff; tracking-tight:tight;">MatchPoint.intelligence</span>
                        <span style="font-size:10px; font-family:'JetBrains Mono', monospace; text-transform:uppercase; background:rgba(20, 184, 166, 0.15); border:1px solid rgba(20, 184, 166, 0.3); color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;">
                            Gradio Serving
                        </span>
                    </div>
                    <div style="font-size:12px; color:#94a3b8; font-family:'JetBrains Mono', monospace;">
                        Dual-Track ML Engine (LightGBM Champion vs LogReg Baseline) • 31 Fundamental Features
                    </div>
                </div>
            </div>
            """)
        with gr.Column(scale=4):
            gr.HTML("""
            <div style="display:flex; justify-content:flex-end; align-items:center; gap:10px; height:100%; font-family:'JetBrains Mono', monospace; font-size:11px;">
                <div style="background:#0f172a; padding:6px 12px; border-radius:6px; border:1px solid #1e293b; display:flex; align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 8px #10b981;"></span>
                    <span style="color:#94a3b8;">Bundle:</span>
                    <span style="color:#e2e8f0; font-weight:600;">Dual Models Loaded</span>
                </div>
                <div style="background:#0f172a; padding:6px 12px; border-radius:6px; border:1px solid #1e293b; display:flex; align-items:center; gap:6px;">
                    <span style="width:8px; height:8px; border-radius:50%; background:#38bdf8;"></span>
                    <span style="color:#94a3b8;">H2H History:</span>
                    <span style="color:#e2e8f0; font-weight:600;">2000–2026 Ready</span>
                </div>
            </div>
            """)

    gr.HTML("<div style='height:1px; background:#1e293b; margin:8px 0 16px 0;'></div>")

    # Mode Selector Toggle (Segmented Radio)
    with gr.Row():
        mode_radio = gr.Radio(
            choices=[
                "⚡ Upcoming Match Predictor",
                "🔍 Historical Match Backtracker"
            ],
            value="⚡ Upcoming Match Predictor",
            show_label=False,
            container=False,
            elem_classes=["segmented-mode-switch"]
        )

    # =========================================================================
    # Mode A: Upcoming Match Predictor
    # =========================================================================
    with gr.Column(visible=True) as mode_a_container:
        with gr.Row():
            roster_scope_cb = gr.Checkbox(
                value=False,
                label="Unlock All-Time Historical Roster (2000–2026)",
                info="Default is active modern roster (2023–2026)"
            )

        with gr.Row():
            p1_dropdown = gr.Dropdown(
                choices=active_players,
                value=DEFAULT_P1,
                label="Player 1",
                filterable=True,
                scale=5
            )
            swap_btn = gr.Button(
                "⇄ Swap",
                elem_classes=["btn-swap"],
                scale=1,
                min_width=70
            )
            p2_dropdown = gr.Dropdown(
                choices=active_players,
                value=DEFAULT_P2,
                label="Player 2",
                filterable=True,
                scale=5
            )

        with gr.Row():
            surface_dropdown = gr.Dropdown(
                choices=["Hard", "Clay", "Grass", "Carpet"],
                value="Hard",
                label="Court Surface",
                scale=4
            )
            series_dropdown = gr.Dropdown(
                choices=["Grand Slam", "Masters 1000", "ATP500", "ATP250"],
                value="Grand Slam",
                label="Tournament Series",
                scale=4
            )
            predict_btn = gr.Button(
                "⚡ Predict Matchup",
                elem_classes=["btn-primary"],
                scale=4
            )

        mode_a_output = gr.HTML(
            value=predict_matchup(DEFAULT_P1, DEFAULT_P2, "Hard", "Grand Slam"),
            label="Inversion Symmetry & Model Diagnostics"
        )

    # =========================================================================
    # Mode B: Historical Match Backtracker
    # =========================================================================
    with gr.Column(visible=False) as mode_b_container:
        with gr.Row():
            split_dropdown = gr.Dropdown(
                choices=SPLIT_CHOICES,
                value=default_split,
                label="Dataset Split",
                scale=4
            )
            tourn_dropdown = gr.Dropdown(
                choices=initial_tournaments,
                value=default_tournament,
                label="Tournament",
                filterable=True,
                scale=8
            )

        with gr.Row():
            match_dropdown = gr.Dropdown(
                choices=initial_matches,
                value=default_match_id,
                label="Select Matchup",
                filterable=True,
                scale=9
            )
            inspect_btn = gr.Button(
                "🔍 Inspect & Backtrack",
                elem_classes=["btn-primary"],
                scale=3
            )

        mode_b_output = gr.HTML(
            value=inspect_historical_match(default_split, default_tournament, default_match_id),
            label="Historical Match Evaluation"
        )

    # =========================================================================
    # Event Wiring & Callbacks
    # =========================================================================
    # 1. Mode Switch Toggle
    def on_mode_change(selected_mode):
        is_mode_a = (selected_mode == "⚡ Upcoming Match Predictor")
        return gr.update(visible=is_mode_a), gr.update(visible=not is_mode_a)

    mode_radio.change(
        fn=on_mode_change,
        inputs=[mode_radio],
        outputs=[mode_a_container, mode_b_container]
    )

    # 2. Swap Players
    def on_swap_players(cur_p1, cur_p2):
        return cur_p2, cur_p1

    swap_btn.click(
        fn=on_swap_players,
        inputs=[p1_dropdown, p2_dropdown],
        outputs=[p1_dropdown, p2_dropdown]
    )

    # 3. Roster Scope Toggle
    def on_roster_toggle(unlock_all, cur_p1, cur_p2):
        pool = all_players if unlock_all else active_players
        new_p1 = cur_p1 if cur_p1 in pool else pool[0]
        new_p2 = cur_p2 if cur_p2 in pool else (pool[1] if len(pool) > 1 else pool[0])
        return gr.update(choices=pool, value=new_p1), gr.update(choices=pool, value=new_p2)

    roster_scope_cb.change(
        fn=on_roster_toggle,
        inputs=[roster_scope_cb, p1_dropdown, p2_dropdown],
        outputs=[p1_dropdown, p2_dropdown]
    )

    # 4. Predict Upcoming Matchup
    predict_btn.click(
        fn=predict_matchup,
        inputs=[p1_dropdown, p2_dropdown, surface_dropdown, series_dropdown],
        outputs=[mode_a_output]
    )

    # 5. Cascading Split Change (Mode B)
    def on_split_change(new_split):
        tourns = get_tournaments_for_split(new_split)
        first_tourn = tourns[0] if tourns else ""
        matches = get_matches_for_tournament(new_split, first_tourn)
        first_match = matches[0][1] if matches else None
        return (
            gr.update(choices=tourns, value=first_tourn),
            gr.update(choices=matches, value=first_match)
        )

    split_dropdown.change(
        fn=on_split_change,
        inputs=[split_dropdown],
        outputs=[tourn_dropdown, match_dropdown]
    )

    # 6. Cascading Tournament Change (Mode B)
    def on_tournament_change(cur_split, tourn_name):
        matches = get_matches_for_tournament(cur_split, tourn_name)
        new_val = matches[0][1] if matches else None
        return gr.update(choices=matches, value=new_val)

    tourn_dropdown.change(
        fn=on_tournament_change,
        inputs=[split_dropdown, tourn_dropdown],
        outputs=[match_dropdown]
    )

    # 7. Inspect Historical Fixture
    inspect_btn.click(
        fn=inspect_historical_match,
        inputs=[split_dropdown, tourn_dropdown, match_dropdown],
        outputs=[mode_b_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_error=True)
