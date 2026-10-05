# 01A: Matchup Predictor UI & Visual Theme Shell (Mode A UI)

**What to build:** Standalone Gradio app root entrypoint (`gradio_app.py`) with MatchPoint.intelligence dark slate/teal visual styling (`#0b0f19` canvas, `#0f172a` card surfaces, `#14b8a6` accents), Plus Jakarta Sans and JetBrains Mono typography, segmented mode switch toggle (`[⚡ Upcoming Match Predictor]` ⟷ `[🔍 Historical Match Backtracker]`), Mode A input controls (Active vs All-Time roster toggle, Player 1 & Player 2 dropdowns, `⇄` swap button, Surface & Series pickers, Predict button), and glassmorphism diagnostic card container markup.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] Standalone `gradio_app.py` created with `demo.launch(inbrowser=True)`.
- [x] MatchPoint.intelligence theme configured via `gr.themes.Soft(primary_hue="teal", neutral_hue="slate")` and custom CSS overrides.
- [x] Segmented mode toggle dynamically toggles visibility between Mode A and Mode B containers.
- [x] Mode A input grid contains Player 1, Player 2, swap button, Surface, Series, and Roster scope toggle.
- [x] Swap button `⇄` swaps selected values between Player 1 and Player 2 dropdowns.
- [x] Placeholder glassmorphism card layout renders in response to Predict action.
