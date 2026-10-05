# 02A: Historical Backtracker UI & Comparison Card (Mode B UI)

**What to build:** Mode B UI container containing split selector dropdown (`Holdout Test (2024–2026)` vs `Full Dataset (2000–2026)`), tournament selector dropdown, matchup selector dropdown, `[🔍 Inspect & Backtrack]` button, and the structured visual layout for the historical match evaluation card (actual score/winner chip, tri-bar comparison for LightGBM, LogReg, and Bookmaker implied probability baseline, and 5 key historical delta tiles).

**Blocked by:** #2 (01B: Matchup Predictor Serving & Inversion Symmetry)

**Status:** ready-for-agent

- [x] Mode B container populated with split, tournament, matchup selectors, and Inspect button.
- [x] Dark theme glassmorphism card layout defined for historical fixture review.
- [x] Visual indicators styled for actual match winner (green for model success, red for upset).
- [x] Tri-bar comparison layout formatted for LightGBM vs LogReg vs Bookmaker implied odds.
