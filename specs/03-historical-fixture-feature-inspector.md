# 03 — Historical Fixture Replay & 31-Delta Feature Breakdown Inspector

**What to build:** Query and inspect historical ATP matches (2000–2026) to view actual outcomes alongside dual-model pre-match predictions and interactive breakdowns across all 31 fundamental delta features (rolling win rates, fatigue indices, surface mastery, head-to-head records).

**Frontier:** No
**Blocked by:** 01 — Dual-Model Prediction & Inversion Symmetry Engine
**Blocks:** None
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Endpoint `GET /api/v1/fixtures/historical` supporting filtering by season, tournament, surface, round, and player names.
- [ ] Endpoint `GET /api/v1/fixtures/{fixture_id}/features` returning normalized deltas categorized into ranking, rolling form, fatigue, H2H, and tournament flags.
- [ ] Feature importance attribution showing top LightGBM gain drivers for the selected matchup.
- [ ] UI backtracker card showing side-by-side comparison of actual score vs pre-match probabilities and interactive delta radar/bar chart.
