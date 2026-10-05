# 04 — Live Schedule Feed & Pre-Match Prediction Dashboard

**What to build:** View today's active ATP tournament schedule with real-time match cards displaying confirmed lineups, surface/series tags, and automated dual-model win probabilities generated from frozen pre-match feature snapshots.

**Frontier:** No
**Blocked by:** 01 — Dual-Model Prediction & Inversion Symmetry Engine, 02 — Match Lifecycle State Machine & Persistence Schema
**Blocks:** 05 — Odds De-Vigorish, Edge Valuation & Kelly Staking Engine, 06 — Real-Time WebSocket Streaming & Live In-Play Alerting
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Ingestion pipeline pulling upcoming tournament draws and daily schedule into PostgreSQL.
- [ ] Endpoint `GET /api/v1/schedule/live` returning active tournaments, daily order of play, and fixture lifecycle statuses.
- [ ] Automated snapshot freezing at T-30m triggering dual-pass inference to populate `match_predictions`.
- [ ] Responsive dashboard UI showing tournament groupings, fixture cards, dual probability comparison bars, and non-linear divergence alerts ($|P_{\text{LGBM}} - P_{\text{LR}}| > 0.15$).
