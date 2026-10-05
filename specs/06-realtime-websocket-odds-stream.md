# 06 — Real-Time WebSocket Streaming & Live In-Play Alerting

**What to build:** Stream live match score updates, odds movements, and real-time edge recalculations to client dashboards via persistent WebSockets without full-page reloads.

**Frontier:** No
**Blocked by:** 04 — Live Schedule Feed & Pre-Match Prediction Dashboard, 05 — Odds De-Vigorish, Edge Valuation & Kelly Staking Engine
**Blocks:** 07 — Calibration Audit & Population Stability Drift Monitoring
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] WebSocket endpoint `ws://api/v1/stream/predictions` broadcasting structured JSON events (`ODDS_UPDATE`, `STATE_CHANGE`, `SCORE_UPDATE`).
- [ ] Background worker recalculating Power Method de-vig and Kelly stakes on odds ticks and publishing to Redis channel.
- [ ] Frontend WebSocket client with automatic reconnection and live visual flash animations on value edge updates.
- [ ] End-to-end event latency from odds ingest to client DOM update $< 500\,\text{ms}$.
