# 02 — Match Lifecycle State Machine & Persistence Schema

**What to build:** Deterministic match lifecycle state machine and PostgreSQL/TimescaleDB schema that tracks fixtures through formal states (`Scheduled` → `LineupsConfirmed` → `PreMatchFeaturesLocked` → `Completed`/`Retired`/`Cancelled`) with frozen 31-feature snapshots at T-30m and Redis cache storage.

**Frontier:** Yes — can start immediately.
**Blocked by:** None
**Blocks:** 04 — Live Schedule Feed & Pre-Match Prediction Dashboard, 05 — Odds De-Vigorish, Edge Valuation & Kelly Staking Engine
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Database schema migrations for `tournaments`, `fixtures`, `feature_snapshots`, and `match_predictions`.
- [ ] Implement lifecycle state transitions: `Scheduled` → `LineupsConfirmed` → `PreMatchFeaturesLocked` → `LiveInPlay` → `Completed`/`Retired`/`Cancelled`.
- [ ] Invalidation rules: discard pre-match walkovers from rolling windows; record in-play retirements in retirement frequency counters without skewing game totals.
- [ ] Redis cache configured for active match state (`fixture:{id}:live`) and pre-calculated feature vectors (`fixture:{id}:features`).
- [ ] Integration tests verify state machine transition legality and feature freezing at T-30m prior to match start.
