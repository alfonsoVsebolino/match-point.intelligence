# 07 — Calibration Audit & Population Stability Drift Monitoring

**What to build:** Post-match evaluation service auditing dual-model calibration (Brier score & Log-Loss vs bookmaker benchmark) and flagging continuous feature distribution drift using Population Stability Index (PSI).

**Frontier:** No
**Blocked by:** 06 — Real-Time WebSocket Streaming & Live In-Play Alerting
**Blocks:** None
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Post-match scoring worker computing rolling 30-day Brier score and Log-Loss against market benchmarks upon fixture completion.
- [ ] Weekly PSI and Kolmogorov-Smirnov test service evaluating drift across the 28 continuous delta features against baseline distributions.
- [ ] Drift alert system flagging features with $\text{PSI} > 0.25$ for retraining review.
- [ ] Operator audit view rendering 10-bin calibration reliability curves and drift alert logs.
