# 05 — Odds De-Vigorish, Edge Valuation & Kelly Staking Engine

**What to build:** Ingest bookmaker odds, remove overround using Power Method de-vig to extract consensus fair probabilities, calculate model edge ($\Delta = P_{\text{LGBM}} - P_{\text{implied}}$), and determine Quarter-Kelly staking fractions ($f^*$) with confidence gating.

**Frontier:** No
**Blocked by:** 02 — Match Lifecycle State Machine & Persistence Schema, 04 — Live Schedule Feed & Pre-Match Prediction Dashboard
**Blocks:** 06 — Real-Time WebSocket Streaming & Live In-Play Alerting
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Ingest decimal odds $(O_1, O_2)$ and compute consensus fair probability using the Power Method ($q_1^{1/k} + q_2^{1/k} = 1.0$).
- [ ] Compute value edge $\text{Edge}(P_1) = P_{\text{LGBM}}(P_1) - P_{\text{implied}, 1}$ and classify: Significant Value ($\ge +3\%$), Market Alignment ($-2\%$ to $+2\%$), Negative Value ($\le -3\%$).
- [ ] Compute Quarter-Kelly staking fraction $f^* = \max(0, \frac{b \cdot P - (1-P)}{b}) \times 0.25$ capped at 2.5% bankroll risk.
- [ ] Expose edge and staking metrics via `GET /api/v1/fixtures/{fixture_id}/edge`.
- [ ] UI displays live odds, consensus implied probabilities, colored edge badges (+EV green / neutral gray / -EV red), and Kelly bet size sizing.
