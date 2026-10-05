# 01B: Matchup Predictor Serving & Inversion Symmetry (Mode A Backend)

**What to build:** Ingestion and deserialization of `atp_inference_bundle.joblib`, active roster derivation (year >= 2023) vs all-time roster pool, on-the-fly 31-delta feature vector builder `build_match_features(p1, p2, surface, series)`, dual-pass inference through `champion_lgb` and `champion_lr`, numerical inversion symmetry score audit ($|P(P_1) + P(P_2) - 1.0| < 10^{-5}$), non-linear divergence detection ($|P_{\text{LGBM}} - P_{\text{LR}}| > 0.15$), and HTML rendering of the populated diagnostic card with animated probability bars and colored delta tiles.

**Blocked by:** #1 (01A: Matchup Predictor UI & Visual Theme Shell)

**Status:** ready-for-agent

- [x] Load and unpack `atp_inference_bundle.joblib` without upstream notebook execution.
- [x] Implement `build_match_features(p1, p2, surface, series)` and `get_h2h_stats(p1, p2, surface)`.
- [x] Implement dual-pass prediction generating calibrated $P_{\text{LGBM}}$ and $P_{\text{LR}}$.
- [x] Evaluate inversion symmetry $\epsilon_{\text{sym}} = |P(P_1) + P(P_2) - 1.0|$ and display status chip.
- [x] Display non-linear divergence alert banner when $|P_{\text{LGBM}} - P_{\text{LR}}| > 0.15$.
- [x] Render 5 key delta tiles (Rank Δ, Surface WR Δ, Form Div Δ, Fatigue 14d Δ, H2H record) with directional color coding.
- [x] Wire Predict button to execute inference and display populated diagnostic card.
