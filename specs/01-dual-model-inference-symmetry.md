# 01 — Dual-Model Prediction & Inversion Symmetry Engine

**What to build:** Select any two ATP players, surface, and series to compute real-time calibrated win probabilities from LightGBM champion and Regularized Logistic Regression baseline. Guarantee mathematical inversion symmetry ($P(P_1) = [M(X) + (1 - M(-X))]/2$) and flag non-linear divergence ($|P_{\text{LGBM}} - P_{\text{LR}}| > 0.15$).

**Frontier:** Yes — can start immediately.
**Blocked by:** None
**Blocks:** 03 — Historical Fixture Replay & 31-Delta Feature Breakdown Inspector, 04 — Live Schedule Feed & Pre-Match Prediction Dashboard
**Status:** ready-for-agent

## Acceptance Criteria
- [ ] Export and load serialized model artifact bundle (`atp_inference_bundle.joblib`) containing `champion_lgb`, `champion_lr`, encoders, and player snapshots.
- [ ] Calculate 31 fundamental delta features ($P_1 - P_2$) on-the-fly for any valid player pair and surface/series context.
- [ ] Serve predictions via REST endpoint (`POST /api/v1/predict`) returning LGBM win probability, LogReg win probability, divergence flag, and symmetry verification score.
- [ ] Enforce inversion symmetry: swapping $P_1 \leftrightarrow P_2$ yields exact complement probability ($|P(P_1) + P(P_2) - 1.0| < 10^{-5}$).
- [ ] Interactive UI allowing player dropdown selection, surface/series picker, dual probability comparison bars, and symmetry swap test button.
