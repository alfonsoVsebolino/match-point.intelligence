# 02B: Historical Backtracker Backend & Data Filtering (Mode B Backend)

**What to build:** Ingestion pipeline loading `test_features.csv`, reactive cascading dropdown update functions (split change updates tournaments; tournament change updates matchups sorted chronologically), fixture feature extraction from precomputed row index, bookmaker implied probability derivation from `implied_prob_diff` ($P_{\text{implied}} = (implied\_prob\_diff + 1.0) / 2.0$), outcome calibration audit against recorded `Winner`, and wiring the Inspect button to render the populated historical card.

**Blocked by:** #3 (02A: Historical Backtracker UI & Comparison Card)

**Status:** ready-for-agent

- [x] Load `test_features.csv` into memory upon app startup.
- [x] Implement reactive cascading callbacks updating tournament and matchup lists dynamically.
- [x] Ingest precomputed 31-delta feature vector for selected historical fixture row.
- [x] Run dual-model inference on historical feature vector.
- [x] Compute bookmaker implied win probability from `implied_prob_diff`.
- [x] Compare champion model prediction ($P > 0.5$) with recorded `Winner` to set winner badge color.
- [x] Render populated historical card on Inspect button click.
