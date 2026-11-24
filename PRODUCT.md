# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **ML & Data-Science Peers (Showcase Audience):** Evaluating model rigor, dual-model calibration (`LightGBM` vs `Logistic Regression`), TreeSHAP log-odds explainability, leakage-free feature engineering, and out-of-sample 2000–2026 backtesting.
- **Quantitative Tennis Bettors & Edge Seekers:** Scanning daily ATP tour slates for `+EV` spots, Power-Method de-vigged fair odds, model divergence signals (`>15%`), and disciplined Quarter-Kelly stake sizing (`2.5%` cap).

## Product Purpose

`matchpoint.ml` is an explainable ATP tennis match forecasting and quantitative edge web application. It transforms 2000–2026 match history, daily tour schedules, and dual-model ML inference into transparent win probabilities, per-match feature attributions, and calibrated staking telemetry.

## Positioning

Unlike opaque sportsbook picks or static Jupyter notebooks, `matchpoint.ml` pairs `<15ms` dual-pass inversion-symmetrized inference (`|P(P1) + P(P2) - 1.0| < 1e-5`) across a non-linear LightGBM champion and a regularized Logistic Regression baseline with native TreeSHAP attributions, automatic divergence flagging (`>15%`), Power-Method de-vigging, and leak-free historical match replay.

## Operating Context

- **Runtime & Deployment:** Stateless FastAPI + Uvicorn service ([src/server.py](src/server.py), [Dockerfile](Dockerfile)) designed for Google Cloud Run, decoupled from an out-of-band daily Kaggle/ESPN sync worker ([pipeline/atp_daily_sync.py](pipeline/atp_daily_sync.py)).
- **Domain Terminology:** Governed by [GLOSSARY.md](GLOSSARY.md) (Calendar-Pinned Matchup, Fantasy Matchup, Published Fixture, Snapshot Freeze, Schedule Refresh Status, Synced Player State, Historical Backtracker Split, Tournament Season) and [docs/adr/0001-calendar-pinned-vs-fantasy-target-date.md](docs/adr/0001-calendar-pinned-vs-fantasy-target-date.md).
- **Workflow Entry Points:**
  1. Daily Schedule Board (`#slate`) — scan published tour fixtures, value edges, and divergence alerts.
  2. Match Dossier (`#match/<id>`) — focused inspection of a scheduled fixture.
  3. H2H Matchup Arena (`#arena`) — custom and All-Time Legends (`2000–2026`) matchup simulation with optional decimal odds.
  4. Historical Match Backtracker (`#backtracker`) — chronological audit (`Holdout Test 2024–2026` and `Full Dataset 2000–2026`) against recorded outcomes and closing market probabilities.

## Capabilities and Constraints

- **Preserve Core ML & Quantitative Math:** Keep all 31 delta predictors, dual-pass inversion symmetrization, TreeSHAP tug-of-war attributions, deterministic OVR/trait/6-axis radar profiles, Power-Method de-vigging ($q_1^{1/k} + q_2^{1/k} = 1.0$), and Quarter-Kelly sizing (capped at `2.5%`).
- **Preserve All 3 Platform Modes & API Contracts:** Keep Daily Slate, H2H Arena, and Historical Backtracker functional against `/healthz`, `/api/v1/meta`, `/api/v1/predict`, `/api/v1/fixtures/historical`, `/api/v1/fixtures/historical/inspect`, `/api/v1/schedule/live`, and `/api/v1/schedule/refresh`, plus Python compatibility in `src/atp_service.py` and `src/atp_engine.py`.
- **External Fonts / CDNs Permitted:** The legacy zero-CDN constraint in `web/styles.css` and `PROJECT.md` is superseded; external font/asset CDNs (e.g., Google Fonts) are allowed (update `tests/e2e/cdp_runner.mjs` external-request checks accordingly when adopting).
- **Missing Asset Notice:** `Sogea-Regular.woff2` referenced in `web/tokens.css` is not currently present in the repository.

## Brand Commitments

- **Canonical Product Name:** `matchpoint.ml` (superseding older `MatchPoint.intelligence` and `baseline.ml` references in docs/tokens).
- **User-Pinned Visual System:** Must strictly adhere to `web/tokens.css` and `.agents/rules/design-systems.md` (Risograph Option A palette, 3-tier typography, and domain invariants).

## Evidence on Hand

- **Trained Models & Metrics:** `models/atp_inference_bundle.joblib` (Holdout evaluation reported in `src/server.py`: AUC `0.7171`, LogLoss `0.6149`, Brier `0.2139`, `184` trees, `31` features).
- **Data & Verification:** 2000–2026 ATP match and feature CSVs (`data/`), FastAPI server (`src/server.py`), unit/UI/E2E test suites (`tests/`), and static visual evaluation prototypes (`design-plans/visual-eval/`).
- **Explicit Absences (Do Not Fabricate):** No live wagering PnL track record, user testimonials, or `Sogea-Regular.woff2` font binary currently exists on disk.

## Product Principles

1. **Explainability Alongside Prediction:** Every probability must remain inspectable through TreeSHAP drivers, baseline vs. champion divergence, and raw delta vectors.
2. **Progressive Disclosure Over Data Walls:** Surface the match verdict, market edge, divergence flag, and key deltas first; tuck high-density scouting, radar, and 31-feature tables into on-demand views.
3. **Symmetry and Calibration Integrity:** Never alter or approximate the symmetric dual-pass probabilities, de-vig solver, or Kelly cap for visual convenience.
4. **Honest Fallback States:** Empty or unseeded schedules and cold-start players must communicate status clearly rather than failing silently or presenting broken metrics.

## Accessibility & Inclusion

- Meet WCAG AA text contrast across all surfaces (specifically `--accent-text` `#C43269` for inline accent text on cream backgrounds).
- Support full keyboard navigation and clear focus states across mode tabs, filters, collapsible sidebars, and interactive chart readouts, without relying on color alone to convey favored/underdog or `+EV`/`-EV` status.
