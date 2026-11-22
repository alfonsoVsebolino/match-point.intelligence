# Project: MatchPoint.intelligence — Neon ESPN Broadcast Web Platform

## Architecture
- **Stateless Cloud Run Web Service (`server.py`)**: Single FastAPI + Uvicorn process listening on `0.0.0.0:${PORT:-8080}`, serving both `/api/v1/*` JSON endpoints and static frontend assets (`web/` mounted at `/` and `/static`), with `GET /healthz` for Cloud Run health checks.
- **Thread-Safe ML Inference & Explainability Core (`atp_engine.py`, `atp_service.py`, `atp_data.py`)**:
  - Pre-trained `atp_inference_bundle.joblib` (LightGBM Champion + Regularized Logistic Regression Baseline, 31 delta predictors).
  - Dual-pass inversion symmetrization (`|P(P1) + P(P2) - 1.0| < 1e-5`) and non-linear divergence check (`|P_LGBM - P_LR| > 0.15`).
  - In-memory H2H pair index built at startup/`reload_state()` for `<15ms` inference latency.
  - Native LightGBM TreeSHAP (`champion_lgb.booster_.predict(X, pred_contrib=True)`) for bidirectional "Tug-of-War" feature contributions + LogReg linear contribution comparison.
  - Deterministic Arcade player OVR (`0–99`), Playstyle Trait Chips, and 6-Axis Radar profile derivation from `latest_player_state` + `player_surface_wr`.
  - Power-Method de-vigging (`q1^(1/k) + q2^(1/k) = 1.0`), Edge % (`P_LGBM - P_implied`), and Quarter-Kelly (`0.25 * f*`, capped at `2.5%`).
- **Offloaded "Hydrogen" Worker (`atp_daily_sync.py` + `.github/workflows/daily_atp_sync.yml`)**:
  - Routine heavy tasks (daily Kaggle dataset sync, 68k+ rolling feature recomputation, `player_state_cache.joblib`, batch ESPN schedule predictions) remain decoupled from web request latency and run out-of-band via GitHub Actions / Hydrogen worker / Cloud Run Job.
  - Cloud Run web app handles missing/unseeded `data/` gracefully with clean empty-state telemetry and an optional lightweight ESPN scoreboard refresh (`POST /api/v1/schedule/refresh` without Kaggle pull).
- **Dedicated Frontend (`web/` — ESPN Broadcast Scoreboard + Dark Neon Arena)**:
  - Pure HTML5 + Vanilla CSS + Vanilla ES6 JS + custom neon SVG/Canvas charts (zero external CDN dependency for airgapped/E2E reliability).
  - Deep obsidian canvas (`#060913`), ESPN-style top live-match ticker strip, liberal neon accents (`#00f5ff` cyan, `#ccff00` tennis-volt lime, `#ff2bd6` hot magenta), unified 3-mode workspace (Daily Schedule Board, H2H Matchup Arena, Historical Backtracker), and Full Arcade Broadcast Visualization Suite.

## Feature Inventory
Every Step 0 Survey feature & locked-in user requirement is assigned to a milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Cloud Run Healthcheck & Stateless Web Entry | FastAPI `server.py` on `0.0.0.0:${PORT:-8080}`, `GET /healthz`, static `web/` serving, graceful missing-`data/` handling | M1, M3 | `ORIGINAL_REQUEST.md` |
| F2 | Thread-Safe Dual-Model Symmetrized Inference + H2H Index | Remove `atp_service.py` monkeypatching, add in-memory H2H lookup (`<15ms` warm), preserve `atp_service` & `atp_engine` HTML backward compatibility | M1 | `explorer_survey_backend`, `specs/01` |
| F3 | LightGBM TreeSHAP Tug-of-War & Full 31-Feature Payload | Compute `pred_contrib=True` bidirectional contributions + all 31 delta features + category grouping for `/api/v1/predict` and historical inspect | M1, M2 | `ORIGINAL_REQUEST.md`, `explorer_survey_backend` |
| F4 | Arcade Player Profile Engine | Compute deterministic OVR (`0–99`), Playstyle Trait Chips, and 6-Axis Radar vectors (`Short Form`, `Med Form`, `Surface Mastery`, `Rank Power`, `Clutch / TB`, `Stamina / Freshness`) for P1 & P2 | M1, M2 | `ORIGINAL_REQUEST.md` |
| F5 | Power-Method De-Vig, Edge & Quarter-Kelly Engine | Solve `q1^(1/k) + q2^(1/k) = 1.0`, compute Edge % and Quarter-Kelly (`0.25 * f*`, cap `2.5%`) for live/optional odds and `atp_daily_sync.py` | M1, M2, M3 | `spec_miner_survey_specs`, `specs/05` |
| F6 | Historical Match Backtracker (2000–2026) | Cascading Split → Tournament → Matchup endpoints & UI with actual winner/score, `PREDICTION ACCURATE` vs `UPSET`, bookmaker implied prob, and full Arcade breakdown | M1, M2 | `explorer_survey_ui`, `specs/03` |
| F7 | Daily/Live Schedule Board & ESPN Ticker Strip | `GET /api/v1/schedule/live` + `POST /api/v1/schedule/refresh` (skip Kaggle), ESPN top ticker strip, tournament scoreboard cards, series & `>=3%` edge filters | M1, M2 | `ORIGINAL_REQUEST.md`, `specs/04` |
| F8 | ESPN Broadcast Scoreboard + Dark Neon Arena & Full Arcade Suite | Dedicated frontend (`web/`): Neon FIFA/TopSpin Trading Cards, 6-Axis SVG Radar Duel, TreeSHAP Tug-of-War, Tale-of-the-Tape & Divergence Gauge, Collapsible 31-Feature Matrix | M2 | `ORIGINAL_REQUEST.md` |
| F9 | Cloud Run Dockerization & Hydrogen Worker Offload | Production `Dockerfile`, `.dockerignore`, `requirements.txt`, `atp_daily_sync.py` Power-Method alignment, `.github/workflows/daily_atp_sync.yml` + Cloud Run/Hydrogen docs in `README.md` | M3 | `ORIGINAL_REQUEST.md` |
| F10 | Opaque-Box E2E Test Suite & Tier 5 Hardening | Independent requirement-driven E2E tests (Tiers 1–4) + `TEST_INFRA.md` + `TEST_READY.md`, plus Tier 5 adversarial hardening & whole-project Forensic Audit | E2E_ORCH, M4 | `ORIGINAL_REQUEST.md` |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Backend Inference, TreeSHAP & FastAPI Server | `atp_engine.py`, `atp_service.py`, `server.py`, `requirements.txt`, `test_server.py`, `test_atp_service.py` | none | DONE (`f724b63d18319103253b6a7ba2d4c4e9`; 67 passed, p95<5ms, Auditor CLEAN) |
| E2E_ORCH | E2E Testing Track (Opaque-Box Tiers 1–4) | `tests_e2e/`, `TEST_INFRA.md`, `TEST_READY.md` | none (runs parallel against Interface Contracts) | DONE (`9d66c982ee00b2dab3e7061b6cb3e078`; 112 tests, Auditor CLEAN) |
| M2 | Dedicated Frontend: Neon ESPN & Arcade UI | `web/index.html`, `web/styles.css`, `web/app.js`, `web/charts.js`, `web/arcade_cards.js` (dedicated frontend specialist subagent) | none (parallel against Interface Contracts) | DONE (`f89ee03b6bc9445a9d196119d2cb2967`; 5/5 verifiers PASS, Auditor PASS) |
| M3 | Cloud Run Container & Hydrogen Worker Offload | `Dockerfile`, `.dockerignore`, `atp_daily_sync.py` (de-vig), `test_daily_sync.py`, `.github/workflows/daily_atp_sync.yml`, `README.md` | none (parallel; M4 runs integrated container/E2E checks) | DONE (`4eec425df7adfb0bdd3dd701abf8befd`; 20/20 `test_daily_sync.py`, Auditor CLEAN) |
| M4 | Final Milestone: E2E Pass (Tiers 1–4) & Tier 5 Hardening | Pass 100% of `TEST_READY.md` suite (including live server/browser & container checks) + Tier 5 Challenger hardening + whole-project Audit | M1, M2, M3, E2E_ORCH | IN_PROGRESS (`26688d0ccdad650c42d72dbcd3d9878`) |

## Code Layout
- `server.py` — FastAPI application entrypoint (`app`), API routes (`/healthz`, `/api/v1/*`), static file serving (`web/`)
- `atp_engine.py` — Bundle loader, H2H index, dual-pass symmetric inference, TreeSHAP contributions, Arcade player profile & radar metrics, Power-Method de-vig/Kelly, structured historical inspector
- `atp_service.py` — Thread-safe Python facade (`predict`, `inspect`, `get_daily_schedule`, `sync_daily_schedule`) preserving backward compatibility with `tk_app.py` and tests
- `atp_data.py` — CSV/JSON dataset access and split/tournament/matchup helpers
- `atp_daily_sync.py` — Offloaded Hydrogen daily Kaggle sync & ESPN schedule batch predictor
- `atp_cards.py`, `gradio_app.py`, `tk_app.py` — Existing UIs (must remain intact and passing existing tests)
- `web/` — Dedicated frontend static bundle:
  - `web/index.html` — Semantic ESPN Broadcast Arena shell + Ticker + 3 Modes + Arcade Suite containers
  - `web/styles.css` — Obsidian `#060913` + Neon Cyan `#00f5ff` / Volt `#ccff00` / Magenta `#ff2bd6` design tokens & responsive layout
  - `web/app.js` — State store, `/api/v1/*` client, mode routing, ticker interactions, filter/edge logic, collapsible 31-feature matrix
  - `web/arcade_cards.js` — Neon FIFA/TopSpin Trading Cards + Trait Chips + Tale-of-the-Tape renderer
  - `web/charts.js` — Custom SVG 6-Axis Neon Spider/Radar Duel Chart, Bidirectional TreeSHAP Tug-of-War, Dual-Model Divergence Gauge
- `Dockerfile`, `.dockerignore`, `requirements.txt` — Google Cloud Run container packaging
- `tests_e2e/` — Requirement-driven opaque-box E2E tests (`TEST_INFRA.md`, `TEST_READY.md`)

## Interface Contracts

### Backward Compatibility (Existing Python Modules & Tests)
- `atp_service.predict(p1, p2, surface, series)` MUST remain thread-safe and return all legacy keys (`p1`, `p2`, `surface`, `series`, `prob_lgb`, `prob_lr`, `key_stats`, `sym_score`, `is_divergent`, `div_delta`, `is_historical`) or `{"error": str}` without monkeypatching `atp_engine.render_diagnostic_card`.
- `atp_service.inspect(split, tournament, match_idx)` MUST remain thread-safe and return all legacy keys (`p1`, `p2`, `surface`, `series`, `prob_lgb`, `prob_lr`, `key_stats`, `sym_score`, `is_divergent`, `div_delta`, `bm_prob`, `actual_winner`, `score_str`, `date_str`, `round_name`, `tournament`, `is_historical`) or `{"error": str}`.
- `atp_engine.predict_matchup(...)` and `atp_engine.inspect_historical_match(...)` MUST continue returning HTML strings for `gradio_app.py`.

### HTTP / REST API (`server.py` on `0.0.0.0:${PORT:-8080}`)

1. **`GET /healthz`** → `200 OK`
   ```json
   {
     "status": "ok",
     "models_loaded": true,
     "players_total": 1823,
     "players_active": 488,
     "sync_status": { "status": "static", "last_synced_at": null, "latest_match_date": "2026-07-26" },
     "schedule_fixtures_count": 0
   }
   ```

2. **`GET /api/v1/meta`** → `200 OK`
   - Query params: optional `all_time: bool = false`
   - Response:
   ```json
   {
     "active_players": ["Alcaraz C.", "Sinner J."],
     "all_players": ["..."],
     "surfaces": ["Hard", "Clay", "Grass", "Carpet"],
     "series": ["Grand Slam", "Masters 1000", "ATP500", "ATP250", "Masters Cup", "Masters", "International Gold", "International"],
     "splits": ["Holdout Test (2024–2026)", "Full Dataset (2000–2026)"],
     "default_p1": "Alcaraz C.",
     "default_p2": "Sinner J.",
     "sync_status": { "...": "..." },
     "model_metrics": { "lgbm_logloss": 0.6149, "lgbm_brier": 0.2139, "lgbm_auc": 0.7171, "trees": 184, "features_count": 31 }
   }
   ```

3. **`POST /api/v1/predict`** → `200 OK` (or `400`/`422` on invalid input: missing/identical players, invalid surface/series)
   - Request JSON:
   ```json
   {
     "p1": "Alcaraz C.",
     "p2": "Sinner J.",
     "surface": "Hard",
     "series": "Grand Slam",
     "odd_1": 1.85,
     "odd_2": 2.05
   }
   ```
   (`odd_1`, `odd_2` optional nullable floats `> 1.0`)
   - Response JSON:
   ```json
   {
     "p1": "Alcaraz C.",
     "p2": "Sinner J.",
     "surface": "Hard",
     "series": "Grand Slam",
     "prob_lgb": 0.56,
     "prob_lr": 0.54,
     "sym_score": 0.0,
     "raw_asym": 0.002,
     "is_divergent": false,
     "div_delta": 0.02,
     "p1_known": true,
     "p2_known": true,
     "key_stats": { "rank_diff": 0.0, "surface_winrate_diff": 0.0, "form_divergence_diff": 0.0, "matches_14d_diff": 0, "h2h_p1_wins": 0, "h2h_p2_wins": 0 },
     "player_cards": {
       "p1": { "name": "Alcaraz C.", "ovr": 95, "rank": 1, "pts": 11000, "surface_winrate": 0.82, "short_winrate": 0.9, "career_winrate": 0.81, "days_since_last": 5, "matches_14d": 3, "traits": ["STREAK FIRE", "CLUTCH TIEBREAK"], "radar": { "short_form": 90, "med_form": 85, "surface_mastery": 82, "rank_power": 98, "clutch_tb": 78, "stamina_freshness": 84 } },
       "p2": { "...": "..." }
     },
     "treeshap": {
       "bias": 0.0,
       "p1_total_contrib": 0.24,
       "contributions": [
         { "feature": "pts_diff", "label": "ATP Ranking Points Δ", "category": "Ranking", "raw_delta": 1200.0, "shap": 0.14, "favors": "p1" }
       ]
     },
     "features_31": [
       { "name": "rank_diff", "label": "ATP Rank Δ", "category": "Ranking", "value": -1.0, "shap": 0.08, "lr_coef": -0.32, "favors": "p1" }
     ],
     "edge": {
       "odd_1": 1.85,
       "odd_2": 2.05,
       "implied_prob_p1": 0.525,
       "implied_prob_p2": 0.475,
       "overround": 1.028,
       "power_k": 0.97,
       "edge_p1": 0.035,
       "edge_p2": -0.035,
       "signal": "VALUE_P1",
       "kelly_p1_pct": 1.03,
       "kelly_p2_pct": 0.0
     }
   }
   ```
   *(When `odd_1` or `odd_2` is omitted/invalid, `"edge": null`.)*

4. **`GET /api/v1/fixtures/historical`** → `200 OK`
   - Query params: `split: str = "Holdout Test (2024–2026)"`, `tournament: str | None = None`
   - Response: `{ "split": "...", "tournaments": ["..."], "selected_tournament": "...", "matches": [{"label": "...", "match_idx": "123"}] }`

5. **`GET /api/v1/fixtures/historical/inspect`** → `200 OK` (or `404`/`400` if invalid)
   - Query params: `split: str`, `tournament: str`, `match_idx: str`
   - Response: Same rich payload as `POST /api/v1/predict` (`player_cards`, `treeshap`, `features_31`, `key_stats`, `prob_lgb`, `prob_lr`, `is_divergent`, `div_delta`, `sym_score`), PLUS `is_historical: true`, `bm_prob`, `actual_winner`, `score_str`, `date_str`, `round_name`, `tournament`, `prediction_accurate`.

6. **`GET /api/v1/schedule/live`** → `200 OK`
   - Response:
   ```json
   {
     "generated_at": null,
     "target_date": null,
     "source": "none",
     "sync_status": { "...": "..." },
     "kpis": { "total_fixtures": 0, "live_count": 0, "value_edges_count": 0, "divergent_count": 0 },
     "fixtures": []
   }
   ```
   *(Each fixture in `fixtures` includes `fixture_id`, `tournament`, `round_name`, `surface`, `series`, `status`, `status_detail`, `p1`, `p2`, `prob_lgb`, `prob_lr`, `is_divergent`, `div_delta`, `odd_1`, `odd_2`, `bm_prob`, `edge_lgb_p1`, `kelly_p1_pct`, `snapshot_locked`, `cold_start`.)*

7. **`POST /api/v1/schedule/refresh`** → `200 OK` (soft fallback if ESPN unreachable: returns current/empty schedule with `refresh_error` field instead of 500)
   - Runs lightweight ESPN scoreboard schedule fetch + batch inference (`skip_kaggle=True`, `allow_espn=True`) and returns the updated `GET /api/v1/schedule/live` payload + `refreshed: bool` + `refresh_error: str | null`.
