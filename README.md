<h1 align="center">MatchPoint.intelligence</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white" alt="Python: 3.11+"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn: 1.6.1"></a>
  <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen" alt="LightGBM: 4.0+"></a>
  <a href="https://cloud.google.com/run"><img src="https://img.shields.io/badge/Google%20Cloud%20Run-Ready-4285F4?logo=googlecloud&logoColor=white" alt="Google Cloud Run Ready"></a>
</p>

End-to-end ATP tennis match forecasting platform featuring a **Neon ESPN Broadcast Web App** (`FastAPI` + `Uvicorn` + Full Arcade Visualization Suite), a **stateless Google Cloud Run** container runtime, and a decoupled **"Hydrogen" routine-task offload worker** for automated daily Kaggle sync and schedule prediction.

---

## 1. Overview & System Topology

MatchPoint.intelligence splits real-time serving from heavy batch data engineering:

```
┌────────────────────────────────────────────────────────────────────────────┐
│            OFFLOADED "HYDROGEN" WORKER (GitHub Actions / Cloud Run Job)    │
│  pipeline/atp_daily_sync.py + .github/workflows/daily_atp_sync.yml         │
│  • Scheduled twice daily at 06:00 & 14:00 UTC                              │
│  • Pulls dissfya/atp-tennis-2000-2023daily-pull via kagglehub              │
│  • Recomputes leakage-free rolling features & data/player_state_cache      │
│  • Ingests ESPN ATP schedule & outputs Power-Method de-vig + Kelly JSON/CSV│
└────────────────────────────────────┬───────────────────────────────────────┘
                                     │ Refreshed state / fixtures
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│          STATELESS GOOGLE CLOUD RUN WEB SERVICE (server.py :8080)          │
│  • Serves web/ Neon ESPN Broadcast Arena & /api/v1/* JSON API              │
│  • Dual-pass inversion-symmetrized LightGBM + LogReg inference (<15ms H2H) │
│  • LightGBM TreeSHAP Tug-of-War, Arcade OVR Cards, 6-Axis Radar & Edge/Kelly│
│  • Readiness probe: GET /healthz                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Neon ESPN Broadcast Web App

The primary web interface (`web/`) pairs an ESPN-inspired dark obsidian scoreboard (`#060913`) and live ticker strip with liberal electric neon accents (`#00f5ff` cyan, `#ccff00` tennis-volt lime, `#ff2bd6` hot magenta) and the **Full Arcade Broadcast Visualization Suite**:

- **Neon FIFA / TopSpin Player Trading Cards**: Deterministic `0–99` OVR rating, ATP rank/points, win rates, and playstyle trait chips (`TOP 10 ELITE`, `STREAK FIRE`, `SURFACE SPECIALIST`, `CLUTCH TIEBREAK`, `COMEBACK KING`, `BAGEL HUNTER`, `IRONMAN`, `FRESH LEGS`, `COLD START`).
- **6-Axis Neon Spider / Radar Duel**: Head-to-head comparison across `Short Form`, `Med Form`, `Surface Mastery`, `Rank Power`, `Clutch / TB`, and `Stamina / Freshness`.
- **LightGBM TreeSHAP Tug-of-War**: Bidirectional `pred_contrib=True` feature contributions showing exactly which predictors push win probability toward Player 1 vs Player 2.
- **Dual-Model Divergence Gauge & Tale-of-the-Tape**: Real-time LightGBM vs Regularized Logistic Regression comparison with automatic divergence alerts (`|P_LGBM - P_LR| > 0.15`) and inversion symmetry auditing.
- **Collapsible 31-Feature Matrix**: Complete breakdown of all 31 delta predictors (`P1 - P2`) grouped by category with SHAP and LogReg coefficient alignment.
- **Power-Method De-Vig & Quarter-Kelly Engine**: Solves $q_1^{1/k} + q_2^{1/k} = 1.0$ via `scipy.optimize.brentq` when decimal odds (`odd_1`, `odd_2`) are supplied, reporting market edge ($\Delta = P_{\text{LGBM}} - P_{\text{implied}}$), signal (`VALUE_P1`, `VALUE_P2`, `ALIGNMENT`, `NEGATIVE`), and Quarter-Kelly bankroll sizing ($0.25 \cdot f^*$, strictly capped at `2.5%`).

### Launch Locally

```bash
# Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# Start the FastAPI + Uvicorn broadcast server
uvicorn src.server:app --host 0.0.0.0 --port 8080
# Or:
python src/server.py
```

Open `http://127.0.0.1:8080` in your browser.

> **Note on `scikit-learn`**: `requirements.txt` pins `scikit-learn>=1.6.0,<1.7.0` (`1.6.1`) for clean deserialization of `models/atp_inference_bundle.joblib`.

### HTTP / REST API Endpoints (`src/server.py`)

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Neon ESPN Broadcast Web App (`web/index.html`) |
| `GET` | `/healthz` | Cloud Run readiness & liveness check (`models_loaded`, player counts, sync status) |
| `GET` | `/api/v1/meta` | Active/all player pools, valid surfaces/series, dataset splits, and model metrics |
| `POST` | `/api/v1/predict` | Symmetrized dual-model prediction, Arcade cards, TreeSHAP, 31 features, and optional Power-Method Edge/Kelly (`odd_1`, `odd_2`) |
| `GET` | `/api/v1/fixtures/historical` | Cascading tournament and match list for 2000–2026 historical replay |
| `GET` | `/api/v1/fixtures/historical/inspect` | Full match backtracker payload with actual winner, score, bookmaker prob, and Arcade breakdown |
| `GET` | `/api/v1/schedule/live` | Latest normalized daily schedule fixtures and KPIs (`total_fixtures`, `live_count`, `value_edges_count`, `divergent_count`) |
| `POST` | `/api/v1/schedule/refresh` | Lightweight ESPN scoreboard fetch + batch inference (`skip_kaggle=True`), with soft fallback on upstream errors |

---

## 3. Stateless Google Cloud Run Deployment

The production container (`Dockerfile` + `.dockerignore`) is designed for **Google Cloud Run**:

- **Base image**: `python:3.11-slim` with `libgomp1` (for LightGBM) and `ca-certificates`.
- **Stateless runtime**: Runs as non-root user `appuser` with writable `/app/data/predictions_history` so optional runtime schedule refreshes work cleanly without root privileges, while handling unseeded `data/` directories gracefully.
- **Dynamic `$PORT` binding**: Executes `uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8080}` (default `8080`).
- **Built-in Healthcheck**: Python `urllib.request` probe targeting `http://127.0.0.1:${PORT:-8080}/healthz` (no `curl` needed).

### Local Docker Build & Run

```bash
docker build -t matchpoint-intelligence:latest .
docker run --rm -p 8080:8080 -e PORT=8080 matchpoint-intelligence:latest
curl http://127.0.0.1:8080/healthz
```

### Deploy to Google Cloud Run

```bash
# Build & deploy directly from source using Dockerfile
gcloud run deploy matchpoint-intelligence \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi
```

---

## 4. Decoupled "Hydrogen" Routine Worker Architecture

Heavy batch workflows are intentionally decoupled from the Cloud Run request path and executed out-of-band by the **Hydrogen Worker** (`pipeline/atp_daily_sync.py` + `.github/workflows/daily_atp_sync.yml`, or an equivalent Cloud Run Job):

1. **Kaggle Daily Pull**: Downloads the unversioned `dissfya/atp-tennis-2000-2023daily-pull` release via `kagglehub` (skipping feature recompute automatically if SHA-256 is unchanged).
2. **Rolling Feature & State Cache Refresh**: Recomputes leakage-free rolling features (`data/test_features.csv`, `data/match_scores.csv`) and updates `data/player_state_cache.joblib` while keeping trained model weights (`models/atp_inference_bundle.joblib`) fixed.
3. **Batch ESPN Schedule Predictions + Power-Method Edge/Kelly**: Fetches published ATP singles fixtures (ESPN public scoreboard API or local CSV/JSON drop-in), enforces T-30m frozen pre-match snapshots, computes Power-Method de-vigged implied probabilities and Quarter-Kelly stakes, and writes `data/daily_predictions_latest.json`, `data/daily_predictions_latest.csv`, and `data/predictions_history/predictions_YYYY-MM-DD.csv`.

### CLI Usage (`pipeline/atp_daily_sync.py`)

```bash
# Run full daily Kaggle sync + published schedule prediction
python pipeline/atp_daily_sync.py run-daily

# Run schedule prediction only (skip Kaggle download, matching Cloud Run lightweight refresh)
python pipeline/atp_daily_sync.py run-daily --no-kaggle

# Sync Kaggle results & refresh player state cache only
python pipeline/atp_daily_sync.py sync

# Predict a custom CSV/JSON schedule file without external API calls
python pipeline/atp_daily_sync.py predict-schedule --schedule-file data/upcoming_schedule.csv --no-espn
```

### Automated GitHub Actions Schedule

[.github/workflows/daily_atp_sync.yml](.github/workflows/daily_atp_sync.yml) runs automatically twice daily (`06:00` and `14:00` UTC) and supports manual `workflow_dispatch` inputs:
- `target_date`: Optional `YYYY-MM-DD` override.
- `force_recompute`: Force feature recomputation even when the raw Kaggle CSV hash is unchanged.
- `skip_kaggle`: Pass `--no-kaggle` to refresh schedule predictions from cached state only.
- **Optional Cloud Run Hook**: Set repository variable `vars.CLOUD_RUN_DEPLOY_HOOK` (and optional secret `CLOUD_RUN_HOOK_TOKEN`) to trigger a Cloud Run schedule refresh or redeploy after new predictions are committed.

---

## 5. Alternative Local Consoles & ETL

### Gradio & Tkinter Consoles
```bash
# Interactive Gradio web prototype
python -m app.gradio_app

# Native desktop Tkinter app (Linux requires: sudo apt install python3-tk)
python -m app.tk_app
```

### Kaggle ETL & Feature Engineering Notebook
To inspect or re-run the 7-phase data cleaning, score parsing, rolling statistics computation, and feature selection:
```bash
jupyter notebook notebooks/eda_and_feature_engineering.ipynb
```

---

## 6. Modeling Benchmarks & 31 Delta Predictors

### Model Performance (2024–2026 Holdout Test Set)

Models predict the binary outcome $y \in \{0, 1\}$ ($1$ = Player 1 wins) using strictly fundamental predictors:

| Metric | Baseline: Logistic Regression (L2) | Champion: LightGBM | Market Benchmark (Bookmaker Implied) |
| :--- | :---: | :---: | :---: |
| **Log-Loss** | `0.6382` | **`0.6149`** | `0.5980` |
| **Brier Score** | `0.2241` | **`0.2139`** | `0.2030` |
| **ROC-AUC** | `0.6842` | **`0.7171`** | — |
| **Accuracy** | `62.80%` | **`65.13%`** | `66.40%` |

### 31 Fundamental Delta Predictors ($P_1 - P_2$)

- **Ranking & Momentum (2)**: `rank_diff`, `pts_diff`
- **Multi-Horizon Rolling Win Rates (17)**: Short (10 matches), Medium (365 days), and Career horizons tracking match win rate, tiebreak win rate, comeback rate post-set-1 loss, bagel/breadstick dominance, average games per set, and retirement frequency.
- **Contextual & Fatigue (5)**: `surface_winrate_diff`, `days_since_last_diff`, `matches_14d_diff`, `matches_30d_diff`, `form_divergence_diff`
- **Head-to-Head Encounters (4)**: `h2h_winrate`, `h2h_match_count`, `h2h_surface_winrate`, `h2h_surface_count` (`0.5` cold-start baseline)
- **Match Context (3)**: `surface_encoded`, `series_encoded`, `is_grand_slam`

Detailed derivations and protocols:
- [MODELING_PLAN.md](MODELING_PLAN.md) — Expanding-window cross-validation and calibration benchmarks.
- [docs/app_interface/PLATFORM_SPECS.md](docs/app_interface/PLATFORM_SPECS.md) — Platform specifications, Power-Method de-vigging, and Quarter-Kelly staking math.
- [docs/logistic_regression/GUIDE.md](docs/logistic_regression/GUIDE.md) — Regularized Logistic Regression mathematical formulation.

---

## 7. Repository Structure

```
.
├── src/                                  # Cloud Run backend & inference core
│   ├── __init__.py                       # Package marker
│   ├── server.py                         # FastAPI + Uvicorn Cloud Run web server (/healthz, /api/v1/*)
│   ├── atp_engine.py                     # Symmetrized inference, H2H index, TreeSHAP, Arcade cards, Power-Method & Kelly
│   ├── atp_service.py                    # Thread-safe Python facade for web, desktop, and tests
│   └── atp_cards.py                      # HTML diagnostic card renderer for Gradio fallback
├── app/                                  # Standalone offline UI applications
│   ├── __init__.py                       # Package marker
│   ├── gradio_app.py                     # Legacy standalone Gradio console
│   └── tk_app.py                         # Native Tkinter desktop GUI
├── web/                                  # Dedicated Neon ESPN Broadcast & Arcade UI static bundle
├── Dockerfile                            # Stateless Google Cloud Run container (python:3.11-slim)
├── .dockerignore                         # Excludes notebooks, dev tests, docs, offline apps, and raw Kaggle CSVs
├── requirements.txt                      # Runtime & serving dependencies
├── pipeline/                             # Data pipeline, feature engineering, dataset access, and daily sync package
│   ├── __init__.py                       # Package marker
│   ├── atp_data.py                       # Dataset splits, sync telemetry, and schedule JSON loader
│   ├── atp_features.py                   # Leakage-free 7-phase rolling feature & score parser
│   ├── atp_daily_sync.py                 # Hydrogen worker: Kaggle sync, ESPN ingestion, Power-Method & Kelly batch predictions
│   └── sync_data.py                      # Kaggle ETL & data synchronization helper
├── models/                               # Versioned trained ML model artifacts
│   └── atp_inference_bundle.joblib       # Serialized LightGBM, LogReg, and feature encoders
├── data/                                 # Feature datasets, player state cache, and daily schedule predictions
│   ├── train_features.csv                # 31 engineered predictors (2000–2023 training set)
│   ├── test_features.csv                 # 31 engineered predictors (2024–2026 holdout set)
│   └── match_scores.csv                  # Historical match score strings for replay
├── notebooks/                            # Exploratory analysis & daily loader Jupyter notebooks
│   ├── eda_and_feature_engineering.ipynb # 7-phase ETL, rolling metrics, and leakage-free splits notebook
│   └── load.ipynb                        # Daily Kaggle pull & schedule prediction loader notebook
├── .github/workflows/daily_atp_sync.yml  # Hydrogen automated twice-daily GitHub Actions offload worker
├── pytest.ini                            # Pytest configuration (`pythonpath = .`, `testpaths = tests`)
├── tests/                                # Organized test suite
│   ├── __init__.py                       # Test suite package marker
│   ├── unit/                             # Unit & API integration tests
│   │   ├── test_server.py                # FastAPI endpoint & contract tests
│   │   ├── test_atp_service.py           # Thread-safe inference & TreeSHAP unit tests
│   │   ├── test_daily_sync.py            # Kaggle sync, ESPN parser, Power-Method de-vig & Quarter-Kelly tests
│   │   └── test_pipeline.py              # Pipeline sanity test
│   ├── ui/                               # Desktop & legacy UI regression tests
│   │   ├── test_gradio_app.py            # Gradio app regression tests
│   │   └── test_tk_app.py                # Tkinter desktop app tests
│   └── e2e/                              # Opaque-box E2E verification suite (Tiers 1–4)
├── docs/                                 # Prototype UI specs & modeling guides
└── specs/                                # Modular platform engineering specifications (01–07)
```
