<h1 align="center">MatchPoint.intelligence</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python: 3.10+"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn: 1.6.1"></a>
  <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen" alt="LightGBM: 4.0+"></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas: 2.0+"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white" alt="NumPy: 1.24+"></a>
</p>

End-to-end machine learning engine and architecture roadmap for professional tennis match outcome forecasting, calibrated probabilistic serving, and live market edge analysis.

---

## 1. Overview & System Topology

Dual-phase architecture combining an empirical ML feature engine with a high-throughput serving pipeline:

```
[ Kaggle Raw Match Data (2000–2026) ]
                 │
                 ▼
[ 7-Phase ETL & Feature Engineering Engine ]
- Zero future-to-past leakage
- 31 Fundamental Delta Predictors (P1 - P2)
- Expanding-Window Validation Splits (2000–2023 Train | 2024–2026 Holdout)
                 │
                 ▼
     [ Dual-Track Model Suite ]
     ├── Champion: LightGBM (Non-linear gradient booster)
     └── Baseline: Regularized Logistic Regression (L2 + RobustScaler)
                 │
                 ▼
[ Serialized Bundle: atp_inference_bundle.joblib ]
                 │
                 ▼ (Roadmap)
[ Real-Time Web Console & Serving Stack ]
- REST & WebSocket Odds Streams (Pinnacle/Betfair)
- Inversion Symmetry & Divergence Guardrails
- Power-Method De-Vigorish & Quarter-Kelly Staking Engine
```

---

## 2. Setup & Execution

### 2.1 Dependencies & Environment

- **Python**: `>= 3.10` (tested on 3.11 / 3.12)
- **Install via `requirements.txt`**:

```bash
pip install -r requirements.txt
```

> **! Note on `scikit-learn`**: Pinned to `>=1.6.0,<1.7.0` (1.6.1) to avoid pickle incompatibility when loading `atp_inference_bundle.joblib`.

### 2.2 Tiered Execution

#### Fast-Track: Offline Evaluation & Inference
Pre-computed features and serialized models are committed to the repository for instant reproduction:

```python
import joblib
import pandas as pd

# Load test features (2024-2026 temporal holdout)
test_df = pd.read_csv("test_features.csv")

# Load model bundle
bundle = joblib.load("atp_inference_bundle.joblib")
champion_lgb = bundle["champion_lgb"]
champion_lr = bundle["champion_lr"]

print("Artifacts loaded successfully.")
```

#### Pop-Out GUI: Interactive Gradio Match Predictor
Launch the standalone web console with dual-model inference, inversion symmetry audit, and historical holdout backtracker:

```bash
python gradio_app.py
```

#### Native Desktop GUI: Tkinter Match Predictor
Launch the standalone high-performance desktop application with dual-model win probabilities, 5-metric key delta tiles, and historical backtracker:

```bash
python tk_app.py
```

> **Note for Linux Users**: Tkinter requires the system-level Tcl/Tk package. If not already installed in your Linux environment, install it via:
> ```bash
> sudo apt install python3-tk
> ```


#### Full-Pipeline: Kaggle ETL & Feature Engineering
To re-run the 7-phase data cleaning, score parsing, rolling statistics computation, and feature selection:

```bash
# Via Jupyter Notebook
jupyter notebook eda_and_feature_engineering.ipynb

# Or headless script execution
python eda_and_feature_engineering.py
```
*Note: Public Kaggle dataset is automatically fetched on first run via `kagglehub`.*

---

## 3. Modeling Benchmarks & 31 Delta Predictors

### 3.1 Model Performance (2024–2026 Holdout Test Set)

Models predict the binary outcome $y \in \{0, 1\}$ ($1$ = Player 1 wins) using strictly fundamental predictors:

| Metric | Baseline: Logistic Regression (L2) | Champion: LightGBM | Market Benchmark (Bookmaker Implied) |
| :--- | :---: | :---: | :---: |
| **Log-Loss** | `0.6382` | **`0.6149`** | `0.5980` |
| **Brier Score** | `0.2241` | **`0.2115`** | `0.2030` |
| **ROC-AUC** | `0.6842` | **`0.7171`** | — |
| **Accuracy** | `62.80%` | **`65.13%`** | `66.40%` |

Detailed derivations, regularization sweeps, and loss curves:
- [MODELING_PLAN.md](MODELING_PLAN.md) — Expanding-window cross-validation and evaluation protocols.
- [docs/logistic_regression/GUIDE.md](docs/logistic_regression/GUIDE.md) — Mathematical formulation, quasi-Newton L-BFGS dynamics, and scaling.

### 3.2 31 Fundamental Delta Predictors ($P_1 - P_2$)

Predictors are computed symmetrically such that swapping player entry preserves inversion symmetry:
- **Ranking & Momentum (2)**: `rank_diff`, `pts_diff`
- **Multi-Horizon Rolling Win Rates (17)**: Short (10 matches), Medium (52 weeks / 365 days), and Career horizons tracking match win rate, tiebreak win rate, comeback rate post-set-1 loss, bagel/breadstick dominance, average games per set, and retirement frequency.
- **Contextual & Fatigue (5)**: `surface_winrate_diff`, `days_since_last_diff`, `matches_14d_diff`, `matches_30d_diff`, `form_divergence_diff`
- **Head-to-Head Encounters (4)**: `h2h_winrate`, `h2h_match_count`, `h2h_surface_winrate`, `h2h_surface_count` (with `0.5` cold-start baselines)
- **Categorical / Tournament Flags (3)**: `surface_encoded`, `series_encoded`, `is_grand_slam`

---

## 4. Repository Structure & Artifacts

```
.
├── eda_and_feature_engineering.ipynb   # 7-phase ETL, rolling metrics, and leakage-free splits
├── eda_and_feature_engineering.py      # Standalone headless pipeline script
├── train_features.csv                  # 31 engineered predictors (2000–2023 training set)
├── test_features.csv                   # 31 engineered predictors (2024–2026 holdout set)
├── atp_inference_bundle.joblib         # Serialized LightGBM, LogReg, and feature encoders
├── CONTEXT.md                          # Domain language and terminology definitions
├── MODELING_PLAN.md                    # Temporal validation design and calibration benchmarks
├── docs/                               # Proof-of-concept UI prototypes & educational guides
│   ├── app_interface/                  # Web-console layout prototype (index.html) & platform specs
│   ├── lgbm/                           # LightGBM champion performance report (index.html)
│   └── logistic_regression/            # Baseline mathematical derivations and learning guides
└── specs/                              # Formal engineering specs for production platform (01–07)
```

> **Note on UI Artifacts**: HTML dashboards under `docs/` serve as interactive proof-of-concept prototypes informing the target full-stack web console.

---

## 5. Web-Console Platform Vision & Roadmap

The project is progressing from offline experimentation to an interactive match prediction and market analytics platform:

- **Dual-Model Inference & Inversion Symmetry**: Real-time evaluation enforcing exact complement probabilities ($|P(P_1) + P(P_2) - 1.0| < 10^{-5}$) with automated non-linear divergence alerts ($|P_{\text{LGBM}} - P_{\text{LR}}| > 0.15$).
- **Match Lifecycle State Machine**: Deterministic progression (`Scheduled` → `LineupsConfirmed` → `PreMatchFeaturesLocked` → `Completed`) with automated feature vector freezing at T-30m.
- **Historical Fixture Replay & Feature Breakdown**: Querying historical ATP matches (2000–2026) with interactive radar/bar comparisons across all 31 delta metrics against actual outcomes.
- **Live Tournament Schedule Dashboard**: Real-time daily order of play with dual probability comparison bars and match cards.
- **Odds De-Vigorish & Quarter-Kelly Staking**: Consensus fair value estimation using Power Method margin removal ($q_1^{1/k} + q_2^{1/k} = 1.0$), model edge calculation ($\Delta = P_{\text{model}} - P_{\text{implied}}$), and fractional Kelly bankroll sizing.
- **Streaming & Drift Operations**: Low-latency WebSocket odds updates and automated Population Stability Index (PSI) drift monitoring.

For complete architectural designs, schemas, and API definitions:
- [PLATFORM_SPECS.md](docs/app_interface/PLATFORM_SPECS.md) — System architecture, database schemas, and state machine.
- [specs/](specs/) — Modular implementation specifications.
