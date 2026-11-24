# Risograph Unified Dossier Implementation Plan (`matchpoint.ml`)

> **Execution Mode:** Self-contained implementation specification for an autonomous engineering subagent with zero prior conversation context. Read every referenced governing file before modifying code, and preserve all ML, quantitative math, and FastAPI backend contracts verbatim.

---

## 1. Header & Baseline State

| Field | Value |
|---|---|
| **Pre-Capture Git Baseline** | `43408ce` (`new-look` branch — `add: design tokens, instructions for frontend pass`) |
| **Canonical Product Name** | `matchpoint.ml` (supersedes legacy `MatchPoint.intelligence` and `baseline.ml` names in UI copy) |
| **Governing Product Spec** | [`PRODUCT.md`](../PRODUCT.md) |
| **Canonical Design Tokens** | [`web/tokens.css`](../web/tokens.css) (reconciled note on `@src/styles/tokens.css` below) |
| **UI Invariants & Rules** | [`.agents/rules/design-systems.md`](../.agents/rules/design-systems.md), [`.agents/skills/risograph-ui/SKILL.md`](../.agents/skills/risograph-ui/SKILL.md) |
| **Approved Interactive Reference Comp** | [`design-plans/visual-eval/unified-dossier.html`](./visual-eval/unified-dossier.html) & [`design-plans/visual-eval/eval-risograph.css`](./visual-eval/eval-risograph.css) |
| **Target Production Files (To Be Updated in Implementation Phase)** | `web/index.html`, `web/styles.css`, `web/app.js`, `web/arcade_cards.js`, `web/charts.js`, `tests/e2e/cdp_runner.mjs`, `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier4_e2e_scenarios.py` |
| **Backend Contract Files (Read-Only / Preserve Verbatim)** | [`src/server.py`](../src/server.py), [`src/atp_service.py`](../src/atp_service.py), [`src/atp_engine.py`](../src/atp_engine.py) |

### Scope Boundaries
- **Preserve Core ML & Quantitative Math:** All 31 delta predictors (`features_31`), dual-pass inversion symmetrization (`|P(P1) + P(P2) - 1.0| < 1e-5`), TreeSHAP tug-of-war attributions (`treeshap`), deterministic OVR/trait/6-axis radar profiles (`player_cards`), Power-Method de-vigging ($q_1^{1/k} + q_2^{1/k} = 1.0$), and Quarter-Kelly sizing (capped at `2.5%`).
- **Preserve All FastAPI Routes & Schemas:** `/`, `/healthz`, `/api/v1/meta`, `/api/v1/predict`, `/api/v1/fixtures/historical`, `/api/v1/fixtures/historical/inspect`, `/api/v1/schedule/live`, `/api/v1/schedule/refresh` in `src/server.py` (no FastAPI path changes required; SPA routing uses URL `#hash` fragments).

---

## 2. Design & Token Migration Spec

### 2.1 Token Wiring & Path Reconciliation (`web/tokens.css` vs `@src/styles/tokens.css`)
- **Path Reconciliation:** `.agents/AGENTS.md`, `.agents/rules/design-systems.md`, `.agents/skills/risograph-ui/SKILL.md`, and line 1 of `web/tokens.css` reference `@src/styles/tokens.css`, which does **not** exist in this repository (`src/` is Python backend code). The canonical token file on disk is **`web/tokens.css`**, served by `src/server.py` (`app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")`) at `/static/tokens.css`.
- **Root CSS Import Order in `web/index.html`:** Load `/static/tokens.css` **before** `/static/styles.css`:
  ```html
  <link rel="stylesheet" href="/static/tokens.css">
  <link rel="stylesheet" href="/static/styles.css">
  ```
- **Legacy `:root` Removal:** Strip the entire dark-neon `:root` block from `web/styles.css` (`--bg-canvas: #060913`, `--bg-panel: #0c1222`, `--neon-cyan: #00f5ff`, `--neon-magenta: #ff2bd6`, `--neon-volt: #ccff00`, `--neon-amber: #f59e0b`, `--glow-cyan`, `--glow-magenta`, `--glow-volt`) so `web/tokens.css` is the single source of truth.

### 2.2 60 / 30 / 10 Risograph Option A Color & Surface Migration
Replace all dark-neon CSS variables, `#FFFFFF` / `rgba(255, 255, 255, ...)` literals, blurred SaaS drop shadows (`box-shadow: 0 12px 34px rgba(...)`, `text-shadow`, `backdrop-filter: blur(...)`), and hardcoded neon SVG hexes (`#00f5ff`, `#ff2bd6`, `#ccff00`, `#f59e0b`, `#ffffff`, `feDropShadow` filters `#neonCyanGlow` / `#neonMagentaGlow`) across `web/styles.css`, `web/charts.js`, and `web/arcade_cards.js`:

| Budget / Role | Canonical Token (`web/tokens.css`) | Hex / Value | Usage Mandate |
|---|---|---|---|
| **60% Cream Paper** | `--bg-canvas` | `#F8DBC1` | Page backdrop, hero anchor clean-paper firewall |
| | `--bg-surface` | `#FCE7D4` | Elevated cards, print plates, button fills on cream |
| | `--bg-recessed` | `#EFCCA9` | Probability tracks, inputs, select wells, delta pills, table headers |
| | `--border-subtle` | `#E2BE9C` | Hairline row dividers, recessed borders |
| **30% Riso Teal** | `--ink-primary` | `#1D646C` | Primary ink, `1.5px` card borders, court lines, net cord, card shadows |
| | `--ink-deep` | `#13454B` | High-density tabular figures, card titles, primary headings |
| | `--ink-muted` | `#3E7D84` | Axis ticks, timestamps, secondary labels, **underdog** prob bars & SHAP fills |
| | `--ink-wash` | `rgba(29, 100, 108, 0.10)` | Neutral / consensus badge backgrounds |
| **<=10% Fluorescent Pink** | `--accent-signal` | `#EA4F88` | Primary CTAs, active filter/nav pills, **favored player** prob bar, `+EV` badges, court shot-bounce hotspot, net center strap |
| | `--accent-text` | `#C43269` | WCAG AA inline pink text on `#F8DBC1` or `#FCE7D4` |
| | `--accent-wash` | `rgba(234, 79, 136, 0.14)` | Active toggle tint, divergent >15% badge wash |
| **Domain Semantics** | `--surface-hard` | `#246B9E` | Hard court badge & court schematic stroke |
| | `--surface-clay` | `#B85335` | Clay court badge & court schematic stroke |
| | `--surface-grass` | `#2B7A4B` | Grass court badge & court schematic stroke |
| | `--state-fade` | `#B54434` | Negative EV / market fade / historical upset (never `#FF0000`) |
| **Texture & Shadow** | `--mesh-print-plate` | Radial pink/teal + cream linear gradient | Contained strictly inside featured Print Plate verdict/slate cards |
| | `--shadow-riso-card` | `4px 4px 0px var(--ink-primary)` | Hard offset Risograph print shadow on primary cards (no blurred drop shadows) |

### 2.3 Tactile Craft, Dividers & Single-Favored-Pink Invariants
1. **Root Paper-Grain Overlay (`web/index.html` + `web/styles.css`):** Add an inline SVG `<svg class="paper-grain-overlay" aria-hidden="true">` inside `<body>` with `<filter id="risoGrainUnified"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch" /></filter><rect width="100%" height="100%" filter="url(#risoGrainUnified)" />` styled at `position: fixed; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 999; opacity: 0.04; mix-blend-mode: multiply;` (within the `3%–5%` rule).
2. **Diamond-Mesh Tennis Net Dividers:** Separate major sections (e.g., Verdict Print Plate -> Progressive Disclosure Tabs) with a diamond-mesh tennis net SVG divider (`height="22"`, `--ink-primary` cord + `0.45` opacity diamond `<pattern>` with unique per-mount ID `netPat_<uid>`, plus `--accent-signal` center strap).
3. **Single-Favored-Pink Probability Bar Rule:**
   - In both `web/arcade_cards.js` (`renderScheduleCard`, `renderModelProbBars`, `renderSplitH2HBanner`) and `web/charts.js` (`renderRadarChart`, `renderTreeShapTugOfWar`), NEVER color both P1 and P2 in bright accent colors or hardcode cyan/magenta.
   - The **favored player** (`prob_lgb >= 0.5` for Champion; or player pulled positively in favored-oriented bars) uses `--accent-signal` (`#EA4F88`) bar fill and `--accent-text` (`#C43269`) inline text; the **underdog** uses `--ink-muted` (`#3E7D84`) bar fill and muted text.
   - In `web/charts.js`:
     - **`renderRadarChart`:** Use `#1D646C` rings/spokes (`stroke-opacity="0.24"` / `"0.28"`), favored/P1 polygon in `rgba(234, 79, 136, 0.20)` with `#EA4F88` stroke, P2 polygon in `rgba(62, 125, 132, 0.18)` with `#3E7D84` dashed stroke (`stroke-dasharray="4 3"`), zero `<feDropShadow>` neon glow filters, and vertex/label hotspots highlighting active state in `--accent-text` (`#C43269`).
     - **`renderDivergenceGauge`:** Replace `#f59e0b`, `#00f5ff`, `#ccff00`, `#ffffff`, `#94a3b8` with `--accent-signal` (`#EA4F88`) when `isDivergent` (`>15%`), `--ink-primary` (`#1D646C`) when consensus, `--bg-recessed` / `rgba(29, 100, 108, 0.22)` track, `--ink-deep` (`#13454B`) readout text, and `--ink-muted` (`#3E7D84`) caption text.
     - **`renderTreeShapTugOfWar`:** Center-anchored diverging bars where favored-direction pull uses `--accent-signal` (`#EA4F88`) and opposing pull uses `--ink-muted` (`#3E7D84`). Keep `data-testid="treeshap-panel"` and `#treeshapBars`.

### 2.4 Three-Tier Typography & Numeric Discipline
- **`--font-display` (`Sogea` / `Space Grotesk` fallback):**
  - Strictly `< 10%` of page text; ONLY for the `matchpoint.ml` wordmark and 2–4 word hero/section anchors (`Today's court slate.`, `H2H Arena.`, `Historical backtracker.`, `<P1> vs <P2>.`).
  - **Hard floor:** `>= 32px` (`2rem`) computed `font-size`.
  - **Clean Paper Firewall:** Sentence case or lowercase, flat `--ink-primary` (`#1D646C`) with at most one focal word/period in `--accent-signal` (`#EA4F88`), zero gradients/outlines/text-shadows, placed exclusively on flat `--bg-canvas` (`#F8DBC1`) with `>= 1.5x` cap-height vertical breathing room.
  - **Forbidden:** Never use `--font-display` inside match cards, navigation links, buttons, badges, or data tables.
- **`--font-sans` (`Plus Jakarta Sans`):** `12px–16px` for player names, tournament metadata, filter pills, tabs, SHAP feature labels, and body copy.
- **`.metric-tabular` / `--font-mono` (`JetBrains Mono` + `font-variant-numeric: tabular-nums; letter-spacing: -0.02em;`):** Mandatory on every win probability, decimal odd, `+EV`/`-EV` edge, Brier/LogLoss/AUC metric, Kelly stake, OVR, rank/pts, 5-key-delta tile value, TreeSHAP score, and LR coefficient.

### 2.5 `[PENDING MAIN-SESSION FONT FACTS & CRITIQUE]`
> **[PENDING MAIN-SESSION FONT FACTS & CRITIQUE]**
> - **Confirmed Policy (`PRODUCT.md`):** External font/asset CDNs (Google Fonts: `Plus Jakarta Sans:wght@400;500;600;700`, `JetBrains Mono:wght@500;700`, `Space Grotesk` fallback) are permitted; `Sogea-Regular.woff2` referenced in `web/tokens.css` is not currently present in the repository.
> - **Pending Main-Session Patch:** Exact `Sogea-Regular.woff2` binary availability/hosting path (e.g. local `/static/fonts/` vs CDN vs `Space Grotesk` promotion), offline/airgapped E2E Chromium font-loading behavior, and any `/impeccable critique` revisions on `design-plans/visual-eval/unified-dossier.html` will be patched into this section prior to implementation handoff.

---

## 3. Information Architecture, Hash Routing & Unified Dossier Spec

### 3.1 Hash SPA Router (`#slate`, `#match/<id>`, `#arena`, `#backtracker`)
Because `src/server.py` serves only `/` (`FileResponse(index.html)`), `/static/*`, `/healthz`, and `/api/v1/*`, client-side navigation MUST use URL hash routing with `window.addEventListener("hashchange", syncRoute)` in `web/app.js` (zero FastAPI path-routing changes needed, safe on browser refresh):
- **`#slate` (Default Landing Route):** If `window.location.hash` is empty on boot, `history.replaceState(null, "", "#slate")` and activate Today's Slate view.
- **`#match/<id>`:** Editorial fixture drill-down from any Slate card or ticker item. Top mode nav keeps `Today's Slate` highlighted as active context.
- **`#arena`:** Custom / All-Time Legends H2H Workbench (collapsed control rail + Unified Result Dossier).
- **`#backtracker`:** Historical Match Backtracker Workbench (open control rail + Historical Verdict Banner + Unified Result Dossier).

### 3.2 Shared Chrome: Ticker Strip & Clean Single-Row Masthead
- **Top ATP Wire Ticker (`#espnTickerStrip`, `data-testid="espn-ticker-strip"`, `#tickerTrack`, `data-testid="ticker-track"`):** Keep visible at the top of the shell for live/published fixture scanning and E2E contract compatibility, restyled in Risograph Cream/Teal (`--bg-recessed` strip, `--ink-primary` borders, `--ink-deep` text, `--accent-signal` live dot & `+EV` tag; zero dark `#090e1c` or `#00f5ff`). Clicking a ticker item routes to `#match/<fixture_id>` (or loads fixture into dossier).
- **Clean Single-Row Masthead:** Left lockup renders the `matchpoint.ml` wordmark (`--font-display`, `>=32px`, focal dot `.` in `--accent-signal`), center renders the 3 mode tabs (`#tabSchedule` `data-testid="tab-schedule"`, `#tabArena` `data-testid="tab-arena"`, `#tabBacktracker` `data-testid="tab-backtracker"`), right renders compact telemetry chips (`#chipModelsStatus`, `#chipSyncStatus`, `#chipRosterCount`).

### 3.3 View 1: `#slate` — Today's Court Slate (Option 3 Default)
- **Hero Anchor:** Flat `--bg-canvas` band with `Today's court slate.` (`--font-display`, `>=32px`) and concise 1-line lead.
- **4-KPI Strip:**
  1. `Published Fixtures` (`#kpiTotalFixtures`)
  2. `Live on Court` (`#kpiLiveCount`)
  3. `Value Edges (>=3%)` (`#kpiValueEdges`, `--accent-text`)
  4. `Champion AUC / LL` (`#kpiModelMetrics`, e.g., `0.717 / 0.615` from `/api/v1/meta`)
  *(Preserve `#kpiDivergentCount` and `#kpiSyncTime` data in masthead/sub-strip or KPI metadata if needed without cluttering the 4-box primary strip.)*
- **Filter Toolbar:** Series filter pills (`All Series`, `Grand Slam`, `Masters 1000`, `ATP500`, `ATP250`, plus `Masters Cup` when present in fixtures), `Value Edges Only (>=3%)` checkbox pill (`#onlyEdgeToggle`, `data-testid="only-edge-toggle"`), and `Refresh ESPN Feed` button (`#refreshScheduleBtn`, `data-testid="refresh-schedule-btn"`).
- **2-Column Print Plate Fixture Grid (`#scheduleCardGrid`, `data-testid="schedule-card-grid"`):**
  - Cards use `.riso-card` (`print-plate` class on featured/`+EV` fixtures), `1.5px solid var(--ink-primary)`, `--shadow-riso-card`.
  - Show tournament, round/status pill, surface/series/divergence/EV badges, favored vs underdog probabilities + odds, **single-favored-pink probability bar**, LogReg split or 1/4 Kelly footer, and `Open Dossier →` action.
  - Clicking any card routes to `#match/<fixture_id>`.
- **Deterministic Featured Fallback Slate (Zero Dead Screen on Boot):**
  - When `/api/v1/schedule/live` returns `fixtures: []` (between tournaments or unseeded local boot), do **not** strand the user on a dead empty box. Instead:
    1. Show a clear, honest status banner (`#scheduleNoticeBanner` / fallback notice: `"Live ESPN Feed Empty — Deterministic Featured Fallback Slate Active"`).
    2. Populate `#scheduleCardGrid` deterministically using curated Tour Spotlight pairs (e.g., `Alcaraz C. vs Sinner J.`, `Sinner J. vs Djokovic N.`, `Zverev A. vs Rublev A.` scored via `POST /api/v1/predict` using `/api/v1/meta` defaults, or deterministic client fallback if offline) marked clearly with `Deterministic Fallback` status pills.
    3. Keep `Refresh ESPN Feed` (`#refreshScheduleBtn`) active so real ESPN updates replace the fallback slate seamlessly, and keep `#scheduleEmptyState` (`data-testid="schedule-empty-state"`) in DOM (toggled hidden when fallback cards render, or reachable if filter yields 0).

### 3.4 View 2: `#match/<id>` — Match Editorial Dossier (Option 1)
- **Breadcrumb & Action Bar:** `← Back to Today's Slate` button (`#slate`) on left; `Open in Workbench →` button on right (copies fixture's `p1`, `p2`, `surface`, `series`, `odd_1`, `odd_2` into `#arena` controls, expands/syncs state, and routes to `#arena`).
- **Hero Title:** `<P1> vs <P2>.` (`--font-display`, `>=32px`, flat `--bg-canvas`) + tournament/round/surface subtitle.
- **Unified Result Dossier (`#arenaStage`, `data-testid="arena-stage"`):** Reused identically across `#match/<id>`, `#arena`, and `#backtracker`:
  1. **Focal Verdict Print Plate (`--mesh-print-plate`):**
     - Meta badge row: Surface badge (`--surface-hard` / `--surface-clay` / `--surface-grass`), Series pill, `+EV` / `Fade` / `Fair` badge, Divergence badge (`DIVERGENT >15%` or `CONSENSUS`), and Symmetry verification chip (`#symmetryChip`, `data-testid="symmetry-chip"`).
     - Split H2H Banner (`#splitH2HBanner`, `data-testid="split-h2h-banner"`) + Favored vs Underdog probability blocks + **single-favored-pink probability bar** + Dual-Model comparison bars (`#modelProbBars`: LightGBM Champion vs LogReg Baseline).
     - Power-Method De-Vig & Quarter-Kelly panel (`#kellyEdgePanel`, `data-testid="kelly-edge-panel"`): De-Vig Implied, Overround / $k$, Model Edge, `1/4 Kelly (Cap 2.5%)` meter.
     - Court Schematic & Divergence Gauge box (`#divergenceGauge`, `data-testid="divergence-gauge"`): Risograph court SVG tinted by surface with `--accent-signal` shot-bounce hotspot + divergence arc/readout.
     - **Always-Visible 5 Key Delta Strip (`#keyStatsTiles`, `data-testid="key-stats-tiles"`):** 1-row 5-column strip (`repeat(5, minmax(0, 1fr))` on desktop, `2-col` on mobile) displaying `Rank Δ`, `Surface WR Δ`, `Form Div Δ`, `Fatigue 14D Δ`, and `H2H Record` directly inside the Verdict Print Plate—never buried behind tabs.
  2. **Diamond-Mesh Tennis Net Divider SVG.**
  3. **3 Progressive Disclosure Tabs (`role="tablist"`):**
     - **Tab 1 — `1. TreeSHAP Drivers` (Default Active):** `#treeshapPanel` (`data-testid="treeshap-panel"`). Bidirectional log-odds tug-of-war bars (pink toward favored, muted teal toward underdog), Base Bias & Net P1 Pull summary pills.
     - **Tab 2 — `2. Scouting Cards & 6-Axis Radar`:** `#tradingCardsDeck` (`data-testid="trading-cards-deck"`) containing `#playerCardP1` (`data-testid="player-card-p1"`) and `#playerCardP2` (`data-testid="player-card-p2"`) with OVR shield, rank/pts, 5 stat cells (`SURF WR`, `10M FORM`, `CAREER`, `REST`, `14D LOAD`), and trait chips; plus `#radarChartContainer` (`data-testid="radar-chart"`) interactive 6-axis SVG radar and `#radarReadout` panel.
     - **Tab 3 — `3. 31-Feature Matrix`:** `#featuresMatrixSection` (`data-testid="features-matrix-section"`) containing category filter pills (`All`, `Ranking`, `Short Form (10M)`, `Med Form (365d)`, `Career`, `Context & Fatigue`, `Head-to-Head`, `Match Context`) and `#features31Table` (`data-testid="features-31-table"`) with all 31 rows (`#`, `Predictor`, `Category`, `Raw Value / Δ`, `TreeSHAP`, `LR Coef`, `Favors`).

### 3.5 View 3A & 3B: `#arena` & `#backtracker` — Workbench Rail + Unified Dossier (Option 2 + Option 1)
- **Layout:** `.workbench-toolbar` above `.workbench-grid` (`grid-template-columns: 310px 1fr` on desktop; collapses to `1fr` at `<=960px` or when `.is-rail-collapsed` is active, with sticky `top: 16px` on `.rail-stack`).
- **`#arena` (`#panelArena`, `data-testid="panel-arena"`):**
  - **Defaults Collapsed:** `.workbench-grid.is-rail-collapsed` on initial load so the Unified Result Dossier fills 100% width immediately.
  - `.workbench-toolbar` renders toggle button (`#arenaRailToggleBtn`, `aria-expanded="false"`, `aria-controls="arenaRail"`, label `"Show Player & Matchup Controls"` / `"Hide Matchup Sidebar"`) alongside `#arenaCollapsedSummary` (`Active: <P1> vs <P2> • <Surface> • <Series> • Odds ...`).
  - Inside `#arenaRail`:
    - `All-Time Legends Pool (2000–2026)` toggle (`#allTimeToggle`, `data-testid="all-time-toggle"`)
    - Player 1 search input (`#p1Filter`) & select (`#p1Select`, `data-testid="p1-select"`)
    - Player 2 search input (`#p2Filter`) & select (`#p2Select`, `data-testid="p2-select"`)
    - Swap button (`#swapPlayersBtn`, `data-testid="swap-players-btn"`)
    - Surface select (`#surfaceSelect`, `data-testid="surface-select"`), Series select (`#seriesSelect`, `data-testid="series-select"`)
    - Optional Decimal Odds inputs (`#odd1Input` `data-testid="odd1-input"`, `#odd2Input` `data-testid="odd2-input"`)
    - Submit CTA (`#predictBtn`, `data-testid="predict-btn"`, `"Analyze Matchup"`)
    - Error banner (`#arenaErrorBanner`, `data-testid="arena-error-banner"`)
    - Featured Matchup Presets card (e.g., `Federer R. vs Nadal R.`, `Sinner J. vs Djokovic N.`, `Alcaraz C. vs Sinner J.`).
- **`#backtracker` (`#panelBacktracker`, `data-testid="panel-backtracker"`):**
  - **Starts Open:** `.workbench-grid` without `.is-rail-collapsed` initially, with `.workbench-toolbar` toggle (`#btRailToggleBtn`, `aria-expanded="true"`, `aria-controls="btRail"`, `"Hide Archive Selectors"` / `"Show Archive Selectors"`) and `#btCollapsedSummary`.
  - Inside `#btRail`: cascading Dataset Split (`#histSplitSelect`, `data-testid="hist-split-select"`), Tournament (`#histTournamentSelect`, `data-testid="hist-tournament-select"`), Matchup (`#histMatchSelect`, `data-testid="hist-match-select"`), `Inspect Match` CTA (`#histInspectBtn`, `data-testid="hist-inspect-btn"`), error banner (`#histErrorBanner`), and Split Telemetry card.
  - Above the Unified Result Dossier in `.canvas-stack`: **Historical Result Verdict Banner** (`#historicalResultBanner`, `data-testid="historical-result-banner"`) showing `PREDICTION ACCURATE` / `UPSET` badge, tournament/round/date, actual winner, scoreline, and bookmaker implied closing probability bar (`bm_prob`).

---

## 4. Preserved Contracts & E2E Alignment

### 4.1 Preserved `data-testid` & Element ID Inventory
Keep all existing `id` and `data-testid` hooks intact across `web/index.html`, `web/app.js`, `web/arcade_cards.js`, and `web/charts.js`:
- **Chrome & Modes:** `espn-ticker-strip` (`#espnTickerStrip`), `ticker-track` (`#tickerTrack`), `tab-schedule` (`#tabSchedule`), `tab-arena` (`#tabArena`), `tab-backtracker` (`#tabBacktracker`), `panel-schedule` (`#panelSchedule`), `panel-arena` (`#panelArena`), `panel-backtracker` (`#panelBacktracker`).
- **Schedule:** `refresh-schedule-btn` (`#refreshScheduleBtn`), `only-edge-toggle` (`#onlyEdgeToggle`), `schedule-card-grid` (`#scheduleCardGrid`), `schedule-empty-state` (`#scheduleEmptyState`).
- **Arena Controls:** `all-time-toggle` (`#allTimeToggle`), `p1-select` (`#p1Select`), `p2-select` (`#p2Select`), `odd1-input` (`#odd1Input`), `odd2-input` (`#odd2Input`), `surface-select` (`#surfaceSelect`), `series-select` (`#seriesSelect`), `swap-players-btn` (`#swapPlayersBtn`), `predict-btn` (`#predictBtn`), `arena-error-banner` (`#arenaErrorBanner`).
- **Backtracker Controls:** `hist-split-select` (`#histSplitSelect`), `hist-tournament-select` (`#histTournamentSelect`), `hist-match-select` (`#histMatchSelect`), `hist-inspect-btn` (`#histInspectBtn`), `historical-result-banner` (`#historicalResultBanner`).
- **Unified Dossier (`#arenaStage`):** `arena-stage` (`#arenaStage`), `split-h2h-banner` (`#splitH2HBanner`), `symmetry-chip` (`#symmetryChip`), `trading-cards-deck` (`#tradingCardsDeck`), `player-card-p1` (`#playerCardP1`), `player-card-p2` (`#playerCardP2`), `radar-chart` (`#radarChartContainer`), `tale-of-the-tape` (`#divergenceAndTapePanel`), `divergence-gauge` (`#divergenceGauge`), `kelly-edge-panel` (`#kellyEdgePanel`), `key-stats-tiles` (`#keyStatsTiles`), `treeshap-panel` (`#treeshapPanel`), `features-matrix-section` (`#featuresMatrixSection`), `toggle-features-matrix-btn` (`#toggleFeaturesMatrixBtn`), `features-31-table` (`#features31Table`).

### 4.2 Required E2E & Tier-1 Test Updates
During implementation, update [`tests/e2e/cdp_runner.mjs`](../tests/e2e/cdp_runner.mjs), [`tests/e2e/test_tier1_features.py`](../tests/e2e/test_tier1_features.py), and [`tests/e2e/test_tier4_e2e_scenarios.py`](../tests/e2e/test_tier4_e2e_scenarios.py) for the new Risograph + Unified Dossier UX:

1. **Theme Token Assertions (`cdp_runner.mjs`, `test_tier1_features.py`, `test_tier4_e2e_scenarios.py`):**
   - `cdp_runner.mjs` `inspectCssColors()` & `boot_and_theme`: Replace dark-neon assertions (`rgb(6, 9, 19)`, `#00f5ff`, `#ccff00`, `#ff2bd6`) with Risograph Option A assertions: `bodyBg === 'rgb(248, 219, 193)'` (`#F8DBC1`), presence of `--ink-primary` (`#1D646C`), `--accent-signal` (`#EA4F88`), `--bg-surface` (`#FCE7D4`), and absence of legacy `#060913` / `#00f5ff` / `#ff2bd6` / `#ccff00` or pure `#ffffff`.
   - Update `test_f8_styles_css_defines_obsidian_and_neon_palette`, `test_f8_live_chromium_boot_and_theme` in `tests/e2e/test_tier1_features.py`, and `test_scenario_1_ephemeral_port_boot_healthz_and_neon_arena` in `tests/e2e/test_tier4_e2e_scenarios.py` to match the Risograph Option A tokens.
2. **Default `#slate` Route & Explicit Mode Switching:**
   - Because `/` now lands on `#slate` by default (instead of `#arena`), `h2h_odds_swap_matrix`, `alltime_legacy_and_divergence`, `invalid_odds_ui`, `historical_backtracker_flow`, and `daily_schedule_filters_and_ticker` must explicitly click `tab-arena` / `tab-backtracker` / `tab-schedule` (note: most already call `findClickable` / `clickControl`, but `daily_schedule_filters_and_ticker` should match `"Today's Slate"` as well as `"Daily Schedule"`, and `navigateAndHydrate()` should treat hydrated `#slate` fixture cards as ready).
3. **Collapsed `#arenaRail` Expansion Before Control Interaction:**
   - In `h2h_odds_swap_matrix`, `alltime_legacy_and_divergence`, and `invalid_odds_ui`, `#arenaWorkbenchGrid` defaults to `.is-rail-collapsed` (`#arenaRail` is `display: none`). Before calling `findSelect` / `findInput` for `p1Select`, `p2Select`, `odd1Input`, `odd2Input`, `allTimeToggle`, `surfaceSelect`, `seriesSelect`, `cdp_runner.mjs` must check for a collapsed rail toggle button (`#arenaRailToggleBtn` matching `/show\s+player|matchup\s+controls|sidebar/i` with `aria-expanded="false"`) and click it so controls become visible.
4. **Progressive Disclosure Tabs in `checkArcadeAndCharts()` & `expandAndCount31Features()`:**
   - Because Tab 2 (`2. Scouting Cards & 6-Axis Radar`) and Tab 3 (`3. 31-Feature Matrix`) are progressively disclosed instead of all visible simultaneously:
     - Ensure `checkArcadeAndCharts()` (or each scenario before calling it) inspects Tab 1 (`TreeSHAP Drivers`) for `hasTreeShap`, clicks Tab 2 (`Scouting Cards & 6-Axis Radar`) before checking `ovrCount >= 2`, `hasRadarGeometry`, and `radarAxesMatched >= 6`, and
     - Update `expandAndCount31Features()` to click Tab 3 (`3. 31-Feature Matrix`, or `#toggleFeaturesMatrixBtn` if preserved inside Tab 3) before counting visible `tbody tr` rows (`>= 31`).
5. **External Google Fonts CDN Allowlist (`externalRequests`):**
   - Per `PRODUCT.md` ("External Fonts / CDNs Permitted"), update `cdp_runner.mjs` `Network.requestWillBeSent` (line 1190) to allowlist `fonts.googleapis.com` and `fonts.gstatic.com` (while still failing on any unapproved external host), and update `test_f8_html_contains_zero_external_cdn_links` in `tests/e2e/test_tier1_features.py` to permit only allowlisted Google Fonts links (see `[PENDING MAIN-SESSION FONT FACTS & CRITIQUE]` regarding offline/airgapped Chromium test behavior).
6. **Schedule Drill-Down Compatibility (`daily_schedule_filters_and_ticker`):**
   - Ensure fixture cards / `#match/<id>` dossier expose accessible controls matching `/open\s+dossier|open\s+in\s+workbench|inspect|open\s+in\s+arena/i` so `daily_schedule_filters_and_ticker` transitions cleanly.

---

## 5. Verification Criteria (Run After Implementation)

Execute the following checks after completing all frontend and E2E updates:

1. **Unit & Backend Contract Tests:**
   ```bash
   pytest tests/unit/
   ```
   - Must pass 100% with zero skips or relaxed math/API assertions.
2. **Tier 1–3 Feature, Boundary & Combination Tests:**
   ```bash
   pytest tests/e2e/test_tier1_features.py tests/e2e/test_tier2_boundaries.py tests/e2e/test_tier3_combinations.py
   ```
   - Must pass 100% (including updated Risograph token & font checks).
3. **Tier 4 Full E2E Headless Chromium Scenarios:**
   ```bash
   pytest tests/e2e/test_tier4_e2e_scenarios.py
   ```
   - Scenarios 1–6 (`boot_and_theme`, `h2h_odds_swap_matrix`, `alltime_legacy_and_divergence`, `historical_backtracker_flow`, `daily_schedule_filters_and_ticker`, concurrent HTTP load) must all pass with `consoleErrors == []`, `pageExceptions == []`, `httpErrors == []`, `failedRequests == []`, and zero unapproved `externalRequests`.
4. **Mobile Viewport Overflow & Invalid Odds Checks:**
   Start `python -m src.server` (or ephemeral port) and run:
   ```bash
   node tests/e2e/cdp_runner.mjs --base-url http://127.0.0.1:8080 --scenario mobile_viewport_layout --width 390 --height 844
   node tests/e2e/cdp_runner.mjs --base-url http://127.0.0.1:8080 --scenario invalid_odds_ui --width 1440 --height 900
   ```
   - Assert `scrollWidth <= innerWidth + 2` at `390x844` across `#slate`, `#match/<id>`, `#arena`, and `#backtracker`, and zero `NaN` / `Infinity` / `undefined%` strings.
5. **Visual Craft & Design-System Audit (`/impeccable audit` / `detect`):**
   - Zero `#FFFFFF` pure-white backgrounds or dark-neon hexes (`#060913`, `#00f5ff`, `#ff2bd6`, `#ccff00`) anywhere in `web/`.
   - Zero `--font-display` usage below `32px`, inside cards/nav/buttons/tables, or over mesh gradients.
   - Single-favored-pink probability bar rule verified across `#slate`, `#match/<id>`, `#arena`, and `#backtracker`.
   - All numeric telemetry wrapped in `.metric-tabular` (`JetBrains Mono` + `tabular-nums`), and inline pink text uses `--accent-text` (`#C43269`) for WCAG AA contrast.
