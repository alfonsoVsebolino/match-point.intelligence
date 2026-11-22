# E2E Testing Infrastructure (`tests_e2e/`)

This repository includes an independent, requirement-driven, opaque-box E2E test suite in `tests_e2e/` covering Tiers 1–4 against the contracts in `PROJECT.md` and `.agents/teamwork/ORIGINAL_REQUEST.md`.

## 1. Environment & Tooling
- **Python Runtime**: `.venv/bin/python` (Python 3.11.2, `pytest 9.2.0`, `httpx 0.28.1`, `fastapi 0.165.0`, `uvicorn 0.54.0`, `pydantic 2.14.4`).
- **Node & Real Browser Runtime**: `node` v24.22.3 + `/usr/bin/chromium` (`Chromium 156`).
  - Browser tests run via `tests_e2e/cdp_runner.mjs` using Node v24 native `fetch` + `WebSocket` against Chromium's DevTools Protocol (`--headless --no-sandbox --disable-gpu --remote-debugging-port=...`) with ephemeral `--user-data-dir` directories cleaned up after every run.
  - Zero external npm packages, zero Playwright/Selenium dependencies, zero external CDN dependencies.
- **`NO_PROXY` IPv6 Workaround**:
  - In environments where `NO_PROXY` contains bracketed IPv6 (`[::1]`), `httpx.Client()` raises `InvalidURL: Invalid port: ':1]'` if `trust_env=True`.
  - All HTTP clients in `tests_e2e/conftest.py` pass `trust_env=False` and sanitize `NO_PROXY=""` / `no_proxy=""` in spawned server environments. Running pytest with `NO_PROXY="" no_proxy=""` is also recommended.

## 2. Harness Architecture (`tests_e2e/conftest.py` & `tests_e2e/cdp_runner.mjs`)
- **Live Server Lifecycle (`live_server` & `start_ephemeral_server`)**:
  - Allocates an unused TCP port on `127.0.0.1` via `free_port()`, sets `PORT=<port>`, launches `uvicorn server:app` (or `python server.py` via `use_python_entrypoint=True`), and polls `GET /healthz` until `200 OK` (up to 25s).
  - Fails loudly via `pytest.fail` / `RuntimeError` with full captured stdout/stderr if `server.py` is missing or fails to boot (zero silent skips or pass-by-default stubs).
- **Isolated In-Process Client (`create_inprocess_client`)**:
  - Used strictly with `tmp_path` + `monkeypatch` when testing mocked ESPN HTTP responses, T-30m snapshot locking, corrupted `daily_predictions_latest.json`, or populated schedules without mutating repository `data/`.
- **Opaque-Box Mathematical Oracles**:
  - `solve_power_devig(odd_1, odd_2)`: Independent bisection solver (`< 1e-9` tolerance) for $q_1^{1/k} + q_2^{1/k} = 1.0$.
  - `compute_quarter_kelly_pct(prob, odd)`: Independent Quarter-Kelly formula ($0.25 \cdot f^* \cdot 100$) clamped to `[0.0, 2.5]`.
  - `assert_valid_player_card`, `assert_valid_treeshap`, `assert_valid_features_31`: Schema and invariant validators aligned with `PROJECT.md`.

## 3. UI & Accessibility Contract Notes for `web/`
The browser driver `tests_e2e/cdp_runner.mjs` locates UI elements semantically (accessible labels, button/tab text, `<select>` options, input names/labels, SVG elements, computed CSS) rather than coupling to private class names:
- **Canvas & Neon Theme**: `document.body` background `#060913` (`rgb(6, 9, 19)`); CSS defines `#00f5ff`, `#ccff00`, `#ff2bd6`; zero external CDN URLs (`http://` / `https://` off-host requests fail the test).
- **Top Ticker & 3 Modes**: ESPN-style top live-match ticker strip + mode switcher controls for **Daily Schedule**, **H2H Arena / Predictor**, and **Historical Backtracker**.
- **H2H Controls**: P1 & P2 selectors, `Swap` (`⇄`) button, All-Time Historical Roster (`2000–2026`) toggle, Surface & Series selectors, optional `odd_1` / `odd_2` inputs, Predict trigger, collapsible 31-Feature Matrix toggle, SVG 6-axis Radar, TreeSHAP tug-of-war, Arcade Trading Cards, and Divergence gauge.
- **Schedule & Backtracker Controls**: Split / Tournament / Matchup cascading selectors + Inspect trigger; Refresh Schedule button + Series filter pills (`All`, `Grand Slam`, ...) + `>=3%` Value Edge toggle.

## 4. Running the Suite
```bash
# Full E2E suite (Tiers 1-4)
NO_PROXY="" no_proxy="" .venv/bin/pytest tests_e2e -v

# Individual tiers
NO_PROXY="" no_proxy="" .venv/bin/pytest tests_e2e -m tier1 -v
NO_PROXY="" no_proxy="" .venv/bin/pytest tests_e2e -m tier2 -v
NO_PROXY="" no_proxy="" .venv/bin/pytest tests_e2e -m tier3 -v
NO_PROXY="" no_proxy="" .venv/bin/pytest tests_e2e -m tier4 -v

# Collection check only
.venv/bin/pytest tests_e2e --collect-only -q
```
