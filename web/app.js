/* ==========================================================================
   MatchPoint.intelligence — Application Controller & API Integration
   Pure Vanilla ES6 JS • Zero External Dependencies
   ========================================================================== */

(function (global) {
  "use strict";

  const MP = (global.MP = global.MP || {});
  const esc = MP.escapeHtml;

  const state = {
    mode: "arena", // "schedule" | "arena" | "backtracker"
    health: null,
    meta: {
      active_players: [],
      all_players: [],
      surfaces: ["Hard", "Clay", "Grass", "Carpet"],
      series: ["Grand Slam", "Masters 1000", "ATP500", "ATP250"],
      splits: [],
      default_p1: "Alcaraz C.",
      default_p2: "Sinner J.",
      sync_status: {},
      model_metrics: {},
    },
    schedule: {
      generated_at: null,
      target_date: null,
      source: "none",
      sync_status: {},
      kpis: { total_fixtures: 0, live_count: 0, value_edges_count: 0, divergent_count: 0 },
      fixtures: [],
    },
    scheduleSeriesFilter: "all",
    scheduleOnlyEdges: false,
    allTimePool: false,
    arenaResult: null,
    histResult: null,
    activeStageData: null,
    matrixCategory: "All",
    matrixExpanded: false,
  };

  const els = {};

  function cacheElements() {
    const ids = [
      "tickerTrack",
      "chipModelsStatus",
      "chipSyncStatus",
      "chipRosterCount",
      "tabSchedule",
      "tabScheduleBadge",
      "tabArena",
      "tabBacktracker",
      "panelSchedule",
      "panelArena",
      "panelBacktracker",
      "refreshScheduleBtn",
      "kpiTotalFixtures",
      "kpiLiveCount",
      "kpiValueEdges",
      "kpiDivergentCount",
      "kpiModelMetrics",
      "kpiSyncTime",
      "onlyEdgeToggle",
      "scheduleNoticeBanner",
      "scheduleCardGrid",
      "scheduleEmptyState",
      "emptyRefreshBtn",
      "emptyGoArenaBtn",
      "allTimeToggle",
      "arenaForm",
      "p1Filter",
      "p1Select",
      "odd1Input",
      "swapPlayersBtn",
      "surfaceSelect",
      "seriesSelect",
      "predictBtn",
      "p2Filter",
      "p2Select",
      "odd2Input",
      "arenaErrorBanner",
      "histSplitSelect",
      "histTournamentSelect",
      "histMatchSelect",
      "histInspectBtn",
      "histErrorBanner",
      "arenaStage",
      "historicalResultBanner",
      "histStandbyPrompt",
      "stageContent",
      "splitH2HBanner",
      "playerCardP1",
      "playerCardP2",
      "radarLegendP1",
      "radarLegendP2",
      "radarChartContainer",
      "radarReadout",
      "divergenceGauge",
      "modelProbBars",
      "kellyEdgePanel",
      "keyStatsTiles",
      "treeshapSummary",
      "treeshapBars",
      "toggleFeaturesMatrixBtn",
      "featuresMatrixBody",
      "matrixCategoryPills",
      "features31Tbody",
    ];
    ids.forEach((id) => {
      els[id] = document.getElementById(id);
    });
  }

  function parseApiError(payload, statusCode) {
    if (payload && typeof payload === "object") {
      if (Array.isArray(payload.detail)) {
        const msgs = payload.detail
          .map((item) => {
            if (typeof item === "string") return item;
            if (item && item.msg) {
              const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
              return loc ? `${loc}: ${item.msg}` : item.msg;
            }
            return JSON.stringify(item);
          })
          .filter(Boolean);
        if (msgs.length) return msgs.join(" | ");
      }
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        return payload.detail;
      }
      if (typeof payload.error === "string" && payload.error.trim()) {
        return payload.error;
      }
    }
    return `Request failed (HTTP ${statusCode})`;
  }

  async function apiFetchJson(url, options = {}) {
    const res = await fetch(url, options);
    let data = null;
    try {
      data = await res.json();
    } catch (_err) {
      data = null;
    }
    if (!res.ok) {
      throw new Error(parseApiError(data, res.status));
    }
    return data;
  }

  function parseOptionalOdd(inputEl) {
    if (!inputEl) return null;
    const raw = String(inputEl.value || "").trim();
    if (!raw) return null;
    const num = Number(raw);
    if (!Number.isFinite(num)) return null;
    return num;
  }

  function setButtonLoading(btn, loading) {
    if (!btn) return;
    btn.disabled = Boolean(loading);
    btn.classList.toggle("is-loading", Boolean(loading));
  }

  function showBanner(el, msg) {
    if (!el) return;
    if (!msg) {
      el.classList.add("is-hidden");
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    el.classList.remove("is-hidden");
  }

  function formatSyncTimestamp(syncStatus) {
    const ss = syncStatus || {};
    const ts = ss.last_synced_at ?? ss.synced_at ?? null;
    const status = String(ss.status || "static").toUpperCase();
    const latestDate = ss.latest_match_date || "";
    if (ts) {
      return `${status} • ${String(ts).slice(0, 16).replace("T", " ")}`;
    }
    if (latestDate) {
      return `${status} • DATA ${latestDate}`;
    }
    return status;
  }

  function updateMastheadTelemetry() {
    const h = state.health || {};
    const m = state.meta || {};
    const syncStatus = h.sync_status || m.sync_status || {};

    if (els.chipModelsStatus) {
      const valEl = els.chipModelsStatus.querySelector(".chip-val");
      if (valEl) {
        valEl.textContent = h.models_loaded === false ? "MODELS STANDBY" : "LGBM + LR ONLINE";
      }
    }

    if (els.chipSyncStatus) {
      const valEl = els.chipSyncStatus.querySelector(".chip-val");
      if (valEl) {
        valEl.textContent = formatSyncTimestamp(syncStatus);
      }
    }

    if (els.chipRosterCount) {
      const valEl = els.chipRosterCount.querySelector(".chip-val");
      if (valEl) {
        const act = h.players_active ?? (m.active_players ? m.active_players.length : "--");
        const tot = h.players_total ?? (m.all_players ? m.all_players.length : "--");
        valEl.textContent = `${act} ACTIVE / ${tot} TOTAL`;
      }
    }
  }

  function switchMode(targetMode) {
    state.mode = targetMode;

    const tabs = [
      { mode: "schedule", tab: els.tabSchedule, panel: els.panelSchedule },
      { mode: "arena", tab: els.tabArena, panel: els.panelArena },
      { mode: "backtracker", tab: els.tabBacktracker, panel: els.panelBacktracker },
    ];

    tabs.forEach(({ mode, tab, panel }) => {
      const active = mode === targetMode;
      if (tab) {
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
      }
      if (panel) {
        panel.classList.toggle("is-hidden", !active);
        panel.setAttribute("aria-hidden", active ? "false" : "true");
      }
    });

    if (targetMode === "schedule") {
      if (els.arenaStage) els.arenaStage.classList.add("is-hidden");
    } else if (targetMode === "arena") {
      if (els.arenaStage) els.arenaStage.classList.remove("is-hidden");
      if (els.histStandbyPrompt) els.histStandbyPrompt.classList.add("is-hidden");
      if (state.arenaResult) {
        renderArenaStage(state.arenaResult);
      }
    } else if (targetMode === "backtracker") {
      if (els.arenaStage) els.arenaStage.classList.remove("is-hidden");
      if (state.histResult) {
        if (els.histStandbyPrompt) els.histStandbyPrompt.classList.add("is-hidden");
        renderArenaStage(state.histResult);
      } else {
        if (els.historicalResultBanner) els.historicalResultBanner.classList.add("is-hidden");
        if (els.stageContent) els.stageContent.classList.add("is-hidden");
        if (els.histStandbyPrompt) els.histStandbyPrompt.classList.remove("is-hidden");
      }
    }
  }

  /* ========================================================================
     Ticker & Daily Schedule Board
     ======================================================================== */
  function renderTicker() {
    if (!els.tickerTrack) return;
    const fixtures = state.schedule.fixtures || [];
    if (fixtures.length === 0) {
      els.tickerTrack.innerHTML = `
        <div class="ticker-standby">
          LIVE FEED STANDBY &bull; NO PUBLISHED FIXTURES &bull; SYNC OR RUN CUSTOM H2H ARENA
        </div>
      `;
      return;
    }

    els.tickerTrack.innerHTML = fixtures
      .map((fx) => {
        const sm = MP.getStatusPillMeta(fx.status, fx.snapshot_locked);
        const lgbP1 = MP.safeNum(fx.prob_lgb, 0.5);
        const lgbP2 = 1 - lgbP1;
        const hasVal =
          fx.edge_lgb_p1 !== null &&
          fx.edge_lgb_p1 !== undefined &&
          Math.abs(Number(fx.edge_lgb_p1)) >= 0.03;

        return `
          <button type="button" class="ticker-item ${hasVal ? "has-value" : ""}" data-fixture-id="${esc(fx.fixture_id)}">
            <span class="status-pill ${sm.cls}">${sm.label}</span>
            <span class="ticker-players">${esc(fx.p1)} vs ${esc(fx.p2)}</span>
            ${fx.status_detail ? `<span class="ticker-detail">${esc(fx.status_detail)}</span>` : ""}
            <span class="ticker-split">${(lgbP1 * 100).toFixed(0)}% / <span class="p2-pct">${(lgbP2 * 100).toFixed(0)}%</span></span>
            ${hasVal ? `<span class="value-alert-pill">VALUE</span>` : ""}
          </button>
        `;
      })
      .join("");
  }

  function renderScheduleBoard() {
    const sched = state.schedule || {};
    const fixtures = Array.isArray(sched.fixtures) ? sched.fixtures : [];
    const kpis = sched.kpis || {};

    if (els.tabScheduleBadge) {
      els.tabScheduleBadge.textContent = String(fixtures.length);
    }
    if (els.kpiTotalFixtures) els.kpiTotalFixtures.textContent = String(kpis.total_fixtures ?? fixtures.length);
    if (els.kpiLiveCount) els.kpiLiveCount.textContent = String(kpis.live_count ?? 0);
    if (els.kpiValueEdges) els.kpiValueEdges.textContent = String(kpis.value_edges_count ?? 0);
    if (els.kpiDivergentCount) els.kpiDivergentCount.textContent = String(kpis.divergent_count ?? 0);

    if (els.kpiModelMetrics) {
      const mm = state.meta.model_metrics || {};
      if (mm.lgbm_auc !== undefined) {
        els.kpiModelMetrics.textContent = `AUC ${Number(mm.lgbm_auc).toFixed(3)} • LL ${Number(mm.lgbm_logloss).toFixed(3)} • ${mm.trees ?? 0}T • ${mm.features_count ?? 31}F`;
      } else {
        els.kpiModelMetrics.textContent = "LGBM + LR Active";
      }
    }

    if (els.kpiSyncTime) {
      if (sched.generated_at) {
        els.kpiSyncTime.textContent = String(sched.generated_at).slice(0, 16).replace("T", " ");
      } else {
        els.kpiSyncTime.textContent = formatSyncTimestamp(sched.sync_status || state.meta.sync_status);
      }
    }

    if (fixtures.length === 0) {
      if (els.scheduleCardGrid) els.scheduleCardGrid.innerHTML = "";
      if (els.scheduleEmptyState) els.scheduleEmptyState.classList.remove("is-hidden");
      return;
    }

    if (els.scheduleEmptyState) els.scheduleEmptyState.classList.add("is-hidden");

    const filtered = fixtures.filter((fx) => {
      if (state.scheduleSeriesFilter !== "all") {
        if (String(fx.series || "").toLowerCase() !== state.scheduleSeriesFilter.toLowerCase()) {
          return false;
        }
      }
      if (state.scheduleOnlyEdges) {
        if (fx.edge_lgb_p1 === null || fx.edge_lgb_p1 === undefined) return false;
        if (Math.abs(Number(fx.edge_lgb_p1)) < 0.03) return false;
      }
      return true;
    });

    if (!els.scheduleCardGrid) return;
    if (filtered.length === 0) {
      els.scheduleCardGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-state-badge">FILTER ACTIVE</div>
          <h3>No Fixtures Match Active Filters</h3>
          <p>Reset the tournament series pill or toggle off Value Edges Only to view all ${fixtures.length} published fixtures.</p>
        </div>
      `;
      return;
    }

    els.scheduleCardGrid.innerHTML = filtered.map((fx) => MP.renderScheduleCard(fx)).join("");
  }

  async function refreshSchedule() {
    setButtonLoading(els.refreshScheduleBtn, true);
    setButtonLoading(els.emptyRefreshBtn, true);
    showBanner(els.scheduleNoticeBanner, "");
    try {
      const res = await apiFetchJson("/api/v1/schedule/refresh", { method: "POST" });
      state.schedule = res;
      renderTicker();
      renderScheduleBoard();
      if (res.refresh_error) {
        showBanner(els.scheduleNoticeBanner, `ESPN Live Refresh Notice: ${res.refresh_error}`);
      } else if (res.refreshed === false) {
        showBanner(els.scheduleNoticeBanner, "Schedule preserved from local cache (no new ESPN update).");
      }
    } catch (err) {
      showBanner(els.scheduleNoticeBanner, `Schedule refresh fallback: ${err.message}`);
    } finally {
      setButtonLoading(els.refreshScheduleBtn, false);
      setButtonLoading(els.emptyRefreshBtn, false);
    }
  }

  /* ========================================================================
     Mode 2: H2H Matchup Arena Selectors & Prediction
     ======================================================================== */
  function getActivePlayerPool() {
    return state.allTimePool
      ? state.meta.all_players || []
      : state.meta.active_players || [];
  }

  function ensureSelectHasOption(selectEl, value, suffixLabel = "") {
    if (!selectEl || !value) return;
    const strVal = String(value);
    const existing = Array.from(selectEl.options).find((opt) => opt.value === strVal);
    if (!existing) {
      const opt = document.createElement("option");
      opt.value = strVal;
      opt.textContent = suffixLabel ? `${strVal} ${suffixLabel}` : strVal;
      opt.dataset.temp = "true";
      selectEl.insertBefore(opt, selectEl.firstChild);
    }
    selectEl.value = strVal;
  }

  function populatePlayerSelect(selectEl, filterQuery, preferredValue) {
    if (!selectEl) return;
    const pool = getActivePlayerPool();
    const q = String(filterQuery || "").trim().toLowerCase();
    const list = q ? pool.filter((name) => name.toLowerCase().includes(q)) : pool;

    selectEl.innerHTML = "";
    if (list.length === 0) {
      const emptyOpt = document.createElement("option");
      emptyOpt.value = "";
      emptyOpt.textContent = "No matching players";
      selectEl.appendChild(emptyOpt);
      if (preferredValue && !q) {
        ensureSelectHasOption(selectEl, preferredValue, "(Cold Start)");
      }
      return;
    }

    const frag = document.createDocumentFragment();
    list.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      frag.appendChild(opt);
    });
    selectEl.appendChild(frag);

    if (preferredValue && list.includes(preferredValue)) {
      selectEl.value = preferredValue;
    } else if (preferredValue && !q) {
      ensureSelectHasOption(selectEl, preferredValue, "(Cold Start)");
    } else {
      selectEl.selectedIndex = 0;
    }
  }

  function populateContextSelects() {
    if (els.surfaceSelect) {
      const surfaces = state.meta.surfaces || ["Hard", "Clay", "Grass", "Carpet"];
      els.surfaceSelect.innerHTML = surfaces
        .map((s) => `<option value="${esc(s)}">${esc(s)}</option>`)
        .join("");
      if (surfaces.includes("Hard")) els.surfaceSelect.value = "Hard";
    }

    if (els.seriesSelect) {
      const series = state.meta.series || ["Grand Slam", "Masters 1000", "ATP500", "ATP250"];
      els.seriesSelect.innerHTML = series
        .map((s) => `<option value="${esc(s)}">${esc(s)}</option>`)
        .join("");
      if (series.includes("Grand Slam")) els.seriesSelect.value = "Grand Slam";
    }
  }

  async function runArenaPredict() {
    if (!els.p1Select || !els.p2Select) return;
    const p1 = String(els.p1Select.value || "").trim();
    const p2 = String(els.p2Select.value || "").trim();
    const surface = String(els.surfaceSelect ? els.surfaceSelect.value : "Hard");
    const series = String(els.seriesSelect ? els.seriesSelect.value : "Grand Slam");
    const odd1 = parseOptionalOdd(els.odd1Input);
    const odd2 = parseOptionalOdd(els.odd2Input);

    showBanner(els.arenaErrorBanner, "");
    setButtonLoading(els.predictBtn, true);

    try {
      const data = await apiFetchJson("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          p1,
          p2,
          surface,
          series,
          odd_1: odd1,
          odd_2: odd2,
        }),
      });
      state.arenaResult = data;
      if (state.mode === "arena") {
        renderArenaStage(data);
      }
    } catch (err) {
      showBanner(els.arenaErrorBanner, err.message);
    } finally {
      setButtonLoading(els.predictBtn, false);
    }
  }

  function loadFixtureIntoArena(fx) {
    if (!fx) return;
    switchMode("arena");
    if (els.p1Filter) els.p1Filter.value = "";
    if (els.p2Filter) els.p2Filter.value = "";

    populatePlayerSelect(els.p1Select, "", fx.p1);
    populatePlayerSelect(els.p2Select, "", fx.p2);
    ensureSelectHasOption(els.p1Select, fx.p1, "(Cold Start)");
    ensureSelectHasOption(els.p2Select, fx.p2, "(Cold Start)");

    if (els.surfaceSelect) ensureSelectHasOption(els.surfaceSelect, fx.surface || "Hard");
    if (els.seriesSelect) ensureSelectHasOption(els.seriesSelect, fx.series || "ATP250");

    if (els.odd1Input) {
      els.odd1Input.value =
        fx.odd_1 !== null && fx.odd_1 !== undefined ? String(fx.odd_1) : "";
    }
    if (els.odd2Input) {
      els.odd2Input.value =
        fx.odd_2 !== null && fx.odd_2 !== undefined ? String(fx.odd_2) : "";
    }

    runArenaPredict();
    if (els.arenaStage && typeof els.arenaStage.scrollIntoView === "function") {
      els.arenaStage.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /* ========================================================================
     Mode 3: Historical Match Backtracker (2000-2026)
     ======================================================================== */
  function populateHistoricalSplits() {
    if (!els.histSplitSelect) return;
    const splits = state.meta.splits || [];
    if (splits.length === 0) {
      els.histSplitSelect.innerHTML = `<option value="">No dataset splits available</option>`;
      return;
    }
    els.histSplitSelect.innerHTML = splits
      .map((s) => `<option value="${esc(s)}">${esc(s)}</option>`)
      .join("");
  }

  async function fetchHistoricalList(split, tournament = null) {
    if (!split) return;
    showBanner(els.histErrorBanner, "");
    const reqSeq = (fetchHistoricalList.seq = (fetchHistoricalList.seq || 0) + 1);
    // Invalidate the stale matchup list immediately so a match_idx from the previous
    // tournament cannot be inspected against the newly selected tournament.
    if (els.histMatchSelect) {
      els.histMatchSelect.innerHTML = `<option value="">Loading matchups…</option>`;
    }
    if (els.histInspectBtn) els.histInspectBtn.disabled = true;
    try {
      const params = new URLSearchParams({ split });
      if (tournament) params.set("tournament", tournament);
      const data = await apiFetchJson(`/api/v1/fixtures/historical?${params.toString()}`);
      if (reqSeq !== fetchHistoricalList.seq) return;

      const tournaments = Array.isArray(data.tournaments) ? data.tournaments : [];
      const matches = Array.isArray(data.matches) ? data.matches : [];

      if (els.histTournamentSelect) {
        if (tournaments.length === 0) {
          els.histTournamentSelect.innerHTML = `<option value="">No tournaments available</option>`;
        } else {
          els.histTournamentSelect.innerHTML = tournaments
            .map((t) => `<option value="${esc(t)}">${esc(t)}</option>`)
            .join("");
          if (data.selected_tournament && tournaments.includes(data.selected_tournament)) {
            els.histTournamentSelect.value = data.selected_tournament;
          }
        }
      }

      if (els.histMatchSelect) {
        if (matches.length === 0) {
          els.histMatchSelect.innerHTML = `<option value="">No matches available</option>`;
          if (els.histInspectBtn) els.histInspectBtn.disabled = true;
        } else {
          els.histMatchSelect.innerHTML = matches
            .map((m) => `<option value="${esc(m.match_idx)}">${esc(m.label)}</option>`)
            .join("");
          if (els.histInspectBtn) els.histInspectBtn.disabled = false;
        }
      }
    } catch (err) {
      showBanner(els.histErrorBanner, err.message);
      if (els.histInspectBtn) els.histInspectBtn.disabled = true;
    }
  }

  async function inspectHistoricalMatch() {
    const split = els.histSplitSelect ? els.histSplitSelect.value : "";
    const tournament = els.histTournamentSelect ? els.histTournamentSelect.value : "";
    const matchIdx = els.histMatchSelect ? els.histMatchSelect.value : "";

    showBanner(els.histErrorBanner, "");
    if (!split || !matchIdx) {
      showBanner(els.histErrorBanner, "Select a valid split, tournament, and matchup to inspect.");
      return;
    }

    setButtonLoading(els.histInspectBtn, true);
    try {
      const params = new URLSearchParams({
        split,
        tournament,
        match_idx: matchIdx,
      });
      const data = await apiFetchJson(`/api/v1/fixtures/historical/inspect?${params.toString()}`);
      state.histResult = data;
      if (state.mode === "backtracker") {
        if (els.histStandbyPrompt) els.histStandbyPrompt.classList.add("is-hidden");
        renderArenaStage(data);
      }
    } catch (err) {
      showBanner(els.histErrorBanner, err.message);
    } finally {
      setButtonLoading(els.histInspectBtn, false);
    }
  }

  /* ========================================================================
     Full Arcade Broadcast Stage Renderer & 31-Feature Matrix
     ======================================================================== */
  function renderFeaturesMatrixTable() {
    if (!els.features31Tbody) return;
    const data = state.activeStageData;
    const rows = data && Array.isArray(data.features_31) ? data.features_31 : [];
    const cat = state.matrixCategory;

    const filtered = rows
      .map((item, idx) => ({ ...item, _idx: idx + 1 }))
      .filter((item) => cat === "All" || item.category === cat);

    if (filtered.length === 0) {
      els.features31Tbody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center; color:var(--text-muted); padding:18px;">
            No predictors match category "${esc(cat)}".
          </td>
        </tr>
      `;
      return;
    }

    els.features31Tbody.innerHTML = filtered
      .map((item) => {
        const val = MP.safeNum(item.value, 0);
        const shap = MP.safeNum(item.shap, 0);
        const lrCoef = MP.safeNum(item.lr_coef, 0);
        const favors = item.favors || "neutral";
        const favorsLabel = favors === "p1" ? "P1" : favors === "p2" ? "P2" : "NEUTRAL";
        const shapCls = favors === "p1" ? "fav-p1" : favors === "p2" ? "fav-p2" : "fav-neutral";
        const valStr = Number.isInteger(val) ? String(val) : val.toFixed(4);

        return `
          <tr>
            <td class="num-col">${item._idx}</td>
            <td>
              <strong>${esc(item.label || item.name)}</strong>
              <span class="feat-name-sub">${esc(item.name)}</span>
            </td>
            <td><span class="series-badge">${esc(item.category)}</span></td>
            <td class="num-col">${esc(valStr)}</td>
            <td class="num-col ${shapCls}">${shap > 0 ? "+" : ""}${shap.toFixed(5)}</td>
            <td class="num-col">${lrCoef > 0 ? "+" : ""}${lrCoef.toFixed(4)}</td>
            <td><span class="favors-badge ${esc(favors)}">${favorsLabel}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  function renderArenaStage(data) {
    if (!data) return;
    state.activeStageData = data;
    if (els.stageContent) els.stageContent.classList.remove("is-hidden");

    MP.renderHistoricalBanner(els.historicalResultBanner, data);
    MP.renderSplitH2HBanner(els.splitH2HBanner, data);
    MP.renderTradingCardsDeck(els.playerCardP1, els.playerCardP2, data);

    if (els.radarLegendP1) els.radarLegendP1.textContent = data.p1 || "P1";
    if (els.radarLegendP2) els.radarLegendP2.textContent = data.p2 || "P2";

    MP.renderRadarChart(els.radarChartContainer, els.radarReadout, data);
    MP.renderDivergenceGauge(els.divergenceGauge, data);
    MP.renderModelProbBars(els.modelProbBars, data);
    MP.renderKellyEdgePanel(els.kellyEdgePanel, data);
    MP.renderKeyStatsTiles(els.keyStatsTiles, data);
    MP.renderTreeShapTugOfWar(els.treeshapSummary, els.treeshapBars, data);
    renderFeaturesMatrixTable();
  }

  /* ========================================================================
     Event Wiring & Bootstrap
     ======================================================================== */
  function bindEvents() {
    if (els.tabSchedule) els.tabSchedule.addEventListener("click", () => switchMode("schedule"));
    if (els.tabArena) els.tabArena.addEventListener("click", () => switchMode("arena"));
    if (els.tabBacktracker) els.tabBacktracker.addEventListener("click", () => switchMode("backtracker"));

    if (els.refreshScheduleBtn) els.refreshScheduleBtn.addEventListener("click", refreshSchedule);
    if (els.emptyRefreshBtn) els.emptyRefreshBtn.addEventListener("click", refreshSchedule);
    if (els.emptyGoArenaBtn) els.emptyGoArenaBtn.addEventListener("click", () => switchMode("arena"));

    // Ticker item click delegation
    if (els.tickerTrack) {
      els.tickerTrack.addEventListener("click", (e) => {
        const btn = e.target.closest(".ticker-item");
        if (!btn) return;
        const fid = btn.getAttribute("data-fixture-id");
        const fx = (state.schedule.fixtures || []).find((item) => item.fixture_id === fid);
        if (fx) loadFixtureIntoArena(fx);
      });
    }

    // Schedule card click delegation
    if (els.scheduleCardGrid) {
      els.scheduleCardGrid.addEventListener("click", (e) => {
        const card = e.target.closest(".schedule-card");
        if (!card) return;
        const fid = card.getAttribute("data-fixture-id");
        const fx = (state.schedule.fixtures || []).find((item) => item.fixture_id === fid);
        if (fx) loadFixtureIntoArena(fx);
      });
    }

    // Schedule series filter pills
    if (els.panelSchedule) {
      const pillsContainer = els.panelSchedule.querySelector(".series-pills");
      if (pillsContainer) {
        pillsContainer.addEventListener("click", (e) => {
          const pill = e.target.closest(".filter-pill");
          if (!pill) return;
          pillsContainer.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("is-active"));
          pill.classList.add("is-active");
          state.scheduleSeriesFilter = pill.getAttribute("data-series") || "all";
          renderScheduleBoard();
        });
      }
    }

    if (els.onlyEdgeToggle) {
      els.onlyEdgeToggle.addEventListener("change", () => {
        state.scheduleOnlyEdges = Boolean(els.onlyEdgeToggle.checked);
        renderScheduleBoard();
      });
    }

    // Arena controls
    if (els.allTimeToggle) {
      els.allTimeToggle.addEventListener("change", () => {
        state.allTimePool = Boolean(els.allTimeToggle.checked);
        const curP1 = els.p1Select ? els.p1Select.value : state.meta.default_p1;
        const curP2 = els.p2Select ? els.p2Select.value : state.meta.default_p2;
        populatePlayerSelect(els.p1Select, els.p1Filter ? els.p1Filter.value : "", curP1);
        populatePlayerSelect(els.p2Select, els.p2Filter ? els.p2Filter.value : "", curP2);
      });
    }

    if (els.p1Filter) {
      els.p1Filter.addEventListener("input", () => {
        const cur = els.p1Select ? els.p1Select.value : "";
        populatePlayerSelect(els.p1Select, els.p1Filter.value, cur);
      });
    }

    if (els.p2Filter) {
      els.p2Filter.addEventListener("input", () => {
        const cur = els.p2Select ? els.p2Select.value : "";
        populatePlayerSelect(els.p2Select, els.p2Filter.value, cur);
      });
    }

    if (els.swapPlayersBtn) {
      els.swapPlayersBtn.addEventListener("click", () => {
        const p1Val = els.p1Select ? els.p1Select.value : "";
        const p2Val = els.p2Select ? els.p2Select.value : "";
        if (els.p1Filter) els.p1Filter.value = "";
        if (els.p2Filter) els.p2Filter.value = "";
        populatePlayerSelect(els.p1Select, "", p2Val);
        populatePlayerSelect(els.p2Select, "", p1Val);
        ensureSelectHasOption(els.p1Select, p2Val, "(Cold Start)");
        ensureSelectHasOption(els.p2Select, p1Val, "(Cold Start)");

        if (els.odd1Input && els.odd2Input) {
          const o1 = els.odd1Input.value;
          els.odd1Input.value = els.odd2Input.value;
          els.odd2Input.value = o1;
        }

        // Re-run the prediction so the displayed probabilities invert with the swapped players.
        if (state.arenaResult) runArenaPredict();
      });
    }

    if (els.arenaForm) {
      els.arenaForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runArenaPredict();
      });
    }

    // Backtracker cascading selects
    if (els.histSplitSelect) {
      els.histSplitSelect.addEventListener("change", () => {
        fetchHistoricalList(els.histSplitSelect.value, null);
      });
    }

    if (els.histTournamentSelect) {
      els.histTournamentSelect.addEventListener("change", () => {
        const split = els.histSplitSelect ? els.histSplitSelect.value : "";
        fetchHistoricalList(split, els.histTournamentSelect.value);
      });
    }

    if (els.histInspectBtn) {
      els.histInspectBtn.addEventListener("click", inspectHistoricalMatch);
    }

    // Collapsible 31-Feature Matrix
    if (els.toggleFeaturesMatrixBtn && els.featuresMatrixBody) {
      els.toggleFeaturesMatrixBtn.addEventListener("click", () => {
        state.matrixExpanded = !state.matrixExpanded;
        els.featuresMatrixBody.classList.toggle("is-hidden", !state.matrixExpanded);
        els.toggleFeaturesMatrixBtn.setAttribute("aria-expanded", state.matrixExpanded ? "true" : "false");
        els.toggleFeaturesMatrixBtn.textContent = state.matrixExpanded
          ? "Hide 31-Feature Matrix"
          : "Show 31-Feature Matrix";
      });
    }

    if (els.matrixCategoryPills) {
      els.matrixCategoryPills.addEventListener("click", (e) => {
        const pill = e.target.closest(".filter-pill");
        if (!pill) return;
        els.matrixCategoryPills.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("is-active"));
        pill.classList.add("is-active");
        state.matrixCategory = pill.getAttribute("data-category") || "All";
        renderFeaturesMatrixTable();
      });
    }
  }

  async function init() {
    cacheElements();
    bindEvents();
    switchMode("arena");

    const [healthRes, metaRes, schedRes] = await Promise.allSettled([
      apiFetchJson("/healthz"),
      apiFetchJson("/api/v1/meta"),
      apiFetchJson("/api/v1/schedule/live"),
    ]);

    if (healthRes.status === "fulfilled" && healthRes.value) {
      state.health = healthRes.value;
    }
    if (metaRes.status === "fulfilled" && metaRes.value) {
      state.meta = metaRes.value;
    }
    if (schedRes.status === "fulfilled" && schedRes.value) {
      state.schedule = schedRes.value;
    }

    updateMastheadTelemetry();
    renderTicker();
    renderScheduleBoard();

    populateContextSelects();
    populatePlayerSelect(els.p1Select, "", state.meta.default_p1);
    populatePlayerSelect(els.p2Select, "", state.meta.default_p2);
    ensureSelectHasOption(els.p1Select, state.meta.default_p1);
    ensureSelectHasOption(els.p2Select, state.meta.default_p2);

    populateHistoricalSplits();
    if (state.meta.splits && state.meta.splits.length > 0) {
      fetchHistoricalList(state.meta.splits[0], null);
    }

    // Auto-run default matchup so Arena Stage looks alive immediately
    if (state.meta.default_p1 && state.meta.default_p2 && state.meta.default_p1 !== state.meta.default_p2) {
      await runArenaPredict();
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
