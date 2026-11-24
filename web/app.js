/* ==========================================================================
   matchpoint.ml — Hash SPA Router, Unified Dossier & API Controller
   Pure Vanilla ES6 JS • Preserves All FastAPI Contracts & ML Telemetry
   ========================================================================== */

(function (global) {
  "use strict";

  const MP = (global.MP = global.MP || {});
  const esc = MP.escapeHtml;

  const FALLBACK_SPECS = [
    {
      fixture_id: "fallback-alcaraz-sinner",
      tournament: "ATP Finals • Turin (Spotlight)",
      round_name: "Semifinal",
      surface: "Hard",
      series: "Masters Cup",
      p1: "Alcaraz C.",
      p2: "Sinner J.",
      odd_1: 1.62,
      odd_2: 2.35,
      status: "Deterministic Fallback",
      status_detail: "Spotlight Pair",
    },
    {
      fixture_id: "fallback-sinner-djokovic",
      tournament: "Australian Open (Spotlight)",
      round_name: "Semifinal",
      surface: "Hard",
      series: "Grand Slam",
      p1: "Sinner J.",
      p2: "Djokovic N.",
      odd_1: 1.60,
      odd_2: 2.42,
      status: "Deterministic Fallback",
      status_detail: "Spotlight Pair",
    },
    {
      fixture_id: "fallback-ruud-tsitsipas",
      tournament: "Monte-Carlo Masters (Spotlight)",
      round_name: "Quarterfinal",
      surface: "Clay",
      series: "Masters 1000",
      p1: "Ruud C.",
      p2: "Tsitsipas S.",
      odd_1: 1.80,
      odd_2: 2.02,
      status: "Deterministic Fallback",
      status_detail: "Spotlight Pair",
    },
    {
      fixture_id: "fallback-zverev-rublev",
      tournament: "Vienna Open (Spotlight)",
      round_name: "Quarterfinal",
      surface: "Hard",
      series: "ATP500",
      p1: "Zverev A.",
      p2: "Rublev A.",
      odd_1: null,
      odd_2: null,
      status: "Deterministic Fallback",
      status_detail: "Model Only",
    },
  ];

  const ARENA_PRESETS = [
    {
      label: "Federer R. vs Nadal R.",
      tag: "Clay Classic",
      p1: "Federer R.",
      p2: "Nadal R.",
      surface: "Clay",
      series: "Grand Slam",
      odd1: 3.10,
      odd2: 1.40,
      legend: true,
    },
    {
      label: "Sinner J. vs Djokovic N.",
      tag: "Divergence Check",
      p1: "Sinner J.",
      p2: "Djokovic N.",
      surface: "Hard",
      series: "Grand Slam",
      odd1: 1.60,
      odd2: 2.42,
      legend: false,
    },
    {
      label: "Alcaraz C. vs Sinner J.",
      tag: "Default Spotlight",
      p1: "Alcaraz C.",
      p2: "Sinner J.",
      surface: "Hard",
      series: "Grand Slam",
      odd1: 1.85,
      odd2: 2.05,
      legend: false,
    },
  ];

  const state = {
    route: "slate", // "slate" | "match" | "arena" | "backtracker"
    selectedMatchId: null,
    activeMatchFixture: null,
    dossierTab: "drivers", // "drivers" | "scouting" | "matrix"
    arenaRailOpen: false,
    btRailOpen: true,
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
    fallbackFixtures: [],
    usingFallbackSlate: false,
    dossierCache: {},
    scheduleSeriesFilter: "all",
    scheduleOnlyEdges: false,
    allTimePool: false,
    arenaResult: null,
    histResult: null,
    activeStageData: null,
    matrixCategory: "All",
    matrixExpanded: true,
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
      "panelMatch",
      "panelArena",
      "panelBacktracker",
      "matchDossierMount",
      "arenaDossierMount",
      "btDossierMount",
      "backToSlateBtn",
      "openInWorkbenchBtn",
      "matchHeroTitle",
      "matchHeroSubtitle",
      "refreshScheduleBtn",
      "kpiTotalFixtures",
      "kpiLiveCount",
      "kpiValueEdges",
      "kpiDivergentCount",
      "kpiModelMetrics",
      "kpiSyncTime",
      "onlyEdgeToggle",
      "onlyEdgeToggleLabel",
      "scheduleNoticeBanner",
      "scheduleCardGrid",
      "scheduleEmptyState",
      "emptyRefreshBtn",
      "emptyGoArenaBtn",
      "arenaRailToggleBtn",
      "arenaCollapsedSummary",
      "arenaWorkbenchGrid",
      "arenaRail",
      "allTimeToggle",
      "allTimeToggleLabel",
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
      "arenaPresets",
      "btRailToggleBtn",
      "btCollapsedSummary",
      "btWorkbenchGrid",
      "btRail",
      "histSplitSelect",
      "histTournamentSelect",
      "histMatchSelect",
      "histInspectBtn",
      "histErrorBanner",
      "btSplitMeta",
      "arenaStage",
      "historicalResultBanner",
      "histStandbyPrompt",
      "stageContent",
      "splitH2HBanner",
      "playerCardP1",
      "playerCardP2",
      "radarSwatchP1",
      "radarSwatchP2",
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
      "subtabBtnDrivers",
      "subtabBtnScouting",
      "subtabBtnMatrix",
      "treeshapPanel",
      "scoutingTabPanel",
      "featuresMatrixSection",
      "toggleFeaturesMatrixBtn",
      "featuresMatrixBody",
      "matrixCategoryPills",
      "matrixCountBadge",
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
        valEl.textContent = h.models_loaded === false ? "STANDBY" : "LGBM + LR";
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
        valEl.textContent = `${act} / ${tot}`;
      }
    }
  }

  /* ========================================================================
     Rail Collapse & Summary Sync (#arena & #backtracker)
     ======================================================================== */
  function syncArenaRailUI() {
    if (els.arenaWorkbenchGrid) {
      els.arenaWorkbenchGrid.classList.toggle("is-rail-collapsed", !state.arenaRailOpen);
    }
    if (els.arenaRailToggleBtn) {
      els.arenaRailToggleBtn.setAttribute("aria-expanded", state.arenaRailOpen ? "true" : "false");
      els.arenaRailToggleBtn.innerHTML = state.arenaRailOpen
        ? "&times; Hide Matchup Sidebar"
        : "&#9776; Show Player &amp; Matchup Controls";
    }
    if (els.arenaCollapsedSummary) {
      els.arenaCollapsedSummary.classList.toggle("is-hidden", state.arenaRailOpen);
      const p1 = els.p1Select ? els.p1Select.value : state.meta.default_p1;
      const p2 = els.p2Select ? els.p2Select.value : state.meta.default_p2;
      const surf = els.surfaceSelect ? els.surfaceSelect.value : "Hard";
      const ser = els.seriesSelect ? els.seriesSelect.value : "Grand Slam";
      const o1 = parseOptionalOdd(els.odd1Input);
      const o2 = parseOptionalOdd(els.odd2Input);
      const oddsTxt = o1 && o2 && o1 > 1 && o2 > 1 ? ` • Odds ${o1.toFixed(2)} / ${o2.toFixed(2)}` : "";
      els.arenaCollapsedSummary.innerHTML = `Active: <strong class="player-name-nowrap" style="color:var(--ink-deep);">${esc(p1)} vs ${esc(p2)}</strong> &bull; ${esc(surf)} &bull; ${esc(ser)}${esc(oddsTxt)}`;
    }
  }

  function syncBtRailUI() {
    if (els.btWorkbenchGrid) {
      els.btWorkbenchGrid.classList.toggle("is-rail-collapsed", !state.btRailOpen);
    }
    if (els.btRailToggleBtn) {
      els.btRailToggleBtn.setAttribute("aria-expanded", state.btRailOpen ? "true" : "false");
      els.btRailToggleBtn.innerHTML = state.btRailOpen
        ? "&times; Hide Archive Selectors"
        : "&#9776; Show Archive Selectors";
    }
    if (els.btCollapsedSummary) {
      els.btCollapsedSummary.classList.toggle("is-hidden", state.btRailOpen);
      const tourn = els.histTournamentSelect ? els.histTournamentSelect.value : "";
      const mOpt = els.histMatchSelect && els.histMatchSelect.selectedOptions[0];
      const mLabel = mOpt ? mOpt.textContent : "";
      els.btCollapsedSummary.innerHTML = tourn
        ? `Active: <strong style="color:var(--ink-deep);">${esc(tourn)}</strong>${mLabel ? ` &bull; ${esc(mLabel)}` : ""}`
        : "";
    }
  }

  /* ========================================================================
     Progressive Disclosure Tabs inside #arenaStage
     ======================================================================== */
  function setDossierTab(tabName) {
    state.dossierTab = tabName || "drivers";
    const tabs = [
      { key: "drivers", btn: els.subtabBtnDrivers, panel: els.treeshapPanel },
      { key: "scouting", btn: els.subtabBtnScouting, panel: els.scoutingTabPanel },
      { key: "matrix", btn: els.subtabBtnMatrix, panel: els.featuresMatrixSection },
    ];
    tabs.forEach(({ key, btn, panel }) => {
      const active = key === state.dossierTab;
      if (btn) {
        btn.classList.toggle("is-active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
        btn.setAttribute("tabindex", active ? "0" : "-1");
      }
      if (panel) {
        panel.classList.toggle("is-hidden", !active);
      }
    });
  }

  /* ========================================================================
     Hash SPA Router (#slate, #match/<id>, #arena, #backtracker)
     ======================================================================== */
  function parseHash() {
    const raw = (window.location.hash || "#slate").replace(/^#/, "");
    if (raw.startsWith("match/")) {
      return { route: "match", matchId: decodeURIComponent(raw.slice("match/".length)) };
    }
    if (raw === "arena") return { route: "arena", matchId: null };
    if (raw === "backtracker") return { route: "backtracker", matchId: null };
    return { route: "slate", matchId: null };
  }

  function navigateToHash(hashStr) {
    if (window.location.hash !== hashStr) {
      history.pushState(null, "", hashStr);
    }
    syncRoute();
  }

  function mountArenaStageInto(mountEl) {
    if (!els.arenaStage || !mountEl) return;
    if (els.arenaStage.parentElement !== mountEl) {
      mountEl.appendChild(els.arenaStage);
    }
  }

  function syncRoute() {
    const { route, matchId } = parseHash();
    state.route = route;
    state.selectedMatchId = matchId;

    const navContext = route === "match" ? "slate" : route;
    const navTabs = [
      { key: "slate", tab: els.tabSchedule },
      { key: "arena", tab: els.tabArena },
      { key: "backtracker", tab: els.tabBacktracker },
    ];
    navTabs.forEach(({ key, tab }) => {
      const active = key === navContext;
      if (tab) {
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
      }
    });

    const panels = [
      { key: "slate", panel: els.panelSchedule },
      { key: "match", panel: els.panelMatch },
      { key: "arena", panel: els.panelArena },
      { key: "backtracker", panel: els.panelBacktracker },
    ];
    panels.forEach(({ key, panel }) => {
      const active = key === route;
      if (panel) {
        panel.classList.toggle("is-hidden", !active);
        panel.setAttribute("aria-hidden", active ? "false" : "true");
      }
    });

    if (route === "slate") {
      if (els.arenaStage) els.arenaStage.classList.add("is-hidden");
      if (els.historicalResultBanner) els.historicalResultBanner.classList.add("is-hidden");
    } else if (route === "match") {
      mountArenaStageInto(els.matchDossierMount);
      if (els.arenaStage) els.arenaStage.classList.remove("is-hidden");
      if (els.historicalResultBanner) els.historicalResultBanner.classList.add("is-hidden");
      if (els.histStandbyPrompt) els.histStandbyPrompt.classList.add("is-hidden");
      openMatchDossier(matchId);
    } else if (route === "arena") {
      mountArenaStageInto(els.arenaDossierMount);
      if (els.arenaStage) els.arenaStage.classList.remove("is-hidden");
      if (els.historicalResultBanner) els.historicalResultBanner.classList.add("is-hidden");
      if (els.histStandbyPrompt) els.histStandbyPrompt.classList.add("is-hidden");
      syncArenaRailUI();
      if (state.arenaResult) {
        renderArenaStage(state.arenaResult);
      }
    } else if (route === "backtracker") {
      mountArenaStageInto(els.btDossierMount);
      if (els.arenaStage) els.arenaStage.classList.remove("is-hidden");
      syncBtRailUI();
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
     Ticker, Live Schedule & Deterministic Featured Fallback Slate
     ======================================================================== */
  function getActiveSlateFixtures() {
    const live = Array.isArray(state.schedule.fixtures) ? state.schedule.fixtures : [];
    if (live.length > 0) {
      state.usingFallbackSlate = false;
      return live;
    }
    if (state.fallbackFixtures.length > 0) {
      state.usingFallbackSlate = true;
      return state.fallbackFixtures;
    }
    state.usingFallbackSlate = false;
    return [];
  }

  async function buildDeterministicFallbackSlate() {
    const scored = await Promise.all(
      FALLBACK_SPECS.map(async (spec) => {
        try {
          const payload = await apiFetchJson("/api/v1/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              p1: spec.p1,
              p2: spec.p2,
              surface: spec.surface,
              series: spec.series,
              odd_1: spec.odd_1,
              odd_2: spec.odd_2,
            }),
          });
          state.dossierCache[spec.fixture_id] = payload;
          const edge = payload.edge || null;
          return {
            fixture_id: spec.fixture_id,
            tournament: spec.tournament,
            round_name: spec.round_name,
            surface: spec.surface,
            series: spec.series,
            status: spec.status,
            status_detail: spec.status_detail,
            p1: spec.p1,
            p2: spec.p2,
            prob_lgb: payload.prob_lgb,
            prob_lr: payload.prob_lr,
            is_divergent: Boolean(payload.is_divergent),
            div_delta: payload.div_delta,
            odd_1: spec.odd_1,
            odd_2: spec.odd_2,
            bm_prob: edge ? edge.implied_prob_p1 : null,
            edge_lgb_p1: edge ? edge.edge_p1 : null,
            kelly_p1_pct: edge ? Math.max(edge.kelly_p1_pct || 0, edge.kelly_p2_pct || 0) : null,
            snapshot_locked: false,
            cold_start: payload.p1_known === false || payload.p2_known === false,
            is_fallback: true,
          };
        } catch (_err) {
          return {
            ...spec,
            prob_lgb: 0.5,
            prob_lr: 0.5,
            is_divergent: false,
            div_delta: 0.0,
            bm_prob: null,
            edge_lgb_p1: null,
            kelly_p1_pct: null,
            snapshot_locked: false,
            cold_start: false,
            is_fallback: true,
          };
        }
      })
    );
    state.fallbackFixtures = scored;
  }

  function renderTicker() {
    if (!els.tickerTrack) return;
    const fixtures = getActiveSlateFixtures();
    if (fixtures.length === 0) {
      els.tickerTrack.innerHTML = `
        <div class="ticker-standby">
          LIVE FEED STANDBY &bull; DETERMINISTIC SPOTLIGHT LOADING &bull; CLICK H2H ARENA FOR CUSTOM INFERENCE
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
            <span class="player-name-nowrap">${esc(fx.p1)}</span>
            <span>vs</span>
            <span class="player-name-nowrap">${esc(fx.p2)}</span>
            <span class="ticker-split">${(lgbP1 * 100).toFixed(0)}% / ${(lgbP2 * 100).toFixed(0)}%</span>
            ${hasVal ? `<span class="value-alert-pill">+EV</span>` : ""}
          </button>
        `;
      })
      .join("");
  }

  function renderScheduleBoard(customNotice = "") {
    const sched = state.schedule || {};
    const fixtures = getActiveSlateFixtures();
    const isFallback = state.usingFallbackSlate;

    const liveCount = isFallback
      ? 0
      : (sched.kpis && typeof sched.kpis.live_count === "number"
          ? sched.kpis.live_count
          : fixtures.filter((f) => String(f.status || "").toLowerCase().includes("live")).length);
    const edgeCount = isFallback
      ? fixtures.filter((f) => f.edge_lgb_p1 !== null && Math.abs(Number(f.edge_lgb_p1)) >= 0.03).length
      : (sched.kpis && typeof sched.kpis.value_edges_count === "number"
          ? sched.kpis.value_edges_count
          : 0);
    const divCount = isFallback
      ? fixtures.filter((f) => Boolean(f.is_divergent)).length
      : (sched.kpis && typeof sched.kpis.divergent_count === "number"
          ? sched.kpis.divergent_count
          : 0);

    if (els.tabScheduleBadge) els.tabScheduleBadge.textContent = String(fixtures.length);
    if (els.kpiTotalFixtures) els.kpiTotalFixtures.textContent = String(fixtures.length);
    if (els.kpiLiveCount) els.kpiLiveCount.textContent = String(liveCount);
    if (els.kpiValueEdges) els.kpiValueEdges.textContent = String(edgeCount);
    if (els.kpiDivergentCount) els.kpiDivergentCount.textContent = String(divCount);

    if (els.kpiModelMetrics) {
      const mm = state.meta.model_metrics || {};
      if (mm.lgbm_auc !== undefined) {
        els.kpiModelMetrics.textContent = `${Number(mm.lgbm_auc).toFixed(3)} / ${Number(mm.lgbm_logloss).toFixed(3)}`;
      } else {
        els.kpiModelMetrics.textContent = "0.717 / 0.615";
      }
    }

    if (els.kpiSyncTime) {
      if (sched.generated_at) {
        els.kpiSyncTime.textContent = String(sched.generated_at).slice(0, 16).replace("T", " ");
      } else {
        els.kpiSyncTime.textContent = formatSyncTimestamp(sched.sync_status || state.meta.sync_status);
      }
    }

    if (customNotice) {
      showBanner(els.scheduleNoticeBanner, customNotice);
    } else if (isFallback) {
      showBanner(
        els.scheduleNoticeBanner,
        "Live ESPN Feed Empty — Deterministic Featured Fallback Slate Active (scored live via LightGBM + LogReg)"
      );
    } else {
      showBanner(els.scheduleNoticeBanner, "");
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
        <div class="riso-card empty-state" style="grid-column: 1 / -1;">
          <span class="badge badge-neutral">FILTER ACTIVE</span>
          <h2>No Fixtures Match Active Filters</h2>
          <p class="meta-caption">Reset the tournament series pill or uncheck Value Edges Only to view all ${fixtures.length} published fixtures.</p>
          <div class="empty-state-actions">
            <button type="button" id="resetSlateFiltersBtn" class="btn-secondary-riso">Reset Slate Filters</button>
          </div>
        </div>
      `;
      const resetBtn = document.getElementById("resetSlateFiltersBtn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          state.scheduleSeriesFilter = "all";
          state.scheduleOnlyEdges = false;
          if (els.onlyEdgeToggle) els.onlyEdgeToggle.checked = false;
          if (els.onlyEdgeToggleLabel) els.onlyEdgeToggleLabel.classList.remove("is-on");
          if (els.panelSchedule) {
            els.panelSchedule.querySelectorAll(".series-pills .filter-pill").forEach((p) => {
              p.classList.toggle("is-active", p.getAttribute("data-series") === "all");
            });
          }
          renderScheduleBoard();
        });
      }
      return;
    }

    els.scheduleCardGrid.innerHTML = filtered.map((fx, idx) => MP.renderScheduleCard(fx, idx)).join("");
  }

  async function refreshSchedule() {
    setButtonLoading(els.refreshScheduleBtn, true);
    setButtonLoading(els.emptyRefreshBtn, true);
    showBanner(els.scheduleNoticeBanner, "");
    try {
      const res = await apiFetchJson("/api/v1/schedule/refresh", { method: "POST" });
      state.schedule = res;
      if ((!res.fixtures || res.fixtures.length === 0) && state.fallbackFixtures.length === 0) {
        await buildDeterministicFallbackSlate();
      }
      renderTicker();
      let notice = "";
      if (res.refresh_error) {
        notice = state.usingFallbackSlate
          ? `Live ESPN Feed Empty — Deterministic Featured Fallback Slate Active (${res.refresh_error})`
          : `ESPN Live Refresh Notice: ${res.refresh_error}`;
      } else if (res.refreshed === false && !state.usingFallbackSlate) {
        notice = "Schedule preserved from local cache (no new ESPN update).";
      }
      renderScheduleBoard(notice);
    } catch (err) {
      if (getActiveSlateFixtures().length === 0) {
        await buildDeterministicFallbackSlate();
      }
      renderTicker();
      renderScheduleBoard(`Schedule refresh fallback: ${err.message}`);
    } finally {
      setButtonLoading(els.refreshScheduleBtn, false);
      setButtonLoading(els.emptyRefreshBtn, false);
    }
  }

  /* ========================================================================
     View 2 (#match/<id>): Editorial Match Dossier
     ======================================================================== */
  async function openMatchDossier(matchId) {
    const fixtures = getActiveSlateFixtures();
    const fx = fixtures.find((item) => item.fixture_id === matchId) || fixtures[0];
    if (!fx) {
      if (els.matchHeroTitle) {
        els.matchHeroTitle.innerHTML = `Matchup dossier<span class="focal-accent">.</span>`;
      }
      return;
    }
    state.activeMatchFixture = fx;

    if (els.matchHeroTitle) {
      els.matchHeroTitle.innerHTML = `<span class="player-name-nowrap">${esc(fx.p1)}</span> vs <span class="player-name-nowrap">${esc(fx.p2)}</span><span class="focal-accent">.</span>`;
    }
    if (els.matchHeroSubtitle) {
      els.matchHeroSubtitle.textContent = `${fx.tournament} (${fx.round_name}) • ${fx.surface} Court • ${fx.series} • Symmetrized Dual-Model Dossier.`;
    }

    const cached = state.dossierCache[fx.fixture_id];
    if (cached) {
      renderArenaStage(cached);
      return;
    }

    try {
      const data = await apiFetchJson("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          p1: fx.p1,
          p2: fx.p2,
          surface: fx.surface || "Hard",
          series: fx.series || "ATP250",
          odd_1: fx.odd_1 ?? null,
          odd_2: fx.odd_2 ?? null,
        }),
      });
      state.dossierCache[fx.fixture_id] = data;
      if (state.route === "match") {
        renderArenaStage(data);
      }
    } catch (_err) {
      // Leave stage readable if request fails
    }
  }

  function loadFixtureIntoArena(fx) {
    if (!fx) return;
    const allPlayers = state.meta.all_players || [];
    const activePlayers = state.meta.active_players || [];
    const needsAllTime =
      (allPlayers.includes(fx.p1) && !activePlayers.includes(fx.p1)) ||
      (allPlayers.includes(fx.p2) && !activePlayers.includes(fx.p2));

    if (needsAllTime && els.allTimeToggle && !els.allTimeToggle.checked) {
      els.allTimeToggle.checked = true;
      state.allTimePool = true;
      if (els.allTimeToggleLabel) els.allTimeToggleLabel.classList.add("is-on");
    }

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

    if (state.dossierCache[fx.fixture_id]) {
      state.arenaResult = state.dossierCache[fx.fixture_id];
    }

    navigateToHash("#arena");
    syncArenaRailUI();
    runArenaPredict();
  }

  /* ========================================================================
     View 3A (#arena): H2H Matchup Arena Selectors, Presets & Prediction
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

  function renderArenaPresets() {
    if (!els.arenaPresets) return;
    const curP1 = els.p1Select ? els.p1Select.value : "";
    const curP2 = els.p2Select ? els.p2Select.value : "";
    const curSurf = els.surfaceSelect ? els.surfaceSelect.value : "";

    els.arenaPresets.innerHTML = ARENA_PRESETS.map((p, idx) => {
      const isSel = p.p1 === curP1 && p.p2 === curP2 && p.surface === curSurf;
      return `
        <button type="button" class="preset-btn ${isSel ? "is-selected" : ""}" data-preset-idx="${idx}">
          <span class="player-name-nowrap">${esc(p.label)}</span>
          <span class="metric-tabular inline-pink">${esc(p.tag)}</span>
        </button>
      `;
    }).join("");
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
    renderArenaPresets();
    syncArenaRailUI();

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
      if (state.route === "arena") {
        renderArenaStage(data);
      }
    } catch (err) {
      showBanner(els.arenaErrorBanner, err.message);
    } finally {
      setButtonLoading(els.predictBtn, false);
    }
  }

  /* ========================================================================
     View 3B (#backtracker): Historical Match Backtracker (2000-2026)
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
    updateBtSplitMeta();
  }

  function updateBtSplitMeta() {
    if (!els.btSplitMeta || !els.histSplitSelect) return;
    const split = els.histSplitSelect.value || "";
    if (split.includes("Holdout")) {
      els.btSplitMeta.textContent = "Chronological holdout (2024–2026) • Champion AUC 0.717 / LogLoss 0.615 • Zero look-ahead leakage.";
    } else {
      els.btSplitMeta.textContent = "Full ATP Tour archive (2000–2026) • Temporal replay with point-in-time feature states.";
    }
  }

  async function fetchHistoricalList(split, tournament = null) {
    if (!split) return;
    showBanner(els.histErrorBanner, "");
    updateBtSplitMeta();
    const reqSeq = (fetchHistoricalList.seq = (fetchHistoricalList.seq || 0) + 1);
    if (!tournament && els.histTournamentSelect) {
      els.histTournamentSelect.innerHTML = `<option value="">Loading tournaments…</option>`;
    }
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
      syncBtRailUI();
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
      syncBtRailUI();
      if (state.route === "backtracker") {
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
     Unified Result Dossier Renderer & 31-Feature Matrix
     ======================================================================== */
  function renderFeaturesMatrixTable() {
    if (!els.features31Tbody) return;
    const data = state.activeStageData;
    const rows = data && Array.isArray(data.features_31) ? data.features_31 : [];
    const cat = state.matrixCategory;
    const p1Favored = data ? MP.safeNum(data.prob_lgb, 0.5) >= 0.5 : true;

    const filtered = rows
      .map((item, idx) => ({ ...item, _idx: idx + 1 }))
      .filter((item) => cat === "All" || item.category === cat);

    if (els.matrixCountBadge) {
      els.matrixCountBadge.textContent = `${filtered.length} / ${rows.length || 31} Predictors`;
    }

    if (filtered.length === 0) {
      els.features31Tbody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align:center; color:var(--ink-primary); padding:18px;">
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

        let toneCls = "fav-neutral";
        if (favors === "p1") toneCls = p1Favored ? "fav-favored" : "fav-underdog";
        else if (favors === "p2") toneCls = !p1Favored ? "fav-favored" : "fav-underdog";

        const valStr = Number.isInteger(val) ? String(val) : val.toFixed(4);

        return `
          <tr>
            <td class="num">${item._idx}</td>
            <td>
              <strong>${esc(item.label || item.name)}</strong>
              <span class="feat-name-sub">${esc(item.name)}</span>
            </td>
            <td><span class="series-badge">${esc(item.category)}</span></td>
            <td class="num">${esc(valStr)}</td>
            <td class="num ${toneCls}">${shap > 0 ? "+" : ""}${shap.toFixed(5)}</td>
            <td class="num">${lrCoef > 0 ? "+" : ""}${lrCoef.toFixed(4)}</td>
            <td><span class="favors-badge ${toneCls}">${favorsLabel}</span></td>
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

    const p1Favored = MP.safeNum(data.prob_lgb, 0.5) >= 0.5;
    if (els.radarLegendP1) els.radarLegendP1.textContent = data.p1 || "P1";
    if (els.radarLegendP2) els.radarLegendP2.textContent = data.p2 || "P2";
    if (els.radarSwatchP1) {
      els.radarSwatchP1.className = `legend-swatch ${p1Favored ? "is-favored" : "is-underdog"}`;
    }
    if (els.radarSwatchP2) {
      els.radarSwatchP2.className = `legend-swatch ${!p1Favored ? "is-favored" : "is-underdog"}`;
    }

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
    window.addEventListener("hashchange", syncRoute);

    if (els.tabSchedule) els.tabSchedule.addEventListener("click", () => navigateToHash("#slate"));
    if (els.tabArena) els.tabArena.addEventListener("click", () => navigateToHash("#arena"));
    if (els.tabBacktracker) els.tabBacktracker.addEventListener("click", () => navigateToHash("#backtracker"));

    if (els.backToSlateBtn) els.backToSlateBtn.addEventListener("click", () => navigateToHash("#slate"));
    if (els.openInWorkbenchBtn) {
      els.openInWorkbenchBtn.addEventListener("click", () => {
        if (state.activeMatchFixture) {
          loadFixtureIntoArena(state.activeMatchFixture);
        } else {
          navigateToHash("#arena");
        }
      });
    }

    if (els.refreshScheduleBtn) els.refreshScheduleBtn.addEventListener("click", refreshSchedule);
    if (els.emptyRefreshBtn) els.emptyRefreshBtn.addEventListener("click", refreshSchedule);
    if (els.emptyGoArenaBtn) els.emptyGoArenaBtn.addEventListener("click", () => navigateToHash("#arena"));

    // Ticker item click -> open Match Dossier (#match/<id>)
    if (els.tickerTrack) {
      els.tickerTrack.addEventListener("click", (e) => {
        const btn = e.target.closest(".ticker-item");
        if (!btn) return;
        const fid = btn.getAttribute("data-fixture-id");
        if (fid) navigateToHash(`#match/${encodeURIComponent(fid)}`);
      });
    }

    // Schedule card click -> open Match Dossier (#match/<id>)
    if (els.scheduleCardGrid) {
      els.scheduleCardGrid.addEventListener("click", (e) => {
        const card = e.target.closest(".schedule-card");
        if (!card) return;
        const fid = card.getAttribute("data-fixture-id");
        if (fid) navigateToHash(`#match/${encodeURIComponent(fid)}`);
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
        if (els.onlyEdgeToggleLabel) {
          els.onlyEdgeToggleLabel.classList.toggle("is-on", state.scheduleOnlyEdges);
        }
        renderScheduleBoard();
      });
    }

    // Collapsible Workbench Rails
    if (els.arenaRailToggleBtn) {
      els.arenaRailToggleBtn.addEventListener("click", () => {
        state.arenaRailOpen = !state.arenaRailOpen;
        syncArenaRailUI();
      });
    }
    if (els.btRailToggleBtn) {
      els.btRailToggleBtn.addEventListener("click", () => {
        state.btRailOpen = !state.btRailOpen;
        syncBtRailUI();
      });
    }

    // Arena controls
    if (els.allTimeToggle) {
      els.allTimeToggle.addEventListener("change", () => {
        state.allTimePool = Boolean(els.allTimeToggle.checked);
        if (els.allTimeToggleLabel) {
          els.allTimeToggleLabel.classList.toggle("is-on", state.allTimePool);
        }
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
        if (state.arenaResult) runArenaPredict();
      });
    }

    if (els.arenaForm) {
      els.arenaForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runArenaPredict();
      });
    }

    if (els.arenaPresets) {
      els.arenaPresets.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-preset-idx]");
        if (!btn) return;
        const idx = Number(btn.getAttribute("data-preset-idx"));
        const p = ARENA_PRESETS[idx];
        if (!p) return;
        loadFixtureIntoArena({
          p1: p.p1,
          p2: p.p2,
          surface: p.surface,
          series: p.series,
          odd_1: p.odd1,
          odd_2: p.odd2,
        });
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

    if (els.histMatchSelect) {
      els.histMatchSelect.addEventListener("change", syncBtRailUI);
    }

    if (els.histInspectBtn) {
      els.histInspectBtn.addEventListener("click", inspectHistoricalMatch);
    }

    // Progressive Disclosure Tabs inside #arenaStage (click + arrow keys)
    const subtabBtns = [els.subtabBtnDrivers, els.subtabBtnScouting, els.subtabBtnMatrix].filter(Boolean);
    subtabBtns.forEach((btn, idx) => {
      btn.addEventListener("click", () => {
        setDossierTab(btn.getAttribute("data-dossier-tab"));
      });
      btn.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const dir = e.key === "ArrowRight" ? 1 : -1;
          const next = subtabBtns[(idx + dir + subtabBtns.length) % subtabBtns.length];
          next.focus();
          setDossierTab(next.getAttribute("data-dossier-tab"));
        }
      });
    });

    // Collapsible 31-Feature Matrix inner body toggle
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
        const pill = e.target.closest(".cat-pill");
        if (!pill) return;
        els.matrixCategoryPills.querySelectorAll(".cat-pill").forEach((p) => p.classList.remove("is-active"));
        pill.classList.add("is-active");
        state.matrixCategory = pill.getAttribute("data-category") || "All";
        renderFeaturesMatrixTable();
      });
    }
  }

  async function init() {
    cacheElements();
    bindEvents();

    if (!window.location.hash) {
      history.replaceState(null, "", "#slate");
    }
    setDossierTab("drivers");
    syncRoute();

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

    if (!state.schedule.fixtures || state.schedule.fixtures.length === 0) {
      await buildDeterministicFallbackSlate();
    }

    renderTicker();
    renderScheduleBoard();

    populateContextSelects();
    populatePlayerSelect(els.p1Select, "", state.meta.default_p1);
    populatePlayerSelect(els.p2Select, "", state.meta.default_p2);
    ensureSelectHasOption(els.p1Select, state.meta.default_p1);
    ensureSelectHasOption(els.p2Select, state.meta.default_p2);
    renderArenaPresets();
    syncArenaRailUI();

    populateHistoricalSplits();
    if (state.meta.splits && state.meta.splits.length > 0) {
      fetchHistoricalList(state.meta.splits[0], null);
    }

    // Pre-score default Arena matchup so Arena & #match routes hydrate cleanly
    if (state.meta.default_p1 && state.meta.default_p2 && state.meta.default_p1 !== state.meta.default_p2) {
      await runArenaPredict();
    }

    syncRoute();
  }

  document.addEventListener("DOMContentLoaded", init);
})(window);
