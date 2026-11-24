/* ==========================================================================
   matchpoint.ml — Risograph Scouting Cards, Verdict Banners & Slate Renderers
   Strict 60/30/10 Cream Paper / Riso Teal / Single-Favored-Pink Discipline
   ========================================================================== */

(function (global) {
  "use strict";

  const MP = (global.MP = global.MP || {});

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function safeNum(val, fallback = 0) {
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
  }

  function formatPct(prob, decimals = 1) {
    return `${(safeNum(prob, 0) * 100).toFixed(decimals)}%`;
  }

  function formatSignedPct(val, decimals = 1) {
    const pct = safeNum(val, 0) * 100;
    const sign = pct > 0 ? "+" : "";
    return `${sign}${pct.toFixed(decimals)}%`;
  }

  function surfaceBadgeClass(surface) {
    const s = String(surface || "").toLowerCase();
    if (s === "clay") return "surface-clay";
    if (s === "grass") return "surface-grass";
    if (s === "carpet") return "surface-carpet";
    return "surface-hard";
  }

  function getStatusPillMeta(status, snapshotLocked) {
    const s = String(status || "").toLowerCase();
    if (s.includes("fallback")) return { cls: "status-locked", label: "FALLBACK" };
    if (s.includes("live")) return { cls: "status-live", label: "LIVE" };
    if (snapshotLocked || s.includes("lock")) return { cls: "status-locked", label: "LOCKED" };
    if (s.includes("final") || s.includes("complete") || s.includes("finish")) {
      return { cls: "status-final", label: "FINAL" };
    }
    return { cls: "status-scheduled", label: "SCHEDULED" };
  }

  function buildSingleFavoredProbBar(p1Prob, ariaLabel) {
    const clampedP1 = Math.max(0, Math.min(1, safeNum(p1Prob, 0.5)));
    const p1Pct = (clampedP1 * 100).toFixed(2);
    const p2Pct = ((1 - clampedP1) * 100).toFixed(2);
    const p1Favored = clampedP1 >= 0.5;
    const leftCls = p1Favored ? "prob-fill-favored" : "prob-fill-underdog";
    const rightCls = p1Favored ? "prob-fill-underdog" : "prob-fill-favored";
    return `
      <div class="prob-track" aria-label="${escapeHtml(ariaLabel)}">
        <div class="${leftCls}" style="width:${p1Pct}%"></div>
        <div class="${rightCls}" style="width:${p2Pct}%"></div>
      </div>
    `;
  }

  function animateOvrCounter(el, targetOvr) {
    if (!el) return;
    const goal = Math.max(40, Math.min(99, Math.round(safeNum(targetOvr, 50))));
    const start = Math.max(40, goal - 16);
    const duration = 380;
    const t0 = performance.now();

    function step(now) {
      const progress = Math.min(1, (now - t0) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (goal - start) * eased);
      el.textContent = String(current);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = String(goal);
      }
    }
    requestAnimationFrame(step);
  }

  function renderHistoricalBanner(container, data) {
    if (!container) return;
    if (!data || !data.is_historical) {
      container.classList.add("is-hidden");
      container.innerHTML = "";
      return;
    }

    container.classList.remove("is-hidden");
    const accurate = Boolean(data.prediction_accurate);
    container.classList.toggle("is-upset", !accurate);

    const verdictCls = accurate ? "verdict-accurate" : "verdict-upset";
    const verdictText = accurate ? "PREDICTION ACCURATE" : "UPSET";
    const winner = escapeHtml(data.actual_winner || "Unknown");
    const score = escapeHtml(data.score_str ?? "Score N/A");
    const dateStr = escapeHtml(data.date_str || "N/A");
    const tournament = escapeHtml(data.tournament || "ATP Tour");
    const roundName = escapeHtml(data.round_name || "Main Draw");

    let bmHtml = "";
    if (typeof data.bm_prob === "number" && Number.isFinite(data.bm_prob)) {
      const bmP1 = Math.max(0, Math.min(1, data.bm_prob));
      const bmP2 = 1 - bmP1;
      bmHtml = `
        <div class="model-bar-row">
          <div class="model-bar-title">
            <span>MARKET IMPLIED CLOSING &bull; <span class="player-name-nowrap">${escapeHtml(data.p1)}</span>: <strong class="metric-tabular ${bmP1 >= 0.5 ? "inline-pink" : ""}">${formatPct(bmP1)}</strong></span>
            <span><span class="player-name-nowrap">${escapeHtml(data.p2)}</span>: <strong class="metric-tabular ${bmP2 > 0.5 ? "inline-pink" : ""}">${formatPct(bmP2)}</strong></span>
          </div>
          ${buildSingleFavoredProbBar(bmP1, "Bookmaker Implied Probability")}
        </div>
      `;
    }

    container.innerHTML = `
      <div class="hist-banner-top">
        <span class="badge ${verdictCls}">${verdictText}</span>
        <div class="hist-match-facts">
          <span>EVENT: <strong>${tournament}</strong> (${roundName})</span>
          <span>DATE: <strong class="metric-tabular">${dateStr}</strong></span>
          <span>WINNER: <strong class="player-name-nowrap">${winner}</strong></span>
          <span>SCORE: <strong class="metric-tabular">${score}</strong></span>
        </div>
      </div>
      ${bmHtml}
    `;
  }

  function renderSplitH2HBanner(container, data) {
    if (!container || !data) return;
    const p1Prob = Math.max(0, Math.min(1, safeNum(data.prob_lgb, 0.5)));
    const p2Prob = 1 - p1Prob;
    const p1Favored = p1Prob >= 0.5;
    const surfCls = surfaceBadgeClass(data.surface);
    const symScore = safeNum(data.sym_score, 0).toFixed(6);
    const rawAsym = safeNum(data.raw_asym, 0).toFixed(4);
    const divPct = (safeNum(data.div_delta, 0) * 100).toFixed(1);
    const isDivergent = Boolean(data.is_divergent || safeNum(data.div_delta, 0) > 0.15);
    const coldStart = data.p1_known === false || data.p2_known === false;
    const coldLabel =
      data.p1_known === false && data.p2_known === false
        ? "COLD START (P1 & P2)"
        : data.p1_known === false
          ? "COLD START (P1)"
          : "COLD START (P2)";

    let evBadge = `<span class="badge badge-neutral">Model Only &bull; No Market Odds</span>`;
    if (data.edge) {
      const sig = String(data.edge.signal || "");
      const e1 = safeNum(data.edge.edge_p1, 0);
      const e2 = safeNum(data.edge.edge_p2, 0);
      if (sig === "VALUE_P1") {
        evBadge = `<span class="badge badge-ev">+${(e1 * 100).toFixed(1)}% EV (${escapeHtml(data.p1)})</span>`;
      } else if (sig === "VALUE_P2") {
        evBadge = `<span class="badge badge-ev">+${(e2 * 100).toFixed(1)}% EV (${escapeHtml(data.p2)})</span>`;
      } else if (sig === "NEGATIVE") {
        const favEdge = p1Favored ? e1 : e2;
        evBadge = `<span class="badge badge-fade">${(favEdge * 100).toFixed(1)}% Market Fade</span>`;
      } else {
        const favEdge = p1Favored ? e1 : e2;
        evBadge = `<span class="badge badge-neutral metric-tabular">${favEdge >= 0 ? "+" : ""}${(favEdge * 100).toFixed(1)}% Fair Line</span>`;
      }
    }

    const divBadge = isDivergent
      ? `<span class="divergent-badge">DIVERGENT &gt;15% (${divPct}%)</span>`
      : `<span class="badge badge-neutral metric-tabular">CONSENSUS (&Delta; ${divPct}%)</span>`;

    container.innerHTML = `
      <div class="verdict-meta-row">
        <span class="surface-badge ${surfCls}">${escapeHtml(data.surface)} Court</span>
        <span class="series-badge">${escapeHtml(data.series)}</span>
        ${evBadge}
        ${divBadge}
        ${coldStart ? `<span class="cold-start-badge">${escapeHtml(coldLabel)}</span>` : ""}
        <span id="symmetryChip" data-testid="symmetry-chip" class="symmetry-chip" title="Raw asymmetry before dual-pass averaging: ${rawAsym}">
          SYM &Delta; ${symScore} &bull; VERIFIED
        </span>
      </div>

      <div class="h2h-players-row">
        <div class="banner-side left ${p1Favored ? "is-favored" : ""}">
          <span class="banner-player-tag">PLAYER 1 &bull; ${p1Favored ? "FAVORED" : "UNDERDOG"}</span>
          <div class="banner-player-name player-name-nowrap">${escapeHtml(data.p1)}</div>
          <div class="banner-win-prob" data-prob="p1">${formatPct(p1Prob)}</div>
        </div>

        <span class="vs-emblem">VS</span>

        <div class="banner-side right ${!p1Favored ? "is-favored" : ""}">
          <span class="banner-player-tag">PLAYER 2 &bull; ${!p1Favored ? "FAVORED" : "UNDERDOG"}</span>
          <div class="banner-player-name player-name-nowrap">${escapeHtml(data.p2)}</div>
          <div class="banner-win-prob" data-prob="p2">${formatPct(p2Prob)}</div>
        </div>
      </div>
    `;
  }

  function renderSingleTradingCard(el, card, isKnown, sideLabel, isFavored) {
    if (!el) return;
    const c = card || {};
    const ovr = Math.max(40, Math.min(99, Math.round(safeNum(c.ovr, 50))));
    const rank = safeNum(c.rank, 500);
    const pts = Math.round(safeNum(c.pts, 0));
    const traits = Array.isArray(c.traits) ? c.traits : [];

    const traitsHtml = traits.length
      ? traits
          .map((t) => `<span class="trait-chip">${escapeHtml(t)}</span>`)
          .join("")
      : `<span class="trait-chip">BASELINE PROFILE</span>`;

    const knownBadge =
      isKnown === false ? `<span class="cold-start-badge">COLD START</span>` : "";

    el.innerHTML = `
      <div class="arcade-card-top">
        <div class="ovr-shield">
          <span class="ovr-num metric-tabular ${isFavored ? "is-favored" : ""}" data-ovr-target="${ovr}">${ovr}</span>
          <span class="ovr-label">OVR</span>
        </div>
        <div class="card-identity">
          <h3 class="card-player-name player-name-nowrap">${escapeHtml(c.name || sideLabel)}</h3>
          <div class="card-rank-line">
            <span>RANK <strong class="metric-tabular">#${escapeHtml(rank)}</strong></span>
            <span>PTS <strong class="metric-tabular">${escapeHtml(pts.toLocaleString())}</strong></span>
            <span>${isFavored ? '<strong class="inline-pink">Favored</strong>' : '<span class="inline-muted">Underdog</span>'}</span>
            ${knownBadge}
          </div>
        </div>
      </div>

      <div class="card-stats-grid">
        <div class="card-stat-cell">
          <span class="card-stat-label">SURF WR</span>
          <strong class="card-stat-val metric-tabular">${formatPct(c.surface_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">10M FORM</span>
          <strong class="card-stat-val metric-tabular">${formatPct(c.short_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">CAREER</span>
          <strong class="card-stat-val metric-tabular">${formatPct(c.career_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">REST</span>
          <strong class="card-stat-val metric-tabular">${Math.round(safeNum(c.days_since_last, 14))}d</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">14D LOAD</span>
          <strong class="card-stat-val metric-tabular">${Math.round(safeNum(c.matches_14d, 0))}M</strong>
        </div>
      </div>

      <div class="card-traits-row" aria-label="Playstyle Traits">
        ${traitsHtml}
      </div>
    `;

    animateOvrCounter(el.querySelector(".ovr-num"), ovr);
  }

  function renderTradingCardsDeck(cardP1El, cardP2El, data) {
    if (!data) return;
    const cards = data.player_cards || {};
    const p1Favored = safeNum(data.prob_lgb, 0.5) >= 0.5;
    renderSingleTradingCard(cardP1El, cards.p1 || { name: data.p1 }, data.p1_known, data.p1, p1Favored);
    renderSingleTradingCard(cardP2El, cards.p2 || { name: data.p2 }, data.p2_known, data.p2, !p1Favored);
  }

  function renderModelProbBars(container, data) {
    if (!container || !data) return;
    const lgbP1 = Math.max(0, Math.min(1, safeNum(data.prob_lgb, 0.5)));
    const lgbP2 = 1 - lgbP1;
    const lrP1 = Math.max(0, Math.min(1, safeNum(data.prob_lr, 0.5)));
    const lrP2 = 1 - lrP1;

    container.innerHTML = `
      <div class="model-bar-row">
        <div class="model-bar-title">
          <span>LIGHTGBM CHAMPION</span>
          <span class="metric-tabular">
            <strong class="${lgbP1 >= 0.5 ? "inline-pink" : "inline-muted"}">${formatPct(lgbP1)}</strong>
            /
            <strong class="${lgbP2 > 0.5 ? "inline-pink" : "inline-muted"}">${formatPct(lgbP2)}</strong>
          </span>
        </div>
        ${buildSingleFavoredProbBar(lgbP1, "LightGBM Champion win probability split")}
      </div>

      <div class="model-bar-row">
        <div class="model-bar-title">
          <span>LOGREG BASELINE</span>
          <span class="metric-tabular">
            <strong class="${lrP1 >= 0.5 ? "inline-pink" : "inline-muted"}">${formatPct(lrP1)}</strong>
            /
            <strong class="${lrP2 > 0.5 ? "inline-pink" : "inline-muted"}">${formatPct(lrP2)}</strong>
          </span>
        </div>
        ${buildSingleFavoredProbBar(lrP1, "Logistic Regression baseline win probability split")}
      </div>
    `;
  }

  function renderKellyEdgePanel(container, data) {
    if (!container) return;
    const edge = data ? data.edge : null;
    if (!edge) {
      container.innerHTML = `
        <div class="kelly-prompt">
          Enter Decimal Odds &gt; 1.0 in H2H Arena controls to unlock Power-Method De-Vig &amp; Quarter-Kelly sizing (2.5% cap).
        </div>
      `;
      return;
    }

    const signal = String(edge.signal || "ALIGNMENT");
    let signalCls = "badge-neutral";
    if (signal === "VALUE_P1" || signal === "VALUE_P2") signalCls = "badge-ev";
    else if (signal === "NEGATIVE") signalCls = "badge-fade";

    const kellyP1 = Math.min(2.5, Math.max(0, safeNum(edge.kelly_p1_pct, 0)));
    const kellyP2 = Math.min(2.5, Math.max(0, safeNum(edge.kelly_p2_pct, 0)));
    const maxKelly = Math.max(kellyP1, kellyP2);
    const meterWidth = Math.min(100, (maxKelly / 2.5) * 100);

    container.innerHTML = `
      <div class="hist-banner-top">
        <span class="kpi-label">POWER-METHOD DE-VIG &amp; QUARTER-KELLY SIZING</span>
        <span class="badge ${signalCls}">${escapeHtml(signal)}</span>
      </div>
      <div class="kelly-strip">
        <div class="kelly-cell">
          <span>De-Vig Implied</span>
          <strong class="metric-tabular">${formatPct(edge.implied_prob_p1)} / ${formatPct(edge.implied_prob_p2)}</strong>
        </div>
        <div class="kelly-cell">
          <span>Overround / k</span>
          <strong class="metric-tabular">${safeNum(edge.overround, 1).toFixed(3)} / ${safeNum(edge.power_k, 1).toFixed(3)}</strong>
        </div>
        <div class="kelly-cell">
          <span>Model Edge (P1/P2)</span>
          <strong class="metric-tabular ${edge.edge_p1 >= 0.03 || edge.edge_p2 >= 0.03 ? "inline-pink" : ""}">
            ${formatSignedPct(edge.edge_p1)} / ${formatSignedPct(edge.edge_p2)}
          </strong>
        </div>
        <div class="kelly-cell">
          <span>1/4 Kelly (Cap 2.5%)</span>
          <strong class="metric-tabular ${maxKelly > 0 ? "inline-pink" : ""}">${kellyP1.toFixed(2)}% / ${kellyP2.toFixed(2)}%</strong>
        </div>
      </div>
      <div class="kelly-meter-track" title="Quarter-Kelly stake intensity relative to 2.5% cap">
        <div class="kelly-meter-fill" style="width:${meterWidth.toFixed(1)}%"></div>
      </div>
    `;
  }

  function renderKeyStatsTiles(container, data) {
    if (!container || !data) return;
    const ks = data.key_stats || {};
    const rankDiff = safeNum(ks.rank_diff, 0);
    const surfDiff = safeNum(ks.surface_winrate_diff, 0);
    const formDivDiff = safeNum(ks.form_divergence_diff, 0);
    const load14Diff = safeNum(ks.matches_14d_diff, 0);
    const h2hP1 = Math.round(safeNum(ks.h2h_p1_wins, 0));
    const h2hP2 = Math.round(safeNum(ks.h2h_p2_wins, 0));

    const p1Favored = safeNum(data.prob_lgb, 0.5) >= 0.5;
    const favorsColor = (favorsP1) => {
      if (favorsP1 === null) return "fav-neutral";
      return favorsP1 === p1Favored ? "fav-favored" : "fav-underdog";
    };

    const rankFavCls = rankDiff === 0 ? "fav-neutral" : favorsColor(rankDiff < 0);
    const rankSub = rankDiff < 0 ? "Favors P1" : rankDiff > 0 ? "Favors P2" : "Even";

    const surfFavCls = surfDiff === 0 ? "fav-neutral" : favorsColor(surfDiff > 0);
    const formFavCls = formDivDiff === 0 ? "fav-neutral" : favorsColor(formDivDiff > 0);
    const loadFavCls = load14Diff === 0 ? "fav-neutral" : favorsColor(load14Diff < 0);
    const loadSub = load14Diff > 0 ? "P1 heavier load" : load14Diff < 0 ? "P2 heavier load" : "Equal load";
    const h2hFavCls = h2hP1 === h2hP2 ? "fav-neutral" : favorsColor(h2hP1 > h2hP2);

    container.innerHTML = `
      <div class="delta-pill">
        <span class="delta-tile-label">Rank &Delta; (P1&minus;P2)</span>
        <strong class="delta-tile-val metric-tabular ${rankFavCls}">${rankDiff > 0 ? "+" : ""}${rankDiff.toFixed(0)}</strong>
        <span class="delta-tile-sub">${rankSub}</span>
      </div>
      <div class="delta-pill">
        <span class="delta-tile-label">${escapeHtml(data.surface || "Surface")} WR &Delta;</span>
        <strong class="delta-tile-val metric-tabular ${surfFavCls}">${formatSignedPct(surfDiff)}</strong>
        <span class="delta-tile-sub">Court edge</span>
      </div>
      <div class="delta-pill">
        <span class="delta-tile-label">Form Div &Delta;</span>
        <strong class="delta-tile-val metric-tabular ${formFavCls}">${formatSignedPct(formDivDiff)}</strong>
        <span class="delta-tile-sub">Short &minus; Career</span>
      </div>
      <div class="delta-pill">
        <span class="delta-tile-label">Fatigue 14D &Delta;</span>
        <strong class="delta-tile-val metric-tabular ${loadFavCls}">${load14Diff > 0 ? "+" : ""}${load14Diff.toFixed(0)}M</strong>
        <span class="delta-tile-sub">${loadSub}</span>
      </div>
      <div class="delta-pill">
        <span class="delta-tile-label">H2H Record</span>
        <strong class="delta-tile-val metric-tabular ${h2hFavCls}">${h2hP1} &ndash; ${h2hP2}</strong>
        <span class="delta-tile-sub">P1 &ndash; P2 Count</span>
      </div>
    `;
  }

  function renderScheduleCard(fx, index = 0) {
    const statusMeta = getStatusPillMeta(fx.status, fx.snapshot_locked);
    const surfCls = surfaceBadgeClass(fx.surface);
    const lgbP1 = Math.max(0, Math.min(1, safeNum(fx.prob_lgb, 0.5)));
    const lgbP2 = 1 - lgbP1;
    const lrP1 = Math.max(0, Math.min(1, safeNum(fx.prob_lr, 0.5)));
    const lrP2 = 1 - lrP1;
    const p1Favored = lgbP1 >= 0.5;

    const favoredName = p1Favored ? fx.p1 : fx.p2;
    const underdogName = p1Favored ? fx.p2 : fx.p1;
    const favProb = p1Favored ? lgbP1 : lgbP2;
    const dogProb = p1Favored ? lgbP2 : lgbP1;
    const favOdd = p1Favored ? fx.odd_1 : fx.odd_2;
    const dogOdd = p1Favored ? fx.odd_2 : fx.odd_1;

    const hasEdgeVal = fx.edge_lgb_p1 !== null && fx.edge_lgb_p1 !== undefined;
    const favEdge = hasEdgeVal ? (p1Favored ? Number(fx.edge_lgb_p1) : -Number(fx.edge_lgb_p1)) : null;
    const hasValueEdge = favEdge !== null && Math.abs(Number(fx.edge_lgb_p1)) >= 0.03;

    let edgeBadge = `<span class="badge badge-neutral">No Odds</span>`;
    if (hasEdgeVal) {
      if (Number(fx.edge_lgb_p1) >= 0.03 || Number(fx.edge_lgb_p1) <= -0.03) {
        const bestEdge = Math.max(Number(fx.edge_lgb_p1), -Number(fx.edge_lgb_p1));
        edgeBadge = `<span class="badge badge-ev">+${(bestEdge * 100).toFixed(1)}% EV</span>`;
      } else if (favEdge <= -0.015) {
        edgeBadge = `<span class="badge badge-fade">${(favEdge * 100).toFixed(1)}% Fade</span>`;
      } else {
        edgeBadge = `<span class="badge badge-neutral metric-tabular">${favEdge >= 0 ? "+" : ""}${(favEdge * 100).toFixed(1)}% Fair</span>`;
      }
    }

    const kellyStr =
      fx.kelly_p1_pct !== null && fx.kelly_p1_pct !== undefined
        ? `${Math.min(2.5, safeNum(fx.kelly_p1_pct, 0)).toFixed(2)}%`
        : "—";

    const favOddStr = favOdd !== null && favOdd !== undefined ? safeNum(favOdd).toFixed(2) : null;
    const dogOddStr = dogOdd !== null && dogOdd !== undefined ? safeNum(dogOdd).toFixed(2) : null;

    const lrFavProb = p1Favored ? lrP1 : lrP2;
    const lrDogProb = 1 - lrFavProb;
    const usePrintPlate = index === 0;

    return `
      <button type="button" class="riso-card ${usePrintPlate ? "print-plate" : ""} schedule-card" data-fixture-id="${escapeHtml(fx.fixture_id)}" aria-label="Open dossier for ${escapeHtml(fx.p1)} vs ${escapeHtml(fx.p2)}">
        <div class="schedule-card-top">
          <div>
            <div class="schedule-meta-title">${escapeHtml(fx.tournament)}</div>
            <div class="schedule-meta-round">${escapeHtml(fx.round_name)}${fx.status_detail ? ` &bull; ${escapeHtml(fx.status_detail)}` : ""}</div>
          </div>
          <span class="status-pill ${statusMeta.cls}">${statusMeta.label}${fx.snapshot_locked ? " &#128274;" : ""}</span>
        </div>

        <div class="schedule-badges-row">
          <span class="surface-badge ${surfCls}">${escapeHtml(fx.surface)}</span>
          <span class="series-badge">${escapeHtml(fx.series)}</span>
          ${fx.cold_start ? `<span class="cold-start-badge">COLD START</span>` : ""}
          ${fx.is_divergent ? `<span class="divergent-badge">DIVERGENT &gt;15% (&Delta;${(safeNum(fx.div_delta) * 100).toFixed(1)}%)</span>` : ""}
          ${edgeBadge}
        </div>

        <div class="schedule-matchup-box">
          <div class="matchup-player-row">
            <span class="matchup-player-name player-name-nowrap">${escapeHtml(favoredName)}</span>
            <span class="matchup-odds-chip metric-tabular inline-pink">${formatPct(favProb)}${favOddStr ? ` (Odds ${favOddStr})` : ""}</span>
          </div>
          <div class="matchup-player-row">
            <span class="matchup-player-name is-underdog player-name-nowrap">${escapeHtml(underdogName)}</span>
            <span class="matchup-odds-chip metric-tabular inline-muted">${formatPct(dogProb)}${dogOddStr ? ` (Odds ${dogOddStr})` : ""}</span>
          </div>
          ${buildSingleFavoredProbBar(favProb, "Win probability split")}
        </div>

        <div class="schedule-card-footer">
          <span class="metric-tabular">LR Split: ${formatPct(lrFavProb)} / ${formatPct(lrDogProb)}</span>
          ${hasValueEdge ? `<span>1/4 Kelly: <strong class="metric-tabular inline-pink">${kellyStr}</strong></span>` : ""}
          <span class="inline-pink">Open Dossier &rarr;</span>
        </div>
      </button>
    `;
  }

  MP.escapeHtml = escapeHtml;
  MP.safeNum = safeNum;
  MP.formatPct = formatPct;
  MP.formatSignedPct = formatSignedPct;
  MP.surfaceBadgeClass = surfaceBadgeClass;
  MP.getStatusPillMeta = getStatusPillMeta;
  MP.renderHistoricalBanner = renderHistoricalBanner;
  MP.renderSplitH2HBanner = renderSplitH2HBanner;
  MP.renderTradingCardsDeck = renderTradingCardsDeck;
  MP.renderModelProbBars = renderModelProbBars;
  MP.renderKellyEdgePanel = renderKellyEdgePanel;
  MP.renderKeyStatsTiles = renderKeyStatsTiles;
  MP.renderScheduleCard = renderScheduleCard;
})(window);
