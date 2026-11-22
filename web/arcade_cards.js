/* ==========================================================================
   MatchPoint.intelligence — Arcade Cards, Split Banners & Tale-of-the-Tape
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
    if (s.includes("live")) return { cls: "status-live", label: "LIVE" };
    if (snapshotLocked || s.includes("lock")) return { cls: "status-locked", label: "LOCKED" };
    if (s.includes("final") || s.includes("complete") || s.includes("finish")) {
      return { cls: "status-final", label: "FINAL" };
    }
    return { cls: "status-scheduled", label: "SCHEDULED" };
  }

  function animateOvrCounter(el, targetOvr) {
    if (!el) return;
    const goal = Math.max(0, Math.min(99, Math.round(safeNum(targetOvr, 50))));
    const start = Math.max(40, goal - 18);
    const duration = 420;
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
    const verdictText = accurate ? "PREDICTION ACCURATE" : "UPSET / DIVERGENT";
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
        <div class="split-bar-group">
          <div class="split-bar-labels">
            <span>MARKET IMPLIED BASELINE &bull; ${escapeHtml(data.p1)}: <strong>${formatPct(bmP1)}</strong></span>
            <span>${escapeHtml(data.p2)}: <strong>${formatPct(bmP2)}</strong></span>
          </div>
          <div class="split-bar-track" aria-label="Market Implied Probability">
            <div class="split-bar-fill-p1" style="width:${(bmP1 * 100).toFixed(2)}%"></div>
            <div class="split-bar-fill-p2" style="width:${(bmP2 * 100).toFixed(2)}%"></div>
          </div>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="hist-banner-top">
        <span class="hist-verdict-badge ${verdictCls}">${verdictText}</span>
        <div class="hist-match-facts">
          <span>EVENT: <strong>${tournament}</strong> (${roundName})</span>
          <span>DATE: <strong>${dateStr}</strong></span>
          <span>WINNER: <strong>${winner}</strong></span>
          <span>SCORE: <strong>${score}</strong></span>
        </div>
      </div>
      ${bmHtml}
    `;
  }

  function renderSplitH2HBanner(container, data) {
    if (!container || !data) return;
    const p1Prob = Math.max(0, Math.min(1, safeNum(data.prob_lgb, 0.5)));
    const p2Prob = 1 - p1Prob;
    const surfCls = surfaceBadgeClass(data.surface);
    const symScore = safeNum(data.sym_score, 0).toFixed(6);
    const rawAsym = safeNum(data.raw_asym, 0).toFixed(4);
    const coldStart = data.p1_known === false || data.p2_known === false;
    const coldLabel =
      data.p1_known === false && data.p2_known === false
        ? "COLD START (P1 & P2)"
        : data.p1_known === false
          ? "COLD START (P1)"
          : "COLD START (P2)";

    container.innerHTML = `
      <div class="banner-side left">
        <span class="banner-player-tag">PLAYER 1 &bull; CYAN</span>
        <div class="banner-player-name">${escapeHtml(data.p1)}</div>
        <div class="banner-win-prob">${formatPct(p1Prob)}</div>
      </div>

      <div class="banner-center">
        <span class="vs-emblem">VS</span>
        <div class="banner-context-pills">
          <span class="surface-badge ${surfCls}">${escapeHtml(data.surface)}</span>
          <span class="series-badge">${escapeHtml(data.series)}</span>
          ${coldStart ? `<span class="cold-start-badge">${escapeHtml(coldLabel)}</span>` : ""}
        </div>
        <div id="symmetryChip" data-testid="symmetry-chip" class="symmetry-chip" title="Raw asymmetry before dual-pass averaging: ${rawAsym}">
          SYM &Delta; ${symScore} &bull; VERIFIED
        </div>
      </div>

      <div class="banner-side right">
        <span class="banner-player-tag">PLAYER 2 &bull; MAGENTA</span>
        <div class="banner-player-name">${escapeHtml(data.p2)}</div>
        <div class="banner-win-prob">${formatPct(p2Prob)}</div>
      </div>
    `;
  }

  function renderSingleTradingCard(el, card, isKnown, sideLabel) {
    if (!el) return;
    const c = card || {};
    const ovr = Math.max(40, Math.min(99, Math.round(safeNum(c.ovr, 50))));
    const rank = safeNum(c.rank, 500);
    const pts = Math.round(safeNum(c.pts, 0));
    const traits = Array.isArray(c.traits) ? c.traits : [];

    const traitsHtml = traits.length
      ? traits
          .map((t) => {
            const cls = String(t).toUpperCase().includes("COLD") ? "trait-chip trait-cold" : "trait-chip";
            return `<span class="${cls}">${escapeHtml(t)}</span>`;
          })
          .join("")
      : `<span class="trait-chip trait-muted">BASELINE PROFILE</span>`;

    const knownBadge =
      isKnown === false ? `<span class="cold-start-badge">COLD START</span>` : "";

    el.innerHTML = `
      <div class="arcade-card-top">
        <div class="ovr-shield">
          <span class="ovr-num" data-ovr-target="${ovr}">${ovr}</span>
          <span class="ovr-label">OVR</span>
        </div>
        <div class="card-identity">
          <h3 class="card-player-name">${escapeHtml(c.name || sideLabel)}</h3>
          <div class="card-rank-line">
            <span>RANK <strong>#${escapeHtml(rank)}</strong></span>
            <span>PTS <strong>${escapeHtml(pts.toLocaleString())}</strong></span>
            ${knownBadge}
          </div>
        </div>
      </div>

      <div class="card-stats-grid">
        <div class="card-stat-cell">
          <span class="card-stat-label">SURF WR</span>
          <strong class="card-stat-val">${formatPct(c.surface_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">10M FORM</span>
          <strong class="card-stat-val">${formatPct(c.short_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">CAREER</span>
          <strong class="card-stat-val">${formatPct(c.career_winrate)}</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">REST</span>
          <strong class="card-stat-val">${Math.round(safeNum(c.days_since_last, 14))}d</strong>
        </div>
        <div class="card-stat-cell">
          <span class="card-stat-label">14D LOAD</span>
          <strong class="card-stat-val">${Math.round(safeNum(c.matches_14d, 0))}M</strong>
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
    renderSingleTradingCard(cardP1El, cards.p1 || { name: data.p1 }, data.p1_known, data.p1);
    renderSingleTradingCard(cardP2El, cards.p2 || { name: data.p2 }, data.p2_known, data.p2);
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
          <span><strong class="fav-p1">${formatPct(lgbP1)}</strong> / <strong class="fav-p2">${formatPct(lgbP2)}</strong></span>
        </div>
        <div class="split-bar-track" aria-label="LightGBM win probability split">
          <div class="split-bar-fill-p1" style="width:${(lgbP1 * 100).toFixed(2)}%"></div>
          <div class="split-bar-fill-p2" style="width:${(lgbP2 * 100).toFixed(2)}%"></div>
        </div>
      </div>

      <div class="model-bar-row">
        <div class="model-bar-title">
          <span>LOGREG BASELINE</span>
          <span><strong class="fav-p1">${formatPct(lrP1)}</strong> / <strong class="fav-p2">${formatPct(lrP2)}</strong></span>
        </div>
        <div class="split-bar-track" aria-label="Logistic Regression win probability split">
          <div class="split-bar-fill-lr-p1" style="width:${(lrP1 * 100).toFixed(2)}%"></div>
          <div class="split-bar-fill-p2" style="width:${(lrP2 * 100).toFixed(2)}%"></div>
        </div>
      </div>
    `;
  }

  function renderKellyEdgePanel(container, data) {
    if (!container) return;
    const edge = data ? data.edge : null;
    if (!edge) {
      container.innerHTML = `
        <div class="kelly-prompt">
          Enter Decimal Odds &gt; 1.0 in Matchup Arena to unlock Power-Method De-Vig &amp; Quarter-Kelly sizing
        </div>
      `;
      return;
    }

    const signal = String(edge.signal || "ALIGNMENT");
    let signalCls = "status-scheduled";
    if (signal === "VALUE_P1" || signal === "VALUE_P2") signalCls = "value-alert-pill";
    else if (signal === "NEGATIVE") signalCls = "status-live";

    const kellyP1 = safeNum(edge.kelly_p1_pct, 0);
    const kellyP2 = safeNum(edge.kelly_p2_pct, 0);
    const maxKelly = Math.max(kellyP1, kellyP2);
    const meterWidth = Math.min(100, (maxKelly / 2.5) * 100);

    container.innerHTML = `
      <div class="hist-banner-top">
        <span class="kpi-label">POWER-METHOD DE-VIG &amp; QUARTER-KELLY SIZING</span>
        <span class="${signalCls}">${escapeHtml(signal)}</span>
      </div>
      <div class="kelly-grid">
        <div class="kelly-metric">
          <span>DE-VIG IMPLIED</span>
          <strong>${formatPct(edge.implied_prob_p1)} / ${formatPct(edge.implied_prob_p2)}</strong>
        </div>
        <div class="kelly-metric">
          <span>OVERROUND / K</span>
          <strong>${safeNum(edge.overround, 1).toFixed(3)} / ${safeNum(edge.power_k, 1).toFixed(3)}</strong>
        </div>
        <div class="kelly-metric">
          <span>MODEL EDGE (P1 / P2)</span>
          <strong class="${edge.edge_p1 >= 0.03 || edge.edge_p2 >= 0.03 ? "edge-highlight" : ""}">
            ${formatSignedPct(edge.edge_p1)} / ${formatSignedPct(edge.edge_p2)}
          </strong>
        </div>
        <div class="kelly-metric">
          <span>1/4 KELLY (CAP 2.5%)</span>
          <strong class="edge-highlight">${kellyP1.toFixed(2)}% / ${kellyP2.toFixed(2)}%</strong>
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

    // Negative rank_diff means P1 has a numerically smaller (better) ATP rank
    const rankFav = rankDiff < 0 ? "fav-p1" : rankDiff > 0 ? "fav-p2" : "fav-neutral";
    const rankSub = rankDiff < 0 ? "Favors P1" : rankDiff > 0 ? "Favors P2" : "Even";

    const surfFav = surfDiff > 0 ? "fav-p1" : surfDiff < 0 ? "fav-p2" : "fav-neutral";
    const formFav = formDivDiff > 0 ? "fav-p1" : formDivDiff < 0 ? "fav-p2" : "fav-neutral";
    const loadFav = load14Diff < 0 ? "fav-p1" : load14Diff > 0 ? "fav-p2" : "fav-neutral";
    const loadSub = load14Diff > 0 ? "P1 heavier load" : load14Diff < 0 ? "P2 heavier load" : "Equal load";
    const h2hFav = h2hP1 > h2hP2 ? "fav-p1" : h2hP2 > h2hP1 ? "fav-p2" : "fav-neutral";

    container.innerHTML = `
      <div class="delta-tile">
        <span class="delta-tile-label">RANK &Delta;</span>
        <strong class="delta-tile-val ${rankFav}">${rankDiff > 0 ? "+" : ""}${rankDiff.toFixed(0)}</strong>
        <span class="delta-tile-sub">${rankSub}</span>
      </div>
      <div class="delta-tile">
        <span class="delta-tile-label">SURFACE WR &Delta;</span>
        <strong class="delta-tile-val ${surfFav}">${formatSignedPct(surfDiff)}</strong>
        <span class="delta-tile-sub">Court edge</span>
      </div>
      <div class="delta-tile">
        <span class="delta-tile-label">FORM DIV &Delta;</span>
        <strong class="delta-tile-val ${formFav}">${formatSignedPct(formDivDiff)}</strong>
        <span class="delta-tile-sub">Short - Career</span>
      </div>
      <div class="delta-tile">
        <span class="delta-tile-label">FATIGUE 14D &Delta;</span>
        <strong class="delta-tile-val ${loadFav}">${load14Diff > 0 ? "+" : ""}${load14Diff.toFixed(0)}M</strong>
        <span class="delta-tile-sub">${loadSub}</span>
      </div>
      <div class="delta-tile">
        <span class="delta-tile-label">H2H RECORD</span>
        <strong class="delta-tile-val ${h2hFav}">${h2hP1} &ndash; ${h2hP2}</strong>
        <span class="delta-tile-sub">P1 &ndash; P2</span>
      </div>
    `;
  }

  function renderScheduleCard(fx) {
    const statusMeta = getStatusPillMeta(fx.status, fx.snapshot_locked);
    const surfCls = surfaceBadgeClass(fx.surface);
    const lgbP1 = Math.max(0, Math.min(1, safeNum(fx.prob_lgb, 0.5)));
    const lgbP2 = 1 - lgbP1;
    const lrP1 = Math.max(0, Math.min(1, safeNum(fx.prob_lr, 0.5)));
    const lrP2 = 1 - lrP1;

    const hasEdge = fx.edge_lgb_p1 !== null && fx.edge_lgb_p1 !== undefined && Math.abs(Number(fx.edge_lgb_p1)) >= 0.03;
    const edgeStr =
      fx.edge_lgb_p1 !== null && fx.edge_lgb_p1 !== undefined
        ? formatSignedPct(fx.edge_lgb_p1)
        : "—";
    const kellyStr =
      fx.kelly_p1_pct !== null && fx.kelly_p1_pct !== undefined
        ? `${safeNum(fx.kelly_p1_pct, 0).toFixed(2)}%`
        : "—";
    const bmStr =
      fx.bm_prob !== null && fx.bm_prob !== undefined
        ? formatPct(fx.bm_prob)
        : "—";

    const odd1Str = fx.odd_1 !== null && fx.odd_1 !== undefined ? safeNum(fx.odd_1).toFixed(2) : "—";
    const odd2Str = fx.odd_2 !== null && fx.odd_2 !== undefined ? safeNum(fx.odd_2).toFixed(2) : "—";

    return `
      <button type="button" class="schedule-card ${hasEdge ? "has-edge" : ""}" data-fixture-id="${escapeHtml(fx.fixture_id)}">
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
          ${fx.is_divergent ? `<span class="divergent-badge">DIVERGENT &Delta;${(safeNum(fx.div_delta) * 100).toFixed(1)}%</span>` : ""}
          ${hasEdge ? `<span class="value-alert-pill">VALUE EDGE</span>` : ""}
        </div>

        <div class="schedule-matchup-box">
          <div class="matchup-player-row">
            <span class="matchup-player-name p1">${escapeHtml(fx.p1)}</span>
            <span class="matchup-odds-chip">Odds ${odd1Str} &bull; Imp ${bmStr}</span>
          </div>
          <div class="matchup-player-row">
            <span class="matchup-player-name p2">${escapeHtml(fx.p2)}</span>
            <span class="matchup-odds-chip">Odds ${odd2Str}</span>
          </div>
        </div>

        <div class="split-bar-group">
          <div class="split-bar-labels">
            <span>LGBM: <strong class="fav-p1">${formatPct(lgbP1)}</strong> / <strong class="fav-p2">${formatPct(lgbP2)}</strong></span>
            <span>LR: ${formatPct(lrP1)} / ${formatPct(lrP2)}</span>
          </div>
          <div class="split-bar-track">
            <div class="split-bar-fill-p1" style="width:${(lgbP1 * 100).toFixed(1)}%"></div>
            <div class="split-bar-fill-p2" style="width:${(lgbP2 * 100).toFixed(1)}%"></div>
          </div>
          <div class="split-bar-track" style="height:5px; opacity:0.85;">
            <div class="split-bar-fill-lr-p1" style="width:${(lrP1 * 100).toFixed(1)}%"></div>
            <div class="split-bar-fill-p2" style="width:${(lrP2 * 100).toFixed(1)}%"></div>
          </div>
        </div>

        <div class="schedule-card-footer">
          <span>P1 EDGE: <strong class="${hasEdge ? "edge-highlight" : ""}">${edgeStr}</strong></span>
          <span>1/4 KELLY: <strong class="${hasEdge ? "edge-highlight" : ""}">${kellyStr}</strong></span>
          <span class="fav-p1">OPEN ARENA &rarr;</span>
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
