/* ==========================================================================
   matchpoint.ml — Risograph SVG Charts: 6-Axis Radar, Divergence Court & TreeSHAP
   Pure Vanilla JS + SVG • Zero External Charting Libraries • Strict 60/30/10 Tokens
   ========================================================================== */

(function (global) {
  "use strict";

  const MP = (global.MP = global.MP || {});

  function esc(val) {
    if (typeof MP.escapeHtml === "function") return MP.escapeHtml(val);
    if (val === null || val === undefined) return "";
    return String(val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function num(val, fallback = 0) {
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
  }

  const RADAR_AXES = [
    { key: "short_form", label: "Short Form" },
    { key: "med_form", label: "Med Form" },
    { key: "surface_mastery", label: "Surface Mastery" },
    { key: "rank_power", label: "Rank Power" },
    { key: "clutch_tb", label: "Clutch / TB" },
    { key: "stamina_freshness", label: "Stamina / Freshness" },
  ];

  const COUNT_FEATURES = new Set([
    "h2h_match_count",
    "h2h_surface_count",
    "surface_encoded",
    "series_encoded",
    "is_grand_slam",
  ]);

  function polarPoint(cx, cy, radius, idx, total) {
    const angle = -Math.PI / 2 + (idx * 2 * Math.PI) / total;
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      angle,
    };
  }

  function renderRadarChart(container, readoutEl, data) {
    if (!container || !data) return;

    const p1Radar = (data.player_cards && data.player_cards.p1 && data.player_cards.p1.radar) || {};
    const p2Radar = (data.player_cards && data.player_cards.p2 && data.player_cards.p2.radar) || {};
    const p1Favored = num(data.prob_lgb, 0.5) >= 0.5;

    const p1Stroke = p1Favored ? "#EA4F88" : "#3E7D84";
    const p1Fill = p1Favored ? "rgba(234, 79, 136, 0.20)" : "rgba(62, 125, 132, 0.18)";
    const p1Dash = p1Favored ? "" : 'stroke-dasharray="4 3"';

    const p2Stroke = !p1Favored ? "#EA4F88" : "#3E7D84";
    const p2Fill = !p1Favored ? "rgba(234, 79, 136, 0.20)" : "rgba(62, 125, 132, 0.18)";
    const p2Dash = !p1Favored ? "" : 'stroke-dasharray="4 3"';

    const cx = 245;
    const cy = 175;
    const maxR = 102;
    const levels = [25, 50, 75, 100];
    const n = RADAR_AXES.length;

    const ringPolygons = levels
      .map((lvl) => {
        const r = (lvl / 100) * maxR;
        const pts = RADAR_AXES.map((_, i) => {
          const pt = polarPoint(cx, cy, r, i, n);
          return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
        }).join(" ");
        return `<polygon points="${pts}" fill="none" stroke="#1D646C" stroke-opacity="${lvl === 100 ? "0.35" : "0.20"}" stroke-width="${lvl === 100 ? "1.3" : "1"}" />`;
      })
      .join("");

    const spokes = RADAR_AXES.map((_, i) => {
      const outer = polarPoint(cx, cy, maxR, i, n);
      return `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="#1D646C" stroke-opacity="0.26" stroke-width="1" />`;
    }).join("");

    const levelLabels = levels
      .map((lvl) => {
        const y = cy - (lvl / 100) * maxR;
        return `<text x="${cx + 4}" y="${(y - 2).toFixed(1)}" fill="#1D646C" fill-opacity="0.72" font-size="9" font-family="var(--font-mono)">${lvl}</text>`;
      })
      .join("");

    const p1Pts = [];
    const p2Pts = [];
    const vertexGroups = [];

    RADAR_AXES.forEach((axis, i) => {
      const v1 = Math.max(0, Math.min(100, num(p1Radar[axis.key], 50)));
      const v2 = Math.max(0, Math.min(100, num(p2Radar[axis.key], 50)));

      const pt1 = polarPoint(cx, cy, (v1 / 100) * maxR, i, n);
      const pt2 = polarPoint(cx, cy, (v2 / 100) * maxR, i, n);
      p1Pts.push(`${pt1.x.toFixed(1)},${pt1.y.toFixed(1)}`);
      p2Pts.push(`${pt2.x.toFixed(1)},${pt2.y.toFixed(1)}`);

      const labelPos = polarPoint(cx, cy, maxR + 24, i, n);
      let anchor = "middle";
      if (Math.cos(labelPos.angle) > 0.25) anchor = "start";
      else if (Math.cos(labelPos.angle) < -0.25) anchor = "end";

      const dy = Math.sin(labelPos.angle) > 0.4 ? 10 : Math.sin(labelPos.angle) < -0.4 ? -4 : 4;

      vertexGroups.push(`
        <g class="radar-axis-hotspot" tabindex="0" role="button"
           data-axis-key="${esc(axis.key)}"
           data-axis-label="${esc(axis.label)}"
           data-v1="${v1.toFixed(1)}"
           data-v2="${v2.toFixed(1)}"
           aria-label="${esc(axis.label)}: ${esc(data.p1)} ${v1.toFixed(1)}, ${esc(data.p2)} ${v2.toFixed(1)}">
          <circle cx="${pt1.x.toFixed(1)}" cy="${pt1.y.toFixed(1)}" r="${p1Favored ? "4.5" : "3.5"}" fill="${p1Stroke}" stroke="#FCE7D4" stroke-width="1.2" />
          <circle cx="${pt2.x.toFixed(1)}" cy="${pt2.y.toFixed(1)}" r="${!p1Favored ? "4.5" : "3.5"}" fill="${p2Stroke}" stroke="#FCE7D4" stroke-width="1.2" />
          <circle cx="${labelPos.x.toFixed(1)}" cy="${labelPos.y.toFixed(1)}" r="22" fill="transparent" />
          <text class="radar-axis-label" x="${labelPos.x.toFixed(1)}" y="${(labelPos.y + dy).toFixed(1)}" text-anchor="${anchor}">
            ${esc(axis.label)}
          </text>
        </g>
      `);
    });

    const dogPolygon = p1Favored
      ? `<polygon points="${p2Pts.join(" ")}" fill="${p2Fill}" stroke="${p2Stroke}" stroke-width="1.9" ${p2Dash} />`
      : `<polygon points="${p1Pts.join(" ")}" fill="${p1Fill}" stroke="${p1Stroke}" stroke-width="1.9" ${p1Dash} />`;
    const favPolygon = p1Favored
      ? `<polygon points="${p1Pts.join(" ")}" fill="${p1Fill}" stroke="${p1Stroke}" stroke-width="2.3" />`
      : `<polygon points="${p2Pts.join(" ")}" fill="${p2Fill}" stroke="${p2Stroke}" stroke-width="2.3" />`;

    container.innerHTML = `
      <svg class="radar-svg" viewBox="0 0 490 350" role="img" aria-label="6-Axis Scouting Radar Duel comparing ${esc(data.p1)} and ${esc(data.p2)}">
        ${ringPolygons}
        ${spokes}
        ${levelLabels}
        ${dogPolygon}
        ${favPolygon}
        ${vertexGroups.join("")}
      </svg>
    `;

    function updateReadout(target) {
      if (!readoutEl || !target) return;
      const label = target.getAttribute("data-axis-label") || "";
      const v1 = Number(target.getAttribute("data-v1") || 0);
      const v2 = Number(target.getAttribute("data-v2") || 0);
      const delta = v1 - v2;
      const leader = delta > 0 ? `${esc(data.p1)} (+${delta.toFixed(1)})` : delta < 0 ? `${esc(data.p2)} (+${Math.abs(delta).toFixed(1)})` : "Even (0.0)";

      container.querySelectorAll(".radar-axis-hotspot").forEach((el) => el.classList.remove("is-active"));
      target.classList.add("is-active");

      readoutEl.innerHTML = `
        <div class="meta-caption">Selected Scouting Axis</div>
        <div style="margin:2px 0 8px; font-size:15px; font-weight:700; color:var(--ink-deep);">${esc(label)}</div>
        <div class="fixture-row">
          <span class="player-name-nowrap">${esc(data.p1)}</span>
          <strong class="metric-tabular ${p1Favored ? "inline-pink" : ""}">${v1.toFixed(1)} / 100</strong>
        </div>
        <div class="fixture-row" style="margin-top:4px;">
          <span class="player-name-nowrap">${esc(data.p2)}</span>
          <strong class="metric-tabular ${!p1Favored ? "inline-pink" : ""}">${v2.toFixed(1)} / 100</strong>
        </div>
        <div style="margin-top:9px; padding-top:8px; border-top:1px solid var(--border-subtle);" class="meta-caption">
          Axis Advantage: <strong class="metric-tabular" style="color:var(--ink-deep);">${leader}</strong>
        </div>
      `;
    }

    const hotspots = container.querySelectorAll(".radar-axis-hotspot");
    hotspots.forEach((el) => {
      el.addEventListener("mouseenter", () => updateReadout(el));
      el.addEventListener("focus", () => updateReadout(el));
      el.addEventListener("click", () => updateReadout(el));
      el.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          updateReadout(el);
        }
      });
    });

    if (hotspots.length > 0) {
      updateReadout(hotspots[0]);
    }
  }

  function renderDivergenceGauge(container, data) {
    if (!container || !data) return;
    const divPct = Math.max(0, Math.min(100, num(data.div_delta, 0) * 100));
    const isDivergent = Boolean(data.is_divergent || divPct > 15);
    const maxScalePct = 35;
    const clampedRatio = Math.min(1, divPct / maxScalePct);

    // Semi-circle divergence arc
    const r = 64;
    const cx = 90;
    const cy = 88;
    const circumference = Math.PI * r;
    const fillLen = circumference * clampedRatio;
    const threshRatio = 15 / maxScalePct;
    const threshAngle = Math.PI * (1 - threshRatio);
    const tx1 = cx + (r - 6) * Math.cos(threshAngle);
    const ty1 = cy - (r - 6) * Math.sin(threshAngle);
    const tx2 = cx + (r + 7) * Math.cos(threshAngle);
    const ty2 = cy - (r + 7) * Math.sin(threshAngle);

    const strokeColor = isDivergent ? "#EA4F88" : "#1D646C";
    const statusBadge = isDivergent
      ? `<span class="divergent-badge">DIVERGENT &gt;15%</span>`
      : `<span class="badge badge-neutral">CONSENSUS</span>`;

    // Surface-tinted Risograph Court Schematic with model hotspot
    const surf = String(data.surface || "Hard");
    const courtStroke = surf === "Clay" ? "#B85335" : surf === "Grass" ? "#2B7A4B" : "#246B9E";
    const courtTint =
      surf === "Clay"
        ? "rgba(184, 83, 53, 0.09)"
        : surf === "Grass"
          ? "rgba(43, 122, 75, 0.09)"
          : "rgba(36, 107, 158, 0.08)";
    const lgbP1 = num(data.prob_lgb, 0.5);
    const hotspotX = lgbP1 >= 0.5 ? 162 : 78;
    const hotspotY = Math.round(65 - Math.min(34, divPct * 1.1));

    container.innerHTML = `
      <div class="meta-caption" style="font-weight:700; color:var(--ink-deep);">Court Schematic &amp; Divergence Gauge</div>
      <svg viewBox="0 0 240 106" width="100%" height="96" role="img" aria-label="${esc(surf)} court schematic with model bounce hotspot">
        <rect x="16" y="10" width="208" height="86" fill="${courtTint}" stroke="${courtStroke}" stroke-width="1.6" />
        <line x1="16" y1="22" x2="224" y2="22" stroke="${courtStroke}" stroke-width="1.1" />
        <line x1="16" y1="84" x2="224" y2="84" stroke="${courtStroke}" stroke-width="1.1" />
        <line x1="66" y1="22" x2="66" y2="84" stroke="${courtStroke}" stroke-width="1.1" />
        <line x1="174" y1="22" x2="174" y2="84" stroke="${courtStroke}" stroke-width="1.1" />
        <line x1="66" y1="53" x2="174" y2="53" stroke="${courtStroke}" stroke-width="1.1" />
        <line x1="120" y1="6" x2="120" y2="100" stroke="#1D646C" stroke-width="1.8" stroke-dasharray="3 2" />
        <circle cx="${hotspotX}" cy="${hotspotY}" r="12" fill="rgba(234, 79, 136, 0.22)" stroke="#EA4F88" stroke-width="1.4" />
        <circle cx="${hotspotX}" cy="${hotspotY}" r="4.5" fill="#EA4F88" />
      </svg>

      <svg width="180" height="104" viewBox="0 0 180 104" role="img" aria-label="Dual-Model Divergence Gauge: ${divPct.toFixed(1)}%">
        <path d="M 26 88 A 64 64 0 0 1 154 88" fill="none" stroke="rgba(29, 100, 108, 0.22)" stroke-width="10" stroke-linecap="round" />
        <path d="M 26 88 A 64 64 0 0 1 154 88" fill="none" stroke="${strokeColor}" stroke-width="10" stroke-linecap="round"
              stroke-dasharray="${fillLen.toFixed(1)} ${circumference.toFixed(1)}" />
        <line x1="${tx1.toFixed(1)}" y1="${ty1.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}"
              stroke="#13454B" stroke-width="2" />
        <text x="90" y="70" text-anchor="middle" fill="#13454B" font-family="var(--font-mono)" font-size="19" font-weight="700">
          ${divPct.toFixed(1)}%
        </text>
        <text x="90" y="88" text-anchor="middle" fill="#3E7D84" font-family="var(--font-mono)" font-size="8.5" font-weight="600">
          |LGBM &minus; LR| (15% MARK)
        </text>
      </svg>
      ${statusBadge}
    `;
  }

  function renderTreeShapTugOfWar(summaryEl, barsEl, data) {
    if (!barsEl || !data) return;
    const treeshap = data.treeshap || {};
    const contributions = Array.isArray(treeshap.contributions) ? treeshap.contributions : [];
    const bias = num(treeshap.bias, 0);
    const totalContrib = num(treeshap.p1_total_contrib, 0);
    const p1Favored = num(data.prob_lgb, 0.5) >= 0.5;

    if (summaryEl) {
      const totalFavorsP1 = totalContrib > 0;
      const totalCls =
        totalContrib === 0
          ? "fav-neutral"
          : totalFavorsP1 === p1Favored
            ? "fav-favored"
            : "fav-underdog";
      summaryEl.innerHTML = `
        <span class="summary-pill">BASE BIAS E[f(x)]: <strong class="metric-tabular">${bias >= 0 ? "+" : ""}${bias.toFixed(4)}</strong></span>
        <span class="summary-pill">NET P1 PULL: <strong class="metric-tabular ${totalCls}">${totalContrib >= 0 ? "+" : ""}${totalContrib.toFixed(4)}</strong></span>
      `;
    }

    if (contributions.length === 0) {
      barsEl.innerHTML = `<div class="kelly-prompt">No TreeSHAP feature contributions returned for this matchup.</div>`;
      return;
    }

    const topItems = contributions.slice(0, 12);
    const maxAbsShap =
      topItems.reduce((acc, item) => Math.max(acc, Math.abs(num(item.shap, 0))), 0) || 1;

    barsEl.innerHTML = topItems
      .map((item) => {
        const shap = num(item.shap, 0);
        const favors = item.favors || (shap > 1e-9 ? "p1" : shap < -1e-9 ? "p2" : "neutral");
        const pctWidth = Math.min(48, (Math.abs(shap) / maxAbsShap) * 48);
        const rawDelta = num(item.raw_delta, 0);
        const deltaStr = Number.isInteger(rawDelta) ? String(rawDelta) : rawDelta.toFixed(3);
        const isCount = COUNT_FEATURES.has(item.feature);
        const rawPrefix = isCount ? "Val" : "\u0394";
        const shapStr = `${shap > 0 ? "+" : ""}${shap.toFixed(4)}`;

        let barHtml = "";
        let valCls = "fav-neutral";
        if (favors === "p1") {
          const toneCls = p1Favored ? "shap-bar-favored" : "shap-bar-underdog";
          barHtml = `<div class="shap-bar-fill shap-bar-left ${toneCls}" style="width:${pctWidth.toFixed(2)}%"></div>`;
          valCls = p1Favored ? "fav-favored" : "fav-underdog";
        } else if (favors === "p2") {
          const toneCls = !p1Favored ? "shap-bar-favored" : "shap-bar-underdog";
          barHtml = `<div class="shap-bar-fill shap-bar-right ${toneCls}" style="width:${pctWidth.toFixed(2)}%"></div>`;
          valCls = !p1Favored ? "fav-favored" : "fav-underdog";
        }

        const pullTarget = favors === "p1" ? data.p1 : favors === "p2" ? data.p2 : "Neutral";
        return `
          <div class="shap-row">
            <div class="shap-feat-info">
              <span class="shap-feat-label">${esc(item.label || item.feature)}</span>
              <span class="shap-cat-badge">${esc(item.category || "Predictor")} &bull; &rarr; <span class="player-name-nowrap">${esc(pullTarget)}</span></span>
            </div>
            <div class="shap-raw-delta metric-tabular" title="Feature value or P1-P2 delta">${rawPrefix} ${esc(deltaStr)}</div>
            <div class="shap-tug-track" aria-label="SHAP tug bar">
              <span class="shap-center-line" aria-hidden="true"></span>
              ${barHtml}
            </div>
            <div class="shap-val-cell metric-tabular ${valCls}">${esc(shapStr)}</div>
          </div>
        `;
      })
      .join("");
  }

  MP.renderRadarChart = renderRadarChart;
  MP.renderDivergenceGauge = renderDivergenceGauge;
  MP.renderTreeShapTugOfWar = renderTreeShapTugOfWar;
})(window);
