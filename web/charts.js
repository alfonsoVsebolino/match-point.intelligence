/* ==========================================================================
   MatchPoint.intelligence — Custom Neon SVG Radar, Divergence Gauge & TreeSHAP
   Pure Vanilla JS + SVG • Zero External Charting Libraries
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

    const cx = 220;
    const cy = 185;
    const maxR = 112;
    const levels = [20, 40, 60, 80, 100];
    const n = RADAR_AXES.length;

    const ringPolygons = levels
      .map((lvl) => {
        const r = (lvl / 100) * maxR;
        const pts = RADAR_AXES.map((_, i) => {
          const pt = polarPoint(cx, cy, r, i, n);
          return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
        }).join(" ");
        return `<polygon points="${pts}" fill="none" stroke="rgba(148,163,184,${lvl === 100 ? "0.35" : "0.18"})" stroke-width="${lvl === 100 ? "1.4" : "1"}" />`;
      })
      .join("");

    const spokes = RADAR_AXES.map((_, i) => {
      const outer = polarPoint(cx, cy, maxR, i, n);
      return `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="rgba(148,163,184,0.22)" stroke-width="1" />`;
    }).join("");

    const levelLabels = levels
      .map((lvl) => {
        const y = cy - (lvl / 100) * maxR;
        return `<text x="${cx + 4}" y="${(y - 2).toFixed(1)}" fill="rgba(148,163,184,0.6)" font-size="8.5" font-family="var(--font-mono)">${lvl}</text>`;
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

      const labelPos = polarPoint(cx, cy, maxR + 28, i, n);
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
          <circle cx="${pt1.x.toFixed(1)}" cy="${pt1.y.toFixed(1)}" r="4.5" fill="#00f5ff" stroke="#060913" stroke-width="1.5" />
          <circle cx="${pt2.x.toFixed(1)}" cy="${pt2.y.toFixed(1)}" r="4.5" fill="#ff2bd6" stroke="#060913" stroke-width="1.5" />
          <circle cx="${labelPos.x.toFixed(1)}" cy="${labelPos.y.toFixed(1)}" r="22" fill="transparent" />
          <text class="radar-axis-label" x="${labelPos.x.toFixed(1)}" y="${(labelPos.y + dy).toFixed(1)}" text-anchor="${anchor}">
            ${esc(axis.label)}
          </text>
        </g>
      `);
    });

    container.innerHTML = `
      <svg class="radar-svg" viewBox="0 0 440 370" role="img" aria-label="6-Axis Scouting Radar Duel comparing ${esc(data.p1)} and ${esc(data.p2)}">
        <defs>
          <filter id="neonCyanGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="3.5" flood-color="#00f5ff" flood-opacity="0.7" />
          </filter>
          <filter id="neonMagentaGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="0" stdDeviation="3.5" flood-color="#ff2bd6" flood-opacity="0.7" />
          </filter>
        </defs>
        ${ringPolygons}
        ${spokes}
        ${levelLabels}
        <polygon points="${p2Pts.join(" ")}" fill="rgba(255,43,214,0.18)" stroke="#ff2bd6" stroke-width="2.2" filter="url(#neonMagentaGlow)" />
        <polygon points="${p1Pts.join(" ")}" fill="rgba(0,245,255,0.20)" stroke="#00f5ff" stroke-width="2.4" filter="url(#neonCyanGlow)" />
        ${vertexGroups.join("")}
      </svg>
    `;

    function updateReadout(target) {
      if (!readoutEl || !target) return;
      const label = target.getAttribute("data-axis-label") || "";
      const v1 = Number(target.getAttribute("data-v1") || 0);
      const v2 = Number(target.getAttribute("data-v2") || 0);
      const delta = v1 - v2;
      const deltaStr = `${delta > 0 ? "+" : ""}${delta.toFixed(1)}`;
      const leaderCls = delta > 0 ? "fav-p1" : delta < 0 ? "fav-p2" : "fav-neutral";

      container.querySelectorAll(".radar-axis-hotspot").forEach((el) => el.classList.remove("is-active"));
      target.classList.add("is-active");

      readoutEl.innerHTML = `
        <strong>${esc(label)}:</strong>
        <span class="fav-p1">${esc(data.p1)} ${v1.toFixed(1)}</span>
        vs
        <span class="fav-p2">${esc(data.p2)} ${v2.toFixed(1)}</span>
        (<span class="${leaderCls}">&Delta; ${deltaStr}</span>)
      `;
    }

    const hotspots = container.querySelectorAll(".radar-axis-hotspot");
    hotspots.forEach((el) => {
      el.addEventListener("mouseenter", () => updateReadout(el));
      el.addEventListener("focus", () => updateReadout(el));
      el.addEventListener("click", () => updateReadout(el));
    });

    if (hotspots.length > 0) {
      updateReadout(hotspots[0]);
    }
  }

  function renderDivergenceGauge(container, data) {
    if (!container || !data) return;
    const divPct = Math.max(0, Math.min(100, num(data.div_delta, 0) * 100));
    const isDivergent = Boolean(data.is_divergent || divPct > 15);
    const maxScalePct = 35; // 35% full scale arc
    const clampedRatio = Math.min(1, divPct / maxScalePct);

    // Semi-circle arc from (20, 95) to (160, 95), r = 70
    const r = 70;
    const cx = 90;
    const cy = 95;
    const circumference = Math.PI * r;
    const fillLen = circumference * clampedRatio;
    const threshRatio = 15 / maxScalePct;
    const threshAngle = Math.PI * (1 - threshRatio);
    const tx1 = cx + (r - 6) * Math.cos(threshAngle);
    const ty1 = cy - (r - 6) * Math.sin(threshAngle);
    const tx2 = cx + (r + 7) * Math.cos(threshAngle);
    const ty2 = cy - (r + 7) * Math.sin(threshAngle);

    const strokeColor = isDivergent ? "#f59e0b" : "#00f5ff";
    const statusBadge = isDivergent
      ? `<span class="divergent-badge">DIVERGENT &gt;15%</span>`
      : `<span class="surface-badge surface-hard">CONSENSUS</span>`;

    container.innerHTML = `
      <svg width="180" height="112" viewBox="0 0 180 112" role="img" aria-label="Dual-Model Divergence Gauge: ${divPct.toFixed(1)}%">
        <path d="M 20 95 A 70 70 0 0 1 160 95" fill="none" stroke="rgba(148,163,184,0.2)" stroke-width="11" stroke-linecap="round" />
        <path d="M 20 95 A 70 70 0 0 1 160 95" fill="none" stroke="${strokeColor}" stroke-width="11" stroke-linecap="round"
              stroke-dasharray="${fillLen.toFixed(1)} ${circumference.toFixed(1)}" />
        <line x1="${tx1.toFixed(1)}" y1="${ty1.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}"
              stroke="#ccff00" stroke-width="2.2" />
        <text x="90" y="76" text-anchor="middle" fill="#ffffff" font-family="var(--font-mono)" font-size="20" font-weight="800">
          ${divPct.toFixed(1)}%
        </text>
        <text x="90" y="94" text-anchor="middle" fill="#94a3b8" font-family="var(--font-mono)" font-size="8.5">
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

    if (summaryEl) {
      const totalCls = totalContrib > 0 ? "fav-p1" : totalContrib < 0 ? "fav-p2" : "fav-neutral";
      summaryEl.innerHTML = `
        <span class="summary-pill">BASE BIAS: <strong>${bias >= 0 ? "+" : ""}${bias.toFixed(4)}</strong></span>
        <span class="summary-pill">NET P1 PULL: <strong class="${totalCls}">${totalContrib >= 0 ? "+" : ""}${totalContrib.toFixed(4)}</strong></span>
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
        const pctWidth = Math.min(50, (Math.abs(shap) / maxAbsShap) * 50);
        const rawDelta = num(item.raw_delta, 0);
        const deltaStr = Number.isInteger(rawDelta) ? String(rawDelta) : rawDelta.toFixed(3);
        const shapStr = `${shap > 0 ? "+" : ""}${shap.toFixed(4)}`;

        let barHtml = "";
        let valCls = "fav-neutral";
        if (favors === "p1") {
          barHtml = `<div class="shap-bar-fill shap-bar-p1" style="width:${pctWidth.toFixed(2)}%"></div>`;
          valCls = "fav-p1";
        } else if (favors === "p2") {
          barHtml = `<div class="shap-bar-fill shap-bar-p2" style="width:${pctWidth.toFixed(2)}%"></div>`;
          valCls = "fav-p2";
        }

        return `
          <div class="shap-row">
            <div class="shap-feat-info">
              <span class="shap-feat-label">${esc(item.label || item.feature)}</span>
              <span class="shap-cat-badge">${esc(item.category || "Predictor")}</span>
            </div>
            <div class="shap-raw-delta" title="Raw feature delta">&Delta; ${esc(deltaStr)}</div>
            <div class="shap-tug-track" aria-label="SHAP tug bar">
              <span class="shap-center-line" aria-hidden="true"></span>
              ${barHtml}
            </div>
            <div class="shap-val-cell ${valCls}">${esc(shapStr)}</div>
          </div>
        `;
      })
      .join("");
  }

  MP.renderRadarChart = renderRadarChart;
  MP.renderDivergenceGauge = renderDivergenceGauge;
  MP.renderTreeShapTugOfWar = renderTreeShapTugOfWar;
})(window);
