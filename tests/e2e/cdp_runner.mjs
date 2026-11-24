#!/usr/bin/env node
/**
 * Pure Node v24 ESM Headless Chromium CDP Runner (zero external npm dependencies).
 *
 * Usage:
 *   node tests/e2e/cdp_runner.mjs --base-url http://127.0.0.1:8080 --scenario boot_and_theme --width 1440 --height 900
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

for (const proxyKey of [
  'HTTP_PROXY',
  'http_proxy',
  'HTTPS_PROXY',
  'https_proxy',
  'ALL_PROXY',
  'all_proxy',
  'NODE_USE_ENV_PROXY',
]) {
  delete process.env[proxyKey];
}
process.env.NO_PROXY = '127.0.0.1,localhost,::1,[::1]';
process.env.no_proxy = '127.0.0.1,localhost,::1,[::1]';

function parseArgs(argv) {
  const args = {
    baseUrl: '',
    scenario: '',
    width: 1440,
    height: 900,
    timeoutMs: 45000,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const val = argv[i + 1];
    if (key === '--base-url' && val) {
      args.baseUrl = val.replace(/\/+$/, '');
      i += 1;
    } else if (key === '--scenario' && val) {
      args.scenario = val;
      i += 1;
    } else if (key === '--width' && val) {
      args.width = Number.parseInt(val, 10) || 1440;
      i += 1;
    } else if (key === '--height' && val) {
      args.height = Number.parseInt(val, 10) || 900;
      i += 1;
    } else if (key === '--timeout-ms' && val) {
      args.timeoutMs = Number.parseInt(val, 10) || 45000;
      i += 1;
    }
  }
  if (!args.baseUrl || !args.scenario) {
    throw new Error('Required arguments: --base-url <url> --scenario <name>');
  }
  return args;
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close((err) => {
        if (err) reject(err);
        else resolve(port);
      });
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function raceWithTimeout(promise, ms, label) {
  let timer = null;
  try {
    const timeoutPromise = new Promise((_, reject) => {
      timer = setTimeout(() => {
        reject(new Error(`${label} exceeded timeout ${ms}ms`));
      }, ms);
      timer.unref?.();
    });
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timer !== null) {
      clearTimeout(timer);
    }
  }
}

async function waitForCdpTarget(cdpPort, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  let lastErr = '';
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${cdpPort}/json/list`);
      if (res.ok) {
        const targets = await res.json();
        const pageTarget = targets.find(
          (t) => t.type === 'page' && t.webSocketDebuggerUrl
        );
        if (pageTarget) return pageTarget;
      } else {
        lastErr = `HTTP ${res.status}`;
      }
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
    await sleep(120);
  }
  throw new Error(`Timed out waiting for Chromium CDP target on port ${cdpPort}: ${lastErr}`);
}

class CdpClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.closed = false;

    this.ready = new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(), { once: true });
      this.ws.addEventListener('error', (ev) => {
        reject(new Error(`CDP WebSocket error: ${ev.message || 'connection failed'}`));
      }, { once: true });
    });

    this.ws.addEventListener('message', (event) => {
      let msg;
      try {
        msg = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (typeof msg.id === 'number') {
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          if (msg.error) {
            entry.reject(new Error(`CDP ${entry.method} error: ${JSON.stringify(msg.error)}`));
          } else {
            entry.resolve(msg.result || {});
          }
        }
      } else if (msg.method) {
        const handlers = this.listeners.get(msg.method);
        if (handlers) {
          for (const handler of handlers) {
            try {
              handler(msg.params || {});
            } catch {
              // Ignore listener exceptions
            }
          }
        }
      }
    });

    this.ws.addEventListener('close', () => {
      this.closed = true;
      for (const [, entry] of this.pending) {
        entry.reject(new Error(`CDP WebSocket closed before ${entry.method} completed`));
      }
      this.pending.clear();
    });
  }

  on(eventName, handler) {
    if (!this.listeners.has(eventName)) {
      this.listeners.set(eventName, new Set());
    }
    this.listeners.get(eventName).add(handler);
  }

  async send(method, params = {}, timeoutMs = 20000) {
    await this.ready;
    if (this.closed) {
      throw new Error(`CDP connection already closed when calling ${method}`);
    }
    const id = this.nextId;
    this.nextId += 1;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP ${method} timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);
      timer.unref?.();

      this.pending.set(id, {
        method,
        resolve: (res) => {
          clearTimeout(timer);
          resolve(res);
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });

      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    try {
      this.ws.close();
    } catch {
      // Ignore close error
    }
  }
}

/**
 * Shared accessible DOM helper snippet injected/executed inside the browser page.
 * Avoids private/brittle class selectors and relies on roles, labels, visible text,
 * option values, SVG geometry, and computed styles.
 */
const DOM_HELPERS_JS = `
(() => {
  if (window.__E2E_HELPERS__) return true;

  const norm = (s) => String(s || '').replace(/\\s+/g, ' ').trim();

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || Number.parseFloat(style.opacity) === 0) {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function getText(el) {
    if (!el) return '';
    return norm(el.innerText || el.textContent || '');
  }

  function getAssociatedLabelText(el) {
    if (!el) return '';
    const parts = [];
    if (el.getAttribute('aria-label')) parts.push(el.getAttribute('aria-label'));
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      for (const id of labelledBy.split(/\\s+/)) {
        const ref = document.getElementById(id);
        if (ref) parts.push(getText(ref));
      }
    }
    if (el.id) {
      const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lbl) parts.push(getText(lbl));
    }
    const parentLabel = el.closest('label');
    if (parentLabel) parts.push(getText(parentLabel));
    if (el.getAttribute('placeholder')) parts.push(el.getAttribute('placeholder'));
    if (el.getAttribute('name')) parts.push(el.getAttribute('name'));
    if (el.id) parts.push(el.id);
    if (el.getAttribute('data-testid')) parts.push(el.getAttribute('data-testid'));
    if (el.previousElementSibling && /^(label|span|div|p|strong|small)$/i.test(el.previousElementSibling.tagName)) {
      parts.push(getText(el.previousElementSibling));
    }
    const parent = el.parentElement;
    if (parent) {
      const siblingLabel = parent.querySelector('label, [class*="label" i]');
      if (siblingLabel && siblingLabel !== el) parts.push(getText(siblingLabel));
    }
    return norm(parts.join(' '));
  }

  function matchAny(text, patterns) {
    const t = norm(text);
    return patterns.some((p) => (p instanceof RegExp ? p.test(t) : t.toLowerCase().includes(String(p).toLowerCase())));
  }

  function findClickable(patterns, { visibleOnly = true } = {}) {
    const candidates = Array.from(
      document.querySelectorAll('button, [role="tab"], [role="button"], a, summary, input[type="button"], input[type="submit"]')
    );
    const filtered = candidates.filter((el) => (!visibleOnly || isVisible(el)) && !el.disabled);
    for (const el of filtered) {
      const txt = norm([getText(el), el.getAttribute('aria-label'), el.getAttribute('title'), el.value].filter(Boolean).join(' '));
      if (matchAny(txt, patterns)) return el;
    }
    return null;
  }

  function clickControl(patterns, { visibleOnly = true } = {}) {
    const el = findClickable(patterns, { visibleOnly });
    if (!el) {
      const available = Array.from(document.querySelectorAll('button, [role="tab"], [role="button"], summary'))
        .filter((x) => isVisible(x))
        .map((x) => getText(x) || x.getAttribute('aria-label') || '')
        .filter(Boolean);
      throw new Error('Clickable control not found for patterns ' + patterns.map(String).join(', ') + '. Visible controls: ' + JSON.stringify(available));
    }
    el.scrollIntoView({ block: 'center', inline: 'nearest' });
    el.click();
    return getText(el) || el.getAttribute('aria-label') || el.tagName;
  }

  function findSelect({ labelPatterns = [], optionPatterns = [], index = 0, visibleOnly = true } = {}) {
    const selects = Array.from(document.querySelectorAll('select')).filter((el) => !visibleOnly || isVisible(el));
    let matches = selects;
    if (optionPatterns.length > 0) {
      matches = matches.filter((sel) => {
        const opts = Array.from(sel.options).map((o) => norm(o.textContent + ' ' + o.value));
        return optionPatterns.some((pat) => opts.some((ot) => (pat instanceof RegExp ? pat.test(ot) : ot.toLowerCase().includes(String(pat).toLowerCase()))));
      });
    }
    if (labelPatterns.length > 0) {
      const byLabel = matches.filter((sel) => matchAny(getAssociatedLabelText(sel), labelPatterns));
      if (byLabel.length === 0) return null;
      matches = byLabel;
    }
    return matches[index] || null;
  }

  function setSelectOption(sel, target) {
    if (!sel) throw new Error('Select element is null for target: ' + target);
    const opts = Array.from(sel.options);
    let found = null;
    if (target instanceof RegExp) {
      found = opts.find((o) => target.test(norm(o.textContent)) || target.test(norm(o.value)));
    } else {
      const lower = String(target).toLowerCase();
      found =
        opts.find((o) => norm(o.value) === target || norm(o.textContent) === target) ||
        opts.find((o) => norm(o.textContent).toLowerCase().includes(lower) || norm(o.value).toLowerCase().includes(lower));
    }
    if (!found) {
      throw new Error('Option matching ' + target + ' not found in select (' + getAssociatedLabelText(sel) + '). First 10 options: ' + JSON.stringify(opts.slice(0, 10).map((o) => o.textContent)));
    }
    sel.value = found.value;
    sel.dispatchEvent(new Event('input', { bubbles: true }));
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    return norm(found.textContent || found.value);
  }

  function findInput(labelPatterns, { index = 0, visibleOnly = true, types = ['text', 'number', 'search'] } = {}) {
    const inputs = Array.from(document.querySelectorAll('input')).filter((el) => {
      const t = (el.type || 'text').toLowerCase();
      if (!types.includes(t)) return false;
      return !visibleOnly || isVisible(el);
    });
    const matched = inputs.filter((el) => matchAny(getAssociatedLabelText(el), labelPatterns));
    return matched[index] || null;
  }

  function setInputValue(el, value) {
    if (!el) throw new Error('Input element is null');
    const proto = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    if (proto && proto.set) {
      proto.set.call(el, String(value));
    } else {
      el.value = String(value);
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.dispatchEvent(new Event('blur', { bubbles: true }));
  }

  function setToggle(patterns, desired = true) {
    const checkboxes = Array.from(document.querySelectorAll('input[type="checkbox"]'));
    for (const cb of checkboxes) {
      if (matchAny(getAssociatedLabelText(cb), patterns)) {
        if (Boolean(cb.checked) !== Boolean(desired)) {
          cb.click();
        }
        return Boolean(cb.checked);
      }
    }
    const switchBtn = Array.from(document.querySelectorAll('button[aria-pressed], [role="switch"]')).filter((el) => isVisible(el));
    for (const btn of switchBtn) {
      const lbl = norm(getText(btn) + ' ' + getAssociatedLabelText(btn));
      if (matchAny(lbl, patterns)) {
        const cur = btn.getAttribute('aria-pressed') === 'true' || btn.getAttribute('aria-checked') === 'true';
        if (cur !== Boolean(desired)) btn.click();
        return Boolean(desired);
      }
    }
    const plainBtn = findClickable(patterns);
    if (plainBtn) {
      plainBtn.click();
      return true;
    }
    throw new Error('Toggle/checkbox not found for patterns: ' + patterns.map(String).join(', '));
  }

  function getSelectedText(el) {
    if (!el) return '';
    if (el.tagName === 'SELECT') {
      const opt = el.options[el.selectedIndex];
      return norm(opt ? opt.textContent : el.value);
    }
    return norm(el.value || getText(el));
  }

  async function waitFor(predicate, timeoutMs = 15000, desc = 'condition') {
    const deadline = Date.now() + timeoutMs;
    let lastError = null;
    while (Date.now() < deadline) {
      try {
        const val = await predicate();
        if (val) return val;
      } catch (err) {
        lastError = err;
      }
      await new Promise((r) => setTimeout(r, 120));
    }
    throw new Error('Timed out after ' + timeoutMs + 'ms waiting for ' + desc + (lastError ? ' (' + lastError.message + ')' : ''));
  }

  function inspectCssColors() {
    const tokens = [];
    for (const sheet of Array.from(document.styleSheets)) {
      try {
        for (const rule of Array.from(sheet.cssRules || [])) {
          tokens.push(rule.cssText || '');
        }
      } catch {
        // Cross-origin or inaccessible stylesheet
      }
    }
    for (const styleEl of Array.from(document.querySelectorAll('style'))) {
      tokens.push(styleEl.textContent || '');
    }
    const joined = tokens.join('\\n').toLowerCase();
    const rootStyle = window.getComputedStyle(document.documentElement);
    const inkPrimary = (rootStyle.getPropertyValue('--ink-primary') || '').trim().toLowerCase();
    const accentSignal = (rootStyle.getPropertyValue('--accent-signal') || '').trim().toLowerCase();
    const bgSurface = (rootStyle.getPropertyValue('--bg-surface') || '').trim().toLowerCase();
    return {
      hasTealInk: inkPrimary === '#1d646c' || /#1d646c|29,\\s*100,\\s*108/.test(joined),
      hasSignalPink: accentSignal === '#ea4f88' || /#ea4f88|234,\\s*79,\\s*136/.test(joined),
      hasSurfaceCream: bgSurface === '#fce7d4' || /#fce7d4|252,\\s*231,\\s*212/.test(joined),
      hasLegacyNeon: /#060913|#00f5ff|#ccff00|#ff2bd6/.test(joined),
      bodyBg: window.getComputedStyle(document.body).backgroundColor,
    };
  }

  async function ensureArenaRailExpanded() {
    await waitFor(() => {
      const toggleBtn = document.getElementById('arenaRailToggleBtn');
      const p1Sel = document.getElementById('p1Select');
      return (toggleBtn && isVisible(toggleBtn)) || (p1Sel && isVisible(p1Sel));
    }, 5000, 'Arena controls or rail toggle');
    const toggleBtn = document.getElementById('arenaRailToggleBtn');
    if (toggleBtn && isVisible(toggleBtn) && toggleBtn.getAttribute('aria-expanded') === 'false') {
      toggleBtn.click();
    }
  }

  function checkArcadeAndCharts() {
    // Check TreeSHAP first on Tab 1 if present
    const driversTab = findClickable([/1\\.\\s*treeshap|treeshap\\s*drivers/i]);
    const scoutingTab = findClickable([/2\\.\\s*scouting|6-axis\\s*radar/i]);

    let hasTreeShap = /treeshap|tug-of-war|shap|feature\\s+impact|contributions?/i.test(getText(document.body));
    if (!hasTreeShap && driversTab) {
      driversTab.click();
      hasTreeShap = /treeshap|tug-of-war|shap|feature\\s+impact|contributions?/i.test(getText(document.body));
    }

    if (scoutingTab && scoutingTab.getAttribute('aria-selected') !== 'true') {
      scoutingTab.click();
    }

    const bodyText = getText(document.body);
    const ovrMatches = bodyText.match(/\\bOVR\\b/gi) || [];
    const svgs = Array.from(document.querySelectorAll('svg')).filter((svg) => {
      if (!isVisible(svg)) return false;
      const r = svg.getBoundingClientRect();
      return r.width >= 80 && r.height >= 80;
    });
    const radarAxes = [
      /short\\s*form/i,
      /med\\s*form/i,
      /surface\\s*mastery/i,
      /rank\\s*power/i,
      /clutch|tiebreak|\\btb\\b/i,
      /stamina|freshness/i,
    ];
    const radarAxesMatched = radarAxes.filter((re) => re.test(bodyText)).length;
    const hasRadarGeometry = svgs.some((svg) => svg.querySelectorAll('polygon, path, circle, line').length >= 4);
    return {
      ovrCount: ovrMatches.length,
      svgCount: svgs.length,
      hasRadarGeometry,
      radarAxesMatched,
      hasTreeShap,
    };
  }

  function expandAndCount31Features() {
    const matrixTab = findClickable([/3\\.\\s*31[- ]?feature|31[- ]?feature\\s*matrix/i]);
    if (matrixTab && matrixTab.getAttribute('aria-selected') !== 'true') {
      matrixTab.click();
    }
    const toggle = findClickable([/31[- ]?feature|feature\\s+matrix|all\\s+31|full\\s+matrix|show\\s+all\\s+features/i]);
    if (toggle) {
      const details = toggle.closest('details');
      const isExpanded = toggle.getAttribute('aria-expanded') === 'true' || (details && details.open);
      if (!isExpanded) toggle.click();
    }
    const tables = Array.from(document.querySelectorAll('table')).filter((t) => isVisible(t));
    let maxRows = 0;
    for (const tbl of tables) {
      const rows = tbl.querySelectorAll('tbody tr').length || tbl.querySelectorAll('tr').length;
      if (rows > maxRows) maxRows = rows;
    }
    const featEls = document.querySelectorAll('[data-feature], [data-feature-name]').length;
    return Math.max(maxRows, featEls);
  }

  window.__E2E_HELPERS__ = {
    norm,
    isVisible,
    getText,
    getAssociatedLabelText,
    findClickable,
    clickControl,
    findSelect,
    setSelectOption,
    findInput,
    setInputValue,
    setToggle,
    getSelectedText,
    waitFor,
    inspectCssColors,
    ensureArenaRailExpanded,
    checkArcadeAndCharts,
    expandAndCount31Features,
  };
  return true;
})()
`;

async function evalPage(cdp, fnBody) {
  await cdp.send('Runtime.evaluate', {
    expression: DOM_HELPERS_JS,
    returnByValue: true,
  });
  const res = await cdp.send('Runtime.evaluate', {
    expression: `(async () => { const H = window.__E2E_HELPERS__; ${fnBody} })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (res.exceptionDetails) {
    const desc =
      res.exceptionDetails.exception?.description ||
      res.exceptionDetails.text ||
      JSON.stringify(res.exceptionDetails);
    throw new Error(`Browser evaluation exception: ${desc}`);
  }
  return res.result?.value;
}

async function navigateAndHydrate(cdp, baseUrl, telemetry) {
  let loadFired = false;
  cdp.on('Page.loadEventFired', () => {
    loadFired = true;
  });

  await cdp.send('Page.navigate', { url: `${baseUrl}/` });

  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    if (loadFired && telemetry.inFlight.size === 0) {
      const ready = await evalPage(cdp, `
        return document.readyState === 'complete' &&
          (document.body && (document.body.innerText || '').trim().length > 20);
      `);
      if (ready) break;
    }
    await sleep(150);
  }

  // Wait for /api/v1/meta hydration (select options or non-loading state)
  await evalPage(cdp, `
    await H.waitFor(() => {
      const selects = document.querySelectorAll('select');
      const hasOptions = Array.from(selects).some((s) => s.options && s.options.length >= 2);
      const hasContent = H.getText(document.body).length > 50;
      return hasOptions || hasContent;
    }, 15000, 'initial app hydration');
  `);
}

const SCENARIOS = {
  async boot_and_theme(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);
    const data = await evalPage(cdp, `
      const css = H.inspectCssColors();
      if (css.bodyBg !== 'rgb(248, 219, 193)') {
        throw new Error('Expected body background rgb(248, 219, 193) (#F8DBC1), got ' + css.bodyBg);
      }
      if (!css.hasTealInk || !css.hasSignalPink || !css.hasSurfaceCream || css.hasLegacyNeon) {
        throw new Error('Risograph Option A tokens check failed: ' + JSON.stringify(css));
      }

      // ATP Wire top live ticker strip (do not accept bare <header> or <nav>)
      const tickerCandidates = Array.from(
        document.querySelectorAll('[aria-label*="ticker" i], [id*="ticker" i], [class*="ticker" i], [role="marquee"], aside, header, nav')
      ).filter((el) => H.isVisible(el));
      const topTicker = tickerCandidates.find((el) => {
        const r = el.getBoundingClientRect();
        const txt = H.getText(el);
        if (!(r.top < 220 && r.height > 16 && txt.length > 5)) return false;
        const tag = el.tagName.toLowerCase();
        const attrMeta = [el.id, el.className, el.getAttribute('aria-label'), el.getAttribute('role')]
          .filter(Boolean)
          .join(' ');
        const hasTickerAttr = /ticker|marquee|wire/i.test(attrMeta);
        const hasTickerContent = /ticker|live|schedule|\\bvs\\b|standby|atp\\s*wire/i.test(txt);
        if (tag === 'header' || tag === 'nav') {
          return hasTickerAttr || hasTickerContent;
        }
        return hasTickerAttr || hasTickerContent;
      });
      if (!topTicker) {
        throw new Error('ESPN-style top ticker strip not found or not visible');
      }

      return {
        bodyBg: css.bodyBg,
        hasTealInk: css.hasTealInk,
        hasSignalPink: css.hasSignalPink,
        hasSurfaceCream: css.hasSurfaceCream,
        hasLegacyNeon: css.hasLegacyNeon,
        tickerText: H.getText(topTicker).slice(0, 160),
      };
    `);
    return data;
  },

  async h2h_odds_swap_matrix(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    const data = await evalPage(cdp, `
      // Ensure H2H Arena mode
      const arenaTab = H.findClickable([/h2h|matchup\\s*arena|arena/i]);
      if (arenaTab) arenaTab.click();
      await H.ensureArenaRailExpanded();

      const p1Sel = await H.waitFor(() => {
        const s = H.findSelect({ optionPatterns: [/Alcaraz/i], labelPatterns: [/p1|player\\s*1|left/i], index: 0 });
        return s || H.findSelect({ optionPatterns: [/Alcaraz/i], index: 0 });
      }, 10000, 'P1 player select');

      const p2Sel = H.findSelect({ optionPatterns: [/Sinner/i], labelPatterns: [/p2|player\\s*2|right/i], index: 0 }) ||
        H.findSelect({ optionPatterns: [/Sinner/i], index: 1 }) ||
        H.findSelect({ optionPatterns: [/Sinner/i], index: 0 });

      const p1Initial = H.getSelectedText(p1Sel);
      const p2Initial = H.getSelectedText(p2Sel);
      if (!/Alcaraz/i.test(p1Initial) || !/Sinner/i.test(p2Initial)) {
        throw new Error('Expected default players Alcaraz C. & Sinner J., got P1=' + p1Initial + ', P2=' + p2Initial);
      }

      const odd1Input = H.findInput([/odd.?1|p1.?odd|player.?1.?odd/i]) || H.findInput([/odd/i], { index: 0 });
      const odd2Input = H.findInput([/odd.?2|p2.?odd|player.?2.?odd/i]) || H.findInput([/odd/i], { index: 1 });
      if (!odd1Input || !odd2Input) {
        throw new Error('Could not find odd_1 / odd_2 inputs');
      }
      H.setInputValue(odd1Input, '1.85');
      H.setInputValue(odd2Input, '2.05');

      H.clickControl([/^predict$|run\\s*predict|analyze|calculate/i]);

      // Wait for Edge/Kelly & probabilities & Arcade cards
      await H.waitFor(() => {
        const txt = H.getText(document.body);
        const pcts = txt.match(/\\b\\d{1,2}(?:\\.\\d+)?%/g) || [];
        const hasEdgeKelly = /edge|kelly|value/i.test(txt);
        const arcade = H.checkArcadeAndCharts();
        return pcts.length >= 2 && hasEdgeKelly && arcade.ovrCount >= 2 && arcade.hasRadarGeometry;
      }, 15000, 'prediction, Edge/Kelly, Arcade cards, and Radar SVG');

      const arcade = H.checkArcadeAndCharts();
      if (arcade.radarAxesMatched < 6) {
        throw new Error('Expected all 6 radar axes labels, matched ' + arcade.radarAxesMatched + ': ' + JSON.stringify(arcade));
      }
      if (!arcade.hasTreeShap) {
        throw new Error('TreeSHAP tug-of-war section not visible');
      }

      // Expand 31-Feature Matrix and assert 31 rows
      const featureRows = await H.waitFor(() => {
        const count = H.expandAndCount31Features();
        return count >= 31 ? count : false;
      }, 8000, '31-feature matrix expansion');

      // Capture P1 win probability before swap
      function extractMainProbPair() {
        const probEls = Array.from(document.querySelectorAll('[data-prob], [data-testid*="prob" i], [class*="prob" i], [id*="prob" i], .prob, strong, span, div'))
          .filter((el) => H.isVisible(el) && el.children.length === 0)
          .map((el) => H.getText(el))
          .filter((t) => /^\\d{1,2}(?:\\.\\d+)?%$/.test(t))
          .map((t) => Number.parseFloat(t.replace('%', '')));
        for (let i = 0; i < probEls.length - 1; i += 1) {
          const a = probEls[i];
          const b = probEls[i + 1];
          if (a > 0 && b > 0 && Math.abs((a + b) - 100) <= 0.5) {
            return [a, b];
          }
        }
        return null;
      }

      const beforePair = extractMainProbPair();
      if (!beforePair) {
        throw new Error('Could not extract complementary P1/P2 probability pair before swap');
      }

      // Click Swap (⇄)
      H.clickControl([/⇄|swap|invert/i]);

      // Wait for players to swap and probabilities to invert
      const afterPair = await H.waitFor(() => {
        const p1Now = H.getSelectedText(p1Sel);
        const p2Now = H.getSelectedText(p2Sel);
        if (!/Sinner/i.test(p1Now) || !/Alcaraz/i.test(p2Now)) return false;
        const pair = extractMainProbPair();
        if (!pair) return false;
        if (Math.abs((pair[0] + beforePair[0]) - 100) <= 1.2) {
          return pair;
        }
        return false;
      }, 10000, 'player swap and inverted probabilities');

      return {
        p1Initial,
        p2Initial,
        featureRows,
        beforePair,
        afterPair,
        arcade,
      };
    `);

    return data;
  },

  async alltime_legacy_and_divergence(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    const data = await evalPage(cdp, `
      const arenaTab = H.findClickable([/h2h|matchup\\s*arena|arena/i]);
      if (arenaTab) arenaTab.click();
      await H.ensureArenaRailExpanded();

      const p1Sel = await H.waitFor(
        () => H.findSelect({ optionPatterns: [/Alcaraz/i], labelPatterns: [/p1|player\\s*1|left/i], index: 0 }) || H.findSelect({ optionPatterns: [/Alcaraz/i], index: 0 }),
        10000,
        'P1 player select'
      );
      const initialOptionCount = p1Sel.options.length;

      // Toggle All-Time Historical Roster (2000-2026)
      H.setToggle([/all[- ]?time|historical\\s*roster|2000.?2026/i], true);

      const expandedCount = await H.waitFor(() => {
        return p1Sel.options.length > 1000 ? p1Sel.options.length : false;
      }, 10000, 'all-time roster (> 1000 options)');

      const p2Sel = H.findSelect({ optionPatterns: [/Federer|Nadal|Sinner/i], labelPatterns: [/p2|player\\s*2|right/i], index: 0 }) ||
        H.findSelect({ optionPatterns: [/Federer|Nadal|Sinner/i], index: 1 });
      const surfSel = H.findSelect({ optionPatterns: [/^Clay$/i, /^Hard$/i] });
      const seriesSel = H.findSelect({ optionPatterns: [/Grand Slam/i] });

      H.setSelectOption(p1Sel, 'Federer R.');
      H.setSelectOption(p2Sel, 'Nadal R.');
      if (surfSel) H.setSelectOption(surfSel, 'Clay');
      if (seriesSel) H.setSelectOption(seriesSel, 'Grand Slam');

      H.clickControl([/^predict$|run\\s*predict|analyze|calculate/i]);

      await H.waitFor(() => {
        const txt = H.getText(document.body);
        const hasCards = /Federer/i.test(txt) && /Nadal/i.test(txt);
        if (!hasCards) return false;

        const h2hSelectors = [
          '[data-testid*="h2h" i]',
          '[id*="h2h" i]',
          '[class*="h2h" i]',
          '[data-testid*="key-stats" i]',
          '[id*="key-stats" i]',
          '[id*="keystats" i]',
          '[class*="key-stats" i]',
          '[data-testid*="tape" i]',
          '[class*="tape" i]',
          '[data-testid*="card" i]',
          '[class*="card" i]',
        ].join(', ');
        const regionTexts = Array.from(document.querySelectorAll(h2hSelectors))
          .filter((el) => H.isVisible(el))
          .map((el) => H.getText(el));
        const labelContexts = Array.from(
          document.querySelectorAll('div, section, article, span, td, th, dt, dd, p, strong, li')
        )
          .filter((el) => H.isVisible(el) && el.children.length <= 4 && /\\bh2h\\b|head[- ]to[- ]head/i.test(H.getText(el)))
          .map((el) => {
            const box = el.closest('section, article, div, tr') || el.parentElement || el;
            const t = H.getText(box);
            return t.length <= 1500 ? t : H.getText(el);
          });
        const h2hText = [...regionTexts, ...labelContexts].join(' ');
        const has16 = /\\b16\\b/.test(h2hText);
        const has40or24 = /\\b40\\b/.test(h2hText) || /\\b24\\b/.test(h2hText);
        return has16 && has40or24;
      }, 12000, 'Federer vs Nadal H2H 40 / 16 / 24 inside H2H/key-stats/card region');

      // Now Sinner J. vs Djokovic N. on Hard (Grand Slam) -> Divergence alert/gauge
      H.setSelectOption(p1Sel, 'Sinner J.');
      H.setSelectOption(p2Sel, 'Djokovic N.');
      if (surfSel) H.setSelectOption(surfSel, 'Hard');
      if (seriesSel) H.setSelectOption(seriesSel, 'Grand Slam');

      H.clickControl([/^predict$|run\\s*predict|analyze|calculate/i]);

      const divergenceFound = await H.waitFor(() => {
        const txt = H.getText(document.body);
        const hasPlayers = /Sinner/i.test(txt) && /Djokovic/i.test(txt);
        const hasDiv = /divergen|disagree|>\\s*15%|non-linear/i.test(txt);
        return hasPlayers && hasDiv ? true : false;
      }, 12000, 'Sinner vs Djokovic divergence indicator');

      return {
        initialOptionCount,
        expandedCount,
        divergenceFound,
      };
    `);

    return data;
  },

  async historical_backtracker_flow(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    const data = await evalPage(cdp, `
      H.clickControl([/historical|backtrack|replay/i]);

      const splitSel = await H.waitFor(
        () => H.findSelect({ optionPatterns: [/Holdout|Full Dataset/i] }),
        10000,
        'Historical split select'
      );
      H.setSelectOption(splitSel, /Full Dataset/i);

      const tournSel = await H.waitFor(() => {
        const allSelects = Array.from(document.querySelectorAll('select')).filter((s) => H.isVisible(s));
        const candidate = allSelects.find((s) => s !== splitSel && s.options.length > 5 && !/Holdout|Full Dataset/i.test(s.options[0]?.textContent || ''));
        return candidate || false;
      }, 12000, 'Historical tournament select populated');

      // Pick first non-placeholder tournament
      const firstTourn = Array.from(tournSel.options).find((o) => o.value && !/^select/i.test(o.textContent.trim()));
      if (!firstTourn) throw new Error('No selectable tournament found');
      H.setSelectOption(tournSel, firstTourn.value);

      const matchSel = await H.waitFor(() => {
        const allSelects = Array.from(document.querySelectorAll('select')).filter((s) => H.isVisible(s));
        const candidate = allSelects.find((s) => s !== splitSel && s !== tournSel && s.options.length >= 1);
        if (!candidate) return false;
        const validOpt = Array.from(candidate.options).find((o) => o.value && !/^select/i.test(o.textContent.trim()));
        return validOpt ? candidate : false;
      }, 12000, 'Historical match select populated');

      const firstMatch = Array.from(matchSel.options).find((o) => o.value && !/^select/i.test(o.textContent.trim()));
      H.setSelectOption(matchSel, firstMatch.value);

      H.clickControl([/inspect|replay|analyze|load\\s*match/i]);

      const inspected = await H.waitFor(() => {
        const txt = H.getText(document.body);
        const hasOutcome = /PREDICTION ACCURATE|UPSET/i.test(txt);
        const hasWinner = /winner/i.test(txt);
        const hasScore = /\\d+-\\d+/.test(txt);
        const hasBm = /bookmaker|implied/i.test(txt);
        const arcade = H.checkArcadeAndCharts();
        if (hasOutcome && hasWinner && hasScore && hasBm && arcade.ovrCount >= 2 && arcade.hasRadarGeometry && arcade.hasTreeShap) {
          return { arcade };
        }
        return false;
      }, 15000, 'Historical inspect rich payload');

      const featureRows = await H.waitFor(() => {
        const count = H.expandAndCount31Features();
        return count >= 31 ? count : false;
      }, 8000, 'Historical 31-feature matrix expansion');

      return {
        tournament: firstTourn.textContent.trim(),
        match: firstMatch.textContent.trim(),
        featureRows,
        ...inspected,
      };
    `);

    return data;
  },

  async daily_schedule_filters_and_ticker(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    await evalPage(cdp, `
      const schedTab = document.getElementById('tabSchedule') || H.findClickable([/today'?s\\s*slate|daily\\s*schedule|live\\s*board/i]);
      if (!schedTab) throw new Error('Today\\'s Slate / Daily Schedule tab not found');
      schedTab.click();

      await H.waitFor(() => {
        const txt = H.getText(document.body);
        return /slate|schedule|fixtures|no\\s+(?:live|daily|upcoming)|refresh/i.test(txt);
      }, 8000, 'Daily Schedule mode render');

      const refreshBtn = H.findClickable([/refresh/i]);
      if (!refreshBtn) throw new Error('Refresh Schedule button not found');
      refreshBtn.click();
      return true;
    `);

    // Allow browser event loop to dispatch any fetch, then wait until telemetry.inFlightCount === 0 (up to 8s)
    await sleep(150);
    const refreshDeadline = Date.now() + 8000;
    while (Date.now() < refreshDeadline) {
      if (telemetry.inFlightCount === 0) {
        break;
      }
      await sleep(100);
    }
    if (telemetry.inFlightCount !== 0) {
      throw new Error(`Refresh network requests did not settle within 8000ms (inFlightCount=${telemetry.inFlightCount})`);
    }

    const data = await evalPage(cdp, `
      await H.waitFor(() => Boolean(H.findClickable([/refresh/i])), 4000, 'Refresh button re-enabled after request');

      // Test Series filter pills (Grand Slam, All) if rendered
      let testedFilters = [];
      const gsFilter = H.findClickable([/^Grand Slam$/i]);
      if (gsFilter) {
        gsFilter.click();
        testedFilters.push('Grand Slam');
      }
      const allFilter = H.findClickable([/^All(\\s+Series|\\s+Matches)?$/i]);
      if (allFilter) {
        allFilter.click();
        testedFilters.push('All');
      }

      // Test >=3% Value Edge toggle if rendered
      const edgeCtrl = H.findClickable([/3%|value\\s*edge/i]);
      if (edgeCtrl) {
        edgeCtrl.click();
        edgeCtrl.click();
        testedFilters.push('>=3% Edge');
      } else {
        try {
          H.setToggle([/3%|value\\s*edge/i], true);
          H.setToggle([/3%|value\\s*edge/i], false);
          testedFilters.push('>=3% Edge checkbox');
        } catch {
          // Optional if no checkbox present
        }
      }

      if (testedFilters.length === 0) {
        throw new Error('No Series filter pills (Grand Slam / All) or Value Edge toggle controls found on Daily Schedule Board');
      }

      // If any fixture/ticker inspect action exists, click it and verify transition
      let inspectedFromSchedule = false;
      const inspectAction = H.findClickable([/open\\s+dossier|open\\s+in\\s+workbench|inspect|open\\s+in\\s+arena|load\\s+arena/i]);
      if (inspectAction) {
        inspectAction.click();
        inspectedFromSchedule = true;
      }

      return {
        testedFilters,
        inspectedFromSchedule,
      };
    `);

    return data;
  },

  async mobile_viewport_layout(cdp, baseUrl, telemetry) {
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    const data = await evalPage(cdp, `
      const routes = ['#slate', '#arena', '#backtracker'];
      for (const r of routes) {
        window.location.hash = r;
        await new Promise((res) => setTimeout(res, 180));
        const sw = document.documentElement.scrollWidth;
        const iw = window.innerWidth;
        if (sw > iw + 2) {
          throw new Error('Horizontal overflow on mobile viewport (' + r + '): scrollWidth=' + sw + ' > innerWidth=' + iw);
        }
      }
      window.location.hash = '#slate';
      await new Promise((res) => setTimeout(res, 120));
      const scrollWidth = document.documentElement.scrollWidth;
      const innerWidth = window.innerWidth;
      const bodyRect = document.body.getBoundingClientRect();
      if (bodyRect.height < 200) {
        throw new Error('Page body collapsed on mobile viewport: height=' + bodyRect.height);
      }
      return { scrollWidth, innerWidth, bodyHeight: bodyRect.height };
    `);

    return data;
  },

  async invalid_odds_ui(cdp, baseUrl, telemetry) {
    telemetry.allowInvalidPredictHttpError = true;
    await navigateAndHydrate(cdp, baseUrl, telemetry);

    const data = await evalPage(cdp, `
      const arenaTab = H.findClickable([/h2h|matchup\\s*arena|arena/i]);
      if (arenaTab) arenaTab.click();
      await H.ensureArenaRailExpanded();

      const odd1Input = await H.waitFor(
        () => H.findInput([/odd.?1|p1.?odd|player.?1.?odd/i]) || H.findInput([/odd/i], { index: 0 }),
        10000,
        'odd_1 input'
      );
      const odd2Input = H.findInput([/odd.?2|p2.?odd|player.?2.?odd/i]) || H.findInput([/odd/i], { index: 1 });
      if (!odd1Input || !odd2Input) throw new Error('Odds inputs not found');

      H.setInputValue(odd1Input, '1.0');
      H.setInputValue(odd2Input, '2.0');

      H.clickControl([/^predict$|run\\s*predict|analyze|calculate/i]);
      await new Promise((r) => setTimeout(r, 1500));

      const bodyText = H.getText(document.body);
      if (/\\bNaN\\b|NaN%|Infinity|-Infinity|undefined%/i.test(bodyText)) {
        throw new Error('Invalid numeric display detected in UI for odd_1=1.0: ' + bodyText.match(/(?:NaN|Infinity|undefined)[^\\n]*/i)?.[0]);
      }
      return { checked: true };
    `);

    return data;
  },
};

async function main() {
  const args = parseArgs(process.argv);
  const scenarioFn = SCENARIOS[args.scenario];
  if (!scenarioFn) {
    throw new Error(`Unknown scenario: ${args.scenario}. Available: ${Object.keys(SCENARIOS).join(', ')}`);
  }

  const chromeBin = process.env.CHROME_BIN || '/usr/bin/chromium';
  const cdpPort = await getFreePort();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'matchpoint-cdp-'));

  const chromeArgs = [
    '--headless',
    '--no-sandbox',
    '--disable-gpu',
    '--disable-dev-shm-usage',
    '--no-first-run',
    '--no-default-browser-check',
    `--remote-debugging-port=${cdpPort}`,
    `--user-data-dir=${userDataDir}`,
    `--window-size=${args.width},${args.height}`,
    'about:blank',
  ];

  const chromeProc = spawn(chromeBin, chromeArgs, {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  });

  let chromeStderr = '';
  chromeProc.stderr.on('data', (chunk) => {
    chromeStderr += chunk.toString();
  });

  let cdp = null;
  let cleanedUp = false;

  async function cleanupChromeAndTempDir() {
    if (cleanedUp) return;
    cleanedUp = true;

    if (cdp) {
      cdp.close();
    }

    if (chromeProc && chromeProc.pid) {
      const stillAlive = chromeProc.exitCode === null && chromeProc.signalCode === null;
      if (stillAlive) {
        const waitForExit = (timeoutMs) =>
          new Promise((resolve) => {
            if (chromeProc.exitCode !== null || chromeProc.signalCode !== null) {
              resolve();
              return;
            }
            const timer = setTimeout(() => {
              chromeProc.off('exit', onDone);
              resolve();
            }, timeoutMs);
            timer.unref?.();
            function onDone() {
              clearTimeout(timer);
              resolve();
            }
            chromeProc.once('exit', onDone);
          });

        try {
          process.kill(-chromeProc.pid, 'SIGTERM');
        } catch {
          try {
            chromeProc.kill('SIGTERM');
          } catch {
            // Ignore kill error
          }
        }
        await waitForExit(400);

        if (chromeProc.exitCode === null && chromeProc.signalCode === null) {
          try {
            process.kill(-chromeProc.pid, 'SIGKILL');
          } catch {
            try {
              chromeProc.kill('SIGKILL');
            } catch {
              // Ignore kill error
            }
          }
          await waitForExit(400);
        }
      }
      try {
        chromeProc.stdout?.destroy();
        chromeProc.stderr?.destroy();
      } catch {
        // Ignore stream destroy error
      }
    }

    try {
      fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 });
    } catch {
      // Ignore temp dir cleanup errors
    }
  }

  const onSignal = async (sig) => {
    await cleanupChromeAndTempDir();
    process.exit(sig === 'SIGINT' ? 130 : 143);
  };
  process.once('SIGINT', onSignal);
  process.once('SIGTERM', onSignal);

  const telemetry = {
    consoleErrors: [],
    pageExceptions: [],
    externalRequests: [],
    httpErrors: [],
    failedRequests: [],
    expectedInvalidResponses: [],
    allowInvalidPredictHttpError: false,
    inFlight: new Set(),
    inFlightCount: 0,
    requestUrls: new Map(),
  };

  try {
    const target = await waitForCdpTarget(cdpPort, 15000);
    cdp = new CdpClient(target.webSocketDebuggerUrl);

    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('Network.enable');
    await cdp.send('DOM.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: args.width,
      height: args.height,
      deviceScaleFactor: args.width <= 450 ? 2 : 1,
      mobile: args.width <= 450,
    });

    cdp.on('Runtime.consoleAPICalled', (params) => {
      if (params.type === 'error') {
        const text = (params.args || [])
          .map((a) => a.value ?? a.description ?? '')
          .join(' ')
          .trim();
        telemetry.consoleErrors.push({
          text: text || 'console.error',
          stackTrace: params.stackTrace || null,
        });
      }
    });

    cdp.on('Runtime.exceptionThrown', (params) => {
      const details = params.exceptionDetails || {};
      telemetry.pageExceptions.push({
        text: details.exception?.description || details.text || 'Uncaught exception',
        lineNumber: details.lineNumber,
        columnNumber: details.columnNumber,
      });
    });

    cdp.on('Network.requestWillBeSent', (params) => {
      const reqId = params.requestId;
      const url = params.request?.url || '';
      if (reqId) {
        telemetry.inFlight.add(reqId);
        telemetry.inFlightCount = telemetry.inFlight.size;
        telemetry.requestUrls.set(reqId, url);
      }
      if (url.startsWith('http://') || url.startsWith('https://')) {
        try {
          const parsed = new URL(url);
          const allowedHosts = new Set([
            '127.0.0.1',
            'localhost',
            'fonts.googleapis.com',
            'fonts.gstatic.com',
          ]);
          if (!allowedHosts.has(parsed.hostname)) {
            telemetry.externalRequests.push({ url, type: params.type || 'Other' });
          }
        } catch {
          telemetry.externalRequests.push({ url, type: params.type || 'InvalidURL' });
        }
      }
    });

    cdp.on('Network.loadingFinished', (params) => {
      if (params.requestId) {
        telemetry.inFlight.delete(params.requestId);
        telemetry.inFlightCount = telemetry.inFlight.size;
      }
    });

    cdp.on('Network.loadingFailed', (params) => {
      if (params.requestId) {
        telemetry.inFlight.delete(params.requestId);
        telemetry.inFlightCount = telemetry.inFlight.size;
      }
      if (!params.canceled) {
        telemetry.failedRequests.push({
          url: telemetry.requestUrls.get(params.requestId) || '',
          errorText: params.errorText || 'loadingFailed',
          type: params.type || 'Other',
        });
      }
    });

    cdp.on('Network.responseReceived', (params) => {
      const status = params.response?.status ?? 0;
      const url = params.response?.url || '';
      if (status === 404) {
        try {
          if (new URL(url).pathname.endsWith('/favicon.ico')) {
            return;
          }
        } catch {
          if (url.endsWith('/favicon.ico')) {
            return;
          }
        }
      }
      if (status >= 400) {
        if (
          telemetry.allowInvalidPredictHttpError &&
          url.includes('/api/v1/predict') &&
          (status === 400 || status === 422)
        ) {
          telemetry.expectedInvalidResponses.push({ url, status });
        } else {
          telemetry.httpErrors.push({
            url,
            status,
            statusText: params.response?.statusText || '',
            type: params.type || 'Other',
          });
        }
      }
    });

    const scenarioData = await raceWithTimeout(
      scenarioFn(cdp, args.baseUrl, telemetry),
      args.timeoutMs,
      `Scenario '${args.scenario}'`
    );

    const errors = [];
    if (telemetry.externalRequests.length > 0) {
      errors.push(`External CDN/network requests detected: ${JSON.stringify(telemetry.externalRequests)}`);
    }
    if (telemetry.consoleErrors.length > 0) {
      errors.push(`Console errors: ${JSON.stringify(telemetry.consoleErrors)}`);
    }
    if (telemetry.pageExceptions.length > 0) {
      errors.push(`Uncaught page exceptions: ${JSON.stringify(telemetry.pageExceptions)}`);
    }
    if (telemetry.httpErrors.length > 0) {
      errors.push(`HTTP >= 400 responses: ${JSON.stringify(telemetry.httpErrors)}`);
    }
    if (telemetry.failedRequests.length > 0) {
      errors.push(`Failed network requests: ${JSON.stringify(telemetry.failedRequests)}`);
    }

    const result = {
      ok: errors.length === 0,
      scenario: args.scenario,
      viewport: { width: args.width, height: args.height },
      data: scenarioData,
      errors,
      telemetry: {
        consoleErrors: telemetry.consoleErrors,
        pageExceptions: telemetry.pageExceptions,
        externalRequests: telemetry.externalRequests,
        httpErrors: telemetry.httpErrors,
        failedRequests: telemetry.failedRequests,
        expectedInvalidResponses: telemetry.expectedInvalidResponses,
      },
    };

    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = result.ok ? 0 : 1;
  } catch (err) {
    const result = {
      ok: false,
      scenario: args.scenario,
      viewport: { width: args.width, height: args.height },
      error: err instanceof Error ? err.message : String(err),
      stack: err instanceof Error ? err.stack : null,
      chromeStderr: chromeStderr.slice(-2000),
      telemetry: {
        consoleErrors: telemetry.consoleErrors,
        pageExceptions: telemetry.pageExceptions,
        externalRequests: telemetry.externalRequests,
        httpErrors: telemetry.httpErrors,
        failedRequests: telemetry.failedRequests,
        expectedInvalidResponses: telemetry.expectedInvalidResponses,
      },
    };
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = 1;
  } finally {
    process.off('SIGINT', onSignal);
    process.off('SIGTERM', onSignal);
    await cleanupChromeAndTempDir();
  }
}

main();
