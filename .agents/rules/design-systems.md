---
trigger: glob
globs: ["**/*.html", "**/*.css", "**/*.scss", "**/*.jsx", "**/*.tsx", "**/*.vue", "**/*.svelte"]
description: Enforces the locked Risograph Option A color palette, Sogea/Jakarta/JetBrains typography hierarchy, and tennis domain UI rules for baseline.ml.
---

# matchpoint.ml — Design System & UI Invariants

All UI components, styles, and layouts MUST strictly consume tokens defined in @src/styles/tokens.css. Never introduce dark-mode themes, arbitrary Tailwind default colors, or generic sportsbook neon palettes.

## 1. The 60 / 30 / 10 Color Budget (Non-Negotiable)
- **60% Cream Paper (`--bg-canvas` `#F8DBC1`, `--bg-surface` `#FCE7D4`, `--bg-recessed` `#EFCCA9`):** All page backdrops, card interiors, and recessed bar tracks. Never use `#FFFFFF` pure white.
- **30% Riso Teal (`--ink-primary` `#1D646C`, `--ink-deep` `#13454B`, `--ink-muted` `#3E7D84`):** All primary copy, borders, table rules, underdog probability bars, and SVG court/racket schematics.
- **10% Fluorescent Pink (`--accent-signal` `#EA4F88`):** Strictly capped at <=10% viewport area. Reserve exclusively for:
  1. Primary CTAs and active filter pills.
  2. The **favored player's** win-probability bar fill (underdog stays in `--ink-muted`).
  3. Positive expected value (`+EV`) badges and primary model calibration curves.
  4. Court shot-bounce hotspots and racket sweet-spot overlays.
- **Inline Pink Text:** When rendering pink text directly on `#F8DBC1` or `#FCE7D4`, use `--accent-text` (`#C43269`) for WCAG AA legibility.

## 2. Three-Tier Typography & "One Diva Per Stage" Rule
- **Display & Wordmark (`--font-display` / `Sogea`):**
  - **Budget:** `< 10%` of page text. Use ONLY for the `baseline.ml` wordmark and 2–4 word hero/section anchors.
  - **Hard Floor:** NEVER render `Sogea` below `32px` (`2rem`).
  - **Casing & Styling:** Use sentence case or lowercase (`Sharp court edge.`, `Today's court slate.`). Render in flat `--ink-primary` (`#1D646C`) with at most one focal word/period in `--accent-signal`. Never apply gradients, outlines, or drop shadows to `Sogea` text.
  - **Clean Paper Firewall:** Keep `Sogea` headlines on flat `--bg-canvas` (`#F8DBC1`) with `>= 1.5x` cap-height breathing room. Never place `Sogea` directly over busy mesh gradients or court illustrations.
- **UI & Analytical Copy (`--font-sans` / `Plus Jakarta Sans`):** All player names, tournament metadata, filter pills, SHAP feature descriptions, and body copy (`12px–16px`).
- **Telemetry & Box-Score Figures (`--font-mono` / `JetBrains Mono`):** MANDATORY for all win probabilities (`64.8%`), decimal/American odds (`1.54`), `+EV`/`-EV` edges (`+4.2%`), Brier/Log-Loss metrics, CPI court speed ratings, and Kelly stakes. Always enforce `font-variant-numeric: tabular-nums`.

## 3. Risograph Tactile Craft & Tennis Artifacts
- Maintain a subtle paper-grain overlay (`feTurbulence` noise at `3%–5%` opacity with `mix-blend-mode: multiply`) on the root viewport.
- Contain `--mesh-print-plate` radial gradients and halftone dot fills inside featured Print Plate cards and Hero visual containers.
- Use hard offset Risograph borders (`1.5px solid var(--ink-primary)` with `box-shadow: var(--shadow-riso-card)`) rather than blurred SaaS drop shadows.
- Separate major landing sections with a diamond-mesh tennis net SVG divider strip (`--ink-primary` cord with `--accent-signal` center strap).

## 4. Forbidden Anti-Patterns
- Do NOT use pure red (`#FF0000`) for `-EV` or underdog states; use `--state-fade` (`#B54434`).
- Do NOT color both players' probability bars in Fluorescent Pink.
- Do NOT use `Sogea` inside match cards, navigation links, buttons, or data tables.
