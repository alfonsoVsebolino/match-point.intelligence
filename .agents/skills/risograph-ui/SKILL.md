---
name: risograph-ui
description: Scaffolds new frontend views or refactors existing UI components to conform to the baseline.ml Risograph Option A design system (Cream Paper, Riso Teal, Fluorescent Pink, and Sogea/Plus Jakarta Sans/JetBrains Mono). Use when building or restyling frontend pages, match cards, charts, or calculators.
---

# Risograph UI Implementation & Audit Workflow

Follow these steps in exact order whenever creating or migrating UI components for `baseline.ml`:

## Step 1: Token & Font Wiring
1. Verify `@src/styles/tokens.css` is imported at the application root.
2. Confirm Google Fonts (`Plus Jakarta Sans:wght@400;500;600;700` and `JetBrains Mono:wght@500;700`) and local `/fonts/Sogea-Regular.woff2` are loaded.
3. Ensure the root layout includes the subtle SVG `feTurbulence` paper-grain overlay (`pointer-events: none; mix-blend-mode: multiply`).

## Step 2: Component Token Replacement
1. Replace all hardcoded background hex/Tailwind colors with `--bg-canvas` (`#F8DBC1`), `--bg-surface` (`#FCE7D4`), or `--bg-recessed` (`#EFCCA9`).
2. Replace all text and border colors with `--ink-primary` (`#1D646C`), `--ink-deep` (`#13454B`), or `--ink-muted` (`#3E7D84`).
3. Audit `--accent-signal` (`#EA4F88`) usage: strip it from any non-favored bars, secondary buttons, or decorative borders so total pink coverage stays under 10%.

## Step 3: Typography & Numeric Verification
1. Inspect every element using `--font-display` (`Sogea`): confirm computed `font-size >= 32px`, phrase length `<= 4 words`, and flat `--bg-canvas` background clearance.
2. Wrap all probabilities, odds, EV percentages, calibration metrics, and currency/stake outputs in `.metric-tabular` (`JetBrains Mono` + `tabular-nums`).

## Step 4: Visual Craft Check
1. Apply `1.5px solid var(--ink-primary)` borders and `--shadow-riso-card` (`4px 4px 0px #1D646C`) to primary cards.
2. Verify surface badges (`Hard`, `Clay`, `Grass`) use their desaturated semantic tokens (`--surface-hard`, `--surface-clay`, `--surface-grass`) so they do not compete with `--accent-signal`.
