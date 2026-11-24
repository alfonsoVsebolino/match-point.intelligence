def render_diagnostic_card(
    p1: str, p2: str, surface: str, series: str,
    prob_lgb: float, prob_lr: float, key_stats: dict,
    sym_score: float = 0.0, raw_asym: float = 0.0,
    is_divergent: bool = False, div_delta: float = 0.0,
    bm_prob: float = None, actual_winner: str = None,
    score_str: str = None, date_str: str = None,
    round_name: str = None, tournament: str = None,
    is_historical: bool = False
) -> str:
    p2_prob_lgb = 1.0 - prob_lgb
    p2_prob_lr = 1.0 - prob_lr
    
    # Surface & Series colors
    surf_colors = {
        "Hard": "#38bdf8", "Clay": "#fb923c",
        "Grass": "#4ade80", "Carpet": "#a78bfa"
    }
    surf_col = surf_colors.get(surface, "#2dd4bf")
    
    # Mode B: Winner verification & Score display
    winner_badge_html = ""
    if is_historical and actual_winner:
        model_favors_p1 = prob_lgb >= 0.5
        actual_p1_won = actual_winner == p1
        correct = (model_favors_p1 == actual_p1_won)
        
        badge_bg = "rgba(16, 185, 129, 0.15)" if correct else "rgba(244, 63, 94, 0.15)"
        badge_border = "#10b981" if correct else "#f43f5e"
        badge_text = "#10b981" if correct else "#f43f5e"
        badge_icon = "✓" if correct else "✗"
        
        score_html = f"<span style='color:#94a3b8; font-size:12px; margin-left:8px; font-family:\"JetBrains Mono\", monospace;'>Score: <strong style='color:#f8fafc;'>{score_str}</strong></span>" if score_str else ""
        
        winner_badge_html = f"""
        <div style="display:flex; align-items:center; justify-content:space-between; background:{badge_bg}; border:1px solid {badge_border}; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:{badge_text}; font-weight:800; font-size:16px;">{badge_icon}</span>
                <span style="color:#e2e8f0; font-size:13px; font-weight:600;">Actual Winner: <strong style="color:#ffffff;">{actual_winner}</strong></span>
                {score_html}
            </div>
            <span style="font-family:'JetBrains Mono', monospace; font-size:11px; padding:2px 8px; border-radius:4px; background:{badge_border}; color:#0b0f19; font-weight:700;">
                {'PREDICTION ACCURATE' if correct else 'UPSET / DIVERGENT'}
            </span>
        </div>
        """
        
    # Divergence alert
    divergence_banner_html = ""
    if is_divergent:
        divergence_banner_html = f"""
        <div style="display:flex; align-items:center; gap:8px; background:rgba(245, 158, 11, 0.12); border:1px solid #f59e0b; border-radius:8px; padding:10px 14px; margin-bottom:14px;">
            <span style="color:#f59e0b; font-size:16px;">⚠️</span>
            <div style="font-size:12px; color:#fde68a; font-family:'Plus Jakarta Sans', sans-serif;">
                <strong>Non-linear divergence detected (|ΔP| = {div_delta*100:.1f}%)</strong>: LightGBM captures non-linear interactions unmodeled by regularized Logistic Regression.
            </div>
        </div>
        """
        
    # Implied Bookmaker Prob Bar (Historical mode)
    bm_html = ""
    if bm_prob is not None:
        bm_p2_prob = 1.0 - bm_prob
        bm_html = f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#94a3b8;">Market Implied (Bookmaker Odds)</span>
                <span style="color:#cbd5e1; font-weight:600;">{bm_prob*100:.1f}% vs {bm_p2_prob*100:.1f}%</span>
            </div>
            <div style="height:6px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{bm_prob*100:.1f}%; background:#94a3b8; transition:width 0.4s ease;"></div>
                <div style="width:{bm_p2_prob*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>
        """
        
    # Subtitle with Date / Tournament info
    sub_title = f"{series} • {surface} Court"
    if is_historical and tournament and date_str:
        sub_title = f"{date_str} • {tournament} ({round_name}) • {surface}"

    # Mode A: Symmetry badge
    symmetry_chip_html = ""
    if not is_historical:
        symmetry_chip_html = f"""
        <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3); color:#10b981; font-weight:600;" title="Raw Asymmetry: {raw_asym*100:.2f}% | Symmetrized Error: {sym_score:.1e}">
            Symmetry: Δ &lt; 1e-5 (Raw Δ: {raw_asym*100:.1f}%)
        </span>
        """

    card_html = f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:20px; font-family:'Plus Jakarta Sans', sans-serif; color:#f8fafc; box-shadow:0 10px 25px -5px rgba(0, 0, 0, 0.4);">
        <!-- Match Header -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; border-bottom:1px solid #1e293b; padding-bottom:12px;">
            <div>
                <div style="font-size:18px; font-weight:800; tracking-tight:tight; color:#ffffff;">
                    {p1} <span style="color:#64748b; font-weight:400;">vs</span> {p2}
                </div>
                <div style="font-size:12px; color:#94a3b8; margin-top:2px; font-family:'JetBrains Mono', monospace;">
                    {sub_title}
                </div>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
                <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:rgba(20, 184, 166, 0.1); border:1px solid rgba(20, 184, 166, 0.3); color:{surf_col}; font-weight:700;">
                    {surface}
                </span>
                <span style="font-size:11px; font-family:'JetBrains Mono', monospace; padding:3px 8px; border-radius:6px; background:#1e293b; color:#cbd5e1; font-weight:600;">
                    {series}
                </span>
                {symmetry_chip_html}
            </div>
        </div>

        {winner_badge_html}
        {divergence_banner_html}

        <!-- Dual Model Probability Bars -->
        <!-- LightGBM Champion -->
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#2dd4bf; font-weight:700;">★ LightGBM Champion</span>
                <span>
                    <strong style="color:#2dd4bf;">{prob_lgb*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p1})</span>
                    <span style="color:#64748b; margin:0 4px;">|</span>
                    <strong style="color:#94a3b8;">{p2_prob_lgb*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p2})</span>
                </span>
            </div>
            <div style="height:12px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{prob_lgb*100:.1f}%; background:linear-gradient(90deg, #0d9488, #2dd4bf); transition:width 0.4s ease;"></div>
                <div style="width:{p2_prob_lgb*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>

        <!-- Logistic Regression Baseline -->
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px; font-family:'JetBrains Mono', monospace;">
                <span style="color:#60a5fa; font-weight:600;">Logistic Regression Baseline</span>
                <span>
                    <strong style="color:#60a5fa;">{prob_lr*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p1})</span>
                    <span style="color:#64748b; margin:0 4px;">|</span>
                    <strong style="color:#94a3b8;">{p2_prob_lr*100:.1f}%</strong>
                    <span style="color:#64748b;"> ({p2})</span>
                </span>
            </div>
            <div style="height:8px; background:#1e293b; border-radius:999px; overflow:hidden; display:flex;">
                <div style="width:{prob_lr*100:.1f}%; background:linear-gradient(90deg, #2563eb, #60a5fa); transition:width 0.4s ease;"></div>
                <div style="width:{p2_prob_lr*100:.1f}%; background:#334155; transition:width 0.4s ease;"></div>
            </div>
        </div>

        {bm_html}

        <!-- 5 Key Delta Summary Tiles -->
        <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-top:16px; background:#0b0f19; padding:12px; border-radius:8px; border:1px solid #1e293b; text-align:center;">
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Rank Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['rank_diff'] <= 0 else '#ef4444'};">
                    {key_stats['rank_diff']:+.0f}
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Surface WR Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['surface_winrate_diff'] >= 0 else '#ef4444'};">
                    {key_stats['surface_winrate_diff']*100:+.1f}%
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Form Div Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:{'#10b981' if key_stats['form_divergence_diff'] >= 0 else '#ef4444'};">
                    {key_stats['form_divergence_diff']:+.2f}
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">Fatigue 14d Δ</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:#e2e8f0;">
                    {key_stats['matches_14d_diff']:+.0f} m
                </div>
            </div>
            <div>
                <div style="color:#64748b; font-size:11px; text-transform:uppercase; font-family:'JetBrains Mono', monospace; font-weight:600;">H2H Record</div>
                <div style="font-weight:800; font-size:15px; margin-top:2px; font-family:'JetBrains Mono', monospace; color:#38bdf8;">
                    {key_stats['h2h_p1_wins']} - {key_stats['h2h_p2_wins']}
                </div>
            </div>
        </div>
    </div>
    """
    return card_html
