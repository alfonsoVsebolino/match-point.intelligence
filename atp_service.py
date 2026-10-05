"""Structured prediction service: reuses atp_engine, returns dicts instead of HTML."""

import re
import atp_engine as eng

def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()

def _capture(fn, *args):
    box, orig = {}, eng.render_diagnostic_card
    eng.render_diagnostic_card = lambda **kw: box.update(kw) or box
    try:
        r = fn(*args)
    finally:
        eng.render_diagnostic_card = orig
    return {"error": _strip_tags(r)} if isinstance(r, str) else box

def predict(p1: str, p2: str, surface: str, series: str) -> dict:
    """Card kwargs dict (prob_lgb, prob_lr, key_stats, ...) or {'error': str}."""
    return _capture(eng.predict_matchup, p1, p2, surface, series)

def inspect(split: str, tournament: str, match_idx) -> dict:
    """Historical card kwargs dict or {'error': str}."""
    return _capture(eng.inspect_historical_match, split, tournament, "" if match_idx is None else str(match_idx))
