"""Thread-safe structured prediction service delegating directly to atp_engine."""

from src import atp_engine as eng


def predict(
    p1: str,
    p2: str,
    surface: str,
    series: str,
    odd_1: float | None = None,
    odd_2: float | None = None,
) -> dict:
    """Structured prediction dict (prob_lgb, prob_lr, treeshap, player_cards, ...) or {'error': str}."""
    return eng.predict_match_structured(p1, p2, surface, series, odd_1=odd_1, odd_2=odd_2)


def inspect(split: str, tournament: str, match_idx) -> dict:
    """Structured historical inspection dict or {'error': str}."""
    match_idx_str = "" if match_idx is None else str(match_idx)
    return eng.inspect_historical_structured(split, tournament, match_idx_str)


def get_daily_schedule() -> dict:
    """Return latest saved daily schedule predictions and sync status."""
    from pipeline import atp_data
    return atp_data.get_daily_predictions()


def sync_daily_schedule(
    skip_kaggle: bool = False,
    schedule_file: str | None = None,
    allow_espn: bool = True,
) -> dict:
    """Run daily Kaggle sync + published schedule prediction and return structured results."""
    from pipeline import atp_daily_sync
    return atp_daily_sync.run_daily_pipeline(
        skip_kaggle=skip_kaggle,
        schedule_file=schedule_file,
        allow_espn=allow_espn,
    )
