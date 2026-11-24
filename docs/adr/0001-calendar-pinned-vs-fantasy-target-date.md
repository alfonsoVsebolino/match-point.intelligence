---
status: accepted
---

# Keep target_date optional to separate Calendar-Pinned and Fantasy Matchups

We keep `target_date` optional (`None` by default) on `POST /api/v1/predict` rather than defaulting omitted dates to today's UTC date, while propagating any explicit `target_date` through both feature construction and Arcade player card freshness. Defaulting omitted requests to today would introduce severe retirement-gap bias into `champion_lr` (`days_since_last_diff`) and max out Arcade `rust_penalty` whenever users run cross-era fantasy matchups in the All-Time Legends pool, whereas keeping `None` era-neutral preserves fantasy parity and pinning `fixture.date` from the Daily Schedule guarantees Schedule-to-Arena parity and honest inactivity counters when a calendar date is supplied.

## Considered Options

- **Default `target_date` to today's UTC date for all prediction requests**: Guarantees zero stale freshness for active players on manual lookups, but penalizes retired legends with multi-year rest gaps in cross-era fantasy matchups.
- **Keep `target_date` optional (`None` = Fantasy / Era-Neutral, explicit `YYYY-MM-DD` = Calendar-Pinned)**: Preserves fair cross-era fantasy comparisons by default while guaranteeing exact probability and freshness parity for Daily Schedule fixtures and date-pinned Arena evaluations.

## Consequences

- Loading a fixture from the Daily Schedule into Matchup Arena must explicitly pass `fixture.date` as `target_date` and expose a clearable date pin in the Arena UI.
- Both `build_match_features` and `_build_player_card` must apply the same `target_date` fatigue decay (`matches_14d` / `matches_30d` set to `0` once `days_since > 14` / `> 30`) so model inputs and visual scouting traits never disagree.
- Manual Arena matchups for active players without a pinned date continue to use each player's latest synced snapshot rather than today's wall-clock date unless the user pins a date.
