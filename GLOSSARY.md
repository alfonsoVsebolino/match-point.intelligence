# MatchPoint.intelligence

ATP match prediction, live schedule telemetry, and historical audit arena combining dual-model probabilities, bookmaker de-vigging, and scouting cards.

## Language

**Calendar-Pinned Matchup**:
A prediction evaluated relative to an explicit match date so player rest and short-term workload reflect time elapsed up to that date.
_Avoid_: Live prediction, dated matchup

**Fantasy Matchup**:
A date-neutral prediction across eras or unscheduled pairings evaluated from each player's last recorded tour snapshot without calendar-gap penalties.
_Avoid_: Hypothetical match, static prediction

**Published Fixture**:
An officially scheduled, in-progress, or completed ATP tour match surfaced on the Daily Schedule Board with resolved player identities and optional bookmaker odds.
_Avoid_: Scheduled game, event row

**Snapshot Freeze**:
Locking a published fixture's pre-match model probabilities and features near match start so later schedule refreshes update status without overwriting pre-match forecasts.
_Avoid_: Locked match, frozen odds

**Schedule Refresh Status**:
The explicit outcome state of an upstream schedule refresh distinguishing fresh updates, active-scoreboard fallbacks, preserved snapshots, empty feeds, and upstream errors.
_Avoid_: Refresh boolean, sync flag

**Synced Player State**:
The up-to-date post-match roster telemetry produced by the daily dataset sync and bundled for deterministic container startup.
_Avoid_: Cached players, latest stats

**Historical Backtracker Split**:
A chronological partition of historical ATP matches (Holdout Test or Full Dataset) inspected to audit pre-match model outputs against actual outcomes.
_Avoid_: Dataset slice, CSV split

**Tournament Season**:
A calendar tour year used to scope historical tournaments and matches when auditing recurring annual events.
_Avoid_: Tour year, date range
