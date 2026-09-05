"""Contact Forecast Phase 2: ordering and window construction for 2026.

Phase 1's `forecast.windows.order_eligible_events` refuses 2026 by design --
its `assert_ledger_schema` ends with `assert_forecast_seasons_allowed`, which
has no override parameter. That refusal is correct and stays exactly as
sealed. This module is the Phase 2 shim around it.

## How the frozen checks are reused without being weakened

`assert_ledger_schema` runs, in order: required columns, unique non-null
`event_id`, no nulls in the chronological sort key, finite run values, the
contact-stage identity `surprise == observed - deserved`, and THEN the Phase 1
season guard as its final statement. So a `ForecastSeasonError` from it proves
every other check already passed.

`assert_schema_for_phase2` therefore calls the frozen function, catches ONLY
`ForecastSeasonError`, and substitutes the Phase 2 season guard. Any other
failure propagates untouched. The dependency on the season guard being last is
not left implicit: `test_a_schema_violation_still_raises_before_the_season_guard`
pins it, so a future reordering upstream would fail the tests here rather than
silently skipping a check.

The ordering arithmetic itself (sort, then a 1-based `bbe_index` per hitter-
season) is the only logic duplicated from Phase 1, and
`test_phase2_ordering_matches_the_frozen_function` proves the duplicate is
byte-identical to `order_eligible_events` on a Phase-1-authorized season.

Everything downstream -- `build_windows`, `assert_windows_causal`,
`history_events`, `build_window_features`, `fit_shrinkage`,
`build_baseline_predictions` -- is the FROZEN Phase 1 code, called unmodified.
None of those carries a season guard.
"""

from __future__ import annotations

import pandas as pd

from forecast.forecast_config import CHRONOLOGICAL_SORT_KEY, ForecastSeasonError
from forecast.phase2.phase2_config import assert_phase2_seasons_allowed
from forecast.windows import ForecastWindowError, assert_ledger_schema


def assert_schema_for_phase2(ledger: pd.DataFrame) -> None:
    """Run every frozen ledger check, with the Phase 2 season guard substituted.

    Raises:
        ForecastWindowError: On any schema, uniqueness, nullity, finiteness or
            contact-stage-identity violation -- propagated unchanged from the
            frozen check.
        Phase2SeasonError: If the season is outside the Phase 2 authorization.
    """
    try:
        assert_ledger_schema(ledger)
    except ForecastSeasonError:
        # Expected for 2026: the Phase 1 guard refuses it, and it is the LAST
        # statement in the frozen check, so everything before it has passed.
        assert_phase2_seasons_allowed(sorted(int(s) for s in ledger["season"].unique()))
        return
    # Phase 1 accepted the season outright (a development season); still hold it
    # to the Phase 2 authorization.
    assert_phase2_seasons_allowed(sorted(int(s) for s in ledger["season"].unique()))


def order_events_for_phase2(ledger: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and assign a 1-based `bbe_index` per hitter-season.

    Identical arithmetic to `forecast.windows.order_eligible_events`, differing
    only in which season guard runs. Equivalence is proved by test.

    Raises:
        ForecastWindowError: If `game_date` will not parse, or if the sort key
            does not uniquely order a hitter-season.
    """
    assert_schema_for_phase2(ledger)

    ordered = ledger.copy()
    parsed = pd.to_datetime(ordered["game_date"], errors="coerce")
    if parsed.isna().any():
        raise ForecastWindowError(
            f"{int(parsed.isna().sum())} game_date value(s) could not be parsed as dates"
        )
    ordered["game_date"] = parsed

    group_key = ["batter", "season"]
    duplicated = ordered.duplicated(subset=group_key + list(CHRONOLOGICAL_SORT_KEY))
    if duplicated.any():
        raise ForecastWindowError(
            f"{int(duplicated.sum())} row(s) share a (batter, season, "
            f"{', '.join(CHRONOLOGICAL_SORT_KEY)}) key, so chronological order within "
            "those hitter-seasons is ambiguous"
        )

    ordered = ordered.sort_values(group_key + list(CHRONOLOGICAL_SORT_KEY)).reset_index(drop=True)
    ordered["bbe_index"] = ordered.groupby(group_key, sort=False).cumcount() + 1
    return ordered


def cutoff_boundary_events(ordered: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """The batted ball AT the cutoff for each hitter who reached it.

    Supplies the forecast ledger's cutoff timestamp and the identifier of the
    hitter's Nth resolved batted ball, so a prediction can be tied to the exact
    event it was made after.
    """
    at_cutoff = ordered[ordered["bbe_index"] == int(cutoff)]
    return pd.DataFrame(
        {
            "batter": at_cutoff["batter"].to_numpy(),
            "season": at_cutoff["season"].astype(int).to_numpy(),
            "cutoff_event_id": at_cutoff["event_id"].to_numpy(),
            "cutoff_game_date": at_cutoff["game_date"].to_numpy(),
            "cutoff_game_pk": at_cutoff["game_pk"].to_numpy(),
        }
    )


def horizon_completion(ordered: pd.DataFrame, *, cutoff: int, horizon: int) -> pd.DataFrame:
    """Per hitter: resolved-BBE count and whether the forward horizon completed.

    A hitter who reached the cutoff always gets a forecast. Whether the target
    exists is a separate question, answered here, so pending hitters can be
    carried rather than dropped.
    """
    counts = (
        ordered.groupby(["batter", "season"], sort=False)["bbe_index"]
        .max()
        .rename("n_resolved_bbe_season")
        .reset_index()
    )
    counts["reached_cutoff"] = counts["n_resolved_bbe_season"] >= int(cutoff)
    counts["completed_horizon"] = counts["n_resolved_bbe_season"] >= int(cutoff) + int(horizon)
    counts["resolved_bbe_since_cutoff"] = (counts["n_resolved_bbe_season"] - int(cutoff)).clip(
        lower=0
    )
    return counts
