"""FanGraphs Guts-based fixed run-value table for Version 0.2 Contact Luck.

Source: FanGraphs' "Guts!" constants page
(https://www.fangraphs.com/guts.aspx?type=cn), which publishes each season's
wOBA scale, league wOBA, and wOBA event weights derived from that season's
actual run-scoring environment (the same constants underlying wOBA/wRC+).
This module recovers each outcome's runs-above-average value via the
standard linear-weights identity:

    run_value(event) = (wOBA_event_weight - league_wOBA) / wOBA_scale

An out has `wOBA_event_weight = 0` by definition (it contributes no positive
batting value in the wOBA formula).

This is DIFFERENT from -- and more empirically grounded than -- the Version
0.1 ordinal placeholder (`mlb_luck_score.scoring.raw_luck`, now legacy:
out=0/single=1/double=2/triple=3/home_run=4). It reflects each outcome's
actual average run value relative to a league-average plate appearance, in
real runs, for each of four real seasons.

`DEFAULT_RUN_VALUE_MAP` fixes ONE table for Version 0.2 by averaging the
four season-specific tables (2021-2024) with equal weight -- a
context-neutral, season-agnostic reference, not a season-specific or
situation-specific (base/out state, park, weather) run-value model. See
`mlb_luck_score.scoring.contact_luck` module docstring for why context is
deliberately excluded from Version 0.2.

The constants in `SEASON_WOBA_CONSTANTS` are published values transcribed by
hand from FanGraphs' Guts! page; they are not fetched live and must be
updated manually if FanGraphs revises them or when new seasons are added.
"""

from __future__ import annotations

from typing import NamedTuple

from mlb_luck_score.config import CLASS_ORDER


class SeasonWobaConstants(NamedTuple):
    """One season's published FanGraphs Guts wOBA constants."""

    league_woba: float
    woba_scale: float
    single: float
    double: float
    triple: float
    home_run: float


#: Published FanGraphs Guts constants for 2021-2024. See module docstring
#: for source and caveats (hand-transcribed, not fetched live).
SEASON_WOBA_CONSTANTS: dict[int, SeasonWobaConstants] = {
    2021: SeasonWobaConstants(
        league_woba=0.314,
        woba_scale=1.209,
        single=0.879,
        double=1.242,
        triple=1.568,
        home_run=2.007,
    ),
    2022: SeasonWobaConstants(
        league_woba=0.310,
        woba_scale=1.259,
        single=0.884,
        double=1.261,
        triple=1.601,
        home_run=2.072,
    ),
    2023: SeasonWobaConstants(
        league_woba=0.318,
        woba_scale=1.204,
        single=0.883,
        double=1.244,
        triple=1.569,
        home_run=2.004,
    ),
    2024: SeasonWobaConstants(
        league_woba=0.310,
        woba_scale=1.242,
        single=0.882,
        double=1.254,
        triple=1.590,
        home_run=2.050,
    ),
}

#: Provenance/methodology metadata -- attach this (or a copy) to any
#: artifact/report derived from `DEFAULT_RUN_VALUE_MAP` so the source is
#: never separated from the numbers.
SOURCE_METADATA: dict[str, str] = {
    "source": "FanGraphs Guts! constants table (https://www.fangraphs.com/guts.aspx?type=cn)",
    "methodology": (
        "Linear weights: run_value(event) = (wOBA_event_weight - league_wOBA) / wOBA_scale; "
        "an out has wOBA_event_weight = 0."
    ),
    "seasons_used": "2021, 2022, 2023, 2024 (equal-weighted arithmetic mean)",
    "caveat": (
        "Hand-transcribed published constants, not fetched live. Context-neutral: ignores "
        "base/out state, park, weather, and defense -- see "
        "mlb_luck_score.scoring.contact_luck module docstring."
    ),
}


def compute_season_run_values(season: int) -> dict[str, float]:
    """Compute one season's outcome -> run-value table from its Guts constants.

    Args:
        season: A year present in `SEASON_WOBA_CONSTANTS`.

    Returns:
        A dict keyed by outcome class (matching `mlb_luck_score.config.
        CLASS_ORDER`) mapping to that season's run value, in runs.

    Raises:
        ValueError: If `season` has no configured constants.
    """
    if season not in SEASON_WOBA_CONSTANTS:
        raise ValueError(
            f"No FanGraphs Guts constants configured for season {season}. "
            f"Configured seasons: {sorted(SEASON_WOBA_CONSTANTS)}"
        )
    constants = SEASON_WOBA_CONSTANTS[season]
    woba_event_weight = {
        "out": 0.0,
        "single": constants.single,
        "double": constants.double,
        "triple": constants.triple,
        "home_run": constants.home_run,
    }
    return {
        outcome: (weight - constants.league_woba) / constants.woba_scale
        for outcome, weight in woba_event_weight.items()
    }


def compute_default_run_value_map(
    seasons: tuple[int, ...] = tuple(sorted(SEASON_WOBA_CONSTANTS)),
) -> dict[str, float]:
    """Equal-weighted arithmetic mean of `seasons`' run-value tables.

    Args:
        seasons: Which seasons' tables to average. Defaults to every season
            configured in `SEASON_WOBA_CONSTANTS` (2021-2024).

    Returns:
        A dict keyed by `mlb_luck_score.config.CLASS_ORDER` outcome classes.
    """
    per_season_tables = [compute_season_run_values(season) for season in seasons]
    return {
        outcome: sum(table[outcome] for table in per_season_tables) / len(per_season_tables)
        for outcome in CLASS_ORDER
    }


#: The fixed Version 0.2 context-neutral run-value reference table -- the
#: arithmetic mean of the 2021-2024 season-specific tables above. Computed
#: PROGRAMMATICALLY from `SEASON_WOBA_CONSTANTS` (not hardcoded), so it
#: always stays consistent with the published constants above. Approximate
#: expected values (verified in `tests/test_run_values.py`): out=-0.254916,
#: single=0.463266, double=0.763026, triple=1.033068, home_run=1.400288.
DEFAULT_RUN_VALUE_MAP: dict[str, float] = compute_default_run_value_map()
