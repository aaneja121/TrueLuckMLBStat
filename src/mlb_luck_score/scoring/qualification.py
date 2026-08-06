"""Contact Luck v0.11 Phase 5: qualification rules.

Defines qualification status BEFORE any player is ranked, using PREDETERMINED
thresholds chosen for external defensibility (round numbers, loosely modeled
on MLB's own "games played"-style qualification conventions, adapted to this
project's "eligible batted balls" unit) -- NOT by searching for values that
happen to produce an appealing 2024 leaderboard. This module was written and
its thresholds fixed BEFORE `mlb_luck_score.scoring.run_season_aggregation`
was ever run against real data (see that module's docstring for the exact
order of operations).

`QUALIFICATION_THRESHOLD_SETS` provides three PREDETERMINED, documented
candidate sets (`"primary"`, `"strict"`, `"lenient"`) rather than one -- per
the task's explicit alternative to a single hand-picked threshold, `mlb_luck_
score.models.evaluate_aggregation_stability`'s Phase 6 sensitivity analysis
reports how player rankings shift across all three, so a reader can judge
threshold-sensitivity directly rather than trusting one arbitrary cutoff.

## The five statuses, checked in this fixed precedence order

  1. `NOT_REPORTABLE`: zero eligible batted balls -- there is nothing to
     compute a rate from.
  2. `SMALL_SAMPLE`: below the volume bar (`min_eligible_batted_balls`/
     `min_games`) -- not enough plays to say anything with confidence,
     regardless of how clean the available plays are.
  3. `INSUFFICIENT_COMPONENT_COVERAGE`: clears the volume bar, but too much
     of the player's profile falls outside BOTH the defense and advancement
     models entirely (`component_coverage_fraction` below `min_component_
     coverage`) -- a genuinely different problem from "too few plays."
  4. `PROVISIONALLY_QUALIFIED`: clears volume AND coverage, but too much of
     the covered value comes from provisional/limited-evidence pathways
     (`share_of_value_from_provisional_components` above `max_provisional_
     value_share`) or too many plays needed missing-input fallback
     (`missing_input_frequency` above `max_missing_input_frequency`).
  5. `QUALIFIED`: clears every threshold -- eligible for the MAIN rankings.

Per the task's explicit instruction, players below `QUALIFIED` still get
their computed values reported (`mlb_luck_score.scoring.
run_season_aggregation`'s player-season output never drops a row) -- this
module only assigns the STATUS LABEL that determines whether a row appears
in the main qualified rankings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STATUS_QUALIFIED = "qualified"
STATUS_PROVISIONALLY_QUALIFIED = "provisionally_qualified"
STATUS_SMALL_SAMPLE = "small_sample"
STATUS_INSUFFICIENT_COMPONENT_COVERAGE = "insufficient_component_coverage"
STATUS_NOT_REPORTABLE = "not_reportable"
QUALIFICATION_STATUS_VALUES: tuple[str, ...] = (
    STATUS_QUALIFIED,
    STATUS_PROVISIONALLY_QUALIFIED,
    STATUS_SMALL_SAMPLE,
    STATUS_INSUFFICIENT_COMPONENT_COVERAGE,
    STATUS_NOT_REPORTABLE,
)


@dataclass(frozen=True)
class QualificationThresholds:
    """One documented, predetermined candidate threshold set.

    `min_eligible_batted_balls`/`min_games` loosely mirror MLB's own batting
    -title qualification convention (3.1 PA per team game over a 162-game
    season, roughly 500 PA) -- SCALED DOWN because this project's unit is
    "eligible batted balls" (fair contact-luck-eligible balls in play), a
    strict SUBSET of plate appearances (excludes walks/strikeouts/HBP/sac
    bunts/etc.), not a like-for-like replacement for it. `min_games` uses
    the commonly-referenced "played (at least) X% of the season" framing
    rather than re-deriving it from any 2024-specific distribution.
    """

    label: str
    min_eligible_batted_balls: int
    min_games: int
    min_component_coverage: float
    max_provisional_value_share: float
    max_missing_input_frequency: float


QUALIFICATION_THRESHOLD_SETS: dict[str, QualificationThresholds] = {
    "primary": QualificationThresholds(
        label="primary",
        min_eligible_batted_balls=200,
        min_games=100,
        min_component_coverage=0.50,
        max_provisional_value_share=0.60,
        max_missing_input_frequency=0.30,
    ),
    "strict": QualificationThresholds(
        label="strict",
        min_eligible_batted_balls=300,
        min_games=120,
        min_component_coverage=0.70,
        max_provisional_value_share=0.40,
        max_missing_input_frequency=0.15,
    ),
    "lenient": QualificationThresholds(
        label="lenient",
        min_eligible_batted_balls=100,
        min_games=50,
        min_component_coverage=0.30,
        max_provisional_value_share=0.80,
        max_missing_input_frequency=0.50,
    ),
}

DEFAULT_THRESHOLD_SET = "primary"

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "eligible_batted_balls",
    "games",
    "component_coverage_fraction",
    "share_of_value_from_provisional_components",
    "missing_input_frequency",
)


class QualificationError(ValueError):
    """Raised when inputs to the qualification layer are invalid."""


def assign_qualification_status(
    summary: pd.DataFrame,
    *,
    thresholds: QualificationThresholds | str = DEFAULT_THRESHOLD_SET,
) -> pd.Series:
    """Assign one of `QUALIFICATION_STATUS_VALUES` per row of `summary`
    (`mlb_luck_score.scoring.aggregate_attribution.aggregate_to_batter_
    season`'s output), using the fixed precedence order documented above.

    Args:
        summary: Must have `REQUIRED_SUMMARY_COLUMNS`.
        thresholds: A `QualificationThresholds` instance, or one of
            `QUALIFICATION_THRESHOLD_SETS`' keys.

    Returns:
        A `str`-valued Series, index-aligned with `summary`.
    """
    if isinstance(thresholds, str):
        if thresholds not in QUALIFICATION_THRESHOLD_SETS:
            raise QualificationError(
                f"Unknown threshold set {thresholds!r}; choose one of "
                f"{list(QUALIFICATION_THRESHOLD_SETS)} or pass a QualificationThresholds instance"
            )
        thresholds = QUALIFICATION_THRESHOLD_SETS[thresholds]

    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in summary.columns]
    if missing:
        raise QualificationError(f"summary is missing required column(s): {missing}")

    eligible = summary["eligible_batted_balls"].fillna(0)
    games = summary["games"].fillna(0)
    coverage = summary["component_coverage_fraction"]
    provisional_share = summary["share_of_value_from_provisional_components"]
    missing_input_freq = summary["missing_input_frequency"]

    not_reportable = eligible <= 0
    meets_volume = (eligible >= thresholds.min_eligible_batted_balls) & (
        games >= thresholds.min_games
    )
    meets_coverage = coverage.fillna(0.0) >= thresholds.min_component_coverage
    meets_provisional_share = (
        provisional_share.fillna(0.0) <= thresholds.max_provisional_value_share
    )
    meets_missing_input = missing_input_freq.fillna(1.0) <= thresholds.max_missing_input_frequency

    status = pd.Series(STATUS_QUALIFIED, index=summary.index, dtype=object)
    status = status.where(
        meets_missing_input & meets_provisional_share, STATUS_PROVISIONALLY_QUALIFIED
    )
    status = status.where(meets_coverage, STATUS_INSUFFICIENT_COMPONENT_COVERAGE)
    status = status.where(meets_volume, STATUS_SMALL_SAMPLE)
    status = status.where(~not_reportable, STATUS_NOT_REPORTABLE)
    return status


def summarize_qualification_counts(status: pd.Series) -> dict[str, int]:
    """Row counts by qualification status -- always includes every value in
    `QUALIFICATION_STATUS_VALUES`, even ones with zero rows (so a report
    never silently omits an empty category).
    """
    counts = status.value_counts().to_dict()
    return {value: int(counts.get(value, 0)) for value in QUALIFICATION_STATUS_VALUES}


def compare_threshold_sets(
    summary: pd.DataFrame,
    *,
    threshold_set_names: tuple[str, ...] = ("primary", "strict", "lenient"),
) -> pd.DataFrame:
    """Assign qualification status under EVERY named threshold set side by
    side -- the sensitivity-analysis input `mlb_luck_score.models.
    evaluate_aggregation_stability` reports on, rather than committing to
    one hand-picked cutoff without showing how much it matters.
    """
    out = (
        summary[["batter", "season"]].copy()
        if "batter" in summary.columns
        else pd.DataFrame(index=summary.index)
    )
    for name in threshold_set_names:
        out[f"status_{name}"] = assign_qualification_status(summary, thresholds=name).to_numpy()
    return out


def qualified_mask(status: pd.Series) -> pd.Series:
    """`True` only for `STATUS_QUALIFIED` -- the main rankings' membership test."""
    return status == STATUS_QUALIFIED
