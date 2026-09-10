"""Pitching Contact Luck -- the pitcher-side re-aggregation of the SAME plays.

**This module trains nothing and predicts nothing.** Contact Luck is
`observed run value - expected run value` on a batted ball; that quantity is
a property of the contact, not of the batter, so the pitcher-side metric
needs no new model and no retraining. It re-groups the frozen play-level
attribution ledger by `pitcher` instead of `batter` and flips the sign.

## Why the sign flips

`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP` is stated from the
BATTER's perspective: a home run is positive, an out is negative. The same
batted ball has the opposite meaning for the pitcher who allowed it, so

    pitching_contact_luck = -1 x batting_contact_luck

for every signed quantity, applied to the total, each of the four additive
components, and every per-100 rate. Positive pitching Contact Luck therefore
means "outcomes more favorable to the PITCHER than the contact predicted",
which is the same sentence the batter-side metric makes about the batter.

The flip is applied ONCE, to a named list of signed columns
(`SIGNED_VALUE_COLUMNS`), and never inside a component computation -- so the
season accounting identity (`aggregate_attribution.verify_season_identity`)
survives it unchanged: if `a + b + c + d = T` then `-a + -b + -c + -d = -T`.

## What this module deliberately does NOT do

It does not redefine Contact Luck (`RESEARCH_RULES.md`, "Never silently
redefine the Luck Score"). The headline pitcher number is constructed
identically to the batter number -- same run values, same expected values,
same four-way decomposition, same ledger -- so the two remain directly
comparable. In particular the `defensive_execution_component` is KEPT in the
pitcher total: it is the defense playing behind that pitcher, which he does
not control, and excluding it would produce a different quantity that this
project has not defined or validated.
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season
from mlb_luck_score.scoring.aggregation_uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    METRIC_COLUMNS,
    bootstrap_batter_season_intervals,
)
from mlb_luck_score.scoring.qualification import (
    STATUS_QUALIFIED,
    QualificationThresholds,
    assign_qualification_status,
    summarize_qualification_counts,
)

#: Every signed, runs-denominated column produced by `aggregate_to_batter_
#: season` that must be negated to change perspective from batter to
#: pitcher. Counts (plays, games), fractions (coverage, provisional share)
#: and status strings are perspective-free and are NEVER negated.
#:
#: The four component totals are negated ALONGSIDE the total they sum to,
#: which is what keeps `aggregate_attribution.verify_season_identity` true
#: on the pitcher side. Negating the total alone silently inverts the
#: decomposition relative to its own headline number.
SIGNED_VALUE_COLUMNS: tuple[str, ...] = (
    "total_observed_minus_expected_runs",
    "total_contact_component_runs",
    "total_unexplained_residual_component_runs",
    "total_defensive_execution_component_runs",
    "total_advancement_component_runs",
    "total_observed_minus_expected_excluding_provisional_runs",
    "observed_minus_expected_per_100",
    "contact_component_per_100",
    "unexplained_residual_component_per_100",
    "defensive_execution_component_per_100",
    "advancement_execution_component_per_100",
    "observed_minus_expected_excluding_provisional_per_100",
)


#: Pitcher-specific qualification threshold sets. **Predetermined and fixed
#: BEFORE any pitcher Contact Luck value was computed or examined** -- the
#: derivation below uses only EXPOSURE (how many eligible batted balls and
#: games a pitcher accumulated), never a luck value, a rank, or a
#: leaderboard shape, per `RESEARCH_RULES.md`'s rule against tuning a
#: threshold to what a result looks like.
#:
#: ## Why the batter set cannot be reused
#:
#: `qualification.QUALIFICATION_THRESHOLD_SETS["primary"]` requires >=100
#: games, mirroring a batting-title convention. Measured on real 2021-2024
#: development data: **0 of 3,494 pitcher-seasons reach 100 games** (max 80,
#: median 19). Reused unchanged it would disqualify every pitcher who has
#: ever thrown a pitch, including a 674-BBE workhorse starter.
#:
#: ## How `pitcher_primary`'s 450 was derived
#:
#: The batter set "loosely mirrors MLB's own batting-title convention"; the
#: pitcher analogue is MLB's ERA-title rule (1 inning pitched per team game,
#: 162 IP), which qualifies roughly 40-60 pitchers per season. Innings
#: pitched CANNOT be reconstructed from this project's batted-ball-only
#: table (strikeouts and walks are not rows here), so the bar is set on the
#: project's own unit -- eligible batted balls -- at the value whose
#: qualifying COUNT matches what MLB's rule admits. Measured qualifiers per
#: season, 2021/2022/2023/2024:
#:
#:     400 BBE -> 70, 82, 76, 85  (mean 78.2)  -- looser than the ERA title
#:     450 BBE -> 41, 51, 57, 62  (mean 52.8)  -- matches it
#:     500 BBE -> 25, 34, 31, 32  (mean 30.5)  -- stricter than it
#:
#: ## Relievers are excluded by exposure, not by choice
#:
#: No reliever reaches a starter-scale bar: across 2021-2024 the
#: highest-volume pitcher-season with >=50 appearances faced **311** eligible
#: batted balls, and 309 of the 313 pitcher-seasons at >=400 BBE pitched in
#: <=35 games. `pitcher_primary` therefore describes STARTING PITCHERS. That
#: is a documented property to state in any output built on it, never a
#: silent filter -- a reliever-scale threshold set is deliberately deferred
#: rather than guessed at here, because a per-100 rate over ~160 batted balls
#: is a materially different precision claim.
#:
#: `min_games` is a low guard (15), well under the 27-game 10th percentile of
#: the >=450 BBE population, so it never binds for the intended population --
#: it exists only to reject a pathological one-appearance row that somehow
#: cleared the volume bar, mirroring the batter set's two-part volume test.
#:
#: The component-quality thresholds are inherited UNCHANGED from the batter
#: set: they describe how well the four component models covered a set of
#: plays, which is a property of the plays, not of who threw them.
PITCHER_QUALIFICATION_THRESHOLD_SETS: dict[str, QualificationThresholds] = {
    "pitcher_primary": QualificationThresholds(
        label="pitcher_primary",
        min_eligible_batted_balls=450,
        min_games=15,
        min_component_coverage=0.50,
        max_provisional_value_share=0.60,
        max_missing_input_frequency=0.30,
    ),
    "pitcher_inclusive": QualificationThresholds(
        label="pitcher_inclusive",
        min_eligible_batted_balls=300,
        min_games=10,
        min_component_coverage=0.50,
        max_provisional_value_share=0.60,
        max_missing_input_frequency=0.30,
    ),
}

#: The threshold set every pitcher-side report defaults to.
DEFAULT_PITCHER_THRESHOLD_SET = "pitcher_primary"


def aggregate_to_pitcher_season(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    pitcher_column: str = "pitcher",
    season_column: str = "season",
    game_column: str = "game_pk",
) -> pd.DataFrame:
    """Aggregate the play-level attribution ledger to one row per (pitcher, season).

    Args:
        df: Prepared DataFrame with `event_id`, `pitcher_column`,
            `season_column`, `game_column` -- same index/order as `ledger`.
        ledger: `mlb_luck_score.scoring.attribution_ledger.
            build_attribution_ledger`'s output.
        confidence: `mlb_luck_score.scoring.component_confidence.
            build_play_level_confidence`'s output (long format).

    Returns:
        One row per (pitcher, season), with the same columns
        `aggregate_to_batter_season` produces, except that every column in
        `SIGNED_VALUE_COLUMNS` carries the pitcher's sign convention.
    """
    summary = aggregate_to_batter_season(
        df,
        ledger,
        confidence,
        batter_column=pitcher_column,
        season_column=season_column,
        game_column=game_column,
    )
    for column in SIGNED_VALUE_COLUMNS:
        summary[column] = -summary[column]
    return summary


def _interval_column_stems() -> tuple[str, ...]:
    """Every `<stem>_ci_low`/`<stem>_ci_high` pair the bootstrap produces."""
    return tuple(stem for metric in METRIC_COLUMNS for stem in (metric, f"{metric}_per_100"))


def bootstrap_pitcher_season_intervals(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    pitcher_column: str = "pitcher",
    season_column: str = "season",
    game_column: str = "game_pk",
    **bootstrap_kwargs: object,
) -> pd.DataFrame:
    """Game-clustered bootstrap intervals for every (pitcher, season).

    Reuses `bootstrap_batter_season_intervals` unchanged -- resampling is
    over `game_pk` within the grouping key, and which key that is does not
    change the procedure -- then converts each interval to the pitcher's
    sign convention.

    **Negation reverses an interval**: `[lo, hi]` becomes `[-hi, -lo]`, not
    `[-lo, -hi]`. Negating the two endpoints in place would leave every
    interval with its low bound above its high bound, which no accounting
    identity elsewhere would catch.

    Returns:
        `bootstrap_batter_season_intervals`' columns, keyed by
        `pitcher_column`, with every interval on the pitcher's sign
        convention and its bounds correctly ordered.
    """
    intervals = bootstrap_batter_season_intervals(
        df,
        ledger,
        batter_column=pitcher_column,
        season_column=season_column,
        game_column=game_column,
        **bootstrap_kwargs,  # type: ignore[arg-type]
    )
    for stem in _interval_column_stems():
        low_col, high_col = f"{stem}_ci_low", f"{stem}_ci_high"
        if low_col not in intervals.columns or high_col not in intervals.columns:
            continue
        # Read both endpoints BEFORE writing either, so the swap cannot
        # clobber the value it still needs.
        low, high = intervals[low_col].copy(), intervals[high_col].copy()
        intervals[low_col] = -high
        intervals[high_col] = -low
    return intervals


def build_pitcher_season_table(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    pitcher_column: str = "pitcher",
    season_column: str = "season",
    game_column: str = "game_pk",
    n_bootstrap_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    threshold_set: QualificationThresholds | str = DEFAULT_PITCHER_THRESHOLD_SET,
) -> pd.DataFrame:
    """One row per (pitcher, season): totals, intervals, qualification status.

    Mirrors the last steps of `mlb_luck_score.scoring.run_season_aggregation.
    build_player_season_report` -- aggregate, bootstrap, merge, qualify --
    on the pitcher's sign convention and threshold set. It trains nothing:
    pass the `scoring_df`/`ledger`/`confidence` that runner already exposes
    on its `SeasonAggregationArtifacts`, so the four component models are
    fit exactly once no matter how many ways the same plays are grouped.

    The table is NEVER filtered by qualification status -- same contract as
    the batter table, so a non-qualifying pitcher keeps his own row.

    Returns:
        The aggregated table. `.attrs["threshold_set"]` records which
        threshold set produced `qualification_status`, because the label is
        not recoverable from the column itself.
    """
    resolved = (
        PITCHER_QUALIFICATION_THRESHOLD_SETS[threshold_set]
        if isinstance(threshold_set, str)
        else threshold_set
    )

    summary = aggregate_to_pitcher_season(
        df,
        ledger,
        confidence,
        pitcher_column=pitcher_column,
        season_column=season_column,
        game_column=game_column,
    )
    intervals = bootstrap_pitcher_season_intervals(
        df,
        ledger,
        pitcher_column=pitcher_column,
        season_column=season_column,
        game_column=game_column,
        n_reps=n_bootstrap_reps,
        seed=bootstrap_seed,
    )
    table = summary.merge(intervals, on=[pitcher_column, season_column], how="left")
    table["qualification_status"] = assign_qualification_status(table, thresholds=resolved)
    table.attrs["threshold_set"] = resolved.label
    return table


#: Totals compared by `verify_batter_side_reproduction`. Rates are excluded
#: because they are exact functions of these and the (perspective-free)
#: eligible-play count, so they add no independent evidence.
REPRODUCTION_CHECK_COLUMNS: tuple[str, ...] = (
    "total_observed_minus_expected_runs",
    "total_contact_component_runs",
    "total_unexplained_residual_component_runs",
    "total_defensive_execution_component_runs",
    "total_advancement_component_runs",
)


def verify_batter_side_reproduction(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    batter_column: str = "batter",
    season_column: str = "season",
    game_column: str = "game_pk",
    atol: float = 1e-9,
) -> dict[str, object]:
    """Prove the pitcher path adds no arithmetic of its own.

    Runs `aggregate_to_pitcher_season` keyed on `batter_column` -- the SAME
    key the frozen batter pipeline uses -- undoes the sign flip, and
    compares every total in `REPRODUCTION_CHECK_COLUMNS` against
    `aggregate_to_batter_season`'s own output on identical inputs.

    This is a self-check to run and RECORD alongside any real pitcher
    output, not a substitute for the test suite: if it ever reports
    `reproduces: False`, the pitcher numbers from that run are not
    trustworthy and the difference must be explained before they are used.

    Returns:
        `reproduces` (all totals agree within `atol`), `n_rows` compared,
        `max_abs_difference`, and `worst_column`.
    """
    batter_side = aggregate_to_batter_season(
        df,
        ledger,
        confidence,
        batter_column=batter_column,
        season_column=season_column,
        game_column=game_column,
    )
    via_pitcher_path = aggregate_to_pitcher_season(
        df,
        ledger,
        confidence,
        pitcher_column=batter_column,
        season_column=season_column,
        game_column=game_column,
    )
    if not batter_side[[batter_column, season_column]].equals(
        via_pitcher_path[[batter_column, season_column]]
    ):
        return {
            "reproduces": False,
            "n_rows": int(len(batter_side)),
            "max_abs_difference": float("nan"),
            "worst_column": "row keys differ between the two paths",
        }

    worst_column, max_difference = "", 0.0
    for column in REPRODUCTION_CHECK_COLUMNS:
        # Undo the sign flip: the pitcher path negated these on the way out.
        difference = float(
            (batter_side[column] - (-via_pitcher_path[column])).abs().max(skipna=True)
        )
        if difference > max_difference:
            worst_column, max_difference = column, difference

    return {
        "reproduces": bool(max_difference <= atol),
        "n_rows": int(len(batter_side)),
        "max_abs_difference": max_difference,
        "worst_column": worst_column,
    }


#: Recorded verbatim in every pitcher report. `pitcher_primary` describes
#: STARTING pitchers -- see `PITCHER_QUALIFICATION_THRESHOLD_SETS` for the
#: exposure measurements behind that.
RELIEVER_SCOPE_NOTE = (
    "pitcher_primary is a starter-scale threshold set. Across 2021-2024 no "
    "pitcher-season with >=50 appearances faced more than 311 eligible batted "
    "balls, and 309 of the 313 pitcher-seasons at >=400 came in <=35 games, so "
    "relievers do not reach it. Relievers are excluded by exposure, not by "
    "choice; a reliever-scale threshold set is deferred, not assumed."
)


#: The metric a leaderboard is ordered by, and the one the qualified
#: -population reference point is computed on.
HEADLINE_RATE_COLUMN = "observed_minus_expected_per_100"


def qualified_population_reference(
    table: pd.DataFrame,
    *,
    rate_column: str = HEADLINE_RATE_COLUMN,
    status_column: str = "qualification_status",
) -> dict[str, object]:
    """Where the CENTRE of a qualified board actually sits, in runs/100.

    Zero is the point at which a batted ball matched its expectation, NOT
    the point at which a qualified pitcher matched his peers. Those differ,
    and the gap is a selection effect rather than a modelling defect:
    clearing a starter-scale exposure bar requires having kept a rotation
    spot all season, and favorable realized outcomes are part of why a
    pitcher keeps one. Measured on real 2024 data, the same conditioning
    moves the mean from -0.520 (all 854 pitcher-seasons) to +0.544 (the 61
    qualified), and it moves the BATTER mean the same direction on the same
    plays (-1.033 -> -0.138), so this is a property of qualification, not
    of the pitcher side.

    Reported so a reader is never left inferring that zero is the neutral
    point of the board in front of them.

    **The score itself is never recentred.** `score_is_recentered` is
    always False: subtracting this reference would redefine Contact Luck
    (`RESEARCH_RULES.md`, "Never silently redefine the Luck Score"), break
    comparability with every batter number, and turn a descriptive fact
    into a different metric. This is context recorded ALONGSIDE the score,
    exactly as the confidence report is descriptive and never dampens it.

    Returns:
        `n_qualified`, `mean_runs_per_100`, `median_runs_per_100`,
        `sd_runs_per_100`, and `score_is_recentered` (always False). The
        three statistics are None when no row qualifies.
    """
    qualified = table[table[status_column] == STATUS_QUALIFIED]
    values = (
        qualified[rate_column].dropna()
        if rate_column in qualified.columns
        else pd.Series(dtype=float)
    )
    have = len(values) > 0
    return {
        "n_qualified": int(len(qualified)),
        "mean_runs_per_100": float(values.mean()) if have else None,
        "median_runs_per_100": float(values.median()) if have else None,
        "sd_runs_per_100": float(values.std()) if len(values) > 1 else None,
        "computed_on": "qualified rows only",
        "score_is_recentered": False,
        "interpretation": (
            "Zero means the batted balls matched expectation; it does NOT mean "
            "average among qualified pitchers. Compare a row against "
            "mean_runs_per_100 to read it relative to the board it appears on."
        ),
    }


def build_pitcher_report(
    table: pd.DataFrame,
    reproduction: dict[str, object],
    *,
    threshold_set_label: str,
    seasons: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Assemble the machine-readable summary for a pitcher-side run.

    `pitcher_values_trustworthy` is the field a reader keys on, and it is
    gated on `verify_batter_side_reproduction`: if the pitcher path stopped
    reproducing the frozen batter path, the numbers from that run are not
    trustworthy no matter how reasonable the leaderboard looks. A failed
    self-check invalidates the run rather than being recorded as a footnote
    under an otherwise healthy-looking summary.
    """
    counts = summarize_qualification_counts(table["qualification_status"])
    return {
        "metric": "pitching_contact_luck",
        "construction": (
            "Identical to batting Contact Luck (same run values, same expected "
            "values, same four-way decomposition, same ledger), re-grouped by "
            "pitcher and negated. No model is trained, refit or recalibrated."
        ),
        "sign_convention": (
            "pitching_contact_luck = -1 x batting_contact_luck. Positive means "
            "outcomes more favorable to the PITCHER than the contact predicted."
        ),
        "seasons": list(seasons) if seasons is not None else None,
        "qualification_threshold_set": threshold_set_label,
        "qualification_counts": counts,
        "pitcher_season_row_count": int(len(table)),
        "reliever_scope": RELIEVER_SCOPE_NOTE,
        "qualified_population_reference": qualified_population_reference(table),
        "batter_side_reproduction": reproduction,
        "pitcher_values_trustworthy": bool(reproduction.get("reproduces", False)),
    }
