"""Pitching Contact Luck: the pitcher-side re-aggregation of the SAME plays.

Contact Luck is a property of the batted ball (`observed run value -
expected run value`), not of the batter -- nothing in the per-play quantity
is batter-specific. The pitcher-side metric therefore introduces no new
model: it re-groups the frozen play-level ledger by `pitcher` and flips the
sign, because a batted ball that becomes an out is favorable to the pitcher
and unfavorable to the batter.

These tests pin the two things that re-grouping alone does NOT get right:
the sign convention, and what happens to a bootstrap interval when you
negate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season
from mlb_luck_score.scoring.aggregation_uncertainty import bootstrap_batter_season_intervals
from mlb_luck_score.scoring.pitching_contact_luck import (
    PITCHER_QUALIFICATION_THRESHOLD_SETS,
    aggregate_to_pitcher_season,
    bootstrap_pitcher_season_intervals,
    build_pitcher_report,
    build_pitcher_season_table,
    verify_batter_side_reproduction,
)


@pytest.fixture
def ledger_with_pitchers(v011_full_ledger):
    """The shared v0.11 ledger with a `pitcher` column set EQUAL to `batter`.

    Holding the grouping key identical is deliberate: it isolates the one
    thing the pitcher path adds (the sign convention) from the thing it
    reuses unchanged (the grouping), so a failure here can only be about
    sign.
    """
    df, ledger, confidence = v011_full_ledger
    df = df.copy()
    df["pitcher"] = df["batter"]
    return df, ledger, confidence


def test_pitcher_season_total_is_the_negation_of_the_batter_side_total(ledger_with_pitchers):
    df, ledger, confidence = ledger_with_pitchers

    batter_side = aggregate_to_batter_season(df, ledger, confidence)
    pitcher_side = aggregate_to_pitcher_season(df, ledger, confidence)

    merged = batter_side.merge(
        pitcher_side,
        left_on=["batter", "season"],
        right_on=["pitcher", "season"],
        suffixes=("_bat", "_pit"),
    )
    assert len(merged) == len(batter_side) > 0

    np.testing.assert_allclose(
        merged["total_observed_minus_expected_runs_pit"].to_numpy(),
        -merged["total_observed_minus_expected_runs_bat"].to_numpy(),
        atol=1e-12,
    )


def test_season_accounting_identity_holds_on_the_pitcher_side(ledger_with_pitchers):
    """The four additive components must still reconstruct the total.

    Negating the total without negating its parts would silently break the
    decomposition -- the components would sum to the wrong sign of the
    headline number, and every component chart built on it would be wrong.
    """
    df, ledger, confidence = ledger_with_pitchers

    pitcher_side = aggregate_to_pitcher_season(df, ledger, confidence)

    reconstructed = (
        pitcher_side["total_contact_component_runs"]
        + pitcher_side["total_unexplained_residual_component_runs"]
        + pitcher_side["total_defensive_execution_component_runs"]
        + pitcher_side["total_advancement_component_runs"]
    )
    np.testing.assert_allclose(
        pitcher_side["total_observed_minus_expected_runs"].to_numpy(),
        reconstructed.to_numpy(),
        atol=1e-9,
    )


def test_negating_an_interval_swaps_its_bounds_it_does_not_invert_them(ledger_with_pitchers):
    """`[lo, hi]` negated is `[-hi, -lo]`, never `[-lo, -hi]`.

    Negating both endpoints in place would leave every interval with its
    low bound above its high bound -- an interval that reads backwards on
    every player page, and one no identity check elsewhere would catch.
    """
    df, ledger, confidence = ledger_with_pitchers

    batter_ci = bootstrap_batter_season_intervals(df, ledger, n_reps=40, seed=7)
    pitcher_ci = bootstrap_pitcher_season_intervals(df, ledger, n_reps=40, seed=7)

    merged = batter_ci.merge(
        pitcher_ci,
        left_on=["batter", "season"],
        right_on=["pitcher", "season"],
        suffixes=("_bat", "_pit"),
    )
    assert len(merged) == len(batter_ci) > 0

    low = merged["observed_minus_expected_ci_low_pit"].to_numpy()
    high = merged["observed_minus_expected_ci_high_pit"].to_numpy()
    finite = np.isfinite(low) & np.isfinite(high)
    assert finite.any()

    # The pitcher's low bound is the negation of the batter's HIGH bound.
    np.testing.assert_allclose(
        low[finite],
        -merged["observed_minus_expected_ci_high_bat"].to_numpy()[finite],
        atol=1e-12,
    )
    # And an interval never reads backwards.
    assert (low[finite] <= high[finite]).all()


def _summary_row(*, bbe: int, games: int) -> dict[str, float | int]:
    """A qualification-summary row with clean component quality.

    Coverage/provisional/missing-input are set to values that pass every
    threshold, so these tests isolate the VOLUME bars -- the only part of
    qualification that had to be re-derived for pitchers.
    """
    return {
        "eligible_batted_balls": bbe,
        "games": games,
        "component_coverage_fraction": 0.95,
        "share_of_value_from_provisional_components": 0.10,
        "missing_input_frequency": 0.05,
    }


def test_batter_games_bar_would_disqualify_every_real_pitcher():
    """The batter set requires 100 games; no pitcher-season reaches it.

    Measured on real 2021-2024 development data: 0 of 3,494 pitcher-seasons
    reach 100 games (max 80, median 19). This test pins the REASON a
    pitcher-specific threshold set has to exist at all.
    """
    from mlb_luck_score.scoring.qualification import (
        QUALIFICATION_THRESHOLD_SETS,
        assign_qualification_status,
    )

    workhorse_starter = pd.DataFrame([_summary_row(bbe=674, games=42)])
    status = assign_qualification_status(
        workhorse_starter, thresholds=QUALIFICATION_THRESHOLD_SETS["primary"]
    )
    assert status.iloc[0] == "small_sample"


def test_pitcher_primary_qualifies_a_starter_and_not_a_reliever():
    """The starter-scale bar admits qualified starters, not relievers.

    Relievers are excluded by EXPOSURE, not by choice: across 2021-2024 the
    highest-volume reliever-season faced 311 eligible batted balls, so no
    reliever reaches a starter-scale bar. That is a documented property of
    this threshold set, not a hidden filter.
    """
    from mlb_luck_score.scoring.qualification import assign_qualification_status

    rows = pd.DataFrame(
        [
            _summary_row(bbe=470, games=30),  # qualified starter
            _summary_row(bbe=165, games=62),  # full-season reliever
            _summary_row(bbe=311, games=71),  # highest-volume reliever observed
        ]
    )
    status = assign_qualification_status(
        rows, thresholds=PITCHER_QUALIFICATION_THRESHOLD_SETS["pitcher_primary"]
    )
    assert list(status) == ["qualified", "small_sample", "small_sample"]


@pytest.fixture
def ledger_with_distinct_pitchers(v011_full_ledger):
    """The shared ledger with `pitcher` assigned INDEPENDENTLY of `batter`.

    The sign tests above hold pitcher == batter to isolate the sign flip.
    This fixture does the opposite: it makes the grouping genuinely
    different, so a bug that silently grouped by the wrong key would change
    the row count and be caught.
    """
    df, ledger, confidence = v011_full_ledger
    df = df.copy()
    rng = np.random.default_rng(11)
    df["pitcher"] = rng.integers(5000, 5006, size=len(df))
    return df, ledger, confidence


def test_pitcher_season_table_is_one_row_per_pitcher_season_with_intervals_and_status(
    ledger_with_distinct_pitchers,
):
    df, ledger, confidence = ledger_with_distinct_pitchers

    table = build_pitcher_season_table(df, ledger, confidence, n_bootstrap_reps=40)

    expected_rows = df.groupby(["pitcher", "season"], dropna=False).ngroups
    assert len(table) == expected_rows > 0
    assert not table.duplicated(subset=["pitcher", "season"]).any()

    # The bootstrap merged onto the right key -- a wrong-key merge leaves
    # these all null rather than failing loudly.
    assert table["observed_minus_expected_ci_low"].notna().all()
    assert table["observed_minus_expected_ci_high"].notna().all()

    # Qualification ran, using the pitcher threshold set by default.
    assert table["qualification_status"].notna().all()
    assert table.attrs["threshold_set"] == "pitcher_primary"


def test_reproduction_check_confirms_the_pitcher_path_matches_the_frozen_batter_path(
    v011_full_ledger,
):
    """Keyed on `batter`, the pitcher path must reproduce the batter table.

    This is the evidence that re-grouping introduces no arithmetic of its
    own: run the new code path on the SAME key the frozen pipeline uses,
    undo the sign flip, and every total must match to floating-point noise.
    A mismatch means the pitcher path is doing something the batter path
    does not, and no pitcher number should be trusted until it is explained.
    """
    df, ledger, confidence = v011_full_ledger

    report = verify_batter_side_reproduction(df, ledger, confidence)

    assert report["reproduces"] is True
    assert report["n_rows"] > 0
    assert report["max_abs_difference"] < 1e-9


def test_report_refuses_to_vouch_for_numbers_when_the_reproduction_check_fails():
    """A failed self-check must invalidate the run, not sit in a footnote.

    If the pitcher path stops reproducing the frozen batter path, every
    pitcher number from that run is suspect. The report has to say so in
    the field a reader keys on, not merely record the raw check result
    somewhere further down.
    """
    table = pd.DataFrame(
        {
            "pitcher": [1, 2],
            "season": [2024, 2024],
            "qualification_status": ["qualified", "small_sample"],
        }
    )
    failed = {
        "reproduces": False,
        "n_rows": 2,
        "max_abs_difference": 0.004,
        "worst_column": "total_contact_component_runs",
    }

    report = build_pitcher_report(table, failed, threshold_set_label="pitcher_primary")

    assert report["pitcher_values_trustworthy"] is False
    assert report["batter_side_reproduction"]["reproduces"] is False


def test_report_records_the_qualified_population_reference_point():
    """Zero is not the neutral point on a qualified board -- say the number.

    The reference is computed from QUALIFIED rows only: a leaderboard reader
    is comparing a pitcher against the pitchers he appears beside, not
    against the 793 pitcher-seasons that never cleared the bar. Including
    them would report a reference no displayed row is measured against.

    `score_is_recentered` is pinned False on purpose: this is descriptive
    context recorded ALONGSIDE the score. Subtracting it would redefine
    Contact Luck (RESEARCH_RULES.md, "Never silently redefine the Luck
    Score") and break comparability with every batter number.
    """
    table = pd.DataFrame(
        {
            "pitcher": [1, 2, 3],
            "season": [2024, 2024, 2024],
            "qualification_status": ["qualified", "qualified", "small_sample"],
            "observed_minus_expected_per_100": [2.0, 4.0, -50.0],
        }
    )
    passing = {"reproduces": True, "n_rows": 3, "max_abs_difference": 0.0, "worst_column": ""}

    report = build_pitcher_report(table, passing, threshold_set_label="pitcher_primary")

    reference = report["qualified_population_reference"]
    assert reference["n_qualified"] == 2
    assert reference["mean_runs_per_100"] == pytest.approx(3.0)
    assert reference["score_is_recentered"] is False
