"""Contact Forecast R1: temporal window construction and leakage.

Every claim this study makes about "features saw only events before the
cutoff" reduces to the index arithmetic in `forecast.windows`. These tests
check it directly, and the two `*_mutation_*` tests check it the strong way:
by rewriting the data a window must not be able to see and proving the
window does not move.

Synthetic data only -- no network, no real Statcast rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import windows as w
from forecast.forecast_config import ForecastSeasonError

_BBE_PER_GAME = 4


def make_ledger(
    *,
    batter: int = 100001,
    season: int = 2023,
    n_events: int = 200,
    start_date: str = "2023-04-01",
    observed_slope: float = 0.01,
    deserved_slope: float = 0.005,
    first_game_pk: int = 700001,
) -> pd.DataFrame:
    """A synthetic single-hitter-season ledger with hand-computable values.

    Event `i` (1-based, chronological) carries `observed_run_value =
    i * observed_slope` and `expected_run_value = i * deserved_slope`, so any
    window sum is an arithmetic series a test can state in closed form.
    Events are packed `_BBE_PER_GAME` to a game, one game per calendar day.
    """
    index = np.arange(1, n_events + 1)
    game_offset = (index - 1) // _BBE_PER_GAME
    observed = index * observed_slope
    deserved = index * deserved_slope
    return pd.DataFrame(
        {
            "event_id": [f"{batter}-{season}-{i}" for i in index],
            "batter": batter,
            "season": season,
            "game_date": (
                pd.to_datetime(start_date) + pd.to_timedelta(game_offset, unit="D")
            ).strftime("%Y-%m-%d"),
            "game_pk": first_game_pk + game_offset,
            "at_bat_number": ((index - 1) % _BBE_PER_GAME) + 1,
            "pitch_number": 1,
            "observed_contact_result_run_value": observed,
            "baseline_expected_contact_run_value": deserved,
            "contact_result_surprise": observed - deserved,
        }
    )


def test_ordering_assigns_a_contiguous_one_based_index() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=120))
    assert ordered["bbe_index"].tolist() == list(range(1, 121))
    assert ordered["game_date"].is_monotonic_increasing


def test_ordering_is_chronological_even_when_input_rows_are_shuffled() -> None:
    ledger = make_ledger(n_events=80)
    shuffled = ledger.sample(frac=1.0, random_state=7).reset_index(drop=True)

    ordered = w.order_eligible_events(shuffled)

    assert ordered["event_id"].tolist() == [f"100001-2023-{i}" for i in range(1, 81)]


def test_ordering_indexes_each_hitter_season_independently() -> None:
    ledger = pd.concat(
        [
            make_ledger(batter=1, season=2023, n_events=60),
            make_ledger(batter=2, season=2023, n_events=90, first_game_pk=800001),
            make_ledger(batter=1, season=2024, n_events=70, start_date="2024-04-01"),
        ],
        ignore_index=True,
    )
    ledger["event_id"] = [f"e{i}" for i in range(len(ledger))]

    ordered = w.order_eligible_events(ledger)

    sizes = ordered.groupby(["batter", "season"])["bbe_index"].max().to_dict()
    assert sizes == {(1, 2023): 60, (1, 2024): 70, (2, 2023): 90}


def test_doubleheader_within_a_day_is_ordered_by_ascending_game_pk() -> None:
    """The documented tie-break: same calendar day, lower game_pk comes first."""
    ledger = make_ledger(n_events=8)
    # Collapse two games onto one day, with the SECOND game given the lower
    # game_pk in row order, so only the tie-break can produce the right order.
    ledger.loc[:3, "game_date"] = "2023-04-01"
    ledger.loc[4:, "game_date"] = "2023-04-01"
    ledger.loc[:3, "game_pk"] = 900002
    ledger.loc[4:, "game_pk"] = 900001

    ordered = w.order_eligible_events(ledger)

    assert ordered["game_pk"].tolist() == [900001] * 4 + [900002] * 4
    assert ordered["bbe_index"].tolist() == list(range(1, 9))


def test_targets_match_a_hand_computed_arithmetic_series() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=200))

    windows = w.build_windows(ordered, cutoffs=(50,), horizons=(100,))

    row = windows.iloc[0]
    # Target window is events 51..150: sum(i) = 100 * (51 + 150) / 2 = 10050.
    assert row["target_realized_rv_per_100"] == pytest.approx(100 * 0.01 * 10050 / 100)
    assert row["target_deserved_rv_per_100"] == pytest.approx(100 * 0.005 * 10050 / 100)


def test_window_indices_bracket_exactly_the_declared_cutoff_and_horizon() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=300))

    windows = w.build_windows(ordered, cutoffs=(150,), horizons=(50,))

    row = windows.iloc[0]
    assert (row["history_start_index"], row["history_end_index"]) == (1, 150)
    assert (row["target_start_index"], row["target_end_index"]) == (151, 200)


def test_hitter_season_is_included_only_with_cutoff_plus_horizon_events() -> None:
    """The inclusion bar is exact: 149 events does not make a 100+50 window."""
    short = w.order_eligible_events(make_ledger(batter=1, n_events=149))
    exact = w.order_eligible_events(make_ledger(batter=2, n_events=150))

    assert w.build_windows(short, cutoffs=(100,), horizons=(50,)).empty
    assert len(w.build_windows(exact, cutoffs=(100,), horizons=(50,))) == 1


def test_one_hitter_season_populates_every_cell_it_qualifies_for() -> None:
    """Overlapping cells per hitter are expected -- and are why we cluster."""
    ordered = w.order_eligible_events(make_ledger(n_events=300))

    windows = w.build_windows(ordered, cutoffs=(50, 100, 150, 200), horizons=(50, 100))

    assert len(windows) == 8
    assert windows["batter"].nunique() == 1


def test_mutating_events_after_the_target_window_cannot_move_a_target() -> None:
    """The strong leakage check: rewrite the unseeable future, expect no change.

    If any target were computed over the whole hitter-season -- an easy
    mistake with a groupby -- this test fails immediately.
    """
    baseline = w.build_windows(
        w.order_eligible_events(make_ledger(n_events=300)), cutoffs=(50,), horizons=(100,)
    )

    tampered_ledger = make_ledger(n_events=300)
    beyond_target = tampered_ledger.index >= 150  # 0-based: events 151..300
    tampered_ledger.loc[beyond_target, "observed_contact_result_run_value"] = 999.0
    tampered_ledger.loc[beyond_target, "baseline_expected_contact_run_value"] = -999.0
    tampered_ledger.loc[beyond_target, "contact_result_surprise"] = 1998.0
    tampered = w.build_windows(
        w.order_eligible_events(tampered_ledger), cutoffs=(50,), horizons=(100,)
    )

    pd.testing.assert_frame_equal(baseline, tampered)


def test_mutating_events_after_the_cutoff_cannot_move_the_history_slice() -> None:
    """The same check for the feature side: `history_events` sees only 1..K."""
    baseline = w.history_events(w.order_eligible_events(make_ledger(n_events=200)), 100)

    tampered_ledger = make_ledger(n_events=200)
    tampered_ledger.loc[tampered_ledger.index >= 100, "observed_contact_result_run_value"] = 999.0
    tampered_ledger.loc[
        tampered_ledger.index >= 100, "baseline_expected_contact_run_value"
    ] = -999.0
    tampered_ledger.loc[tampered_ledger.index >= 100, "contact_result_surprise"] = 1998.0
    tampered = w.history_events(w.order_eligible_events(tampered_ledger), 100)

    pd.testing.assert_frame_equal(baseline, tampered)


def test_history_events_returns_exactly_the_pre_cutoff_rows() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=200))

    history = w.history_events(ordered, 75)

    assert history["bbe_index"].max() == 75
    assert len(history) == 75
    assert (history["cutoff"] == 75).all()


def test_history_events_rejects_a_non_positive_cutoff() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=60))
    with pytest.raises(w.ForecastWindowError, match="positive integer"):
        w.history_events(ordered, 0)


def test_causality_assertion_catches_an_overlapping_target_window() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=200))
    windows = w.build_windows(ordered, cutoffs=(100,), horizons=(50,))

    windows.loc[0, "target_start_index"] = 100  # one event back inside history

    with pytest.raises(w.ForecastWindowError, match="LEAKAGE"):
        w.assert_windows_causal(windows)


def test_causality_assertion_catches_a_target_starting_before_history_ends() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=200))
    windows = w.build_windows(ordered, cutoffs=(100,), horizons=(50,))

    windows.loc[0, "target_start_date"] = pd.Timestamp("2023-04-01")

    with pytest.raises(w.ForecastWindowError, match="LEAKAGE"):
        w.assert_windows_causal(windows)


def test_causality_assertion_catches_a_target_past_the_end_of_the_season() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=200))
    windows = w.build_windows(ordered, cutoffs=(100,), horizons=(50,))

    windows.loc[0, "n_eligible_bbe_season"] = 149

    with pytest.raises(w.ForecastWindowError, match="past the hitter-season"):
        w.assert_windows_causal(windows)


def test_schema_check_rejects_a_missing_column() -> None:
    ledger = make_ledger(n_events=60).drop(columns=["baseline_expected_contact_run_value"])
    with pytest.raises(w.ForecastWindowError, match="missing required column"):
        w.assert_ledger_schema(ledger)


def test_schema_check_rejects_a_duplicated_event_id() -> None:
    ledger = make_ledger(n_events=60)
    ledger.loc[5, "event_id"] = ledger.loc[4, "event_id"]
    with pytest.raises(w.ForecastWindowError, match="duplicated event_id"):
        w.assert_ledger_schema(ledger)


def test_schema_check_rejects_a_null_in_the_ordering_key() -> None:
    ledger = make_ledger(n_events=60)
    ledger.loc[3, "at_bat_number"] = None
    with pytest.raises(w.ForecastWindowError, match="ordering column"):
        w.assert_ledger_schema(ledger)


def test_schema_check_rejects_an_unscored_batted_ball() -> None:
    """Unscored rows must be dropped upstream WITH a count, never carried as NaN."""
    ledger = make_ledger(n_events=60)
    ledger.loc[10, "baseline_expected_contact_run_value"] = np.nan
    with pytest.raises(w.ForecastWindowError, match="must be scored"):
        w.assert_ledger_schema(ledger)


def test_schema_check_rejects_a_violated_contact_stage_identity() -> None:
    """A ledger where surprise != Rc - E0 is not the frozen methodology."""
    ledger = make_ledger(n_events=60)
    ledger.loc[7, "contact_result_surprise"] = ledger.loc[7, "contact_result_surprise"] + 0.5
    with pytest.raises(w.ForecastWindowError, match="Contact-stage identity violated"):
        w.assert_ledger_schema(ledger)


def test_schema_check_rejects_a_sealed_season() -> None:
    ledger = make_ledger(n_events=60, season=2025, start_date="2025-04-01")
    with pytest.raises(ForecastSeasonError, match="permanently sealed"):
        w.assert_ledger_schema(ledger)


def test_ordering_rejects_an_ambiguous_sort_key() -> None:
    ledger = make_ledger(n_events=60)
    ledger.loc[5, ["game_date", "game_pk", "at_bat_number", "pitch_number"]] = ledger.loc[
        4, ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    ].to_numpy()
    with pytest.raises(w.ForecastWindowError, match="ambiguous"):
        w.order_eligible_events(ledger)


def test_build_windows_requires_an_ordered_frame() -> None:
    with pytest.raises(w.ForecastWindowError, match="order_eligible_events first"):
        w.build_windows(make_ledger(n_events=200))


def test_inclusion_summary_reports_the_survivorship_condition() -> None:
    ledger = pd.concat(
        [
            make_ledger(batter=b, n_events=n, first_game_pk=700001 + 1000 * b)
            for b, n in [(1, 300), (2, 300), (3, 120)]
        ],
        ignore_index=True,
    )
    ordered = w.order_eligible_events(ledger)

    summary = w.summarize_inclusion(ordered, cutoffs=(100,), horizons=(100,))

    row = summary.iloc[0]
    assert row["n_reached_cutoff"] == 3  # all three hitters reach 100 BBE
    assert row["n_included"] == 2  # only two survive to 200
    assert row["n_excluded"] == 1
    assert row["exclusion_rate"] == pytest.approx(1 / 3)


def test_doubleheader_boundary_count_flags_only_a_real_boundary() -> None:
    """The count isolates windows where the game_pk tie-break decides the cutoff."""
    clean = w.order_eligible_events(make_ledger(n_events=120))
    assert (
        w.count_doubleheader_boundary_windows(clean, cutoffs=(50,)).iloc[0][
            "n_boundary_inside_doubleheader"
        ]
        == 0
    )

    # Events 49-52 already share one game on one calendar day. Split that
    # game in two, keeping the date, so the cutoff at 50 falls between the
    # day's two games -- exactly where the game_pk tie-break decides which
    # side of the cutoff a batted ball lands on. The sort is date-first, so
    # the second game's game_pk only has to exceed the first's.
    ledger = make_ledger(n_events=120)
    ledger.loc[50:51, "game_pk"] = 999999
    straddled = w.order_eligible_events(ledger)

    assert (
        w.count_doubleheader_boundary_windows(straddled, cutoffs=(50,)).iloc[0][
            "n_boundary_inside_doubleheader"
        ]
        == 1
    )


def test_empty_cell_produces_an_empty_table_with_the_right_columns() -> None:
    ordered = w.order_eligible_events(make_ledger(n_events=60))

    windows = w.build_windows(ordered, cutoffs=(200,), horizons=(100,))

    assert windows.empty
    assert list(windows.columns) == list(w.WINDOW_COLUMNS)


def test_unresolved_batted_balls_are_dropped_to_match_production_denominator() -> None:
    """`field_error`/`fielders_choice` carry a null outcome_class and no run value.

    Production's `eligible_batted_balls` is `sum(resolved)` (`mlb_luck_
    score.scoring.aggregate_attribution`), so a rate this package calls
    "per 100 eligible batted balls" has to exclude them too or it would not
    be the same quantity the published metric reports.
    """
    ledger = make_ledger(n_events=60)
    ledger["outcome_class"] = "out"
    ledger.loc[[3, 11], "outcome_class"] = None
    ledger.loc[
        [3, 11],
        [
            "observed_contact_result_run_value",
            "baseline_expected_contact_run_value",
            "contact_result_surprise",
        ],
    ] = np.nan

    kept, report = w.select_scored_eligible_events(ledger)

    assert report == {
        "n_input_rows": 60,
        "n_unresolved_outcome_class": 2,
        "n_dropped_for_missing_run_value": 0,
        "n_kept": 58,
    }
    assert len(kept) == 58
    assert kept["outcome_class"].notna().all()


def test_dropping_unresolved_events_renumbers_the_index_contiguously() -> None:
    """Why the exclusion must precede ordering: it changes what event 50 is."""
    ledger = make_ledger(n_events=60)
    ledger["outcome_class"] = "out"
    ledger.loc[[0, 1], "outcome_class"] = None
    ledger.loc[
        [0, 1],
        [
            "observed_contact_result_run_value",
            "baseline_expected_contact_run_value",
            "contact_result_surprise",
        ],
    ] = np.nan

    kept, _ = w.select_scored_eligible_events(ledger)
    ordered = w.order_eligible_events(kept)

    assert ordered["bbe_index"].tolist() == list(range(1, 59))
    assert ordered.iloc[0]["event_id"] == "100001-2023-3"


def test_selector_requires_outcome_class() -> None:
    with pytest.raises(w.ForecastWindowError, match="no outcome_class column"):
        w.select_scored_eligible_events(make_ledger(n_events=10))


def test_selector_refuses_a_fully_unresolved_ledger() -> None:
    ledger = make_ledger(n_events=10)
    ledger["outcome_class"] = None
    with pytest.raises(w.ForecastWindowError, match="dropped as unresolved"):
        w.select_scored_eligible_events(ledger)
