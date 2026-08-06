"""Tests for the Contact Luck v0.11 game_pk-clustered bootstrap (Phase 4).

No test touches the network -- uses the shared `v011_full_ledger` fixture
(`tests/conftest.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.aggregation_uncertainty import (
    AggregationUncertaintyError,
    bootstrap_batter_season_intervals,
    verify_bootstrap_replicate_additivity,
)


def test_bootstrap_output_is_deterministic_given_same_seed(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    first = bootstrap_batter_season_intervals(df, ledger, n_reps=200, seed=123)
    second = bootstrap_batter_season_intervals(df, ledger, n_reps=200, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_output_differs_with_different_seed(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    first = bootstrap_batter_season_intervals(df, ledger, n_reps=200, seed=1)
    second = bootstrap_batter_season_intervals(df, ledger, n_reps=200, seed=2)
    # At least one interval bound should differ somewhere across all rows.
    diffs = (
        first["observed_minus_expected_ci_low"].to_numpy()
        != second["observed_minus_expected_ci_low"].to_numpy()
    )
    assert diffs.any()


def test_game_clustered_resampling_preserves_within_replicate_additivity(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    assert verify_bootstrap_replicate_additivity(df, ledger, n_reps=200, seed=42)


def test_single_game_batter_has_a_degenerate_zero_width_interval():
    # A cluster bootstrap with exactly ONE cluster cannot estimate between
    # -cluster variability -- the honest result is a point interval (width
    # zero), not a fabricated spread.
    df = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "batter": [1, 1],
            "season": [2024, 2024],
            "game_pk": [10, 10],
        }
    )
    ledger = pd.DataFrame(
        {
            "component_eligibility_status": [
                "defense_opportunity_unavailable;advancement_not_modeled_for_this_play"
            ]
            * 2,
            "observed_contact_result_run_value": [0.5, 0.3],
            "baseline_expected_contact_run_value": [0.1, 0.1],
            "observed_final_run_value": [0.5, 0.3],
            "defensive_execution_contribution": [np.nan, np.nan],
            "advancement_execution_contribution": [0.0, 0.0],
            "unexplained_residual": [0.4, 0.2],
        }
    )
    result = bootstrap_batter_season_intervals(df, ledger, n_reps=200, seed=42)
    row = result.iloc[0]
    assert row["n_games"] == 1
    assert row["observed_minus_expected_ci_low"] == pytest.approx(
        row["observed_minus_expected_ci_high"]
    )


def test_interval_width_shrinks_as_game_count_increases():
    # More independent games -> less relative sampling variability in the
    # per-game-clustered bootstrap, for a fixed per-game value distribution.
    rng = np.random.default_rng(0)

    def _make_inputs(n_games: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        per_game_value = rng.normal(0.1, 0.5, size=n_games)
        df = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(n_games)],
                "batter": [1] * n_games,
                "season": [2024] * n_games,
                "game_pk": list(range(n_games)),
            }
        )
        ledger = pd.DataFrame(
            {
                "component_eligibility_status": [
                    "defense_opportunity_unavailable;advancement_not_modeled_for_this_play"
                ]
                * n_games,
                "observed_contact_result_run_value": per_game_value,
                "baseline_expected_contact_run_value": [0.0] * n_games,
                "observed_final_run_value": per_game_value,
                "defensive_execution_contribution": [np.nan] * n_games,
                "advancement_execution_contribution": [0.0] * n_games,
                "unexplained_residual": per_game_value,
            }
        )
        return df, ledger

    df_small, ledger_small = _make_inputs(10)
    df_large, ledger_large = _make_inputs(100)

    small_result = bootstrap_batter_season_intervals(df_small, ledger_small, n_reps=500, seed=42)
    large_result = bootstrap_batter_season_intervals(df_large, ledger_large, n_reps=500, seed=42)

    small_width_per_100 = (
        small_result["observed_minus_expected_per_100_ci_high"].iloc[0]
        - small_result["observed_minus_expected_per_100_ci_low"].iloc[0]
    )
    large_width_per_100 = (
        large_result["observed_minus_expected_per_100_ci_high"].iloc[0]
        - large_result["observed_minus_expected_per_100_ci_low"].iloc[0]
    )
    assert large_width_per_100 < small_width_per_100


def test_bootstrap_rejects_mismatched_index(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    with pytest.raises(AggregationUncertaintyError, match="identical index"):
        bootstrap_batter_season_intervals(df, ledger.iloc[:-1], n_reps=50)


def test_bootstrap_rejects_non_positive_n_reps(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    with pytest.raises(AggregationUncertaintyError, match="n_reps"):
        bootstrap_batter_season_intervals(df, ledger, n_reps=0)


def test_bootstrap_covers_every_batter_season_in_full_ledger(v011_full_ledger):
    df, ledger, _ = v011_full_ledger
    result = bootstrap_batter_season_intervals(df, ledger, n_reps=100, seed=42)
    expected_keys = set(zip(df["batter"], df["season"], strict=True))
    actual_keys = set(zip(result["batter"], result["season"], strict=True))
    assert actual_keys == expected_keys
