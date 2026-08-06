"""Tests for the Contact Luck v0.11 aggregation stability/sensitivity module (Phase 6/8).

No test touches the network or real data -- hand-built player-season tables
exercise the pure-computation functions directly, without requiring a full
model-training run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.models.evaluate_aggregation_stability import (
    AggregationStabilityError,
    compute_component_covariance,
    compute_interval_width_by_sample_size,
    compute_provisional_exclusion_sensitivity,
    compute_qualification_threshold_sensitivity,
    compute_split_half_reliability,
)
from mlb_luck_score.scoring.run_season_aggregation import SeasonAggregationArtifacts


def _player_season_row(**overrides) -> dict:
    base = {
        "batter": 1,
        "season": 2024,
        "eligible_batted_balls": 250,
        "games": 100,
        "n_games": 100,
        "component_coverage_fraction": 0.8,
        "share_of_value_from_provisional_components": 0.2,
        "missing_input_frequency": 0.1,
        "total_observed_minus_expected_runs": 5.0,
        "total_contact_component_runs": 2.0,
        "total_unexplained_residual_component_runs": 1.0,
        "total_defensive_execution_component_runs": 1.5,
        "total_advancement_component_runs": 0.5,
        "total_observed_minus_expected_excluding_provisional_runs": 4.5,
        "observed_minus_expected_per_100": 2.0,
        "observed_minus_expected_excluding_provisional_per_100": 1.8,
        "observed_minus_expected_per_100_ci_low": 1.0,
        "observed_minus_expected_per_100_ci_high": 3.0,
    }
    base.update(overrides)
    return base


def test_compute_interval_width_by_sample_size_orders_bins_by_games():
    rows = []
    for n_games, width in [(10, 4.0), (50, 2.0), (150, 0.5)]:
        for _ in range(5):
            rows.append(
                _player_season_row(
                    n_games=n_games,
                    observed_minus_expected_per_100_ci_low=0.0,
                    observed_minus_expected_per_100_ci_high=width,
                )
            )
    player_season = pd.DataFrame(rows)
    result = compute_interval_width_by_sample_size(player_season, n_bins=3)
    assert len(result) == 3
    widths = [row["median_interval_width"] for row in result]
    assert widths == sorted(widths, reverse=True)  # width should shrink as n_games grows


def test_compute_interval_width_by_sample_size_handles_empty_input():
    player_season = pd.DataFrame(
        columns=[
            "n_games",
            "observed_minus_expected_per_100_ci_low",
            "observed_minus_expected_per_100_ci_high",
        ]
    )
    assert compute_interval_width_by_sample_size(player_season) == []


def test_compute_qualification_threshold_sensitivity_reports_all_sets():
    player_season = pd.DataFrame([_player_season_row(batter=i) for i in range(30)])
    result = compute_qualification_threshold_sensitivity(player_season, top_n=5)
    assert set(result["qualified_counts_by_threshold_set"].keys()) == {
        "primary",
        "strict",
        "lenient",
    }
    assert "strict" in result["top_5_overlap_fraction_vs_primary"]


def test_compute_provisional_exclusion_sensitivity_flags_biggest_movers():
    rows = [
        _player_season_row(
            batter=1,
            observed_minus_expected_per_100=5.0,
            observed_minus_expected_excluding_provisional_per_100=1.0,  # big mover
        ),
        _player_season_row(
            batter=2,
            observed_minus_expected_per_100=2.0,
            observed_minus_expected_excluding_provisional_per_100=1.9,  # small mover
        ),
    ]
    player_season = pd.DataFrame(rows)
    result = compute_provisional_exclusion_sensitivity(player_season, top_n=1)
    movers = result["top_1_biggest_movers_when_provisional_excluded"]
    assert len(movers) == 1
    assert movers[0]["batter"] == 1


def test_compute_component_covariance_returns_matrices_for_enough_players():
    player_season = pd.DataFrame([_player_season_row(batter=i) for i in range(5)])
    result = compute_component_covariance(player_season)
    assert result["n_players"] == 5
    assert result["covariance"] is not None
    assert result["correlation"] is not None


def test_compute_component_covariance_handles_too_few_players():
    player_season = pd.DataFrame([_player_season_row(batter=1)])
    result = compute_component_covariance(player_season)
    assert result["covariance"] is None


def test_compute_split_half_reliability_rejects_unknown_split():
    df = pd.DataFrame({"game_date": ["2024-04-01"], "game_pk": [1], "event_id": ["e1"]})
    artifacts = SeasonAggregationArtifacts(
        scoring_df=df,
        ledger=pd.DataFrame(),
        confidence=pd.DataFrame(),
        player_season=pd.DataFrame(),
        report={},
    )
    with pytest.raises(AggregationStabilityError, match="Unknown split"):
        compute_split_half_reliability(artifacts, split="quarterly")
