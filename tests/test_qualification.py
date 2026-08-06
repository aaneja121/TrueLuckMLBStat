"""Tests for the Contact Luck v0.11 qualification rules (Phase 5).

No test touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.scoring.qualification import (
    QUALIFICATION_THRESHOLD_SETS,
    STATUS_INSUFFICIENT_COMPONENT_COVERAGE,
    STATUS_NOT_REPORTABLE,
    STATUS_PROVISIONALLY_QUALIFIED,
    STATUS_QUALIFIED,
    STATUS_SMALL_SAMPLE,
    QualificationError,
    QualificationThresholds,
    assign_qualification_status,
    compare_threshold_sets,
    qualified_mask,
    summarize_qualification_counts,
)


def _summary_row(**overrides) -> dict:
    base = {
        "batter": 1,
        "season": 2024,
        "eligible_batted_balls": 250,
        "games": 120,
        "component_coverage_fraction": 0.8,
        "share_of_value_from_provisional_components": 0.2,
        "missing_input_frequency": 0.1,
    }
    base.update(overrides)
    return base


def test_fully_clean_row_is_qualified():
    summary = pd.DataFrame([_summary_row()])
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_QUALIFIED


def test_zero_eligible_plays_is_not_reportable():
    summary = pd.DataFrame(
        [_summary_row(eligible_batted_balls=0, games=0, component_coverage_fraction=None)]
    )
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_NOT_REPORTABLE


def test_below_volume_threshold_is_small_sample():
    summary = pd.DataFrame([_summary_row(eligible_batted_balls=50, games=30)])
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_SMALL_SAMPLE


def test_low_coverage_with_adequate_volume_is_insufficient_component_coverage():
    summary = pd.DataFrame([_summary_row(component_coverage_fraction=0.1)])
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_INSUFFICIENT_COMPONENT_COVERAGE


def test_high_provisional_share_with_adequate_coverage_is_provisionally_qualified():
    summary = pd.DataFrame([_summary_row(share_of_value_from_provisional_components=0.9)])
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_PROVISIONALLY_QUALIFIED


def test_high_missing_input_frequency_is_provisionally_qualified():
    summary = pd.DataFrame([_summary_row(missing_input_frequency=0.9)])
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_PROVISIONALLY_QUALIFIED


def test_volume_failure_takes_precedence_over_coverage_and_provisional_failures():
    # Fails EVERY check simultaneously -- must report the single highest
    # -precedence reason (small_sample), not coverage or provisional.
    summary = pd.DataFrame(
        [
            _summary_row(
                eligible_batted_balls=10,
                games=5,
                component_coverage_fraction=0.05,
                share_of_value_from_provisional_components=0.99,
                missing_input_frequency=0.99,
            )
        ]
    )
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_SMALL_SAMPLE


def test_boundary_values_are_inclusive():
    # Exactly AT the primary threshold must count as meeting it (>=, <=).
    thresholds = QUALIFICATION_THRESHOLD_SETS["primary"]
    summary = pd.DataFrame(
        [
            _summary_row(
                eligible_batted_balls=thresholds.min_eligible_batted_balls,
                games=thresholds.min_games,
                component_coverage_fraction=thresholds.min_component_coverage,
                share_of_value_from_provisional_components=thresholds.max_provisional_value_share,
                missing_input_frequency=thresholds.max_missing_input_frequency,
            )
        ]
    )
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_QUALIFIED


def test_just_below_volume_boundary_is_small_sample():
    thresholds = QUALIFICATION_THRESHOLD_SETS["primary"]
    summary = pd.DataFrame(
        [_summary_row(eligible_batted_balls=thresholds.min_eligible_batted_balls - 1)]
    )
    status = assign_qualification_status(summary)
    assert status.iloc[0] == STATUS_SMALL_SAMPLE


def test_custom_thresholds_instance_is_respected():
    custom = QualificationThresholds(
        label="custom",
        min_eligible_batted_balls=5,
        min_games=2,
        min_component_coverage=0.0,
        max_provisional_value_share=1.0,
        max_missing_input_frequency=1.0,
    )
    summary = pd.DataFrame([_summary_row(eligible_batted_balls=5, games=2)])
    status = assign_qualification_status(summary, thresholds=custom)
    assert status.iloc[0] == STATUS_QUALIFIED


def test_unknown_threshold_set_name_raises():
    summary = pd.DataFrame([_summary_row()])
    with pytest.raises(QualificationError, match="Unknown threshold set"):
        assign_qualification_status(summary, thresholds="nonexistent")


def test_missing_required_column_raises():
    summary = pd.DataFrame([{"eligible_batted_balls": 100}])
    with pytest.raises(QualificationError, match="missing required column"):
        assign_qualification_status(summary)


def test_summarize_qualification_counts_includes_zero_count_statuses():
    summary = pd.DataFrame([_summary_row()])  # everyone qualified
    status = assign_qualification_status(summary)
    counts = summarize_qualification_counts(status)
    assert counts[STATUS_QUALIFIED] == 1
    assert counts[STATUS_NOT_REPORTABLE] == 0
    assert counts[STATUS_SMALL_SAMPLE] == 0
    assert set(counts.keys()) == {
        STATUS_QUALIFIED,
        STATUS_PROVISIONALLY_QUALIFIED,
        STATUS_SMALL_SAMPLE,
        STATUS_INSUFFICIENT_COMPONENT_COVERAGE,
        STATUS_NOT_REPORTABLE,
    }


def test_qualified_mask_matches_status_equality():
    summary = pd.DataFrame([_summary_row(), _summary_row(eligible_batted_balls=10, games=5)])
    status = assign_qualification_status(summary)
    mask = qualified_mask(status)
    assert mask.tolist() == [True, False]


def test_compare_threshold_sets_can_disagree_across_sets():
    # A row that clears "lenient" but not "strict" demonstrates genuine
    # threshold sensitivity -- this is the exact case Phase 6's sensitivity
    # analysis is built to surface.
    summary = pd.DataFrame([_summary_row(eligible_batted_balls=150, games=60)])
    comparison = compare_threshold_sets(summary)
    assert comparison["status_lenient"].iloc[0] == STATUS_QUALIFIED
    assert comparison["status_strict"].iloc[0] != STATUS_QUALIFIED
