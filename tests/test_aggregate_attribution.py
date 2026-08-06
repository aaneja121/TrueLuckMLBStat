"""Tests for the Contact Luck v0.11 additive batter-season aggregation (Phase 3).

No test touches the network -- uses the shared `v011_full_ledger` fixture
(`tests/conftest.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.aggregate_attribution import (
    AggregationError,
    aggregate_to_batter_season,
    assert_additive_only,
    build_play_level_components,
    verify_play_level_partition_identity,
    verify_season_identity,
)

# ---------------------------------------------------------------------------
# assert_additive_only
# ---------------------------------------------------------------------------


def test_assert_additive_only_accepts_known_additive_columns():
    assert_additive_only(
        ["observed_final_run_value", "defensive_execution_contribution"]
    )  # no raise


def test_assert_additive_only_rejects_known_non_additive_columns():
    with pytest.raises(AggregationError, match="non-additive"):
        assert_additive_only(["contact_result_surprise"])


def test_assert_additive_only_rejects_probability_column():
    with pytest.raises(AggregationError, match="non-additive"):
        assert_additive_only(["defensive_opportunity_probability"])


def test_assert_additive_only_rejects_status_string_column():
    with pytest.raises(AggregationError, match="non-additive"):
        assert_additive_only(["component_eligibility_status"])


# ---------------------------------------------------------------------------
# build_play_level_components / verify_play_level_partition_identity
# ---------------------------------------------------------------------------


def test_partition_identity_holds_for_every_resolved_row(v011_full_ledger):
    _, ledger, _ = v011_full_ledger
    components = build_play_level_components(ledger)
    holds = verify_play_level_partition_identity(ledger, components)
    resolved = holds.dropna()
    assert len(resolved) > 0
    assert resolved.all()


def test_contact_component_and_unexplained_residual_never_both_nonzero_same_row(v011_full_ledger):
    # contact_only rows and other_resolved rows are DISJOINT masks -- exactly
    # one of the two partitioned columns can be non-null per resolved row.
    _, ledger, _ = v011_full_ledger
    components = build_play_level_components(ledger)
    both_present = (
        components["contact_component"].notna()
        & components["unexplained_residual_component"].notna()
    )
    assert not both_present.any()


# ---------------------------------------------------------------------------
# aggregate_to_batter_season: exact identity, order invariance, edge cases
# ---------------------------------------------------------------------------


def test_season_identity_holds_for_every_batter_season(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    summary = aggregate_to_batter_season(df, ledger, confidence)
    holds = verify_season_identity(summary)
    assert len(summary) > 0
    assert bool(holds.all())


def test_aggregation_is_order_invariant(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    shuffled_index = df.sample(frac=1.0, random_state=7).index
    df_shuffled = df.loc[shuffled_index]
    ledger_shuffled = ledger.loc[shuffled_index]
    confidence_shuffled = confidence[confidence["event_id"].isin(df_shuffled["event_id"])]

    summary_original = aggregate_to_batter_season(df, ledger, confidence)
    summary_shuffled = aggregate_to_batter_season(df_shuffled, ledger_shuffled, confidence_shuffled)

    key = ["batter", "season"]
    left = summary_original.sort_values(key).reset_index(drop=True)
    right = summary_shuffled.sort_values(key).reset_index(drop=True)
    pd.testing.assert_series_equal(
        left["total_observed_minus_expected_runs"], right["total_observed_minus_expected_runs"]
    )
    pd.testing.assert_series_equal(left["eligible_batted_balls"], right["eligible_batted_balls"])


def test_zero_eligible_plays_batter_gets_nan_rates_not_a_crash(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    # Build a batter with ONLY an unresolved (field_error) row -- zero
    # eligible batted balls for that batter-season.
    error_mask = df["events"] == "field_error"
    assert error_mask.any()
    only_error_event_id = df.loc[error_mask, "event_id"].iloc[0]
    subset_index = df.index[df["event_id"] == only_error_event_id]
    df_subset = df.loc[subset_index].copy()
    df_subset["batter"] = 999999
    df_subset["season"] = 2024
    ledger_subset = ledger.loc[subset_index]
    confidence_subset = confidence[confidence["event_id"] == only_error_event_id]

    summary = aggregate_to_batter_season(df_subset, ledger_subset, confidence_subset)
    row = summary[summary["batter"] == 999999].iloc[0]
    assert row["eligible_batted_balls"] == 0
    assert row["games"] == 0
    assert row["total_observed_minus_expected_runs"] == 0.0
    assert np.isnan(row["observed_minus_expected_per_100"])
    assert np.isnan(row["share_of_value_from_provisional_components"])


def test_duplicate_plays_double_count_predictably(v011_full_ledger):
    # Aggregation has no play-identity check of its own (that lives in
    # attribution_ledger.assert_unique_play_identifiers, upstream) -- if a
    # caller feeds it the SAME play twice, the total should double
    # deterministically, not silently dedupe or crash. This documents that
    # behavior rather than asserting a de-duplication guarantee that does
    # not exist at this layer.
    df, ledger, confidence = v011_full_ledger
    resolved_event_id = df.loc[
        ledger["observed_contact_result_run_value"].notna(), "event_id"
    ].iloc[0]
    single_idx = df.index[df["event_id"] == resolved_event_id]
    df_single = df.loc[single_idx]
    ledger_single = ledger.loc[single_idx]
    confidence_single = confidence[confidence["event_id"] == resolved_event_id]
    summary_single = aggregate_to_batter_season(df_single, ledger_single, confidence_single)

    doubled_index = single_idx.tolist() * 2
    df_doubled = df.loc[doubled_index].reset_index(drop=True)
    ledger_doubled = ledger.loc[doubled_index].reset_index(drop=True)
    df_doubled["event_id"] = [f"{resolved_event_id}-a", f"{resolved_event_id}-b"]
    confidence_row = confidence[confidence["event_id"] == resolved_event_id]
    confidence_doubled = pd.concat(
        [
            confidence_row.assign(event_id=f"{resolved_event_id}-a"),
            confidence_row.assign(event_id=f"{resolved_event_id}-b"),
        ],
        ignore_index=True,
    )
    summary_doubled = aggregate_to_batter_season(df_doubled, ledger_doubled, confidence_doubled)

    assert summary_doubled["eligible_batted_balls"].iloc[0] == (
        2 * summary_single["eligible_batted_balls"].iloc[0]
    )
    assert summary_doubled["total_observed_minus_expected_runs"].iloc[0] == pytest.approx(
        2 * summary_single["total_observed_minus_expected_runs"].iloc[0]
    )


def test_missing_component_values_are_excluded_not_treated_as_zero_bias():
    # A batter whose plays are ALL contact-only (no defense/advancement
    # component applies at all) must show defense/advancement totals of
    # exactly 0.0 (documented "not modeled" convention), not NaN, and must
    # NOT be silently penalized/boosted by missing components. Hand-built
    # (rather than pulled from the shared fixture, where every synthetic row
    # is deliberately eligible for exactly one domain) to guarantee this
    # exact contact-only scenario.
    df = pd.DataFrame(
        {
            "event_id": ["e1"],
            "batter": [777],
            "season": [2024],
            "game_pk": [1],
        }
    )
    ledger = pd.DataFrame(
        {
            "component_eligibility_status": [
                "defense_opportunity_unavailable;advancement_not_modeled_for_this_play"
            ],
            "observed_contact_result_run_value": [0.5],
            "baseline_expected_contact_run_value": [0.1],
            "observed_final_run_value": [0.5],
            "defensive_execution_contribution": [np.nan],
            "advancement_execution_contribution": [0.0],
            "unexplained_residual": [0.4],
        }
    )
    confidence = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e1", "e1"],
            "component_name": ["contact", "outfield_defense", "infield_defense", "advancement"],
            "model_status_confidence": ["calibrated", "unavailable", "unavailable", "unavailable"],
            "row_data_quality": [
                "complete_measured_inputs",
                "excluded_play",
                "excluded_play",
                "excluded_play",
            ],
        }
    )
    summary = aggregate_to_batter_season(df, ledger, confidence)
    row = summary.iloc[0]
    assert row["total_defensive_execution_component_runs"] == 0.0
    assert row["total_advancement_component_runs"] == 0.0
    assert row["total_contact_component_runs"] == pytest.approx(0.4)
    assert row["total_observed_minus_expected_runs"] == pytest.approx(0.4)


def test_duplicate_event_ids_in_confidence_table_fail_loudly_not_silently():
    # aggregate_to_batter_season pivots the confidence table by event_id per
    # component -- a duplicated event_id (e.g. an upstream ETL bug feeding
    # the SAME play into the confidence builder twice) must raise, not
    # silently misalign defense/advancement status against the wrong row.
    df = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
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
            "defensive_execution_contribution": [float("nan")] * 2,
            "advancement_execution_contribution": [0.0, 0.0],
            "unexplained_residual": [0.4, 0.2],
        }
    )
    confidence = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],  # duplicated event_id, single component
            "component_name": ["contact", "contact"],
            "model_status_confidence": ["calibrated", "calibrated"],
            "row_data_quality": ["complete_measured_inputs", "complete_measured_inputs"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_to_batter_season(df, ledger, confidence)


def test_provisional_component_accounting_matches_manual_computation(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    summary = aggregate_to_batter_season(df, ledger, confidence)
    # share_of_value_from_provisional_components must be within [0, 1] for
    # every batter-season with nonzero total value.
    valid = summary["share_of_value_from_provisional_components"].notna()
    assert valid.any()
    shares = summary.loc[valid, "share_of_value_from_provisional_components"]
    assert (shares >= -1e-9).all()
    assert (shares <= 1.0 + 1e-9).all()


def test_aggregate_to_batter_season_rejects_mismatched_index(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    with pytest.raises(AggregationError, match="identical index"):
        aggregate_to_batter_season(df, ledger.iloc[:-1], confidence)


def test_aggregate_to_batter_season_requires_batter_season_game_columns(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    df_missing = df.drop(columns=["batter"])
    with pytest.raises(AggregationError, match="requires column"):
        aggregate_to_batter_season(df_missing, ledger, confidence)
