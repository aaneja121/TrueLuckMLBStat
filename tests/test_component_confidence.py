"""Tests for the Contact Luck v0.11 confidence taxonomy (Phase 1-2).

No test touches the network -- uses the shared `v011_full_ledger` fixture
(`tests/conftest.py`), itself built entirely from `test_attribution_ledger`'s
synthetic fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.scoring.component_confidence import (
    COMPONENTS,
    CONFIDENCE_RECORD_COLUMNS,
    DATA_QUALITY_VALUES,
    DOMAIN_VALUES,
    MODEL_STATUS_CALIBRATED,
    MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE,
    MODEL_STATUS_NOT_CALIBRATED,
    MODEL_STATUS_PROVISIONAL,
    MODEL_STATUS_UNAVAILABLE,
    MODEL_STATUS_VALUES,
    SUPPORT_VALUES,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    TIER_UNAVAILABLE,
    ComponentConfidenceError,
    build_play_level_confidence,
    derive_confidence_tier,
    normalize_model_status,
)

# ---------------------------------------------------------------------------
# normalize_model_status / derive_confidence_tier: pure unit tests
# ---------------------------------------------------------------------------


def test_normalize_model_status_maps_known_values():
    assert normalize_model_status("calibrated") == MODEL_STATUS_CALIBRATED
    assert (
        normalize_model_status("calibrated_with_limited_subgroup_evidence (see x.json)")
        == MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE
    )
    assert normalize_model_status("not_calibrated") == MODEL_STATUS_NOT_CALIBRATED
    assert normalize_model_status("provisional") == MODEL_STATUS_PROVISIONAL


def test_normalize_model_status_maps_true_false_strings():
    # mlb_luck_score.scoring.run_attribution_ledger._read_existing_gate_status
    # cites compare_opportunity_models.py's raw "passes_basic_validation"
    # boolean, stringified -- normalize_model_status must handle both.
    assert normalize_model_status("True (see opportunity_model_comparison_detail.json)") == (
        MODEL_STATUS_CALIBRATED
    )
    assert normalize_model_status("False (see opportunity_model_comparison_detail.json)") == (
        MODEL_STATUS_NOT_CALIBRATED
    )


def test_normalize_model_status_never_guesses_calibrated_for_unknown_input():
    assert normalize_model_status(None) == MODEL_STATUS_UNAVAILABLE
    assert normalize_model_status("not_available_missing_file:x.json") == MODEL_STATUS_UNAVAILABLE
    assert normalize_model_status("garbage") == MODEL_STATUS_UNAVAILABLE


def test_derive_confidence_tier_unavailable_model_status_is_always_unavailable_tier():
    assert derive_confidence_tier(MODEL_STATUS_UNAVAILABLE, "strong") == TIER_UNAVAILABLE
    assert derive_confidence_tier(MODEL_STATUS_UNAVAILABLE, "insufficient") == TIER_UNAVAILABLE


def test_derive_confidence_tier_not_calibrated_is_always_low():
    assert derive_confidence_tier(MODEL_STATUS_NOT_CALIBRATED, "strong") == TIER_LOW


def test_derive_confidence_tier_insufficient_support_caps_at_low_even_if_calibrated():
    # An uncertain subgroup must NEVER inherit "high" just because the
    # model's aggregate metrics are calibrated -- the exact rule the task
    # asked this taxonomy to enforce.
    assert derive_confidence_tier(MODEL_STATUS_CALIBRATED, "insufficient") == TIER_LOW


def test_derive_confidence_tier_calibrated_and_strong_support_is_high():
    assert derive_confidence_tier(MODEL_STATUS_CALIBRATED, "strong") == TIER_HIGH


def test_derive_confidence_tier_limited_evidence_is_medium_not_high():
    assert (
        derive_confidence_tier(MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE, "limited")
        == TIER_MEDIUM
    )


# ---------------------------------------------------------------------------
# build_play_level_confidence: structural + integration tests
# ---------------------------------------------------------------------------


def test_confidence_table_has_expected_columns(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    assert list(confidence.columns) == list(CONFIDENCE_RECORD_COLUMNS)


def test_confidence_table_row_count_is_events_times_components(v011_full_ledger):
    df, ledger, confidence = v011_full_ledger
    assert len(confidence) == len(df) * len(COMPONENTS)
    assert set(confidence["component_name"].unique()) == set(COMPONENTS)


def test_confidence_table_every_value_is_in_its_own_taxonomy(v011_full_ledger):
    _, _, confidence = v011_full_ledger
    assert set(confidence["model_status_confidence"].unique()) <= set(MODEL_STATUS_VALUES)
    assert set(confidence["row_data_quality"].unique()) <= set(DATA_QUALITY_VALUES)
    assert set(confidence["statistical_support"].unique()) <= set(SUPPORT_VALUES)
    assert set(confidence["domain_status"].unique()) <= set(DOMAIN_VALUES)


def test_ineligible_rows_get_unavailable_status_not_calibrated(v011_full_ledger):
    df, _, confidence = v011_full_ledger
    ineligible_outfield_event_ids = set(
        df.loc[~df["outfield_opportunity_eligible"].astype(bool), "event_id"]
    )
    rows = confidence[
        (confidence["component_name"] == "outfield_defense")
        & confidence["event_id"].isin(ineligible_outfield_event_ids)
    ]
    assert len(rows) > 0
    assert (rows["model_status_confidence"] == MODEL_STATUS_UNAVAILABLE).all()
    assert (~rows["eligible"]).all()
    assert (~rows["prediction_available"]).all()


def test_reason_codes_are_never_empty_for_ineligible_rows(v011_full_ledger):
    _, _, confidence = v011_full_ledger
    ineligible = confidence[~confidence["eligible"]]
    assert len(ineligible) > 0
    assert ineligible["reason_codes"].apply(lambda codes: len(codes) > 0).all()


def test_confidence_propagates_supplied_model_status_verbatim_via_normalization(v011_full_ledger):
    _, _, confidence = v011_full_ledger
    infield_rows = confidence[
        (confidence["component_name"] == "infield_defense") & confidence["eligible"]
    ]
    assert len(infield_rows) > 0
    assert (
        infield_rows["model_status_confidence"] == MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE
    ).all()


def test_build_play_level_confidence_rejects_mismatched_index():
    df = pd.DataFrame({"event_id": ["a", "b"]})
    ledger = pd.DataFrame({"observed_contact_result_run_value": [1.0]}, index=[0])
    with pytest.raises(ComponentConfidenceError, match="identical index"):
        build_play_level_confidence(df, ledger)


def test_build_play_level_confidence_requires_event_id_column():
    df = pd.DataFrame({"x": [1, 2]})
    ledger = pd.DataFrame({"observed_contact_result_run_value": [1.0, 2.0]})
    with pytest.raises(ComponentConfidenceError, match="event_id"):
        build_play_level_confidence(df, ledger)
