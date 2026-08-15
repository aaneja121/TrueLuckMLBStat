"""Unit tests for the Version 1.4.0 play-ledger schema (`play_ledger_schema.py`).

Pure hand-built DataFrames -- no model training, no real data, no network.
`tests/test_play_ledger_export.py` covers the exporter's projection logic
against a real (synthetic-data) trained model; this file covers only the
schema contract itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mlb_luck_score.scoring.play_ledger_schema import (
    PLAY_LEDGER_COLUMNS,
    PLAY_LEDGER_NULLABLE_COLUMNS,
    PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS,
    PROBABILITY_COLUMNS,
    RESULT_LINKED_COLUMNS,
    PlayLedgerValidationError,
    resolved_rows,
    validate_play_ledger,
)


def _resolved_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "at_bat_number": 10,
        "pitch_number": 3,
        "game_date": "2024-06-01",
        "season": 2024,
        "batter_id": 12345,
        "stand": "L",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "is_scored": True,
        # Full-precision values from the reviewed dashboard/demo_fixture.json
        # (Riley Greene, hard_contact_out) -- internally consistent (unlike
        # rounded placeholder probabilities would be) so the identity/cross-
        # check validations in validate_play_ledger pass exactly.
        "outcome_class": "out",
        "observed_run_value": -0.25491574127584554,
        "p_out": 0.05692444081479338,
        "p_single": 0.0020191686532899677,
        "p_double": 0.019210357764228986,
        "p_triple": 0.003476039197145578,
        "p_home_run": 0.9183699935705423,
        "expected_run_value": 1.290656107968476,
    }
    row["contact_luck_runs"] = row["observed_run_value"] - row["expected_run_value"]
    row.update(overrides)
    return row


def _unresolved_row(**overrides: Any) -> dict[str, Any]:
    # Empirically, `attribution_ledger.build_attribution_ledger` nulls the
    # ENTIRE ledger-derived row (probabilities + expected_run_value too, not
    # just outcome_class/observed_run_value/contact_luck_runs) for an
    # eligible-but-unresolved play -- see play_ledger_schema.py's module
    # docstring on `PLAY_LEDGER_NULLABLE_COLUMNS`.
    row = _resolved_row(
        play_id="700002-20-1",
        at_bat_number=20,
        pitch_number=1,
        is_scored=False,
        outcome_class=pd.NA,
        observed_run_value=pd.NA,
        contact_luck_runs=pd.NA,
        p_out=pd.NA,
        p_single=pd.NA,
        p_double=pd.NA,
        p_triple=pd.NA,
        p_home_run=pd.NA,
        expected_run_value=pd.NA,
    )
    row.update(overrides)
    return row


def _df(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[list(PLAY_LEDGER_COLUMNS)]


def test_column_set_covers_every_required_group():
    for col in (
        "play_id",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "game_date",
        "season",
        "batter_id",
        "stand",
        "launch_speed",
        "launch_angle",
        "bb_type",
        "spray_angle_approx",
        "hit_distance_sc",
        "is_scored",
        "outcome_class",
        "observed_run_value",
        "expected_run_value",
        "contact_luck_runs",
    ):
        assert col in PLAY_LEDGER_COLUMNS
    for col in PROBABILITY_COLUMNS:
        assert col in PLAY_LEDGER_COLUMNS


def test_presentation_overlay_fields_not_in_canonical_schema():
    """batter_name/batter_team/opponent_team are presentation overlays applied
    AFTER scoring (Phase 2 contract amendment, Section 3) -- they must never
    be required canonical scoring fields."""
    for col in ("batter_name", "batter_team", "opponent_team"):
        assert col not in PLAY_LEDGER_COLUMNS


def test_play_ledger_version_is_not_a_row_column():
    """`play_ledger_version` is FILE-level provenance now (see
    play_ledger_metadata.py), never repeated on every row."""
    assert "play_ledger_version" not in PLAY_LEDGER_COLUMNS


def test_no_favorable_boolean_field_in_schema():
    """Per the explicit Phase 1/2 decision: display sign must be derived
    from `contact_luck_runs`, never stored as a separate boolean."""
    assert "favorable" not in PLAY_LEDGER_COLUMNS


def test_no_component_attribution_fields_in_v1_4_0_schema():
    for col in (
        "contact_component",
        "defensive_execution_component",
        "advancement_execution_component",
        "unexplained_residual_component",
        "component_eligibility_status",
        "component_confidence_status",
    ):
        assert col not in PLAY_LEDGER_COLUMNS


def test_no_raw_description_text_in_schema():
    assert "des" not in PLAY_LEDGER_COLUMNS
    assert "description" not in PLAY_LEDGER_COLUMNS


def test_result_linked_columns_cover_probabilities_expectation_and_result():
    # Empirically wider than just outcome_class/observed_run_value/
    # contact_luck_runs -- see play_ledger_schema.py's module docstring.
    expected = {
        "p_out",
        "p_single",
        "p_double",
        "p_triple",
        "p_home_run",
        "expected_run_value",
        "outcome_class",
        "observed_run_value",
        "contact_luck_runs",
    }
    assert set(RESULT_LINKED_COLUMNS) == expected
    assert set(RESULT_LINKED_COLUMNS).issubset(PLAY_LEDGER_NULLABLE_COLUMNS)


def test_identity_columns_are_never_nullable():
    for col in (
        "play_id",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "batter_id",
        "season",
        "game_date",
    ):
        assert col in PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS


def test_exact_column_count_and_order():
    """Phase 2.5 amendment: `at_bat_number`/`pitch_number` were added as
    native identity fields (previously only derivable by parsing `play_id`),
    bringing the canonical schema to 23 columns."""
    assert list(PLAY_LEDGER_COLUMNS) == [
        "play_id",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "game_date",
        "season",
        "batter_id",
        "stand",
        "launch_speed",
        "launch_angle",
        "bb_type",
        "spray_angle_approx",
        "hit_distance_sc",
        "is_scored",
        "outcome_class",
        "observed_run_value",
        "p_out",
        "p_single",
        "p_double",
        "p_triple",
        "p_home_run",
        "expected_run_value",
        "contact_luck_runs",
    ]
    assert len(PLAY_LEDGER_COLUMNS) == 23


def test_stand_is_the_one_always_present_contact_field():
    """Empirically (real 2024 development data), `launch_speed`/`launch_
    angle`/`bb_type`/`spray_angle_approx`/`hit_distance_sc` can each be
    independently missing on an `is_eligible=True` row (`missing_contact_
    data`) -- only `stand` was observed always present. See
    `PLAY_LEDGER_NULLABLE_COLUMNS`'s docstring."""
    assert "stand" in PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS
    for col in ("launch_speed", "launch_angle", "bb_type", "spray_angle_approx", "hit_distance_sc"):
        assert col in PLAY_LEDGER_NULLABLE_COLUMNS


def test_valid_resolved_row_passes():
    df = _df(_resolved_row())
    validate_play_ledger(df)  # no raise


def test_valid_unresolved_row_passes():
    df = _df(_unresolved_row())
    validate_play_ledger(df)  # no raise


def test_mixed_resolved_and_unresolved_rows_pass():
    df = _df(_resolved_row(), _unresolved_row())
    validate_play_ledger(df)  # no raise


def test_missing_column_raises():
    df = _df(_resolved_row()).drop(columns=["launch_speed"])
    with pytest.raises(PlayLedgerValidationError, match="missing column"):
        validate_play_ledger(df)


def test_null_play_id_raises():
    df = _df(_resolved_row(play_id=pd.NA))
    with pytest.raises(PlayLedgerValidationError, match="null play_id"):
        validate_play_ledger(df)


def test_duplicate_play_id_raises():
    df = _df(_resolved_row(), _resolved_row())
    with pytest.raises(PlayLedgerValidationError, match="duplicate play_id"):
        validate_play_ledger(df)


def test_null_required_non_nullable_column_raises():
    df = _df(_resolved_row(stand=pd.NA))
    with pytest.raises(PlayLedgerValidationError, match="stand"):
        validate_play_ledger(df)


def test_null_contact_fields_are_allowed_independent_of_resolution():
    """Missing raw contact measurement (`missing_contact_data`) is
    independent of outcome resolution -- a resolved row can still have a
    null `launch_speed`/`launch_angle`/etc."""
    df = _df(
        _resolved_row(
            launch_speed=pd.NA,
            launch_angle=pd.NA,
            bb_type=pd.NA,
            spray_angle_approx=pd.NA,
            hit_distance_sc=pd.NA,
        )
    )
    validate_play_ledger(df)  # no raise -- still fully resolved/scored


def test_is_scored_true_matches_resolved_row():
    df = _df(_resolved_row())
    validate_play_ledger(df)  # no raise
    assert bool(df["is_scored"].iloc[0]) is True


def test_is_scored_false_matches_unresolved_row():
    df = _df(_unresolved_row())
    validate_play_ledger(df)  # no raise
    assert bool(df["is_scored"].iloc[0]) is False


def test_is_scored_never_null():
    assert "is_scored" in PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS
    assert "is_scored" not in PLAY_LEDGER_NULLABLE_COLUMNS


def test_is_scored_mismatch_with_observed_run_value_raises():
    """is_scored is a pure derivation of observed_run_value.notna() (Phase
    2.5 amendment: the SAME column aggregate_to_batter_season's own
    denominator rule uses) -- it must never disagree with it, even if every
    other field is internally consistent."""
    df = _df(_resolved_row(is_scored=False))
    with pytest.raises(PlayLedgerValidationError, match="is_scored"):
        validate_play_ledger(df)

    df2 = _df(_unresolved_row(is_scored=True))
    with pytest.raises(PlayLedgerValidationError, match="is_scored"):
        validate_play_ledger(df2)


def test_observed_run_value_outcome_class_equivalence_invariant():
    """Explicit Phase 2.5 invariant: observed_run_value.notna() ==
    outcome_class.notna() == is_scored == "all scoring-linked fields
    populated", for the currently supported scoring contract."""
    df = _df(_resolved_row(), _unresolved_row())
    validate_play_ledger(df)  # no raise -- proves the equivalence holds

    observed_resolved = df["observed_run_value"].notna()
    outcome_resolved = df["outcome_class"].notna()
    is_scored = df["is_scored"].astype("boolean").fillna(False)
    all_scoring_fields_populated = df[list(RESULT_LINKED_COLUMNS)].notna().all(axis=1)

    assert (observed_resolved == outcome_resolved).all()
    assert (observed_resolved == is_scored).all()
    assert (observed_resolved == all_scoring_fields_populated).all()


def test_outcome_class_resolution_equivalence_violation_raises():
    """A row where outcome_class and observed_run_value disagree about
    resolution must be rejected even if is_scored happens to be internally
    consistent with one of them -- this is a SEPARATE, explicit check from
    the is_scored derivation check."""
    df = _df(
        _resolved_row(
            outcome_class=pd.NA,  # now disagrees with observed_run_value (still set)
            is_scored=True,  # kept consistent with observed_run_value.notna() (still True),
            # so the is_scored check passes and this test isolates the SEPARATE
            # outcome_class/observed_run_value equivalence check below.
        )
    )
    with pytest.raises(PlayLedgerValidationError, match="outcome_class.notna"):
        validate_play_ledger(df)


@pytest.mark.parametrize(
    "overrides",
    [
        # NOTE: overriding outcome_class or observed_run_value ALONE is
        # deliberately NOT parametrized here -- under the Phase 2.5
        # amendment those two fields each have their OWN, earlier-firing,
        # more specific check (is_scored derivation /
        # test_is_scored_mismatch_with_observed_run_value_raises, and
        # outcome_class/observed_run_value equivalence /
        # test_outcome_class_resolution_equivalence_violation_raises) that
        # catches the inconsistency before the generic PARTIAL check below
        # would. This parametrize covers the remaining RESULT_LINKED_COLUMNS
        # (contact_luck_runs / expected_run_value / a probability column)
        # that have no such dedicated check of their own.
        {"contact_luck_runs": pd.NA},  # the other 8 still set
        {"expected_run_value": pd.NA},  # the other 8 still set
        {"p_out": pd.NA},  # the other 8 still set
    ],
)
def test_partial_result_linked_resolution_raises(overrides):
    df = _df(_resolved_row(**overrides))
    with pytest.raises(PlayLedgerValidationError, match="PARTIAL"):
        validate_play_ledger(df)


def test_probability_out_of_range_raises():
    df = _df(_resolved_row(p_out=1.5))
    with pytest.raises(PlayLedgerValidationError, match=r"\[0, 1\]"):
        validate_play_ledger(df)


def test_probabilities_not_summing_to_one_raises():
    df = _df(_resolved_row(p_out=0.9, p_single=0.9, p_double=0.0, p_triple=0.0, p_home_run=0.0))
    with pytest.raises(PlayLedgerValidationError, match="sum"):
        validate_play_ledger(df)


def test_contact_luck_identity_violation_raises():
    df = _df(_resolved_row(contact_luck_runs=12345.0))
    with pytest.raises(PlayLedgerValidationError, match="contact_luck_runs"):
        validate_play_ledger(df)


def test_resolved_rows_helper_filters_correctly():
    df = _df(_resolved_row(), _unresolved_row())
    resolved = resolved_rows(df)
    assert len(resolved) == 1
    assert resolved["play_id"].iloc[0] == "700001-10-3"


def test_resolved_rows_helper_never_uses_len_of_full_df():
    """The published `eligible_batted_balls` denominator excludes unresolved
    rows -- `len(df)` alone would silently over-count it (see Phase 2's
    empirical finding). This test locks in that `resolved_rows` is the
    correct, and only correct, way to reproduce that count."""
    df = _df(_resolved_row(), _unresolved_row(), _unresolved_row(play_id="700003-1-1"))
    assert len(df) == 3
    assert len(resolved_rows(df)) == 1
