"""Tests for the Version 1.4.0 play-ledger exporter (Phase 3: pure projection).

Uses a LOCAL, self-contained synthetic fixture (not `tests/conftest.py`'s
shared `v011_full_ledger`, so nothing here can affect any other test file)
that mirrors that fixture's own construction exactly -- a real, frozen-
architecture contact model trained on synthetic data via `mlb_luck_score.
models.train_contact_model.train_model`, then `build_attribution_ledger`
called for real. Since Phase 3, `build_attribution_ledger` itself returns
the contact model's native `p_out`..`p_home_run` columns (see `attribution_
ledger.py`'s module docstring), so this fixture no longer needs a second
`predict_proba_ordered` call the way Phase 2's did.

No network access. No real Statcast data. No 2025/2026 data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import compute_eligibility
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger
from mlb_luck_score.scoring.play_ledger_export import PlayLedgerExportError, build_play_ledger
from mlb_luck_score.scoring.play_ledger_schema import (
    PLAY_LEDGER_COLUMNS,
    PlayLedgerValidationError,
    resolved_rows,
    validate_play_ledger,
)
from test_attribution_ledger import _synthetic_ground_ball_df, _synthetic_outfield_df


@pytest.fixture(scope="module")
def synthetic_ledger() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(df, ledger)` -- df includes at least one genuine unresolved
    (`field_error`) row, mirroring `tests/conftest.py`'s shared `v011_full_
    ledger` fixture. `ledger` already carries native `p_*` columns (Phase
    3), so no second probability-prediction call is needed here.
    """
    outfield_df = _synthetic_outfield_df()
    ground_df = _synthetic_ground_ball_df()
    df = pd.concat([outfield_df, ground_df], ignore_index=True)

    rng = np.random.default_rng(0)
    df["batter"] = rng.integers(1000, 1010, size=len(df))
    df["game_pk"] = rng.integers(700000, 700010, size=len(df))
    df["at_bat_number"] = np.arange(1, len(df) + 1)
    df["pitch_number"] = 1
    df["game_date"] = pd.Timestamp("2024-04-01") + pd.to_timedelta(
        rng.integers(0, 175, size=len(df)), unit="D"
    )
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    df["season"] = 2024

    df = compute_eligibility(df)
    assert (df["outcome_class"].isna() & df["is_eligible"]).any(), (
        "fixture must contain at least one genuine eligible-but-unresolved row "
        "(e.g. field_error) to exercise the nullable RESULT_LINKED_COLUMNS contract"
    )

    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)

    ledger = build_attribution_ledger(df, contact_trained)

    from mlb_luck_score.data.clean_batted_balls import build_event_id

    df["event_id"] = build_event_id(df)

    return df, ledger


def test_build_play_ledger_passes_schema_validation(synthetic_ledger):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    validate_play_ledger(result)  # no raise
    assert list(result.columns) == list(PLAY_LEDGER_COLUMNS)
    assert len(result) == len(df)


def test_play_id_matches_event_id_and_is_unique(synthetic_ledger):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    assert result["play_id"].tolist() == df["event_id"].tolist()
    assert result["play_id"].is_unique


def test_at_bat_number_and_pitch_number_copied_verbatim(synthetic_ledger):
    """Phase 2.5 amendment: at_bat_number/pitch_number are native identity
    columns, copied from `df` bit-for-bit (not re-derived by parsing
    `play_id`)."""
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    pd.testing.assert_series_equal(
        result["at_bat_number"].astype("int64"),
        df["at_bat_number"].astype("int64"),
        check_names=False,
        check_index=False,
    )
    pd.testing.assert_series_equal(
        result["pitch_number"].astype("int64"),
        df["pitch_number"].astype("int64"),
        check_names=False,
        check_index=False,
    )


def test_unresolved_rows_have_the_full_scoring_group_null(synthetic_ledger):
    """Empirically (see play_ledger_schema.py's module docstring), the
    ledger nulls probabilities/expected_run_value TOGETHER with the result
    fields for an unresolved row -- not just outcome_class/observed_run_
    value/contact_luck_runs. Only the CONTACT fields (sourced from `df`,
    never touched by the ledger's nulling step) remain populated.
    """
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    unresolved = result["outcome_class"].isna()
    assert unresolved.any(), "fixture must exercise the unresolved-row path"
    assert result.loc[unresolved, "observed_run_value"].isna().all()
    assert result.loc[unresolved, "contact_luck_runs"].isna().all()
    assert result.loc[unresolved, "expected_run_value"].isna().all()
    assert result.loc[unresolved, "p_out"].isna().all()
    # CONTACT fields come from `df` directly, never nulled by the ledger.
    assert result.loc[unresolved, "launch_speed"].notna().all()
    assert result.loc[unresolved, "bb_type"].notna().all()


def test_resolved_rows_have_all_three_result_linked_columns_non_null(synthetic_ledger):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    resolved = result["outcome_class"].notna()
    assert resolved.any()
    assert result.loc[resolved, "observed_run_value"].notna().all()
    assert result.loc[resolved, "contact_luck_runs"].notna().all()


def test_resolved_rows_subset_matches_resolved_rows_helper(synthetic_ledger):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    via_helper = resolved_rows(result)
    via_mask = result[result["outcome_class"].notna()]
    assert len(via_helper) == len(via_mask)
    assert via_helper["play_id"].tolist() == via_mask["play_id"].tolist()


def test_exporter_copies_ledger_values_verbatim_bit_for_bit(synthetic_ledger):
    """The exporter must COPY the ledger's own already-scored values, never
    recompute them -- assert bit-for-bit equality (not merely
    `pytest.approx`) between the exported columns and their ledger source
    columns for every resolved row, proving direct passthrough (pure
    projection, Phase 3) rather than any independent formula call.
    """
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    resolved = ledger["observed_contact_result_run_value"].notna()

    pd.testing.assert_series_equal(
        result.loc[resolved, "observed_run_value"].astype("float64"),
        ledger.loc[resolved, "observed_contact_result_run_value"].astype("float64"),
        check_names=False,
        check_index=False,
    )
    pd.testing.assert_series_equal(
        result.loc[resolved, "expected_run_value"].astype("float64"),
        ledger.loc[resolved, "baseline_expected_contact_run_value"].astype("float64"),
        check_names=False,
        check_index=False,
    )
    pd.testing.assert_series_equal(
        result.loc[resolved, "contact_luck_runs"].astype("float64"),
        ledger.loc[resolved, "contact_result_surprise"].astype("float64"),
        check_names=False,
        check_index=False,
    )
    for cls in ("out", "single", "double", "triple", "home_run"):
        pd.testing.assert_series_equal(
            result.loc[resolved, f"p_{cls}"].astype("float64"),
            ledger.loc[resolved, f"p_{cls}"].astype("float64"),
            check_names=False,
            check_index=False,
        )


def test_contact_luck_reconciles_to_observed_minus_expected_for_every_resolved_row(
    synthetic_ledger,
):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    resolved = result["outcome_class"].notna()
    sub = result.loc[resolved]
    reconstructed = sub["observed_run_value"].astype("float64") - sub["expected_run_value"].astype(
        "float64"
    )
    np.testing.assert_allclose(
        sub["contact_luck_runs"].astype("float64").to_numpy(), reconstructed.to_numpy(), atol=1e-9
    )


def test_play_ledger_version_is_not_in_exported_columns(synthetic_ledger):
    """`play_ledger_version` is FILE-level provenance now (see
    play_ledger_metadata.py) -- the exporter must never stamp it onto rows."""
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    assert "play_ledger_version" not in result.columns


def test_no_presentation_overlay_columns_in_exported_ledger(synthetic_ledger):
    """batter_name/batter_team/opponent_team are presentation overlays, not
    canonical scoring truth -- the exporter accepts no such parameters and
    produces no such columns (Phase 2 contract amendment, Section 3)."""
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    for col in ("batter_name", "batter_team", "opponent_team"):
        assert col not in result.columns


def test_build_play_ledger_no_longer_accepts_contact_proba_or_overlay_kwargs(synthetic_ledger):
    """Phase 3 removed the `contact_proba` parameter entirely -- it is a
    positional-only two-argument function now."""
    df, ledger = synthetic_ledger
    with pytest.raises(TypeError):
        build_play_ledger(df, ledger, ledger[["p_out"]])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        names = pd.Series("Test Player", index=df.index)
        build_play_ledger(df, ledger, batter_names=names)  # type: ignore[call-arg]


def test_is_scored_matches_resolution_on_exported_ledger(synthetic_ledger):
    df, ledger = synthetic_ledger
    result = build_play_ledger(df, ledger)
    resolved = result["outcome_class"].notna()
    assert resolved.any() and (~resolved).any(), (
        "fixture must contain both resolved and unresolved rows"
    )
    pd.testing.assert_series_equal(
        result["is_scored"].astype("boolean"),
        resolved.astype("boolean"),
        check_names=False,
    )


def test_misaligned_index_raises(synthetic_ledger):
    df, ledger = synthetic_ledger
    bad_ledger = ledger.reset_index(drop=True)
    bad_ledger.index = bad_ledger.index + 10_000
    with pytest.raises(PlayLedgerExportError, match="index"):
        build_play_ledger(df, bad_ledger)


def test_ledger_missing_probability_column_raises(synthetic_ledger):
    df, ledger = synthetic_ledger
    bad_ledger = ledger.drop(columns=["p_out"])
    with pytest.raises(PlayLedgerExportError, match="p_out"):
        build_play_ledger(df, bad_ledger)


def test_ledger_missing_observed_run_value_column_raises(synthetic_ledger):
    df, ledger = synthetic_ledger
    bad_ledger = ledger.drop(columns=["observed_contact_result_run_value"])
    with pytest.raises(PlayLedgerExportError, match="observed_contact_result_run_value"):
        build_play_ledger(df, bad_ledger)


def test_validate_play_ledger_rejects_partial_result_linked_resolution():
    base_row = {
        "play_id": "1-1-1",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "game_date": "2024-04-01",
        "season": 2024,
        "batter_id": 1,
        "stand": "R",
        "launch_speed": 90.0,
        "launch_angle": 20.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": 0.0,
        "hit_distance_sc": 300.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": -0.25,
        "p_out": 0.5,
        "p_single": 0.2,
        "p_double": 0.15,
        "p_triple": 0.05,
        "p_home_run": 0.1,
        # partial: everything else resolved, expected_run_value not. Not
        # outcome_class/observed_run_value (those have their own, earlier-
        # firing, dedicated equivalence checks -- see
        # tests/test_play_ledger_schema.py's
        # test_outcome_class_resolution_equivalence_violation_raises).
        "expected_run_value": pd.NA,
        "contact_luck_runs": -0.1,
    }
    df = pd.DataFrame([base_row])[list(PLAY_LEDGER_COLUMNS)]
    with pytest.raises(PlayLedgerValidationError, match="PARTIAL"):
        validate_play_ledger(df)


def test_validate_play_ledger_rejects_probabilities_not_summing_to_one():
    base_row = {
        "play_id": "1-1-1",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "game_date": "2024-04-01",
        "season": 2024,
        "batter_id": 1,
        "stand": "R",
        "launch_speed": 90.0,
        "launch_angle": 20.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": 0.0,
        "hit_distance_sc": 300.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": -0.25,
        "p_out": 0.5,
        "p_single": 0.5,
        "p_double": 0.5,
        "p_triple": 0.0,
        "p_home_run": 0.0,
        "expected_run_value": 0.1,
        "contact_luck_runs": -0.35,
    }
    df = pd.DataFrame([base_row])[list(PLAY_LEDGER_COLUMNS)]
    with pytest.raises(PlayLedgerValidationError, match="sum"):
        validate_play_ledger(df)


def test_validate_play_ledger_rejects_contact_luck_identity_violation():
    base_row = {
        "play_id": "1-1-1",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "game_date": "2024-04-01",
        "season": 2024,
        "batter_id": 1,
        "stand": "R",
        "launch_speed": 90.0,
        "launch_angle": 20.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": 0.0,
        "hit_distance_sc": 300.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": -0.25,
        "p_out": 0.5,
        "p_single": 0.2,
        "p_double": 0.15,
        "p_triple": 0.05,
        "p_home_run": 0.1,
        "expected_run_value": 0.1,
        "contact_luck_runs": 999.0,  # deliberately wrong
    }
    df = pd.DataFrame([base_row])[list(PLAY_LEDGER_COLUMNS)]
    with pytest.raises(PlayLedgerValidationError, match="contact_luck_runs"):
        validate_play_ledger(df)
