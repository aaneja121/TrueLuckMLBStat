"""Contact Luck v1.0, Task 33 categories 7+8: schema drift and system guards.

Category 7 targets `run_v1_final_evaluation.assert_raw_statcast_schema_
compatible`. Category 8 targets `evaluation.v1_system_evaluation`'s
predeclared system-level checks, reusing the existing `v011_full_ledger`/
`v012_public_score_artifacts` synthetic fixtures (already validated,
correct data) as the "known-good" case and deliberately-corrupted copies as
the failure case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import run_v1_final_evaluation as rv1
import v1_system_evaluation as sysval

# ---------------------------------------------------------------------------
# Category 7: schema drift
# ---------------------------------------------------------------------------


def _minimal_valid_raw_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": np.arange(1, n + 1),
            "game_date": ["2025-03-18"] * n,
            "at_bat_number": np.arange(1, n + 1),
            "pitch_number": [1] * n,
            "batter": np.arange(100, 100 + n),
            "events": ["single"] * n,
            "launch_speed": [90.0] * n,
            "launch_angle": [12.0] * n,
            "hit_distance_sc": [180.0] * n,
            "hc_x": [130.0] * n,
            "hc_y": [150.0] * n,
        }
    )


def test_schema_check_passes_on_a_well_formed_raw_frame() -> None:
    result = rv1.assert_raw_statcast_schema_compatible(_minimal_valid_raw_df())
    assert result["missing_required_columns"] == []
    assert result["incompatible_numeric_columns"] == []


def test_schema_check_reports_extra_columns_without_failing() -> None:
    df = _minimal_valid_raw_df()
    df["some_new_savant_field"] = "harmless"
    result = rv1.assert_raw_statcast_schema_compatible(df)
    assert "some_new_savant_field" in result["extra_columns_present"]


def test_schema_check_fails_on_missing_required_identifier_column() -> None:
    df = _minimal_valid_raw_df().drop(columns=["game_pk"])
    with pytest.raises(rv1.SchemaDriftError, match="identifiers"):
        rv1.assert_raw_statcast_schema_compatible(df)


def test_schema_check_fails_on_renamed_essential_numeric_column() -> None:
    """A rename looks exactly like a missing column -- `launch_speed` simply
    isn't present under the name every downstream feature expects.
    """
    df = _minimal_valid_raw_df().rename(columns={"launch_speed": "launchSpeed"})
    with pytest.raises(rv1.SchemaDriftError, match="renamed"):
        rv1.assert_raw_statcast_schema_compatible(df)


def test_schema_check_fails_on_type_incompatible_numeric_column() -> None:
    df = _minimal_valid_raw_df()
    df["launch_speed"] = ["89.5 mph"] * len(df)
    with pytest.raises(rv1.SchemaDriftError, match="non-numeric"):
        rv1.assert_raw_statcast_schema_compatible(df)


def test_type_incompatibility_message_differs_from_missingness_message() -> None:
    missing_df = _minimal_valid_raw_df().drop(columns=["game_pk"])
    incompatible_df = _minimal_valid_raw_df()
    incompatible_df["launch_speed"] = ["nope"] * len(incompatible_df)

    with pytest.raises(rv1.SchemaDriftError) as missing_exc:
        rv1.assert_raw_statcast_schema_compatible(missing_df)
    with pytest.raises(rv1.SchemaDriftError) as incompatible_exc:
        rv1.assert_raw_statcast_schema_compatible(incompatible_df)

    assert str(missing_exc.value) != str(incompatible_exc.value)
    assert "non-numeric" not in str(missing_exc.value)
    assert "non-numeric" in str(incompatible_exc.value)


def test_type_incompatible_column_is_never_silently_coerced() -> None:
    """The original, uncoerced dataframe is what the caller keeps -- this
    function must never mutate its input to "fix" the problem.
    """
    df = _minimal_valid_raw_df()
    df["launch_speed"] = ["89.5 mph"] * len(df)
    original_values = df["launch_speed"].tolist()
    with pytest.raises(rv1.SchemaDriftError):
        rv1.assert_raw_statcast_schema_compatible(df)
    assert df["launch_speed"].tolist() == original_values


def test_missing_optional_non_essential_column_does_not_fail() -> None:
    """`stand`/`bb_type` etc. are optional and NOT in `EXPECTED_NUMERIC_RAW_
    COLUMNS` -- their absence is not schema drift by this check's own scope
    (`clean_batted_balls.ensure_optional_columns` backfills them safely).
    """
    df = _minimal_valid_raw_df()
    assert "stand" not in df.columns
    result = rv1.assert_raw_statcast_schema_compatible(df)
    assert result["missing_required_columns"] == []


# ---------------------------------------------------------------------------
# Category 8: system guards
# ---------------------------------------------------------------------------


def test_verify_event_ids_passes_on_valid_ledger(v011_full_ledger) -> None:
    df, ledger, _confidence = v011_full_ledger
    result = sysval.verify_event_ids(df)
    assert result["unique_non_null_event_ids"] is True


def test_verify_event_ids_fails_on_duplicate_event_id(v011_full_ledger) -> None:
    df, _ledger, _confidence = v011_full_ledger
    corrupted = df.copy()
    corrupted.loc[corrupted.index[1], "event_id"] = corrupted.loc[corrupted.index[0], "event_id"]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        sysval.verify_event_ids(corrupted)


def test_verify_event_ids_fails_on_null_event_id(v011_full_ledger) -> None:
    df, _ledger, _confidence = v011_full_ledger
    corrupted = df.copy()
    corrupted.loc[corrupted.index[0], "event_id"] = None
    with pytest.raises(ValueError):
        sysval.verify_event_ids(corrupted)


def test_verify_routing_exclusivity_passes_on_valid_data(v011_full_ledger) -> None:
    df, _ledger, _confidence = v011_full_ledger
    result = sysval.verify_routing_exclusivity(df)
    assert result["mutually_exclusive"] is True


def test_verify_routing_exclusivity_fails_when_both_domains_eligible(v011_full_ledger) -> None:
    df, _ledger, _confidence = v011_full_ledger
    corrupted = df.copy()
    corrupted["outfield_opportunity_eligible"] = True
    corrupted["infield_opportunity_eligible"] = True
    with pytest.raises(sysval.SystemEvaluationError, match="both outfield and infield"):
        sysval.verify_routing_exclusivity(corrupted)


def test_verify_no_experimental_components_passes_on_frozen_ledger(v011_full_ledger) -> None:
    _df, ledger, _confidence = v011_full_ledger
    result = sysval.verify_no_experimental_components_in_official_total(ledger)
    assert result["no_weather_or_alignment_columns_in_official_ledger"] is True


def test_verify_no_experimental_components_fails_on_weather_column(v011_full_ledger) -> None:
    _df, ledger, _confidence = v011_full_ledger
    corrupted = ledger.copy()
    corrupted["weather_run_value_effect"] = 0.0
    with pytest.raises(sysval.SystemEvaluationError, match="[Ee]xperimental"):
        sysval.verify_no_experimental_components_in_official_total(corrupted)


def test_verify_no_experimental_components_fails_on_alignment_column(v011_full_ledger) -> None:
    _df, ledger, _confidence = v011_full_ledger
    corrupted = ledger.copy()
    corrupted["alignment_interaction_effect"] = 0.0
    with pytest.raises(sysval.SystemEvaluationError):
        sysval.verify_no_experimental_components_in_official_total(corrupted)


def test_verify_play_level_accounting_passes_on_valid_ledger(v011_full_ledger) -> None:
    _df, ledger, _confidence = v011_full_ledger
    result = sysval.verify_play_level_accounting(ledger)
    assert result["identity_holds_for_every_resolved_row"] is True
    assert result["abs_reconciliation_error"]["max"] < 1e-6


def test_verify_play_level_accounting_fails_when_identity_broken(v011_full_ledger) -> None:
    _df, ledger, _confidence = v011_full_ledger
    corrupted = ledger.copy()
    corrupted["unexplained_residual"] = corrupted["unexplained_residual"] + 100.0
    with pytest.raises(sysval.SystemEvaluationError, match="identity failed"):
        sysval.verify_play_level_accounting(corrupted)


def test_verify_season_level_accounting_passes_on_valid_summary(
    v012_public_score_artifacts,
) -> None:
    artifacts = v012_public_score_artifacts
    result = sysval.verify_season_level_accounting(artifacts.player_season)
    assert result["identity_holds_for_every_row"] is True


def test_verify_season_level_accounting_fails_when_identity_broken(
    v012_public_score_artifacts,
) -> None:
    artifacts = v012_public_score_artifacts
    corrupted = artifacts.player_season.copy()
    corrupted["total_unexplained_residual_component_runs"] = (
        corrupted["total_unexplained_residual_component_runs"] + 100.0
    )
    with pytest.raises(sysval.SystemEvaluationError, match="identity failed"):
        sysval.verify_season_level_accounting(corrupted)


def test_verify_reproducibility_passes_for_a_deterministic_function() -> None:
    def _deterministic(n: int) -> pd.DataFrame:
        return pd.DataFrame({"x": range(n)})

    result = sysval.verify_reproducibility(_deterministic, 5)
    assert result["deterministic"] is True


def test_verify_reproducibility_fails_for_a_nondeterministic_function() -> None:
    rng = np.random.default_rng()

    def _nondeterministic(n: int) -> pd.DataFrame:
        return pd.DataFrame({"x": rng.random(n)})

    with pytest.raises(sysval.SystemEvaluationError, match="identical"):
        sysval.verify_reproducibility(_nondeterministic, 5)


def test_qualification_and_ranking_contract_passes_on_valid_table(
    v012_public_score_artifacts,
) -> None:
    from mlb_luck_score.scoring.leaderboard import assign_official_ranks
    from mlb_luck_score.scoring.public_score_table import build_public_score_table

    table = build_public_score_table(v012_public_score_artifacts)
    table = assign_official_ranks(table)
    result = sysval.verify_qualification_and_ranking_contract(table)
    assert result["schema_valid"] is True
    assert result["no_rank_on_non_qualified_rows"] is True
    assert result["point_estimates_always_paired_with_intervals"] is True


def test_non_qualified_row_with_a_rank_fails_the_contract(v012_public_score_artifacts) -> None:
    """A non-qualified row can never carry an official rank -- enforced in
    LAYERS: `mlb_luck_score.scoring.public_score_schema.
    validate_public_score_table` (called first, inside `verify_
    qualification_and_ranking_contract`) already strictly ties `official_
    rank_eligible` to `qualification_status == "qualified"`, so a corrupted
    row that merely sets a rank value is caught there, one layer before
    this function's own redundant `non_qualified_ranked` check could ever
    run. Both layers exist; this test confirms the outer one (whichever
    fires first) actually rejects the row -- see `PublicScoreSchemaError`.
    """
    from mlb_luck_score.scoring.leaderboard import assign_official_ranks
    from mlb_luck_score.scoring.public_score_schema import PublicScoreSchemaError
    from mlb_luck_score.scoring.public_score_table import build_public_score_table
    from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED

    table = build_public_score_table(v012_public_score_artifacts)
    table = assign_official_ranks(table)
    non_qualified = table[table["qualification_status"] != STATUS_QUALIFIED]
    if non_qualified.empty:
        pytest.skip("synthetic fixture has no non-qualified rows to corrupt")
    corrupted = table.copy()
    corrupted.loc[non_qualified.index[0], "official_rank_favorable"] = 1
    with pytest.raises(PublicScoreSchemaError, match="official_rank_favorable"):
        sysval.verify_qualification_and_ranking_contract(corrupted)


def test_point_estimate_without_interval_fails_the_contract(v012_public_score_artifacts) -> None:
    from mlb_luck_score.scoring.leaderboard import assign_official_ranks
    from mlb_luck_score.scoring.public_score_table import build_public_score_table

    table = build_public_score_table(v012_public_score_artifacts)
    table = assign_official_ranks(table)
    has_estimate = table[table["contact_luck_runs_per_100"].notna()]
    if has_estimate.empty:
        pytest.skip("synthetic fixture has no rows with a point estimate to corrupt")
    corrupted = table.copy()
    corrupted.loc[has_estimate.index[0], "lower_95_interval"] = None
    corrupted.loc[has_estimate.index[0], "upper_95_interval"] = None
    with pytest.raises(sysval.SystemEvaluationError, match="interval"):
        sysval.verify_qualification_and_ranking_contract(corrupted)
