"""Tests for the Contact Luck v0.10 play-level attribution ledger.

No test touches the network -- synthetic fixtures only. The central claim
under test is the EXACT accounting identity:

    observed_final_run_value - baseline_expected_contact_run_value
        == unexplained_residual
           + defensive_execution_contribution
           + advancement_execution_contribution

across every realistic eligibility scenario: contact-only (no opportunity/
advancement model applies), outfield defense only, outfield defense +
advancement, infield defense only, and reached-on-error (contact result
unavailable, whole row excluded from the identity check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import (
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.features.build_contact_features import (
    INFIELD_OPPORTUNITY_TARGET_COLUMN,
    add_advancement_contact_probability_features,
    add_advancement_features,
    add_opportunity_features_by_domain,
    add_outfield_opportunity_features,
)
from mlb_luck_score.models.train_advancement_model import train_advancement_model
from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring.attribution_ledger import (
    FINAL_BASE_RUN_VALUE_MAP,
    STATUS_ADVANCEMENT_MODELED,
    STATUS_ADVANCEMENT_NOT_MODELED,
    STATUS_CONTACT_RESULT_UNAVAILABLE,
    STATUS_DEFENSE_INFIELD,
    STATUS_DEFENSE_OUTFIELD,
    STATUS_DEFENSE_UNAVAILABLE,
    AttributionLedgerError,
    assert_unique_play_identifiers,
    build_attribution_ledger,
    compute_expected_advancement_value_runs,
    compute_opportunity_adjusted_expected_run_value,
    verify_attribution_identity,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

# ---------------------------------------------------------------------------
# Pure-math unit tests
# ---------------------------------------------------------------------------


def test_final_base_run_value_map_reuses_default_run_value_map_exactly():
    assert FINAL_BASE_RUN_VALUE_MAP[0] == DEFAULT_RUN_VALUE_MAP["out"]
    assert FINAL_BASE_RUN_VALUE_MAP[1] == DEFAULT_RUN_VALUE_MAP["single"]
    assert FINAL_BASE_RUN_VALUE_MAP[2] == DEFAULT_RUN_VALUE_MAP["double"]
    assert FINAL_BASE_RUN_VALUE_MAP[3] == DEFAULT_RUN_VALUE_MAP["triple"]
    assert FINAL_BASE_RUN_VALUE_MAP[4] == DEFAULT_RUN_VALUE_MAP["home_run"]


def test_opportunity_adjusted_expected_run_value_certain_out():
    contact_proba = pd.DataFrame(
        {"out": [1.0], "single": [0.0], "double": [0.0], "triple": [0.0], "home_run": [0.0]}
    )
    p_out_opportunity = pd.Series([1.0])
    eo = compute_opportunity_adjusted_expected_run_value(contact_proba, p_out_opportunity)
    assert eo.iloc[0] == pytest.approx(DEFAULT_RUN_VALUE_MAP["out"])


def test_opportunity_adjusted_expected_run_value_certain_single_given_not_out():
    contact_proba = pd.DataFrame(
        {"out": [0.5], "single": [0.5], "double": [0.0], "triple": [0.0], "home_run": [0.0]}
    )
    p_out_opportunity = pd.Series([0.0])
    eo = compute_opportunity_adjusted_expected_run_value(contact_proba, p_out_opportunity)
    # p_out_opportunity=0 -> entirely the "not out" conditional expectation,
    # which here is 100% single (the only non-out mass).
    assert eo.iloc[0] == pytest.approx(DEFAULT_RUN_VALUE_MAP["single"])


def test_expected_advancement_value_runs_matches_final_base_map():
    proba = pd.DataFrame(
        {
            "held_at_first": [1.0],
            "advanced_to_second": [0.0],
            "advanced_to_third": [0.0],
            "inside_the_park_home_run": [0.0],
            "retired_while_advancing": [0.0],
        }
    )
    ea = compute_expected_advancement_value_runs(proba)
    assert ea.iloc[0] == pytest.approx(DEFAULT_RUN_VALUE_MAP["single"])


def test_expected_advancement_value_runs_rejects_wrong_columns():
    bad = pd.DataFrame({"a": [0.5], "b": [0.5]})
    with pytest.raises(AttributionLedgerError, match="Expected advancement_proba columns"):
        compute_expected_advancement_value_runs(bad)


# ---------------------------------------------------------------------------
# verify_attribution_identity: direct tests on hand-built ledgers
# ---------------------------------------------------------------------------


def _hand_ledger(**overrides) -> pd.DataFrame:
    base = {
        "baseline_expected_contact_run_value": [0.1],
        "observed_final_run_value": [0.5],
        "observed_contact_result_run_value": [0.5],
        "unexplained_residual": [0.0],
        "defensive_execution_contribution": [0.4],
        "advancement_execution_contribution": [0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_identity_holds_for_a_balanced_ledger():
    ledger = _hand_ledger()
    holds = verify_attribution_identity(ledger)
    assert bool(holds.iloc[0]) is True


def test_identity_fails_for_an_unbalanced_ledger():
    ledger = _hand_ledger(defensive_execution_contribution=[0.1])  # wrong on purpose
    holds = verify_attribution_identity(ledger)
    assert bool(holds.iloc[0]) is False


def test_identity_is_na_for_unresolved_contact_result():
    ledger = _hand_ledger(observed_contact_result_run_value=[float("nan")])
    holds = verify_attribution_identity(ledger)
    assert holds.iloc[0] is pd.NA


def test_identity_tolerance_is_tight():
    # A genuine logic bug (off by 0.01) must NOT slip through the default tolerance.
    ledger = _hand_ledger(unexplained_residual=[0.01])
    holds = verify_attribution_identity(ledger)
    assert bool(holds.iloc[0]) is False


def test_identity_treats_nan_contributions_as_not_applicable_zero():
    # defensive_execution_contribution NaN (no opportunity model available) --
    # unexplained_residual alone must absorb the full gap.
    ledger = _hand_ledger(
        defensive_execution_contribution=[float("nan")],
        unexplained_residual=[0.4],
    )
    holds = verify_attribution_identity(ledger)
    assert bool(holds.iloc[0]) is True


# ---------------------------------------------------------------------------
# End-to-end: build_attribution_ledger against real trained models
# ---------------------------------------------------------------------------


def _synthetic_outfield_df(n_per_class: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    profiles = {
        "out": (85.0, 25.0, 250.0, "field_out"),
        "single": (90.0, 8.0, 180.0, "single"),
        "double": (98.0, 18.0, 320.0, "double"),
        "triple": (100.0, 20.0, 370.0, "triple"),
        "home_run": (105.0, 28.0, 410.0, "home_run"),
    }
    des_map = {
        "field_out": "Someone flies out to center fielder Other Player.",
        "single": "Someone singles on a fly ball to center fielder Other Player.",
        "double": "Someone doubles (1) on a fly ball to center fielder Other Player.",
        "triple": "Someone triples (1) on a fly ball to center fielder Other Player.",
        "home_run": "Someone homers (1) on a fly ball to right field.",
    }
    for speed, angle, dist, events in profiles.values():
        for i in range(n_per_class):
            rows.append(
                {
                    "event_id": f"of-{events}-{i}",
                    "game_pk": 4000 + i,
                    "launch_speed": speed + rng.normal(0, 1.0),
                    "launch_angle": angle + rng.normal(0, 1.0),
                    "spray_angle_approx": rng.uniform(-15, 15),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "fly_ball",
                    "stand": "R",
                    "venue": "Synthetic Park",
                    "events": events,
                    "des": des_map[events],
                    "hit_location": 8,
                    "of_fielding_alignment": "Standard",
                    "fielder_7": 111,
                    "fielder_8": 222,
                    "fielder_9": 333,
                    "sprint_speed": 27.0 + rng.normal(0, 1.5),
                    "eligible_for_training": True,
                    "season": 2021 + (i % 4),
                }
            )
    return pd.DataFrame(rows)


def _synthetic_ground_ball_df(n_per_class: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    profiles = {
        "field_out": (90.0, -5.0, "field_out", 6, "Routine grounder to short."),
        "single": (85.0, 2.0, "single", 4, "Ground ball single up the middle."),
        "field_error": (88.0, 1.0, "field_error", 6, "Reaches on a throwing error."),
    }
    for speed, angle, events, hit_location, des in profiles.values():
        for i in range(n_per_class):
            rows.append(
                {
                    "event_id": f"if-{events}-{i}",
                    "game_pk": 5000 + i,
                    "launch_speed": speed + rng.normal(0, 1.0),
                    "launch_angle": angle + rng.normal(0, 1.0),
                    "spray_angle_approx": rng.uniform(-15, 15),
                    "hit_distance_sc": 60.0 + rng.normal(0, 5),
                    "bb_type": "ground_ball",
                    "stand": "R",
                    "venue": "Synthetic Park",
                    "hit_location": hit_location,
                    "events": events,
                    "des": des,
                    "pitcher": 1,
                    "fielder_2": 202,
                    "fielder_3": 303,
                    "fielder_4": 404,
                    "fielder_5": 505,
                    "fielder_6": 606,
                    "if_fielding_alignment": "Standard",
                    "surface_type": "Grass",
                    "sprint_speed": 27.0 + rng.normal(0, 1.5),
                    "outs_when_up": 1,
                    "on_1b": None,
                    "eligible_for_training": True,
                    "season": 2021 + (i % 4),
                }
            )
    return pd.DataFrame(rows)


ADVANCEMENT_NUMERIC = [
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "estimated_hang_time_s",
    "landing_x_ft",
    "landing_y_ft",
    "sprint_speed",
    "contact_p_out",
    "contact_p_single",
    "contact_p_double",
    "contact_p_triple",
    "contact_p_home_run",
]
ADVANCEMENT_CATEGORICAL = ["hit_type_group", "assigned_outfield_position", "stand"]


@pytest.fixture(scope="module")
def full_ledger():
    outfield_df = _synthetic_outfield_df()
    ground_df = _synthetic_ground_ball_df()
    df = pd.concat([outfield_df, ground_df], ignore_index=True)

    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)

    # Version 0.10 hardening: add_opportunity_features_by_domain routes each
    # row to at most one of add_outfield_opportunity_features/add_infield_
    # opportunity_features and reassembles in original order, rather than
    # hand-splitting eligible rows (the discipline that was skipped once
    # already -- see build_contact_features.INFIELD_OPPORTUNITY_TARGET_
    # COLUMN's docstring for the bug that caught).
    df = add_opportunity_features_by_domain(df)

    df = add_advancement_features(df)

    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)

    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba_all = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    df = add_advancement_contact_probability_features(df, contact_proba_all)

    outfield_elig = df[df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_elig, class_weight=None)

    infield_elig = df[df["infield_opportunity_eligible"].astype(bool)]
    infield_trained = train_opportunity_model(
        infield_elig,
        numeric_features=[
            "launch_speed",
            "launch_angle",
            "spray_angle_approx",
            "hit_distance_sc",
            "sprint_speed",
            "outs_when_up",
            "on_1b_occupied",
        ],
        categorical_features=[
            "stand",
            "if_fielding_alignment",
            "assigned_infield_position",
            "surface_type",
        ],
        class_weight=None,
        target_column=INFIELD_OPPORTUNITY_TARGET_COLUMN,
    )

    advancement_elig = df[df["advancement_eligible"].astype(bool)]
    advancement_trained = train_advancement_model(
        advancement_elig,
        numeric_features=ADVANCEMENT_NUMERIC,
        categorical_features=ADVANCEMENT_CATEGORICAL,
    )

    ledger = build_attribution_ledger(
        df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status="calibrated",
        infield_confidence_status="calibrated_with_limited_subgroup_evidence",
        advancement_confidence_status="calibrated_with_limited_subgroup_evidence",
    )
    return df, ledger


def test_ledger_has_all_expected_columns(full_ledger):
    _, ledger = full_ledger
    for col in (
        "baseline_expected_contact_run_value",
        "observed_contact_result_run_value",
        "contact_result_surprise",
        "defensive_opportunity_probability",
        "defensive_execution_contribution",
        "expected_advancement_value",
        "advancement_execution_contribution",
        "observed_final_run_value",
        "unexplained_residual",
        "component_eligibility_status",
        "component_confidence_status",
    ):
        assert col in ledger.columns


def test_ledger_preserves_row_count_and_index(full_ledger):
    df, ledger = full_ledger
    assert len(ledger) == len(df)
    assert ledger.index.equals(df.index)


def test_identity_holds_for_every_resolved_row(full_ledger):
    _, ledger = full_ledger
    holds = verify_attribution_identity(ledger)
    resolved = holds.dropna()
    assert len(resolved) > 0
    assert resolved.all(), f"identity failed for {int((~resolved).sum())} of {len(resolved)} rows"


def test_no_row_in_realistic_fixture_is_eligible_for_both_domains(full_ledger):
    df, _ = full_ledger
    both = df["outfield_opportunity_eligible"].astype(bool) & df[
        "infield_opportunity_eligible"
    ].astype(bool)
    assert not both.any()


def test_outfield_rows_get_outfield_defense_status(full_ledger):
    df, ledger = full_ledger
    outfield_mask = df["outfield_opportunity_eligible"].astype(bool)
    assert (
        ledger.loc[outfield_mask, "component_eligibility_status"]
        .str.startswith(STATUS_DEFENSE_OUTFIELD)
        .all()
    )


def test_infield_rows_get_infield_defense_status(full_ledger):
    df, ledger = full_ledger
    infield_mask = df["infield_opportunity_eligible"].astype(bool) & (df["events"] != "field_error")
    assert (
        ledger.loc[infield_mask, "component_eligibility_status"]
        .str.startswith(STATUS_DEFENSE_INFIELD)
        .all()
    )


def test_reached_on_error_rows_are_marked_unavailable_and_excluded_from_identity(full_ledger):
    df, ledger = full_ledger
    error_mask = df["events"] == "field_error"
    assert error_mask.any()
    assert (
        ledger.loc[error_mask, "component_eligibility_status"] == STATUS_CONTACT_RESULT_UNAVAILABLE
    ).all()
    assert ledger.loc[error_mask, "observed_contact_result_run_value"].isna().all()
    holds = verify_attribution_identity(ledger)
    assert holds.loc[error_mask].isna().all()


def test_field_error_status_override_does_not_corrupt_other_rows_values(full_ledger):
    # The field_error status override touches ONLY the status string column
    # for unresolved rows -- every OTHER (resolved, non-field_error) row's
    # accounting values must be fully populated and untouched by that
    # override, and every field_error row's accounting values must be
    # cleanly null (never a stale/leaked value that could be double-counted
    # by a caller that ignores component_eligibility_status).
    df, ledger = full_ledger
    error_mask = df["events"] == "field_error"
    non_error_mask = ~error_mask
    core_cols = [
        "baseline_expected_contact_run_value",
        "observed_contact_result_run_value",
        "contact_result_surprise",
        "observed_final_run_value",
        "unexplained_residual",
    ]
    for col in core_cols:
        assert ledger.loc[non_error_mask, col].notna().all(), (
            f"{col} has a null value for a non-field_error row -- field_error handling "
            "elsewhere in the ledger must never leak into unrelated rows"
        )
        assert ledger.loc[error_mask, col].isna().all(), (
            f"{col} has a non-null value for a field_error row -- these rows must stay "
            "cleanly excluded, not carry a stale computed value"
        )


def test_advancement_eligible_rows_get_advancement_modeled_status(full_ledger):
    df, ledger = full_ledger
    adv_mask = df["advancement_eligible"].astype(bool)
    assert (
        ledger.loc[adv_mask, "component_eligibility_status"]
        .str.contains(STATUS_ADVANCEMENT_MODELED)
        .all()
    )
    assert ledger.loc[adv_mask, "expected_advancement_value"].notna().all()


def test_ground_ball_rows_get_advancement_not_modeled_and_zero_contribution(full_ledger):
    df, ledger = full_ledger
    ground_mask = df["bb_type"] == "ground_ball"
    non_error_ground = ground_mask & (df["events"] != "field_error")
    # field_error rows have no resolved outcome_class, so their status is
    # overridden to STATUS_CONTACT_RESULT_UNAVAILABLE instead -- see
    # test_reached_on_error_rows_are_marked_unavailable_and_excluded_from_identity.
    assert (
        ledger.loc[non_error_ground, "component_eligibility_status"]
        .str.contains(STATUS_ADVANCEMENT_NOT_MODELED)
        .all()
    )
    assert (ledger.loc[non_error_ground, "advancement_execution_contribution"] == 0.0).all()
    # observed_final_run_value must equal observed_contact_result_run_value
    # exactly when advancement is not modeled.
    pd.testing.assert_series_equal(
        ledger.loc[non_error_ground, "observed_final_run_value"],
        ledger.loc[non_error_ground, "observed_contact_result_run_value"],
        check_names=False,
    )


def test_contact_result_surprise_matches_existing_formula(full_ledger):
    _, ledger = full_ledger
    resolved = ledger["observed_contact_result_run_value"].notna()
    computed = (
        ledger.loc[resolved, "observed_contact_result_run_value"]
        - ledger.loc[resolved, "baseline_expected_contact_run_value"]
    )
    pd.testing.assert_series_equal(
        ledger.loc[resolved, "contact_result_surprise"], computed, check_names=False
    )


def test_no_opportunity_or_advancement_model_supplied_yields_pure_contact_ledger():
    df = _synthetic_outfield_df(n_per_class=5)
    df = compute_eligibility(df)
    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)

    ledger = build_attribution_ledger(df, contact_trained)
    assert ledger["defensive_opportunity_probability"].isna().all()
    assert ledger["defensive_execution_contribution"].isna().all()
    assert (ledger["advancement_execution_contribution"] == 0.0).all()
    assert (ledger["component_eligibility_status"].str.startswith(STATUS_DEFENSE_UNAVAILABLE)).all()

    holds = verify_attribution_identity(ledger)
    resolved = holds.dropna()
    assert len(resolved) > 0
    assert resolved.all()
    # With nothing else available, unexplained_residual collapses to
    # contact_result_surprise exactly.
    resolved_mask = ledger["observed_contact_result_run_value"].notna()
    pd.testing.assert_series_equal(
        ledger.loc[resolved_mask, "unexplained_residual"],
        ledger.loc[resolved_mask, "contact_result_surprise"],
        check_names=False,
    )


def test_missing_required_columns_raises():
    with pytest.raises(AttributionLedgerError, match="requires column"):
        build_attribution_ledger(pd.DataFrame({"bb_type": ["fly_ball"]}), contact_trained=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Version 0.10 hardening: play-identifier and domain-routing fail-fasts
# ---------------------------------------------------------------------------


def _minimal_ledger_input(**overrides) -> pd.DataFrame:
    base = {
        "event_id": ["a", "b"],
        "outcome_class": ["out", "single"],
        "bb_type": ["fly_ball", "fly_ball"],
        "events": ["field_out", "single"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_assert_unique_play_identifiers_accepts_clean_ids():
    assert_unique_play_identifiers(pd.DataFrame({"event_id": ["a", "b", "c"]}))  # must not raise


def test_assert_unique_play_identifiers_rejects_duplicates():
    with pytest.raises(AttributionLedgerError, match="duplicate"):
        assert_unique_play_identifiers(pd.DataFrame({"event_id": ["a", "a", "b"]}))


def test_assert_unique_play_identifiers_rejects_nulls():
    with pytest.raises(AttributionLedgerError, match="null"):
        assert_unique_play_identifiers(pd.DataFrame({"event_id": ["a", None]}))


def test_assert_unique_play_identifiers_rejects_missing_column():
    with pytest.raises(AttributionLedgerError, match="missing the play-identifier column"):
        assert_unique_play_identifiers(pd.DataFrame({"other_col": [1]}))


def test_missing_event_id_raises():
    df = pd.DataFrame({"outcome_class": ["out"], "bb_type": ["fly_ball"], "events": ["field_out"]})
    with pytest.raises(AttributionLedgerError, match="requires column"):
        build_attribution_ledger(df, contact_trained=None)  # type: ignore[arg-type]


def test_duplicate_event_id_raises_before_scoring():
    df = _minimal_ledger_input(event_id=["dup", "dup"])
    with pytest.raises(AttributionLedgerError, match="duplicate"):
        build_attribution_ledger(df, contact_trained=None)  # type: ignore[arg-type]


def test_null_event_id_raises_before_scoring():
    df = _minimal_ledger_input(event_id=["a", None])
    with pytest.raises(AttributionLedgerError, match="null"):
        build_attribution_ledger(df, contact_trained=None)  # type: ignore[arg-type]


def test_row_eligible_for_both_domains_raises_before_scoring():
    df = _minimal_ledger_input(
        outfield_opportunity_eligible=[True, False],
        infield_opportunity_eligible=[True, False],
    )
    with pytest.raises(AttributionLedgerError, match="BOTH outfield and infield"):
        build_attribution_ledger(df, contact_trained=None)  # type: ignore[arg-type]


def test_confidence_status_reflects_supplied_gate_verdicts(full_ledger):
    df, ledger = full_ledger
    outfield_mask = df["outfield_opportunity_eligible"].astype(bool)
    assert (
        ledger.loc[outfield_mask, "component_confidence_status"]
        .str.contains("defense=calibrated;")
        .all()
    )


def test_confidence_status_reports_not_supplied_when_omitted():
    df = _synthetic_outfield_df(n_per_class=5)
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_outfield_opportunity_features(df)
    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)
    outfield_elig = df[df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_elig, class_weight=None)

    ledger = build_attribution_ledger(df, contact_trained, outfield_trained=outfield_trained)
    outfield_mask = df["outfield_opportunity_eligible"].astype(bool)
    assert (
        ledger.loc[outfield_mask, "component_confidence_status"]
        .str.contains("defense=not_supplied")
        .all()
    )
