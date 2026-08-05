"""Tests for the Contact Luck v0.9 advancement component report.

No test touches the network -- synthetic outfield-air-ball-shaped fixtures only.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import add_advancement_eligibility, compute_eligibility
from mlb_luck_score.features.build_contact_features import (
    add_advancement_contact_probability_features,
    add_advancement_features,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_NOT_CALIBRATED,
)
from mlb_luck_score.models.train_advancement_model import train_advancement_model
from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model
from mlb_luck_score.scoring import advancement_execution
from mlb_luck_score.scoring.advancement_execution import (
    AdvancementComponentError,
    build_advancement_component_report,
)

NUMERIC = [
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
CATEGORICAL = ["hit_type_group", "assigned_outfield_position", "stand"]


def _synthetic_outfield_df(n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    profiles = {
        "single": (88.0, 10.0, 200.0, "single"),
        "double": (98.0, 15.0, 320.0, "double"),
        "triple": (100.0, 20.0, 370.0, "triple"),
        "home_run": (105.0, 28.0, 410.0, "home_run"),
    }
    for speed, angle, dist, events in profiles.values():
        for i in range(n_per_class):
            rows.append(
                {
                    "game_pk": 3000 + i,
                    "launch_speed": speed + rng.normal(0, 1.0),
                    "launch_angle": angle + rng.normal(0, 1.0),
                    "spray_angle_approx": rng.uniform(-15, 15),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "fly_ball",
                    "stand": "R",
                    "venue": "Synthetic Park",
                    "events": events,
                    "des": {
                        "single": "Someone singles on a fly ball to center fielder Other Player.",
                        "double": "Someone doubles (1) on a fly ball to center fielder Other Player.",
                        "triple": "Someone triples (1) on a fly ball to center fielder Other Player.",
                        "home_run": "Someone homers (1) on a fly ball to right field.",
                    }[events],
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


@pytest.fixture
def trained_models():
    df = _synthetic_outfield_df()
    df = compute_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_advancement_features(df)

    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)

    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    df = add_advancement_contact_probability_features(df, contact_proba)

    advancement_elig = df[df["advancement_eligible"].astype(bool)]
    advancement_trained = train_advancement_model(
        advancement_elig, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    return df, contact_trained, advancement_trained


def test_report_has_all_expected_columns(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    for col in (
        "expected_contact_base",
        "actual_final_batter_base",
        "advancement_opportunity",
        "batter_runner_advancement_execution",
        "defensive_advancement_effect",
        "residual_uncertainty",
        "advancement_status",
    ):
        assert col in report.columns


def test_report_preserves_row_count_and_index(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    assert len(report) == len(df)
    assert report.index.equals(df.index)


def test_eligible_rows_have_all_six_components_populated(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible_mask = df["advancement_eligible"].astype(bool)
    eligible_report = report.loc[eligible_mask]
    for col in (
        "expected_contact_base",
        "actual_final_batter_base",
        "advancement_opportunity",
        "batter_runner_advancement_execution",
        "defensive_advancement_effect",
        "residual_uncertainty",
    ):
        assert eligible_report[col].notna().all()


def test_defensive_advancement_effect_is_negation_of_batter_execution(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible = report[report["advancement_opportunity"].notna()]
    pd.testing.assert_series_equal(
        eligible["defensive_advancement_effect"],
        -eligible["batter_runner_advancement_execution"],
        check_names=False,
    )


def test_batter_execution_equals_actual_minus_opportunity(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible = report[report["advancement_opportunity"].notna()]
    computed = eligible["actual_final_batter_base"] - eligible["advancement_opportunity"]
    pd.testing.assert_series_equal(
        eligible["batter_runner_advancement_execution"], computed, check_names=False
    )


def test_residual_uncertainty_is_non_negative(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible = report[report["residual_uncertainty"].notna()]
    assert (eligible["residual_uncertainty"] >= 0).all()


def test_expected_contact_base_is_finite_for_every_row(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    assert np.isfinite(report["expected_contact_base"]).all()


def test_trivial_home_run_rows_get_excluded_status(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    home_run_rows = df[df["events"] == "home_run"]
    assert (
        report.loc[home_run_rows.index, "advancement_status"] == "excluded_trivial_home_run"
    ).all()
    assert report.loc[home_run_rows.index, "advancement_opportunity"].isna().all()


def test_eligible_rows_get_calibrated_status_when_overall_calibrated(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible_mask = df["advancement_eligible"].astype(bool)
    assert (report.loc[eligible_mask, "advancement_status"] == "calibrated_advancement_model").all()


def test_eligible_rows_get_provisional_status_when_not_calibrated(trained_models):
    df, contact_trained, advancement_trained = trained_models
    report = build_advancement_component_report(
        df, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_NOT_CALIBRATED
    )
    eligible_mask = df["advancement_eligible"].astype(bool)
    assert (
        report.loc[eligible_mask, "advancement_status"] == "provisional_advancement_model"
    ).all()


def test_components_are_never_summed_into_one_score():
    source = inspect.getsource(advancement_execution)
    assert "expected_contact_base +" not in source
    assert "+ expected_contact_base" not in source
    assert "advancement_opportunity +" not in source
    assert "residual_uncertainty +" not in source


def test_raises_on_unexpected_exclusion_reason(trained_models):
    df, contact_trained, advancement_trained = trained_models
    bad = df.copy()
    bad.loc[bad.index[0], "advancement_exclusion_reason"] = "not_a_real_reason"
    bad.loc[bad.index[0], "advancement_eligible"] = False
    with pytest.raises(AdvancementComponentError, match="Unhandled advancement_exclusion_reason"):
        build_advancement_component_report(
            bad, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
        )


def test_rejects_non_outfield_air_ball_rows(trained_models):
    df, contact_trained, advancement_trained = trained_models
    bad = df.copy()
    bad.loc[bad.index[0], "bb_type"] = "ground_ball"
    with pytest.raises(AdvancementComponentError, match="outfield air-ball rows"):
        build_advancement_component_report(
            bad, contact_trained, advancement_trained, overall_status=OVERALL_STATUS_CALIBRATED
        )
