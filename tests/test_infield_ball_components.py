"""Tests for the Contact Luck v0.8 infield component report.

No test touches the network -- synthetic infield-opportunity-shaped
fixtures only.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import add_infield_opportunity_eligibility, compute_eligibility
from mlb_luck_score.features.build_contact_features import add_infield_opportunity_features
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    OVERALL_STATUS_NOT_CALIBRATED,
)
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring import infield_ball_components
from mlb_luck_score.scoring.infield_ball_components import (
    InfieldBallComponentError,
    build_infield_ball_component_report,
)


def _synthetic_ground_ball_df(n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    # (launch_speed, launch_angle, events, hit_location, des)
    profiles = {
        "field_out": (90.0, -5.0, "field_out", 6, "Routine grounder to short."),
        "single": (85.0, 2.0, "single", 4, "Ground ball single up the middle."),
        "double": (100.0, -2.0, "double", 5, "Ground ball double down the line."),
        "field_error": (88.0, 1.0, "field_error", 6, "Reaches on a throwing error."),
    }
    for speed, angle, events, hit_location, des in profiles.values():
        for i in range(n_per_class):
            rows.append(
                {
                    "game_pk": 2000 + i,
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
                    "season": 2021 + (i % 4),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_models():
    df = _synthetic_ground_ball_df()
    df = compute_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_infield_opportunity_features(df)

    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)

    infield_elig = df[df["infield_opportunity_eligible"].astype(bool)]
    infield_trained = train_opportunity_model(infield_elig, class_weight=None)
    return df, contact_trained, infield_trained


def test_report_has_all_expected_columns(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    for col in (
        "expected_run_value_contact_model",
        "actual_run_value",
        "residual_contact_luck_runs",
        "p_out_opportunity",
        "actual_out",
        "reached_on_error",
        "defensive_execution_probability",
        "batter_perspective_infield_execution",
        "infield_opportunity_status",
    ):
        assert col in report.columns


def test_report_preserves_row_count_and_index(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    assert len(report) == len(df)
    assert report.index.equals(df.index)


def test_reached_on_error_rows_have_null_run_value_columns(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    error_rows = report[report["reached_on_error"]]
    assert len(error_rows) > 0
    assert error_rows["expected_run_value_contact_model"].isna().all()
    assert error_rows["actual_run_value"].isna().all()
    assert error_rows["residual_contact_luck_runs"].isna().all()


def test_reached_on_error_rows_still_have_opportunity_and_execution(trained_models):
    # field_error is eligible/safe (y_out=0) -- opportunity/execution columns
    # should be populated even though the run-value columns are null.
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    error_rows = report[report["reached_on_error"]]
    assert error_rows["p_out_opportunity"].notna().all()
    assert (error_rows["actual_out"] == 0.0).all()


def test_eligible_rows_get_calibrated_status_when_overall_calibrated(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible_mask = df["infield_opportunity_eligible"].astype(bool)
    assert (
        report.loc[eligible_mask, "infield_opportunity_status"] == "calibrated_infield_opportunity"
    ).all()


def test_eligible_rows_get_insufficient_evidence_status_for_limited_evidence(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df,
        contact_trained,
        infield_trained,
        overall_status=OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    )
    eligible_mask = df["infield_opportunity_eligible"].astype(bool)
    assert (
        report.loc[eligible_mask, "infield_opportunity_status"] == "insufficient_evidence"
    ).all()


def test_eligible_rows_get_provisional_status_when_not_calibrated(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_NOT_CALIBRATED
    )
    eligible_mask = df["infield_opportunity_eligible"].astype(bool)
    assert (
        report.loc[eligible_mask, "infield_opportunity_status"] == "provisional_infield_opportunity"
    ).all()


def test_defensive_execution_consistent_with_p_out_and_actual_out(trained_models):
    df, contact_trained, infield_trained = trained_models
    report = build_infield_ball_component_report(
        df, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
    )
    eligible = report[report["p_out_opportunity"].notna()]
    computed = eligible["actual_out"] - eligible["p_out_opportunity"]
    pd.testing.assert_series_equal(
        eligible["defensive_execution_probability"], computed, check_names=False, check_exact=False
    )


def test_components_are_never_summed_into_one_score():
    source = inspect.getsource(infield_ball_components)
    assert "residual_contact_luck_runs +" not in source
    assert "+ residual_contact_luck_runs" not in source
    assert "defensive_execution_probability +" not in source
    assert "+ defensive_execution_probability" not in source


def test_raises_on_unexpected_exclusion_reason(trained_models):
    df, contact_trained, infield_trained = trained_models
    bad = df.copy()
    bad.loc[bad.index[0], "infield_opportunity_exclusion_reason"] = "not_a_real_reason"
    bad.loc[bad.index[0], "infield_opportunity_eligible"] = False
    with pytest.raises(
        InfieldBallComponentError, match="Unhandled infield_opportunity_exclusion_reason"
    ):
        build_infield_ball_component_report(
            bad, contact_trained, infield_trained, overall_status=OVERALL_STATUS_CALIBRATED
        )
