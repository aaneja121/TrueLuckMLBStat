"""Tests for the Contact Luck v0.7 air-ball component report.

No test touches the network -- synthetic outfield-opportunity-shaped
fixtures only.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import add_outfield_opportunity_eligibility
from mlb_luck_score.features.build_contact_features import add_outfield_opportunity_features
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring import air_ball_components
from mlb_luck_score.scoring.air_ball_components import (
    AirBallComponentError,
    build_air_ball_component_report,
)


def _synthetic_air_ball_df(n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    profiles = {
        "out": (85.0, 25.0, 250.0, "field_out"),
        "single": (92.0, 8.0, 180.0, "single"),
        "double": (98.0, 20.0, 320.0, "double"),
        "home_run": (105.0, 28.0, 410.0, "home_run"),
    }
    for outcome, (speed, angle, dist, events) in profiles.items():
        for i in range(n_per_class):
            rows.append(
                {
                    "game_pk": 1000 + i,
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.uniform(-20, 20),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "fly_ball",
                    "stand": "R",
                    "venue": "Synthetic Park",
                    "hit_location": 8,
                    "of_fielding_alignment": "Standard",
                    "if_fielding_alignment": "Standard",
                    "fielder_7": 111,
                    "fielder_8": 222,
                    "fielder_9": 333,
                    "events": events,
                    "outcome_class": outcome,
                    "eligible_for_training": True,
                    "season": 2021 + (i % 4),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_models():
    df = _synthetic_air_ball_df()
    df = add_outfield_opportunity_eligibility(df)
    df = add_outfield_opportunity_features(df)
    elig = df[df["outfield_opportunity_eligible"]]

    contact_trained = train_model(elig, class_weight=None)
    opportunity_trained = train_opportunity_model(elig, class_weight=None)
    return elig, contact_trained, opportunity_trained


def test_report_has_all_four_component_columns(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    for col in (
        "expected_run_value_contact_model",
        "actual_run_value",
        "residual_contact_luck_runs",
        "p_out_opportunity",
        "converted_to_out",
        "defensive_execution",
        "batter_favorable_defensive_circumstance",
    ):
        assert col in report.columns


def test_report_preserves_row_count_and_index(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    assert len(report) == len(elig)
    assert report.index.equals(elig.index)


def test_report_includes_id_columns_when_present(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    assert "game_pk" in report.columns


def test_residual_contact_luck_equals_actual_minus_expected(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    computed = report["actual_run_value"] - report["expected_run_value_contact_model"]
    pd.testing.assert_series_equal(
        report["residual_contact_luck_runs"], computed, check_names=False
    )


def test_defensive_execution_consistent_with_p_out_and_converted(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    computed = report["converted_to_out"].astype(float) - report["p_out_opportunity"]
    pd.testing.assert_series_equal(
        report["defensive_execution"], computed, check_names=False, check_exact=False
    )


def test_all_probabilities_and_run_values_finite(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    report = build_air_ball_component_report(elig, contact_trained, opportunity_trained)
    assert np.isfinite(report["p_out_opportunity"]).all()
    assert np.isfinite(report["expected_run_value_contact_model"]).all()
    assert np.isfinite(report["residual_contact_luck_runs"]).all()


def test_components_are_never_summed_into_one_score():
    # Contract check: the module never adds the four components together --
    # only reports them as separate columns. See module docstring.
    source = inspect.getsource(air_ball_components)
    # No line should add residual_contact_luck_runs to defensive_execution
    # or p_out_opportunity (the only plausible "combine into one score" bug).
    assert "residual_contact_luck_runs +" not in source
    assert "+ residual_contact_luck_runs" not in source
    assert "defensive_execution +" not in source
    assert "+ defensive_execution" not in source


def test_raises_on_unknown_outcome_class(trained_models):
    elig, contact_trained, opportunity_trained = trained_models
    bad = elig.copy()
    bad.loc[bad.index[0], "outcome_class"] = "not_a_real_outcome"
    with pytest.raises(AirBallComponentError, match="run_value_map"):
        build_air_ball_component_report(bad, contact_trained, opportunity_trained)
