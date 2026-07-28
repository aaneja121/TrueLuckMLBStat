"""Tests for the Contact Luck v0.5.1 controlled-perturbation testing utilities.

No test touches the network -- a small real-shaped synthetic model + data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.weather_perturbation import (
    check_directional_effect,
    mean_predicted_probability,
    override_columns,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    # hit_distance_sc genuinely DRIVES the outcome label here (short ->
    # "out", long -> "home_run") so a trained model reliably learns the
    # expected monotonic relationship -- needed to test that
    # check_directional_effect's pass/fail logic itself is correct, as
    # opposed to accidentally testing an untrained/random relationship.
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        distance = float(rng.uniform(50.0, 450.0))
        if distance < 150.0:
            outcome = "out"
        elif distance < 250.0:
            outcome = "single"
        elif distance < 320.0:
            outcome = "double"
        elif distance < 380.0:
            outcome = "triple"
        else:
            outcome = "home_run"
        rows.append(
            {
                "launch_speed": 95.0 + rng.normal(0, 5),
                "launch_angle": 20.0 + rng.normal(0, 8),
                "spray_angle_approx": rng.uniform(-44, 44),
                "hit_distance_sc": distance,
                "bb_type": "fly_ball",
                "stand": "R" if i % 2 == 0 else "L",
                "outcome_class": outcome,
                "flag_col": "A" if i % 2 == 0 else "B",
                "value_col": float(i),
                "eligible_for_training": True,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_model(synthetic_df: pd.DataFrame):
    return train_model(synthetic_df, class_weight=None)


def test_override_columns_applies_to_all_rows_by_default(synthetic_df: pd.DataFrame):
    result = override_columns(synthetic_df, {"value_col": 999.0})
    assert (result["value_col"] == 999.0).all()


def test_override_columns_respects_mask(synthetic_df: pd.DataFrame):
    mask = synthetic_df["flag_col"] == "A"
    result = override_columns(synthetic_df, {"value_col": -1.0}, mask=mask)
    assert (result.loc[mask, "value_col"] == -1.0).all()
    assert (result.loc[~mask, "value_col"] == synthetic_df.loc[~mask, "value_col"]).all()


def test_override_columns_leaves_non_overridden_columns_untouched(synthetic_df: pd.DataFrame):
    result = override_columns(synthetic_df, {"value_col": 0.0})
    pd.testing.assert_series_equal(result["launch_speed"], synthetic_df["launch_speed"])


def test_override_columns_ignores_unknown_column(synthetic_df: pd.DataFrame):
    result = override_columns(synthetic_df, {"nonexistent_col": 1.0})
    assert "nonexistent_col" not in result.columns


def test_mean_predicted_probability_is_between_zero_and_one(
    trained_model, synthetic_df: pd.DataFrame
):
    prob = mean_predicted_probability(trained_model, synthetic_df, "home_run")
    assert 0.0 <= prob <= 1.0


def test_check_directional_effect_passes_when_direction_matches(trained_model, synthetic_df):
    # hit_distance_sc strongly drives home_run probability in this synthetic
    # setup indirectly via launch conditions -- use it directly as a proxy
    # perturbation to build a deterministic, known-direction test.
    low_overrides = {"hit_distance_sc": 50.0}
    high_overrides = {"hit_distance_sc": 450.0}
    result = check_directional_effect(
        trained_model,
        synthetic_df,
        low_overrides=low_overrides,
        high_overrides=high_overrides,
        expect_high_greater=True,
        outcome_class="home_run",
        label="distance sanity check",
    )
    assert result.mean_prob_high > result.mean_prob_low
    assert result.passed is True
    assert result.delta == pytest.approx(result.mean_prob_high - result.mean_prob_low)


def test_check_directional_effect_fails_when_direction_is_wrong(trained_model, synthetic_df):
    low_overrides = {"hit_distance_sc": 50.0}
    high_overrides = {"hit_distance_sc": 450.0}
    # Deliberately assert the WRONG expected direction.
    result = check_directional_effect(
        trained_model,
        synthetic_df,
        low_overrides=low_overrides,
        high_overrides=high_overrides,
        expect_high_greater=False,
        outcome_class="home_run",
        label="deliberately wrong expectation",
    )
    assert result.passed is False


def test_check_directional_effect_reports_sample_size_with_mask(trained_model, synthetic_df):
    mask = synthetic_df["flag_col"] == "A"
    result = check_directional_effect(
        trained_model,
        synthetic_df,
        low_overrides={"hit_distance_sc": 50.0},
        high_overrides={"hit_distance_sc": 450.0},
        expect_high_greater=True,
        mask=mask,
        label="masked",
    )
    assert result.sample_size == int(mask.sum())


def test_check_directional_effect_full_sample_size_without_mask(trained_model, synthetic_df):
    result = check_directional_effect(
        trained_model,
        synthetic_df,
        low_overrides={"hit_distance_sc": 50.0},
        high_overrides={"hit_distance_sc": 450.0},
        expect_high_greater=True,
        label="unmasked",
    )
    assert result.sample_size == len(synthetic_df)
