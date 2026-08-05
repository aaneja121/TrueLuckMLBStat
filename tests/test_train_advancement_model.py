"""Tests for the Contact Luck v0.9 multinomial batter-runner advancement model.

No test touches the network -- synthetic advancement-shaped fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.eligibility import ADVANCEMENT_LABELS
from mlb_luck_score.models.train_advancement_model import (
    MODEL_TYPE_HGB,
    MODEL_TYPE_LOGISTIC,
    VARIANT_CLASS_BALANCED,
    VARIANT_UNWEIGHTED,
    evaluate_advancement_model,
    predict_advancement_proba,
    train_advancement_model,
    validate_advancement_probabilities,
)


def _synthetic_advancement_df(n_per_class: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    # Deterministic relationship: faster sprint speed -> more advancement.
    profiles = {
        "held_at_first": 23.0,
        "advanced_to_second": 26.0,
        "advanced_to_third": 28.5,
        "retired_while_advancing": 25.0,
        "inside_the_park_home_run": 29.5,
    }
    for label, speed in profiles.items():
        for i in range(n_per_class):
            rows.append(
                {
                    "launch_speed": 95.0 + rng.normal(0, 3),
                    "launch_angle": 20.0 + rng.normal(0, 3),
                    "spray_angle_approx": rng.uniform(-15, 15),
                    "hit_distance_sc": 300.0 + rng.normal(0, 10),
                    "estimated_hang_time_s": 4.0 + rng.normal(0, 0.3),
                    "landing_x_ft": rng.normal(0, 10),
                    "landing_y_ft": 300.0,
                    "wall_distance_in_spray_direction": 380.0,
                    "absolute_distance_to_wall": 80.0,
                    "outs_when_up": i % 3,
                    "on_1b_occupied": 0,
                    "on_2b_occupied": 0,
                    "on_3b_occupied": 0,
                    "assigned_outfield_position": "8",
                    "of_fielding_alignment": "Standard",
                    "stand": "R",
                    "sprint_speed": speed + rng.normal(0, 0.5),
                    "contact_p_out": 0.3,
                    "contact_p_single": 0.3,
                    "contact_p_double": 0.2,
                    "contact_p_triple": 0.1,
                    "contact_p_home_run": 0.1,
                    "batter_final_base": label,
                    "season": 2021 + (i % 4),
                }
            )
    return pd.DataFrame(rows)


NUMERIC = [
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "sprint_speed",
    "contact_p_out",
    "contact_p_single",
    "contact_p_double",
    "contact_p_triple",
    "contact_p_home_run",
]
CATEGORICAL = ["assigned_outfield_position", "of_fielding_alignment", "stand"]


@pytest.fixture
def advancement_df() -> pd.DataFrame:
    return _synthetic_advancement_df()


def test_train_advancement_model_produces_valid_multiclass_probabilities(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    proba = predict_advancement_proba(trained, advancement_df[NUMERIC + CATEGORICAL].head(10))
    validate_advancement_probabilities(proba)
    assert list(proba.columns) == list(ADVANCEMENT_LABELS)
    assert len(proba) == 10


def test_model_learns_faster_speed_means_more_advancement(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    slow_row = advancement_df[advancement_df["batter_final_base"] == "held_at_first"].iloc[[0]]
    fast_row = advancement_df[advancement_df["batter_final_base"] == "advanced_to_third"].iloc[[0]]
    p_slow = predict_advancement_proba(trained, slow_row[NUMERIC + CATEGORICAL]).iloc[0]
    p_fast = predict_advancement_proba(trained, fast_row[NUMERIC + CATEGORICAL]).iloc[0]
    assert p_slow["held_at_first"] > p_fast["held_at_first"]
    assert p_fast["advanced_to_third"] > p_slow["advanced_to_third"]


def test_default_variant_is_unweighted(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    assert trained.variant == VARIANT_UNWEIGHTED
    assert trained.class_weight is None


def test_class_balanced_variant_is_explicitly_labeled(advancement_df):
    trained = train_advancement_model(
        advancement_df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        class_weight="balanced",
    )
    assert trained.variant == VARIANT_CLASS_BALANCED


def test_validate_advancement_probabilities_rejects_wrong_columns():
    bad = pd.DataFrame({"a": [0.5], "b": [0.5]})
    with pytest.raises(ValueError, match="Expected columns"):
        validate_advancement_probabilities(bad)


def test_validate_advancement_probabilities_rejects_rows_not_summing_to_one():
    bad = pd.DataFrame(
        {label: [0.5 if label == ADVANCEMENT_LABELS[0] else 0.0] for label in ADVANCEMENT_LABELS}
    )
    with pytest.raises(ValueError, match="do not sum to 1"):
        validate_advancement_probabilities(bad)


def test_evaluate_advancement_model_returns_expected_keys(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    metrics = evaluate_advancement_model(trained, advancement_df)
    assert metrics["sample_count"] == len(advancement_df)
    assert "multiclass_log_loss" in metrics
    assert set(metrics["per_class_brier_score"].keys()) == set(ADVANCEMENT_LABELS)
    assert set(metrics["class_frequencies"].keys()) == set(ADVANCEMENT_LABELS)
    assert metrics["confusion_matrix"]["labels"] == list(ADVANCEMENT_LABELS)
    assert "feature_missingness" in metrics


def test_hgb_model_type_produces_valid_probabilities(advancement_df):
    trained = train_advancement_model(
        advancement_df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        model_type=MODEL_TYPE_HGB,
    )
    assert trained.model_type == MODEL_TYPE_HGB
    proba = predict_advancement_proba(trained, advancement_df[NUMERIC + CATEGORICAL].head(10))
    validate_advancement_probabilities(proba)


def test_default_model_type_is_logistic(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    assert trained.model_type == MODEL_TYPE_LOGISTIC


def test_unknown_model_type_rejected(advancement_df):
    with pytest.raises(ValueError, match="model_type"):
        train_advancement_model(
            advancement_df,
            numeric_features=NUMERIC,
            categorical_features=CATEGORICAL,
            model_type="random_forest",
        )


def test_leakage_columns_are_rejected(advancement_df):
    from mlb_luck_score.features.build_contact_features import LeakageError

    with pytest.raises(LeakageError):
        train_advancement_model(
            advancement_df,
            numeric_features=[*NUMERIC, "woba_value"],
            categorical_features=CATEGORICAL,
        )


def test_class_order_defaults_to_advancement_labels(advancement_df):
    trained = train_advancement_model(
        advancement_df, numeric_features=NUMERIC, categorical_features=CATEGORICAL
    )
    assert trained.class_order == list(ADVANCEMENT_LABELS)
