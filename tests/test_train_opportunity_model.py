"""Tests for the Contact Luck v0.7A binary outfield-opportunity model.

No test touches the network -- synthetic outfield-opportunity-shaped
fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import ProtectedSeasonError, assert_seasons_allowed
from mlb_luck_score.models.train_opportunity_model import (
    MODEL_TYPE_HGB,
    MODEL_TYPE_LOGISTIC,
    VARIANT_CLASS_BALANCED,
    VARIANT_UNWEIGHTED,
    evaluate_opportunity_model,
    predict_opportunity_proba,
    train_opportunity_model,
    validate_opportunity_probabilities,
)


def _synthetic_opportunity_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        # Deterministic relationship: shallow/short hang time -> almost always
        # converted; deep/long hang time -> rarely converted -- gives the
        # model real, learnable signal (mirrors test_weather_perturbation.py's
        # approach of encoding a real relationship into the fixture).
        deep = i % 2 == 0
        hang_time = 5.5 + rng.normal(0, 0.3) if deep else 2.0 + rng.normal(0, 0.3)
        distance = 380.0 + rng.normal(0, 10) if deep else 150.0 + rng.normal(0, 10)
        converted = 0 if deep else 1
        rows.append(
            {
                "launch_speed": 95.0 + rng.normal(0, 2),
                "launch_angle": 28.0 + rng.normal(0, 2),
                "hit_distance_sc": distance,
                "estimated_hang_time_s": hang_time,
                "landing_x_ft": rng.normal(0, 20),
                "landing_y_ft": distance,
                "wall_distance_in_spray_direction": 350.0,
                "absolute_distance_to_wall": 350.0 - distance,
                "bb_type": "fly_ball",
                "of_fielding_alignment": "Standard",
                "assigned_outfield_position": "8",
                "converted_to_out": converted,
                "season": 2021 + (i % 4),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def opportunity_df() -> pd.DataFrame:
    return _synthetic_opportunity_df()


def test_train_opportunity_model_produces_binary_probabilities(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df, class_weight=None)
    feature_cols = trained.numeric_features + trained.categorical_features
    p_out = predict_opportunity_proba(trained, opportunity_df[feature_cols].head(10))
    validate_opportunity_probabilities(p_out)
    assert p_out.name == "p_out"
    assert len(p_out) == 10


def test_opportunity_model_learns_the_real_relationship(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df, class_weight=None)
    feature_cols = trained.numeric_features + trained.categorical_features
    deep_row = opportunity_df[opportunity_df["converted_to_out"] == 0].iloc[[0]]
    shallow_row = opportunity_df[opportunity_df["converted_to_out"] == 1].iloc[[0]]
    p_deep = predict_opportunity_proba(trained, deep_row[feature_cols]).iloc[0]
    p_shallow = predict_opportunity_proba(trained, shallow_row[feature_cols]).iloc[0]
    # Deep balls (label=0, i.e. NOT converted) should have LOWER p_out than
    # shallow ones (label=1, converted).
    assert p_deep < p_shallow


def test_default_variant_is_unweighted(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df)
    assert trained.variant == VARIANT_UNWEIGHTED
    assert trained.class_weight is None


def test_class_balanced_variant_is_explicitly_labeled(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df, class_weight="balanced")
    assert trained.variant == VARIANT_CLASS_BALANCED


def test_validate_opportunity_probabilities_rejects_out_of_range():
    bad = pd.Series([0.5, 1.5, -0.1])
    with pytest.raises(ValueError, match="out of \\[0, 1\\] range"):
        validate_opportunity_probabilities(bad)


def test_validate_opportunity_probabilities_rejects_non_finite():
    bad = pd.Series([0.5, float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        validate_opportunity_probabilities(bad)


def test_evaluate_opportunity_model_returns_expected_keys(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df)
    metrics = evaluate_opportunity_model(trained, opportunity_df)
    assert metrics["sample_count"] == len(opportunity_df)
    assert "binary_log_loss" in metrics
    assert "brier_score" in metrics
    assert "converted_to_out_rate" in metrics
    assert "feature_missingness" in metrics


def test_train_opportunity_model_requires_both_classes():
    df = _synthetic_opportunity_df()
    df["converted_to_out"] = 1  # only one class present
    with pytest.raises(ValueError, match="both converted_to_out"):
        train_opportunity_model(df)


def test_custom_feature_list_is_respected(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(
        opportunity_df,
        numeric_features=["launch_speed", "estimated_hang_time_s"],
        categorical_features=["bb_type"],
    )
    assert trained.numeric_features == ["launch_speed", "estimated_hang_time_s"]
    assert trained.categorical_features == ["bb_type"]


def test_hgb_model_type_produces_valid_probabilities(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df, class_weight=None, model_type=MODEL_TYPE_HGB)
    assert trained.model_type == MODEL_TYPE_HGB
    feature_cols = trained.numeric_features + trained.categorical_features
    p_out = predict_opportunity_proba(trained, opportunity_df[feature_cols].head(10))
    validate_opportunity_probabilities(p_out)


def test_default_model_type_is_logistic(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df)
    assert trained.model_type == MODEL_TYPE_LOGISTIC


def test_unknown_model_type_rejected(opportunity_df: pd.DataFrame):
    with pytest.raises(ValueError, match="model_type"):
        train_opportunity_model(opportunity_df, model_type="random_forest")


def test_hgb_learns_the_real_relationship(opportunity_df: pd.DataFrame):
    trained = train_opportunity_model(opportunity_df, class_weight=None, model_type=MODEL_TYPE_HGB)
    feature_cols = trained.numeric_features + trained.categorical_features
    deep_row = opportunity_df[opportunity_df["converted_to_out"] == 0].iloc[[0]]
    shallow_row = opportunity_df[opportunity_df["converted_to_out"] == 1].iloc[[0]]
    p_deep = predict_opportunity_proba(trained, deep_row[feature_cols]).iloc[0]
    p_shallow = predict_opportunity_proba(trained, shallow_row[feature_cols]).iloc[0]
    assert p_deep < p_shallow


def test_training_rejects_protected_2025_season_without_explicit_flag():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed([2024, 2025])
    assert_seasons_allowed([2024, 2025], allow_final_evaluation=True)
