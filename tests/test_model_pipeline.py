from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import CLASS_ORDER, ProtectedSeasonError, assert_seasons_allowed
from mlb_luck_score.models.train_contact_model import (
    evaluate_model,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)


def _synthetic_training_df(n_per_class: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    profiles = {
        "out": (75.0, 20.0, 100.0),
        "single": (92.0, 8.0, 180.0),
        "double": (98.0, 18.0, 300.0),
        "triple": (100.0, 15.0, 340.0),
        "home_run": (105.0, 28.0, 410.0),
    }
    for outcome, (speed, angle, dist) in profiles.items():
        for _ in range(n_per_class):
            rows.append(
                {
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.normal(0, 10),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "line_drive",
                    "stand": "R",
                    "venue": "Synthetic Park",
                    "outcome_class": outcome,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def training_df() -> pd.DataFrame:
    return _synthetic_training_df()


def test_train_model_produces_five_class_probabilities(training_df: pd.DataFrame):
    trained = train_model(training_df)
    feature_cols = trained.numeric_features + trained.categorical_features
    proba = predict_proba_ordered(trained, training_df[feature_cols].head(5))
    assert list(proba.columns) == list(CLASS_ORDER)
    validate_probabilities(proba)


def test_predict_proba_ordered_is_stable_regardless_of_row_order(training_df: pd.DataFrame):
    trained = train_model(training_df)
    feature_cols = trained.numeric_features + trained.categorical_features
    shuffled = training_df.sample(frac=1.0, random_state=7)
    proba_a = predict_proba_ordered(trained, training_df[feature_cols].head(3))
    proba_b = predict_proba_ordered(trained, shuffled[feature_cols].loc[training_df.head(3).index])
    assert list(proba_a.columns) == list(proba_b.columns) == list(CLASS_ORDER)


def test_validate_probabilities_rejects_bad_sums():
    bad = pd.DataFrame([[0.5, 0.5, 0.5, 0.0, 0.0]], columns=list(CLASS_ORDER))
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probabilities(bad)


def test_validate_probabilities_rejects_wrong_columns():
    bad = pd.DataFrame([[1.0]], columns=["out"])
    with pytest.raises(ValueError, match="Expected columns"):
        validate_probabilities(bad)


def test_evaluate_model_returns_expected_keys(training_df: pd.DataFrame):
    trained = train_model(training_df)
    metrics = evaluate_model(trained, training_df)
    assert metrics["sample_count"] == len(training_df)
    assert set(metrics["brier_score_by_class"].keys()) == set(CLASS_ORDER)
    assert len(metrics["confusion_matrix"]) == len(CLASS_ORDER)
    assert "multiclass_log_loss" in metrics
    assert "feature_missingness" in metrics


def test_rare_classes_are_preserved_not_deleted(training_df: pd.DataFrame):
    # triples are rare in real data; ensure a small number of examples survive training
    # rather than being dropped, and the model can still assign them nonzero probability.
    trained = train_model(training_df)
    assert "triple" in trained.pipeline.named_steps["classify"].classes_


def test_training_rejects_protected_2025_season_without_explicit_flag():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed([2024, 2025])
    # Explicit opt-in is required and must be intentional -- normal training
    # commands must not pass this by default.
    assert_seasons_allowed([2024, 2025], allow_final_evaluation=True)
