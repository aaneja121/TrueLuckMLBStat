from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.features.build_contact_features import (
    LeakageError,
    assert_no_leakage,
    build_preprocessing_pipeline,
    select_available_features,
)


def test_assert_no_leakage_rejects_known_leakage_columns():
    for col in ("events", "outcome_class", "woba_value", "delta_home_win_exp"):
        with pytest.raises(LeakageError):
            assert_no_leakage(["launch_speed", col])


def test_assert_no_leakage_allows_clean_feature_list():
    assert_no_leakage(["launch_speed", "launch_angle", "bb_type", "stand"])


def test_select_available_features_drops_missing_and_sparse_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [90.0, 91.0, 92.0, 93.0],
            "launch_angle": [10.0, None, None, None],  # 25% present -> below threshold
            "bb_type": ["line_drive"] * 4,
            "stand": ["R"] * 4,
        }
    )
    numeric, categorical = select_available_features(df)
    assert "launch_speed" in numeric
    assert "launch_angle" not in numeric  # too sparse
    assert "spray_angle_approx" not in numeric  # absent entirely
    assert "bb_type" in categorical
    assert "stand" in categorical
    assert "venue" not in categorical  # absent entirely


def test_selected_features_never_include_leakage_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [90.0],
            "bb_type": ["line_drive"],
            "stand": ["R"],
            "events": ["single"],
            "outcome_class": ["single"],
        }
    )
    numeric, categorical = select_available_features(df)
    assert "events" not in numeric + categorical
    assert "outcome_class" not in numeric + categorical


def test_preprocessing_pipeline_fits_and_transforms():
    df = pd.DataFrame(
        {
            "launch_speed": [90.0, 100.0, None, 85.0],
            "launch_angle": [10.0, 25.0, 15.0, -5.0],
            "bb_type": ["line_drive", "fly_ball", "ground_ball", None],
            "stand": ["R", "L", "R", "R"],
        }
    )
    pipeline = build_preprocessing_pipeline(["launch_speed", "launch_angle"], ["bb_type", "stand"])
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 2  # numeric cols + at least one one-hot column


def test_preprocessing_pipeline_handles_unseen_categories_at_predict_time():
    train_df = pd.DataFrame(
        {"launch_speed": [90.0, 95.0], "bb_type": ["line_drive", "fly_ball"], "stand": ["R", "L"]}
    )
    pipeline = build_preprocessing_pipeline(["launch_speed"], ["bb_type", "stand"])
    pipeline.fit(train_df)

    unseen_df = pd.DataFrame({"launch_speed": [88.0], "bb_type": ["popup"], "stand": ["R"]})
    transformed = pipeline.transform(unseen_df)  # must not raise
    assert transformed.shape[0] == 1
