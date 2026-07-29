from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.features.build_contact_features import (
    OPPORTUNITY_CATEGORICAL_FEATURES,
    OPPORTUNITY_NUMERIC_FEATURES,
    OPPORTUNITY_TARGET_COLUMN,
    LeakageError,
    add_opportunity_target,
    add_outfield_opportunity_features,
    assert_no_leakage,
    build_preprocessing_pipeline,
    select_available_features,
    select_opportunity_features,
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


# ---------------------------------------------------------------------------
# Version 0.7A: outfield-opportunity features
# ---------------------------------------------------------------------------


def test_opportunity_features_are_not_leakage_columns():
    assert set(OPPORTUNITY_NUMERIC_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp"}
    )
    assert set(OPPORTUNITY_CATEGORICAL_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp"}
    )


def test_opportunity_features_never_include_responsible_outfielder_id():
    # The opportunity model represents an AVERAGE outfielder's difficulty --
    # training on fielder identity would make it encode that fielder's
    # actual skill instead. See select_opportunity_features's docstring.
    assert "responsible_outfielder_id" not in OPPORTUNITY_NUMERIC_FEATURES
    assert "responsible_outfielder_id" not in OPPORTUNITY_CATEGORICAL_FEATURES


def test_add_opportunity_target_maps_out_to_one():
    df = pd.DataFrame({"outcome_class": ["out", "single", "double", None]})
    out = add_opportunity_target(df)
    assert out[OPPORTUNITY_TARGET_COLUMN].tolist()[:3] == [1, 0, 0]
    assert pd.isna(out[OPPORTUNITY_TARGET_COLUMN].iloc[3])


def test_add_outfield_opportunity_features_is_noop_without_required_columns():
    df = pd.DataFrame({"bb_type": ["fly_ball"]})
    result = add_outfield_opportunity_features(df)
    pd.testing.assert_frame_equal(df, result)


def test_add_outfield_opportunity_features_computes_hang_time_and_landing():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0, None],
            "launch_angle": [30.0, 20.0],
            "hit_distance_sc": [350.0, 200.0],
            "spray_angle_approx": [0.0, -20.0],
            "outcome_class": ["out", "single"],
            "assigned_outfield_position": pd.array([8, pd.NA], dtype="Int64"),
        }
    )
    out = add_outfield_opportunity_features(df)
    assert out["estimated_hang_time_s"].iloc[0] > 0
    assert pd.isna(out["estimated_hang_time_s"].iloc[1])  # missing launch_speed
    assert out["landing_y_ft"].iloc[0] == pytest.approx(350.0)
    assert out[OPPORTUNITY_TARGET_COLUMN].tolist() == [1, 0]
    assert out["assigned_outfield_position"].iloc[0] == "8"
    assert pd.isna(out["assigned_outfield_position"].iloc[1])


def test_select_opportunity_features_drops_missing_and_sparse_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0, 96.0, 97.0, 98.0],
            "estimated_hang_time_s": [3.5, None, None, None],  # 25% present
            "bb_type": ["fly_ball"] * 4,
        }
    )
    numeric, categorical = select_opportunity_features(df)
    assert "launch_speed" in numeric
    assert "estimated_hang_time_s" not in numeric  # too sparse
    assert "landing_x_ft" not in numeric  # absent entirely
    assert "bb_type" in categorical
    assert "of_fielding_alignment" not in categorical  # absent entirely


def test_select_opportunity_features_never_include_leakage_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0],
            "bb_type": ["fly_ball"],
            "events": ["field_out"],
            "outcome_class": ["out"],
        }
    )
    numeric, categorical = select_opportunity_features(df)
    assert "events" not in numeric + categorical
    assert "outcome_class" not in numeric + categorical
