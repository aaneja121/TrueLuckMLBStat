from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.features.build_contact_features import (
    ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
    ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
    ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES,
    ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES,
    ADVANCEMENT_SPEED_CATEGORICAL_FEATURES,
    ADVANCEMENT_SPEED_NUMERIC_FEATURES,
    INFIELD_CATEGORICAL_FEATURES,
    INFIELD_NUMERIC_FEATURES,
    INFIELD_OPPORTUNITY_TARGET_COLUMN,
    INFIELD_TARGET_COLUMN,
    NEAR_WALL_CATEGORICAL_FEATURES,
    NEAR_WALL_NUMERIC_FEATURES,
    OPPORTUNITY_CATEGORICAL_FEATURES,
    OPPORTUNITY_NUMERIC_FEATURES,
    OUTFIELD_OPPORTUNITY_TARGET_COLUMN,
    LeakageError,
    OpportunityDomainRoutingError,
    add_advancement_contact_probability_features,
    add_advancement_features,
    add_infield_opportunity_features,
    add_opportunity_features_by_domain,
    add_opportunity_target,
    add_outfield_opportunity_features,
    assert_no_leakage,
    build_preprocessing_pipeline,
    select_advancement_features,
    select_available_features,
    select_infield_opportunity_features,
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
    assert out[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].tolist()[:3] == [1, 0, 0]
    assert pd.isna(out[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[3])


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
    assert out[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].tolist() == [1, 0]
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


def test_near_wall_features_extend_opportunity_features_with_wall_geometry():
    assert set(OPPORTUNITY_NUMERIC_FEATURES) <= set(NEAR_WALL_NUMERIC_FEATURES)
    assert set(OPPORTUNITY_CATEGORICAL_FEATURES) <= set(NEAR_WALL_CATEGORICAL_FEATURES)
    assert "wall_height_in_spray_direction" in NEAR_WALL_NUMERIC_FEATURES
    assert "wall_segment_label" in NEAR_WALL_CATEGORICAL_FEATURES


def test_near_wall_features_are_not_leakage_columns():
    assert_no_leakage(list(NEAR_WALL_NUMERIC_FEATURES) + list(NEAR_WALL_CATEGORICAL_FEATURES))


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


# ---------------------------------------------------------------------------
# Version 0.8: infield-opportunity features
# ---------------------------------------------------------------------------


def test_infield_features_are_not_leakage_columns():
    assert set(INFIELD_NUMERIC_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp"}
    )
    assert set(INFIELD_CATEGORICAL_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp"}
    )


def test_infield_features_never_include_responsible_infielder_id():
    assert "responsible_infielder_id" not in INFIELD_NUMERIC_FEATURES
    assert "responsible_infielder_id" not in INFIELD_CATEGORICAL_FEATURES


def test_add_infield_opportunity_features_is_noop_without_y_out():
    df = pd.DataFrame({"bb_type": ["ground_ball"]})
    result = add_infield_opportunity_features(df)
    pd.testing.assert_frame_equal(df, result)


def test_add_infield_opportunity_features_computes_context_and_copies_target():
    df = pd.DataFrame(
        {
            "y_out": pd.array([1, 0], dtype="Int64"),
            "on_1b": pd.array([123, pd.NA], dtype="Int64"),
            "assigned_infield_position": pd.array([6, pd.NA], dtype="Int64"),
        }
    )
    out = add_infield_opportunity_features(df)
    assert out["on_1b_occupied"].tolist() == [1, 0]
    assert out[INFIELD_TARGET_COLUMN].tolist() == [1, 0]
    assert out[INFIELD_OPPORTUNITY_TARGET_COLUMN].tolist() == [1, 0]
    # Version 0.10 hardening: the infield builder must NEVER write the
    # outfield's target column name -- the two domains no longer collide.
    assert "outfield_converted_to_out" not in out.columns
    assert "converted_to_out" not in out.columns
    assert out["assigned_infield_position"].iloc[0] == "6"
    assert pd.isna(out["assigned_infield_position"].iloc[1])


def test_add_infield_opportunity_features_defaults_on_1b_occupied_when_absent():
    df = pd.DataFrame({"y_out": pd.array([1], dtype="Int64")})
    out = add_infield_opportunity_features(df)
    assert out["on_1b_occupied"].tolist() == [0]


# ---------------------------------------------------------------------------
# Version 0.10 hardening: the outfield/infield target-column collision
# ---------------------------------------------------------------------------


def _combined_domain_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_class": ["out", "single", None, None],
            "y_out": pd.array([pd.NA, pd.NA, 1, 0], dtype="Int64"),
            "outfield_opportunity_eligible": [True, True, False, False],
            "infield_opportunity_eligible": [False, False, True, True],
            "launch_speed": [95.0, 90.0, 88.0, 85.0],
            "launch_angle": [30.0, 10.0, -5.0, 2.0],
            "hit_distance_sc": [350.0, 200.0, 60.0, 65.0],
            "spray_angle_approx": [0.0, -10.0, 5.0, -5.0],
        }
    )


def test_calling_both_builders_sequentially_does_not_overwrite_either_target():
    combined = _combined_domain_df()

    result = add_infield_opportunity_features(add_outfield_opportunity_features(combined))
    assert result[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[0] == 1
    assert result[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[1] == 0
    assert pd.isna(result[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[2])
    assert pd.isna(result[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[3])
    assert pd.isna(result[INFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[0])
    assert pd.isna(result[INFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[1])
    assert result[INFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[2] == 1
    assert result[INFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[3] == 0

    # Same values regardless of call order -- there is nothing left for
    # either call to clobber.
    reversed_result = add_outfield_opportunity_features(add_infield_opportunity_features(combined))
    assert reversed_result[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[0] == 1
    assert reversed_result[INFIELD_OPPORTUNITY_TARGET_COLUMN].iloc[2] == 1


def test_opportunity_features_by_domain_routes_each_row_to_at_most_one_builder():
    combined = _combined_domain_df()
    out = add_opportunity_features_by_domain(combined)

    # Outfield rows (0, 1) got the outfield builder's columns...
    assert out.loc[0, OUTFIELD_OPPORTUNITY_TARGET_COLUMN] == 1
    assert out.loc[1, OUTFIELD_OPPORTUNITY_TARGET_COLUMN] == 0
    assert out.loc[0, "estimated_hang_time_s"] > 0
    # ...and never the infield builder's target.
    assert pd.isna(out.loc[0, INFIELD_OPPORTUNITY_TARGET_COLUMN])
    assert pd.isna(out.loc[1, INFIELD_OPPORTUNITY_TARGET_COLUMN])

    # Infield rows (2, 3) got the infield builder's columns...
    assert out.loc[2, INFIELD_OPPORTUNITY_TARGET_COLUMN] == 1
    assert out.loc[3, INFIELD_OPPORTUNITY_TARGET_COLUMN] == 0
    assert "on_1b_occupied" in out.columns
    # ...and never the outfield builder's target.
    assert pd.isna(out.loc[2, OUTFIELD_OPPORTUNITY_TARGET_COLUMN])
    assert pd.isna(out.loc[3, OUTFIELD_OPPORTUNITY_TARGET_COLUMN])


def test_opportunity_features_by_domain_raises_on_both_domains_eligible():
    combined = _combined_domain_df()
    combined.loc[0, "infield_opportunity_eligible"] = True  # now eligible for BOTH domains
    with pytest.raises(OpportunityDomainRoutingError, match="BOTH outfield and infield"):
        add_opportunity_features_by_domain(combined)


def test_opportunity_features_by_domain_raises_on_duplicate_index():
    combined = _combined_domain_df()
    combined.index = [0, 0, 1, 2]
    with pytest.raises(OpportunityDomainRoutingError, match="duplicate value"):
        add_opportunity_features_by_domain(combined)


def test_opportunity_features_by_domain_raises_on_missing_eligibility_columns():
    df = pd.DataFrame({"launch_speed": [90.0]})
    with pytest.raises(OpportunityDomainRoutingError, match="requires column"):
        add_opportunity_features_by_domain(df)


def test_opportunity_features_by_domain_preserves_row_identity_and_order():
    combined = _combined_domain_df()
    shuffled = combined.sample(frac=1.0, random_state=0)  # scramble both index and row order
    out = add_opportunity_features_by_domain(shuffled)
    assert out.index.tolist() == shuffled.index.tolist()
    for idx in shuffled.index:
        assert out.loc[idx, "launch_speed"] == shuffled.loc[idx, "launch_speed"]


def test_select_infield_opportunity_features_drops_missing_and_sparse_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0, 96.0, 97.0, 98.0],
            "sprint_speed": [27.0, None, None, None],  # 25% present -> below threshold
            "stand": ["R"] * 4,
        }
    )
    numeric, categorical = select_infield_opportunity_features(df)
    assert "launch_speed" in numeric
    assert "sprint_speed" not in numeric  # too sparse
    assert "hit_distance_sc" not in numeric  # absent entirely
    assert "stand" in categorical
    assert "if_fielding_alignment" not in categorical  # absent entirely


def test_select_infield_opportunity_features_never_include_leakage_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0],
            "stand": ["R"],
            "events": ["field_out"],
            "outcome_class": ["out"],
        }
    )
    numeric, categorical = select_infield_opportunity_features(df)
    assert "events" not in numeric + categorical
    assert "outcome_class" not in numeric + categorical


# ---------------------------------------------------------------------------
# Version 0.9: batter-runner advancement features
# ---------------------------------------------------------------------------


def test_advancement_context_features_are_not_leakage_columns():
    assert set(ADVANCEMENT_CONTEXT_NUMERIC_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp", "des"}
    )
    assert set(ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES).isdisjoint(
        {"events", "outcome_class", "woba_value", "delta_home_win_exp", "des"}
    )


def test_advancement_speed_features_add_only_sprint_speed():
    assert set(ADVANCEMENT_SPEED_NUMERIC_FEATURES) - set(ADVANCEMENT_CONTEXT_NUMERIC_FEATURES) == {
        "sprint_speed"
    }
    assert ADVANCEMENT_SPEED_CATEGORICAL_FEATURES == ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES


def test_advancement_nonlinear_features_identical_to_speed_features():
    assert ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES == ADVANCEMENT_SPEED_NUMERIC_FEATURES
    assert ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES == ADVANCEMENT_SPEED_CATEGORICAL_FEATURES


def test_advancement_features_never_include_des_or_matched_clause():
    # The target label is reconstructed from des -- des and any parser
    # audit-trail column must never be a model input (task safeguard #7).
    forbidden = {
        "des",
        "advancement_matched_clause",
        "advancement_parse_status",
        "advancement_parse_failure_reason",
        "batter_final_base",
    }
    assert forbidden.isdisjoint(ADVANCEMENT_CONTEXT_NUMERIC_FEATURES)
    assert forbidden.isdisjoint(ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES)
    assert forbidden.isdisjoint(ADVANCEMENT_SPEED_NUMERIC_FEATURES)


def test_add_advancement_features_is_noop_without_required_columns():
    df = pd.DataFrame({"bb_type": ["fly_ball"]})
    result = add_advancement_features(df)
    pd.testing.assert_frame_equal(df, result)


def test_add_advancement_features_computes_hang_time_landing_and_base_occupancy():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0],
            "launch_angle": [25.0],
            "hit_distance_sc": [350.0],
            "spray_angle_approx": [0.0],
            "on_1b": pd.array([123], dtype="Int64"),
            "on_2b": pd.array([pd.NA], dtype="Int64"),
            "on_3b": pd.array([pd.NA], dtype="Int64"),
            "assigned_outfield_position": pd.array([8], dtype="Int64"),
        }
    )
    out = add_advancement_features(df)
    assert out["estimated_hang_time_s"].iloc[0] > 0
    assert out["on_1b_occupied"].tolist() == [1]
    assert out["on_2b_occupied"].tolist() == [0]
    assert out["on_3b_occupied"].tolist() == [0]
    assert out["assigned_outfield_position"].iloc[0] == "8"


def test_add_advancement_features_defaults_base_occupancy_when_absent():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0],
            "launch_angle": [25.0],
            "hit_distance_sc": [350.0],
            "spray_angle_approx": [0.0],
        }
    )
    out = add_advancement_features(df)
    assert out["on_1b_occupied"].tolist() == [0]
    assert out["on_2b_occupied"].tolist() == [0]
    assert out["on_3b_occupied"].tolist() == [0]


def test_add_advancement_contact_probability_features_attaches_five_columns():
    df = pd.DataFrame({"a": [1, 2]})
    contact_proba = pd.DataFrame(
        {
            "out": [0.2, 0.3],
            "single": [0.3, 0.2],
            "double": [0.2, 0.2],
            "triple": [0.1, 0.1],
            "home_run": [0.2, 0.2],
        }
    )
    out = add_advancement_contact_probability_features(df, contact_proba)
    assert out["contact_p_out"].tolist() == [0.2, 0.3]
    assert out["contact_p_home_run"].tolist() == [0.2, 0.2]


def test_add_advancement_contact_probability_features_rejects_mismatched_index():
    df = pd.DataFrame({"a": [1]}, index=[0])
    contact_proba = pd.DataFrame(
        {"out": [0.2], "single": [0.2], "double": [0.2], "triple": [0.2], "home_run": [0.2]},
        index=[1],
    )
    with pytest.raises(ValueError, match="identical index"):
        add_advancement_contact_probability_features(df, contact_proba)


def test_select_advancement_features_drops_missing_and_sparse_columns():
    df = pd.DataFrame(
        {
            "launch_speed": [95.0, 96.0, 97.0, 98.0],
            "sprint_speed": [27.0, None, None, None],  # 25% present -> below threshold
            "stand": ["R"] * 4,
        }
    )
    numeric, categorical = select_advancement_features(
        df,
        numeric_candidates=["launch_speed", "sprint_speed", "hit_distance_sc"],
        categorical_candidates=["stand", "if_fielding_alignment"],
    )
    assert "launch_speed" in numeric
    assert "sprint_speed" not in numeric  # too sparse
    assert "hit_distance_sc" not in numeric  # absent entirely
    assert "stand" in categorical
    assert "if_fielding_alignment" not in categorical  # absent entirely


def test_select_advancement_features_never_include_leakage_columns():
    df = pd.DataFrame({"launch_speed": [95.0], "des": ["Someone singles."], "stand": ["R"]})
    with pytest.raises(LeakageError):
        select_advancement_features(
            df, numeric_candidates=["launch_speed", "des"], categorical_candidates=["stand"]
        )
