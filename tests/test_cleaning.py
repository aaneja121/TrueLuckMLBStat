from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.data.clean_batted_balls import (
    CleaningError,
    add_spray_features,
    build_event_id,
    clean_batted_balls,
    ensure_optional_columns,
    validate_required_columns,
)


def test_empty_input_returns_empty_output():
    result = clean_batted_balls(pd.DataFrame())
    assert result.empty


def test_missing_required_columns_raises_clear_error():
    df = pd.DataFrame({"launch_speed": [90.0]})
    with pytest.raises(CleaningError, match="missing required column"):
        validate_required_columns(df)
    with pytest.raises(CleaningError):
        clean_batted_balls(df)


def test_missing_optional_columns_are_added_as_null_without_crashing():
    df = pd.DataFrame(
        {
            "game_pk": [1],
            "game_date": ["2022-04-05"],
            "at_bat_number": [1],
            "pitch_number": [1],
            "batter": [1],
            "events": ["single"],
        }
    )
    out = ensure_optional_columns(df)
    assert "launch_speed" in out.columns
    assert out["launch_speed"].isna().all()

    cleaned = clean_batted_balls(df)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "events"] == "single"


def test_duplicate_records_are_removed(raw_statcast_df: pd.DataFrame):
    cleaned = clean_batted_balls(raw_statcast_df)
    # game_pk=3, at_bat_number=2, pitch_number=1 appears twice in the fixture
    dup_rows = cleaned[(cleaned["game_pk"] == 3) & (cleaned["at_bat_number"] == 2)]
    assert len(dup_rows) == 1


def test_event_id_is_deterministic():
    df = pd.DataFrame({"game_pk": [100, 100], "at_bat_number": [5, 5], "pitch_number": [2, 2]})
    ids = build_event_id(df)
    assert ids.iloc[0] == ids.iloc[1]
    assert ids.iloc[0] == "100-5-2"


def test_ambiguous_events_are_kept_but_excluded_from_training(raw_statcast_df: pd.DataFrame):
    cleaned = clean_batted_balls(raw_statcast_df)
    fc_rows = cleaned[cleaned["events"] == "fielders_choice"]
    error_rows = cleaned[cleaned["events"] == "field_error"]
    assert len(fc_rows) == 1
    assert len(error_rows) == 1
    assert bool(fc_rows.iloc[0]["eligible_for_training"]) is False
    assert bool(error_rows.iloc[0]["eligible_for_training"]) is False
    assert fc_rows.iloc[0]["outcome_class"] is pd.NA or pd.isna(fc_rows.iloc[0]["outcome_class"])


def test_only_eligible_events_survive_cleaning(raw_statcast_df: pd.DataFrame):
    cleaned = clean_batted_balls(raw_statcast_df)
    assert "strikeout" not in cleaned["events"].to_numpy()
    assert "walk" not in cleaned["events"].to_numpy()


def test_spray_features_are_stable_for_identical_inputs():
    df = pd.DataFrame(
        {
            "hc_x": [130.0, 130.0, None],
            "hc_y": [150.0, 150.0, 100.0],
            "stand": ["R", "R", "L"],
        }
    )
    out = add_spray_features(df)
    assert out.loc[0, "spray_angle_approx"] == out.loc[1, "spray_angle_approx"]
    assert out.loc[0, "spray_sector"] == out.loc[1, "spray_sector"]
    assert pd.isna(out.loc[2, "spray_angle_approx"])
    assert out.loc[2, "spray_sector"] is None or pd.isna(out.loc[2, "spray_sector"])


def test_invalid_game_date_marks_ineligible_but_keeps_row(raw_statcast_df: pd.DataFrame):
    cleaned = clean_batted_balls(raw_statcast_df)
    bad_date_rows = cleaned[cleaned["game_date"] == "not-a-date"]
    assert len(bad_date_rows) == 1
    assert bool(bad_date_rows.iloc[0]["eligible_for_training"]) is False
    assert bad_date_rows.iloc[0]["training_exclusion_reason"] == "invalid_game_date"
    assert pd.isna(bad_date_rows.iloc[0]["season"])


def test_logs_and_output_row_counts_are_consistent(raw_statcast_df: pd.DataFrame):
    cleaned = clean_batted_balls(raw_statcast_df)
    assert len(cleaned) == int(cleaned["is_eligible"].sum())
    assert cleaned["eligible_for_training"].sum() <= len(cleaned)
