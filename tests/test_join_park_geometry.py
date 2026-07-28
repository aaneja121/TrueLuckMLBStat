"""Tests for the Contact Luck v0.4 per-play park-geometry join pipeline.

No test touches the network -- synthetic venue-joined-shaped fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.data.join_park_geometry import (
    GEOMETRY_STATUS_MISSING_SPRAY_ANGLE,
    GEOMETRY_STATUS_MISSING_VENUE_ID,
    GEOMETRY_STATUS_MISSING_VENUE_METADATA,
    GEOMETRY_STATUS_OK,
    GEOMETRY_STATUS_OUTSIDE_RANGE,
    GEOMETRY_STATUS_TEMPORARY_VENUE,
    build_geometry_join_report,
    join_park_geometry,
)
from mlb_luck_score.data.park_geometry import TEMPORARY_OR_SPECIAL_VENUE_IDS


def _row(**overrides) -> dict:
    base = {
        "game_pk": 1,
        "game_date": "2021-06-01",
        "venue_id": 2.0,  # Camden Yards, pre-2022 config
        "venue_name": "Oriole Park at Camden Yards",
        "has_venue_metadata": True,
        "spray_angle_approx": -45.0,  # left field line
        "hit_distance_sc": 340,
        "bb_type": "fly_ball",
        "outcome_class": "home_run",
        "eligible_for_training": True,
        "season": 2021,
    }
    base.update(overrides)
    return base


@pytest.fixture
def venue_joined_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(game_pk=1, spray_angle_approx=-45.0, hit_distance_sc=340),  # near LF wall (330)
            _row(game_pk=2, spray_angle_approx=0.0, hit_distance_sc=250),  # deep CF, well short
            _row(game_pk=3, spray_angle_approx=60.0, hit_distance_sc=300),  # outside +-45 range
            _row(game_pk=4, spray_angle_approx=np.nan, hit_distance_sc=300),  # missing spray angle
            # has_venue_metadata=False with a non-null venue_id -- a hypothetical case
            # distinct from a genuinely missing venue_id (unmatched left-join row).
            _row(game_pk=5, has_venue_metadata=False, venue_id=2.0),
            _row(game_pk=6, venue_id=-1.0, venue_name="MLB Field of Dreams (Dyersville, Iowa)"),
            _row(game_pk=7, hit_distance_sc=None),  # missing hit distance
            _row(game_pk=8, has_venue_metadata=False, venue_id=np.nan),  # genuinely unmatched row
        ]
    )


def test_join_preserves_every_row(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    assert len(joined) == len(venue_joined_df)


def test_join_produces_no_duplicate_rows(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    assert joined["game_pk"].tolist() == venue_joined_df["game_pk"].tolist()
    assert joined["game_pk"].is_unique


def test_ok_status_and_wall_lookup_for_normal_row(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 1].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_OK
    assert row["has_park_geometry"]
    assert row["wall_distance_in_spray_direction"] == 333.0  # Camden Yards LF line, pre-2022
    assert row["wall_segment_label"] == "left_field_line"


def test_distance_margin_calculation(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 1].iloc[0]
    expected_margin = 340 - 333.0
    assert row["projected_distance_to_wall_margin"] == pytest.approx(expected_margin)
    assert row["absolute_distance_to_wall"] == pytest.approx(abs(expected_margin))
    assert bool(row["projected_beyond_wall"]) is True
    assert bool(row["near_wall_10ft"]) is True  # margin=7ft
    assert bool(row["near_wall_5ft"]) is False


def test_deep_center_field_short_of_wall_not_beyond(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 2].iloc[0]
    assert row["has_park_geometry"]
    assert row["wall_distance_in_spray_direction"] == 400.0
    assert row["projected_distance_to_wall_margin"] == pytest.approx(250 - 400.0)
    assert bool(row["projected_beyond_wall"]) is False
    assert bool(row["near_wall_20ft"]) is False


def test_outside_modeled_angular_range_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 3].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_OUTSIDE_RANGE
    assert not row["has_park_geometry"]
    assert pd.isna(row["wall_distance_in_spray_direction"])
    assert pd.isna(row["projected_distance_to_wall_margin"])


def test_missing_spray_angle_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 4].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_MISSING_SPRAY_ANGLE
    assert not row["has_park_geometry"]


def test_missing_venue_metadata_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 5].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_MISSING_VENUE_METADATA
    assert not row["has_park_geometry"]
    assert not bool(row["geometry_uncertain"])  # explicitly not "uncertain" -- just not applicable


def test_genuinely_missing_venue_id_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 8].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_MISSING_VENUE_ID
    assert not row["has_park_geometry"]


def test_field_of_dreams_is_temporary_venue_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 6].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_TEMPORARY_VENUE
    assert not row["has_park_geometry"]
    assert bool(row["temporary_or_special_venue"]) is True
    assert not bool(row["geometry_uncertain"])  # temporary venues are excluded from "uncertain"


def test_missing_hit_distance_still_resolves_geometry_but_no_margin(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    row = joined.loc[joined["game_pk"] == 7].iloc[0]
    assert row["geometry_status"] == GEOMETRY_STATUS_OK
    assert row["has_park_geometry"]
    assert not pd.isna(row["wall_distance_in_spray_direction"])
    assert pd.isna(row["projected_distance_to_wall_margin"])
    assert pd.isna(row["near_wall_5ft"])  # unknown, not False


def test_missing_venue_id_column_raises():
    df = pd.DataFrame({"game_date": ["2021-01-01"], "spray_angle_approx": [0.0]})
    with pytest.raises(ValueError, match="has_venue_metadata"):
        join_park_geometry(df)


def test_temporary_or_special_venue_flag_independent_of_other_status(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    non_temp_rows = joined.loc[joined["game_pk"] != 6]
    assert not non_temp_rows["temporary_or_special_venue"].any()


def test_all_temporary_venue_ids_are_covered_by_the_special_set():
    assert -1 in TEMPORARY_OR_SPECIAL_VENUE_IDS


def test_coverage_report_counts_add_up(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    report = build_geometry_join_report(joined)
    assert report.total_rows == len(venue_joined_df)
    assert sum(report.counts_by_status.values()) == report.total_rows
    assert report.rows_with_geometry == report.counts_by_status.get(GEOMETRY_STATUS_OK, 0)
    assert report.coverage_rate == pytest.approx(report.rows_with_geometry / report.total_rows)


def test_coverage_report_by_season_and_venue(venue_joined_df: pd.DataFrame):
    joined = join_park_geometry(venue_joined_df)
    report = build_geometry_join_report(joined)
    assert 2021 in report.counts_by_season or "2021" in report.counts_by_season
    assert any("Camden Yards" in k for k in report.counts_by_venue)


def test_high_wall_indicator_uses_config_specific_height():
    # Camden Yards left-field wall height differs pre/post 2022 (7ft vs 13ft);
    # neither crosses the 15ft high-wall threshold, but Fenway's Green Monster does.
    df = pd.DataFrame(
        [
            _row(game_pk=10, venue_id=3.0, venue_name="Fenway Park", spray_angle_approx=-45.0),
        ]
    )
    joined = join_park_geometry(df)
    row = joined.iloc[0]
    assert row["wall_height_in_spray_direction"] == pytest.approx(37.17)
    assert bool(row["high_wall_indicator"]) is True


def test_geometry_config_id_reflects_effective_date():
    df = pd.DataFrame(
        [
            _row(game_pk=1, game_date="2021-06-01", venue_id=2.0, spray_angle_approx=-45.0),
            _row(game_pk=2, game_date="2023-06-01", venue_id=2.0, spray_angle_approx=-45.0),
        ]
    )
    joined = join_park_geometry(df)
    assert (
        joined.loc[joined["game_pk"] == 1, "geometry_config_id"].iloc[0] == "camden_yards_pre2022"
    )
    assert (
        joined.loc[joined["game_pk"] == 2, "geometry_config_id"].iloc[0] == "camden_yards_2022_2024"
    )
