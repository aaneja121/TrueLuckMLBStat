"""Tests for the Version 0.8 sprint-speed join. No test touches the
network -- this module never does."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_luck_score.data.join_sprint_speed import (
    JoinSprintSpeedError,
    build_sprint_speed_join_report,
    join_sprint_speed,
    load_sprint_speed,
)


def _cleaned_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": [1, 2, 3, 4],
            "season": [2021, 2021, 2022, 2022],
            "sprint_speed": [None, None, None, None],  # pre-existing all-null optional column
            "hp_to_1b": [None, None, None, None],
        }
    )


def _sprint_speed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "season": [2021, 2021, 2022],
            "sprint_speed": [28.5, 25.0, 29.9],
            "hp_to_1b": [4.1, 4.4, 4.0],
        }
    )


# -- load_sprint_speed --------------------------------------------------------


def test_load_sprint_speed_raises_clear_error_for_missing_season(tmp_path: Path):
    with pytest.raises(JoinSprintSpeedError, match="season 2021"):
        load_sprint_speed(tmp_path, [2021])


def test_load_sprint_speed_tags_each_row_with_its_season(tmp_path: Path):
    for season, ids in ((2021, [1, 2]), (2022, [3])):
        pd.DataFrame(
            {"player_id": ids, "sprint_speed": [27.0] * len(ids), "hp_to_1b": [4.2] * len(ids)}
        ).to_parquet(tmp_path / f"sprint_speed_{season}.parquet", index=False)
    combined = load_sprint_speed(tmp_path, [2021, 2022])
    assert sorted(combined["season"].tolist()) == [2021, 2021, 2022]


def test_load_sprint_speed_deduplicates_player_season_rows(tmp_path: Path):
    pd.DataFrame(
        {"player_id": [1, 1], "sprint_speed": [27.0, 28.0], "hp_to_1b": [4.2, 4.1]}
    ).to_parquet(tmp_path / "sprint_speed_2021.parquet", index=False)
    combined = load_sprint_speed(tmp_path, [2021])
    assert len(combined) == 1


# -- join_sprint_speed ---------------------------------------------------------


def test_join_matches_by_batter_and_season():
    joined = join_sprint_speed(_cleaned_df(), _sprint_speed_df())
    row = joined.loc[joined["batter"] == 1].iloc[0]
    assert row["sprint_speed"] == pytest.approx(28.5)
    assert row["hp_to_1b"] == pytest.approx(4.1)
    assert bool(row["has_sprint_speed"])


def test_join_preserves_unmatched_rows_with_false_flag_and_nulls():
    joined = join_sprint_speed(_cleaned_df(), _sprint_speed_df())
    row = joined.loc[joined["batter"] == 4].iloc[0]
    assert not bool(row["has_sprint_speed"])
    assert pd.isna(row["sprint_speed"])
    assert pd.isna(row["hp_to_1b"])


def test_join_preserves_row_count():
    joined = join_sprint_speed(_cleaned_df(), _sprint_speed_df())
    assert len(joined) == len(_cleaned_df())


def test_join_replaces_preexisting_all_null_sprint_speed_column_cleanly():
    # No sprint_speed_x/sprint_speed_y collision columns.
    joined = join_sprint_speed(_cleaned_df(), _sprint_speed_df())
    assert "sprint_speed_x" not in joined.columns
    assert "sprint_speed_y" not in joined.columns


def test_join_does_not_match_across_seasons():
    # batter=3 only has a sprint-speed row for season 2022, not 2021.
    cleaned = pd.DataFrame({"batter": [3], "season": [2021]})
    joined = join_sprint_speed(cleaned, _sprint_speed_df())
    assert not bool(joined.iloc[0]["has_sprint_speed"])


# -- build_sprint_speed_join_report --------------------------------------------


def test_join_report_counts():
    joined = join_sprint_speed(_cleaned_df(), _sprint_speed_df())
    report = build_sprint_speed_join_report(joined)
    assert report.total_rows == 4
    assert report.matched_rows == 3
    assert report.unmatched_rows == 1
    assert report.match_rate == pytest.approx(0.75)
    assert report.unique_batters == 4
    assert report.unique_batters_matched == 3
