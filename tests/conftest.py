"""Shared synthetic fixtures. No test in this suite requires network access."""

from __future__ import annotations

import pandas as pd
import pytest


def _row(
    game_pk: int,
    at_bat_number: int,
    pitch_number: int,
    game_date: str,
    events: str | None,
    *,
    launch_speed: float | None = 90.0,
    launch_angle: float | None = 15.0,
    hit_distance_sc: float | None = 250.0,
    hc_x: float | None = 130.0,
    hc_y: float | None = 150.0,
    bb_type: str | None = "line_drive",
    stand: str | None = "R",
    venue: str | None = "Synthetic Park",
    description: str | None = None,
    batter: int = 1,
) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "batter": batter,
        "pitcher": 999,
        "player_name": "Test Player",
        "home_team": "AAA",
        "away_team": "BBB",
        "venue": venue,
        "inning": 3,
        "inning_topbot": "Top",
        "outs_when_up": 1,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "events": events,
        "description": description,
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "hit_distance_sc": hit_distance_sc,
        "hit_location": 7,
        "hc_x": hc_x,
        "hc_y": hc_y,
        "bb_type": bb_type,
        "stand": stand,
        "p_throws": "R",
        "if_fielding_alignment": "Standard",
        "of_fielding_alignment": "Standard",
        "estimated_ba_using_speedangle": 0.5,
        "estimated_woba_using_speedangle": 0.6,
        "woba_value": 1.0,
        "delta_home_win_exp": 0.01,
        "bat_speed": 70.0,
        "sprint_speed": 27.0,
    }


@pytest.fixture
def raw_statcast_rows() -> list[dict]:
    """Synthetic raw Statcast-shaped rows spanning eligible/ineligible events."""
    return [
        _row(1, 1, 1, "2022-04-05", "single", hc_x=140.0, hc_y=160.0),
        _row(1, 2, 1, "2022-04-05", "double", hc_x=90.0, hc_y=120.0),
        _row(1, 3, 1, "2022-04-05", "triple", hc_x=170.0, hc_y=100.0),
        _row(1, 4, 1, "2022-04-05", "home_run", hc_x=125.0, hc_y=50.0),
        _row(1, 5, 1, "2022-04-05", "field_out"),
        _row(2, 1, 1, "2023-05-10", "strikeout", launch_speed=None, launch_angle=None),
        _row(2, 2, 1, "2023-05-10", "walk", launch_speed=None, launch_angle=None),
        _row(2, 3, 1, "2023-05-10", "field_error"),
        _row(2, 4, 1, "2023-05-10", "fielders_choice"),
        _row(2, 5, 1, "2023-05-10", "fielders_choice_out"),
        _row(3, 1, 1, "2024-06-01", "single", launch_speed=None),  # missing contact data
        _row(3, 2, 1, "2024-06-01", "single"),
        _row(3, 2, 1, "2024-06-01", "single"),  # exact duplicate of the row above
        _row(3, 3, 1, "not-a-date", "double"),  # malformed date
    ]


@pytest.fixture
def raw_statcast_df(raw_statcast_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(raw_statcast_rows)
