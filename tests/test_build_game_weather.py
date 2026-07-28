"""Tests for the Contact Luck v0.5 game-weather table builder.

No test touches the network -- synthetic schedule-weather/ASOS-shaped
fixtures only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.data.build_game_weather import (
    MAX_AFTER_FALLBACK_MINUTES,
    MAX_OFFSET_MINUTES_FAIR,
    ROOF_STATUS_FIXED_INDOOR,
    ROOF_STATUS_OUTDOOR_OPEN_AIR,
    ROOF_STATUS_RETRACTABLE_CLOSED,
    ROOF_STATUS_RETRACTABLE_OPEN,
    ROOF_STATUS_UNKNOWN,
    WEATHER_STATUS_EFFECTIVE_UNAVAILABLE_INDOOR,
    WEATHER_STATUS_MISSING_START_TIME,
    WEATHER_STATUS_NO_OBSERVATION,
    WEATHER_STATUS_OK,
    WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE,
    GameWeatherError,
    build_game_weather_row,
    build_game_weather_table,
    classify_roof_status,
    find_nearest_observation,
)
from mlb_luck_score.data.venue_environment import get_venue_environment

FENWAY = get_venue_environment(3)  # Open
TROPICANA = get_venue_environment(12)  # roof_type "Dome" would come from metadata
ROGERS = get_venue_environment(14)  # Retractable


def _obs_df(times, **cols) -> pd.DataFrame:
    data = {"valid": pd.to_datetime(times, utc=True)}
    data.update(cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Roof-status classification
# ---------------------------------------------------------------------------


def test_classify_roof_status_open():
    assert classify_roof_status("Open", "Sunny") == ROOF_STATUS_OUTDOOR_OPEN_AIR


def test_classify_roof_status_dome():
    assert classify_roof_status("Dome", "Roof Closed") == ROOF_STATUS_FIXED_INDOOR
    assert classify_roof_status("Dome", None) == ROOF_STATUS_FIXED_INDOOR


def test_classify_roof_status_retractable_closed():
    assert classify_roof_status("Retractable", "Roof Closed") == ROOF_STATUS_RETRACTABLE_CLOSED
    assert classify_roof_status("Retractable", "roof closed") == ROOF_STATUS_RETRACTABLE_CLOSED


def test_classify_roof_status_retractable_open():
    assert classify_roof_status("Retractable", "Sunny") == ROOF_STATUS_RETRACTABLE_OPEN
    assert classify_roof_status("Retractable", "Rain") == ROOF_STATUS_RETRACTABLE_OPEN


def test_classify_roof_status_never_infers_closed_from_precipitation():
    # Rain alone must NOT be interpreted as "roof closed" -- only an
    # explicit "Roof Closed" string does that.
    assert classify_roof_status("Retractable", "Rain") == ROOF_STATUS_RETRACTABLE_OPEN


def test_classify_roof_status_missing_roof_type():
    assert classify_roof_status(None, "Sunny") == ROOF_STATUS_UNKNOWN


def test_classify_roof_status_retractable_missing_condition():
    assert classify_roof_status("Retractable", None) == ROOF_STATUS_UNKNOWN


# ---------------------------------------------------------------------------
# Observation-time matching
# ---------------------------------------------------------------------------


def test_find_nearest_observation_prefers_observation_before_start():
    obs = _obs_df(
        ["2024-07-01 21:00:00", "2024-07-01 22:00:00", "2024-07-01 23:00:00"],
        tmpf=[70.0, 75.0, 78.0],
    )
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")
    match = find_nearest_observation(obs, start)
    assert match is not None
    assert match.tmpf == 75.0
    assert match.offset_minutes == pytest.approx(10.0)
    assert match.match_quality == "good"


def test_find_nearest_observation_fair_quality_beyond_one_hour():
    obs = _obs_df(["2024-07-01 20:00:00"], tmpf=[70.0])
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")  # 130 min offset
    match = find_nearest_observation(obs, start)
    assert match is not None
    assert match.match_quality == "fair"
    assert match.offset_minutes == pytest.approx(130.0)


def test_find_nearest_observation_none_beyond_max_offset():
    obs = _obs_df(["2024-07-01 10:00:00"], tmpf=[70.0])
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")  # way more than MAX_OFFSET_MINUTES_FAIR
    assert find_nearest_observation(obs, start) is None


def test_find_nearest_observation_falls_back_to_after_start():
    obs = _obs_df(["2024-07-01 22:30:00"], tmpf=[70.0])
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")  # observation 20 min AFTER start
    match = find_nearest_observation(obs, start)
    assert match is not None
    assert match.offset_minutes == pytest.approx(-20.0)  # negative = after start


def test_find_nearest_observation_after_fallback_respects_its_own_limit():
    obs = _obs_df(
        [f"2024-07-01 23:{MAX_AFTER_FALLBACK_MINUTES + 30:02d}:00"]
        if MAX_AFTER_FALLBACK_MINUTES + 30 < 60
        else ["2024-07-02 00:30:00"],
        tmpf=[70.0],
    )
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")
    assert find_nearest_observation(obs, start) is None


def test_find_nearest_observation_empty_dataframe():
    obs = pd.DataFrame({"valid": pd.to_datetime([], utc=True), "tmpf": []})
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")
    assert find_nearest_observation(obs, start) is None


def test_find_nearest_observation_boundary_at_max_offset_fair():
    obs = _obs_df(["2024-07-01 19:10:00"], tmpf=[70.0])
    start = pd.Timestamp("2024-07-01 22:10:00", tz="UTC")  # exactly MAX_OFFSET_MINUTES_FAIR
    match = find_nearest_observation(obs, start)
    assert match is not None
    assert match.offset_minutes == pytest.approx(float(MAX_OFFSET_MINUTES_FAIR))


# ---------------------------------------------------------------------------
# build_game_weather_row scenarios
# ---------------------------------------------------------------------------


def _game(**overrides) -> dict:
    base = {
        "game_pk": 1,
        "game_date": "2024-07-01",
        "venue_id": 3,
        "scheduled_start_time_utc": "2024-07-01T22:10:00Z",
        "weather_condition": "Sunny",
        "weather_temp_f": 75.0,
        "weather_wind_raw": "8 mph, Out To CF",
    }
    base.update(overrides)
    return base


def _obs():
    return _obs_df(
        ["2024-07-01 22:00:00"],
        tmpf=[75.0],
        dwpf=[65.0],
        relh=[70.0],
        drct=[230.0],
        sknt=[10.0],
        alti=[29.9],
        mslp=[1012.0],
    )


def test_outdoor_open_air_row_has_effective_conditions():
    row = build_game_weather_row(_game(), "Open", FENWAY, _obs())
    assert row["weather_status"] == WEATHER_STATUS_OK
    assert row["roof_status"] == ROOF_STATUS_OUTDOOR_OPEN_AIR
    assert row["effective_temperature_c"] is not None
    assert row["air_density_kg_m3"] is not None
    assert row["effective_wind_speed_mps"] is not None


def test_fixed_indoor_row_suppresses_effective_conditions_and_wind():
    row = build_game_weather_row(_game(venue_id=12), "Dome", TROPICANA, _obs())
    assert row["weather_status"] == WEATHER_STATUS_EFFECTIVE_UNAVAILABLE_INDOOR
    assert row["roof_status"] == ROOF_STATUS_FIXED_INDOOR
    assert row["effective_temperature_c"] is None
    assert row["effective_relative_humidity_pct"] is None
    assert row["effective_pressure_hpa"] is None
    assert row["air_density_kg_m3"] is None
    assert row["effective_wind_speed_mps"] == 0.0
    # Raw external weather IS preserved even though effective is unavailable.
    assert row["temperature_c"] is not None


def test_retractable_closed_row_suppresses_wind_but_preserves_raw():
    row = build_game_weather_row(
        _game(venue_id=14, weather_condition="Roof Closed", weather_wind_raw="0 mph, None"),
        "Retractable",
        ROGERS,
        _obs(),
    )
    assert row["roof_status"] == ROOF_STATUS_RETRACTABLE_CLOSED
    assert row["effective_wind_speed_mps"] == 0.0
    assert row["effective_temperature_c"] is None
    assert row["temperature_c"] is not None


def test_retractable_open_row_behaves_like_outdoor():
    row = build_game_weather_row(
        _game(venue_id=14, weather_condition="Cloudy"), "Retractable", ROGERS, _obs()
    )
    assert row["roof_status"] == ROOF_STATUS_RETRACTABLE_OPEN
    assert row["weather_status"] == WEATHER_STATUS_OK
    assert row["effective_temperature_c"] is not None


def test_missing_venue_environment_status():
    row = build_game_weather_row(_game(venue_id=5340), None, None, None)
    assert row["weather_status"] == WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE
    assert row["effective_temperature_c"] is None


def test_missing_start_time_status():
    row = build_game_weather_row(_game(scheduled_start_time_utc=None), "Open", FENWAY, _obs())
    assert row["weather_status"] == WEATHER_STATUS_MISSING_START_TIME


def test_no_observation_within_window_but_mlb_temp_present():
    far_obs = _obs_df(
        ["2024-06-01 12:00:00"],
        tmpf=[70.0],
        dwpf=[60],
        relh=[60],
        drct=[0],
        sknt=[5],
        alti=[29.9],
        mslp=[1012.0],
    )
    row = build_game_weather_row(_game(), "Open", FENWAY, far_obs)
    # MLB-reported temp/wind are captured (raw temperature_c is populated),
    # but no station match within the allowed window means no pressure/
    # humidity/air-density -- outdoor "ok" status requires the full
    # accounting, so this must NOT be silently marked "ok".
    assert row["weather_status"] == WEATHER_STATUS_NO_OBSERVATION
    assert row["temperature_c"] is not None
    assert row["pressure_hpa"] is None
    assert row["air_density_kg_m3"] is None


def test_no_data_at_all_gives_no_observation_status():
    game = _game(weather_condition=None, weather_temp_f=None, weather_wind_raw=None)
    empty_obs = pd.DataFrame({"valid": pd.to_datetime([], utc=True)})
    row = build_game_weather_row(game, "Open", FENWAY, empty_obs)
    assert row["weather_status"] == WEATHER_STATUS_NO_OBSERVATION


def test_station_distance_and_timezone_populated():
    row = build_game_weather_row(_game(), "Open", FENWAY, _obs())
    assert row["station_distance_km"] == pytest.approx(FENWAY.station_distance_km())
    assert row["venue_timezone"] == "America/New_York"


def test_local_start_time_is_converted_from_utc():
    row = build_game_weather_row(_game(), "Open", FENWAY, _obs())
    # 2024-07-01T22:10:00Z in America/New_York (EDT, UTC-4) is 18:10 local.
    assert "18:10:00" in row["local_start_time"]


# ---------------------------------------------------------------------------
# build_game_weather_table (row preservation, duplicate rejection)
# ---------------------------------------------------------------------------


def test_build_table_preserves_every_row():
    schedule_df = pd.DataFrame([_game(game_pk=1), _game(game_pk=2, venue_id=12)])
    table = build_game_weather_table(
        schedule_df, {FENWAY.weather_station_id: _obs()}, {1: "Open", 2: "Dome"}
    )
    assert len(table) == 2
    assert set(table["game_pk"]) == {1, 2}


def test_build_table_rejects_duplicate_game_pk():
    schedule_df = pd.DataFrame([_game(game_pk=1), _game(game_pk=1)])
    with pytest.raises(GameWeatherError, match="duplicate"):
        build_game_weather_table(schedule_df, {}, {})


def test_build_table_no_row_duplication_for_shared_station():
    # Two venues sharing the same station (e.g. LGA) must not produce
    # duplicate or cross-contaminated rows.
    schedule_df = pd.DataFrame([_game(game_pk=1, venue_id=3), _game(game_pk=2, venue_id=3)])
    table = build_game_weather_table(
        schedule_df, {FENWAY.weather_station_id: _obs()}, {1: "Open", 2: "Open"}
    )
    assert len(table) == 2
    assert table["game_pk"].is_unique
