"""Tests for the Contact Luck v0.5 weather physics module.

No test touches the network -- pure-function unit tests only.
"""

from __future__ import annotations

import math

import pytest

from mlb_luck_score.data.weather_physics import (
    DEFAULT_REFERENCE_AIR_DENSITY_KG_M3,
    WindObservation,
    fahrenheit_to_celsius,
    inhg_to_hpa,
    moist_air_density_kg_m3,
    mph_to_mps,
    parse_wind_text,
    pressure_at_elevation_hpa,
    saturation_vapor_pressure_hpa,
    wind_relative_components,
    zero_wind_components,
)

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def test_fahrenheit_to_celsius_freezing():
    assert fahrenheit_to_celsius(32.0) == pytest.approx(0.0)


def test_fahrenheit_to_celsius_boiling():
    assert fahrenheit_to_celsius(212.0) == pytest.approx(100.0)


def test_fahrenheit_to_celsius_missing():
    assert fahrenheit_to_celsius(None) is None
    assert fahrenheit_to_celsius(float("nan")) is None


def test_mph_to_mps():
    assert mph_to_mps(10.0) == pytest.approx(4.4704)
    assert mph_to_mps(None) is None


def test_inhg_to_hpa_standard_pressure():
    # Standard sea-level pressure: 29.92 inHg ~ 1013.2 hPa
    assert inhg_to_hpa(29.92) == pytest.approx(1013.2, abs=0.5)
    assert inhg_to_hpa(None) is None


# ---------------------------------------------------------------------------
# Moist-air density
# ---------------------------------------------------------------------------


def test_saturation_vapor_pressure_at_zero_c():
    # Well-known reference value: es(0C) ~ 6.11 hPa
    assert saturation_vapor_pressure_hpa(0.0) == pytest.approx(6.11, abs=0.02)


def test_moist_air_density_matches_standard_atmosphere_at_sea_level_dry():
    rho = moist_air_density_kg_m3(15.0, 1013.25, 0.0)
    assert rho == pytest.approx(DEFAULT_REFERENCE_AIR_DENSITY_KG_M3, abs=0.001)


def test_moist_air_density_humid_air_is_less_dense_than_dry_air():
    # Water vapor is less dense than dry air at the same T/P -- humid air is
    # LESS dense than dry air, a well-known (if counterintuitive) fact.
    dry = moist_air_density_kg_m3(30.0, 1013.25, 0.0)
    humid = moist_air_density_kg_m3(30.0, 1013.25, 100.0)
    assert humid < dry


def test_moist_air_density_missing_inputs_returns_none():
    assert moist_air_density_kg_m3(None, 1013.25, 50.0) is None
    assert moist_air_density_kg_m3(20.0, None, 50.0) is None
    assert moist_air_density_kg_m3(20.0, 1013.25, None) is None
    assert moist_air_density_kg_m3(float("nan"), 1013.25, 50.0) is None


def test_moist_air_density_rejects_invalid_humidity():
    with pytest.raises(ValueError, match="relative_humidity_pct"):
        moist_air_density_kg_m3(20.0, 1013.25, 150.0)
    with pytest.raises(ValueError, match="relative_humidity_pct"):
        moist_air_density_kg_m3(20.0, 1013.25, -1.0)


def test_moist_air_density_boundary_humidity_values_do_not_raise():
    assert moist_air_density_kg_m3(20.0, 1013.25, 0.0) is not None
    assert moist_air_density_kg_m3(20.0, 1013.25, 100.0) is not None


def test_pressure_at_elevation_decreases_with_altitude():
    sea_level = pressure_at_elevation_hpa(1013.25, 0.0, 15.0)
    coors_elevation = pressure_at_elevation_hpa(1013.25, 1609.344, 15.0)
    assert sea_level == pytest.approx(1013.25, abs=0.01)
    assert coors_elevation < sea_level
    # Real-world sanity: Coors Field's actual station pressure is commonly
    # cited around 830-845 hPa.
    assert 800.0 < coors_elevation < 860.0


def test_pressure_at_elevation_missing_inputs_returns_none():
    assert pressure_at_elevation_hpa(None, 100.0, 15.0) is None
    assert pressure_at_elevation_hpa(1013.25, None, 15.0) is None
    assert pressure_at_elevation_hpa(1013.25, 100.0, None) is None


# ---------------------------------------------------------------------------
# Wind-text parsing (MLB Stats API convention)
# ---------------------------------------------------------------------------


def test_parse_wind_text_out_to_center():
    w = parse_wind_text("8 mph, Out To CF")
    assert w is not None
    assert w.movement_bearing_degrees == 0.0
    assert w.speed_mps == pytest.approx(8 * 0.44704)
    assert w.parse_status == "ok"


def test_parse_wind_text_in_from_center():
    w = parse_wind_text("10 mph, In From CF")
    assert w is not None
    assert w.movement_bearing_degrees == pytest.approx(
        -180.0
    ) or w.movement_bearing_degrees == pytest.approx(180.0)


def test_parse_wind_text_out_to_left_and_right():
    left = parse_wind_text("5 mph, Out To LF")
    right = parse_wind_text("5 mph, Out To RF")
    assert left is not None and right is not None
    assert left.movement_bearing_degrees == -45.0
    assert right.movement_bearing_degrees == 45.0


def test_parse_wind_text_left_to_right_crosswind():
    w = parse_wind_text("6 mph, L To R")
    assert w is not None
    assert w.movement_bearing_degrees == 90.0


def test_parse_wind_text_right_to_left_crosswind():
    w = parse_wind_text("6 mph, R To L")
    assert w is not None
    assert w.movement_bearing_degrees == -90.0


def test_parse_wind_text_calm():
    w = parse_wind_text("0 mph, None")
    assert w is not None
    assert w.speed_mps == 0.0
    assert w.movement_bearing_degrees is None
    assert w.parse_status == "calm"


def test_parse_wind_text_missing_returns_none():
    assert parse_wind_text(None) is None
    assert parse_wind_text("") is None


def test_parse_wind_text_unparseable_direction():
    w = parse_wind_text("5 mph, Sideways")
    assert w is not None
    assert w.parse_status == "unparseable"
    assert w.movement_bearing_degrees is None


def test_parse_wind_text_malformed_string():
    w = parse_wind_text("not a wind string")
    assert w is not None
    assert w.parse_status == "unparseable"
    assert w.speed_mps == 0.0


def test_in_from_and_out_to_are_opposite_bearings():
    for segment in ("LF", "CF", "RF"):
        out_bearing = parse_wind_text(f"5 mph, Out To {segment}").movement_bearing_degrees
        in_bearing = parse_wind_text(f"5 mph, In From {segment}").movement_bearing_degrees
        diff = abs(((out_bearing - in_bearing + 180) % 360) - 180)
        assert diff == pytest.approx(180.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Wind-relative decomposition
# ---------------------------------------------------------------------------


def test_wind_blowing_out_to_center_is_pure_following_for_ball_to_center():
    wind = parse_wind_text("10 mph, Out To CF")
    components = wind_relative_components(wind, 0.0)
    assert components is not None
    assert components.following_wind_mps == pytest.approx(wind.speed_mps)
    assert components.headwind_mps == pytest.approx(-wind.speed_mps)
    assert components.crosswind_mps == pytest.approx(0.0, abs=1e-9)


def test_wind_blowing_in_from_center_is_pure_headwind_for_ball_to_center():
    wind = parse_wind_text("10 mph, In From CF")
    components = wind_relative_components(wind, 0.0)
    assert components is not None
    assert components.following_wind_mps == pytest.approx(-wind.speed_mps)
    assert components.headwind_mps == pytest.approx(wind.speed_mps)
    assert components.crosswind_mps == pytest.approx(0.0, abs=1e-9)


def test_left_to_right_crosswind_is_pure_crosswind_for_ball_to_center():
    wind = parse_wind_text("6 mph, L To R")
    components = wind_relative_components(wind, 0.0)
    assert components is not None
    assert components.following_wind_mps == pytest.approx(0.0, abs=1e-9)
    assert components.crosswind_mps == pytest.approx(wind.speed_mps)


def test_right_to_left_crosswind_is_negative_crosswind_for_ball_to_center():
    wind = parse_wind_text("6 mph, R To L")
    components = wind_relative_components(wind, 0.0)
    assert components is not None
    assert components.crosswind_mps == pytest.approx(-wind.speed_mps)


def test_wind_component_magnitude_splits_evenly_at_45_degrees():
    wind = parse_wind_text("10 mph, Out To CF")  # bearing 0
    components = wind_relative_components(wind, 45.0)  # ball to RF line
    assert components is not None
    expected = wind.speed_mps * math.cos(math.radians(45.0))
    assert components.following_wind_mps == pytest.approx(expected)
    assert components.crosswind_mps == pytest.approx(-expected)


def test_zero_wind_speed_gives_zero_components_via_helper():
    components = zero_wind_components()
    assert components.following_wind_mps == 0.0
    assert components.headwind_mps == 0.0
    assert components.crosswind_mps == 0.0


def test_wind_relative_components_none_when_wind_none():
    assert wind_relative_components(None, 0.0) is None


def test_wind_relative_components_none_when_spray_angle_missing():
    wind = parse_wind_text("10 mph, Out To CF")
    assert wind_relative_components(wind, None) is None
    assert wind_relative_components(wind, float("nan")) is None


def test_wind_relative_components_none_when_direction_unknown():
    calm = parse_wind_text("0 mph, None")
    assert wind_relative_components(calm, 0.0) is None


def test_wind_relative_components_stadium_orientation_rotation_consistency():
    # A wind vector "Out To RF" (bearing +45) relative to a ball hit exactly
    # to RF (spray_angle=45) must be pure following wind (rotation cancels).
    wind = parse_wind_text("8 mph, Out To RF")
    components = wind_relative_components(wind, 45.0)
    assert components is not None
    assert components.following_wind_mps == pytest.approx(wind.speed_mps)
    assert components.crosswind_mps == pytest.approx(0.0, abs=1e-9)


def test_wind_observation_dataclass_direct_construction():
    # Exercises the dataclass directly (not just via parse_wind_text), as
    # used by mlb_luck_score.data.join_weather_features.
    wind = WindObservation(
        speed_mps=5.0, movement_bearing_degrees=10.0, raw_text="", parse_status="ok"
    )
    components = wind_relative_components(wind, 10.0)
    assert components is not None
    assert components.following_wind_mps == pytest.approx(5.0)
