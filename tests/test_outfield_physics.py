"""Tests for the Contact Luck v0.7A outfield-opportunity physics estimates.

Pure-function unit tests only -- no network, no real Statcast data.
"""

from __future__ import annotations

import math

import pytest

from mlb_luck_score.data.outfield_physics import (
    MIN_HANG_TIME_SECONDS,
    estimate_hang_time_seconds,
    estimate_landing_coordinates_ft,
    solve_launch_speed_for_matched_range_mph,
)


def test_hang_time_positive_for_typical_fly_ball():
    # ~95 mph, 30 degrees -- a routine fly ball.
    t = estimate_hang_time_seconds(95.0, 30.0)
    assert t > 0
    # Sanity range: real MLB fly balls hang roughly 3-6 seconds.
    assert 2.0 < t < 7.0


def test_hang_time_increases_with_launch_angle_at_fixed_speed():
    low_angle = estimate_hang_time_seconds(95.0, 15.0)
    high_angle = estimate_hang_time_seconds(95.0, 60.0)
    assert high_angle > low_angle


def test_hang_time_increases_with_speed_at_fixed_angle():
    slow = estimate_hang_time_seconds(80.0, 30.0)
    fast = estimate_hang_time_seconds(105.0, 30.0)
    assert fast > slow


def test_hang_time_floored_for_non_positive_launch_angle():
    t = estimate_hang_time_seconds(95.0, 0.0)
    assert t == MIN_HANG_TIME_SECONDS
    t_negative = estimate_hang_time_seconds(95.0, -5.0)
    assert t_negative == MIN_HANG_TIME_SECONDS


def test_hang_time_nan_propagates():
    assert math.isnan(estimate_hang_time_seconds(math.nan, 30.0))
    assert math.isnan(estimate_hang_time_seconds(95.0, math.nan))


def test_hang_time_matches_vacuum_formula_by_hand():
    # v0 = 100 mph = 44.704 m/s, theta = 45 deg -> t = 2*v0*sin(theta)/g
    v0_mps = 100.0 * 0.44704
    expected = 2.0 * v0_mps * math.sin(math.radians(45.0)) / 9.80665
    assert estimate_hang_time_seconds(100.0, 45.0) == pytest.approx(expected)


def test_landing_coordinates_center_field_is_pure_y():
    x, y = estimate_landing_coordinates_ft(350.0, 0.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(350.0)


def test_landing_coordinates_negative_angle_is_left_field_side():
    x, _y = estimate_landing_coordinates_ft(300.0, -45.0)
    assert x < 0


def test_landing_coordinates_positive_angle_is_right_field_side():
    x, _y = estimate_landing_coordinates_ft(300.0, 45.0)
    assert x > 0


def test_landing_coordinates_distance_preserved():
    x, y = estimate_landing_coordinates_ft(400.0, 30.0)
    assert math.hypot(x, y) == pytest.approx(400.0)


def test_landing_coordinates_nan_propagates():
    x, y = estimate_landing_coordinates_ft(math.nan, 0.0)
    assert math.isnan(x) and math.isnan(y)
    x, y = estimate_landing_coordinates_ft(300.0, math.nan)
    assert math.isnan(x) and math.isnan(y)


def test_solved_launch_speed_reproduces_target_distance():
    # Forward-check: plug the solved v0 back into the same vacuum range
    # formula (R = v0^2 * sin(2*theta) / g) and confirm it lands back on the
    # target distance.
    target_distance_ft = 350.0
    angle = 30.0
    v0_mph = solve_launch_speed_for_matched_range_mph(target_distance_ft, angle)
    v0_mps = v0_mph * 0.44704
    theta_rad = math.radians(angle)
    range_m = v0_mps**2 * math.sin(2.0 * theta_rad) / 9.80665
    range_ft = range_m / 0.3048
    assert range_ft == pytest.approx(target_distance_ft, rel=1e-9)


def test_solved_launch_speed_lower_for_higher_angle_below_45_degrees():
    # Both 18 and 40 degrees are below the 45-degree max-range angle, so
    # reaching the SAME distance at the higher angle requires LESS exit
    # velocity (sin(2*theta) is still increasing in this range).
    target_distance_ft = 380.0
    low_angle_speed = solve_launch_speed_for_matched_range_mph(target_distance_ft, 18.0)
    high_angle_speed = solve_launch_speed_for_matched_range_mph(target_distance_ft, 40.0)
    assert high_angle_speed < low_angle_speed


def test_matched_trajectory_higher_angle_has_greater_hang_time():
    # The core invariant the trajectory-matched perturbation check in
    # compare_near_wall_models relies on: for two trajectories reaching the
    # SAME distance, the higher-angle one spends more time in the air, even
    # though it needs less exit velocity to get there.
    target_distance_ft = 380.0
    low_angle, high_angle = 18.0, 40.0
    low_speed = solve_launch_speed_for_matched_range_mph(target_distance_ft, low_angle)
    high_speed = solve_launch_speed_for_matched_range_mph(target_distance_ft, high_angle)
    low_hang_time = estimate_hang_time_seconds(low_speed, low_angle)
    high_hang_time = estimate_hang_time_seconds(high_speed, high_angle)
    assert high_hang_time > low_hang_time


def test_solved_launch_speed_nan_for_degenerate_angle():
    assert math.isnan(solve_launch_speed_for_matched_range_mph(350.0, 0.0))
    assert math.isnan(solve_launch_speed_for_matched_range_mph(350.0, 90.0))


def test_solved_launch_speed_nan_propagates():
    assert math.isnan(solve_launch_speed_for_matched_range_mph(math.nan, 30.0))
    assert math.isnan(solve_launch_speed_for_matched_range_mph(350.0, math.nan))
