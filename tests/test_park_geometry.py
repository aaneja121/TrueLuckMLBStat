"""Tests for the Contact Luck v0.4 reviewed park-geometry reference table.

No test touches the network -- these exercise the schema, validation,
effective-date resolution, and piecewise-linear interpolation logic in
`mlb_luck_score.data.park_geometry` against small synthetic geometry tables
(plus a few checks against the real, git-tracked reviewed table itself,
which is offline reference data, not a download).
"""

from __future__ import annotations

import pytest

from mlb_luck_score.data.clean_batted_balls import _spray_angle_degrees
from mlb_luck_score.data.park_geometry import (
    PARK_GEOMETRY_CONFIGS,
    PARK_GEOMETRY_POINTS,
    STANDARD_ANGLES,
    TEMPORARY_OR_SPECIAL_VENUE_IDS,
    ParkGeometryPoint,
    ParkGeometryValidationError,
    interpolate_wall_geometry,
    resolve_geometry_config,
    validate_geometry_points,
)

_SOURCE_KWARGS = {
    "source_name": "Synthetic test source",
    "source_reference": "https://example.invalid/test",
    "source_accessed_date": "2026-01-01",
    "review_status": "agent_sourced_pending_human_review",
}


def _pt(
    venue_id, config_id, start, end, angle, distance, height=None, label="center_field", temp=False
):
    return ParkGeometryPoint(
        venue_id=venue_id,
        venue_name=f"Venue {venue_id}",
        geometry_config_id=config_id,
        effective_start_date=start,
        effective_end_date=end,
        spray_angle_degrees=angle,
        wall_distance_feet=distance,
        wall_height_feet=height,
        segment_label=label,
        special_structure=None,
        temporary_venue=temp,
        **_SOURCE_KWARGS,
        notes="",
    )


def _standard_config(venue_id, config_id, start, end, distances, heights=(None,) * 5):
    labels = (
        "left_field_line",
        "left_center",
        "center_field",
        "right_center",
        "right_field_line",
    )
    return [
        _pt(venue_id, config_id, start, end, angle, dist, height, label)
        for angle, label, dist, height in zip(
            STANDARD_ANGLES, labels, distances, heights, strict=True
        )
    ]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_real_reviewed_table_is_valid():
    # The module already validates this at import time; re-running here
    # documents the expectation explicitly and fails loudly if it regresses.
    validate_geometry_points(PARK_GEOMETRY_POINTS)


def test_real_table_covers_no_temporary_venues():
    covered_venues = {p.venue_id for p in PARK_GEOMETRY_POINTS}
    assert covered_venues.isdisjoint(TEMPORARY_OR_SPECIAL_VENUE_IDS)


def test_validation_rejects_missing_source_metadata():
    bad_point = _pt(999, "cfg_a", "2021-01-01", None, -45.0, 330.0)
    points = list(_standard_config(999, "cfg_a", "2021-01-01", None, (330, 370, 400, 370, 330)))
    points[0] = ParkGeometryPoint(
        **{**bad_point.__dict__, "source_name": "", "source_reference": ""}
    )
    with pytest.raises(ParkGeometryValidationError, match="missing source metadata"):
        validate_geometry_points(points)


def test_validation_rejects_overlapping_configs_for_same_venue():
    cfg_a = _standard_config(999, "cfg_a", "2021-01-01", "2022-12-31", (330, 370, 400, 370, 330))
    cfg_b = _standard_config(
        999, "cfg_b", "2022-06-01", None, (335, 375, 405, 375, 335)
    )  # overlaps
    with pytest.raises(ParkGeometryValidationError, match="overlapping"):
        validate_geometry_points(cfg_a + cfg_b)


def test_validation_accepts_adjacent_non_overlapping_configs():
    cfg_a = _standard_config(999, "cfg_a", "2021-01-01", "2022-03-31", (330, 370, 400, 370, 330))
    cfg_b = _standard_config(999, "cfg_b", "2022-04-01", None, (335, 375, 405, 375, 335))
    validate_geometry_points(cfg_a + cfg_b)  # must not raise


def test_validation_rejects_temporary_venue_with_geometry():
    temp_venue_id = next(iter(TEMPORARY_OR_SPECIAL_VENUE_IDS))
    points = _standard_config(
        temp_venue_id, "cfg_temp", "2021-01-01", None, (330, 370, 400, 370, 330)
    )
    with pytest.raises(ParkGeometryValidationError, match="TEMPORARY_OR_SPECIAL_VENUE_IDS"):
        validate_geometry_points(points)


def test_validation_rejects_wrong_angle_set():
    points = _standard_config(999, "cfg_a", "2021-01-01", None, (330, 370, 400, 370, 330))
    points[0] = _pt(999, "cfg_a", "2021-01-01", None, -44.0, 330.0)  # wrong angle
    with pytest.raises(ParkGeometryValidationError, match="STANDARD_ANGLES"):
        validate_geometry_points(points)


# ---------------------------------------------------------------------------
# Effective-date resolution
# ---------------------------------------------------------------------------


def test_resolve_geometry_config_picks_correct_era():
    points = _standard_config(
        999, "cfg_a", "2021-01-01", "2022-03-31", (330, 370, 400, 370, 330)
    ) + _standard_config(999, "cfg_b", "2022-04-01", None, (335, 375, 405, 375, 335))
    validate_geometry_points(points)

    from mlb_luck_score.data import park_geometry as pg

    configs = pg._build_configs(points)
    match_2021 = next(
        c
        for c in configs
        if c.effective_start_date <= "2021-06-01" <= (c.effective_end_date or "9999-12-31")
    )
    assert match_2021.geometry_config_id == "cfg_a"


def test_resolve_geometry_config_boundary_dates():
    # Real Camden Yards configuration boundary: pre-2022 ends 2022-04-06,
    # 2022-2024 config starts 2022-04-07.
    assert resolve_geometry_config(2, "2022-04-06").geometry_config_id == "camden_yards_pre2022"
    assert resolve_geometry_config(2, "2022-04-07").geometry_config_id == "camden_yards_2022_2024"


def test_resolve_geometry_config_multi_config_venue_camden_yards():
    assert resolve_geometry_config(2, "2021-06-01").geometry_config_id == "camden_yards_pre2022"
    assert resolve_geometry_config(2, "2023-06-01").geometry_config_id == "camden_yards_2022_2024"
    assert resolve_geometry_config(2, "2024-06-01").geometry_config_id == "camden_yards_2022_2024"


def test_resolve_geometry_config_returns_none_for_unconfigured_venue():
    assert resolve_geometry_config(999_999, "2022-06-01") is None


def test_resolve_geometry_config_returns_none_for_missing_inputs():
    assert resolve_geometry_config(None, "2022-06-01") is None
    assert resolve_geometry_config(2, None) is None


def test_field_of_dreams_has_no_geometry_configuration():
    # venue_id=-1 (mlb_luck_score.data.game_metadata_overrides.FIELD_OF_DREAMS_VENUE_ID)
    # must never resolve a geometry config -- deliberately unavailable, not fabricated.
    assert resolve_geometry_config(-1, "2021-08-12") is None


@pytest.mark.parametrize("temp_venue_id", sorted(TEMPORARY_OR_SPECIAL_VENUE_IDS))
def test_temporary_venues_have_no_geometry_configuration(temp_venue_id):
    assert resolve_geometry_config(temp_venue_id, "2023-06-01") is None


# ---------------------------------------------------------------------------
# Spray-angle sign convention
# ---------------------------------------------------------------------------


def test_sign_convention_matches_clean_batted_balls_left():
    # hc_x < origin -> negative angle (third-base/left side), per
    # clean_batted_balls._spray_angle_degrees and the park_geometry docstring.
    angle = _spray_angle_degrees(hc_x=90.0, hc_y=150.0)
    assert angle < 0


def test_sign_convention_matches_clean_batted_balls_right():
    angle = _spray_angle_degrees(hc_x=170.0, hc_y=150.0)
    assert angle > 0


def test_sign_convention_matches_clean_batted_balls_center():
    angle = _spray_angle_degrees(hc_x=125.42, hc_y=100.0)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_standard_angles_match_eligibility_sector_boundary():
    from mlb_luck_score.data.clean_batted_balls import _SPRAY_SECTOR_EDGES

    assert STANDARD_ANGLES[0] == _SPRAY_SECTOR_EDGES[0]
    assert STANDARD_ANGLES[-1] == _SPRAY_SECTOR_EDGES[-1]


# ---------------------------------------------------------------------------
# Left/center/right wall lookup + interpolation
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_config():
    points = _standard_config(
        999,
        "cfg_a",
        "2021-01-01",
        None,
        (330.0, 370.0, 400.0, 370.0, 330.0),
        (10.0, None, 8.0, None, 10.0),
    )
    from mlb_luck_score.data import park_geometry as pg

    return pg._build_configs(points)[0]


def test_wall_lookup_left_field_line(synthetic_config):
    result = interpolate_wall_geometry(synthetic_config, -45.0)
    assert result is not None
    assert result.wall_distance_feet == 330.0
    assert result.wall_height_feet == 10.0
    assert result.distance_source_type == "measured"
    assert result.segment_label == "left_field_line"


def test_wall_lookup_center_field(synthetic_config):
    result = interpolate_wall_geometry(synthetic_config, 0.0)
    assert result is not None
    assert result.wall_distance_feet == 400.0
    assert result.distance_source_type == "measured"


def test_wall_lookup_right_field_line(synthetic_config):
    result = interpolate_wall_geometry(synthetic_config, 45.0)
    assert result is not None
    assert result.wall_distance_feet == 330.0
    assert result.distance_source_type == "measured"


def test_piecewise_interpolation_midpoint(synthetic_config):
    # Halfway between left_field_line (-45, 330) and left_center (-22.5, 370).
    result = interpolate_wall_geometry(synthetic_config, -33.75)
    assert result is not None
    assert result.wall_distance_feet == pytest.approx(350.0)
    assert result.distance_source_type == "interpolated"
    assert result.nearest_point_distance_degrees == pytest.approx(11.25)


def test_interpolation_returns_none_height_when_a_neighbor_is_missing(synthetic_config):
    # Between left_field_line (height=10.0) and left_center (height=None).
    result = interpolate_wall_geometry(synthetic_config, -30.0)
    assert result is not None
    assert result.wall_height_feet is None


def test_interpolation_boundary_point_is_measured_not_interpolated(synthetic_config):
    result = interpolate_wall_geometry(synthetic_config, 22.5)
    assert result is not None
    assert result.distance_source_type == "measured"
    assert result.nearest_point_distance_degrees == 0.0


def test_no_extrapolation_outside_modeled_range(synthetic_config):
    assert interpolate_wall_geometry(synthetic_config, 45.001) is None
    assert interpolate_wall_geometry(synthetic_config, -45.001) is None
    assert interpolate_wall_geometry(synthetic_config, 90.0) is None
    assert interpolate_wall_geometry(synthetic_config, -90.0) is None


def test_interpolation_within_range_never_none(synthetic_config):
    for angle in (-45.0, -30.0, -22.5, 0.0, 22.5, 30.0, 45.0):
        assert interpolate_wall_geometry(synthetic_config, angle) is not None


# ---------------------------------------------------------------------------
# Real-table sanity checks
# ---------------------------------------------------------------------------


def test_real_table_has_thirty_primary_venues():
    venues = {c.venue_id for c in PARK_GEOMETRY_CONFIGS}
    assert len(venues) == 30


def test_real_table_multi_config_venues_are_the_three_documented_ones():
    from collections import Counter

    counts = Counter(c.venue_id for c in PARK_GEOMETRY_CONFIGS)
    multi = {venue_id for venue_id, n in counts.items() if n > 1}
    assert multi == {2, 14, 2394}  # Camden Yards, Rogers Centre, Comerica Park


def test_real_table_camden_yards_wall_height_increased_in_2022():
    pre = resolve_geometry_config(2, "2021-06-01")
    post = resolve_geometry_config(2, "2022-06-01")
    pre_left_line = next(p for p in pre.points if p.segment_label == "left_field_line")
    post_left_line = next(p for p in post.points if p.segment_label == "left_field_line")
    assert post_left_line.wall_height_feet > pre_left_line.wall_height_feet
