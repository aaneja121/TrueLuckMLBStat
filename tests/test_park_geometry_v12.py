"""Tests for the Version 1.2 park-geometry table (`mlb_luck_score.data.park_geometry_v12`).

The v1.2 table is the frozen v1.0 table plus maintainer-approved corrections and new
2025-2027 configurations, recorded in `docs/reviews/park_geometry_review_2026-09-29.md`.
These tests pin every approved correction, prove the frozen table is untouched, and check
that review statuses never claim more review than happened.
"""

from __future__ import annotations

import pytest

from mlb_luck_score.data import park_geometry as v10
from mlb_luck_score.data import park_geometry_v12 as v12

LF, LC, CF, RC, RF = v10.STANDARD_ANGLES


def _point(config_id: str, angle: float) -> v10.ParkGeometryPoint:
    matches = [
        p
        for p in v12.PARK_GEOMETRY_POINTS_V12
        if p.geometry_config_id == config_id and p.spray_angle_degrees == angle
    ]
    assert len(matches) == 1, f"{config_id} @ {angle}: {len(matches)} points"
    return matches[0]


def _frozen_point(config_id: str, angle: float) -> v10.ParkGeometryPoint:
    return next(
        p
        for p in v10.PARK_GEOMETRY_POINTS
        if p.geometry_config_id == config_id and p.spray_angle_degrees == angle
    )


# --------------------------------------------------------------------------
# The frozen v1.0 table is never modified
# --------------------------------------------------------------------------


def test_the_frozen_table_is_unchanged_by_building_v12() -> None:
    assert _frozen_point("pnc_park_v1", LF).wall_distance_feet == 320.0
    assert _frozen_point("oracle_park_v1", CF).wall_height_feet == 8.5
    assert _frozen_point("camden_yards_2022_2024", LF).effective_end_date is None
    assert all(p.review_status == v10.REVIEW_STATUS_AGENT_SOURCED for p in v10.PARK_GEOMETRY_POINTS)


def test_v12_passes_the_frozen_validator() -> None:
    v10.validate_geometry_points(v12.PARK_GEOMETRY_POINTS_V12)


# --------------------------------------------------------------------------
# Every approved correction is applied (worksheet "Confirmed corrections")
# --------------------------------------------------------------------------

DISTANCE_CORRECTIONS = [
    ("pnc_park_v1", LF, 325.0),
    ("camden_yards_pre2022", LC, 364.0),
    ("camden_yards_2022_2024", LC, 398.0),
    ("progressive_field_v1", CF, 405.0),
    ("petco_park_v1", LC, 386.0),
    ("petco_park_v1", RF, 322.0),
    ("citi_field_v1", LC, 370.0),
    ("citi_field_v1", RC, 380.0),
    ("american_family_field_v1", LF, 342.0),
    ("american_family_field_v1", LC, 370.0),
    ("american_family_field_v1", RF, 345.0),
    ("guaranteed_rate_field_v1", LC, 377.0),
    ("guaranteed_rate_field_v1", RC, 372.0),
    ("rogers_centre_2023_2024", LC, 381.0),
    ("rogers_centre_2023_2024", RC, 372.0),
    ("chase_field_v1", LC, 376.0),
    ("chase_field_v1", RC, 376.0),
    ("chase_field_v1", RF, 335.0),
    ("nationals_park_v1", LF, 336.0),
]

HEIGHT_CORRECTIONS = [
    ("wrigley_field_v1", LF, 15.0),
    ("wrigley_field_v1", RF, 15.0),
    ("petco_park_v1", LF, 4.0),
    ("petco_park_v1", LC, 7.0),
    ("oracle_park_v1", CF, 10.0),
    ("oracle_park_v1", RC, 24.0),
    ("rogers_centre_2023_2024", LC, 12.75),
    ("rogers_centre_2023_2024", RC, 10.75),
    ("dodger_stadium_v1", LF, 4.5),
    ("dodger_stadium_v1", CF, 8.0),
    ("dodger_stadium_v1", RF, 4.5),
    ("chase_field_v1", LF, 7.5),
    ("chase_field_v1", CF, 25.0),
    ("chase_field_v1", RF, 7.5),
    ("target_field_v1", LF, 8.0),
    ("target_field_v1", LC, 8.0),
    ("target_field_v1", CF, 8.0),
    ("target_field_v1", RC, 8.0),
    ("target_field_v1", RF, 23.0),
    ("minute_maid_park_v1", LC, 25.0),
    ("minute_maid_park_v1", CF, 10.0),
    ("minute_maid_park_v1", RC, 10.0),
    ("minute_maid_park_v1", RF, 7.0),
    ("gabp_v1", LF, 12.0),
    ("gabp_v1", CF, 8.0),
    ("gabp_v1", RF, 8.0),
    *[
        (cfg, angle, 8.0)
        for cfg in ("citi_field_v1", "american_family_field_v1", "t_mobile_park_v1")
        for angle in v10.STANDARD_ANGLES
    ],
]


@pytest.mark.parametrize(("config_id", "angle", "expected"), DISTANCE_CORRECTIONS)
def test_each_approved_distance_correction_is_applied(
    config_id: str, angle: float, expected: float
) -> None:
    assert _point(config_id, angle).wall_distance_feet == expected
    assert _frozen_point(config_id, angle).wall_distance_feet != expected


@pytest.mark.parametrize(("config_id", "angle", "expected"), HEIGHT_CORRECTIONS)
def test_each_approved_height_correction_is_applied(
    config_id: str, angle: float, expected: float
) -> None:
    assert _point(config_id, angle).wall_height_feet == expected


def test_oracle_right_field_stays_24_ft_and_its_distances_stand() -> None:
    assert _point("oracle_park_v1", RF).wall_height_feet == 24.0
    assert [_point("oracle_park_v1", a).wall_distance_feet for a in v10.STANDARD_ANGLES] == [
        339.0,
        399.0,
        391.0,
        415.0,
        309.0,
    ]


def test_dodger_stadium_keeps_true_distances_over_posted_signs() -> None:
    assert [_point("dodger_stadium_v1", a).wall_distance_feet for a in v10.STANDARD_ANGLES] == [
        330.0,
        375.0,
        400.0,
        375.0,
        330.0,
    ]
    # Alley heights depend on the bullpen positions, which no source pins down.
    assert _point("dodger_stadium_v1", LC).wall_height_feet is None
    assert _point("dodger_stadium_v1", RC).wall_height_feet is None


def test_unreviewed_values_are_carried_over_unchanged() -> None:
    assert _point("fenway_park_v1", CF).wall_distance_feet == 390.0  # official 389, not approved
    assert _point("busch_stadium_v1", LC).wall_distance_feet == 375.0
    assert _point("kauffman_stadium_v1", CF).wall_height_feet == 9.0


def test_every_corrected_point_records_its_source() -> None:
    for config_id, angle, _ in DISTANCE_CORRECTIONS + HEIGHT_CORRECTIONS:
        point = _point(config_id, angle)
        assert "v1.2" in point.notes, f"{config_id} @ {angle} has no v1.2 correction note"


# --------------------------------------------------------------------------
# New and re-dated configurations
# --------------------------------------------------------------------------


def test_camden_2022_config_ends_before_2025() -> None:
    assert _point("camden_yards_2022_2024", LF).effective_end_date == "2024-12-31"


@pytest.mark.parametrize(
    ("config_id", "venue_id", "start", "end", "distances"),
    [
        ("kauffman_stadium_2026", 7, "2026-01-01", None, (330.0, 379.0, 410.0, 379.0, 330.0)),
        ("camden_yards_2025", 2, "2025-01-01", None, (333.0, 363.0, 400.0, 373.0, 318.0)),
        (
            "sutter_health_park_2025_2027",
            2529,
            "2025-01-01",
            "2027-12-31",
            (330.0, 380.0, 403.0, 380.0, 325.0),
        ),
        (
            "las_vegas_ballpark_2026_2027",
            5355,
            "2026-01-01",
            "2027-12-31",
            (340.0, 380.0, 415.0, 380.0, 340.0),
        ),
    ],
)
def test_new_configurations(
    config_id: str, venue_id: int, start: str, end: str | None, distances: tuple[float, ...]
) -> None:
    points = [_point(config_id, a) for a in v10.STANDARD_ANGLES]
    assert {p.venue_id for p in points} == {venue_id}
    assert {p.effective_start_date for p in points} == {start}
    assert {p.effective_end_date for p in points} == {end}
    assert tuple(p.wall_distance_feet for p in points) == distances


def test_new_configuration_heights() -> None:
    assert all(
        _point("kauffman_stadium_2026", a).wall_height_feet == 8.5 for a in v10.STANDARD_ANGLES
    )
    assert _point("camden_yards_2025", LF).wall_height_feet == 8.0
    assert _point("camden_yards_2025", LC).wall_height_feet == 6.92


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("venue_id", "game_date", "expected"),
    [
        (2, "2021-06-01", "camden_yards_pre2022"),
        (2, "2024-09-01", "camden_yards_2022_2024"),
        (2, "2025-05-01", "camden_yards_2025"),
        (7, "2025-07-01", "kauffman_stadium_v1"),
        (7, "2026-07-01", "kauffman_stadium_2026"),
        (12, "2026-09-27", "tropicana_field_v1"),  # the relocated Red Sox home game
        (5355, "2026-06-08", "las_vegas_ballpark_2026_2027"),
        (2529, "2027-09-01", "sutter_health_park_2025_2027"),
    ],
)
def test_lookup_resolves_by_venue_and_date(venue_id: int, game_date: str, expected: str) -> None:
    config = v12.resolve_geometry_config_v12(venue_id, game_date)
    assert config is not None
    assert config.geometry_config_id == expected


@pytest.mark.parametrize(
    ("venue_id", "game_date"),
    [
        (5445, "2026-08-13"),  # Field of Dreams
        (5340, "2026-04-25"),  # Mexico City
        (2735, "2026-08-23"),  # Williamsport
        (2529, "2028-04-01"),  # Sutter after the A's leave
        (None, "2026-05-01"),
        (2, None),
    ],
)
def test_excluded_or_unknown_venues_have_no_geometry(
    venue_id: int | None, game_date: str | None
) -> None:
    assert v12.resolve_geometry_config_v12(venue_id, game_date) is None


def test_no_excluded_venue_has_points() -> None:
    venues = {p.venue_id for p in v12.PARK_GEOMETRY_POINTS_V12}
    assert not venues & v12.V12_EXCLUDED_VENUE_IDS


# --------------------------------------------------------------------------
# Review status never overstates review
# --------------------------------------------------------------------------


def test_review_statuses_are_from_the_documented_set() -> None:
    allowed = {
        v10.REVIEW_STATUS_AGENT_SOURCED,
        v12.REVIEW_STATUS_MAINTAINER_OFFICIAL,
        v12.REVIEW_STATUS_MAINTAINER_SECONDARY,
    }
    assert {p.review_status for p in v12.PARK_GEOMETRY_POINTS_V12} <= allowed


@pytest.mark.parametrize(
    ("config_id", "angle"),
    [
        ("gabp_v1", LC),
        ("gabp_v1", RC),
        ("tropicana_field_v1", LC),
        ("tropicana_field_v1", RC),
        ("angel_stadium_v1", RC),
        ("sutter_health_park_2025_2027", LC),
        ("sutter_health_park_2025_2027", RC),
        ("las_vegas_ballpark_2026_2027", CF),
        ("dodger_stadium_v1", CF),
    ],
)
def test_points_without_an_official_figure_are_marked_secondary(
    config_id: str, angle: float
) -> None:
    assert _point(config_id, angle).review_status == v12.REVIEW_STATUS_MAINTAINER_SECONDARY


@pytest.mark.parametrize(
    ("config_id", "angle"),
    [
        ("fenway_park_v1", CF),
        ("comerica_park_pre2023", LF),
        ("kauffman_stadium_v1", LC),
        ("oakland_coliseum_v1", LC),
    ],
)
def test_points_nobody_reviewed_keep_the_agent_status(config_id: str, angle: float) -> None:
    assert _point(config_id, angle).review_status == v10.REVIEW_STATUS_AGENT_SOURCED


def test_officially_confirmed_points_are_marked_official() -> None:
    assert _point("pnc_park_v1", LF).review_status == v12.REVIEW_STATUS_MAINTAINER_OFFICIAL
    assert _point("wrigley_field_v1", LC).review_status == v12.REVIEW_STATUS_MAINTAINER_OFFICIAL
    assert _point("oracle_park_v1", RF).review_status == v12.REVIEW_STATUS_MAINTAINER_OFFICIAL


# --------------------------------------------------------------------------
# Wall-height review (docs/reviews/park_wall_height_review_2026-10-01.md)
# --------------------------------------------------------------------------

_ALL = v10.STANDARD_ANGLES

#: Every height the maintainer approved on 2026-10-01 (sourced rows plus the
#: accepted suggestions for contested points).
HEIGHT_REVIEW_2026_10_01: list[tuple[str, float, float]] = [
    *[("truist_park_v1", a, h) for a, h in zip(_ALL, (6.0, 8.67, 8.67, 16.0, 16.0), strict=True)],
    *[
        ("citizens_bank_park_v1", a, h)
        for a, h in zip(_ALL, (10.5, 10.5, 6.0, 13.25, 13.25), strict=True)
    ],
    *[("globe_life_field_v1", a, 8.0) for a in _ALL],
    *[("wrigley_field_v1", a, 11.5) for a in (LC, CF, RC)],
    ("camden_yards_pre2022", LC, 7.0),
    ("camden_yards_2022_2024", LC, 13.0),
    *[
        (cfg, a, h)
        for cfg in ("camden_yards_pre2022", "camden_yards_2022_2024", "camden_yards_2025")
        for a, h in ((CF, 7.0), (RC, 7.0), (RF, 21.0))
    ],
    ("fenway_park_v1", LC, 17.0),
    ("fenway_park_v1", CF, 17.0),
    ("fenway_park_v1", RC, 5.0),
    ("fenway_park_v1", RF, 3.0),
    ("coors_field_v1", LC, 8.0),
    ("coors_field_v1", CF, 8.0),
    ("coors_field_v1", RF, 16.5),
    ("comerica_park_pre2023", CF, 8.5),
    ("comerica_park_pre2023", RC, 13.0),
    ("comerica_park_pre2023", RF, 8.5),
    ("loandepot_park_v1", LF, 7.0),
    ("loandepot_park_v1", LC, 11.5),
    ("loandepot_park_v1", RF, 7.0),
    ("petco_park_v1", CF, 7.0),
    ("petco_park_v1", RC, 7.0),
    *[
        ("oakland_coliseum_v1", a, h)
        for a, h in zip(_ALL, (8.0, 15.0, 8.0, 15.0, 8.0), strict=True)
    ],
    *[("busch_stadium_v1", a, 8.0) for a in _ALL],
    *[
        ("tropicana_field_v1", a, h)
        for a, h in zip(_ALL, (11.0, 11.0, 9.0, 11.0, 11.0), strict=True)
    ],
    *[("angel_stadium_v1", a, h) for a, h in zip(_ALL, (5.0, 8.0, 8.0, 8.0, 5.0), strict=True)],
    ("yankee_stadium_v1", LC, 8.0),
    ("yankee_stadium_v1", CF, 8.0),
    ("yankee_stadium_v1", RC, 8.0),
    ("progressive_field_v1", LC, 19.0),
    ("progressive_field_v1", RC, 9.0),
    ("gabp_v1", LC, 12.0),
    ("gabp_v1", RC, 8.0),
    ("chase_field_v1", LC, 7.5),
    ("chase_field_v1", RC, 7.5),
    ("pnc_park_v1", LC, 10.0),
    ("pnc_park_v1", RC, 21.0),
]

#: Points the review deliberately left without a height.
HEIGHTS_LEFT_EMPTY_2026_10_01: list[tuple[str, float]] = [
    *[(cfg, a) for cfg in ("comerica_park_pre2023", "comerica_park_2023_2024") for a in (LF, LC)],
    ("loandepot_park_v1", CF),
    ("loandepot_park_v1", RC),
    *[("nationals_park_v1", a) for a in _ALL],
    ("oracle_park_v1", LC),
    ("dodger_stadium_v1", LC),
    ("dodger_stadium_v1", RC),
    *[("sutter_health_park_2025_2027", a) for a in _ALL],
    *[("las_vegas_ballpark_2026_2027", a) for a in _ALL],
]


def test_height_review_covers_75_points_once() -> None:
    keys = [(c, a) for c, a, _ in HEIGHT_REVIEW_2026_10_01]
    assert len(keys) == len(set(keys)) == 75


@pytest.mark.parametrize(("config_id", "angle", "expected"), HEIGHT_REVIEW_2026_10_01)
def test_each_reviewed_height_is_applied(config_id: str, angle: float, expected: float) -> None:
    assert _point(config_id, angle).wall_height_feet == expected


@pytest.mark.parametrize(("config_id", "angle", "_"), HEIGHT_REVIEW_2026_10_01)
def test_each_reviewed_height_cites_the_height_worksheet(
    config_id: str, angle: float, _: float
) -> None:
    notes = _point(config_id, angle).notes
    assert "park_wall_height_review_2026-10-01.md" in notes
    assert "approved 2026-10-01" in notes


@pytest.mark.parametrize(("config_id", "angle"), HEIGHTS_LEFT_EMPTY_2026_10_01)
def test_points_the_review_left_empty_have_no_height(config_id: str, angle: float) -> None:
    assert _point(config_id, angle).wall_height_feet is None


def test_height_review_changes_no_distance_or_distance_status() -> None:
    for config_id, angle, _ in HEIGHT_REVIEW_2026_10_01:
        if config_id == "camden_yards_2025":
            continue  # a v1.2-only configuration, absent from the frozen table
        point = _point(config_id, angle)
        frozen = _frozen_point(config_id, angle)
        assert point.wall_distance_feet in (frozen.wall_distance_feet,) or any(
            c == config_id and a == angle for c, a, _ in DISTANCE_CORRECTIONS
        )
    assert _point("nationals_park_v1", LF).review_status == v12.REVIEW_STATUS_MAINTAINER_OFFICIAL
    assert _point("angel_stadium_v1", RC).review_status == v12.REVIEW_STATUS_MAINTAINER_SECONDARY


def test_height_count_after_the_review() -> None:
    with_height = sum(p.wall_height_feet is not None for p in v12.PARK_GEOMETRY_POINTS_V12)
    assert with_height == 86 + 75
