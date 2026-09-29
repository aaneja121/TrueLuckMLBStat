"""Contact Luck v1.2 (candidate): maintainer-reviewed outfield-wall geometry.

A separately versioned geometry table for the Version 1.2 park/weather work
(`docs/plans/v1_2_park_weather_plan.md`). It is built FROM the frozen v1.0 table
(`mlb_luck_score.data.park_geometry.PARK_GEOMETRY_POINTS`) by applying the
corrections the maintainer approved in
`docs/reviews/park_geometry_review_2026-09-29.md`, and by adding configurations
for venues and wall changes after 2024.

## The frozen v1.0 table is never edited

`park_geometry.py` is a frozen Version 1.0 input
(`evaluation.v1_final_evaluation_manifest.FROZEN_ARTIFACT_RELATIVE_PATHS`). It
is what the sealed 2025 evaluation and every 2026 snapshot were computed with,
so its known errors stay in it as a faithful record. This module only imports
it. Nothing in production scoring reads this module yet: adopting it is a
separate, explicit v1.2 decision.

## What changed, and on whose authority

Every override below cites the worksheet decision and the source behind it, and
the corrected point's `notes` gain a "v1.2:" line saying so. The rules applied,
all decided by the maintainer on 2026-09-29:

- **Measured over posted.** Where a posted sign differs from an independently
  sourced true distance, the true distance is used (e.g. Dodger Stadium).
- **The ±22.5 degree point is the power alley:** a point the team explicitly calls
  the power alley, otherwise the point the team labels left-/right-center.
- **A team figure beats an unsourced "measured" figure.**
- Heights are the barrier a ball must clear, including structure above the wall.

## Review status describes the wall DISTANCE

`ParkGeometryPoint` has one `review_status`, so in this table it records how the
point's **wall distance** was reviewed:

- `REVIEW_STATUS_MAINTAINER_OFFICIAL`: the distance matches an official team or
  league source recorded in the worksheet, and the maintainer approved it.
- `REVIEW_STATUS_MAINTAINER_SECONDARY`: the maintainer approved it, but no
  official figure exists; it rests on agreeing secondary sources.
- `REVIEW_STATUS_AGENT_SOURCED` (from v1.0): nobody reviewed it. Carried over
  unchanged.

A wall HEIGHT changed here cites its own source in the point's `notes`. A height
carried over from v1.0 without a "v1.2:" height note is unreviewed.

## Deliberately excluded venues

`V12_EXCLUDED_VENUE_IDS` extends v1.0's temporary/special set with the one-off
neutral sites of 2025-2027 (maintainer decision: about six games a season are
not worth geometry built on thin data) and with Steinbrenner Field, which was
used only in the sealed 2025 season that v1.2 never scores. Games there resolve
to no geometry, the same unknown-venue path as v1.0.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from mlb_luck_score.data.park_geometry import (
    PARK_GEOMETRY_POINTS,
    REVIEW_STATUS_AGENT_SOURCED,
    STANDARD_ANGLES,
    TEMPORARY_OR_SPECIAL_VENUE_IDS,
    ParkGeometryConfig,
    ParkGeometryPoint,
    ParkGeometryValidationError,
    _build_configs,
    _cfg,
    validate_geometry_points,
)

REVIEW_STATUS_MAINTAINER_OFFICIAL: Final = "maintainer_approved_official_source"
REVIEW_STATUS_MAINTAINER_SECONDARY: Final = "maintainer_approved_secondary_source"

#: Date of the maintainer's review in the worksheet.
V12_REVIEW_DATE: Final = "2026-09-29"
_WORKSHEET: Final = "docs/reviews/park_geometry_review_2026-09-29.md"

#: v1.0's temporary/special venues, plus venues v1.2 deliberately leaves
#: without geometry.
V12_EXCLUDED_VENUE_IDS: frozenset[int] = TEMPORARY_OR_SPECIAL_VENUE_IDS | frozenset(
    {
        5445,  # Field of Dreams (Dyersville) -- 2026 and 2027 single games
        6130,  # Bristol Motor Speedway -- 2025 only (sealed)
        2397,  # Tokyo Dome -- 2025 only (sealed)
        2523,  # George M. Steinbrenner Field -- Rays 2025 only (sealed)
    }
)

LF, LC, CF, RC, RF = STANDARD_ANGLES


#: (config_id, spray angle) -> (field, new value, source). `field` is
#: "wall_distance_feet", "wall_height_feet" or "effective_end_date".
_Override = tuple[str, float | int | str | None, str]
_OVERRIDES: dict[tuple[str, float], list[_Override]] = {}


def _add(
    config_id: str, angles: tuple[float, ...], field: str, value: float | str | None, source: str
) -> None:
    for angle in angles:
        _OVERRIDES.setdefault((config_id, angle), []).append((field, value, source))


_ALL = STANDARD_ANGLES
_D, _H = "wall_distance_feet", "wall_height_feet"

# --- distances -------------------------------------------------------------
_add("pnc_park_v1", (LF,), _D, 325.0, "MLB.com PNC Park guide (2026-03-25): 'left field, 325 feet'")
_add("camden_yards_pre2022", (LC,), _D, 364.0, "Orioles ground rules: '364 feet to left center'")
_add(
    "camden_yards_2022_2024",
    (LC,),
    _D,
    398.0,
    "Orioles ground rules (2022): 'left-center corner: 398 ft.'; rule A",
)
_add("progressive_field_v1", (CF,), _D, 405.0, "Guardians ballparks page: 'center field: 405 ft.'")
_add(
    "petco_park_v1",
    (LC,),
    _D,
    386.0,
    "Padres release 2014-11-06 (wall in ~34-38 in. for 2015); MLB.com guide 'left-center, 386 feet'",
)
_add("petco_park_v1", (RF,), _D, 322.0, "MLB.com Petco Park guide: 'right-field line, 322 feet'")
_add("citi_field_v1", (LC,), _D, 370.0, "Mets ballpark guide: 'Left Center Field 370 feet'")
_add("citi_field_v1", (RC,), _D, 380.0, "Mets ballpark guide: 'Right Center Field 380 feet'")
_add(
    "american_family_field_v1",
    (LF,),
    _D,
    342.0,
    "Brewers ground rules: 'Left-Field Foul Line: 342 feet'",
)
_add(
    "american_family_field_v1",
    (LC,),
    _D,
    370.0,
    "Brewers ground rules: 'Left-Field Power Alley: 370 feet'",
)
_add(
    "american_family_field_v1",
    (RF,),
    _D,
    345.0,
    "Brewers ground rules: 'Right-Field Foul Line: 345 feet'; rule B",
)
_add(
    "guaranteed_rate_field_v1",
    (LC,),
    _D,
    377.0,
    "White Sox ballpark page: 'Left Centerfield: 377 feet'",
)
_add(
    "guaranteed_rate_field_v1",
    (RC,),
    _D,
    372.0,
    "White Sox ballpark page: 'Right Centerfield: 372 feet'",
)
_add(
    "rogers_centre_2023_2024",
    (LC,),
    _D,
    381.0,
    "Blue Jays history page: 'Left Center Power Alley: 381 feet'; rule A",
)
_add(
    "rogers_centre_2023_2024",
    (RC,),
    _D,
    372.0,
    "Blue Jays history page: 'Right Center Power Alley: 372 feet'; rule A",
)
_add("chase_field_v1", (LC, RC), _D, 376.0, "D-backs facts page: 'LCF: 376' ... RCF: 376''")
_add("chase_field_v1", (RF,), _D, 335.0, "D-backs facts page: 'RF: 335''")
_add("nationals_park_v1", (LF,), _D, 336.0, "Nationals facts page: 'Left Field: 336 Feet'")

# --- heights ---------------------------------------------------------------
_add("wrigley_field_v1", (LF, RF), _H, 15.0, "Cubs history page: 'In corners - 15.0 feet'")
_add(
    "petco_park_v1",
    (LF,),
    _H,
    4.0,
    "Maintainer-confirmed 2026-09-29; Ballpark Digest 2009 '4 feet in the left field corner'",
)
_add(
    "petco_park_v1", (LC,), _H, 7.0, "Padres release 2014-11-06: wall 'lowered to seven-feet tall'"
)
_add(
    "oracle_park_v1",
    (CF,),
    _H,
    10.0,
    "NBC Sports Bay Area 2020-07-28: center wall 'lengthened to 10 feet tall'",
)
_add(
    "oracle_park_v1",
    (RC,),
    _H,
    24.0,
    "Maintainer decision: 415 point is the end of the 24-ft brick wall (MLB.com guide)",
)
_add(
    "rogers_centre_2023_2024",
    (LC,),
    _H,
    12.75,
    "Blue Jays history page: power-alley 'wall: 12 feet 9 inches'",
)
_add(
    "rogers_centre_2023_2024",
    (RC,),
    _H,
    10.75,
    "Blue Jays history page: power-alley 'wall: 10 feet 9 inches'",
)
_add(
    "dodger_stadium_v1",
    (LF, RF),
    _H,
    4.5,
    "Dodgers history page: foul pole to bullpens '55 inches high (about 4.5 feet)'",
)
_add(
    "dodger_stadium_v1",
    (CF,),
    _H,
    8.0,
    "Dodgers history page: 'From bullpen to bullpen, the fence is 8 feet high'",
)
_add("chase_field_v1", (LF, RF), _H, 7.5, "D-backs facts page: 'LF: 7'6\" ... RF: 7'6\"'")
_add("chase_field_v1", (CF,), _H, 25.0, "D-backs facts page: 'CF: 25''")
_add(
    "target_field_v1",
    (LF, LC, CF, RC),
    _H,
    8.0,
    "Minnesota Ballpark Authority: walls '8' from the left field foul pole to right center field'",
)
_add(
    "target_field_v1",
    (RF,),
    _H,
    23.0,
    "Minnesota Ballpark Authority: '23' from right center field to the right field foul pole'",
)
_add("minute_maid_park_v1", (LC,), _H, 25.0, "Astros facts page: 'Left-center - 25 feet'")
_add("minute_maid_park_v1", (CF,), _H, 10.0, "Astros facts page: 'Center field - 10 feet'")
_add("minute_maid_park_v1", (RC,), _H, 10.0, "Astros facts page: 'Right-center - 10 feet'")
_add("minute_maid_park_v1", (RF,), _H, 7.0, "Astros facts page: 'Right field - 7 feet'")
_add("gabp_v1", (LF,), _H, 12.0, "Reds dimensions FAQ: left field 'with a 12-foot wall'")
_add("gabp_v1", (CF, RF), _H, 8.0, "Reds dimensions FAQ: center and right 'wall height of 8 feet'")
_add(
    "citi_field_v1",
    _ALL,
    _H,
    8.0,
    "Mets ballpark guide: '8 feet consistent from Foul Pole to Foul Pole'",
)
_add(
    "american_family_field_v1",
    _ALL,
    _H,
    8.0,
    "Brewers ground rules: '/8 feet' at every listed point",
)
_add(
    "t_mobile_park_v1",
    _ALL,
    _H,
    8.0,
    "Mariners release 2012-10-02: 'eight-feet from foul pole to foul pole'",
)

# --- dates -----------------------------------------------------------------
_add(
    "camden_yards_2022_2024",
    _ALL,
    "effective_end_date",
    "2024-12-31",
    "Orioles: wall changed again 'Prior to the 2025 season'",
)


#: Wall-DISTANCE review status per (config, angle). Anything absent keeps
#: REVIEW_STATUS_AGENT_SOURCED. See the worksheet for each source.
_OFFICIAL: dict[str, tuple[float, ...]] = {
    "angel_stadium_v1": (LF, LC, CF, RF),
    "truist_park_v1": _ALL,
    "camden_yards_pre2022": _ALL,
    "camden_yards_2022_2024": _ALL,
    "fenway_park_v1": (LF, LC, RC, RF),  # CF: official 389 vs stored 390, not approved
    "wrigley_field_v1": _ALL,
    "guaranteed_rate_field_v1": _ALL,
    "gabp_v1": (LF, CF, RF),
    "progressive_field_v1": _ALL,
    "coors_field_v1": _ALL,
    "comerica_park_2023_2024": _ALL,
    "minute_maid_park_v1": _ALL,
    "dodger_stadium_v1": (LF, RF),
    "loandepot_park_v1": _ALL,
    "american_family_field_v1": _ALL,
    "target_field_v1": _ALL,
    "citi_field_v1": _ALL,
    "yankee_stadium_v1": _ALL,
    "citizens_bank_park_v1": _ALL,
    "pnc_park_v1": _ALL,
    "petco_park_v1": _ALL,
    "oracle_park_v1": _ALL,
    "t_mobile_park_v1": _ALL,
    "busch_stadium_v1": _ALL,
    "tropicana_field_v1": (LF, CF, RF),
    "globe_life_field_v1": _ALL,
    "rogers_centre_pre2023": _ALL,
    "rogers_centre_2023_2024": _ALL,
    "nationals_park_v1": _ALL,
    "chase_field_v1": _ALL,
    "kauffman_stadium_2026": _ALL,
    "camden_yards_2025": _ALL,
    "sutter_health_park_2025_2027": (LF, CF, RF),
}
_SECONDARY: dict[str, tuple[float, ...]] = {
    "angel_stadium_v1": (RC,),
    "gabp_v1": (LC, RC),
    "tropicana_field_v1": (LC, RC),
    "dodger_stadium_v1": (LC, CF, RC),  # true distances; posted 385 / 395 / 385
    "sutter_health_park_2025_2027": (LC, RC),  # Stats API only
    "las_vegas_ballpark_2026_2027": _ALL,
}


def _new_configs() -> tuple[ParkGeometryPoint, ...]:
    """Configurations for wall changes and venues after 2024 (worksheet gap pass)."""
    return (
        *_cfg(
            7,
            "Kauffman Stadium",
            "kauffman_stadium_2026",
            "2026-01-01",
            None,
            (330.0, 379.0, 410.0, 379.0, 330.0),
            (8.5, 8.5, 8.5, 8.5, 8.5),
            source_name="Royals ballparks history page",
            source_reference="https://www.mlb.com/royals/history/ballparks",
            notes="v1.2: 'Foul Poles 330 feet ... Left-Center Gap 379 feet Center Field 410 feet ... "
            "Fence Height 8.5 feet'. Corners 347/344 and straightaway 364 are on the deferred list.",
            source_accessed_date=V12_REVIEW_DATE,
        ),
        *_cfg(
            2,
            "Oriole Park at Camden Yards",
            "camden_yards_2025",
            "2025-01-01",
            None,
            (333.0, 363.0, 400.0, 373.0, 318.0),
            (8.0, 6.92, None, None, None),
            source_name="Orioles ground rules",
            source_reference="https://www.mlb.com/orioles/ballpark/ground-rules",
            notes="v1.2: 2025 wall 'to 8 feet near the left field foul pole and to 6 feet, 11 inches "
            "closer to the left-center bullpens'; 'left-center: 363 ft.' (rule A). LC height 6.92 "
            "pending an official diagram. Corner 373 and bullpen 376 are on the deferred list.",
            source_accessed_date=V12_REVIEW_DATE,
        ),
        *_cfg(
            2529,
            "Sutter Health Park",
            "sutter_health_park_2025_2027",
            "2025-01-01",
            "2027-12-31",
            (330.0, 380.0, 403.0, 380.0, 325.0),
            (None, None, None, None, None),
            source_name="MLB.com Sutter Health Park guide (lines, CF); MLB Stats API fieldInfo (alleys)",
            source_reference="https://www.mlb.com/news/featured/sutter-health-park-guide-capacity-seating-chart-parking-and-more",
            notes="v1.2: Athletics' home 2025-2027. Alleys 380/380 come only from the Stats API -- low "
            "confidence.",
            source_accessed_date=V12_REVIEW_DATE,
        ),
        *_cfg(
            5355,
            "Las Vegas Ballpark",
            "las_vegas_ballpark_2026_2027",
            "2026-01-01",
            "2027-12-31",
            (340.0, 380.0, 415.0, 380.0, 340.0),
            (None, None, None, None, None),
            source_name="MLB Stats API fieldInfo; Wikipedia",
            source_reference="https://statsapi.mlb.com/api/v1/venues/5355?hydrate=fieldInfo",
            notes="v1.2: six Athletics home games in each of 2026 and 2027. No official dimensions "
            "published; secondary sources agree.",
            source_accessed_date=V12_REVIEW_DATE,
        ),
    )


def _status(config_id: str, angle: float) -> str:
    if angle in _OFFICIAL.get(config_id, ()):
        return REVIEW_STATUS_MAINTAINER_OFFICIAL
    if angle in _SECONDARY.get(config_id, ()):
        return REVIEW_STATUS_MAINTAINER_SECONDARY
    return REVIEW_STATUS_AGENT_SOURCED


def _apply(point: ParkGeometryPoint) -> ParkGeometryPoint:
    changes: dict[str, object] = {}
    notes = [point.notes] if point.notes else []
    for field, value, source in _OVERRIDES.get(
        (point.geometry_config_id, point.spray_angle_degrees), []
    ):
        changes[field] = value
        notes.append(
            f"v1.2: {field} -> {value} ({source}; approved {V12_REVIEW_DATE}, {_WORKSHEET})."
        )
    status = _status(point.geometry_config_id, point.spray_angle_degrees)
    return replace(point, **changes, review_status=status, notes=" ".join(notes))  # type: ignore[arg-type]


def build_v12_points() -> tuple[ParkGeometryPoint, ...]:
    """Build the v1.2 table: frozen v1.0 points with approved overrides, plus new configs.

    Raises:
        ParkGeometryValidationError: If an override names a point that does not exist,
            the result fails the v1.0 validator, or an excluded venue has points.
    """
    known = {(p.geometry_config_id, p.spray_angle_degrees) for p in PARK_GEOMETRY_POINTS}
    missing = sorted(key for key in _OVERRIDES if key not in known)
    if missing:
        raise ParkGeometryValidationError(f"v1.2 overrides name unknown points: {missing}")

    points = tuple(_apply(p) for p in PARK_GEOMETRY_POINTS) + tuple(
        _apply(p) for p in _new_configs()
    )
    validate_geometry_points(points)
    excluded = sorted({p.venue_id for p in points} & V12_EXCLUDED_VENUE_IDS)
    if excluded:
        raise ParkGeometryValidationError(f"excluded venues have v1.2 geometry: {excluded}")
    return points


#: The v1.2 geometry table, validated at import time.
PARK_GEOMETRY_POINTS_V12: tuple[ParkGeometryPoint, ...] = build_v12_points()
PARK_GEOMETRY_CONFIGS_V12: tuple[ParkGeometryConfig, ...] = _build_configs(PARK_GEOMETRY_POINTS_V12)


def resolve_geometry_config_v12(
    venue_id: int | None, game_date: str | None
) -> ParkGeometryConfig | None:
    """The v1.2 configuration in effect for `venue_id` on `game_date`, or `None`.

    Same contract as `park_geometry.resolve_geometry_config`, against the v1.2
    table: `None` means geometry unavailable, and callers must never guess the
    nearest configuration.
    """
    if venue_id is None or game_date is None:
        return None
    try:
        venue = int(venue_id)
    except (TypeError, ValueError):
        return None
    matches = [
        c
        for c in PARK_GEOMETRY_CONFIGS_V12
        if c.venue_id == venue
        and c.effective_start_date <= game_date
        and (c.effective_end_date is None or game_date <= c.effective_end_date)
    ]
    if len(matches) > 1:
        raise ParkGeometryValidationError(
            f"venue_id={venue} date={game_date} matches {[c.geometry_config_id for c in matches]}"
        )
    return matches[0] if matches else None
