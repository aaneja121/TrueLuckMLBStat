"""Contact Luck v0.4: reviewed outfield-wall-geometry reference data.

This module is the ONE place that defines physical outfield-wall geometry
(wall distance and, where known, wall height, as a function of spray angle)
for MLB venues. It follows the same pattern as `mlb_luck_score.data.
game_metadata_overrides`: a small, hand-curated, git-tracked table of
reference data (not a downloaded dataset), where every record documents its
source. Do not duplicate or reimplement this table elsewhere.

## Sourcing and review status

Every point in `PARK_GEOMETRY_POINTS` below was gathered by an AI coding
agent (Claude Code) from publicly available, cited sources (primarily
individual-ballpark Wikipedia articles and their own citations, cross-checked
against independent web searches where a figure looked internally
inconsistent -- see `source_reference`/`source_name`/`source_accessed_date`
on each record). **This is NOT the same as human review.** Every record's
`review_status` is `REVIEW_STATUS_AGENT_SOURCED` ("agent_sourced_pending_
human_review") -- a documented Version 0.4 research placeholder, not a
validated ground truth. A maintainer should spot-check records (especially
ones flagged in `notes` as simplified or uncertain) before treating this
table as authoritative. This matches CLAUDE.md's "use conservative
scientific language" and "document assumptions" rules.

## What is modeled

For each venue and each real, documented configuration-change era, five
discrete "reviewed wall points" are recorded at standard spray angles
(`_STANDARD_ANGLES`/`_STANDARD_LABELS` below): the left-field line, the
left-center power alley, straightaway center field, the right-center power
alley, and the right-field line. This is a deliberate simplification -- see
"Not modeled" below -- not a claim that MLB outfield walls are literal
5-point polygons. Wall DISTANCE is populated for essentially every point
(publicly published on team sites/media guides); wall HEIGHT is populated
only where a source explicitly gave a height for that specific location
(many sources give only one headline height, e.g. Fenway's 37-foot Green
Monster in left field) -- height is left `None` elsewhere rather than
guessed.

Three venues have MORE THAN ONE geometry configuration because their real
outfield walls changed during the 2021-2024 development window, each a
well-documented, dated renovation:

  - Oriole Park at Camden Yards (venue_id=2): the left-field wall was moved
    back and raised before Opening Day 2022.
  - Rogers Centre (venue_id=14): the outfield wall was reconfigured (moved
    in, staggered heights) before the 2023 season.
  - Comerica Park (venue_id=2394): the outfield fences were adjusted (and
    previously-posted distances corrected via laser measurement) before the
    2023 season.

See each `notes` field for specifics and citations.

## Not modeled (Version 0.4 scope)

- Full wall curvature between the five reviewed points (piecewise-linear
  interpolation only -- see `interpolate_wall_geometry`).
- Foul-territory polygons, exact roof/catwalk effects, or any
  outcome-derived "effective" distance.
- Temporary/neutral-site venues (spring-training/emergency home venues, the
  London Series, Mexico City Series, the Little League Classic field, the
  Rickwood Field tribute game, and the MLB Field of Dreams games) --
  deliberately left WITHOUT geometry entries (see
  `TEMPORARY_OR_SPECIAL_VENUE_IDS`) rather than fabricated. Field of Dreams
  (`venue_id=-1`, see `mlb_luck_score.data.game_metadata_overrides`)
  explicitly stays geometry-unavailable per that module's documented
  extension point.
- Minor localized wall irregularities that don't reduce to a single point on
  the standard 5-angle grid (e.g. Citizens Bank Park's zigzag "Monty's
  Angle" in left-center, Citi Field's right-field "nook" whose exact
  pre-2023 sub-dimensions could not be pinned down from available sources)
  are folded into the nearest standard point with a `notes` caveat, not
  separately modeled.

## Spray-angle convention

Matches `mlb_luck_score.data.clean_batted_balls._spray_angle_degrees`
exactly (verified against that module's docstring and tests): 0 degrees is
straightaway center field, negative angles are the third-base/left-field
side, positive angles are the first-base/right-field side. The standard
angle grid (`_STANDARD_ANGLES = (-45, -22.5, 0, 22.5, 45)`) also matches
`mlb_luck_score.data.clean_batted_balls._SPRAY_SECTOR_EDGES`'s +-45 degree
fair-territory boundary, so no interpolation is ever attempted outside the
angular range this repository already treats as modeled fair territory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

#: Standard reviewed-point spray angles (degrees), shared by every venue for
#: consistency. See module docstring for the sign convention. -45/+45 match
#: `mlb_luck_score.data.clean_batted_balls._SPRAY_SECTOR_EDGES`'s modeled
#: fair-territory boundary.
STANDARD_ANGLES: tuple[float, ...] = (-45.0, -22.5, 0.0, 22.5, 45.0)
STANDARD_LABELS: tuple[str, ...] = (
    "left_field_line",
    "left_center",
    "center_field",
    "right_center",
    "right_field_line",
)

#: Documented review status for every record in this Version 0.4 table --
#: see module docstring "Sourcing and review status". This is the ONLY
#: status currently used; the constant exists so a future human-reviewed
#: pass has an obvious place to record an upgraded status without changing
#: every call site.
REVIEW_STATUS_AGENT_SOURCED = "agent_sourced_pending_human_review"

#: Venue ids that are temporary, neutral-site, or otherwise special fields
#: used during 2021-2024 where reliable, reviewable wall-geometry dimensions
#: were not sourced for this Version 0.4 pass. Deliberately NOT populated
#: with geometry -- see module docstring "Not modeled". `-1` is the Field of
#: Dreams sentinel from `mlb_luck_score.data.game_metadata_overrides`.
TEMPORARY_OR_SPECIAL_VENUE_IDS: frozenset[int] = frozenset(
    {
        -1,  # MLB Field of Dreams (Dyersville, Iowa) -- 2021 & 2022
        2756,  # Sahlen Field, Buffalo -- Blue Jays temporary home, 2021
        2536,  # TD Ballpark, Dunedin -- Blue Jays temporary home, 2021
        5381,  # London Stadium -- London Series, 2023-2024
        5340,  # Estadio Alfredo Harp Helu -- Mexico City Series, 2023-2024
        2735,  # Bowman Field, Williamsport (Little League Classic; venue
        # name/sponsor changed across 2021 "BB&T Ballpark", 2022-2023
        # "Muncy Bank Ballpark", 2024 "Journey Bank Ballpark" -- same
        # physical field, one venue_id)
        3949,  # Rickwood Field -- 2024 MLB tribute game (temporary
        # modern-era fence configuration for the one-off game; not verified)
    }
)


class ParkGeometryValidationError(ValueError):
    """Raised when the reviewed geometry table (or a candidate table) is invalid."""


@dataclass(frozen=True)
class ParkGeometryPoint:
    """One reviewed wall-distance/height point for one venue geometry configuration.

    Flat, one-row-per-point schema (see README.md/CLAUDE.md "v0.4" section
    for the field list this mirrors). Multiple points sharing the same
    `geometry_config_id` together describe one era's wall geometry for one
    venue; multiple `geometry_config_id`s for the same `venue_id` describe
    that venue's wall changing over time (see module docstring).

    Attributes:
        venue_id: MLB Stats API venue id (matches `mlb_luck_score.data.
            join_venue_metadata`'s `venue_id` column; `-1` for the Field of
            Dreams sentinel, though that venue has no geometry points here).
        venue_name: Human-readable venue name, for readability/debugging
            only -- `venue_id` is the join key.
        geometry_config_id: Stable id for one venue-era (e.g.
            "camden_yards_2022"). Unique per (venue, era); every point
            sharing this id must share the same effective dates.
        effective_start_date: ISO date (YYYY-MM-DD), inclusive -- first date
            this configuration is believed to apply.
        effective_end_date: ISO date, inclusive, or `None` if this is the
            most recent known configuration (still in effect as of the
            `source_accessed_date` on this record).
        spray_angle_degrees: One of `STANDARD_ANGLES`.
        wall_distance_feet: Distance from home plate to the wall along this
            spray direction, in feet.
        wall_height_feet: Wall height at this point, in feet, or `None` if
            no source gave a height for this specific location (see module
            docstring -- never guessed).
        segment_label: One of `STANDARD_LABELS`.
        special_structure: Name of a notable physical feature at/near this
            point (e.g. "Green Monster"), or `None`.
        temporary_venue: Always `False` for entries in this table --
            temporary/special venues are deliberately excluded (see
            `TEMPORARY_OR_SPECIAL_VENUE_IDS`) rather than given a geometry
            row. Kept as an explicit field (rather than inferred) so a
            future temporary-venue entry, if ever reliably sourced, is
            unambiguous about its own status.
        source_name: Short human-readable source description.
        source_reference: URL or citation for `source_name`.
        source_accessed_date: ISO date this source was consulted.
        review_status: See `REVIEW_STATUS_AGENT_SOURCED` / module docstring.
        notes: Free-form caveats -- simplifications, unresolved
            sub-dimensions, cross-checks performed, etc. Required to be
            non-empty when a simplification was made (not enforced in code,
            reviewed by convention like `game_metadata_overrides.py`).
    """

    venue_id: int
    venue_name: str
    geometry_config_id: str
    effective_start_date: str
    effective_end_date: str | None
    spray_angle_degrees: float
    wall_distance_feet: float
    wall_height_feet: float | None
    segment_label: str
    special_structure: str | None
    temporary_venue: bool
    source_name: str
    source_reference: str
    source_accessed_date: str
    review_status: str
    notes: str = field(default="")


def _cfg(
    venue_id: int,
    venue_name: str,
    config_id: str,
    start: str,
    end: str | None,
    distances: tuple[float, float, float, float, float],
    heights: tuple[float | None, float | None, float | None, float | None, float | None],
    *,
    source_name: str,
    source_reference: str,
    source_accessed_date: str = "2026-07-27",
    special_structures: tuple[str | None, str | None, str | None, str | None, str | None] = (
        None,
        None,
        None,
        None,
        None,
    ),
    notes: str = "",
) -> tuple[ParkGeometryPoint, ...]:
    """Build the 5 standard-angle points for one venue geometry configuration."""
    return tuple(
        ParkGeometryPoint(
            venue_id=venue_id,
            venue_name=venue_name,
            geometry_config_id=config_id,
            effective_start_date=start,
            effective_end_date=end,
            spray_angle_degrees=angle,
            wall_distance_feet=distance,
            wall_height_feet=height,
            segment_label=label,
            special_structure=structure,
            temporary_venue=False,
            source_name=source_name,
            source_reference=source_reference,
            source_accessed_date=source_accessed_date,
            review_status=REVIEW_STATUS_AGENT_SOURCED,
            notes=notes,
        )
        for angle, label, distance, height, structure in zip(
            STANDARD_ANGLES, STANDARD_LABELS, distances, heights, special_structures, strict=True
        )
    )


#: Reviewed park-geometry reference table for the 30 primary 2021-2024 MLB
#: venues. See module docstring for sourcing/review-status caveats. Every
#: `venue_id` below is a real MLB Stats API venue id observed in
#: `data/processed/cleaned_development_data_with_venue.parquet`. Distances
#: prefer an "actual/measured" figure over a rounded marketing-posted one
#: where a source distinguished the two (documented per-record in `notes`
#: where relevant).
PARK_GEOMETRY_POINTS: tuple[ParkGeometryPoint, ...] = (
    # Angel Stadium (Los Angeles Angels)
    *_cfg(
        1,
        "Angel Stadium",
        "angel_stadium_v1",
        "2018-01-01",
        None,
        (347.0, 390.0, 396.0, 370.0, 350.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Angel Stadium",
        source_reference="https://en.wikipedia.org/wiki/Angel_Stadium",
        notes=(
            "347/390/396/370/350 confirmed via independent web-search cross-check "
            "(2018 offseason wall reconfiguration); right-center has an additional "
            "shallow marker (365 ft) not represented on the standard 5-point grid."
        ),
    ),
    # Truist Park (Atlanta Braves), opened 2017
    *_cfg(
        4705,
        "Truist Park",
        "truist_park_v1",
        "2017-01-01",
        None,
        (335.0, 385.0, 400.0, 375.0, 325.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Truist Park",
        source_reference="https://en.wikipedia.org/wiki/Truist_Park",
    ),
    # Oriole Park at Camden Yards (Baltimore Orioles) -- TWO configurations
    *_cfg(
        2,
        "Oriole Park at Camden Yards",
        "camden_yards_pre2022",
        "2000-01-01",
        "2022-04-06",
        (333.0, 376.0, 400.0, 373.0, 318.0),
        (7.0, None, None, None, None),
        source_name="Wikipedia: Oriole Park at Camden Yards",
        source_reference="https://en.wikipedia.org/wiki/Oriole_Park_at_Camden_Yards",
        notes=(
            "Exact original installation date of this configuration not verified; "
            "confirmed in effect through end of 2021 season. Left-field wall height "
            "7 ft. effective_end_date is the day before the 2022 Opening Day "
            "(2022-04-07 per MLB_REGULAR_SEASON_DATE_RANGES); construction was "
            "completed 'by Opening Day 2022' per source."
        ),
    ),
    *_cfg(
        2,
        "Oriole Park at Camden Yards",
        "camden_yards_2022_2024",
        "2022-04-07",
        None,
        (333.0, 384.0, 400.0, 373.0, 318.0),
        (13.0, None, None, None, None),
        source_name="Wikipedia: Oriole Park at Camden Yards",
        source_reference="https://en.wikipedia.org/wiki/Oriole_Park_at_Camden_Yards",
        notes=(
            "Left-field wall moved back up to 26.5 ft and raised from 7 ft to ~13 ft "
            "before Opening Day 2022 (straightaway left 384 ft, left-center reported "
            "up to 398 ft in some sources -- 384 ft used here as the left_center grid "
            "point). Source also documents a FURTHER 2024 sub-segment adjustment (part "
            "of the left-field wall moved in to 363 ft over an ~8-ft-wide segment) that "
            "is NOT separately modeled here -- insufficient point-level detail to place "
            "it on the standard angle grid; the 2022-2024 config is used as-is for all "
            "of 2022-2024, a documented simplification, not a fabricated 2024 config."
        ),
    ),
    # Fenway Park (Boston Red Sox)
    *_cfg(
        3,
        "Fenway Park",
        "fenway_park_v1",
        "1940-01-01",
        None,
        (310.0, 379.0, 390.0, 380.0, 302.0),
        (37.17, None, None, None, None),
        source_name="Wikipedia: Fenway Park",
        source_reference="https://en.wikipedia.org/wiki/Fenway_Park",
        special_structures=("Green Monster", None, None, None, None),
        notes=(
            "Center field posted as 389 ft 9 in (~390 ft used here). Deep right-center "
            "(bullpen area, 420 ft) is a distinct deeper sub-point not represented on "
            "the standard 5-point grid -- 380 ft ('right center') used instead."
        ),
    ),
    # Wrigley Field (Chicago Cubs)
    *_cfg(
        17,
        "Wrigley Field",
        "wrigley_field_v1",
        "2015-01-01",
        None,
        (355.0, 368.0, 400.0, 368.0, 353.0),
        (11.5, None, None, None, 11.5),
        source_name="Wikipedia: Wrigley Field",
        source_reference="https://en.wikipedia.org/wiki/Wrigley_Field",
        notes="Bleacher wall height 11.5 ft (corners were 15 ft prior to 2015 renovation).",
    ),
    # Guaranteed Rate Field / Rate Field (Chicago White Sox)
    *_cfg(
        4,
        "Guaranteed Rate Field",
        "guaranteed_rate_field_v1",
        "2001-01-01",
        None,
        (330.0, 375.0, 400.0, 375.0, 335.0),
        (8.0, 8.0, 8.0, 8.0, 8.0),
        source_name="Wikipedia: Guaranteed Rate Field (Rate Field)",
        source_reference="https://en.wikipedia.org/wiki/Guaranteed_Rate_Field",
        notes="Left-center/right-center distances are 'not posted' on the wall per source.",
    ),
    # Great American Ball Park (Cincinnati Reds)
    *_cfg(
        2602,
        "Great American Ball Park",
        "gabp_v1",
        "2003-01-01",
        None,
        (328.0, 379.0, 404.0, 370.0, 325.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Great American Ball Park",
        source_reference="https://en.wikipedia.org/wiki/Great_American_Ball_Park",
    ),
    # Progressive Field (Cleveland Guardians)
    *_cfg(
        5,
        "Progressive Field",
        "progressive_field_v1",
        "2015-01-01",
        None,
        (325.0, 370.0, 400.0, 375.0, 325.0),
        (19.0, None, 9.0, None, 9.0),
        source_name="Wikipedia: Progressive Field",
        source_reference="https://en.wikipedia.org/wiki/Progressive_Field",
        special_structures=("Little Green Monster", None, None, None, None),
        notes="Center field also reported as deep as 410 ft in some sources; 400 ft used.",
    ),
    # Coors Field (Colorado Rockies)
    *_cfg(
        19,
        "Coors Field",
        "coors_field_v1",
        "2016-01-01",
        None,
        (347.0, 390.0, 415.0, 375.0, 350.0),
        (13.0, None, None, 16.5, None),
        source_name="Wikipedia: Coors Field",
        source_reference="https://en.wikipedia.org/wiki/Coors_Field",
        notes=(
            "Wall heights in left field (13 ft) and right-center (16.5 ft) raised "
            "before the 2016 season; distances unchanged since then per source."
        ),
    ),
    # Comerica Park (Detroit Tigers) -- TWO configurations
    *_cfg(
        2394,
        "Comerica Park",
        "comerica_park_pre2023",
        "2003-01-01",
        "2023-03-29",
        (345.0, 370.0, 420.0, 365.0, 330.0),
        (None, None, None, None, None),
        source_name="Web search synthesis of Comerica Park dimension-change coverage",
        source_reference=(
            "https://www.mlb.com/news/comerica-park-dimensions-history ; "
            "https://www.espn.com/mlb/story/_/id/35428089/"
            "tigers-change-comerica-park-dimensions-encourage-offense"
        ),
        notes="In effect 2003 through end of 2022 season, per cited coverage of the 2023 change.",
    ),
    *_cfg(
        2394,
        "Comerica Park",
        "comerica_park_2023_2024",
        "2023-03-30",
        None,
        (342.0, 370.0, 412.0, 365.0, 330.0),
        (None, None, 7.0, 7.0, 7.0),
        source_name="Wikipedia: Comerica Park + web search synthesis",
        source_reference="https://en.wikipedia.org/wiki/Comerica_Park",
        notes=(
            "Before the 2023 season: center field fence moved in 10 ft to 412 ft, "
            "center/right-center/right field wall height lowered to 7 ft, and the "
            "previously-posted left-field distance (345 ft) was corrected to the "
            "actual laser-measured 342 ft (a labeling correction, not a physical wall "
            "move, per source)."
        ),
    ),
    # Minute Maid Park (Houston Astros)
    *_cfg(
        2392,
        "Minute Maid Park",
        "minute_maid_park_v1",
        "2017-01-01",
        None,
        (315.0, 366.0, 409.0, 370.0, 326.0),
        (19.0, None, None, None, None),
        source_name="Wikipedia: Minute Maid Park",
        source_reference="https://en.wikipedia.org/wiki/Minute_Maid_Park",
        special_structures=("Crawford Boxes", None, None, None, None),
        notes=(
            "Center field moved from 436 ft to 409 ft when Tal's Hill was removed "
            "before the 2017 season. Deep sub-points (399 ft deep left-center, 408 ft "
            "deep right-center) are not represented on the standard 5-point grid."
        ),
    ),
    # Kauffman Stadium (Kansas City Royals)
    *_cfg(
        7,
        "Kauffman Stadium",
        "kauffman_stadium_v1",
        "2009-01-01",
        "2025-12-31",
        (330.0, 387.0, 410.0, 387.0, 330.0),
        (9.0, 9.0, 9.0, 9.0, 9.0),
        source_name="Web search synthesis (Wikipedia figures for this park were internally "
        "inconsistent on first fetch and were corrected against a second, independent search)",
        source_reference="https://en.wikipedia.org/wiki/Kauffman_Stadium",
        notes=(
            "In effect 2009-2025 per source (superseded by an announced 2026 change, "
            "outside this repo's 2021-2024 development window and 2025 final-test "
            "season, so not modeled here). An automated first-pass extraction of this "
            "page mislabeled the 2026 alley figure (379 ft) as the left-field-line "
            "distance; corrected via an independent web search before being recorded."
        ),
    ),
    # Dodger Stadium (Los Angeles Dodgers)
    *_cfg(
        22,
        "Dodger Stadium",
        "dodger_stadium_v1",
        "1969-01-01",
        None,
        (330.0, 375.0, 400.0, 375.0, 330.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Dodger Stadium",
        source_reference="https://en.wikipedia.org/wiki/Dodger_Stadium",
        notes=(
            "Center field has been posted/marked as 395 ft since 1973 but is actually "
            "400 ft per source ('as has been the case since 1969'); the actual 400 ft "
            "figure is used here (see module docstring: prefer measured over posted)."
        ),
    ),
    # loanDepot park (Miami Marlins)
    *_cfg(
        4169,
        "loanDepot park",
        "loandepot_park_v1",
        "2020-01-01",
        None,
        (344.0, 386.0, 400.0, 387.0, 335.0),
        (None, None, None, None, None),
        source_name="Wikipedia: loanDepot park",
        source_reference="https://en.wikipedia.org/wiki/LoanDepot_Park",
        notes=(
            "Renamed from Marlins Park on 2021-03-31; no dimension change. Wall height "
            "varies 6-16 ft around the park per source -- no single representative "
            "height recorded rather than guessing one."
        ),
    ),
    # American Family Field (Milwaukee Brewers)
    *_cfg(
        32,
        "American Family Field",
        "american_family_field_v1",
        "2001-01-01",
        None,
        (344.0, 371.0, 400.0, 374.0, 337.0),
        (None, None, None, None, None),
        source_name="Wikipedia: American Family Field",
        source_reference="https://en.wikipedia.org/wiki/American_Family_Field",
        notes=(
            "Source gives both an 'actual' and a rounded 'posted' figure for left field "
            "(344 actual / 342 posted, as of 2021) and right field (337 actual / 345 "
            "posted); actual/measured values used here per module docstring convention."
        ),
    ),
    # Target Field (Minnesota Twins)
    *_cfg(
        3312,
        "Target Field",
        "target_field_v1",
        "2010-01-01",
        None,
        (339.0, 377.0, 404.0, 367.0, 328.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Target Field",
        source_reference="https://en.wikipedia.org/wiki/Target_Field",
        notes=(
            "Center field is asymmetric (left corner 411 ft, right corner 403 ft per "
            "source); 404 ft used as a single representative center-field point, a "
            "documented simplification -- interpolation does not reproduce this "
            "asymmetry."
        ),
    ),
    # Citi Field (New York Mets)
    *_cfg(
        3289,
        "Citi Field",
        "citi_field_v1",
        "2012-01-01",
        None,
        (335.0, 358.0, 408.0, 375.0, 330.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Citi Field",
        source_reference="https://en.wikipedia.org/wiki/Citi_Field",
        notes=(
            "A right-field wall 'nook' was removed in 2023, straightening that section "
            "of fence; available sources gave inconsistent/conflicting figures for the "
            "exact pre-2023 right-field distance, so it was NOT modeled as a separate "
            "2021-2022 configuration (would require fabricating a number). The "
            "post-2023 figure (330 ft) is used for the entire 2021-2024 window as a "
            "documented simplification -- right-field-line distance/margin features "
            "for 2021-2022 Citi Field plays carry this known imprecision."
        ),
    ),
    # Yankee Stadium (New York Yankees), current (2009) ballpark
    *_cfg(
        3313,
        "Yankee Stadium",
        "yankee_stadium_v1",
        "2009-01-01",
        None,
        (318.0, 399.0, 408.0, 385.0, 314.0),
        (8.42, None, None, None, 8.0),
        source_name="Wikipedia: Yankee Stadium",
        source_reference="https://en.wikipedia.org/wiki/Yankee_Stadium",
        notes="Wall height ~8.42 ft from left-field pole to the bullpens, tapering to 8 ft in right.",
    ),
    # Oakland Coliseum (Oakland/Athletics)
    *_cfg(
        10,
        "Oakland Coliseum",
        "oakland_coliseum_v1",
        "1996-01-01",
        None,
        (330.0, 388.0, 400.0, 388.0, 330.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Oakland Coliseum",
        source_reference="https://en.wikipedia.org/wiki/Oakland_Coliseum",
        notes="Configuration since the 1996 'Mount Davis' reconfiguration; used through the 2024 A's tenancy.",
    ),
    # Citizens Bank Park (Philadelphia Phillies)
    *_cfg(
        2681,
        "Citizens Bank Park",
        "citizens_bank_park_v1",
        "2004-01-01",
        None,
        (329.0, 374.0, 401.0, 369.0, 330.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Citizens Bank Park",
        source_reference="https://en.wikipedia.org/wiki/Citizens_Bank_Park",
        special_structures=(None, "Monty's Angle", None, None, None),
        notes=(
            "Left-center ('Monty's Angle') is a real zigzag wall reported at 409/381/387 "
            "ft across its width; 374 ft (the standard power-alley figure) is used as a "
            "single representative point -- the zigzag itself is not modeled."
        ),
    ),
    # PNC Park (Pittsburgh Pirates)
    *_cfg(
        31,
        "PNC Park",
        "pnc_park_v1",
        "2001-01-01",
        None,
        (320.0, 383.0, 399.0, 375.0, 320.0),
        (6.0, None, 10.0, None, 21.0),
        source_name="Wikipedia: PNC Park",
        source_reference="https://en.wikipedia.org/wiki/PNC_Park",
        notes=(
            "Right-field wall is 21 ft as a tribute to Roberto Clemente's uniform "
            "number. Deep left-center sub-point (410 ft) not represented on the "
            "standard grid."
        ),
    ),
    # Petco Park (San Diego Padres)
    *_cfg(
        2680,
        "Petco Park",
        "petco_park_v1",
        "2013-01-01",
        None,
        (336.0, 390.0, 396.0, 391.0, 331.0),
        (None, None, None, None, 8.0),
        source_name="Wikipedia: Petco Park",
        source_reference="https://en.wikipedia.org/wiki/Petco_Park",
        notes="Right-field wall height lowered from 11 ft to 8 ft before the 2013 season.",
    ),
    # Oracle Park (San Francisco Giants)
    *_cfg(
        2395,
        "Oracle Park",
        "oracle_park_v1",
        "2020-01-01",
        None,
        (339.0, 399.0, 391.0, 415.0, 309.0),
        (8.0, None, 8.5, 20.0, 24.0),
        source_name="Wikipedia: Oracle Park + web search cross-check",
        source_reference="https://en.wikipedia.org/wiki/Oracle_Park",
        special_structures=(None, None, None, "Triples Alley", None),
        notes=(
            "Bullpens relocated and fences adjusted before the 2020 season (left-center "
            "404->399 ft, center 399->391 ft, right-center 'Triples Alley' 421->415 ft). "
            "Center-field wall height reported as 7-10 ft in different sources; 8.5 ft "
            "used as a midpoint rather than a single cited figure -- lower confidence "
            "than other height values in this table."
        ),
    ),
    # T-Mobile Park (Seattle Mariners)
    *_cfg(
        680,
        "T-Mobile Park",
        "t_mobile_park_v1",
        "2013-01-01",
        None,
        (331.0, 378.0, 401.0, 381.0, 326.0),
        (None, None, None, None, None),
        source_name="Wikipedia: T-Mobile Park",
        source_reference="https://en.wikipedia.org/wiki/T-Mobile_Park",
    ),
    # Busch Stadium (St. Louis Cardinals)
    *_cfg(
        2889,
        "Busch Stadium",
        "busch_stadium_v1",
        "2006-01-01",
        None,
        (336.0, 375.0, 400.0, 375.0, 335.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Busch Stadium",
        source_reference="https://en.wikipedia.org/wiki/Busch_Stadium",
    ),
    # Tropicana Field (Tampa Bay Rays)
    *_cfg(
        12,
        "Tropicana Field",
        "tropicana_field_v1",
        "1998-01-01",
        None,
        (315.0, 370.0, 404.0, 370.0, 322.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Tropicana Field",
        source_reference="https://en.wikipedia.org/wiki/Tropicana_Field",
    ),
    # Globe Life Field (Texas Rangers)
    *_cfg(
        5325,
        "Globe Life Field",
        "globe_life_field_v1",
        "2020-01-01",
        None,
        (329.0, 372.0, 407.0, 374.0, 326.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Globe Life Field",
        source_reference="https://en.wikipedia.org/wiki/Globe_Life_Field",
    ),
    # Rogers Centre (Toronto Blue Jays) -- TWO configurations
    *_cfg(
        14,
        "Rogers Centre",
        "rogers_centre_pre2023",
        "2000-01-01",
        "2023-03-29",
        (328.0, 375.0, 400.0, 375.0, 328.0),
        (10.0, 10.0, 10.0, 10.0, 10.0),
        source_name="Web search synthesis of Rogers Centre pre-2023 dimension coverage",
        source_reference="https://www.mlbtraderumors.com/2023/01/blue-jays-announce-rogers-centres-new-outfield-dimensions.html",
        notes=(
            "Uniform 328 ft down the lines, 375 ft to the gaps, 400 ft to center, "
            "roughly 10 ft walls throughout, per coverage of the 2023 change. Exact "
            "original installation date not verified; confirmed in effect through end "
            "of 2022 season."
        ),
    ),
    *_cfg(
        14,
        "Rogers Centre",
        "rogers_centre_2023_2024",
        "2023-03-30",
        None,
        (328.0, 368.0, 400.0, 359.0, 328.0),
        (14.33, 11.17, 8.0, 14.33, 12.58),
        source_name="Wikipedia: Rogers Centre + web search synthesis",
        source_reference="https://en.wikipedia.org/wiki/Rogers_Centre",
        notes=(
            "Outfield wall reconfigured before the 2023 season: left-center in 7 ft to "
            "368 ft, right-center in 16 ft to 359 ft, center field unchanged at 400 ft. "
            "Wall height now staggered: 8 ft in center, 11.17 ft in left-center, 14.33 "
            "ft down the left-field line and in right-center, 12.58 ft down the "
            "right-field line."
        ),
    ),
    # Nationals Park (Washington Nationals)
    *_cfg(
        3309,
        "Nationals Park",
        "nationals_park_v1",
        "2008-01-01",
        None,
        (337.0, 377.0, 402.0, 370.0, 335.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Nationals Park",
        source_reference="https://en.wikipedia.org/wiki/Nationals_Park",
    ),
    # Chase Field (Arizona Diamondbacks)
    *_cfg(
        15,
        "Chase Field",
        "chase_field_v1",
        "1998-01-01",
        None,
        (330.0, 374.0, 407.0, 374.0, 334.0),
        (None, None, None, None, None),
        source_name="Wikipedia: Chase Field",
        source_reference="https://en.wikipedia.org/wiki/Chase_Field",
        notes="Deep left-/right-center sub-points (413 ft each) not represented on the standard grid.",
    ),
)


def validate_geometry_points(points: Sequence[ParkGeometryPoint]) -> None:
    """Validate a candidate reviewed-geometry table.

    Checks (raising `ParkGeometryValidationError`, listing every problem
    found rather than stopping at the first one):
      - Every record has a non-empty `source_note`-equivalent (`notes` may
        be empty, but `source_name`/`source_reference` must not be).
      - No two configurations for the SAME `venue_id` have overlapping
        `[effective_start_date, effective_end_date]` date ranges (an
        open-ended `effective_end_date=None` is treated as extending to
        infinity).
      - Every point within one `geometry_config_id` shares the same venue
        id/name and effective dates (internal consistency).
      - No venue id in `TEMPORARY_OR_SPECIAL_VENUE_IDS` has geometry points
        (those venues are deliberately left geometry-unavailable).
    """
    problems: list[str] = []

    by_config: dict[str, list[ParkGeometryPoint]] = {}
    for p in points:
        by_config.setdefault(p.geometry_config_id, []).append(p)
        if not p.source_name or not p.source_reference:
            problems.append(f"{p.geometry_config_id}/{p.segment_label}: missing source metadata")
        if p.venue_id in TEMPORARY_OR_SPECIAL_VENUE_IDS:
            problems.append(
                f"{p.geometry_config_id}: venue_id={p.venue_id} is in "
                "TEMPORARY_OR_SPECIAL_VENUE_IDS and must not have geometry points"
            )

    configs_by_venue: dict[int, list[tuple[str, str, str | None]]] = {}
    for config_id, config_points in by_config.items():
        venues = {p.venue_id for p in config_points}
        starts = {p.effective_start_date for p in config_points}
        ends = {p.effective_end_date for p in config_points}
        if len(venues) != 1:
            problems.append(f"{config_id}: points disagree on venue_id ({venues})")
        if len(starts) != 1 or len(ends) != 1:
            problems.append(f"{config_id}: points disagree on effective dates")
        angles = sorted(p.spray_angle_degrees for p in config_points)
        if angles != sorted(STANDARD_ANGLES):
            problems.append(f"{config_id}: angle set {angles} does not match STANDARD_ANGLES")
        if venues and starts and ends:
            venue_id = next(iter(venues))
            configs_by_venue.setdefault(venue_id, []).append(
                (config_id, next(iter(starts)), next(iter(ends)))
            )

    for venue_id, configs in configs_by_venue.items():
        ordered = sorted(configs, key=lambda c: c[1])
        for (id_a, _start_a, end_a), (id_b, start_b, _end_b) in zip(
            ordered, ordered[1:], strict=False
        ):
            end_a_date = date.max.isoformat() if end_a is None else end_a
            if end_a_date >= start_b:
                problems.append(
                    f"venue_id={venue_id}: configs '{id_a}' (ends {end_a}) and '{id_b}' "
                    f"(starts {start_b}) have overlapping effective-date ranges"
                )

    if problems:
        raise ParkGeometryValidationError(
            f"{len(problems)} problem(s) found in geometry reference table:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


# Fail fast at import time -- mirrors mlb_luck_score.eligibility's module-level
# assertions. A broken reviewed table must never silently ship.
validate_geometry_points(PARK_GEOMETRY_POINTS)


@dataclass(frozen=True)
class ParkGeometryConfig:
    """One resolved venue-era geometry configuration (grouped points)."""

    venue_id: int
    venue_name: str
    geometry_config_id: str
    effective_start_date: str
    effective_end_date: str | None
    points: tuple[ParkGeometryPoint, ...]  # sorted by spray_angle_degrees
    review_status: str


def _build_configs(points: Sequence[ParkGeometryPoint]) -> tuple[ParkGeometryConfig, ...]:
    by_config: dict[str, list[ParkGeometryPoint]] = {}
    for p in points:
        by_config.setdefault(p.geometry_config_id, []).append(p)

    configs = []
    for config_id, config_points in by_config.items():
        ordered = tuple(sorted(config_points, key=lambda p: p.spray_angle_degrees))
        first = ordered[0]
        configs.append(
            ParkGeometryConfig(
                venue_id=first.venue_id,
                venue_name=first.venue_name,
                geometry_config_id=config_id,
                effective_start_date=first.effective_start_date,
                effective_end_date=first.effective_end_date,
                points=ordered,
                review_status=first.review_status,
            )
        )
    return tuple(configs)


#: Resolved configurations built from `PARK_GEOMETRY_POINTS` at import time.
PARK_GEOMETRY_CONFIGS: tuple[ParkGeometryConfig, ...] = _build_configs(PARK_GEOMETRY_POINTS)


def resolve_geometry_config(
    venue_id: int | None, game_date: str | None
) -> ParkGeometryConfig | None:
    """Find the geometry configuration in effect for `venue_id` on `game_date`.

    Args:
        venue_id: A venue id, or `None`/NaN-like if unknown.
        game_date: An ISO date string (`YYYY-MM-DD`, e.g. from a `game_date`
            column), or `None` if unknown.

    Returns:
        The matching `ParkGeometryConfig`, or `None` if no configuration is
        available for this venue at all, or none covers this specific date
        (e.g. the date falls in an un-reviewed gap) -- callers must treat
        `None` as "geometry unavailable/uncertain", never guess the nearest
        configuration (see module docstring / README.md v0.4 section).

    Raises:
        ParkGeometryValidationError: If more than one configuration matches
            (would indicate a validation bug -- `validate_geometry_points`
            should have caught this already at import time).
    """
    if venue_id is None or game_date is None:
        return None
    try:
        venue_id_int = int(venue_id)
    except (TypeError, ValueError):
        return None

    matches = [
        c
        for c in PARK_GEOMETRY_CONFIGS
        if c.venue_id == venue_id_int
        and c.effective_start_date <= game_date
        and (c.effective_end_date is None or game_date <= c.effective_end_date)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ParkGeometryValidationError(
            f"venue_id={venue_id_int} date={game_date} matches {len(matches)} geometry "
            f"configurations: {[c.geometry_config_id for c in matches]} -- this should "
            "have been rejected by validate_geometry_points at import time."
        )
    return matches[0]


@dataclass(frozen=True)
class GeometryLookupResult:
    """Result of interpolating wall geometry at one spray angle.

    Attributes:
        wall_distance_feet: Interpolated (or exact, if `spray_angle_degrees`
            matched a reviewed point) wall distance, in feet.
        wall_height_feet: Interpolated wall height, or `None` if either
            bracketing reviewed point lacks a height value.
        segment_label: Label of the nearest reviewed point, or a
            "between X and Y" description when strictly between two points.
        distance_source_type: `"measured"` if `spray_angle_degrees` exactly
            matched a reviewed point (within floating-point tolerance),
            else `"interpolated"`.
        nearest_point_distance_degrees: Angular distance (degrees) to the
            nearest reviewed point -- a simple interpolation-confidence
            indicator (0 at a reviewed point, up to 11.25 degrees at the
            midpoint between two standard-grid points).
        geometry_config_id: Which configuration this came from.
    """

    wall_distance_feet: float
    wall_height_feet: float | None
    segment_label: str
    distance_source_type: str
    nearest_point_distance_degrees: float
    geometry_config_id: str


def interpolate_wall_geometry(
    config: ParkGeometryConfig, spray_angle_degrees: float
) -> GeometryLookupResult | None:
    """Piecewise-linearly interpolate wall distance/height at `spray_angle_degrees`.

    Never extrapolates: if `spray_angle_degrees` falls outside
    `[STANDARD_ANGLES[0], STANDARD_ANGLES[-1]]` (i.e. outside the +-45
    degree modeled fair-territory range), returns `None` -- callers must
    treat this as "not applicable" (e.g. a play that will map to
    `geometry_status="outside_modeled_angular_range"`), never as a reason to
    hold the nearest boundary value flat.

    Args:
        config: A resolved `ParkGeometryConfig` (see `resolve_geometry_config`).
        spray_angle_degrees: The play's spray angle, same convention as
            `mlb_luck_score.data.clean_batted_balls.spray_angle_approx`.

    Returns:
        A `GeometryLookupResult`, or `None` if outside the modeled range.
    """
    points = config.points  # sorted by angle
    lo, hi = points[0].spray_angle_degrees, points[-1].spray_angle_degrees
    if spray_angle_degrees < lo or spray_angle_degrees > hi:
        return None

    for p in points:
        if abs(p.spray_angle_degrees - spray_angle_degrees) < 1e-9:
            return GeometryLookupResult(
                wall_distance_feet=p.wall_distance_feet,
                wall_height_feet=p.wall_height_feet,
                segment_label=p.segment_label,
                distance_source_type="measured",
                nearest_point_distance_degrees=0.0,
                geometry_config_id=config.geometry_config_id,
            )

    left = max(
        (p for p in points if p.spray_angle_degrees < spray_angle_degrees),
        key=lambda p: p.spray_angle_degrees,
    )
    right = min(
        (p for p in points if p.spray_angle_degrees > spray_angle_degrees),
        key=lambda p: p.spray_angle_degrees,
    )

    span = right.spray_angle_degrees - left.spray_angle_degrees
    weight_right = (spray_angle_degrees - left.spray_angle_degrees) / span

    distance = left.wall_distance_feet + weight_right * (
        right.wall_distance_feet - left.wall_distance_feet
    )
    if left.wall_height_feet is not None and right.wall_height_feet is not None:
        height = left.wall_height_feet + weight_right * (
            right.wall_height_feet - left.wall_height_feet
        )
    else:
        height = None

    nearest_distance_degrees = min(
        abs(spray_angle_degrees - left.spray_angle_degrees),
        abs(spray_angle_degrees - right.spray_angle_degrees),
    )
    nearest_label = (
        left.segment_label
        if nearest_distance_degrees == abs(spray_angle_degrees - left.spray_angle_degrees)
        else right.segment_label
    )

    return GeometryLookupResult(
        wall_distance_feet=distance,
        wall_height_feet=height,
        segment_label=f"between {left.segment_label} and {right.segment_label} (nearest: {nearest_label})",
        distance_source_type="interpolated",
        nearest_point_distance_degrees=nearest_distance_degrees,
        geometry_config_id=config.geometry_config_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the reviewed park-geometry reference table and report a summary. "
            "There is no separate 'build from raw source' step for this table (unlike "
            "venue metadata) -- it is small, hand-curated reference data edited directly "
            "in this module's source code, like mlb_luck_score.data."
            "game_metadata_overrides. This command is both 'make build-park-geometry' "
            "and 'make validate-park-geometry' in the Makefile."
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    parser.parse_args(argv)

    try:
        validate_geometry_points(PARK_GEOMETRY_POINTS)
    except ParkGeometryValidationError as exc:
        logger.error(str(exc))
        return 2

    venues = sorted({c.venue_id for c in PARK_GEOMETRY_CONFIGS})
    multi_config_venues = sorted(
        {
            c.venue_id
            for c in PARK_GEOMETRY_CONFIGS
            if sum(1 for c2 in PARK_GEOMETRY_CONFIGS if c2.venue_id == c.venue_id) > 1
        }
    )
    logger.info("Geometry reference table is valid.")
    logger.info("Reviewed points: %d", len(PARK_GEOMETRY_POINTS))
    logger.info("Geometry configurations: %d", len(PARK_GEOMETRY_CONFIGS))
    logger.info("Venues covered: %d", len(venues))
    logger.info(
        "Venues with more than one configuration (real dimension changes): %s", multi_config_venues
    )
    logger.info(
        "Temporary/special venues (deliberately NOT covered): %d -- %s",
        len(TEMPORARY_OR_SPECIAL_VENUE_IDS),
        sorted(TEMPORARY_OR_SPECIAL_VENUE_IDS),
    )
    review_statuses = sorted({p.review_status for p in PARK_GEOMETRY_POINTS})
    logger.info("Review status(es) present: %s (see module docstring)", review_statuses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
