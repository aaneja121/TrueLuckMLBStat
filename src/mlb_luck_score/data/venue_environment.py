"""Contact Luck v0.5: reviewed venue-environment reference data.

This module is the ONE place that defines each venue's fixed physical/
environmental properties needed for weather matching: approximate location,
stadium elevation, IANA timezone, and the nearest reliable historical
weather-observation station. It follows the same pattern as `mlb_luck_score.
data.park_geometry` and `mlb_luck_score.data.game_metadata_overrides`: a
small, hand-curated, git-tracked table of reference data (not a downloaded
dataset), where every record documents its source. Do not duplicate or
reimplement this table elsewhere.

Roof/retractable-roof classification is deliberately NOT duplicated here --
`mlb_luck_score.data.download_game_metadata` already fetches a per-venue
`roof_type` ("Open" / "Retractable" / "Dome") from the MLB Stats API
`/venues/{id}?hydrate=fieldInfo` endpoint, and `mlb_luck_score.data.
build_game_weather` combines that with the PER-GAME weather condition text
(e.g. `"Roof Closed"`) to determine actual open/closed status for
retractable-roof venues on a given day. Duplicating a second, possibly
inconsistent roof classification here would violate "keep reference data
in ONE place" (see CLAUDE.md).

## Sourcing and review status

- `latitude`/`longitude`: approximate venue coordinates from general
  public knowledge of these well-known, stable public landmarks. These are
  NOT independently re-verified per-venue against a citation for this pass
  (unlike `mlb_luck_score.data.park_geometry`, which cites a specific
  source per record) -- treat them as accurate to roughly city-block
  precision, sufficient for computing an approximate weather-station
  distance, not for anything requiring survey-grade precision.
  `review_status="agent_sourced_pending_human_review"`, same caveat as the
  park-geometry table.
- `elevation_m`: for Coors Field (venue_id=19) ONLY, this is the
  famous, well-documented, MLB/Rockies-publicized stadium elevation
  (5,280 ft = 1609.344 m -- the literal "mile-high" row of purple seats).
  For every other venue, `elevation_m` is copied from its assigned weather
  station's own reported elevation (see below) as a documented proxy, NOT
  an independently surveyed stadium elevation -- most assigned stations are
  in the same metro area and same broad elevation band as their venue, but
  this is a real, acknowledged simplification (see `elevation_source`).
- `weather_station_id`, `weather_station_latitude/longitude/elevation_m`:
  VERIFIED programmatically -- every station below was queried directly
  against the Iowa Environmental Mesonet ASOS archive
  (`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`, a public,
  documented, no-API-key historical METAR/ASOS archive) and confirmed to
  return real historical observations for the 2021-2024 window, with its
  coordinates/elevation read directly from that response (i.e. from NOAA/
  Environment Canada's own station metadata, not a secondary source).
- `timezone`: IANA timezone name for the venue's metro area, used to
  convert each game's local start time for weather-observation matching.
  Arizona (Chase Field) does not observe daylight saving time --
  `America/Phoenix` handles this correctly via the `zoneinfo` database.

## Coverage

Covers the 30 primary 2021-2024 MLB venues (matches `mlb_luck_score.data.
park_geometry.PARK_GEOMETRY_POINTS`'s venue coverage exactly). Temporary/
neutral-site venues (`mlb_luck_score.data.park_geometry.
TEMPORARY_OR_SPECIAL_VENUE_IDS`) are deliberately NOT covered here either --
same rationale as park geometry: do not fabricate environmental reference
data for one-off venues rather than guess a station/elevation for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

REVIEW_STATUS_AGENT_SOURCED = "agent_sourced_pending_human_review"

ELEVATION_SOURCE_STADIUM_SPECIFIC = "stadium_specific_published_value"
ELEVATION_SOURCE_STATION_PROXY = "nearest_weather_station_elevation_proxy"


@dataclass(frozen=True)
class VenueEnvironment:
    """Fixed physical/environmental reference data for one venue.

    Attributes:
        venue_id: MLB Stats API venue id (matches `mlb_luck_score.data.
            join_venue_metadata`'s `venue_id` column).
        venue_name: Human-readable name, for readability/debugging only.
        latitude / longitude: Approximate venue coordinates (decimal
            degrees) -- see module docstring for precision caveat.
        elevation_m: Venue elevation in meters -- see `elevation_source`.
        elevation_source: `ELEVATION_SOURCE_STADIUM_SPECIFIC` (Coors Field
            only) or `ELEVATION_SOURCE_STATION_PROXY` (every other venue).
        timezone: IANA timezone name for local game-time conversion.
        weather_station_id: Station identifier usable with the Iowa
            Environmental Mesonet ASOS request endpoint (3-letter US ASOS
            id, or 4-letter ICAO for the one Canadian venue).
        weather_station_name: Human-readable station name/airport.
        weather_station_latitude / _longitude / _elevation_m: The
            station's own reported coordinates/elevation, verified via a
            live query (see module docstring) -- used to compute
            `station_distance_km`.
        source_note: Provenance summary, required non-empty.
        review_status: See module docstring.
    """

    venue_id: int
    venue_name: str
    latitude: float
    longitude: float
    elevation_m: float
    elevation_source: str
    timezone: str
    weather_station_id: str
    weather_station_name: str
    weather_station_latitude: float
    weather_station_longitude: float
    weather_station_elevation_m: float
    source_note: str
    review_status: str = REVIEW_STATUS_AGENT_SOURCED
    fallback_station_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.source_note:
            raise ValueError(
                f"VenueEnvironment for venue_id={self.venue_id} has no source_note -- "
                "every record must document how it was verified."
            )

    def station_distance_km(self) -> float:
        """Great-circle (haversine) distance from the venue to its assigned station."""
        return _haversine_km(
            self.latitude,
            self.longitude,
            self.weather_station_latitude,
            self.weather_station_longitude,
        )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


#: Reviewed venue-environment reference table for the 30 primary 2021-2024
#: MLB venues. See module docstring for sourcing/review-status caveats.
_STATION_VERIFICATION_NOTE = (
    "Weather station coordinates/elevation verified via a live query against "
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py, confirmed to return real "
    "2024 observations."
)

VENUE_ENVIRONMENTS: tuple[VenueEnvironment, ...] = (
    VenueEnvironment(
        1,
        "Angel Stadium",
        33.8003,
        -117.8827,
        16.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "SNA",
        "John Wayne Airport, Santa Ana CA",
        33.6757,
        -117.8682,
        16.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        4705,
        "Truist Park",
        33.8908,
        -84.4678,
        256.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "FTY",
        "Fulton County Airport (Charlie Brown Field), Atlanta GA",
        33.7800,
        -84.5200,
        256.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2,
        "Oriole Park at Camden Yards",
        39.2839,
        -76.6218,
        42.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "BWI",
        "Baltimore/Washington International Airport",
        39.1733,
        -76.6841,
        42.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        3,
        "Fenway Park",
        42.3467,
        -71.0972,
        9.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "BOS",
        "Boston Logan International Airport",
        42.3606,
        -71.0097,
        9.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        17,
        "Wrigley Field",
        41.9484,
        -87.6553,
        205.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "ORD",
        "Chicago O'Hare International Airport",
        41.9602,
        -87.9316,
        205.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        4,
        "Guaranteed Rate Field",
        41.8299,
        -87.6338,
        188.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "MDW",
        "Chicago Midway International Airport",
        41.7860,
        -87.7524,
        188.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2602,
        "Great American Ball Park",
        39.0975,
        -84.5074,
        155.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "LUK",
        "Cincinnati Municipal Airport (Lunken Field)",
        39.1033,
        -84.4186,
        155.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        5,
        "Progressive Field",
        41.4962,
        -81.6852,
        178.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "BKL",
        "Cleveland Burke Lakefront Airport",
        41.5175,
        -81.6833,
        178.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        19,
        "Coors Field",
        39.7559,
        -104.9942,
        1609.344,
        ELEVATION_SOURCE_STADIUM_SPECIFIC,
        "America/Denver",
        "BKF",
        "Buckley Space Force Base, Aurora CO",
        39.7017,
        -104.7517,
        1726.0,
        "Elevation is the famous, widely-published Coors Field figure (5,280 ft = one mile "
        "above sea level, marked by a row of purple seats in the upper deck) -- this is the "
        "one venue where the station-proxy elevation (Buckley SFB, ~1726 m, further from "
        "downtown Denver in the foothills) would be materially wrong for a park whose whole "
        "notoriety is altitude/air-density effects, so the actual stadium figure is used "
        "instead. " + _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2394,
        "Comerica Park",
        42.3390,
        -83.0485,
        190.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Detroit",
        "DET",
        "Coleman A. Young International Airport, Detroit MI",
        42.4092,
        -83.0099,
        190.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2392,
        "Minute Maid Park",
        29.7573,
        -95.3555,
        14.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "HOU",
        "William P. Hobby Airport, Houston TX",
        29.6375,
        -95.2824,
        14.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        7,
        "Kauffman Stadium",
        39.0517,
        -94.4803,
        227.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "MKC",
        "Kansas City Downtown Airport",
        39.1230,
        -94.5930,
        227.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        22,
        "Dodger Stadium",
        34.0739,
        -118.2400,
        236.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "BUR",
        "Hollywood Burbank Airport",
        34.2007,
        -118.3587,
        236.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        4169,
        "loanDepot park",
        25.7781,
        -80.2196,
        4.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "MIA",
        "Miami International Airport",
        25.7880,
        -80.3169,
        4.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        32,
        "American Family Field",
        43.0280,
        -87.9712,
        204.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "MKE",
        "Milwaukee Mitchell International Airport",
        42.9472,
        -87.8967,
        204.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        3312,
        "Target Field",
        44.9817,
        -93.2777,
        265.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "MSP",
        "Minneapolis-St Paul International Airport",
        44.8854,
        -93.2313,
        265.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        3289,
        "Citi Field",
        40.7571,
        -73.8458,
        9.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "LGA",
        "LaGuardia Airport, New York NY",
        40.7794,
        -73.8803,
        9.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        3313,
        "Yankee Stadium",
        40.8296,
        -73.9262,
        9.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "LGA",
        "LaGuardia Airport, New York NY",
        40.7794,
        -73.8803,
        9.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        10,
        "Oakland Coliseum",
        37.7516,
        -122.2005,
        2.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "OAK",
        "Oakland International Airport",
        37.7178,
        -122.2330,
        2.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2681,
        "Citizens Bank Park",
        39.9061,
        -75.1665,
        2.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "PHL",
        "Philadelphia International Airport",
        39.8734,
        -75.2266,
        2.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        31,
        "PNC Park",
        40.4468,
        -80.0057,
        382.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "AGC",
        "Allegheny County Airport, Pittsburgh PA",
        40.3547,
        -79.9217,
        382.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2680,
        "Petco Park",
        32.7073,
        -117.1566,
        9.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "SAN",
        "San Diego International Airport",
        32.7339,
        -117.1845,
        9.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2395,
        "Oracle Park",
        37.7786,
        -122.3893,
        5.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "SFO",
        "San Francisco International Airport",
        37.6190,
        -122.3749,
        5.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        680,
        "T-Mobile Park",
        47.5914,
        -122.3325,
        5.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Los_Angeles",
        "BFI",
        "Boeing Field / King County International Airport, Seattle WA",
        47.5300,
        -122.3000,
        5.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        2889,
        "Busch Stadium",
        38.6226,
        -90.1928,
        171.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "STL",
        "St. Louis Lambert International Airport",
        38.7525,
        -90.3734,
        171.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        12,
        "Tropicana Field",
        27.7683,
        -82.6534,
        3.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "PIE",
        "St. Pete-Clearwater International Airport",
        27.9100,
        -82.6874,
        3.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        5325,
        "Globe Life Field",
        32.7473,
        -97.0842,
        192.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Chicago",
        "GKY",
        "Arlington Municipal Airport, Arlington TX",
        32.6639,
        -97.0943,
        192.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        14,
        "Rogers Centre",
        43.6414,
        -79.3894,
        173.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Toronto",
        "CYYZ",
        "Toronto Pearson International Airport",
        43.6772,
        -79.6306,
        173.0,
        "Toronto Pearson (CYYZ) is ~20 km from downtown Toronto/Rogers Centre -- the "
        "nearest Environment Canada ASOS-equivalent station available through Iowa "
        "Environmental Mesonet; a closer downtown station (e.g. Billy Bishop City Airport) "
        "was not confirmed available through this archive. " + _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        3309,
        "Nationals Park",
        38.8730,
        -77.0074,
        20.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/New_York",
        "DCA",
        "Ronald Reagan Washington National Airport",
        38.8472,
        -77.0345,
        20.0,
        _STATION_VERIFICATION_NOTE,
    ),
    VenueEnvironment(
        15,
        "Chase Field",
        33.4455,
        -112.0667,
        337.0,
        ELEVATION_SOURCE_STATION_PROXY,
        "America/Phoenix",
        "PHX",
        "Phoenix Sky Harbor International Airport",
        33.4343,
        -112.0116,
        337.0,
        _STATION_VERIFICATION_NOTE,
    ),
)


def validate_venue_environments(venues: tuple[VenueEnvironment, ...]) -> None:
    """Validate a candidate venue-environment table.

    Checks: unique `venue_id`, non-empty `source_note` (enforced per-record
    by `__post_init__` already, re-checked here for a candidate list built
    without going through the dataclass), and that no venue appears in
    `mlb_luck_score.data.park_geometry.TEMPORARY_OR_SPECIAL_VENUE_IDS`.
    """
    from mlb_luck_score.data.park_geometry import TEMPORARY_OR_SPECIAL_VENUE_IDS

    problems: list[str] = []
    seen: set[int] = set()
    for v in venues:
        if v.venue_id in seen:
            problems.append(f"duplicate venue_id={v.venue_id}")
        seen.add(v.venue_id)
        if not v.source_note:
            problems.append(f"venue_id={v.venue_id}: missing source_note")
        if v.venue_id in TEMPORARY_OR_SPECIAL_VENUE_IDS:
            problems.append(
                f"venue_id={v.venue_id} is in TEMPORARY_OR_SPECIAL_VENUE_IDS and must not "
                "have environment reference data"
            )
    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in venue-environment table:\n" + "\n".join(problems)
        )


validate_venue_environments(VENUE_ENVIRONMENTS)

VENUE_ENVIRONMENT_BY_ID: dict[int, VenueEnvironment] = {v.venue_id: v for v in VENUE_ENVIRONMENTS}


def get_venue_environment(venue_id: int | None) -> VenueEnvironment | None:
    """Look up reviewed environment reference data for a venue, or `None` if unavailable."""
    if venue_id is None:
        return None
    try:
        return VENUE_ENVIRONMENT_BY_ID.get(int(venue_id))
    except (TypeError, ValueError):
        return None
