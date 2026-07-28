"""Tests for the Contact Luck v0.5 reviewed venue-environment reference table."""

from __future__ import annotations

import pytest

from mlb_luck_score.data.park_geometry import TEMPORARY_OR_SPECIAL_VENUE_IDS
from mlb_luck_score.data.venue_environment import (
    VENUE_ENVIRONMENTS,
    VenueEnvironment,
    get_venue_environment,
    validate_venue_environments,
)


def test_real_table_is_valid():
    validate_venue_environments(VENUE_ENVIRONMENTS)


def test_real_table_covers_thirty_venues():
    assert len({v.venue_id for v in VENUE_ENVIRONMENTS}) == 30


def test_real_table_covers_no_temporary_venues():
    covered = {v.venue_id for v in VENUE_ENVIRONMENTS}
    assert covered.isdisjoint(TEMPORARY_OR_SPECIAL_VENUE_IDS)


def test_get_venue_environment_known_venue():
    env = get_venue_environment(3)
    assert env is not None
    assert env.venue_name == "Fenway Park"
    assert env.timezone == "America/New_York"


def test_get_venue_environment_unknown_venue_returns_none():
    assert get_venue_environment(999_999) is None


def test_get_venue_environment_handles_none_and_float_ids():
    assert get_venue_environment(None) is None
    assert get_venue_environment(3.0).venue_name == "Fenway Park"


def test_station_distance_km_is_nonnegative_and_bounded():
    for env in VENUE_ENVIRONMENTS:
        distance = env.station_distance_km()
        assert distance >= 0.0
        # Every assigned station should be within ~25km of its venue (verified
        # against real coordinates -- see module docstring).
        assert distance < 25.0


def test_coors_field_uses_stadium_specific_elevation():
    coors = get_venue_environment(19)
    assert coors is not None
    assert coors.elevation_m == pytest.approx(1609.344, abs=0.01)


def test_validation_rejects_duplicate_venue_id():
    base = VENUE_ENVIRONMENTS[0]
    duplicate = VenueEnvironment(
        venue_id=base.venue_id,
        venue_name=base.venue_name,
        latitude=base.latitude,
        longitude=base.longitude,
        elevation_m=base.elevation_m,
        elevation_source=base.elevation_source,
        timezone=base.timezone,
        weather_station_id=base.weather_station_id,
        weather_station_name=base.weather_station_name,
        weather_station_latitude=base.weather_station_latitude,
        weather_station_longitude=base.weather_station_longitude,
        weather_station_elevation_m=base.weather_station_elevation_m,
        source_note="duplicate test entry",
    )
    with pytest.raises(ValueError, match="duplicate venue_id"):
        validate_venue_environments((base, duplicate))


def test_validation_rejects_missing_source_note():
    with pytest.raises(ValueError, match="source_note"):
        VenueEnvironment(
            venue_id=999,
            venue_name="Test Venue",
            latitude=0.0,
            longitude=0.0,
            elevation_m=0.0,
            elevation_source="nearest_weather_station_elevation_proxy",
            timezone="UTC",
            weather_station_id="XXX",
            weather_station_name="Test Station",
            weather_station_latitude=0.0,
            weather_station_longitude=0.0,
            weather_station_elevation_m=0.0,
            source_note="",
        )


def test_validation_rejects_temporary_venue_entry():
    temp_id = next(iter(TEMPORARY_OR_SPECIAL_VENUE_IDS))
    bad = VenueEnvironment(
        venue_id=temp_id,
        venue_name="Temp Venue",
        latitude=0.0,
        longitude=0.0,
        elevation_m=0.0,
        elevation_source="nearest_weather_station_elevation_proxy",
        timezone="UTC",
        weather_station_id="XXX",
        weather_station_name="Test Station",
        weather_station_latitude=0.0,
        weather_station_longitude=0.0,
        weather_station_elevation_m=0.0,
        source_note="test",
    )
    with pytest.raises(ValueError, match="TEMPORARY_OR_SPECIAL_VENUE_IDS"):
        validate_venue_environments((bad,))


def test_every_venue_timezone_is_a_valid_iana_name():
    from zoneinfo import ZoneInfo

    for env in VENUE_ENVIRONMENTS:
        ZoneInfo(env.timezone)  # raises if invalid


def test_arizona_timezone_does_not_observe_dst():
    # Chase Field -- America/Phoenix never shifts UTC offset across the year,
    # unlike every other US venue's timezone.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    chase = get_venue_environment(15)
    assert chase is not None
    tz = ZoneInfo(chase.timezone)
    winter_offset = datetime(2024, 1, 15, 12, tzinfo=tz).utcoffset()
    summer_offset = datetime(2024, 7, 15, 12, tzinfo=tz).utcoffset()
    assert winter_offset == summer_offset
