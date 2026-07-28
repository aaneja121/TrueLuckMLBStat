"""Contact Luck v0.5: build the reproducible game_pk-keyed weather table.

Usage:

    python -m mlb_luck_score.data.build_game_weather \\
        --seasons 2021 2022 2023 2024 --raw-dir data/raw \\
        --output data/processed/game_weather.parquet

Combines the raw caches produced by `mlb_luck_score.data.
download_historical_weather` (per-season MLB schedule weather, per-station
-season ASOS observations) with `mlb_luck_score.data.venue_environment`
(station assignment/timezone/elevation) and each game's `roof_type` (from
the existing `mlb_luck_score.data.download_game_metadata` cache) into one
row per `game_pk`.

Time matching: for each game, the nearest ASOS observation AT OR BEFORE the
game's scheduled UTC start time is used (falling back to the nearest
observation shortly AFTER start only if no prior observation exists within
`MAX_OFFSET_MINUTES_FAIR`); the exact offset is preserved
(`weather_time_offset_minutes`), and a game is never matched to an
observation farther away than `MAX_OFFSET_MINUTES_FAIR` (3 hours) -- beyond
that, `weather_status="no_observation_within_window"` and every derived
weather field is left null, never filled with a distant observation.

Roof handling: `roof_status` is one of `outdoor_open_air`,
`retractable_roof_open`, `retractable_roof_closed`, `fixed_indoor`, or
`roof_status_unknown` (see `classify_roof_status`). For
`retractable_roof_closed`/`fixed_indoor`/`roof_status_unknown` games, the
RAW external observation is preserved, but `effective_temperature_c`/
`effective_relative_humidity_pct`/`effective_pressure_hpa`/`air_density_kg_m3`
are left null (no reviewed indoor-climate assumption exists in this
repository -- see CLAUDE.md "never fabricate") and `effective_wind_speed_mps`
is forced to 0.0 (a closed roof genuinely blocks outdoor wind -- this is a
physical certainty, not a guess).

This module never touches the network -- it only combines already-downloaded
raw caches. This module never touches 2025 (`assert_seasons_allowed`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    assert_seasons_allowed,
    game_metadata_path,
    schedule_weather_path,
    station_weather_path,
)
from mlb_luck_score.data.venue_environment import VenueEnvironment, get_venue_environment
from mlb_luck_score.data.weather_physics import (
    fahrenheit_to_celsius,
    moist_air_density_kg_m3,
    parse_wind_text,
    pressure_at_elevation_hpa,
)

logger = logging.getLogger(__name__)

#: Observation-time-matching tolerances (minutes). A "good" match is within
#: one hour of first pitch; "fair" is within three hours (routine ASOS
#: reports are hourly, so three hours is already a generous, documented
#: upper bound -- see module docstring "Do not silently match a game to an
#: observation many hours away").
MAX_OFFSET_MINUTES_GOOD = 60
MAX_OFFSET_MINUTES_FAIR = 180
#: How far AFTER game start a fallback observation may be used, only when no
#: observation before game start exists within `MAX_OFFSET_MINUTES_FAIR`.
MAX_AFTER_FALLBACK_MINUTES = 60

ROOF_STATUS_OUTDOOR_OPEN_AIR = "outdoor_open_air"
ROOF_STATUS_RETRACTABLE_OPEN = "retractable_roof_open"
ROOF_STATUS_RETRACTABLE_CLOSED = "retractable_roof_closed"
ROOF_STATUS_FIXED_INDOOR = "fixed_indoor"
ROOF_STATUS_UNKNOWN = "roof_status_unknown"
#: Roof statuses where outdoor weather does not act on the ball and no
#: reviewed indoor-climate assumption exists -- effective conditions unavailable.
_EFFECTIVE_CONDITIONS_UNAVAILABLE_STATUSES = frozenset(
    {ROOF_STATUS_RETRACTABLE_CLOSED, ROOF_STATUS_FIXED_INDOOR, ROOF_STATUS_UNKNOWN}
)
#: Roof statuses where wind is physically blocked from acting on the ball.
_WIND_SUPPRESSED_STATUSES = frozenset({ROOF_STATUS_RETRACTABLE_CLOSED, ROOF_STATUS_FIXED_INDOOR})

WEATHER_STATUS_OK = "ok"
WEATHER_STATUS_EFFECTIVE_UNAVAILABLE_INDOOR = "effective_conditions_unavailable_indoor"
WEATHER_STATUS_NO_OBSERVATION = "no_observation_within_window"
WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE = "venue_environment_unavailable"
WEATHER_STATUS_MISSING_SCHEDULE_WEATHER = "missing_schedule_weather"
WEATHER_STATUS_MISSING_START_TIME = "missing_scheduled_start_time"


def classify_roof_status(roof_type: str | None, weather_condition: str | None) -> str:
    """Classify a game's roof status from the venue's `roof_type` and reported condition text.

    Args:
        roof_type: `"Open"` / `"Retractable"` / `"Dome"` / `None` (from
            `mlb_luck_score.data.download_game_metadata`).
        weather_condition: The game's reported weather condition text (e.g.
            `"Sunny"`, `"Roof Closed"`) from MLB schedule weather.

    Returns:
        One of the `ROOF_STATUS_*` constants. Never guesses roof status from
        precipitation/condition text for a non-retractable venue -- only
        `"Retractable"` venues use the condition text at all, and only to
        distinguish open vs. closed (an explicit `"Roof Closed"` string),
        never inferred from e.g. `"Rain"` alone.
    """
    if roof_type is None or (isinstance(roof_type, float)):
        return ROOF_STATUS_UNKNOWN
    if roof_type == "Open":
        return ROOF_STATUS_OUTDOOR_OPEN_AIR
    if roof_type == "Dome":
        return ROOF_STATUS_FIXED_INDOOR
    if roof_type == "Retractable":
        if weather_condition is None or (isinstance(weather_condition, float)):
            return ROOF_STATUS_UNKNOWN
        if str(weather_condition).strip().lower() == "roof closed":
            return ROOF_STATUS_RETRACTABLE_CLOSED
        return ROOF_STATUS_RETRACTABLE_OPEN
    return ROOF_STATUS_UNKNOWN


@dataclass(frozen=True)
class ObservationMatch:
    observation_time_utc: pd.Timestamp
    offset_minutes: float
    match_quality: str
    tmpf: float | None
    dwpf: float | None
    relh: float | None
    drct: float | None
    sknt: float | None
    alti: float | None
    mslp: float | None


def _match_quality(offset_minutes: float) -> str:
    if offset_minutes <= MAX_OFFSET_MINUTES_GOOD:
        return "good"
    return "fair"


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _observation_match_from_row(
    row: pd.Series, offset_minutes: float, match_quality: str
) -> ObservationMatch:
    observation_time = row["valid"]
    assert isinstance(observation_time, pd.Timestamp)  # noqa: S101 -- guaranteed by caller's dtype
    return ObservationMatch(
        observation_time_utc=observation_time,
        offset_minutes=offset_minutes,
        match_quality=match_quality,
        tmpf=_float_or_none(row.get("tmpf")),
        dwpf=_float_or_none(row.get("dwpf")),
        relh=_float_or_none(row.get("relh")),
        drct=_float_or_none(row.get("drct")),
        sknt=_float_or_none(row.get("sknt")),
        alti=_float_or_none(row.get("alti")),
        mslp=_float_or_none(row.get("mslp")),
    )


def find_nearest_observation(
    station_obs: pd.DataFrame, scheduled_start_utc: pd.Timestamp
) -> ObservationMatch | None:
    """Find the nearest usable ASOS observation for one game's start time.

    Args:
        station_obs: One station's observations (must have a `valid` column
            of UTC timestamps), any order.
        scheduled_start_utc: The game's scheduled UTC start time.

    Returns:
        An `ObservationMatch`, or `None` if no observation exists within
        `MAX_OFFSET_MINUTES_FAIR` before start (or `MAX_AFTER_FALLBACK_MINUTES`
        after, used only when no prior observation qualifies).
    """
    if station_obs.empty:
        return None

    before = station_obs[station_obs["valid"] <= scheduled_start_utc]
    if not before.empty:
        row = cast("pd.Series", before.loc[before["valid"].idxmax()])
        observation_time = cast("pd.Timestamp", row["valid"])
        offset_minutes = pd.Timedelta(scheduled_start_utc - observation_time).total_seconds() / 60.0
        if offset_minutes <= MAX_OFFSET_MINUTES_FAIR:
            return _observation_match_from_row(row, offset_minutes, _match_quality(offset_minutes))

    after = station_obs[station_obs["valid"] > scheduled_start_utc]
    if not after.empty:
        row = cast("pd.Series", after.loc[after["valid"].idxmin()])
        observation_time = cast("pd.Timestamp", row["valid"])
        offset_minutes = pd.Timedelta(observation_time - scheduled_start_utc).total_seconds() / 60.0
        if offset_minutes <= MAX_AFTER_FALLBACK_MINUTES:
            # Negative: observation is AFTER game start.
            return _observation_match_from_row(row, -offset_minutes, _match_quality(offset_minutes))
    return None


def build_game_weather_row(
    game: dict[str, Any],
    roof_type: str | None,
    venue_env: VenueEnvironment | None,
    station_obs: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build one row of the game-weather table for a single game."""
    game_pk = game.get("game_pk")
    game_date = game.get("game_date")
    venue_id = game.get("venue_id")
    scheduled_start_raw = game.get("scheduled_start_time_utc")
    weather_condition = game.get("weather_condition")
    weather_temp_f = game.get("weather_temp_f")
    weather_wind_raw = game.get("weather_wind_raw")

    row: dict[str, Any] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "scheduled_start_time_utc": scheduled_start_raw,
        "local_start_time": None,
        "venue_id": venue_id,
        "venue_timezone": None,
        "weather_station_id": None,
        "observation_time_utc": None,
        "weather_time_offset_minutes": None,
        "station_distance_km": None,
        "temperature_c": None,
        "relative_humidity_pct": None,
        "pressure_hpa": None,
        "wind_speed_mps": None,
        "wind_direction_from_degrees": None,
        "wind_movement_bearing_degrees": None,
        "precipitation": None,
        "external_weather_available": False,
        "roof_status": classify_roof_status(roof_type, weather_condition),
        "roof_type": roof_type,
        "weather_condition_raw": weather_condition,
        "effective_temperature_c": None,
        "effective_relative_humidity_pct": None,
        "effective_pressure_hpa": None,
        "effective_wind_speed_mps": None,
        "effective_wind_movement_bearing_degrees": None,
        "air_density_kg_m3": None,
        "weather_status": None,
        "weather_match_quality": None,
        "source_name": "MLB Stats API schedule weather + Iowa Environmental Mesonet ASOS archive",
        "source_reference": (
            "https://statsapi.mlb.com/api/v1/schedule?hydrate=weather ; "
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        ),
        "retrieved_at": None,
    }

    if venue_env is None:
        row["weather_status"] = WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE
        return row

    row["venue_timezone"] = venue_env.timezone
    row["weather_station_id"] = venue_env.weather_station_id
    row["station_distance_km"] = venue_env.station_distance_km()

    if scheduled_start_raw is None or (isinstance(scheduled_start_raw, float)):
        row["weather_status"] = WEATHER_STATUS_MISSING_START_TIME
        return row

    scheduled_start_utc = pd.Timestamp(scheduled_start_raw)
    if scheduled_start_utc.tzinfo is None:
        scheduled_start_utc = scheduled_start_utc.tz_localize("UTC")
    else:
        scheduled_start_utc = scheduled_start_utc.tz_convert("UTC")
    local_start = scheduled_start_utc.tz_convert(ZoneInfo(venue_env.timezone))
    row["local_start_time"] = local_start.isoformat()

    wind_obs = parse_wind_text(weather_wind_raw)
    if wind_obs is not None:
        row["wind_speed_mps"] = wind_obs.speed_mps
        row["wind_movement_bearing_degrees"] = wind_obs.movement_bearing_degrees
        row["external_weather_available"] = True
    if weather_temp_f is not None and not (
        isinstance(weather_temp_f, float) and pd.isna(weather_temp_f)
    ):
        row["temperature_c"] = fahrenheit_to_celsius(float(weather_temp_f))
        row["external_weather_available"] = True

    match = None
    if station_obs is not None and not station_obs.empty:
        match = find_nearest_observation(station_obs, scheduled_start_utc)

    if match is not None:
        row["observation_time_utc"] = match.observation_time_utc.isoformat()
        row["weather_time_offset_minutes"] = match.offset_minutes
        row["weather_match_quality"] = match.match_quality
        row["relative_humidity_pct"] = match.relh
        row["wind_direction_from_degrees"] = match.drct
        # Prefer the MLB-reported temperature (already matched exactly to the
        # game); fall back to the station observation's temperature if MLB's
        # own field was missing.
        if row["temperature_c"] is None and match.tmpf is not None and not pd.isna(match.tmpf):
            row["temperature_c"] = fahrenheit_to_celsius(float(match.tmpf))
        if match.mslp is not None and not pd.isna(match.mslp):
            row["pressure_hpa"] = float(match.mslp)
        row["external_weather_available"] = True

    roof_status = row["roof_status"]

    if roof_status in _EFFECTIVE_CONDITIONS_UNAVAILABLE_STATUSES:
        row["weather_status"] = (
            WEATHER_STATUS_EFFECTIVE_UNAVAILABLE_INDOOR
            if row["external_weather_available"]
            else WEATHER_STATUS_NO_OBSERVATION
        )
        if roof_status in _WIND_SUPPRESSED_STATUSES:
            row["effective_wind_speed_mps"] = 0.0
            row["effective_wind_movement_bearing_degrees"] = None
        return row

    # outdoor_open_air or retractable_roof_open: outdoor conditions ARE the
    # effective playing conditions.
    if not row["external_weather_available"] or match is None:
        row["weather_status"] = WEATHER_STATUS_NO_OBSERVATION
        return row

    row["effective_temperature_c"] = row["temperature_c"]
    row["effective_relative_humidity_pct"] = row["relative_humidity_pct"]
    row["effective_wind_speed_mps"] = row["wind_speed_mps"]
    row["effective_wind_movement_bearing_degrees"] = row["wind_movement_bearing_degrees"]

    if row["pressure_hpa"] is not None and row["temperature_c"] is not None:
        row["effective_pressure_hpa"] = pressure_at_elevation_hpa(
            row["pressure_hpa"], venue_env.elevation_m, row["temperature_c"]
        )

    if (
        row["effective_temperature_c"] is not None
        and row["effective_pressure_hpa"] is not None
        and row["effective_relative_humidity_pct"] is not None
    ):
        row["air_density_kg_m3"] = moist_air_density_kg_m3(
            row["effective_temperature_c"],
            row["effective_pressure_hpa"],
            row["effective_relative_humidity_pct"],
        )

    row["weather_status"] = WEATHER_STATUS_OK
    return row


def build_game_weather_table(
    schedule_weather_df: pd.DataFrame,
    station_observations: dict[str, pd.DataFrame],
    roof_type_by_game: dict[int, str | None],
) -> pd.DataFrame:
    """Build the full game_pk-keyed weather table.

    Args:
        schedule_weather_df: Combined MLB schedule-weather rows across all
            requested seasons (see `mlb_luck_score.data.
            download_historical_weather`). Must have one row per `game_pk`
            -- duplicate `game_pk`s are rejected (see
            `GameWeatherError`).
        station_observations: `{station_id: observations_df}`, each
            covering all requested seasons for that station.
        roof_type_by_game: `{game_pk: roof_type}` from the existing game
            -metadata cache.

    Returns:
        One row per input game, in the schema documented in the module
        docstring. Every input row is preserved.
    """
    dup = schedule_weather_df["game_pk"][schedule_weather_df["game_pk"].duplicated()]
    if not dup.empty:
        raise GameWeatherError(
            f"schedule_weather_df has {dup.nunique()} duplicate game_pk value(s): "
            f"{sorted(dup.unique().tolist())[:10]}"
        )

    rows = []
    for raw_game in schedule_weather_df.to_dict(orient="records"):
        game = cast("dict[str, Any]", raw_game)
        venue_id = game.get("venue_id")
        venue_env = get_venue_environment(venue_id) if venue_id is not None else None
        station_obs = (
            station_observations.get(venue_env.weather_station_id)
            if venue_env is not None
            else None
        )
        game_pk = game.get("game_pk")
        roof_type = roof_type_by_game.get(int(game_pk)) if game_pk is not None else None
        rows.append(build_game_weather_row(game, roof_type, venue_env, station_obs))

    result = pd.DataFrame(rows)
    result["retrieved_at"] = datetime.now(UTC).isoformat()
    return result


class GameWeatherError(ValueError):
    """Raised for unrecoverable game-weather build problems."""


def load_schedule_weather(raw_dir: Path, seasons: list[int]) -> pd.DataFrame:
    missing = [s for s in seasons if not schedule_weather_path(raw_dir, s).exists()]
    if missing:
        raise GameWeatherError(
            f"Missing schedule-weather cache file(s) for season(s) {missing}. Run "
            "`make download-weather-data` first."
        )
    frames = [pd.read_parquet(schedule_weather_path(raw_dir, s)) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_station_observations(raw_dir: Path, seasons: list[int]) -> dict[str, pd.DataFrame]:
    from mlb_luck_score.data.venue_environment import VENUE_ENVIRONMENTS

    station_ids = sorted({v.weather_station_id for v in VENUE_ENVIRONMENTS})
    result: dict[str, pd.DataFrame] = {}
    for station in station_ids:
        frames = []
        for season in seasons:
            path = station_weather_path(raw_dir, station, season)
            if path.exists():
                frames.append(pd.read_parquet(path))
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["valid"] = pd.to_datetime(combined["valid"], utc=True)
            result[station] = combined.sort_values("valid").reset_index(drop=True)
        else:
            logger.warning(
                "No station-observation cache files found for station=%s seasons=%s",
                station,
                seasons,
            )
    return result


def load_roof_type_by_game(raw_dir: Path, seasons: list[int]) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for season in seasons:
        path = game_metadata_path(raw_dir, season)
        if not path.exists():
            raise GameWeatherError(
                f"Missing game-metadata cache file for season {season} (expected {path}). "
                "Run `make download-game-metadata` first."
            )
        df = pd.read_parquet(path)
        for game_pk, roof_type in zip(df["game_pk"], df["roof_type"], strict=True):
            if pd.notna(game_pk):
                result[int(game_pk)] = roof_type if pd.notna(roof_type) else None
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR / "game_weather.parquet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="Required to use the protected 2025 season. Never pass this for routine use.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    assert_seasons_allowed(tuple(args.seasons), allow_final_evaluation=args.allow_final_evaluation)

    if args.output.exists() and not args.overwrite:
        logger.error("Output file %s already exists. Pass --overwrite to replace it.", args.output)
        return 2

    try:
        schedule_weather_df = load_schedule_weather(args.raw_dir, args.seasons)
        station_observations = load_station_observations(args.raw_dir, args.seasons)
        roof_type_by_game = load_roof_type_by_game(args.raw_dir, args.seasons)
        table = build_game_weather_table(
            schedule_weather_df, station_observations, roof_type_by_game
        )
    except GameWeatherError as exc:
        logger.error(str(exc))
        return 2

    logger.info("Built game-weather table: %d games", len(table))
    logger.info(
        "Counts by weather_status: %s", table["weather_status"].value_counts(dropna=False).to_dict()
    )
    logger.info(
        "Counts by roof_status: %s", table["roof_status"].value_counts(dropna=False).to_dict()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(table), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
