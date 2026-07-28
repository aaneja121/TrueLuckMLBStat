"""Contact Luck v0.5: join the game-weather table onto every eligible batted-ball event.

Usage:

    python -m mlb_luck_score.data.join_weather_features \\
        --input data/processed/cleaned_development_data_with_geometry.parquet \\
        --game-weather data/processed/game_weather.parquet \\
        --output data/processed/cleaned_development_data_with_weather.parquet

Joins `mlb_luck_score.data.build_game_weather`'s game_pk-keyed table onto
every row by `game_pk`, then computes PER-PLAY following/head/crosswind
components (`mlb_luck_score.data.weather_physics.wind_relative_components`)
using each play's own `spray_angle_approx` -- the game-level effective wind
is a single park-relative vector; how much of it is a tailwind vs. a
crosswind depends on which direction THIS specific ball was hit.

Every input row is PRESERVED -- rows without usable weather get explicit
status columns (never dropped, never silently filled with a guessed
value). This module never touches the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_luck_score.config import PROCESSED_DATA_DIR
from mlb_luck_score.data.build_game_weather import (
    ROOF_STATUS_FIXED_INDOOR,
    ROOF_STATUS_RETRACTABLE_CLOSED,
    WEATHER_STATUS_OK,
    WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE,
)
from mlb_luck_score.data.weather_physics import (
    DEFAULT_REFERENCE_AIR_DENSITY_KG_M3,
    WindComponents,
    WindObservation,
    wind_relative_components,
    zero_wind_components,
)

logger = logging.getLogger(__name__)

_INDOOR_ROOF_STATUSES = frozenset({ROOF_STATUS_FIXED_INDOOR, ROOF_STATUS_RETRACTABLE_CLOSED})

#: Columns pulled from the game-weather table onto every play (game-level,
#: broadcast to all rows sharing a `game_pk`).
_GAME_WEATHER_JOIN_COLUMNS: tuple[str, ...] = (
    "weather_status",
    "weather_match_quality",
    "weather_time_offset_minutes",
    "station_distance_km",
    "roof_status",
    "effective_temperature_c",
    "effective_relative_humidity_pct",
    "effective_pressure_hpa",
    "effective_wind_speed_mps",
    "effective_wind_movement_bearing_degrees",
    "air_density_kg_m3",
)


class JoinWeatherFeaturesError(ValueError):
    """Raised for unrecoverable weather-feature join problems."""


def join_weather_features(df: pd.DataFrame, game_weather_df: pd.DataFrame) -> pd.DataFrame:
    """Join `game_weather_df` onto every row of `df` by `game_pk`.

    Args:
        df: Cleaned batted-ball events (must have `game_pk` and
            `spray_angle_approx` columns).
        game_weather_df: Output of `mlb_luck_score.data.build_game_weather.
            build_game_weather_table` (one row per `game_pk`).

    Returns:
        A copy of `df` with weather columns added. Every input row is
        preserved (see module docstring).
    """
    required = ("game_pk", "spray_angle_approx")
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise JoinWeatherFeaturesError(f"join_weather_features requires column(s) {missing_cols}")

    dup = game_weather_df["game_pk"][game_weather_df["game_pk"].duplicated()]
    if not dup.empty:
        raise JoinWeatherFeaturesError(
            f"game_weather_df has {dup.nunique()} duplicate game_pk value(s) -- refusing to "
            f"join: {sorted(dup.unique().tolist())[:10]}"
        )

    join_cols = ["game_pk", *_GAME_WEATHER_JOIN_COLUMNS]
    available_join_cols = [c for c in join_cols if c in game_weather_df.columns]
    out = df.merge(
        game_weather_df[available_join_cols], on="game_pk", how="left", validate="many_to_one"
    )

    out["has_weather_data"] = out["weather_status"].notna() & (
        out["weather_status"] != WEATHER_STATUS_VENUE_ENVIRONMENT_UNAVAILABLE
    )
    out["has_effective_weather"] = out["weather_status"] == WEATHER_STATUS_OK
    out["indoor_indicator"] = out["roof_status"].isin(_INDOOR_ROOF_STATUSES)

    out["temperature_c"] = out["effective_temperature_c"]
    out["humidity_pct"] = out["effective_relative_humidity_pct"]
    out["pressure_hpa"] = out["effective_pressure_hpa"]
    out["wind_speed_mps"] = out["effective_wind_speed_mps"]

    out["air_density_deviation_from_reference"] = (
        out["air_density_kg_m3"] - DEFAULT_REFERENCE_AIR_DENSITY_KG_M3
    )

    n = len(out)
    following = np.full(n, np.nan)
    headwind = np.full(n, np.nan)
    crosswind = np.full(n, np.nan)

    speeds = out["effective_wind_speed_mps"].to_numpy()
    bearings = out["effective_wind_movement_bearing_degrees"].to_numpy()
    angles = out["spray_angle_approx"].to_numpy()

    for i in range(n):
        speed = speeds[i]
        angle = angles[i]
        if pd.isna(speed) or pd.isna(angle):
            continue
        components: WindComponents | None
        if speed == 0.0:
            components = zero_wind_components()
        else:
            bearing = bearings[i]
            if pd.isna(bearing):
                continue
            wind = WindObservation(
                speed_mps=float(speed),
                movement_bearing_degrees=float(bearing),
                raw_text="",
                parse_status="ok",
            )
            components = wind_relative_components(wind, float(angle))
            if components is None:
                continue
        following[i] = components.following_wind_mps
        headwind[i] = components.headwind_mps
        crosswind[i] = components.crosswind_mps

    out["following_wind_mps"] = following
    out["headwind_mps"] = headwind
    out["crosswind_mps"] = crosswind

    out["weather_uncertain"] = out["has_weather_data"] & ~out["has_effective_weather"]

    return out


@dataclass(frozen=True)
class WeatherJoinReport:
    """Coverage report for a weather-feature join."""

    total_rows: int
    rows_with_weather_data: int
    rows_with_effective_weather: int
    coverage_rate: float
    effective_coverage_rate: float
    counts_by_status: dict[str, int]
    counts_by_roof_status: dict[str, int]
    counts_by_match_quality: dict[str, int]
    counts_by_season: dict[str, int]
    counts_by_venue: dict[str, int]


def build_weather_join_report(joined_df: pd.DataFrame) -> WeatherJoinReport:
    """Compute coverage/health statistics for a weather-feature join."""
    total_rows = len(joined_df)
    rows_with_weather = int(joined_df["has_weather_data"].sum())
    rows_with_effective = int(joined_df["has_effective_weather"].sum())

    def _counts(col: str) -> dict[str, int]:
        if col not in joined_df.columns:
            return {}
        return {str(k): int(v) for k, v in joined_df[col].value_counts(dropna=False).items()}

    by_season: dict[str, int] = {}
    if "season" in joined_df.columns:
        cross = joined_df.groupby("season", dropna=False)["has_effective_weather"].sum()
        totals = joined_df.groupby("season", dropna=False).size()
        by_season = {str(s): int(cross.get(s, 0)) for s in totals.index}

    by_venue: dict[str, int] = {}
    if "venue_name" in joined_df.columns:
        cross = joined_df.groupby("venue_name", dropna=False)["has_effective_weather"].sum()
        by_venue = {str(k): int(v) for k, v in cross.items()}

    return WeatherJoinReport(
        total_rows=total_rows,
        rows_with_weather_data=rows_with_weather,
        rows_with_effective_weather=rows_with_effective,
        coverage_rate=rows_with_weather / total_rows if total_rows else 0.0,
        effective_coverage_rate=rows_with_effective / total_rows if total_rows else 0.0,
        counts_by_status=_counts("weather_status"),
        counts_by_roof_status=_counts("roof_status"),
        counts_by_match_quality=_counts("weather_match_quality"),
        counts_by_season=by_season,
        counts_by_venue=by_venue,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned batted-ball dataset")
    parser.add_argument(
        "--game-weather",
        type=Path,
        default=PROCESSED_DATA_DIR / "game_weather.parquet",
        help="Output of build_game_weather",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_development_data_with_weather.parquet",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        logger.error("Output file %s already exists. Pass --overwrite to replace it.", args.output)
        return 2

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    game_weather_df = pd.read_parquet(args.game_weather)
    logger.info("Input rows: %d | game-weather rows: %d", len(df), len(game_weather_df))

    try:
        joined_df = join_weather_features(df, game_weather_df)
    except JoinWeatherFeaturesError as exc:
        logger.error(str(exc))
        return 2

    report = build_weather_join_report(joined_df)
    logger.info(
        "Rows with weather data: %d / %d (%.4f); with EFFECTIVE weather: %d / %d (%.4f)",
        report.rows_with_weather_data,
        report.total_rows,
        report.coverage_rate,
        report.rows_with_effective_weather,
        report.total_rows,
        report.effective_coverage_rate,
    )
    logger.info("Counts by weather_status: %s", report.counts_by_status)
    logger.info("Counts by roof_status: %s", report.counts_by_roof_status)
    logger.info("Counts by weather_match_quality: %s", report.counts_by_match_quality)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(args.output, index=False)
    logger.info("Wrote %d joined rows to %s", len(joined_df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
