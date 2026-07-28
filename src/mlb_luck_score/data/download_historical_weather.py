"""Contact Luck v0.5: download historical weather data (2021-2024 only).

Usage (requires internet access):

    python -m mlb_luck_score.data.download_historical_weather \\
        --seasons 2021 2022 2023 2024 --output-dir data/raw

Downloads TWO independent, complementary raw sources, cached one file per
season (schedule weather) or per station-season (station observations) so a
partial failure never loses already-downloaded data:

1. **MLB schedule weather** (`schedule_weather_<season>.parquet`): per-game
   `temp`/`wind`/`condition` text, hydrated directly from the public MLB
   Stats API `/schedule` endpoint (`hydrate=weather`) -- the SAME endpoint
   and chunked-date-range pattern `mlb_luck_score.data.
   download_game_metadata` already uses for venue metadata, just with one
   extra query parameter. This is the primary source for temperature, wind
   speed/direction (already expressed relative to the park, e.g. `"8 mph,
   Out To CF"`), and roof/sky condition text (e.g. `"Roof Closed"`) -- see
   `mlb_luck_score.data.build_game_weather` for how these are parsed.

2. **Historical station observations** (`asos_weather_<station>_<season>.
   parquet`): routine hourly METAR/ASOS observations (temperature, dewpoint,
   relative humidity, wind, altimeter setting, sea-level pressure) for each
   venue's assigned station (see `mlb_luck_score.data.venue_environment`),
   fetched from the public Iowa Environmental Mesonet historical archive
   (`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`, no API key
   required -- verified working directly against this endpoint before this
   module was written). This is the ONLY source of humidity and pressure in
   this repository -- MLB's own weather field does not include them.
   `report_type=3` restricts to routine (non-special) hourly reports so
   every row has a complete observation rather than the 5-minute wind-only
   "special" reports this archive also contains.

Both are resumable (an existing output file is skipped unless
`--overwrite`), retry transient failures a bounded number of times, and
never touch 2025 (`assert_seasons_allowed`; `mlb_luck_score.config.
MLB_REGULAR_SEASON_DATE_RANGES` also has no entry for 2025, as defense in
depth). Raw cache files are never modified in place by any other module.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    IOWA_MESONET_ASOS_BASE_URL,
    MLB_REGULAR_SEASON_DATE_RANGES,
    MLB_STATS_API_BASE_URL,
    RAW_DATA_DIR,
    ProtectedSeasonError,
    assert_seasons_allowed,
    schedule_weather_path,
    station_weather_path,
)
from mlb_luck_score.data.download_game_metadata import _get_json_with_retries
from mlb_luck_score.data.download_statcast import DownloadValidationError, build_date_chunks
from mlb_luck_score.data.venue_environment import VENUE_ENVIRONMENTS

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DAYS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
#: Polite minimum spacing between requests to the ASOS archive -- observed
#: empirically that rapid-fire requests with no spacing intermittently
#: return empty results (treated as a transient failure and retried, but
#: spacing requests avoids triggering it in the first place).
ASOS_REQUEST_SPACING_SECONDS = 1.2
ASOS_REQUEST_TIMEOUT_SECONDS = 30

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_EXISTING = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_PLANNED = "planned"

#: Columns requested from the ASOS archive. See module docstring for units:
#: tmpf/dwpf in Fahrenheit, relh in percent, drct in compass degrees (FROM),
#: sknt in knots, alti in inches of mercury (station altimeter setting),
#: mslp in hectopascals (sea-level-reduced pressure).
ASOS_DATA_FIELDS: tuple[str, ...] = ("tmpf", "dwpf", "relh", "drct", "sknt", "alti", "mslp")

SCHEDULE_WEATHER_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "scheduled_start_time_utc",
    "venue_id",
    "weather_condition",
    "weather_temp_f",
    "weather_wind_raw",
)


class HistoricalWeatherError(ValueError):
    """Raised for unrecoverable historical-weather download problems."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of attempting to download one cache file."""

    key: str
    status: str
    rows: int
    path: Path
    detail: str = ""


# ---------------------------------------------------------------------------
# 1. MLB schedule weather
# ---------------------------------------------------------------------------


def _fetch_schedule_weather_chunk(
    start: date, end: date, **retry_kwargs: Any
) -> list[dict[str, Any]]:
    data = _get_json_with_retries(
        f"{MLB_STATS_API_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "gameType": "R",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "hydrate": "weather",
        },
        **retry_kwargs,
    )
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        games.extend(day.get("games", []))
    return games


def _parse_schedule_weather_game(game: dict[str, Any]) -> dict[str, Any]:
    venue = game.get("venue") or {}
    weather = game.get("weather") or {}
    return {
        "game_pk": game.get("gamePk"),
        "game_date": game.get("officialDate"),
        # Full scheduled UTC start time (e.g. "2024-07-01T19:07:00Z") -- distinct from
        # `officialDate` (a date only) -- needed to match a weather observation by LOCAL
        # game time (see mlb_luck_score.data.build_game_weather).
        "scheduled_start_time_utc": game.get("gameDate"),
        "venue_id": venue.get("id"),
        "weather_condition": weather.get("condition"),
        "weather_temp_f": weather.get("temp"),
        "weather_wind_raw": weather.get("wind"),
    }


def build_season_schedule_weather(
    season: int,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Download and assemble one season's MLB schedule-weather table."""
    if season not in MLB_REGULAR_SEASON_DATE_RANGES:
        raise DownloadValidationError(
            f"No regular-season date range configured for {season} in "
            "mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES."
        )
    start_str, end_str = MLB_REGULAR_SEASON_DATE_RANGES[season]
    start, end = date.fromisoformat(start_str), date.fromisoformat(end_str)
    chunks = build_date_chunks(start, end, chunk_days)

    raw_games: list[dict[str, Any]] = []
    for chunk in chunks:
        logger.info(
            "[schedule-weather season %d] fetching %s to %s", season, chunk.start, chunk.end
        )
        raw_games.extend(
            _fetch_schedule_weather_chunk(
                chunk.start, chunk.end, max_retries=max_retries, backoff_seconds=backoff_seconds
            )
        )

    rows = [_parse_schedule_weather_game(g) for g in raw_games]
    games_df = pd.DataFrame(rows, columns=list(SCHEDULE_WEATHER_COLUMNS))
    before = len(games_df)
    games_df = games_df.drop_duplicates(subset=["game_pk"], keep="first")
    if before != len(games_df):
        logger.info(
            "[schedule-weather season %d] dropped %d duplicate game_pk row(s)",
            season,
            before - len(games_df),
        )
    return games_df.sort_values("game_pk").reset_index(drop=True)


def download_schedule_weather(
    seasons: list[int],
    output_dir: Path,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_final_evaluation: bool = False,
) -> list[DownloadResult]:
    """Download one MLB schedule-weather Parquet file per requested season."""
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)

    results: list[DownloadResult] = []
    for season in seasons:
        path = schedule_weather_path(output_dir, season)
        key = f"schedule_weather season={season}"

        if path.exists() and not overwrite:
            logger.info(
                "[%s] already exists at %s; skipping (pass --overwrite to redo).", key, path
            )
            results.append(DownloadResult(key, STATUS_SKIPPED_EXISTING, -1, path))
            continue
        if dry_run:
            logger.info("[DRY RUN][%s] would fetch schedule weather into %s.", key, path)
            results.append(DownloadResult(key, STATUS_PLANNED, 0, path))
            continue
        try:
            season_df = build_season_schedule_weather(
                season, chunk_days=chunk_days, max_retries=max_retries
            )
        except (DownloadValidationError, RuntimeError) as exc:
            logger.error("[%s] failed: %s", key, exc)
            results.append(DownloadResult(key, STATUS_FAILED, 0, path, detail=str(exc)))
            continue
        if season_df.empty:
            logger.error("[%s] no games found; refusing to write an empty file.", key)
            results.append(DownloadResult(key, STATUS_FAILED, 0, path, detail="empty result"))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(path, index=False)
        logger.info("[%s] saved %d game(s) to %s", key, len(season_df), path)
        results.append(DownloadResult(key, STATUS_DOWNLOADED, len(season_df), path))
    return results


# ---------------------------------------------------------------------------
# 2. Historical station (ASOS/METAR) observations
# ---------------------------------------------------------------------------


def _fetch_asos_csv(
    station: str,
    start: date,
    end: date,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> str:
    """GET one station's routine hourly ASOS observations for [start, end], retrying transient failures.

    An empty (header-only) response is treated as a transient failure and
    retried -- observed empirically that this archive intermittently
    returns an empty body under rapid request rates even for a station/date
    range known to have data (see `ASOS_REQUEST_SPACING_SECONDS`).
    """
    params: dict[str, str | int] = {
        "station": station,
        "data": ",".join(ASOS_DATA_FIELDS),
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "3",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                IOWA_MESONET_ASOS_BASE_URL, params=params, timeout=ASOS_REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            text = response.text
            lines = text.strip().splitlines()
            if len(lines) > 1:
                return text
            last_error = HistoricalWeatherError(
                f"Empty response for station={station} {start}..{end} (header-only, "
                f"possible transient rate-limit)"
            )
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            last_error = exc
        logger.warning(
            "ASOS request for station=%s %s..%s failed on attempt %d/%d: %s",
            station,
            start,
            end,
            attempt,
            max_retries,
            last_error,
        )
        if attempt < max_retries:
            time.sleep(backoff_seconds * attempt)
    raise HistoricalWeatherError(
        f"Failed to fetch ASOS data for station={station} {start}..{end} after "
        f"{max_retries} attempts"
    ) from last_error


def parse_asos_csv(text: str) -> pd.DataFrame:
    """Parse the ASOS archive's `onlycomma` CSV text into a typed DataFrame."""
    from io import StringIO

    df = pd.read_csv(StringIO(text), na_values=["M"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df


def build_station_season_weather(
    station: str,
    season: int,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Download and parse one station's routine ASOS observations for one season."""
    if season not in MLB_REGULAR_SEASON_DATE_RANGES:
        raise DownloadValidationError(
            f"No regular-season date range configured for {season} in "
            "mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES."
        )
    start_str, end_str = MLB_REGULAR_SEASON_DATE_RANGES[season]
    start, end = date.fromisoformat(start_str), date.fromisoformat(end_str)
    text = _fetch_asos_csv(
        station, start, end, max_retries=max_retries, backoff_seconds=backoff_seconds
    )
    return parse_asos_csv(text)


def download_station_weather(
    seasons: list[int],
    output_dir: Path,
    *,
    stations: list[str] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_final_evaluation: bool = False,
) -> list[DownloadResult]:
    """Download one ASOS observation Parquet file per requested station-season.

    Args:
        stations: Station ids to download. Defaults to every station in
            `mlb_luck_score.data.venue_environment.VENUE_ENVIRONMENTS`
            (deduplicated -- some venues share a station, e.g. the two New
            York ballparks both use LaGuardia).
    """
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)

    if stations is None:
        seen: list[str] = []
        for v in VENUE_ENVIRONMENTS:
            if v.weather_station_id not in seen:
                seen.append(v.weather_station_id)
        stations = seen

    results: list[DownloadResult] = []
    for station in stations:
        for season in seasons:
            path = station_weather_path(output_dir, station, season)
            key = f"asos station={station} season={season}"

            if path.exists() and not overwrite:
                logger.info(
                    "[%s] already exists at %s; skipping (pass --overwrite to redo).", key, path
                )
                results.append(DownloadResult(key, STATUS_SKIPPED_EXISTING, -1, path))
                continue
            if dry_run:
                logger.info("[DRY RUN][%s] would fetch ASOS observations into %s.", key, path)
                results.append(DownloadResult(key, STATUS_PLANNED, 0, path))
                continue
            try:
                station_df = build_station_season_weather(station, season, max_retries=max_retries)
            except (DownloadValidationError, HistoricalWeatherError) as exc:
                logger.error("[%s] failed: %s", key, exc)
                results.append(DownloadResult(key, STATUS_FAILED, 0, path, detail=str(exc)))
                time.sleep(ASOS_REQUEST_SPACING_SECONDS)
                continue
            if station_df.empty:
                logger.error("[%s] no observations found; refusing to write an empty file.", key)
                results.append(DownloadResult(key, STATUS_FAILED, 0, path, detail="empty result"))
                time.sleep(ASOS_REQUEST_SPACING_SECONDS)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            station_df.to_parquet(path, index=False)
            logger.info("[%s] saved %d observation(s) to %s", key, len(station_df), path)
            results.append(DownloadResult(key, STATUS_DOWNLOADED, len(station_df), path))
            time.sleep(ASOS_REQUEST_SPACING_SECONDS)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("schedule", "stations"),
        default=["schedule", "stations"],
        help="Which source(s) to download. Defaults to both.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="Required to download the protected 2025 season. Never pass this for routine use.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dry_run:
        logger.info(
            "This command requires internet access to reach the MLB Stats API and/or the "
            "Iowa Environmental Mesonet ASOS archive."
        )

    all_results: list[DownloadResult] = []
    try:
        if "schedule" in args.sources:
            all_results += download_schedule_weather(
                args.seasons,
                args.output_dir,
                chunk_days=args.chunk_days,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                allow_final_evaluation=args.allow_final_evaluation,
            )
        if "stations" in args.sources:
            all_results += download_station_weather(
                args.seasons,
                args.output_dir,
                max_retries=args.max_retries,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                allow_final_evaluation=args.allow_final_evaluation,
            )
    except ProtectedSeasonError as exc:
        logger.error(str(exc))
        return 2

    logger.info("=== Summary ===")
    by_status: dict[str, int] = {}
    for result in all_results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
        logger.info(
            "%s: %s (rows=%d, path=%s)%s",
            result.key,
            result.status,
            result.rows,
            result.path,
            f" -- {result.detail}" if result.detail else "",
        )
    logger.info("Counts by status: %s", by_status)

    if any(r.status == STATUS_FAILED for r in all_results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
