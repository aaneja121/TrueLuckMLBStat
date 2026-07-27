"""Download per-game venue metadata from the public MLB Stats API.

Usage (requires internet access):

    python -m mlb_luck_score.data.download_game_metadata \\
        --seasons 2021 2022 2023 2024 --output-dir data/raw

Statcast itself does not include a usable venue field (see CLAUDE.md) --
this downloads game-level metadata keyed by `game_pk` from the public MLB
Stats API (https://statsapi.mlb.com/api/v1, no API key required):

    game_pk, game_date, venue_id, venue_name, home_team, away_team,
    roof_type, surface_type, is_neutral_site

One raw Parquet file PER SEASON (`game_metadata_<season>.parquet`), fetched
in small date chunks over each season's regular-season window (see
`mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES`) via the `/schedule`
endpoint (`gameType=R` restricts to regular season), then enriched with
`roofType`/`turfType` from the `/venues/{id}` endpoint (hydrated with
`fieldInfo`) for every unique venue encountered.

`is_neutral_site` is a DERIVED heuristic, not a raw API field: for each
(season, home team), the most common venue across that team's home games
that season is treated as its "true" home park; any game at a different
venue is flagged neutral. A small number of real games (e.g. the MLB Field
of Dreams games) have no venue at all in the API response -- these rows are
kept with all venue fields null rather than dropped or guessed.

Resumable: if a season's output file already exists, that season is
SKIPPED unless `--overwrite` is passed. A failure downloading one season
does not abort the others.

2025 is a protected final-test season: this command refuses to download it
unless `--allow-final-evaluation` is explicitly passed, and additionally
has no configured date range for 2025 in
`mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` as defense in depth.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    MLB_REGULAR_SEASON_DATE_RANGES,
    MLB_STATS_API_BASE_URL,
    RAW_DATA_DIR,
    ProtectedSeasonError,
    assert_seasons_allowed,
    game_metadata_path,
)
from mlb_luck_score.data.download_statcast import DownloadValidationError, build_date_chunks

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DAYS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_EXISTING = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_PLANNED = "planned"

GAME_METADATA_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "venue_id",
    "venue_name",
    "home_team",
    "away_team",
    "roof_type",
    "surface_type",
    "is_neutral_site",
)


@dataclass(frozen=True)
class SeasonMetadataResult:
    """Outcome of attempting to download one season's game-metadata file."""

    season: int
    status: str
    rows: int
    path: Path
    detail: str = ""


def _get_json_with_retries(
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """GET a JSON endpoint, retrying transient failures a bounded number of times."""
    import time

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            last_error = exc
            logger.warning(
                "Request to %s failed on attempt %d/%d: %s", url, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts") from last_error


def _fetch_schedule_chunk(start: date, end: date, **retry_kwargs: Any) -> list[dict[str, Any]]:
    """Fetch raw game entries for one date chunk via the /schedule endpoint."""
    data = _get_json_with_retries(
        f"{MLB_STATS_API_BASE_URL}/schedule",
        params={
            "sportId": 1,
            "gameType": "R",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        },
        **retry_kwargs,
    )
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        games.extend(day.get("games", []))
    return games


def _parse_schedule_game(game: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw /schedule game entry into a metadata row (venue fields may be null)."""
    venue = game.get("venue") or {}
    home_team = (game.get("teams", {}).get("home", {}) or {}).get("team", {}) or {}
    away_team = (game.get("teams", {}).get("away", {}) or {}).get("team", {}) or {}
    return {
        "game_pk": game.get("gamePk"),
        "game_date": game.get("officialDate"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "home_team": home_team.get("name"),
        "away_team": away_team.get("name"),
    }


def fetch_venue_details(venue_id: int, **retry_kwargs: Any) -> dict[str, Any]:
    """Fetch roof/surface details for one venue via the /venues/{id} endpoint."""
    data = _get_json_with_retries(
        f"{MLB_STATS_API_BASE_URL}/venues/{venue_id}",
        params={"hydrate": "fieldInfo"},
        **retry_kwargs,
    )
    venues = data.get("venues") or [{}]
    field_info = venues[0].get("fieldInfo", {})
    return {
        "roof_type": field_info.get("roofType"),
        "surface_type": field_info.get("turfType"),
    }


def compute_neutral_site_flags(games_df: pd.DataFrame) -> pd.Series:
    """Flag games played somewhere other than the home team's usual season venue.

    For each (season, home_team), the most common venue_id across that
    team's home games that season is treated as its home park; games at a
    different venue are flagged neutral. Games with a missing venue_id are
    never flagged (there's nothing to compare).
    """
    working = games_df.assign(_season=pd.to_datetime(games_df["game_date"]).dt.year)

    primary_venue: dict[Any, Any] = {}
    for key, group in working.groupby(["_season", "home_team"], dropna=True):
        venue_counts = group["venue_id"].dropna()
        if not venue_counts.empty:
            primary_venue[key] = Counter(venue_counts).most_common(1)[0][0]

    flags = []
    for _, row in working.iterrows():
        venue_id = row["venue_id"]
        if pd.isna(venue_id):
            flags.append(False)
            continue
        expected_venue = primary_venue.get((row["_season"], row["home_team"]))
        flags.append(expected_venue is not None and venue_id != expected_venue)

    return pd.Series(flags, index=games_df.index)


def build_season_metadata(
    season: int,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    venue_cache: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Download and assemble one season's game-metadata table."""
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
        logger.info("[season %d] fetching schedule %s to %s", season, chunk.start, chunk.end)
        raw_games.extend(
            _fetch_schedule_chunk(
                chunk.start, chunk.end, max_retries=max_retries, backoff_seconds=backoff_seconds
            )
        )

    rows = [_parse_schedule_game(g) for g in raw_games]
    games_df = pd.DataFrame(
        rows, columns=["game_pk", "game_date", "venue_id", "venue_name", "home_team", "away_team"]
    )
    before_dedup = len(games_df)
    games_df = games_df.drop_duplicates(subset=["game_pk"], keep="first")
    n_duplicates = before_dedup - len(games_df)
    if n_duplicates:
        logger.info("[season %d] dropped %d duplicate game_pk row(s)", season, n_duplicates)

    venue_cache = {} if venue_cache is None else venue_cache
    unique_venue_ids = [int(v) for v in games_df["venue_id"].dropna().unique()]
    for venue_id in unique_venue_ids:
        if venue_id not in venue_cache:
            logger.info("Fetching venue details for venue_id=%d", venue_id)
            venue_cache[venue_id] = fetch_venue_details(
                venue_id, max_retries=max_retries, backoff_seconds=backoff_seconds
            )

    games_df["roof_type"] = games_df["venue_id"].map(
        lambda v: venue_cache.get(int(v), {}).get("roof_type") if pd.notna(v) else None
    )
    games_df["surface_type"] = games_df["venue_id"].map(
        lambda v: venue_cache.get(int(v), {}).get("surface_type") if pd.notna(v) else None
    )
    games_df["is_neutral_site"] = compute_neutral_site_flags(games_df)

    games_df = games_df.sort_values("game_pk").reset_index(drop=True)
    return games_df[list(GAME_METADATA_COLUMNS)]


def download_game_metadata(
    seasons: list[int],
    output_dir: Path,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_final_evaluation: bool = False,
) -> list[SeasonMetadataResult]:
    """Download one game-metadata Parquet file per requested season."""
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)

    venue_cache: dict[int, dict[str, Any]] = {}
    results: list[SeasonMetadataResult] = []
    for season in seasons:
        path = game_metadata_path(output_dir, season)

        if path.exists() and not overwrite:
            logger.info(
                "[season %d] output already exists at %s; skipping (pass --overwrite to redo).",
                season,
                path,
            )
            results.append(SeasonMetadataResult(season, STATUS_SKIPPED_EXISTING, -1, path))
            continue

        if dry_run:
            start_str, end_str = MLB_REGULAR_SEASON_DATE_RANGES.get(season, (None, None))
            logger.info(
                "[DRY RUN][season %d] would fetch schedule %s to %s into %s "
                "(no network access performed).",
                season,
                start_str,
                end_str,
                path,
            )
            results.append(SeasonMetadataResult(season, STATUS_PLANNED, 0, path))
            continue

        try:
            season_df = build_season_metadata(
                season, chunk_days=chunk_days, max_retries=max_retries, venue_cache=venue_cache
            )
        except (DownloadValidationError, RuntimeError) as exc:
            logger.error("[season %d] failed: %s", season, exc)
            results.append(SeasonMetadataResult(season, STATUS_FAILED, 0, path, detail=str(exc)))
            continue

        if season_df.empty:
            logger.error("[season %d] no games found; refusing to write an empty file.", season)
            results.append(
                SeasonMetadataResult(season, STATUS_FAILED, 0, path, detail="empty result")
            )
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(path, index=False)
        logger.info("[season %d] saved %d game(s) to %s", season, len(season_df), path)
        results.append(SeasonMetadataResult(season, STATUS_DOWNLOADED, len(season_df), path))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
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
        logger.info("This command requires internet access to reach the MLB Stats API.")

    try:
        results = download_game_metadata(
            args.seasons,
            args.output_dir,
            chunk_days=args.chunk_days,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            allow_final_evaluation=args.allow_final_evaluation,
        )
    except ProtectedSeasonError as exc:
        logger.error(str(exc))
        return 2

    logger.info("=== Summary ===")
    for result in results:
        logger.info(
            "season %d: %s (rows=%d, path=%s)%s",
            result.season,
            result.status,
            result.rows,
            result.path,
            f" -- {result.detail}" if result.detail else "",
        )

    if any(r.status == STATUS_FAILED for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
