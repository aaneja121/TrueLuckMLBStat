"""Download full regular-season Statcast data for the development seasons.

Usage (requires internet access -- see README.md for storage/runtime notes
before running this for real):

    python -m mlb_luck_score.data.download_development_data \\
        --seasons 2021 2022 2023 2024 --output-dir data/raw

Downloads ONE raw Parquet file PER SEASON (never one combined file), using
`mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` for each season's
date range, in the same small-chunk, bounded-retry, deduplicated,
deterministically-sorted style as `mlb_luck_score.data.download_statcast`
(this module reuses that downloader's chunking/retry/dedup logic directly
rather than reimplementing it).

Resumable: if a season's output file already exists, that season is
SKIPPED (not re-downloaded) unless `--overwrite` is passed. Re-running this
command after an interruption or a partial failure only downloads the
seasons that are still missing or that failed last time. A failure
downloading one season does NOT abort the others -- failures are collected
and reported at the end via a non-zero exit code, so one bad season never
blocks progress on the rest.

This command NEVER writes to the separate one-week bootstrap sample file
(`statcast_2024_sample.parquet` by default, produced by
`mlb_luck_score.data.download_statcast`) -- every file this command writes
uses the `statcast_<season>_regular_season.parquet` naming convention
(see `mlb_luck_score.config.development_raw_path`), which cannot collide
with the sample filename.

2025 is a protected final-test season: this command refuses to download it
unless `--allow-final-evaluation` is explicitly passed, and additionally
has no configured date range for 2025 in
`mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` as defense in depth.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    MLB_REGULAR_SEASON_DATE_RANGES,
    RAW_DATA_DIR,
    ProtectedSeasonError,
    assert_seasons_allowed,
    development_raw_path,
)
from mlb_luck_score.data.download_statcast import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_MAX_RETRIES,
    DownloadValidationError,
    build_date_chunks,
    download_statcast_range,
    save_dataframe,
)

logger = logging.getLogger(__name__)

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_EXISTING = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_PLANNED = "planned"


@dataclass(frozen=True)
class SeasonDownloadResult:
    """Outcome of attempting to download one season's raw file."""

    season: int
    status: str
    rows: int
    path: Path
    detail: str = ""


def season_date_range(season: int) -> tuple[date, date]:
    """Look up the documented regular-season date range for `season`.

    Raises:
        DownloadValidationError: If no date range is configured (this is
            the case for 2025 by design -- see module docstring).
    """
    if season not in MLB_REGULAR_SEASON_DATE_RANGES:
        raise DownloadValidationError(
            f"No authoritative regular-season date range configured for {season} in "
            "mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES. Add one there first."
        )
    start_str, end_str = MLB_REGULAR_SEASON_DATE_RANGES[season]
    return date.fromisoformat(start_str), date.fromisoformat(end_str)


def download_development_data(
    seasons: list[int],
    output_dir: Path,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_final_evaluation: bool = False,
) -> list[SeasonDownloadResult]:
    """Download one raw Parquet file per requested development season.

    Each season is attempted independently: a download failure for one
    season is logged and recorded, but does not prevent the remaining
    seasons from being attempted (see module docstring on resumability).
    """
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)

    results: list[SeasonDownloadResult] = []
    for season in seasons:
        path = development_raw_path(output_dir, season)

        if path.exists() and not overwrite:
            logger.info(
                "[season %d] output already exists at %s; skipping "
                "(pass --overwrite to redownload).",
                season,
                path,
            )
            results.append(SeasonDownloadResult(season, STATUS_SKIPPED_EXISTING, -1, path))
            continue

        try:
            start, end = season_date_range(season)
        except DownloadValidationError as exc:
            logger.error("[season %d] %s", season, exc)
            results.append(SeasonDownloadResult(season, STATUS_FAILED, 0, path, detail=str(exc)))
            continue

        chunks = build_date_chunks(start, end, chunk_days)

        if dry_run:
            logger.info(
                "[DRY RUN][season %d] would download %d chunk(s) from %s to %s into %s "
                "(no network access performed).",
                season,
                len(chunks),
                start,
                end,
                path,
            )
            for chunk in chunks:
                logger.info(
                    "[DRY RUN][season %d]   chunk: %s to %s", season, chunk.start, chunk.end
                )
            results.append(SeasonDownloadResult(season, STATUS_PLANNED, len(chunks), path))
            continue

        logger.info(
            "=== [season %d] downloading %s to %s (%d chunk(s)) ===",
            season,
            start,
            end,
            len(chunks),
        )
        try:
            df = download_statcast_range(start, end, chunk_days=chunk_days, max_retries=max_retries)
        except RuntimeError as exc:
            logger.error(
                "[season %d] download failed after retries: %s. "
                "Re-run this command later to retry -- already-downloaded seasons will "
                "be skipped automatically.",
                season,
                exc,
            )
            results.append(SeasonDownloadResult(season, STATUS_FAILED, 0, path, detail=str(exc)))
            continue

        if df.empty:
            logger.error("[season %d] no data downloaded; refusing to write an empty file.", season)
            results.append(
                SeasonDownloadResult(season, STATUS_FAILED, 0, path, detail="empty result")
            )
            continue

        save_dataframe(df, path)
        results.append(SeasonDownloadResult(season, STATUS_DOWNLOADED, len(df), path))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(DEVELOPMENT_SEASONS),
        help="Seasons to download (default: %(default)s)",
    )
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Redownload seasons whose output already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the per-season download plan; no network access",
    )
    parser.add_argument(
        "--chunk-size",
        "--chunk-days",
        dest="chunk_days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help="Number of days per download chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Bounded retry count per chunk on transient failure",
    )
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help=(
            "Required to download the protected 2025 final-test season. "
            "Never pass this for routine development, tuning, or iteration."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dry_run:
        logger.info(
            "This command requires internet access to reach Baseball Savant via pybaseball."
        )

    try:
        results = download_development_data(
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
