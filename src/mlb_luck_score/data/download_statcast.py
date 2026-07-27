"""Download raw Statcast batted-ball data via pybaseball.

Usage (requires internet access -- pybaseball scrapes public Baseball
Savant CSV endpoints):

    python -m mlb_luck_score.data.download_statcast \\
        --start-date 2024-04-01 --end-date 2024-04-07 \\
        --output data/raw/statcast_2024_sample.parquet

Downloads happen in small date chunks (see --chunk-days) to stay polite to
the upstream service and to bound the size/time of any single request.
Transient network failures are retried a bounded number of times per chunk.

By default this refuses to touch the 2025 season (the protected final-test
year) or overwrite an existing output file -- see --allow-final-evaluation
and --overwrite.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from mlb_luck_score.config import (
    DEFAULT_SAMPLE_END_DATE,
    DEFAULT_SAMPLE_START_DATE,
    ProtectedSeasonError,
    assert_seasons_allowed,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_DAYS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

#: Natural-key columns used to deduplicate rows across overlapping chunk
#: boundaries. Falls back to full-row dedup if these aren't all present.
_DEDUP_KEY_COLUMNS = ("game_pk", "at_bat_number", "pitch_number")
_SORT_COLUMNS = ("game_date", "game_pk", "at_bat_number", "pitch_number")


class DownloadValidationError(ValueError):
    """Raised for invalid CLI arguments (bad dates, backwards range, etc.)."""


@dataclass(frozen=True)
class DateChunk:
    start: date
    end: date


def parse_date(value: str, *, label: str) -> date:
    """Parse a YYYY-MM-DD date string, raising a clear error if malformed."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise DownloadValidationError(
            f"Invalid {label} '{value}': expected YYYY-MM-DD format"
        ) from exc


def validate_date_range(start: date, end: date) -> None:
    """Reject an end date earlier than the start date."""
    if end < start:
        raise DownloadValidationError(
            f"--end-date ({end}) cannot be earlier than --start-date ({start})"
        )


def build_date_chunks(start: date, end: date, chunk_days: int) -> list[DateChunk]:
    """Split [start, end] into small, inclusive date chunks."""
    if chunk_days < 1:
        raise DownloadValidationError("--chunk-days must be a positive integer")

    chunks = []
    current = start
    step = timedelta(days=chunk_days - 1)
    one_day = timedelta(days=1)
    while current <= end:
        chunk_end = min(current + step, end)
        chunks.append(DateChunk(current, chunk_end))
        current = chunk_end + one_day
    return chunks


def _download_chunk_with_retries(
    chunk: DateChunk, *, max_retries: int, backoff_seconds: float
) -> pd.DataFrame:
    """Download one date chunk via pybaseball, retrying transient failures."""
    import pybaseball  # imported lazily so --dry-run needs no network stack

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading chunk %s to %s (attempt %d/%d)",
                chunk.start,
                chunk.end,
                attempt,
                max_retries,
            )
            frame = pybaseball.statcast(
                start_dt=chunk.start.isoformat(), end_dt=chunk.end.isoformat()
            )
            return frame
        except Exception as exc:  # noqa: BLE001 - genuinely want to retry any transient failure
            last_error = exc
            logger.warning(
                "Chunk %s to %s failed on attempt %d/%d: %s",
                chunk.start,
                chunk.end,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(
        f"Failed to download chunk {chunk.start} to {chunk.end} after {max_retries} attempts"
    ) from last_error


def download_statcast_range(
    start: date,
    end: date,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Download and combine Statcast data for a date range.

    Requires internet access -- this calls out to Baseball Savant via
    pybaseball. Downloads happen in `chunk_days`-sized pieces, are retried on
    transient failure, then combined, deduplicated, and sorted
    deterministically.
    """
    chunks = build_date_chunks(start, end, chunk_days)
    logger.info("Downloading %d chunk(s) covering %s to %s", len(chunks), start, end)

    frames = []
    for chunk in chunks:
        frame = _download_chunk_with_retries(
            chunk, max_retries=max_retries, backoff_seconds=backoff_seconds
        )
        logger.info("Chunk %s to %s returned %d rows", chunk.start, chunk.end, len(frame))
        if not frame.empty:
            frames.append(frame)

    if not frames:
        logger.warning("No rows returned for %s to %s", start, end)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined)

    dedup_keys = [c for c in _DEDUP_KEY_COLUMNS if c in combined.columns]
    if dedup_keys:
        combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")
    logger.info("Deduplicated %d -> %d rows", before_dedup, len(combined))

    sort_cols = [c for c in _SORT_COLUMNS if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    return combined


def save_dataframe(df: pd.DataFrame, output: Path) -> None:
    """Save to Parquet by default, or CSV if `output` ends in .csv."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
    logger.info("Saved %d rows to %s", len(df), output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw Statcast batted-ball data (requires internet access). "
            "Defaults to a small one-week 2024 sample window so repository "
            "bootstrap never triggers a full-season download."
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_SAMPLE_START_DATE, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_SAMPLE_END_DATE, help="YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/statcast_sample.parquet"),
        help="Output path (.parquet by default, .csv if the filename ends in .csv)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting an existing output file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the download plan without any network access",
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

    try:
        start = parse_date(args.start_date, label="--start-date")
        end = parse_date(args.end_date, label="--end-date")
        validate_date_range(start, end)
        assert_seasons_allowed(
            [start.year, end.year], allow_final_evaluation=args.allow_final_evaluation
        )
    except (DownloadValidationError, ProtectedSeasonError) as exc:
        logger.error(str(exc))
        return 2

    if args.dry_run:
        chunks = build_date_chunks(start, end, args.chunk_days)
        logger.info(
            "[DRY RUN] Would download %d chunk(s) from %s to %s into %s "
            "(no network access performed).",
            len(chunks),
            start,
            end,
            args.output,
        )
        for chunk in chunks:
            logger.info("[DRY RUN]   chunk: %s to %s", chunk.start, chunk.end)
        return 0

    if args.output.exists() and not args.overwrite:
        logger.error("Output %s already exists. Pass --overwrite to replace it.", args.output)
        return 2

    logger.info("This command requires internet access to reach Baseball Savant via pybaseball.")
    df = download_statcast_range(
        start,
        end,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries,
    )
    if df.empty:
        logger.error("No data downloaded; refusing to write an empty output file.")
        return 1

    save_dataframe(df, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
