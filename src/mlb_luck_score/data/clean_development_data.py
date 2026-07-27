"""Combine per-season development raw files into one cleaned processed table.

Usage:

    python -m mlb_luck_score.data.clean_development_data \\
        --seasons 2021 2022 2023 2024 \\
        --raw-dir data/raw \\
        --output data/processed/cleaned_development_data.parquet

Reads the per-season raw Parquet files produced by
`mlb_luck_score.data.download_development_data`
(`statcast_<season>_regular_season.parquet`), cleans each season
independently through the exact same pipeline as
`mlb_luck_score.data.clean_batted_balls` (eligibility, outcome mapping,
spray features -- nothing is duplicated here), logs row totals and
training-exclusion reasons SEPARATELY BY SEASON, then combines every
season into one deterministically-deduplicated, sorted table with a
`season` column.

Ambiguous events (`field_error`, `fielders_choice`) are preserved in the
combined output but remain excluded from training with their specific
reason, exactly as in the single-file pipeline -- no label is invented.

Fails clearly (before doing any work) if any requested season's raw file is
missing. Never includes the protected 2025 season, whether requested (see
`mlb_luck_score.config.assert_seasons_allowed`) or found unexpectedly inside
a raw file's `game_date` values.

This module never downloads anything and never touches the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ProtectedSeasonError,
    assert_seasons_allowed,
    development_raw_path,
)
from mlb_luck_score.data.clean_batted_balls import (
    CleaningError,
    clean_batted_balls,
    read_raw_table,
    save_dataframe,
)

logger = logging.getLogger(__name__)


def _check_all_raw_files_exist(seasons: Sequence[int], raw_dir: Path) -> None:
    """Fail clearly (listing every missing season at once) before any work."""
    missing = [
        (season, development_raw_path(raw_dir, season))
        for season in seasons
        if not development_raw_path(raw_dir, season).exists()
    ]
    if missing:
        details = "; ".join(f"season {season} (expected {path})" for season, path in missing)
        raise CleaningError(
            f"Missing raw data file(s) for {len(missing)} of {len(seasons)} requested "
            f"season(s): {details}. Run `make download-development-data` first, or check "
            "--raw-dir."
        )


def clean_one_season(season: int, raw_dir: Path) -> pd.DataFrame:
    """Clean a single season's raw file, logging season-specific counts."""
    path = development_raw_path(raw_dir, season)
    if not path.exists():
        raise CleaningError(f"Missing raw data file for season {season}: {path}")

    logger.info("=== [season %d] cleaning %s ===", season, path)
    raw = read_raw_table(path)
    cleaned = clean_batted_balls(raw)

    n_eligible = len(cleaned)
    n_training_eligible = int(cleaned["eligible_for_training"].sum()) if n_eligible else 0
    logger.info(
        "[season %d] eligible fair-batted-ball events: %d | training-eligible: %d",
        season,
        n_eligible,
        n_training_eligible,
    )
    if n_eligible:
        reason_counts = (
            cleaned.loc[~cleaned["eligible_for_training"], "training_exclusion_reason"]
            .value_counts(dropna=False)
            .to_dict()
        )
        logger.info("[season %d] training-exclusion reason counts: %s", season, reason_counts)

        unexpected_seasons = set(cleaned["season"].dropna().unique().tolist()) - {season}
        if unexpected_seasons:
            logger.warning(
                "[season %d] file %s contains row(s) tagged with unexpected season(s) %s "
                "(game_date outside the expected season range?)",
                season,
                path,
                sorted(unexpected_seasons),
            )

    return cleaned


def combine_cleaned_seasons(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-season cleaned frames with deterministic cross-file dedup.

    Deduplication uses `event_id` (built deterministically from
    `game_pk`/`at_bat_number`/`pitch_number`), so an event appearing in more
    than one season's raw file -- which should not happen since `game_pk`
    is globally unique, but is checked defensively -- is kept exactly once.
    """
    non_empty = [df for df in frames.values() if not df.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)
    before = len(combined)
    dedup_keys = ["event_id"] if "event_id" in combined.columns else None
    combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
    n_cross_file_duplicates = before - len(combined)
    if n_cross_file_duplicates:
        logger.warning(
            "Removed %d cross-file duplicate event(s) (same event_id appeared in "
            "multiple season files).",
            n_cross_file_duplicates,
        )

    sort_cols = [c for c in ("season", "event_id") if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def clean_development_data(
    seasons: Sequence[int],
    raw_dir: Path,
    *,
    allow_final_evaluation: bool = False,
) -> pd.DataFrame:
    """Combine per-season raw Statcast files into one cleaned development table."""
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)
    _check_all_raw_files_exist(seasons, raw_dir)

    frames = {season: clean_one_season(season, raw_dir) for season in seasons}
    combined = combine_cleaned_seasons(frames)

    if not combined.empty:
        observed_seasons = tuple(
            sorted(int(s) for s in combined["season"].dropna().unique().tolist())
        )
        # Defense in depth: refuse to proceed if the combined data ever contains
        # 2025, even if the caller only asked for allowed seasons -- this should
        # be unreachable in practice since each raw file is season-scoped.
        assert_seasons_allowed(observed_seasons, allow_final_evaluation=allow_final_evaluation)

    n_total = len(combined)
    n_training_eligible = int(combined["eligible_for_training"].sum()) if n_total else 0
    logger.info(
        "Combined development dataset: %d rows across %d requested season(s) "
        "(%d training-eligible).",
        n_total,
        len(seasons),
        n_training_eligible,
    )
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DATA_DIR / "cleaned_development_data.parquet"
    )
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="Required to include the protected 2025 season. Never pass this for routine use.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        combined = clean_development_data(
            args.seasons, args.raw_dir, allow_final_evaluation=args.allow_final_evaluation
        )
    except (CleaningError, ProtectedSeasonError) as exc:
        logger.error(str(exc))
        return 2

    if combined.empty:
        logger.error("Combined cleaned dataset is empty; refusing to write output.")
        return 1

    save_dataframe(combined, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
