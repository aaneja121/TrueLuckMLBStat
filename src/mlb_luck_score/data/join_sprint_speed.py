"""Join the Baseball Savant Sprint Speed leaderboard onto per-play rows.

Usage:

    python -m mlb_luck_score.data.join_sprint_speed \\
        --cleaned-input data/processed/cleaned_development_data_with_geometry.parquet \\
        --raw-dir data/raw \\
        --output data/processed/cleaned_development_data_with_sprint_speed.parquet

Loads the per-season `sprint_speed_<season>.parquet` files produced by
`mlb_luck_score.data.download_sprint_speed` and left-joins `sprint_speed`/
`hp_to_1b` onto the cleaned batted-ball table by `(batter, season)` --
`player_id` (Savant leaderboard) and `batter` (Statcast per-play) are the SAME
MLBAM player-ID space. A batter/season with no leaderboard row (below
Savant's own `min_opp` qualification threshold, or simply not in the
leaderboard for that season) is KEPT, not dropped, with `has_sprint_speed`
explicitly `False` and `sprint_speed`/`hp_to_1b` null -- this is expected and
reported, not an error (see `build_sprint_speed_join_report`).

This module never downloads anything and never touches the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    RAW_DATA_DIR,
    sprint_speed_path,
)

logger = logging.getLogger(__name__)

SPRINT_SPEED_JOIN_COLUMNS: tuple[str, ...] = ("player_id", "season", "sprint_speed", "hp_to_1b")


class JoinSprintSpeedError(ValueError):
    """Raised for unrecoverable sprint-speed join problems."""


def load_sprint_speed(raw_dir: Path, seasons: Sequence[int]) -> pd.DataFrame:
    """Load and combine per-season Sprint Speed leaderboard files, tagging each with its season.

    Raises:
        JoinSprintSpeedError: If any requested season's file is missing (all
            missing seasons are listed at once).
    """
    missing = [
        (season, sprint_speed_path(raw_dir, season))
        for season in seasons
        if not sprint_speed_path(raw_dir, season).exists()
    ]
    if missing:
        details = "; ".join(f"season {season} (expected {path})" for season, path in missing)
        raise JoinSprintSpeedError(
            f"Missing sprint-speed leaderboard file(s) for {len(missing)} of {len(seasons)} "
            f"requested season(s): {details}. Run `make download-sprint-speed` first."
        )

    frames = []
    for season in seasons:
        season_df = pd.read_parquet(sprint_speed_path(raw_dir, season))
        season_df = season_df.assign(season=season)
        frames.append(season_df)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["player_id", "season"], keep="first")
    n_duplicates = before - len(combined)
    if n_duplicates:
        logger.info(
            "Dropped %d duplicate (player_id, season) row(s) across sprint-speed files.",
            n_duplicates,
        )
    return combined


def join_sprint_speed(cleaned_df: pd.DataFrame, sprint_speed_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join sprint speed/hp_to_1b onto `cleaned_df` by `(batter, season)`.

    Rows without a match are preserved with `has_sprint_speed=False` and null
    `sprint_speed`/`hp_to_1b` -- never dropped. `mlb_luck_score.data.
    clean_batted_balls` already carries an all-null `sprint_speed` OPTIONAL
    column (declared, but never populated by the per-pitch Statcast pull) --
    dropped here first so the real leaderboard values replace it cleanly
    instead of colliding into `sprint_speed_x`/`sprint_speed_y`.
    """
    base_df = cleaned_df.drop(columns=["sprint_speed", "hp_to_1b"], errors="ignore")
    join_cols = sprint_speed_df[list(SPRINT_SPEED_JOIN_COLUMNS)].rename(
        columns={"player_id": "batter"}
    )
    joined = base_df.merge(join_cols, on=["batter", "season"], how="left", validate="many_to_one")
    joined["has_sprint_speed"] = joined["sprint_speed"].notna()
    return joined


@dataclass(frozen=True)
class SprintSpeedJoinReport:
    """Coverage report for a sprint-speed join."""

    total_rows: int
    matched_rows: int
    unmatched_rows: int
    match_rate: float
    unique_batters: int
    unique_batters_matched: int


def build_sprint_speed_join_report(joined_df: pd.DataFrame) -> SprintSpeedJoinReport:
    """Compute coverage statistics for a sprint-speed join."""
    total_rows = len(joined_df)
    matched_rows = int(joined_df["has_sprint_speed"].sum())
    unmatched_rows = total_rows - matched_rows
    match_rate = matched_rows / total_rows if total_rows else 0.0
    unique_batters = int(joined_df["batter"].nunique())
    unique_batters_matched = int(joined_df.loc[joined_df["has_sprint_speed"], "batter"].nunique())
    return SprintSpeedJoinReport(
        total_rows=total_rows,
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        match_rate=match_rate,
        unique_batters=unique_batters,
        unique_batters_matched=unique_batters_matched,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-input", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cleaned_df = pd.read_parquet(args.cleaned_input)
    try:
        sprint_speed_df = load_sprint_speed(args.raw_dir, args.seasons)
    except JoinSprintSpeedError as exc:
        logger.error(str(exc))
        return 2

    joined = join_sprint_speed(cleaned_df, sprint_speed_df)
    report = build_sprint_speed_join_report(joined)
    logger.info(
        "Sprint-speed join: %d/%d rows matched (%.2f%%), %d/%d unique batters matched",
        report.matched_rows,
        report.total_rows,
        100 * report.match_rate,
        report.unique_batters_matched,
        report.unique_batters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False)
    logger.info("Saved %d rows to %s", len(joined), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
