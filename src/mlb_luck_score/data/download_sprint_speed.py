"""Download the public Baseball Savant Sprint Speed leaderboard, per season.

Usage (requires internet access):

    python -m mlb_luck_score.data.download_sprint_speed \\
        --seasons 2021 2022 2023 2024 --output-dir data/raw

Statcast's per-PITCH data (`mlb_luck_score.data.download_statcast`) has NO
batter-speed field at all -- sprint speed is exposed only as a separate,
SEASON-LEVEL leaderboard (`pybaseball.statcast_sprint_speed`, Baseball
Savant's own "Sprint Speed" leaderboard: "feet per second in a player's
fastest one-second window," computed from ~top two-thirds of a player's
qualifying runs that season). This is the Version 0.8 infield-opportunity
model's ONLY source of runner-speed information -- see
`mlb_luck_score.data.join_sprint_speed` for how it's joined onto per-play rows
by `batter`+`season`, and README.md "Infield opportunity (Version 0.8)" for
why a per-play speed measurement does not exist in public data.

One raw Parquet file PER SEASON (`sprint_speed_<season>.parquet`), columns
`player_id` (MLBAM ID, same space as Statcast's `batter` column),
`sprint_speed` (ft/sec), and `hp_to_1b` (home-to-first time in seconds,
Savant's OWN season-level proxy for "how fast can this batter reach first" --
reported alongside `sprint_speed` since it is directly relevant to the
Version 0.8 timing-margin candidate, `infield_time_margin_proxy_v08_candidate`).

`min_opp` (default 10, matching `pybaseball.statcast_sprint_speed`'s own
default) is a QUALIFICATION threshold Savant itself uses -- batters below it
are not published on the leaderboard at all, so they will be missing here.
This is Savant's own documented data limitation, not something this module
can work around; `mlb_luck_score.data.join_sprint_speed` reports what
fraction of eligible rows fail to match a sprint-speed row for exactly this
reason.

Resumable: if a season's output file already exists, that season is SKIPPED
unless `--overwrite` is passed. A failure downloading one season does not
abort the others.

2025 is a protected final-test season: this command refuses to download it
unless `--allow-final-evaluation` is explicitly passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    RAW_DATA_DIR,
    ProtectedSeasonError,
    assert_seasons_allowed,
    sprint_speed_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_OPPORTUNITIES = 10

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_EXISTING = "skipped_existing"
STATUS_FAILED = "failed"
STATUS_PLANNED = "planned"

SPRINT_SPEED_COLUMNS: tuple[str, ...] = ("player_id", "sprint_speed", "hp_to_1b")


@dataclass(frozen=True)
class SeasonSprintSpeedResult:
    """Outcome of attempting to download one season's Sprint Speed leaderboard."""

    season: int
    status: str
    rows: int
    path: Path
    detail: str = ""


def fetch_season_sprint_speed(
    season: int, *, min_opp: int = DEFAULT_MIN_OPPORTUNITIES
) -> pd.DataFrame:
    """Fetch and normalize one season's Sprint Speed leaderboard via `pybaseball`."""
    import pybaseball

    raw = pybaseball.statcast_sprint_speed(season, min_opp=min_opp)
    if raw.empty:
        return pd.DataFrame(columns=list(SPRINT_SPEED_COLUMNS))
    return raw[list(SPRINT_SPEED_COLUMNS)].reset_index(drop=True)


def download_sprint_speed(
    seasons: list[int],
    output_dir: Path,
    *,
    min_opp: int = DEFAULT_MIN_OPPORTUNITIES,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_final_evaluation: bool = False,
) -> list[SeasonSprintSpeedResult]:
    """Download one Sprint Speed leaderboard Parquet file per requested season."""
    assert_seasons_allowed(tuple(seasons), allow_final_evaluation=allow_final_evaluation)

    results: list[SeasonSprintSpeedResult] = []
    for season in seasons:
        path = sprint_speed_path(output_dir, season)

        if path.exists() and not overwrite:
            logger.info(
                "[season %d] output already exists at %s; skipping (pass --overwrite to redo).",
                season,
                path,
            )
            results.append(SeasonSprintSpeedResult(season, STATUS_SKIPPED_EXISTING, -1, path))
            continue

        if dry_run:
            logger.info(
                "[DRY RUN][season %d] would fetch Sprint Speed leaderboard (min_opp=%d) into "
                "%s (no network access performed).",
                season,
                min_opp,
                path,
            )
            results.append(SeasonSprintSpeedResult(season, STATUS_PLANNED, 0, path))
            continue

        try:
            season_df = fetch_season_sprint_speed(season, min_opp=min_opp)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            logger.error("[season %d] failed: %s", season, exc)
            results.append(SeasonSprintSpeedResult(season, STATUS_FAILED, 0, path, detail=str(exc)))
            continue

        if season_df.empty:
            logger.error("[season %d] no rows returned; refusing to write an empty file.", season)
            results.append(
                SeasonSprintSpeedResult(season, STATUS_FAILED, 0, path, detail="empty result")
            )
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(path, index=False)
        logger.info("[season %d] saved %d player(s) to %s", season, len(season_df), path)
        results.append(SeasonSprintSpeedResult(season, STATUS_DOWNLOADED, len(season_df), path))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--min-opp", type=int, default=DEFAULT_MIN_OPPORTUNITIES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        logger.info("This command requires internet access to reach Baseball Savant.")

    try:
        results = download_sprint_speed(
            args.seasons,
            args.output_dir,
            min_opp=args.min_opp,
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
