"""Join per-game venue metadata to the cleaned development dataset.

Usage:

    python -m mlb_luck_score.data.join_venue_metadata \\
        --cleaned-input data/processed/cleaned_development_data.parquet \\
        --metadata-dir data/raw \\
        --output data/processed/cleaned_development_data_with_venue.parquet

Loads the per-season `game_metadata_<season>.parquet` files produced by
`mlb_luck_score.data.download_game_metadata`, applies a small set of
hand-reviewed corrections (see `mlb_luck_score.data.
game_metadata_overrides` -- the raw per-season cache files themselves are
NEVER modified), validates that no `game_pk` maps to more than one venue,
and left-joins `venue_id`, `venue_name`, `roof_type`, `surface_type`, and
`is_neutral_site` onto the cleaned batted-ball table by `game_pk`. Rows
whose game has no venue metadata even after overrides are KEPT, not
dropped, with `has_venue_metadata` explicitly `False` and the venue columns
null.

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
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    game_metadata_path,
)
from mlb_luck_score.data.game_metadata_overrides import GAME_METADATA_OVERRIDES

logger = logging.getLogger(__name__)

VENUE_JOIN_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "venue_id",
    "venue_name",
    "roof_type",
    "surface_type",
    "is_neutral_site",
)


class JoinVenueMetadataError(ValueError):
    """Raised for unrecoverable venue-metadata join problems."""


def load_game_metadata(raw_dir: Path, seasons: Sequence[int]) -> pd.DataFrame:
    """Load and combine per-season game-metadata files.

    Raises:
        JoinVenueMetadataError: If any requested season's metadata file is
            missing (all missing seasons are listed at once).
    """
    missing = [
        (season, game_metadata_path(raw_dir, season))
        for season in seasons
        if not game_metadata_path(raw_dir, season).exists()
    ]
    if missing:
        details = "; ".join(f"season {season} (expected {path})" for season, path in missing)
        raise JoinVenueMetadataError(
            f"Missing game-metadata file(s) for {len(missing)} of {len(seasons)} requested "
            f"season(s): {details}. Run `make download-game-metadata` first."
        )

    frames = [pd.read_parquet(game_metadata_path(raw_dir, season)) for season in seasons]
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["game_pk"], keep="first")
    n_duplicates = before - len(combined)
    if n_duplicates:
        logger.info(
            "Dropped %d duplicate game_pk row(s) across season metadata files.", n_duplicates
        )
    return apply_metadata_overrides(combined)


def apply_metadata_overrides(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the small, hand-reviewed corrections in `game_metadata_overrides`.

    The raw per-season cache files on disk are never touched by this --
    overrides are applied fresh to an in-memory copy every time metadata is
    loaded, so re-running `make download-game-metadata` always produces the
    same untouched raw API response, and reviewing/updating an override only
    requires editing `game_metadata_overrides.py`.
    """
    if not GAME_METADATA_OVERRIDES:
        return metadata_df

    out = metadata_df.copy()
    applied_game_pks: list[int] = []
    for game_pk, override in GAME_METADATA_OVERRIDES.items():
        mask = out["game_pk"] == game_pk
        if not mask.any():
            continue
        out.loc[mask, "venue_id"] = override.venue_id
        out.loc[mask, "venue_name"] = override.venue_name
        out.loc[mask, "roof_type"] = override.roof_type
        out.loc[mask, "surface_type"] = override.surface_type
        out.loc[mask, "is_neutral_site"] = override.is_neutral_site
        applied_game_pks.append(game_pk)

    if applied_game_pks:
        logger.info(
            "Applied %d hand-reviewed game-metadata override(s) for game_pk(s) %s "
            "(see mlb_luck_score.data.game_metadata_overrides).",
            len(applied_game_pks),
            applied_game_pks,
        )
    return out


def validate_no_conflicting_venues(metadata_df: pd.DataFrame) -> None:
    """Fail clearly if any `game_pk` is mapped to more than one distinct venue.

    Raises:
        JoinVenueMetadataError: Listing every conflicting `game_pk`.
    """
    with_venue = metadata_df.dropna(subset=["venue_id"])
    venue_counts_per_game = with_venue.groupby("game_pk")["venue_id"].nunique()
    conflicting = venue_counts_per_game[venue_counts_per_game > 1]
    if not conflicting.empty:
        raise JoinVenueMetadataError(
            f"{len(conflicting)} game_pk(s) map to more than one venue -- refusing to join: "
            f"{conflicting.index.tolist()}"
        )


def join_venue_metadata(cleaned_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join venue metadata onto `cleaned_df` by `game_pk`.

    Validates there are no conflicting venue mappings before joining. Rows
    without a venue match are preserved with `has_venue_metadata=False` and
    null venue columns -- never dropped.
    """
    validate_no_conflicting_venues(metadata_df)

    venue_cols = metadata_df[list(VENUE_JOIN_COLUMNS)].drop_duplicates(subset=["game_pk"])
    joined = cleaned_df.merge(venue_cols, on="game_pk", how="left", validate="many_to_one")
    joined["has_venue_metadata"] = joined["venue_id"].notna()
    return joined


@dataclass(frozen=True)
class VenueJoinReport:
    """Coverage report for a venue-metadata join."""

    total_games: int
    total_event_rows: int
    matched_rows: int
    unmatched_rows: int
    venue_match_rate: float
    unmatched_games: int
    duplicate_mappings: int
    counts_by_venue: dict[str, int]
    counts_by_season_venue: dict[str, int]


def build_venue_join_report(joined_df: pd.DataFrame, metadata_df: pd.DataFrame) -> VenueJoinReport:
    """Compute coverage/health statistics for a venue-metadata join."""
    total_games = int(joined_df["game_pk"].nunique())
    total_event_rows = len(joined_df)
    matched_rows = int(joined_df["has_venue_metadata"].sum())
    unmatched_rows = total_event_rows - matched_rows
    venue_match_rate = matched_rows / total_event_rows if total_event_rows else 0.0
    unmatched_games = int(joined_df.loc[~joined_df["has_venue_metadata"], "game_pk"].nunique())

    with_venue = metadata_df.dropna(subset=["venue_id"])
    venue_counts_per_game = with_venue.groupby("game_pk")["venue_id"].nunique()
    duplicate_mappings = int((venue_counts_per_game > 1).sum())

    counts_by_venue = {
        str(name): int(count)
        for name, count in joined_df["venue_name"].value_counts(dropna=True).items()
    }

    counts_by_season_venue: dict[str, int] = {}
    if "season" in joined_df.columns:
        season_venue_df = (
            joined_df.dropna(subset=["venue_name"])
            .groupby(["season", "venue_name"])
            .size()
            .reset_index(name="row_count")
        )
        seasons = season_venue_df["season"].to_numpy()
        venues = season_venue_df["venue_name"].to_numpy()
        row_counts = season_venue_df["row_count"].to_numpy()
        for season_value, venue_value, row_count in zip(seasons, venues, row_counts, strict=True):
            counts_by_season_venue[f"{season_value}|{venue_value}"] = int(row_count)

    return VenueJoinReport(
        total_games=total_games,
        total_event_rows=total_event_rows,
        matched_rows=matched_rows,
        unmatched_rows=unmatched_rows,
        venue_match_rate=venue_match_rate,
        unmatched_games=unmatched_games,
        duplicate_mappings=duplicate_mappings,
        counts_by_venue=counts_by_venue,
        counts_by_season_venue=counts_by_season_venue,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-input", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEVELOPMENT_SEASONS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_development_data_with_venue.parquet",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cleaned_df = pd.read_parquet(args.cleaned_input)

    try:
        metadata_df = load_game_metadata(args.metadata_dir, args.seasons)
        joined_df = join_venue_metadata(cleaned_df, metadata_df)
    except JoinVenueMetadataError as exc:
        logger.error(str(exc))
        return 2

    report = build_venue_join_report(joined_df, metadata_df)
    logger.info("Total games: %d", report.total_games)
    logger.info("Total event rows: %d", report.total_event_rows)
    logger.info(
        "Venue match rate: %.4f (%d matched, %d unmatched rows)",
        report.venue_match_rate,
        report.matched_rows,
        report.unmatched_rows,
    )
    logger.info("Unmatched games: %d", report.unmatched_games)
    logger.info("Duplicate game->venue mappings: %d", report.duplicate_mappings)
    logger.info("Counts by venue: %s", report.counts_by_venue)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(args.output, index=False)
    logger.info("Wrote %d joined rows to %s", len(joined_df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
