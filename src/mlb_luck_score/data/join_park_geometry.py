"""Contact Luck v0.4: join reviewed park geometry onto every eligible batted-ball event.

Usage:

    python -m mlb_luck_score.data.join_park_geometry \\
        --input data/processed/cleaned_development_data_with_venue.parquet \\
        --output data/processed/cleaned_development_data_with_geometry.parquet

Joins the reviewed wall-geometry table (`mlb_luck_score.data.park_geometry`)
onto the venue-joined cleaned development dataset (see
`mlb_luck_score.data.join_venue_metadata`) by `venue_id` + `game_date` +
`spray_angle_approx`. Every input row is PRESERVED -- rows without usable
geometry get explicit status columns (never dropped, never silently filled
with a guessed value; see `GEOMETRY_STATUS_*` constants and module
docstring "Missing/non-applicable geometry" below).

This module never downloads anything and never touches the network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_luck_score.config import PROCESSED_DATA_DIR
from mlb_luck_score.data.park_geometry import (
    PARK_GEOMETRY_CONFIGS,
    TEMPORARY_OR_SPECIAL_VENUE_IDS,
    GeometryLookupResult,
    ParkGeometryConfig,
    interpolate_wall_geometry,
)

logger = logging.getLogger(__name__)

#: Mutually exclusive per-row geometry status reasons. See module docstring
#: "Missing/non-applicable geometry" -- these are checked in this priority
#: order (first match wins) so every row gets exactly one status.
GEOMETRY_STATUS_OK = "ok"
GEOMETRY_STATUS_MISSING_VENUE_ID = "missing_venue_id"
GEOMETRY_STATUS_MISSING_VENUE_METADATA = "missing_venue_metadata"
GEOMETRY_STATUS_TEMPORARY_VENUE = "temporary_or_special_venue"
GEOMETRY_STATUS_MISSING_GAME_DATE = "missing_game_date"
GEOMETRY_STATUS_NO_CONFIG_FOR_DATE = "no_geometry_reviewed_for_venue_or_date"
GEOMETRY_STATUS_MISSING_SPRAY_ANGLE = "missing_spray_angle"
GEOMETRY_STATUS_OUTSIDE_RANGE = "outside_modeled_angular_range"

#: Wall height (feet) at/above which `high_wall_indicator` is True. A
#: documented Version 0.4 research placeholder, not a validated threshold --
#: chosen to separate genuinely tall walls (Green Monster 37 ft, Coors
#: right-center 16.5 ft, PNC/Rogers Centre segments ~13-21 ft) from the
#: typical 6-10 ft outfield wall.
DEFAULT_HIGH_WALL_THRESHOLD_FT = 15.0

#: `abs(projected_distance_to_wall_margin)` thresholds (feet) for the
#: near-wall boolean flags.
NEAR_WALL_THRESHOLDS_FT: tuple[int, ...] = (5, 10, 20)


def _resolve_row(
    venue_id: Any,
    has_venue_metadata: Any,
    game_date: Any,
    spray_angle: Any,
    configs_by_venue: dict[int, tuple[ParkGeometryConfig, ...]],
) -> tuple[str, ParkGeometryConfig | None, GeometryLookupResult | None]:
    """Resolve one row's geometry status/config/lookup. Pure function, no pandas state."""
    if venue_id is None or (isinstance(venue_id, float) and pd.isna(venue_id)):
        return GEOMETRY_STATUS_MISSING_VENUE_ID, None, None

    venue_id_int = int(venue_id)

    if not bool(has_venue_metadata):
        return GEOMETRY_STATUS_MISSING_VENUE_METADATA, None, None

    if venue_id_int in TEMPORARY_OR_SPECIAL_VENUE_IDS:
        return GEOMETRY_STATUS_TEMPORARY_VENUE, None, None

    if game_date is None or (isinstance(game_date, float) and pd.isna(game_date)):
        return GEOMETRY_STATUS_MISSING_GAME_DATE, None, None

    game_date_str = str(game_date)[:10]
    candidates = configs_by_venue.get(venue_id_int, ())
    config = None
    for c in candidates:
        if c.effective_start_date <= game_date_str and (
            c.effective_end_date is None or game_date_str <= c.effective_end_date
        ):
            config = c
            break
    if config is None:
        return GEOMETRY_STATUS_NO_CONFIG_FOR_DATE, None, None

    if spray_angle is None or (isinstance(spray_angle, float) and pd.isna(spray_angle)):
        return GEOMETRY_STATUS_MISSING_SPRAY_ANGLE, config, None

    lookup = interpolate_wall_geometry(config, float(spray_angle))
    if lookup is None:
        return GEOMETRY_STATUS_OUTSIDE_RANGE, config, None

    return GEOMETRY_STATUS_OK, config, lookup


def join_park_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Join reviewed park geometry onto every row of `df`.

    Args:
        df: Cleaned development data already joined with venue metadata
            (see `mlb_luck_score.data.join_venue_metadata`) -- must have
            `venue_id`, `has_venue_metadata`, `game_date`, and
            `spray_angle_approx` columns. `hit_distance_sc` is used if
            present to compute distance-margin columns.

    Returns:
        A copy of `df` with geometry columns added. Every input row is
        preserved (see module docstring).
    """
    required = ("venue_id", "has_venue_metadata", "game_date", "spray_angle_approx")
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"join_park_geometry requires column(s) {missing_cols} -- run "
            "mlb_luck_score.data.join_venue_metadata first."
        )

    out = df.copy()
    hit_distance = (
        out["hit_distance_sc"]
        if "hit_distance_sc" in out.columns
        else pd.Series([pd.NA] * len(out), index=out.index)
    )

    configs_by_venue_lists: dict[int, list[ParkGeometryConfig]] = {}
    for c in PARK_GEOMETRY_CONFIGS:
        configs_by_venue_lists.setdefault(c.venue_id, []).append(c)
    configs_by_venue: dict[int, tuple[ParkGeometryConfig, ...]] = {
        k: tuple(v) for k, v in configs_by_venue_lists.items()
    }

    n = len(out)
    statuses = np.empty(n, dtype=object)
    config_ids: list[Any] = [pd.NA] * n
    distances = np.full(n, np.nan)
    heights = np.full(n, np.nan)
    segment_labels: list[Any] = [pd.NA] * n
    source_types: list[Any] = [pd.NA] * n
    interp_degrees = np.full(n, np.nan)

    venue_ids = out["venue_id"].to_numpy()
    has_venue = out["has_venue_metadata"].to_numpy()
    game_dates = out["game_date"].to_numpy()
    spray_angles = out["spray_angle_approx"].to_numpy()

    for i in range(n):
        status, config, lookup = _resolve_row(
            venue_ids[i], has_venue[i], game_dates[i], spray_angles[i], configs_by_venue
        )
        statuses[i] = status
        if config is not None:
            config_ids[i] = config.geometry_config_id
        if lookup is not None:
            distances[i] = lookup.wall_distance_feet
            heights[i] = lookup.wall_height_feet if lookup.wall_height_feet is not None else np.nan
            segment_labels[i] = lookup.segment_label
            source_types[i] = lookup.distance_source_type
            interp_degrees[i] = lookup.nearest_point_distance_degrees

    out["geometry_config_id"] = pd.array(config_ids, dtype="string")
    out["geometry_status"] = statuses
    out["has_park_geometry"] = out["geometry_status"] == GEOMETRY_STATUS_OK
    out["wall_distance_in_spray_direction"] = distances
    out["wall_height_in_spray_direction"] = heights
    out["wall_segment_label"] = pd.array(segment_labels, dtype="string")
    out["wall_distance_source_type"] = pd.array(source_types, dtype="string")
    out["wall_interpolation_distance_degrees"] = interp_degrees

    hit_distance_num = pd.to_numeric(hit_distance, errors="coerce")
    has_margin_inputs = out["has_park_geometry"] & hit_distance_num.notna()
    margin = pd.Series(np.nan, index=out.index)
    margin.loc[has_margin_inputs] = (
        hit_distance_num.loc[has_margin_inputs]
        - out.loc[has_margin_inputs, "wall_distance_in_spray_direction"]
    )
    out["projected_distance_to_wall_margin"] = margin
    out["absolute_distance_to_wall"] = margin.abs()

    for threshold in NEAR_WALL_THRESHOLDS_FT:
        col = f"near_wall_{threshold}ft"
        flag = pd.array([pd.NA] * n, dtype="boolean")
        flag[has_margin_inputs.to_numpy()] = (
            margin.loc[has_margin_inputs].abs() <= threshold
        ).to_numpy()
        out[col] = flag

    projected_beyond = pd.array([pd.NA] * n, dtype="boolean")
    projected_beyond[has_margin_inputs.to_numpy()] = (margin.loc[has_margin_inputs] > 0).to_numpy()
    out["projected_beyond_wall"] = projected_beyond

    has_height = out["has_park_geometry"] & pd.Series(heights, index=out.index).notna()
    high_wall = pd.array([pd.NA] * n, dtype="boolean")
    high_wall[has_height.to_numpy()] = (
        pd.Series(heights, index=out.index).loc[has_height] >= DEFAULT_HIGH_WALL_THRESHOLD_FT
    ).to_numpy()
    out["high_wall_indicator"] = high_wall

    out["temporary_or_special_venue"] = out["venue_id"].apply(
        lambda v: (not pd.isna(v)) and int(v) in TEMPORARY_OR_SPECIAL_VENUE_IDS
    )
    out["geometry_uncertain"] = (
        out["has_venue_metadata"].astype(bool)
        & ~out["temporary_or_special_venue"]
        & (out["geometry_status"] != GEOMETRY_STATUS_OK)
    )

    return out


@dataclass(frozen=True)
class GeometryJoinReport:
    """Coverage report for a park-geometry join."""

    total_rows: int
    rows_with_geometry: int
    coverage_rate: float
    counts_by_status: dict[str, int]
    counts_by_season: dict[str, int]
    counts_by_venue: dict[str, int]
    counts_by_bb_type: dict[str, int]
    counts_by_outcome_class: dict[str, int]
    counts_by_training_eligibility: dict[str, int]
    n_temporary_or_special_venue_rows: int
    n_configurations_used: int


def build_geometry_join_report(joined_df: pd.DataFrame) -> GeometryJoinReport:
    """Compute coverage/health statistics for a park-geometry join."""
    total_rows = len(joined_df)
    rows_with_geometry = int(joined_df["has_park_geometry"].sum())

    def _counts(col: str) -> dict[str, int]:
        if col not in joined_df.columns:
            return {}
        return {str(k): int(v) for k, v in joined_df[col].value_counts(dropna=False).items()}

    by_season: dict[str, int] = {}
    if "season" in joined_df.columns:
        cross = joined_df.groupby("season", dropna=False)["has_park_geometry"].sum()
        totals = joined_df.groupby("season", dropna=False).size()
        by_season = {str(season): int(cross.get(season, 0)) for season in totals.index}

    by_venue: dict[str, int] = {}
    if "venue_name" in joined_df.columns:
        cross = joined_df.groupby("venue_name", dropna=False)["has_park_geometry"].sum()
        by_venue = {str(k): int(v) for k, v in cross.items()}

    by_bb_type = {}
    if "bb_type" in joined_df.columns:
        cross = joined_df.groupby("bb_type", dropna=False)["has_park_geometry"].sum()
        by_bb_type = {str(k): int(v) for k, v in cross.items()}

    by_outcome = {}
    if "outcome_class" in joined_df.columns:
        cross = joined_df.groupby("outcome_class", dropna=False)["has_park_geometry"].sum()
        by_outcome = {str(k): int(v) for k, v in cross.items()}

    by_training = {}
    if "eligible_for_training" in joined_df.columns:
        cross = joined_df.groupby("eligible_for_training", dropna=False)["has_park_geometry"].sum()
        by_training = {str(k): int(v) for k, v in cross.items()}

    return GeometryJoinReport(
        total_rows=total_rows,
        rows_with_geometry=rows_with_geometry,
        coverage_rate=rows_with_geometry / total_rows if total_rows else 0.0,
        counts_by_status=_counts("geometry_status"),
        counts_by_season=by_season,
        counts_by_venue=by_venue,
        counts_by_bb_type=by_bb_type,
        counts_by_outcome_class=by_outcome,
        counts_by_training_eligibility=by_training,
        n_temporary_or_special_venue_rows=int(joined_df["temporary_or_special_venue"].sum()),
        n_configurations_used=int(joined_df["geometry_config_id"].nunique()),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Venue-joined cleaned dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "cleaned_development_data_with_geometry.parquet",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting an existing output file."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        logger.error("Output file %s already exists. Pass --overwrite to replace it.", args.output)
        return 2

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    logger.info("Input rows: %d", len(df))

    joined_df = join_park_geometry(df)
    report = build_geometry_join_report(joined_df)

    logger.info(
        "Rows with usable geometry: %d / %d (%.4f)",
        report.rows_with_geometry,
        report.total_rows,
        report.coverage_rate,
    )
    logger.info("Counts by geometry_status: %s", report.counts_by_status)
    logger.info("Configurations used: %d", report.n_configurations_used)
    logger.info("Temporary/special-venue rows: %d", report.n_temporary_or_special_venue_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(args.output, index=False)
    logger.info("Wrote %d joined rows to %s", len(joined_df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
