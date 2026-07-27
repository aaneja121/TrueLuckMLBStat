"""Build a cleaned, one-row-per-eligible-event batted-ball table.

Usage:

    python -m mlb_luck_score.data.clean_batted_balls \\
        --input data/raw/statcast_2024_sample.parquet \\
        --output data/processed/cleaned_batted_balls.parquet

This module never downloads anything and never touches the network -- it
only transforms a raw Statcast extract (as produced by
`mlb_luck_score.data.download_statcast`) into the cleaned event table used
by feature engineering and model training.

Eligibility and outcome-class labeling are delegated entirely to
`mlb_luck_score.eligibility` -- this module does not duplicate that logic.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_luck_score.eligibility import compute_eligibility

logger = logging.getLogger(__name__)

#: Columns without which we cannot build a stable event identifier or run
#: eligibility logic. Missing any of these is a hard error.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "batter",
    "events",
)

#: Columns retained when present, added as an all-null column (and logged)
#: when absent. Missing optional columns must never crash the pipeline.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "pitcher",
    "player_name",
    "home_team",
    "away_team",
    "venue",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "description",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hit_location",
    "hc_x",
    "hc_y",
    "bb_type",
    "stand",
    "p_throws",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "delta_home_win_exp",
    "bat_speed",
    "sprint_speed",
)

# hc_x/hc_y are Statcast's raw visualization coordinates for where a batted
# ball was fielded, in an arbitrary pixel-like coordinate system centered
# roughly on home plate. The constants below (125.42, 198.27) are the
# widely-used community approximation (originally popularized in public
# sabermetrics tooling) for converting them to a spray angle in degrees,
# where 0 = straight up the middle, negative = third-base side, positive =
# first-base side. This conversion is APPROXIMATE and has not been
# independently verified against Baseball Savant's internal geometry; treat
# `spray_angle_approx` and everything derived from it as directional, not
# precise.
_HC_X_ORIGIN = 125.42
_HC_Y_ORIGIN = 198.27

_SPRAY_SECTOR_EDGES = (-45.0, -27.0, -9.0, 9.0, 27.0, 45.0)
_SPRAY_SECTOR_LABELS = ("left", "left_center", "center", "right_center", "right")
_PULL_SECTORS_FOR_RIGHTY = {"left", "left_center"}
_OPPO_SECTORS_FOR_RIGHTY = {"right", "right_center"}


class CleaningError(ValueError):
    """Raised for unrecoverable input problems (missing required columns)."""


def read_raw_table(path: Path) -> pd.DataFrame:
    """Read a raw Statcast extract, handling empty/malformed files clearly."""
    if not path.exists():
        raise CleaningError(f"Input file does not exist: {path}")
    if path.stat().st_size == 0:
        logger.warning("Input file %s is empty (0 bytes); returning an empty table.", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise CleaningError(f"Could not parse input file {path}: {exc}") from exc

    if df.empty:
        logger.warning("Input file %s parsed to zero rows.", path)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CleaningError(
            f"Input is missing required column(s): {missing}. "
            f"Required columns are: {list(REQUIRED_COLUMNS)}"
        )


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing optional columns as null, logging their absence."""
    out = df.copy()
    missing = [c for c in OPTIONAL_COLUMNS if c not in out.columns]
    if missing:
        logger.info("Optional column(s) absent from input, adding as null: %s", missing)
        for col in missing:
            out[col] = pd.array([None] * len(out), dtype="object")
    return out


def build_event_id(df: pd.DataFrame) -> pd.Series:
    """Deterministically build a stable event id from natural identifiers."""
    return (
        df["game_pk"].astype("Int64").astype(str)
        + "-"
        + df["at_bat_number"].astype("Int64").astype(str)
        + "-"
        + df["pitch_number"].astype("Int64").astype(str)
    )


def _spray_angle_degrees(hc_x: float, hc_y: float) -> float:
    if pd.isna(hc_x) or pd.isna(hc_y):
        return math.nan
    return math.degrees(math.atan2(hc_x - _HC_X_ORIGIN, _HC_Y_ORIGIN - hc_y))


def _spray_sector(angle: float) -> str | None:
    if pd.isna(angle):
        return None
    clipped = min(max(angle, _SPRAY_SECTOR_EDGES[0]), _SPRAY_SECTOR_EDGES[-1])
    idx = int(np.searchsorted(_SPRAY_SECTOR_EDGES, clipped, side="right") - 1)
    idx = min(max(idx, 0), len(_SPRAY_SECTOR_LABELS) - 1)
    return _SPRAY_SECTOR_LABELS[idx]


def add_spray_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add approximate spray-angle/sector/pull-oppo features from hc_x/hc_y.

    See the module-level docstring note on `_HC_X_ORIGIN` / `_HC_Y_ORIGIN`:
    this is a documented approximation, not a verified geometric transform.
    """
    out = df.copy()
    out["spray_angle_approx"] = [
        _spray_angle_degrees(x, y) for x, y in zip(out["hc_x"], out["hc_y"], strict=True)
    ]
    out["spray_sector"] = [_spray_sector(a) for a in out["spray_angle_approx"]]

    is_pull = pd.array([pd.NA] * len(out), dtype="boolean")
    is_center = pd.array([pd.NA] * len(out), dtype="boolean")
    is_oppo = pd.array([pd.NA] * len(out), dtype="boolean")

    for i, (sector, stand) in enumerate(zip(out["spray_sector"], out["stand"], strict=True)):
        if sector is None or pd.isna(stand):
            continue
        if sector == "center":
            is_center[i], is_pull[i], is_oppo[i] = True, False, False
        elif stand == "R":
            is_pull[i] = sector in _PULL_SECTORS_FOR_RIGHTY
            is_oppo[i] = sector in _OPPO_SECTORS_FOR_RIGHTY
            is_center[i] = False
        elif stand == "L":
            is_pull[i] = sector in _OPPO_SECTORS_FOR_RIGHTY
            is_oppo[i] = sector in _PULL_SECTORS_FOR_RIGHTY
            is_center[i] = False

    out["is_pull"] = is_pull
    out["is_center"] = is_center
    out["is_opposite_field"] = is_oppo
    return out


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `game_date` into a `season` year column without silent coercion.

    Rows with an unparseable `game_date` get `season = <NA>` and are flagged
    ineligible for training (their season role can't be determined) rather
    than being silently dropped or assigned a guessed year.
    """
    out = df.copy()
    parsed = pd.to_datetime(out["game_date"], errors="coerce")
    n_invalid = int(parsed.isna().sum())
    if n_invalid:
        logger.warning(
            "%d row(s) have an unparseable game_date; marking ineligible for training.",
            n_invalid,
        )
    out["season"] = parsed.dt.year.astype("Int64")
    out["_invalid_game_date"] = parsed.isna()
    return out


def clean_batted_balls(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw Statcast extract.

    Returns one cleaned row per eligible fair-batted-ball event (per
    `mlb_luck_score.eligibility`), with engineered spray features, a
    deterministic `event_id`, a `season` column, and training-eligibility
    flags/reasons. Ineligible rows are dropped entirely (they were never
    fair-batted-ball events); rows that ARE eligible events but that fail a
    training-quality check are retained with `eligible_for_training=False`
    and a specific `training_exclusion_reason`.
    """
    if raw.empty:
        logger.warning("Received an empty raw table; returning an empty cleaned table.")
        return raw.copy()

    validate_required_columns(raw)
    n_input = len(raw)

    df = ensure_optional_columns(raw)

    n_before_dedup = len(df)
    dedup_keys = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in df.columns]
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    n_duplicates = n_before_dedup - len(df)
    if n_duplicates:
        logger.info("Dropped %d duplicate raw record(s).", n_duplicates)

    df = compute_eligibility(df)
    n_eligible = int(df["is_eligible"].sum())
    df = df[df["is_eligible"]].copy()
    logger.info("Input rows: %d | eligible fair-batted-ball events: %d", n_input, n_eligible)

    df["event_id"] = build_event_id(df)
    df = add_spray_features(df)
    df = add_season(df)

    df.loc[df["_invalid_game_date"], "eligible_for_training"] = False
    df.loc[df["_invalid_game_date"], "training_exclusion_reason"] = "invalid_game_date"
    df = df.drop(columns=["_invalid_game_date"])

    n_training_eligible = int(df["eligible_for_training"].sum())
    reason_counts = (
        df.loc[~df["eligible_for_training"], "training_exclusion_reason"]
        .value_counts(dropna=False)
        .to_dict()
    )
    logger.info("Training-eligible rows: %d", n_training_eligible)
    logger.info("Training-exclusion reason counts: %s", reason_counts)

    n_missing_required = int(df[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum())
    logger.info("Rows with a missing required identifier field: %d", n_missing_required)

    sort_cols = [c for c in ("event_id",) if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def save_dataframe(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
    logger.info("Wrote %d cleaned rows to %s", len(df), output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw Statcast extract path")
    parser.add_argument("--output", type=Path, required=True, help="Cleaned output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        raw = read_raw_table(args.input)
        cleaned = clean_batted_balls(raw)
    except CleaningError as exc:
        logger.error(str(exc))
        return 2

    save_dataframe(cleaned, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
