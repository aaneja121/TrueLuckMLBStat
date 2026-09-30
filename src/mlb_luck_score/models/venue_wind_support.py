"""Contact Luck v1.2 step 2: per-venue station-wind check.

Question: at each venue, does station (ASOS) wind relate to in-park carry?
The answer decides ONLY whether that venue's wind-direction perturbation
check applies in the v1.2 evaluation. It is never a model input and selects
no model. The rule below was frozen in `docs/plans/v1_2_park_weather_plan.md`
("Step 2") and committed before any per-venue confidence interval was
computed; change it there first, never here alone.

Rule, per venue (`venue_id`), on development seasons 2021-2024 only:

  - Rows (`select_wind_check_rows`): eligible fly balls with measured
    distance, air density and following wind, in outdoor or roof-open games,
    launch speed 90-115 mph and launch angle 20-40 degrees.
  - Estimator (`estimate_venue_coefficients`): OLS of `hit_distance_sc` on
    following wind and air density with launch-speed x launch-angle cell
    fixed effects (2 mph x 2 degree cells, within-cell demeaning; cells with
    fewer than `MIN_ROWS_PER_CELL` rows dropped).
  - Interval: game_pk-clustered bootstrap, 500 reps, seed 42, 95% percentile
    CI -- the `compare_near_wall_calibration_gate` convention.
  - Three-way status (`classify_venue_wind`, v0.7D pattern): supported (whole
    CI above 0), contradicted (whole CI below 0), else insufficient evidence.
    Insufficient evidence is never a pass.
  - Bonferroni sensitivity interval (normal approximation from the bootstrap
    SE, alpha split across all venues in the table) is REPORTED, not gating.

Only the direction is tested: a supported venue's slope says nothing about
in-park wind magnitude, and airport wind stays a noisy proxy for it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from mlb_luck_score.config import DEVELOPMENT_SEASONS, PROCESSED_DATA_DIR, TABLES_DIR
from mlb_luck_score.data.build_game_weather import (
    ROOF_STATUS_OUTDOOR_OPEN_AIR,
    ROOF_STATUS_RETRACTABLE_OPEN,
)
from mlb_luck_score.data.weather_physics import MPH_TO_MPS
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
)

logger = logging.getLogger(__name__)

STATUS_WIND_SUPPORTED = "wind_supported"
STATUS_WIND_CONTRADICTED = "wind_contradicted"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

#: Roof states with outdoor wind reaching the field. Closed/indoor games have
#: no station wind to test, and unknown roof status is never guessed.
WIND_ROOF_STATUSES = frozenset({ROOF_STATUS_OUTDOOR_OPEN_AIR, ROOF_STATUS_RETRACTABLE_OPEN})

#: The step 1 carry band (plan, "Step 1 re-scope"), fixed before step 2.
LAUNCH_SPEED_RANGE_MPH = (90.0, 115.0)
LAUNCH_ANGLE_RANGE_DEG = (20.0, 40.0)
LAUNCH_SPEED_CELL_MPH = 2.0
LAUNCH_ANGLE_CELL_DEG = 2.0
MIN_ROWS_PER_CELL = 5

DEFAULT_INPUT = PROCESSED_DATA_DIR / "cleaned_development_data_with_weather.parquet"
OUTPUT_FILENAME = "v1_2_venue_wind_check.csv"

_REQUIRED_COLUMNS = (
    "season",
    "game_pk",
    "venue_id",
    "venue_name",
    "bb_type",
    "eligible_for_training",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
    "roof_status",
    "air_density_kg_m3",
    "following_wind_mps",
)
_WIND = "following_wind_mps"
_DENSITY = "air_density_kg_m3"
_DISTANCE = "hit_distance_sc"


def select_wind_check_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen row filter. Refuses any non-development season."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    seasons = {int(s) for s in pd.unique(df["season"].dropna())}
    outside = seasons - set(DEVELOPMENT_SEASONS)
    if outside:
        raise ValueError(
            f"season(s) {sorted(outside)} are not development seasons "
            f"{DEVELOPMENT_SEASONS}; this check reads 2021-2024 only"
        )

    eligible = df["eligible_for_training"].astype("boolean").fillna(False).astype(bool)
    mask = (
        eligible
        & (df["bb_type"] == "fly_ball").fillna(False).astype(bool)
        & df["roof_status"].isin(WIND_ROOF_STATUSES)
        & df[_DISTANCE].notna()
        & df[_DENSITY].notna()
        & df[_WIND].notna()
        & df["launch_speed"].between(*LAUNCH_SPEED_RANGE_MPH)
        & df["launch_angle"].between(*LAUNCH_ANGLE_RANGE_DEG)
    )
    return df.loc[mask].copy()


def demean_within_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Demean distance, wind and density within launch-speed x launch-angle
    cells (the fixed effects), dropping cells below `MIN_ROWS_PER_CELL`.
    Callers pass a single venue's rows."""
    out = df.copy()
    ls_cell = np.floor(out["launch_speed"].to_numpy(dtype=float) / LAUNCH_SPEED_CELL_MPH)
    la_cell = np.floor(out["launch_angle"].to_numpy(dtype=float) / LAUNCH_ANGLE_CELL_DEG)
    out["_cell"] = (
        pd.Series(ls_cell, index=out.index).astype(int).astype(str)
        + "_"
        + pd.Series(la_cell, index=out.index).astype(int).astype(str)
    )
    size = out.groupby("_cell")[_DISTANCE].transform("size")
    out = out.loc[size >= MIN_ROWS_PER_CELL].copy()
    for col in (_DISTANCE, _WIND, _DENSITY):
        out[f"{col}_dm"] = out[col] - out.groupby("_cell")[col].transform("mean")
    return out.drop(columns="_cell")


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _design(dm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([dm[f"{_WIND}_dm"].to_numpy(float), dm[f"{_DENSITY}_dm"].to_numpy(float)])
    return x, dm[f"{_DISTANCE}_dm"].to_numpy(float)


def estimate_venue_coefficients(df: pd.DataFrame) -> dict[str, float]:
    """Point estimates for one venue's selected rows: ft per mph of following
    wind and ft per +0.01 kg/m^3 of air density."""
    x, y = _design(demean_within_cells(df))
    beta = _ols(x, y)
    return {
        "ft_per_mph_following_wind": float(beta[0] * MPH_TO_MPS),
        "ft_per_001_density": float(beta[1] * 0.01),
    }


def classify_venue_wind(n_plays: int, n_games: int, ci_low: float, ci_high: float) -> str:
    """Three-way status; see the module docstring."""
    if n_plays < MIN_SUBGROUP_PLAYS or n_games < MIN_SUBGROUP_GAMES:
        return STATUS_INSUFFICIENT_EVIDENCE
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return STATUS_INSUFFICIENT_EVIDENCE
    if ci_low > 0:
        return STATUS_WIND_SUPPORTED
    if ci_high < 0:
        return STATUS_WIND_CONTRADICTED
    return STATUS_INSUFFICIENT_EVIDENCE


def wind_check_applies(status: str) -> bool:
    """Only a supported venue gets a wind-direction perturbation check."""
    return status == STATUS_WIND_SUPPORTED


def _bootstrap_wind_slopes(dm: pd.DataFrame, *, n_reps: int, seed: int) -> np.ndarray:
    """Game_pk-clustered bootstrap of the wind slope (ft per mph). Resamples
    games of the already-demeaned rows, so cell means stay those of the full
    venue sample."""
    x, y = _design(dm)
    games = dm["game_pk"].to_numpy()
    unique_games = np.unique(games)
    game_to_rows = {g: np.where(games == g)[0] for g in unique_games}
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_reps)
    for rep in range(n_reps):
        sampled = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([game_to_rows[g] for g in sampled])
        slopes[rep] = _ols(x[rows], y[rows])[0] * MPH_TO_MPS
    return slopes


def run_venue_wind_check(
    df: pd.DataFrame,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> pd.DataFrame:
    """One row per venue with estimates, CIs, status and applicability."""
    rows = select_wind_check_rows(df)
    alpha = (1.0 - ci) / 2.0
    records: list[dict[str, Any]] = []
    for venue_id, venue_rows in rows.groupby("venue_id", sort=True):
        dm = demean_within_cells(venue_rows)
        n_plays = len(dm)
        n_games = int(dm["game_pk"].nunique())
        record: dict[str, Any] = {
            "venue_id": venue_id,
            "venue_name": " / ".join(
                sorted(venue_rows["venue_name"].dropna().astype(str).unique())
            ),
            "n_plays": n_plays,
            "n_games": n_games,
            "seasons": ",".join(str(s) for s in sorted(venue_rows["season"].unique())),
            "ft_per_mph_following_wind": float("nan"),
            "ft_per_001_density": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_se": float("nan"),
        }
        if n_games >= 2 and n_plays >= 3:
            x, y = _design(dm)
            beta = _ols(x, y)
            slopes = _bootstrap_wind_slopes(dm, n_reps=n_reps, seed=seed)
            record.update(
                ft_per_mph_following_wind=float(beta[0] * MPH_TO_MPS),
                ft_per_001_density=float(beta[1] * 0.01),
                ci_low=float(np.quantile(slopes, alpha)),
                ci_high=float(np.quantile(slopes, 1.0 - alpha)),
                bootstrap_se=float(np.std(slopes, ddof=1)),
            )
        record["status"] = classify_venue_wind(
            n_plays, n_games, record["ci_low"], record["ci_high"]
        )
        record["wind_check_applies"] = wind_check_applies(record["status"])
        records.append(record)

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    z = NormalDist().inv_cdf(1.0 - alpha / len(table))
    table["bonferroni_ci_low"] = table["ft_per_mph_following_wind"] - z * table["bootstrap_se"]
    table["bonferroni_ci_high"] = table["ft_per_mph_following_wind"] + z * table["bootstrap_se"]
    table["bonferroni_status"] = [
        classify_venue_wind(int(n_plays), int(n_games), float(lo), float(hi))
        for n_plays, n_games, lo, hi in zip(
            table["n_plays"],
            table["n_games"],
            table["bonferroni_ci_low"],
            table["bonferroni_ci_high"],
            strict=True,
        )
    ]
    return table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    df = pd.read_parquet(args.input, columns=list(_REQUIRED_COLUMNS))
    try:
        table = run_venue_wind_check(df)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / OUTPUT_FILENAME
    table.to_csv(out_path, index=False)
    logger.info("wrote %s (%d venues)", out_path, len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
