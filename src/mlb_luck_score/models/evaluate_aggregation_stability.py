"""Contact Luck v0.11 Phase 6: aggregation stability and sensitivity analysis.

Descriptive DEVELOPMENT analysis of `mlb_luck_score.scoring.
run_season_aggregation`'s real 2021-2024 output -- NOT an untouched
validation exercise. This module reads the SAME scored plays `run_season_
aggregation.build_player_season_report` already produced (no retraining,
no re-scoring) and asks purely descriptive reliability/sensitivity
questions:

  - Split-half reliability: does a player's per-100 rate in one half of
    their scored plays correlate with the other half? Two splits are
    reported -- calendar first-half-of-season vs. second-half (by `game_
    date`), and odd-`game_pk` vs. even-`game_pk` (a standard, deterministic
    quasi-random split-half proxy). Both require `min_eligible_each_half`
    eligible plays in BOTH halves for a player to be included, to avoid a
    correlation dominated by single-digit-sample noise.
  - Interval width vs. sample size: does the game_pk-clustered bootstrap's
    interval narrow as `n_games` grows, as it should for any honest
    sampling-variability estimate?
  - Qualification-threshold sensitivity: how much does the QUALIFIED set
    and the top-N ranking change across `mlb_luck_score.scoring.
    qualification.QUALIFICATION_THRESHOLD_SETS`'s three PREDETERMINED
    candidate sets (chosen and fixed before this analysis ever ran)?
  - Provisional-pathway sensitivity: how much does a player's per-100 rate
    and rank change if provisional/limited-evidence component value is
    EXCLUDED entirely? Reports the largest movers explicitly, not just an
    aggregate correlation, so a reader can see WHICH players are most
    exposed to the provisional/near-wall gap this codebase has documented
    since Version 0.7C.
  - Component covariance at the player-season level.

Every threshold and split rule here was fixed in `mlb_luck_score.scoring.
qualification`/this module's own docstring BEFORE any real player-season
number was examined for this analysis -- see `run_season_aggregation`'s
module docstring for the exact order of operations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_luck_score.config import TABLES_DIR
from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season
from mlb_luck_score.scoring.qualification import (
    QUALIFICATION_THRESHOLD_SETS,
    assign_qualification_status,
    qualified_mask,
)
from mlb_luck_score.scoring.run_attribution_ledger import RunAttributionLedgerError
from mlb_luck_score.scoring.run_season_aggregation import (
    SeasonAggregationArtifacts,
    build_player_season_report,
)

logger = logging.getLogger(__name__)

#: A player-season must have at least this many eligible plays in BOTH
#: halves of a split-half comparison to be included in that comparison --
#: otherwise a handful of plays would dominate the correlation with pure
#: sampling noise. Documented, predetermined (not tuned to the real result).
MIN_ELIGIBLE_EACH_HALF = 20

#: How many top-ranked qualified players to compare across threshold sets /
#: with-vs-without-provisional-value rankings.
TOP_N_FOR_RANKING_COMPARISON = 25

RANKING_METRIC = "observed_minus_expected_per_100"


class AggregationStabilityError(ValueError):
    """Raised when inputs to this module's analyses are invalid."""


def _half_season_masks(scoring_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    game_date = pd.to_datetime(scoring_df["game_date"])
    median_date = game_date.median()
    first_half = game_date <= median_date
    return first_half, ~first_half


def _odd_even_masks(scoring_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    odd = scoring_df["game_pk"].astype("int64") % 2 == 1
    return odd, ~odd


def _aggregate_subset(
    scoring_df: pd.DataFrame, ledger: pd.DataFrame, confidence: pd.DataFrame, mask: pd.Series
) -> pd.DataFrame:
    idx = scoring_df.index[mask]
    subset_confidence = confidence[confidence["event_id"].isin(scoring_df.loc[idx, "event_id"])]
    return aggregate_to_batter_season(scoring_df.loc[idx], ledger.loc[idx], subset_confidence)


def compute_split_half_reliability(
    artifacts: SeasonAggregationArtifacts,
    *,
    split: str = "calendar",
    metric: str = RANKING_METRIC,
    min_eligible_each_half: int = MIN_ELIGIBLE_EACH_HALF,
) -> dict[str, Any]:
    """Pearson/Spearman correlation of `metric` between two halves of each
    player's scored plays -- `split="calendar"` (first vs. second half of
    the season by `game_date`) or `split="odd_even"` (odd vs. even
    `game_pk`).
    """
    if split not in ("calendar", "odd_even"):
        raise AggregationStabilityError(
            f"Unknown split {split!r}; expected 'calendar' or 'odd_even'"
        )

    mask_a, mask_b = (
        _half_season_masks(artifacts.scoring_df)
        if split == "calendar"
        else _odd_even_masks(artifacts.scoring_df)
    )
    summary_a = _aggregate_subset(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence, mask_a
    )
    summary_b = _aggregate_subset(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence, mask_b
    )

    merged = summary_a.merge(summary_b, on=["batter", "season"], suffixes=("_a", "_b"), how="inner")
    merged = merged[
        (merged["eligible_batted_balls_a"] >= min_eligible_each_half)
        & (merged["eligible_batted_balls_b"] >= min_eligible_each_half)
    ]

    n = len(merged)
    if n < 2:
        return {
            "split": split,
            "metric": metric,
            "min_eligible_each_half": min_eligible_each_half,
            "n_players_compared": n,
            "pearson_r": None,
            "spearman_r": None,
        }

    col_a, col_b = f"{metric}_a", f"{metric}_b"
    pearson_r = float(merged[col_a].corr(merged[col_b], method="pearson"))
    spearman_r = float(merged[col_a].corr(merged[col_b], method="spearman"))
    return {
        "split": split,
        "metric": metric,
        "min_eligible_each_half": min_eligible_each_half,
        "n_players_compared": n,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }


def compute_interval_width_by_sample_size(
    player_season: pd.DataFrame, *, metric: str = RANKING_METRIC, n_bins: int = 5
) -> list[dict[str, Any]]:
    """Median bootstrap interval width for `metric`, binned by `n_games`
    quantile -- should DECREASE as `n_games` increases for a well-behaved
    sampling-variability estimate.
    """
    ci_low_col, ci_high_col = f"{metric}_ci_low", f"{metric}_ci_high"
    working = player_season[["n_games", ci_low_col, ci_high_col]].dropna()
    if working.empty:
        return []
    working = working.assign(width=working[ci_high_col] - working[ci_low_col])
    try:
        working["bin"] = pd.qcut(working["n_games"], q=n_bins, duplicates="drop")
    except ValueError:
        working["bin"] = "all"

    rows: list[dict[str, Any]] = []
    for bin_label, group in working.groupby("bin", observed=True):
        rows.append(
            {
                "n_games_bin": str(bin_label),
                "n_players": int(len(group)),
                "median_n_games": float(group["n_games"].median()),
                "median_interval_width": float(group["width"].median()),
            }
        )
    rows.sort(key=lambda r: float(r["median_n_games"]))
    return rows


def compute_qualification_threshold_sensitivity(
    player_season: pd.DataFrame,
    *,
    metric: str = RANKING_METRIC,
    top_n: int = TOP_N_FOR_RANKING_COMPARISON,
) -> dict[str, Any]:
    """Compares the QUALIFIED set and top-`top_n` ranking across every
    `QUALIFICATION_THRESHOLD_SETS` entry.
    """
    statuses = {
        name: assign_qualification_status(player_season, thresholds=name)
        for name in QUALIFICATION_THRESHOLD_SETS
    }
    qualified_counts = {name: int(qualified_mask(s).sum()) for name, s in statuses.items()}

    top_n_batters = {}
    for name, status in statuses.items():
        qualified_rows = player_season[qualified_mask(status).to_numpy()]
        ranked = qualified_rows.sort_values(metric, ascending=False).head(top_n)
        top_n_batters[name] = set(zip(ranked["batter"], ranked["season"], strict=True))

    overlap_vs_primary = {
        name: (
            len(top_n_batters[name] & top_n_batters["primary"]) / len(top_n_batters["primary"])
            if top_n_batters.get("primary")
            else None
        )
        for name in top_n_batters
        if name != "primary"
    }

    return {
        "qualified_counts_by_threshold_set": qualified_counts,
        f"top_{top_n}_overlap_fraction_vs_primary": overlap_vs_primary,
    }


def compute_provisional_exclusion_sensitivity(
    player_season: pd.DataFrame,
    *,
    metric: str = RANKING_METRIC,
    top_n: int = TOP_N_FOR_RANKING_COMPARISON,
) -> dict[str, Any]:
    """Compares `metric` (the full per-100 rate) against `mlb_luck_score.
    scoring.aggregate_attribution.aggregate_to_batter_season`'s already
    -computed `observed_minus_expected_excluding_provisional_per_100` --
    the SAME signed total with provisional/limited-evidence defense and
    advancement contributions excluded (see that function's docstring for
    the exact row-level definition) -- and reports the rank correlation plus
    the players whose value changes most.
    """
    working = player_season[["batter", "season", "eligible_batted_balls", metric]].copy()
    working["non_provisional_per_100"] = player_season[
        "observed_minus_expected_excluding_provisional_per_100"
    ]
    working["delta"] = working["non_provisional_per_100"] - working[metric]

    valid = working.dropna(subset=[metric, "non_provisional_per_100"])
    rank_corr = (
        float(valid[metric].corr(valid["non_provisional_per_100"], method="spearman"))
        if len(valid) >= 2
        else None
    )

    biggest_movers = (
        valid.assign(abs_delta=valid["delta"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(top_n)[
            [
                "batter",
                "season",
                "eligible_batted_balls",
                metric,
                "non_provisional_per_100",
                "delta",
            ]
        ]
        .to_dict(orient="records")
    )

    return {
        "metric": metric,
        "rank_correlation_full_vs_non_provisional": rank_corr,
        "n_players_compared": int(len(valid)),
        f"top_{top_n}_biggest_movers_when_provisional_excluded": biggest_movers,
    }


def compute_component_covariance(player_season: pd.DataFrame) -> dict[str, Any]:
    """Player-season-level covariance/correlation among the four additive
    components -- descriptive only, no causal claim.
    """
    cols = [
        "total_contact_component_runs",
        "total_unexplained_residual_component_runs",
        "total_defensive_execution_component_runs",
        "total_advancement_component_runs",
    ]
    working = player_season[cols].dropna()
    if len(working) < 2:
        return {"n_players": len(working), "covariance": None, "correlation": None}
    return {
        "n_players": int(len(working)),
        "covariance": working.cov().round(6).to_dict(),
        "correlation": working.corr().round(4).to_dict(),
    }


def build_stability_report(artifacts: SeasonAggregationArtifacts) -> dict[str, Any]:
    player_season = artifacts.player_season
    return {
        "split_half_reliability": {
            "calendar": compute_split_half_reliability(artifacts, split="calendar"),
            "odd_even_game_pk": compute_split_half_reliability(artifacts, split="odd_even"),
        },
        "interval_width_by_sample_size": compute_interval_width_by_sample_size(player_season),
        "qualification_threshold_sensitivity": compute_qualification_threshold_sensitivity(
            player_season
        ),
        "provisional_exclusion_sensitivity": compute_provisional_exclusion_sensitivity(
            player_season
        ),
        "component_covariance": compute_component_covariance(player_season),
        "note": (
            "Descriptive DEVELOPMENT analysis over 2021-2024 only -- not an untouched "
            "validation exercise. See module docstring."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_development_data_with_sprint_speed.parquet"),
        help="Cleaned development data joined with venue+geometry+sprint speed",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    artifacts = build_player_season_report(args.input, output_dir=args.output_dir)
    if not artifacts.report["season_identity_holds_for_every_row"]:
        raise RunAttributionLedgerError(  # pragma: no cover -- defensive, should never fire
            "Season identity failed upstream; refusing to build a stability report on top of it"
        )

    report = build_stability_report(artifacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "aggregation_stability_v011_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info(
        "split_half_calendar_pearson_r=%s",
        report["split_half_reliability"]["calendar"]["pearson_r"],
    )
    logger.info(
        "split_half_odd_even_pearson_r=%s",
        report["split_half_reliability"]["odd_even_game_pk"]["pearson_r"],
    )
    logger.info("Saved report to %s", report_path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
