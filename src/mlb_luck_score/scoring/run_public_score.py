"""Contact Luck v0.12 Phase 7-8: the public score CLI and development review tables.

FREEZES Versions 0.2-0.11 completely: this script calls `mlb_luck_score.
scoring.run_season_aggregation.build_player_season_report` (Version 0.11,
unchanged) to get the frozen player-season artifacts, then layers the
Version 0.12 public contract on top -- `mlb_luck_score.scoring.
public_score_table.build_public_score_table` (schema population), `mlb_luck_
score.scoring.leaderboard.assign_official_ranks` (ranking policy). No step in
this script refits, recalibrates, or retunes any model or threshold, and
Phase 8's review tables are explicitly DEVELOPMENT-ONLY diagnostics, not an
invitation to adjust qualification thresholds or models based on how the
2024 leaderboard looks (the task's own explicit instruction).

## 2025 protection

This script never widens the season set `run_season_aggregation` already
refuses to exceed -- there is no `--allow-final-evaluation`-style flag here,
and none should ever be added without a genuine, separately-named, one-time
final-evaluation mechanism.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_luck_score.config import TABLES_DIR
from mlb_luck_score.scoring.leaderboard import (
    TIE_BREAK_COLUMN,
    assign_official_ranks,
    least_favorable_leaderboard,
    most_favorable_leaderboard,
)
from mlb_luck_score.scoring.public_score_schema import PUBLIC_SCORE_SCHEMA_VERSION
from mlb_luck_score.scoring.public_score_table import build_public_score_table
from mlb_luck_score.scoring.qualification import QUALIFICATION_THRESHOLD_SETS, STATUS_QUALIFIED
from mlb_luck_score.scoring.run_season_aggregation import (
    SeasonAggregationArtifacts,
    build_player_season_report,
)

logger = logging.getLogger(__name__)

#: Below this eligible-play count, a row is "single-digit-sample" -- Phase 8
#: explicitly asks this script to confirm such rows never carry an official
#: rank (they never should, given qualification's much higher volume bar,
#: but this is checked explicitly rather than assumed).
SINGLE_DIGIT_SAMPLE_THRESHOLD = 10

REVIEW_TOP_N = 25


class RunPublicScoreError(ValueError):
    """Raised when this script's own preconditions are not met."""


def _flatten_for_tabular_export(df: pd.DataFrame) -> pd.DataFrame:
    """CSV/Parquet cannot hold native Python dict cells -- JSON-stringify the
    two dict-valued columns for those exports ONLY; the JSON outputs keep the
    richly-typed structure.
    """
    out = df.copy()
    for col in ("component_status_reason_codes", "model_version"):
        if col in out.columns:
            out[col] = out[col].apply(json.dumps)
    return out


def assert_no_single_digit_sample_players_ranked(public_score_table: pd.DataFrame) -> None:
    """Fail loudly if any row with `eligible_batted_balls < SINGLE_DIGIT_
    SAMPLE_THRESHOLD` carries an official rank -- Phase 8's explicit check.
    """
    ranked = (
        public_score_table["official_rank_favorable"].notna()
        | public_score_table["official_rank_unfavorable"].notna()
    )
    single_digit = public_score_table["eligible_batted_balls"] < SINGLE_DIGIT_SAMPLE_THRESHOLD
    bad = ranked & single_digit
    if bad.any():
        raise RunPublicScoreError(
            f"{int(bad.sum())} single-digit-sample row(s) carry an official rank -- this "
            "should be structurally impossible given qualification's volume threshold"
        )


def build_score_card(
    public_score_table: pd.DataFrame, artifacts: SeasonAggregationArtifacts
) -> dict[str, Any]:
    """A compact, machine-readable model/score card -- NOT the full table."""
    qualified = public_score_table["qualification_status"] == STATUS_QUALIFIED
    qualification_counts = public_score_table["qualification_status"].value_counts().to_dict()
    return {
        "score_version": PUBLIC_SCORE_SCHEMA_VERSION,
        "model_version": public_score_table["model_version"].iloc[0]
        if len(public_score_table)
        else {},
        "generated_at": public_score_table["generated_at"].iloc[0]
        if len(public_score_table)
        else None,
        "data_through_date": (
            sorted(public_score_table["data_through_date"].dropna().unique().tolist())
        ),
        "seasons": sorted(int(s) for s in public_score_table["season"].unique().tolist()),
        "row_count": int(len(public_score_table)),
        "qualified_count": int(qualified.sum()),
        "qualification_counts": {str(k): int(v) for k, v in qualification_counts.items()},
        "ranking_method": "competition_ranking_min_with_batter_id_tiebreak",
        "official_metric": "contact_luck_runs_per_100",
        "interval_method": "game_pk_clustered_percentile_bootstrap_95pct",
        "season_identity_holds_for_every_row": artifacts.report.get(
            "season_identity_holds_for_every_row"
        ),
    }


def build_review_tables(
    public_score_table: pd.DataFrame, artifacts: SeasonAggregationArtifacts
) -> dict[str, Any]:
    """Phase 8: development-only 2024 review tables. NOT permission to retune
    qualification thresholds or models based on how they look.
    """
    qualified = public_score_table[public_score_table["qualification_status"] == STATUS_QUALIFIED]
    non_qualified = public_score_table[
        public_score_table["qualification_status"] != STATUS_QUALIFIED
    ]

    widest = qualified.assign(
        _width=qualified["upper_95_interval"] - qualified["lower_95_interval"]
    ).sort_values(["_width", TIE_BREAK_COLUMN], ascending=[False, True])
    narrowest = qualified.assign(
        _width=qualified["upper_95_interval"] - qualified["lower_95_interval"]
    ).sort_values(["_width", TIE_BREAK_COLUMN], ascending=[True, True])

    largest_provisional_share = public_score_table[
        public_score_table["eligible_batted_balls"] > 0
    ].sort_values(
        ["share_of_value_from_provisional_components", TIE_BREAK_COLUMN], ascending=[False, True]
    )

    crosses_zero = qualified[qualified["interval_interpretation"] == "overlaps_zero"]

    non_qualified_extreme = non_qualified.dropna(subset=["contact_luck_runs_per_100"]).sort_values(
        "contact_luck_runs_per_100", key=lambda s: s.abs(), ascending=False
    )

    provisional_comparison = artifacts.player_season[
        [
            "batter",
            "season",
            "observed_minus_expected_per_100",
            "observed_minus_expected_excluding_provisional_per_100",
        ]
    ].rename(columns={"batter": "batter_id"})
    comparison = qualified[["batter_id", "season", "contact_luck_runs_per_100"]].merge(
        provisional_comparison, on=["batter_id", "season"], how="left"
    )
    comparison["rank_official"] = comparison["contact_luck_runs_per_100"].rank(
        method="min", ascending=False
    )
    comparison["rank_excluding_provisional"] = comparison[
        "observed_minus_expected_excluding_provisional_per_100"
    ].rank(method="min", ascending=False)
    comparison["rank_delta"] = (
        comparison["rank_excluding_provisional"] - comparison["rank_official"]
    )
    comparison = comparison.sort_values(TIE_BREAK_COLUMN)

    review_display_cols = [
        "batter_id",
        "season",
        "contact_luck_runs_per_100",
        "lower_95_interval",
        "upper_95_interval",
        "interval_interpretation",
        "eligible_batted_balls",
        "games",
        "share_of_value_from_provisional_components",
        "qualification_status",
    ]

    def _records(frame: pd.DataFrame, n: int | None = None) -> list[dict[str, Any]]:
        cols = [c for c in review_display_cols if c in frame.columns]
        result = frame[cols]
        if n is not None:
            result = result.head(n)
        return [
            {str(k): v for k, v in record.items()} for record in result.to_dict(orient="records")
        ]

    return {
        "top_qualified_favorable": _records(
            qualified.sort_values(
                ["contact_luck_runs_per_100", TIE_BREAK_COLUMN], ascending=[False, True]
            ),
            REVIEW_TOP_N,
        ),
        "bottom_qualified_unfavorable": _records(
            qualified.sort_values(
                ["contact_luck_runs_per_100", TIE_BREAK_COLUMN], ascending=[True, True]
            ),
            REVIEW_TOP_N,
        ),
        "widest_qualified_intervals": _records(widest, REVIEW_TOP_N),
        "narrowest_qualified_intervals": _records(narrowest, REVIEW_TOP_N),
        "largest_provisional_component_shares": _records(largest_provisional_share, REVIEW_TOP_N),
        "qualified_intervals_crossing_zero": {
            "count": int(len(crosses_zero)),
            "fraction_of_qualified": (
                float(len(crosses_zero) / len(qualified)) if len(qualified) else None
            ),
            "sample": _records(crosses_zero, REVIEW_TOP_N),
        },
        "non_qualified_extreme_point_estimates": _records(non_qualified_extreme, REVIEW_TOP_N),
        "ranking_with_vs_without_provisional_components": comparison.head(len(comparison)).to_dict(
            orient="records"
        ),
    }


def run(artifacts: SeasonAggregationArtifacts) -> dict[str, Any]:
    """Build everything this script writes to disk, returned as one dict of
    DataFrames/dicts for `main` to serialize.
    """
    table = build_public_score_table(artifacts)
    table = assign_official_ranks(table)
    assert_no_single_digit_sample_players_ranked(table)

    favorable = most_favorable_leaderboard(table)
    unfavorable = least_favorable_leaderboard(table)
    score_card = build_score_card(table, artifacts)
    review_tables = build_review_tables(table, artifacts)

    return {
        "public_score_table": table,
        "favorable_leaderboard": favorable,
        "unfavorable_leaderboard": unfavorable,
        "score_card": score_card,
        "review_tables": review_tables,
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
    parser.add_argument(
        "--threshold-set", choices=list(QUALIFICATION_THRESHOLD_SETS), default="primary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    artifacts = build_player_season_report(
        args.input, output_dir=args.output_dir, threshold_set=args.threshold_set
    )
    results = run(artifacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = results["public_score_table"]
    tabular = _flatten_for_tabular_export(table)
    tabular.to_csv(args.output_dir / "public_score_v012.csv", index=False)
    tabular.to_parquet(args.output_dir / "public_score_v012.parquet", index=False)
    table.to_json(args.output_dir / "public_score_v012.json", orient="records", indent=2)

    results["favorable_leaderboard"].to_json(
        args.output_dir / "public_score_v012_favorable_leaderboard.json", orient="records", indent=2
    )
    results["unfavorable_leaderboard"].to_json(
        args.output_dir / "public_score_v012_unfavorable_leaderboard.json",
        orient="records",
        indent=2,
    )
    (args.output_dir / "public_score_v012_scorecard.json").write_text(
        json.dumps(results["score_card"], indent=2, default=str)
    )
    (args.output_dir / "public_score_v012_review_tables.json").write_text(
        json.dumps(results["review_tables"], indent=2, default=str)
    )

    logger.info("score_card=%s", json.dumps(results["score_card"], default=str))
    logger.info("Saved public score outputs to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
