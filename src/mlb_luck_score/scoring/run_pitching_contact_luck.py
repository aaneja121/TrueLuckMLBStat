"""Pitching Contact Luck: end-to-end pitcher-season CLI (research spike).

Trains nothing of its own. It calls `mlb_luck_score.scoring.
run_season_aggregation.build_player_season_report` UNCHANGED -- which fits
the four component models on `TRAIN_SEASONS` exactly as the frozen batter
pipeline does -- and then re-aggregates the `scoring_df`/`ledger`/
`confidence` that runner already exposes on its `SeasonAggregationArtifacts`.
That dataclass exists precisely so a second re-aggregation can be built
"from the SAME scored plays without retraining a second copy of the four
component models"; this is a second consumer of that contract, alongside
`mlb_luck_score.models.evaluate_aggregation_stability`.

Consequences worth being explicit about:

- **No model changes.** Nothing here retrains, recalibrates, reselects a
  feature, or re-chooses a threshold. The pitcher numbers come from the same
  frozen predictions the batter numbers come from.
- **2025 protection is inherited, not re-implemented.** `build_player_season_
  report` calls `assert_seasons_allowed` and independently re-checks every
  season present in the input. This script exposes NO flag that could reach
  a final-test season, exactly like the batter runner.
- **Every run self-checks.** `verify_batter_side_reproduction` re-runs the
  pitcher code path on the `batter` key and confirms it reproduces the frozen
  batter table. `pitcher_values_trustworthy` in the report is gated on it.

Version 0.13 research spike -- provisional. The threshold derivation and the
starter-only scope are documented in `mlb_luck_score.scoring.
pitching_contact_luck`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mlb_luck_score.config import TABLES_DIR
from mlb_luck_score.scoring.aggregation_uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
)
from mlb_luck_score.scoring.pitching_contact_luck import (
    DEFAULT_PITCHER_THRESHOLD_SET,
    PITCHER_QUALIFICATION_THRESHOLD_SETS,
    build_pitcher_report,
    build_pitcher_season_table,
    verify_batter_side_reproduction,
)
from mlb_luck_score.scoring.run_season_aggregation import build_player_season_report

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_development_data_with_sprint_speed.parquet"),
        help="Cleaned development data joined with venue+geometry+sprint speed",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--threshold-set",
        choices=list(PITCHER_QUALIFICATION_THRESHOLD_SETS),
        default=DEFAULT_PITCHER_THRESHOLD_SET,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    # Trains the four component models ONCE, on TRAIN_SEASONS, via the
    # frozen batter runner. Everything below reuses its scored plays.
    artifacts = build_player_season_report(
        args.input,
        output_dir=args.output_dir,
        n_bootstrap_reps=args.n_bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    logger.info(
        "Reusing %d scored plays from the frozen batter pipeline", len(artifacts.scoring_df)
    )

    reproduction = verify_batter_side_reproduction(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence
    )
    logger.info("batter_side_reproduction=%s", reproduction)

    table = build_pitcher_season_table(
        artifacts.scoring_df,
        artifacts.ledger,
        artifacts.confidence,
        n_bootstrap_reps=args.n_bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        threshold_set=args.threshold_set,
    )
    logger.info("Aggregated to %d pitcher-season rows", len(table))

    seasons = tuple(sorted(int(s) for s in table["season"].unique()))
    report = build_pitcher_report(
        table,
        reproduction,
        threshold_set_label=str(table.attrs["threshold_set"]),
        seasons=seasons,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "pitcher_season_pitching_contact_luck_v013.json"
    table.to_json(table_path, orient="records", indent=2)
    report_path = args.output_dir / "pitching_contact_luck_v013_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("qualification_counts=%s", report["qualification_counts"])
    logger.info("pitcher_values_trustworthy=%s", report["pitcher_values_trustworthy"])
    logger.info("Saved pitcher-season table to %s", table_path)
    logger.info("Saved report to %s", report_path)

    if not report["pitcher_values_trustworthy"]:
        logger.error(
            "Batter-side reproduction FAILED (%s); pitcher values from this run are not "
            "trustworthy and must not be used until the difference is explained.",
            reproduction,
        )
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
