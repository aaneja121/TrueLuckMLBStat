"""Contact Luck v0.11 Phase 7: end-to-end player-season aggregation CLI.

Freezes Version 0.2-0.10 completely: this script trains the SAME four
component models, on the SAME season splits, via the SAME real winner
-selection procedures, as `mlb_luck_score.scoring.run_attribution_ledger`
(Version 0.10's own real-data runner) -- it duplicates that glue rather than
importing or modifying that script's internals, so nothing about how those
models are fit changes by one line. See `run_attribution_ledger`'s module
docstring for the exact model choices, their real (already-computed)
calibration statuses, and why the near-wall specialist is not used.

Everything from `mlb_luck_score.scoring.component_confidence` onward is NEW
(Version 0.11): confidence records, additive season aggregation, game
-clustered bootstrap intervals, and qualification status. No step in this
new part ever refits, recalibrates, or tunes a model -- it only reads the
frozen ledger's already-computed predictions.

## Order of operations (why this matters for Phase 5's thresholds)

`mlb_luck_score.scoring.qualification`'s threshold sets were written and
fixed BEFORE this script was ever run against real data. Running this
script is the FIRST time any real 2024 player-season leaderboard from this
pipeline is produced -- the thresholds could not have been tuned to it.

## 2025 protection

Identical to `run_attribution_ledger`: `assert_seasons_allowed` on the exact
train/validation season set, plus an independent re-check of every season
value actually present in the loaded input file against `SCRIPT_SEASONS`.
No `--allow-final-evaluation` flag exists here, on purpose.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_luck_score.config import (
    TABLES_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.eligibility import (
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.features.build_contact_features import (
    ADVANCEMENT_CONTACT_PROBABILITY_FEATURES,
    add_advancement_features,
    add_opportunity_features_by_domain,
)
from mlb_luck_score.models.compare_advancement_models import (
    fit_advancement_contact_model,
    get_advancement_rows,
    run_advancement_model_selection,
)
from mlb_luck_score.models.compare_infield_opportunity import run_infield_model_selection
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring.aggregate_attribution import (
    aggregate_to_batter_season,
    verify_season_identity,
)
from mlb_luck_score.scoring.aggregation_uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    BootstrapDesign,
    bootstrap_batter_season_intervals,
)
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger
from mlb_luck_score.scoring.component_confidence import build_play_level_confidence
from mlb_luck_score.scoring.qualification import (
    QUALIFICATION_THRESHOLD_SETS,
    assign_qualification_status,
    summarize_qualification_counts,
)
from mlb_luck_score.scoring.run_attribution_ledger import (
    _read_existing_gate_status,
    _validate_no_final_test_seasons,
)

logger = logging.getLogger(__name__)


class RunSeasonAggregationError(ValueError):
    """Raised when this script's own preconditions are not met."""


@dataclass
class SeasonAggregationArtifacts:
    """Every intermediate object `build_player_season_report` produces --
    exposed (not just the final player-season table) so `mlb_luck_score.
    models.evaluate_aggregation_stability` (Phase 6) can build split-half/
    odd-even re-aggregations from the SAME scored plays without retraining
    a second copy of the four component models.
    """

    scoring_df: pd.DataFrame
    ledger: pd.DataFrame
    confidence: pd.DataFrame
    player_season: pd.DataFrame
    report: dict[str, Any]


def build_player_season_report(
    input_path: Path,
    *,
    output_dir: Path,
    train_seasons: tuple[int, ...] = TRAIN_SEASONS,
    validation_seasons: tuple[int, ...] = VALIDATION_SEASONS,
    n_bootstrap_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    threshold_set: str = "primary",
) -> SeasonAggregationArtifacts:
    """Train every component model (identical to `run_attribution_ledger`),
    build the Version 0.10 ledger, then layer Version 0.11's confidence,
    aggregation, bootstrap, and qualification machinery on top.

    Returns:
        A `SeasonAggregationArtifacts`. `.player_season` is the
        machine-readable output described in Phase 7 (one row per (batter,
        season), NEVER filtered by qualification status -- see `mlb_luck_
        score.scoring.qualification`'s docstring); `.report` is the Phase 9
        summary dict `main` writes to disk.
    """
    assert_seasons_allowed(tuple(train_seasons) + tuple(validation_seasons))

    df = pd.read_parquet(input_path)
    observed_seasons = _validate_no_final_test_seasons(df)
    logger.info(
        "Loaded %d rows from %s; seasons observed: %s", len(df), input_path, observed_seasons
    )

    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)

    contact_train_df = df[
        df["eligible_for_training"].fillna(False) & df["season"].isin(train_seasons)
    ]
    contact_trained = train_model(contact_train_df, class_weight=None)
    logger.info(
        "Contact model (baseline_v02) trained on %d rows, seasons %s",
        len(contact_train_df),
        train_seasons,
    )

    outfield_train_df = df[
        df["outfield_opportunity_eligible"].astype(bool) & df["season"].isin(train_seasons)
    ]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)
    logger.info("Outfield opportunity model trained on %d rows", len(outfield_train_df))

    infield_elig = df[df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, _infield_selection_metrics, infield_trained_by_candidate = (
        run_infield_model_selection(infield_elig)
    )
    infield_trained = infield_trained_by_candidate[infield_winner]
    logger.info("Infield opportunity model selection winner: %s", infield_winner)

    advancement_contact_trained = fit_advancement_contact_model(df)
    advancement_elig = get_advancement_rows(df, advancement_contact_trained)
    advancement_winner, _advancement_selection_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]
    logger.info("Advancement model selection winner: %s", advancement_winner)

    prob_cols = list(ADVANCEMENT_CONTACT_PROBABILITY_FEATURES)
    for col in prob_cols:
        df[col] = np.nan
    df.loc[advancement_elig.index, prob_cols] = advancement_elig[prob_cols]

    outfield_status_raw = _read_existing_gate_status(
        output_dir / "opportunity_model_comparison_detail.json",
        status_key=(
            "validation_summary",
            "per_candidate",
            "measured_contact_only_v07",
            "passes_basic_validation",
        ),
    )
    infield_status_raw = _read_existing_gate_status(
        output_dir / "infield_opportunity_detail.json",
        status_key=("gate_summary", "overall_status"),
    )
    advancement_status_raw = _read_existing_gate_status(
        output_dir / "advancement_detail.json",
        status_key=("gate_summary", "overall_status"),
    )

    scoring_df = df[df["season"].isin(validation_seasons)].copy()
    logger.info("Scoring %d rows from seasons %s", len(scoring_df), validation_seasons)

    ledger = build_attribution_ledger(
        scoring_df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status=outfield_status_raw,
        infield_confidence_status=infield_status_raw,
        advancement_confidence_status=advancement_status_raw,
    )

    confidence = build_play_level_confidence(
        scoring_df,
        ledger,
        outfield_model_version="measured_contact_only_v07",
        infield_model_version=infield_winner,
        advancement_model_version=advancement_winner,
        outfield_model_status=outfield_status_raw,
        infield_model_status=infield_status_raw,
        advancement_model_status=advancement_status_raw,
    )
    logger.info("Built %d play-level confidence records", len(confidence))

    summary = aggregate_to_batter_season(scoring_df, ledger, confidence)
    logger.info("Aggregated to %d batter-season rows", len(summary))

    season_identity_holds = verify_season_identity(summary)
    if not bool(season_identity_holds.all()):
        bad = int((~season_identity_holds).sum())
        raise RunSeasonAggregationError(
            f"Season-level accounting identity failed for {bad} of {len(summary)} rows"
        )

    bootstrap = bootstrap_batter_season_intervals(
        scoring_df, ledger, n_reps=n_bootstrap_reps, seed=bootstrap_seed
    )
    logger.info(
        "Bootstrapped %d batter-season intervals (n_reps=%d)", len(bootstrap), n_bootstrap_reps
    )

    player_season = summary.merge(bootstrap, on=["batter", "season"], how="left")
    player_season["qualification_status"] = assign_qualification_status(
        player_season, thresholds=threshold_set
    ).to_numpy()

    design = BootstrapDesign(n_reps=n_bootstrap_reps, seed=bootstrap_seed)
    report: dict[str, Any] = {
        "seasons": {
            "train_seasons": list(train_seasons),
            "validation_seasons": list(validation_seasons),
            "observed_in_input": observed_seasons,
            "final_test_season_touched": False,
        },
        "model_selection_winners": {
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
        },
        "component_model_status": {
            "outfield": outfield_status_raw,
            "infield": infield_status_raw,
            "advancement": advancement_status_raw,
        },
        "bootstrap_design": {
            "method": design.method,
            "resampling_unit": design.resampling_unit,
            "n_reps": design.n_reps,
            "seed": design.seed,
            "alpha": design.alpha,
            "refits_models": design.refits_models,
        },
        "qualification_threshold_set": threshold_set,
        "qualification_counts": summarize_qualification_counts(
            player_season["qualification_status"]
        ),
        "player_season_row_count": int(len(player_season)),
        "season_identity_holds_for_every_row": bool(season_identity_holds.all()),
        "total_scored_plays": int(len(scoring_df)),
    }
    return SeasonAggregationArtifacts(
        scoring_df=scoring_df,
        ledger=ledger,
        confidence=confidence,
        player_season=player_season,
        report=report,
    )


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
        "--threshold-set", choices=list(QUALIFICATION_THRESHOLD_SETS), default="primary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    artifacts = build_player_season_report(
        args.input,
        output_dir=args.output_dir,
        n_bootstrap_reps=args.n_bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        threshold_set=args.threshold_set,
    )
    player_season, report = artifacts.player_season, artifacts.report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_season_path = args.output_dir / "player_season_attribution_v011.json"
    player_season.to_json(player_season_path, orient="records", indent=2)

    report_path = args.output_dir / "season_aggregation_v011_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("qualification_counts=%s", report["qualification_counts"])
    logger.info(
        "season_identity_holds_for_every_row=%s", report["season_identity_holds_for_every_row"]
    )
    logger.info("Saved player-season table to %s", player_season_path)
    logger.info("Saved report to %s", report_path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
