"""Contact Luck v1.1: train the frozen Version 1.0 component models on
`TRAIN_SEASONS` (2021-2023, UNCHANGED) and score isolated 2026 data.

Mirrors `evaluation.run_v1_final_evaluation.train_and_score_2025` exactly --
same model choices, same training seasons, same order of operations, same
frozen `run_*_model_selection` winner-selection procedures -- targeting 2026
instead of 2025. Duplicates the small `_apply_eligibility_pipeline` glue
(6 lines) rather than importing it from `run_v1_final_evaluation`, matching
this codebase's established convention of keeping each season-hardcoded
runner's glue self-contained (see that module's own docstring for why it
duplicates `run_season_aggregation`'s glue rather than extending it).

Never refits, recalibrates, or tunes anything based on what the 2026 numbers
look like -- every model choice, feature set, eligibility rule, and
qualification threshold is read from the existing frozen code, exactly as
Version 1.0 did for 2025.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prospective_config import PROSPECTIVE_SEASON
from run_v1_final_evaluation import _read_existing_gate_status

from mlb_luck_score.config import TABLES_DIR, TRAIN_SEASONS, assert_seasons_allowed
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
from mlb_luck_score.models.compare_near_wall_models import (
    get_near_wall_rows,
    run_near_wall_model_selection,
)
from mlb_luck_score.models.train_advancement_model import TrainedAdvancementModel
from mlb_luck_score.models.train_contact_model import TrainedModel, train_model
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    train_opportunity_model,
)
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
    assign_qualification_status,
    summarize_qualification_counts,
)
from mlb_luck_score.scoring.run_season_aggregation import SeasonAggregationArtifacts

logger = logging.getLogger(__name__)


class ProspectiveScoringError(ValueError):
    """Raised when a Version 1.1 scoring precondition is violated."""


def _apply_eligibility_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)
    return df


@dataclass
class TrainedComponents:
    """Every trained model object this module produces, exposed so callers
    can feed them to `evaluation.v1_system_evaluation`'s generic per
    -component checks without retraining anything.
    """

    contact_trained: TrainedModel
    outfield_trained: TrainedOpportunityModel
    infield_trained: TrainedOpportunityModel
    advancement_trained: TrainedAdvancementModel
    advancement_contact_trained: TrainedModel
    near_wall_trained: TrainedOpportunityModel
    infield_winner: str
    advancement_winner: str
    near_wall_winner: str
    development_df: pd.DataFrame


def train_and_score_2026(
    development_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    *,
    n_bootstrap_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    threshold_set: str = "primary",
    gate_status_dir: Path = TABLES_DIR,
) -> tuple[SeasonAggregationArtifacts, TrainedComponents]:
    """Train the four frozen component models on `TRAIN_SEASONS` (2021-2023,
    UNCHANGED), then score the isolated 2026 dataset.

    Raises:
        ProspectiveScoringError: if `development_df` contains any season
            outside `TRAIN_SEASONS`, or `scoring_df` contains anything other
            than `PROSPECTIVE_SEASON` -- this function must never silently
            train on, or score, the wrong season.
    """
    assert_seasons_allowed(TRAIN_SEASONS, allow_final_evaluation=False)

    dev_seasons_observed = sorted(int(s) for s in development_df["season"].dropna().unique())
    outside_train = [s for s in dev_seasons_observed if s not in TRAIN_SEASONS]
    if outside_train:
        raise ProspectiveScoringError(
            f"Training data contains season(s) outside TRAIN_SEASONS {TRAIN_SEASONS}: "
            f"{outside_train}"
        )

    scoring_seasons_observed = sorted(int(s) for s in scoring_df["season"].dropna().unique())
    if scoring_seasons_observed != [PROSPECTIVE_SEASON]:
        raise ProspectiveScoringError(
            f"Scoring dataset must contain ONLY season {PROSPECTIVE_SEASON}; observed "
            f"{scoring_seasons_observed}"
        )

    dev_df = _apply_eligibility_pipeline(development_df)
    score_df = _apply_eligibility_pipeline(scoring_df)

    contact_train_df = dev_df[dev_df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_train_df, class_weight=None)

    outfield_train_df = dev_df[dev_df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)

    infield_elig = dev_df[dev_df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, _infield_metrics, infield_trained_by_candidate = run_infield_model_selection(
        infield_elig
    )
    infield_trained = infield_trained_by_candidate[infield_winner]

    advancement_contact_trained = fit_advancement_contact_model(dev_df)
    advancement_elig = get_advancement_rows(dev_df, advancement_contact_trained)
    advancement_winner, _advancement_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]

    near_wall_dev = get_near_wall_rows(dev_df)
    near_wall_winner, _near_wall_metrics, near_wall_trained_by_candidate = (
        run_near_wall_model_selection(near_wall_dev)
    )
    near_wall_trained = near_wall_trained_by_candidate[near_wall_winner]

    prob_cols = list(ADVANCEMENT_CONTACT_PROBABILITY_FEATURES)
    for col in prob_cols:
        score_df[col] = np.nan
    score_advancement_elig = get_advancement_rows(score_df, advancement_contact_trained)
    score_df.loc[score_advancement_elig.index, prob_cols] = score_advancement_elig[prob_cols]

    outfield_status_raw = _read_existing_gate_status(
        gate_status_dir / "opportunity_model_comparison_detail.json",
        status_key=(
            "validation_summary",
            "per_candidate",
            "measured_contact_only_v07",
            "passes_basic_validation",
        ),
    )
    infield_status_raw = _read_existing_gate_status(
        gate_status_dir / "infield_opportunity_detail.json",
        status_key=("gate_summary", "overall_status"),
    )
    advancement_status_raw = _read_existing_gate_status(
        gate_status_dir / "advancement_detail.json",
        status_key=("gate_summary", "overall_status"),
    )

    ledger = build_attribution_ledger(
        score_df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status=outfield_status_raw,
        infield_confidence_status=infield_status_raw,
        advancement_confidence_status=advancement_status_raw,
    )
    confidence = build_play_level_confidence(
        score_df,
        ledger,
        outfield_model_version="measured_contact_only_v07",
        infield_model_version=infield_winner,
        advancement_model_version=advancement_winner,
        outfield_model_status=outfield_status_raw,
        infield_model_status=infield_status_raw,
        advancement_model_status=advancement_status_raw,
    )
    summary = aggregate_to_batter_season(score_df, ledger, confidence)

    season_identity_holds = verify_season_identity(summary)
    if not bool(season_identity_holds.all()):
        bad = int((~season_identity_holds).sum())
        raise ProspectiveScoringError(
            f"Season-level accounting identity failed for {bad} of {len(summary)} rows"
        )

    bootstrap = bootstrap_batter_season_intervals(
        score_df, ledger, n_reps=n_bootstrap_reps, seed=bootstrap_seed
    )
    player_season = summary.merge(bootstrap, on=["batter", "season"], how="left")
    player_season["qualification_status"] = assign_qualification_status(
        player_season, thresholds=threshold_set
    ).to_numpy()

    design = BootstrapDesign(n_reps=n_bootstrap_reps, seed=bootstrap_seed)
    report: dict[str, Any] = {
        "seasons": {
            "train_seasons": list(TRAIN_SEASONS),
            "scoring_seasons": [PROSPECTIVE_SEASON],
            "observed_in_training_input": dev_seasons_observed,
            "observed_in_scoring_input": scoring_seasons_observed,
        },
        "model_selection_winners": {
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
            "near_wall_specialist": (
                f"{near_wall_winner} (informational only; Version 0.7 stays provisional/frozen)"
            ),
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
        "total_scored_plays": int(len(score_df)),
    }

    artifacts = SeasonAggregationArtifacts(
        scoring_df=score_df,
        ledger=ledger,
        confidence=confidence,
        player_season=player_season,
        report=report,
    )
    trained = TrainedComponents(
        contact_trained=contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        advancement_contact_trained=advancement_contact_trained,
        near_wall_trained=near_wall_trained,
        infield_winner=infield_winner,
        advancement_winner=advancement_winner,
        near_wall_winner=near_wall_winner,
        development_df=dev_df,
    )
    return artifacts, trained
