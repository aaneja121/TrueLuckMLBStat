"""Contact Luck v0.6: does the defense's STARTING ALIGNMENT (not execution)
improve out-of-sample probability quality beyond the currently-selected
production baseline?

This estimates whether the defense's pre-pitch positioning changed the
expected outcome of a batted ball -- deliberately kept separate from
defensive EXECUTION (reaction, route, pickup, transfer, throw), which public
data cannot observe at all. With only Statcast's `if_fielding_alignment` /
`of_fielding_alignment` labels available, this is scoped as
ALIGNMENT-AWARE POSITIONING, not exact defender positioning: coarse,
pre-pitch category labels ("Standard"/"Strategic"/"Infield shift"/"Infield
shade" for the infield; "Standard"/"Strategic"/"4th outfielder" for the
outfield), never exact coordinates, pre-contact movement, reaction time,
route efficiency, or a judgment of whether the alignment chosen was
strategically appropriate. See README.md "Alignment-aware positioning
(Version 0.6)" for the full limitation statement.

Controlled variants, trained on the SAME 2021-2023 rows and evaluated on the
SAME untouched 2024 validation rows (2025 never touched), all with
`class_weight=None` (see CLAUDE.md -- never class-weighted):

  - `baseline_v02`: the currently-selected production model, unchanged.
  - `alignment_labels_v06`: `baseline_v02` + the two raw alignment labels
    (`mlb_luck_score.features.build_contact_features.
    ALIGNMENT_LABELS_CATEGORICAL_FEATURES`).
  - `alignment_interactions_v06`: `alignment_labels_v06` + physically
    motivated interaction terms (`ALIGNMENT_INTERACTION_NUMERIC_FEATURES` /
    `ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES`) -- infield alignment x
    batter handedness, infield alignment x spray direction, outfield
    alignment x launch angle, outfield alignment x projected distance, and
    infield-shift x pull-side-ground-ball.

`position_depth_v06` (baseline + public per-fielder positioning
coordinates/depths) is documented but NOT implemented: no reliable, publicly
downloadable per-play or per-fielder-position coordinate/depth dataset was
found to exist -- Baseball Savant's public data exposes only the coarse
`if_fielding_alignment`/`of_fielding_alignment` category labels used above,
not exact or average fielder (x, y) positions joinable to individual plays.
This is exactly the public-data limitation the task anticipated (see module
docstring's "alignment-aware positioning, not exact defender positioning").
If a reliable public source is ever found, `position_depth_v06` should be
added here as a fourth candidate under the same adoption rule -- it is a
scope gap, not a design decision to exclude numeric depth data on purpose.

Reuses the Version 0.3/0.4/0.5 calibration/by-venue/material-regression/
paired-bootstrap machinery rather than reimplementing it, and adds Version
-0.6-specific batted-ball-type/pull-oppo/shift-status/handedness/base-state
subgroup calibration, a report of the plays whose predicted probabilities
changed the most, and controlled-perturbation directional checks (reusing
`mlb_luck_score.models.weather_perturbation`, which is generic despite its
module name) for "does a shifted infield actually predict fewer pull-side
ground-ball singles" style physical plausibility.

ADOPTION RULE (`recommend_alignment_adoption`): every candidate must pass ALL
of -- (1) 2024 log-loss improvement, (2) a paired bootstrap interval that
supports the improvement, (3) no material overall/home-run ECE regression,
(4) no material batted-ball-type/handedness/pull-oppo/shift/base-state
subgroup regression, (5) no reliably-sampled venue triggers the existing
venue-regression rule, (6) controlled counterfactual perturbation checks
behave in the physically expected direction, (7) the positioning
counterfactual (`mlb_luck_score.scoring.positioning_attribution`) is stable
-- well-formed predictions on genuinely re-derived "typical alignment" rows,
AND `positioning_effect` is (by construction) ~0 for rows whose actual
alignment already IS the typical one, a strong automatable check against the
same class of stale-derived-feature bug documented in `mlb_luck_score.
features.build_contact_features.generate_standardized_environment_rows`.
The task's explicit constraint -- "The model should not be adopted merely
because it predicts shifted ground balls more accurately; its probabilities
must remain calibrated across ALL major groups" -- is why criterion (4)
checks every subgroup, not just the shift-related ones, and why nothing here
ever recommends adoption from an improvement restricted to one subgroup.

Usage:

    python -m mlb_luck_score.models.compare_alignment_aware \\
        --input data/processed/cleaned_development_data_with_venue.parquet \\
        --output-dir outputs/tables --figures-dir outputs/figures/alignment_aware
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_luck_score.config import CLASS_ORDER, TABLES_DIR, TRAIN_SEASONS, VALIDATION_SEASONS
from mlb_luck_score.features.build_contact_features import (
    ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES,
    ALIGNMENT_INTERACTION_NUMERIC_FEATURES,
    ALIGNMENT_LABELS_CATEGORICAL_FEATURES,
    STANDARD_ALIGNMENT_LABEL,
    add_alignment_interaction_features,
    generate_typical_alignment_rows,
)
from mlb_luck_score.models.compare_geometry_aware import (
    BOOTSTRAP_METRICS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    PairedBootstrapResult,
    compute_paired_bootstrap,
)
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_HIGH_DISTANCE_QUANTILE,
    DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    DEFAULT_MIN_VENUE_SAMPLES,
    VARIANT_BASELINE_V02,
    _prepare_venue_column,
    _summarize_variant,
    compute_subgroup_calibration,
    find_material_venue_regressions,
)
from mlb_luck_score.models.compare_weather_aware import find_material_subgroup_regressions
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.models.weather_perturbation import (
    DirectionalCheckResult,
    mean_predicted_probability,
)
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

logger = logging.getLogger(__name__)

VARIANT_ALIGNMENT_LABELS_V06 = "alignment_labels_v06"
VARIANT_ALIGNMENT_INTERACTIONS_V06 = "alignment_interactions_v06"
ALIGNMENT_CANDIDATE_VARIANTS: tuple[str, ...] = (
    VARIANT_ALIGNMENT_LABELS_V06,
    VARIANT_ALIGNMENT_INTERACTIONS_V06,
)

#: Adoption-rule thresholds. Documented Version 0.6 research placeholders,
#: not validated thresholds -- consistent with every prior version's margins.
DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN = 0.0
DEFAULT_MIN_ALIGNMENT_COVERAGE = 0.5
#: Minimum rows required in a perturbation-check subgroup (e.g. real
#: pull-side ground balls) before that check is run at all -- too few rows
#: makes the mean-probability comparison noise, not signal.
MIN_PERTURBATION_SAMPLE_SIZE = 30
#: Positioning-effect-at-standard-alignment tolerance: this should be
#: EXACTLY zero by construction (actual alignment == typical alignment for
#: these rows), so any deviation beyond floating-point noise indicates the
#: typical-alignment counterfactual is not correctly holding non-alignment
#: features fixed -- the same class of bug `generate_standardized_
#: environment_rows` documents having caught once already (Version 0.5).
POSITIONING_EFFECT_ZERO_TOLERANCE = 1e-6


def _prepare_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    return add_alignment_interaction_features(df)


def override_alignment_and_recompute(
    df: pd.DataFrame,
    *,
    if_alignment: str | None = None,
    of_alignment: str | None = None,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Override raw alignment label(s) on masked rows and RECOMPUTE every derived column.

    Unlike `mlb_luck_score.models.weather_perturbation.override_columns`
    (which only sets the named columns verbatim), alignment interaction
    features (`if_alignment_shift_indicator`, `if_alignment_x_stand`, etc.)
    are FUNCTIONS of the raw alignment labels -- overriding the raw label
    without recomputing its derived columns would feed `alignment_
    interactions_v06` an internally inconsistent row (e.g. a raw label of
    "Standard" alongside a stale `if_alignment_shift_indicator=1.0`).  This
    always recomputes via `add_alignment_interaction_features` after
    applying the override, for both `generate_typical_alignment_rows`-style
    "Standard" overrides and controlled-perturbation scenarios that override
    to any other real, previously-seen alignment value.
    """
    out = df.copy()
    active_mask = mask if mask is not None else pd.Series(True, index=out.index)
    if if_alignment is not None and "if_fielding_alignment" in out.columns:
        out.loc[active_mask, "if_fielding_alignment"] = if_alignment
    if of_alignment is not None and "of_fielding_alignment" in out.columns:
        out.loc[active_mask, "of_fielding_alignment"] = of_alignment
    return add_alignment_interaction_features(out)


def compute_alignment_subgroup_calibration(
    y_true: pd.Series, proba_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Version 0.6 required evaluation-group calibration.

    Covers every group the task named: batted-ball type, pull-side vs.
    opposite-field, shifted vs. standard alignment, LHB vs. RHB, and bases
    empty vs. runners on. Every subgroup mask comes from `val_df` (shared
    across all variants), so a variant's subgroup ECE is directly comparable
    to another's. (Per-venue calibration is reported separately by
    `_summarize_variant`'s existing `calibration_by_venue`, and "plays whose
    probabilities changed most" by `find_most_changed_plays` below -- neither
    fits this label->mask dict shape.)
    """
    subgroups: dict[str, dict[str, Any]] = {}

    if "bb_type" in val_df.columns:
        for bb_type in ("ground_ball", "line_drive", "fly_ball", "popup"):
            mask = (val_df["bb_type"] == bb_type).to_numpy()
            subgroups[f"bb_type_{bb_type}"] = compute_subgroup_calibration(
                y_true, proba_df, mask, f"bb_type_{bb_type}"
            )

    if "is_pull" in val_df.columns:
        pull_mask = val_df["is_pull"].fillna(False).astype(bool).to_numpy()
        subgroups["pull_side"] = compute_subgroup_calibration(
            y_true, proba_df, pull_mask, "pull_side"
        )
    if "is_opposite_field" in val_df.columns:
        oppo_mask = val_df["is_opposite_field"].fillna(False).astype(bool).to_numpy()
        subgroups["opposite_field"] = compute_subgroup_calibration(
            y_true, proba_df, oppo_mask, "opposite_field"
        )

    if (
        "if_alignment_shift_indicator" in val_df.columns
        and "of_alignment_shift_indicator" in val_df.columns
    ):
        if_shift = val_df["if_alignment_shift_indicator"]
        of_shift = val_df["of_alignment_shift_indicator"]
        shifted_mask = ((if_shift == 1.0) | (of_shift == 1.0)).fillna(False).to_numpy()
        standard_mask = ((if_shift == 0.0) & (of_shift == 0.0)).fillna(False).to_numpy()
        subgroups["shifted_alignment"] = compute_subgroup_calibration(
            y_true, proba_df, shifted_mask, "shifted_alignment"
        )
        subgroups["standard_alignment"] = compute_subgroup_calibration(
            y_true, proba_df, standard_mask, "standard_alignment"
        )

    if "stand" in val_df.columns:
        subgroups["lhb"] = compute_subgroup_calibration(
            y_true, proba_df, (val_df["stand"] == "L").to_numpy(), "lhb"
        )
        subgroups["rhb"] = compute_subgroup_calibration(
            y_true, proba_df, (val_df["stand"] == "R").to_numpy(), "rhb"
        )

    base_cols = [c for c in ("on_1b", "on_2b", "on_3b") if c in val_df.columns]
    if base_cols:
        runners_on_mask = val_df[base_cols].notna().any(axis=1).to_numpy()
        subgroups["bases_empty"] = compute_subgroup_calibration(
            y_true, proba_df, ~runners_on_mask, "bases_empty"
        )
        subgroups["runners_on"] = compute_subgroup_calibration(
            y_true, proba_df, runners_on_mask, "runners_on"
        )

    return subgroups


def find_most_changed_plays(
    baseline_proba: pd.DataFrame,
    candidate_proba: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """The `top_n` validation rows whose predicted probabilities shifted the most.

    "Shift" is total variation distance between the two probability vectors
    (`0.5 * sum(|candidate - baseline|)`, in `[0, 1]`) -- a single number
    combining every class's change, not just one class. Returned for manual
    qualitative review (see notebook `08_alignment_positioning_analysis.
    ipynb`) -- this is a REQUIRED evaluation group per the task, but it is
    not itself an automated adoption criterion (there is no principled
    threshold for "how much a play's probabilities should change"; the point
    is to let a maintainer read the actual plays and features).
    """
    if not baseline_proba.index.equals(candidate_proba.index):
        raise ValueError("baseline_proba and candidate_proba must share the same row index")

    labels = list(CLASS_ORDER)
    tvd = 0.5 * (candidate_proba[labels] - baseline_proba[labels]).abs().sum(axis=1)
    top_idx = tvd.sort_values(ascending=False).head(top_n).index

    report_cols = [
        c
        for c in (
            "game_pk",
            "if_fielding_alignment",
            "of_fielding_alignment",
            "bb_type",
            "stand",
            "spray_sector",
            "is_pull",
            "launch_angle",
            "hit_distance_sc",
            "venue_id",
        )
        if c in val_df.columns
    ]

    records: list[dict[str, Any]] = []
    for idx in top_idx:
        record: dict[str, Any] = {"probability_shift_tvd": float(tvd.loc[idx])}
        record.update({c: val_df.loc[idx, c] for c in report_cols})
        record.update({f"baseline_p_{cls}": float(baseline_proba.loc[idx, cls]) for cls in labels})
        record.update(
            {f"candidate_p_{cls}": float(candidate_proba.loc[idx, cls]) for cls in labels}
        )
        records.append(record)
    return records


def run_alignment_aware_comparison(
    joined_df: pd.DataFrame,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
    high_distance_quantile: float = DEFAULT_HIGH_DISTANCE_QUANTILE,
    figures_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, TrainedModel], dict[str, pd.DataFrame]]:
    """Train and evaluate `baseline_v02` + both v0.6 candidates on IDENTICAL rows.

    Args:
        joined_df: Cleaned development data joined with venue metadata (see
            `mlb_luck_score.data.join_venue_metadata`) -- alignment labels
            come straight from the cleaned Statcast columns, no separate
            join step is required for them.
        min_venue_samples: Minimum rows for a venue's calibration figures to
            be marked reliable.
        high_distance_quantile: Percentile threshold defining "high
            projected-distance contact".
        figures_dir: If given, save one calibration plot set per variant.

    Returns:
        `(comparison, trained_models, proba_by_variant)`.
    """
    required = ("if_fielding_alignment", "of_fielding_alignment", "venue_id")
    missing = [c for c in required if c not in joined_df.columns]
    if missing:
        raise ValueError(
            f"joined_df is missing column(s) {missing} -- run "
            "mlb_luck_score.data.join_venue_metadata.join_venue_metadata first."
        )

    df = _prepare_alignment_columns(joined_df)
    df = _prepare_venue_column(df)

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the alignment-aware comparison."
        )
    logger.info(
        "Identical rows for all 3 variants: %d training rows, %d validation rows",
        len(train_df),
        len(val_df),
    )

    variant_kwargs: dict[str, dict[str, Any]] = {
        VARIANT_BASELINE_V02: {},
        VARIANT_ALIGNMENT_LABELS_V06: {
            "extra_categorical_features": ALIGNMENT_LABELS_CATEGORICAL_FEATURES,
        },
        VARIANT_ALIGNMENT_INTERACTIONS_V06: {
            "extra_numeric_features": ALIGNMENT_INTERACTION_NUMERIC_FEATURES,
            "extra_categorical_features": ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES,
        },
    }

    alignment_coverage_mask = (
        val_df["if_fielding_alignment"].notna() & val_df["of_fielding_alignment"].notna()
    )

    comparison: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, TrainedModel] = {}
    proba_by_variant: dict[str, pd.DataFrame] = {}

    for variant, kwargs in variant_kwargs.items():
        logger.info("=== [%s] training on %d rows ===", variant, len(train_df))
        trained = train_model(train_df, class_weight=None, **kwargs)
        feature_cols = trained.numeric_features + trained.categorical_features
        proba_df = predict_proba_ordered(trained, val_df[feature_cols])
        validate_probabilities(proba_df)

        summary = _summarize_variant(
            variant,
            trained,
            proba_df,
            val_df,
            min_venue_samples=min_venue_samples,
            high_distance_quantile=high_distance_quantile,
            figures_dir=figures_dir,
        )
        y_true = val_df["outcome_class"].astype(str)
        summary["alignment_subgroups"] = compute_alignment_subgroup_calibration(
            y_true, proba_df, val_df
        )
        summary["alignment_coverage_rate"] = float(alignment_coverage_mask.mean())

        comparison[variant] = summary
        trained_models[variant] = trained
        proba_by_variant[variant] = proba_df

    for candidate_name in ALIGNMENT_CANDIDATE_VARIANTS:
        comparison[candidate_name]["most_changed_plays"] = find_most_changed_plays(
            proba_by_variant[VARIANT_BASELINE_V02], proba_by_variant[candidate_name], val_df
        )

    return comparison, trained_models, proba_by_variant


def run_alignment_perturbation_checks(
    trained: TrainedModel, val_df: pd.DataFrame
) -> dict[str, DirectionalCheckResult]:
    """Controlled-perturbation directional checks for one candidate's trained model.

    Args:
        trained: The candidate's fitted model.
        val_df: Validation rows, already through `_prepare_alignment_columns`.

    Checks (each only run if its subgroup has at least
    `MIN_PERTURBATION_SAMPLE_SIZE` real rows):

      1. Among REAL pull-side ground balls, an "Infield shift" infield
         alignment should predict a LOWER mean `single` probability than a
         "Standard" alignment -- the entire physical rationale for shifting
         is to convert pulled ground balls that would otherwise be singles.
      2. Among REAL pull-side fly balls, a "Strategic" outfield alignment
         should predict a LOWER mean `double` probability than a "Standard"
         outfield alignment.
    """
    results: dict[str, DirectionalCheckResult] = {}
    if "if_fielding_alignment" not in val_df.columns or "is_pull" not in val_df.columns:
        return results

    is_pull = val_df["is_pull"].fillna(False).astype(bool)

    pull_groundball_mask = is_pull & (val_df["bb_type"] == "ground_ball")
    n_pull_groundball = int(pull_groundball_mask.sum())
    if n_pull_groundball >= MIN_PERTURBATION_SAMPLE_SIZE:
        subset = val_df[pull_groundball_mask]
        standard_subset = override_alignment_and_recompute(
            subset, if_alignment=STANDARD_ALIGNMENT_LABEL
        )
        shifted_subset = override_alignment_and_recompute(subset, if_alignment="Infield shift")
        mean_standard = mean_predicted_probability(trained, standard_subset, "single")
        mean_shifted = mean_predicted_probability(trained, shifted_subset, "single")
        delta = mean_shifted - mean_standard
        results["infield_shift_reduces_pull_groundball_singles"] = DirectionalCheckResult(
            label="infield alignment (pull-side ground balls): Standard vs Infield shift",
            outcome_class="single",
            low_label="Standard",
            high_label="Infield shift",
            mean_prob_low=mean_standard,
            mean_prob_high=mean_shifted,
            delta=delta,
            expect_high_greater=False,
            passed=delta < 0,
            sample_size=n_pull_groundball,
        )

    pull_flyball_mask = is_pull & (val_df["bb_type"] == "fly_ball")
    n_pull_flyball = int(pull_flyball_mask.sum())
    if n_pull_flyball >= MIN_PERTURBATION_SAMPLE_SIZE:
        subset = val_df[pull_flyball_mask]
        standard_subset = override_alignment_and_recompute(
            subset, of_alignment=STANDARD_ALIGNMENT_LABEL
        )
        strategic_subset = override_alignment_and_recompute(subset, of_alignment="Strategic")
        mean_standard = mean_predicted_probability(trained, standard_subset, "double")
        mean_strategic = mean_predicted_probability(trained, strategic_subset, "double")
        delta = mean_strategic - mean_standard
        results["outfield_strategic_reduces_pull_flyball_doubles"] = DirectionalCheckResult(
            label="outfield alignment (pull-side fly balls): Standard vs Strategic",
            outcome_class="double",
            low_label="Standard",
            high_label="Strategic",
            mean_prob_low=mean_standard,
            mean_prob_high=mean_strategic,
            delta=delta,
            expect_high_greater=False,
            passed=delta < 0,
            sample_size=n_pull_flyball,
        )

    return results


def check_positioning_counterfactual_stable(
    trained: TrainedModel, val_df: pd.DataFrame
) -> dict[str, Any]:
    """Automated checks for the positioning counterfactual's stability (criterion 7).

    1. `typical_predictions_well_formed`: predicting on genuinely re-derived
       "typical alignment" rows (`generate_typical_alignment_rows`, NOT the
       model's regular actual-alignment predictions -- see `mlb_luck_score.
       models.compare_weather_variants.check_standardized_stability_for_
       candidate` for why re-deriving matters) produces finite, valid
       probability rows.
    2. `positioning_effect_zero_at_standard_alignment`: for rows whose
       ACTUAL alignment already IS the typical ("Standard"/"Standard")
       alignment, `positioning_effect` (`mlb_luck_score.scoring.
       positioning_attribution`) must be ~0 by construction (actual and
       typical rows are then IDENTICAL after `add_alignment_interaction_
       features` recomputation) -- any nonzero deviation beyond floating
       -point noise means the counterfactual generator is not actually
       holding non-alignment features fixed, the same bug class Version 0.5
       caught in `generate_standardized_environment_rows`.
    """
    feature_cols = trained.numeric_features + trained.categorical_features
    typical_df = generate_typical_alignment_rows(val_df)
    typical_proba = predict_proba_ordered(trained, typical_df[feature_cols])
    try:
        validate_probabilities(typical_proba)
        typical_predictions_well_formed = True
    except ValueError:
        typical_predictions_well_formed = False

    actual_proba = predict_proba_ordered(trained, val_df[feature_cols])
    expected_actual = compute_expected_run_value_vectorized(actual_proba)
    expected_typical = compute_expected_run_value_vectorized(typical_proba)
    positioning_effect = expected_actual - expected_typical

    already_standard_mask = (
        (val_df["if_fielding_alignment"] == STANDARD_ALIGNMENT_LABEL)
        & (val_df["of_fielding_alignment"] == STANDARD_ALIGNMENT_LABEL)
    ).to_numpy()
    max_abs_effect_at_standard = (
        float(np.abs(positioning_effect.to_numpy()[already_standard_mask]).max())
        if already_standard_mask.any()
        else 0.0
    )
    zero_at_standard = max_abs_effect_at_standard <= POSITIONING_EFFECT_ZERO_TOLERANCE

    return {
        "typical_predictions_well_formed": typical_predictions_well_formed,
        "positioning_effect_zero_at_standard_alignment": zero_at_standard,
        "max_abs_positioning_effect_at_standard_alignment": max_abs_effect_at_standard,
        "stable": typical_predictions_well_formed and zero_at_standard,
    }


def recommend_alignment_adoption(
    comparison: dict[str, dict[str, Any]],
    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]],
    perturbation_by_candidate: dict[str, dict[str, DirectionalCheckResult]],
    counterfactual_stability_by_candidate: dict[str, dict[str, Any]],
    *,
    candidate_variants: tuple[str, ...] = ALIGNMENT_CANDIDATE_VARIANTS,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    meaningful_log_loss_regression_margin: float = DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN,
    min_alignment_coverage: float = DEFAULT_MIN_ALIGNMENT_COVERAGE,
) -> dict[str, Any]:
    """Apply the Version 0.6 adoption rule (7 criteria) to every candidate.

    All criteria combined with AND -- a candidate that improves shifted
    -ground-ball prediction specifically but regresses ANY other required
    subgroup, or fails ANY controlled-perturbation check, is NOT recommended
    (see module docstring's explicit statement of the task's calibration
    -across-all-groups constraint).
    """
    baseline = comparison[VARIANT_BASELINE_V02]
    per_candidate: dict[str, Any] = {}

    for candidate_name in candidate_variants:
        if candidate_name not in comparison:
            continue
        candidate = comparison[candidate_name]
        bootstrap = bootstrap_by_candidate.get(candidate_name, {})

        improves_log_loss = candidate["multiclass_log_loss"] < baseline["multiclass_log_loss"]

        log_loss_bootstrap = bootstrap.get("log_loss")
        bootstrap_supports = (
            log_loss_bootstrap is not None
            and log_loss_bootstrap["ci_high"] <= meaningful_log_loss_regression_margin
        )

        ece_delta = candidate["expected_calibration_error"] - baseline["expected_calibration_error"]
        ece_not_worse = ece_delta <= absolute_margin

        hr_ece_delta = candidate["home_run_ece"] - baseline["home_run_ece"]
        hr_ece_not_worse = hr_ece_delta <= absolute_margin

        material_venue_regressions = find_material_venue_regressions(
            baseline["calibration_by_venue"],
            candidate["calibration_by_venue"],
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
        )
        no_venue_regressions = not material_venue_regressions

        material_subgroup_regressions = find_material_subgroup_regressions(
            baseline.get("alignment_subgroups", {}),
            candidate.get("alignment_subgroups", {}),
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
        )
        no_subgroup_regressions = not material_subgroup_regressions

        coverage = candidate.get("alignment_coverage_rate")
        sufficient_coverage = coverage is None or coverage >= min_alignment_coverage

        perturbation_results = perturbation_by_candidate.get(candidate_name, {})
        perturbation_failures = [
            name for name, result in perturbation_results.items() if not result.passed
        ]
        perturbation_checks_passed = bool(perturbation_results) and not perturbation_failures

        counterfactual_stability = counterfactual_stability_by_candidate.get(candidate_name, {})
        counterfactual_stable = bool(counterfactual_stability.get("stable", False))

        passes_all_automated = (
            improves_log_loss
            and bootstrap_supports
            and ece_not_worse
            and hr_ece_not_worse
            and no_venue_regressions
            and no_subgroup_regressions
            and sufficient_coverage
            and perturbation_checks_passed
            and counterfactual_stable
        )

        per_candidate[candidate_name] = {
            "improves_log_loss": improves_log_loss,
            "log_loss_delta": candidate["multiclass_log_loss"] - baseline["multiclass_log_loss"],
            "bootstrap_supports_improvement": bootstrap_supports,
            "log_loss_bootstrap": log_loss_bootstrap,
            "ece_not_materially_worse": ece_not_worse,
            "ece_delta": ece_delta,
            "home_run_ece_not_materially_worse": hr_ece_not_worse,
            "home_run_ece_delta": hr_ece_delta,
            "no_material_venue_regressions": no_venue_regressions,
            "material_venue_regressions": material_venue_regressions,
            "no_material_subgroup_regressions": no_subgroup_regressions,
            "material_subgroup_regressions": material_subgroup_regressions,
            "sufficient_alignment_coverage": sufficient_coverage,
            "alignment_coverage_rate": coverage,
            "perturbation_checks_passed": perturbation_checks_passed,
            "perturbation_failures": perturbation_failures,
            "perturbation_detail": {k: v.__dict__ for k, v in perturbation_results.items()},
            "positioning_counterfactual_stable": counterfactual_stable,
            "positioning_counterfactual_detail": counterfactual_stability,
            "passes_all_automated_criteria": passes_all_automated,
        }

    passing = [
        name for name, result in per_candidate.items() if result["passes_all_automated_criteria"]
    ]
    best_candidate = None
    if passing:
        best_candidate = min(passing, key=lambda name: comparison[name]["multiclass_log_loss"])

    return {
        "per_candidate": per_candidate,
        "recommend_adopt_any_v06_candidate": bool(passing),
        "best_candidate": best_candidate,
        "note": (
            "All 7 adoption criteria are combined with AND -- a candidate that predicts "
            "shifted ground balls better but regresses any other required subgroup, or "
            "fails any controlled-perturbation direction check, is never recommended."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with venue metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--min-venue-samples", type=int, default=DEFAULT_MIN_VENUE_SAMPLES)
    parser.add_argument(
        "--high-distance-quantile", type=float, default=DEFAULT_HIGH_DISTANCE_QUANTILE
    )
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        comparison, trained_models, proba_by_variant = run_alignment_aware_comparison(
            df,
            min_venue_samples=args.min_venue_samples,
            high_distance_quantile=args.high_distance_quantile,
            figures_dir=args.figures_dir,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    val_df = _prepare_alignment_columns(val_df)
    val_df = _prepare_venue_column(val_df)
    y_true = val_df["outcome_class"].astype(str)

    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]] = {}
    perturbation_by_candidate: dict[str, dict[str, DirectionalCheckResult]] = {}
    counterfactual_stability_by_candidate: dict[str, dict[str, Any]] = {}

    for candidate_name in ALIGNMENT_CANDIDATE_VARIANTS:
        logger.info(
            "Running paired bootstrap for %s vs %s ...", candidate_name, VARIANT_BASELINE_V02
        )
        bootstrap_by_candidate[candidate_name] = compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_BASELINE_V02],
            proba_by_variant[candidate_name],
            val_df["game_pk"],
            n_reps=args.n_bootstrap_reps,
            seed=args.bootstrap_seed,
            metrics=BOOTSTRAP_METRICS,
        )
        logger.info("Running controlled-perturbation checks for %s ...", candidate_name)
        trained = trained_models[candidate_name]
        perturbation_by_candidate[candidate_name] = run_alignment_perturbation_checks(
            trained, val_df
        )
        counterfactual_stability_by_candidate[candidate_name] = (
            check_positioning_counterfactual_stable(trained, val_df)
        )

    recommendation = recommend_alignment_adoption(
        comparison,
        bootstrap_by_candidate,
        perturbation_by_candidate,
        counterfactual_stability_by_candidate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "alignment_aware_comparison_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "comparison": comparison,
                "bootstrap": bootstrap_by_candidate,
                "recommendation": recommendation,
            },
            indent=2,
            default=str,
        )
    )

    for variant, summary in comparison.items():
        logger.info(
            "[%s] log_loss=%.6f ece=%.6f home_run_ece=%.6f n=%d",
            variant,
            summary["multiclass_log_loss"],
            summary["expected_calibration_error"],
            summary["home_run_ece"],
            summary["sample_count"],
        )
    for candidate_name, result in recommendation["per_candidate"].items():
        logger.info(
            "[%s] passes_all_automated_criteria=%s perturbation_failures=%s",
            candidate_name,
            result["passes_all_automated_criteria"],
            result["perturbation_failures"],
        )
    logger.info(
        "recommend_adopt_any_v06_candidate=%s best_candidate=%s",
        recommendation["recommend_adopt_any_v06_candidate"],
        recommendation["best_candidate"],
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
