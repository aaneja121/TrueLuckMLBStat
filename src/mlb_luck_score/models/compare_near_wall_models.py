"""Contact Luck v0.7C: near-wall opportunity-difficulty specialist.

Version 0.7A's `measured_contact_only_v07` (`mlb_luck_score.models.
compare_opportunity_models`) has real, honestly-reported calibration
weaknesses for wall-adjacent plays on real 2024 data: ECE 0.180-0.254 within
the 5/10/20-ft wall bands, vs. an overall 0.018 -- roughly 10-15x worse,
concentrated exactly where judging defense is hardest (ambiguous
warning-track plays). This module builds a GATED architecture to address
that specifically, without touching open-field scoring at all:

  - `open_field_v07`: the EXISTING, unchanged `measured_contact_only_v07`
    model, used for plays that are NOT within 20 ft of the wall (see
    `mlb_luck_score.models.outfield_gating`). No retraining here.
  - `near_wall_v07c_candidate`: a specialist trained ONLY on wall-adjacent
    plays, using the near-wall feature set (`mlb_luck_score.features.
    build_contact_features.NEAR_WALL_NUMERIC_FEATURES`/`NEAR_WALL_
    CATEGORICAL_FEATURES` -- adds `wall_height_in_spray_direction`,
    `projected_distance_to_wall_margin`, `wall_segment_label` to the
    Version 0.7A set).

## Model selection: logistic baseline vs. one unweighted nonlinear model

Two candidates are compared with the IDENTICAL near-wall feature set (see
`mlb_luck_score.models.train_opportunity_model.MODEL_TYPE_HGB`'s docstring
for why no hand-engineered interaction terms are added to the logistic
candidate -- the point of this comparison is model CLASS, not feature
representation):

  - `near_wall_logistic_v07c`: `LogisticRegression`, `class_weight=None`.
  - `near_wall_hgb_v07c`: `HistGradientBoostingClassifier`, `class_weight=
    None` -- can find nonlinear interactions among hang time/launch angle/
    wall distance/wall height/spray direction automatically, unlike a
    linear model.

## Season split -- REUSES the existing post-hoc-calibration split

Per the task's explicit instruction ("2021-2022 for fitting, 2023 for model
selection, 2024 for the final comparison"), this module reuses
`mlb_luck_score.config.CALIBRATION_BASE_TRAIN_SEASONS`/`CALIBRATION_FIT_
SEASONS`/`CALIBRATION_EVAL_SEASONS` (2021-2022 / 2023 / 2024) rather than
inventing new near-wall-specific season constants -- these are semantically
IDENTICAL to what a fit/select/final-eval three-way split needs, already
documented and tested for exactly this purpose (`mlb_luck_score.models.
compare_models`'s post-hoc calibration design). 2025 remains untouched
throughout (`assert_seasons_allowed`).

`open_field_v07`, in contrast, is evaluated here EXACTLY as already trained
in `mlb_luck_score.models.compare_opportunity_models` (fit on the standard
`TRAIN_SEASONS` = 2021-2023) -- it is not retrained on the 2021-2022-only
split. The final 2024 comparison is: "does routing wall-adjacent plays to
the freshly-trained near-wall specialist improve on what the EXISTING
open-field model would have predicted for those same plays," which is
exactly what the gated architecture needs to answer.

## Required checks (all combined, see `summarize_near_wall_validation`)

Log loss/ECE within the 5/10/20-ft wall bands; calibration by wall height,
venue, spray sector, and opportunity-time bucket; a game-level paired
bootstrap (near-wall specialist vs. `open_field_v07`, on the SAME near-wall
2024 rows); an architectural no-open-field-regression check (`open_field_
v07`'s predictions for open-field-gated rows are BIT-IDENTICAL whether
reached via the gate or directly -- gating only routes rows, it cannot
alter the model or its inputs); and controlled-perturbation directional
checks (wall distance close-vs-far; launch angle low-vs-high, i.e. short
-vs-long hang time, with the DEPENDENT `projected_distance_to_wall_margin`/
`absolute_distance_to_wall`/`estimated_hang_time_s` columns RECOMPUTED after
each override -- see `_override_wall_distance_and_recompute`/`_override_
launch_angle_and_recompute_hang_time`'s docstrings for a real bug this
caught during development: overriding a raw feature without recomputing its
dependents fed the model an internally-inconsistent row and produced a
spurious backwards wall-distance result).

On real 2024 data, the FIXED wall-distance perturbation check passes
cleanly (close wall -> materially lower P(out) than far wall, as expected).
The launch-angle/hang-time check, however, is genuinely backwards from the
open-field intuition even after the fix -- and this may be a real, different
physical relationship specific to the near-wall CONDITIONAL population, not
a bug: among plays that all travel roughly the same (long) distance to
reach the wall by construction of the gate, achieving that distance via a
LOW launch angle requires much higher exit velocity (a flat, hard-hit
line-drive double/triple that gives the fielder little time to react) than
achieving it via a HIGH launch angle (a softer, higher-arcing fly ball the
fielder has more time to track) -- the reverse of the open-field-wide
correlation between hang time and out probability, where distance itself
varies freely. This is reported as an unresolved, documented finding
-- per CLAUDE.md's confounding-by-indication guidance, a backwards
-signed perturbation is treated as a signal to investigate further, not a
check to relax to force a pass -- and is exactly why `near_wall_specialist_
calibrated` stays `False` until it is understood.

## Reporting rule until/unless this succeeds

Per the task: "Outfield execution score available for calibrated open-field
opportunities; provisional or unavailable for wall-adjacent opportunities."
`summarize_near_wall_validation`'s `near_wall_specialist_calibrated` boolean
drives EXACTLY this in `mlb_luck_score.scoring.gated_outfield_report` -- if
it's `False`, near-wall rows are still scored (never silently blanked) but
explicitly labeled `"provisional_near_wall"`, never presented as equivalent
-confidence to open-field scores.

Usage:

    python -m mlb_luck_score.models.compare_near_wall_models \\
        --input data/processed/cleaned_development_data_with_geometry.parquet \\
        --output-dir outputs/tables
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
from sklearn.metrics import log_loss

from mlb_luck_score.config import (
    CALIBRATION_BASE_TRAIN_SEASONS,
    CALIBRATION_EVAL_SEASONS,
    CALIBRATION_FIT_SEASONS,
    TABLES_DIR,
    assert_seasons_allowed,
)
from mlb_luck_score.data.outfield_physics import estimate_hang_time_seconds
from mlb_luck_score.eligibility import add_outfield_opportunity_eligibility
from mlb_luck_score.features.build_contact_features import (
    NEAR_WALL_CATEGORICAL_FEATURES,
    NEAR_WALL_NUMERIC_FEATURES,
    OPPORTUNITY_TARGET_COLUMN,
    add_geometry_interaction_features,
    add_outfield_opportunity_features,
)
from mlb_luck_score.models.compare_geometry_aware import _bool_mask
from mlb_luck_score.models.compare_opportunity_models import (
    VARIANT_MEASURED_CONTACT_ONLY,
    compute_binary_calibration_by_venue,
    compute_binary_calibration_table,
    compute_binary_ece,
    compute_binary_subgroup_calibration,
    compute_opportunity_time_buckets,
)
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_MIN_VENUE_SAMPLES,
    _prepare_venue_column,
)
from mlb_luck_score.models.outfield_gating import (
    GATE_NEAR_WALL,
    GATE_OPEN_FIELD,
    assign_outfield_opportunity_gate,
)
from mlb_luck_score.models.train_opportunity_model import (
    MODEL_TYPE_HGB,
    MODEL_TYPE_LOGISTIC,
    TrainedOpportunityModel,
    predict_opportunity_proba,
    train_opportunity_model,
    validate_opportunity_probabilities,
)
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult, override_columns

logger = logging.getLogger(__name__)

VARIANT_OPEN_FIELD = VARIANT_MEASURED_CONTACT_ONLY
VARIANT_NEAR_WALL_LOGISTIC = "near_wall_logistic_v07c"
VARIANT_NEAR_WALL_HGB = "near_wall_hgb_v07c"
NEAR_WALL_SELECTION_CANDIDATES: tuple[str, ...] = (
    VARIANT_NEAR_WALL_LOGISTIC,
    VARIANT_NEAR_WALL_HGB,
)
#: The winning candidate is reported under this name in the final 2024
#: comparison -- "the" near-wall specialist, per the task's naming.
VARIANT_NEAR_WALL_FINAL = "near_wall_v07c_candidate"

_MODEL_TYPE_BY_CANDIDATE: dict[str, str] = {
    VARIANT_NEAR_WALL_LOGISTIC: MODEL_TYPE_LOGISTIC,
    VARIANT_NEAR_WALL_HGB: MODEL_TYPE_HGB,
}

DEFAULT_MIN_SUBGROUP_SAMPLES = 100
MATERIAL_ECE_ABSOLUTE_THRESHOLD = 0.05
DEFAULT_N_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

#: Controlled-perturbation scenario magnitudes -- chosen within the real
#: observed range for near-wall plays (2021-2024: wall distance
#: 302-419 ft, launch angle 12-54 deg). Documented Version 0.7C research
#: placeholders, not validated thresholds.
CLOSE_WALL_DISTANCE_FT = 320.0
FAR_WALL_DISTANCE_FT = 400.0
LOW_LAUNCH_ANGLE_DEG = 18.0
HIGH_LAUNCH_ANGLE_DEG = 40.0


def _prepare_near_wall_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_outfield_opportunity_eligibility(df)
    out = add_geometry_interaction_features(out)
    out = add_outfield_opportunity_features(out)
    return out


def get_near_wall_rows(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Outfield-opportunity-eligible rows gated to `GATE_NEAR_WALL`."""
    required = (
        "if_fielding_alignment",
        "of_fielding_alignment",
        "wall_distance_in_spray_direction",
    )
    missing = [c for c in required if c not in joined_df.columns]
    if missing:
        raise ValueError(
            f"joined_df is missing column(s) {missing} -- run mlb_luck_score.data."
            "join_venue_metadata AND mlb_luck_score.data.join_park_geometry first."
        )
    df = _prepare_near_wall_columns(joined_df)
    df = _prepare_venue_column(df) if "venue_id" in df.columns else df
    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]
    gate = assign_outfield_opportunity_gate(eligible)
    return eligible[gate == GATE_NEAR_WALL]


def _mean_predicted_p_out(trained: TrainedOpportunityModel, df: pd.DataFrame) -> float:
    feature_cols = trained.numeric_features + trained.categorical_features
    return float(predict_opportunity_proba(trained, df[feature_cols]).mean())


def compute_wall_height_buckets(wall_height: pd.Series, *, q: int = 3) -> pd.Series:
    """Tercile buckets of wall height -- coarser than opportunity-time quartiles because
    `wall_height_in_spray_direction` coverage is sparse (~12% of near-wall rows have a
    reviewed height, see README.md "Outfield opportunity and execution" -- Version 0.4's
    park-geometry table has height data for a minority of venues/eras).
    """
    try:
        return pd.qcut(wall_height, q=q, labels=["short", "medium", "tall"])
    except ValueError:
        return pd.Series(pd.NA, index=wall_height.index)


def compute_near_wall_subgroups(
    y_true: np.ndarray, p_out: pd.Series, val_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Required Version 0.7C subgroup calibration: wall bands, wall height, spray sector,
    opportunity-time bucket. Per-venue calibration is reported separately (`compute_binary_
    calibration_by_venue`).
    """
    subgroups: dict[str, dict[str, Any]] = {}

    for col in ("near_wall_5ft", "near_wall_10ft", "near_wall_20ft"):
        if col in val_df.columns:
            subgroups[col] = compute_binary_subgroup_calibration(
                y_true, p_out, _bool_mask(val_df[col]), col
            )

    if "wall_height_in_spray_direction" in val_df.columns:
        buckets = compute_wall_height_buckets(val_df["wall_height_in_spray_direction"])
        for bucket in ("short", "medium", "tall"):
            mask = (buckets == bucket).to_numpy()
            subgroups[f"wall_height_{bucket}"] = compute_binary_subgroup_calibration(
                y_true, p_out, mask, f"wall_height_{bucket}"
            )

    if "spray_sector" in val_df.columns:
        for sector in ("left", "left_center", "center", "right_center", "right"):
            mask = (val_df["spray_sector"] == sector).to_numpy()
            subgroups[f"spray_sector_{sector}"] = compute_binary_subgroup_calibration(
                y_true, p_out, mask, f"spray_sector_{sector}"
            )

    if "estimated_hang_time_s" in val_df.columns:
        buckets = compute_opportunity_time_buckets(val_df["estimated_hang_time_s"])
        for bucket in ("q1_shortest", "q2", "q3", "q4_longest"):
            mask = (buckets == bucket).to_numpy()
            subgroups[f"opportunity_time_{bucket}"] = compute_binary_subgroup_calibration(
                y_true, p_out, mask, f"opportunity_time_{bucket}"
            )

    return subgroups


def run_near_wall_model_selection(
    near_wall_df: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, TrainedOpportunityModel]]:
    """Fit both near-wall candidates on `CALIBRATION_BASE_TRAIN_SEASONS`, select
    on `CALIBRATION_FIT_SEASONS` by log loss (tie-broken by ECE).

    Returns:
        `(winner_variant, selection_metrics_by_candidate, trained_by_candidate)`.
    """
    fit_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    selection_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_FIT_SEASONS)]
    if fit_df.empty or selection_df.empty:
        raise ValueError(
            f"Need non-empty fit ({CALIBRATION_BASE_TRAIN_SEASONS}) and selection "
            f"({CALIBRATION_FIT_SEASONS}) near-wall rows to run model selection."
        )
    logger.info(
        "Near-wall model selection: %d fit rows, %d selection rows", len(fit_df), len(selection_df)
    )

    selection_metrics: dict[str, dict[str, Any]] = {}
    trained_by_candidate: dict[str, TrainedOpportunityModel] = {}

    for candidate, model_type in _MODEL_TYPE_BY_CANDIDATE.items():
        trained = train_opportunity_model(
            fit_df,
            class_weight=None,
            feature_set_label=candidate,
            numeric_features=NEAR_WALL_NUMERIC_FEATURES,
            categorical_features=NEAR_WALL_CATEGORICAL_FEATURES,
            model_type=model_type,
        )
        feature_cols = trained.numeric_features + trained.categorical_features
        p_out = predict_opportunity_proba(trained, selection_df[feature_cols])
        validate_opportunity_probabilities(p_out)
        y_true = selection_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
        table = compute_binary_calibration_table(y_true, p_out)
        selection_metrics[candidate] = {
            "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
            "expected_calibration_error": compute_binary_ece(table),
        }
        trained_by_candidate[candidate] = trained

    winner = min(
        NEAR_WALL_SELECTION_CANDIDATES,
        key=lambda c: (
            selection_metrics[c]["binary_log_loss"],
            selection_metrics[c]["expected_calibration_error"],
        ),
    )
    logger.info("Near-wall model selection winner: %s (%s)", winner, selection_metrics[winner])
    return winner, selection_metrics, trained_by_candidate


def _override_wall_distance_and_recompute(
    df: pd.DataFrame, new_wall_distance_ft: float
) -> pd.DataFrame:
    """Override `wall_distance_in_spray_direction` and RECOMPUTE its dependents.

    `projected_distance_to_wall_margin`/`absolute_distance_to_wall` are
    DERIVED from `wall_distance_in_spray_direction` (`mlb_luck_score.data.
    join_park_geometry`: `margin = hit_distance_sc - wall_distance_in_spray_
    direction`). A naive `override_columns` call that overrides only
    `wall_distance_in_spray_direction` leaves those two columns STALE
    (reflecting the ORIGINAL wall distance), feeding the model an internally
    -inconsistent row -- the exact same class of bug `generate_standardized_
    environment_rows`/`generate_typical_alignment_rows` document (see
    CLAUDE.md). Verified: this was the actual cause of an initial backwards
    -signed wall-distance perturbation result during Version 0.7C
    development.
    """
    out = override_columns(df, {"wall_distance_in_spray_direction": new_wall_distance_ft})
    margin = out["hit_distance_sc"].astype(float) - new_wall_distance_ft
    out["projected_distance_to_wall_margin"] = margin
    out["absolute_distance_to_wall"] = margin.abs()
    return out


def _override_launch_angle_and_recompute_hang_time(
    df: pd.DataFrame, new_launch_angle_deg: float
) -> pd.DataFrame:
    """Override `launch_angle` and RECOMPUTE `estimated_hang_time_s` to match.

    `estimated_hang_time_s` is ITSELF derived from `launch_speed`/
    `launch_angle` (`mlb_luck_score.data.outfield_physics.
    estimate_hang_time_seconds`) -- overriding hang time directly (as an
    earlier version of this check did) creates the same internal
    -inconsistency bug as `_override_wall_distance_and_recompute` describes,
    just one feature removed: the model would see a hang time that
    contradicts the row's own (unchanged) launch_angle/launch_speed. This
    perturbs the genuinely independent variable (launch_angle) and
    recomputes the dependent one instead.
    """
    out = override_columns(df, {"launch_angle": new_launch_angle_deg})
    out["estimated_hang_time_s"] = [
        estimate_hang_time_seconds(speed, new_launch_angle_deg)
        for speed in out["launch_speed"].astype(float)
    ]
    return out


def run_near_wall_perturbation_checks(
    trained: TrainedOpportunityModel, near_wall_df: pd.DataFrame
) -> dict[str, DirectionalCheckResult]:
    """Controlled-perturbation directional checks for the near-wall specialist.

    1. A FARTHER wall should mean a HIGHER P(out) than a CLOSE wall (holding
       everything else fixed, including the dependent wall-margin columns --
       see `_override_wall_distance_and_recompute`) -- a ball hit the same
       distance is more likely to clear a close wall (home run, not an out)
       or bang off it for extra bases than to be a routine catch; farther
       from the wall, the same batted ball is comparatively more catchable.
    2. A LOW launch angle (short resulting hang time, recomputed via
       `_override_launch_angle_and_recompute_hang_time`) should mean a
       HIGHER P(out) than a HIGH launch angle (long hang time) -- same
       directional expectation as Version 0.7A's opportunity-time subgroup,
       reconfirmed within the near-wall-specific model.
    """
    results: dict[str, DirectionalCheckResult] = {}

    close_df = _override_wall_distance_and_recompute(near_wall_df, CLOSE_WALL_DISTANCE_FT)
    far_df = _override_wall_distance_and_recompute(near_wall_df, FAR_WALL_DISTANCE_FT)
    mean_close = _mean_predicted_p_out(trained, close_df)
    mean_far = _mean_predicted_p_out(trained, far_df)
    results["wall_distance_direction"] = DirectionalCheckResult(
        label="near-wall: wall distance close vs far",
        outcome_class="out",
        low_label=f"close wall ({CLOSE_WALL_DISTANCE_FT} ft)",
        high_label=f"far wall ({FAR_WALL_DISTANCE_FT} ft)",
        mean_prob_low=mean_close,
        mean_prob_high=mean_far,
        delta=mean_far - mean_close,
        expect_high_greater=True,
        passed=(mean_far - mean_close) > 0,
        sample_size=len(near_wall_df),
    )

    low_angle_df = _override_launch_angle_and_recompute_hang_time(
        near_wall_df, LOW_LAUNCH_ANGLE_DEG
    )
    high_angle_df = _override_launch_angle_and_recompute_hang_time(
        near_wall_df, HIGH_LAUNCH_ANGLE_DEG
    )
    mean_short = _mean_predicted_p_out(trained, low_angle_df)
    mean_long = _mean_predicted_p_out(trained, high_angle_df)
    results["hang_time_direction"] = DirectionalCheckResult(
        label="near-wall: launch angle low (short hang time) vs high (long hang time)",
        outcome_class="out",
        low_label=f"low launch angle ({LOW_LAUNCH_ANGLE_DEG} deg)",
        high_label=f"high launch angle ({HIGH_LAUNCH_ANGLE_DEG} deg)",
        mean_prob_low=mean_short,
        mean_prob_high=mean_long,
        delta=mean_long - mean_short,
        expect_high_greater=False,
        passed=(mean_long - mean_short) < 0,
        sample_size=len(near_wall_df),
    )

    return results


def compute_near_wall_paired_bootstrap(
    y_true: np.ndarray,
    baseline_p_out: pd.Series,
    candidate_p_out: pd.Series,
    game_pks: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> dict[str, dict[str, Any]]:
    """Game-level paired bootstrap: `near_wall_v07c_candidate` vs. `open_field_v07`,
    evaluated on the SAME near-wall rows -- unlike Version 0.7A (which had no baseline),
    `open_field_v07` IS a natural baseline here (what would have happened had these
    wall-adjacent plays kept using the open-field model).
    """
    y_arr = np.asarray(y_true)
    baseline_arr = baseline_p_out.to_numpy()
    candidate_arr = candidate_p_out.to_numpy()
    game_pk_arr = game_pks.to_numpy()
    unique_games = np.unique(game_pk_arr)
    game_to_rows = {g: np.where(game_pk_arr == g)[0] for g in unique_games}

    rng = np.random.default_rng(seed)
    log_loss_deltas = np.empty(n_reps)
    ece_deltas = np.empty(n_reps)
    eps = 1e-15

    def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
        p_clip = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))

    def _ece(y: np.ndarray, p: np.ndarray) -> float:
        table = compute_binary_calibration_table(y, pd.Series(p))
        return compute_binary_ece(table) if not table.empty else float("nan")

    for rep in range(n_reps):
        sampled_games = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([game_to_rows[g] for g in sampled_games])
        y_rep = y_arr[rows]
        log_loss_deltas[rep] = _log_loss(y_rep, candidate_arr[rows]) - _log_loss(
            y_rep, baseline_arr[rows]
        )
        ece_deltas[rep] = _ece(y_rep, candidate_arr[rows]) - _ece(y_rep, baseline_arr[rows])

    alpha = (1.0 - ci) / 2.0
    point_log_loss_delta = _log_loss(y_arr, candidate_arr) - _log_loss(y_arr, baseline_arr)
    point_ece_delta = _ece(y_arr, candidate_arr) - _ece(y_arr, baseline_arr)

    return {
        "log_loss_delta": {
            "point_estimate": point_log_loss_delta,
            "ci_low": float(np.nanquantile(log_loss_deltas, alpha)),
            "ci_high": float(np.nanquantile(log_loss_deltas, 1.0 - alpha)),
            "n_reps": n_reps,
            "seed": seed,
            "resampling_unit": "game_pk",
        },
        "ece_delta": {
            "point_estimate": point_ece_delta,
            "ci_low": float(np.nanquantile(ece_deltas, alpha)),
            "ci_high": float(np.nanquantile(ece_deltas, 1.0 - alpha)),
            "n_reps": n_reps,
            "seed": seed,
            "resampling_unit": "game_pk",
        },
    }


def check_no_open_field_regression(
    open_field_trained: TrainedOpportunityModel, joined_df: pd.DataFrame
) -> dict[str, Any]:
    """Verify introducing the gated architecture does not change open-field scoring.

    `open_field_v07` is `measured_contact_only_v07`, UNCHANGED -- this check confirms
    (a) predictions for gate==open_field rows are BIT-IDENTICAL whether the row-preparation
    path is `mlb_luck_score.models.compare_opportunity_models._prepare_opportunity_columns`
    (the original, standalone Version 0.7A pipeline) or THIS module's `_prepare_near_wall_
    columns` (which additionally computes gate-relevant columns) -- gating must only ROUTE
    rows, never alter the features an open-field row is scored on, and (b) `open_field_v07`'s
    calibration restricted to its own gated subset is not materially worse than the
    whole-population figure already reported in `compare_opportunity_models`.
    """
    from mlb_luck_score.models.compare_opportunity_models import _prepare_opportunity_columns

    standalone_df = _prepare_opportunity_columns(joined_df)
    standalone_df = (
        _prepare_venue_column(standalone_df)
        if "venue_id" in standalone_df.columns
        else standalone_df
    )
    standalone_eligible = standalone_df[standalone_df["outfield_opportunity_eligible"].astype(bool)]

    gated_df = _prepare_near_wall_columns(joined_df)
    gated_df = _prepare_venue_column(gated_df) if "venue_id" in gated_df.columns else gated_df
    gated_eligible = gated_df[gated_df["outfield_opportunity_eligible"].astype(bool)]
    gate = assign_outfield_opportunity_gate(gated_eligible)
    open_field_df = gated_eligible[
        (gate == GATE_OPEN_FIELD) & gated_eligible["season"].isin(CALIBRATION_EVAL_SEASONS)
    ]
    standalone_open_field_df = standalone_eligible.loc[open_field_df.index]

    feature_cols = open_field_trained.numeric_features + open_field_trained.categorical_features
    p_out_via_gate = predict_opportunity_proba(open_field_trained, open_field_df[feature_cols])
    p_out_standalone = predict_opportunity_proba(
        open_field_trained, standalone_open_field_df[feature_cols]
    )
    identical = bool(np.allclose(p_out_via_gate.to_numpy(), p_out_standalone.to_numpy(), atol=0.0))

    y_true = open_field_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    table = compute_binary_calibration_table(y_true, p_out_via_gate)
    ece = compute_binary_ece(table)
    log_loss_value = float(log_loss(y_true, p_out_via_gate.to_numpy(), labels=[0, 1]))

    return {
        "predictions_identical_via_gate": identical,
        "gated_subset_sample_count": int(len(open_field_df)),
        "gated_subset_binary_log_loss": log_loss_value,
        "gated_subset_ece": ece,
        "ece_within_threshold": ece <= MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    }


def find_material_subgroup_issues(
    subgroups: dict[str, dict[str, Any]],
    *,
    absolute_threshold: float = MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    min_sample_size: int = DEFAULT_MIN_SUBGROUP_SAMPLES,
) -> list[str]:
    flagged = []
    for label, sub in subgroups.items():
        ece = sub.get("ece")
        n = sub.get("sample_count", 0)
        if ece is None or n < min_sample_size:
            continue
        if ece > absolute_threshold:
            flagged.append(label)
    return flagged


def run_near_wall_final_comparison(
    near_wall_df: pd.DataFrame,
    open_field_trained: TrainedOpportunityModel,
    winner_variant: str,
    winner_trained: TrainedOpportunityModel,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
) -> dict[str, dict[str, Any]]:
    """Evaluate `open_field_v07` and the winning near-wall specialist on `CALIBRATION_EVAL_SEASONS`
    (2024), restricted to near-wall rows -- the "final comparison within this phase."
    """
    final_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_EVAL_SEASONS)]
    if final_df.empty:
        raise ValueError(
            f"Need non-empty final-comparison ({CALIBRATION_EVAL_SEASONS}) near-wall rows."
        )
    y_true = final_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()

    comparison: dict[str, dict[str, Any]] = {}
    for variant, trained in (
        (VARIANT_OPEN_FIELD, open_field_trained),
        (VARIANT_NEAR_WALL_FINAL, winner_trained),
    ):
        feature_cols = trained.numeric_features + trained.categorical_features
        p_out = predict_opportunity_proba(trained, final_df[feature_cols])
        validate_opportunity_probabilities(p_out)
        table = compute_binary_calibration_table(y_true, p_out)

        summary: dict[str, Any] = {
            "variant": variant,
            "underlying_model": winner_variant if variant == VARIANT_NEAR_WALL_FINAL else variant,
            "sample_count": int(len(final_df)),
            "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
            "expected_calibration_error": compute_binary_ece(table),
            "near_wall_subgroups": compute_near_wall_subgroups(y_true, p_out, final_df),
        }
        if "venue_id" in final_df.columns:
            venue_table = compute_binary_calibration_by_venue(
                y_true, p_out, final_df["venue_id"], min_reliable_samples=min_venue_samples
            )
            summary["calibration_by_venue"] = venue_table.to_dict(orient="records")
        comparison[variant] = summary

    return comparison


def summarize_near_wall_validation(
    comparison: dict[str, dict[str, Any]],
    bootstrap: dict[str, dict[str, Any]],
    perturbation_results: dict[str, DirectionalCheckResult],
    open_field_regression_check: dict[str, Any],
) -> dict[str, Any]:
    """Combine every required Version 0.7C check into one pass/fail summary.

    `near_wall_specialist_calibrated` is the single boolean `mlb_luck_score.
    scoring.gated_outfield_report` reads to decide between `"available_
    near_wall_calibrated"` and `"provisional_near_wall"`.
    """
    specialist = comparison[VARIANT_NEAR_WALL_FINAL]
    baseline = comparison[VARIANT_OPEN_FIELD]

    wall_band_issues = find_material_subgroup_issues(
        {
            k: v
            for k, v in specialist["near_wall_subgroups"].items()
            if k in ("near_wall_5ft", "near_wall_10ft", "near_wall_20ft")
        }
    )
    other_subgroup_issues = find_material_subgroup_issues(
        {
            k: v
            for k, v in specialist["near_wall_subgroups"].items()
            if k not in ("near_wall_5ft", "near_wall_10ft", "near_wall_20ft")
        }
    )
    venue_issues = [
        row["venue_id"]
        for row in specialist.get("calibration_by_venue", [])
        if row.get("reliable") and (row.get("ece") or 0) > MATERIAL_ECE_ABSOLUTE_THRESHOLD
    ]

    improves_log_loss = specialist["binary_log_loss"] < baseline["binary_log_loss"]
    bootstrap_supports = bootstrap.get("log_loss_delta", {}).get("ci_high", float("inf")) <= 0.0
    perturbation_failures = [name for name, r in perturbation_results.items() if not r.passed]
    perturbation_checks_passed = bool(perturbation_results) and not perturbation_failures

    calibrated = (
        improves_log_loss
        and bootstrap_supports
        and not wall_band_issues
        and not other_subgroup_issues
        and not venue_issues
        and perturbation_checks_passed
        and open_field_regression_check.get("predictions_identical_via_gate", False)
    )

    return {
        "improves_log_loss_vs_open_field": improves_log_loss,
        "log_loss_delta": specialist["binary_log_loss"] - baseline["binary_log_loss"],
        "bootstrap_supports_improvement": bootstrap_supports,
        "wall_band_ece_issues": wall_band_issues,
        "other_subgroup_ece_issues": other_subgroup_issues,
        "venue_ece_issues": venue_issues,
        "perturbation_checks_passed": perturbation_checks_passed,
        "perturbation_failures": perturbation_failures,
        "perturbation_detail": {k: v.__dict__ for k, v in perturbation_results.items()},
        "open_field_regression_check": open_field_regression_check,
        "near_wall_specialist_calibrated": calibrated,
        "reporting_rule": (
            "Outfield execution score available for calibrated open-field opportunities; "
            "provisional or unavailable for wall-adjacent opportunities."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with venue metadata AND park geometry",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--min-venue-samples", type=int, default=DEFAULT_MIN_VENUE_SAMPLES)
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    assert_seasons_allowed(
        CALIBRATION_BASE_TRAIN_SEASONS + CALIBRATION_FIT_SEASONS + CALIBRATION_EVAL_SEASONS
    )

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        near_wall_df = get_near_wall_rows(df)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    logger.info("Near-wall-gated rows: %d", len(near_wall_df))

    winner_variant, selection_metrics, trained_by_candidate = run_near_wall_model_selection(
        near_wall_df
    )

    # open_field_v07 == measured_contact_only_v07, trained EXACTLY as in
    # mlb_luck_score.models.compare_opportunity_models: on ALL outfield
    # -opportunity-eligible TRAIN_SEASONS (2021-2023) rows, INCLUDING
    # near-wall ones -- NOT retrained on a gate-restricted subset. Training
    # on gate==open_field rows only would silently change the model (it
    # would never see near-wall examples during fitting), which is exactly
    # the kind of accidental regression `check_no_open_field_regression` is
    # meant to catch, not cause.
    from mlb_luck_score.config import TRAIN_SEASONS
    from mlb_luck_score.models.compare_opportunity_models import _prepare_opportunity_columns
    from mlb_luck_score.models.compare_park_aware import _prepare_venue_column as _pv

    prepared = _prepare_opportunity_columns(df)
    prepared = _pv(prepared) if "venue_id" in prepared.columns else prepared
    eligible = prepared[prepared["outfield_opportunity_eligible"].astype(bool)]
    open_field_train_rows = eligible[eligible["season"].isin(TRAIN_SEASONS)]
    open_field_trained = train_opportunity_model(open_field_train_rows, class_weight=None)

    winner_trained = trained_by_candidate[winner_variant]
    comparison = run_near_wall_final_comparison(
        near_wall_df,
        open_field_trained,
        winner_variant,
        winner_trained,
        min_venue_samples=args.min_venue_samples,
    )

    final_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_EVAL_SEASONS)]
    y_true = final_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    open_field_feature_cols = (
        open_field_trained.numeric_features + open_field_trained.categorical_features
    )
    winner_feature_cols = winner_trained.numeric_features + winner_trained.categorical_features
    baseline_p_out = predict_opportunity_proba(
        open_field_trained, final_df[open_field_feature_cols]
    )
    candidate_p_out = predict_opportunity_proba(winner_trained, final_df[winner_feature_cols])

    bootstrap = compute_near_wall_paired_bootstrap(
        y_true,
        baseline_p_out,
        candidate_p_out,
        final_df["game_pk"],
        n_reps=args.n_bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    perturbation_results = run_near_wall_perturbation_checks(winner_trained, final_df)
    open_field_regression_check = check_no_open_field_regression(open_field_trained, df)

    validation_summary = summarize_near_wall_validation(
        comparison, bootstrap, perturbation_results, open_field_regression_check
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "near_wall_comparison_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "selection_metrics": selection_metrics,
                "winner_variant": winner_variant,
                "comparison": comparison,
                "bootstrap": bootstrap,
                "validation_summary": validation_summary,
            },
            indent=2,
            default=str,
        )
    )

    logger.info("Model selection metrics: %s", selection_metrics)
    logger.info("Winner: %s", winner_variant)
    for variant, summary in comparison.items():
        logger.info(
            "[%s] log_loss=%.6f ece=%.6f n=%d",
            variant,
            summary["binary_log_loss"],
            summary["expected_calibration_error"],
            summary["sample_count"],
        )
    logger.info(
        "near_wall_specialist_calibrated=%s", validation_summary["near_wall_specialist_calibrated"]
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
