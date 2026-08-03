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
alter the model or its inputs); and a controlled-perturbation PLAUSIBILITY
SUITE (see next section) -- NOT a single global launch-angle monotonicity
rule.

## Launch angle does NOT have one global expected direction near the wall

An earlier version of this module required "low launch angle (short hang
time) -> higher P(out)" as a single, GLOBAL, always-true rule -- copied
directly from the open-field intuition. That produced a genuinely backwards
-looking result even after fixing a real dependent-feature bug (see
`_override_wall_distance_and_recompute`'s docstring). On reflection this was
the WRONG kind of check to begin with, not just a buggy implementation of a
correct one: within the near-wall CONDITIONAL population, launch angle
interacts with exit velocity, projected distance, hang time, wall height,
and margin-to-wall (short of it, at it, or beyond it) -- a higher launch
angle gives more opportunity time, but can also push the projected intercept
point toward or past the wall, INTO a defender's unreachable area or over
the fence entirely. The physically sound expected direction is therefore
CONDITIONAL on where the ball is projected to land relative to the wall, not
a single global monotonic rule.

`run_near_wall_perturbation_checks` replaces that one global rule with:

  1. `wall_distance_direction` (unchanged, still REQUIRED, still global): a
     farther wall means the SAME batted ball is comparatively more
     catchable -- this direction genuinely is clear and unconditional
     across the whole near-wall population, and continues to pass cleanly
     on real 2024 data.
  2. `launch_angle_proxy_short_of_wall` (DESCRIPTIVE ONLY -- see "Launch
     angle is not the same variable as opportunity time" below for why this
     is no longer required, band-restricted to `BAND_SHORT_OF_WALL`):
     overrides ONLY `launch_angle` (via `_override_launch_angle_and_
     recompute_hang_time`) and recomputes `estimated_hang_time_s` from it,
     but leaves `launch_speed` -- and therefore the row's landing distance
     and every wall-geometry feature derived from it -- at whatever the
     REAL row happened to have. That is a proxy for "more opportunity time"
     confounded with "how hard this ball was hit to go this distance at
     this angle," not a clean test of opportunity time alone. Kept and
     renamed (from `hang_time_direction_short_of_wall`) for continuity with
     prior reporting, but demoted to descriptive: it is reported, never
     used to gate `near_wall_specialist_calibrated`.
  3. `trajectory_matched_opportunity_time_short_of_wall` (REQUIRED,
     band-restricted to `BAND_SHORT_OF_WALL`): the corrected replacement
     for check 2 as the actual required signal. Built via
     `_override_launch_angle_matched_trajectory`, which holds landing
     distance (and therefore `wall_distance_in_spray_direction`,
     `projected_distance_to_wall_margin`, `absolute_distance_to_wall`,
     `landing_x_ft`/`landing_y_ft` -- everything spray-direction/wall
     -geometry related) fixed at the row's REAL value, and instead SOLVES
     for the exit velocity (`mlb_luck_score.data.outfield_physics.
     solve_launch_speed_for_matched_range_mph`, the same vacuum projectile
     model as `estimate_hang_time_seconds`) that would still reach that
     same distance at the new launch angle. Because two trajectories are
     matched to the SAME range, the higher-angle one is guaranteed (and
     verified at runtime -- see `trajectory_match_invariants`) to have
     strictly greater estimated opportunity time. This isolates "does more
     opportunity time increase P(out), landing location and wall context
     truly held fixed" from the exit-velocity confound that made check 2
     backwards-looking, and is expected to hold: `expect_high_greater=True`
     (a NON-STRICT `>=` pass condition -- "should not reduce," not "must
     strictly increase," matching the softer physical claim that more time
     to react cannot make a play harder, though it may not measurably help
     every row).
  4. `launch_angle_proxy_effect_at_wall` / `launch_angle_proxy_effect_
     beyond_wall` (DESCRIPTIVE ONLY, band-restricted, no required sign,
     renamed from `hang_time_effect_at_wall`/`hang_time_effect_beyond_
     wall` for the same "this is a launch-angle proxy, not hang time
     itself" reason as check 2): for rows projected AT the wall
     (`BAND_AT_WALL`) or BEYOND it (`BAND_BEYOND_WALL`), the launch-angle
     effect is reported (mean P(out) at low vs. high angle, and the delta)
     but NOT required to point a specific direction -- a real interaction
     with wall clearance is plausible and expected here (see module
     docstring above), so forcing a sign would repeat the original mistake
     in a narrower disguise. The trajectory-matched construction (check 3)
     is NOT extended to these bands: for rows already at or beyond the
     wall, changing launch angle while holding distance fixed does not
     eliminate the wall-clearance confound the way it does for
     `BAND_SHORT_OF_WALL` -- a matched trajectory could cross from
     "caught" to "over the fence" purely by construction, which is exactly
     the interaction these bands are meant to surface, not average away.
  5. `compute_launch_angle_partial_dependence`: a classical partial
     -dependence curve (mean predicted P(out) across a launch-angle grid,
     globally and per band) plus a basic smoothness diagnostic (count of
     sign changes in the discrete derivative) -- flagged, not
     auto-rejected, if the curve reverses direction more than once across
     the grid, which would suggest an unstable/discontinuous fit rather
     than a genuine, single interaction effect. Meant for the SAME kind of
     manual inspection the notebook already does for calibration curves and
     example plays -- not a fully automated pass/fail gate, matching this
     project's established "physical plausibility is a judgment call, not
     100% automatable" pattern (see `mlb_luck_score.models.
     compare_geometry_aware`/`compare_weather_aware`).
  6. `compute_grouped_opportunity_time_response`: a REAL-DATA (never
     synthetic/counterfactual) grouped response curve -- rows' own natural
     `estimated_hang_time_s` binned into quartiles within each wall band and
     `bb_type` (`fly_ball`/`line_drive`) stratum, reporting mean predicted
     P(out) per bin alongside the bin's mean `hit_distance_sc` (so a reader
     can see whether landing distance is reasonably stable across bins --
     genuinely controlling for it -- or itself trends with the bin, a sign
     the curve is still entangled with distance rather than isolating
     opportunity time). DESCRIPTIVE ONLY, reported for manual review.

## Launch angle is not the same variable as opportunity time

A pure "vary opportunity time alone, holding launch angle/exit velocity/
distance/wall context fixed" perturbation is NOT physically constructible in
this codebase's feature model: `estimated_hang_time_s` is fully DETERMINED by
`launch_speed`/`launch_angle` (`estimate_hang_time_seconds`) -- there is no
free, independent lever for hang time itself. Directly overriding
`estimated_hang_time_s` while leaving `launch_speed`/`launch_angle` at their
real values would recreate the exact internal-inconsistency bug class
`_override_launch_angle_and_recompute_hang_time`'s docstring (and CLAUDE.md)
already warns about, just with the override and the raw feature swapped.
Rather than build that invalid row, two real substitutes are used instead,
per the design above: (a) the trajectory-matched check (3), which varies
hang time only THROUGH a physically valid, jointly-solved change to launch
angle AND exit velocity, and (b) the grouped real-data response curve (6),
which uses REAL rows' own natural hang-time variation rather than any
synthetic override at all.

`near_wall_specialist_calibrated` (see `summarize_near_wall_validation`)
requires checks 1 and 3 to pass; checks 2, 4, 5, and 6 are reported for
review but do not block it -- they exist to distinguish a real, interesting
interaction (or a genuine confound) from a broken counterfactual, which is
exactly what they're for and not something a single fixed threshold could
safely automate. The ORIGINAL required check (2, then named `hang_time_
direction_short_of_wall`) is deliberately no longer, by itself, load-bearing
for calibration -- see "Launch angle is not the same variable as opportunity
time" above for why it was demoted rather than simply sign-flipped.

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
from dataclasses import dataclass, field
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
from mlb_luck_score.data.outfield_physics import (
    estimate_hang_time_seconds,
    solve_launch_speed_for_matched_range_mph,
)
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

#: Wall-intercept bands -- see `classify_wall_intercept_band`. `AT_WALL_
#: TOLERANCE_FT` (10 ft) matches the existing `near_wall_10ft` convention
#: (`mlb_luck_score.data.join_park_geometry`) rather than inventing a new
#: threshold; chosen empirically to give a reasonably balanced 3-way split
#: on real 2021-2024 near-wall data (short=9,924 / at=15,863 /
#: beyond=5,718 of 31,505 rows).
AT_WALL_TOLERANCE_FT = 10.0
BAND_SHORT_OF_WALL = "short_of_wall"
BAND_AT_WALL = "at_wall"
BAND_BEYOND_WALL = "beyond_wall"
ALL_WALL_INTERCEPT_BANDS: tuple[str, ...] = (BAND_SHORT_OF_WALL, BAND_AT_WALL, BAND_BEYOND_WALL)

#: Grid for the launch-angle partial-dependence diagnostic -- spans the
#: real observed near-wall launch-angle range (12-54 deg).
LAUNCH_ANGLE_PDP_GRID = tuple(np.linspace(12.0, 54.0, 15))
#: More than this many direction reversals in the discrete derivative of
#: the PDP curve is flagged as potentially unstable/discontinuous rather
#: than a genuine single interaction effect. A documented Version 0.7C
#: research placeholder, not a validated threshold -- see module docstring.
MAX_PLAUSIBLE_PDP_SIGN_CHANGES = 1
#: A wall-intercept band needs at least this many rows before its
#: launch-angle effect (required or descriptive) is computed at all --
#: matches `DEFAULT_MIN_SUBGROUP_SAMPLES` for consistency with every other
#: subgroup-reliability threshold in this module.
MIN_BAND_SAMPLE_SIZE = DEFAULT_MIN_SUBGROUP_SAMPLES


def classify_wall_intercept_band(
    margin: pd.Series, *, tolerance_ft: float = AT_WALL_TOLERANCE_FT
) -> pd.Series:
    """Classify each row's projected landing point relative to the wall.

    `margin` is `projected_distance_to_wall_margin` (`mlb_luck_score.data.
    join_park_geometry`: `hit_distance_sc - wall_distance_in_spray_
    direction`) -- positive means the ball is projected to land BEYOND the
    wall (home-run territory), negative means SHORT of it (in the
    fielder's reachable area, at least in terms of raw distance).

    Returns a `str`-valued Series: `BAND_SHORT_OF_WALL` (`margin <
    -tolerance_ft`), `BAND_AT_WALL` (`-tolerance_ft <= margin <=
    tolerance_ft`), or `BAND_BEYOND_WALL` (`margin > tolerance_ft`).
    """
    band = pd.Series(BAND_AT_WALL, index=margin.index, dtype=object)
    band.loc[margin < -tolerance_ft] = BAND_SHORT_OF_WALL
    band.loc[margin > tolerance_ft] = BAND_BEYOND_WALL
    return band


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

    NOTE this is a LAUNCH-ANGLE PROXY, not a clean opportunity-time
    perturbation: `launch_speed` (and therefore the row's landing distance
    and every wall-geometry feature derived from it) is left at the REAL
    row's value, so the counterfactual ball silently lands somewhere
    different than the real one did. See `_override_launch_angle_matched_
    trajectory` for the corrected, distance-preserving alternative, and
    module docstring "Launch angle is not the same variable as opportunity
    time" for why both are kept (one descriptive, one required).
    """
    out = override_columns(df, {"launch_angle": new_launch_angle_deg})
    out["estimated_hang_time_s"] = [
        estimate_hang_time_seconds(speed, new_launch_angle_deg)
        for speed in out["launch_speed"].astype(float)
    ]
    return out


def _override_launch_angle_matched_trajectory(
    df: pd.DataFrame, new_launch_angle_deg: float
) -> pd.DataFrame:
    """Build a TRAJECTORY-MATCHED counterfactual: same landing distance (and
    therefore the same spray-direction wall geometry) as the real row, but a
    different launch angle, with exit velocity SOLVED (`mlb_luck_score.data.
    outfield_physics.solve_launch_speed_for_matched_range_mph`, the same
    vacuum projectile model as `estimate_hang_time_seconds`) so the ball
    still reaches the row's REAL `hit_distance_sc`.

    Unlike `_override_launch_angle_and_recompute_hang_time` (which changes
    launch angle while implicitly leaving exit velocity -- and therefore the
    landing point -- at whatever the real row had), this holds `hit_
    distance_sc` fixed, so `landing_x_ft`/`landing_y_ft`, `wall_distance_in_
    spray_direction`, `projected_distance_to_wall_margin`, and `absolute_
    distance_to_wall` (all derived from distance/spray angle, NOT from
    launch angle or exit velocity -- see `mlb_luck_score.data.
    outfield_physics`/`mlb_luck_score.data.join_park_geometry`) are correct
    and unchanged with no recomputation needed. Only `launch_speed`,
    `launch_angle`, and `estimated_hang_time_s` (derived from the first two)
    change.
    """
    out = override_columns(df, {"launch_angle": new_launch_angle_deg})
    solved_speed = [
        solve_launch_speed_for_matched_range_mph(distance, new_launch_angle_deg)
        for distance in out["hit_distance_sc"].astype(float)
    ]
    out["launch_speed"] = solved_speed
    out["estimated_hang_time_s"] = [
        estimate_hang_time_seconds(speed, new_launch_angle_deg) for speed in solved_speed
    ]
    return out


@dataclass(frozen=True)
class BandedEffectResult:
    """A launch-angle low-vs-high effect reported WITHOUT a required sign.

    Used for `BAND_AT_WALL`/`BAND_BEYOND_WALL`, where the physically
    plausible direction genuinely depends on the interaction of launch
    angle with exit velocity, wall height, and margin-to-wall -- see module
    docstring "Launch angle does NOT have one global expected direction
    near the wall". Reported for manual review, never auto-passed or
    auto-failed.
    """

    band: str
    outcome_class: str
    low_label: str
    high_label: str
    mean_prob_low: float
    mean_prob_high: float
    delta: float
    sample_size: int


@dataclass(frozen=True)
class PartialDependenceDiagnostic:
    """Mean predicted P(out) across a launch-angle grid, plus a smoothness diagnostic.

    `n_sign_changes` counts direction reversals in the discrete derivative
    of the curve -- more than `MAX_PLAUSIBLE_PDP_SIGN_CHANGES` is flagged
    (`flagged_as_erratic`) as POSSIBLY indicating an unstable/discontinuous
    fit rather than a genuine single interaction effect. This is a
    diagnostic for manual inspection (see module docstring), not an
    automated pass/fail gate.
    """

    band: str
    grid: list[float]
    mean_p_out: list[float]
    n_sign_changes: int
    flagged_as_erratic: bool


@dataclass(frozen=True)
class TrajectoryMatchInvariantCheck:
    """Runtime verification that a matched-trajectory pair actually has the
    opportunity-time relationship it's supposed to by construction.

    `_override_launch_angle_matched_trajectory` solves exit velocity so two
    trajectories reach the SAME distance -- which mathematically guarantees
    (for the vacuum projectile model, `0 < angle < 90`) that the higher
    -angle trajectory has strictly greater estimated hang time. This records
    that verification rather than silently assuming it -- `run_near_wall_
    perturbation_checks` raises if `verified` would be `False`, since that
    would indicate a bug in the trajectory-matching itself, not a genuine
    physical finding worth reporting.
    """

    band: str
    mean_hang_time_low_angle_s: float
    mean_hang_time_high_angle_s: float
    verified: bool


@dataclass(frozen=True)
class OpportunityTimeResponseBin:
    """One bin of a REAL-DATA (never synthetic/counterfactual) grouped response
    curve for `estimated_hang_time_s`, within a wall-intercept band and `bb_type`.

    Uses rows' own natural variation in hang time (binned into quartiles via
    `mlb_luck_score.models.compare_opportunity_models.
    compute_opportunity_time_buckets`), never an override -- see module
    docstring "Launch angle is not the same variable as opportunity time".
    `mean_hit_distance_ft` is reported alongside so a reader can check
    whether landing distance is reasonably stable across bins (genuinely
    controlling for it) or itself trends with the bin (a sign this curve is
    still entangled with distance, not a clean isolated opportunity-time
    effect). Descriptive only -- never gates `near_wall_specialist_
    calibrated`.
    """

    band: str
    bb_type: str
    bucket: str
    mean_estimated_hang_time_s: float
    mean_hit_distance_ft: float
    mean_predicted_p_out: float
    sample_size: int


@dataclass(frozen=True)
class NearWallPerturbationSuite:
    """The full Version 0.7C plausibility suite -- required checks, descriptive
    banded effects, partial-dependence diagnostics, trajectory-match runtime
    verifications, and real-data grouped opportunity-time response curves.
    See module docstring.
    """

    required: dict[str, DirectionalCheckResult]
    descriptive: dict[str, BandedEffectResult]
    partial_dependence: dict[str, PartialDependenceDiagnostic]
    trajectory_match_invariants: dict[str, TrajectoryMatchInvariantCheck] = field(
        default_factory=dict
    )
    opportunity_time_response: list[OpportunityTimeResponseBin] = field(default_factory=list)


def compute_launch_angle_partial_dependence(
    trained: TrainedOpportunityModel,
    df: pd.DataFrame,
    *,
    grid: tuple[float, ...] = LAUNCH_ANGLE_PDP_GRID,
    band: str = "all",
) -> PartialDependenceDiagnostic:
    """Classical partial-dependence curve for launch angle, over `df`.

    For each grid value, overrides EVERY row's launch angle to that value
    (recomputing `estimated_hang_time_s` -- see `_override_launch_angle_
    and_recompute_hang_time`) and takes the mean predicted P(out) -- the
    standard PDP definition (average marginal effect holding the joint
    distribution of every other feature fixed at its real values).
    """
    mean_p_out = [
        _mean_predicted_p_out(trained, _override_launch_angle_and_recompute_hang_time(df, angle))
        for angle in grid
    ]
    diffs = np.diff(mean_p_out)
    nonzero_diffs = diffs[diffs != 0]
    sign_changes = (
        int(np.sum(np.diff(np.sign(nonzero_diffs)) != 0)) if len(nonzero_diffs) > 1 else 0
    )
    return PartialDependenceDiagnostic(
        band=band,
        grid=list(grid),
        mean_p_out=mean_p_out,
        n_sign_changes=sign_changes,
        flagged_as_erratic=sign_changes > MAX_PLAUSIBLE_PDP_SIGN_CHANGES,
    )


#: `bb_type` values eligible for the near-wall gate -- see `mlb_luck_score.
#: eligibility.OUTFIELD_AIR_BALL_TYPES`. Used to stratify the grouped
#: opportunity-time response curve (`compute_grouped_opportunity_time_
#: response`) so a batted-ball-type mix shift can't masquerade as an
#: opportunity-time effect.
_NEAR_WALL_BB_TYPES: tuple[str, ...] = ("fly_ball", "line_drive")
_OPPORTUNITY_TIME_BUCKET_LABELS: tuple[str, ...] = ("q1_shortest", "q2", "q3", "q4_longest")


def compute_grouped_opportunity_time_response(
    trained: TrainedOpportunityModel, near_wall_df: pd.DataFrame
) -> list[OpportunityTimeResponseBin]:
    """REAL-DATA grouped response curve: mean predicted P(out) by `estimated_
    hang_time_s` quartile, within each wall-intercept band and `bb_type`.

    Never overrides or synthesizes a row -- bins rows by their OWN natural
    hang-time variation (`compute_opportunity_time_buckets`) within each
    (wall band, bb_type) stratum, and reports each bin's mean predicted
    P(out) alongside its mean REAL `hit_distance_sc` -- the latter lets a
    reader check whether distance is genuinely comparable across bins
    (stratifying/controlling for it, per module docstring "Launch angle is
    not the same variable as opportunity time") rather than confounding the
    curve. Descriptive only.
    """
    band = classify_wall_intercept_band(near_wall_df["projected_distance_to_wall_margin"])
    feature_cols = trained.numeric_features + trained.categorical_features
    results: list[OpportunityTimeResponseBin] = []

    for wall_band in ALL_WALL_INTERCEPT_BANDS:
        band_df = near_wall_df[(band == wall_band).to_numpy()]
        for bb_type in _NEAR_WALL_BB_TYPES:
            subset = band_df[band_df["bb_type"] == bb_type]
            if len(subset) < MIN_BAND_SAMPLE_SIZE:
                continue
            buckets = compute_opportunity_time_buckets(subset["estimated_hang_time_s"])
            p_out = predict_opportunity_proba(trained, subset[feature_cols])
            for bucket_label in _OPPORTUNITY_TIME_BUCKET_LABELS:
                mask = (buckets == bucket_label).to_numpy()
                n = int(mask.sum())
                if n == 0:
                    continue
                results.append(
                    OpportunityTimeResponseBin(
                        band=wall_band,
                        bb_type=bb_type,
                        bucket=bucket_label,
                        mean_estimated_hang_time_s=float(
                            subset["estimated_hang_time_s"].to_numpy()[mask].mean()
                        ),
                        mean_hit_distance_ft=float(
                            subset["hit_distance_sc"].to_numpy()[mask].mean()
                        ),
                        mean_predicted_p_out=float(p_out.to_numpy()[mask].mean()),
                        sample_size=n,
                    )
                )
    return results


def run_near_wall_perturbation_checks(
    trained: TrainedOpportunityModel, near_wall_df: pd.DataFrame
) -> NearWallPerturbationSuite:
    """The Version 0.7C controlled-perturbation plausibility suite -- see module docstring
    "Launch angle does NOT have one global expected direction near the wall" and "Launch
    angle is not the same variable as opportunity time" for the full design and history.

    Required (gate `near_wall_specialist_calibrated`):
      1. `wall_distance_direction`: a FARTHER wall should mean a HIGHER
         P(out) than a CLOSE wall (holding everything else fixed, including
         the dependent wall-margin columns -- see `_override_wall_distance_
         and_recompute`) -- unconditionally clear across the whole near
         -wall population.
      2. `trajectory_matched_opportunity_time_short_of_wall`: among rows
         whose OWN projected landing point is clearly short of the wall
         (`BAND_SHORT_OF_WALL`), a trajectory matched to the SAME landing
         distance but a HIGHER launch angle (exit velocity solved via
         `_override_launch_angle_matched_trajectory` -- strictly more
         estimated opportunity time by construction, verified via
         `trajectory_match_invariants`) should not have a LOWER P(out) than
         the low-angle matched trajectory.

    Descriptive only (reported, never gating):
      3. `launch_angle_proxy_short_of_wall`: the ORIGINAL required check
         (renamed from `hang_time_direction_short_of_wall`), which varies
         launch angle WITHOUT matching exit velocity/distance -- confounded
         with "how hard the ball was hit to go this far at this angle," per
         module docstring, so demoted to descriptive.
      4. `launch_angle_proxy_effect_at_wall` / `launch_angle_proxy_effect_
         beyond_wall`: the SAME (unmatched) low-vs-high launch-angle
         comparison, restricted to `BAND_AT_WALL`/`BAND_BEYOND_WALL` rows,
         with NO required sign -- see module docstring for why a real
         interaction is plausible here.
      5. `partial_dependence`: `compute_launch_angle_partial_dependence`,
         globally and per band.
      6. `opportunity_time_response`: `compute_grouped_opportunity_time_
         response`, real-data hang-time-quartile response curves by band and
         `bb_type`.
    """
    required: dict[str, DirectionalCheckResult] = {}
    descriptive: dict[str, BandedEffectResult] = {}
    trajectory_match_invariants: dict[str, TrajectoryMatchInvariantCheck] = {}

    close_df = _override_wall_distance_and_recompute(near_wall_df, CLOSE_WALL_DISTANCE_FT)
    far_df = _override_wall_distance_and_recompute(near_wall_df, FAR_WALL_DISTANCE_FT)
    mean_close = _mean_predicted_p_out(trained, close_df)
    mean_far = _mean_predicted_p_out(trained, far_df)
    required["wall_distance_direction"] = DirectionalCheckResult(
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

    band = classify_wall_intercept_band(near_wall_df["projected_distance_to_wall_margin"])
    band_masks = {b: (band == b).to_numpy() for b in ALL_WALL_INTERCEPT_BANDS}

    for wall_band in ALL_WALL_INTERCEPT_BANDS:
        mask = band_masks[wall_band]
        n = int(mask.sum())
        if n < MIN_BAND_SAMPLE_SIZE:
            continue
        subset = near_wall_df[mask]
        low_angle_df = _override_launch_angle_and_recompute_hang_time(subset, LOW_LAUNCH_ANGLE_DEG)
        high_angle_df = _override_launch_angle_and_recompute_hang_time(
            subset, HIGH_LAUNCH_ANGLE_DEG
        )
        mean_low = _mean_predicted_p_out(trained, low_angle_df)
        mean_high = _mean_predicted_p_out(trained, high_angle_df)

        if wall_band == BAND_SHORT_OF_WALL:
            descriptive["launch_angle_proxy_short_of_wall"] = BandedEffectResult(
                band=wall_band,
                outcome_class="out",
                low_label=f"low launch angle ({LOW_LAUNCH_ANGLE_DEG} deg), unmatched distance",
                high_label=f"high launch angle ({HIGH_LAUNCH_ANGLE_DEG} deg), unmatched distance",
                mean_prob_low=mean_low,
                mean_prob_high=mean_high,
                delta=mean_high - mean_low,
                sample_size=n,
            )

            matched_low_df = _override_launch_angle_matched_trajectory(subset, LOW_LAUNCH_ANGLE_DEG)
            matched_high_df = _override_launch_angle_matched_trajectory(
                subset, HIGH_LAUNCH_ANGLE_DEG
            )
            mean_hang_low = float(matched_low_df["estimated_hang_time_s"].mean())
            mean_hang_high = float(matched_high_df["estimated_hang_time_s"].mean())
            verified = mean_hang_high > mean_hang_low
            if not verified:
                raise RuntimeError(
                    "Trajectory-matched invariant violated for band "
                    f"{wall_band!r}: the higher-angle matched trajectory "
                    f"(mean hang time {mean_hang_high:.4f}s) did not have "
                    f"strictly greater estimated opportunity time than the "
                    f"lower-angle one ({mean_hang_low:.4f}s). This indicates "
                    "a bug in _override_launch_angle_matched_trajectory or "
                    "solve_launch_speed_for_matched_range_mph, not a "
                    "genuine physical finding."
                )
            trajectory_match_invariants[wall_band] = TrajectoryMatchInvariantCheck(
                band=wall_band,
                mean_hang_time_low_angle_s=mean_hang_low,
                mean_hang_time_high_angle_s=mean_hang_high,
                verified=verified,
            )

            mean_matched_low = _mean_predicted_p_out(trained, matched_low_df)
            mean_matched_high = _mean_predicted_p_out(trained, matched_high_df)
            required["trajectory_matched_opportunity_time_short_of_wall"] = DirectionalCheckResult(
                label=(
                    f"near-wall ({wall_band}): trajectory-matched launch angle "
                    "low vs high (landing distance/wall context held fixed, "
                    "exit velocity solved to match)"
                ),
                outcome_class="out",
                low_label=(
                    f"matched trajectory, low angle ({LOW_LAUNCH_ANGLE_DEG} deg, "
                    f"mean opportunity time {mean_hang_low:.2f}s)"
                ),
                high_label=(
                    f"matched trajectory, high angle ({HIGH_LAUNCH_ANGLE_DEG} deg, "
                    f"mean opportunity time {mean_hang_high:.2f}s)"
                ),
                mean_prob_low=mean_matched_low,
                mean_prob_high=mean_matched_high,
                delta=mean_matched_high - mean_matched_low,
                expect_high_greater=True,
                passed=(mean_matched_high - mean_matched_low) >= 0,
                sample_size=n,
            )
        else:
            descriptive[f"launch_angle_proxy_effect_{wall_band}"] = BandedEffectResult(
                band=wall_band,
                outcome_class="out",
                low_label=f"low launch angle ({LOW_LAUNCH_ANGLE_DEG} deg)",
                high_label=f"high launch angle ({HIGH_LAUNCH_ANGLE_DEG} deg)",
                mean_prob_low=mean_low,
                mean_prob_high=mean_high,
                delta=mean_high - mean_low,
                sample_size=n,
            )

    partial_dependence = {
        "all": compute_launch_angle_partial_dependence(trained, near_wall_df, band="all")
    }
    for wall_band in ALL_WALL_INTERCEPT_BANDS:
        mask = band_masks[wall_band]
        if int(mask.sum()) < MIN_BAND_SAMPLE_SIZE:
            continue
        partial_dependence[wall_band] = compute_launch_angle_partial_dependence(
            trained, near_wall_df[mask], band=wall_band
        )

    opportunity_time_response = compute_grouped_opportunity_time_response(trained, near_wall_df)

    return NearWallPerturbationSuite(
        required=required,
        descriptive=descriptive,
        partial_dependence=partial_dependence,
        trajectory_match_invariants=trajectory_match_invariants,
        opportunity_time_response=opportunity_time_response,
    )


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
    perturbation_suite: NearWallPerturbationSuite,
    open_field_regression_check: dict[str, Any],
) -> dict[str, Any]:
    """Combine every required Version 0.7C check into one pass/fail summary.

    `near_wall_specialist_calibrated` is the single boolean `mlb_luck_score.
    scoring.gated_outfield_report` reads to decide between `"available_
    near_wall_calibrated"` and `"provisional_near_wall"`. Only `perturbation_
    suite.required` (`wall_distance_direction`, `trajectory_matched_
    opportunity_time_short_of_wall`) gates this -- `perturbation_suite.
    descriptive` (the launch-angle-proxy banded effects, including the
    demoted original `launch_angle_proxy_short_of_wall` check), `.partial_
    dependence`, `.trajectory_match_invariants`, and `.opportunity_time_
    response` are all reported for manual review but never block it, per
    module docstring "Launch angle does NOT have one global expected
    direction near the wall" / "Launch angle is not the same variable as
    opportunity time".
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
    perturbation_failures = [
        name for name, r in perturbation_suite.required.items() if not r.passed
    ]
    perturbation_checks_passed = bool(perturbation_suite.required) and not perturbation_failures
    pdp_flags = [
        band
        for band, diag in perturbation_suite.partial_dependence.items()
        if diag.flagged_as_erratic
    ]

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
        "perturbation_required_detail": {
            k: v.__dict__ for k, v in perturbation_suite.required.items()
        },
        "perturbation_descriptive_detail": {
            k: v.__dict__ for k, v in perturbation_suite.descriptive.items()
        },
        "partial_dependence_flags": pdp_flags,
        "partial_dependence_detail": {
            k: v.__dict__ for k, v in perturbation_suite.partial_dependence.items()
        },
        "trajectory_match_invariants_detail": {
            k: v.__dict__ for k, v in perturbation_suite.trajectory_match_invariants.items()
        },
        "opportunity_time_response_detail": [
            b.__dict__ for b in perturbation_suite.opportunity_time_response
        ],
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
    perturbation_suite = run_near_wall_perturbation_checks(winner_trained, final_df)
    open_field_regression_check = check_no_open_field_regression(open_field_trained, df)

    validation_summary = summarize_near_wall_validation(
        comparison, bootstrap, perturbation_suite, open_field_regression_check
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
