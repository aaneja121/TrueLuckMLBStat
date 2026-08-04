"""Contact Luck v0.8: infield opportunity model and combined infield execution.

Version 0.7 is FROZEN (see CLAUDE.md/AGENTS.md "Version 0.7 is now FROZEN") --
nothing in this module touches `mlb_luck_score.models.compare_near_wall_models`/
`compare_near_wall_calibration_gate`'s model, feature set, calibration gate,
predictions, or reporting rule. It REUSES that generation's generic,
already-established building blocks (the v0.7D sample-size-aware calibration
-gate machinery, `mlb_luck_score.models.weather_perturbation`'s directional
-check primitives, the shared `mlb_luck_score.models.train_opportunity_model`
trainer) exactly as those were designed to be reused across opportunity-style
binary models -- it does not modify any of them.

## Phase 1: public-data audit

Audited the real, already-downloaded 2021-2024 cleaned/joined dataset and the
installed `pybaseball` package for every field the task asked about:

  - **Responsible fielder identity**: AVAILABLE. `fielder_2`.."fielder_6"
    (2B/3B/SS/1B/... -- the specific player ID at each position for that
    exact play) plus `pitcher` for position 1 (there is NO separate
    `fielder_1` Statcast column -- the pitcher IS fielder 1). Used ONLY for
    post-hoc individual-defender evaluation, never as a model input (task
    Phase 4).
  - **Responsible fielding position**: AVAILABLE via `hit_location` (1=P,
    2=C, 3=1B, 4=2B, 5=3B, 6=SS) -- real Statcast fielding credit, 99.98%
    coverage on ground balls (38 of 214,032 missing).
  - **`hit_location`**: AVAILABLE, see above.
  - **Batter sprint speed**: NOT a per-pitch Statcast field at all --
    AVAILABLE ONLY as a separate, SEASON-LEVEL public leaderboard
    (`pybaseball.statcast_sprint_speed`, Baseball Savant's own "Sprint
    Speed," ft/sec). Downloaded (`mlb_luck_score.data.download_sprint_speed`,
    2021-2024 only, small leaderboard files, ~560-586 qualified batters per
    season) and joined by `(batter, season)`
    (`mlb_luck_score.data.join_sprint_speed`) -- 98.89% row-level match rate
    on infield-eligible rows (906 of 1,337 unique batters matched; the
    unmatched batters are disproportionately low-PA players below Savant's
    own `min_opp=10` qualification threshold -- Savant's own documented
    limitation, not something this pipeline can work around). `hp_to_1b`
    (home-to-first time, seconds) is ALSO available from the same
    leaderboard and downloaded alongside `sprint_speed`, but not used as a
    separate model feature (redundant/collinear with `sprint_speed` as a
    runner-speed signal -- one clearly documented season-level speed measure
    is what the task asks for, not two).
  - **Infield alignment**: AVAILABLE via `if_fielding_alignment` (Standard/
    Infield shift/Infield shade/Strategic), the SAME column Version 0.6
    (`mlb_luck_score.models.compare_alignment_aware`) already uses -- 99.6%
    coverage on ground balls.
  - **Batted-ball coordinates**: AVAILABLE (`hc_x`/`hc_y`, `spray_angle_
    approx`/`spray_sector` -- the SAME community-approximation conversion
    already used and documented throughout this codebase, e.g.
    `mlb_luck_score.data.clean_batted_balls`).
  - **Estimated landing/fielding location**: `hit_distance_sc` (99.4%
    coverage on ground balls) is Statcast's OWN measured distance from home
    plate to where the ball was ACTUALLY fielded -- a real measured
    quantity, not an estimate this codebase has to construct (unlike the
    outfield model's physics-estimated hang time/landing coordinates, which
    exist because outfield balls are airborne and Statcast doesn't measure
    where they land; ground balls are fielded at a real, recorded distance).
  - **Throws or assists / errors / fielder's choices / force outs / double
    plays / bunts**: all encoded in `events` (a controlled taxonomy: the
    complete list of values ever observed for `bb_type == "ground_ball"` in
    real 2021-2024 data is exactly `{field_out, single, double, triple,
    field_error, force_out, fielders_choice, fielders_choice_out,
    double_play, grounded_into_double_play, sac_bunt}` -- 11 values, no
    `triple_play`, no null `events`) EXCEPT bunts, which have NO dedicated
    `bb_type`/`events` flag at all and must be detected via a `des`
    free-text keyword match (see `mlb_luck_score.eligibility`'s Version 0.8
    section for the exact keyword and real counts: 4,393 of 214,032 ground
    balls mention "bunt," of which 2,364 have an `events` value OTHER than
    `sac_bunt` -- bunt singles/outs/force-outs invisible to any other
    signal).
  - **Errors**: `events == "field_error"` is a controlled, reliable flag
    (used for the `reached_on_error` REPORTING indicator only -- task Phase
    3 explicitly forbids using it as a predictor).
  - **Fielder's choices / force outs / double plays**: all controlled
    `events` values (see above) -- reliably excludable, no ambiguity.
  - **Reviewed or overturned outcomes**: NOT a dedicated column -- inferred
    from `des` text ("challenged"/"overturned"/"review"). 1,199 of 143,377
    eligible plays (0.84%) match. NOT excluded (Statcast's recorded
    `events`/`des` already reflect the FINAL, corrected ruling after any
    review, so the label itself is not made less reliable by having been
    reviewed) -- reported informationally only. See `mlb_luck_score.
    eligibility.add_infield_opportunity_eligibility`'s `reviewed_or_
    overturned` column docstring.
  - **Interference/obstruction/catcher interference/appeal/rundown plays**:
    NO dedicated columns -- all `des`-text keyword matches. Real counts on
    2021-2024 ground balls: "interference" or "obstruction" 113 times
    (BOTH directions matter and neither is visible from `events` alone: 64
    of these are "batter interference" outs recorded as an ordinary
    `field_out` -- a rules-violation penalty, NOT a real defensive
    conversion -- and several are "fan interference" hits recorded as an
    ordinary `single`/`double`/`triple` -- an external event, NOT a real
    defensive failure); "appeal" 0 times; "rundown" 0 times; "catcher
    interference" cannot co-occur with a batted ball by construction
    (Statcast's `catcher_interf` event has no batted-ball trajectory --
    verified zero co-occurrence with `bb_type == "ground_ball"`).

  **UNSAFE TO USE / NOT ASSUMED** (per the task's explicit instruction):
  exact defender starting coordinates, route distance, exchange/transfer
  time, throw velocity, and release location are NOT directly measured
  anywhere in public Statcast data and are NEVER assumed, invented, or
  backed out from other fields in this module.

## Phase 4: Candidate B was investigated and NOT built

`infield_time_margin_proxy_v08_candidate` requires SOME estimate of "how long
does the ball take to reach a fielder/complete the throw" to compare against
the batter-runner's own speed. Unlike the outfield model's hang-time estimate
(`mlb_luck_score.data.outfield_physics.estimate_hang_time_seconds` -- a real,
citable vacuum-projectile-motion formula, Newtonian mechanics, missing only
air resistance), there is NO comparable citable physics for ground-ball ROLL
deceleration: a ball rolling/bouncing on grass or turf loses speed to
friction/bouncing in a way that has no standard closed-form equation this
project can point to, and any such model would require assuming an
unvalidated friction/restitution coefficient -- exactly the kind of invented
assumption the task instructs against, one level removed from "invent a
defender's starting position." `hit_distance_sc` gives the REAL distance the
ball traveled before being fielded, and `sprint_speed`/`hp_to_1b` give a REAL
season-level runner-speed measure, but there is no defensible way to combine
them into a genuine "time margin" without inventing the missing physics.
Per the task's explicit permission ("do not build Candidate B merely for
completeness. Skip it if the required physics or public inputs are too
weak"), Candidate B is SKIPPED -- matching the Version 0.7A precedent of
investigating and not building `typical_position_proxy_v07` for a
different, but analogous, reason (no era-appropriate public data existed
there; no defensible physics exists here).

## Phase 5: model-CLASS comparison (not feature-set comparison)

With only one feature-set candidate (`infield_contact_only_v08`), the
Phase 5 "compare a transparent logistic model with at most one justified
nonlinear candidate" instruction becomes a model-CLASS selection on the
IDENTICAL feature set -- exactly `mlb_luck_score.models.
compare_near_wall_models`'s `near_wall_logistic_v07c` vs. `near_wall_hgb_v07c`
pattern (`LogisticRegression` vs. unweighted `HistGradientBoostingClassifier`,
both `class_weight=None`). Fit on 2021-2022, selected on 2023 by log loss
(tie-broken by ECE), final comparison on 2024. 2025 remains untouched
throughout (`assert_seasons_allowed`).

## Phase 6: absolute-quality bar, declared BEFORE examining 2024 results

Like Version 0.7A (`measured_contact_only_v07`), this is a genuinely NEW
opportunity-difficulty category with no prior public per-play infield
-opportunity baseline to compare against -- CLAUDE.md/AGENTS.md's "a model
with no prior baseline needs an ABSOLUTE quality bar, not a relative one"
applies directly. The bar is the SAME `MATERIAL_ECE_ABSOLUTE_THRESHOLD`
(0.05) already established and reused across Version 0.7A/0.7C/0.7D -- NOT a
new number chosen after looking at infield 2024 results. This module's
subgroup/venue calibration gate REUSES Version 0.7D's sample-size-aware
three-status framework (`mlb_luck_score.models.
compare_near_wall_calibration_gate`) from the start, per the task's explicit
instruction -- never a bare point-estimate ECE threshold. Because there is no
prior baseline model, the reused machinery's PAIRED-comparison slot is filled
with the candidate's own predictions (`baseline_p_out = specialist_p_out`),
which makes the paired delta identically zero and NEVER triggers a "credible
regression" finding -- this leaves only the absolute-ECE-confidence-interval
criterion active, exactly matching the "absolute bar, not relative" rule
rather than inventing a placeholder baseline to diff against.

## Phase 7: physical/logical perturbation checks

  1. `sprint_speed_direction` (REQUIRED): among genuinely otherwise-identical
     batted balls (same contact characteristics, estimated location, fielding
     position, and alignment -- `_override_sprint_speed` changes ONLY
     `sprint_speed`, which has NO derived dependent features anywhere in this
     feature set, so no recomputation is needed), a FASTER batter must not
     show a HIGHER retirement probability than a slower one.
  2. Timing-margin direction check: NOT APPLICABLE -- no timing-margin
     feature was built (Candidate B skipped, see above). Reported as
     `not_applicable`, not silently omitted.
  3. Recompute derived features: trivially satisfied -- NONE of this
     module's numeric features (`launch_speed`, `launch_angle`,
     `spray_angle_approx`, `hit_distance_sc`, `sprint_speed`, `outs_when_up`,
     `on_1b_occupied`) is derived FROM another feature in this set (unlike
     the outfield model's `estimated_hang_time_s`/`landing_x_ft`/
     `landing_y_ft`, which ARE derived from launch speed/angle/distance).
     Perturbing `sprint_speed` therefore cannot create the "stale derived
     feature" bug class documented in CLAUDE.md/AGENTS.md ("Any override of a
     raw feature must recompute every feature DERIVED from it") because
     there is nothing derived from it to go stale.
  4. Exit velocity has NO required monotonic direction (`launch_speed`
     partial dependence, `compute_launch_speed_partial_dependence`, is
     DESCRIPTIVE ONLY) -- per the task's explicit reasoning, a harder-hit
     ground ball can reduce a fielder's reaction time (harder to field
     cleanly) but can also reach a well-positioned defender before it has a
     chance to take a bad hop, so no single global sign is physically
     required.
  5. A REAL-DATA (never synthetic) grouped response curve for `sprint_speed`
     by `assigned_infield_position` (`compute_grouped_sprint_speed_response`)
     corroborates the required perturbation check using rows' own natural
     variation, the SAME "matched real-row" pattern
     `compare_near_wall_models.compute_grouped_opportunity_time_response`
     uses for opportunity time.

## Phase 8: combined infield execution (see `mlb_luck_score.scoring.
infield_execution` for the exact formulas) -- NOT yet claimed to isolate
fielding from throwing, positioning, scorer judgment, or unobserved route
quality; see that module's docstring.

Usage:

    python -m mlb_luck_score.models.compare_infield_opportunity \\
        --input data/processed/cleaned_development_data_with_sprint_speed.parquet \\
        --output-dir outputs/tables \\
        --figures-dir outputs/figures/infield_opportunity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
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
from mlb_luck_score.eligibility import (
    add_infield_opportunity_eligibility,
    summarize_infield_exclusions,
)
from mlb_luck_score.features.build_contact_features import (
    add_infield_opportunity_features,
    select_infield_opportunity_features,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    MATERIAL_ECE_ABSOLUTE_THRESHOLD as _MATERIAL_ECE_ABSOLUTE_THRESHOLD,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    MIN_SUBGROUP_PLAYS,
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    OVERALL_STATUS_NOT_CALIBRATED,
    SUBGROUP_STATUS_CALIBRATED,
    SUBGROUP_STATUS_NOT_CALIBRATED,
    SubgroupCalibrationEvidence,
    compute_adaptive_calibration_table,
    compute_calibration_intercept_slope,
    evaluate_subgroup_calibration,
)
from mlb_luck_score.models.compare_near_wall_models import (
    PartialDependenceDiagnostic,
)
from mlb_luck_score.models.compare_opportunity_models import compute_binary_ece
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

VARIANT_INFIELD_LOGISTIC = "infield_logistic_v08"
VARIANT_INFIELD_HGB = "infield_hgb_v08"
INFIELD_SELECTION_CANDIDATES: tuple[str, ...] = (VARIANT_INFIELD_LOGISTIC, VARIANT_INFIELD_HGB)
VARIANT_INFIELD_FINAL = "infield_contact_only_v08"

_MODEL_TYPE_BY_CANDIDATE: dict[str, str] = {
    VARIANT_INFIELD_LOGISTIC: MODEL_TYPE_LOGISTIC,
    VARIANT_INFIELD_HGB: MODEL_TYPE_HGB,
}

#: Absolute-quality bar (task Phase 6) -- REUSED verbatim from Version
#: 0.7A/0.7C/0.7D, declared here before this module is ever run against real
#: 2024 infield results. See module docstring.
MATERIAL_ECE_ABSOLUTE_THRESHOLD = _MATERIAL_ECE_ABSOLUTE_THRESHOLD

DEFAULT_N_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

#: Controlled-perturbation scenario magnitudes -- chosen within the real
#: observed 2021-2024 infield-eligible sprint-speed range (21.8-30.7 ft/s).
#: Documented Version 0.8 research placeholders, not validated thresholds.
SLOW_SPRINT_SPEED_FTS = 23.0
FAST_SPRINT_SPEED_FTS = 29.0

#: Grid for the launch-speed (exit velocity) partial-dependence diagnostic --
#: spans the real observed infield-eligible launch-speed range (8.6-120.3
#: mph), clipped to the bulk of the distribution.
LAUNCH_SPEED_PDP_GRID = tuple(np.linspace(20.0, 110.0, 15))
MAX_PLAUSIBLE_PDP_SIGN_CHANGES = 2

MIN_BAND_SAMPLE_SIZE = MIN_SUBGROUP_PLAYS


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def get_infield_rows(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Outfield-... no -- INFIELD-opportunity-eligible rows, features attached.

    Args:
        joined_df: Cleaned development data joined with sprint speed
            (`mlb_luck_score.data.join_sprint_speed`) -- venue metadata is
            NOT required (Version 0.8 uses `surface_type`, already present
            in the base cleaned table, not park geometry).

    Returns:
        `infield_opportunity_eligible` rows only, with Version 0.8 feature
        columns attached.
    """
    df = add_infield_opportunity_eligibility(joined_df)
    df = add_infield_opportunity_features(df)
    return df[df["infield_opportunity_eligible"].astype(bool)]


def _mean_predicted_p_out(trained: TrainedOpportunityModel, df: pd.DataFrame) -> float:
    feature_cols = trained.numeric_features + trained.categorical_features
    return float(predict_opportunity_proba(trained, df[feature_cols]).mean())


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def run_infield_model_selection(
    infield_df: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, TrainedOpportunityModel]]:
    """Fit both candidates on `CALIBRATION_BASE_TRAIN_SEASONS`, select on
    `CALIBRATION_FIT_SEASONS` by log loss (tie-broken by ECE).

    Returns:
        `(winner_variant, selection_metrics_by_candidate, trained_by_candidate)`.
    """
    fit_df = infield_df[infield_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    selection_df = infield_df[infield_df["season"].isin(CALIBRATION_FIT_SEASONS)]
    if fit_df.empty or selection_df.empty:
        raise ValueError(
            f"Need non-empty fit ({CALIBRATION_BASE_TRAIN_SEASONS}) and selection "
            f"({CALIBRATION_FIT_SEASONS}) infield-opportunity rows to run model selection."
        )
    logger.info(
        "Infield model selection: %d fit rows, %d selection rows", len(fit_df), len(selection_df)
    )

    numeric_features, categorical_features = select_infield_opportunity_features(fit_df)

    selection_metrics: dict[str, dict[str, Any]] = {}
    trained_by_candidate: dict[str, TrainedOpportunityModel] = {}

    for candidate, model_type in _MODEL_TYPE_BY_CANDIDATE.items():
        trained = train_opportunity_model(
            fit_df,
            class_weight=None,
            feature_set_label=candidate,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_type,
        )
        feature_cols = trained.numeric_features + trained.categorical_features
        p_out = predict_opportunity_proba(trained, selection_df[feature_cols])
        validate_opportunity_probabilities(p_out)
        y_true = selection_df["y_out"].astype(int).to_numpy()
        table = compute_adaptive_calibration_table(y_true, p_out)
        selection_metrics[candidate] = {
            "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
            "expected_calibration_error": compute_binary_ece(table) if not table.empty else None,
        }
        trained_by_candidate[candidate] = trained

    winner = min(
        INFIELD_SELECTION_CANDIDATES,
        key=lambda c: (
            selection_metrics[c]["binary_log_loss"],
            selection_metrics[c]["expected_calibration_error"] or float("inf"),
        ),
    )
    logger.info("Infield model selection winner: %s (%s)", winner, selection_metrics[winner])
    return winner, selection_metrics, trained_by_candidate


# ---------------------------------------------------------------------------
# Phase 7: physical/logical perturbation checks
# ---------------------------------------------------------------------------


def _override_sprint_speed(df: pd.DataFrame, new_sprint_speed_fts: float) -> pd.DataFrame:
    """Override `sprint_speed` ONLY -- no other feature in `INFIELD_NUMERIC_
    FEATURES`/`INFIELD_CATEGORICAL_FEATURES` is derived from it (see module
    docstring Phase 7 check 3), so no recomputation is needed or possible.
    """
    return override_columns(df, {"sprint_speed": new_sprint_speed_fts})


def compute_launch_speed_partial_dependence(
    trained: TrainedOpportunityModel,
    df: pd.DataFrame,
    *,
    grid: tuple[float, ...] = LAUNCH_SPEED_PDP_GRID,
    band: str = "all",
) -> PartialDependenceDiagnostic:
    """Classical partial-dependence curve for exit velocity (`launch_speed`), over `df`.

    DESCRIPTIVE ONLY, no required sign -- see module docstring Phase 7 check
    4 for why exit velocity has no universal monotonic direction.
    """
    mean_p_out = [
        _mean_predicted_p_out(trained, override_columns(df, {"launch_speed": speed}))
        for speed in grid
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


#: Sprint-speed quartile bucket labels -- same convention as `mlb_luck_score.
#: models.compare_opportunity_models.compute_opportunity_time_buckets`.
_SPRINT_SPEED_BUCKET_LABELS: tuple[str, ...] = ("q1_slowest", "q2", "q3", "q4_fastest")


def compute_sprint_speed_buckets(sprint_speed: pd.Series) -> pd.Series:
    """Quartile buckets of real `sprint_speed` -- the sprint-speed-group subgroup."""
    try:
        return pd.qcut(sprint_speed, q=4, labels=list(_SPRINT_SPEED_BUCKET_LABELS))
    except ValueError:
        return pd.Series(pd.NA, index=sprint_speed.index)


@dataclass(frozen=True)
class SprintSpeedResponseBin:
    """One bin of a REAL-DATA (never synthetic) grouped response curve for
    `sprint_speed`, within an infield position -- see module docstring Phase
    7 check 5. Uses rows' own natural variation, never an override.
    """

    position: str
    bucket: str
    mean_sprint_speed_fts: float
    mean_predicted_p_out: float
    sample_size: int


def compute_grouped_sprint_speed_response(
    trained: TrainedOpportunityModel, infield_df: pd.DataFrame
) -> list[SprintSpeedResponseBin]:
    """REAL-DATA grouped response curve: mean predicted P(out) by `sprint_speed`
    quartile, within each infield position -- corroborates
    `sprint_speed_direction` using rows' own natural variation.
    """
    if "sprint_speed" not in infield_df.columns:
        return []
    feature_cols = trained.numeric_features + trained.categorical_features
    results: list[SprintSpeedResponseBin] = []
    for position in sorted(infield_df["assigned_infield_position"].dropna().unique()):
        subset = infield_df[infield_df["assigned_infield_position"] == position]
        subset = subset[subset["sprint_speed"].notna()]
        if len(subset) < MIN_BAND_SAMPLE_SIZE:
            continue
        buckets = compute_sprint_speed_buckets(subset["sprint_speed"])
        p_out = predict_opportunity_proba(trained, subset[feature_cols])
        for bucket_label in _SPRINT_SPEED_BUCKET_LABELS:
            mask = (buckets == bucket_label).to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue
            results.append(
                SprintSpeedResponseBin(
                    position=str(position),
                    bucket=bucket_label,
                    mean_sprint_speed_fts=float(subset["sprint_speed"].to_numpy()[mask].mean()),
                    mean_predicted_p_out=float(p_out.to_numpy()[mask].mean()),
                    sample_size=n,
                )
            )
    return results


@dataclass(frozen=True)
class InfieldPerturbationSuite:
    """The full Version 0.8 controlled-perturbation plausibility suite. See
    module docstring Phase 7 for the full design.
    """

    required: dict[str, DirectionalCheckResult]
    partial_dependence: dict[str, PartialDependenceDiagnostic]
    grouped_sprint_speed_response: list[SprintSpeedResponseBin] = field(default_factory=list)
    timing_margin_check_status: str = "not_applicable"


def run_infield_perturbation_checks(
    trained: TrainedOpportunityModel, infield_df: pd.DataFrame
) -> InfieldPerturbationSuite:
    """The Version 0.8 controlled-perturbation plausibility suite -- see module
    docstring Phase 7 for the full design and reasoning.
    """
    required: dict[str, DirectionalCheckResult] = {}

    if "sprint_speed" in infield_df.columns:
        eval_df = infield_df[infield_df["sprint_speed"].notna()]
        slow_df = _override_sprint_speed(eval_df, SLOW_SPRINT_SPEED_FTS)
        fast_df = _override_sprint_speed(eval_df, FAST_SPRINT_SPEED_FTS)
        mean_slow = _mean_predicted_p_out(trained, slow_df)
        mean_fast = _mean_predicted_p_out(trained, fast_df)
        required["sprint_speed_direction"] = DirectionalCheckResult(
            label="infield: batter sprint speed slow vs fast",
            outcome_class="out",
            low_label=f"slow batter ({SLOW_SPRINT_SPEED_FTS} ft/s)",
            high_label=f"fast batter ({FAST_SPRINT_SPEED_FTS} ft/s)",
            mean_prob_low=mean_slow,
            mean_prob_high=mean_fast,
            delta=mean_fast - mean_slow,
            expect_high_greater=False,
            passed=(mean_fast - mean_slow) <= 0,
            sample_size=len(eval_df),
        )

    partial_dependence = {"all": compute_launch_speed_partial_dependence(trained, infield_df)}
    grouped_response = compute_grouped_sprint_speed_response(trained, infield_df)

    return InfieldPerturbationSuite(
        required=required,
        partial_dependence=partial_dependence,
        grouped_sprint_speed_response=grouped_response,
        timing_margin_check_status="not_applicable",
    )


# ---------------------------------------------------------------------------
# Phase 6: subgroup/venue masks (task Phase 6's required breakdown)
# ---------------------------------------------------------------------------

#: `if_fielding_alignment` categories other than "Standard" grouped as
#: "shifted" for the REQUIRED "standard versus shifted alignment" breakdown
#: (task Phase 6) -- the full per-category breakdown is also computed
#: (`alignment_infield_shift`/`alignment_infield_shade`/`alignment_strategic`)
#: for completeness, but the task's explicit binary framing is the primary one.
_SHIFTED_ALIGNMENT_CATEGORIES: frozenset[str] = frozenset(
    {"Infield shift", "Infield shade", "Strategic"}
)

#: Ground-ball "location region" -- reuses the SAME `spray_sector` categories
#: (`mlb_luck_score.data.clean_batted_balls`) as every other version in this
#: codebase, not a new convention.
_SPRAY_SECTORS: tuple[str, ...] = ("left", "left_center", "center", "right_center", "right")


def build_infield_subgroup_masks(infield_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """All required Version 0.8 subgroup masks (task Phase 6): fielding
    position, pull/center/opposite-field contact, batter handedness,
    sprint-speed groups, standard-vs-shifted alignment, weak/medium/hard
    ground balls, and spray-angle/location regions. Venues are handled
    separately (`build_infield_venue_masks`).
    """
    masks: dict[str, np.ndarray] = {}
    n = len(infield_df)

    if "assigned_infield_position" in infield_df.columns:
        for position in sorted(infield_df["assigned_infield_position"].dropna().unique()):
            masks[f"position_{position}"] = (
                infield_df["assigned_infield_position"] == position
            ).to_numpy()

    if "is_pull" in infield_df.columns:
        masks["pull"] = infield_df["is_pull"].fillna(0).astype(bool).to_numpy()
    if "is_center" in infield_df.columns:
        masks["center"] = infield_df["is_center"].fillna(0).astype(bool).to_numpy()
    if "is_opposite_field" in infield_df.columns:
        masks["opposite_field"] = infield_df["is_opposite_field"].fillna(0).astype(bool).to_numpy()

    if "stand" in infield_df.columns:
        for hand in ("L", "R"):
            masks[f"batter_stand_{hand}"] = (infield_df["stand"] == hand).to_numpy()

    if "sprint_speed" in infield_df.columns:
        buckets = compute_sprint_speed_buckets(infield_df["sprint_speed"])
        for bucket in _SPRINT_SPEED_BUCKET_LABELS:
            masks[f"sprint_speed_{bucket}"] = (buckets == bucket).to_numpy()

    if "if_fielding_alignment" in infield_df.columns:
        alignment = infield_df["if_fielding_alignment"]
        masks["alignment_standard"] = (alignment == "Standard").to_numpy()
        masks["alignment_shifted"] = alignment.isin(_SHIFTED_ALIGNMENT_CATEGORIES).to_numpy()
        masks["alignment_infield_shift"] = (alignment == "Infield shift").to_numpy()
        masks["alignment_infield_shade"] = (alignment == "Infield shade").to_numpy()
        masks["alignment_strategic"] = (alignment == "Strategic").to_numpy()

    if "launch_speed" in infield_df.columns:
        try:
            buckets = pd.qcut(infield_df["launch_speed"], q=3, labels=["weak", "medium", "hard"])
            for bucket in ("weak", "medium", "hard"):
                masks[f"ground_ball_{bucket}"] = (buckets == bucket).to_numpy()
        except ValueError:
            pass

    if "spray_sector" in infield_df.columns:
        for sector in _SPRAY_SECTORS:
            masks[f"spray_sector_{sector}"] = (infield_df["spray_sector"] == sector).to_numpy()

    return {k: v for k, v in masks.items() if v.shape == (n,)}


def build_infield_venue_masks(venue_ids: pd.Series | None) -> dict[str, np.ndarray]:
    """One mask per distinct venue -- "reliably sampled" is determined by
    `has_adequate_support`/status (v0.7D framework), not a separate
    venue-specific sample-size convention.
    """
    if venue_ids is None:
        return {}
    grouped = venue_ids.fillna("__missing_venue__").to_numpy()
    return {f"venue_{v}": (grouped == v) for v in pd.unique(grouped)}


def evaluate_all_infield_subgroups(
    infield_df: pd.DataFrame,
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> list[SubgroupCalibrationEvidence]:
    """Evaluate every required Version 0.8 subgroup with the REUSED v0.7D
    machinery. `baseline_p_out = specialist_p_out` (the candidate compared
    against ITSELF) since there is no prior production infield-opportunity
    model to compare against -- see module docstring Phase 6. This makes the
    paired-delta CI identically zero, so it can never trigger a "credible
    regression" finding, leaving only the absolute-ECE-confidence-interval
    criterion active (the "absolute bar, not relative" rule for a genuinely
    new model category, per CLAUDE.md/AGENTS.md).
    """
    masks = build_infield_subgroup_masks(infield_df)
    game_pks = infield_df["game_pk"]
    return [
        evaluate_subgroup_calibration(
            label,
            mask,
            y_true,
            specialist_p_out,
            specialist_p_out,
            game_pks,
            n_reps=n_reps,
            seed=seed,
            ci=ci,
        )
        for label, mask in masks.items()
    ]


def evaluate_all_infield_venues(
    infield_df: pd.DataFrame,
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> list[SubgroupCalibrationEvidence]:
    """Evaluate every venue with the REUSED v0.7D machinery -- same
    self-baseline convention as `evaluate_all_infield_subgroups`.
    """
    if "venue_id" not in infield_df.columns:
        return []
    masks = build_infield_venue_masks(infield_df["venue_id"])
    game_pks = infield_df["game_pk"]
    return [
        evaluate_subgroup_calibration(
            label,
            mask,
            y_true,
            specialist_p_out,
            specialist_p_out,
            game_pks,
            n_reps=n_reps,
            seed=seed,
            ci=ci,
        )
        for label, mask in masks.items()
    ]


@dataclass(frozen=True)
class ReachedOnErrorComparison:
    """Descriptive-only comparison of reached-on-error vs. non-error safe
    outcomes (task Phase 6) -- NOT a calibration status: restricting to only
    `y_out == 0` rows gives a degenerate calibration table (no outcome
    variation to calibrate against), so this reports mean predicted P(out)
    and sample sizes only. The model cannot distinguish "the defense muffed
    a play it should have made" from "the ball simply beat the defense
    cleanly" from PRE-CONTACT features alone -- both should show SIMILAR mean
    predicted P(out) if the model is not accidentally leaking post-outcome
    information.
    """

    reached_on_error_n: int
    reached_on_error_mean_predicted_p_out: float | None
    non_error_safe_n: int
    non_error_safe_mean_predicted_p_out: float | None


def compute_reached_on_error_comparison(
    infield_df: pd.DataFrame, p_out: pd.Series
) -> ReachedOnErrorComparison:
    error_mask = infield_df["reached_on_error"].astype(bool).to_numpy()
    safe_mask = (infield_df["y_out"].astype(int) == 0).to_numpy()
    non_error_safe_mask = safe_mask & ~error_mask
    p_arr = p_out.to_numpy()

    return ReachedOnErrorComparison(
        reached_on_error_n=int(error_mask.sum()),
        reached_on_error_mean_predicted_p_out=(
            float(p_arr[error_mask].mean()) if error_mask.any() else None
        ),
        non_error_safe_n=int(non_error_safe_mask.sum()),
        non_error_safe_mean_predicted_p_out=(
            float(p_arr[non_error_safe_mask].mean()) if non_error_safe_mask.any() else None
        ),
    )


# ---------------------------------------------------------------------------
# Final 2024 comparison and overall summary
# ---------------------------------------------------------------------------


def run_infield_final_comparison(
    infield_df: pd.DataFrame, winner_trained: TrainedOpportunityModel
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    """Evaluate the winning candidate on `CALIBRATION_EVAL_SEASONS` (2024) --
    the "final comparison" within this phase. Reports overall log loss/Brier/
    ECE/calibration-intercept-slope, prevalence, and missingness -- task
    Phase 6's overall metrics.
    """
    final_df = infield_df[infield_df["season"].isin(CALIBRATION_EVAL_SEASONS)]
    if final_df.empty:
        raise ValueError(f"Need non-empty final-comparison ({CALIBRATION_EVAL_SEASONS}) rows.")

    from sklearn.metrics import brier_score_loss

    feature_cols = winner_trained.numeric_features + winner_trained.categorical_features
    p_out = predict_opportunity_proba(winner_trained, final_df[feature_cols])
    validate_opportunity_probabilities(p_out)
    y_true = final_df["y_out"].astype(int).to_numpy()

    table = compute_adaptive_calibration_table(y_true, p_out)
    intercept, slope = compute_calibration_intercept_slope(y_true, p_out.to_numpy())

    return (
        {
            "variant": VARIANT_INFIELD_FINAL,
            "sample_count": int(len(final_df)),
            "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, p_out.to_numpy())),
            "expected_calibration_error": compute_binary_ece(table) if not table.empty else None,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "outcome_prevalence": float(y_true.mean()),
            "mean_predicted_p_out": float(p_out.mean()),
            "feature_missingness": {
                col: float(final_df[col].isna().mean())
                for col in feature_cols
                if col in final_df.columns
            },
        },
        final_df,
        p_out,
    )


def summarize_infield_calibration(
    overall_comparison: dict[str, Any],
    subgroup_evidence: list[SubgroupCalibrationEvidence],
    venue_evidence: list[SubgroupCalibrationEvidence],
    perturbation_suite: InfieldPerturbationSuite,
    reached_on_error: ReachedOnErrorComparison,
) -> dict[str, Any]:
    """Recompute the Version 0.8 analogue of `near_wall_specialist_calibrated`
    using the v0.7D three-state framework FROM THE START (task Phase 6/9).

    `calibrated_infield_opportunity` (task Phase 9's status) is `True` ONLY
    if: the overall ECE clears the absolute-quality bar
    (`MATERIAL_ECE_ABSOLUTE_THRESHOLD`), every REQUIRED perturbation check
    passes, and no adequately-supported subgroup/venue is credibly
    `not_calibrated`. If those hold but some subgroup/venue lacks adequate
    evidence, the status is `provisional_infield_opportunity` (not a silent
    pass -- see CLAUDE.md/AGENTS.md "Measured miscalibration vs. insufficient
    evidence"). Otherwise `not_calibrated`.
    """
    all_evidence = subgroup_evidence + venue_evidence
    not_calibrated = [e.label for e in all_evidence if e.status == SUBGROUP_STATUS_NOT_CALIBRATED]
    insufficient = [
        e.label
        for e in all_evidence
        if e.status not in (SUBGROUP_STATUS_CALIBRATED, SUBGROUP_STATUS_NOT_CALIBRATED)
    ]
    calibrated = [e.label for e in all_evidence if e.status == SUBGROUP_STATUS_CALIBRATED]

    perturbation_failures = [
        name for name, r in perturbation_suite.required.items() if not r.passed
    ]
    perturbation_checks_passed = bool(perturbation_suite.required) and not perturbation_failures

    overall_ece = overall_comparison.get("expected_calibration_error")
    overall_ece_within_threshold = (
        overall_ece is not None and overall_ece <= MATERIAL_ECE_ABSOLUTE_THRESHOLD
    )

    clears_absolute_bar = (
        overall_ece_within_threshold and perturbation_checks_passed and not not_calibrated
    )

    if not clears_absolute_bar:
        overall_status = OVERALL_STATUS_NOT_CALIBRATED
    elif insufficient:
        overall_status = OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE
    else:
        overall_status = OVERALL_STATUS_CALIBRATED

    calibrated_infield_opportunity = overall_status == OVERALL_STATUS_CALIBRATED

    return {
        "overall_ece": overall_ece,
        "overall_ece_within_threshold": overall_ece_within_threshold,
        "perturbation_checks_passed": perturbation_checks_passed,
        "perturbation_failures": perturbation_failures,
        "perturbation_required_detail": {
            k: v.__dict__ for k, v in perturbation_suite.required.items()
        },
        "partial_dependence_detail": {
            k: v.__dict__ for k, v in perturbation_suite.partial_dependence.items()
        },
        "grouped_sprint_speed_response_detail": [
            b.__dict__ for b in perturbation_suite.grouped_sprint_speed_response
        ],
        "timing_margin_check_status": perturbation_suite.timing_margin_check_status,
        "subgroup_evidence": [asdict(e) for e in subgroup_evidence],
        "venue_evidence": [asdict(e) for e in venue_evidence],
        "not_calibrated_groups": not_calibrated,
        "insufficient_evidence_groups": insufficient,
        "calibrated_groups": calibrated,
        "reached_on_error_comparison": asdict(reached_on_error),
        "overall_status": overall_status,
        "calibrated_infield_opportunity": calibrated_infield_opportunity,
        "measured_miscalibration_vs_insufficient_evidence_note": (
            "'not_calibrated' means CREDIBLE evidence of a problem (a confidence interval "
            "entirely on the wrong side of a line); 'insufficient_evidence' means the data "
            "cannot yet tell -- reported honestly and separately, never treated as a pass or "
            "a fail. See CLAUDE.md/AGENTS.md 'Measured miscalibration vs. insufficient evidence'."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with sprint speed",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
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

    eligibility_df = add_infield_opportunity_eligibility(df)
    exclusions = summarize_infield_exclusions(eligibility_df)
    logger.info("Infield eligibility totals: %s", exclusions.total)

    infield_df = add_infield_opportunity_features(eligibility_df)
    infield_df = infield_df[infield_df["infield_opportunity_eligible"].astype(bool)]
    logger.info("Infield-opportunity-eligible rows: %d", len(infield_df))

    winner_variant, selection_metrics, trained_by_candidate = run_infield_model_selection(
        infield_df
    )
    winner_trained = trained_by_candidate[winner_variant]

    overall_comparison, final_df, p_out = run_infield_final_comparison(infield_df, winner_trained)
    y_true = final_df["y_out"].astype(int).to_numpy()

    perturbation_suite = run_infield_perturbation_checks(winner_trained, final_df)

    subgroup_evidence = evaluate_all_infield_subgroups(
        final_df, y_true, p_out, n_reps=args.n_bootstrap_reps, seed=args.bootstrap_seed
    )
    venue_evidence = evaluate_all_infield_venues(
        final_df, y_true, p_out, n_reps=args.n_bootstrap_reps, seed=args.bootstrap_seed
    )
    reached_on_error = compute_reached_on_error_comparison(final_df, p_out)

    gate_summary = summarize_infield_calibration(
        overall_comparison, subgroup_evidence, venue_evidence, perturbation_suite, reached_on_error
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "infield_opportunity_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "eligibility_exclusions": {
                    "total": exclusions.total,
                    "by_season": exclusions.by_season,
                },
                "winner_variant": winner_variant,
                "selection_metrics": selection_metrics,
                "overall_comparison": overall_comparison,
                "gate_summary": gate_summary,
            },
            indent=2,
            default=str,
        )
    )

    if args.figures_dir is not None:
        from mlb_luck_score.models.compare_near_wall_calibration_gate import (
            plot_subgroup_reliability_diagrams,
        )

        masks = {
            **build_infield_subgroup_masks(final_df),
            **build_infield_venue_masks(
                final_df["venue_id"] if "venue_id" in final_df.columns else None
            ),
        }
        written = plot_subgroup_reliability_diagrams(y_true, p_out, masks, args.figures_dir)
        logger.info("Wrote %d reliability diagrams to %s", len(written), args.figures_dir)

    logger.info("Winner: %s", winner_variant)
    logger.info(
        "[%s] log_loss=%.6f ece=%.6f n=%d",
        VARIANT_INFIELD_FINAL,
        overall_comparison["binary_log_loss"],
        overall_comparison["expected_calibration_error"],
        overall_comparison["sample_count"],
    )
    logger.info("overall_status=%s", gate_summary["overall_status"])
    logger.info("calibrated_infield_opportunity=%s", gate_summary["calibrated_infield_opportunity"])
    logger.info(
        "calibrated=%d not_calibrated=%d insufficient_evidence=%d",
        len(gate_summary["calibrated_groups"]),
        len(gate_summary["not_calibrated_groups"]),
        len(gate_summary["insufficient_evidence_groups"]),
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
