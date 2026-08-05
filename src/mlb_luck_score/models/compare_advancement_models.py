"""Contact Luck v0.9: batter-runner advancement model comparison.

Compares three candidates over the multinomial `batter_final_base` target
(`mlb_luck_score.eligibility.ADVANCEMENT_LABELS`), for `advancement_
eligible` rows only (fair outfield air balls where the batter safely
reaches at least first -- infield hits are explicitly deferred, per the
task):

  - `advancement_context_v09`: multinomial logistic regression, contact/
    context features only (no batter speed).
  - `advancement_speed_v09`: the SAME features plus `sprint_speed`, same
    model class -- isolates the marginal value of adding speed.
  - `advancement_nonlinear_v09_candidate`: the SAME features as `speed_v09`,
    `HistGradientBoostingClassifier` instead of logistic regression -- the
    SAME "model CLASS comparison on an identical feature set" pattern
    Version 0.7C/0.8 already established.

...plus a fourth, NON-learned reference point:

  - `advancement_empirical_baseline`: predicted probabilities are a lookup
    table of empirical outcome frequencies by (hit-type-implied floor base,
    sprint-speed quartile), fit ONLY on `CALIBRATION_BASE_TRAIN_SEASONS`
    (same "fit on train, apply to all" discipline as `mlb_luck_score.data.
    join_weather_features.compute_venue_air_density_baseline`). Falls back
    to the OVERALL training-season frequency distribution for any (group)
    combination unseen in training.

## Phase 1: public-data audit for MODEL INPUTS (labeling audit lives in
`mlb_luck_score.eligibility`'s Version 0.9 module comment)

Every "likely pre-outcome input" the task named was checked against the
real, already-downloaded 2021-2024 dataset:

  - **Modeled contact outcome probabilities**: AVAILABLE -- the EXISTING,
    UNCHANGED Version 0.2 `baseline_v02` 5-class contact model's own
    predicted probabilities for this play (`mlb_luck_score.models.
    train_contact_model.predict_proba_ordered`), attached as 5 numeric
    features (`ADVANCEMENT_CONTACT_PROBABILITY_FEATURES`) by THIS module,
    fit ONLY on `CALIBRATION_BASE_TRAIN_SEASONS` contact-eligible rows (the
    same train-only discipline as the empirical baseline above) and applied
    unchanged to the selection/final-eval splits.
  - **Exit velocity, launch angle, spray direction, estimated landing
    location, estimated hang time, wall proximity**: AVAILABLE -- the SAME
    raw/physics-derived columns Version 0.7A already established.
  - **Responsible outfielder position**: AVAILABLE -- REUSES `mlb_luck_
    score.eligibility.add_outfield_opportunity_eligibility`'s `assigned_
    outfield_position` rather than recomputing it (Version 0.9's eligible
    scope is a subset of Version 0.7A's).
  - **Batter sprint speed**: AVAILABLE -- the SAME Version 0.8 season-level
    Baseball Savant leaderboard (`mlb_luck_score.data.download_sprint_
    speed`/`join_sprint_speed`).
  - **Outfield alignment, venue geometry, outs and base state**: AVAILABLE
    -- `of_fielding_alignment`, the Version 0.4 wall-geometry columns, and
    `outs_when_up`/`on_1b_occupied`/`on_2b_occupied`/`on_3b_occupied`
    (pre-contact game-state indicators computed by `mlb_luck_score.
    features.build_contact_features.add_advancement_features` -- used only
    as CONTEXT for the batter's OWN advancement decision, never to model
    the other runners' advancement, which is out of scope per the task).
  - **Actual initial fielding result, only where it does not leak final
    advancement**: NOT BUILT. Statcast/`des` records exactly ONE terminal
    outcome per play, not a separate "was it fielded cleanly" checkpoint
    distinct from the final result -- there is no way to isolate this
    without it becoming a restatement of (part of) the target itself.
  - **Throwing distance proxy, only if defensibly derived**: NOT BUILT --
    `des` reveals the fielder-to-fielder relay CHAIN (see `mlb_luck_score.
    eligibility`'s multi-throw-relay proxy) but never real coordinates or
    distances; building a proxy would mean inventing defender/throw
    geometry, exactly what CLAUDE.md/AGENTS.md's "no assumed defender
    starting position" rule (and Version 0.7A's/0.8's own skipped position/
    timing proxies) already warns against.

  **UNSAFE TO USE / NEVER A MODEL INPUT**: `des`, the parser's matched
  clause, parse status, and every other Version 0.9 audit-trail column are
  all in `mlb_luck_score.config.LEAKAGE_COLUMNS` -- `assert_no_leakage`
  (called by `select_advancement_features`/`train_advancement_model`)
  enforces this the same way it enforces every other leakage rule in this
  codebase (task safeguard #7).

## Phase 5/6: model selection and absolute-quality bar

Fit on `CALIBRATION_BASE_TRAIN_SEASONS`, selected on `CALIBRATION_FIT_
SEASONS` by multiclass log loss (tie-broken by mean one-vs-rest ECE), final
comparison against the empirical baseline on `CALIBRATION_EVAL_SEASONS`.
2025 remains untouched throughout. Like Version 0.7A/0.8, this is a
genuinely NEW target with no prior per-play baseline -- CLAUDE.md/AGENTS.md's
"a model with no prior baseline needs an ABSOLUTE quality bar, not a
relative one" applies. The SAME `MATERIAL_ECE_ABSOLUTE_THRESHOLD` (0.05,
reused verbatim from Version 0.7A/0.7C/0.7D/0.8) is applied to each
one-vs-rest class's ECE. The v0.7D sample-size-aware THREE-STATE subgroup
framework (`mlb_luck_score.models.compare_near_wall_calibration_gate`) is
reused FROM THE START, looped over EVERY (subgroup, class) pair via one
-vs-rest binarization of `y_true`/predicted probabilities -- the SAME
"self-baseline" trick Version 0.8 uses (`baseline_p_out = specialist_p_out`)
so the paired-regression criterion can never trigger, leaving only the
absolute-ECE-confidence-interval criterion active.

## Phase 7: controlled-perturbation check

`sprint_speed_direction` (REQUIRED): among otherwise-identical rows, a
FASTER batter must not show a LOWER mean "expected advancement index"
(`held_at_first`=1, `advanced_to_second`=2, `advanced_to_third`=3,
`inside_the_park_home_run`=4, `retired_while_advancing`=0 -- retired is
worse than merely holding at first, since it costs a full out) than a
slower one. `sprint_speed` has NO derived-dependent feature anywhere in
this feature set (unlike e.g. Version 0.7C's hang-time/launch-angle
coupling) -- `contact_p_*` comes from an EARLIER, independent model stage;
`estimated_hang_time_s`/`landing_x_ft`/`landing_y_ft` are derived from
`launch_speed`/`launch_angle`/`hit_distance_sc`, none of which `sprint_
speed` touches -- so overriding it needs no recomputation (see CLAUDE.md/
AGENTS.md "Any override of a raw feature must recompute every feature
DERIVED from it" -- verified inapplicable here, not silently assumed).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
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
    ADVANCEMENT_LABELS,
    add_advancement_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
    summarize_advancement_exclusions,
)
from mlb_luck_score.features.build_contact_features import (
    ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
    ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
    ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES,
    ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES,
    ADVANCEMENT_SPEED_CATEGORICAL_FEATURES,
    ADVANCEMENT_SPEED_NUMERIC_FEATURES,
    HIT_TYPE_GROUP_LABELS,
    add_advancement_contact_probability_features,
    add_advancement_features,
    select_advancement_features,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    OVERALL_STATUS_NOT_CALIBRATED,
    SUBGROUP_STATUS_CALIBRATED,
    SUBGROUP_STATUS_NOT_CALIBRATED,
    SubgroupCalibrationEvidence,
    compute_adaptive_calibration_table,
    evaluate_subgroup_calibration,
)
from mlb_luck_score.models.compare_opportunity_models import compute_binary_ece
from mlb_luck_score.models.train_advancement_model import (
    MODEL_TYPE_HGB,
    MODEL_TYPE_LOGISTIC,
    TrainedAdvancementModel,
    predict_advancement_proba,
    train_advancement_model,
    validate_advancement_probabilities,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
)
from mlb_luck_score.models.weather_perturbation import override_columns

logger = logging.getLogger(__name__)

VARIANT_ADVANCEMENT_CONTEXT = "advancement_context_v09"
VARIANT_ADVANCEMENT_SPEED = "advancement_speed_v09"
VARIANT_ADVANCEMENT_NONLINEAR = "advancement_nonlinear_v09_candidate"
VARIANT_ADVANCEMENT_EMPIRICAL = "advancement_empirical_baseline"
ADVANCEMENT_SELECTION_CANDIDATES: tuple[str, ...] = (
    VARIANT_ADVANCEMENT_CONTEXT,
    VARIANT_ADVANCEMENT_SPEED,
    VARIANT_ADVANCEMENT_NONLINEAR,
)

_CANDIDATE_FEATURE_SETS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    VARIANT_ADVANCEMENT_CONTEXT: (
        ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
        ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
        MODEL_TYPE_LOGISTIC,
    ),
    VARIANT_ADVANCEMENT_SPEED: (
        ADVANCEMENT_SPEED_NUMERIC_FEATURES,
        ADVANCEMENT_SPEED_CATEGORICAL_FEATURES,
        MODEL_TYPE_LOGISTIC,
    ),
    VARIANT_ADVANCEMENT_NONLINEAR: (
        ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES,
        ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES,
        MODEL_TYPE_HGB,
    ),
}

#: Absolute-quality bar (task Phase 6) -- REUSED verbatim from Version
#: 0.7A/0.7C/0.7D/0.8. NOT a new number chosen after looking at 2024
#: advancement results.
ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD = MATERIAL_ECE_ABSOLUTE_THRESHOLD

#: Reduced from the 500-rep default used by binary gates elsewhere in this
#: codebase -- this module evaluates a (subgroup x class) CROSS PRODUCT (up
#: to 5x the bootstrap workload of a binary gate for the same subgroup
#: list), and 200 game-clustered resamples is still a statistically
#: meaningful confidence interval. A documented compute-budget choice, not
#: a threshold-loosening one -- the ECE/regression THRESHOLDS themselves are
#: unchanged.
DEFAULT_N_BOOTSTRAP_REPS = 200
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

#: Controlled-perturbation scenario magnitudes -- the SAME real observed
#: infield/outfield-eligible sprint-speed range Version 0.8 used (21.8-30.7
#: ft/s). A documented Version 0.9 research placeholder, not a validated
#: threshold.
SLOW_SPRINT_SPEED_FTS = 23.0
FAST_SPRINT_SPEED_FTS = 29.0

#: "Expected advancement index" value scale for the perturbation check and
#: empirical-baseline reporting -- `retired_while_advancing` is WORSE than
#: `held_at_first` (a full out, not just no extra base), so it is scored
#: BELOW held_at_first, not merely "no advancement."
ADVANCEMENT_BASE_VALUE: dict[str, float] = {
    "retired_while_advancing": 0.0,
    "held_at_first": 1.0,
    "advanced_to_second": 2.0,
    "advanced_to_third": 3.0,
    "inside_the_park_home_run": 4.0,
}

_SPRINT_SPEED_BUCKET_LABELS: tuple[str, ...] = ("q1_slowest", "q2", "q3", "q4_fastest")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_advancement_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full Version 0.9 eligibility/feature pipeline on raw joined data.

    Composes `compute_eligibility` (Version 0.1, for `eligible_for_training`
    -- needed to fit the contact-probability-feature contact model),
    `add_outfield_opportunity_eligibility` (Version 0.7A, for `assigned_
    outfield_position`), and `add_advancement_eligibility` +
    `add_advancement_features` (Version 0.9).
    """
    df = compute_eligibility(raw_df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_advancement_features(df)
    return df


def fit_advancement_contact_model(prepared_df: pd.DataFrame) -> TrainedModel:
    """Fit `baseline_v02` on `CALIBRATION_BASE_TRAIN_SEASONS` contact-eligible rows
    ONLY -- applied unchanged to selection/final-eval rows, same "fit on train,
    apply to all" discipline as `mlb_luck_score.data.join_weather_features.
    compute_venue_air_density_baseline`.
    """
    contact_train_df = prepared_df[
        prepared_df["eligible_for_training"].fillna(False)
        & prepared_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)
    ]
    return train_model(contact_train_df, class_weight=None)


def get_advancement_rows(prepared_df: pd.DataFrame, contact_trained: TrainedModel) -> pd.DataFrame:
    """`advancement_eligible` rows with contact-probability features attached.

    Args:
        prepared_df: Output of `prepare_advancement_data`.
        contact_trained: Output of `fit_advancement_contact_model` (or any
            other `baseline_v02`-shaped `TrainedModel` fit on train-only
            rows).
    """
    eligible = prepared_df[prepared_df["advancement_eligible"].astype(bool)]
    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, eligible[contact_feature_cols])
    return add_advancement_contact_probability_features(eligible, contact_proba)


# ---------------------------------------------------------------------------
# Empirical baseline
# ---------------------------------------------------------------------------


def compute_sprint_speed_buckets(sprint_speed: pd.Series) -> pd.Series:
    """Quartile buckets of real `sprint_speed` -- same convention as `mlb_luck_score.
    models.compare_infield_opportunity.compute_sprint_speed_buckets`.
    """
    try:
        return pd.qcut(sprint_speed, q=4, labels=list(_SPRINT_SPEED_BUCKET_LABELS))
    except ValueError:
        return pd.Series(pd.NA, index=sprint_speed.index)


def _hit_type_group(floor_base: pd.Series) -> pd.Series:
    return floor_base.map(HIT_TYPE_GROUP_LABELS)


@dataclass(frozen=True)
class EmpiricalAdvancementBaseline:
    """A simple, non-learned reference: empirical `batter_final_base` frequencies
    by (hit-type group, sprint-speed quartile), fit ONLY on train-season rows.
    """

    lookup: dict[tuple[str, str], dict[str, float]]
    overall: dict[str, float]


def fit_empirical_advancement_baseline(fit_df: pd.DataFrame) -> EmpiricalAdvancementBaseline:
    """Fit the empirical baseline on `fit_df` (expected: `CALIBRATION_BASE_TRAIN_SEASONS`
    rows only).
    """
    working = fit_df.copy()
    working["_hit_type_group"] = _hit_type_group(working["hit_type_implied_floor_base"])
    working["_speed_bucket"] = compute_sprint_speed_buckets(working["sprint_speed"])

    overall_counts = working["batter_final_base"].value_counts(normalize=True)
    overall = {label: float(overall_counts.get(label, 0.0)) for label in ADVANCEMENT_LABELS}

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    grouped = working.dropna(subset=["_hit_type_group", "_speed_bucket"]).groupby(
        ["_hit_type_group", "_speed_bucket"], observed=True
    )
    for (hit_group, speed_bucket), group in grouped:
        counts = group["batter_final_base"].value_counts(normalize=True)
        lookup[(str(hit_group), str(speed_bucket))] = {
            label: float(counts.get(label, 0.0)) for label in ADVANCEMENT_LABELS
        }

    return EmpiricalAdvancementBaseline(lookup=lookup, overall=overall)


def predict_empirical_baseline_proba(
    baseline: EmpiricalAdvancementBaseline, df: pd.DataFrame
) -> pd.DataFrame:
    """Look up each row's empirical outcome distribution -- falls back to the
    OVERALL train-season distribution for any (hit-type group, speed bucket)
    combination unseen in training (including rows with missing `sprint_speed`).
    """
    hit_type_group = _hit_type_group(df["hit_type_implied_floor_base"])
    speed_bucket = compute_sprint_speed_buckets(df["sprint_speed"])

    rows = []
    for hit_group, bucket in zip(hit_type_group, speed_bucket, strict=True):
        key = (str(hit_group), str(bucket))
        rows.append(baseline.lookup.get(key, baseline.overall))
    return pd.DataFrame(rows, index=df.index)[list(ADVANCEMENT_LABELS)]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def run_advancement_model_selection(
    advancement_df: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, TrainedAdvancementModel]]:
    """Fit all three candidates on `CALIBRATION_BASE_TRAIN_SEASONS`, select on
    `CALIBRATION_FIT_SEASONS` by multiclass log loss (tie-broken by mean
    one-vs-rest ECE).

    Args:
        advancement_df: Output of `get_advancement_rows` (contact
            -probability features already attached).

    Returns:
        `(winner_variant, selection_metrics_by_candidate, trained_by_candidate)`.
    """
    fit_df = advancement_df[advancement_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    selection_df = advancement_df[advancement_df["season"].isin(CALIBRATION_FIT_SEASONS)]
    if fit_df.empty or selection_df.empty:
        raise ValueError(
            f"Need non-empty fit ({CALIBRATION_BASE_TRAIN_SEASONS}) and selection "
            f"({CALIBRATION_FIT_SEASONS}) advancement-eligible rows to run model selection."
        )
    logger.info(
        "Advancement model selection: %d fit rows, %d selection rows",
        len(fit_df),
        len(selection_df),
    )

    selection_metrics: dict[str, dict[str, Any]] = {}
    trained_by_candidate: dict[str, TrainedAdvancementModel] = {}

    for candidate, (
        numeric_candidates,
        categorical_candidates,
        model_type,
    ) in _CANDIDATE_FEATURE_SETS.items():
        numeric_features, categorical_features = select_advancement_features(
            fit_df,
            numeric_candidates=numeric_candidates,
            categorical_candidates=categorical_candidates,
        )
        trained = train_advancement_model(
            fit_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            class_weight=None,
            feature_set_label=candidate,
            model_type=model_type,
        )
        feature_cols = trained.numeric_features + trained.categorical_features
        proba = predict_advancement_proba(trained, selection_df[feature_cols])
        validate_advancement_probabilities(proba)
        y_true = selection_df["batter_final_base"].to_numpy()

        sorted_labels = sorted(ADVANCEMENT_LABELS)
        multiclass_log_loss = float(
            log_loss(y_true, proba[sorted_labels].to_numpy(), labels=sorted_labels)
        )
        per_class_ece = []
        for cls in ADVANCEMENT_LABELS:
            y_binary = (y_true == cls).astype(int)
            table = compute_adaptive_calibration_table(y_binary, proba[cls])
            if not table.empty:
                per_class_ece.append(compute_binary_ece(table))
        mean_ece = float(np.mean(per_class_ece)) if per_class_ece else float("inf")

        selection_metrics[candidate] = {
            "multiclass_log_loss": multiclass_log_loss,
            "mean_one_vs_rest_ece": mean_ece,
        }
        trained_by_candidate[candidate] = trained

    winner = min(
        ADVANCEMENT_SELECTION_CANDIDATES,
        key=lambda c: (
            selection_metrics[c]["multiclass_log_loss"],
            selection_metrics[c]["mean_one_vs_rest_ece"],
        ),
    )
    logger.info("Advancement model selection winner: %s (%s)", winner, selection_metrics[winner])
    return winner, selection_metrics, trained_by_candidate


# ---------------------------------------------------------------------------
# Perturbation check
# ---------------------------------------------------------------------------


def compute_expected_advancement_index(proba: pd.DataFrame) -> pd.Series:
    """`sum(p(label) * ADVANCEMENT_BASE_VALUE[label])` per row."""
    values = np.array([ADVANCEMENT_BASE_VALUE[label] for label in proba.columns])
    return pd.Series(proba.to_numpy() @ values, index=proba.index)


def _override_sprint_speed(df: pd.DataFrame, new_sprint_speed_fts: float) -> pd.DataFrame:
    """Override `sprint_speed` ONLY -- see module docstring Phase 7 for why no
    feature in this set is derived from it, so no recomputation is needed.
    """
    return override_columns(df, {"sprint_speed": new_sprint_speed_fts})


@dataclass(frozen=True)
class AdvancementDirectionalCheckResult:
    label: str
    low_label: str
    high_label: str
    mean_expected_advancement_low: float
    mean_expected_advancement_high: float
    delta: float
    passed: bool
    sample_size: int


def run_advancement_perturbation_checks(
    trained: TrainedAdvancementModel, advancement_df: pd.DataFrame
) -> dict[str, AdvancementDirectionalCheckResult]:
    """The Version 0.9 controlled-perturbation plausibility suite -- see module
    docstring Phase 7 for the full design.
    """
    results: dict[str, AdvancementDirectionalCheckResult] = {}
    if "sprint_speed" not in advancement_df.columns:
        return results

    feature_cols = trained.numeric_features + trained.categorical_features
    eval_df = advancement_df[advancement_df["sprint_speed"].notna()]
    slow_df = _override_sprint_speed(eval_df, SLOW_SPRINT_SPEED_FTS)
    fast_df = _override_sprint_speed(eval_df, FAST_SPRINT_SPEED_FTS)

    slow_proba = predict_advancement_proba(trained, slow_df[feature_cols])
    fast_proba = predict_advancement_proba(trained, fast_df[feature_cols])
    mean_slow = float(compute_expected_advancement_index(slow_proba).mean())
    mean_fast = float(compute_expected_advancement_index(fast_proba).mean())

    results["sprint_speed_direction"] = AdvancementDirectionalCheckResult(
        label="advancement: batter sprint speed slow vs fast",
        low_label=f"slow batter ({SLOW_SPRINT_SPEED_FTS} ft/s)",
        high_label=f"fast batter ({FAST_SPRINT_SPEED_FTS} ft/s)",
        mean_expected_advancement_low=mean_slow,
        mean_expected_advancement_high=mean_fast,
        delta=mean_fast - mean_slow,
        passed=(mean_fast - mean_slow) >= 0,
        sample_size=len(eval_df),
    )
    return results


# ---------------------------------------------------------------------------
# Subgroup/venue masks (task's required breakdown: sprint-speed group,
# contact type, field sector, outfielder position, wall proximity, venue)
# ---------------------------------------------------------------------------


def build_advancement_subgroup_masks(advancement_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """All required Version 0.9 subgroup masks EXCEPT venue (`build_advancement_venue_masks`)
    and final-base outcome (that dimension is the per-class one-vs-rest loop itself, in
    `evaluate_all_advancement_subgroups`).
    """
    masks: dict[str, np.ndarray] = {}
    n = len(advancement_df)

    if "sprint_speed" in advancement_df.columns:
        buckets = compute_sprint_speed_buckets(advancement_df["sprint_speed"])
        for bucket in _SPRINT_SPEED_BUCKET_LABELS:
            masks[f"sprint_speed_{bucket}"] = (buckets == bucket).to_numpy()

    if "hit_type_implied_floor_base" in advancement_df.columns:
        hit_group = _hit_type_group(advancement_df["hit_type_implied_floor_base"])
        for group_label in HIT_TYPE_GROUP_LABELS.values():
            masks[f"contact_type_{group_label}"] = (hit_group == group_label).to_numpy()

    if "spray_sector" in advancement_df.columns:
        for sector in ("left", "left_center", "center", "right_center", "right"):
            masks[f"spray_sector_{sector}"] = (advancement_df["spray_sector"] == sector).to_numpy()

    if "assigned_outfield_position" in advancement_df.columns:
        for position in sorted(advancement_df["assigned_outfield_position"].dropna().unique()):
            masks[f"outfielder_position_{position}"] = (
                advancement_df["assigned_outfield_position"] == position
            ).to_numpy()

    if "absolute_distance_to_wall" in advancement_df.columns:
        near_wall = advancement_df["absolute_distance_to_wall"] <= 20.0
        masks["wall_proximity_near_wall_20ft"] = near_wall.fillna(False).to_numpy()
        masks["wall_proximity_open_field"] = (~near_wall.fillna(True)).to_numpy()

    return {k: v for k, v in masks.items() if v.shape == (n,)}


def build_advancement_venue_masks(venue_ids: pd.Series | None) -> dict[str, np.ndarray]:
    if venue_ids is None:
        return {}
    grouped = venue_ids.fillna("__missing_venue__").to_numpy()
    return {f"venue_{v}": (grouped == v) for v in pd.unique(grouped)}


def evaluate_all_advancement_subgroups(
    advancement_df: pd.DataFrame,
    proba: pd.DataFrame,
    *,
    include_venues: bool = True,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> list[SubgroupCalibrationEvidence]:
    """Evaluate every required subgroup x class (one-vs-rest) pair with the REUSED
    v0.7D machinery. `baseline_p_out = specialist_p_out` (self-baseline) since there
    is no prior production advancement model -- see module docstring Phase 6.
    """
    masks = build_advancement_subgroup_masks(advancement_df)
    if include_venues and "venue_id" in advancement_df.columns:
        masks = {**masks, **build_advancement_venue_masks(advancement_df["venue_id"])}

    y_true = advancement_df["batter_final_base"].to_numpy()
    game_pks = advancement_df["game_pk"]

    evidence: list[SubgroupCalibrationEvidence] = []
    for cls in ADVANCEMENT_LABELS:
        y_binary = (y_true == cls).astype(int)
        cls_proba = proba[cls]
        for label, mask in masks.items():
            evidence.append(
                evaluate_subgroup_calibration(
                    f"{label}__class_{cls}",
                    mask,
                    y_binary,
                    cls_proba,
                    cls_proba,
                    game_pks,
                    n_reps=n_reps,
                    seed=seed,
                    ci=ci,
                )
            )
    return evidence


# ---------------------------------------------------------------------------
# Final 2024 comparison and overall summary
# ---------------------------------------------------------------------------


def run_advancement_final_comparison(
    advancement_df: pd.DataFrame,
    winner_trained: TrainedAdvancementModel,
    empirical_baseline: EmpiricalAdvancementBaseline,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate the winning candidate AND the empirical baseline on
    `CALIBRATION_EVAL_SEASONS` (2024).
    """
    final_df = advancement_df[advancement_df["season"].isin(CALIBRATION_EVAL_SEASONS)]
    if final_df.empty:
        raise ValueError(f"Need non-empty final-comparison ({CALIBRATION_EVAL_SEASONS}) rows.")

    feature_cols = winner_trained.numeric_features + winner_trained.categorical_features
    winner_proba = predict_advancement_proba(winner_trained, final_df[feature_cols])
    validate_advancement_probabilities(winner_proba)
    empirical_proba = predict_empirical_baseline_proba(empirical_baseline, final_df)

    y_true = final_df["batter_final_base"].to_numpy()
    sorted_labels = sorted(ADVANCEMENT_LABELS)

    def _summarize(proba: pd.DataFrame, variant: str) -> dict[str, Any]:
        multiclass_log_loss = float(
            log_loss(y_true, proba[sorted_labels].to_numpy(), labels=sorted_labels)
        )
        per_class_ece: dict[str, float | None] = {}
        for cls in ADVANCEMENT_LABELS:
            y_binary = (y_true == cls).astype(int)
            table = compute_adaptive_calibration_table(y_binary, proba[cls])
            per_class_ece[cls] = compute_binary_ece(table) if not table.empty else None
        expected_index = compute_expected_advancement_index(proba)
        return {
            "variant": variant,
            "sample_count": int(len(final_df)),
            "multiclass_log_loss": multiclass_log_loss,
            "per_class_ece": per_class_ece,
            "mean_expected_advancement_index": float(expected_index.mean()),
            "class_frequencies": {cls: int((y_true == cls).sum()) for cls in ADVANCEMENT_LABELS},
        }

    overall_comparison = {
        winner_trained.feature_set_label: _summarize(
            winner_proba, winner_trained.feature_set_label
        ),
        VARIANT_ADVANCEMENT_EMPIRICAL: _summarize(empirical_proba, VARIANT_ADVANCEMENT_EMPIRICAL),
    }
    return overall_comparison, final_df, winner_proba


def summarize_advancement_calibration(
    overall_comparison: dict[str, Any],
    winner_variant: str,
    subgroup_evidence: list[SubgroupCalibrationEvidence],
    perturbation_results: dict[str, AdvancementDirectionalCheckResult],
) -> dict[str, Any]:
    """Recompute the Version 0.9 analogue of `calibrated_infield_opportunity` using the
    v0.7D three-state framework from the start (task Phase 6/9's absolute-quality-bar
    rule).

    `calibrated_advancement_model` is `True` ONLY if: the winner BEATS the empirical
    baseline's multiclass log loss, EVERY one-vs-rest class's overall ECE clears
    `ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD`, the required perturbation check
    passes, and no adequately-supported subgroup/class pair is credibly `not_calibrated`.
    If those hold but some pairs lack adequate evidence, the status is `calibrated_with_
    limited_subgroup_evidence` -- NOT a silent pass (see CLAUDE.md/AGENTS.md "Measured
    miscalibration vs. insufficient evidence").

    The `beats_empirical_baseline` check exists because a real, non-hypothetical failure
    mode was caught during Version 0.9 development: an earlier feature set omitted the
    batter's own hit type (`hit_type_group`) as an explicit categorical feature, and ALL
    THREE learned candidates scored WORSE (multiclass log loss ~0.44-0.53) than the
    trivial `advancement_empirical_baseline` (~0.18) on real 2024 data -- driven largely
    by catastrophic near-zero probability collapse on the rarest class (`inside_the_park_
    home_run`, n=20 in 2024) from the tree-based candidate specifically. Adding `hit_type_
    group` fixed the underlying feature gap (`advancement_speed_v09`'s log loss dropped to
    ~0.155, beating the baseline), but this check stays in place permanently as a
    real-data-motivated sanity gate: a model that cannot even beat a two-column frequency
    lookup table has no business being called "calibrated," no matter what its own
    internal ECE looks like in isolation.
    """
    winner_summary = overall_comparison[winner_variant]
    empirical_summary = overall_comparison.get(VARIANT_ADVANCEMENT_EMPIRICAL)
    beats_empirical_baseline = (
        empirical_summary is not None
        and winner_summary["multiclass_log_loss"] < empirical_summary["multiclass_log_loss"]
    )
    per_class_ece = winner_summary["per_class_ece"]
    ece_within_threshold = {
        cls: (ece is not None and ece <= ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD)
        for cls, ece in per_class_ece.items()
    }
    overall_ece_within_threshold = all(ece_within_threshold.values())

    perturbation_failures = [name for name, r in perturbation_results.items() if not r.passed]
    perturbation_checks_passed = bool(perturbation_results) and not perturbation_failures

    not_calibrated = [
        e.label for e in subgroup_evidence if e.status == SUBGROUP_STATUS_NOT_CALIBRATED
    ]
    insufficient = [
        e.label
        for e in subgroup_evidence
        if e.status not in (SUBGROUP_STATUS_CALIBRATED, SUBGROUP_STATUS_NOT_CALIBRATED)
    ]
    calibrated = [e.label for e in subgroup_evidence if e.status == SUBGROUP_STATUS_CALIBRATED]

    clears_absolute_bar = (
        beats_empirical_baseline
        and overall_ece_within_threshold
        and perturbation_checks_passed
        and not not_calibrated
    )
    if not clears_absolute_bar:
        overall_status = OVERALL_STATUS_NOT_CALIBRATED
    elif insufficient:
        overall_status = OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE
    else:
        overall_status = OVERALL_STATUS_CALIBRATED

    return {
        "winner_variant": winner_variant,
        "beats_empirical_baseline": beats_empirical_baseline,
        "winner_multiclass_log_loss": winner_summary["multiclass_log_loss"],
        "empirical_baseline_multiclass_log_loss": (
            empirical_summary["multiclass_log_loss"] if empirical_summary is not None else None
        ),
        "per_class_ece": per_class_ece,
        "ece_within_threshold_by_class": ece_within_threshold,
        "overall_ece_within_threshold": overall_ece_within_threshold,
        "perturbation_checks_passed": perturbation_checks_passed,
        "perturbation_failures": perturbation_failures,
        "perturbation_detail": {k: asdict(v) for k, v in perturbation_results.items()},
        "not_calibrated_pairs": not_calibrated,
        "insufficient_evidence_pairs": insufficient,
        "calibrated_pairs": calibrated,
        "overall_status": overall_status,
        "calibrated_advancement_model": overall_status == OVERALL_STATUS_CALIBRATED,
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

    prepared = prepare_advancement_data(df)
    exclusions = summarize_advancement_exclusions(prepared)
    logger.info("Advancement eligibility totals: %s", exclusions.total)

    contact_trained = fit_advancement_contact_model(prepared)
    advancement_df = get_advancement_rows(prepared, contact_trained)
    logger.info("Advancement-eligible rows: %d", len(advancement_df))

    winner_variant, selection_metrics, trained_by_candidate = run_advancement_model_selection(
        advancement_df
    )
    winner_trained = trained_by_candidate[winner_variant]

    fit_df = advancement_df[advancement_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    empirical_baseline = fit_empirical_advancement_baseline(fit_df)

    overall_comparison, final_df, winner_proba = run_advancement_final_comparison(
        advancement_df, winner_trained, empirical_baseline
    )
    perturbation = run_advancement_perturbation_checks(winner_trained, final_df)
    subgroup_evidence = evaluate_all_advancement_subgroups(
        final_df, winner_proba, n_reps=args.n_bootstrap_reps, seed=args.bootstrap_seed
    )
    gate_summary = summarize_advancement_calibration(
        overall_comparison, winner_variant, subgroup_evidence, perturbation
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "advancement_detail.json"
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

    logger.info("Winner: %s", winner_variant)
    logger.info(
        "multiclass_log_loss=%.6f beats_empirical_baseline=%s",
        gate_summary["winner_multiclass_log_loss"],
        gate_summary["beats_empirical_baseline"],
    )
    logger.info("overall_status=%s", gate_summary["overall_status"])
    logger.info("calibrated_advancement_model=%s", gate_summary["calibrated_advancement_model"])
    logger.info(
        "calibrated=%d not_calibrated=%d insufficient_evidence=%d",
        len(gate_summary["calibrated_pairs"]),
        len(gate_summary["not_calibrated_pairs"]),
        len(gate_summary["insufficient_evidence_pairs"]),
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
