"""Contact Luck v0.7A: evaluate the outfield-opportunity-difficulty model.

## Public-data audit (the task's required "practical checkpoint")

Before any model code was written, the full 119-column raw Statcast schema
already downloaded for this project (2021-2024) and every fielding-related
function in the installed `pybaseball` package (`statcast_outfield_
catch_prob`, `statcast_outfield_directional_oaa`, `statcast_outfielder_jump`,
`statcast_outs_above_average`, `statcast_fielding`) were audited for six
specific fields:

  - Defender starting location: NOT AVAILABLE. No column, no pybaseball
    function exposes it.
  - Defender endpoint: NOT AVAILABLE, same reason.
  - Opportunity time (hang time): NOT AVAILABLE as a measured field. See
    `mlb_luck_score.data.outfield_physics.estimate_hang_time_seconds` for a
    physics-derived ESTIMATE used instead.
  - Distance needed: NOT AVAILABLE (requires a real starting location).
  - Catch-probability inputs: NOT AVAILABLE per-play. Every pybaseball
    catch-probability/OAA/jump function returns a SEASON-LEVEL AGGREGATE
    leaderboard (players binned into 1-5 "star" difficulty categories), not
    a per-play value -- and per the task's modeling rule, these are
    outcome-derived anyway, so they could only ever be validation/comparison
    metrics, never model inputs, regardless of granularity.
  - Responsible fielder: AVAILABLE. `hit_location` (91.75% coverage among
    real 2024 air balls; ~100% for outs/singles/triples/sac-flies, ~0.1% for
    home runs since nobody fields one) plus `fielder_7`/`fielder_8`/
    `fielder_9` (100% coverage, the specific player ID at each outfield
    position for that exact play) together give a real individual-defender
    identity -- see `mlb_luck_score.eligibility.
    add_outfield_opportunity_eligibility`.

**Conclusion**: exact defender coordinates/movement/distance-needed cannot
be measured from public data. Per the task's explicit fallback, this module
implements only the CONSERVATIVE, measured/estimated-only candidate:

  - `measured_contact_only_v07`: landing-location estimate, estimated hang
    time, wall proximity (reused from Version 0.4 park geometry), exit
    velocity/launch angle, batted-ball type, and coarse outfield-alignment
    label (`mlb_luck_score.features.build_contact_features.
    OPPORTUNITY_NUMERIC_FEATURES`/`OPPORTUNITY_CATEGORICAL_FEATURES`). NO
    assumed defender starting coordinates.

A second candidate, `typical_position_proxy_v07` (an assumed average
starting depth/angle per outfield position, yielding `estimated_coverage_
distance_ft`/`estimated_lateral_distance_ft`/`estimated_depth_change_ft`/
`estimated_travel_direction`/`position_proxy_confidence`/`starting_position_
source`), was investigated but is **NOT implemented**: the only citable
public figures found (MLB.com/Statcast-sourced 2015-2016 league averages --
316 ft average CF depth, 294 ft average RF depth) are 5-9 years stale
relative to this project's 2021-2024 training window, and multiple
independent sources describe a SUSTAINED, DIRECTIONAL trend toward deeper
outfield positioning since then (one cited example: the Astros moved their
center fielders back 16 feet specifically during 2021-2022), meaning the
stale figures would introduce a known, systematic bias, not just noise.
Per CLAUDE.md's park-geometry provenance rule ("if a source is ambiguous or
conflicting... use the best-supported figure with a caveat or leave [it]
without... rather than guessing"), no defensible source exists for THIS
project's era, so `typical_position_proxy_v07` is not built. If a
citable, era-appropriate source is ever found, it should be added as a
second entry in `_CANDIDATE_FEATURE_BUILDERS` below under the SAME adoption
rule -- the module is structured to support it without restructuring.

## Required-but-unavailable evaluation dimension

The task also requires evaluating "forward, lateral, and backward defender
movement" as a subgroup. This is NOT COMPUTED, for the identical reason
`typical_position_proxy_v07` was not built: movement direction requires a
real starting position, which does not exist in public data. This is a
documented gap (`MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE` below), not a
silently-dropped requirement.

## What this module does NOT do

Unlike `mlb_luck_score.models.compare_park_aware`/`compare_geometry_aware`/
etc., there is no PRIOR production model here to compare against --
`measured_contact_only_v07` is a genuinely new opportunity-difficulty score,
not a candidate replacing an existing default. So this module reports
calibration/validation diagnostics for the model on its own terms (absolute
quality bars, plus a bootstrap CI on its own metrics) rather than a
`recommend_adopt`-style relative comparison. See
`summarize_opportunity_validation`.

Usage:

    python -m mlb_luck_score.models.compare_opportunity_models \\
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
from sklearn.metrics import brier_score_loss, log_loss

from mlb_luck_score.config import TABLES_DIR, TRAIN_SEASONS, VALIDATION_SEASONS
from mlb_luck_score.eligibility import add_outfield_opportunity_eligibility
from mlb_luck_score.features.build_contact_features import (
    OPPORTUNITY_TARGET_COLUMN,
    add_geometry_interaction_features,
    add_outfield_opportunity_features,
)
from mlb_luck_score.models.compare_geometry_aware import _bool_mask
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_MIN_VENUE_SAMPLES,
    _prepare_venue_column,
)
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    predict_opportunity_proba,
    train_opportunity_model,
    validate_opportunity_probabilities,
)

logger = logging.getLogger(__name__)

VARIANT_MEASURED_CONTACT_ONLY = "measured_contact_only_v07"
ALL_V07A_CANDIDATES: tuple[str, ...] = (VARIANT_MEASURED_CONTACT_ONLY,)

#: Recorded explicitly (rather than silently omitted) wherever subgroup
#: results are reported -- see module docstring "Required-but-unavailable
#: evaluation dimension".
MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE = (
    "forward/lateral/backward defender movement direction requires a real defender "
    "starting position, which is not available in public Statcast data -- see module "
    "docstring's public-data audit. Not computed."
)

DEFAULT_N_BINS = 10
MIN_RELIABLE_BIN_SAMPLES = 20
DEFAULT_MIN_SUBGROUP_SAMPLES = 100
#: An individual defender's per-play sample must be at least this large
#: before their calibration is reported at all -- "individual defenders
#: only when sample sizes are adequate" per the task.
MIN_DEFENDER_SAMPLES = 100
#: Absolute-quality bar (not a relative-to-baseline margin, since there is
#: no baseline model here) -- a subgroup/venue ECE above this is flagged for
#: review. A documented Version 0.7A research placeholder, like every other
#: material-regression threshold in this codebase.
MATERIAL_ECE_ABSOLUTE_THRESHOLD = 0.05

DEFAULT_N_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

_CANDIDATE_FEATURE_BUILDERS: dict[str, tuple[tuple[str, ...] | None, tuple[str, ...] | None]] = {
    VARIANT_MEASURED_CONTACT_ONLY: (None, None),  # None -> use select_opportunity_features default
}


def _prepare_opportunity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_outfield_opportunity_eligibility(df)
    if "wall_distance_in_spray_direction" in out.columns:
        out = add_geometry_interaction_features(out)
    out = add_outfield_opportunity_features(out)
    return out


def compute_binary_calibration_table(
    y_true: np.ndarray,
    p_out: pd.Series,
    *,
    n_bins: int = DEFAULT_N_BINS,
    min_reliable_bin_samples: int = MIN_RELIABLE_BIN_SAMPLES,
) -> pd.DataFrame:
    """Binary-target calibration table: mean predicted `P(out)` vs. observed out rate per bin.

    Same structure/formula as `mlb_luck_score.models.calibrate_model.
    compute_calibration_table`, specialized to a single probability column
    instead of a `CLASS_ORDER`-columned DataFrame (that function hard
    -requires 5 class columns, so it cannot be reused directly for a binary
    target).
    """
    predicted = p_out.to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, n_bins - 1)

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(predicted[mask].mean())
        obs_freq = float(y_true[mask].mean())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "mean_predicted_probability": mean_pred,
                "observed_frequency": obs_freq,
                "sample_count": count,
                "abs_calibration_error": abs(mean_pred - obs_freq),
                "reliable": count >= min_reliable_bin_samples,
            }
        )
    return pd.DataFrame(rows)


def compute_binary_ece(calibration_table: pd.DataFrame) -> float:
    """Sample-count-weighted mean absolute calibration error for a binary calibration table."""
    if calibration_table.empty:
        return float("nan")
    weights = calibration_table["sample_count"].to_numpy()
    errors = calibration_table["abs_calibration_error"].to_numpy()
    return float(np.average(errors, weights=weights))


def compute_binary_subgroup_calibration(
    y_true: np.ndarray, p_out: pd.Series, mask: np.ndarray, label: str
) -> dict[str, Any]:
    """Calibration/log-loss summary restricted to rows where `mask` is True."""
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "sample_count": 0, "ece": None, "log_loss": None}
    y_sub = y_true[mask]
    p_sub = p_out.to_numpy()[mask]
    table = compute_binary_calibration_table(y_sub, pd.Series(p_sub))
    ece = compute_binary_ece(table) if not table.empty else float("nan")
    ll = float(log_loss(y_sub, p_sub, labels=[0, 1])) if len(set(y_sub.tolist())) > 1 else None
    return {"label": label, "sample_count": n, "ece": ece, "log_loss": ll}


def compute_opportunity_time_buckets(estimated_hang_time_s: pd.Series) -> pd.Series:
    """Quartile buckets of estimated hang time -- the opportunity-time-range subgroup."""
    try:
        return pd.qcut(estimated_hang_time_s, q=4, labels=["q1_shortest", "q2", "q3", "q4_longest"])
    except ValueError:
        return pd.Series(pd.NA, index=estimated_hang_time_s.index)


def compute_opportunity_subgroups(
    y_true: np.ndarray, p_out: pd.Series, val_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Version 0.7A required-evaluation-group calibration.

    Covers batted-ball type, LF/CF/RF, near-wall plays, and opportunity
    -time ranges. Per-venue and per-defender calibration are reported
    separately (`compute_binary_calibration_by_venue`/`compute_defender_
    subgroup_calibration`) since they don't fit this label->mask shape.
    "Forward/lateral/backward defender movement" is NOT computed -- see
    `MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE`.
    """
    subgroups: dict[str, dict[str, Any]] = {}

    if "bb_type" in val_df.columns:
        for bb_type in ("fly_ball", "line_drive"):
            mask = (val_df["bb_type"] == bb_type).to_numpy()
            subgroups[f"bb_type_{bb_type}"] = compute_binary_subgroup_calibration(
                y_true, p_out, mask, f"bb_type_{bb_type}"
            )

    if "assigned_outfield_position" in val_df.columns:
        position_numeric = pd.to_numeric(val_df["assigned_outfield_position"], errors="coerce")
        for position, label in ((7, "left_field"), (8, "center_field"), (9, "right_field")):
            mask = (position_numeric == position).to_numpy()
            subgroups[label] = compute_binary_subgroup_calibration(y_true, p_out, mask, label)

    for col in ("near_wall_5ft", "near_wall_10ft", "near_wall_20ft"):
        if col in val_df.columns:
            subgroups[col] = compute_binary_subgroup_calibration(
                y_true, p_out, _bool_mask(val_df[col]), col
            )

    if "estimated_hang_time_s" in val_df.columns:
        buckets = compute_opportunity_time_buckets(val_df["estimated_hang_time_s"])
        for bucket in ("q1_shortest", "q2", "q3", "q4_longest"):
            mask = (buckets == bucket).to_numpy()
            subgroups[f"opportunity_time_{bucket}"] = compute_binary_subgroup_calibration(
                y_true, p_out, mask, f"opportunity_time_{bucket}"
            )

    return subgroups


def compute_binary_calibration_by_venue(
    y_true: np.ndarray,
    p_out: pd.Series,
    venue_ids: pd.Series,
    *,
    min_reliable_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
) -> pd.DataFrame:
    """Per-venue binary calibration -- same shape/reliability rule as `mlb_luck_score.
    models.compare_park_aware.compute_calibration_by_venue`, specialized to a single
    probability column.
    """
    grouped_venue = venue_ids.fillna("__missing_venue__").to_numpy()
    p_out_arr = p_out.to_numpy()
    rows: list[dict[str, Any]] = []
    for venue_value in pd.unique(grouped_venue):
        mask = grouped_venue == venue_value
        n = int(mask.sum())
        if n == 0:
            continue
        table = compute_binary_calibration_table(y_true[mask], pd.Series(p_out_arr[mask]))
        ece = compute_binary_ece(table) if not table.empty else float("nan")
        rows.append(
            {
                "venue_id": venue_value,
                "sample_count": n,
                "ece": ece,
                "reliable": n >= min_reliable_samples,
            }
        )
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def compute_defender_subgroup_calibration(
    y_true: np.ndarray,
    p_out: pd.Series,
    responsible_outfielder_id: pd.Series,
    *,
    min_samples: int = MIN_DEFENDER_SAMPLES,
) -> pd.DataFrame:
    """Per-defender calibration AND actual-vs-predicted out rate, for reliably-sampled defenders only.

    Reported for two purposes: (1) the task's "individual defenders only
    when sample sizes are adequate" evaluation requirement, and (2) this is
    the SAME aggregation Version 0.7B's execution scoring is built on
    (actual outs vs. opportunity-implied expected outs, by fielder) -- see
    `mlb_luck_score.scoring.defensive_execution`.
    """
    # Normalize pandas' pd.NA sentinel to plain None -- `pd.NA == x` returns
    # pd.NA (not False), which raises "boolean value of NA is ambiguous"
    # inside a raw numpy `==` comparison below. `None == x` returns a normal
    # False, so this side-steps the issue entirely (same class of bug
    # documented in `mlb_luck_score.models.compare_geometry_aware._bool_mask`).
    notna_mask = responsible_outfielder_id.notna()
    fielder_series = responsible_outfielder_id.where(notna_mask, None)
    fielder_arr = fielder_series.to_numpy()
    p_out_arr = p_out.to_numpy()
    rows: list[dict[str, Any]] = []
    for fielder_id in pd.unique(fielder_arr[notna_mask.to_numpy()]):
        mask = fielder_arr == fielder_id
        n = int(mask.sum())
        if n < min_samples:
            continue
        table = compute_binary_calibration_table(y_true[mask], pd.Series(p_out_arr[mask]))
        ece = compute_binary_ece(table) if not table.empty else float("nan")
        rows.append(
            {
                "responsible_outfielder_id": fielder_id,
                "sample_count": n,
                "actual_out_rate": float(y_true[mask].mean()),
                "mean_predicted_p_out": float(p_out_arr[mask].mean()),
                "ece": ece,
            }
        )
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def compute_opportunity_bootstrap(
    y_true: np.ndarray,
    p_out: pd.Series,
    game_pks: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> dict[str, dict[str, Any]]:
    """Game-level paired bootstrap CI for `measured_contact_only_v07`'s OWN metrics.

    Unlike `mlb_luck_score.models.compare_geometry_aware.
    compute_paired_bootstrap`, there is no baseline model to compute a DELTA
    against here (see module docstring) -- this reports a bootstrap
    confidence interval on the model's own log-loss/ECE, i.e. "how much
    would this metric plausibly vary under a different, similarly-sized
    sample of games," not a candidate-vs-baseline comparison. Resamples
    GAMES (not rows) with replacement, matching the task's game-level
    -paired-bootstrap requirement and every other bootstrap in this
    codebase.
    """
    y_arr = np.asarray(y_true)
    p_arr = p_out.to_numpy()
    game_pk_arr = game_pks.to_numpy()
    unique_games = np.unique(game_pk_arr)
    game_to_rows = {g: np.where(game_pk_arr == g)[0] for g in unique_games}

    rng = np.random.default_rng(seed)
    log_loss_reps = np.empty(n_reps)
    ece_reps = np.empty(n_reps)

    for rep in range(n_reps):
        sampled_games = rng.choice(unique_games, size=len(unique_games), replace=True)
        rows = np.concatenate([game_to_rows[g] for g in sampled_games])
        y_rep, p_rep = y_arr[rows], p_arr[rows]
        eps = 1e-15
        p_clip = np.clip(p_rep, eps, 1 - eps)
        log_loss_reps[rep] = float(
            -np.mean(y_rep * np.log(p_clip) + (1 - y_rep) * np.log(1 - p_clip))
        )
        table = compute_binary_calibration_table(y_rep, pd.Series(p_rep))
        ece_reps[rep] = compute_binary_ece(table) if not table.empty else float("nan")

    alpha = (1.0 - ci) / 2.0
    point_log_loss = float(log_loss(y_arr, p_arr, labels=[0, 1]))
    point_table = compute_binary_calibration_table(y_arr, pd.Series(p_arr))
    point_ece = compute_binary_ece(point_table)

    return {
        "log_loss": {
            "point_estimate": point_log_loss,
            "ci_low": float(np.nanquantile(log_loss_reps, alpha)),
            "ci_high": float(np.nanquantile(log_loss_reps, 1.0 - alpha)),
            "n_reps": n_reps,
            "seed": seed,
            "resampling_unit": "game_pk",
        },
        "ece": {
            "point_estimate": point_ece,
            "ci_low": float(np.nanquantile(ece_reps, alpha)),
            "ci_high": float(np.nanquantile(ece_reps, 1.0 - alpha)),
            "n_reps": n_reps,
            "seed": seed,
            "resampling_unit": "game_pk",
        },
    }


def run_opportunity_model_evaluation(
    joined_df: pd.DataFrame,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
) -> tuple[dict[str, dict[str, Any]], dict[str, TrainedOpportunityModel], dict[str, pd.Series]]:
    """Train and evaluate every registered Version 0.7A candidate on IDENTICAL rows.

    Args:
        joined_df: Cleaned development data joined with park geometry
            (`mlb_luck_score.data.join_park_geometry`) -- venue metadata
            must already be joined too (geometry join requires it). Wall
            -proximity features degrade gracefully (dropped, not an error)
            if geometry columns are absent.
        min_venue_samples: Minimum rows for a venue's calibration to be
            marked reliable.

    Returns:
        `(comparison, trained_models, p_out_by_variant)`.
    """
    if "if_fielding_alignment" not in joined_df.columns:
        raise ValueError(
            "joined_df is missing 'if_fielding_alignment'/'of_fielding_alignment' -- these "
            "come straight from cleaned Statcast columns, check the input file."
        )

    df = _prepare_opportunity_columns(joined_df)
    df = _prepare_venue_column(df) if "venue_id" in df.columns else df

    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]
    train_df = eligible[eligible["season"].isin(TRAIN_SEASONS)]
    val_df = eligible[eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "outfield-opportunity-eligible rows to run this evaluation."
        )
    logger.info(
        "Outfield-opportunity-eligible rows: %d training, %d validation", len(train_df), len(val_df)
    )

    comparison: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, TrainedOpportunityModel] = {}
    p_out_by_variant: dict[str, pd.Series] = {}

    for variant, (numeric_features, categorical_features) in _CANDIDATE_FEATURE_BUILDERS.items():
        logger.info("=== [%s] training on %d rows ===", variant, len(train_df))
        trained = train_opportunity_model(
            train_df,
            class_weight=None,
            feature_set_label=variant,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        feature_cols = trained.numeric_features + trained.categorical_features
        p_out = predict_opportunity_proba(trained, val_df[feature_cols])
        validate_opportunity_probabilities(p_out)

        y_true = val_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
        table = compute_binary_calibration_table(y_true, p_out)
        summary: dict[str, Any] = {
            "variant": variant,
            "features": feature_cols,
            "sample_count": int(len(val_df)),
            "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, p_out.to_numpy())),
            "expected_calibration_error": compute_binary_ece(table),
            "converted_to_out_rate": float(y_true.mean()),
            "mean_predicted_p_out": float(p_out.mean()),
            "opportunity_subgroups": compute_opportunity_subgroups(y_true, p_out, val_df),
            "movement_direction_subgroup_note": MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE,
        }
        if "venue_id" in val_df.columns:
            venue_table = compute_binary_calibration_by_venue(
                y_true, p_out, val_df["venue_id"], min_reliable_samples=min_venue_samples
            )
            summary["calibration_by_venue"] = venue_table.to_dict(orient="records")
        if "responsible_outfielder_id" in val_df.columns:
            defender_table = compute_defender_subgroup_calibration(
                y_true, p_out, val_df["responsible_outfielder_id"]
            )
            summary["calibration_by_defender"] = defender_table.to_dict(orient="records")

        comparison[variant] = summary
        trained_models[variant] = trained
        p_out_by_variant[variant] = p_out

    return comparison, trained_models, p_out_by_variant


def find_material_subgroup_issues(
    subgroups: dict[str, dict[str, Any]],
    *,
    absolute_threshold: float = MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    min_sample_size: int = DEFAULT_MIN_SUBGROUP_SAMPLES,
) -> list[str]:
    """Subgroups whose ECE exceeds an ABSOLUTE quality bar (no baseline to compare against)."""
    flagged = []
    for label, sub in subgroups.items():
        ece = sub.get("ece")
        n = sub.get("sample_count", 0)
        if ece is None or n < min_sample_size:
            continue
        if ece > absolute_threshold:
            flagged.append(label)
    return flagged


def summarize_opportunity_validation(
    comparison: dict[str, dict[str, Any]],
    bootstrap_by_variant: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Absolute-quality validation summary for every trained Version 0.7A candidate.

    No `recommend_adopt`-style boolean here -- there is no prior production
    model to adopt OVER (see module docstring). Reports whether each
    candidate clears an absolute calibration quality bar overall and by
    subgroup/venue, for a maintainer to read before treating its
    probabilities as the opportunity-difficulty score feeding Version 0.7B.
    """
    per_candidate: dict[str, Any] = {}
    for variant, summary in comparison.items():
        subgroup_issues = find_material_subgroup_issues(summary.get("opportunity_subgroups", {}))
        venue_issues = [
            row["venue_id"]
            for row in summary.get("calibration_by_venue", [])
            if row.get("reliable") and (row.get("ece") or 0) > MATERIAL_ECE_ABSOLUTE_THRESHOLD
        ]
        overall_ece_ok = summary["expected_calibration_error"] <= MATERIAL_ECE_ABSOLUTE_THRESHOLD
        per_candidate[variant] = {
            "overall_ece": summary["expected_calibration_error"],
            "overall_ece_within_threshold": overall_ece_ok,
            "subgroup_ece_issues": subgroup_issues,
            "venue_ece_issues": venue_issues,
            "bootstrap": bootstrap_by_variant.get(variant),
            "passes_basic_validation": overall_ece_ok and not subgroup_issues and not venue_issues,
        }
    return {
        "per_candidate": per_candidate,
        "typical_position_proxy_v07_built": False,
        "typical_position_proxy_v07_note": (
            "Not built -- no era-appropriate (2021-2024), citable public source for typical "
            "outfielder starting depth was found. See module docstring for the sources checked "
            "and why they were judged too stale/biased to use."
        ),
        "movement_direction_subgroup_note": MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE,
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

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        comparison, _trained_models, p_out_by_variant = run_opportunity_model_evaluation(
            df, min_venue_samples=args.min_venue_samples
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    df_prepared = _prepare_opportunity_columns(df)
    df_prepared = (
        _prepare_venue_column(df_prepared) if "venue_id" in df_prepared.columns else df_prepared
    )
    eligible = df_prepared[df_prepared["outfield_opportunity_eligible"].astype(bool)]
    val_df = eligible[eligible["season"].isin(VALIDATION_SEASONS)]
    y_true = val_df[OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()

    bootstrap_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in ALL_V07A_CANDIDATES:
        logger.info("Running bootstrap CI for %s ...", variant)
        bootstrap_by_variant[variant] = compute_opportunity_bootstrap(
            y_true,
            p_out_by_variant[variant],
            val_df["game_pk"],
            n_reps=args.n_bootstrap_reps,
            seed=args.bootstrap_seed,
        )

    validation_summary = summarize_opportunity_validation(comparison, bootstrap_by_variant)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "opportunity_model_comparison_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "comparison": comparison,
                "bootstrap": bootstrap_by_variant,
                "validation_summary": validation_summary,
            },
            indent=2,
            default=str,
        )
    )

    for variant, summary in comparison.items():
        logger.info(
            "[%s] log_loss=%.6f ece=%.6f brier=%.6f n=%d",
            variant,
            summary["binary_log_loss"],
            summary["expected_calibration_error"],
            summary["brier_score"],
            summary["sample_count"],
        )
    logger.info("Validation summary: %s", validation_summary)
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
