"""Contact Luck v0.7C: the gated outfield opportunity/execution report.

Routes every outfield-opportunity-eligible play to the right model (or no
opportunity model at all) via `mlb_luck_score.models.outfield_gating`, and
labels each row's `execution_availability_status` accordingly -- implementing
the task's reporting rule verbatim:

    "Outfield execution score available for calibrated open-field
    opportunities; provisional or unavailable for wall-adjacent
    opportunities."

Four possible statuses:

  - `STATUS_AVAILABLE_OPEN_FIELD`: `mlb_luck_score.models.outfield_gating.
    GATE_OPEN_FIELD` rows, scored with `open_field_v07` (== `measured_
    contact_only_v07`, unchanged) -- calibration already verified acceptable
    (`mlb_luck_score.models.compare_opportunity_models`).
  - `STATUS_AVAILABLE_NEAR_WALL_CALIBRATED`: `GATE_NEAR_WALL` rows, scored
    with the Version 0.7C near-wall specialist, ONLY IF `mlb_luck_score.
    models.compare_near_wall_models.summarize_near_wall_validation`'s
    `near_wall_specialist_calibrated` is `True`.
  - `STATUS_PROVISIONAL_NEAR_WALL`: `GATE_NEAR_WALL` rows, scored with the
    near-wall specialist, but `near_wall_specialist_calibrated` is `False`
    -- the score is COMPUTED and reported (never silently blanked), but
    explicitly labeled lower-confidence, per the task's "provisional... for
    wall-adjacent opportunities" instruction. As of this writing, this is
    the status near-wall rows actually get: the near-wall specialist shows
    a large, real, bootstrap-confirmed calibration improvement over
    `open_field_v07` but has NOT cleared every required check (see that
    module's docstring for the specific unresolved perturbation-check
    finding).
  - `STATUS_UNAVAILABLE_GEOMETRY_FALLBACK`: `GATE_CONTACT_ONLY_FALLBACK`
    rows -- park geometry itself is unreliable for this play, so NEITHER
    opportunity model has trustworthy inputs. `p_out_opportunity`,
    `defensive_execution`, and `batter_favorable_defensive_circumstance`
    are `NaN` for these rows -- only the EXISTING Version 0.2 contact
    -model expectation and residual contact luck are reported (`converted_
    to_out`, the actual observed outcome, is still reported -- it needs no
    model, only the real result).

This module does not decide `near_wall_specialist_calibrated` itself --
callers pass it in (typically read from `mlb_luck_score.models.
compare_near_wall_models.summarize_near_wall_validation`'s real-data
output), so this module doesn't need to re-run the full near-wall
comparison/validation pipeline just to build a report.
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.models.outfield_gating import (
    GATE_CONTACT_ONLY_FALLBACK,
    GATE_NEAR_WALL,
    GATE_OPEN_FIELD,
    assign_outfield_opportunity_gate,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    validate_probabilities,
)
from mlb_luck_score.models.train_opportunity_model import TrainedOpportunityModel
from mlb_luck_score.scoring.air_ball_components import _ID_COLUMNS, build_air_ball_component_report
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

STATUS_AVAILABLE_OPEN_FIELD = "available_open_field"
STATUS_AVAILABLE_NEAR_WALL_CALIBRATED = "available_near_wall_calibrated"
STATUS_PROVISIONAL_NEAR_WALL = "provisional_near_wall"
STATUS_UNAVAILABLE_GEOMETRY_FALLBACK = "unavailable_geometry_fallback"

_REPORT_COLUMNS: tuple[str, ...] = (
    "expected_run_value_contact_model",
    "actual_run_value",
    "residual_contact_luck_runs",
    "p_out_opportunity",
    "converted_to_out",
    "defensive_execution",
    "batter_favorable_defensive_circumstance",
)


def _build_contact_only_segment(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    *,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
) -> pd.DataFrame:
    """Contact-only fallback: contact expectation/residual luck ARE computed;
    opportunity difficulty and defensive execution are NOT (`NaN`) -- park geometry
    is unreliable for these rows, so neither opportunity model has trustworthy inputs.
    """
    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    validate_probabilities(contact_proba)
    expected_run_value = compute_expected_run_value_vectorized(contact_proba, run_value_map)

    actual_run_value = df["outcome_class"].astype(str).map(run_value_map).astype(float)
    residual_contact_luck_runs = actual_run_value - expected_run_value

    converted_to_out = (
        df["converted_to_out"].astype(float)
        if "converted_to_out" in df.columns
        else (df["outcome_class"] == "out").astype(float)
    )

    report = pd.DataFrame(
        {
            "expected_run_value_contact_model": expected_run_value,
            "actual_run_value": actual_run_value,
            "residual_contact_luck_runs": residual_contact_luck_runs,
            "p_out_opportunity": float("nan"),
            "converted_to_out": converted_to_out,
            "defensive_execution": float("nan"),
            "batter_favorable_defensive_circumstance": float("nan"),
        },
        index=df.index,
    )
    for col in reversed(_ID_COLUMNS):
        if col in df.columns:
            report.insert(0, col, df[col])
    return report


def build_gated_outfield_report(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    open_field_trained: TrainedOpportunityModel,
    near_wall_trained: TrainedOpportunityModel,
    *,
    near_wall_specialist_calibrated: bool,
) -> pd.DataFrame:
    """Build the gated per-play report for every row of `df`.

    Args:
        df: `outfield_opportunity_eligible` rows, already through
            `mlb_luck_score.eligibility.add_outfield_opportunity_
            eligibility`, `mlb_luck_score.features.build_contact_features.
            add_geometry_interaction_features` (if geometry columns are
            present), and `add_outfield_opportunity_features` -- must have
            every feature column all three trained models need.
        contact_trained: Fitted 5-class contact model (typically
            `baseline_v02`).
        open_field_trained: Fitted Version 0.7A `measured_contact_only_v07`.
        near_wall_trained: Fitted Version 0.7C near-wall specialist (the
            selection winner from `mlb_luck_score.models.
            compare_near_wall_models.run_near_wall_model_selection`).
        near_wall_specialist_calibrated: `mlb_luck_score.models.
            compare_near_wall_models.summarize_near_wall_validation`'s
            `near_wall_specialist_calibrated` -- drives whether near-wall
            rows get `STATUS_AVAILABLE_NEAR_WALL_CALIBRATED` or `STATUS_
            PROVISIONAL_NEAR_WALL`.

    Returns:
        A DataFrame (same rows/index as `df`, in `df`'s original order)
        with `_REPORT_COLUMNS` plus `execution_availability_status`.
    """
    gate = assign_outfield_opportunity_gate(df)
    segments: list[pd.DataFrame] = []

    open_field_df = df[gate == GATE_OPEN_FIELD]
    if len(open_field_df):
        report = build_air_ball_component_report(open_field_df, contact_trained, open_field_trained)
        report["execution_availability_status"] = STATUS_AVAILABLE_OPEN_FIELD
        segments.append(report)

    near_wall_df = df[gate == GATE_NEAR_WALL]
    if len(near_wall_df):
        report = build_air_ball_component_report(near_wall_df, contact_trained, near_wall_trained)
        report["execution_availability_status"] = (
            STATUS_AVAILABLE_NEAR_WALL_CALIBRATED
            if near_wall_specialist_calibrated
            else STATUS_PROVISIONAL_NEAR_WALL
        )
        segments.append(report)

    fallback_df = df[gate == GATE_CONTACT_ONLY_FALLBACK]
    if len(fallback_df):
        report = _build_contact_only_segment(fallback_df, contact_trained)
        report["execution_availability_status"] = STATUS_UNAVAILABLE_GEOMETRY_FALLBACK
        segments.append(report)

    if not segments:
        return pd.DataFrame(columns=[*_REPORT_COLUMNS, "execution_availability_status"])

    combined = pd.concat(segments)
    return combined.loc[df.index]
