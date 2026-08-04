"""Contact Luck v0.8: per-play infield component report and reporting statuses.

For each fair GROUND-ball row (`bb_type == "ground_ball"`), assembles the SAME
kind of side-by-side, NEVER-SUMMED component report `mlb_luck_score.scoring.
air_ball_components` builds for outfield air balls (task Phase 8/9):

  1. `expected_run_value_contact_model` / `residual_contact_luck_runs` -- the
     EXISTING Version 0.2 contact-model expectation and raw contact luck.
     UNAVAILABLE (`NaN`) for reached-on-error rows specifically: Version
     0.1's `mlb_luck_score.eligibility.map_outcome_class` deliberately leaves
     `outcome_class` null for `field_error` (the batter-runner's resulting
     BASE can't be reliably determined from public data), so there is
     nothing to look up in `run_value_map` for these rows -- reported as
     unavailable, never guessed.
  2. `p_out_opportunity` -- the Version 0.8 `infield_contact_only_v08`
     model's `P(an average MLB infielder converts this into an out)`
     (`mlb_luck_score.models.train_opportunity_model`).
  3. `defensive_execution_probability` / `batter_perspective_infield_
     execution` -- `mlb_luck_score.scoring.infield_execution`.
  4. `reached_on_error` -- REPORTING ONLY (task Phase 3), never a predictor.
  5. `infield_opportunity_status` -- one of five explicit statuses (task
     Phase 9), see `build_infield_ball_component_report`'s docstring.

**These are reported side by side, NEVER summed into one score** -- same
reasoning as `air_ball_components`'s module docstring (this contact model
does not currently include infield-opportunity/execution features, so its
relationship to `defensive_execution_probability` is not yet a clean
decomposition).

This module does NOT yet claim to isolate pickup, transfer, footwork, or
throwing execution -- see `mlb_luck_score.models.compare_infield_opportunity`
module docstring Phase 8, and `mlb_luck_score.scoring.infield_execution`'s
"Scope" section.
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.eligibility import (
    REASON_AMBIGUOUS_INFIELD_EVENT,
    REASON_APPEAL_PLAY,
    REASON_BUNT_EXCLUDED,
    REASON_EXCLUDED_STRATEGIC_PLAY,
    REASON_INTERFERENCE_OR_OBSTRUCTION,
    REASON_MISSING_INFIELD_CONTACT_DATA,
    REASON_MISSING_INFIELD_POSITION,
    REASON_OUTFIELD_CREDITED_HIT_LOCATION,
    REASON_RUNDOWN_PLAY,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    validate_probabilities,
)
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    predict_opportunity_proba,
    validate_opportunity_probabilities,
)
from mlb_luck_score.scoring.infield_execution import compute_infield_execution
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

STATUS_CALIBRATED_INFIELD_OPPORTUNITY = "calibrated_infield_opportunity"
STATUS_PROVISIONAL_INFIELD_OPPORTUNITY = "provisional_infield_opportunity"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
STATUS_EXCLUDED_STRATEGIC_PLAY = "excluded_strategic_play"
STATUS_UNAVAILABLE_MISSING_INPUTS = "unavailable_missing_inputs"

#: `infield_opportunity_exclusion_reason` values (`mlb_luck_score.
#: eligibility.add_infield_opportunity_eligibility`) that mean "this is a
#: real play, but a STRATEGIC one the straightforward batter-runner model was
#: never trained to represent" -- task's explicit "Do not score excluded
#: fielder's-choice or double-play situations using a model trained on
#: straightforward batter-runner plays."
_STRATEGIC_EXCLUSION_REASONS: frozenset[str] = frozenset(
    {
        REASON_EXCLUDED_STRATEGIC_PLAY,
        REASON_BUNT_EXCLUDED,
        REASON_INTERFERENCE_OR_OBSTRUCTION,
        REASON_APPEAL_PLAY,
        REASON_RUNDOWN_PLAY,
    }
)
#: `infield_opportunity_exclusion_reason` values that mean "a required input
#: is genuinely missing/unusable," not a strategic-play distinction.
_MISSING_INPUT_REASONS: frozenset[str] = frozenset(
    {
        REASON_MISSING_INFIELD_POSITION,
        REASON_OUTFIELD_CREDITED_HIT_LOCATION,
        REASON_MISSING_INFIELD_CONTACT_DATA,
        REASON_AMBIGUOUS_INFIELD_EVENT,
    }
)

_ID_COLUMNS: tuple[str, ...] = ("game_pk", "at_bat_number", "pitch_number")


class InfieldBallComponentError(ValueError):
    """Raised when inputs to the infield component report are invalid."""


def _infield_opportunity_status_for_row(exclusion_reason: object, overall_status: str) -> str:
    is_missing = exclusion_reason is None or (
        isinstance(exclusion_reason, float) and pd.isna(exclusion_reason)
    )
    if is_missing:
        if overall_status == OVERALL_STATUS_CALIBRATED:
            return STATUS_CALIBRATED_INFIELD_OPPORTUNITY
        if overall_status == OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE:
            return STATUS_INSUFFICIENT_EVIDENCE
        return STATUS_PROVISIONAL_INFIELD_OPPORTUNITY
    if exclusion_reason in _STRATEGIC_EXCLUSION_REASONS:
        return STATUS_EXCLUDED_STRATEGIC_PLAY
    if exclusion_reason in _MISSING_INPUT_REASONS:
        return STATUS_UNAVAILABLE_MISSING_INPUTS
    raise InfieldBallComponentError(
        f"Unhandled infield_opportunity_exclusion_reason: {exclusion_reason!r} -- update "
        "_STRATEGIC_EXCLUSION_REASONS/_MISSING_INPUT_REASONS in "
        "mlb_luck_score.scoring.infield_ball_components."
    )


def build_infield_ball_component_report(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    infield_trained: TrainedOpportunityModel,
    *,
    overall_status: str,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
) -> pd.DataFrame:
    """Build the per-play infield component report for every row of `df`.

    Args:
        df: Fair GROUND-ball rows (`bb_type == "ground_ball"`), already
            through `mlb_luck_score.eligibility.compute_eligibility` (for
            `outcome_class`) AND `add_infield_opportunity_eligibility` +
            `mlb_luck_score.features.build_contact_features.
            add_infield_opportunity_features` (for `infield_opportunity_
            eligible`, `infield_opportunity_exclusion_reason`, `y_out`,
            `reached_on_error`, and the Version 0.8 feature columns) -- must
            also have every feature column `contact_trained` and
            `infield_trained` need.
        contact_trained: A fitted 5-class contact model, typically
            `baseline_v02`.
        infield_trained: A fitted Version 0.8 `infield_contact_only_v08`
            model (the model-selection winner from `mlb_luck_score.models.
            compare_infield_opportunity.run_infield_model_selection`).
        overall_status: `mlb_luck_score.models.compare_infield_opportunity.
            summarize_infield_calibration`'s `overall_status` -- drives
            `infield_opportunity_status` for every ELIGIBLE row (see
            `_infield_opportunity_status_for_row`): `calibrated` ->
            `calibrated_infield_opportunity`; `calibrated_with_limited_
            subgroup_evidence` -> `insufficient_evidence` (some subgroups
            lack evidence -- reported honestly, never as an unconditional
            pass); anything else -> `provisional_infield_opportunity`.
        run_value_map: Fixed run-value table, same as `mlb_luck_score.
            scoring.air_ball_components` -- never a redefinition.

    Returns:
        A DataFrame (same index as `df`) with any available `_ID_COLUMNS`,
        `expected_run_value_contact_model`, `actual_run_value`, `residual_
        contact_luck_runs` (all three `NaN` for reached-on-error rows --
        see module docstring), `p_out_opportunity`, `actual_out`,
        `reached_on_error`, `defensive_execution_probability`, `batter_
        perspective_infield_execution` (the last four `NaN` for
        non-eligible rows), and `infield_opportunity_status`.
    """
    if "bb_type" in df.columns and not (df["bb_type"] == "ground_ball").all():
        raise InfieldBallComponentError(
            "build_infield_ball_component_report expects only bb_type == 'ground_ball' rows."
        )

    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    validate_probabilities(contact_proba)
    expected_run_value = compute_expected_run_value_vectorized(contact_proba, run_value_map)

    outcome_class = (
        df["outcome_class"] if "outcome_class" in df.columns else pd.Series(pd.NA, index=df.index)
    )
    actual_run_value = outcome_class.astype(object).map(run_value_map).astype(float)
    residual_contact_luck_runs = actual_run_value - expected_run_value

    eligible_mask = df["infield_opportunity_eligible"].astype(bool)
    p_out_opportunity = pd.Series(float("nan"), index=df.index)
    actual_out = pd.Series(float("nan"), index=df.index)
    defensive_execution_probability = pd.Series(float("nan"), index=df.index)
    batter_perspective_infield_execution = pd.Series(float("nan"), index=df.index)

    if eligible_mask.any():
        eligible_df = df[eligible_mask]
        infield_feature_cols = (
            infield_trained.numeric_features + infield_trained.categorical_features
        )
        p_out_eligible = predict_opportunity_proba(
            infield_trained, eligible_df[infield_feature_cols]
        )
        validate_opportunity_probabilities(p_out_eligible)
        y_out_eligible = eligible_df["y_out"].astype(int)
        execution = compute_infield_execution(y_out_eligible, p_out_eligible)

        p_out_opportunity.loc[eligible_mask] = p_out_eligible
        actual_out.loc[eligible_mask] = y_out_eligible.astype(float)
        defensive_execution_probability.loc[eligible_mask] = execution[
            "defensive_execution_probability"
        ]
        batter_perspective_infield_execution.loc[eligible_mask] = execution[
            "batter_perspective_infield_execution"
        ]

    reached_on_error = (
        df["reached_on_error"].astype(bool)
        if "reached_on_error" in df.columns
        else pd.Series(False, index=df.index)
    )
    actual_run_value = actual_run_value.mask(reached_on_error)
    residual_contact_luck_runs = residual_contact_luck_runs.mask(reached_on_error)
    expected_run_value_reported = expected_run_value.mask(reached_on_error)

    exclusion_reason = (
        df["infield_opportunity_exclusion_reason"]
        if "infield_opportunity_exclusion_reason" in df.columns
        else pd.Series(None, index=df.index, dtype=object)
    )
    infield_opportunity_status = exclusion_reason.apply(
        lambda reason: _infield_opportunity_status_for_row(reason, overall_status)
    )

    report = pd.DataFrame(
        {
            "expected_run_value_contact_model": expected_run_value_reported,
            "actual_run_value": actual_run_value,
            "residual_contact_luck_runs": residual_contact_luck_runs,
            "p_out_opportunity": p_out_opportunity,
            "actual_out": actual_out,
            "reached_on_error": reached_on_error,
            "defensive_execution_probability": defensive_execution_probability,
            "batter_perspective_infield_execution": batter_perspective_infield_execution,
            "infield_opportunity_status": infield_opportunity_status,
        },
        index=df.index,
    )
    for col in reversed(_ID_COLUMNS):
        if col in df.columns:
            report.insert(0, col, df[col])
    return report
