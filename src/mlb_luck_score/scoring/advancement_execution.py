"""Contact Luck v0.9: per-play batter-runner advancement component report.

For each fair OUTFIELD air-ball row (`bb_type` in `mlb_luck_score.
eligibility.ADVANCEMENT_ELIGIBLE_BB_TYPES`) where the batter safely reaches
at least first, assembles SIX components side by side -- **never summed**,
same reasoning as `mlb_luck_score.scoring.air_ball_components`/
`infield_ball_components` (this contact model does not currently include
advancement features, so its relationship to the advancement components is
not yet a clean decomposition; per the task's explicit "Do not automatically
sum those components yet"):

  1. `expected_contact_base` -- the EXISTING, UNCHANGED Version 0.2 contact
     model's OWN prediction of how many bases this contact was worth BEFORE
     anything about fielding/advancement happened:
     `sum(contact_p_<label> * bases(<label>))` (`out`=0, `single`=1,
     `double`=2, `triple`=3, `home_run`=4).
  2. `actual_final_batter_base` -- the REAL observed outcome, `mlb_luck_
     score.eligibility.ADVANCEMENT_BASE_VALUE[batter_final_base]`.
  3. `advancement_opportunity` -- the Version 0.9 model's own predicted
     `P(batter_final_base = <label>)` distribution, collapsed to an expected
     value via the SAME `ADVANCEMENT_BASE_VALUE` scale (`mlb_luck_score.
     models.compare_advancement_models.compute_expected_advancement_index`)
     -- "what base would an AVERAGE batter-runner be expected to reach,
     given this contact and context."
  4. `batter_runner_advancement_execution` = `actual_final_batter_base -
     advancement_opportunity` -- positive: this batter-runner advanced
     FURTHER than an average batter-runner would in the same situation
     (favorable for the batter). Negative: fell short (e.g. held up, or was
     thrown out attempting an extra base).
  5. `defensive_advancement_effect` = `-batter_runner_advancement_execution`
     -- the SAME quantity from the DEFENSE's perspective, the identical
     sign-flip convention `mlb_luck_score.scoring.defensive_execution`/
     `infield_execution` already use for their own batter-vs-defense
     perspective pairs. This is NOT a separate causal estimate of "how much
     of the outcome was the defense's doing" -- a single joint model cannot
     cleanly separate batter-speed effects from defensive-positioning
     effects without assuming a counterfactual "average defense" that this
     module does not construct (see "Scope" below).
  6. `residual_uncertainty` -- the VARIANCE of the predicted base-value
     distribution around `advancement_opportunity`:
     `sum(p(label) * (bases(label) - advancement_opportunity)^2)`. How much
     genuine outcome variance remains even after conditioning on everything
     the model knows -- large for a genuinely uncertain play, small for a
     near-deterministic one. Same units (bases^2) as the other components,
     unlike Shannon entropy.

## Scope: this does NOT separate batter execution from defensive execution
(task's own framing)

`batter_runner_advancement_execution`/`defensive_advancement_effect` are a
single BINARY-PERSPECTIVE-FLIP pair, not two independently-estimated causal
effects -- see component 5's note above. Isolating a genuine "what would an
average batter have done against THIS specific defense" counterfactual
would require a materially different, two-sided model this phase does not
build. This mirrors `mlb_luck_score.scoring.infield_execution`'s "Scope"
section (does not isolate pickup/transfer/footwork/throwing) and `air_ball_
components`' "not yet a clean decomposition" caveat -- the same caution,
applied to a new domain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_luck_score.eligibility import (
    REASON_NOT_ADVANCEMENT_SAFE_EVENT,
    REASON_NOT_OUTFIELD_AIR_BALL,
    REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY,
    REASON_UNRESOLVED_DES_PARSE,
)
from mlb_luck_score.models.compare_advancement_models import (
    ADVANCEMENT_BASE_VALUE,
    compute_expected_advancement_index,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
)
from mlb_luck_score.models.train_advancement_model import (
    TrainedAdvancementModel,
    predict_advancement_proba,
    validate_advancement_probabilities,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    validate_probabilities,
)

STATUS_CALIBRATED_ADVANCEMENT = "calibrated_advancement_model"
STATUS_PROVISIONAL_ADVANCEMENT = "provisional_advancement_model"
STATUS_EXCLUDED_TRIVIAL_HOME_RUN = "excluded_trivial_home_run"
STATUS_UNAVAILABLE_MISSING_INPUTS = "unavailable_missing_inputs"
STATUS_UNAVAILABLE_NOT_OUTFIELD_AIR_BALL = "unavailable_not_outfield_air_ball"

#: `mlb_luck_score.eligibility` `advancement_exclusion_reason` values that
#: mean "a required input is genuinely missing/unusable" (including an
#: unresolved des parse, since safeguard #3 says an unresolved play is
#: never force-labeled -- reported here as unavailable, not guessed).
_MISSING_INPUT_REASONS: frozenset[str] = frozenset(
    {REASON_NOT_ADVANCEMENT_SAFE_EVENT, REASON_UNRESOLVED_DES_PARSE}
)

_ID_COLUMNS: tuple[str, ...] = ("game_pk", "at_bat_number", "pitch_number")

#: `out` contributes 0 bases -- same value scale as `ADVANCEMENT_BASE_VALUE`,
#: extended with the contact model's own `out` class.
_CONTACT_LABEL_BASE_VALUE: dict[str, float] = {
    "out": 0.0,
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
    "home_run": 4.0,
}


class AdvancementComponentError(ValueError):
    """Raised when inputs to the advancement component report are invalid."""


def _advancement_status_for_row(exclusion_reason: object, overall_status: str) -> str:
    """`overall_status` collapses to a boolean gate -- the SAME pattern
    `mlb_luck_score.scoring.infield_ball_components` uses (see that
    module's fix note): every eligible row gets `calibrated_advancement_
    model` ONLY if `overall_status` is fully `OVERALL_STATUS_CALIBRATED`;
    any other status gets `provisional_advancement_model`. The score is
    still computed and reported for every eligible row.
    """
    is_missing = exclusion_reason is None or (
        isinstance(exclusion_reason, float) and pd.isna(exclusion_reason)
    )
    if is_missing:
        return (
            STATUS_CALIBRATED_ADVANCEMENT
            if overall_status == OVERALL_STATUS_CALIBRATED
            else STATUS_PROVISIONAL_ADVANCEMENT
        )
    if exclusion_reason == REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY:
        return STATUS_EXCLUDED_TRIVIAL_HOME_RUN
    if exclusion_reason == REASON_NOT_OUTFIELD_AIR_BALL:
        return STATUS_UNAVAILABLE_NOT_OUTFIELD_AIR_BALL
    if exclusion_reason in _MISSING_INPUT_REASONS:
        return STATUS_UNAVAILABLE_MISSING_INPUTS
    raise AdvancementComponentError(
        f"Unhandled advancement_exclusion_reason: {exclusion_reason!r} -- update "
        "mlb_luck_score.scoring.advancement_execution accordingly."
    )


def build_advancement_component_report(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    advancement_trained: TrainedAdvancementModel,
    *,
    overall_status: str,
) -> pd.DataFrame:
    """Build the per-play advancement component report for every row of `df`.

    Args:
        df: Fair OUTFIELD air-ball rows (`bb_type` in `mlb_luck_score.
            eligibility.ADVANCEMENT_ELIGIBLE_BB_TYPES`), already through
            `mlb_luck_score.eligibility.add_advancement_eligibility` +
            `mlb_luck_score.features.build_contact_features.
            add_advancement_features` (for `advancement_eligible`,
            `advancement_exclusion_reason`, `batter_final_base`, and the
            Version 0.9 feature columns) -- must also have every feature
            column `contact_trained` and `advancement_trained` need,
            including `mlb_luck_score.features.build_contact_features.
            add_advancement_contact_probability_features`'s output.
        contact_trained: A fitted 5-class contact model, typically
            `baseline_v02`.
        advancement_trained: A fitted Version 0.9 advancement model (the
            model-selection winner from `mlb_luck_score.models.
            compare_advancement_models.run_advancement_model_selection`).
        overall_status: `mlb_luck_score.models.compare_advancement_models.
            summarize_advancement_calibration`'s `overall_status` -- drives
            `advancement_status` for every ELIGIBLE row (see
            `_advancement_status_for_row`).

    Returns:
        A DataFrame (same index as `df`) with any available `_ID_COLUMNS`,
        the six components described in the module docstring (all `NaN` for
        non-eligible rows), and `advancement_status`.
    """
    if "bb_type" in df.columns and not df["bb_type"].isin(("fly_ball", "line_drive")).all():
        raise AdvancementComponentError(
            "build_advancement_component_report expects only outfield air-ball rows "
            "(bb_type in {'fly_ball', 'line_drive'})."
        )

    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    validate_probabilities(contact_proba)
    contact_values = np.array([_CONTACT_LABEL_BASE_VALUE[c] for c in contact_proba.columns])
    expected_contact_base = pd.Series(contact_proba.to_numpy() @ contact_values, index=df.index)

    eligible_mask = df["advancement_eligible"].astype(bool)
    actual_final_batter_base = pd.Series(float("nan"), index=df.index)
    advancement_opportunity = pd.Series(float("nan"), index=df.index)
    batter_runner_advancement_execution = pd.Series(float("nan"), index=df.index)
    defensive_advancement_effect = pd.Series(float("nan"), index=df.index)
    residual_uncertainty = pd.Series(float("nan"), index=df.index)

    if eligible_mask.any():
        eligible_df = df[eligible_mask]
        advancement_feature_cols = (
            advancement_trained.numeric_features + advancement_trained.categorical_features
        )
        proba = predict_advancement_proba(
            advancement_trained, eligible_df[advancement_feature_cols]
        )
        validate_advancement_probabilities(proba)

        opportunity = compute_expected_advancement_index(proba)
        actual = eligible_df["batter_final_base"].map(ADVANCEMENT_BASE_VALUE).astype(float)
        base_values = np.array([ADVANCEMENT_BASE_VALUE[label] for label in proba.columns])
        variance = pd.Series(
            (
                proba.to_numpy()
                * (base_values[np.newaxis, :] - opportunity.to_numpy()[:, np.newaxis]) ** 2
            ).sum(axis=1),
            index=eligible_df.index,
        )

        advancement_opportunity.loc[eligible_mask] = opportunity
        actual_final_batter_base.loc[eligible_mask] = actual
        batter_runner_advancement_execution.loc[eligible_mask] = actual - opportunity
        defensive_advancement_effect.loc[eligible_mask] = opportunity - actual
        residual_uncertainty.loc[eligible_mask] = variance

    exclusion_reason = (
        df["advancement_exclusion_reason"]
        if "advancement_exclusion_reason" in df.columns
        else pd.Series(None, index=df.index, dtype=object)
    )
    advancement_status = exclusion_reason.apply(
        lambda reason: _advancement_status_for_row(reason, overall_status)
    )

    report = pd.DataFrame(
        {
            "expected_contact_base": expected_contact_base,
            "actual_final_batter_base": actual_final_batter_base,
            "advancement_opportunity": advancement_opportunity,
            "batter_runner_advancement_execution": batter_runner_advancement_execution,
            "defensive_advancement_effect": defensive_advancement_effect,
            "residual_uncertainty": residual_uncertainty,
            "advancement_status": advancement_status,
        },
        index=df.index,
    )
    for col in reversed(_ID_COLUMNS):
        if col in df.columns:
            report.insert(0, col, df[col])
    return report
