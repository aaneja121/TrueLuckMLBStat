"""Contact Luck v0.7: per-play air-ball component report.

For each outfield-opportunity-eligible play (`mlb_luck_score.eligibility.
add_outfield_opportunity_eligibility`), assembles FOUR SEPARATE, NAMED
quantities:

  1. `expected_run_value_contact_model` / `residual_contact_luck_runs` --
     the EXISTING Version 0.2 contact-model expectation and raw contact
     luck (`mlb_luck_score.scoring.contact_luck.
     compute_raw_contact_luck_runs`'s formula, vectorized here rather than
     reimplemented -- same `selected production baseline` model, same fixed
     run-value table).
  2. `p_out_opportunity` -- the Version 0.7A opportunity-difficulty model's
     `P(an average MLB outfielder converts this into an out)` (`mlb_luck_
     score.models.train_opportunity_model`).
  3. `defensive_execution` / `batter_favorable_defensive_circumstance` --
     the Version 0.7B comparison of the actual result against opportunity
     difficulty (`mlb_luck_score.scoring.defensive_execution`).

**These four quantities are DELIBERATELY NOT COMBINED into one score here.**
Per the task's explicit instruction, this module reports them side by side
for the SAME play so a reader can inspect how they relate, but does not sum,
weight, or otherwise merge them into a single "true luck" number -- that
combination (if it ever happens) requires resolving real double-counting
questions first (e.g. `residual_contact_luck_runs` is computed from a
contact model that does NOT currently include opportunity/execution
features, so its relationship to `defensive_execution` is not yet a clean
decomposition -- see `mlb_luck_score.scoring.weather_attribution`'s module
docstring for the same double-counting caution applied to weather).
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.features.build_contact_features import OPPORTUNITY_TARGET_COLUMN
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
from mlb_luck_score.scoring.defensive_execution import compute_defensive_execution
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

#: Identifier columns copied through to the report (when present) purely so
#: individual rows can be traced back to a real play -- never used as model
#: features anywhere in this module.
_ID_COLUMNS: tuple[str, ...] = ("game_pk", "at_bat_number", "pitch_number")


class AirBallComponentError(ValueError):
    """Raised when inputs to the air-ball component report are invalid."""


def build_air_ball_component_report(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    opportunity_trained: TrainedOpportunityModel,
    *,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
) -> pd.DataFrame:
    """Build the four-component report for every row of `df`.

    Args:
        df: Rows already filtered to `outfield_opportunity_eligible`
            (`mlb_luck_score.eligibility.add_outfield_opportunity_
            eligibility`) and through `mlb_luck_score.features.
            build_contact_features.add_outfield_opportunity_features` (for
            `converted_to_out` and the opportunity feature columns) -- must
            also have every feature column `contact_trained` and
            `opportunity_trained` need.
        contact_trained: A fitted 5-class contact model (`mlb_luck_score.
            models.train_contact_model.train_model`) -- typically
            `baseline_v02`, the selected production model.
        opportunity_trained: A fitted Version 0.7A opportunity model
            (`mlb_luck_score.models.train_opportunity_model.
            train_opportunity_model`) -- typically `measured_contact_
            only_v07`.
        run_value_map: Fixed run-value table. Defaults to `mlb_luck_score.
            scoring.run_values.DEFAULT_RUN_VALUE_MAP` -- the SAME table
            `mlb_luck_score.scoring.contact_luck` uses, never a
            redefinition (see CLAUDE.md).

    Returns:
        A DataFrame (same index as `df`) with any available `_ID_COLUMNS`,
        `expected_run_value_contact_model`, `actual_run_value`,
        `residual_contact_luck_runs`, `p_out_opportunity`,
        `converted_to_out`, `defensive_execution`, and `batter_favorable_
        defensive_circumstance`. See module docstring: these are reported
        together, never summed into one score.

    Raises:
        AirBallComponentError: If `outcome_class` contains a value not in
            `run_value_map` (should not happen for `eligible_for_training`
            rows, which `outfield_opportunity_eligible` is a subset of).
    """
    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    validate_probabilities(contact_proba)
    expected_run_value = compute_expected_run_value_vectorized(contact_proba, run_value_map)

    actual_run_value = df["outcome_class"].astype(str).map(run_value_map)
    if actual_run_value.isna().any():
        missing = df.loc[actual_run_value.isna(), "outcome_class"].unique().tolist()
        raise AirBallComponentError(
            f"outcome_class contains value(s) not in run_value_map: {missing}"
        )
    residual_contact_luck_runs = actual_run_value - expected_run_value

    opportunity_feature_cols = (
        opportunity_trained.numeric_features + opportunity_trained.categorical_features
    )
    p_out_opportunity = predict_opportunity_proba(opportunity_trained, df[opportunity_feature_cols])
    validate_opportunity_probabilities(p_out_opportunity)

    converted_to_out = df[OPPORTUNITY_TARGET_COLUMN].astype(int)
    execution = compute_defensive_execution(converted_to_out, p_out_opportunity)

    report = pd.DataFrame(
        {
            "expected_run_value_contact_model": expected_run_value,
            "actual_run_value": actual_run_value.astype(float),
            "residual_contact_luck_runs": residual_contact_luck_runs,
            "p_out_opportunity": p_out_opportunity,
            "converted_to_out": converted_to_out,
            "defensive_execution": execution["defensive_execution"],
            "batter_favorable_defensive_circumstance": execution[
                "batter_favorable_defensive_circumstance"
            ],
        },
        index=df.index,
    )
    for col in reversed(_ID_COLUMNS):
        if col in df.columns:
            report.insert(0, col, df[col])
    return report
