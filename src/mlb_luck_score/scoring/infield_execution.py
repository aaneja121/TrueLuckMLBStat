"""Contact Luck v0.8: per-play combined infield execution scoring.

Compares the ACTUAL result of an infield-opportunity-eligible play against
the Version 0.8 `infield_contact_only_v08` opportunity-difficulty model's
prediction (task Phase 8):

    defensive_execution_probability = actual_out - predicted_out_probability

Where `actual_out` is 1 if the batter-runner was retired on this play, 0
otherwise (`mlb_luck_score.eligibility.add_infield_opportunity_eligibility`'s
`y_out` -- a reached-on-error play is `actual_out = 0`, since the defense did
NOT complete the conversion, per task Phase 3), and `predicted_out_
probability` is `mlb_luck_score.models.train_opportunity_model.
predict_opportunity_proba`'s output for the Version 0.8 model -- `P(an
average MLB infielder converts this opportunity into an out)`.

This is the SAME formula/sign-convention pattern as `mlb_luck_score.scoring.
defensive_execution` (Version 0.7B, outfield) -- a deliberately parallel,
NOT shared, implementation: the two scoring domains (outfield air balls,
infield ground balls) use different opportunity models and different
eligibility populations, and CLAUDE.md's "prefer small, reviewable changes"/
"Version 0.7 is frozen" both argue against retrofitting a shared abstraction
onto the existing, working Version 0.7B module just to avoid this small
duplication.

## Sign convention (defense's perspective)

  - `defensive_execution_probability` near 0: the play went about as an
    average infielder would be expected to.
  - Strongly POSITIVE: an out was made on a LOW-probability opportunity
    (e.g. `predicted_out_probability=0.5`, out made -> `+0.5`) -- this
    defense outperformed an average infielder on this play.
  - Strongly NEGATIVE: a HIGH-probability opportunity was NOT converted
    (e.g. `predicted_out_probability=0.95`, no out -> `-0.95`) -- this
    defense underperformed an average infielder on this play.

## Batter's perspective is the OPPOSITE sign (task Phase 8)

    batter_perspective_infield_execution = predicted_out_probability - actual_out

  - POSITIVE `batter_perspective_infield_execution`: the defense converted
    LESS successfully than an average defense would be expected to convert
    the opportunity -- favorable circumstance for the batter.
  - NEGATIVE: the defense converted MORE successfully than expected --
    unfavorable circumstance for the batter.

## Scope: this does NOT isolate fielding from throwing (task Phase 8)

This is a BINARY, opportunity-relative execution measure -- it says nothing
about pickup, transfer, footwork, or throwing execution specifically (none of
which can be distinguished in public per-play data; see `mlb_luck_score.
models.compare_infield_opportunity` module docstring's Phase 1 audit). A
clean barehand play and a workmanlike fielding play of the exact same
opportunity difficulty both register as whatever `actual_out - predicted_
out_probability` computes to. It also does NOT yet claim to separate
positioning, scorer judgment, or unobserved route quality from execution --
see that module's Phase 8 section for the full caveat.

## Deliberately separate from residual contact luck

Same caution as `mlb_luck_score.scoring.defensive_execution`'s module
docstring: `defensive_execution_probability` is NOT the same quantity as
`mlb_luck_score.scoring.contact_luck.compute_raw_contact_luck_runs`. See
`mlb_luck_score.scoring.infield_ball_components` for how all these pieces are
reported side by side WITHOUT being summed into one score.
"""

from __future__ import annotations

import pandas as pd


class InfieldExecutionError(ValueError):
    """Raised when inputs to the infield-execution calculation are invalid."""


def compute_infield_execution(
    actual_out: pd.Series, predicted_out_probability: pd.Series
) -> pd.DataFrame:
    """Per-play combined infield execution: `actual_out - predicted_out_probability`.

    Args:
        actual_out: 1 if the batter-runner was retired, 0 otherwise, for each
            infield-opportunity-eligible play (`y_out`).
        predicted_out_probability: The Version 0.8 model's predicted
            `P(out)` for the SAME rows, in the SAME order.

    Returns:
        A DataFrame (same index as the inputs) with `defensive_execution_
        probability` (defense's perspective -- see module docstring) and
        `batter_perspective_infield_execution` (`= -defensive_execution_
        probability`, the batter's perspective).

    Raises:
        InfieldExecutionError: If the two inputs have mismatched index/
            length, `predicted_out_probability` is outside `[0, 1]`, or
            `actual_out` contains a value other than 0/1.
    """
    if len(actual_out) != len(predicted_out_probability) or not actual_out.index.equals(
        predicted_out_probability.index
    ):
        raise InfieldExecutionError(
            "actual_out and predicted_out_probability must have the identical index/row "
            "order -- they must describe the SAME plays."
        )

    p_arr = predicted_out_probability.to_numpy(dtype=float)
    if ((p_arr < 0) | (p_arr > 1)).any():
        raise InfieldExecutionError("predicted_out_probability must be within [0, 1]")

    actual_arr = actual_out.to_numpy()
    if not set(pd.unique(actual_arr)).issubset({0, 1}):
        raise InfieldExecutionError("actual_out must contain only 0/1 values")

    defensive_execution_probability = actual_out.astype(float) - predicted_out_probability.astype(
        float
    )
    return pd.DataFrame(
        {
            "defensive_execution_probability": defensive_execution_probability,
            "batter_perspective_infield_execution": -defensive_execution_probability,
        },
        index=actual_out.index,
    )
