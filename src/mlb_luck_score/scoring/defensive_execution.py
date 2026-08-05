"""Contact Luck v0.7B: per-play defensive-execution scoring.

Compares the ACTUAL result of an outfield-opportunity-eligible play against
the Version 0.7A opportunity-difficulty model's prediction:

    defensive_execution = actual_out_indicator - p_out_opportunity

Where `actual_out_indicator` is 1 if the batter-runner was put out on this
play, 0 otherwise (`mlb_luck_score.features.build_contact_features.
OUTFIELD_OPPORTUNITY_TARGET_COLUMN`), and `p_out_opportunity` is `mlb_luck_
score.models.train_opportunity_model.predict_opportunity_proba`'s output --
`P(an average MLB outfielder converts this opportunity into an out)`.

## Sign convention (fielder's perspective)

  - `defensive_execution` near 0: the play went about as an average
    outfielder would be expected to (a routine catch stays near-neutral,
    since `p_out_opportunity` is already high and the out was made).
  - `defensive_execution` strongly POSITIVE: an out was made on a
    LOW-probability opportunity (e.g. `p_out_opportunity=0.15`, out was
    made -> `execution = 0.85`) -- strong POSITIVE defensive execution
    (this fielder outperformed an average outfielder on this play).
  - `defensive_execution` strongly NEGATIVE: a HIGH-probability opportunity
    was NOT converted (e.g. `p_out_opportunity=0.9`, no out ->
    `execution = -0.9`) -- strong NEGATIVE defensive execution (this
    fielder underperformed an average outfielder on this play).

## Batter's perspective is the OPPOSITE sign

Per the task's explicit framing: "unexpectedly strong defensive execution is
unfavorable circumstance [for the batter]; poor defense is favorable
circumstance." `compute_defensive_execution` also returns `batter_favorable_
defensive_circumstance = -defensive_execution` for exactly this reason --
strong defense (positive `defensive_execution`) is NEGATIVE circumstance for
the batter, and vice versa. Downstream Contact-Luck-style reasoning about
"favorable/unfavorable circumstance for the batter" should use THIS column,
not `defensive_execution` directly, to avoid a sign-convention mistake.

## Scope: execution here means "was the opportunity converted," not HOW

This is a BINARY, opportunity-relative execution measure -- it says nothing
about reaction time, route efficiency, or throwing accuracy specifically
(none of which are in public data; see `mlb_luck_score.models.
compare_opportunity_models` module docstring's audit). A spectacular diving
catch and a workmanlike catch of the exact same difficulty both register as
whatever `1 - p_out_opportunity` computes to -- this module cannot and does
not distinguish HOW an out was made, only THAT one was (or wasn't), relative
to how often an average outfielder would have made it.

## Deliberately separate from residual contact luck

`defensive_execution` is NOT the same quantity as `mlb_luck_score.scoring.
contact_luck.compute_raw_contact_luck_runs` (`actual_run_value -
expected_run_value`, from the 5-class contact model). The contact model's
`expected_run_value` already reflects, in aggregate, how outfielders
typically perform against a given launch_speed/launch_angle/etc. profile --
`defensive_execution` decomposes THAT aggregate further into "how hard was
this specific opportunity" (Version 0.7A) and "did the specific defender
convert it" (this module), for outfield air balls only. See
`mlb_luck_score.scoring.air_ball_components` for how all of these pieces are
reported side by side WITHOUT being summed into one score.
"""

from __future__ import annotations

import pandas as pd


class DefensiveExecutionError(ValueError):
    """Raised when inputs to the defensive-execution calculation are invalid."""


def compute_defensive_execution(
    actual_out_indicator: pd.Series, p_out_opportunity: pd.Series
) -> pd.DataFrame:
    """Per-play defensive-execution value: `actual_out_indicator - p_out_opportunity`.

    Args:
        actual_out_indicator: 1 if the batter-runner was put out, 0
            otherwise, for each outfield-opportunity-eligible play.
        p_out_opportunity: The Version 0.7A model's predicted `P(out)` for
            the SAME rows, in the SAME order.

    Returns:
        A DataFrame (same index as the inputs) with `defensive_execution`
        (fielder's perspective -- see module docstring) and `batter_
        favorable_defensive_circumstance` (`= -defensive_execution`, the
        batter's perspective).

    Raises:
        DefensiveExecutionError: If the two inputs have mismatched index/
            length, `p_out_opportunity` is outside `[0, 1]`, or `actual_out_
            indicator` contains a value other than 0/1.
    """
    if len(actual_out_indicator) != len(p_out_opportunity) or not actual_out_indicator.index.equals(
        p_out_opportunity.index
    ):
        raise DefensiveExecutionError(
            "actual_out_indicator and p_out_opportunity must have the identical index/row "
            "order -- they must describe the SAME plays."
        )

    p_arr = p_out_opportunity.to_numpy(dtype=float)
    if ((p_arr < 0) | (p_arr > 1)).any():
        raise DefensiveExecutionError("p_out_opportunity must be within [0, 1]")

    actual_arr = actual_out_indicator.to_numpy()
    if not set(pd.unique(actual_arr)).issubset({0, 1}):
        raise DefensiveExecutionError("actual_out_indicator must contain only 0/1 values")

    defensive_execution = actual_out_indicator.astype(float) - p_out_opportunity.astype(float)
    return pd.DataFrame(
        {
            "defensive_execution": defensive_execution,
            "batter_favorable_defensive_circumstance": -defensive_execution,
        },
        index=actual_out_indicator.index,
    )
