"""Contact Luck v0.6: per-play alignment-positioning run-value attribution.

Only meaningful once an alignment-aware candidate has passed the Version 0.6
adoption rule (`mlb_luck_score.models.compare_alignment_aware.
recommend_alignment_adoption`) -- this module provides the reusable
computation, but a caller should not treat its output as a validated,
headline Contact Luck component unless that gate has actually been passed
(see README.md "Alignment-aware positioning (Version 0.6)"). Building and
testing this infrastructure does not itself constitute adoption.

    expected_run_value_actual_alignment  = E[run_value | actual starting alignment]
    expected_run_value_typical_alignment = E[run_value | typical ("Standard") alignment,
                                                           every other feature held constant]
    positioning_effect = expected_run_value_actual_alignment
                        - expected_run_value_typical_alignment

Positive `positioning_effect` means the defense's ACTUAL starting alignment
that play made contact MORE favorable to the batter than a typical
("Standard"/"Standard") alignment would have; negative means LESS favorable
(the shift/strategic alignment worked as intended, from the batter's
perspective). See `mlb_luck_score.features.build_contact_features.
generate_typical_alignment_rows` for the exact typical-alignment values.

This is EXPLICITLY an external-circumstance contribution -- "did today's
defensive alignment happen to be more or less favorable than usual for this
specific batted ball" -- NOT raw residual contact luck (that is still
`actual_run_value - expected_run_value` from the SELECTED model's own
prediction, `mlb_luck_score.scoring.contact_luck.
compute_raw_contact_luck_runs`) and NOT a measure of defensive EXECUTION
(reaction, route, pickup, transfer, throw -- entirely unobserved by public
data; see module docstring of `mlb_luck_score.models.
compare_alignment_aware`). A well-executed play behind a bad alignment and a
poorly-executed play behind a good alignment can have the same
`positioning_effect` -- this module only decomposes "where the fielders
started," never how well they moved from there.

This is DELIBERATELY KEPT SEPARATE from `mlb_luck_score.scoring.
contact_luck.compute_raw_contact_luck_runs` -- do NOT add
`positioning_effect` on top of `raw_contact_luck_runs`; that would
double-count alignment's contribution (once implicitly, through the
baseline model's own prediction if it uses alignment features; once
explicitly, through this attribution). Same pattern as `mlb_luck_score.
scoring.weather_attribution`, which this module mirrors structurally.

Uses the SAME fixed run-value formula as `mlb_luck_score.scoring.
contact_luck.compute_expected_run_value` (`mlb_luck_score.scoring.
run_values.DEFAULT_RUN_VALUE_MAP`), via `mlb_luck_score.scoring.
weather_attribution.compute_expected_run_value_vectorized` (already generic
despite that module's name) -- not a redefinition (see CLAUDE.md "Never
silently redefine the Luck Score").
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized


class PositioningAttributionError(ValueError):
    """Raised when inputs to the positioning-attribution calculation are invalid."""


def compute_positioning_attribution(
    actual_proba_df: pd.DataFrame,
    typical_proba_df: pd.DataFrame,
    *,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
) -> pd.DataFrame:
    """Per-play alignment-positioning run-value attribution.

    Args:
        actual_proba_df: Predicted probabilities (columns in
            `mlb_luck_score.config.CLASS_ORDER`) using each play's ACTUAL
            starting alignment features -- every other (non-alignment)
            feature identical to `typical_proba_df`'s inputs.
        typical_proba_df: Predicted probabilities using the Version 0.6
            typical alignment (see `mlb_luck_score.features.
            build_contact_features.generate_typical_alignment_rows`) for the
            SAME rows, in the SAME order.
        run_value_map: Fixed run-value table. Defaults to
            `mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`.

    Returns:
        A DataFrame (same index as the inputs) with
        `expected_run_value_actual_alignment`,
        `expected_run_value_typical_alignment`, and `positioning_effect`
        columns.

    Raises:
        PositioningAttributionError: If the two inputs have different
            indices/lengths (a mismatched pair cannot be meaningfully
            compared).
    """
    if len(actual_proba_df) != len(typical_proba_df) or not actual_proba_df.index.equals(
        typical_proba_df.index
    ):
        raise PositioningAttributionError(
            "actual_proba_df and typical_proba_df must have the identical index/row order -- "
            "they must be predictions for the SAME rows, differing only in alignment features."
        )

    expected_actual = compute_expected_run_value_vectorized(actual_proba_df, run_value_map)
    expected_typical = compute_expected_run_value_vectorized(typical_proba_df, run_value_map)

    return pd.DataFrame(
        {
            "expected_run_value_actual_alignment": expected_actual,
            "expected_run_value_typical_alignment": expected_typical,
            "positioning_effect": expected_actual - expected_typical,
        },
        index=actual_proba_df.index,
    )
