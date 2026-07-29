"""Contact Luck v0.7C: gating logic for the outfield opportunity/execution pipeline.

Routes each `outfield_opportunity_eligible` row to exactly one of three
gates:

  - `GATE_OPEN_FIELD`: park geometry is reliable AND the play is NOT within
    20 ft of the wall (`near_wall_20ft=False`, see `mlb_luck_score.data.
    join_park_geometry`). Scored with `open_field_v07` (== the Version 0.7A
    `measured_contact_only_v07` model, unchanged) -- calibration was already
    verified acceptable here (see `mlb_luck_score.models.
    compare_opportunity_models` real-data results).
  - `GATE_NEAR_WALL`: park geometry is reliable AND the play IS within 20 ft
    of the wall. Scored with the Version 0.7C near-wall specialist
    (`mlb_luck_score.models.compare_near_wall_models`) -- this is exactly
    the subgroup Version 0.7A's calibration was materially worse for (ECE
    0.18-0.25 vs. an overall 0.018).
  - `GATE_CONTACT_ONLY_FALLBACK`: park geometry is UNRELIABLE for this row
    (missing entirely, flagged `geometry_uncertain`, or missing
    `wall_distance_in_spray_direction`) -- neither the open-field nor the
    near-wall model has trustworthy wall-proximity inputs for this play, so
    it falls back to contact-only scoring (Version 0.2's `residual_
    contact_luck_runs` only; NO opportunity-difficulty or defensive
    -execution score is reported -- see `mlb_luck_score.scoring.
    gated_outfield_report`).

This module ONLY assigns gates -- it does not train or apply any model. See
`mlb_luck_score.scoring.gated_outfield_report` for how the gate assignment
is used to route rows to the right model (or no opportunity model at all)
and to set the reporting availability status the task calls for:
"Outfield execution score available for calibrated open-field opportunities;
provisional or unavailable for wall-adjacent opportunities."
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.models.compare_geometry_aware import _bool_mask

GATE_OPEN_FIELD = "open_field"
GATE_NEAR_WALL = "near_wall"
GATE_CONTACT_ONLY_FALLBACK = "contact_only_fallback"

ALL_GATES: tuple[str, ...] = (GATE_OPEN_FIELD, GATE_NEAR_WALL, GATE_CONTACT_ONLY_FALLBACK)


def assign_outfield_opportunity_gate(df: pd.DataFrame) -> pd.Series:
    """Assign one of `ALL_GATES` to every row of `df`.

    Must be called AFTER `mlb_luck_score.data.join_park_geometry.
    join_park_geometry` and `mlb_luck_score.features.build_contact_features.
    add_geometry_interaction_features` (for the stringified `near_wall_20ft`/
    `geometry_uncertain` columns -- see `mlb_luck_score.models.
    compare_geometry_aware._bool_mask`'s docstring for why a naive
    `.astype(bool)` on those columns would be wrong). If geometry was never
    joined at all (no `has_park_geometry` column), every row falls back to
    `GATE_CONTACT_ONLY_FALLBACK` -- there is no wall-proximity information
    to gate on.

    Args:
        df: Outfield-opportunity-eligible rows (or any superset -- gating
            does not require `outfield_opportunity_eligible` to already be
            filtered, though callers typically apply it first).

    Returns:
        A `str`-valued Series (same index as `df`), one of `ALL_GATES` per
        row.
    """
    if "has_park_geometry" not in df.columns:
        return pd.Series(GATE_CONTACT_ONLY_FALLBACK, index=df.index)

    has_geometry = df["has_park_geometry"].astype(bool).to_numpy()

    geometry_uncertain = (
        _bool_mask(df["geometry_uncertain"])
        if "geometry_uncertain" in df.columns
        else pd.Series(False, index=df.index).to_numpy()
    )

    if "wall_distance_in_spray_direction" in df.columns:
        wall_distance_missing = df["wall_distance_in_spray_direction"].isna().to_numpy()
    else:
        wall_distance_missing = pd.Series(True, index=df.index).to_numpy()

    geometry_reliable = has_geometry & ~geometry_uncertain & ~wall_distance_missing

    near_wall = (
        _bool_mask(df["near_wall_20ft"])
        if "near_wall_20ft" in df.columns
        else pd.Series(False, index=df.index).to_numpy()
    )

    gate = pd.Series(GATE_OPEN_FIELD, index=df.index, dtype=object)
    gate.loc[~geometry_reliable] = GATE_CONTACT_ONLY_FALLBACK
    gate.loc[geometry_reliable & near_wall] = GATE_NEAR_WALL
    return gate
