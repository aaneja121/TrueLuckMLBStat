"""Contact Luck v1.3.1: validator/view-model for the "Try It Yourself"
counterfactual EV/launch-angle grid on the `/demo/` page.

Reads ONLY the committed `dashboard/demo_counterfactual_grid.json` --
produced once, offline, by `demo/build_counterfactual_grid.py` (which
imports `mlb_luck_score` freely; this module must never do that -- see
`tests/test_dashboard_isolation.py`). This module validates structure
(every grid cell is a legitimate probability distribution, the axis arrays
are sorted and match the declared grid shape, the recorded original
coordinate is actually present in the axis arrays at the recorded index)
and builds a small view-model for the template; the browser fetches the
RAW committed JSON file directly (copied byte-for-byte into `dist/demo/` by
`build.py`) rather than anything re-serialized here, so there is only ever
one copy of the (large) grid data on the wire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CLASS_ORDER",
    "CounterfactualContentError",
    "CounterfactualGridData",
    "CounterfactualReferencePlay",
    "load_counterfactual_grid_data",
]

#: Duplicated from `mlb_luck_score.config.CLASS_ORDER`, not imported -- see
#: this module's docstring on the read-only isolation boundary.
CLASS_ORDER: tuple[str, ...] = ("out", "single", "double", "triple", "home_run")

REQUIRED_EXAMPLE_IDS: tuple[str, ...] = ("hard_contact_out", "weak_contact_single")
REQUIRED_VARIED_FEATURES: tuple[str, ...] = ("launch_speed", "launch_angle")
REQUIRED_FIXED_FEATURE_NAMES: tuple[str, ...] = (
    "hit_distance_sc",
    "spray_angle_approx",
    "bb_type",
    "stand",
)

_PROBABILITY_SUM_TOLERANCE = 1e-3


class CounterfactualContentError(Exception):
    """Raised when the committed counterfactual grid is missing, malformed, or
    internally inconsistent -- never a reason to fabricate a placeholder grid.
    """


@dataclass(frozen=True)
class CounterfactualReferencePlay:
    example_id: str
    fixed_context: dict[str, Any]
    original_exit_velocity_mph: float
    original_launch_angle_deg: float
    original_outcome_class: str
    original_grid_index: dict[str, int]
    exit_velocity_values: list[float]
    launch_angle_values: list[float]
    grid_shape: dict[str, int]


@dataclass(frozen=True)
class CounterfactualGridData:
    reference_plays: dict[str, CounterfactualReferencePlay]
    run_value_table: dict[str, float]
    raw_path: Path


def _validate_probability_row(cell: list[Any], where: str) -> None:
    if len(cell) != len(CLASS_ORDER):
        raise CounterfactualContentError(
            f"{where}: expected {len(CLASS_ORDER)} probabilities, got {len(cell)}"
        )
    for v in cell:
        if not isinstance(v, int | float) or v != v or v in (float("inf"), float("-inf")):
            raise CounterfactualContentError(f"{where}: non-finite probability {v!r}")
        if v < 0.0 or v > 1.0:
            raise CounterfactualContentError(f"{where}: probability out of [0, 1] ({v!r})")
    total = sum(cell)
    if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise CounterfactualContentError(f"{where}: probabilities sum to {total!r}, expected ~1.0")


def _build_reference_play(raw: dict[str, Any]) -> CounterfactualReferencePlay:
    example_id = raw.get("example_id", "<unknown>")

    fixed_context = raw.get("fixed_context", {})
    missing = [f for f in REQUIRED_FIXED_FEATURE_NAMES if f not in fixed_context]
    if missing:
        raise CounterfactualContentError(f"{example_id}: fixed_context missing {missing}")

    ev_values = raw["exit_velocity_values"]
    la_values = raw["launch_angle_values"]
    if ev_values != sorted(ev_values):
        raise CounterfactualContentError(f"{example_id}: exit_velocity_values not sorted")
    if la_values != sorted(la_values):
        raise CounterfactualContentError(f"{example_id}: launch_angle_values not sorted")

    grid_shape = raw["grid_shape"]
    n_ev, n_la = grid_shape["n_ev"], grid_shape["n_la"]
    if len(ev_values) != n_ev or len(la_values) != n_la:
        raise CounterfactualContentError(
            f"{example_id}: grid_shape {grid_shape} does not match axis array lengths "
            f"(len(ev)={len(ev_values)}, len(la)={len(la_values)})"
        )

    grid = raw["grid"]
    if len(grid) != n_ev * n_la:
        raise CounterfactualContentError(
            f"{example_id}: grid length {len(grid)} != n_ev({n_ev}) * n_la({n_la})"
        )
    for i, cell in enumerate(grid):
        _validate_probability_row(cell, f"{example_id}[row {i}]")

    original_grid_index = raw["original_grid_index"]
    ev_index, la_index = original_grid_index["ev_index"], original_grid_index["la_index"]
    if not (0 <= ev_index < n_ev):
        raise CounterfactualContentError(
            f"{example_id}: ev_index {ev_index} out of range [0, {n_ev})"
        )
    if not (0 <= la_index < n_la):
        raise CounterfactualContentError(
            f"{example_id}: la_index {la_index} out of range [0, {n_la})"
        )

    real_ev = raw["original_exit_velocity_mph"]
    real_la = raw["original_launch_angle_deg"]
    if ev_values[ev_index] != real_ev:
        raise CounterfactualContentError(
            f"{example_id}: exit_velocity_values[{ev_index}]={ev_values[ev_index]!r} != "
            f"original_exit_velocity_mph={real_ev!r}"
        )
    if la_values[la_index] != real_la:
        raise CounterfactualContentError(
            f"{example_id}: launch_angle_values[{la_index}]={la_values[la_index]!r} != "
            f"original_launch_angle_deg={real_la!r}"
        )

    return CounterfactualReferencePlay(
        example_id=example_id,
        fixed_context=fixed_context,
        original_exit_velocity_mph=float(real_ev),
        original_launch_angle_deg=float(real_la),
        original_outcome_class=raw["original_outcome_class"],
        original_grid_index=original_grid_index,
        exit_velocity_values=ev_values,
        launch_angle_values=la_values,
        grid_shape=grid_shape,
    )


def load_counterfactual_grid_data(path: Path) -> CounterfactualGridData:
    """Load and validate the committed counterfactual grid at `path`.

    Raises:
        CounterfactualContentError: if the file is missing/malformed,
            doesn't contain exactly the two required reference plays, has
            an invalid probability distribution anywhere in either grid, or
            its `original_grid_index` doesn't actually point at the
            recorded real coordinate within its own axis arrays.
    """
    if not path.exists():
        raise CounterfactualContentError(
            f"Counterfactual grid not found at {path}. It is committed reference data, "
            "produced by `demo/build_counterfactual_grid.py` -- it is never regenerated by "
            "the dashboard build."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CounterfactualContentError(
            f"failed to parse counterfactual grid {path}: {exc}"
        ) from exc

    semantics = raw.get("counterfactual_semantics", {})
    if tuple(semantics.get("varied_features", [])) != REQUIRED_VARIED_FEATURES:
        raise CounterfactualContentError(
            f"counterfactual_semantics.varied_features {semantics.get('varied_features')!r} != "
            f"{REQUIRED_VARIED_FEATURES!r}"
        )
    fixed_features = set(semantics.get("fixed_features", []))
    missing_fixed = set(REQUIRED_FIXED_FEATURE_NAMES) - fixed_features
    if missing_fixed:
        raise CounterfactualContentError(
            f"counterfactual_semantics.fixed_features is missing {sorted(missing_fixed)}"
        )

    raw_plays = raw.get("reference_plays", {})
    if tuple(sorted(raw_plays)) != tuple(sorted(REQUIRED_EXAMPLE_IDS)):
        raise CounterfactualContentError(
            f"Counterfactual grid at {path} has example_ids {sorted(raw_plays)}, expected "
            f"exactly {sorted(REQUIRED_EXAMPLE_IDS)}"
        )

    reference_plays = {
        example_id: _build_reference_play(rp) for example_id, rp in raw_plays.items()
    }

    run_value_table = raw.get("model_configuration", {}).get("run_value_table", {})
    if set(run_value_table) != set(CLASS_ORDER):
        raise CounterfactualContentError(
            f"model_configuration.run_value_table keys {set(run_value_table)} != {set(CLASS_ORDER)}"
        )

    return CounterfactualGridData(
        reference_plays=reference_plays,
        run_value_table=run_value_table,
        raw_path=path,
    )
