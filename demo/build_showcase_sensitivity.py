"""Contact Luck v1.4.1: Showcase Plays "What if?" sensitivity artifact
generator.

    real 2026 play_ledger.parquet row (a showcase play)
            v
    THIS MODULE: retrain the frozen contact model, score the play's OWN
    real coordinate, gate on reconciliation, build an EV/LA grid
            v
    showcase-sensitivity/<play_id>.json

REUSES `demo/build_counterfactual_grid.py`'s EXACT grid-construction
machinery (`build_axis_values`, `find_axis_index`, `build_grid_dataframe`,
`compute_integer_slider_bounds`, `bb_type_support_bounds`,
`model_configuration`, `counterfactual_semantics`) -- this module never
reimplements a second grid-construction algorithm or a second simulator.
The only genuinely new logic here is (1) sourcing a reference play from a
REAL 2026 canonical `play_ledger.parquet` row instead of the committed
2021-2024 dev-data reference plays, and (2) reconciling the freshly
-retrained model's own-coordinate score against THAT row's already-recorded
`p_out`..`p_home_run`/`expected_run_value` instead of against `dashboard/
demo_fixture.json`.

## The mandatory reconciliation gate (never weakened)

For any candidate play, `build_sensitivity_artifact` computes the frozen
model's own probabilities/expected run value AT THE PLAY'S REAL (launch_
speed, launch_angle) coordinate and requires it to reproduce that play's
CANONICAL `p_out`..`p_home_run`/`expected_run_value` within
`RECONCILIATION_TOLERANCE`. If it does not, `ShowcaseSensitivityReconciliationError`
is raised -- there is no fallback, no partial artifact, no silently-widened
tolerance. A caller must never catch this and ship a downgraded artifact in
its place; see `scripts/generate_production_explorer_artifacts.py`'s own
docstring for how a real production run responds to this (aborts sensitivity
generation for the whole run, never ships a partially-reconciled set).

## Why retraining here is safe -- NOT "a second potentially divergent model"

This codebase never persists a trained model object between runs (see
`mlb_luck_score.models.train_contact_model` / CLAUDE.md's "What 'frozen
artifact' means here"): `prospective/prospective_scoring.py`'s own
canonical-ledger scoring ALREADY retrains `baseline_v02` fresh, every
snapshot run, from the exact same frozen dev-data file (`TRAIN_SEASONS`
2021-2023, `RANDOM_SEED=42`, `class_weight=None`) this module trains from
(`train_frozen_contact_model` mirrors that recipe exactly, and mirrors
`demo/build_counterfactual_grid.py`'s own identical recipe). Calling
`train_model()` a second time, in a separate process, with the IDENTICAL
training data/seed/code, is the SAME "frozen model" by this codebase's own
definition, not a divergent one -- and the reconciliation gate above is
what actually PROVES that on every real showcase play, rather than merely
asserting it. Empirically (real 2026-08-16 production data, six real
showcase plays spanning both favorable home runs and unfavorable outs):
max observed gap was ~1.2e-7, far inside `RECONCILIATION_TOLERANCE`.

## Interactive selection policy (the per-group cap)

`select_interactive_candidates` picks up to `MAX_INTERACTIVE_PER_GROUP`
(6) showcase rows per group (favorable/unfavorable), in EXISTING rank
order (never re-sorted -- the true top-12 ranking is untouched by this
module), filtered to `interactive_available` rows only (raw model-input
eligibility, `demo.build_play_explorer_fixture.is_showcase_interactive_eligible`).
A true season extreme that lacks required inputs, or that ranks below the
6th eligible play in its group, simply never gets a sensitivity grid --
see `demo/build_play_explorer_fixture.py`'s "Interactive eligibility"
docstring section for why this never removes it from the showcase ranking
itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_counterfactual_grid as ccg  # noqa: E402

from mlb_luck_score.config import CLASS_ORDER, TRAIN_SEASONS, assert_seasons_allowed  # noqa: E402
from mlb_luck_score.eligibility import compute_eligibility  # noqa: E402
from mlb_luck_score.models.train_contact_model import (  # noqa: E402
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.scoring.contact_luck import compute_expected_run_value  # noqa: E402
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP  # noqa: E402

__all__ = [
    "MAX_INTERACTIVE_PER_GROUP",
    "RECONCILIATION_TOLERANCE",
    "REQUIRED_SENSITIVITY_INPUT_COLUMNS",
    "ShowcaseSensitivityError",
    "ShowcaseSensitivityReconciliationError",
    "build_sensitivity_artifact",
    "generate_showcase_sensitivity_artifacts",
    "is_interactive_eligible",
    "select_interactive_candidates",
    "train_frozen_contact_model",
]

#: Generous relative to the ~1e-7 cross-process retraining noise floor
#: empirically observed on real 2026-08-16 data (see module docstring) --
#: tight enough that a genuine divergence (a code/config drift, a training
#: -data mismatch) is still caught, loose enough to absorb ordinary
#: cross-machine floating-point noise from sklearn's solver.
RECONCILIATION_TOLERANCE = 1e-4

#: At most this many interactive showcase plays PER GROUP (favorable/
#: unfavorable) -- Section 5's "at most 6 favorable + 6 unfavorable = 12
#: interactive plays maximum."
MAX_INTERACTIVE_PER_GROUP = 6

#: Mirrors `demo.build_play_explorer_fixture.SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS`
#: exactly (duplicated, not imported, to keep this module's own contract
#: self-contained and independently testable -- `tests/
#: test_showcase_sensitivity.py` asserts the two stay identical).
REQUIRED_SENSITIVITY_INPUT_COLUMNS: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "bb_type",
    "stand",
)


class ShowcaseSensitivityError(RuntimeError):
    """Raised for any failure building a showcase sensitivity artifact --
    never a reason to fabricate, approximate, or partially write a grid.
    """


class ShowcaseSensitivityReconciliationError(ShowcaseSensitivityError):
    """Raised when a freshly-retrained frozen contact model's OWN
    -COORDINATE score for a real showcase play does not reproduce that
    play's canonical `p_out`..`p_home_run`/`expected_run_value` within
    `RECONCILIATION_TOLERANCE` -- the mandatory gate. Never weakened, never
    caught-and-downgraded by a caller.
    """


def is_interactive_eligible(row: pd.Series) -> bool:
    return all(pd.notna(row[col]) for col in REQUIRED_SENSITIVITY_INPUT_COLUMNS)


def select_interactive_candidates(
    showcase_rows: list[dict[str, Any]], *, max_per_group: int = MAX_INTERACTIVE_PER_GROUP
) -> list[dict[str, Any]]:
    """Up to `max_per_group` rows per `group` from `showcase_rows`
    (`demo.build_play_explorer_fixture.build_showcase_rows`'s output),
    filtered to `interactive_available` (raw input eligibility) and taken
    in EXISTING rank order -- never re-sorted, so this always picks the
    highest-ranked eligible plays in each group, never an arbitrary subset.
    """
    selected: list[dict[str, Any]] = []
    for group in ("favorable", "unfavorable"):
        eligible = [
            row
            for row in showcase_rows
            if row["group"] == group and row["interactive_available"]
        ]
        eligible.sort(key=lambda r: r["rank"])
        selected.extend(eligible[:max_per_group])
    return selected


def train_frozen_contact_model(dev_data_path: Path) -> tuple[TrainedModel, pd.DataFrame]:
    """Retrains the EXACT SAME frozen `baseline_v02` contact model
    production scoring uses -- same `TRAIN_SEASONS`, same fixed random
    seed (`train_model`'s own `RANDOM_SEED`), same `class_weight=None` --
    from the SAME frozen dev-data file already required as a Version 1.1
    input (`scripts/ensure_frozen_inputs.py`). Never persists a model
    file; this IS what "frozen model" means throughout this codebase (see
    module docstring). Returns `(trained, train_df)` -- the training
    dataframe is also needed for `bb_type_support_bounds` (per-`bb_type`
    empirical slider support), exactly as `demo/build_counterfactual_grid.py`
    uses it.
    """
    assert_seasons_allowed(TRAIN_SEASONS)
    if not dev_data_path.exists():
        raise ShowcaseSensitivityError(
            f"{dev_data_path} not found -- this module requires the same frozen 2021-2024 "
            "development dataset the normal publish path already fetches via "
            "scripts/ensure_frozen_inputs.py."
        )
    full = pd.read_parquet(dev_data_path)
    train_raw = full[full["season"].isin(TRAIN_SEASONS)]
    train_elig = compute_eligibility(train_raw)
    train_df = train_elig[train_elig["eligible_for_training"].fillna(False)]
    if train_df.empty:
        raise ShowcaseSensitivityError(f"No training-eligible rows found for {TRAIN_SEASONS}")
    trained = train_model(train_df, class_weight=None)
    return trained, train_df


def build_sensitivity_artifact(
    trained: TrainedModel, train_df: pd.DataFrame, play_row: pd.Series
) -> dict[str, Any]:
    """Build ONE play's sensitivity artifact -- reuses `demo/
    build_counterfactual_grid.py`'s exact axis/grid machinery, and enforces
    the mandatory reconciliation gate against `play_row`'s own canonical
    `p_out`..`p_home_run`/`expected_run_value`.

    Raises:
        ShowcaseSensitivityError: if `play_row` is missing a required
            model input.
        ShowcaseSensitivityReconciliationError: if the gate fails.
    """
    if not is_interactive_eligible(play_row):
        raise ShowcaseSensitivityError(
            f"play_id={play_row['play_id']!r} is missing required model input(s) -- not "
            "interactive-eligible."
        )

    bb_type = str(play_row["bb_type"])
    fixed_context = {
        "hit_distance_sc": float(play_row["hit_distance_sc"]),
        "spray_angle_approx": float(play_row["spray_angle_approx"]),
        "bb_type": bb_type,
        "stand": str(play_row["stand"]),
    }
    real_ev = float(play_row["launch_speed"])
    real_la = float(play_row["launch_angle"])
    feature_cols = trained.numeric_features + trained.categorical_features

    original_record = dict(fixed_context)
    original_record["launch_speed"] = real_ev
    original_record["launch_angle"] = real_la
    x = pd.DataFrame([original_record])[feature_cols]
    proba = predict_proba_ordered(trained, x)
    validate_probabilities(proba)
    original_probabilities = {cls: float(proba.iloc[0][cls]) for cls in CLASS_ORDER}
    original_expected_rv = compute_expected_run_value(
        original_probabilities, run_value_map=DEFAULT_RUN_VALUE_MAP
    )

    canonical_probabilities = {cls: float(play_row[f"p_{cls}"]) for cls in CLASS_ORDER}
    canonical_expected_rv = float(play_row["expected_run_value"])

    for cls in CLASS_ORDER:
        gap = abs(original_probabilities[cls] - canonical_probabilities[cls])
        if gap > RECONCILIATION_TOLERANCE:
            raise ShowcaseSensitivityReconciliationError(
                f"play_id={play_row['play_id']!r}: p_{cls} ({original_probabilities[cls]!r}) "
                f"does not reconcile with the canonical play ledger "
                f"({canonical_probabilities[cls]!r}, gap={gap!r}) -- refusing to ship a "
                "sensitivity artifact for this play."
            )
    rv_gap = abs(original_expected_rv - canonical_expected_rv)
    if rv_gap > RECONCILIATION_TOLERANCE:
        raise ShowcaseSensitivityReconciliationError(
            f"play_id={play_row['play_id']!r}: expected_run_value ({original_expected_rv!r}) "
            f"does not reconcile with the canonical play ledger ({canonical_expected_rv!r}, "
            f"gap={rv_gap!r})."
        )

    ev_p1, ev_p99 = ccg.bb_type_support_bounds(train_df, bb_type, "launch_speed")
    la_p1, la_p99 = ccg.bb_type_support_bounds(train_df, bb_type, "launch_angle")
    ev_values = ccg.build_axis_values(ev_p1, ev_p99, real_ev)
    la_values = ccg.build_axis_values(la_p1, la_p99, real_la)

    grid_df = ccg.build_grid_dataframe(fixed_context, ev_values, la_values, list(feature_cols))
    proba_df = predict_proba_ordered(trained, grid_df)
    validate_probabilities(proba_df)
    grid = [
        [round(float(r[c]), ccg.GRID_CELL_PROBABILITY_DECIMALS) for c in CLASS_ORDER]
        for _, r in proba_df.iterrows()
    ]

    ev_index = ccg.find_axis_index(ev_values, real_ev)
    la_index = ccg.find_axis_index(la_values, real_la)
    grid_row_index = ev_index * len(la_values) + la_index
    grid[grid_row_index] = [original_probabilities[cls] for cls in CLASS_ORDER]

    return {
        "play_id": str(play_row["play_id"]),
        "model_configuration": ccg.model_configuration(trained),
        "counterfactual_semantics": ccg.counterfactual_semantics(trained),
        "fixed_context": fixed_context,
        "original_exit_velocity_mph": real_ev,
        "original_launch_angle_deg": real_la,
        "original_probabilities": original_probabilities,
        "original_expected_run_value": original_expected_rv,
        "original_grid_index": {"ev_index": ev_index, "la_index": la_index},
        "exit_velocity_values": ev_values,
        "launch_angle_values": la_values,
        "grid_shape": {"n_ev": len(ev_values), "n_la": len(la_values)},
        "grid": grid,
    }


def generate_showcase_sensitivity_artifacts(
    *,
    play_ledger: pd.DataFrame,
    showcase_rows: list[dict[str, Any]],
    dev_data_path: Path,
    output_dir: Path,
    max_per_group: int = MAX_INTERACTIVE_PER_GROUP,
) -> dict[str, Any]:
    """Select up to `max_per_group` interactive-eligible showcase rows per
    group, build each one's sensitivity artifact, and write
    `showcase-sensitivity/<play_id>.json` for each.

    Atomic per this module's fail-loud policy: if ANY selected candidate
    fails the reconciliation gate, this function raises
    `ShowcaseSensitivityReconciliationError` and writes NOTHING (no partial
    directory) -- see module docstring. The caller (`scripts/
    generate_production_explorer_artifacts.py`) does not catch this; a
    real reconciliation failure aborts that whole publish run rather than
    silently shipping a downgraded interactive layer.

    Returns:
        `{"interactive_play_ids": [...], "output_dir": str}` --
        `interactive_play_ids` is empty (and `output_dir` is never created)
        if no showcase row is interactive-eligible.
    """
    candidates = select_interactive_candidates(showcase_rows, max_per_group=max_per_group)
    if not candidates:
        return {"interactive_play_ids": [], "output_dir": str(output_dir)}

    trained, train_df = train_frozen_contact_model(dev_data_path)

    ledger_by_play_id = play_ledger.set_index("play_id", drop=False)
    artifacts: dict[str, dict[str, Any]] = {}
    for row in candidates:
        play_id = row["play_id"]
        play_row = ledger_by_play_id.loc[play_id]
        if isinstance(play_row, pd.DataFrame):
            raise ShowcaseSensitivityError(f"duplicate play_id={play_id!r} in play_ledger")
        artifacts[play_id] = build_sensitivity_artifact(trained, train_df, play_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for play_id, artifact in artifacts.items():
        (output_dir / f"{play_id}.json").write_text(
            json.dumps(artifact, sort_keys=True, separators=(",", ":"))
        )

    return {"interactive_play_ids": sorted(artifacts.keys()), "output_dir": str(output_dir)}
