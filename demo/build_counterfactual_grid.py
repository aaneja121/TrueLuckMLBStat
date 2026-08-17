"""Contact Luck v1.3.1: builds the committed `/demo/` "Try It Yourself"
counterfactual grid.

Like `demo/build_demo_fixture.py` (see that module's docstring for the full
read-only-boundary rationale -- the same applies here verbatim), this is a
ONE-TIME (or occasionally-rerun) offline generation step, run manually by a
maintainer -- never by `dashboard/build.py`, never by CI, and never as part
of `make check`.

CETERIS-PARIBUS MODEL SENSITIVITY, NOT A PHYSICS SIMULATOR
------------------------------------------------------------------
This module deliberately does NOT attempt to reconstruct a physically
consistent batted ball for each counterfactual (launch_speed, launch_angle)
pair. An earlier design considered recomputing `hit_distance_sc` from a
calibrated vacuum-projectile range formula as EV/LA changed; that was
rejected (see the task that produced this module) because `hit_distance_sc`
is an INDEPENDENT raw Statcast input to the frozen `baseline_v02` model --
never a function of `launch_speed`/`launch_angle` anywhere in this codebase
(unlike, say, `estimated_hang_time_s`, which genuinely IS derived from them
elsewhere -- see `mlb_luck_score.data.outfield_physics`). Introducing a new
auxiliary calibrated-physics relationship solely for this simulator would
make its behavior depend on a model Contact Luck itself never used.

Instead, every counterfactual grid point holds ALL active `baseline_v02`
features fixed at the selected reference play's real recorded values EXCEPT
`launch_speed` and `launch_angle`, which are varied directly. See
`GRID_VARIED_FEATURES` / `GRID_FIXED_FEATURE_NAMES` below and each
generated reference play's own `counterfactual_semantics` block. This is
intentionally a "how does the frozen model's expected-outcome distribution
respond to a changed launch_speed/launch_angle, holding every other model
input fixed" question -- NOT a claim that the resulting feature vector
describes a reconstructed physical trajectory. `dashboard/templates/`
copy must reflect this distinction; see the task's UI-copy guidance.

FROZEN MODEL, SAME AS v1.3.0
------------------------------------------------------------------
Trains the exact same frozen Version 0.2 `baseline_v02` model
(`train_contact_model.train_model`, unweighted probability baseline) on the
exact same `TRAIN_SEASONS` (2021-2023) as `demo/build_demo_fixture.py`, and
reuses the SAME two reference plays (identified by the same `event_id`s) --
this module never selects, retrains, or recalibrates anything.

EXPLICIT AXIS ARRAYS, NOT IMPLIED INTEGER RANGES
------------------------------------------------------------------
Real reference plays don't always have integer exit velocities (Greene:
107.6 mph, Lindor: 60.4 mph). Rather than either (a) expanding the whole
grid to 0.1 mph resolution just to land on these two values, or (b)
interpolating between neighboring integer grid points, each axis is an
EXPLICIT, sorted array (`build_axis_values`): every integer within that
play's empirically-supported bounds, PLUS the exact real value inserted if
it isn't already an integer. The grid stays a compact, direct-lookup-only
integer lattice everywhere else; only the one real coordinate is a genuine
non-integer grid point. `original_grid_index` records that point's exact
array position so "Reset to real play" is a real lookup, not a
reconstruction. The browser must index into `exit_velocity_values`/
`launch_angle_values` by POSITION -- never assume `values[i] == i + min`.

ONE FULL-PRECISION CELL PER REFERENCE PLAY
------------------------------------------------------------------
Every grid cell is rounded to `GRID_CELL_PROBABILITY_DECIMALS` EXCEPT the
one at `original_grid_index`, which is overwritten with the exact,
full-precision `original_probabilities` values. This means "Reset to real
play" needs no separate frontend special case and no second probability
source -- the SAME direct-lookup-by-index path every slider position uses
also happens to be exact at that one position, because that specific cell
IS the full-precision value, not a rounded approximation of it.

DETERMINISM AND FAIL-LOUD RECONCILIATION
------------------------------------------------------------------
The generator computes each reference play's probabilities/expected run
value AT ITS OWN REAL (launch_speed, launch_angle) COORDINATE directly
(independent of the grid -- see `original_probabilities`/
`original_expected_run_value` below) and asserts, before writing any
output, that these reconcile with the ALREADY-REVIEWED `dashboard/
demo_fixture.json` within tight numerical tolerance. If a future code
change (a different `DEFAULT_RUN_VALUE_MAP`, a different default
`class_weight`, different selected features, ...) would silently change
what this simulator shows, this assertion fails loudly instead of shipping
a quietly-inconsistent artifact. This artifact is committed, reviewed
reference data -- the SAME source data/configuration must produce
byte-identical JSON on every regeneration (no generated-at/timestamp
field; see `tests/test_counterfactual_grid.py`'s byte-determinism test).

Usage (requires the full 2021-2024 development dataset already cached
locally -- see README.md "Full development dataset" /
`make clean-development-data`; no network access):

    .venv/bin/python demo/build_counterfactual_grid.py

Writes `dashboard/demo_counterfactual_grid.json`, committed to git as
reviewed reference data, the same convention as `dashboard/demo_fixture.json`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import CLASS_ORDER, TRAIN_SEASONS, assert_seasons_allowed
from mlb_luck_score.models.train_contact_model import (
    RANDOM_SEED,
    VARIANT_UNWEIGHTED,
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.scoring.contact_luck import compute_expected_run_value
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

logger = logging.getLogger(__name__)

DEMO_COUNTERFACTUAL_GRID_VERSION = "1.3.1"
DEFAULT_INPUT_PATH = Path("data/processed/cleaned_development_data.parquet")
DEFAULT_OUTPUT_PATH = Path("dashboard/demo_counterfactual_grid.json")
#: The already-reviewed v1.3.0 fixture this generator's original-coordinate
#: output must reconcile with -- see module docstring.
DEMO_FIXTURE_PATH = Path("dashboard/demo_fixture.json")

_RECONCILIATION_TOLERANCE = 1e-9

#: Reuses the EXACT SAME reference-play identifiers as
#: `demo/build_demo_fixture.py` -- this module selects nothing new.
REFERENCE_PLAY_SPECS: tuple[dict[str, str], ...] = (
    {
        "example_id": "hard_contact_out",
        "event_id": "746378-62-5",
        "batter_name": "Riley Greene",
    },
    {
        "example_id": "weak_contact_single",
        "event_id": "746861-44-5",
        "batter_name": "Francisco Lindor",
    },
)

#: The two `baseline_v02` features this simulator varies.
GRID_VARIED_FEATURES: tuple[str, ...] = ("launch_speed", "launch_angle")

#: The `baseline_v02` features held fixed at the reference play's real
#: recorded values for EVERY grid point. `stand` and `spray_angle_approx`
#: are genuinely independent of launch_speed/launch_angle in this codebase
#: (see module docstring); `bb_type` is a raw Statcast field with no
#: launch_angle-derived formula anywhere in this codebase, so the slider
#: bounds below are chosen (per-bb_type empirical support) specifically so
#: `bb_type` never becomes an implausible label for the varied coordinate.
GRID_FIXED_FEATURE_NAMES: tuple[str, ...] = (
    "hit_distance_sc",
    "spray_angle_approx",
    "bb_type",
    "stand",
)

#: Empirical support percentiles used to bound each reference play's
#: sliders -- see module docstring / task discussion for why this (Option
#: A: constrain sliders to supported region) was chosen over a fabricated
#: confidence score.
SUPPORT_PERCENTILE_LOW = 0.01
SUPPORT_PERCENTILE_HIGH = 0.99

#: Decimal places kept for each `grid` cell's probabilities (display-only
#: values -- see `_build_reference_play`'s `grid` construction). Chosen to
#: keep the shipped artifact reasonably small while remaining far more
#: precise than anything the UI ever displays (a percentage to 1 decimal
#: place needs at most ~3-4 decimal places of underlying probability).
GRID_CELL_PROBABILITY_DECIMALS = 6


class CounterfactualGridBuildError(RuntimeError):
    """Raised when the counterfactual grid cannot be built as specified --
    never a reason to fall back to a fabricated or partial grid.
    """


def bb_type_support_bounds(
    train_df: pd.DataFrame, bb_type: str, column: str
) -> tuple[float, float]:
    values = train_df.loc[train_df["bb_type"] == bb_type, column].dropna()
    if values.empty:
        raise CounterfactualGridBuildError(
            f"No training-eligible rows found for bb_type={bb_type!r}, column={column!r} -- "
            "cannot compute empirical slider support."
        )
    return float(values.quantile(SUPPORT_PERCENTILE_LOW)), float(
        values.quantile(SUPPORT_PERCENTILE_HIGH)
    )


def compute_integer_slider_bounds(
    p_low: float, p_high: float, real_value: float
) -> tuple[int, int]:
    """`[min(p_low, real_value), max(p_high, real_value)]`, widened outward
    to integers -- guarantees the reference play's own real coordinate is
    always reachable (needed for "Reset to real play") while staying
    anchored to real empirical support, never an arbitrary range.
    """
    lo = math.floor(min(p_low, real_value))
    hi = math.ceil(max(p_high, real_value))
    return lo, hi


def build_axis_values(p_low: float, p_high: float, real_value: float) -> list[float]:
    """Explicit, sorted, deduplicated slider-axis values for one EV or LA
    axis: every integer in `compute_integer_slider_bounds(p_low, p_high,
    real_value)`, PLUS `real_value` itself if it isn't already one of those
    integers (e.g. Greene's real exit velocity, 107.6, is never an integer
    -- see module docstring's "explicit allowed-value arrays" design).

    This -- not a uniform 0.1-step grid, and not interpolation -- is how
    "Reset to real play" and exact reconciliation with `demo_fixture.json`
    stay possible while keeping the grid a compact, direct-lookup-only
    integer lattice everywhere else. The browser indexes into this array by
    POSITION, never by assuming `array[i] == i + min`.

    Returns:
        A sorted list mixing `int` (regular integer steps) and, at most,
        one `float` (the inserted real value, only when it isn't already
        integer-valued) -- matches the task's own
        `[65, 66, ..., 107, 107.6, 108, ..., 111]` illustration.
    """
    lo, hi = compute_integer_slider_bounds(p_low, p_high, real_value)
    values: list[float] = list(range(lo, hi + 1))
    if not any(math.isclose(v, real_value, abs_tol=1e-9) for v in values):
        values.append(real_value)
        values.sort()
    return values


def find_axis_index(values: list[float], target: float) -> int:
    """The exact index of `target` within `values` (built by
    `build_axis_values`), matched by close-enough float equality rather
    than exact `==` (guards against any residual floating-point noise
    between how `target` and `values` were each computed, even though both
    ultimately trace back to the same source row's `launch_speed`/
    `launch_angle`).
    """
    for i, v in enumerate(values):
        if math.isclose(v, target, abs_tol=1e-9):
            return i
    raise CounterfactualGridBuildError(f"{target!r} not found in axis values {values!r}")


def build_grid_dataframe(
    fixed_context: dict[str, Any],
    ev_values: list[float],
    la_values: list[float],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build the counterfactual feature rows for one reference play's full
    grid: `launch_speed`/`launch_angle` vary per `ev_values`/`la_values`
    (row-major: `launch_speed` outer, `launch_angle` inner -- matching the
    flat `grid` array's index convention `ev_index * len(la_values) +
    la_index`), every other feature in `feature_cols` is copied verbatim
    from `fixed_context` on EVERY row. This is the ONLY place a
    counterfactual row is assembled -- kept separate from model
    training/scoring so it stays directly unit-testable (see
    `tests/test_counterfactual_grid.py`'s fixed-feature-invariant checks)
    without needing real data or a trained model.
    """
    records = []
    for ev in ev_values:
        for la in la_values:
            record = dict(fixed_context)
            record["launch_speed"] = float(ev)
            record["launch_angle"] = float(la)
            records.append(record)
    return pd.DataFrame(records)[feature_cols]


def _score_single(
    trained: TrainedModel, feature_cols: list[str], record: dict[str, Any]
) -> dict[str, float]:
    x = pd.DataFrame([record])[feature_cols]
    proba = predict_proba_ordered(trained, x)
    validate_probabilities(proba)
    row = proba.iloc[0]
    return {cls: float(row[cls]) for cls in CLASS_ORDER}


def _build_reference_play(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    trained: TrainedModel,
    spec: dict[str, str],
    reviewed_fixture_by_example_id: dict[str, Any],
) -> dict[str, Any]:
    event_id = spec["event_id"]
    matches = df[df["event_id"] == event_id]
    if len(matches) != 1:
        raise CounterfactualGridBuildError(
            f"Expected exactly one row for event_id={event_id!r}, found {len(matches)}."
        )
    row = matches.iloc[0]

    des_text = str(row["des"])
    if not des_text.startswith(spec["batter_name"]):
        raise CounterfactualGridBuildError(
            f"event_id={event_id!r}: expected des text to start with batter_name="
            f"{spec['batter_name']!r}, got des={des_text!r}."
        )

    fixed_context: dict[str, Any] = {
        "hit_distance_sc": float(row["hit_distance_sc"]),
        "spray_angle_approx": float(row["spray_angle_approx"]),
        "bb_type": str(row["bb_type"]),
        "stand": str(row["stand"]),
    }
    real_ev = float(row["launch_speed"])
    real_la = float(row["launch_angle"])
    outcome_class = str(row["outcome_class"])

    feature_cols = trained.numeric_features + trained.categorical_features

    # Score the EXACT real coordinate directly, independent of the grid
    # (built further below) -- this is what must reconcile with the
    # already-reviewed demo_fixture.json. The grid is later verified to
    # reproduce this SAME value at its own exact array index (see
    # find_axis_index below), but computing it here first, standalone,
    # means that check has an independent ground truth to compare against.
    original_record = dict(fixed_context)
    original_record["launch_speed"] = real_ev
    original_record["launch_angle"] = real_la
    original_probabilities = _score_single(trained, feature_cols, original_record)
    original_expected_rv = compute_expected_run_value(
        original_probabilities, run_value_map=DEFAULT_RUN_VALUE_MAP
    )

    reviewed = reviewed_fixture_by_example_id.get(spec["example_id"])
    if reviewed is None:
        raise CounterfactualGridBuildError(
            f"example_id={spec['example_id']!r} not found in {DEMO_FIXTURE_PATH} -- "
            "cannot verify reconciliation with the reviewed v1.3.0 fixture."
        )
    reviewed_probs = reviewed["expectation"]["probabilities"]
    for cls in CLASS_ORDER:
        gap = abs(original_probabilities[cls] - reviewed_probs[cls])
        if gap > _RECONCILIATION_TOLERANCE:
            raise CounterfactualGridBuildError(
                f"example_id={spec['example_id']!r}: probability for {cls!r} "
                f"({original_probabilities[cls]!r}) does not reconcile with the reviewed "
                f"v1.3.0 fixture ({reviewed_probs[cls]!r}, gap={gap!r}). Refusing to write "
                "an inconsistent counterfactual grid."
            )
    reviewed_rv = reviewed["expectation"]["expected_run_value"]
    rv_gap = abs(original_expected_rv - reviewed_rv)
    if rv_gap > _RECONCILIATION_TOLERANCE:
        raise CounterfactualGridBuildError(
            f"example_id={spec['example_id']!r}: expected_run_value ({original_expected_rv!r}) "
            f"does not reconcile with the reviewed v1.3.0 fixture ({reviewed_rv!r}, "
            f"gap={rv_gap!r})."
        )

    ev_p1, ev_p99 = bb_type_support_bounds(train_df, fixed_context["bb_type"], "launch_speed")
    la_p1, la_p99 = bb_type_support_bounds(train_df, fixed_context["bb_type"], "launch_angle")
    ev_values = build_axis_values(ev_p1, ev_p99, real_ev)
    la_values = build_axis_values(la_p1, la_p99, real_la)

    grid_df = build_grid_dataframe(fixed_context, ev_values, la_values, feature_cols)
    proba_df = predict_proba_ordered(trained, grid_df)
    validate_probabilities(proba_df)
    # Rounded to GRID_CELL_PROBABILITY_DECIMALS -- these are the ORDINARY
    # exploratory cells, display-only, never reconciliation-tested. Keeps
    # the shipped artifact reasonably sized without sacrificing any
    # precision the UI could ever show (a probability displayed to 1
    # decimal place of a percentage needs far less than 6 decimal places
    # of underlying precision). The ONE exception -- the cell at the real
    # reference coordinate -- is overwritten below with full precision, so
    # a single direct-lookup path (no frontend special case, no second
    # probability source) exactly reconciles with demo_fixture.json.
    grid = [
        [round(float(r[c]), GRID_CELL_PROBABILITY_DECIMALS) for c in CLASS_ORDER]
        for _, r in proba_df.iterrows()
    ]

    # The real coordinate is a genuine point IN the grid (not just
    # separately-scored metadata) -- overwrite exactly that one cell with
    # the full-precision `original_probabilities` already computed above,
    # so "Reset to real play" (a plain grid lookup at original_grid_index,
    # the SAME lookup path every other slider position uses) reproduces
    # the reviewed v1.3.0 fixture exactly, not merely within a rounding
    # tolerance. Every other cell stays rounded.
    ev_index = find_axis_index(ev_values, real_ev)
    la_index = find_axis_index(la_values, real_la)
    grid_row_index = ev_index * len(la_values) + la_index
    grid[grid_row_index] = [original_probabilities[cls] for cls in CLASS_ORDER]

    return {
        "example_id": spec["example_id"],
        "fixed_context": fixed_context,
        "original_exit_velocity_mph": real_ev,
        "original_launch_angle_deg": real_la,
        "original_outcome_class": outcome_class,
        "original_probabilities": original_probabilities,
        "original_expected_run_value": original_expected_rv,
        "original_grid_index": {"ev_index": ev_index, "la_index": la_index},
        "exit_velocity_values": ev_values,
        "launch_angle_values": la_values,
        "grid_shape": {"n_ev": len(ev_values), "n_la": len(la_values)},
        "grid": grid,
    }


def model_configuration(trained: TrainedModel) -> dict[str, Any]:
    if trained.variant != VARIANT_UNWEIGHTED:
        raise CounterfactualGridBuildError(
            f"Refusing to build the counterfactual grid from a non-default model variant "
            f"{trained.variant!r} -- see CLAUDE.md 'Never use class_weight=\"balanced\"'."
        )
    return {
        "contact_model_variant": trained.variant,
        "class_weight": trained.class_weight,
        "train_seasons": list(TRAIN_SEASONS),
        "class_order": list(CLASS_ORDER),
        "numeric_features": list(trained.numeric_features),
        "categorical_features": list(trained.categorical_features),
        "random_seed": RANDOM_SEED,
        "run_value_table": dict(DEFAULT_RUN_VALUE_MAP),
        "run_value_source": (
            "FanGraphs Guts! constants (2021-2024 average) -- see "
            "mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP"
        ),
        "package_version": package_version,
        "scikit_learn_version": sklearn.__version__,
    }


def counterfactual_semantics(trained: TrainedModel) -> dict[str, Any]:
    """Adapted to the ACTUAL active feature set (`trained.numeric_features`
    + `trained.categorical_features`), not a hardcoded guess -- if a future
    code change alters which features `select_available_features` selects
    (e.g. `venue` regains enough coverage to be included), this block --
    and the fingerprint over it -- changes too, rather than silently
    describing a stale feature set.
    """
    active_features = list(trained.numeric_features) + list(trained.categorical_features)
    fixed_features = [f for f in active_features if f not in GRID_VARIED_FEATURES]
    missing = set(GRID_FIXED_FEATURE_NAMES) - set(fixed_features)
    if missing:
        raise CounterfactualGridBuildError(
            f"Expected fixed feature(s) {sorted(missing)} are not part of the model's "
            f"active feature set {active_features} -- this module's hardcoded "
            "GRID_FIXED_FEATURE_NAMES assumption is stale and must be revisited."
        )
    return {
        "type": "ceteris_paribus_model_sensitivity",
        "varied_features": list(GRID_VARIED_FEATURES),
        "fixed_features": fixed_features,
        "interpretation": (
            "Model sensitivity only; not a physical ball-flight reconstruction. Every "
            "grid point holds every active baseline_v02 feature EXCEPT launch_speed/"
            "launch_angle fixed at the selected reference play's real recorded values."
        ),
    }


def _config_fingerprint(model_configuration: dict[str, Any], semantics: dict[str, Any]) -> str:
    """SHA-256 over a canonical JSON encoding of BOTH `model_configuration`
    and `counterfactual_semantics` -- a future change to either the frozen
    model's configuration OR to which features this simulator varies/fixes
    changes this fingerprint, exactly like `demo/build_demo_fixture.py`'s
    fingerprint (see `tests/test_counterfactual_grid.py`, which recomputes
    this offline from the committed artifact's own recorded blocks and
    compares).
    """
    payload = {"model_configuration": model_configuration, "counterfactual_semantics": semantics}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_counterfactual_grid(
    input_path: Path = DEFAULT_INPUT_PATH,
    fixture_path: Path = DEMO_FIXTURE_PATH,
) -> dict[str, Any]:
    assert_seasons_allowed(TRAIN_SEASONS)

    if not input_path.exists():
        raise CounterfactualGridBuildError(
            f"{input_path} not found. This script requires the full 2021-2024 development "
            "dataset (see README.md 'Full development dataset' / `make clean-development-data`) "
            "-- it is never downloaded automatically."
        )
    if not fixture_path.exists():
        raise CounterfactualGridBuildError(
            f"{fixture_path} not found -- this generator must reconcile every reference "
            "play's original coordinate against the already-reviewed v1.3.0 fixture."
        )
    reviewed_fixture = json.loads(fixture_path.read_text())
    reviewed_fixture_by_example_id = {e["example_id"]: e for e in reviewed_fixture["examples"]}

    df = pd.read_parquet(input_path)
    train_df = df[df["eligible_for_training"].astype(bool) & df["season"].isin(TRAIN_SEASONS)]
    if train_df.empty:
        raise CounterfactualGridBuildError(f"No training-eligible rows found for {TRAIN_SEASONS}")

    logger.info(
        "Training the frozen baseline_v02 model on %d rows (seasons %s)",
        len(train_df),
        TRAIN_SEASONS,
    )
    trained = train_model(train_df)

    model_config = model_configuration(trained)
    semantics = counterfactual_semantics(trained)

    reference_plays = {
        spec["example_id"]: _build_reference_play(
            df, train_df, trained, spec, reviewed_fixture_by_example_id
        )
        for spec in REFERENCE_PLAY_SPECS
    }

    return {
        "demo_counterfactual_grid_version": DEMO_COUNTERFACTUAL_GRID_VERSION,
        # Deliberately no generated_at/current-time field: this is a
        # committed, reviewed artifact -- the same source data/config must
        # produce byte-identical output on every regeneration. Git history
        # already records when the reviewed file last changed; a wall
        # -clock timestamp here would only ever be noise in that diff. See
        # tests/test_counterfactual_grid.py's byte-determinism test.
        "generator_script": "demo/build_counterfactual_grid.py",
        "model_configuration": model_config,
        "counterfactual_semantics": semantics,
        "generator_config_fingerprint": _config_fingerprint(model_config, semantics),
        "reference_plays": reference_plays,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--fixture", type=Path, default=DEMO_FIXTURE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    artifact = build_counterfactual_grid(args.input, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")

    total_points = sum(
        rp["grid_shape"]["n_ev"] * rp["grid_shape"]["n_la"]
        for rp in artifact["reference_plays"].values()
    )
    size_bytes = args.output.stat().st_size
    logger.info(
        "Wrote counterfactual grid to %s (%d bytes, %d total grid points)",
        args.output,
        size_bytes,
        total_points,
    )
    for example_id, rp in artifact["reference_plays"].items():
        ev_values = rp["exit_velocity_values"]
        la_values = rp["launch_angle_values"]
        logger.info(
            "  %s: EV [%s..%s] (%d values) x LA [%s..%s] (%d values) = %d points "
            "(real coordinate at index ev=%d, la=%d)",
            example_id,
            ev_values[0],
            ev_values[-1],
            len(ev_values),
            la_values[0],
            la_values[-1],
            len(la_values),
            rp["grid_shape"]["n_ev"] * rp["grid_shape"]["n_la"],
            rp["original_grid_index"]["ev_index"],
            rp["original_grid_index"]["la_index"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
