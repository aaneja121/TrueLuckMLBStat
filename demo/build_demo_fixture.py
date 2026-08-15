"""Contact Luck v1.3.0: builds the committed `/demo/` page fixture.

This is a ONE-TIME (or occasionally-rerun) offline generation step, run
manually by a maintainer -- never by `dashboard/build.py`, never by CI, and
never as part of `make check`. It is the ONLY place in the repository that
scores the two illustrative plays shown on the "Why Contact Luck Exists"
demo page.

Why this has to be a separate top-level module (like `evaluation/` and
`prospective/`) rather than living under `dashboard/`: `dashboard/` is a
read-only presentation layer that is structurally forbidden from importing
`mlb_luck_score` at all (see `tests/test_dashboard_isolation.py`). This
script imports it freely -- it runs the real frozen Version 0.2 contact
model (`mlb_luck_score.models.train_contact_model.train_model`, the
unweighted probability baseline -- see CLAUDE.md "Never use
class_weight='balanced'...") on the real, already-downloaded 2021-2024
development dataset, then scores two specific, hardcoded real plays with the
frozen `mlb_luck_score.scoring.contact_luck.compute_raw_contact_luck_runs`
formula and the frozen `mlb_luck_score.scoring.run_values.
DEFAULT_RUN_VALUE_MAP`. Nothing here retrains, recalibrates, or reselects
any model, feature, or threshold -- it only asks the ALREADY-FROZEN Version
0.2 pipeline what it says about two real, already-recorded 2024 plays.

Usage (requires the full 2021-2024 development dataset -- see README.md
"Full development dataset" / `make clean-development-data`; no network
access, this reads the already-cached local parquet):

    .venv/bin/python demo/build_demo_fixture.py

Writes `dashboard/demo_fixture.json`, which IS committed to git as reviewed
reference data (the same convention as `mlb_luck_score.data.
game_metadata_overrides` / `mlb_luck_score.data.park_geometry`: small,
hand-reviewed, cited data written as a build artifact of real model code,
not a downloaded dataset). `dashboard/build.py` and every other module under
`dashboard/` only ever READ that committed file -- see `dashboard/
demo_content.py` -- never regenerate it.

Determinism and auditability: the two example plays are identified by their
exact Statcast `event_id` (game_pk-at_bat_number-pitch_number), not
re-selected by any search/ranking at run time, so re-running this script
against the same cached dataset reproduces byte-identical probabilities.
`model_configuration` in the output records every frozen input the
scoring depended on (train seasons, class order, selected features, run
-value table, random seed, package/scikit-learn versions), and
`generator_config_fingerprint` is a SHA-256 hash over that block -- see
`tests/test_demo_fixture.py`, which recomputes both the fingerprint and each
example's probabilities/expected-run-value/Contact-Luck value from the
CURRENT live `mlb_luck_score` constants (offline, using only the committed
fixture -- no real data needed for the test) and fails if either has drifted
from what this script actually produced. That test is the mechanism that
makes a future silent change (e.g. to `DEFAULT_RUN_VALUE_MAP`, `CLASS_ORDER`,
or the default `class_weight`) visible rather than leaving the demo showing
stale, no-longer-reproducible numbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import CLASS_ORDER, TRAIN_SEASONS, assert_seasons_allowed
from mlb_luck_score.models.train_contact_model import (
    RANDOM_SEED,
    VARIANT_UNWEIGHTED,
    predict_proba_ordered,
    train_model,
)
from mlb_luck_score.scoring.contact_luck import (
    compute_expected_run_value,
    compute_raw_contact_luck_runs,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

logger = logging.getLogger(__name__)

DEMO_FIXTURE_VERSION = "1.3.0"
DEFAULT_INPUT_PATH = Path("data/processed/cleaned_development_data.parquet")
DEFAULT_OUTPUT_PATH = Path("dashboard/demo_fixture.json")

_OUTCOME_LABELS = {
    "out": "Out",
    "single": "Single",
    "double": "Double",
    "triple": "Triple",
    "home_run": "Home run",
}

#: The two hardcoded, already-real, already-recorded plays this fixture
#: illustrates. Selected once (2024 validation-season data, filtered to
#: clean single-clause `des` text with no additional baserunner narrative,
#: then the single most extreme real `raw_contact_luck_runs` value in each
#: category) -- NOT re-searched at build time, so a rerun of this script is
#: reproducible from these fixed identifiers alone. This selection choses
#: which REAL, already-computed value to display; it does not tune, retrain,
#: or reselect anything about the model itself -- see CLAUDE.md's
#: distinction between illustration and tuning.
EXAMPLE_SPECS: tuple[dict[str, str], ...] = (
    {
        "example_id": "hard_contact_out",
        "narrative_label": "Hard-hit line drive that became an out",
        "contact_description": "Hard line drive",
        "event_id": "746378-62-5",
        # Verified against the row's own `des` text at build time (see
        # `_build_example`'s startswith assertion) -- NOT read from the
        # dataset's `player_name` column, which (a well-known Statcast/
        # pybaseball quirk) names the PITCHER on record for that pitch, not
        # the batter.
        "batter_name": "Riley Greene",
    },
    {
        "example_id": "weak_contact_single",
        "narrative_label": "Weak bloop that became a single",
        "contact_description": "Weak bloop",
        "event_id": "746861-44-5",
        "batter_name": "Francisco Lindor",
    },
)


class DemoFixtureBuildError(RuntimeError):
    """Raised when the demo fixture cannot be built as specified -- never a
    reason to fall back to a fabricated or partial example.
    """


def _team_and_opponent(row: pd.Series) -> tuple[str, str]:
    """The batter's own team and their opponent, from `home_team`/`away_team`/`inning_topbot`."""
    if row["inning_topbot"] == "Top":
        return str(row["away_team"]), str(row["home_team"])
    return str(row["home_team"]), str(row["away_team"])


def _build_example(df: pd.DataFrame, trained: Any, spec: dict[str, str]) -> dict[str, Any]:
    event_id = spec["event_id"]
    matches = df[df["event_id"] == event_id]
    if len(matches) != 1:
        raise DemoFixtureBuildError(
            f"Expected exactly one row for event_id={event_id!r}, found {len(matches)}. "
            "The demo fixture's hardcoded example identifiers must be re-verified against "
            "the current development dataset before proceeding."
        )
    row = matches.iloc[0]

    batter_name = spec["batter_name"]
    des_text = str(row["des"])
    if not des_text.startswith(batter_name):
        raise DemoFixtureBuildError(
            f"event_id={event_id!r}: expected des text to start with batter_name="
            f"{batter_name!r}, got des={des_text!r}. Re-verify this example's identifiers."
        )

    feature_cols = trained.numeric_features + trained.categorical_features
    proba_row = predict_proba_ordered(trained, pd.DataFrame([row[feature_cols]])).iloc[0]
    probabilities = {cls: float(proba_row[cls]) for cls in CLASS_ORDER}

    outcome_class = str(row["outcome_class"])
    if outcome_class not in DEFAULT_RUN_VALUE_MAP:
        raise DemoFixtureBuildError(
            f"event_id={event_id!r} has unrecognized outcome_class={outcome_class!r}"
        )

    expected_run_value = compute_expected_run_value(
        probabilities, run_value_map=DEFAULT_RUN_VALUE_MAP
    )
    observed_run_value = DEFAULT_RUN_VALUE_MAP[outcome_class]
    contact_luck_runs = compute_raw_contact_luck_runs(
        probabilities, outcome_class, run_value_map=DEFAULT_RUN_VALUE_MAP
    )
    reconciliation_gap = abs((observed_run_value - expected_run_value) - contact_luck_runs)
    if reconciliation_gap > 1e-9:
        raise DemoFixtureBuildError(
            f"event_id={event_id!r}: observed_run_value - expected_run_value does not "
            f"reconcile with contact_luck_runs (gap={reconciliation_gap!r})"
        )

    batter_team, opponent_team = _team_and_opponent(row)

    return {
        "example_id": spec["example_id"],
        "narrative_label": spec["narrative_label"],
        "contact_description": spec["contact_description"],
        "provenance": {
            "event_id": event_id,
            "game_pk": int(row["game_pk"]),
            "game_date": str(row["game_date"]),
            "season": int(row["season"]),
            "batter_id": int(row["batter"]),
            "batter_name": batter_name,
            "batter_team": batter_team,
            "opponent_team": opponent_team,
            "batter_stand": str(row["stand"]),
            "pitcher_name": str(row["player_name"]) if pd.notna(row["player_name"]) else None,
            "description": des_text,
        },
        "contact": {
            "exit_velocity_mph": float(row["launch_speed"]),
            "launch_angle_deg": float(row["launch_angle"]),
            "spray_angle_deg": float(row["spray_angle_approx"]),
            "bb_type": str(row["bb_type"]),
            "hit_distance_ft": float(row["hit_distance_sc"]),
        },
        "expectation": {
            "probabilities": probabilities,
            "expected_run_value": expected_run_value,
        },
        "reality": {
            "outcome_class": outcome_class,
            "outcome_label": _OUTCOME_LABELS[outcome_class],
            "observed_run_value": observed_run_value,
        },
        "contact_luck_runs": contact_luck_runs,
    }


def _model_configuration(trained: Any) -> dict[str, Any]:
    if trained.variant != VARIANT_UNWEIGHTED:
        raise DemoFixtureBuildError(
            f"Refusing to build the demo fixture from a non-default model variant "
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


def _config_fingerprint(model_configuration: dict[str, Any]) -> str:
    """SHA-256 over a canonical JSON encoding of `model_configuration`.

    Deterministic across runs (fixed key order via `sort_keys=True`) and
    reproducible offline from the fixture's own `model_configuration` block
    -- `tests/test_demo_fixture.py` recomputes this from the CURRENT live
    `mlb_luck_score` constants and compares, so any future drift (a changed
    run-value table, a changed default class_weight, a changed train-season
    list) is caught as a test failure rather than silently changing what the
    demo page implies without anyone noticing.
    """
    canonical = json.dumps(model_configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_demo_fixture(input_path: Path = DEFAULT_INPUT_PATH) -> dict[str, Any]:
    assert_seasons_allowed(TRAIN_SEASONS)

    if not input_path.exists():
        raise DemoFixtureBuildError(
            f"{input_path} not found. This script requires the full 2021-2024 development "
            "dataset (see README.md 'Full development dataset' / `make clean-development-data`) "
            "-- it is never downloaded automatically."
        )

    df = pd.read_parquet(input_path)
    train_df = df[df["eligible_for_training"].astype(bool) & df["season"].isin(TRAIN_SEASONS)]
    if train_df.empty:
        raise DemoFixtureBuildError(f"No training-eligible rows found for seasons {TRAIN_SEASONS}")

    logger.info(
        "Training the frozen baseline_v02 model on %d rows (seasons %s)",
        len(train_df),
        TRAIN_SEASONS,
    )
    trained = train_model(train_df)

    model_configuration = _model_configuration(trained)
    examples = [_build_example(df, trained, spec) for spec in EXAMPLE_SPECS]

    return {
        "demo_fixture_version": DEMO_FIXTURE_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator_script": "demo/build_demo_fixture.py",
        "model_configuration": model_configuration,
        "generator_config_fingerprint": _config_fingerprint(model_configuration),
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    fixture = build_demo_fixture(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote demo fixture to %s", args.output)
    for example in fixture["examples"]:
        logger.info(
            "  %s: %s -- contact_luck_runs=%+.4f",
            example["example_id"],
            example["provenance"]["batter_name"],
            example["contact_luck_runs"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
