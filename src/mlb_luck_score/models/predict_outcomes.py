"""Load a trained contact model artifact and predict outcome probabilities.

Usage:

    python -m mlb_luck_score.models.predict_outcomes \\
        --input data/processed/cleaned_batted_balls.parquet \\
        --artifact-dir artifacts --model-version 0.1.0 \\
        --output outputs/tables/predictions.parquet

    python -m mlb_luck_score.models.predict_outcomes --demo

`--demo` runs an end-to-end synthetic walkthrough (fake contact rows -> tiny
model -> prediction -> raw luck -> provisional public score -> confidence
report) with no real data or network access required. It exists to make the
full pipeline runnable and inspectable immediately after repository
bootstrap, and is exercised by `make demo`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from mlb_luck_score.config import CLASS_ORDER, DEFAULT_VALUE_MAP
from mlb_luck_score.models.train_contact_model import TrainedModel, predict_proba_ordered
from mlb_luck_score.models.train_contact_model import (
    validate_probabilities as _validate_probabilities,
)
from mlb_luck_score.scoring.confidence import compute_confidence
from mlb_luck_score.scoring.public_score import raw_luck_to_public_score
from mlb_luck_score.scoring.raw_luck import compute_expected_value, compute_raw_luck

logger = logging.getLogger(__name__)

validate_probabilities = _validate_probabilities


def load_model(artifact_dir: Path, model_version: str) -> TrainedModel:
    """Load a saved pipeline + its metadata back into a `TrainedModel`."""
    model_path = artifact_dir / f"contact_model_v{model_version}.joblib"
    metadata_path = artifact_dir / f"contact_model_v{model_version}_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifact/metadata not found for version {model_version} in {artifact_dir}. "
            "Run `make train` (or the train_contact_model module) first."
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    variant = metadata.get("variant", "unknown_legacy_artifact")
    if "class_balanced" in variant:
        logger.warning(
            "Loaded artifact variant '%s' is a labeled comparison model with known "
            "miscalibrated probabilities -- do not use it for the Contact Luck score.",
            variant,
        )
    return TrainedModel(
        pipeline=pipeline,
        numeric_features=metadata["numeric_features"],
        categorical_features=metadata["categorical_features"],
        class_order=metadata["class_order"],
        class_weight=metadata.get("class_weight"),
        variant=variant,
    )


def predict(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities for `df`, validated and column-ordered."""
    feature_cols = trained.numeric_features + trained.categorical_features
    proba_df = predict_proba_ordered(trained, df[feature_cols])
    validate_probabilities(proba_df)
    return proba_df


def _build_synthetic_demo_data() -> pd.DataFrame:
    """Small synthetic training set spanning all five outcome classes."""
    rng_rows = [
        # launch_speed, launch_angle, spray_angle_approx, hit_distance_sc, bb_type, stand, venue, outcome
        (105.0, 28.0, 2.0, 410.0, "fly_ball", "R", "Synthetic Park", "home_run"),
        (98.0, 22.0, -18.0, 320.0, "fly_ball", "R", "Synthetic Park", "double"),
        (92.0, 12.0, 10.0, 180.0, "line_drive", "L", "Synthetic Park", "single"),
        (70.0, 45.0, 0.0, 90.0, "popup", "R", "Synthetic Park", "out"),
        (88.0, -5.0, -30.0, 60.0, "ground_ball", "L", "Synthetic Park", "out"),
        (101.0, 18.0, 25.0, 340.0, "line_drive", "R", "Synthetic Park", "triple"),
        (75.0, 30.0, 5.0, 140.0, "fly_ball", "L", "Synthetic Park", "out"),
        (110.0, 26.0, -3.0, 420.0, "fly_ball", "R", "Synthetic Park", "home_run"),
        (95.0, 15.0, -22.0, 300.0, "line_drive", "L", "Synthetic Park", "double"),
        (90.0, 10.0, 3.0, 160.0, "line_drive", "R", "Synthetic Park", "single"),
        (65.0, -10.0, 12.0, 40.0, "ground_ball", "R", "Synthetic Park", "out"),
        (99.0, 20.0, 28.0, 330.0, "line_drive", "L", "Synthetic Park", "triple"),
    ]
    return pd.DataFrame(
        rng_rows,
        columns=[
            "launch_speed",
            "launch_angle",
            "spray_angle_approx",
            "hit_distance_sc",
            "bb_type",
            "stand",
            "venue",
            "outcome_class",
        ],
    )


def run_demo() -> None:
    """Run the full synthetic pipeline walkthrough and print a report."""
    from mlb_luck_score.models.train_contact_model import train_model

    logger.info("Building a tiny synthetic dataset (not real Statcast data)...")
    demo_df = _build_synthetic_demo_data()
    trained = train_model(demo_df)

    example = demo_df.iloc[[0]].copy()
    observed_outcome = str(example["outcome_class"].iloc[0])
    proba_df = predict(trained, example)
    probabilities: dict[str, float] = {cls: float(proba_df.iloc[0][cls]) for cls in CLASS_ORDER}

    raw_luck = compute_raw_luck(probabilities, observed_outcome, value_map=DEFAULT_VALUE_MAP)
    public_score = raw_luck_to_public_score(raw_luck)
    confidence = compute_confidence(example.iloc[0])

    print("\n=== Contact Luck Prototype v0.1 -- synthetic demo ===")
    print("This uses fabricated data and an untuned toy model. It demonstrates")
    print("plumbing only -- it is not a scientific result.\n")
    print(f"Observed outcome: {observed_outcome}")
    print("Predicted probability distribution:")
    for cls in CLASS_ORDER:
        print(f"  {cls:>9s}: {probabilities[cls]:.3f}")
    expected_value = compute_expected_value(probabilities, value_map=DEFAULT_VALUE_MAP)
    print(f"Expected ordinal value: {expected_value:.3f}")
    print(f"Actual ordinal value:   {DEFAULT_VALUE_MAP[observed_outcome]:.3f}")
    print(f"Preliminary raw luck:   {raw_luck:+.3f}  (research placeholder, additive only)")
    print(f"Preliminary public score: {public_score:+.1f} / 100 (provisional, non-additive)")
    print(f"Data-completeness label: {confidence.label}")
    print(
        "\nDISCLAIMER: This repository is a research prototype, not a validated "
        "public baseball statistic. See README.md and CLAUDE.md."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Cleaned batted-ball table")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--output", type=Path, help="Where to save predictions")
    parser.add_argument(
        "--demo", action="store_true", help="Run a self-contained synthetic demo (no data needed)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --demo is passed")

    trained = load_model(args.artifact_dir, args.model_version)
    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    proba_df = predict(trained, df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    if args.output.suffix.lower() == ".csv":
        result.to_csv(args.output, index=False)
    else:
        result.to_parquet(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
