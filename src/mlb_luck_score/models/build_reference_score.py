"""Build the Version 0.2 empirical public-score reference artifact.

Usage:

    python -m mlb_luck_score.models.build_reference_score \\
        --input data/processed/cleaned_development_data.parquet \\
        --output-dir artifacts

Trains the **unweighted** probability baseline (`class_weight=None` --
never the class-balanced comparison model, see
`mlb_luck_score.models.train_contact_model` module docstring) on
`TRAIN_SEASONS` (2021-2023 by default), predicts on `VALIDATION_SEASONS`
(2024 by default, training-eligible rows only), computes each play's
Version 0.2 raw contact luck in runs (see `mlb_luck_score.scoring.
contact_luck`), and stores the resulting positive/negative raw-luck
reference distributions (as quantile tables) in a `mlb_luck_score.scoring.
empirical_score.ReferenceScoreArtifact`, saved as JSON under the
(git-ignored) artifacts directory.

2025 is never touched: this command only ever uses `TRAIN_SEASONS`/
`VALIDATION_SEASONS` and refuses to proceed if either includes 2025 unless
`--allow-final-evaluation` is explicitly passed -- which would defeat the
purpose of a genuinely out-of-sample reference and should not be done
routinely.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import (
    ARTIFACTS_DIR,
    CLASS_ORDER,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model
from mlb_luck_score.scoring.empirical_score import (
    DEFAULT_SCORING_VERSION,
    ReferenceScoreArtifact,
    save_reference_artifact,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP, SOURCE_METADATA

logger = logging.getLogger(__name__)

DEFAULT_N_QUANTILE_POINTS = 101
REFERENCE_ARTIFACT_FILENAME_TEMPLATE = "reference_score_v{version}.json"


def build_reference_artifact(
    df: pd.DataFrame,
    *,
    train_seasons: tuple[int, ...] = TRAIN_SEASONS,
    reference_seasons: tuple[int, ...] = VALIDATION_SEASONS,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
    n_quantile_points: int = DEFAULT_N_QUANTILE_POINTS,
    scoring_version: str = DEFAULT_SCORING_VERSION,
    allow_final_evaluation: bool = False,
    extra_numeric_features: Sequence[str] = (),
    extra_categorical_features: Sequence[str] = (),
) -> ReferenceScoreArtifact:
    """Train the unweighted baseline and build the empirical reference.

    Args:
        df: A cleaned batted-ball table (single-file or combined
            development dataset) with `eligible_for_training`, `season`,
            and `outcome_class` columns.
        train_seasons: Seasons to train the base model on. Defaults to
            `TRAIN_SEASONS` (2021-2023).
        reference_seasons: Seasons whose predictions build the reference
            distribution. Defaults to `VALIDATION_SEASONS` (2024).
        run_value_map: Fixed run-value table. Defaults to `DEFAULT_RUN_VALUE_MAP`.
        n_quantile_points: Number of evenly-spaced percentile points (0-100
            inclusive) to store per side.
        scoring_version: Version tag recorded in the artifact. Use a
            distinct version (e.g. "0.3.0") when building a reference for a
            different model variant so it never overwrites another
            version's artifact file (see `mlb_luck_score.models.
            build_reference_score.REFERENCE_ARTIFACT_FILENAME_TEMPLATE`).
        allow_final_evaluation: Must be True to permit 2025 in either season
            list -- never pass this for routine use.
        extra_numeric_features: Additional numeric features beyond the
            standard set (e.g. `mlb_luck_score.features.
            build_contact_features.WEATHER_VECTOR_NUMERIC_FEATURES` to
            build a weather-aware reference artifact -- see
            `mlb_luck_score.models.compare_weather_aware`). Empty by
            default.
        extra_categorical_features: Additional categorical features beyond
            the standard set (e.g. `("venue_id",)` to build a park-aware
            reference artifact -- see `mlb_luck_score.models.
            compare_park_aware`). Empty by default.

    Returns:
        A `ReferenceScoreArtifact` ready to save or score against.

    Raises:
        ValueError: If there are no training or no reference rows, or if
            the reference distribution has no positive or no negative
            raw-luck values.
    """
    assert_seasons_allowed(
        tuple(train_seasons) + tuple(reference_seasons),
        allow_final_evaluation=allow_final_evaluation,
    )

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(train_seasons)]
    reference_df = training_eligible[training_eligible["season"].isin(reference_seasons)]

    if train_df.empty:
        raise ValueError(f"No training-eligible rows for training seasons {train_seasons}")
    if reference_df.empty:
        raise ValueError(f"No training-eligible rows for reference seasons {reference_seasons}")

    logger.info(
        "Training unweighted baseline on %d rows (seasons %s)", len(train_df), train_seasons
    )
    trained = train_model(
        train_df,
        class_weight=None,
        extra_numeric_features=extra_numeric_features,
        extra_categorical_features=extra_categorical_features,
    )
    feature_cols = trained.numeric_features + trained.categorical_features
    proba_df = predict_proba_ordered(trained, reference_df[feature_cols])

    run_value_series = pd.Series(run_value_map).reindex(list(CLASS_ORDER))
    expected_run_value = proba_df[list(CLASS_ORDER)].to_numpy() @ run_value_series.to_numpy()
    actual_run_value = reference_df["outcome_class"].astype(str).map(run_value_map).to_numpy()
    raw_luck_arr = actual_run_value - expected_run_value

    n_total = len(raw_luck_arr)
    positive = raw_luck_arr[raw_luck_arr > 0]
    negative_abs = np.abs(raw_luck_arr[raw_luck_arr < 0])
    n_zero = int((raw_luck_arr == 0).sum())

    if len(positive) == 0 or len(negative_abs) == 0:
        raise ValueError(
            "Reference distribution has no positive or no negative raw-luck values -- "
            "cannot build a two-sided percentile mapping from this data."
        )

    percentile_grid = np.linspace(0.0, 100.0, n_quantile_points)
    positive_quantiles = np.quantile(positive, percentile_grid / 100.0).tolist()
    negative_quantiles = np.quantile(negative_abs, percentile_grid / 100.0).tolist()

    logger.info(
        "Reference distribution: %d total (%d positive, %d negative, %d zero)",
        n_total,
        len(positive),
        len(negative_abs),
        n_zero,
    )

    return ReferenceScoreArtifact(
        scoring_version=scoring_version,
        model_variant=trained.variant,
        training_seasons=tuple(sorted(int(s) for s in train_df["season"].unique())),
        reference_seasons=tuple(sorted(int(s) for s in reference_df["season"].unique())),
        class_order=tuple(CLASS_ORDER),
        run_value_table=dict(run_value_map),
        sample_counts={
            "total": n_total,
            "positive": int(len(positive)),
            "negative": int(len(negative_abs)),
            "zero": n_zero,
        },
        quantile_percentiles=percentile_grid.tolist(),
        positive_quantile_values=positive_quantiles,
        negative_quantile_values=negative_quantiles,
        created_at_utc=datetime.now(UTC).isoformat(),
        package_version=package_version,
        scikit_learn_version=sklearn.__version__,
        source_metadata=dict(SOURCE_METADATA),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned batted-ball table")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--scoring-version", default=DEFAULT_SCORING_VERSION)
    parser.add_argument("--n-quantile-points", type=int, default=DEFAULT_N_QUANTILE_POINTS)
    parser.add_argument(
        "--extra-numeric-features",
        nargs="*",
        default=(),
        help="Extra numeric features beyond the standard set (e.g. air_density_kg_m3).",
    )
    parser.add_argument(
        "--extra-categorical-features",
        nargs="*",
        default=(),
        help="Extra categorical features beyond the standard set (e.g. venue_id).",
    )
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="Required to use the protected 2025 season. Never pass this for routine use.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        artifact = build_reference_artifact(
            df,
            n_quantile_points=args.n_quantile_points,
            scoring_version=args.scoring_version,
            allow_final_evaluation=args.allow_final_evaluation,
            extra_numeric_features=tuple(args.extra_numeric_features),
            extra_categorical_features=tuple(args.extra_categorical_features),
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    output_path = args.output_dir / REFERENCE_ARTIFACT_FILENAME_TEMPLATE.format(
        version=args.scoring_version
    )
    save_reference_artifact(artifact, output_path)
    logger.info("Saved reference artifact to %s", output_path)
    logger.info("Sample counts: %s", artifact.sample_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
