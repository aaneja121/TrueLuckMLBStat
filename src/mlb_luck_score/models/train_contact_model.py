"""Train the Version 0.1 baseline multiclass contact outcome model.

Usage:

    python -m mlb_luck_score.models.train_contact_model \\
        --input data/processed/cleaned_batted_balls.parquet \\
        --output-dir artifacts

Baseline model: multinomial logistic regression over the Version 0.1 contact
feature set (see `mlb_luck_score.features.build_contact_features`). This is
a deliberately simple baseline -- no claim of strong predictive performance
is made without actual validation results, which this command prints and
saves alongside the model artifact.

IMPORTANT -- class_weight and probability quality: `train_model` defaults to
`class_weight=None` (`VARIANT_UNWEIGHTED`). This is deliberate and was
verified empirically (see `mlb_luck_score.models.compare_models` and
CLAUDE.md): setting `class_weight="balanced"` reweights the loss inversely
to training-class frequency, which shifts the fitted model's priors away
from the true class distribution. On real 2021-2023/2024 data this produced
severe probability miscalibration (e.g. rows the model called ~54% likely
to be a triple were observed to be a triple ~3.6% of the time) even though
the *unweighted* model's aggregate predicted-vs-observed frequencies matched
almost exactly and its log loss was ~40% lower. `class_weight="balanced"` is
preserved ONLY as `VARIANT_CLASS_BALANCED`, an explicitly-labeled comparison
model -- it must never be used to produce probabilities for the Contact Luck
score.

2025 is a protected final-test season: this command refuses to train or
validate on it unless `--allow-final-evaluation` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import (
    CLASS_ORDER,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.features.build_contact_features import (
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    select_available_features,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_FILENAME_TEMPLATE = "contact_model_v{version}.joblib"
METADATA_FILENAME_TEMPLATE = "contact_model_v{version}_metadata.json"
EVALUATION_FILENAME_TEMPLATE = "contact_model_v{version}_evaluation.json"

#: The candidate Contact Luck probability baseline: no class weighting, so
#: predicted probabilities reflect the true training-class distribution.
VARIANT_UNWEIGHTED = "unweighted_probability_baseline"
#: Preserved ONLY as an explicitly-labeled comparison model. Verified to
#: produce severely miscalibrated probabilities (see module docstring) --
#: never use this variant's output as a Contact Luck probability.
VARIANT_CLASS_BALANCED = "class_balanced_comparison_only"


@dataclass
class TrainedModel:
    pipeline: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    class_order: list[str] = field(default_factory=lambda: list(CLASS_ORDER))
    class_weight: str | None = None
    variant: str = VARIANT_UNWEIGHTED


def _prepare_xy(
    df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = numeric_features + categorical_features
    x = df[feature_cols]
    y = df[TARGET_COLUMN].astype(str)
    return x, y


def train_model(
    train_df: pd.DataFrame,
    *,
    include_optional_features: bool = False,
    class_weight: str | None = None,
    extra_numeric_features: Sequence[str] = (),
    extra_categorical_features: Sequence[str] = (),
) -> TrainedModel:
    """Fit the baseline logistic-regression pipeline on training-eligible rows.

    Args:
        train_df: Training-eligible rows (already filtered to
            `eligible_for_training` and the desired training seasons).
        include_optional_features: Whether to include optional features
            (sprint speed, alignment classifications) when available.
        class_weight: Passed straight through to `LogisticRegression`.
            Defaults to `None` -- the probability-safe choice, and the one
            that should be used for the Contact Luck probability baseline.
            `"balanced"` is supported ONLY to produce the explicitly-labeled
            `VARIANT_CLASS_BALANCED` comparison model; it is verified (see
            module docstring) to severely distort predicted probabilities.
            Every triple must be KEPT in the training data regardless of
            `class_weight` -- this argument changes how the loss weights
            classes during fitting, not which rows are used.
        extra_numeric_features: Additional numeric features beyond the
            standard set (e.g. `mlb_luck_score.features.
            build_contact_features.GEOMETRY_NUMERIC_FEATURES` for a
            geometry-aware comparison variant -- see `mlb_luck_score.models.
            compare_geometry_aware`). Empty by default, so ordinary callers
            see no behavior change.
        extra_categorical_features: Additional categorical features beyond
            the standard set (e.g. `("venue_id",)` for a park-aware
            comparison variant -- see `mlb_luck_score.models.
            compare_park_aware`). Empty by default, so ordinary callers see
            no behavior change. Subject to the same missingness/leakage
            checks as every other feature (see `select_available_features`).
    """
    numeric_features, categorical_features = select_available_features(
        train_df,
        include_optional=include_optional_features,
        extra_numeric_features=extra_numeric_features,
        extra_categorical_features=extra_categorical_features,
    )
    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    if class_weight == "balanced":
        logger.warning(
            "Training with class_weight='balanced' (%s). This variant is a "
            "labeled COMPARISON MODEL ONLY -- its predicted probabilities are "
            "known to be severely miscalibrated and must NOT be used for the "
            "Contact Luck score.",
            VARIANT_CLASS_BALANCED,
        )
        variant = VARIANT_CLASS_BALANCED
    elif class_weight is None:
        variant = VARIANT_UNWEIGHTED
    else:
        variant = f"custom_class_weight[{class_weight}]"

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classify", classifier)])

    x_train, y_train = _prepare_xy(train_df, numeric_features, categorical_features)
    missing_classes = set(CLASS_ORDER) - set(y_train.unique())
    if missing_classes:
        logger.warning(
            "Training data does not contain example(s) of class(es): %s. "
            "The model will not be able to predict these classes.",
            sorted(missing_classes),
        )
    pipeline.fit(x_train, y_train)

    return TrainedModel(
        pipeline=pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_weight=class_weight,
        variant=variant,
    )


def reorder_proba_columns(
    raw_proba: np.ndarray, classes: list[str], index: pd.Index
) -> pd.DataFrame:
    """Reindex a raw (n_rows, n_classes) probability array to `CLASS_ORDER`.

    Shared by any estimator whose `.classes_` attribute orders classes
    alphabetically (every sklearn classifier does this, including
    `CalibratedClassifierCV` -- see `mlb_luck_score.models.calibrate_model`)
    rather than in the project's canonical `CLASS_ORDER`. Any class absent
    from `classes` (e.g. never seen in training) is filled with 0.0.
    """
    proba_df = pd.DataFrame(raw_proba, columns=classes, index=index)
    for cls in CLASS_ORDER:
        if cls not in proba_df.columns:
            proba_df[cls] = 0.0
    return proba_df[list(CLASS_ORDER)]


def predict_proba_ordered(trained: TrainedModel, x: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities as a DataFrame with columns in `CLASS_ORDER`.

    The underlying sklearn classifier orders `classes_` alphabetically,
    which does not match `CLASS_ORDER` -- this reindexes to the canonical
    order (filling any class absent from training with 0.0 probability) so
    every consumer downstream can rely on stable column order.
    """
    raw_proba = trained.pipeline.predict_proba(x)
    classes = list(trained.pipeline.named_steps["classify"].classes_)
    return reorder_proba_columns(raw_proba, classes, x.index)


def validate_probabilities(proba_df: pd.DataFrame, *, atol: float = 1e-6) -> None:
    """Validate that every row is finite and sums to ~1 across CLASS_ORDER."""
    if list(proba_df.columns) != list(CLASS_ORDER):
        raise ValueError(f"Expected columns {CLASS_ORDER}, got {list(proba_df.columns)}")
    values = proba_df.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Predicted probabilities contain non-finite values")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=atol))[0]
        raise ValueError(f"Predicted probabilities do not sum to 1 for row(s): {bad[:10].tolist()}")


def evaluate_model(trained: TrainedModel, eval_df: pd.DataFrame) -> dict[str, Any]:
    """Compute Version 0.1 evaluation metrics on a held-out split.

    Reports log loss, one-vs-rest Brier score per class, a confusion
    matrix, class frequencies, sample counts, and missingness -- but makes
    no claim of strong predictive performance beyond what these numbers
    show.
    """
    x_eval, y_eval = _prepare_xy(eval_df, trained.numeric_features, trained.categorical_features)
    proba_df = predict_proba_ordered(trained, x_eval)
    validate_probabilities(proba_df)

    y_true = y_eval.to_numpy()
    labels = list(CLASS_ORDER)

    metrics: dict[str, Any] = {
        "variant": trained.variant,
        "class_weight": trained.class_weight,
        "sample_count": int(len(eval_df)),
    }

    try:
        # sklearn's log_loss silently assumes y_prob columns are ordered
        # lexicographically by label, regardless of the `labels` argument's
        # order (verified: passing our CLASS_ORDER-ordered columns as-is
        # silently produces wrong results with no error, only a warning).
        # Reorder columns to lexicographic order to match what it expects.
        sorted_labels = sorted(labels)
        metrics["multiclass_log_loss"] = float(
            log_loss(y_true, proba_df[sorted_labels].to_numpy(), labels=sorted_labels)
        )
    except ValueError as exc:
        logger.warning("Could not compute log loss: %s", exc)
        metrics["multiclass_log_loss"] = None

    brier_by_class = {}
    for cls in labels:
        binary_true = (y_true == cls).astype(int)
        brier_by_class[cls] = float(brier_score_loss(binary_true, proba_df[cls].to_numpy()))
    metrics["brier_score_by_class"] = brier_by_class

    predicted_labels = proba_df.idxmax(axis=1).to_numpy()
    metrics["confusion_matrix"] = confusion_matrix(y_true, predicted_labels, labels=labels).tolist()
    metrics["confusion_matrix_labels"] = labels

    metrics["outcome_class_frequencies"] = (
        pd.Series(y_true).value_counts(normalize=True).reindex(labels).fillna(0.0).to_dict()
    )

    feature_cols = trained.numeric_features + trained.categorical_features
    metrics["feature_missingness"] = {
        col: float(eval_df[col].isna().mean()) for col in feature_cols if col in eval_df.columns
    }

    return metrics


def save_artifact(
    trained: TrainedModel,
    *,
    output_dir: Path,
    model_version: str,
    train_seasons: list[int],
    n_train_rows: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME_TEMPLATE.format(version=model_version)
    metadata_path = output_dir / METADATA_FILENAME_TEMPLATE.format(version=model_version)

    joblib.dump(trained.pipeline, model_path)

    metadata = {
        "model_version": model_version,
        "variant": trained.variant,
        "class_weight": trained.class_weight,
        "training_seasons": train_seasons,
        "numeric_features": trained.numeric_features,
        "categorical_features": trained.categorical_features,
        "class_order": trained.class_order,
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "package_version": package_version,
        "scikit_learn_version": sklearn.__version__,
        "n_train_rows": n_train_rows,
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)
    return model_path, metadata_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned batted-ball table")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=list(TRAIN_SEASONS))
    parser.add_argument(
        "--validation-seasons", type=int, nargs="+", default=list(VALIDATION_SEASONS)
    )
    parser.add_argument("--include-optional-features", action="store_true")
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help=(
            "'none' (default) trains the unweighted probability baseline -- use this for "
            "the Contact Luck score. 'balanced' trains the explicitly-labeled comparison "
            "model only; its probabilities are verified to be severely miscalibrated and "
            "must not be used for scoring (see module docstring)."
        ),
    )
    parser.add_argument(
        "--allow-final-evaluation",
        action="store_true",
        help="Required to train or validate on the protected 2025 season.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    assert_seasons_allowed(
        args.train_seasons + args.validation_seasons,
        allow_final_evaluation=args.allow_final_evaluation,
    )

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    training_eligible = df[df["eligible_for_training"].astype(bool)]

    train_df = training_eligible[training_eligible["season"].isin(args.train_seasons)]
    if train_df.empty:
        logger.error(
            "No training-eligible rows found for seasons %s in %s", args.train_seasons, args.input
        )
        return 2
    logger.info("Training on %d rows from seasons %s", len(train_df), args.train_seasons)

    class_weight = None if args.class_weight == "none" else args.class_weight
    if class_weight == "balanced":
        logger.warning(
            "--class-weight=balanced was requested: this produces the "
            "VARIANT_CLASS_BALANCED comparison model. Its probabilities are verified to be "
            "severely miscalibrated (see module docstring) -- do not use this artifact for "
            "the Contact Luck score."
        )
    trained = train_model(
        train_df,
        include_optional_features=args.include_optional_features,
        class_weight=class_weight,
    )

    model_path, metadata_path = save_artifact(
        trained,
        output_dir=args.output_dir,
        model_version=args.model_version,
        train_seasons=args.train_seasons,
        n_train_rows=len(train_df),
    )

    eval_df = training_eligible[training_eligible["season"].isin(args.validation_seasons)]
    if eval_df.empty:
        logger.warning(
            "No validation-eligible rows found for seasons %s; skipping evaluation.",
            args.validation_seasons,
        )
    else:
        metrics = evaluate_model(trained, eval_df)
        eval_path = args.output_dir / EVALUATION_FILENAME_TEMPLATE.format(
            version=args.model_version
        )
        eval_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Validation metrics (seasons %s): %s", args.validation_seasons, metrics)
        logger.info("Saved evaluation metrics to %s", eval_path)

    logger.info("Model artifact: %s", model_path)
    logger.info("Metadata: %s", metadata_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
