"""Contact Luck v0.9: train the multinomial batter-runner advancement model.

Predicts `P(batter_final_base = <label>)` for `advancement_eligible` rows
(`mlb_luck_score.eligibility.add_advancement_eligibility`) over the 5-class
target `batter_final_base` (`mlb_luck_score.eligibility.ADVANCEMENT_LABELS`)
-- a genuinely DIFFERENT target from both the 5-class contact model's
`outcome_class`/`CLASS_ORDER` (`mlb_luck_score.models.train_contact_model`)
and the various binary opportunity models (`mlb_luck_score.models.
train_opportunity_model`). Deliberately NOT built on top of either: `train_
contact_model` is hardcoded to `TARGET_COLUMN`/`CLASS_ORDER` throughout, and
`train_opportunity_model` is hardcoded to a binary target -- forcing a
DIFFERENTLY-LABELED 5-class problem through either would be more convoluted
than a small, focused, parallel implementation, exactly the "prefer small,
reviewable changes" case for a new module (see CLAUDE.md).

IMPORTANT -- class_weight and probability quality: the SAME lesson as
`train_contact_model`/`train_opportunity_model` applies. `train_advancement_
model` defaults to `class_weight=None` for the identical reason (`"balanced"`
was verified, for the 5-class CONTACT model, to severely distort predicted
probabilities). There is no reason to expect a different 5-class multinomial
problem behaves differently, so the same default applies without
re-deriving the result from scratch; if a future change wants to test
`class_weight="balanced"` here, verify calibration first via `mlb_luck_score.
models.compare_advancement_models` before ever using such a variant's
probabilities.

Uses `sklearn.linear_model.LogisticRegression`'s default multinomial
(softmax) handling for >2 classes -- no special parameter needed, this IS
"multinomial logistic regression" as the task requires, not a one-vs-rest
decomposition. `HistGradientBoostingClassifier` (Version 0.7C/0.8's
established nonlinear-candidate choice) also natively supports multiclass
softmax output.

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import TRAIN_SEASONS, VALIDATION_SEASONS, assert_seasons_allowed
from mlb_luck_score.eligibility import ADVANCEMENT_LABELS
from mlb_luck_score.features.build_contact_features import build_preprocessing_pipeline

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_FILENAME_TEMPLATE = "advancement_model_v{version}.joblib"
METADATA_FILENAME_TEMPLATE = "advancement_model_v{version}_metadata.json"
EVALUATION_FILENAME_TEMPLATE = "advancement_model_v{version}_evaluation.json"

TARGET_COLUMN = "batter_final_base"

VARIANT_UNWEIGHTED = "unweighted_advancement_baseline"
#: Preserved for the same reason as `train_contact_model.VARIANT_CLASS_
#: BALANCED` -- an explicitly-labeled comparison model, never used to
#: produce the advancement probabilities.
VARIANT_CLASS_BALANCED = "class_balanced_advancement_comparison_only"

MODEL_TYPE_LOGISTIC = "logistic"
MODEL_TYPE_HGB = "hgb"


@dataclass
class TrainedAdvancementModel:
    pipeline: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    class_order: list[str] = field(default_factory=lambda: list(ADVANCEMENT_LABELS))
    class_weight: str | None = None
    variant: str = VARIANT_UNWEIGHTED
    feature_set_label: str = "advancement_context_v09"
    model_type: str = MODEL_TYPE_LOGISTIC


def _prepare_xy(
    df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = numeric_features + categorical_features
    x = df[feature_cols]
    y = df[TARGET_COLUMN].astype(str)
    return x, y


def train_advancement_model(
    train_df: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    class_weight: str | None = None,
    feature_set_label: str = "advancement_context_v09",
    model_type: str = MODEL_TYPE_LOGISTIC,
) -> TrainedAdvancementModel:
    """Fit the multinomial advancement model pipeline.

    Args:
        train_df: Rows already filtered to `advancement_eligible` and the
            desired training seasons.
        numeric_features / categorical_features: The candidate feature list
            -- ALWAYS required explicitly (unlike `train_opportunity_model`,
            there is no single "default" Version 0.9 feature set; see
            `mlb_luck_score.features.build_contact_features.
            select_advancement_features`). Subject to the same
            missingness/leakage checks as every other feature set
            (`assert_no_leakage`).
        class_weight: Passed straight through to the underlying classifier.
            Defaults to `None` -- see module docstring.
        feature_set_label: A human-readable label for which candidate this
            is (e.g. `"advancement_context_v09"`), recorded on the returned
            `TrainedAdvancementModel` for reporting only.
        model_type: `MODEL_TYPE_LOGISTIC` (default) or `MODEL_TYPE_HGB`.
    """
    from mlb_luck_score.features.build_contact_features import assert_no_leakage

    assert_no_leakage(list(numeric_features) + list(categorical_features))
    selected_numeric = [c for c in numeric_features if c in train_df.columns]
    selected_categorical = [c for c in categorical_features if c in train_df.columns]

    logger.info("Advancement numeric features: %s", selected_numeric)
    logger.info("Advancement categorical features: %s", selected_categorical)

    if model_type not in (MODEL_TYPE_LOGISTIC, MODEL_TYPE_HGB):
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if class_weight == "balanced":
        logger.warning(
            "Training with class_weight='balanced' (%s). This variant is a labeled "
            "COMPARISON MODEL ONLY -- do NOT use its probabilities for real scoring.",
            VARIANT_CLASS_BALANCED,
        )
        variant = VARIANT_CLASS_BALANCED
    elif class_weight is None:
        variant = VARIANT_UNWEIGHTED
    else:
        variant = f"custom_class_weight[{class_weight}]"

    preprocessor = build_preprocessing_pipeline(selected_numeric, selected_categorical)
    classifier: LogisticRegression | HistGradientBoostingClassifier
    if model_type == MODEL_TYPE_HGB:
        classifier = HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=RANDOM_SEED
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=RANDOM_SEED
        )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classify", classifier)])

    x_train, y_train = _prepare_xy(train_df, selected_numeric, selected_categorical)
    missing_classes = set(ADVANCEMENT_LABELS) - set(y_train.unique())
    if missing_classes:
        logger.warning(
            "Training data does not contain example(s) of class(es): %s. The model "
            "will not be able to predict these classes.",
            sorted(missing_classes),
        )
    pipeline.fit(x_train, y_train)

    return TrainedAdvancementModel(
        pipeline=pipeline,
        numeric_features=selected_numeric,
        categorical_features=selected_categorical,
        class_weight=class_weight,
        variant=variant,
        feature_set_label=feature_set_label,
        model_type=model_type,
    )


def reorder_advancement_proba_columns(
    raw_proba: np.ndarray, classes: list[str], index: pd.Index
) -> pd.DataFrame:
    """Reindex a raw (n_rows, n_classes) probability array to `ADVANCEMENT_LABELS`.

    Same reasoning as `train_contact_model.reorder_proba_columns` -- sklearn
    orders `.classes_` alphabetically, not in the project's canonical
    `ADVANCEMENT_LABELS` order. Any class absent from `classes` (e.g. never
    observed in training -- realistic here, since `inside_the_park_home_run`
    and `retired_while_advancing` are both rare) is filled with 0.0.
    """
    proba_df = pd.DataFrame(raw_proba, columns=classes, index=index)
    for cls in ADVANCEMENT_LABELS:
        if cls not in proba_df.columns:
            proba_df[cls] = 0.0
    return proba_df[list(ADVANCEMENT_LABELS)]


def predict_advancement_proba(trained: TrainedAdvancementModel, x: pd.DataFrame) -> pd.DataFrame:
    """Predict probabilities as a DataFrame with columns in `ADVANCEMENT_LABELS` order."""
    raw_proba = trained.pipeline.predict_proba(x)
    classes = list(trained.pipeline.named_steps["classify"].classes_)
    return reorder_advancement_proba_columns(raw_proba, classes, x.index)


def validate_advancement_probabilities(proba_df: pd.DataFrame, *, atol: float = 1e-6) -> None:
    """Validate that every row is finite, in [0, 1], and sums to ~1 across `ADVANCEMENT_LABELS`."""
    if list(proba_df.columns) != list(ADVANCEMENT_LABELS):
        raise ValueError(f"Expected columns {ADVANCEMENT_LABELS}, got {list(proba_df.columns)}")
    values = proba_df.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Predicted probabilities contain non-finite values")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Predicted probabilities out of [0, 1] range")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=atol))[0]
        raise ValueError(f"Predicted probabilities do not sum to 1 for row(s): {bad[:10].tolist()}")


def evaluate_advancement_model(
    trained: TrainedAdvancementModel, eval_df: pd.DataFrame
) -> dict[str, Any]:
    """Compute Version 0.9 evaluation metrics (multiclass log loss, per-class one-vs-rest
    Brier score, confusion matrix, class frequencies, sample counts, missingness) on a
    held-out split. Makes no claim of strong predictive performance beyond these numbers.
    """
    x_eval, y_eval = _prepare_xy(eval_df, trained.numeric_features, trained.categorical_features)
    proba_df = predict_advancement_proba(trained, x_eval)
    validate_advancement_probabilities(proba_df)

    y_true = y_eval.to_numpy()
    labels = list(ADVANCEMENT_LABELS)

    metrics: dict[str, Any] = {
        "variant": trained.variant,
        "feature_set_label": trained.feature_set_label,
        "model_type": trained.model_type,
        "class_weight": trained.class_weight,
        "sample_count": int(len(eval_df)),
    }

    try:
        # Same sklearn log_loss lexicographic-column-order gotcha documented
        # in train_contact_model.evaluate_model -- reorder to match.
        sorted_labels = sorted(labels)
        metrics["multiclass_log_loss"] = float(
            log_loss(y_true, proba_df[sorted_labels].to_numpy(), labels=sorted_labels)
        )
    except ValueError as exc:
        metrics["multiclass_log_loss"] = None
        metrics["multiclass_log_loss_error"] = str(exc)

    per_class_brier: dict[str, float] = {}
    for cls in labels:
        y_binary = (y_true == cls).astype(int)
        per_class_brier[cls] = float(brier_score_loss(y_binary, proba_df[cls].to_numpy()))
    metrics["per_class_brier_score"] = per_class_brier

    metrics["class_frequencies"] = {cls: int((y_true == cls).sum()) for cls in labels}
    cm = confusion_matrix(y_true, proba_df.idxmax(axis=1).to_numpy(), labels=labels)
    metrics["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}

    feature_cols = trained.numeric_features + trained.categorical_features
    metrics["feature_missingness"] = {
        col: float(eval_df[col].isna().mean()) for col in feature_cols if col in eval_df.columns
    }
    return metrics


def save_artifact(
    trained: TrainedAdvancementModel,
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
        "feature_set_label": trained.feature_set_label,
        "model_type": trained.model_type,
        "class_weight": trained.class_weight,
        "training_seasons": train_seasons,
        "numeric_features": trained.numeric_features,
        "categorical_features": trained.categorical_features,
        "target_column": TARGET_COLUMN,
        "class_order": list(ADVANCEMENT_LABELS),
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "package_version": package_version,
        "scikit_learn_version": sklearn.__version__,
        "n_train_rows": n_train_rows,
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved advancement model to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)
    return model_path, metadata_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned, geometry+sprint-speed-joined batted-ball table",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default="0.9.0")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=list(TRAIN_SEASONS))
    parser.add_argument(
        "--validation-seasons", type=int, nargs="+", default=list(VALIDATION_SEASONS)
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

    from mlb_luck_score.eligibility import (
        add_advancement_eligibility,
        add_outfield_opportunity_eligibility,
        compute_eligibility,
    )
    from mlb_luck_score.features.build_contact_features import (
        ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
        ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
        add_advancement_contact_probability_features,
        add_advancement_features,
        select_advancement_features,
    )
    from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_advancement_features(df)
    eligible = df[df["advancement_eligible"].astype(bool)]

    train_df = eligible[eligible["season"].isin(args.train_seasons)]
    if train_df.empty:
        logger.error(
            "No advancement-eligible rows found for seasons %s in %s",
            args.train_seasons,
            args.input,
        )
        return 2
    logger.info(
        "Training on %d advancement-eligible rows from seasons %s",
        len(train_df),
        args.train_seasons,
    )

    # Standalone CLI default: fit baseline_v02 on the SAME train seasons'
    # contact-eligible rows (mirroring mlb_luck_score.models.
    # compare_advancement_models' own data prep) so `advancement_context_
    # v09`'s contact-probability features are available. The comparison
    # module is the primary entry point for multi-candidate evaluation --
    # this CLI trains one model with one feature set for convenience.
    contact_train_df = df[
        df["eligible_for_training"].fillna(False) & df["season"].isin(args.train_seasons)
    ]
    contact_trained = train_model(contact_train_df, class_weight=None)
    contact_proba = predict_proba_ordered(
        contact_trained,
        train_df[
            list(contact_trained.numeric_features) + list(contact_trained.categorical_features)
        ],
    )
    train_df = add_advancement_contact_probability_features(train_df, contact_proba)

    numeric_features, categorical_features = select_advancement_features(
        train_df,
        numeric_candidates=ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
        categorical_candidates=ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
    )
    trained = train_advancement_model(
        train_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_set_label="advancement_context_v09",
    )

    model_path, metadata_path = save_artifact(
        trained,
        output_dir=args.output_dir,
        model_version=args.model_version,
        train_seasons=args.train_seasons,
        n_train_rows=len(train_df),
    )

    eval_df = eligible[eligible["season"].isin(args.validation_seasons)]
    if eval_df.empty:
        logger.warning(
            "No advancement-eligible validation rows found for seasons %s; skipping evaluation.",
            args.validation_seasons,
        )
    else:
        eval_contact_feature_cols = list(contact_trained.numeric_features) + list(
            contact_trained.categorical_features
        )
        eval_contact_proba = predict_proba_ordered(
            contact_trained, eval_df[eval_contact_feature_cols]
        )
        eval_df = add_advancement_contact_probability_features(eval_df, eval_contact_proba)
        metrics = evaluate_advancement_model(trained, eval_df)
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
