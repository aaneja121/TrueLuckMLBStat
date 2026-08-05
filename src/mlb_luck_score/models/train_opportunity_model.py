"""Contact Luck v0.7A: train the binary outfield-opportunity-difficulty model.

Predicts `P(an average MLB outfielder converts this batted ball into an out)`
for `outfield_opportunity_eligible` rows (`mlb_luck_score.eligibility.
add_outfield_opportunity_eligibility`) -- a SEPARATE binary target
(`outfield_converted_to_out`, `mlb_luck_score.features.build_contact_
features.OUTFIELD_OPPORTUNITY_TARGET_COLUMN`, the default `target_column`)
from the 5-class contact model's `outcome_class`. This trainer is reused
as-is for the Version 0.8 infield model too, by passing `target_column=
mlb_luck_score.features.build_contact_features.INFIELD_OPPORTUNITY_
TARGET_COLUMN` explicitly -- see that constant's docstring for why the two
domains no longer share one column name.
Deliberately NOT built on top of `mlb_luck_score.models.train_contact_model`:
that module is hardcoded to the 5-class target and `CLASS_ORDER` throughout
(`_prepare_xy`, `reorder_proba_columns`, `evaluate_model`'s confusion
matrix), and forcing a binary problem through that machinery would be more
convoluted than a small, focused, parallel implementation -- exactly the
"prefer small, reviewable changes" case for a new module rather than
overloading an existing one (see CLAUDE.md).

IMPORTANT -- class_weight and probability quality: the SAME lesson as
`mlb_luck_score.models.train_contact_model` applies here. `train_opportunity_
model` defaults to `class_weight=None` for the identical reason (`class_
weight="balanced"` was verified, for the 5-class model, to severely distort
predicted probabilities -- see that module's docstring and CLAUDE.md). There
is no reason to expect binary logistic regression behaves differently, so
the same default applies without re-deriving the result from scratch; if a
future change wants to test `class_weight="balanced"` for this model, verify
calibration first via `mlb_luck_score.models.compare_opportunity_models`
before ever using such a variant's probabilities as the opportunity score.

2025 is a protected final-test season: this command refuses to train or
validate on it unless `--allow-final-evaluation` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from mlb_luck_score import __version__ as package_version
from mlb_luck_score.config import TRAIN_SEASONS, VALIDATION_SEASONS, assert_seasons_allowed
from mlb_luck_score.features.build_contact_features import (
    OUTFIELD_OPPORTUNITY_TARGET_COLUMN,
    build_preprocessing_pipeline,
    select_opportunity_features,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_FILENAME_TEMPLATE = "opportunity_model_v{version}.joblib"
METADATA_FILENAME_TEMPLATE = "opportunity_model_v{version}_metadata.json"
EVALUATION_FILENAME_TEMPLATE = "opportunity_model_v{version}_evaluation.json"

VARIANT_UNWEIGHTED = "unweighted_opportunity_baseline"
#: Preserved for the same reason as `mlb_luck_score.models.
#: train_contact_model.VARIANT_CLASS_BALANCED` -- an explicitly-labeled
#: comparison model, never used to produce the opportunity probability.
VARIANT_CLASS_BALANCED = "class_balanced_opportunity_comparison_only"

MODEL_TYPE_LOGISTIC = "logistic"
#: Version 0.7C: an UNWEIGHTED nonlinear alternative for the near-wall
#: specialist comparison (`mlb_luck_score.models.compare_near_wall_models`).
#: `HistGradientBoostingClassifier` can find nonlinear interactions among
#: hang time/launch angle/wall distance/wall height/spray direction
#: automatically, unlike `LogisticRegression` -- see that module's docstring
#: for why both candidates use the IDENTICAL feature set rather than giving
#: the nonlinear model an unfair advantage via hand-engineered interactions.
#: Fed through the SAME preprocessing pipeline (numeric scaling + one-hot
#: categorical encoding) as the logistic candidate for a fair, apples
#: -to-apples feature representation -- trees split on the resulting 0/1
#: dummy columns just fine, at some efficiency cost vs. HGB's native
#: categorical support, which is irrelevant here since fairness of
#: comparison matters more than raw efficiency.
MODEL_TYPE_HGB = "hgb"


@dataclass
class TrainedOpportunityModel:
    pipeline: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    class_weight: str | None = None
    variant: str = VARIANT_UNWEIGHTED
    feature_set_label: str = "measured_contact_only_v07"
    model_type: str = MODEL_TYPE_LOGISTIC
    #: Which column this SPECIFIC trained model was fit against -- recorded
    #: per-instance (rather than assumed to be the module-level default)
    #: since the same trainer is reused for the infield domain with a
    #: different target column (see `train_opportunity_model`'s
    #: `target_column` parameter).
    target_column: str = OUTFIELD_OPPORTUNITY_TARGET_COLUMN


def _prepare_xy(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    target_column: str = OUTFIELD_OPPORTUNITY_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = numeric_features + categorical_features
    x = df[feature_cols]
    y = df[target_column].astype(int)
    return x, y


def train_opportunity_model(
    train_df: pd.DataFrame,
    *,
    class_weight: str | None = None,
    feature_set_label: str = "measured_contact_only_v07",
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    model_type: str = MODEL_TYPE_LOGISTIC,
    target_column: str = OUTFIELD_OPPORTUNITY_TARGET_COLUMN,
) -> TrainedOpportunityModel:
    """Fit the binary opportunity-difficulty model pipeline.

    Args:
        train_df: Rows already filtered to `outfield_opportunity_eligible`
            (or `infield_opportunity_eligible`, if `target_column` is set to
            the infield column) and the desired training seasons.
        class_weight: Passed straight through to the underlying classifier
            (`LogisticRegression` or `HistGradientBoostingClassifier`, both
            of which support this argument). Defaults to `None` -- see
            module docstring.
        feature_set_label: A human-readable label for which candidate's
            feature set this is (e.g. `"measured_contact_only_v07"`),
            recorded on the returned `TrainedOpportunityModel` for reporting
            -- purely descriptive, does not affect feature selection.
        numeric_features / categorical_features: If given, used AS THE
            CANDIDATE FEATURE LIST directly (still subject to the same
            missingness/leakage checks as `select_opportunity_features`).
            If omitted, defaults to `select_opportunity_features(train_df)`
            -- i.e. the Version 0.7A `measured_contact_only_v07` set. This
            indirection exists so a future candidate (e.g. Version 0.7C's
            near-wall specialist) can reuse this trainer with its own
            feature list without duplicating the training/evaluation logic.
        model_type: `MODEL_TYPE_LOGISTIC` (default) or `MODEL_TYPE_HGB` --
            see that constant's docstring.
        target_column: Which binary label column to fit against. Defaults
            to the Version 0.7A outfield target
            (`OUTFIELD_OPPORTUNITY_TARGET_COLUMN`). The Version 0.8 infield
            model reuses this SAME trainer by passing `target_column=
            mlb_luck_score.features.build_contact_features.
            INFIELD_OPPORTUNITY_TARGET_COLUMN` explicitly -- the two domains
            no longer share one column name (see that constant's docstring
            for the collision bug this replaced), so callers must say which
            one they mean rather than relying on an implicit shared default.
    """
    if numeric_features is None or categorical_features is None:
        selected_numeric, selected_categorical = select_opportunity_features(train_df)
    else:
        from mlb_luck_score.features.build_contact_features import assert_no_leakage

        assert_no_leakage(list(numeric_features) + list(categorical_features))
        selected_numeric = [c for c in numeric_features if c in train_df.columns]
        selected_categorical = [c for c in categorical_features if c in train_df.columns]

    logger.info("Opportunity numeric features: %s", selected_numeric)
    logger.info("Opportunity categorical features: %s", selected_categorical)

    if model_type not in (MODEL_TYPE_LOGISTIC, MODEL_TYPE_HGB):
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if class_weight == "balanced":
        logger.warning(
            "Training with class_weight='balanced' (%s). This variant is a labeled "
            "COMPARISON MODEL ONLY -- do NOT use its probabilities as the opportunity score.",
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
            class_weight=class_weight,
            random_state=RANDOM_SEED,
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_SEED,
        )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classify", classifier)])

    x_train, y_train = _prepare_xy(
        train_df, selected_numeric, selected_categorical, target_column=target_column
    )
    if y_train.nunique() < 2:
        raise ValueError(
            f"Training data must contain both {target_column}=0 and =1 rows to fit a "
            "binary opportunity model."
        )
    pipeline.fit(x_train, y_train)

    return TrainedOpportunityModel(
        pipeline=pipeline,
        numeric_features=selected_numeric,
        categorical_features=selected_categorical,
        class_weight=class_weight,
        variant=variant,
        feature_set_label=feature_set_label,
        target_column=target_column,
        model_type=model_type,
    )


def predict_opportunity_proba(trained: TrainedOpportunityModel, x: pd.DataFrame) -> pd.Series:
    """Predict `P(converted_to_out=1)` -- the opportunity-difficulty-implied out probability.

    Returns a single probability Series (not a class-ordered DataFrame, as
    `mlb_luck_score.models.train_contact_model.predict_proba_ordered` does
    for the 5-class model) -- a binary problem only needs one number.
    `LogisticRegression.classes_` is `[0, 1]` (sorted), so column index 1 is
    always `P(class=1)` regardless of class balance.
    """
    raw_proba = trained.pipeline.predict_proba(x)
    classes = list(trained.pipeline.named_steps["classify"].classes_)
    positive_idx = classes.index(1)
    return pd.Series(raw_proba[:, positive_idx], index=x.index, name="p_out")


def validate_opportunity_probabilities(p_out: pd.Series, *, name: str = "p_out") -> None:
    """Validate that every predicted probability is finite and in `[0, 1]`."""
    values = p_out.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if (values < 0).any() or (values > 1).any():
        bad = np.where((values < 0) | (values > 1))[0]
        raise ValueError(f"{name} out of [0, 1] range for row(s): {bad[:10].tolist()}")


def evaluate_opportunity_model(
    trained: TrainedOpportunityModel, eval_df: pd.DataFrame
) -> dict[str, Any]:
    """Compute Version 0.7A evaluation metrics (binary log loss, Brier score) on a held-out split."""
    x_eval, y_eval = _prepare_xy(
        eval_df,
        trained.numeric_features,
        trained.categorical_features,
        target_column=trained.target_column,
    )
    p_out = predict_opportunity_proba(trained, x_eval)
    validate_opportunity_probabilities(p_out)

    y_true = y_eval.to_numpy()
    metrics: dict[str, Any] = {
        "variant": trained.variant,
        "feature_set_label": trained.feature_set_label,
        "class_weight": trained.class_weight,
        "sample_count": int(len(eval_df)),
    }
    metrics["binary_log_loss"] = float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(y_true, p_out.to_numpy()))
    metrics["converted_to_out_rate"] = float(y_true.mean())
    metrics["mean_predicted_p_out"] = float(p_out.mean())

    feature_cols = trained.numeric_features + trained.categorical_features
    metrics["feature_missingness"] = {
        col: float(eval_df[col].isna().mean()) for col in feature_cols if col in eval_df.columns
    }
    return metrics


def save_artifact(
    trained: TrainedOpportunityModel,
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
        "class_weight": trained.class_weight,
        "training_seasons": train_seasons,
        "numeric_features": trained.numeric_features,
        "categorical_features": trained.categorical_features,
        "target_column": trained.target_column,
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "package_version": package_version,
        "scikit_learn_version": sklearn.__version__,
        "n_train_rows": n_train_rows,
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved opportunity model to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)
    return model_path, metadata_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Cleaned, geometry-joined batted-ball table"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default="0.7.0")
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

    from mlb_luck_score.eligibility import add_outfield_opportunity_eligibility
    from mlb_luck_score.features.build_contact_features import add_outfield_opportunity_features

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    df = add_outfield_opportunity_eligibility(df)
    df = add_outfield_opportunity_features(df)
    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]

    train_df = eligible[eligible["season"].isin(args.train_seasons)]
    if train_df.empty:
        logger.error(
            "No outfield-opportunity-eligible rows found for seasons %s in %s",
            args.train_seasons,
            args.input,
        )
        return 2
    logger.info(
        "Training on %d opportunity-eligible rows from seasons %s",
        len(train_df),
        args.train_seasons,
    )

    trained = train_opportunity_model(train_df, class_weight=None)

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
            "No opportunity-eligible validation rows found for seasons %s; skipping evaluation.",
            args.validation_seasons,
        )
    else:
        metrics = evaluate_opportunity_model(trained, eval_df)
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
