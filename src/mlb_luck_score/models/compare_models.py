"""Controlled comparison of contact-model variants for probability quality.

Background: an earlier baseline trained `LogisticRegression(class_weight=
"balanced")` and, on real 2021-2024 Statcast data, produced severely
miscalibrated probabilities (e.g. rows the model called ~54% likely to be a
triple were observed to be a triple ~3.6% of the time on high-sample-count
validation bins). A controlled comparison (same 2021-2023 training rows,
same untouched 2024 validation rows, changing ONLY `class_weight`) confirmed
`class_weight="balanced"` as the cause: it reweights the loss inversely to
training-class frequency, which shifts the model's fitted probabilities away
from the true class prior. See `mlb_luck_score.models.train_contact_model`
module docstring and CLAUDE.md.

This module runs that comparison (and a couple of useful benchmarks)
end-to-end and reports the numbers honestly -- it does not pick a winner for
you beyond what the metrics show:

  - `unweighted`        -- `LogisticRegression(class_weight=None)`. The
                            candidate Contact Luck probability baseline.
  - `class_balanced`    -- `LogisticRegression(class_weight="balanced")`.
                            Preserved ONLY as an explicitly-labeled
                            comparison model -- verified miscalibrated, must
                            never be used for the Contact Luck score.
  - `naive_prevalence`  -- predicts the training-set marginal class
                            distribution for every row, ignoring all
                            features. A sanity-check floor: a real model
                            that can't beat this on calibration/log-loss has
                            a serious problem.
  - `unweighted_calibrated` (optional, `--include-post-hoc-calibration`) --
                            time-ordered post-hoc isotonic calibration: base
                            model trained on 2021-2022 ONLY, calibration
                            layer fit on 2023 ONLY (never used for base
                            training), evaluated on 2024 (never used for
                            base training or calibration fitting). Tests
                            whether the unweighted model needs/benefits from
                            post-hoc calibration on top of already being
                            unweighted.

For every variant this reports: multiclass log loss, Brier score by class,
a full calibration table, overall and per-class expected calibration error
(ECE), argmax-based accuracy/recall by class (explicitly a SEPARATE concern
from calibration -- a model can have great recall and terrible calibration,
or vice versa), and saves one calibration plot set per variant.

2025 is never touched: this module only uses TRAIN_SEASONS/
VALIDATION_SEASONS and CALIBRATION_BASE_TRAIN_SEASONS/CALIBRATION_FIT_SEASONS/
CALIBRATION_EVAL_SEASONS, none of which include 2025 (see
`mlb_luck_score.config`).

Usage:

    python -m mlb_luck_score.models.compare_models \\
        --input data/processed/cleaned_development_data.parquet \\
        --output-dir outputs/tables \\
        --figures-dir outputs/figures/model_comparison \\
        --include-post-hoc-calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, recall_score

from mlb_luck_score.config import (
    CALIBRATION_BASE_TRAIN_SEASONS,
    CALIBRATION_EVAL_SEASONS,
    CALIBRATION_FIT_SEASONS,
    CLASS_ORDER,
    TABLES_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.models.baseline_models import (
    predict_proba_prevalence,
    train_prevalence_baseline,
)
from mlb_luck_score.models.calibrate_model import (
    compute_calibration_table,
    compute_expected_calibration_error,
    compute_expected_calibration_error_by_class,
    fit_calibrated_classifier,
    plot_calibration_curves,
    predict_proba_calibrated,
)
from mlb_luck_score.models.train_contact_model import (
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)

logger = logging.getLogger(__name__)

VARIANT_UNWEIGHTED = "unweighted"
VARIANT_CLASS_BALANCED = "class_balanced_comparison_only"
VARIANT_NAIVE_PREVALENCE = "naive_prevalence"
VARIANT_UNWEIGHTED_CALIBRATED = "unweighted_post_hoc_calibrated"


def _recall_by_class(
    y_true: Any, y_pred: Any, *, class_order: tuple[str, ...] = CLASS_ORDER
) -> dict[str, float]:
    """Argmax classification recall per class -- NOT a calibration measure.

    Reported alongside calibration metrics specifically so the two are
    never conflated: a model can have high recall on a class while being
    badly miscalibrated on that same class's predicted probabilities (this
    is exactly what class_weight="balanced" does for rare classes).
    """
    scores = recall_score(y_true, y_pred, labels=list(class_order), average=None, zero_division=0)
    return dict(zip(class_order, (float(s) for s in scores), strict=True))


def summarize_variant(
    variant: str,
    proba_df: pd.DataFrame,
    y_true: pd.Series,
    *,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute the full metric set for one model variant's predictions."""
    validate_probabilities(proba_df)
    y_true_arr = y_true.astype(str).to_numpy()
    labels = list(CLASS_ORDER)
    sorted_labels = sorted(labels)

    log_loss_value = float(
        log_loss(y_true_arr, proba_df[sorted_labels].to_numpy(), labels=sorted_labels)
    )
    brier_by_class = {
        cls: float(brier_score_loss((y_true_arr == cls).astype(int), proba_df[cls].to_numpy()))
        for cls in labels
    }

    calibration_table = compute_calibration_table(y_true_arr, proba_df)
    ece_overall = compute_expected_calibration_error(calibration_table)
    ece_by_class = compute_expected_calibration_error_by_class(calibration_table)

    predicted_labels = proba_df.idxmax(axis=1).to_numpy()
    recall_by_class = _recall_by_class(y_true_arr, predicted_labels)
    accuracy = float((predicted_labels == y_true_arr).mean())

    if figures_dir is not None:
        plot_calibration_curves(calibration_table, figures_dir / variant)

    return {
        "variant": variant,
        "sample_count": int(len(y_true_arr)),
        "multiclass_log_loss": log_loss_value,
        "brier_score_by_class": brier_by_class,
        "expected_calibration_error": ece_overall,
        "expected_calibration_error_by_class": ece_by_class,
        "argmax_accuracy": accuracy,
        "argmax_recall_by_class": recall_by_class,
        "calibration_table": calibration_table.to_dict(orient="records"),
    }


def run_comparison(
    df: pd.DataFrame,
    *,
    include_post_hoc_calibration: bool = False,
    figures_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the full controlled model comparison; return one summary dict per variant.

    Args:
        df: A cleaned batted-ball table (single-file or combined
            development dataset) with `eligible_for_training`, `season`,
            and `outcome_class` columns.
        include_post_hoc_calibration: Also fit and evaluate the time-ordered
            post-hoc-calibrated variant (slower: fits an extra base model
            plus an isotonic calibration layer).
        figures_dir: If given, save one calibration-plot set per variant
            under `figures_dir/<variant>/`.

    Returns:
        A list of per-variant summary dicts (see `summarize_variant`).
    """
    training_eligible = df[df["eligible_for_training"].astype(bool)]

    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the model comparison -- run `make clean-development-data` first."
        )

    results: list[dict[str, Any]] = []

    logger.info(
        "=== [%s] class_weight=None, train=%s (%d rows) ===",
        VARIANT_UNWEIGHTED,
        TRAIN_SEASONS,
        len(train_df),
    )
    unweighted = train_model(train_df, class_weight=None)
    unweighted_features = unweighted.numeric_features + unweighted.categorical_features
    unweighted_proba = predict_proba_ordered(unweighted, val_df[unweighted_features])
    results.append(
        summarize_variant(
            VARIANT_UNWEIGHTED, unweighted_proba, val_df["outcome_class"], figures_dir=figures_dir
        )
    )

    logger.info(
        "=== [%s] class_weight='balanced', train=%s (%d rows) ===",
        VARIANT_CLASS_BALANCED,
        TRAIN_SEASONS,
        len(train_df),
    )
    balanced = train_model(train_df, class_weight="balanced")
    balanced_features = balanced.numeric_features + balanced.categorical_features
    balanced_proba = predict_proba_ordered(balanced, val_df[balanced_features])
    results.append(
        summarize_variant(
            VARIANT_CLASS_BALANCED, balanced_proba, val_df["outcome_class"], figures_dir=figures_dir
        )
    )

    logger.info("=== [%s] no features, train=%s ===", VARIANT_NAIVE_PREVALENCE, TRAIN_SEASONS)
    prevalence = train_prevalence_baseline(train_df)
    prevalence_proba = predict_proba_prevalence(prevalence, len(val_df), index=val_df.index)
    results.append(
        summarize_variant(
            VARIANT_NAIVE_PREVALENCE,
            prevalence_proba,
            val_df["outcome_class"],
            figures_dir=figures_dir,
        )
    )

    if include_post_hoc_calibration:
        calibration_seasons = (
            *CALIBRATION_BASE_TRAIN_SEASONS,
            *CALIBRATION_FIT_SEASONS,
            *CALIBRATION_EVAL_SEASONS,
        )
        assert_seasons_allowed(calibration_seasons)

        cal_train_df = training_eligible[
            training_eligible["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)
        ]
        cal_fit_df = training_eligible[training_eligible["season"].isin(CALIBRATION_FIT_SEASONS)]
        cal_eval_df = training_eligible[training_eligible["season"].isin(CALIBRATION_EVAL_SEASONS)]

        if cal_train_df.empty or cal_fit_df.empty or cal_eval_df.empty:
            logger.warning(
                "Skipping post-hoc calibration comparison -- missing rows for base=%s "
                "fit=%s eval=%s.",
                CALIBRATION_BASE_TRAIN_SEASONS,
                CALIBRATION_FIT_SEASONS,
                CALIBRATION_EVAL_SEASONS,
            )
        else:
            logger.info(
                "=== [%s] base=%s (%d rows) -> fit=%s (%d rows) -> eval=%s (%d rows) ===",
                VARIANT_UNWEIGHTED_CALIBRATED,
                CALIBRATION_BASE_TRAIN_SEASONS,
                len(cal_train_df),
                CALIBRATION_FIT_SEASONS,
                len(cal_fit_df),
                CALIBRATION_EVAL_SEASONS,
                len(cal_eval_df),
            )
            base = train_model(cal_train_df, class_weight=None)
            feature_cols = base.numeric_features + base.categorical_features
            calibrated = fit_calibrated_classifier(
                base.pipeline,
                cal_fit_df[feature_cols],
                cal_fit_df["outcome_class"].astype(str),
            )
            calibrated_proba = predict_proba_calibrated(calibrated, cal_eval_df[feature_cols])
            results.append(
                summarize_variant(
                    VARIANT_UNWEIGHTED_CALIBRATED,
                    calibrated_proba,
                    cal_eval_df["outcome_class"],
                    figures_dir=figures_dir,
                )
            )

    return results


def build_comparison_summary_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten per-variant summaries into one comparison DataFrame."""
    rows = []
    for r in results:
        row = {
            "variant": r["variant"],
            "sample_count": r["sample_count"],
            "multiclass_log_loss": r["multiclass_log_loss"],
            "expected_calibration_error": r["expected_calibration_error"],
            "argmax_accuracy": r["argmax_accuracy"],
        }
        for cls in CLASS_ORDER:
            row[f"brier_{cls}"] = r["brier_score_by_class"][cls]
        for cls in CLASS_ORDER:
            row[f"recall_{cls}"] = r["argmax_recall_by_class"].get(cls, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned batted-ball table")
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument(
        "--include-post-hoc-calibration",
        action="store_true",
        help="Also fit/evaluate the time-ordered post-hoc-calibrated variant (slower).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        results = run_comparison(
            df,
            include_post_hoc_calibration=args.include_post_hoc_calibration,
            figures_dir=args.figures_dir,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    summary = build_comparison_summary_table(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    detail_path = args.output_dir / "model_comparison_detail.json"
    detail_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
