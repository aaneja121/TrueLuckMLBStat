"""Contact Luck v0.3: does adding venue_id improve out-of-sample probability quality?

Compares two controlled variants on the SAME 2021-2023 training rows and
the SAME untouched 2024 validation rows (2025 never touched):

  - `baseline_v02`: the currently-selected model -- unweighted
    (`class_weight=None`), features = launch_speed/launch_angle/
    spray_angle_approx/hit_distance_sc + bb_type/stand. Identical to
    `mlb_luck_score.models.train_contact_model.train_model()`'s default.
  - `park_aware_v03_candidate`: the EXACT same model and features, PLUS one
    additional categorical feature, `venue_id` (joined from
    `mlb_luck_score.data.join_venue_metadata`). `venue_id` is known before
    the pitch is thrown -- it is not derived from the play's outcome and is
    not a target-leakage column (see `mlb_luck_score.config.
    LEAKAGE_COLUMNS`).

Reports, for both variants: multiclass log loss, Brier score by class,
expected calibration error (ECE) overall/by class/by venue, sample counts
by venue, a dedicated home-run calibration check, and subgroup calibration
for fly balls and high-projected-distance contact. Accuracy is reported
only as a SECONDARY number -- see `mlb_luck_score.models.compare_models`
module docstring for why classification accuracy/recall must never be
conflated with probability calibration.

Does NOT include published park factors (e.g. altitude or Green Monster
adjustments) -- only the venue's bare categorical identity. Does NOT use
any feature derived from the 2024 outcomes, and never touches 2025.

ADOPTION RULE: do NOT automatically prefer `park_aware_v03_candidate`.
`recommend_park_aware_adoption` implements one transparent, documented
rule (log loss AND ECE both improve, AND no venue shows a "material"
calibration regression -- flagged if a reliably-sampled venue's ECE
worsens by more than an absolute margin OR a relative-to-baseline margin,
see `find_material_venue_regressions`) -- inspect the full by-venue table
yourself before deciding; a rule-based recommendation is a starting point,
not a substitute for judgment.

Usage:

    python -m mlb_luck_score.models.compare_park_aware \\
        --input data/processed/cleaned_development_data_with_venue.parquet \\
        --output-dir outputs/tables --figures-dir outputs/figures/park_aware
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb_luck_score.config import CLASS_ORDER, TABLES_DIR, TRAIN_SEASONS, VALIDATION_SEASONS
from mlb_luck_score.models.calibrate_model import (
    compute_calibration_table,
    compute_expected_calibration_error,
    compute_expected_calibration_error_by_class,
    plot_calibration_curves,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)

logger = logging.getLogger(__name__)

VARIANT_BASELINE_V02 = "baseline_v02"
VARIANT_PARK_AWARE_V03_CANDIDATE = "park_aware_v03_candidate"

DEFAULT_MIN_VENUE_SAMPLES = 100
DEFAULT_HIGH_DISTANCE_QUANTILE = 0.75
#: Scale-sensitive "material venue regression" thresholds. A fixed absolute
#: threshold alone is not appropriate here: real per-venue ECE values in
#: this dataset run roughly 0.01-0.07, so a single large absolute cutoff
#: (e.g. 0.15) would never fire in practice and would miss real, meaningful
#: regressions (verified: Fenway Park's per-venue ECE moved 0.016 -> 0.025,
#: a real +58% relative increase, which a purely-absolute high threshold
#: did not catch). A venue is flagged if it worsens by MORE THAN
#: `DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN` in absolute ECE, OR by more than
#: `DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN` (a fraction, e.g. 0.50 = 50%)
#: relative to its own baseline ECE -- whichever is more sensitive at that
#: venue's scale. Only venues with at least `DEFAULT_MIN_VENUE_SAMPLES`
#: baseline rows (see `reliable` in `compute_calibration_by_venue`) are
#: eligible to be flagged at all.
DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN = 0.01
DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN = 0.50
_MISSING_VENUE_LABEL = "__missing_venue__"


def _prepare_venue_column(df: pd.DataFrame) -> pd.DataFrame:
    """Cast `venue_id` to a clean nullable-string categorical for training.

    Real missing values (a game with no venue metadata) stay as actual
    nulls so the standard categorical imputer treats them as a "missing"
    category rather than the literal string "nan" or "<NA>".
    """
    out = df.copy()
    out["venue_id"] = out["venue_id"].astype("Int64").astype("string")
    return out


def compute_calibration_by_venue(
    y_true: pd.Series,
    proba_df: pd.DataFrame,
    venue_ids: pd.Series,
    *,
    min_reliable_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
) -> pd.DataFrame:
    """Per-venue calibration summary: overall ECE and home-run-only ECE.

    Missing venue_id values are grouped into a single `"__missing_venue__"`
    bucket rather than dropped.
    """
    y_true_arr = np.asarray(y_true).astype(str)
    grouped_venue = venue_ids.fillna(_MISSING_VENUE_LABEL).to_numpy()

    rows: list[dict[str, Any]] = []
    for venue_value in pd.unique(grouped_venue):
        mask = grouped_venue == venue_value
        n = int(mask.sum())
        if n == 0:
            continue
        sub_proba = proba_df.iloc[mask]
        table = compute_calibration_table(y_true_arr[mask], sub_proba)
        if table.empty:
            continue
        overall_ece = compute_expected_calibration_error(table)
        hr_table = table[table["outcome_class"] == "home_run"]
        hr_ece = (
            compute_expected_calibration_error(hr_table) if not hr_table.empty else float("nan")
        )
        rows.append(
            {
                "venue_id": venue_value,
                "sample_count": n,
                "ece_overall": overall_ece,
                "ece_home_run": hr_ece,
                "reliable": n >= min_reliable_samples,
            }
        )
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def compute_subgroup_calibration(
    y_true: pd.Series, proba_df: pd.DataFrame, mask: np.ndarray, label: str
) -> dict[str, Any]:
    """Calibration/log-loss summary restricted to rows where `mask` is True."""
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "sample_count": 0, "ece": None, "log_loss": None}

    y_true_arr = np.asarray(y_true).astype(str)[mask]
    sub_proba = proba_df.iloc[mask]
    table = compute_calibration_table(y_true_arr, sub_proba)
    ece = compute_expected_calibration_error(table) if not table.empty else float("nan")
    sorted_labels = sorted(CLASS_ORDER)
    ll = float(log_loss(y_true_arr, sub_proba[sorted_labels].to_numpy(), labels=sorted_labels))
    return {"label": label, "sample_count": n, "ece": ece, "log_loss": ll}


def _summarize_variant(
    variant: str,
    trained: TrainedModel,
    proba_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    min_venue_samples: int,
    high_distance_quantile: float,
    figures_dir: Path | None,
) -> dict[str, Any]:
    validate_probabilities(proba_df)
    y_true = val_df["outcome_class"].astype(str)
    y_true_arr = y_true.to_numpy()
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
    accuracy = float((predicted_labels == y_true_arr).mean())

    calibration_by_venue = compute_calibration_by_venue(
        y_true, proba_df, val_df["venue_id"], min_reliable_samples=min_venue_samples
    )

    fly_ball_mask = (val_df["bb_type"] == "fly_ball").to_numpy()
    fly_ball_summary = compute_subgroup_calibration(y_true, proba_df, fly_ball_mask, "fly_ball")

    if "hit_distance_sc" in val_df.columns and val_df["hit_distance_sc"].notna().any():
        distance_threshold = float(val_df["hit_distance_sc"].quantile(high_distance_quantile))
        # A handful of rows have a missing hit_distance_sc even though they passed
        # eligibility (only launch_speed/launch_angle are required) -- treat those as
        # "not in the high-distance subgroup" rather than propagating NaN/NA into the mask.
        high_distance_mask = (
            (val_df["hit_distance_sc"] >= distance_threshold).fillna(False).to_numpy(dtype=bool)
        )
    else:
        distance_threshold = float("nan")
        high_distance_mask = np.zeros(len(val_df), dtype=bool)
    high_distance_summary = compute_subgroup_calibration(
        y_true,
        proba_df,
        high_distance_mask,
        f"hit_distance_sc>=p{int(high_distance_quantile * 100)}",
    )
    high_distance_summary["distance_threshold_ft"] = distance_threshold

    if figures_dir is not None:
        plot_calibration_curves(calibration_table, figures_dir / variant)

    return {
        "variant": variant,
        "features": trained.numeric_features + trained.categorical_features,
        "sample_count": int(len(y_true_arr)),
        "multiclass_log_loss": log_loss_value,
        "brier_score_by_class": brier_by_class,
        "expected_calibration_error": ece_overall,
        "expected_calibration_error_by_class": ece_by_class,
        "home_run_ece": ece_by_class.get("home_run", float("nan")),
        "argmax_accuracy_secondary": accuracy,
        "calibration_by_venue": calibration_by_venue.to_dict(orient="records"),
        "fly_ball_subgroup": fly_ball_summary,
        "high_distance_subgroup": high_distance_summary,
    }


def run_park_aware_comparison(
    joined_df: pd.DataFrame,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
    high_distance_quantile: float = DEFAULT_HIGH_DISTANCE_QUANTILE,
    figures_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate both variants; return one summary dict per variant.

    Args:
        joined_df: Cleaned development data already joined with venue
            metadata (see `mlb_luck_score.data.join_venue_metadata`), with
            an `eligible_for_training`, `season`, `outcome_class`, and
            `venue_id` column.
        min_venue_samples: Minimum rows for a venue's calibration figures to
            be marked reliable.
        high_distance_quantile: Percentile threshold (of validation-set
            `hit_distance_sc`) defining "high projected-distance contact".
        figures_dir: If given, save one calibration plot set per variant.

    Returns:
        `{"baseline_v02": {...}, "park_aware_v03_candidate": {...}}`.
    """
    if "venue_id" not in joined_df.columns:
        raise ValueError(
            "joined_df has no 'venue_id' column -- run "
            "mlb_luck_score.data.join_venue_metadata.join_venue_metadata first."
        )

    training_eligible = joined_df[joined_df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the park-aware comparison."
        )

    train_df = _prepare_venue_column(train_df)
    val_df = _prepare_venue_column(val_df)

    logger.info(
        "=== [%s] training on %d rows (no venue feature) ===", VARIANT_BASELINE_V02, len(train_df)
    )
    baseline = train_model(train_df, class_weight=None)
    baseline_features = baseline.numeric_features + baseline.categorical_features
    baseline_proba = predict_proba_ordered(baseline, val_df[baseline_features])
    baseline_summary = _summarize_variant(
        VARIANT_BASELINE_V02,
        baseline,
        baseline_proba,
        val_df,
        min_venue_samples=min_venue_samples,
        high_distance_quantile=high_distance_quantile,
        figures_dir=figures_dir,
    )

    logger.info(
        "=== [%s] training on %d rows (+ venue_id feature) ===",
        VARIANT_PARK_AWARE_V03_CANDIDATE,
        len(train_df),
    )
    park_aware = train_model(train_df, class_weight=None, extra_categorical_features=("venue_id",))
    if "venue_id" not in park_aware.categorical_features:
        raise ValueError(
            "venue_id was not included in the park-aware model's features (likely too much "
            "missing venue data) -- cannot run a meaningful comparison."
        )
    park_aware_features = park_aware.numeric_features + park_aware.categorical_features
    park_aware_proba = predict_proba_ordered(park_aware, val_df[park_aware_features])
    park_aware_summary = _summarize_variant(
        VARIANT_PARK_AWARE_V03_CANDIDATE,
        park_aware,
        park_aware_proba,
        val_df,
        min_venue_samples=min_venue_samples,
        high_distance_quantile=high_distance_quantile,
        figures_dir=figures_dir,
    )

    return {
        VARIANT_BASELINE_V02: baseline_summary,
        VARIANT_PARK_AWARE_V03_CANDIDATE: park_aware_summary,
    }


def find_material_venue_regressions(
    baseline_by_venue: list[dict[str, Any]] | pd.DataFrame,
    candidate_by_venue: list[dict[str, Any]] | pd.DataFrame,
    *,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
) -> list[str]:
    """Venues where the candidate's calibration worsens materially vs. baseline.

    Scale-sensitive by design (see module-level constants): a venue with at
    least `DEFAULT_MIN_VENUE_SAMPLES` baseline rows (`reliable_baseline`) is
    flagged if EITHER:

      - its overall ECE increases by more than `absolute_margin` (in
        absolute ECE units), OR
      - its overall ECE increases by more than `relative_margin` relative
        to its own baseline ECE (e.g. 0.50 = a 50% relative increase).

    Using only an absolute threshold is inappropriate here: real per-venue
    ECE values are typically small (roughly 0.01-0.07 in this dataset), so
    a single large absolute cutoff would rarely fire and would miss real
    regressions that are small in absolute terms but large relative to that
    venue's own baseline. Using only a relative threshold has the opposite
    problem for venues whose baseline ECE is already tiny (a jump from
    0.001 to 0.003 is a 200% relative increase but negligible in practice)
    -- the OR combination catches both failure modes. This is a starting
    point for judgment, not a substitute for reading the full table.
    """
    baseline_df = pd.DataFrame(baseline_by_venue)
    candidate_df = pd.DataFrame(candidate_by_venue)
    if baseline_df.empty or candidate_df.empty:
        return []

    merged = baseline_df.merge(
        candidate_df, on="venue_id", suffixes=("_baseline", "_candidate"), how="inner"
    )
    reliable = merged[merged["reliable_baseline"]].copy()
    if reliable.empty:
        return []

    baseline_ece = reliable["ece_overall_baseline"]
    candidate_ece = reliable["ece_overall_candidate"]
    absolute_delta = candidate_ece - baseline_ece

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_delta = absolute_delta / baseline_ece
    relative_delta = relative_delta.replace([np.inf, -np.inf], np.nan)

    flagged_mask = (absolute_delta > absolute_margin) | (
        relative_delta.fillna(-np.inf) > relative_margin
    )
    return [str(v) for v in reliable.loc[flagged_mask, "venue_id"].tolist()]


def recommend_park_aware_adoption(
    comparison: dict[str, dict[str, Any]],
    *,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
) -> dict[str, Any]:
    """Apply one transparent, documented adoption rule (not a final decision).

    Recommends `park_aware_v03_candidate` only if BOTH multiclass log loss
    and overall ECE improve on 2024 validation data AND no venue shows a
    "material" calibration regression (see `find_material_venue_regressions`
    for the scale-sensitive absolute-OR-relative rule).
    """
    baseline = comparison[VARIANT_BASELINE_V02]
    candidate = comparison[VARIANT_PARK_AWARE_V03_CANDIDATE]

    improves_log_loss = candidate["multiclass_log_loss"] < baseline["multiclass_log_loss"]
    improves_ece = candidate["expected_calibration_error"] < baseline["expected_calibration_error"]
    material_regressions = find_material_venue_regressions(
        baseline["calibration_by_venue"],
        candidate["calibration_by_venue"],
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
    )

    adopt = improves_log_loss and improves_ece and not material_regressions
    return {
        "recommend_adopt_park_aware_v03": adopt,
        "improves_log_loss": improves_log_loss,
        "improves_ece": improves_ece,
        "log_loss_delta": candidate["multiclass_log_loss"] - baseline["multiclass_log_loss"],
        "ece_delta": candidate["expected_calibration_error"]
        - baseline["expected_calibration_error"],
        "material_venue_regressions": material_regressions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with venue metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--min-venue-samples", type=int, default=DEFAULT_MIN_VENUE_SAMPLES)
    parser.add_argument(
        "--high-distance-quantile", type=float, default=DEFAULT_HIGH_DISTANCE_QUANTILE
    )
    parser.add_argument(
        "--material-absolute-margin",
        type=float,
        default=DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
        help="Flag a reliably-sampled venue if its ECE worsens by more than this (absolute).",
    )
    parser.add_argument(
        "--material-relative-margin",
        type=float,
        default=DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
        help="Flag a reliably-sampled venue if its ECE worsens by more than this fraction "
        "relative to its own baseline ECE (e.g. 0.50 = 50%%).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        comparison = run_park_aware_comparison(
            df,
            min_venue_samples=args.min_venue_samples,
            high_distance_quantile=args.high_distance_quantile,
            figures_dir=args.figures_dir,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    recommendation = recommend_park_aware_adoption(
        comparison,
        absolute_margin=args.material_absolute_margin,
        relative_margin=args.material_relative_margin,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "park_aware_comparison_detail.json"
    detail_path.write_text(
        json.dumps({**comparison, "recommendation": recommendation}, indent=2, default=str)
    )

    for variant, summary in comparison.items():
        logger.info(
            "[%s] log_loss=%.6f ece=%.6f home_run_ece=%.6f accuracy(secondary)=%.4f n=%d",
            variant,
            summary["multiclass_log_loss"],
            summary["expected_calibration_error"],
            summary["home_run_ece"],
            summary["argmax_accuracy_secondary"],
            summary["sample_count"],
        )
    logger.info("Recommendation: %s", recommendation)
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
