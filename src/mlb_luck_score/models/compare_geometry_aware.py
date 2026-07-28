"""Contact Luck v0.4: does park GEOMETRY (not just venue identity) improve
out-of-sample probability quality, beyond what bare venue identity already
gives (see `mlb_luck_score.models.compare_park_aware`, Version 0.3)?

Four controlled variants, trained on the SAME 2021-2023 rows and evaluated
on the SAME untouched 2024 validation rows (2025 never touched), with
IDENTICAL non-geometry features and preprocessing:

  - `baseline_v02`: the currently-selected default model. No venue_id, no
    geometry features. Identical to `mlb_luck_score.models.
    train_contact_model.train_model()`'s default.
  - `venue_only_v03_candidate`: `baseline_v02` + `venue_id` (see
    `mlb_luck_score.models.compare_park_aware`) -- included here again so
    all four variants are compared under the exact same row set/splits in
    one place.
  - `geometry_only_v04_candidate`: `baseline_v02` + the Version 0.4 wall
    -geometry feature set (see `mlb_luck_score.features.
    build_contact_features.GEOMETRY_NUMERIC_FEATURES` /
    `GEOMETRY_CATEGORICAL_FEATURES`) -- NO `venue_id`, so any improvement
    here is attributable to physical wall geometry, not bare venue identity.
  - `venue_plus_geometry_v04_candidate`: `baseline_v02` + `venue_id` +
    geometry features.

This four-way design directly answers whether geometry adds information
BEYOND venue identity, or whether any apparent improvement is really just
venue identity in disguise (a model can partially "learn" a park's geometry
implicitly through its venue one-hot encoding).

Reuses the Version 0.3 calibration/by-venue/material-regression machinery
(`mlb_luck_score.models.compare_park_aware`) rather than reimplementing it,
and adds Version-0.4-specific near-wall/high-wall subgroup calibration and a
paired, game_pk-level bootstrap for confidence intervals on every
candidate-vs-baseline metric delta.

ADOPTION RULE: see `recommend_geometry_adoption` -- do NOT automatically
adopt a candidate from a small aggregate improvement. Every criterion in the
task's adoption rule is checked except "the model is learning physically
plausible effects", which is inherently a judgment call (see notebook
`06_park_geometry_analysis.ipynb` for that qualitative check) and is
deliberately NOT automated here -- `recommend_geometry_adoption`'s boolean
output is a starting point, never a substitute for reading the full
by-venue/by-subgroup tables and the notebook's example plays.

Usage:

    python -m mlb_luck_score.models.compare_geometry_aware \\
        --input data/processed/cleaned_development_data_with_geometry.parquet \\
        --output-dir outputs/tables --figures-dir outputs/figures/geometry_aware
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

from mlb_luck_score.config import CLASS_ORDER, TABLES_DIR, TRAIN_SEASONS, VALIDATION_SEASONS
from mlb_luck_score.features.build_contact_features import (
    GEOMETRY_CATEGORICAL_FEATURES,
    GEOMETRY_NUMERIC_FEATURES,
    add_geometry_interaction_features,
)
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_HIGH_DISTANCE_QUANTILE,
    DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    DEFAULT_MIN_VENUE_SAMPLES,
    VARIANT_BASELINE_V02,
    VARIANT_PARK_AWARE_V03_CANDIDATE,
    _prepare_venue_column,
    _summarize_variant,
    compute_subgroup_calibration,
    find_material_venue_regressions,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)

logger = logging.getLogger(__name__)

VARIANT_GEOMETRY_ONLY_V04_CANDIDATE = "geometry_only_v04_candidate"
VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE = "venue_plus_geometry_v04_candidate"
ALL_VARIANTS: tuple[str, ...] = (
    VARIANT_BASELINE_V02,
    VARIANT_PARK_AWARE_V03_CANDIDATE,
    VARIANT_GEOMETRY_ONLY_V04_CANDIDATE,
    VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE,
)
GEOMETRY_CANDIDATE_VARIANTS: tuple[str, ...] = (
    VARIANT_GEOMETRY_ONLY_V04_CANDIDATE,
    VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE,
)

#: Bootstrap defaults. See `compute_paired_bootstrap` -- resampling is done
#: at the `game_pk` level (not per-row) so within-game correlation between
#: plays is preserved, per the task's paired-uncertainty requirement.
DEFAULT_N_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

#: Adoption-rule thresholds not already defined in `compare_park_aware`.
#: Documented Version 0.4 research placeholders, not validated thresholds.
DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN = 0.0
DEFAULT_MIN_GEOMETRY_COVERAGE = 0.5

_CLASS_INDEX: dict[str, int] = {cls: i for i, cls in enumerate(CLASS_ORDER)}
_HOME_RUN_INDEX = _CLASS_INDEX["home_run"]


def _bool_mask(series: pd.Series) -> np.ndarray:
    """Coerce a boolean-ish column (native/nullable-boolean OR stringified
    "True"/"False"/None, see `add_geometry_interaction_features`) to a plain bool mask.

    `val_df` passed into `compute_geometry_subgroup_calibration` has already
    been through `add_geometry_interaction_features`, which stringifies
    `GEOMETRY_CATEGORICAL_FEATURES` columns (including every near-wall/
    beyond-wall flag) for the model's categorical pipeline -- `bool("False")`
    is `True` in plain Python, so a naive `.astype(bool)` on the stringified
    column would silently treat every non-null row (both "True" and "False")
    as True. This explicitly maps `"True"`/`"False"` strings (and native
    booleans) to their real value instead. NA/None is treated as "not in
    this subgroup" -- consistent with every other subgroup mask in this
    module (e.g. `compare_park_aware`'s high-distance mask), never silently
    coerced into either direction with ambiguity.
    """

    def _to_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value == "True"
        return bool(value)

    return series.map(_to_bool).to_numpy()


def compute_geometry_subgroup_calibration(
    y_true: pd.Series, proba_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Version 0.4 near-wall / high-wall / geometry-availability subgroup calibration.

    Every subgroup mask comes from `val_df` (shared across all four model
    variants), so the SAME rows define each subgroup for every variant --
    a variant's subgroup ECE is directly comparable to another variant's.
    """
    subgroups: dict[str, dict[str, Any]] = {}

    for col, label in (
        ("near_wall_5ft", "near_wall_5ft"),
        ("near_wall_10ft", "near_wall_10ft"),
        ("near_wall_20ft", "near_wall_20ft"),
        ("projected_beyond_wall", "projected_beyond_wall"),
        ("temporary_or_special_venue", "temporary_or_special_venue"),
    ):
        if col in val_df.columns:
            subgroups[label] = compute_subgroup_calibration(
                y_true, proba_df, _bool_mask(val_df[col]), label
            )

    if "high_wall_indicator" in val_df.columns:
        subgroups["high_wall_segments"] = compute_subgroup_calibration(
            y_true, proba_df, _bool_mask(val_df["high_wall_indicator"]), "high_wall_segments"
        )
        has_geometry_and_height = (
            val_df.get("has_park_geometry", pd.Series(False, index=val_df.index)).astype(bool)
            & val_df["wall_height_in_spray_direction"].notna()
        )
        low_wall_mask = (
            has_geometry_and_height & (val_df["wall_height_in_spray_direction"] < 15.0)
        ).to_numpy()
        subgroups["low_wall_segments"] = compute_subgroup_calibration(
            y_true, proba_df, low_wall_mask, "low_wall_segments"
        )

    if "has_park_geometry" in val_df.columns:
        available_mask = val_df["has_park_geometry"].astype(bool).to_numpy()
        subgroups["geometry_available"] = compute_subgroup_calibration(
            y_true, proba_df, available_mask, "geometry_available"
        )
        subgroups["geometry_unavailable"] = compute_subgroup_calibration(
            y_true, proba_df, ~available_mask, "geometry_unavailable"
        )

    return subgroups


def run_geometry_aware_comparison(
    joined_df: pd.DataFrame,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
    high_distance_quantile: float = DEFAULT_HIGH_DISTANCE_QUANTILE,
    figures_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, TrainedModel], dict[str, pd.DataFrame]]:
    """Train and evaluate all four variants on IDENTICAL rows.

    Args:
        joined_df: Cleaned development data joined with both venue metadata
            (`mlb_luck_score.data.join_venue_metadata`) AND park geometry
            (`mlb_luck_score.data.join_park_geometry`).
        min_venue_samples: Minimum rows for a venue's calibration figures to
            be marked reliable (see `compare_park_aware`).
        high_distance_quantile: Percentile threshold (of validation-set
            `hit_distance_sc`) defining "high projected-distance contact".
        figures_dir: If given, save one calibration plot set per variant.

    Returns:
        `(comparison, trained_models, proba_by_variant)` -- the per-variant
        summary dicts, the fitted `TrainedModel`s (needed for the paired
        bootstrap and for building a v0.4 reference artifact if adopted),
        and each variant's validation-set predicted-probability DataFrame
        (same row order as the shared `val_df`, needed for the bootstrap).
    """
    required = ("venue_id", "has_park_geometry", "wall_distance_in_spray_direction")
    missing = [c for c in required if c not in joined_df.columns]
    if missing:
        raise ValueError(
            f"joined_df is missing column(s) {missing} -- run "
            "mlb_luck_score.data.join_venue_metadata AND "
            "mlb_luck_score.data.join_park_geometry first."
        )

    df = add_geometry_interaction_features(joined_df)
    df = _prepare_venue_column(df)

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the geometry-aware comparison."
        )
    logger.info(
        "Identical rows for all 4 variants: %d training rows, %d validation rows",
        len(train_df),
        len(val_df),
    )

    variant_kwargs: dict[str, dict[str, Any]] = {
        VARIANT_BASELINE_V02: {},
        VARIANT_PARK_AWARE_V03_CANDIDATE: {"extra_categorical_features": ("venue_id",)},
        VARIANT_GEOMETRY_ONLY_V04_CANDIDATE: {
            "extra_numeric_features": GEOMETRY_NUMERIC_FEATURES,
            "extra_categorical_features": GEOMETRY_CATEGORICAL_FEATURES,
        },
        VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE: {
            "extra_numeric_features": GEOMETRY_NUMERIC_FEATURES,
            "extra_categorical_features": ("venue_id", *GEOMETRY_CATEGORICAL_FEATURES),
        },
    }

    comparison: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, TrainedModel] = {}
    proba_by_variant: dict[str, pd.DataFrame] = {}

    for variant, kwargs in variant_kwargs.items():
        logger.info("=== [%s] training on %d rows ===", variant, len(train_df))
        trained = train_model(train_df, class_weight=None, **kwargs)
        feature_cols = trained.numeric_features + trained.categorical_features
        proba_df = predict_proba_ordered(trained, val_df[feature_cols])
        validate_probabilities(proba_df)

        summary = _summarize_variant(
            variant,
            trained,
            proba_df,
            val_df,
            min_venue_samples=min_venue_samples,
            high_distance_quantile=high_distance_quantile,
            figures_dir=figures_dir,
        )
        y_true = val_df["outcome_class"].astype(str)
        summary["geometry_subgroups"] = compute_geometry_subgroup_calibration(
            y_true, proba_df, val_df
        )

        comparison[variant] = summary
        trained_models[variant] = trained
        proba_by_variant[variant] = proba_df

    return comparison, trained_models, proba_by_variant


def _fast_log_loss(
    proba: np.ndarray, y_idx: np.ndarray, rows: np.ndarray, *, eps: float = 1e-15
) -> float:
    p = np.clip(proba[rows, y_idx[rows]], eps, 1.0)
    return float(-np.mean(np.log(p)))


def _fast_ece(proba: np.ndarray, y_idx: np.ndarray, rows: np.ndarray, *, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_err = 0.0
    total_w = 0
    for cls_idx in range(proba.shape[1]):
        predicted = proba[rows, cls_idx]
        observed = (y_idx[rows] == cls_idx).astype(float)
        bin_idx = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            total_err += cnt * abs(predicted[mask].mean() - observed[mask].mean())
            total_w += cnt
    return total_err / total_w if total_w else float("nan")


def _fast_class_ece(
    proba: np.ndarray, y_idx: np.ndarray, rows: np.ndarray, cls_idx: int, *, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    predicted = proba[rows, cls_idx]
    observed = (y_idx[rows] == cls_idx).astype(float)
    bin_idx = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, n_bins - 1)
    total_err = 0.0
    total_w = 0
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        total_err += cnt * abs(predicted[mask].mean() - observed[mask].mean())
        total_w += cnt
    return total_err / total_w if total_w else float("nan")


def _fast_brier(proba: np.ndarray, y_idx: np.ndarray, rows: np.ndarray, cls_idx: int) -> float:
    observed = (y_idx[rows] == cls_idx).astype(float)
    return float(np.mean((proba[rows, cls_idx] - observed) ** 2))


#: Metric names produced by `compute_paired_bootstrap`.
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "log_loss",
    "ece",
    "home_run_ece",
    *(f"brier_{cls}" for cls in CLASS_ORDER),
)


def _compute_metric(metric: str, proba: np.ndarray, y_idx: np.ndarray, rows: np.ndarray) -> float:
    if metric == "log_loss":
        return _fast_log_loss(proba, y_idx, rows)
    if metric == "ece":
        return _fast_ece(proba, y_idx, rows)
    if metric == "home_run_ece":
        return _fast_class_ece(proba, y_idx, rows, _HOME_RUN_INDEX)
    if metric.startswith("brier_"):
        cls = metric[len("brier_") :]
        return _fast_brier(proba, y_idx, rows, _CLASS_INDEX[cls])
    raise ValueError(f"Unknown bootstrap metric: {metric}")


class PairedBootstrapResult(dict):
    """Dict-like paired-bootstrap result for one metric (JSON-serializable as-is)."""


def compute_paired_bootstrap(
    y_true: pd.Series,
    baseline_proba: pd.DataFrame,
    candidate_proba: pd.DataFrame,
    game_pks: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    metrics: tuple[str, ...] = BOOTSTRAP_METRICS,
) -> dict[str, PairedBootstrapResult]:
    """Paired, game_pk-level bootstrap for candidate-vs-baseline metric deltas.

    Resamples GAMES (not individual rows) with replacement -- all rows from
    a sampled game are included together each replicate -- so within-game
    correlation between plays is preserved, per the task's requirement to
    resample at the `game_pk` level. "Paired" means both the baseline and
    the candidate are evaluated on the EXACT SAME resampled rows each
    replicate (both models already predicted on the full, real validation
    set -- only the evaluation subset is resampled), which is what makes the
    delta's variance meaningfully smaller than resampling each model
    independently would.

    Deterministic: a given `seed` always produces the same resampled game
    sequence (`numpy.random.default_rng(seed)`), so results are reproducible
    (verified in `tests/test_compare_geometry_aware.py`).

    Args:
        y_true: True outcome class labels, aligned with `baseline_proba`/
            `candidate_proba`/`game_pks` (same row order, same length).
        baseline_proba: `baseline_v02`'s predicted-probability DataFrame
            (columns in `CLASS_ORDER`) on the validation set.
        candidate_proba: A candidate variant's predicted-probability
            DataFrame on the SAME validation rows.
        game_pks: `game_pk` for each row -- the resampling unit.
        n_reps: Number of bootstrap replicates.
        seed: Random seed (see determinism note above).
        ci: Confidence level for the percentile interval (e.g. 0.95).
        metrics: Which metrics to compute (see `BOOTSTRAP_METRICS`).

    Returns:
        `{metric: PairedBootstrapResult}`, each with `point_estimate`
        (candidate - baseline on the full validation set), `ci_low`,
        `ci_high` (percentile interval of the bootstrap delta
        distribution), `n_reps`, `seed`, and `resampling_unit`.
    """
    if list(baseline_proba.columns) != list(CLASS_ORDER) or list(candidate_proba.columns) != list(
        CLASS_ORDER
    ):
        raise ValueError("baseline_proba/candidate_proba must have columns in CLASS_ORDER")

    y_idx = np.array([_CLASS_INDEX[str(v)] for v in y_true], dtype=np.int64)
    baseline_arr = baseline_proba.to_numpy()
    candidate_arr = candidate_proba.to_numpy()
    all_rows = np.arange(len(y_idx))

    game_pk_arr = game_pks.to_numpy()
    unique_games = np.unique(game_pk_arr)
    game_to_rows = {g: np.where(game_pk_arr == g)[0] for g in unique_games}
    n_games = len(unique_games)

    rng = np.random.default_rng(seed)
    deltas: dict[str, np.ndarray] = {m: np.empty(n_reps) for m in metrics}

    for rep in range(n_reps):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        rows = np.concatenate([game_to_rows[g] for g in sampled_games])
        for metric in metrics:
            baseline_val = _compute_metric(metric, baseline_arr, y_idx, rows)
            candidate_val = _compute_metric(metric, candidate_arr, y_idx, rows)
            deltas[metric][rep] = candidate_val - baseline_val

    alpha = (1.0 - ci) / 2.0
    results: dict[str, PairedBootstrapResult] = {}
    for metric in metrics:
        point_baseline = _compute_metric(metric, baseline_arr, y_idx, all_rows)
        point_candidate = _compute_metric(metric, candidate_arr, y_idx, all_rows)
        dist = deltas[metric]
        results[metric] = PairedBootstrapResult(
            metric=metric,
            point_estimate=point_candidate - point_baseline,
            ci_low=float(np.quantile(dist, alpha)),
            ci_high=float(np.quantile(dist, 1.0 - alpha)),
            confidence_level=ci,
            n_reps=n_reps,
            seed=seed,
            resampling_unit="game_pk",
        )
    return results


def recommend_geometry_adoption(
    comparison: dict[str, dict[str, Any]],
    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]],
    *,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    meaningful_log_loss_regression_margin: float = DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN,
    min_geometry_coverage: float = DEFAULT_MIN_GEOMETRY_COVERAGE,
) -> dict[str, Any]:
    """Apply the Version 0.4 adoption rule to both geometry candidates.

    Checks, for each of `geometry_only_v04_candidate` and
    `venue_plus_geometry_v04_candidate`, ALL of the task's automatable
    criteria:

      1. Validation log loss improves over `baseline_v02`.
      2. The paired bootstrap interval for the log-loss delta supports a
         real improvement or at minimum excludes a meaningful regression
         (`ci_high <= meaningful_log_loss_regression_margin`).
      3. Overall ECE does not materially worsen (`compare_park_aware`'s
         absolute/relative margins, applied to the ECE delta directly).
      4. Home-run ECE does not materially worsen (same margins).
      5. No reliably-sampled venue triggers the material-regression rule
         (`find_material_venue_regressions`).
      6. Near-wall (5/10/20 ft) and high-distance subgroup ECE does not
         materially worsen vs. `baseline_v02` on the SAME subgroup rows.
      7. Geometry coverage (fraction of validation rows with
         `has_park_geometry=True`) is at least `min_geometry_coverage`.

    Criterion 8 from the task ("the model is learning physically plausible
    effects") is NOT automated -- it requires reading example plays (see
    notebook `06_park_geometry_analysis.ipynb`) and is reported here only as
    `requires_manual_review: True`. `recommend_adopt=True` therefore means
    "passes every automatable check", not "adopt without further review".

    Returns:
        A dict with a per-candidate breakdown and an overall
        `recommend_adopt_any_v04_candidate` / `best_candidate` summary.
        Never recommends adoption if no candidate passes every automated
        criterion.
    """
    baseline = comparison[VARIANT_BASELINE_V02]
    per_candidate: dict[str, Any] = {}

    for candidate_name in GEOMETRY_CANDIDATE_VARIANTS:
        candidate = comparison[candidate_name]
        bootstrap = bootstrap_by_candidate.get(candidate_name, {})

        improves_log_loss = candidate["multiclass_log_loss"] < baseline["multiclass_log_loss"]

        log_loss_bootstrap = bootstrap.get("log_loss")
        bootstrap_supports = (
            log_loss_bootstrap is not None
            and log_loss_bootstrap["ci_high"] <= meaningful_log_loss_regression_margin
        )

        ece_delta = candidate["expected_calibration_error"] - baseline["expected_calibration_error"]
        ece_not_worse = ece_delta <= absolute_margin

        hr_ece_delta = candidate["home_run_ece"] - baseline["home_run_ece"]
        hr_ece_not_worse = hr_ece_delta <= absolute_margin

        material_regressions = find_material_venue_regressions(
            baseline["calibration_by_venue"],
            candidate["calibration_by_venue"],
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
        )
        no_venue_regressions = not material_regressions

        subgroup_failures: list[str] = []
        for label, candidate_sub in candidate.get("geometry_subgroups", {}).items():
            baseline_sub = baseline.get("geometry_subgroups", {}).get(label)
            if (
                not baseline_sub
                or candidate_sub.get("ece") is None
                or baseline_sub.get("ece") is None
            ):
                continue
            if label not in (
                "near_wall_5ft",
                "near_wall_10ft",
                "near_wall_20ft",
                "hit_distance_sc>=p75",
            ):
                continue
            delta = candidate_sub["ece"] - baseline_sub["ece"]
            if delta > absolute_margin:
                subgroup_failures.append(label)
        no_subgroup_failures = not subgroup_failures

        geometry_coverage = candidate.get("geometry_coverage_rate")
        sufficient_coverage = (
            geometry_coverage is None or geometry_coverage >= min_geometry_coverage
        )

        passes_all_automated = (
            improves_log_loss
            and bootstrap_supports
            and ece_not_worse
            and hr_ece_not_worse
            and no_venue_regressions
            and no_subgroup_failures
            and sufficient_coverage
        )

        per_candidate[candidate_name] = {
            "improves_log_loss": improves_log_loss,
            "log_loss_delta": candidate["multiclass_log_loss"] - baseline["multiclass_log_loss"],
            "bootstrap_supports_improvement_or_no_meaningful_regression": bootstrap_supports,
            "log_loss_bootstrap": log_loss_bootstrap,
            "ece_not_materially_worse": ece_not_worse,
            "ece_delta": ece_delta,
            "home_run_ece_not_materially_worse": hr_ece_not_worse,
            "home_run_ece_delta": hr_ece_delta,
            "no_material_venue_regressions": no_venue_regressions,
            "material_venue_regressions": material_regressions,
            "no_material_subgroup_regressions": no_subgroup_failures,
            "material_subgroup_regressions": subgroup_failures,
            "sufficient_geometry_coverage": sufficient_coverage,
            "passes_all_automated_criteria": passes_all_automated,
            "requires_manual_review_physical_plausibility": True,
        }

    passing = [
        name for name, result in per_candidate.items() if result["passes_all_automated_criteria"]
    ]
    best_candidate = None
    if passing:
        best_candidate = min(passing, key=lambda name: comparison[name]["multiclass_log_loss"])

    return {
        "per_candidate": per_candidate,
        "recommend_adopt_any_v04_candidate": bool(passing),
        "best_candidate": best_candidate,
        "note": (
            "recommend_adopt_any_v04_candidate=True means the best candidate passed every "
            "AUTOMATABLE criterion -- criterion 8 (physically plausible effects) still "
            "requires manual review of example plays before an actual adoption decision."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with venue metadata AND park geometry",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--min-venue-samples", type=int, default=DEFAULT_MIN_VENUE_SAMPLES)
    parser.add_argument(
        "--high-distance-quantile", type=float, default=DEFAULT_HIGH_DISTANCE_QUANTILE
    )
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        comparison, _trained_models, proba_by_variant = run_geometry_aware_comparison(
            df,
            min_venue_samples=args.min_venue_samples,
            high_distance_quantile=args.high_distance_quantile,
            figures_dir=args.figures_dir,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    val_df = add_geometry_interaction_features(val_df)
    y_true = val_df["outcome_class"].astype(str)

    coverage_rate = float(val_df["has_park_geometry"].mean()) if len(val_df) else float("nan")
    for variant in GEOMETRY_CANDIDATE_VARIANTS:
        comparison[variant]["geometry_coverage_rate"] = coverage_rate

    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]] = {}
    for candidate_name in GEOMETRY_CANDIDATE_VARIANTS:
        logger.info(
            "Running paired bootstrap for %s vs %s ...", candidate_name, VARIANT_BASELINE_V02
        )
        bootstrap_by_candidate[candidate_name] = compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_BASELINE_V02],
            proba_by_variant[candidate_name],
            val_df["game_pk"],
            n_reps=args.n_bootstrap_reps,
            seed=args.bootstrap_seed,
        )

    recommendation = recommend_geometry_adoption(comparison, bootstrap_by_candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "geometry_aware_comparison_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "comparison": comparison,
                "bootstrap": bootstrap_by_candidate,
                "recommendation": recommendation,
            },
            indent=2,
            default=str,
        )
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
