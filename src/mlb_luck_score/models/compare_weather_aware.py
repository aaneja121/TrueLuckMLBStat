"""Contact Luck v0.5: does WEATHER/AIR-DENSITY awareness improve out-of-sample
probability quality beyond the currently-selected production baseline?

As of this writing, `selected_production_baseline` == `baseline_v02`
(`class_weight=None`, no `venue_id`, no park geometry -- see README.md
"Current status": neither `park_aware_v03_candidate` nor either Version 0.4
geometry candidate was adopted). This module does not hardcode that
assumption beyond using `mlb_luck_score.models.train_contact_model.
train_model()`'s current defaults for `selected_production_baseline` --
if a future version's default ever changes, this module picks it up
automatically, exactly like `mlb_luck_score.models.compare_geometry_aware`
already does relative to `mlb_luck_score.models.compare_park_aware`.

Controlled variants, trained on the SAME 2021-2023 rows and evaluated on
the SAME untouched 2024 validation rows (2025 never touched):

  - `selected_production_baseline`: the current default model, unchanged.
  - `weather_basic_v05_candidate`: baseline + temperature/humidity/air
    density/roof-indoor status.
  - `weather_vector_v05_candidate`: baseline + air density (and its
    deviation from a fixed reference atmosphere) + following/head/crosswind
    (relative to each play's own spray direction) + roof-adjusted
    conditions.
  - `geometry_plus_weather_v05_candidate`: baseline + park-geometry
    features (`mlb_luck_score.features.build_contact_features.
    GEOMETRY_NUMERIC_FEATURES`/`GEOMETRY_CATEGORICAL_FEATURES`) + the
    weather-vector feature set -- included ONLY when the input data has
    park-geometry columns (see `run_weather_aware_comparison`'s
    `include_geometry` parameter); geometry is a documented, NOT adopted,
    Version 0.4 candidate (see `mlb_luck_score.models.
    compare_geometry_aware`), so this variant answers "if geometry were
    combined with weather, would the combination look different from
    either alone?", not "should geometry be adopted".

Reuses the Version 0.3/0.4 calibration/by-venue/material-regression/paired
-bootstrap machinery (`mlb_luck_score.models.compare_park_aware`,
`mlb_luck_score.models.compare_geometry_aware`) rather than reimplementing
it, and adds Version-0.5-specific roof-status/weather-quality/air-density/
wind-strength subgroup calibration plus complete-case vs. all-row
(missingness-aware) reporting.

ADOPTION RULE: see `recommend_weather_adoption` -- checks 8 of the task's
10 adoption criteria automatically; "learned effects are physically
plausible" is NOT automated (see notebook `07_weather_air_density_analysis.
ipynb` for that qualitative check).

Usage:

    python -m mlb_luck_score.models.compare_weather_aware \\
        --input data/processed/cleaned_development_data_with_weather.parquet \\
        --output-dir outputs/tables --figures-dir outputs/figures/weather_aware
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_luck_score.config import TABLES_DIR, TRAIN_SEASONS, VALIDATION_SEASONS
from mlb_luck_score.features.build_contact_features import (
    GEOMETRY_CATEGORICAL_FEATURES,
    GEOMETRY_NUMERIC_FEATURES,
    WEATHER_BASIC_CATEGORICAL_FEATURES,
    WEATHER_BASIC_NUMERIC_FEATURES,
    WEATHER_VECTOR_CATEGORICAL_FEATURES,
    WEATHER_VECTOR_NUMERIC_FEATURES,
    add_geometry_interaction_features,
    add_weather_interaction_features,
    generate_standardized_environment_rows,
)
from mlb_luck_score.models.compare_geometry_aware import (
    BOOTSTRAP_METRICS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    PairedBootstrapResult,
    _bool_mask,
    compute_paired_bootstrap,
)
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_HIGH_DISTANCE_QUANTILE,
    DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    DEFAULT_MIN_VENUE_SAMPLES,
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

VARIANT_SELECTED_PRODUCTION_BASELINE = "selected_production_baseline"
VARIANT_WEATHER_BASIC_V05_CANDIDATE = "weather_basic_v05_candidate"
VARIANT_WEATHER_VECTOR_V05_CANDIDATE = "weather_vector_v05_candidate"
VARIANT_GEOMETRY_PLUS_WEATHER_V05_CANDIDATE = "geometry_plus_weather_v05_candidate"

WEATHER_CANDIDATE_VARIANTS_NO_GEOMETRY: tuple[str, ...] = (
    VARIANT_WEATHER_BASIC_V05_CANDIDATE,
    VARIANT_WEATHER_VECTOR_V05_CANDIDATE,
)

#: Adoption-rule thresholds not already defined in `compare_park_aware`.
DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN = 0.0
DEFAULT_MIN_WEATHER_COVERAGE = 0.5
#: Wind-strength / air-density subgroup thresholds. Documented Version 0.5
#: research placeholders, not validated thresholds.
STRONG_WIND_THRESHOLD_MPS = 4.5  # ~10 mph
HIGH_AIR_DENSITY_QUANTILE = 0.75
LOW_AIR_DENSITY_QUANTILE = 0.25

_NOTABLE_VENUES: dict[int, str] = {
    19: "Coors Field",
    3: "Fenway Park",
    17: "Wrigley Field",
    2395: "Oracle Park",
}


def _prepare_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_weather_interaction_features(df)
    if "wall_distance_in_spray_direction" in out.columns:
        out = add_geometry_interaction_features(out)
    return out


def compute_weather_subgroup_calibration(
    y_true: pd.Series, proba_df: pd.DataFrame, val_df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Version 0.5 roof/weather-quality/air-density/wind-strength subgroup calibration.

    Every subgroup mask comes from `val_df` (shared across all variants),
    so a variant's subgroup ECE is directly comparable to another's.
    """
    subgroups: dict[str, dict[str, Any]] = {}

    if "roof_status" in val_df.columns:
        for status in (
            "outdoor_open_air",
            "retractable_roof_open",
            "retractable_roof_closed",
            "fixed_indoor",
            "roof_status_unknown",
        ):
            mask = (val_df["roof_status"].astype(str) == status).to_numpy()
            subgroups[f"roof_{status}"] = compute_subgroup_calibration(
                y_true, proba_df, mask, f"roof_{status}"
            )

    if "weather_match_quality" in val_df.columns:
        for quality in ("good", "fair"):
            mask = (val_df["weather_match_quality"].astype(str) == quality).to_numpy()
            subgroups[f"weather_quality_{quality}"] = compute_subgroup_calibration(
                y_true, proba_df, mask, f"weather_quality_{quality}"
            )

    if "has_effective_weather" in val_df.columns:
        available_mask = _bool_mask(val_df["has_effective_weather"])
        subgroups["complete_case_weather_available"] = compute_subgroup_calibration(
            y_true, proba_df, available_mask, "complete_case_weather_available"
        )
        subgroups["weather_unavailable"] = compute_subgroup_calibration(
            y_true, proba_df, ~available_mask, "weather_unavailable"
        )

    if "temperature_c" in val_df.columns and val_df["temperature_c"].notna().any():
        temp = val_df["temperature_c"]
        subgroups["high_temperature"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (temp >= temp.quantile(0.75)).fillna(False).to_numpy(),
            "high_temperature",
        )
        subgroups["low_temperature"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (temp <= temp.quantile(0.25)).fillna(False).to_numpy(),
            "low_temperature",
        )

    if "air_density_kg_m3" in val_df.columns and val_df["air_density_kg_m3"].notna().any():
        density = val_df["air_density_kg_m3"]
        subgroups["high_air_density"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (density >= density.quantile(HIGH_AIR_DENSITY_QUANTILE)).fillna(False).to_numpy(),
            "high_air_density",
        )
        subgroups["low_air_density"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (density <= density.quantile(LOW_AIR_DENSITY_QUANTILE)).fillna(False).to_numpy(),
            "low_air_density",
        )

    if "following_wind_mps" in val_df.columns:
        following = val_df["following_wind_mps"]
        subgroups["strong_following_wind"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (following >= STRONG_WIND_THRESHOLD_MPS).fillna(False).to_numpy(),
            "strong_following_wind",
        )
    if "headwind_mps" in val_df.columns:
        headwind = val_df["headwind_mps"]
        subgroups["strong_headwind"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (headwind >= STRONG_WIND_THRESHOLD_MPS).fillna(False).to_numpy(),
            "strong_headwind",
        )
    if "crosswind_mps" in val_df.columns:
        crosswind = val_df["crosswind_mps"].abs()
        subgroups["strong_crosswind"] = compute_subgroup_calibration(
            y_true,
            proba_df,
            (crosswind >= STRONG_WIND_THRESHOLD_MPS).fillna(False).to_numpy(),
            "strong_crosswind",
        )

    if "venue_id" in val_df.columns:
        venue_ids = pd.to_numeric(val_df["venue_id"], errors="coerce")
        for venue_id, name in _NOTABLE_VENUES.items():
            mask = (venue_ids == venue_id).to_numpy()
            subgroups[f"venue_{name}"] = compute_subgroup_calibration(y_true, proba_df, mask, name)

    if "near_wall_10ft" in val_df.columns:
        subgroups["near_wall_10ft"] = compute_subgroup_calibration(
            y_true, proba_df, _bool_mask(val_df["near_wall_10ft"]), "near_wall_10ft"
        )

    return subgroups


def run_weather_aware_comparison(
    joined_df: pd.DataFrame,
    *,
    include_geometry: bool | None = None,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
    high_distance_quantile: float = DEFAULT_HIGH_DISTANCE_QUANTILE,
    figures_dir: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, TrainedModel], dict[str, pd.DataFrame]]:
    """Train and evaluate all applicable variants on IDENTICAL rows.

    Args:
        joined_df: Cleaned development data joined with weather features
            (`mlb_luck_score.data.join_weather_features`), optionally also
            joined with park geometry (`mlb_luck_score.data.
            join_park_geometry`).
        include_geometry: Whether to include
            `geometry_plus_weather_v05_candidate`. Defaults to
            auto-detecting whether `joined_df` has geometry columns.
        min_venue_samples: Minimum rows for a venue's calibration figures to
            be marked reliable.
        high_distance_quantile: Percentile threshold defining "high
            projected-distance contact".
        figures_dir: If given, save one calibration plot set per variant.

    Returns:
        `(comparison, trained_models, proba_by_variant)`, keyed by variant
        name. `comparison[variant]` additionally has a `"complete_case"` key
        with the SAME model's metrics restricted to rows with
        `has_effective_weather=True` (see module docstring "complete-case
        vs. all-row" reporting).
    """
    required = ("temperature_c", "has_effective_weather")
    missing = [c for c in required if c not in joined_df.columns]
    if missing:
        raise ValueError(
            f"joined_df is missing column(s) {missing} -- run "
            "mlb_luck_score.data.join_weather_features.join_weather_features first."
        )

    has_geometry = "wall_distance_in_spray_direction" in joined_df.columns
    if include_geometry is None:
        include_geometry = has_geometry
    if include_geometry and not has_geometry:
        raise ValueError(
            "include_geometry=True but joined_df has no park-geometry columns -- run "
            "mlb_luck_score.data.join_park_geometry.join_park_geometry first."
        )

    df = _prepare_weather_columns(joined_df)
    df = _prepare_venue_column(df) if "venue_id" in df.columns else df

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the weather-aware comparison."
        )
    logger.info(
        "Identical rows for all variants: %d training rows, %d validation rows",
        len(train_df),
        len(val_df),
    )

    variant_kwargs: dict[str, dict[str, Any]] = {
        VARIANT_SELECTED_PRODUCTION_BASELINE: {},
        VARIANT_WEATHER_BASIC_V05_CANDIDATE: {
            "extra_numeric_features": WEATHER_BASIC_NUMERIC_FEATURES,
            "extra_categorical_features": WEATHER_BASIC_CATEGORICAL_FEATURES,
        },
        VARIANT_WEATHER_VECTOR_V05_CANDIDATE: {
            "extra_numeric_features": WEATHER_VECTOR_NUMERIC_FEATURES,
            "extra_categorical_features": WEATHER_VECTOR_CATEGORICAL_FEATURES,
        },
    }
    if include_geometry:
        variant_kwargs[VARIANT_GEOMETRY_PLUS_WEATHER_V05_CANDIDATE] = {
            "extra_numeric_features": (
                *WEATHER_VECTOR_NUMERIC_FEATURES,
                *GEOMETRY_NUMERIC_FEATURES,
            ),
            "extra_categorical_features": (
                *WEATHER_VECTOR_CATEGORICAL_FEATURES,
                *GEOMETRY_CATEGORICAL_FEATURES,
            ),
        }

    complete_case_mask = _bool_mask(val_df["has_effective_weather"])

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
        summary["weather_subgroups"] = compute_weather_subgroup_calibration(
            y_true, proba_df, val_df
        )
        summary["weather_coverage_rate"] = float(complete_case_mask.mean())

        if complete_case_mask.any():
            cc_summary = compute_subgroup_calibration(
                y_true, proba_df, complete_case_mask, "complete_case"
            )
            summary["complete_case"] = cc_summary

        comparison[variant] = summary
        trained_models[variant] = trained
        proba_by_variant[variant] = proba_df

    return comparison, trained_models, proba_by_variant


def find_material_subgroup_regressions(
    baseline_subgroups: dict[str, dict[str, Any]],
    candidate_subgroups: dict[str, dict[str, Any]],
    *,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    min_sample_size: int = 30,
) -> list[str]:
    """Subgroup labels where candidate ECE worsens materially vs. baseline.

    Same scale-sensitive absolute-OR-relative rule as `mlb_luck_score.
    models.compare_park_aware.find_material_venue_regressions`, applied to
    arbitrary named subgroups (roof status, weather quality, wind strength,
    air density, notable venues) instead of venues specifically. A subgroup
    is only eligible to be flagged if it has at least `min_sample_size`
    baseline rows.
    """
    flagged: list[str] = []
    for label, baseline_sub in baseline_subgroups.items():
        candidate_sub = candidate_subgroups.get(label)
        if not candidate_sub:
            continue
        baseline_ece = baseline_sub.get("ece")
        candidate_ece = candidate_sub.get("ece")
        n = baseline_sub.get("sample_count", 0)
        if baseline_ece is None or candidate_ece is None or n < min_sample_size:
            continue
        delta = candidate_ece - baseline_ece
        relative = delta / baseline_ece if baseline_ece else float("inf") if delta > 0 else 0.0
        if delta > absolute_margin or relative > relative_margin:
            flagged.append(label)
    return flagged


def check_standardized_predictions_stable(proba_df: pd.DataFrame) -> bool:
    """Sanity-check that standardized-environment counterfactual predictions are well-formed.

    A basic, automatable stability check (criterion 10 of the task's
    adoption rule): every predicted probability row must be finite and sum
    to ~1. Does NOT check whether the counterfactual SHIFT is itself
    "reasonable" in magnitude -- that requires human judgment (see notebook
    `07_weather_air_density_analysis.ipynb`, "actual vs. standardized
    environment examples").
    """
    try:
        validate_probabilities(proba_df)
    except ValueError:
        return False
    return True


def recommend_weather_adoption(
    comparison: dict[str, dict[str, Any]],
    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]],
    *,
    candidate_variants: tuple[str, ...] = WEATHER_CANDIDATE_VARIANTS_NO_GEOMETRY,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    meaningful_log_loss_regression_margin: float = DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN,
    min_weather_coverage: float = DEFAULT_MIN_WEATHER_COVERAGE,
    standardized_predictions_stable: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Apply the Version 0.5 adoption rule to every weather candidate.

    Checks, for each candidate, ALL of the task's automatable criteria:

      1. Validation log loss improves over `selected_production_baseline`.
      2. The paired bootstrap interval for the log-loss delta supports a
         real improvement or at minimum excludes a meaningful regression.
      3. Overall ECE does not materially worsen.
      4. Home-run ECE does not materially worsen.
      5. No reliably-sampled venue triggers the material-regression rule.
      6. No roof/weather-quality/wind/air-density subgroup triggers the
         material-regression rule (`find_material_subgroup_regressions`).
      7. Weather coverage is at least `min_weather_coverage`, transparently
         reported (`weather_coverage_rate`).
      8. Missing weather does not create selection bias -- approximated by
         checking the complete-case log loss/ECE are not materially better
         than the all-row log loss/ECE for the SAME model (if restricting
         to rows with weather made the model look meaningfully better, that
         is evidence the missing-weather rows are systematically different,
         i.e. potential selection bias).
      10. Standardized-environment counterfactual predictions are
          well-formed (`standardized_predictions_stable`, if provided --
          see `check_standardized_predictions_stable`).

    Criterion 9 ("learned effects are physically plausible") is NOT
    automated -- reported here only as `requires_manual_review: True`.
    `recommend_adopt=True` therefore means "passes every automatable
    check", not "adopt without further review".
    """
    baseline = comparison[VARIANT_SELECTED_PRODUCTION_BASELINE]
    per_candidate: dict[str, Any] = {}

    for candidate_name in candidate_variants:
        if candidate_name not in comparison:
            continue
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

        material_venue_regressions = find_material_venue_regressions(
            baseline["calibration_by_venue"],
            candidate["calibration_by_venue"],
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
        )
        no_venue_regressions = not material_venue_regressions

        material_subgroup_regressions = find_material_subgroup_regressions(
            baseline.get("weather_subgroups", {}),
            candidate.get("weather_subgroups", {}),
            absolute_margin=absolute_margin,
            relative_margin=relative_margin,
        )
        no_subgroup_regressions = not material_subgroup_regressions

        coverage = candidate.get("weather_coverage_rate")
        sufficient_coverage = coverage is None or coverage >= min_weather_coverage

        selection_bias_ok = True
        if "complete_case" in candidate and candidate["complete_case"].get("ece") is not None:
            cc_ece = candidate["complete_case"]["ece"]
            all_row_ece = candidate["expected_calibration_error"]
            # A complete-case ECE meaningfully BETTER than all-row ECE suggests rows with
            # weather available are systematically easier/different, not that the model
            # itself is worse -- flag rather than silently ignore.
            selection_bias_ok = not (all_row_ece - cc_ece > absolute_margin)

        stable = True
        if standardized_predictions_stable is not None:
            stable = standardized_predictions_stable.get(candidate_name, True)

        passes_all_automated = (
            improves_log_loss
            and bootstrap_supports
            and ece_not_worse
            and hr_ece_not_worse
            and no_venue_regressions
            and no_subgroup_regressions
            and sufficient_coverage
            and selection_bias_ok
            and stable
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
            "material_venue_regressions": material_venue_regressions,
            "no_material_subgroup_regressions": no_subgroup_regressions,
            "material_subgroup_regressions": material_subgroup_regressions,
            "sufficient_weather_coverage": sufficient_coverage,
            "weather_coverage_rate": coverage,
            "no_evident_selection_bias": selection_bias_ok,
            "standardized_predictions_stable": stable,
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
        "recommend_adopt_any_v05_candidate": bool(passing),
        "best_candidate": best_candidate,
        "note": (
            "recommend_adopt_any_v05_candidate=True means the best candidate passed every "
            "AUTOMATABLE criterion -- criterion 9 (physically plausible effects) still "
            "requires manual review of example plays before an actual adoption decision."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with weather features (optionally also geometry)",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--min-venue-samples", type=int, default=DEFAULT_MIN_VENUE_SAMPLES)
    parser.add_argument(
        "--high-distance-quantile", type=float, default=DEFAULT_HIGH_DISTANCE_QUANTILE
    )
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--include-geometry",
        action="store_true",
        default=None,
        help="Force-include geometry_plus_weather_v05_candidate (auto-detected by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        comparison, trained_models, proba_by_variant = run_weather_aware_comparison(
            df,
            include_geometry=args.include_geometry,
            min_venue_samples=args.min_venue_samples,
            high_distance_quantile=args.high_distance_quantile,
            figures_dir=args.figures_dir,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    candidate_variants = tuple(v for v in comparison if v != VARIANT_SELECTED_PRODUCTION_BASELINE)

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    val_df = _prepare_weather_columns(val_df)
    y_true = val_df["outcome_class"].astype(str)
    standardized_val_df = generate_standardized_environment_rows(val_df)

    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]] = {}
    standardized_stable: dict[str, bool] = {}
    for candidate_name in candidate_variants:
        logger.info(
            "Running paired bootstrap for %s vs %s ...",
            candidate_name,
            VARIANT_SELECTED_PRODUCTION_BASELINE,
        )
        bootstrap_by_candidate[candidate_name] = compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_SELECTED_PRODUCTION_BASELINE],
            proba_by_variant[candidate_name],
            val_df["game_pk"],
            n_reps=args.n_bootstrap_reps,
            seed=args.bootstrap_seed,
            metrics=BOOTSTRAP_METRICS,
        )
        # Criterion 10 checks GENUINE standardized-environment predictions (not
        # the actual-environment ones already validated inside
        # run_weather_aware_comparison) -- predict on the standardized rows
        # with this candidate's own trained model.
        candidate_trained = trained_models[candidate_name]
        candidate_feature_cols = (
            candidate_trained.numeric_features + candidate_trained.categorical_features
        )
        standardized_proba = predict_proba_ordered(
            candidate_trained, standardized_val_df[candidate_feature_cols]
        )
        standardized_stable[candidate_name] = check_standardized_predictions_stable(
            standardized_proba
        )

    recommendation = recommend_weather_adoption(
        comparison,
        bootstrap_by_candidate,
        candidate_variants=candidate_variants,
        standardized_predictions_stable=standardized_stable,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "weather_aware_comparison_detail.json"
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
            "[%s] log_loss=%.6f ece=%.6f home_run_ece=%.6f accuracy(secondary)=%.4f n=%d "
            "weather_coverage=%.4f",
            variant,
            summary["multiclass_log_loss"],
            summary["expected_calibration_error"],
            summary["home_run_ece"],
            summary["argmax_accuracy_secondary"],
            summary["sample_count"],
            summary.get("weather_coverage_rate", float("nan")),
        )
    logger.info("Recommendation: %s", recommendation)
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
