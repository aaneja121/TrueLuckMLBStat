"""Contact Luck v0.5.1: CORRECTED weather-feature-set comparison.

Version 0.5's `weather_vector_v05_candidate` combined `air_density_kg_m3`
(and its deviation from a fixed reference) with the raw `temperature_c`/
`humidity_pct`/`pressure_hpa` used to derive it. Air density is a
near-deterministic function of those three variables, so including a
derived quantity alongside every variable used to compute it creates severe
multicollinearity: individual linear-model coefficients (and therefore
per-play weather attribution) become unstable and can flip sign even when
the model's AGGREGATE calibration looks fine. This is exactly what Version
0.5's physical-plausibility check found (see README.md "Weather and air
density (Version 0.5)") -- a real, tiny, bootstrap-confirmed log-loss
improvement paired with a weather attribution that correlated the WRONG
direction with air density and following wind.

Version 0.5.1 tests three INDEPENDENT (not cumulative) corrected feature
sets, each using only ONE representation of the temperature/humidity/
pressure/density family:

  - `density_only_v051_candidate`: `air_density_kg_m3` + wind components +
    roof status. NO temperature/humidity/pressure. Tests whether density
    alone is a sufficient physical summary.
  - `components_only_v051_candidate`: temperature + humidity + pressure +
    wind components + roof status. NO derived air density. Tests whether
    the model does better finding its own combination of the raw variables.
  - `density_anomaly_v051_candidate`: each venue's air-density ANOMALY
    (actual density minus that venue's own TRAINING-season-only normal
    density, see `mlb_luck_score.data.join_weather_features.
    add_venue_air_density_anomaly`) + wind components + roof status. NO raw
    density. Separates "was today unusual weather for this specific park"
    from "this park is persistently high/low altitude" -- Coors Field's raw
    density is almost always low, so it mostly just encodes venue identity,
    not day-specific weather.

ADOPTION RULE (`recommend_variant_adoption`): every candidate must pass ALL
of -- (1) 2024 log-loss improvement, (2) a paired bootstrap interval that
supports the improvement, (3-4) no material overall/home-run ECE
regression, (5-6) no material venue/subgroup regression, (7) sufficient
weather coverage, (8) no evident selection bias, (9) CONTROLLED PERTURBATION
checks with the physically correct direction (air-density/anomaly
low-vs-high, strong following-vs-headwind, and -- for the two density
-based candidates -- a Coors Field-specific check), and (10) stable
standardized-condition predictions. These are combined with AND, not OR --
a candidate that improves log loss but fails ANY perturbation check is NOT
recommended, no matter how small or statistically significant the log-loss
delta is. This is a stricter, more literal version of Version 0.5's
"physically plausible effects" criterion: instead of leaving it entirely to
manual notebook inspection, it is now a checkable, automated gate (see
`mlb_luck_score.models.weather_perturbation`) -- though a maintainer should
still read the full detail before actually adopting a candidate.

Usage:

    python -m mlb_luck_score.models.compare_weather_variants \\
        --input data/processed/cleaned_development_data_with_weather.parquet \\
        --output-dir outputs/tables
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
from mlb_luck_score.data.join_weather_features import (
    add_venue_air_density_anomaly,
    compute_venue_air_density_baseline,
)
from mlb_luck_score.features.build_contact_features import (
    WEATHER_COMPONENTS_ONLY_CATEGORICAL_FEATURES,
    WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES,
    WEATHER_DENSITY_ANOMALY_CATEGORICAL_FEATURES,
    WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES,
    WEATHER_DENSITY_ONLY_CATEGORICAL_FEATURES,
    WEATHER_DENSITY_ONLY_NUMERIC_FEATURES,
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
    find_material_venue_regressions,
)
from mlb_luck_score.models.compare_weather_aware import (
    check_standardized_predictions_stable,
    compute_weather_subgroup_calibration,
    find_material_subgroup_regressions,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.models.weather_perturbation import (
    DirectionalCheckResult,
    check_directional_effect,
)

logger = logging.getLogger(__name__)

VARIANT_SELECTED_PRODUCTION_BASELINE = "selected_production_baseline"
VARIANT_DENSITY_ONLY = "density_only_v051_candidate"
VARIANT_COMPONENTS_ONLY = "components_only_v051_candidate"
VARIANT_DENSITY_ANOMALY = "density_anomaly_v051_candidate"

ALL_V051_CANDIDATES: tuple[str, ...] = (
    VARIANT_DENSITY_ONLY,
    VARIANT_COMPONENTS_ONLY,
    VARIANT_DENSITY_ANOMALY,
)

#: Adoption-rule thresholds. Documented Version 0.5.1 research placeholders.
DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN = 0.0
DEFAULT_MIN_WEATHER_COVERAGE = 0.5

#: Controlled-perturbation scenario magnitudes. Chosen to span a physically
#: realistic real-world range (see README.md "Weather and air density
#: (Version 0.5)": real air density across 2021-2024 games ranges roughly
#: 0.95-1.30 kg/m^3) without extrapolating to absurd values. Documented
#: Version 0.5.1 research placeholders, not validated thresholds.
LOW_DENSITY_KG_M3 = 1.05
HIGH_DENSITY_KG_M3 = 1.30
LOW_DENSITY_ANOMALY_KG_M3 = -0.05
HIGH_DENSITY_ANOMALY_KG_M3 = 0.05
STRONG_WIND_SCENARIO_MPS = 5.0

#: Coors Field's venue_id (see `mlb_luck_score.data.park_geometry` /
#: `mlb_luck_score.data.venue_environment`) -- the single venue where a
#: real, well-documented altitude/air-density effect is expected, used for
#: the candidate-specific "sensible effects at Coors" check.
COORS_FIELD_VENUE_ID = 19

_CANDIDATE_FEATURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    VARIANT_DENSITY_ONLY: (
        WEATHER_DENSITY_ONLY_NUMERIC_FEATURES,
        WEATHER_DENSITY_ONLY_CATEGORICAL_FEATURES,
    ),
    VARIANT_COMPONENTS_ONLY: (
        WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES,
        WEATHER_COMPONENTS_ONLY_CATEGORICAL_FEATURES,
    ),
    VARIANT_DENSITY_ANOMALY: (
        WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES,
        WEATHER_DENSITY_ANOMALY_CATEGORICAL_FEATURES,
    ),
}

#: Which column represents "air density" for each candidate's perturbation
#: checks -- `None` for `components_only` (no single density column exists;
#: its density-direction check instead perturbs `pressure_hpa`/`temperature_c`
#: jointly, see `_density_like_scenarios`).
_DENSITY_COLUMN_BY_CANDIDATE: dict[str, str | None] = {
    VARIANT_DENSITY_ONLY: "air_density_kg_m3",
    VARIANT_COMPONENTS_ONLY: None,
    VARIANT_DENSITY_ANOMALY: "air_density_venue_anomaly_kg_m3",
}

_ROOF_NEUTRAL_OVERRIDES: dict[str, Any] = {
    "roof_status": "outdoor_open_air",
    "indoor_indicator": "False",
}
_CALM_WIND_OVERRIDES: dict[str, Any] = {
    "following_wind_mps": 0.0,
    "headwind_mps": 0.0,
    "crosswind_mps": 0.0,
}


def _prepare_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return add_weather_interaction_features(df)


def run_weather_variant_comparison(
    joined_df: pd.DataFrame,
    *,
    min_venue_samples: int = DEFAULT_MIN_VENUE_SAMPLES,
    high_distance_quantile: float = DEFAULT_HIGH_DISTANCE_QUANTILE,
    figures_dir: Path | None = None,
) -> tuple[
    dict[str, dict[str, Any]], dict[str, TrainedModel], dict[str, pd.DataFrame], dict[int, float]
]:
    """Train and evaluate the baseline + all three v0.5.1 candidates on IDENTICAL rows.

    Returns:
        `(comparison, trained_models, proba_by_variant, venue_density_baseline)`.
        `venue_density_baseline` is the fitted (train-only) per-venue air
        -density baseline used by `density_anomaly_v051_candidate` --
        callers doing further analysis (e.g. controlled perturbations) need
        it to build physically meaningful anomaly scenarios.
    """
    required = ("temperature_c", "has_effective_weather", "air_density_kg_m3")
    missing = [c for c in required if c not in joined_df.columns]
    if missing:
        raise ValueError(
            f"joined_df is missing column(s) {missing} -- run "
            "mlb_luck_score.data.join_weather_features.join_weather_features first."
        )

    venue_density_baseline = compute_venue_air_density_baseline(joined_df, TRAIN_SEASONS)
    df = add_venue_air_density_anomaly(joined_df, venue_density_baseline)
    df = _prepare_variant_columns(df)
    df = _prepare_venue_column(df) if "venue_id" in df.columns else df

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin(TRAIN_SEASONS)]
    val_df = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Need non-empty training ({TRAIN_SEASONS}) and validation ({VALIDATION_SEASONS}) "
            "rows to run the weather-variant comparison."
        )
    logger.info(
        "Identical rows for all variants: %d training rows, %d validation rows",
        len(train_df),
        len(val_df),
    )

    variant_kwargs: dict[str, dict[str, Any]] = {VARIANT_SELECTED_PRODUCTION_BASELINE: {}}
    for candidate, (numeric_features, categorical_features) in _CANDIDATE_FEATURES.items():
        variant_kwargs[candidate] = {
            "extra_numeric_features": numeric_features,
            "extra_categorical_features": categorical_features,
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
            from mlb_luck_score.models.compare_park_aware import compute_subgroup_calibration

            summary["complete_case"] = compute_subgroup_calibration(
                y_true, proba_df, complete_case_mask, "complete_case"
            )

        comparison[variant] = summary
        trained_models[variant] = trained
        proba_by_variant[variant] = proba_df

    return comparison, trained_models, proba_by_variant, venue_density_baseline


def _density_like_scenarios(
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str] | None:
    """Return (low_overrides, high_overrides, low_label, high_label) for the density-direction check.

    `None` for `components_only_v051_candidate`, which has no single
    density-like column -- its density-direction check instead perturbs
    temperature and pressure jointly (see `run_perturbation_checks`).
    """
    density_column = _DENSITY_COLUMN_BY_CANDIDATE[candidate]
    if density_column is None:
        return None
    if candidate == VARIANT_DENSITY_ANOMALY:
        low_value, high_value = LOW_DENSITY_ANOMALY_KG_M3, HIGH_DENSITY_ANOMALY_KG_M3
    else:
        low_value, high_value = LOW_DENSITY_KG_M3, HIGH_DENSITY_KG_M3
    low = {density_column: low_value, **_CALM_WIND_OVERRIDES, **_ROOF_NEUTRAL_OVERRIDES}
    high = {density_column: high_value, **_CALM_WIND_OVERRIDES, **_ROOF_NEUTRAL_OVERRIDES}
    return low, high, f"{density_column}={low_value}", f"{density_column}={high_value}"


def run_perturbation_checks(
    trained: TrainedModel, val_df: pd.DataFrame, candidate: str
) -> dict[str, DirectionalCheckResult]:
    """Run every applicable controlled-perturbation directional check for one candidate.

    Args:
        trained: The candidate's fitted model.
        val_df: Validation rows (already through `_prepare_variant_columns`).
        candidate: One of `ALL_V051_CANDIDATES`.

    Returns:
        `{check_name: DirectionalCheckResult}`.
    """
    results: dict[str, DirectionalCheckResult] = {}

    # 1. Density (or anomaly) direction: LOWER density should mean MORE
    # home runs (thinner air carries the ball farther) -- i.e. the "low"
    # scenario's mean home_run probability should be HIGHER than "high"'s,
    # so expect_high_greater=False.
    density_scenarios = _density_like_scenarios(candidate)
    if density_scenarios is not None:
        low_overrides, high_overrides, low_label, high_label = density_scenarios
        results["density_direction"] = check_directional_effect(
            trained,
            val_df,
            low_overrides=low_overrides,
            high_overrides=high_overrides,
            expect_high_greater=False,
            outcome_class="home_run",
            label=f"{candidate}: air density direction",
            low_label=low_label,
            high_label=high_label,
        )
    else:
        # components_only: perturb temperature/pressure jointly in the
        # direction that lowers/raises implied air density (hot+low-pressure
        # = thin air; cold+high-pressure = dense air).
        low_overrides = {
            "temperature_c": 30.0,
            "pressure_hpa": 995.0,
            **_CALM_WIND_OVERRIDES,
            **_ROOF_NEUTRAL_OVERRIDES,
        }
        high_overrides = {
            "temperature_c": 5.0,
            "pressure_hpa": 1030.0,
            **_CALM_WIND_OVERRIDES,
            **_ROOF_NEUTRAL_OVERRIDES,
        }
        results["density_direction"] = check_directional_effect(
            trained,
            val_df,
            low_overrides=low_overrides,
            high_overrides=high_overrides,
            expect_high_greater=False,
            outcome_class="home_run",
            label=f"{candidate}: implied-density direction (temp/pressure)",
            low_label="hot+low-pressure (thin air)",
            high_label="cold+high-pressure (dense air)",
        )

    # 2. Wind direction: a strong FOLLOWING wind should mean MORE home runs
    # than a strong HEADWIND -- expect_high_greater=True (high=following).
    headwind_overrides = {
        "following_wind_mps": -STRONG_WIND_SCENARIO_MPS,
        "headwind_mps": STRONG_WIND_SCENARIO_MPS,
        "crosswind_mps": 0.0,
        **_ROOF_NEUTRAL_OVERRIDES,
    }
    following_overrides = {
        "following_wind_mps": STRONG_WIND_SCENARIO_MPS,
        "headwind_mps": -STRONG_WIND_SCENARIO_MPS,
        "crosswind_mps": 0.0,
        **_ROOF_NEUTRAL_OVERRIDES,
    }
    results["wind_direction"] = check_directional_effect(
        trained,
        val_df,
        low_overrides=headwind_overrides,
        high_overrides=following_overrides,
        expect_high_greater=True,
        outcome_class="home_run",
        label=f"{candidate}: wind direction",
        low_label="strong headwind",
        high_label="strong following wind",
    )

    # 3. Coors Field-specific check (density-based candidates only): among
    # REAL Coors Field rows, the actual (thin-air) scenario should show a
    # HIGHER home-run rate than a hypothetical "Coors with reference/dense
    # air" counterfactual.
    if density_scenarios is not None and "venue_id" in val_df.columns:
        coors_mask = pd.to_numeric(val_df["venue_id"], errors="coerce") == COORS_FIELD_VENUE_ID
        if int(coors_mask.sum()) >= 30:
            coors_df = val_df[coors_mask]
            density_column = _DENSITY_COLUMN_BY_CANDIDATE[candidate]
            assert density_column is not None  # narrowed by density_scenarios check above
            dense_value = (
                HIGH_DENSITY_KG_M3
                if candidate == VARIANT_DENSITY_ONLY
                else HIGH_DENSITY_ANOMALY_KG_M3
            )
            results["coors_field_effect"] = check_directional_effect(
                trained,
                coors_df,
                low_overrides={},  # actual Coors rows, unmodified
                high_overrides={density_column: dense_value},
                expect_high_greater=False,
                outcome_class="home_run",
                label=f"{candidate}: Coors Field actual (thin) vs hypothetical dense air",
                low_label="actual Coors conditions",
                high_label=f"hypothetical dense air ({density_column}={dense_value})",
            )

    return results


def check_standardized_stability_for_candidate(trained: TrainedModel, val_df: pd.DataFrame) -> bool:
    """Predict on genuinely standardized rows with `trained` and validate the output.

    Same lesson as Version 0.5 (see module docstring): must predict on ACTUAL
    standardized-environment rows, not reuse the model's regular
    actual-environment predictions, which would make this check a no-op.
    """
    standardized_val_df = generate_standardized_environment_rows(val_df)
    feature_cols = trained.numeric_features + trained.categorical_features
    standardized_proba = predict_proba_ordered(trained, standardized_val_df[feature_cols])
    return check_standardized_predictions_stable(standardized_proba)


def recommend_variant_adoption(
    comparison: dict[str, dict[str, Any]],
    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]],
    perturbation_by_candidate: dict[str, dict[str, DirectionalCheckResult]],
    standardized_stable_by_candidate: dict[str, bool],
    *,
    candidate_variants: tuple[str, ...] = ALL_V051_CANDIDATES,
    absolute_margin: float = DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
    meaningful_log_loss_regression_margin: float = DEFAULT_MEANINGFUL_LOG_LOSS_REGRESSION_MARGIN,
    min_weather_coverage: float = DEFAULT_MIN_WEATHER_COVERAGE,
) -> dict[str, Any]:
    """Apply the Version 0.5.1 adoption rule to every candidate.

    See module docstring for the full criterion list. All criteria are
    combined with AND -- a tiny, statistically-supported log-loss
    improvement is NECESSARY but never SUFFICIENT; every controlled
    -perturbation directional check must also pass.
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
            selection_bias_ok = not (all_row_ece - cc_ece > absolute_margin)

        perturbation_results = perturbation_by_candidate.get(candidate_name, {})
        perturbation_failures = [
            name for name, result in perturbation_results.items() if not result.passed
        ]
        perturbation_checks_passed = bool(perturbation_results) and not perturbation_failures

        standardized_stable = standardized_stable_by_candidate.get(candidate_name, False)

        passes_all_automated = (
            improves_log_loss
            and bootstrap_supports
            and ece_not_worse
            and hr_ece_not_worse
            and no_venue_regressions
            and no_subgroup_regressions
            and sufficient_coverage
            and selection_bias_ok
            and perturbation_checks_passed
            and standardized_stable
        )

        per_candidate[candidate_name] = {
            "improves_log_loss": improves_log_loss,
            "log_loss_delta": candidate["multiclass_log_loss"] - baseline["multiclass_log_loss"],
            "bootstrap_supports_improvement": bootstrap_supports,
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
            "perturbation_checks_passed": perturbation_checks_passed,
            "perturbation_failures": perturbation_failures,
            "perturbation_detail": {k: v.__dict__ for k, v in perturbation_results.items()},
            "standardized_predictions_stable": standardized_stable,
            "passes_all_automated_criteria": passes_all_automated,
        }

    passing = [
        name for name, result in per_candidate.items() if result["passes_all_automated_criteria"]
    ]
    best_candidate = None
    if passing:
        best_candidate = min(passing, key=lambda name: comparison[name]["multiclass_log_loss"])

    return {
        "per_candidate": per_candidate,
        "recommend_adopt_any_v051_candidate": bool(passing),
        "best_candidate": best_candidate,
        "note": (
            "All criteria (statistical AND controlled-perturbation-based physical-plausibility "
            "checks) are combined with AND -- a candidate is never recommended on log-loss "
            "improvement alone, however small or statistically significant."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with weather features",
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
        comparison, trained_models, proba_by_variant, _venue_baseline = (
            run_weather_variant_comparison(
                df,
                min_venue_samples=args.min_venue_samples,
                high_distance_quantile=args.high_distance_quantile,
                figures_dir=args.figures_dir,
            )
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    training_eligible = df[df["eligible_for_training"].astype(bool)]
    val_df_raw = training_eligible[training_eligible["season"].isin(VALIDATION_SEASONS)]
    venue_baseline = compute_venue_air_density_baseline(df, TRAIN_SEASONS)
    val_df = add_venue_air_density_anomaly(val_df_raw, venue_baseline)
    val_df = _prepare_variant_columns(val_df)
    y_true = val_df["outcome_class"].astype(str)

    bootstrap_by_candidate: dict[str, dict[str, PairedBootstrapResult]] = {}
    perturbation_by_candidate: dict[str, dict[str, DirectionalCheckResult]] = {}
    standardized_stable_by_candidate: dict[str, bool] = {}

    for candidate_name in ALL_V051_CANDIDATES:
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
        logger.info("Running controlled-perturbation checks for %s ...", candidate_name)
        trained = trained_models[candidate_name]
        perturbation_by_candidate[candidate_name] = run_perturbation_checks(
            trained, val_df, candidate_name
        )
        standardized_stable_by_candidate[candidate_name] = (
            check_standardized_stability_for_candidate(trained, val_df)
        )

    recommendation = recommend_variant_adoption(
        comparison,
        bootstrap_by_candidate,
        perturbation_by_candidate,
        standardized_stable_by_candidate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "weather_variants_comparison_detail.json"
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
            "[%s] log_loss=%.6f ece=%.6f home_run_ece=%.6f n=%d",
            variant,
            summary["multiclass_log_loss"],
            summary["expected_calibration_error"],
            summary["home_run_ece"],
            summary["sample_count"],
        )
    for candidate_name, result in recommendation["per_candidate"].items():
        logger.info(
            "[%s] passes_all_automated_criteria=%s perturbation_failures=%s",
            candidate_name,
            result["passes_all_automated_criteria"],
            result["perturbation_failures"],
        )
    logger.info(
        "recommend_adopt_any_v051_candidate=%s best_candidate=%s",
        recommendation["recommend_adopt_any_v051_candidate"],
        recommendation["best_candidate"],
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
