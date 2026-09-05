"""Contact Forecast: run the HistGradientBoosting experiment and write its artifacts.

One nonlinear family, two prespecified formulations, one evaluation on 2024.
2025 stays sealed and 2026 stays behind the Phase 2 gate. No prediction
intervals and no dashboard surface are built here.

## Order of operations, which is itself the anti-overfitting control

1. Load the FROZEN R1 prediction table; verify the R1 and ridge freeze chains.
2. Verify from the frozen R1 artifact that the residual baseline is causal.
3. Write `hgb_candidate_grid.json` -- the complete grid, before any evaluation.
4. Fold 1: fit every candidate on 2022, score on 2023, select.
5. Write `hgb_specification.json` -- formulation, hyperparameters, feature list,
   preprocessing, loss and seed, all frozen, BEFORE 2024 is touched.
6. Fold 2: fit on 2022-2023 at the frozen specification, predict 2024 once.
7. Compare against league mean, shrunk realized, shrunk deserved and the frozen
   full-profile ridge.

2024 chooses nothing: not a hyperparameter, not a feature, not the formulation.
The direct/residual designation was fixed in `forecast.hgb` before either 2024
number existed, and is forced by data availability rather than by a result.

## The preference rule is prespecified

`NONLINEAR_VALUE_RULE` is declared before the comparison runs: HGB is preferred
over the frozen ridge only when its delta against ridge is negative AND the
paired batter-bootstrap interval for that delta excludes zero. A numerically
lower point estimate alone is not sufficient -- ridge wins ties on simplicity
and interpretability.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_OUTPUTS_DIR,
    PRIMARY_HORIZON,
    RANDOM_SEED,
    SEASON_FORWARD_FOLDS,
    assert_path_outside_forbidden_namespaces,
    conditional_forecast_language,
)
from forecast.hgb import (
    HGB_FEATURE_SET_NAME,
    HGB_SELECTION_METRIC,
    HGB_TARGET,
    PRIMARY_FORMULATION,
    RESIDUAL_BASELINE,
    RESIDUAL_TARGET,
    SENSITIVITY_FORMULATION,
    assert_no_later_season_rows,
    assert_residual_baseline_is_causal,
    build_residual_target,
    fit_hgb,
    grid_record,
    hgb_features,
    predict,
    select_hyperparameters,
)
from forecast.metrics import (
    DELTA_SIGN_CONVENTION,
    assert_delta_convention,
    batter_balanced_errors,
    evaluate_predictor,
    paired_delta,
)
from forecast.ridge import fit_ridge as fit_ridge_model
from forecast.ridge import predict as ridge_predict
from forecast.run_ridge_forecast import load_focal_table

logger = logging.getLogger(__name__)

FROZEN_RIDGE_MODEL = "D_full_contact_profile"
RIDGE_PREDICTION_COLUMN = "ridge_D_full_contact_profile"
HGB_DIRECT_COLUMN = "hgb_direct"
HGB_RESIDUAL_COLUMN = "hgb_residual"

#: Benchmarks the selected HGB candidate is measured against, in report order.
BENCHMARKS: tuple[str, ...] = (
    "league_mean",
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
    RIDGE_PREDICTION_COLUMN,
)

#: Declared BEFORE the comparison ran.
NONLINEAR_VALUE_RULE: dict[str, Any] = {
    "question": (
        "Does nonlinear modeling extract information from the contact profile that the "
        "full-profile ridge failed to capture?"
    ),
    "rule": (
        "HGB is preferred over the frozen ridge only if delta_MAE(HGB - ridge) is "
        "negative AND its paired batter-bootstrap interval excludes zero. Otherwise "
        "ridge is preferred on simplicity and interpretability grounds."
    ),
    "explicitly_insufficient": (
        "a numerically lower point estimate on its own is NOT sufficient to call HGB superior"
    ),
    "declared_before_the_comparison_ran": True,
}


class ForecastHGBRunError(RuntimeError):
    """Raised when the HGB experiment cannot run as specified."""


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")


def _full_metrics(frame: pd.DataFrame, *, predictor: str) -> dict[str, Any]:
    measured = evaluate_predictor(frame, target=HGB_TARGET, predictor=predictor)
    balanced_mae, balanced_rmse, n_batters = batter_balanced_errors(
        frame, target=HGB_TARGET, predictor=predictor
    )
    record = measured.as_record()
    record.update(
        {
            "r_squared": _r_squared(
                frame[HGB_TARGET].to_numpy(dtype=float),
                frame[predictor].to_numpy(dtype=float),
            ),
            "batter_balanced_mae": balanced_mae,
            "batter_balanced_rmse": balanced_rmse,
            "n_unique_batters": n_batters,
        }
    )
    return record


def _verify_ridge_freeze(outputs_dir: Path) -> dict[str, Any]:
    """Confirm the ridge stage is sealed and pull the frozen alpha and MAE."""
    manifest_path = outputs_dir / "ridge_freeze_manifest.json"
    if not manifest_path.exists():
        raise ForecastHGBRunError(
            f"No ridge freeze manifest at {manifest_path}. The ridge stage must be "
            "sealed before a nonlinear model is compared against it."
        )
    frozen = json.loads(manifest_path.read_text())
    key = frozen["key_results"]
    return {
        "ridge_freeze_manifest_sha256": frozen["manifest_sha256"],
        "ridge_verdict": frozen["conclusion"]["verdict"],
        "frozen_alpha": float(key["selected_alphas"][FROZEN_RIDGE_MODEL]),
        "frozen_ridge_evaluation_mae": float(key["evaluation_mae_by_model"][FROZEN_RIDGE_MODEL]),
        "frozen_ridge_delta_vs_shrunk_deserved": key["focal_comparison"]["delta_mae"],
    }


def _reproduce_frozen_ridge(
    train: pd.DataFrame, evaluate: pd.DataFrame, *, ridge_provenance: dict[str, Any]
) -> np.ndarray:
    """Refit the frozen full-profile ridge and prove it reproduces its frozen MAE.

    The ridge artifact stores metrics, not per-window predictions, so the model
    is refit from the same frozen table with the same frozen alpha and the same
    frozen code. The reproduction check is what makes that legitimate: if the
    refit does not land on the sealed MAE, the comparison is not against the
    model that was frozen and the run stops.
    """
    fit = fit_ridge_model(
        train, model_name=FROZEN_RIDGE_MODEL, alpha=ridge_provenance["frozen_alpha"]
    )
    predictions = ridge_predict(fit, evaluate)
    reproduced = float(np.mean(np.abs(predictions - evaluate[HGB_TARGET].to_numpy(dtype=float))))
    expected = ridge_provenance.get("frozen_ridge_evaluation_mae")
    ridge_provenance["reproduced_ridge_evaluation_mae"] = reproduced
    if expected is None:
        # Freeze checking was skipped (development only), so there is no sealed
        # value to reproduce. Recorded as unverified rather than silently
        # treated as a passing check.
        ridge_provenance["reproduction_matches_freeze"] = None
        ridge_provenance["reproduction_note"] = (
            "freeze verification skipped; the refit was NOT checked against a sealed MAE"
        )
        return predictions
    if abs(reproduced - expected) > 1e-9:
        raise ForecastHGBRunError(
            f"Refitting the frozen ridge gave MAE {reproduced!r} but the ridge freeze "
            f"records {expected!r}. The comparison would not be against the frozen model."
        )
    ridge_provenance["reproduction_matches_freeze"] = True
    return predictions


def _residual_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Whether the contact-profile correction itself contains learnable signal."""
    actual = frame[RESIDUAL_TARGET].to_numpy(dtype=float)
    predicted = frame["predicted_residual"].to_numpy(dtype=float)
    mae_zero = float(np.mean(np.abs(actual)))
    mae_model = float(np.mean(np.abs(actual - predicted)))
    if float(np.std(predicted)) == 0.0:
        correlation: float | None = None
        correlation_status = "undefined_constant_prediction"
    else:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        correlation_status = "defined"
    return {
        "residual_definition": f"{HGB_TARGET} - {RESIDUAL_BASELINE}",
        "n": int(len(frame)),
        "residual_target_sd": float(np.std(actual, ddof=1)),
        "residual_target_mean": float(np.mean(actual)),
        "mae_predicting_residual_zero": mae_zero,
        "mae_hgb_predicted_residual": mae_model,
        "correlation_predicted_vs_actual_residual": correlation,
        "correlation_status": correlation_status,
        "fraction_of_benchmark_error_recovered": (
            float((mae_zero - mae_model) / mae_zero) if mae_zero > 0 else None
        ),
        "note": (
            "MAE of predicting residual = 0 is exactly the shrunk-deserved benchmark's "
            "own MAE, by construction. A fraction recovered at or below zero means the "
            "correction contains no learnable signal on this season."
        ),
    }


def _stability(frame: pd.DataFrame, *, challenger: str, reference: str) -> dict[str, Any]:
    """Is the advantage broad, or carried by a handful of hitters?"""
    y = frame[HGB_TARGET].to_numpy(dtype=float)
    reference_error = np.abs(frame[reference].to_numpy(dtype=float) - y)
    challenger_error = np.abs(frame[challenger].to_numpy(dtype=float) - y)
    # Positive => the challenger was closer for that hitter.
    improvement = reference_error - challenger_error
    order = np.argsort(improvement)

    trimmed: dict[str, float] = {}
    for k in (1, 5, 10, 25):
        if len(improvement) > 2 * k:
            trimmed[f"mean_excluding_top_{k}_most_improved"] = float(
                np.mean(np.sort(improvement)[:-k])
            )
    return {
        "quantity": f"absolute_error({reference}) - absolute_error({challenger})",
        "sign_meaning": "positive means the challenger was closer for that hitter",
        "n_hitters": int(len(improvement)),
        "mean": float(np.mean(improvement)),
        "median": float(np.median(improvement)),
        "q1": float(np.percentile(improvement, 25)),
        "q3": float(np.percentile(improvement, 75)),
        "p5": float(np.percentile(improvement, 5)),
        "p95": float(np.percentile(improvement, 95)),
        "min": float(np.min(improvement)),
        "max": float(np.max(improvement)),
        "fraction_of_hitters_improved": float(np.mean(improvement > 0)),
        "top_5_most_improved": [float(v) for v in improvement[order][-5:][::-1]],
        "bottom_5_most_worsened": [float(v) for v in improvement[order][:5]],
        "trimmed_means": trimmed,
        "reading": (
            "if the mean collapses toward zero once a handful of hitters are excluded, "
            "the average advantage is carried by extremes rather than being broad"
        ),
    }


def _permutation_importance(
    fit: Any, validate: pd.DataFrame, *, target_column: str
) -> dict[str, Any]:
    """EXPLORATORY only, computed on the validation season, never used to revise.

    Run on the fold-1 model against the 2023 validation season -- the only
    structure permitted for a diagnostic. It is not computed on 2024, and no
    model decision anywhere in this stage consults it.
    """
    result = permutation_importance(
        fit.model,
        validate[fit.features].to_numpy(dtype=float),
        pd.to_numeric(validate[target_column], errors="coerce").to_numpy(dtype=float),
        n_repeats=20,
        random_state=RANDOM_SEED,
        scoring="neg_mean_absolute_error",
    )
    ranked = sorted(
        (
            {
                "feature": name,
                "mean_importance": float(mean),
                "sd": float(sd),
            }
            for name, mean, sd in zip(
                fit.features, result.importances_mean, result.importances_std, strict=True
            )
        ),
        key=lambda row: -row["mean_importance"],
    )
    return {
        "status": "EXPLORATORY -- not causal, not a feature-selection procedure",
        "computed_on": "the 2023 validation season with the fold-1 model",
        "not_computed_on_2024": True,
        "used_to_revise_the_model": False,
        "scoring": "neg_mean_absolute_error",
        "n_repeats": 20,
        "ranking": ranked,
        "caveat": (
            "Permutation importance is conditional on the rest of the design and on "
            "correlations among features. It does not measure a causal effect and "
            "nothing in this stage was changed on the basis of it."
        ),
    }


def run_experiment(
    *,
    outputs_dir: Path = FORECAST_OUTPUTS_DIR,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    verify_freeze_hash: bool = True,
) -> dict[str, Any]:
    """The complete HGB experiment, returning a JSON-ready record."""
    focal, r1_provenance = load_focal_table(
        outputs_dir=outputs_dir, verify_freeze_hash=verify_freeze_hash
    )
    ridge_provenance = (
        _verify_ridge_freeze(outputs_dir)
        if verify_freeze_hash
        else {"frozen_alpha": 100.0, "frozen_ridge_evaluation_mae": None}
    )

    residual_causality: dict[str, Any] = {"verified": False}
    metrics_path = outputs_dir / "r1_metrics.json"
    if metrics_path.exists():
        residual_causality = assert_residual_baseline_is_causal(
            json.loads(metrics_path.read_text())
        )
        residual_causality["verified"] = True

    # Written BEFORE any evaluation runs.
    grid = grid_record()
    (outputs_dir / "hgb_candidate_grid.json").write_text(
        json.dumps(grid, indent=2, sort_keys=True, default=_json_default)
    )
    logger.info(
        "wrote the candidate grid (%d configurations) before evaluation", grid["n_candidates"]
    )

    (fold1_train_seasons, fold1_eval), (fold2_train_seasons, fold2_eval) = SEASON_FORWARD_FOLDS
    fold1_train = focal[focal["season"].astype(int).isin(fold1_train_seasons)]
    fold1_validate = focal[focal["season"].astype(int) == int(fold1_eval)].copy()

    logger.info("fold 1: fit candidates on %s, select on %s", list(fold1_train_seasons), fold1_eval)
    selection = select_hyperparameters(fold1_train, fold1_validate, formulation=PRIMARY_FORMULATION)
    selected_parameters = selection["selected_parameters"]

    fold1_fit = fit_hgb(
        fold1_train, formulation=PRIMARY_FORMULATION, parameters=selected_parameters
    )
    fold1_validate[HGB_DIRECT_COLUMN] = predict(fold1_fit, fold1_validate)

    # Resolved on the TRAINING seasons only. Resolution includes a
    # zero-variance check, which reads the frame it is given -- deriving the
    # recorded feature list from the whole table would let 2024 rows decide
    # which columns are degenerate, and that is exactly the leak the frozen
    # specification exists to exclude. The fitted models already resolve on
    # their own training frames inside `fit_hgb`.
    pre_2024 = focal[focal["season"].astype(int).isin(fold2_train_seasons)]
    features, dropped = hgb_features(pre_2024)
    specification = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "frozen_before_2024_was_touched": True,
        "model_family": "sklearn.ensemble.HistGradientBoostingRegressor",
        "primary_formulation": PRIMARY_FORMULATION,
        "sensitivity_formulation": SENSITIVITY_FORMULATION,
        "formulation_designation": (
            "Direct is primary and residual is the prespecified sensitivity. The "
            "designation is forced by data availability -- 2022 has no causal "
            "shrunk-deserved value, so the residual arm cannot use the prescribed "
            "fit-2022/select-2023 structure -- and was fixed before either 2024 number "
            "was computed."
        ),
        "hyperparameters": selected_parameters,
        "feature_set": HGB_FEATURE_SET_NAME,
        "feature_list": features,
        "dropped_features": dropped,
        "feature_provenance": (
            "inherited verbatim from the frozen ridge Model D definition and resolved "
            "by the same frozen rules; no feature was chosen using 2024 ridge results"
        ),
        "preprocessing": "none (HistGradientBoosting is scale-invariant)",
        "loss": selected_parameters["loss"],
        "random_seed": selected_parameters["random_state"],
        "selection": {
            "metric": HGB_SELECTION_METRIC,
            "fit_seasons": list(fold1_train_seasons),
            "validation_season": fold1_eval,
            "evaluation_season_used_in_selection": False,
        },
        "nonlinear_value_rule": NONLINEAR_VALUE_RULE,
    }
    (outputs_dir / "hgb_specification.json").write_text(
        json.dumps(specification, indent=2, sort_keys=True, default=_json_default)
    )
    logger.info("froze the HGB specification before touching %s", fold2_eval)

    # ---- fold 2: the single evaluation -----------------------------------
    fold2_train = focal[focal["season"].astype(int).isin(fold2_train_seasons)]
    evaluate = focal[focal["season"].astype(int) == int(fold2_eval)].copy()
    causality = assert_no_later_season_rows(fold2_train, fold2_eval)

    direct_fit = fit_hgb(
        fold2_train, formulation=PRIMARY_FORMULATION, parameters=selected_parameters
    )
    evaluate[HGB_DIRECT_COLUMN] = predict(direct_fit, evaluate)

    # Residual arm: 2023 is the only season before 2024 carrying a causal
    # shrunk-deserved baseline, so it is the whole training set.
    residual_train = build_residual_target(
        focal[focal["season"].astype(int).isin(fold2_train_seasons)]
    )
    assert_no_later_season_rows(residual_train, fold2_eval)
    residual_fit = fit_hgb(
        residual_train,
        formulation=SENSITIVITY_FORMULATION,
        parameters=selected_parameters,
        target_column=RESIDUAL_TARGET,
    )
    evaluate = build_residual_target(evaluate)
    evaluate["predicted_residual"] = predict(residual_fit, evaluate)
    evaluate[HGB_RESIDUAL_COLUMN] = evaluate[RESIDUAL_BASELINE].to_numpy(dtype=float) + evaluate[
        "predicted_residual"
    ].to_numpy(dtype=float)

    evaluate[RIDGE_PREDICTION_COLUMN] = _reproduce_frozen_ridge(
        fold2_train, evaluate, ridge_provenance=ridge_provenance
    )

    predictors = [HGB_DIRECT_COLUMN, HGB_RESIDUAL_COLUMN, *BENCHMARKS]
    complete = evaluate.dropna(subset=[HGB_TARGET, *predictors])
    metrics = {name: _full_metrics(complete, predictor=name) for name in predictors}

    deltas: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)
    for benchmark in BENCHMARKS:
        deltas[benchmark] = {}
        for challenger in (HGB_DIRECT_COLUMN, HGB_RESIDUAL_COLUMN):
            delta = paired_delta(
                complete, target=HGB_TARGET, reference=benchmark, challenger=challenger
            )
            assert_delta_convention(delta)
            deltas[benchmark][challenger] = delta.as_record()
            bootstraps[f"{challenger}_vs_{benchmark}"] = run_paired_bootstrap(
                complete,
                target=HGB_TARGET,
                reference=benchmark,
                challenger=challenger,
                design=design,
            )

    verdict = _nonlinear_value_verdict(deltas, bootstraps)
    cell_key = f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "contact_forecast_hgb_v1",
        "model_family": "HistGradientBoostingRegressor -- the only nonlinear family fitted",
        "explicitly_not_implemented": [
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "neural network",
            "ensemble search",
            "model zoo",
            "prediction intervals",
            "dashboard UI",
        ],
        "target": {
            "column": HGB_TARGET,
            "cell": cell_key,
            "conditional_interpretation": conditional_forecast_language(PRIMARY_HORIZON),
        },
        "sign_convention": DELTA_SIGN_CONVENTION,
        "provenance": {
            "r1": r1_provenance,
            "ridge": ridge_provenance,
            "residual_baseline_causality": residual_causality,
        },
        "candidate_grid": grid,
        "specification": specification,
        "fold_1_selection": selection,
        "fold_1_validation_mae": {
            HGB_DIRECT_COLUMN: float(
                np.mean(
                    np.abs(
                        fold1_validate[HGB_DIRECT_COLUMN].to_numpy(dtype=float)
                        - fold1_validate[HGB_TARGET].to_numpy(dtype=float)
                    )
                )
            )
        },
        "fold_2_evaluation": {
            "train_seasons": list(fold2_train_seasons),
            "evaluate_season": int(fold2_eval),
            "n_train_direct": int(len(fold2_train)),
            "n_train_residual": int(len(residual_train)),
            "residual_train_seasons": [int(s) for s in residual_fit.train_seasons],
            "n_evaluate": int(len(complete)),
            "causality": causality,
            "fits": {
                PRIMARY_FORMULATION: direct_fit.as_record(),
                SENSITIVITY_FORMULATION: residual_fit.as_record(),
            },
            "metrics": metrics,
        },
        "deltas_on_evaluation_season": deltas,
        "bootstrap": bootstraps,
        "nonlinear_value_test": verdict,
        "residual_diagnostics": _residual_diagnostics(complete),
        "stability": {
            "vs_shrunk_deserved": _stability(
                complete,
                challenger=HGB_DIRECT_COLUMN,
                reference="shrunk_deserved_persistence",
            ),
            "vs_frozen_ridge": _stability(
                complete, challenger=HGB_DIRECT_COLUMN, reference=RIDGE_PREDICTION_COLUMN
            ),
        },
        "permutation_importance": _permutation_importance(
            fold1_fit, fold1_validate, target_column=HGB_TARGET
        ),
    }


def _nonlinear_value_verdict(deltas: dict[str, Any], bootstraps: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified preference rule and state the answer plainly."""
    cell_key = f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    delta = deltas[RIDGE_PREDICTION_COLUMN][HGB_DIRECT_COLUMN]
    summary = bootstraps[f"{HGB_DIRECT_COLUMN}_vs_{RIDGE_PREDICTION_COLUMN}"]["cells"][cell_key][
        "pooled"
    ]["delta_mae"]
    point_favours_hgb = delta["delta_mae"] < 0
    excludes_zero = summary["ci_crosses_zero"] is False
    prefer_hgb = bool(point_favours_hgb and excludes_zero)
    return {
        **NONLINEAR_VALUE_RULE,
        "delta_mae_hgb_minus_ridge": delta["delta_mae"],
        "pct_change_mae": delta["pct_change_mae"],
        "ci_delta_mae": [summary["ci_lower"], summary["ci_upper"]],
        "ci_crosses_zero": summary["ci_crosses_zero"],
        "point_estimate_favours_hgb": point_favours_hgb,
        "interval_excludes_zero": excludes_zero,
        "preferred_model": "hgb" if prefer_hgb else "ridge",
        "answer": (
            "Yes -- HGB extracts information the full-profile ridge did not, by the "
            "prespecified rule."
            if prefer_hgb
            else "No -- HGB does not materially outperform the full-profile ridge by the "
            "prespecified rule, so ridge is preferred on simplicity and "
            "interpretability grounds."
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value)!r} is not JSON-serializable")


def run(
    *,
    outputs_dir: Path = FORECAST_OUTPUTS_DIR,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    verify_freeze_hash: bool = True,
) -> dict[str, Path]:
    """Run the experiment and write its artifacts plus the report."""
    assert_path_outside_forbidden_namespaces(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(
        outputs_dir=outputs_dir,
        reps=reps,
        seed=seed,
        alpha=alpha,
        verify_freeze_hash=verify_freeze_hash,
    )
    results_path = outputs_dir / "hgb_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=_json_default))

    from forecast.hgb_report import render_hgb_report

    report_path = outputs_dir / "hgb_report.md"
    report_path.write_text(render_hgb_report(results))
    logger.info("wrote %s and %s", results_path, report_path)
    return {
        "results": results_path,
        "report": report_path,
        "grid": outputs_dir / "hgb_candidate_grid.json",
        "specification": outputs_dir / "hgb_specification.json",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_BOOTSTRAP_ALPHA)
    parser.add_argument("--skip-freeze-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    written = run(
        outputs_dir=args.outputs_dir,
        reps=args.bootstrap_reps,
        seed=args.seed,
        alpha=args.alpha,
        verify_freeze_hash=not args.skip_freeze_check,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
