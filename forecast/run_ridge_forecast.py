"""Contact Forecast: run the ridge forecasting experiment and write its artifacts.

Ridge only. No HistGradientBoosting, no nonlinear candidate, no interaction
basis -- the stop condition for this stage is the ridge result, including its
negative ablations.

## What is evaluated, and against what

Target: `target_realized_rv_per_100` in the focal 100 -> 100 cell, conditional
on the hitter accumulating the next 100 resolved eligible BBE.

Four prespecified nested models (`forecast.ridge.FEATURE_SETS`) are compared
against three benchmarks: `league_mean`, `shrunk_realized_persistence` and
`shrunk_deserved_persistence`. The last is the one that matters. R1 already
established that raw realized persistence is worse than the league mean and
that most of shrinkage's benefit is generic regression to the mean, so a model
that beats only those has demonstrated nothing.

    key_delta = MAE(best_ridge) - MAE(shrunk_deserved_persistence)

on R1's frozen sign convention: negative means the forecasting layer adds
value beyond the established luck-adjusted persistence signal.

## Season-forward, twice

    fold 1  train 2022             -> validate 2023   (alpha AND model selection)
    fold 2  train 2022-2023, alpha fixed from fold 1 -> evaluate 2024

2024 participates in no selection of any kind: not alpha, not the winning
model, not feature inclusion. Fold 1's numbers are reported but are labelled
selection-contaminated, because the same season chose the hyperparameter.

## Clustering and balancing are honest no-ops here, and say so

In a single evaluation season at a fixed cutoff every hitter contributes
exactly one window. Batter-clustered resampling therefore coincides with an
ordinary bootstrap, and batter-balanced MAE coincides with window-weighted
MAE, by construction rather than by luck. Both are computed and reported, with
that equivalence asserted in the artifact so neither is mistaken for extra
evidence.

## The R1 package this builds on is frozen

The prediction table is read from the frozen R1 artifacts and its hash is
checked against `r1_freeze_manifest.json`. A ridge result computed against a
drifted R1 package would not be comparable to R1's own benchmarks.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_OUTPUTS_DIR,
    PRIMARY_HORIZON,
    SEASON_FORWARD_FOLDS,
    assert_forecast_seasons_allowed,
    assert_path_outside_forbidden_namespaces,
    conditional_forecast_language,
)
from forecast.freeze_r1 import hash_file
from forecast.metrics import (
    DELTA_SIGN_CONVENTION,
    assert_delta_convention,
    batter_balanced_delta,
    batter_balanced_errors,
    evaluate_predictor,
    paired_delta,
)
from forecast.ridge import (
    ALPHA_GRID,
    COEFFICIENT_INTERPRETATION_CAVEAT,
    FEATURE_SETS,
    MODEL_DESCRIPTIONS,
    PRESPECIFIED_DROPS,
    RIDGE_TARGET,
    SELECTION_METRIC,
    assert_no_later_season_rows,
    collinearity_audit,
    fit_ridge,
    predict,
    resolve_features,
    select_alpha,
)

logger = logging.getLogger(__name__)

#: The benchmark every ridge model is measured against. R1's own primary
#: challenger, so the delta is directly comparable to R1's numbers.
PRIMARY_BENCHMARK = "shrunk_deserved_persistence"

#: Reported alongside it, in report order.
BENCHMARKS: tuple[str, ...] = (
    "league_mean",
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
)


class ForecastRidgeRunError(RuntimeError):
    """Raised when the ridge experiment cannot run as specified."""


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """`1 - SSE/SST` against the EVALUATION set's own mean.

    Computed here rather than added to `forecast.metrics`, which is frozen as
    part of the R1 package and must not change now that R1 is sealed.
    """
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")


def _full_metrics(frame: pd.DataFrame, *, predictor: str) -> dict[str, Any]:
    """Every required metric for one predictor on one evaluation season."""
    measured = evaluate_predictor(frame, target=RIDGE_TARGET, predictor=predictor)
    balanced_mae, balanced_rmse, n_batters = batter_balanced_errors(
        frame, target=RIDGE_TARGET, predictor=predictor
    )
    record = measured.as_record()
    record.update(
        {
            "r_squared": _r_squared(
                frame[RIDGE_TARGET].to_numpy(dtype=float),
                frame[predictor].to_numpy(dtype=float),
            ),
            "batter_balanced_mae": balanced_mae,
            "batter_balanced_rmse": balanced_rmse,
            "n_unique_batters": n_batters,
            "batter_balancing_is_a_no_op": bool(n_batters == len(frame)),
        }
    )
    return record


def load_focal_table(
    *, outputs_dir: Path = FORECAST_OUTPUTS_DIR, verify_freeze_hash: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The frozen R1 prediction table, restricted to the focal cell.

    Raises:
        ForecastRidgeRunError: If the table is missing, or if its hash differs
            from the one recorded in the R1 freeze manifest.
    """
    table_path = outputs_dir / "r1_prediction_table.parquet"
    assert_path_outside_forbidden_namespaces(table_path)
    if not table_path.exists():
        raise ForecastRidgeRunError(
            f"No R1 prediction table at {table_path}. Run `make run-forecast-r1` first."
        )

    provenance: dict[str, Any] = {
        "prediction_table_path": str(table_path),
        "prediction_table_sha256": hash_file(table_path),
    }
    manifest_path = outputs_dir / "r1_freeze_manifest.json"
    if verify_freeze_hash:
        if not manifest_path.exists():
            raise ForecastRidgeRunError(
                f"No R1 freeze manifest at {manifest_path}. R1 must be frozen before a "
                "forecasting model is fitted against it."
            )
        frozen = json.loads(manifest_path.read_text())
        recorded = frozen["artifact_sha256"]["r1_prediction_table.parquet"]
        if recorded != provenance["prediction_table_sha256"]:
            raise ForecastRidgeRunError(
                "The R1 prediction table has drifted from its freeze "
                f"(frozen {recorded[:16]}..., current "
                f"{provenance['prediction_table_sha256'][:16]}...). Ridge results built "
                "on a drifted package are not comparable to R1's benchmarks."
            )
        provenance["r1_freeze_manifest_sha256"] = frozen["manifest_sha256"]
        provenance["r1_conclusion"] = frozen["conclusion"]["verdict"]

    table = pd.read_parquet(table_path)
    # A third layer behind the season guards in scoring and assembly: whatever
    # is actually IN the table must be an authorized development season. 2025
    # is sealed and 2026 is behind the Phase 2 gate, and neither may reach an
    # estimator through a hand-edited parquet.
    assert_forecast_seasons_allowed(sorted(int(s) for s in table["season"].unique()))
    focal = table[
        (table["cutoff"].astype(int) == FOCAL_PRESENTATION_CUTOFF)
        & (table["horizon"].astype(int) == PRIMARY_HORIZON)
    ].copy()
    if focal.empty:
        raise ForecastRidgeRunError("The frozen table has no windows in the focal cell")
    provenance["focal_cell"] = {
        "cutoff": FOCAL_PRESENTATION_CUTOFF,
        "horizon": PRIMARY_HORIZON,
        "n_windows": int(len(focal)),
        "by_season": {
            str(s): int(n) for s, n in focal.groupby(focal["season"].astype(int)).size().items()
        },
    }
    return focal, provenance


def _rank_deficient_diagnostic(
    train: pd.DataFrame, evaluate: pd.DataFrame, *, model_name: str, alpha: float
) -> dict[str, Any]:
    """Fit the DECLARED feature list including the exactly-dependent columns.

    The identifiable parameterization drops `surprise` because it is exactly
    `realized - deserved`. Ridge would still return a unique solution -- the L2
    penalty regularises the singularity away -- but it is a different solution,
    because the penalty distributes weight across the collinear set instead of
    concentrating it. This fits that variant deliberately and reports how much
    the predictions actually move, so "we used the identifiable version"
    is a measured choice rather than an assertion.

    Bypasses `fit_ridge` on purpose: that function refuses a rank-deficient
    design, and it should keep refusing one.
    """
    nominal = FEATURE_SETS[model_name]
    with_dependents = [f for f in nominal if f in train.columns]
    # Keep every declared column except the genuinely degenerate ones.
    usable = [
        f
        for f in with_dependents
        if train[f].to_numpy(dtype=float).max() != train[f].to_numpy(dtype=float).min()
    ]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[usable].to_numpy(dtype=float))
    estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    estimator.fit(x_train, train[RIDGE_TARGET].to_numpy(dtype=float))
    predictions = estimator.predict(scaler.transform(evaluate[usable].to_numpy(dtype=float)))
    y_true = evaluate[RIDGE_TARGET].to_numpy(dtype=float)

    standardized = x_train
    rank = int(np.linalg.matrix_rank(standardized))
    return {
        "model": model_name,
        "features_including_exact_dependencies": usable,
        "rank": rank,
        "n_features": len(usable),
        "is_rank_deficient": bool(rank < len(usable)),
        "mae": float(np.mean(np.abs(predictions - y_true))),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(usable, estimator.coef_, strict=True)
        },
        "note": (
            "Fitted only as a diagnostic. Ridge returns a unique solution despite the "
            "exact dependency, but the coefficients on the collinear set are an "
            "arbitrary split and the predictions differ from the identifiable fit. "
            "The reported models use the identifiable parameterization."
        ),
    }


def run_fold(
    focal: pd.DataFrame,
    *,
    train_seasons: tuple[int, ...],
    evaluate_season: int,
    alphas: dict[str, float] | None,
) -> dict[str, Any]:
    """One season-forward fold: fit every model, evaluate every model and benchmark."""
    train = focal[focal["season"].astype(int).isin(train_seasons)]
    evaluate = focal[focal["season"].astype(int) == int(evaluate_season)].copy()
    if train.empty or evaluate.empty:
        raise ForecastRidgeRunError(
            f"Fold train={train_seasons} evaluate={evaluate_season} has no rows"
        )
    causality = assert_no_later_season_rows(train, evaluate_season)

    fits: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for model_name in FEATURE_SETS:
        if alphas is None:
            selection = select_alpha(
                train,
                evaluate,
                model_name=model_name,
                alphas=ALPHA_GRID,
            )
            selections[model_name] = selection
            alpha = float(selection["selected_alpha"])
        else:
            alpha = float(alphas[model_name])
        fit = fit_ridge(train, model_name=model_name, alpha=alpha)
        evaluate[f"ridge_{model_name}"] = predict(fit, evaluate)
        fits[model_name] = fit

    predictors = [f"ridge_{name}" for name in FEATURE_SETS]
    available_benchmarks = [
        b for b in BENCHMARKS if b in evaluate.columns and evaluate[b].notna().all()
    ]
    complete = evaluate.dropna(subset=[RIDGE_TARGET, *predictors, *available_benchmarks])

    metrics = {
        name: _full_metrics(complete, predictor=name)
        for name in [*predictors, *available_benchmarks]
    }
    return {
        "train_seasons": list(train_seasons),
        "evaluate_season": int(evaluate_season),
        "n_train": int(len(train)),
        "n_evaluate": int(len(complete)),
        "causality": causality,
        "benchmarks_available": available_benchmarks,
        "benchmarks_unavailable": [b for b in BENCHMARKS if b not in available_benchmarks],
        "alpha_selection": selections,
        "fits": {name: fit.as_record() for name, fit in fits.items()},
        "metrics": metrics,
        "_frame": complete,
        "_fits": fits,
    }


def _deltas_against(
    frame: pd.DataFrame, *, challengers: list[str], reference: str
) -> dict[str, Any]:
    """Paired deltas of every ridge model against one benchmark."""
    out: dict[str, Any] = {}
    for challenger in challengers:
        delta = paired_delta(frame, target=RIDGE_TARGET, reference=reference, challenger=challenger)
        assert_delta_convention(delta)
        balanced = batter_balanced_delta(
            frame, target=RIDGE_TARGET, reference=reference, challenger=challenger
        )
        assert_delta_convention(balanced)
        out[challenger] = {
            "delta": delta.as_record(),
            "batter_balanced_delta": balanced.as_record(),
            "batter_balanced_matches_window_weighted": bool(
                np.isclose(balanced.delta_mae, delta.delta_mae)
            ),
        }
    return out


def run_experiment(
    *,
    outputs_dir: Path = FORECAST_OUTPUTS_DIR,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    verify_freeze_hash: bool = True,
) -> dict[str, Any]:
    """The complete ridge experiment, returning a JSON-ready record."""
    focal, provenance = load_focal_table(
        outputs_dir=outputs_dir, verify_freeze_hash=verify_freeze_hash
    )

    (fold1_train, fold1_eval), (fold2_train, fold2_eval) = SEASON_FORWARD_FOLDS
    logger.info("fold 1: train %s -> validate %s", list(fold1_train), fold1_eval)
    fold1 = run_fold(focal, train_seasons=fold1_train, evaluate_season=fold1_eval, alphas=None)
    selected_alphas = {
        name: float(fold1["alpha_selection"][name]["selected_alpha"]) for name in FEATURE_SETS
    }

    # Model selection also happens on fold 1, by the same prespecified metric.
    validation_scores = {
        name: fold1["metrics"][f"ridge_{name}"][SELECTION_METRIC] for name in FEATURE_SETS
    }
    best_model = min(validation_scores, key=lambda name: validation_scores[name])
    logger.info("selected model %s on the %s validation season", best_model, fold1_eval)

    logger.info("fold 2: train %s -> evaluate %s", list(fold2_train), fold2_eval)
    fold2 = run_fold(
        focal,
        train_seasons=fold2_train,
        evaluate_season=fold2_eval,
        alphas=selected_alphas,
    )

    evaluation_frame = fold2["_frame"]
    challengers = [f"ridge_{name}" for name in FEATURE_SETS]
    deltas = {
        benchmark: _deltas_against(evaluation_frame, challengers=challengers, reference=benchmark)
        for benchmark in fold2["benchmarks_available"]
    }

    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)
    bootstraps: dict[str, Any] = {}
    for benchmark in fold2["benchmarks_available"]:
        for challenger in challengers:
            key = f"{challenger}_vs_{benchmark}"
            bootstraps[key] = run_paired_bootstrap(
                evaluation_frame,
                target=RIDGE_TARGET,
                reference=benchmark,
                challenger=challenger,
                design=design,
            )

    key_delta_record = _build_key_delta(
        evaluation_frame,
        fold2=fold2,
        best_model=best_model,
        bootstraps=bootstraps,
    )

    fold2_train_frame = focal[focal["season"].astype(int).isin(fold2_train)]
    diagnostics = _build_diagnostics(
        fold2_train_frame,
        fold2=fold2,
        selected_alphas=selected_alphas,
        evaluation_frame=evaluation_frame,
    )

    for fold in (fold1, fold2):
        fold.pop("_frame", None)
        fold.pop("_fits", None)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "contact_forecast_ridge_v1",
        "model_class": "ridge (regularized linear) only -- no nonlinear candidate fitted",
        "target": {
            "column": RIDGE_TARGET,
            "cell": f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}",
            "conditional_interpretation": conditional_forecast_language(PRIMARY_HORIZON),
        },
        "sign_convention": DELTA_SIGN_CONVENTION,
        "r1_provenance": provenance,
        "protocol": {
            "folds": [
                {
                    "fold": 1,
                    "train": list(fold1_train),
                    "validate": fold1_eval,
                    "purpose": "alpha selection AND model selection",
                    "results_are_selection_contaminated": True,
                },
                {
                    "fold": 2,
                    "train": list(fold2_train),
                    "evaluate": fold2_eval,
                    "purpose": "clean evaluation at the fixed alpha",
                    "participated_in_any_selection": False,
                },
            ],
            "selection_metric": SELECTION_METRIC,
            "alpha_grid": list(ALPHA_GRID),
            "split_rule": (
                "season-forward only; windows and hitters are never split randomly "
                "across train and validation"
            ),
            "standardization": "fit on training seasons only, applied to later seasons",
            "prior_season_features_excluded": (
                "prior-season deserved is structurally unavailable for the 2022 training "
                "season and no imputation scheme was introduced to recover it"
            ),
        },
        "feature_sets": {
            name: {"description": MODEL_DESCRIPTIONS[name], "declared_features": list(features)}
            for name, features in FEATURE_SETS.items()
        },
        "selected_alphas": selected_alphas,
        "validation_scores": validation_scores,
        "selected_model": {
            "model": best_model,
            "selected_on": f"{fold1_eval} validation {SELECTION_METRIC}",
            "note": "selection used no 2024 information of any kind",
        },
        "fold_1_validation": fold1,
        "fold_2_evaluation": fold2,
        "deltas_on_evaluation_season": deltas,
        "bootstrap": bootstraps,
        "key_delta": key_delta_record,
        "diagnostics": diagnostics,
    }


def _build_key_delta(
    frame: pd.DataFrame,
    *,
    fold2: dict[str, Any],
    best_model: str,
    bootstraps: dict[str, Any],
) -> dict[str, Any]:
    """`MAE(best_ridge) - MAE(shrunk_deserved)`, the decision number."""
    challenger = f"ridge_{best_model}"
    if PRIMARY_BENCHMARK not in fold2["benchmarks_available"]:
        return {
            "status": "benchmark_unavailable",
            "benchmark": PRIMARY_BENCHMARK,
            "note": "the primary benchmark has no values in the evaluation season",
        }
    delta = paired_delta(
        frame, target=RIDGE_TARGET, reference=PRIMARY_BENCHMARK, challenger=challenger
    )
    assert_delta_convention(delta)
    boot = bootstraps[f"{challenger}_vs_{PRIMARY_BENCHMARK}"]["cells"][
        f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    ]["pooled"]
    return {
        "definition": f"MAE({challenger}) - MAE({PRIMARY_BENCHMARK})",
        "interpretation": (
            "negative means the forecasting layer adds value beyond the established "
            "luck-adjusted persistence signal"
        ),
        "best_ridge_model": best_model,
        "evaluation_season": fold2["evaluate_season"],
        "n_evaluation_windows": int(len(frame)),
        "mae_best_ridge": delta.mae_challenger,
        "mae_shrunk_deserved": delta.mae_reference,
        "delta_mae": delta.delta_mae,
        "pct_change_mae": delta.pct_change_mae,
        "delta_rmse": delta.delta_rmse,
        "pct_change_rmse": delta.pct_change_rmse,
        "direction_mae": delta.direction_mae,
        "ci_delta_mae": [boot["delta_mae"]["ci_lower"], boot["delta_mae"]["ci_upper"]],
        "ci_crosses_zero": boot["delta_mae"]["ci_crosses_zero"],
        "share_replicates_favouring_benchmark": boot["delta_mae"][
            "share_replicates_favouring_reference"
        ],
        "target_sd": float(frame[RIDGE_TARGET].std(ddof=1)),
        "abs_delta_mae_as_fraction_of_target_sd": float(
            abs(delta.delta_mae) / float(frame[RIDGE_TARGET].std(ddof=1))
        ),
    }


def _build_diagnostics(
    train_frame: pd.DataFrame,
    *,
    fold2: dict[str, Any],
    selected_alphas: dict[str, float],
    evaluation_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Collinearity audit, coefficients, and the rank-deficient comparison."""
    audits: dict[str, Any] = {}
    for model_name in FEATURE_SETS:
        features, drops = resolve_features(FEATURE_SETS[model_name], train_frame)
        audits[model_name] = {
            "retained_features": features,
            "dropped_features": drops,
            **collinearity_audit(train_frame, features),
        }

    unique_batters = int(evaluation_frame["batter"].nunique())
    return {
        "coefficient_caveat": COEFFICIENT_INTERPRETATION_CAVEAT,
        "prespecified_drops": dict(PRESPECIFIED_DROPS),
        "collinearity_by_model": audits,
        "rank_deficient_variant": {
            model_name: _rank_deficient_diagnostic(
                train_frame,
                evaluation_frame,
                model_name=model_name,
                alpha=selected_alphas[model_name],
            )
            for model_name in ("C_results_plus_deserved", "D_full_contact_profile")
        },
        "standardized_coefficients": {
            name: fit["standardized_coefficients"] for name, fit in fold2["fits"].items()
        },
        "clustering_is_a_no_op_here": {
            "n_evaluation_windows": int(len(evaluation_frame)),
            "n_unique_batters": unique_batters,
            "one_window_per_hitter": bool(unique_batters == len(evaluation_frame)),
            "consequence": (
                "in a single evaluation season at a fixed cutoff each hitter has exactly "
                "one window, so batter-clustered resampling coincides with an ordinary "
                "bootstrap and batter-balanced MAE coincides with window-weighted MAE. "
                "Both are reported, and neither is extra evidence."
            ),
        },
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
    """Run the experiment and write its machine-readable artifacts plus the report."""
    assert_path_outside_forbidden_namespaces(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        outputs_dir=outputs_dir,
        reps=reps,
        seed=seed,
        alpha=alpha,
        verify_freeze_hash=verify_freeze_hash,
    )
    results_path = outputs_dir / "ridge_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s", results_path)

    from forecast.ridge_report import render_ridge_report

    report_path = outputs_dir / "ridge_report.md"
    report_path.write_text(render_ridge_report(results))
    logger.info("wrote %s", report_path)
    return {"results": results_path, "report": report_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_BOOTSTRAP_ALPHA)
    parser.add_argument(
        "--skip-freeze-check",
        action="store_true",
        help="Skip the R1 freeze-hash check (development only).",
    )
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
