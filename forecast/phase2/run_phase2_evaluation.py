"""Contact Forecast Phase 2: the first held-out evaluation, on 2026.

Runs the ALREADY-FROZEN specification against a pinned 2026 snapshot. Nothing
here tunes, selects, recalibrates or deploys, and no 2026 number feeds back
into any frozen element.

## Order of operations, which is the control

1. Verify the R1 / ridge / HGB freeze chain and require zero drift. If any
   stage has drifted, stop before a single 2026 byte is read.
2. Load the frozen contract -- model, alpha, feature declaration, benchmarks,
   sign convention -- FROM those verified manifests rather than restating it.
3. Write and hash `phase2_authorization.json`.
4. Pin one immutable 2026 snapshot, verify its integrity hashes, record it.
5. Build features from each hitter's FIRST 100 resolved eligible BBE.
6. Fit the final ridge and both shrinkage benchmarks on permitted development
   seasons only (2022-2024). 2025 is unreachable; 2026 is never fitted on.
7. Persist every forecast BEFORE any outcome is attached.
8. Attach the next-100-BBE target only to hitters who actually completed it.

## Completed versus pending is a hard separation

A hitter who reached 100 resolved BBE gets a forecast. A hitter who has not
since completed 100 MORE resolved BBE has no target and is carried in the
pending artifact -- never evaluated on a partial window, never dropped from
the prediction ledger. A later evaluation attaches outcomes to these frozen
predictions; it does not regenerate them.
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

from forecast.baselines import (
    assert_fit_is_causal,
    build_baseline_predictions,
    fit_shrinkage,
    shrinkage_fit_record,
)
from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.features import build_window_features
from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_DATA_DIR,
    FORECAST_OUTPUTS_DIR,
    PRIMARY_HORIZON,
)
from forecast.freeze_r1 import hash_file, hash_json
from forecast.metrics import (
    DELTA_SIGN_CONVENTION,
    assert_delta_convention,
    batter_balanced_errors,
    evaluate_predictor,
    paired_delta,
)
from forecast.phase2.phase2_config import (
    PHASE2_OUTPUTS_DIR,
    PHASE2_PERMITTED_TRAINING_SEASONS,
    SUCCESS_CLASSIFICATION,
    assert_phase2_path_allowed,
    assert_phase2_seasons_allowed,
    build_authorization_record,
)
from forecast.phase2.phase2_windows import (
    cutoff_boundary_events,
    horizon_completion,
    order_events_for_phase2,
)
from forecast.phase2.snapshot import (
    DESERVED_COLUMN,
    OBSERVED_COLUMN,
    assert_no_full_telescoping_columns,
    load_contact_stage_ledger,
    resolve_snapshot,
)
from forecast.ridge import fit_ridge
from forecast.ridge import predict as ridge_predict
from forecast.windows import build_windows, select_scored_eligible_events

logger = logging.getLogger(__name__)

TARGET = "target_realized_rv_per_100"
FORECAST_COLUMN = "contact_forecast"
BENCHMARKS: tuple[str, ...] = (
    "league_mean",
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
)
PRIMARY_BENCHMARK = "shrunk_deserved_persistence"

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "phase2_authorization.json",
    "phase2_2026_snapshot_manifest.json",
    "phase2_2026_predictions.parquet",
    "phase2_2026_completed_evaluation.parquet",
    "phase2_2026_pending_predictions.parquet",
    "phase2_2026_metrics.json",
    "phase2_2026_bootstrap.json",
    "phase2_2026_stability.json",
    "phase2_2026_distribution_shift.json",
    "phase2_2026_report.md",
)


class Phase2EvaluationError(RuntimeError):
    """Raised when the Phase 2 evaluation cannot run as specified."""


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"{type(value)!r} is not JSON-serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_phase2_path_allowed(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------
# Step 0: verify everything before opening 2026
# --------------------------------------------------------------------------


def verify_freeze_chain(*, research_dir: Path = FORECAST_OUTPUTS_DIR) -> dict[str, Any]:
    """Verify R1, ridge and HGB, and refuse to proceed on any drift.

    Raises:
        Phase2EvaluationError: If any stage has drifted from its freeze.
    """
    from forecast.freeze_hgb import HGB_FREEZE_SPEC
    from forecast.freeze_r1 import verify_freeze
    from forecast.freeze_ridge import RIDGE_FREEZE_SPEC
    from forecast.stage_freeze import verify_stage

    results = {
        "r1": verify_freeze(outputs_dir=research_dir, data_dir=FORECAST_DATA_DIR),
        "ridge": verify_stage(RIDGE_FREEZE_SPEC, outputs_dir=research_dir),
        "hgb": verify_stage(HGB_FREEZE_SPEC, outputs_dir=research_dir),
    }
    drifted = {name: r["drift"] for name, r in results.items() if not r["matches_freeze"]}
    if drifted:
        raise Phase2EvaluationError(
            f"The freeze chain has drifted: {drifted}. 2026 is not opened against a "
            "package that no longer matches what was sealed."
        )
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "zero_drift_confirmed": True,
        "manifest_sha256": {name: r["manifest_sha256"] for name, r in results.items()},
        "detail": results,
    }


def load_frozen_contract(*, research_dir: Path = FORECAST_OUTPUTS_DIR) -> dict[str, Any]:
    """Read every frozen element out of the verified manifests.

    Loaded rather than restated: a value retyped here could drift from what was
    sealed, while a value read from a verified manifest cannot.

    Raises:
        Phase2EvaluationError: If the frozen model is not the expected
            full-contact-profile ridge, or the recorded reasons for choosing it
            no longer hold.
    """
    r1 = json.loads((research_dir / "r1_freeze_manifest.json").read_text())
    ridge = json.loads((research_dir / "ridge_freeze_manifest.json").read_text())
    hgb = json.loads((research_dir / "hgb_freeze_manifest.json").read_text())

    key = ridge["key_results"]
    model = key["selected_model"]["model"]
    if model != "D_full_contact_profile":
        raise Phase2EvaluationError(
            f"The frozen model is {model!r}, not the expected full-contact-profile ridge."
        )
    ridge_mae = key["evaluation_mae_by_model"][model]
    deserved_mae = key["benchmark_mae"][PRIMARY_BENCHMARK]
    if not ridge_mae < deserved_mae:
        raise Phase2EvaluationError(
            f"The recorded reason for the model choice no longer holds: frozen ridge "
            f"MAE {ridge_mae} is not better than shrunk deserved {deserved_mae}."
        )
    nonlinear = hgb["key_results"]["nonlinear_value_test"]
    if nonlinear["preferred_model"] != "ridge" or nonlinear["delta_mae_hgb_minus_ridge"] <= 0:
        raise Phase2EvaluationError(
            "The recorded reason for retaining ridge no longer holds: the HGB stage's "
            f"prespecified rule now reports {nonlinear}."
        )

    return {
        "model": model,
        "model_choice_reasons": {
            "best_frozen_2024_mae": ridge_mae,
            "better_than_shrunk_deserved": deserved_mae,
            "hgb_did_not_outperform_ridge": True,
            "hgb_vs_ridge_point_estimate_favoured_ridge": nonlinear["delta_mae_hgb_minus_ridge"],
            "prespecified_nonlinear_rule_retained_ridge": nonlinear["preferred_model"],
        },
        "alpha": float(key["selected_alphas"][model]),
        "declared_features": list(key["feature_sets"][model]),
        "target": r1["specification"]["targets"]["primary"],
        "benchmarks": list(r1["specification"]["baseline_ladder"]),
        "sign_convention": r1["specification"]["metrics"]["sign_convention"],
        "shrinkage_rule": r1["specification"]["shrinkage"],
        "stage_verdicts": {
            "r1": r1["conclusion"]["verdict"],
            "ridge": ridge["conclusion"]["verdict"],
            "hgb": hgb["conclusion"]["verdict"],
        },
    }


# --------------------------------------------------------------------------
# Development training data (permitted seasons only)
# --------------------------------------------------------------------------


def load_development_training(
    *,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    data_dir: Path = FORECAST_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Focal-cell windows and per-event rows for the permitted development seasons.

    2025 cannot appear: it has no walk-forward ledger, it is absent from the
    frozen prediction table, and `assert_phase2_seasons_allowed` refuses it.

    Raises:
        Phase2EvaluationError: If a forbidden season somehow appears.
    """
    table_path = research_dir / "r1_prediction_table.parquet"
    assert_phase2_path_allowed(table_path)
    table = pd.read_parquet(table_path)
    focal = table[
        (table["cutoff"].astype(int) == FOCAL_PRESENTATION_CUTOFF)
        & (table["horizon"].astype(int) == PRIMARY_HORIZON)
        & (table["season"].astype(int).isin(PHASE2_PERMITTED_TRAINING_SEASONS))
    ].copy()
    assert_phase2_seasons_allowed(sorted(int(s) for s in focal["season"].unique()))

    events: list[pd.DataFrame] = []
    for season in PHASE2_PERMITTED_TRAINING_SEASONS:
        path = data_dir / f"walk_forward_ledger_{season}.parquet"
        assert_phase2_path_allowed(path)
        ledger = pd.read_parquet(path)
        events.append(ledger[ledger[OBSERVED_COLUMN].notna()])
    prior_events = pd.concat(events, ignore_index=True)
    assert_phase2_seasons_allowed(sorted(int(s) for s in prior_events["season"].unique()))

    audit = {
        "permitted_training_seasons": list(PHASE2_PERMITTED_TRAINING_SEASONS),
        "forbidden_seasons_present": [],
        "n_focal_windows": int(len(focal)),
        "focal_windows_by_season": {
            str(s): int(n) for s, n in focal.groupby(focal["season"].astype(int)).size().items()
        },
        "n_prior_events": int(len(prior_events)),
        "prior_events_by_season": {
            str(s): int(n)
            for s, n in prior_events.groupby(prior_events["season"].astype(int)).size().items()
        },
        "source_table_sha256": hash_file(table_path),
    }
    return focal, prior_events, audit


# --------------------------------------------------------------------------
# Forecast generation
# --------------------------------------------------------------------------


def build_2026_forecasts(
    *,
    snapshot: Any,
    development_focal: pd.DataFrame,
    prior_events: pd.DataFrame,
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every 2026 forecast, generated before any outcome is attached."""
    ledger, projection_audit = load_contact_stage_ledger(snapshot)
    assert_no_full_telescoping_columns(ledger)

    resolved, exclusion = select_scored_eligible_events(ledger)
    ordered = order_events_for_phase2(resolved)
    completion = horizon_completion(
        ordered, cutoff=FOCAL_PRESENTATION_CUTOFF, horizon=PRIMARY_HORIZON
    )

    # Features from the FIRST 100 resolved BBE only. `build_window_features` is
    # the frozen Phase 1 builder, and it reads its slice through
    # `history_events`, which cannot see past the cutoff.
    features = build_window_features(ordered, FOCAL_PRESENTATION_CUTOFF)
    if features.empty:
        raise Phase2EvaluationError("No 2026 hitter reached the 100-BBE cutoff")
    features["horizon"] = PRIMARY_HORIZON

    # Benchmarks: the frozen ladder, with shrinkage fit on permitted
    # development seasons only and re-proved causal against 2026.
    realized_fit = fit_shrinkage(prior_events, value_column=OBSERVED_COLUMN, quantity="realized")
    deserved_fit = fit_shrinkage(prior_events, value_column=DESERVED_COLUMN, quantity="deserved")
    assert_fit_is_causal(realized_fit, 2026)
    assert_fit_is_causal(deserved_fit, 2026)
    with_benchmarks = build_baseline_predictions(
        features, prior_events=prior_events, predicted_season=2026
    )

    # The frozen Model D ridge, refit on permitted development seasons at the
    # FROZEN alpha. Alpha is never retuned, and 2026 never enters the fit.
    ridge_fit = fit_ridge(development_focal, model_name=frozen["model"], alpha=frozen["alpha"])
    with_benchmarks[FORECAST_COLUMN] = ridge_predict(ridge_fit, with_benchmarks)

    boundary = cutoff_boundary_events(ordered, FOCAL_PRESENTATION_CUTOFF)
    forecasts = with_benchmarks.merge(boundary, on=["batter", "season"], how="left")
    forecasts = forecasts.merge(
        completion[
            [
                "batter",
                "season",
                "n_resolved_bbe_season",
                "completed_horizon",
                "resolved_bbe_since_cutoff",
            ]
        ],
        on=["batter", "season"],
        how="left",
    )
    forecasts["batter_name"] = _attach_names(snapshot, forecasts["batter"])
    forecasts["snapshot_label"] = snapshot.label
    forecasts["snapshot_data_through"] = snapshot.data_through_date
    forecasts["cutoff"] = FOCAL_PRESENTATION_CUTOFF
    forecasts["horizon"] = PRIMARY_HORIZON

    audit = {
        "contact_stage_projection": projection_audit,
        "resolved_event_exclusion": exclusion,
        "n_events_2026": int(len(ledger)),
        "n_resolved_events_2026": int(len(resolved)),
        "n_hitters_reaching_100_resolved_bbe": int(completion["reached_cutoff"].sum()),
        "n_hitters_reaching_200_resolved_bbe": int(completion["completed_horizon"].sum()),
        "n_completed_100_to_100_windows": int(completion["completed_horizon"].sum()),
        "n_pending_forecasts": int(
            (completion["reached_cutoff"] & ~completion["completed_horizon"]).sum()
        ),
        "shrinkage": {
            "realized": shrinkage_fit_record(realized_fit),
            "deserved": shrinkage_fit_record(deserved_fit),
            "fit_seasons_strictly_earlier_than_2026": True,
        },
        "ridge_fit": ridge_fit.as_record(),
        "alpha_source": "frozen ridge manifest; never retuned on 2026",
        "ordered_events": ordered,
    }
    return forecasts, audit


def _attach_names(snapshot: Any, batters: pd.Series) -> pd.Series:
    """Best-effort batter names from the pinned snapshot's own public score."""
    path = snapshot.outputs_dir / "public_score.parquet"
    if not path.exists():
        return pd.Series([None] * len(batters), index=batters.index, dtype=object)
    assert_phase2_path_allowed(path)
    names = pd.read_parquet(path)[["batter_id", "batter_name"]].drop_duplicates("batter_id")
    lookup = dict(zip(names["batter_id"], names["batter_name"], strict=True))
    return batters.map(lookup)


def attach_targets(
    forecasts: pd.DataFrame, ordered: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecasts into completed evaluations and pending, attaching targets.

    Targets come from the FROZEN `build_windows`, which only emits a window
    when the hitter actually accumulated `cutoff + horizon` resolved batted
    balls. A hitter mid-horizon therefore has no target at all, rather than a
    partial one.
    """
    windows = build_windows(
        ordered, cutoffs=(FOCAL_PRESENTATION_CUTOFF,), horizons=(PRIMARY_HORIZON,)
    )
    targets = windows[
        [
            "batter",
            "season",
            "window_id",
            "target_realized_rv_per_100",
            "target_deserved_rv_per_100",
            "target_start_date",
            "target_end_date",
        ]
    ]
    merged = forecasts.merge(targets, on=["batter", "season"], how="left")
    merged["evaluation_status"] = np.where(merged[TARGET].notna(), "completed", "pending")
    completed = merged[merged["evaluation_status"] == "completed"].copy()
    pending = merged[merged["evaluation_status"] == "pending"].copy()

    inconsistent = int((pending[TARGET].notna()).sum())
    if inconsistent:
        raise Phase2EvaluationError(
            f"{inconsistent} pending forecast(s) carry a target; a hitter mid-horizon "
            "must never be evaluated on a partial window"
        )
    return completed, pending


# --------------------------------------------------------------------------
# Metrics, bootstrap, and the prespecified classification
# --------------------------------------------------------------------------


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")


def _full_metrics(frame: pd.DataFrame, *, predictor: str) -> dict[str, Any]:
    measured = evaluate_predictor(frame, target=TARGET, predictor=predictor)
    balanced_mae, balanced_rmse, n_batters = batter_balanced_errors(
        frame, target=TARGET, predictor=predictor
    )
    record = measured.as_record()
    record.update(
        {
            "r_squared": _r_squared(
                frame[TARGET].to_numpy(dtype=float),
                frame[predictor].to_numpy(dtype=float),
            ),
            "batter_balanced_mae": balanced_mae,
            "batter_balanced_rmse": balanced_rmse,
            "n_unique_batters": n_batters,
        }
    )
    return record


def classify_result(delta_mae: float, ci_lower: float, ci_upper: float) -> dict[str, Any]:
    """Apply the prespecified classification. MAE decides; nothing else may.

    The four cases are mutually exclusive and were fixed before any 2026 number
    existed. Secondary metrics are reported but cannot move a result between
    classes.
    """
    below_zero = ci_upper < 0.0
    above_zero = ci_lower > 0.0
    if delta_mae < 0 and below_zero:
        key = "established_incremental_success"
    elif delta_mae < 0:
        key = "promising_but_inconclusive"
    elif delta_mae > 0 and above_zero:
        key = "evidence_against_incremental_value"
    else:
        key = "no_evidence_of_incremental_improvement"
    return {
        "classification": key,
        "condition": SUCCESS_CLASSIFICATION[key]["condition"],
        "meaning": SUCCESS_CLASSIFICATION[key]["meaning"],
        "deciding_metric": "MAE",
        "secondary_metrics_may_not_reclassify": True,
        "delta_mae": delta_mae,
        "ci_delta_mae": [ci_lower, ci_upper],
    }


def build_metrics(completed: pd.DataFrame, *, bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Every required 2026 metric, plus the frozen comparison and classification."""
    predictors = [FORECAST_COLUMN, *BENCHMARKS]
    metrics = {name: _full_metrics(completed, predictor=name) for name in predictors}

    delta = paired_delta(
        completed, target=TARGET, reference=PRIMARY_BENCHMARK, challenger=FORECAST_COLUMN
    )
    assert_delta_convention(delta)
    cell = bootstrap["cells"][f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"]["pooled"]
    mae_summary = cell["delta_mae"]
    rmse_summary = cell["delta_rmse"]
    favouring_benchmark = mae_summary["share_replicates_favouring_reference"]

    n_unique = int(completed["batter"].nunique())
    return {
        "evaluation_season": 2026,
        "n_completed_windows": int(len(completed)),
        "n_unique_hitters": n_unique,
        "one_window_per_hitter": bool(n_unique == len(completed)),
        "clustering_note": (
            "At the fixed 100->100 view each hitter contributes exactly one completed "
            "window, so the batter-clustered bootstrap and an ordinary hitter-level "
            "bootstrap are IDENTICAL here. Clustering is not additional evidence."
        ),
        "sign_convention": DELTA_SIGN_CONVENTION,
        "by_predictor": metrics,
        "contact_forecast_vs_shrunk_deserved": {
            "definition": f"MAE({FORECAST_COLUMN}) - MAE({PRIMARY_BENCHMARK})",
            "delta_mae": delta.delta_mae,
            "pct_change_mae": delta.pct_change_mae,
            "delta_rmse": delta.delta_rmse,
            "pct_change_rmse": delta.pct_change_rmse,
            "mae_contact_forecast": delta.mae_challenger,
            "mae_shrunk_deserved": delta.mae_reference,
            "rmse_contact_forecast": delta.rmse_challenger,
            "rmse_shrunk_deserved": delta.rmse_reference,
            "direction_mae": delta.direction_mae,
            "ci_delta_mae": [mae_summary["ci_lower"], mae_summary["ci_upper"]],
            "ci_delta_rmse": [rmse_summary["ci_lower"], rmse_summary["ci_upper"]],
            "ci_crosses_zero": mae_summary["ci_crosses_zero"],
            "share_replicates_favouring_shrunk_deserved": favouring_benchmark,
            "share_replicates_favouring_contact_forecast": (
                None if favouring_benchmark is None else 1.0 - favouring_benchmark
            ),
        },
        "prespecified_classification": classify_result(
            delta.delta_mae, mae_summary["ci_lower"], mae_summary["ci_upper"]
        ),
        "forecast_bias_diagnostics": _bias_diagnostics(completed),
    }


def _bias_diagnostics(completed: pd.DataFrame) -> dict[str, Any]:
    """Descriptive calibration only. Nothing is recalibrated on the basis of it."""
    predicted = completed[FORECAST_COLUMN].to_numpy(dtype=float)
    actual = completed[TARGET].to_numpy(dtype=float)
    error = predicted - actual
    slope, intercept = np.polyfit(predicted, actual, deg=1)
    return {
        "mean_prediction": float(predicted.mean()),
        "mean_realized_target": float(actual.mean()),
        "mean_prediction_error": float(error.mean()),
        "median_prediction_error": float(np.median(error)),
        "sd_prediction": float(predicted.std(ddof=1)),
        "sd_realized_target": float(actual.std(ddof=1)),
        "calibration_regression": {
            "model": "realized ~ intercept + slope * prediction",
            "slope": float(slope),
            "intercept": float(intercept),
            "perfect_calibration_would_be": {"slope": 1.0, "intercept": 0.0},
        },
        "status": "DIAGNOSTIC ONLY -- the frozen forecast is not recalibrated",
    }


def build_stability(completed: pd.DataFrame) -> dict[str, Any]:
    """Is any advantage broad, or carried by a handful of hitters?

    Especially important here: the HGB stage's development advantage turned out
    to be carried by a few extreme players, which only the trimmed analysis
    revealed.
    """
    actual = completed[TARGET].to_numpy(dtype=float)
    benchmark_error = np.abs(completed[PRIMARY_BENCHMARK].to_numpy(dtype=float) - actual)
    forecast_error = np.abs(completed[FORECAST_COLUMN].to_numpy(dtype=float) - actual)
    improvement = benchmark_error - forecast_error
    order = np.argsort(improvement)

    names = completed["batter_name"].to_numpy()

    def _named(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "batter": int(completed["batter"].to_numpy()[i]),
                "batter_name": (None if pd.isna(names[i]) else str(names[i])),
                "error_improvement": float(improvement[i]),
            }
            for i in indices
        ]

    sensitivity: dict[str, Any] = {}
    for k in (5, 10):
        keep = np.sort(order)[np.isin(np.sort(order), order[-k:], invert=True)]
        keep = np.array([i for i in range(len(improvement)) if i not in set(order[-k:])])
        trimmed_delta = float(np.mean(forecast_error[keep]) - np.mean(benchmark_error[keep]))
        sensitivity[f"delta_mae_excluding_{k}_most_improved"] = trimmed_delta
    return {
        "quantity": (
            f"error_improvement = abs_error({PRIMARY_BENCHMARK}) - abs_error({FORECAST_COLUMN})"
        ),
        "sign_meaning": "positive means the Contact Forecast was closer for that hitter",
        "n_hitters": int(len(improvement)),
        "fraction_of_hitters_improved": float(np.mean(improvement > 0)),
        "mean": float(np.mean(improvement)),
        "median": float(np.median(improvement)),
        "p10": float(np.percentile(improvement, 10)),
        "q1": float(np.percentile(improvement, 25)),
        "q3": float(np.percentile(improvement, 75)),
        "p90": float(np.percentile(improvement, 90)),
        "five_largest_improvements": _named(order[-5:][::-1]),
        "five_largest_deteriorations": _named(order[:5]),
        "outlier_sensitivity": {
            **sensitivity,
            "status": (
                "SENSITIVITY ONLY -- these hitters are NOT removed from the official "
                "result. The question is whether the improvement is broad or driven "
                "by outliers."
            ),
        },
    }


def build_distribution_shift(
    forecasts: pd.DataFrame, development_focal: pd.DataFrame, *, features: list[str]
) -> dict[str, Any]:
    """Compare 2026 feature distributions with permitted development data.

    Explanatory only. Nothing is recalibrated, transformed, clipped, dropped or
    retrained on the basis of these numbers.
    """
    rows: list[dict[str, Any]] = []
    for feature in features:
        development = pd.to_numeric(development_focal[feature], errors="coerce")
        prospective = pd.to_numeric(forecasts[feature], errors="coerce")
        dev_clean = development.dropna()
        pro_clean = prospective.dropna()
        dev_sd = float(dev_clean.std(ddof=1)) if len(dev_clean) > 1 else float("nan")
        low, high = float(dev_clean.min()), float(dev_clean.max())
        outside = float(((pro_clean < low) | (pro_clean > high)).mean()) if len(pro_clean) else None
        shift = (
            float((pro_clean.mean() - dev_clean.mean()) / dev_sd)
            if dev_sd and np.isfinite(dev_sd) and dev_sd > 0
            else None
        )
        rows.append(
            {
                "feature": feature,
                "development_mean": float(dev_clean.mean()),
                "development_sd": dev_sd,
                "development_missing_fraction": float(development.isna().mean()),
                "prospective_2026_mean": float(pro_clean.mean()) if len(pro_clean) else None,
                "prospective_2026_sd": (
                    float(pro_clean.std(ddof=1)) if len(pro_clean) > 1 else None
                ),
                "prospective_2026_missing_fraction": float(prospective.isna().mean()),
                "standardized_mean_shift": shift,
                "fraction_outside_development_range": outside,
                "material_shift": bool(shift is not None and abs(shift) >= 0.25),
            }
        )
    flagged = [r["feature"] for r in rows if r["material_shift"]]
    return {
        "status": (
            "EXPLANATORY ONLY -- no recalibration, transformation, clipping, dropping "
            "or retraining follows from these diagnostics"
        ),
        "material_shift_threshold_standardized": 0.25,
        "n_features": len(rows),
        "features_with_material_shift": flagged,
        "n_features_with_material_shift": len(flagged),
        "by_feature": rows,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(
    *,
    outputs_dir: Path = PHASE2_OUTPUTS_DIR,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    data_dir: Path = FORECAST_DATA_DIR,
    snapshot_label: str | None = None,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
) -> dict[str, Path]:
    """Verify, authorize, pin, forecast, evaluate, and write every artifact."""
    assert_phase2_path_allowed(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- step 0: verify BEFORE any 2026 byte is read ----------------------
    chain = verify_freeze_chain(research_dir=research_dir)
    frozen = load_frozen_contract(research_dir=research_dir)
    authorization = build_authorization_record(
        frozen_manifests=chain["manifest_sha256"], resolved_contract=frozen
    )
    authorization["freeze_chain_verification"] = chain
    authorization["record_sha256"] = hash_json(
        {k: v for k, v in authorization.items() if k != "record_sha256"}
    )
    _write_json(outputs_dir / "phase2_authorization.json", authorization)
    logger.info("authorization recorded (%s)", authorization["record_sha256"][:16])

    # --- step 1: pin one snapshot ----------------------------------------
    snapshot = resolve_snapshot(label=snapshot_label)
    logger.info("pinned snapshot %s (data through %s)", snapshot.label, snapshot.data_through_date)

    development_focal, prior_events, development_audit = load_development_training(
        research_dir=research_dir, data_dir=data_dir
    )
    forecasts, forecast_audit = build_2026_forecasts(
        snapshot=snapshot,
        development_focal=development_focal,
        prior_events=prior_events,
        frozen=frozen,
    )
    ordered = forecast_audit.pop("ordered_events")

    snapshot_manifest = {
        "pinned_snapshot": snapshot.as_record(),
        "single_snapshot_used_for_entire_evaluation": True,
        "refreshed_during_analysis": False,
        "generated_or_overwritten_a_production_snapshot": False,
        "deployed": False,
        "counts": {
            "n_events_2026": forecast_audit["n_events_2026"],
            "n_resolved_events_2026": forecast_audit["n_resolved_events_2026"],
            "n_hitters_reaching_100_resolved_bbe": forecast_audit[
                "n_hitters_reaching_100_resolved_bbe"
            ],
            "n_hitters_reaching_200_resolved_bbe": forecast_audit[
                "n_hitters_reaching_200_resolved_bbe"
            ],
            "n_evaluable_completed_100_to_100_windows": forecast_audit[
                "n_completed_100_to_100_windows"
            ],
            "n_forecasts_with_pending_outcomes": forecast_audit["n_pending_forecasts"],
        },
        "contact_stage_projection": forecast_audit["contact_stage_projection"],
        "resolved_event_exclusion": forecast_audit["resolved_event_exclusion"],
        "development_training": development_audit,
        "shrinkage": forecast_audit["shrinkage"],
        "ridge_fit": forecast_audit["ridge_fit"],
    }
    _write_json(outputs_dir / "phase2_2026_snapshot_manifest.json", snapshot_manifest)

    # --- forecasts persisted BEFORE outcomes are attached -----------------
    prediction_columns = [
        "batter",
        "batter_name",
        "season",
        "cutoff",
        "horizon",
        "cutoff_event_id",
        "cutoff_game_date",
        "cutoff_game_pk",
        FORECAST_COLUMN,
        *BENCHMARKS,
        "n_resolved_bbe_season",
        "resolved_bbe_since_cutoff",
        "completed_horizon",
        "snapshot_label",
        "snapshot_data_through",
    ]
    predictions_path = outputs_dir / "phase2_2026_predictions.parquet"
    forecasts[prediction_columns].to_parquet(predictions_path, index=False)
    logger.info("persisted %d forecasts before attaching any outcome", len(forecasts))

    completed, pending = attach_targets(forecasts, ordered)
    completed.to_parquet(outputs_dir / "phase2_2026_completed_evaluation.parquet", index=False)
    pending[prediction_columns].to_parquet(
        outputs_dir / "phase2_2026_pending_predictions.parquet", index=False
    )
    logger.info("%d completed evaluations, %d pending", len(completed), len(pending))

    if completed.empty:
        raise Phase2EvaluationError(
            "No 2026 hitter has completed a 100->100 window in the pinned snapshot"
        )

    completed["cutoff"] = FOCAL_PRESENTATION_CUTOFF
    completed["horizon"] = PRIMARY_HORIZON
    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)
    bootstrap = run_paired_bootstrap(
        completed,
        target=TARGET,
        reference=PRIMARY_BENCHMARK,
        challenger=FORECAST_COLUMN,
        design=design,
    )
    bootstrap["clustering_note"] = (
        "Each hitter contributes exactly one completed window here, so the "
        "batter-clustered and ordinary hitter-level bootstraps are identical."
    )
    benchmark_bootstraps = {
        f"{FORECAST_COLUMN}_vs_{benchmark}": run_paired_bootstrap(
            completed,
            target=TARGET,
            reference=benchmark,
            challenger=FORECAST_COLUMN,
            design=design,
        )
        for benchmark in BENCHMARKS
    }
    _write_json(
        outputs_dir / "phase2_2026_bootstrap.json",
        {"primary": bootstrap, "all_benchmarks": benchmark_bootstraps},
    )

    metrics = build_metrics(completed, bootstrap=bootstrap)
    metrics["pending_forecasts"] = {
        "n_pending": int(len(pending)),
        "note": (
            "These hitters received a forecast but have not completed 100 subsequent "
            "resolved BBE. They are never evaluated on a partial window. A later "
            "evaluation attaches outcomes to these frozen predictions rather than "
            "regenerating forecasts with later information."
        ),
    }
    metrics["provenance"] = {
        "authorization_sha256": authorization["record_sha256"],
        "frozen_stage_manifests": chain["manifest_sha256"],
        "snapshot_label": snapshot.label,
        "snapshot_data_through": snapshot.data_through_date,
        "frozen_contract": frozen,
    }
    _write_json(outputs_dir / "phase2_2026_metrics.json", metrics)

    stability = build_stability(completed)
    _write_json(outputs_dir / "phase2_2026_stability.json", stability)

    ridge_features = list(forecast_audit["ridge_fit"]["features"])
    shift = build_distribution_shift(forecasts, development_focal, features=ridge_features)
    _write_json(outputs_dir / "phase2_2026_distribution_shift.json", shift)

    from forecast.phase2.phase2_report import render_phase2_report

    report_path = outputs_dir / "phase2_2026_report.md"
    report_path.write_text(
        render_phase2_report(
            authorization=authorization,
            snapshot_manifest=snapshot_manifest,
            metrics=metrics,
            stability=stability,
            shift=shift,
        )
    )
    logger.info("wrote %s", report_path)

    _write_provenance(outputs_dir, chain=chain, authorization=authorization)
    return {name: outputs_dir / name for name in REQUIRED_ARTIFACTS}


def _write_provenance(
    outputs_dir: Path, *, chain: dict[str, Any], authorization: dict[str, Any]
) -> None:
    """Hash every Phase 2 artifact and chain it to the frozen development stages."""
    hashes = {
        name: hash_file(outputs_dir / name)
        for name in REQUIRED_ARTIFACTS
        if (outputs_dir / name).exists()
    }
    record = {
        "stage": "phase2_2026",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact_sha256": hashes,
        "authorization_sha256": authorization["record_sha256"],
        "provenance_chain": chain["manifest_sha256"],
        "sealed_seasons_never_read": [2025],
        "evaluation_season": 2026,
        "deployed": False,
    }
    record["manifest_sha256"] = hash_json(
        {k: v for k, v in record.items() if k != "generated_at_utc"}
    )
    _write_json(outputs_dir / "phase2_provenance_manifest.json", record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=PHASE2_OUTPUTS_DIR)
    parser.add_argument("--research-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=FORECAST_DATA_DIR)
    parser.add_argument("--snapshot-label", type=str, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_BOOTSTRAP_ALPHA)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    written = run(
        outputs_dir=args.outputs_dir,
        research_dir=args.research_dir,
        data_dir=args.data_dir,
        snapshot_label=args.snapshot_label,
        reps=args.bootstrap_reps,
        seed=args.seed,
        alpha=args.alpha,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
