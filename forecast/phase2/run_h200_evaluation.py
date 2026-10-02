"""Contact Forecast H=200: the held-out 2026 evaluation of the frozen specification.

Runs the ALREADY-FROZEN H=200 specification (`forecast.phase2.h200_spec`)
against the pinned 2026 snapshot, once. Nothing here tunes, selects,
recalibrates, trims or deploys, and no 2026 number feeds back into any frozen
element.

## Order of operations, which is the control

1. Verify the R1 / ridge / HGB chain AND the h200_spec freeze, requiring zero
   drift, before a single 2026 outcome is read.
2. Load the frozen H=200 contract -- alpha, feature list, coefficients,
   intercept, standardization, target, benchmarks, metrics, classification --
   FROM the verified manifest rather than restating it.
3. Re-derive the model by the frozen procedure and prove it reproduces the
   sealed coefficients, intercept and scaler exactly. A mismatch stops the run.
4. Pin the same immutable 2026 snapshot, build features from each hitter's
   FIRST 100 resolved eligible BBE, and persist every forecast BEFORE any
   outcome is attached.
5. Attach the next-200-BBE target only to hitters who actually completed it.
6. Compute the frozen metrics, apply the frozen four-way classification
   mechanically from delta_MAE and its CI, and hash the whole result.

## What this module may not do

Retune alpha, refit on 2026, remove hitters, change features, change the
target, change benchmarks, reinterpret the threshold, read 2025, test another
model family, or touch the dashboard. The classification is applied by
`run_phase2_evaluation.classify_result` -- the same code the H=100 evaluation
used -- so the rule cannot be reinterpreted here.

The frozen H=100 Contact Forecast and its Phase 2 result are untouched. H=200
is a REFIT and a separate specification; the two results are never pooled.
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
    FORECAST_DATA_DIR,
    FORECAST_OUTPUTS_DIR,
)
from forecast.freeze_r1 import hash_file, hash_json
from forecast.metrics import DELTA_SIGN_CONVENTION, assert_delta_convention, paired_delta
from forecast.phase2.h200_disclosure import (
    DISCLOSURE_PROTECTED_ELEMENTS,
    H200_2026_EXPOSURE_DISCLOSURE,
    render_disclosure_markdown,
)
from forecast.phase2.h200_spec import (
    BENCHMARKS,
    H200_CUTOFF,
    H200_HORIZON,
    H200_MODEL,
    H200_OUTPUTS_DIR,
    H200_TARGET,
    PRIMARY_BENCHMARK,
    build_development_windows,
    h200_freeze_spec,
)
from forecast.phase2.phase2_config import (
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
from forecast.phase2.run_h200_report import render_h200_report
from forecast.phase2.run_phase2_evaluation import (
    FORECAST_COLUMN,
    _attach_names,
    _bias_diagnostics,
    _full_metrics,
    _json_default,
    build_distribution_shift,
    build_stability,
    classify_result,
    verify_freeze_chain,
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

H200_EVALUATION_OUTPUTS_DIR = H200_OUTPUTS_DIR
CELL_KEY = f"K{H200_CUTOFF}_H{H200_HORIZON}"

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "h200_2026_authorization.json",
    "h200_2026_exposure_disclosure.json",
    "h200_2026_snapshot_manifest.json",
    "h200_2026_predictions.parquet",
    "h200_2026_completed_evaluation.parquet",
    "h200_2026_pending_predictions.parquet",
    "h200_2026_metrics.json",
    "h200_2026_bootstrap.json",
    "h200_2026_stability.json",
    "h200_2026_distribution_shift.json",
    "h200_2026_survivorship_comparison.json",
    "h200_2026_report.md",
)


class H200EvaluationError(RuntimeError):
    """Raised when the H=200 evaluation cannot run exactly as frozen."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_phase2_path_allowed(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------
# Step 0: verify the H=200 freeze before opening any 2026 outcome
# --------------------------------------------------------------------------


def verify_h200_freeze(*, outputs_dir: Path = H200_OUTPUTS_DIR) -> dict[str, Any]:
    """Verify the H=200 specification freeze and refuse to proceed on any drift.

    Raises:
        H200EvaluationError: If the sealed specification has drifted.
    """
    from forecast.stage_freeze import verify_stage

    result = verify_stage(h200_freeze_spec(), outputs_dir=outputs_dir)
    if not result["matches_freeze"]:
        raise H200EvaluationError(
            f"The H=200 specification has drifted: {result['drift']}. 2026 outcomes "
            "are not opened against a specification that no longer matches what was "
            "sealed."
        )
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "zero_drift_confirmed": True,
        "manifest_sha256": result["manifest_sha256"],
        "detail": result,
    }


def load_h200_contract(*, outputs_dir: Path = H200_OUTPUTS_DIR) -> dict[str, Any]:
    """Read every frozen H=200 element out of the verified specification.

    Loaded rather than restated, for the same reason Phase 2 loaded the H=100
    contract from its manifest: a retyped value could drift from what was
    sealed, a value read from a verified artifact cannot.

    Raises:
        H200EvaluationError: If the specification is not the expected frozen
            H=200 refit, or records an outcome as already opened.
    """
    spec = json.loads((outputs_dir / "h200_specification.json").read_text())
    model = spec["forecast_model_being_evaluated"]

    if spec["provenance_answer"]["answer"] != "REFIT":
        raise H200EvaluationError(
            "The frozen specification no longer records the H=200 model as a REFIT; "
            "its provenance claim must hold before a result may be attached to it."
        )
    if model["architecture"] != H200_MODEL:
        raise H200EvaluationError(
            f"The frozen architecture is {model['architecture']!r}, not {H200_MODEL!r}."
        )
    if spec["season_discipline"]["prospective_outcomes_opened"]:
        raise H200EvaluationError(
            "The frozen specification already records 2026 outcomes as opened; this "
            "evaluation runs exactly once."
        )
    if spec["metrics"]["secondary_may_reclassify"]:
        raise H200EvaluationError(
            "The frozen specification no longer forbids secondary metrics from "
            "reclassifying the result."
        )
    if spec["success_classification"] != SUCCESS_CLASSIFICATION:
        raise H200EvaluationError(
            "The frozen four-way classification no longer matches the prespecified "
            "rule; the success threshold may not be reinterpreted."
        )

    return {
        "model": model["architecture"],
        "alpha": float(model["alpha"]),
        "features": list(model["features"]),
        "coefficients": dict(model["coefficients"]),
        "intercept": float(model["intercept"]),
        "scaler_mean": list(model["scaler_mean"]),
        "scaler_scale": list(model["scaler_scale"]),
        "trained_on_seasons": list(model["trained_on_seasons"]),
        "n_train": int(model["n_train"]),
        "target": spec["design"]["target_column"],
        "target_definition": spec["design"]["target"],
        "cutoff": H200_CUTOFF,
        "horizon": H200_HORIZON,
        "benchmarks": list(spec["benchmarks"]["all"]),
        "primary_benchmark": spec["benchmarks"]["primary"],
        "primary_metric": spec["metrics"]["primary"],
        "sign_convention": spec["metrics"]["sign_convention"],
        "bootstrap": spec["bootstrap"],
        "success_classification": spec["success_classification"],
        "exploratory_power_estimate": spec["exploratory_power_estimate"],
        "spec_version": spec["spec_version"],
    }


def assert_disclosure_elements_frozen(contract: dict[str, Any]) -> dict[str, Any]:
    """Check the disclosure's claim: every protected element came from the seal.

    The disclosure asserts that the 2026 readiness observations modified no
    frozen element. That is checkable rather than merely stated: each element
    below is read out of the verified specification, so anything the readiness
    observations could have touched would show as drift in `verify_h200_freeze`.

    Raises:
        H200EvaluationError: If a protected element is absent from the contract.
    """
    sources = {
        "Model D architecture": contract["model"],
        "feature list": contract["features"],
        "alpha": contract["alpha"],
        "preprocessing procedure": contract["scaler_mean"],
        "target definition": contract["target"],
        "benchmarks": contract["benchmarks"],
        "primary metric": contract["primary_metric"],
        "CI procedure": contract["bootstrap"],
        "success classification": contract["success_classification"],
    }
    missing = [name for name in DISCLOSURE_PROTECTED_ELEMENTS if not sources.get(name)]
    if missing:
        raise H200EvaluationError(
            f"The disclosure protects {missing}, which the frozen specification does "
            "not supply; the claim that they were unmodified cannot be checked."
        )
    return {
        "protected_elements": list(DISCLOSURE_PROTECTED_ELEMENTS),
        "all_loaded_from_the_verified_freeze": True,
        "restated_in_this_module": False,
        "disclosure_sha256": hash_json(dict(H200_2026_EXPOSURE_DISCLOSURE)),
    }


# --------------------------------------------------------------------------
# The frozen model, re-derived and proved identical
# --------------------------------------------------------------------------


def rebuild_frozen_model(contract: dict[str, Any], *, data_dir: Path = FORECAST_DATA_DIR) -> Any:
    """Re-derive the sealed H=200 fit and prove it reproduces the frozen numbers.

    The fit is re-run by the frozen procedure at the FROZEN alpha on the
    permitted development seasons, then compared coefficient by coefficient
    with what was sealed. Nothing is selected, and alpha is never re-tuned.

    Raises:
        H200EvaluationError: If the re-derived fit differs from the seal.
    """
    table, _ = build_development_windows(data_dir=data_dir)
    train = table[table["season"].astype(int).isin(PHASE2_PERMITTED_TRAINING_SEASONS)]
    assert_phase2_seasons_allowed(sorted(int(s) for s in train["season"].unique()))
    fit = fit_ridge(
        train, model_name=contract["model"], alpha=contract["alpha"], target=contract["target"]
    )

    mismatches: dict[str, Any] = {}
    if list(fit.features) != contract["features"]:
        mismatches["features"] = {"frozen": contract["features"], "rebuilt": list(fit.features)}
    if int(fit.n_train) != contract["n_train"]:
        mismatches["n_train"] = {"frozen": contract["n_train"], "rebuilt": int(fit.n_train)}
    if not np.isclose(fit.intercept, contract["intercept"], rtol=0, atol=1e-9):
        mismatches["intercept"] = {"frozen": contract["intercept"], "rebuilt": fit.intercept}
    for name, frozen_value in contract["coefficients"].items():
        rebuilt = fit.coefficients.get(name)
        if rebuilt is None or not np.isclose(rebuilt, frozen_value, rtol=0, atol=1e-9):
            mismatches.setdefault("coefficients", {})[name] = {
                "frozen": frozen_value,
                "rebuilt": rebuilt,
            }
    for label, frozen_seq, rebuilt_seq in (
        ("scaler_mean", contract["scaler_mean"], fit.scaler_mean),
        ("scaler_scale", contract["scaler_scale"], fit.scaler_scale),
    ):
        if not np.allclose(np.asarray(frozen_seq), np.asarray(rebuilt_seq), rtol=0, atol=1e-9):
            mismatches[label] = {"frozen": frozen_seq, "rebuilt": rebuilt_seq}
    if mismatches:
        raise H200EvaluationError(
            f"The re-derived H=200 fit does not reproduce the frozen model: {mismatches}"
        )
    logger.info(
        "re-derived the frozen H=200 fit exactly (alpha=%s, n_train=%d)",
        contract["alpha"],
        fit.n_train,
    )
    return fit, table


# --------------------------------------------------------------------------
# Forecast generation
# --------------------------------------------------------------------------


def build_2026_h200_forecasts(
    *, snapshot: Any, prior_events: pd.DataFrame, fit: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Every 2026 H=200 forecast, generated before any outcome is attached."""
    ledger, projection_audit = load_contact_stage_ledger(snapshot)
    assert_no_full_telescoping_columns(ledger)

    resolved, exclusion = select_scored_eligible_events(ledger)
    ordered = order_events_for_phase2(resolved)
    completion = horizon_completion(ordered, cutoff=H200_CUTOFF, horizon=H200_HORIZON)

    features = build_window_features(ordered, H200_CUTOFF)
    if features.empty:
        raise H200EvaluationError("No 2026 hitter reached the 100-BBE cutoff")
    features["horizon"] = H200_HORIZON

    realized_fit = fit_shrinkage(prior_events, value_column=OBSERVED_COLUMN, quantity="realized")
    deserved_fit = fit_shrinkage(prior_events, value_column=DESERVED_COLUMN, quantity="deserved")
    assert_fit_is_causal(realized_fit, 2026)
    assert_fit_is_causal(deserved_fit, 2026)
    with_benchmarks = build_baseline_predictions(
        features, prior_events=prior_events, predicted_season=2026
    )
    with_benchmarks[FORECAST_COLUMN] = ridge_predict(fit, with_benchmarks)

    boundary = cutoff_boundary_events(ordered, H200_CUTOFF)
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
    forecasts["cutoff"] = H200_CUTOFF
    forecasts["horizon"] = H200_HORIZON

    audit = {
        "contact_stage_projection": projection_audit,
        "resolved_event_exclusion": exclusion,
        "n_events_2026": int(len(ledger)),
        "n_resolved_events_2026": int(len(resolved)),
        "n_hitters_reaching_100_resolved_bbe": int(completion["reached_cutoff"].sum()),
        "n_hitters_reaching_300_resolved_bbe": int(completion["completed_horizon"].sum()),
        "n_completed_100_to_200_windows": int(completion["completed_horizon"].sum()),
        "n_pending_forecasts": int(
            (completion["reached_cutoff"] & ~completion["completed_horizon"]).sum()
        ),
        "shrinkage": {
            "realized": shrinkage_fit_record(realized_fit),
            "deserved": shrinkage_fit_record(deserved_fit),
            "fit_seasons_strictly_earlier_than_2026": True,
        },
        "ridge_fit": fit.as_record(),
        "alpha_source": "frozen h200 specification; never retuned on 2026",
    }
    return forecasts, ordered, audit


def attach_h200_targets(
    forecasts: pd.DataFrame, ordered: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into completed and pending, attaching the next-200-BBE target.

    `build_windows` only emits a window when the hitter actually accumulated
    `cutoff + horizon` resolved batted balls, so a hitter mid-horizon has no
    target rather than a partial one.
    """
    windows = build_windows(ordered, cutoffs=(H200_CUTOFF,), horizons=(H200_HORIZON,))
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
    merged["evaluation_status"] = np.where(merged[H200_TARGET].notna(), "completed", "pending")
    completed = merged[merged["evaluation_status"] == "completed"].copy()
    pending = merged[merged["evaluation_status"] == "pending"].copy()

    inconsistent = int(pending[H200_TARGET].notna().sum())
    if inconsistent:
        raise H200EvaluationError(
            f"{inconsistent} pending forecast(s) carry a target; a hitter mid-horizon "
            "must never be evaluated on a partial window"
        )
    return completed, pending


# --------------------------------------------------------------------------
# Metrics and the prespecified classification
# --------------------------------------------------------------------------


def build_h200_metrics(completed: pd.DataFrame, *, bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Every frozen H=200 metric, plus the mechanically applied classification."""
    predictors = [FORECAST_COLUMN, *BENCHMARKS]
    metrics = {name: _full_metrics(completed, predictor=name) for name in predictors}

    delta = paired_delta(
        completed, target=H200_TARGET, reference=PRIMARY_BENCHMARK, challenger=FORECAST_COLUMN
    )
    assert_delta_convention(delta)
    cell = bootstrap["cells"][CELL_KEY]["pooled"]
    mae_summary = cell["delta_mae"]
    rmse_summary = cell["delta_rmse"]
    favouring_benchmark = mae_summary["share_replicates_favouring_reference"]

    n_unique = int(completed["batter"].nunique())
    return {
        "evaluation_season": 2026,
        "cutoff": H200_CUTOFF,
        "horizon": H200_HORIZON,
        "n_completed_windows": int(len(completed)),
        "n_unique_hitters": n_unique,
        "one_window_per_hitter": bool(n_unique == len(completed)),
        "clustering_note": (
            "At the fixed 100->200 view each hitter contributes exactly one completed "
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


# --------------------------------------------------------------------------
# Survivorship: development versus the held-out season
# --------------------------------------------------------------------------


def build_survivorship_comparison(*, outputs_dir: Path = H200_OUTPUTS_DIR) -> dict[str, Any]:
    """Compare the frozen development and 2026 cutoff-side survivorship audits.

    Both audits were produced before any outcome was opened; this only puts
    them side by side. Nothing is reweighted, matched or adjusted on the basis
    of it, and no hitter is added to or removed from the evaluation.
    """
    development = json.loads((outputs_dir / "h200_development_survivorship.json").read_text())
    prospective = json.loads((outputs_dir / "h200_prospective_cutoff_audit.json").read_text())
    pooled = development["pooled"]

    by_feature = {row["feature"]: row for row in prospective["cutoff_side_differences"]}
    dev_by_feature = {row["feature"]: row for row in pooled["cutoff_side_differences"]}
    side_by_side = [
        {
            "feature": feature,
            "development_standardized_difference": dev_by_feature[feature][
                "standardized_difference"
            ],
            "prospective_2026_standardized_difference": by_feature.get(feature, {}).get(
                "standardized_difference"
            ),
        }
        for feature in dev_by_feature
    ]
    side_by_side.sort(key=lambda row: -abs(row["prospective_2026_standardized_difference"] or 0.0))
    return {
        "status": (
            "DESCRIPTIVE ONLY -- no reweighting, matching or adjustment follows from "
            "this comparison, and no hitter is added to or removed from the evaluation"
        ),
        "development_2022_2024_pooled": {
            "n_reaching_100_resolved_bbe": pooled["n_reaching_100_resolved_bbe"],
            "n_subsequently_reaching_300_total_resolved_bbe": pooled[
                "n_subsequently_reaching_300_total_resolved_bbe"
            ],
            "completion_rate": pooled["completion_rate"],
            "largest_standardized_differences": pooled["largest_standardized_differences"],
        },
        "prospective_2026": {
            "snapshot_label": prospective["snapshot_label"],
            "snapshot_data_through": prospective["snapshot_data_through"],
            "n_reaching_100_resolved_bbe": prospective["n_reaching_100_resolved_bbe"],
            "n_subsequently_reaching_300_total_resolved_bbe": prospective[
                "n_subsequently_reaching_300_total_resolved_bbe"
            ],
            "completion_rate": prospective["completion_rate"],
            "largest_standardized_differences": prospective["largest_standardized_differences"],
        },
        "completion_rate_gap_2026_minus_development": (
            float(prospective["completion_rate"]) - float(pooled["completion_rate"])
        ),
        "cutoff_side_differences_side_by_side": side_by_side,
        "generalization_note": pooled["generalization_note"],
        "development_by_season": {
            season: {
                "n_reaching_100_resolved_bbe": audit["n_reaching_100_resolved_bbe"],
                "n_subsequently_reaching_300_total_resolved_bbe": audit[
                    "n_subsequently_reaching_300_total_resolved_bbe"
                ],
                "completion_rate": audit["completion_rate"],
            }
            for season, audit in development["by_season"].items()
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(
    *,
    outputs_dir: Path = H200_OUTPUTS_DIR,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    data_dir: Path = FORECAST_DATA_DIR,
    snapshot_label: str | None = None,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
) -> dict[str, Path]:
    """Verify, authorize, pin, forecast, evaluate, classify and hash. Once."""
    assert_phase2_path_allowed(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- step 0: verify BEFORE any 2026 outcome is read -------------------
    chain = verify_freeze_chain(research_dir=research_dir)
    h200_chain = verify_h200_freeze(outputs_dir=outputs_dir)
    contract = load_h200_contract(outputs_dir=outputs_dir)
    disclosure_check = assert_disclosure_elements_frozen(contract)

    # The disclosure is recorded (again, idempotently) before the run, so the
    # statement of what had been seen cannot be assembled after the answer.
    disclosure = dict(H200_2026_EXPOSURE_DISCLOSURE)
    disclosure["disclosure_sha256"] = hash_json(dict(H200_2026_EXPOSURE_DISCLOSURE))
    disclosure["checked_against_the_frozen_specification"] = disclosure_check
    _write_json(outputs_dir / "h200_2026_exposure_disclosure.json", disclosure)
    (outputs_dir / "h200_2026_exposure_disclosure.md").write_text(render_disclosure_markdown())

    authorization = build_authorization_record(
        frozen_manifests={**chain["manifest_sha256"], "h200_spec": h200_chain["manifest_sha256"]},
        resolved_contract=contract,
    )
    authorization["freeze_chain_verification"] = chain
    authorization["h200_freeze_verification"] = h200_chain
    authorization["exposure_disclosure"] = disclosure
    authorization["evaluation"] = {
        "specification": contract["spec_version"],
        "cutoff": H200_CUTOFF,
        "horizon": H200_HORIZON,
        "is_the_frozen_h100_contact_forecast": False,
        "h100_phase2_result_modified": False,
        "runs_once": True,
    }
    authorization["record_sha256"] = hash_json(
        {k: v for k, v in authorization.items() if k != "record_sha256"}
    )
    _write_json(outputs_dir / "h200_2026_authorization.json", authorization)
    logger.info("authorization recorded (%s)", authorization["record_sha256"][:16])

    # --- step 1: re-derive the frozen model, then pin one snapshot --------
    fit, development_table = rebuild_frozen_model(contract, data_dir=data_dir)
    snapshot = resolve_snapshot(label=snapshot_label)
    logger.info("pinned snapshot %s (data through %s)", snapshot.label, snapshot.data_through_date)

    prior_events: list[pd.DataFrame] = []
    for season in PHASE2_PERMITTED_TRAINING_SEASONS:
        path = data_dir / f"walk_forward_ledger_{season}.parquet"
        assert_phase2_path_allowed(path)
        ledger = pd.read_parquet(path)
        prior_events.append(ledger[ledger[OBSERVED_COLUMN].notna()])
    events = pd.concat(prior_events, ignore_index=True)
    assert_phase2_seasons_allowed(sorted(int(s) for s in events["season"].unique()))

    forecasts, ordered, forecast_audit = build_2026_h200_forecasts(
        snapshot=snapshot, prior_events=events, fit=fit
    )

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
            "n_hitters_reaching_300_resolved_bbe": forecast_audit[
                "n_hitters_reaching_300_resolved_bbe"
            ],
            "n_evaluable_completed_100_to_200_windows": forecast_audit[
                "n_completed_100_to_200_windows"
            ],
            "n_forecasts_with_pending_outcomes": forecast_audit["n_pending_forecasts"],
        },
        "contact_stage_projection": forecast_audit["contact_stage_projection"],
        "resolved_event_exclusion": forecast_audit["resolved_event_exclusion"],
        "development_training": {
            "permitted_training_seasons": list(PHASE2_PERMITTED_TRAINING_SEASONS),
            "forbidden_seasons_present": [],
            "n_development_windows": int(len(development_table)),
            "n_prior_events": int(len(events)),
        },
        "shrinkage": forecast_audit["shrinkage"],
        "ridge_fit": forecast_audit["ridge_fit"],
    }
    _write_json(outputs_dir / "h200_2026_snapshot_manifest.json", snapshot_manifest)

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
    forecasts[prediction_columns].to_parquet(
        outputs_dir / "h200_2026_predictions.parquet", index=False
    )
    logger.info("persisted %d forecasts before attaching any outcome", len(forecasts))

    completed, pending = attach_h200_targets(forecasts, ordered)
    completed.to_parquet(outputs_dir / "h200_2026_completed_evaluation.parquet", index=False)
    pending[prediction_columns].to_parquet(
        outputs_dir / "h200_2026_pending_predictions.parquet", index=False
    )
    logger.info("%d completed evaluations, %d pending", len(completed), len(pending))

    if completed.empty:
        raise H200EvaluationError(
            "No 2026 hitter has completed a 100->200 window in the pinned snapshot"
        )

    completed["cutoff"] = H200_CUTOFF
    completed["horizon"] = H200_HORIZON
    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)
    bootstrap = run_paired_bootstrap(
        completed,
        target=H200_TARGET,
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
            target=H200_TARGET,
            reference=benchmark,
            challenger=FORECAST_COLUMN,
            design=design,
        )
        for benchmark in BENCHMARKS
    }
    _write_json(
        outputs_dir / "h200_2026_bootstrap.json",
        {"primary": bootstrap, "all_benchmarks": benchmark_bootstraps},
    )

    metrics = build_h200_metrics(completed, bootstrap=bootstrap)
    metrics["pending_forecasts"] = {
        "n_pending": int(len(pending)),
        "note": (
            "These hitters received a forecast but have not completed 200 subsequent "
            "resolved BBE. They are never evaluated on a partial window. A later "
            "evaluation attaches outcomes to these frozen predictions rather than "
            "regenerating forecasts with later information."
        ),
    }
    metrics["exposure_disclosure"] = disclosure
    metrics["provenance"] = {
        "authorization_sha256": authorization["record_sha256"],
        "frozen_stage_manifests": chain["manifest_sha256"],
        "h200_spec_manifest_sha256": h200_chain["manifest_sha256"],
        "snapshot_label": snapshot.label,
        "snapshot_data_through": snapshot.data_through_date,
        "frozen_contract": contract,
    }
    _write_json(outputs_dir / "h200_2026_metrics.json", metrics)

    stability = build_stability(completed)
    _write_json(outputs_dir / "h200_2026_stability.json", stability)

    shift = build_distribution_shift(
        forecasts, development_table, features=list(contract["features"])
    )
    _write_json(outputs_dir / "h200_2026_distribution_shift.json", shift)

    survivorship = build_survivorship_comparison(outputs_dir=outputs_dir)
    _write_json(outputs_dir / "h200_2026_survivorship_comparison.json", survivorship)

    report_path = outputs_dir / "h200_2026_report.md"
    report_path.write_text(
        render_h200_report(
            authorization=authorization,
            snapshot_manifest=snapshot_manifest,
            metrics=metrics,
            stability=stability,
            shift=shift,
            survivorship=survivorship,
            disclosure_markdown=render_disclosure_markdown(),
        )
    )
    logger.info("wrote %s", report_path)

    _write_provenance(outputs_dir, chain=chain, h200_chain=h200_chain, authorization=authorization)
    return {name: outputs_dir / name for name in REQUIRED_ARTIFACTS}


def _write_provenance(
    outputs_dir: Path,
    *,
    chain: dict[str, Any],
    h200_chain: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    """Hash every H=200 result artifact and chain it to the frozen stages."""
    hashes = {
        name: hash_file(outputs_dir / name)
        for name in REQUIRED_ARTIFACTS
        if (outputs_dir / name).exists()
    }
    record = {
        "stage": "h200_2026_evaluation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact_sha256": hashes,
        "authorization_sha256": authorization["record_sha256"],
        "provenance_chain": {
            **chain["manifest_sha256"],
            "h200_spec": h200_chain["manifest_sha256"],
        },
        "sealed_seasons_never_read": [2025],
        "evaluation_season": 2026,
        "cutoff": H200_CUTOFF,
        "horizon": H200_HORIZON,
        "deployed": False,
        "frozen_h100_phase2_result_modified": False,
        "result_is_final_as_computed": True,
    }
    record["manifest_sha256"] = hash_json(
        {k: v for k, v in record.items() if k != "generated_at_utc"}
    )
    _write_json(outputs_dir / "h200_2026_result_manifest.json", record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=H200_OUTPUTS_DIR)
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
        logger.info("%s -> %s", name, path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
