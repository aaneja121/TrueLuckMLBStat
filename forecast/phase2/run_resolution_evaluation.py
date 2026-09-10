"""Contact Forecast: the end-of-season 2026 resolution pass.

Executes the FROZEN resolution specification
(`forecast.phase2.resolution_spec`) once, after the 2026 regular season has
ended. It attaches outcomes to predictions that were sealed at the 2026-09-01
first look; it never regenerates a forecast, never refits, never trims, and
never deploys.

## This module is written before the data exists, on purpose

The specification was frozen on 2026-09-05 and the season does not end until
2026-09-27. Writing the runner now means the code cannot be shaped by a result
nobody has seen: every branch, cohort split and reporting decision is fixed
while the outcomes are still unopened.

`assert_season_has_ended` enforces that directly. The pass refuses to run
before the verified regular-season end date, and refuses a snapshot whose
data-through date does not reach it. There is no flag that overrides either.

## The gate, in the order the specification fixes it

1. The 2026 regular season has ended (verified date, not an assumption).
2. The pinned snapshot reaches that date.
3. The resolution specification verifies with zero drift.
4. R1, ridge, HGB and h200_spec verify with zero drift.
5. Both frozen result manifests verify and both sealed ledgers re-hash.
6. Exactly one end-of-season snapshot is pinned, integrity-verified, read-only.
7. The maintainer authorizes the second look in the moment -- an explicit
   argument with no default, so the pass cannot run by accident or by cron.

## Cohorts

INCREMENTAL is primary and the only cohort the four-way classification
decides: hitters pending at the first look who completed the horizon by
season's end. Their outcomes have never been examined.

FULL_SEASON is secondary and carries a mandatory NOT INDEPENDENT label. It
reuses every already-observed window, so it is reported as an estimate with an
interval and is NEVER classified -- see `classification_applies_to` in the
frozen specification (amendment 2), and `COHORT_CLASSIFICATION_HISTORY`.

NEVER_COMPLETED is reported and never evaluated: no partial windows, no
imputation, no removal from the ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prospective.prospective_config import PROSPECTIVE_2026_SEASON_END_DATE

from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FORECAST_DATA_DIR,
    FORECAST_OUTPUTS_DIR,
)
from forecast.freeze_r1 import hash_file, hash_json
from forecast.metrics import DELTA_SIGN_CONVENTION, assert_delta_convention, paired_delta
from forecast.phase2.h200_spec import H200_OUTPUTS_DIR, h200_freeze_spec
from forecast.phase2.phase2_config import assert_phase2_path_allowed
from forecast.phase2.phase2_windows import horizon_completion, order_events_for_phase2
from forecast.phase2.resolution_spec import (
    FROZEN_PREDICTION_LEDGERS,
    RESOLUTION_OUTPUTS_DIR,
    SEASON_END_VERIFICATION,
    resolution_freeze_spec,
)
from forecast.phase2.run_phase2_evaluation import (
    BENCHMARKS,
    FORECAST_COLUMN,
    PRIMARY_BENCHMARK,
    TARGET,
    _bias_diagnostics,
    _full_metrics,
    _json_default,
    build_distribution_shift,
    build_stability,
    classify_result,
    load_development_training,
    verify_freeze_chain,
)
from forecast.phase2.snapshot import (
    assert_no_full_telescoping_columns,
    load_contact_stage_ledger,
    resolve_snapshot,
)
from forecast.windows import build_windows, select_scored_eligible_events

logger = logging.getLogger(__name__)

#: The prediction columns carried through from the sealed ledger. Every one of
#: these is READ from the frozen parquet and never recomputed.
SEALED_PREDICTION_COLUMNS: tuple[str, ...] = (FORECAST_COLUMN, *BENCHMARKS)

#: Amendment 2 resolved a contradiction this runner surfaced: the frozen
#: specification said both `classification_applies_to = 'the incremental cohort
#: only'` and `cohorts.full_season.four_way_classification_applied = True`.
#: Kept as a historical record. The runner no longer hardcodes the resolution --
#: it reads each cohort's flag from the frozen specification, and
#: `assert_cohort_classification_is_consistent` refuses to proceed if the two
#: fields ever disagree again.
COHORT_CLASSIFICATION_HISTORY: dict[str, Any] = {
    "was_a_contradiction": True,
    "resolved_by": "resolution_spec amendment 2",
    "resolved_as": "the incremental cohort only",
    "amended_before_any_outcome_was_opened": True,
    "runner_reads_the_flag_from_the_specification": True,
    "full_season_receives": (
        "a delta and a paired interval under the NOT INDEPENDENT label, and no classification label"
    ),
}


def assert_cohort_classification_is_consistent(spec: dict[str, Any]) -> dict[str, bool]:
    """Read which cohorts may be classified, and refuse an inconsistent spec.

    The classification may only be applied to the primary cohort. If
    `classification_applies_to` and the per-cohort flags ever disagree again,
    the pass stops rather than picking a reading at run time.

    Raises:
        ResolutionGateError: If the specification is internally inconsistent.
    """
    flags = {
        name: bool(cohort["four_way_classification_applied"])
        for name, cohort in spec["cohorts"].items()
    }
    classified = sorted(name for name, on in flags.items() if on)
    if classified != [spec["primary_cohort"]]:
        raise ResolutionGateError(
            f"The frozen specification says the classification applies to "
            f"{spec['classification_applies_to']!r}, but flags it for {classified}. "
            "A cohort that may not decide the conclusion must not carry a "
            "classification label."
        )
    for name, cohort in spec["cohorts"].items():
        if flags[name] and not cohort["may_decide_the_conclusion"]:
            raise ResolutionGateError(
                f"Cohort {name!r} is flagged for classification but may not decide the conclusion."
            )
    return flags


class ResolutionGateError(RuntimeError):
    """Raised when a gate condition for the resolution pass is not satisfied."""


class ResolutionIntegrityError(RuntimeError):
    """Raised when a sealed prediction or first-look outcome has moved."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_phase2_path_allowed(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------
# Gate conditions 1 and 2: the season is over, and the snapshot reaches it
# --------------------------------------------------------------------------


def assert_season_has_ended(*, today: date | None = None) -> dict[str, Any]:
    """Refuse to run before the verified end of the 2026 regular season.

    There is deliberately no override. Running early would resolve a partial
    season and call it the end-of-season pass.

    Raises:
        ResolutionGateError: If the verified end date has not yet passed.
    """
    current = today or datetime.now(UTC).date()
    end = PROSPECTIVE_2026_SEASON_END_DATE
    if current < end:
        raise ResolutionGateError(
            f"The 2026 regular season ends {end.isoformat()}; today is "
            f"{current.isoformat()}. The resolution pass does not run early, and no "
            f"flag overrides this. Days remaining: {(end - current).days}."
        )
    return {
        "verified_season_end_date": end.isoformat(),
        "evaluated_on": current.isoformat(),
        "season_has_ended": True,
        "source": SEASON_END_VERIFICATION["source"],
    }


def assert_snapshot_reaches_season_end(snapshot: Any) -> dict[str, Any]:
    """Refuse a snapshot whose data does not reach the end of the season.

    Raises:
        ResolutionGateError: If the snapshot stops short of the end date.
    """
    through = date.fromisoformat(str(snapshot.data_through_date))
    end = PROSPECTIVE_2026_SEASON_END_DATE
    if through < end:
        raise ResolutionGateError(
            f"Snapshot {snapshot.label!r} carries data through {through.isoformat()}, "
            f"before the verified regular-season end {end.isoformat()}. A partial "
            "season is not the end-of-season resolution pass."
        )
    return {
        "snapshot_label": snapshot.label,
        "snapshot_data_through": through.isoformat(),
        "reaches_verified_season_end": True,
    }


# --------------------------------------------------------------------------
# Gate conditions 3-5: the chain, and the sealed predictions
# --------------------------------------------------------------------------


def verify_full_chain(
    *,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    h200_dir: Path = H200_OUTPUTS_DIR,
    resolution_dir: Path = RESOLUTION_OUTPUTS_DIR,
) -> dict[str, Any]:
    """Verify R1 -> ridge -> HGB -> H=200 spec -> resolution spec, zero drift.

    Raises:
        ResolutionGateError: If any stage has drifted.
    """
    from forecast.stage_freeze import verify_stage

    upstream = verify_freeze_chain(research_dir=research_dir)
    stages = {
        "h200_spec": verify_stage(h200_freeze_spec(), outputs_dir=h200_dir),
        "resolution_spec": verify_stage(resolution_freeze_spec(), outputs_dir=resolution_dir),
    }
    drifted = {name: r["drift"] for name, r in stages.items() if not r["matches_freeze"]}
    if drifted:
        raise ResolutionGateError(
            f"The freeze chain has drifted: {drifted}. Outcomes are not opened against "
            "a specification that no longer matches what was sealed."
        )
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "zero_drift_confirmed": True,
        "manifest_sha256": {
            **upstream["manifest_sha256"],
            **{name: r["manifest_sha256"] for name, r in stages.items()},
        },
    }


def load_sealed_predictions(horizon_key: str) -> dict[str, Any]:
    """Load one horizon's sealed ledgers, verified by hash against their manifest.

    A prediction that has moved is not the prediction that was made, so the
    pass refuses to proceed on any mismatch rather than resolving whatever
    happens to be on disk.

    Raises:
        ResolutionIntegrityError: If a ledger no longer matches its manifest.
    """
    ledger = FROZEN_PREDICTION_LEDGERS[horizon_key]
    outputs_dir = Path(ledger["outputs_dir"])
    manifest_path = outputs_dir / ledger["result_manifest"]
    assert_phase2_path_allowed(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    recorded: dict[str, str] = manifest["artifact_sha256"]

    hashes: dict[str, str] = {}
    for name in (ledger["pending_ledger"], ledger["completed_ledger"]):
        path = outputs_dir / name
        if not path.exists():
            raise ResolutionIntegrityError(f"Sealed ledger {name} is missing at {path}")
        digest = hash_file(path)
        if digest != recorded.get(name):
            raise ResolutionIntegrityError(
                f"Sealed ledger {name} hashes to {digest}, but the frozen manifest "
                f"recorded {recorded.get(name)}. The predictions have moved."
            )
        hashes[name] = digest

    pending = pd.read_parquet(outputs_dir / ledger["pending_ledger"])
    completed = pd.read_parquet(outputs_dir / ledger["completed_ledger"])
    if len(pending) != ledger["n_pending_at_first_look"]:
        raise ResolutionIntegrityError(
            f"{horizon_key}: expected {ledger['n_pending_at_first_look']} pending "
            f"predictions, found {len(pending)}"
        )
    if len(completed) != ledger["n_completed_at_first_look"]:
        raise ResolutionIntegrityError(
            f"{horizon_key}: expected {ledger['n_completed_at_first_look']} first-look "
            f"completed windows, found {len(completed)}"
        )
    logger.info(
        "%s: %d sealed pending, %d first-look completed, both hash-verified",
        horizon_key,
        len(pending),
        len(completed),
    )
    return {
        "horizon_key": horizon_key,
        "cutoff": int(ledger["cutoff"]),
        "horizon": int(ledger["horizon"]),
        "pending": pending,
        "first_look_completed": completed,
        "ledger_sha256": hashes,
        "result_manifest_sha256": manifest["manifest_sha256"],
    }


# --------------------------------------------------------------------------
# Attaching outcomes to sealed predictions
# --------------------------------------------------------------------------


def attach_end_of_season_outcomes(
    sealed: dict[str, Any], ordered: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach targets to the SEALED pending predictions. Nothing is recomputed.

    The frozen prediction columns travel through untouched: they are read from
    the sealed parquet and merged onto targets, never regenerated from
    end-of-season features. Regenerating them would build the cutoff-side
    window from a longer history and silently change the prediction under test.

    Returns:
        `(resolved, still_pending)` -- pending predictions that now carry a
        target, and those that still do not.
    """
    cutoff, horizon = sealed["cutoff"], sealed["horizon"]
    pending = sealed["pending"]

    windows = build_windows(ordered, cutoffs=(cutoff,), horizons=(horizon,))
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
    before = pending[list(SEALED_PREDICTION_COLUMNS)].copy()
    merged = pending.merge(targets, on=["batter", "season"], how="left")

    # Prove the sealed predictions survived the merge unchanged.
    after = merged.drop_duplicates(subset=["batter", "season"])[list(SEALED_PREDICTION_COLUMNS)]
    if len(after) != len(before) or not np.allclose(
        before.to_numpy(dtype=float), after.to_numpy(dtype=float), rtol=0, atol=0, equal_nan=True
    ):
        raise ResolutionIntegrityError(
            "Attaching outcomes altered a sealed prediction column. Outcomes are "
            "attached to predictions; predictions are never regenerated."
        )

    merged["evaluation_status"] = np.where(merged[TARGET].notna(), "resolved", "pending")
    resolved = merged[merged["evaluation_status"] == "resolved"].copy()
    still_pending = merged[merged["evaluation_status"] == "pending"].copy()
    if int(still_pending[TARGET].notna().sum()):
        raise ResolutionIntegrityError(
            "A still-pending forecast carries a target; a hitter mid-horizon must never "
            "be evaluated on a partial window."
        )
    return resolved, still_pending


def assert_first_look_targets_unchanged(
    sealed: dict[str, Any], ordered: pd.DataFrame
) -> dict[str, Any]:
    """Re-derive the first-look targets at season's end and require they match.

    A window is fixed by the hitter's first `cutoff` and next `horizon`
    resolved batted balls, so later games cannot change it. If one moved, the
    underlying snapshot data was revised retroactively -- an integrity failure,
    not a modelling question.

    Raises:
        ResolutionIntegrityError: If any first-look target has changed.
    """
    cutoff, horizon = sealed["cutoff"], sealed["horizon"]
    first_look = sealed["first_look_completed"]
    windows = build_windows(ordered, cutoffs=(cutoff,), horizons=(horizon,))
    recomputed = windows[["batter", "season", TARGET]].rename(columns={TARGET: "recomputed_target"})
    check = first_look[["batter", "season", TARGET]].merge(
        recomputed, on=["batter", "season"], how="left"
    )
    missing = int(check["recomputed_target"].isna().sum())
    if missing:
        raise ResolutionIntegrityError(
            f"{missing} first-look window(s) no longer resolve at season's end; the "
            "snapshot data changed retroactively."
        )
    drift = ~np.isclose(
        check[TARGET].to_numpy(dtype=float),
        check["recomputed_target"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    )
    if bool(drift.any()):
        raise ResolutionIntegrityError(
            f"{int(drift.sum())} first-look target(s) changed between the 2026-09-01 "
            "snapshot and the end-of-season snapshot."
        )
    return {
        "n_first_look_windows_rechecked": int(len(check)),
        "n_targets_changed": 0,
        "first_look_results_still_valid": True,
    }


# --------------------------------------------------------------------------
# The cohorts
# --------------------------------------------------------------------------


def build_cohorts(
    sealed: dict[str, Any], resolved: pd.DataFrame, still_pending: pd.DataFrame
) -> dict[str, Any]:
    """Split into the three prespecified cohorts.

    INCREMENTAL is the primary cohort and the only one the classification
    decides. FULL_SEASON pools it with the already-observed first-look windows
    and is never independent. NEVER_COMPLETED is reported, never evaluated.
    """
    first_look = sealed["first_look_completed"]
    shared = [
        c
        for c in ("batter", "batter_name", "season", "cutoff", "horizon", TARGET)
        if c in first_look.columns and c in resolved.columns
    ]
    columns = list(dict.fromkeys([*shared, *SEALED_PREDICTION_COLUMNS]))
    full_season = pd.concat(
        [first_look[columns], resolved[columns]], ignore_index=True
    ).drop_duplicates(subset=["batter", "season"])

    overlap = set(first_look["batter"]) & set(resolved["batter"])
    if overlap:
        raise ResolutionIntegrityError(
            f"{len(overlap)} hitter(s) appear in both the first-look and incremental "
            "cohorts; the incremental cohort must be disjoint from what was observed."
        )
    return {
        "incremental": resolved,
        "full_season": full_season,
        "never_completed": still_pending,
        "counts": {
            "n_first_look_completed": int(len(first_look)),
            "n_incremental": int(len(resolved)),
            "n_full_season": int(len(full_season)),
            "n_never_completed": int(len(still_pending)),
            "incremental_disjoint_from_first_look": True,
        },
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate_cohort(
    frame: pd.DataFrame,
    *,
    cohort: str,
    cutoff: int,
    horizon: int,
    design: BootstrapDesign,
    classify: bool,
) -> dict[str, Any]:
    """Metrics, paired interval, and -- only where prespecified -- a class.

    `classify` is False for the full-season cohort. It reuses already-observed
    windows, so it receives an estimate and an interval but never a
    classification label a reader could quote as a result.
    """
    if frame.empty:
        return {"cohort": cohort, "n_windows": 0, "evaluated": False}

    scored = frame.copy()
    scored["cutoff"] = cutoff
    scored["horizon"] = horizon
    predictors = [FORECAST_COLUMN, *BENCHMARKS]
    metrics = {name: _full_metrics(scored, predictor=name) for name in predictors}

    delta = paired_delta(
        scored, target=TARGET, reference=PRIMARY_BENCHMARK, challenger=FORECAST_COLUMN
    )
    assert_delta_convention(delta)
    bootstrap = run_paired_bootstrap(
        scored,
        target=TARGET,
        reference=PRIMARY_BENCHMARK,
        challenger=FORECAST_COLUMN,
        design=design,
    )
    cells: dict[str, Any] = bootstrap["cells"]  # type: ignore[assignment]
    cell: dict[str, Any] = cells[f"K{cutoff}_H{horizon}"]["pooled"]
    mae, rmse = cell["delta_mae"], cell["delta_rmse"]
    favouring_benchmark = mae["share_replicates_favouring_reference"]

    record: dict[str, Any] = {
        "cohort": cohort,
        "evaluated": True,
        "n_windows": int(len(scored)),
        "n_unique_hitters": int(scored["batter"].nunique()),
        "sign_convention": DELTA_SIGN_CONVENTION,
        "by_predictor": metrics,
        "contact_forecast_vs_shrunk_deserved": {
            "definition": f"MAE({FORECAST_COLUMN}) - MAE({PRIMARY_BENCHMARK})",
            "delta_mae": delta.delta_mae,
            "pct_change_mae": delta.pct_change_mae,
            "delta_rmse": delta.delta_rmse,
            "mae_contact_forecast": delta.mae_challenger,
            "mae_shrunk_deserved": delta.mae_reference,
            "rmse_contact_forecast": delta.rmse_challenger,
            "rmse_shrunk_deserved": delta.rmse_reference,
            "direction_mae": delta.direction_mae,
            "ci_delta_mae": [mae["ci_lower"], mae["ci_upper"]],
            "ci_delta_rmse": [rmse["ci_lower"], rmse["ci_upper"]],
            "ci_crosses_zero": mae["ci_crosses_zero"],
            "share_replicates_favouring_contact_forecast": (
                None if favouring_benchmark is None else 1.0 - favouring_benchmark
            ),
        },
        "bootstrap": bootstrap,
        "stability": build_stability(scored),
        "forecast_bias_diagnostics": _bias_diagnostics(scored),
    }
    if classify:
        record["prespecified_classification"] = classify_result(
            delta.delta_mae, mae["ci_lower"], mae["ci_upper"]
        )
        record["may_decide_the_conclusion"] = True
    else:
        record["prespecified_classification"] = None
        record["may_decide_the_conclusion"] = False
        record["mandatory_label"] = (
            "NOT INDEPENDENT -- contains windows already observed at the first look. "
            "Reported as an estimate for precision; it is not a valid significance "
            "test of the prespecified hypothesis and carries no classification."
        )
    return record


# --------------------------------------------------------------------------
# Survivorship: who resolved late, and who never did
# --------------------------------------------------------------------------


def _standardized(a: float | None, b: float | None, pooled_sd: float) -> float | None:
    """A difference in pooled standard-deviation units, or None if undefined."""
    if a is None or b is None or not pooled_sd or not np.isfinite(pooled_sd):
        return None
    return float((a - b) / pooled_sd)


def build_survivorship(
    sealed: dict[str, Any], cohorts: dict[str, Any], ordered: pd.DataFrame
) -> dict[str, Any]:
    """Cutoff-side comparison of incremental, first-look and never-completed.

    The incremental cohort is the SLOWER accumulators by construction: they
    needed the rest of the season to reach a horizon the first-look cohort
    reached by 2026-09-01. They are not exchangeable with it, and this
    quantifies how they differ using only information available at the cutoff.

    Descriptive only. Nothing is reweighted, matched, adjusted or excluded.
    """
    from forecast.features import build_window_features
    from forecast.ridge import FEATURE_SETS, resolve_features

    cutoff = sealed["cutoff"]
    features = build_window_features(ordered, cutoff)
    feature_names, _ = resolve_features(FEATURE_SETS["D_full_contact_profile"], features)
    audited = ["std_realized_rv_per_100", "std_deserved_rv_per_100", *feature_names]

    groups = {
        "first_look_completed": set(sealed["first_look_completed"]["batter"]),
        "incremental": set(cohorts["incremental"]["batter"]),
        "never_completed": set(cohorts["never_completed"]["batter"]),
    }
    frames = {name: features[features["batter"].isin(batters)] for name, batters in groups.items()}

    rows: list[dict[str, Any]] = []
    for name in audited:
        if name not in features.columns:
            continue
        pooled_sd = float(pd.to_numeric(features[name], errors="coerce").std(ddof=1))
        means = {
            group: (
                float(pd.to_numeric(frame[name], errors="coerce").dropna().mean())
                if len(frame)
                else None
            )
            for group, frame in frames.items()
        }

        rows.append(
            {
                "feature": name,
                "means": means,
                "incremental_vs_first_look_standardized": _standardized(
                    means["incremental"], means["first_look_completed"], pooled_sd
                ),
                "incremental_vs_never_completed_standardized": _standardized(
                    means["incremental"], means["never_completed"], pooled_sd
                ),
            }
        )
    rows.sort(key=lambda r: -abs(r["incremental_vs_first_look_standardized"] or 0.0))

    counts = cohorts["counts"]
    n_reached = (
        counts["n_first_look_completed"] + counts["n_incremental"] + counts["n_never_completed"]
    )
    return {
        "status": (
            "DESCRIPTIVE ONLY -- no reweighting, matching, adjustment, imputation or "
            "exclusion follows from this, and no hitter is added to or removed from a "
            "cohort on the basis of it"
        ),
        "end_of_season_completion": {
            "n_hitters_with_a_forecast": int(n_reached),
            "n_completed_by_season_end": int(
                counts["n_first_look_completed"] + counts["n_incremental"]
            ),
            "completion_rate": (
                float(counts["n_first_look_completed"] + counts["n_incremental"]) / n_reached
                if n_reached
                else float("nan")
            ),
        },
        "cohort_sizes": counts,
        "cutoff_side_differences": rows,
        "largest_incremental_vs_first_look": rows[:5],
        "why_this_matters": (
            "The incremental cohort needed the rest of the season to accumulate the "
            "horizon. Differences from the first-look cohort are expected and are not "
            "evidence about model quality."
        ),
        "generalization_note": (
            "This result generalizes only to hitters who accumulated the horizon within "
            "the season. It does not describe those who did not, and it does not "
            "predict playing time, injury, roster survival or demotion."
        ),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "resolution_2026_authorization.json",
    "resolution_2026_cohorts.json",
    "resolution_2026_metrics.json",
    "resolution_2026_survivorship.json",
    "resolution_2026_distribution_shift.json",
    "resolution_2026_incremental.parquet",
    "resolution_2026_full_season.parquet",
    "resolution_2026_never_completed.parquet",
    "resolution_2026_report.md",
)


def frozen_model_features(*, resolution_dir: Path = RESOLUTION_OUTPUTS_DIR) -> list[str]:
    """The selected model's feature list, read from the sealed ridge freeze.

    Never re-derived. `the feature list` is on the specification's
    `frozen_and_untouchable` list, so the only defensible source is the
    manifest that sealed it -- recomputing the same names from live code
    would produce an identical answer today and silently a different one
    the day the code changes, which is exactly what freezing prevents.

    Raises:
        ResolutionIntegrityError: If the manifest does not name the model's
            features, which means the freeze is not what this expects.
    """
    manifest_path = resolution_dir / "ridge_freeze_manifest.json"
    assert_phase2_path_allowed(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    key_results = manifest.get("key_results", {})
    model = (key_results.get("selected_model") or {}).get("model")
    features = (key_results.get("feature_sets") or {}).get(model)
    if not model or not features:
        raise ResolutionIntegrityError(
            f"{manifest_path} does not name the selected model's feature set. The "
            "distribution-shift diagnostic reads the FROZEN feature list and will not "
            "fall back to deriving one."
        )
    return list(features)


def build_cohort_distribution_shift(
    cohorts: dict[str, Any],
    development_focal: pd.DataFrame,
    *,
    features: list[str],
) -> dict[str, Any]:
    """Required analysis 3: how 2026's feature distributions sit against the
    permitted development seasons, per cohort.

    Run per cohort rather than once over 2026, because the incremental cohort
    is not exchangeable with the first-look cohort -- it is the slower
    accumulators by construction. A single pooled diagnostic would describe a
    population that no classification is made about, which is precisely the
    confusion the cohort split exists to avoid.

    Explanatory only, and the same words apply as at the first look: nothing
    is recalibrated, transformed, clipped, dropped or retrained on the basis
    of these numbers. A large shift is a caveat on generalization, never a
    reason to adjust a frozen forecast.
    """
    per_cohort: dict[str, Any] = {}
    for cohort in ("incremental", "full_season"):
        frame = cohorts.get(cohort)
        if frame is None or frame.empty:
            per_cohort[cohort] = {"evaluated": False, "n_windows": 0}
            continue
        missing = [f for f in features if f not in frame.columns]
        if missing:
            raise ResolutionIntegrityError(
                f"cohort {cohort!r} is missing frozen model features {missing}. The "
                "diagnostic compares the FROZEN feature list and does not silently "
                "compare a subset."
            )
        per_cohort[cohort] = {
            "evaluated": True,
            "n_windows": int(len(frame)),
            "features": build_distribution_shift(frame, development_focal, features=features),
        }
    return {
        "purpose": (
            "Descriptive comparison of each cohort's frozen-model feature distributions "
            "against the permitted development seasons."
        ),
        "status": "DIAGNOSTIC ONLY -- nothing is recalibrated, dropped or retrained",
        "generalization_note": (
            "A large shift limits how far the result generalizes. It is never grounds to "
            "adjust the frozen forecast, reweight a cohort, or exclude a hitter."
        ),
        "n_features": len(features),
        "cohorts": per_cohort,
    }


def run(
    *,
    authorized_by: str,
    horizon_key: str = "H200",
    outputs_dir: Path = RESOLUTION_OUTPUTS_DIR,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    data_dir: Path = FORECAST_DATA_DIR,
    snapshot_label: str | None = None,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    today: date | None = None,
) -> dict[str, Path]:
    """Run the resolution pass once, after every gate condition holds.

    Args:
        authorized_by: The maintainer authorizing this second look, in the
            moment. Gate condition 7 -- it has no default, so the pass cannot
            run from a cron job or by accident.

    Raises:
        ResolutionGateError: If any gate condition fails.
    """
    if not authorized_by or not authorized_by.strip():
        raise ResolutionGateError(
            "Gate condition 7: the maintainer must authorize this second look in the "
            "moment, knowing it is a second unadjusted look at 2026."
        )
    assert_phase2_path_allowed(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- gates 1, 3, 4, 5: before a single outcome is read ----------------
    season = assert_season_has_ended(today=today)
    chain = verify_full_chain(research_dir=research_dir, resolution_dir=outputs_dir)
    frozen_spec = json.loads((outputs_dir / "resolution_specification.json").read_text())
    classification_flags = assert_cohort_classification_is_consistent(frozen_spec)
    sealed = load_sealed_predictions(horizon_key)

    # --- gates 2, 6: pin exactly one end-of-season snapshot ---------------
    snapshot = resolve_snapshot(label=snapshot_label)
    snapshot_gate = assert_snapshot_reaches_season_end(snapshot)
    logger.info("pinned %s (data through %s)", snapshot.label, snapshot.data_through_date)

    authorization = {
        "stage": "resolution_2026",
        "authorized_by": authorized_by,
        "authorized_at_utc": datetime.now(UTC).isoformat(),
        "acknowledged": (
            "This is a SECOND unadjusted look at 2026. The two looks together are not a "
            "single 5% test, and this pass cannot confirm the first look."
        ),
        "horizon_key": horizon_key,
        "cutoff": sealed["cutoff"],
        "horizon": sealed["horizon"],
        "gate": {
            "season_has_ended": season,
            "snapshot_reaches_season_end": snapshot_gate,
            "freeze_chain": chain,
            "sealed_ledger_sha256": sealed["ledger_sha256"],
            "result_manifest_sha256": sealed["result_manifest_sha256"],
            "single_snapshot": True,
            "snapshot_refreshed_during_analysis": False,
        },
        "cohort_classification_history": COHORT_CLASSIFICATION_HISTORY,
        "cohort_classification_flags": classification_flags,
        "predictions_regenerated": False,
        "first_look_results_modified": False,
        "deployed": False,
        "is_the_final_look": True,
    }
    authorization["record_sha256"] = hash_json(
        {k: v for k, v in authorization.items() if k != "record_sha256"}
    )
    _write_json(outputs_dir / "resolution_2026_authorization.json", authorization)

    # --- outcomes, attached to sealed predictions -------------------------
    ledger, _ = load_contact_stage_ledger(snapshot)
    assert_no_full_telescoping_columns(ledger)
    resolved_events, _ = select_scored_eligible_events(ledger)
    ordered = order_events_for_phase2(resolved_events)

    integrity = assert_first_look_targets_unchanged(sealed, ordered)
    resolved, still_pending = attach_end_of_season_outcomes(sealed, ordered)
    cohorts = build_cohorts(sealed, resolved, still_pending)
    logger.info(
        "cohorts: incremental=%d full_season=%d never_completed=%d",
        cohorts["counts"]["n_incremental"],
        cohorts["counts"]["n_full_season"],
        cohorts["counts"]["n_never_completed"],
    )

    cohorts["incremental"].to_parquet(
        outputs_dir / "resolution_2026_incremental.parquet", index=False
    )
    cohorts["full_season"].to_parquet(
        outputs_dir / "resolution_2026_full_season.parquet", index=False
    )
    cohorts["never_completed"].to_parquet(
        outputs_dir / "resolution_2026_never_completed.parquet", index=False
    )
    _write_json(
        outputs_dir / "resolution_2026_cohorts.json",
        {"counts": cohorts["counts"], "first_look_integrity": integrity},
    )

    # --- evaluation: the classification decides on the incremental cohort --
    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)
    completion = horizon_completion(ordered, cutoff=sealed["cutoff"], horizon=sealed["horizon"])
    metrics = {
        "horizon_key": horizon_key,
        "cutoff": sealed["cutoff"],
        "horizon": sealed["horizon"],
        "primary_cohort": "incremental",
        "classification_applies_to": frozen_spec["classification_applies_to"],
        "cohort_classification_history": COHORT_CLASSIFICATION_HISTORY,
        "incremental": evaluate_cohort(
            cohorts["incremental"],
            cohort="incremental",
            cutoff=sealed["cutoff"],
            horizon=sealed["horizon"],
            design=design,
            classify=classification_flags["incremental"],
        ),
        "full_season": evaluate_cohort(
            cohorts["full_season"],
            cohort="full_season",
            cutoff=sealed["cutoff"],
            horizon=sealed["horizon"],
            design=design,
            classify=classification_flags["full_season"],
        ),
        "never_completed": {
            "cohort": "never_completed",
            "evaluated": False,
            "n_hitters": int(cohorts["counts"]["n_never_completed"]),
            "note": (
                "Reported, never evaluated. No partial window, no imputation, and no "
                "removal from the prediction ledger."
            ),
        },
        "provenance": {
            "authorization_sha256": authorization["record_sha256"],
            "freeze_chain": chain["manifest_sha256"],
            "sealed_ledger_sha256": sealed["ledger_sha256"],
            "snapshot_label": snapshot.label,
            "snapshot_data_through": snapshot.data_through_date,
        },
        "n_hitters_reaching_cutoff": int(completion["reached_cutoff"].sum()),
    }
    _write_json(outputs_dir / "resolution_2026_metrics.json", metrics)

    survivorship = build_survivorship(sealed, cohorts, ordered)
    _write_json(outputs_dir / "resolution_2026_survivorship.json", survivorship)

    # Required analysis 3. The development frame is loaded here rather than
    # earlier because nothing before this point needs it, and the pass should
    # not read development data at all if it is going to fail a gate.
    development_focal, _prior_events, _development_audit = load_development_training(
        research_dir=research_dir, data_dir=data_dir
    )
    shift = build_cohort_distribution_shift(
        cohorts, development_focal, features=frozen_model_features()
    )
    _write_json(outputs_dir / "resolution_2026_distribution_shift.json", shift)

    from forecast.phase2.run_resolution_report import render_resolution_report

    report_path = outputs_dir / "resolution_2026_report.md"
    report_path.write_text(
        render_resolution_report(
            authorization=authorization,
            metrics=metrics,
            survivorship=survivorship,
            distribution_shift=shift,
        )
    )
    logger.info("wrote %s", report_path)
    _write_provenance(outputs_dir, chain=chain, authorization=authorization)
    return {name: outputs_dir / name for name in REQUIRED_ARTIFACTS}


def _write_provenance(
    outputs_dir: Path, *, chain: dict[str, Any], authorization: dict[str, Any]
) -> None:
    """Hash every resolution artifact and chain it to the frozen stages."""
    record = {
        "stage": "resolution_2026",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact_sha256": {
            name: hash_file(outputs_dir / name)
            for name in REQUIRED_ARTIFACTS
            if (outputs_dir / name).exists()
        },
        "authorization_sha256": authorization["record_sha256"],
        "provenance_chain": chain["manifest_sha256"],
        "sealed_seasons_never_read": [2025],
        "evaluation_season": 2026,
        "is_the_final_look": True,
        "predictions_regenerated": False,
        "deployed": False,
    }
    record["manifest_sha256"] = hash_json(
        {k: v for k, v in record.items() if k != "generated_at_utc"}
    )
    _write_json(outputs_dir / "resolution_2026_result_manifest.json", record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorized-by",
        required=True,
        help="Gate 7: the maintainer authorizing this second unadjusted look, now.",
    )
    parser.add_argument("--horizon-key", choices=sorted(FROZEN_PREDICTION_LEDGERS), default="H200")
    parser.add_argument("--outputs-dir", type=Path, default=RESOLUTION_OUTPUTS_DIR)
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
        authorized_by=args.authorized_by,
        horizon_key=args.horizon_key,
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
