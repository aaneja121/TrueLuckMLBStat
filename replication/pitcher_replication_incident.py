"""Version 0.14: append-only incident recording for the 2025 pitcher
replication, plus the proposed training-frame correction.

## Why this module exists separately

`pitcher_replication_execution.py`, `run_pitcher_replication_2025.py`,
`pitcher_replication_questions.py`, `pitcher_replication_authorization.py`,
`evaluation/run_v1_final_evaluation.py` and the `Makefile` are all hashed by
the sealed pre-execution manifest. Recording a failure must not invalidate
that manifest, so this module is deliberately NOT in
`EXECUTION_SOURCE_RELATIVE_PATHS`. It never mutates the execution-start
receipt or the pre-execution manifest; it only appends.

## The 2026-09-08 incident, in one line

`train_and_score_2025(development_df, evaluation_df)`'s first parameter is
NAMED `development_df` but its CONTRACT is "training data already filtered to
`TRAIN_SEASONS`". Version 1.0's own caller does that filtering
(`full_development_df[full_development_df["season"].isin(TRAIN_SEASONS)]`);
`run_pitcher_replication_2025.run_replication` passed the unfiltered
2021-2024 parquet, so 2024 reached the training argument and the frozen
guard refused.

The guard is a PRECONDITION CHECK, not a filter. It fires before
`_apply_eligibility_pipeline` and before every `train_*` call, so no model
was fit on 2024 -- see `GUARD_FIRES_BEFORE_FITTING`.

## What the correction is, and is not

`select_training_frame` is the exact expression Version 1.0 already uses. It
restores the frozen 2021-2023 training contract that the specification
already fixed; it does not choose it. Nothing scientific changes: not the
metric, denominator, estimators, role boundaries, presentation rules, or
model architecture. See `CORRECTION_IS_EXECUTION_ONLY`.

**2025 is already open.** Nothing in this module may be described as a first
look, and no recovery may run without a fresh explicit maintainer
authorization.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pitcher_replication_freeze import ARTIFACTS_DIR, REPO_ROOT, display_path

#: Append-only. Never rewritten, never truncated: an incident log that can be
#: edited is not evidence.
FAILURE_RECORD_PATH = ARTIFACTS_DIR / "failure_records.jsonl"

#: Where the failed run's already-downloaded 2025 artifacts live. Hashed by
#: BYTES only -- their contents are not opened to record an incident.
REPLICATION_2025_DATA_DIR = REPO_ROOT / "data" / "pitcher_replication" / "2025"

GUARD_FIRES_BEFORE_FITTING = (
    "In evaluation.run_v1_final_evaluation.train_and_score_2025 the season guard raises "
    "BEFORE _apply_eligibility_pipeline and before every train_* call (verified by source "
    "offset: guard 1209, pipeline 1661, first fit 1862). No component model was fit, no "
    "2025 prediction was produced, and no pitcher aggregate, diagnostic or classification "
    "was computed."
)

CORRECTION_IS_EXECUTION_ONLY = (
    "The frozen Version 0.14 specification already fixes training to 2021-2023 and 2025 to "
    "evaluation only. The defect is that the runner passed an unfiltered dataframe to a "
    "parameter whose contract is 'already filtered to TRAIN_SEASONS'. Applying the filter "
    "restores the frozen intent rather than selecting it, so no methodological choice is "
    "involved: no metric, denominator, estimator, threshold, role boundary, presentation "
    "rule or model architecture changes, and nothing is informed by any observed 2025 "
    "result -- none exists."
)


class IncidentError(RuntimeError):
    """Raised when an incident record cannot be written."""


# ---------------------------------------------------------------------------
# The proposed correction
# ---------------------------------------------------------------------------


def select_training_frame(
    development_df: pd.DataFrame, *, season_column: str = "season"
) -> pd.DataFrame:
    """Filter a development dataframe to `TRAIN_SEASONS` (2021-2023).

    This is the expression `evaluation.run_v1_final_evaluation.
    run_final_evaluation` already applies before calling
    `train_and_score_2025`. It is reproduced here so the proposed recovery is
    a named, tested unit rather than an inline edit nobody checked.

    Raises:
        IncidentError: if the frame has no season column, or if filtering
            leaves nothing to train on.
    """
    from mlb_luck_score.config import TRAIN_SEASONS

    if season_column not in development_df.columns:
        raise IncidentError(f"development frame has no {season_column!r} column")
    training = development_df[development_df[season_column].isin(TRAIN_SEASONS)].copy()
    if training.empty:
        raise IncidentError(
            f"filtering to TRAIN_SEASONS {TRAIN_SEASONS} left no training rows -- refusing "
            "to train on an empty frame"
        )
    return training


def training_seasons_of(frame: pd.DataFrame, *, season_column: str = "season") -> set[int]:
    """The distinct seasons present, as a set of ints."""
    return {int(s) for s in frame[season_column].dropna().unique()}


# ---------------------------------------------------------------------------
# Incident recording
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_2025_artifacts(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Byte-level inventory of every 2025 file the failed run created.

    Records size, mtime and SHA-256. **Does not open any file as data** --
    an incident record must not become a reason to look at held-out results.
    """
    data_dir = data_dir if data_dir is not None else REPLICATION_2025_DATA_DIR
    if not data_dir.exists():
        return []
    return [
        {
            "path": display_path(path),
            "bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "sha256": _sha256(path),
        }
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.name != ".gitkeep"
    ]


def build_failure_record(
    *,
    exception_type: str,
    exception_message: str,
    stack_location: str,
    stage_reached: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the incident record from the receipt, the sealed manifest and
    the filesystem. Reads no held-out data.
    """
    import json as _json

    from pitcher_replication_execution import (
        EXECUTION_MANIFEST_PATH,
        EXECUTION_RECEIPT_PATH,
    )

    if not EXECUTION_RECEIPT_PATH.exists():
        raise IncidentError(
            "No execution-start receipt: refusing to record a post-exposure failure for a "
            "run that never opened 2025."
        )
    receipt = _json.loads(EXECUTION_RECEIPT_PATH.read_text())
    manifest = _json.loads(EXECUTION_MANIFEST_PATH.read_text())

    outputs_dir = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
    result_files = (
        [display_path(p) for p in outputs_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
        if outputs_dir.exists()
        else []
    )
    run_seal = ARTIFACTS_DIR / "pitcher_replication_2025_seal.json"

    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "incident": "execution_failure_after_held_out_ingestion",
        "execution_id": receipt["execution_id"],
        "receipt_opened_at_utc": receipt["opened_at_utc"],
        "freeze_content_hash": receipt["freeze_content_hash"],
        "spec_content_hash": receipt["spec_content_hash"],
        "execution_manifest_hash": receipt["execution_manifest_hash"],
        "authorization_content_hash": receipt["authorization_content_hash"],
        "runner_commit": manifest["repository_commit"],
        "exception_type": exception_type,
        "exception_message": exception_message,
        "stack_location": stack_location,
        "stage_reached": stage_reached,
        "guard_fired_before_model_fitting": True,
        "guard_evidence": GUARD_FIRES_BEFORE_FITTING,
        "exposure": {
            "receipt_written": True,
            "held_out_2025_ingested": True,
            "evaluation_dataframe_constructed": True,
            "model_parameters_fit": False,
            "any_model_fit_on_2024": False,
            "predictions_produced": False,
            "pitcher_aggregates_produced": False,
            "questions_a_to_g_computed": False,
            "classification_computed": False,
            "result_artifact_exists": bool(result_files),
            "run_seal_exists": run_seal.exists(),
            "result_files": result_files,
        },
        "artifacts_2025": inventory_2025_artifacts(),
        "receipt_preserved": True,
        "execution_manifest_preserved": True,
        "is_a_first_look": False,
        "note_on_exposure": (
            "2025 is OPEN. This record documents a failure that occurred AFTER held-out "
            "ingestion. Any subsequent run is a resumption, never a first look."
        ),
        "notes": notes or [],
    }


def append_failure_record(record: dict[str, Any], *, path: Path | None = None) -> tuple[Path, str]:
    """Append one incident record. Never rewrites or truncates the log.

    Returns:
        `(path, record_sha256)`.
    """
    path = path if path is not None else FAILURE_RECORD_PATH
    payload = json.dumps(record, sort_keys=True, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(payload + "\n")
    return (path, hashlib.sha256(payload.encode("utf-8")).hexdigest())


def read_failure_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path if path is not None else FAILURE_RECORD_PATH
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
