"""Version 0.14: recovery control for the failed 2025 pitcher replication.

**2025 is already open.** Execution `8edc32d6ca8830ce` opened the held-out
season on 2026-09-08 and failed inside the frozen scorer before any model was
fit. Nothing in this module is a first look, and nothing here may be
described as one.

## Why a THIRD manifest

- `pitcher_replication_freeze.json` -- what we were going to measure.
  Untouched, still validating 15/15.
- `pitcher_replication_execution_manifest.json` -- the code and state the
  FAILED run was sealed with. Preserved byte-identically. It no longer
  validates against the working tree, and that is correct: the runner has
  since been corrected, which is exactly the drift it exists to detect. It
  is never resealed.
- `recovery_manifest.json` (this module) -- the corrected code, the exact
  diff, the already-downloaded 2025 artifacts, and the incident it descends
  from.

## The original authorization is SPENT

`pitcher_replication_authorization` covered one evaluation, and the receipt
records it consumed. This module does NOT reuse it. A resumption needs a
NEW, explicit maintainer authorization bound to this recovery manifest's own
hash -- see `resolve_recovery_authorization`, which currently returns False
because no such authorization exists.

`assert_ready_for_recovery` therefore always refuses today. That is the
intended state.

## One recovery attempt

`write_recovery_receipt` is written immediately before the first
post-failure 2025 read, and once it exists a further attempt refuses. There
is no automatic retry: a second failure requires a new explicit incident
decision, not another quiet look at held-out data.

## Integrity, not inspection

`verify_2025_artifacts` re-hashes the three downloaded files BYTE-FOR-BYTE
and compares against the incident record. It never parses their contents.
Proving the recovery needs no network is a filesystem and hash question, and
is treated as one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pitcher_replication_execution import (
    EXECUTION_MANIFEST_PATH,
    EXECUTION_RECEIPT_PATH,
    ExecutionError,
)
from pitcher_replication_freeze import (
    ARTIFACTS_DIR,
    REPLICATION_DATA_DIR,
    REPLICATION_OUTPUTS_DIR,
    REPO_ROOT,
    build_source_hashes,
    content_hash,
    display_path,
    get_git_commit_hash,
    read_freeze,
    resolve_authorization,
    validate_freeze,
    working_tree_status,
)
from pitcher_replication_incident import FAILURE_RECORD_PATH, read_failure_records

RECOVERY_MANIFEST_VERSION = "1.0.0"

RECOVERY_MANIFEST_PATH = ARTIFACTS_DIR / "recovery_manifest.json"
RECOVERY_RECEIPT_PATH = ARTIFACTS_DIR / "recovery_receipt.json"

#: The corrected runner plus every recovery-control source. Distinct from the
#: research freeze (what is measured) and from the failed run's execution
#: manifest (the code that failed).
RECOVERY_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "replication/run_pitcher_replication_2025.py",
    "replication/pitcher_replication_recovery.py",
    "replication/pitcher_replication_incident.py",
    "replication/pitcher_replication_execution.py",
    "replication/pitcher_replication_questions.py",
    "replication/pitcher_replication_authorization.py",
    "evaluation/run_v1_final_evaluation.py",
)

#: The commit the failed run was sealed at, and the commit that corrects it.
FAILED_RUN_COMMIT = "8e73d61843fe347c24018c69f609253b7b813c72"
CORRECTION_COMMIT = "185f2621de19b75d4489fe83a8cecb93f33e0ca1"

RECOVERY_IS_TECHNICAL_NOT_METHODOLOGICAL = (
    "The frozen Version 0.14 specification already fixed training to 2021-2023 and 2025 to "
    "evaluation only. The defect was that the runner passed an unfiltered dataframe to a "
    "parameter whose contract is 'already filtered to TRAIN_SEASONS'. Applying that filter "
    "RESTORES the frozen intent rather than selecting it. No metric, denominator, "
    "estimator, interval procedure, threshold, role boundary, presentation rule, question "
    "or classification rule changed, and nothing was informed by an observed 2025 result -- "
    "none was computed."
)

RECOVERY_IS_NOT_A_FIRST_LOOK = (
    "2025 was opened by execution 8edc32d6ca8830ce at 2026-09-08T17:00:33Z. Any recovery is "
    "a RESUMPTION after exposure. It must never be reported, committed, or described as a "
    "first look, and the original execution-start receipt must never be deleted or "
    "replaced to make it appear so."
)

NO_RESULT_WAS_COMPUTED = (
    "The failure occurred in the frozen scorer's season precondition guard, which raises "
    "before the eligibility pipeline and before every train_* call. No component model was "
    "fit, no 2025 prediction was produced, no pitcher aggregate was computed, no question "
    "A-G was answered, and no classification was produced. The outputs namespace holds no "
    "result artifact."
)


class RecoveryError(RuntimeError):
    """Raised when recovery control refuses to proceed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The NEW recovery authorization -- absent by design
# ---------------------------------------------------------------------------


def resolve_recovery_authorization(
    manifest_hash: str,
) -> tuple[bool, list[str]]:
    """Is there a maintainer authorization for THIS recovery manifest?

    The original `pitcher_replication_authorization` covered one evaluation
    and is spent; it is deliberately NOT consulted here. A resumption needs
    its own sign-off, recorded the same way and bound to `manifest_hash`.

    Returns:
        `(authorized, reasons)`. Today always `(False, [...])` -- no
        recovery authorization module exists.
    """
    try:
        from pitcher_replication_recovery_authorization import (
            recovery_authorization_binds_to,
        )
    except ImportError:
        return (
            False,
            [
                "no recovery authorization has been recorded; the original one-time "
                "authorization was consumed by execution 8edc32d6ca8830ce and may not be "
                "reused"
            ],
        )
    return recovery_authorization_binds_to(manifest_hash)  # pragma: no cover


# ---------------------------------------------------------------------------
# Integrity of the already-downloaded 2025 artifacts
# ---------------------------------------------------------------------------


def verify_2025_artifacts(data_dir: Path | None = None) -> dict[str, Any]:
    """Re-hash the three downloaded 2025 files and require exact agreement
    with the incident record.

    Byte-level only: the files are never parsed. This is what lets the
    recovery reuse them instead of downloading 2025 a second time.

    Raises:
        RecoveryError: if the incident record is missing, a file is absent,
            or any hash disagrees.
    """
    data_dir = data_dir if data_dir is not None else REPLICATION_DATA_DIR
    records = read_failure_records()
    if not records:
        raise RecoveryError(
            f"No incident record at {FAILURE_RECORD_PATH}; refusing to verify recovery "
            "inputs without the record they must match."
        )
    recorded = {entry["path"].split("/")[-1]: entry for entry in records[-1]["artifacts_2025"]}
    if not recorded:
        raise RecoveryError("The incident record lists no 2025 artifacts to verify.")

    verified: dict[str, Any] = {}
    problems: list[str] = []
    for name, entry in sorted(recorded.items()):
        path = data_dir / name
        if not path.is_file():
            problems.append(f"{name} is missing from {display_path(data_dir)}")
            continue
        live = _sha256_file(path)
        if live != entry["sha256"]:
            problems.append(
                f"{name} bytes changed since the incident (recorded {entry['sha256']}, now {live})"
            )
            continue
        verified[name] = {
            "path": display_path(path),
            "bytes": path.stat().st_size,
            "sha256": live,
            "matches_incident_record": True,
        }
    if problems:
        raise RecoveryError("2025 artifact integrity FAILED: " + "; ".join(problems))
    return verified


def assert_originals_preserved() -> dict[str, str]:
    """The failed run's receipt, its execution manifest and the incident
    record must all still exist. Their hashes are pinned by the recovery
    manifest, so a later change is detectable.

    Raises:
        RecoveryError: if any is missing.
    """
    required = {
        "execution_start_receipt": EXECUTION_RECEIPT_PATH,
        "execution_manifest": EXECUTION_MANIFEST_PATH,
        "incident_record": FAILURE_RECORD_PATH,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RecoveryError(
            f"Original incident evidence is missing: {missing}. These are permanent "
            "records of an exposure that already happened and must never be deleted."
        )
    return {name: _sha256_file(path) for name, path in required.items()}


# ---------------------------------------------------------------------------
# The recovery manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryManifest:
    recovery_manifest_version: str
    original_execution_id: str
    original_receipt_opened_at_utc: str
    original_receipt_sha256: str
    original_execution_manifest_hash: str
    original_execution_manifest_sha256: str
    incident_record_sha256: str
    freeze_content_hash: str
    spec_content_hash: str
    original_authorization_content_hash: str
    failed_run_commit: str
    correction_commit: str
    corrected_runner_sha256: str
    correction_diff_sha256: str
    correction_diff: str
    recovery_source_hashes: dict[str, str]
    artifacts_2025: dict[str, Any]
    repository_commit: str
    working_tree_clean: bool
    intended_output_paths: dict[str, str]
    statements: dict[str, Any]
    sealed_at_utc: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryManifest:
        return cls(**data)

    def manifest_content_hash(self) -> str:
        payload = self.to_dict()
        for volatile in ("sealed_at_utc", "notes"):
            payload.pop(volatile, None)
        return content_hash(payload)


def _correction_diff(repo_root: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            f"{FAILED_RUN_COMMIT}..{CORRECTION_COMMIT}",
            "--",
            "replication/run_pitcher_replication_2025.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RecoveryError(f"could not compute the correction diff: {result.stderr.strip()}")
    if not result.stdout.strip():
        raise RecoveryError(
            "the correction diff is empty -- the recovery manifest must pin a real change"
        )
    return result.stdout


def build_recovery_manifest(
    repo_root: Path = REPO_ROOT, *, notes: list[str] | None = None
) -> RecoveryManifest:
    """Assemble the recovery manifest, running every precondition first.

    Raises:
        RecoveryError / FreezeError: on any failed precondition.
    """
    freeze = read_freeze()
    validate_freeze(freeze, repo_root)
    binds, reasons = resolve_authorization(freeze)
    if not binds:
        raise RecoveryError(f"original authorization no longer binds to the freeze: {reasons}")

    originals = assert_originals_preserved()
    artifacts = verify_2025_artifacts()
    receipt = json.loads(EXECUTION_RECEIPT_PATH.read_text())
    execution_manifest = json.loads(EXECUTION_MANIFEST_PATH.read_text())

    is_clean, dirty = working_tree_status(repo_root)
    if not is_clean:
        raise RecoveryError(
            f"Working tree is not clean; refusing to seal a recovery manifest: {dirty}"
        )
    assert_no_recovery_receipt()

    diff = _correction_diff(repo_root)
    runner = repo_root / "replication" / "run_pitcher_replication_2025.py"

    return RecoveryManifest(
        recovery_manifest_version=RECOVERY_MANIFEST_VERSION,
        original_execution_id=receipt["execution_id"],
        original_receipt_opened_at_utc=receipt["opened_at_utc"],
        original_receipt_sha256=originals["execution_start_receipt"],
        original_execution_manifest_hash=receipt["execution_manifest_hash"],
        original_execution_manifest_sha256=originals["execution_manifest"],
        incident_record_sha256=originals["incident_record"],
        freeze_content_hash=freeze.freeze_content_hash(),
        spec_content_hash=freeze.spec_content_hash,
        original_authorization_content_hash=execution_manifest["authorization_content_hash"],
        failed_run_commit=FAILED_RUN_COMMIT,
        correction_commit=CORRECTION_COMMIT,
        corrected_runner_sha256=_sha256_file(runner),
        correction_diff_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        correction_diff=diff,
        recovery_source_hashes=build_source_hashes(
            repo_root, relative_paths=RECOVERY_SOURCE_RELATIVE_PATHS
        ),
        artifacts_2025=artifacts,
        repository_commit=get_git_commit_hash(repo_root),
        working_tree_clean=is_clean,
        intended_output_paths={
            "raw_data_dir": display_path(REPLICATION_DATA_DIR),
            "outputs_dir": display_path(REPLICATION_OUTPUTS_DIR),
            "artifacts_dir": display_path(ARTIFACTS_DIR),
            "recovery_receipt": display_path(RECOVERY_RECEIPT_PATH),
        },
        statements={
            "artifacts_downloaded_during_failed_first_exposure": True,
            "artifacts_downloaded_note": (
                "The three 2025 files were downloaded by execution "
                f"{receipt['execution_id']} before it failed. The recovery reuses them "
                "and performs no network operation."
            ),
            "no_contact_luck_result_computed_before_failure": True,
            "no_result_note": NO_RESULT_WAS_COMPUTED,
            "recovery_is_technical_not_methodological": True,
            "technical_note": RECOVERY_IS_TECHNICAL_NOT_METHODOLOGICAL,
            "season_2025_already_exposed": True,
            "is_a_first_look": False,
            "first_look_note": RECOVERY_IS_NOT_A_FIRST_LOOK,
            "season_2026_exposed": False,
            "original_authorization_is_spent": True,
            "requires_new_recovery_authorization": True,
            "one_recovery_attempt_only": True,
        },
        sealed_at_utc=datetime.now(UTC).isoformat(),
        notes=notes or [],
    )


def read_recovery_manifest(path: Path | None = None) -> RecoveryManifest:
    path = path if path is not None else RECOVERY_MANIFEST_PATH
    if not path.is_file():
        raise RecoveryError(f"No recovery manifest at {path}")
    return RecoveryManifest.from_dict(json.loads(path.read_text()))


def write_recovery_manifest(
    manifest: RecoveryManifest, *, path: Path | None = None
) -> tuple[Path, str]:
    """Write-once, additive. Never touches the freeze, the original execution
    manifest, the original receipt or the incident record.

    An identical reseal is idempotent. Anything else is refused, and there is
    deliberately NO amendment mechanism: a further code change needs a NEW
    recovery manifest from a new commit.

    Raises:
        RecoveryError: if a different recovery manifest already exists.
    """
    path = path if path is not None else RECOVERY_MANIFEST_PATH
    new_hash = manifest.manifest_content_hash()
    if path.exists():
        existing = read_recovery_manifest(path)
        existing_hash = existing.manifest_content_hash()
        if existing_hash == new_hash:
            return (path, existing_hash)
        raise RecoveryError(
            f"A DIFFERENT recovery manifest already exists at {path} (existing "
            f"{existing_hash}, new {new_hash}). It is write-once and has no amendment "
            "path. If the recovery code changed again, commit it and seal a NEW recovery "
            "manifest so this record of what was sealed survives intact."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str) + "\n")
    return (path, new_hash)


def validate_recovery_manifest(
    manifest: RecoveryManifest, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Re-derive everything the recovery manifest pinned and compare.

    Raises:
        RecoveryError: on recovery-source drift, changed 2025 artifact bytes,
            a changed original receipt or execution manifest, a changed
            incident record, or a changed freeze.
    """
    problems: list[str] = []

    pinned = set(manifest.recovery_source_hashes)
    required = set(RECOVERY_SOURCE_RELATIVE_PATHS)
    added, removed = sorted(required - pinned), sorted(pinned - required)
    if added or removed:
        problems.append(
            f"the recovery source set changed -- newly required: {added or 'none'}; no "
            f"longer required: {removed or 'none'}"
        )
    current = build_source_hashes(repo_root, relative_paths=tuple(pinned))
    drifted = sorted(p for p, h in manifest.recovery_source_hashes.items() if current.get(p) != h)
    if drifted:
        problems.append(f"recovery source file(s) changed since sealing: {drifted}")

    try:
        originals = assert_originals_preserved()
    except RecoveryError as exc:
        problems.append(str(exc))
        originals = {}
    if originals:
        if originals["execution_start_receipt"] != manifest.original_receipt_sha256:
            problems.append("the ORIGINAL execution-start receipt changed since sealing")
        if originals["execution_manifest"] != manifest.original_execution_manifest_sha256:
            problems.append("the ORIGINAL execution manifest changed since sealing")
        if originals["incident_record"] != manifest.incident_record_sha256:
            problems.append("the incident record changed since sealing")

    try:
        live_artifacts = verify_2025_artifacts()
    except RecoveryError as exc:
        problems.append(str(exc))
        live_artifacts = {}
    for name, entry in manifest.artifacts_2025.items():
        if live_artifacts.get(name, {}).get("sha256") != entry["sha256"]:
            problems.append(f"2025 artifact bytes changed since sealing: {name}")

    freeze = read_freeze()
    if freeze.freeze_content_hash() != manifest.freeze_content_hash:
        problems.append("the research freeze changed since this recovery manifest was sealed")
    if freeze.spec_content_hash != manifest.spec_content_hash:
        problems.append("the specification changed since this recovery manifest was sealed")

    if problems:
        raise RecoveryError("Recovery manifest validation FAILED: " + "; ".join(problems))
    return {
        "recovery_manifest_hash": manifest.manifest_content_hash(),
        "recovery_sources_verified": len(pinned),
        "artifacts_2025_verified": len(manifest.artifacts_2025),
        "originals_preserved": True,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# The recovery receipt -- designed, never written by this task
# ---------------------------------------------------------------------------


def assert_no_recovery_receipt(path: Path | None = None) -> None:
    """One recovery attempt only.

    Raises:
        RecoveryError: if a recovery receipt already exists.
    """
    path = path if path is not None else RECOVERY_RECEIPT_PATH
    if path.exists():
        receipt = json.loads(path.read_text())
        raise RecoveryError(
            f"A recovery receipt already exists at {path} (recovery_id "
            f"{receipt.get('recovery_id')}, opened {receipt.get('reopened_at_utc')}). The "
            "recovery authorization permitted ONE attempt and it is spent. A further "
            "attempt requires a new explicit incident decision -- never an automatic "
            "retry."
        )


def write_recovery_receipt(
    manifest: RecoveryManifest, *, path: Path | None = None, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Record that 2025 is about to be RE-opened, before the first read.

    Separate from the original execution-start receipt, which is preserved
    untouched. Not called by any code path in this task.

    Raises:
        RecoveryError: if a recovery receipt already exists.
    """
    path = path if path is not None else RECOVERY_RECEIPT_PATH
    assert_no_recovery_receipt(path)
    receipt = {
        "recovery_id": secrets.token_hex(8),
        "reopened_at_utc": datetime.now(UTC).isoformat(),
        "supersedes_execution_id": manifest.original_execution_id,
        "original_receipt_sha256": manifest.original_receipt_sha256,
        "recovery_manifest_hash": manifest.manifest_content_hash(),
        "freeze_content_hash": manifest.freeze_content_hash,
        "spec_content_hash": manifest.spec_content_hash,
        "repository_commit": get_git_commit_hash(repo_root),
        "is_a_first_look": False,
        "statement": RECOVERY_IS_NOT_A_FIRST_LOOK,
        "one_recovery_attempt_only": True,
        "no_automatic_retry": (
            "A failure after this point requires a new explicit incident decision. Do not "
            "delete this receipt and retry."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


# ---------------------------------------------------------------------------
# The recovery gate
# ---------------------------------------------------------------------------


def assert_ready_for_recovery(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Every gate a recovery run must pass. Refuses today, by design.

    Deliberately does NOT re-validate the failed run's execution manifest:
    that manifest pins the code that failed, the runner has since been
    corrected, and the resulting drift is the very thing it exists to record.
    It is preserved and hash-checked, never resealed.

    Raises:
        RecoveryError: if the recovery authorization is absent, or any
            integrity or state precondition fails.
    """
    freeze = read_freeze()
    validate_freeze(freeze, repo_root)
    assert_originals_preserved()
    verify_2025_artifacts()

    manifest = read_recovery_manifest()
    report = validate_recovery_manifest(manifest, repo_root)

    is_clean, dirty = working_tree_status(repo_root)
    if not is_clean:
        raise RecoveryError(f"Working tree is not clean; refusing to proceed: {dirty}")
    assert_no_recovery_receipt()

    authorized, reasons = resolve_recovery_authorization(manifest.manifest_content_hash())
    if not authorized:
        raise RecoveryError(
            "No maintainer authorization for this recovery. The original one-time "
            "authorization was CONSUMED by execution "
            f"{manifest.original_execution_id} and may not be reused as though unspent. A "
            "resumption needs a new explicit sign-off bound to recovery manifest "
            f"{manifest.manifest_content_hash()}: " + "; ".join(reasons)
        )
    return {**report, "ready": True}  # pragma: no cover -- unreachable until authorized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recovery control for the failed 2025 pitcher replication. Seals and validates "
            "the recovery manifest and reports readiness. Opens no season."
        )
    )
    parser.add_argument("--seal-recovery-manifest", action="store_true")
    parser.add_argument("--check-recovery-readiness", action="store_true")
    parser.add_argument("--verify-2025-artifacts", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_2025_artifacts:
        print(json.dumps(verify_2025_artifacts(), indent=2, sort_keys=True))
        print("Byte hashes only; contents were not parsed. 2025 was NOT re-opened.")
        return 0

    if args.seal_recovery_manifest:
        manifest = build_recovery_manifest()
        path, digest = write_recovery_manifest(manifest)
        report = validate_recovery_manifest(manifest)
        print(f"=== Sealed {path} ===")
        print(f"recovery_manifest_hash = {digest}")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("2025 was NOT re-opened.")
        return 0

    if args.check_recovery_readiness:
        try:
            print(json.dumps(assert_ready_for_recovery(), indent=2, sort_keys=True))
            return 0
        except (RecoveryError, ExecutionError) as exc:
            print(f"NOT READY: {exc}")
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
