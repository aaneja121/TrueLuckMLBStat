"""Version 0.14: execution control for the one-time held-out 2025 pitcher
replication -- the pre-execution manifest, the readiness gate, the
unforgeable authorization token, and the immutable execution-start receipt.

This module opens no season. It reads and writes only small JSON control
artifacts inside `artifacts/pitcher_replication/v0_14/`.

## Why a SECOND manifest, separate from the research freeze

The Version 0.14 research specification was frozen at commit 6276461,
before any runner existed. Amending it to add the runner would rewrite a
pre-registration after the fact and destroy the evidence that the questions
preceded the execution code. So the freeze stays untouched, and this module
adds a PRE-EXECUTION manifest that binds four separate things together:

1. the existing frozen specification (by `freeze_content_hash` and
   `spec_content_hash`, never re-derived);
2. the existing maintainer authorization (by its own content hash);
3. the exact runner implementation and its helper modules (by SHA-256); and
4. the state of the world at sealing time -- commit, clean tree, empty
   namespace, 2025 unopened.

The research freeze answers "what were we going to measure?". This manifest
answers "with exactly what code, under whose sign-off, from what state?".

## The token: `allow_final_evaluation=True` has exactly one road in

`ReplicationAuthorization` is minted ONLY by `run_readiness_checks`, which
adds its token to a process-local set. `require_authorization` rejects
anything not in that set -- a bare `True`, a fabricated token, or a token
from another process. It is that function, and only that function, which
calls `assert_seasons_allowed([2025], allow_final_evaluation=True)`.

This mirrors `evaluation.run_v1_final_evaluation`'s design deliberately: it
is a within-process guard against a caller skipping the gate, not a
cross-process security boundary.

## The receipt marks the authorization CONSUMED at first look

`write_execution_start_receipt` is called after every readiness check passes
and BEFORE the first byte of 2025 is read. Once it exists, a normal rerun
refuses. That is the point: the authorization is spent when the held-out
data is opened, not when a run happens to finish successfully. There is no
automatic retry, and recovery from a post-receipt failure requires the
documented procedure in `RECOVERY_PROCEDURE`, not a silent second look.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pitcher_replication_freeze import (
    ARTIFACTS_DIR,
    FREEZE_PATH,
    REPLICATION_DATA_DIR,
    REPLICATION_OUTPUTS_DIR,
    REPO_ROOT,
    FreezeError,
    assert_2025_protection_intact,
    assert_ready_for_2025,
    build_source_hashes,
    compute_file_sha256,
    content_hash,
    display_path,
    get_git_commit_hash,
    read_freeze,
    resolve_authorization,
    validate_freeze,
    working_tree_status,
)
from pitcher_replication_spec import REPLICATION_SEASON

EXECUTION_MANIFEST_VERSION = "1.0.0"

EXECUTION_MANIFEST_PATH = ARTIFACTS_DIR / "pitcher_replication_execution_manifest.json"
EXECUTION_RECEIPT_PATH = ARTIFACTS_DIR / "execution_start_receipt.json"

#: Every file whose CONTENT this execution manifest pins -- the runner, its
#: helpers, and the authorization record. Distinct from the RESEARCH freeze's
#: `FROZEN_SOURCE_RELATIVE_PATHS`, which pins what is being measured; this
#: pins what does the measuring, plus whose sign-off permits it.
#:
#: `evaluation/run_v1_final_evaluation.py` is pinned because the runner reuses
#: its `train_and_score_2025` -- the already-reviewed frozen 2025 scoring
#: orchestration. Reimplementing that would have been a new research choice.
EXECUTION_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "replication/run_pitcher_replication_2025.py",
    "replication/pitcher_replication_execution.py",
    "replication/pitcher_replication_questions.py",
    "replication/pitcher_replication_authorization.py",
    "evaluation/run_v1_final_evaluation.py",
    "Makefile",
)

#: Control/manifest files permitted to exist in the artifacts namespace
#: before execution. Anything else means a run already happened.
PERMITTED_ARTIFACT_NAMES: frozenset[str] = frozenset(
    {
        FREEZE_PATH.name,
        EXECUTION_MANIFEST_PATH.name,
        "amendments.jsonl",
        ".gitkeep",
    }
)

RECOVERY_PROCEDURE = (
    "If a run fails AFTER the execution-start receipt exists, 2025 has already been "
    "opened and the authorization is spent. Do NOT delete the receipt and re-run. The "
    "documented recovery is: (1) preserve the receipt and every partial output; (2) "
    "report to the maintainer exactly how far the run got and what was observed, "
    "including 'nothing was inspected' if that is true; (3) obtain a fresh, explicit "
    "written authorization that names the failure and permits a resumption, recorded the "
    "same way the original sign-off was; (4) record it as a new authorization revision "
    "with its own receipt. A rerun without that is a second look at held-out data "
    "wearing the first look's paperwork."
)


class ExecutionError(RuntimeError):
    """Raised when execution control refuses to proceed."""


# ---------------------------------------------------------------------------
# Unforgeable authorization token
# ---------------------------------------------------------------------------

#: Tokens minted by `run_readiness_checks` -- the ONLY function permitted to
#: add to this set. Process-local by design.
_ISSUED_REPLICATION_TOKENS: set[str] = set()


@dataclass(frozen=True)
class ReplicationAuthorization:
    """Proof that `run_readiness_checks` actually ran and passed. Every
    function that can reach 2025 requires this OBJECT, never a bare bool.
    """

    token: str
    freeze_content_hash: str
    execution_manifest_hash: str
    granted_at: str


def _mint_authorization(freeze_hash: str, execution_manifest_hash: str) -> ReplicationAuthorization:
    token = secrets.token_hex(16)
    _ISSUED_REPLICATION_TOKENS.add(token)
    return ReplicationAuthorization(
        token=token,
        freeze_content_hash=freeze_hash,
        execution_manifest_hash=execution_manifest_hash,
        granted_at=datetime.now(UTC).isoformat(),
    )


def require_authorization(authorization: Any, fn_name: str) -> None:
    """The single choke point through which `allow_final_evaluation=True`
    becomes reachable.

    Raises:
        ExecutionError: if `authorization` is not a `ReplicationAuthorization`
            minted by `run_readiness_checks` in this process.
    """
    from mlb_luck_score.config import assert_seasons_allowed

    if not isinstance(authorization, ReplicationAuthorization):
        raise ExecutionError(
            f"{fn_name} requires a ReplicationAuthorization minted by "
            f"run_readiness_checks -- got {type(authorization).__name__!r}. A bare "
            "boolean is never accepted, regardless of its value."
        )
    if authorization.token not in _ISSUED_REPLICATION_TOKENS:
        raise ExecutionError(
            f"{fn_name} requires a ReplicationAuthorization minted by "
            "run_readiness_checks -- a fabricated token, or one from a different or "
            "earlier process, is never accepted."
        )
    assert_seasons_allowed([REPLICATION_SEASON], allow_final_evaluation=True)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


def assert_within_namespace(path: Path, root: Path, *, label: str) -> None:
    """Refuse any path outside the isolated replication namespace, including
    via `..` traversal.

    Raises:
        ExecutionError: if `path` is not inside `root`.
    """
    resolved, root_resolved = path.resolve(), root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ExecutionError(
            f"{label} path {resolved} is outside the isolated pitcher-replication "
            f"namespace {root_resolved}. Refusing: this guard is what stops a typo or a "
            "'..' traversal from writing 2025 into a development cache, the Version 1.0 "
            "final-evaluation namespace, prospective/, or a dashboard fixture."
        )


def assert_output_namespace_available() -> dict[str, Any]:
    """The output namespace must hold nothing but permitted control files.

    Raises:
        ExecutionError: if a prior run's data or outputs are present.
    """
    stray_data = (
        sorted(p.name for p in REPLICATION_DATA_DIR.rglob("*") if p.is_file())
        if REPLICATION_DATA_DIR.exists()
        else []
    )
    stray_outputs = (
        sorted(p.name for p in REPLICATION_OUTPUTS_DIR.rglob("*") if p.is_file())
        if REPLICATION_OUTPUTS_DIR.exists()
        else []
    )
    stray_artifacts = (
        sorted(
            p.name
            for p in ARTIFACTS_DIR.rglob("*")
            if p.is_file()
            and p.name not in PERMITTED_ARTIFACT_NAMES
            and "superseded" not in p.parts
        )
        if ARTIFACTS_DIR.exists()
        else []
    )
    stray_data = [n for n in stray_data if n != ".gitkeep"]
    stray_outputs = [n for n in stray_outputs if n != ".gitkeep"]

    if stray_data or stray_outputs or stray_artifacts:
        raise ExecutionError(
            "The pitcher-replication namespace is already populated beyond its permitted "
            f"control files -- data: {stray_data or 'none'}; outputs: "
            f"{stray_outputs or 'none'}; artifacts: {stray_artifacts or 'none'}. A "
            "one-time held-out evaluation refuses to overwrite a previous one."
        )
    return {
        "data_dir_clear": True,
        "outputs_dir_clear": True,
        "artifacts_dir_has_only_control_files": True,
    }


# ---------------------------------------------------------------------------
# Pre-execution manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionManifest:
    execution_manifest_version: str
    freeze_content_hash: str
    spec_content_hash: str
    freeze_repository_commit: str
    authorization_content_hash: str
    authorized_freeze_content_hash: str
    authorized_at_utc: str
    execution_source_hashes: dict[str, str]
    repository_commit: str
    working_tree_clean: bool
    intended_output_paths: dict[str, str]
    replication_season: int
    date_range: list[str]
    preconditions: dict[str, Any]
    sealed_at_utc: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionManifest:
        return cls(**data)

    def manifest_content_hash(self) -> str:
        payload = self.to_dict()
        for volatile in ("sealed_at_utc", "notes"):
            payload.pop(volatile, None)
        return content_hash(payload)


def _authorization_content_hash() -> str:
    from pitcher_replication_authorization import authorization_record

    return content_hash(authorization_record())


def build_execution_manifest(
    repo_root: Path = REPO_ROOT, *, notes: list[str] | None = None
) -> ExecutionManifest:
    """Assemble the pre-execution manifest.

    Runs every precondition first, so a manifest can never be sealed in a
    state that would not have been allowed to run.

    Raises:
        FreezeError / ExecutionError: on any failed precondition.
    """
    from pitcher_replication_authorization import (
        AUTHORIZED_AT_UTC,
        AUTHORIZED_FREEZE_CONTENT_HASH,
    )
    from run_pitcher_replication_2025 import PITCHER_REPLICATION_2025_DATE_RANGE

    freeze = read_freeze()
    validate_freeze(freeze, repo_root)
    binds, reasons = resolve_authorization(freeze)
    if not binds:
        raise ExecutionError(f"authorization does not bind to the freeze: {reasons}")
    protection = assert_2025_protection_intact()
    namespace = assert_output_namespace_available()
    is_clean, dirty = working_tree_status(repo_root)
    if not is_clean:
        raise ExecutionError(
            f"Working tree is not clean; refusing to seal an execution manifest: {dirty}"
        )
    assert_no_execution_receipt()

    return ExecutionManifest(
        execution_manifest_version=EXECUTION_MANIFEST_VERSION,
        freeze_content_hash=freeze.freeze_content_hash(),
        spec_content_hash=freeze.spec_content_hash,
        freeze_repository_commit=freeze.repository_commit,
        authorization_content_hash=_authorization_content_hash(),
        authorized_freeze_content_hash=AUTHORIZED_FREEZE_CONTENT_HASH,
        authorized_at_utc=AUTHORIZED_AT_UTC,
        execution_source_hashes=build_source_hashes(
            repo_root, relative_paths=EXECUTION_SOURCE_RELATIVE_PATHS
        ),
        repository_commit=get_git_commit_hash(repo_root),
        working_tree_clean=is_clean,
        intended_output_paths={
            "raw_data_dir": display_path(REPLICATION_DATA_DIR),
            "outputs_dir": display_path(REPLICATION_OUTPUTS_DIR),
            "artifacts_dir": display_path(ARTIFACTS_DIR),
            "receipt": display_path(EXECUTION_RECEIPT_PATH),
        },
        replication_season=REPLICATION_SEASON,
        date_range=list(PITCHER_REPLICATION_2025_DATE_RANGE),
        preconditions={
            "freeze_validates": True,
            "frozen_sources_verified": len(freeze.source_hashes),
            "authorization_resolves": True,
            "namespace_empty": namespace,
            "season_protection": protection,
            "season_2025_opened": False,
            "season_2026_opened": False,
            "receipt_exists": False,
        },
        sealed_at_utc=datetime.now(UTC).isoformat(),
        notes=notes or [],
    )


def _serialize(obj: Any) -> str:
    return json.dumps(obj.to_dict(), indent=2, sort_keys=True, default=str) + "\n"


def read_execution_manifest(path: Path | None = None) -> ExecutionManifest:
    path = path if path is not None else EXECUTION_MANIFEST_PATH
    if not path.is_file():
        raise ExecutionError(
            f"No pre-execution manifest at {path} -- seal one before any 2025 access "
            "(make seal-pitcher-replication-execution)."
        )
    return ExecutionManifest.from_dict(json.loads(path.read_text()))


def write_execution_manifest(
    manifest: ExecutionManifest, *, path: Path | None = None
) -> tuple[Path, str]:
    """Write-once. An identical reseal is idempotent; changed execution code
    is refused.

    There is deliberately NO amendment mechanism that rewrites the original:
    a changed runner needs a new manifest built from a new commit, so the
    original sealing record can never be silently edited after the fact.

    Raises:
        ExecutionError: if a manifest exists with different content.
    """
    path = path if path is not None else EXECUTION_MANIFEST_PATH
    new_hash = manifest.manifest_content_hash()
    if path.exists():
        existing = read_execution_manifest(path)
        existing_hash = existing.manifest_content_hash()
        if existing_hash == new_hash:
            return (path, existing_hash)
        raise ExecutionError(
            f"A DIFFERENT execution manifest already exists at {path} (existing "
            f"{existing_hash}, new {new_hash}). It is write-once, and there is no "
            "amendment path that rewrites it. The execution code changed after sealing: "
            "commit the change and seal a NEW manifest, so the original record of what "
            "was sealed, and when, survives intact."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(manifest))
    return (path, new_hash)


def validate_execution_manifest(
    manifest: ExecutionManifest, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Re-derive every hash the manifest pinned and compare.

    Raises:
        ExecutionError: on source drift, a changed required source set, or a
            freeze/authorization the manifest does not name.
    """
    problems: list[str] = []

    pinned = set(manifest.execution_source_hashes)
    required = set(EXECUTION_SOURCE_RELATIVE_PATHS)
    added, removed = sorted(required - pinned), sorted(pinned - required)
    if added or removed:
        problems.append(
            f"the execution source set changed -- newly required: {added or 'none'}; no "
            f"longer required: {removed or 'none'}"
        )

    current = build_source_hashes(repo_root, relative_paths=tuple(pinned))
    drifted = sorted(p for p, h in manifest.execution_source_hashes.items() if current.get(p) != h)
    if drifted:
        problems.append(f"execution source file(s) changed since sealing: {drifted}")

    freeze = read_freeze()
    if freeze.freeze_content_hash() != manifest.freeze_content_hash:
        problems.append("the freeze changed since this execution manifest was sealed")
    if freeze.spec_content_hash != manifest.spec_content_hash:
        problems.append("the specification changed since this execution manifest was sealed")
    if _authorization_content_hash() != manifest.authorization_content_hash:
        problems.append("the authorization record changed since this manifest was sealed")

    if problems:
        raise ExecutionError("Execution manifest validation FAILED: " + "; ".join(problems))
    return {
        "execution_manifest_hash": manifest.manifest_content_hash(),
        "execution_sources_verified": len(pinned),
        "freeze_content_hash": manifest.freeze_content_hash,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Execution-start receipt
# ---------------------------------------------------------------------------


def assert_no_execution_receipt(path: Path | None = None) -> None:
    """The authorization is one-time; a receipt means it is already spent.

    `path` resolves from the module attribute at CALL time. That is
    load-bearing: a definition-time default cannot be redirected, and a test
    that believed it had been redirected once wrote a receipt into the real
    namespace.

    Raises:
        ExecutionError: if a receipt exists.
    """
    path = path if path is not None else EXECUTION_RECEIPT_PATH
    if path.exists():
        receipt = json.loads(path.read_text())
        raise ExecutionError(
            "An execution-start receipt already exists at "
            f"{path} (execution_id {receipt.get('execution_id')}, opened at "
            f"{receipt.get('opened_at_utc')}). 2025 has already been opened under this "
            "authorization, which was for ONE evaluation. Refusing a second look.\n\n"
            + RECOVERY_PROCEDURE
        )


def write_execution_start_receipt(
    authorization: ReplicationAuthorization,
    manifest: ExecutionManifest,
    *,
    path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Record that 2025 is about to be opened, BEFORE the first read.

    Called after every readiness check passes and before any ingestion. Once
    written, `assert_no_execution_receipt` refuses a rerun -- the
    authorization is consumed at first look, not at successful completion.

    Raises:
        ExecutionError: if a receipt already exists.
    """
    path = path if path is not None else EXECUTION_RECEIPT_PATH
    assert_no_execution_receipt(path)
    freeze = read_freeze()
    receipt = {
        "execution_id": secrets.token_hex(8),
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "replication_season": REPLICATION_SEASON,
        "freeze_content_hash": freeze.freeze_content_hash(),
        "spec_content_hash": freeze.spec_content_hash,
        "authorization_content_hash": _authorization_content_hash(),
        "authorized_freeze_content_hash": manifest.authorized_freeze_content_hash,
        "execution_manifest_hash": manifest.manifest_content_hash(),
        "repository_commit": get_git_commit_hash(repo_root),
        "authorization_token_granted_at": authorization.granted_at,
        "statement": (
            "Every readiness check passed and no 2025 data had been read at the moment "
            "this receipt was written. The one-time authorization is CONSUMED from this "
            "point forward, whether or not the run completes."
        ),
        "one_time_use": True,
        "recovery_procedure": RECOVERY_PROCEDURE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    path.write_text(payload)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return receipt


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def run_readiness_checks(
    repo_root: Path = REPO_ROOT,
) -> tuple[ReplicationAuthorization, ExecutionManifest, dict[str, Any]]:
    """Every gate that MUST pass before 2025 is touched, in order.

    Returns the ONLY `ReplicationAuthorization` this run will mint -- no
    other function in this package can create one, which is what makes
    `allow_final_evaluation=True` reachable through this path alone.

    Raises:
        FreezeError / ExecutionError: on any failure. Nothing is read or
            downloaded before this returns.
    """
    freeze = read_freeze()
    validate_freeze(freeze, repo_root)
    assert_ready_for_2025(freeze, repo_root=repo_root)

    manifest = read_execution_manifest()
    manifest_report = validate_execution_manifest(manifest, repo_root)

    assert_output_namespace_available()
    assert_no_execution_receipt()

    authorization = _mint_authorization(
        freeze.freeze_content_hash(), manifest.manifest_content_hash()
    )
    report = {
        "freeze_content_hash": freeze.freeze_content_hash(),
        "spec_content_hash": freeze.spec_content_hash,
        "frozen_sources_verified": len(freeze.source_hashes),
        "execution_manifest": manifest_report,
        "authorization_resolves": True,
        "namespace_available": True,
        "receipt_absent": True,
        "ready": True,
    }
    return authorization, manifest, report


def compute_file_hash(path: Path) -> str:
    """Re-exported for the runner's provenance records."""
    return compute_file_sha256(path)


__all__ = [
    "EXECUTION_MANIFEST_PATH",
    "EXECUTION_RECEIPT_PATH",
    "EXECUTION_SOURCE_RELATIVE_PATHS",
    "RECOVERY_PROCEDURE",
    "ExecutionError",
    "ExecutionManifest",
    "FreezeError",
    "ReplicationAuthorization",
    "assert_no_execution_receipt",
    "assert_output_namespace_available",
    "assert_within_namespace",
    "build_execution_manifest",
    "compute_file_hash",
    "read_execution_manifest",
    "require_authorization",
    "run_readiness_checks",
    "validate_execution_manifest",
    "write_execution_manifest",
    "write_execution_start_receipt",
]
