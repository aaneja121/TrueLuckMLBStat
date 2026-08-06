"""Contact Luck v1.1: the per-snapshot manifest and read-only v1.0 seal check.

Records everything a reviewer needs to trust one dated prospective snapshot:
repository commit, frozen model/score versions, the requested vs. observed
2026 date coverage, retrieval provenance, row counts, schema/join-coverage
summaries, the qualification threshold set, deterministic seeds, and (once
outputs are written) their hashes. Computes no predictions itself -- it only
RECORDS the frozen state of the scoring logic (reusing `evaluation.
v1_final_evaluation_manifest`'s artifact-hashing machinery directly, since
Version 1.1 freezes the exact same Version 1.0 code) and the run-specific
facts `prospective.run_v1_1_2026_scoring` gathers along the way.

## Why no "seal" ceremony

Version 1.0's `EvaluationSeal` exists because that was a single, one-time
event that must never silently rerun. Version 1.1 is the opposite: a tool
meant to run repeatedly across a season. Immutability here instead comes from
`prospective.run_v1_1_2026_scoring`'s refuse-to-overwrite-a-different-
snapshot + accept-an-identical-rerun-as-a-no-op logic, driven by
`SnapshotManifest.deterministic_content_hash()` below (deliberately excludes
wall-clock fields like `generated_at`/`retrieval_timestamps` so two runs of
the same `--data-through` date against the same code and the same upstream
data compare equal even though their wall-clock timestamps differ).

## The v1.0 seal check is read-only and non-fatal

`verify_v1_seal_unchanged` NEVER reads 2025 data -- it only re-parses and
re-validates `artifacts/final_evaluation/v1/seal.json`'s own JSON structure
(reusing `evaluation.run_v1_final_evaluation.read_existing_seal`/`SealError`
directly) and reports a result. "Remains unchanged" here means the seal file
still parses as a well-formed `EvaluationSeal` with every required field
present and non-empty -- this module keeps no separate historical reference
copy to diff byte-for-byte against, so it cannot detect a sophisticated
in-place forgery that preserves valid structure. A missing or corrupted seal
file is reported (`verified=False`) but never raises and never blocks 2026
scoring: 2026 scoring does not depend on 2025 data at all, so a v1.0 seal
problem is a finding for a human reviewer, not a reason to stop this run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prospective_config import PROSPECTIVE_SEASON, SEALED_V1_SEAL_PATH
from v1_final_evaluation_manifest import (
    FROZEN_ARTIFACT_RELATIVE_PATHS,
    ManifestError,
    build_artifact_hashes,
    compute_file_sha256,
    get_git_commit_hash,
)
from v1_final_evaluation_manifest import _model_versions as _v1_model_versions
from v1_final_evaluation_manifest import _score_version as _v1_score_version

MANIFEST_VERSION = "1.1.0"

__all__ = [
    "MANIFEST_VERSION",
    "ManifestError",
    "SnapshotConflictError",
    "SnapshotManifest",
    "V1SealVerificationResult",
    "build_snapshot_manifest",
    "compute_file_sha256",
    "verify_v1_seal_unchanged",
    "with_output_hashes",
]


class SnapshotConflictError(ValueError):
    """Raised when a snapshot for the same data-through date already exists
    and differs from the one this run would produce -- see `prospective.
    run_v1_1_2026_scoring` for the refuse-vs-idempotent-no-op logic.
    """


def is_working_tree_clean(repo_root: Path) -> bool:
    """Never raises -- this low-level function only RECORDS whether the tree
    was clean, in `SnapshotManifest.working_tree_clean`. The actual BLOCKING
    enforcement (a dirty tree must refuse a real snapshot run -- see
    CLAUDE.md "Version 1.1: prospective 2026 scoring") lives one layer up,
    in `prospective.prospective_ingestion.run_prospective_guards`
    (`assert_clean_working_tree`, reused from Version 1.0), which every real
    snapshot run goes through before `build_snapshot_manifest` is ever
    called. This function stays non-raising so it can still be called in
    isolation (e.g. for diagnostics) without that side effect.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return not result.stdout.strip()


@dataclass(frozen=True)
class V1SealVerificationResult:
    """Outcome of the read-only, non-fatal Version 1.0 seal-integrity check."""

    verified: bool
    seal_path: str
    seal_exists: bool
    detail: str
    sealed_at: str | None = None
    v1_repository_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_v1_seal_unchanged(
    seal_path: Path = SEALED_V1_SEAL_PATH,
) -> V1SealVerificationResult:
    """Read-only, NEVER raises, NEVER reads any 2025 data other than the
    seal file's own JSON structure. See module docstring for exactly what
    "remains unchanged" means here.
    """
    import sys

    # `read_existing_seal`/`SealError` live in `run_v1_final_evaluation`, which
    # (unlike `v1_final_evaluation_manifest`) imports heavy pipeline modules at
    # module scope -- imported lazily so building this manifest module never
    # requires the full Version 1.0 entry point to be importable.
    if "run_v1_final_evaluation" in sys.modules:
        rv1 = sys.modules["run_v1_final_evaluation"]
    else:
        import run_v1_final_evaluation as rv1  # type: ignore[no-redef]

    if not seal_path.exists():
        return V1SealVerificationResult(
            verified=False,
            seal_path=str(seal_path),
            seal_exists=False,
            detail="No Version 1.0 seal file found at this path.",
        )
    try:
        seal = rv1.read_existing_seal(seal_path)
    except rv1.SealError as exc:
        return V1SealVerificationResult(
            verified=False,
            seal_path=str(seal_path),
            seal_exists=True,
            detail=f"Seal file exists but failed validation: {exc}",
        )
    if seal is None:
        return V1SealVerificationResult(
            verified=False,
            seal_path=str(seal_path),
            seal_exists=False,
            detail="No Version 1.0 seal file found at this path.",
        )
    return V1SealVerificationResult(
        verified=True,
        seal_path=str(seal_path),
        seal_exists=True,
        detail="Seal file parses as a well-formed EvaluationSeal.",
        sealed_at=seal.sealed_at,
        v1_repository_commit=seal.repository_commit,
    )


@dataclass(frozen=True)
class SnapshotManifest:
    manifest_version: str
    repository_commit: str
    working_tree_clean: bool
    prospective_season: int
    model_versions: dict[str, str]
    score_version: str
    qualification_threshold_set: str
    requested_date_range: list[str]
    observed_date_coverage: list[str]
    data_through_date: str
    snapshot_label: str | None
    retrieval_timestamps: dict[str, str]
    source_hashes: dict[str, str]
    raw_row_counts: dict[str, int]
    processed_row_count: int
    schema_checks: dict[str, Any]
    join_coverage: dict[str, Any]
    deterministic_seeds: dict[str, int]
    v1_seal_verification: dict[str, Any]
    generated_at: str
    frozen_artifact_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotManifest:
        return cls(**data)

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def deterministic_content_hash(self) -> str:
        """Hash of everything EXCEPT wall-clock fields (`generated_at`,
        `retrieval_timestamps`), the freshly-assigned `v1_seal_verification`
        (its own `sealed_at`/`v1_repository_commit` are about a DIFFERENT,
        independently-sealed run and carry no information about whether THIS
        2026 snapshot's own inputs/outputs are identical to a prior run of
        the same `--data-through` date), and `output_hashes` (only known
        AFTER writing output files -- excluding it means this hash can be
        compared consistently whether or not output files have been written
        yet). Used to decide whether a rerun is a true idempotent no-op vs. a
        genuine conflict -- see `SnapshotConflictError`.
        """
        payload = self.to_dict()
        payload.pop("generated_at", None)
        payload.pop("retrieval_timestamps", None)
        payload.pop("v1_seal_verification", None)
        payload.pop("output_hashes", None)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot_manifest(
    repo_root: Path,
    *,
    data_through_date: str,
    requested_date_range: tuple[str, str],
    observed_date_coverage: list[str],
    snapshot_label: str | None,
    retrieval_timestamps: dict[str, str],
    source_hashes: dict[str, str],
    raw_row_counts: dict[str, int],
    processed_row_count: int,
    schema_checks: dict[str, Any],
    join_coverage: dict[str, Any],
    qualification_threshold_set: str,
    output_hashes: dict[str, str] | None = None,
    v1_seal_path: Path = SEALED_V1_SEAL_PATH,
) -> SnapshotManifest:
    """Assemble a full snapshot manifest. Frozen model/score versions and
    frozen-code artifact hashes are read directly from `evaluation.
    v1_final_evaluation_manifest` -- Version 1.1 freezes the exact same
    Version 1.0 logic, so nothing about those values is re-derived here.

    `v1_seal_path` is accepted as a genuine parameter (not just read off the
    `prospective_config.SEALED_V1_SEAL_PATH` module global at call time)
    specifically so tests can point it at a synthetic seal fixture -- a
    default bound to a module-level Path at *function-definition* time would
    not pick up a `monkeypatch.setattr` on the module constant.

    Raises:
        ManifestError: if a frozen artifact required by Version 1.0 is
            missing (propagated from `build_artifact_hashes`).
    """
    return SnapshotManifest(
        manifest_version=MANIFEST_VERSION,
        repository_commit=get_git_commit_hash(repo_root),
        working_tree_clean=is_working_tree_clean(repo_root),
        prospective_season=PROSPECTIVE_SEASON,
        model_versions=_v1_model_versions(),
        score_version=_v1_score_version(),
        qualification_threshold_set=qualification_threshold_set,
        requested_date_range=list(requested_date_range),
        observed_date_coverage=list(observed_date_coverage),
        data_through_date=data_through_date,
        snapshot_label=snapshot_label,
        retrieval_timestamps=dict(retrieval_timestamps),
        source_hashes=dict(source_hashes),
        raw_row_counts=dict(raw_row_counts),
        processed_row_count=processed_row_count,
        schema_checks=schema_checks,
        join_coverage=join_coverage,
        deterministic_seeds={"model_random_seed": 42, "bootstrap_seed": 42},
        v1_seal_verification=verify_v1_seal_unchanged(v1_seal_path).to_dict(),
        generated_at=datetime.now(UTC).isoformat(),
        frozen_artifact_hashes=build_artifact_hashes(
            repo_root, relative_paths=FROZEN_ARTIFACT_RELATIVE_PATHS
        ),
        output_hashes=dict(output_hashes or {}),
    )


def with_output_hashes(
    manifest: SnapshotManifest, output_hashes: dict[str, str]
) -> SnapshotManifest:
    """Return a copy of `manifest` with `output_hashes` populated -- output
    files don't exist until after scoring completes, so this is applied as a
    second step rather than threading hashes through `build_snapshot_
    manifest` itself.
    """
    data = manifest.to_dict()
    data["output_hashes"] = dict(output_hashes)
    return SnapshotManifest(**data)
