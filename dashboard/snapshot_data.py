"""Contact Luck v1.2: read-only snapshot discovery, integrity, and precedence.

This module is the ONLY place the dashboard decides which Version 1.1
prospective snapshot directories exist, whether each is trustworthy, and
which one to prefer when more than one exists for the same data-through
date. It never computes a Contact Luck value, never mutates a snapshot
directory, and never reads anything outside
`outputs/prospective/v1_1/<snapshot>/` and
`artifacts/prospective/v1_1/<snapshot>/`.

## Snapshot classification (the deterministic rule)

Every discovered directory pair is classified into exactly one of four
types, decided in this order:

1. `invalid_incomplete_snapshot` -- if ANY integrity check below fails.
   Checked first and unconditionally: a directory that fails integrity is
   never trusted enough to ask "genuine or corrected?" about.
2. `retrospective_backfill` -- integrity passed AND the manifest's
   `snapshot_label` equals the reserved value
   `dashboard_config.RETROSPECTIVE_BACKFILL_LABEL`
   (`"retrospective_backfill"`). No such snapshot exists in this repository
   yet; this branch exists so a future, separately-built backfill mechanism
   is classified correctly the moment it appears (see project memory on the
   2026-08-07 gap).
3. `corrected_prospective_snapshot` -- integrity passed AND `snapshot_label`
   is present but is neither `None` nor the reserved backfill label (e.g.
   `"refreshed"`). This is what `prospective/run_v1_1_2026_scoring.py`
   produces when it is deliberately rerun for the same `--data-through` date
   under a distinct `--snapshot-label` -- both directories stay on disk
   (Version 1.1 never overwrites), and the dashboard treats the later one as
   superseding the earlier for DISPLAY purposes only.
4. `genuine_prospective_snapshot` -- integrity passed AND `snapshot_label`
   is `None`, i.e. the ordinary unlabeled run for a `--data-through` date.

## Snapshot precedence (same data-through date)

Among the VALID (non-invalid) snapshots sharing one `data_through_date`:

- A `retrospective_backfill` is excluded from consideration whenever at
  least one genuine or corrected snapshot exists for that date -- a backfill
  must never supersede a real prospective snapshot. It is only chosen if it
  is the SOLE valid snapshot for that date.
- Among the remaining candidates, the one with the latest `generated_at`
  timestamp wins -- a later corrected/refreshed run supersedes an earlier
  one (or the original unlabeled run) for display, exactly matching how
  `2026-08-08__refreshed` (generated two minutes after `2026-08-08`) is
  meant to supersede it.
- Ties (identical `generated_at`, e.g. synthetic test fixtures) break on
  `directory_name` descending, for full determinism.

## Latest-snapshot resolution (across dates)

`resolve_latest_snapshot` picks the preferred snapshot (per the rule above)
for the maximum `data_through_date` among dates that have at least one
non-backfill preferred snapshot. A retrospective backfill is never chosen as
"latest," even if it is numerically the newest date on disk -- it can still
be browsed via `build_snapshot_history`, just never presented as the current
view.

## Integrity validation

A snapshot is valid only if, in order: both `outputs/.../<name>/` and
`artifacts/.../<name>/` directories exist; `manifest.json` and
`integrity_hashes.json` both exist and parse as JSON; the manifest has every
key this module depends on; the directory name's parsed date/label agree
with the manifest's own `data_through_date`/`snapshot_label` fields; and
every file `integrity_hashes.json` lists exists and its sha256 matches. Any
failure is recorded as a string in `SnapshotIntegrityResult.errors` and the
snapshot is excluded from `valid_snapshots()`, `resolve_preferred_snapshots()`,
and `resolve_latest_snapshot()` -- there is no partial-credit path and no
silent fallback to an invalid newer snapshot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dashboard_config import (
    PROSPECTIVE_ARTIFACTS_ROOT,
    PROSPECTIVE_OUTPUTS_ROOT,
    RETROSPECTIVE_BACKFILL_LABEL,
)

__all__ = [
    "SNAPSHOT_TYPE_CORRECTED",
    "SNAPSHOT_TYPE_GENUINE",
    "SNAPSHOT_TYPE_INVALID",
    "SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL",
    "DiscoveredSnapshot",
    "SnapshotHistoryEntry",
    "SnapshotIntegrityResult",
    "build_snapshot_history",
    "classify_snapshot_type",
    "compute_file_sha256",
    "discover_snapshots",
    "group_by_data_through_date",
    "invalid_snapshots",
    "parse_snapshot_directory_name",
    "resolve_latest_snapshot",
    "resolve_preferred_snapshots",
    "select_preferred_snapshot",
    "valid_snapshots",
]

SNAPSHOT_TYPE_GENUINE = "genuine_prospective_snapshot"
SNAPSHOT_TYPE_CORRECTED = "corrected_prospective_snapshot"
SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL = "retrospective_backfill"
SNAPSHOT_TYPE_INVALID = "invalid_incomplete_snapshot"

_REQUIRED_MANIFEST_KEYS = (
    "data_through_date",
    "snapshot_label",
    "generated_at",
    "score_version",
    "model_versions",
    "repository_commit",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_snapshot_directory_name(directory_name: str) -> tuple[str, str | None]:
    """Split `"2026-08-08__refreshed"` -> `("2026-08-08", "refreshed")`, or
    `"2026-08-08"` -> `("2026-08-08", None)`. Mirrors
    `prospective.run_v1_1_2026_scoring._snapshot_dir_name` exactly, without
    importing it (see module docstring on the read-only boundary). This is
    used only to cross-check against the manifest's own fields, never as the
    sole source of truth -- the manifest wins whenever it loads.
    """
    if "__" in directory_name:
        date_part, label_part = directory_name.split("__", 1)
        return date_part, label_part
    return directory_name, None


def classify_snapshot_type(*, snapshot_label: str | None, integrity_valid: bool) -> str:
    """The deterministic classification rule -- see module docstring."""
    if not integrity_valid:
        return SNAPSHOT_TYPE_INVALID
    if snapshot_label == RETROSPECTIVE_BACKFILL_LABEL:
        return SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL
    if snapshot_label is None:
        return SNAPSHOT_TYPE_GENUINE
    return SNAPSHOT_TYPE_CORRECTED


@dataclass(frozen=True)
class SnapshotIntegrityResult:
    valid: bool
    checked_at: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveredSnapshot:
    """One `outputs/`+`artifacts/` directory pair, as discovered on disk.

    `data_through_date`/`snapshot_label`/`generated_at`/`score_version`/
    `model_versions` come from the manifest when it loaded successfully;
    when it did not, `data_through_date`/`snapshot_label` fall back to the
    directory-name parse purely for diagnostic display, and the others are
    `None` -- `integrity.valid` is always `False` in that case, so nothing
    downstream treats these fallback values as trustworthy.
    """

    directory_name: str
    data_through_date: str
    snapshot_label: str | None
    snapshot_type: str
    outputs_dir: Path
    artifacts_dir: Path
    generated_at: str | None
    score_version: str | None
    model_versions: dict[str, str] | None
    integrity: SnapshotIntegrityResult


def _resolve_integrity_path(
    relative_key: str, outputs_dir: Path, artifacts_dir: Path
) -> Path | None:
    if relative_key.startswith("outputs/"):
        return outputs_dir / relative_key.split("/", 1)[1]
    if relative_key.startswith("artifacts/"):
        return artifacts_dir / relative_key.split("/", 1)[1]
    return None


def _load_and_validate_snapshot(
    outputs_dir: Path, artifacts_dir: Path, directory_name: str
) -> DiscoveredSnapshot:
    errors: list[str] = []
    checked_at = _now_iso()
    parsed_date, parsed_label = parse_snapshot_directory_name(directory_name)

    if not outputs_dir.is_dir():
        errors.append(f"outputs directory missing: {outputs_dir}")
    if not artifacts_dir.is_dir():
        errors.append(f"artifacts directory missing: {artifacts_dir}")

    manifest: dict[str, Any] | None = None
    integrity_hashes: dict[str, str] | None = None

    if not errors:
        manifest_path = artifacts_dir / "manifest.json"
        integrity_path = artifacts_dir / "integrity_hashes.json"
        if not manifest_path.exists():
            errors.append(f"missing manifest.json at {manifest_path}")
        if not integrity_path.exists():
            errors.append(f"missing integrity_hashes.json at {integrity_path}")

        if not errors:
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"manifest.json is not valid JSON: {exc}")
            try:
                integrity_hashes = json.loads(integrity_path.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"integrity_hashes.json is not valid JSON: {exc}")

    if manifest is not None:
        for key in _REQUIRED_MANIFEST_KEYS:
            if key not in manifest:
                errors.append(f"manifest.json missing required key '{key}'")

    if manifest is not None and not errors:
        manifest_date = manifest.get("data_through_date")
        manifest_label = manifest.get("snapshot_label")
        if manifest_date != parsed_date:
            errors.append(
                f"directory name date '{parsed_date}' does not match "
                f"manifest data_through_date '{manifest_date}'"
            )
        if manifest_label != parsed_label:
            errors.append(
                f"directory name label '{parsed_label}' does not match "
                f"manifest snapshot_label '{manifest_label}'"
            )

    if integrity_hashes is not None and not errors:
        for relative_key, expected_hash in integrity_hashes.items():
            file_path = _resolve_integrity_path(relative_key, outputs_dir, artifacts_dir)
            if file_path is None:
                errors.append(f"integrity_hashes.json has unrecognized key: {relative_key}")
                continue
            if not file_path.exists():
                errors.append(f"integrity-listed file missing: {relative_key}")
                continue
            actual_hash = compute_file_sha256(file_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"hash mismatch for {relative_key}: expected {expected_hash}, got {actual_hash}"
                )

    integrity_valid = not errors
    best_label = (
        manifest.get("snapshot_label", parsed_label) if manifest is not None else parsed_label
    )
    snapshot_type = classify_snapshot_type(
        snapshot_label=best_label, integrity_valid=integrity_valid
    )
    best_date = (manifest.get("data_through_date") if manifest is not None else None) or parsed_date

    return DiscoveredSnapshot(
        directory_name=directory_name,
        data_through_date=best_date,
        snapshot_label=best_label,
        snapshot_type=snapshot_type,
        outputs_dir=outputs_dir,
        artifacts_dir=artifacts_dir,
        generated_at=manifest.get("generated_at") if manifest is not None else None,
        score_version=manifest.get("score_version") if manifest is not None else None,
        model_versions=manifest.get("model_versions") if manifest is not None else None,
        integrity=SnapshotIntegrityResult(
            valid=integrity_valid, checked_at=checked_at, errors=tuple(errors)
        ),
    )


def _discover_directory_names(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def discover_snapshots(
    outputs_root: Path = PROSPECTIVE_OUTPUTS_ROOT,
    artifacts_root: Path = PROSPECTIVE_ARTIFACTS_ROOT,
) -> list[DiscoveredSnapshot]:
    """Discover every snapshot directory pair under the two v1.1 namespaces,
    sorted deterministically by directory name. Includes invalid ones (so a
    build can report them) -- callers that only want trustworthy snapshots
    should filter with `valid_snapshots()`.
    """
    names = sorted(
        _discover_directory_names(outputs_root) | _discover_directory_names(artifacts_root)
    )
    return [
        _load_and_validate_snapshot(outputs_root / name, artifacts_root / name, name)
        for name in names
    ]


def valid_snapshots(snapshots: Iterable[DiscoveredSnapshot]) -> list[DiscoveredSnapshot]:
    return [s for s in snapshots if s.integrity.valid]


def invalid_snapshots(snapshots: Iterable[DiscoveredSnapshot]) -> list[DiscoveredSnapshot]:
    return [s for s in snapshots if not s.integrity.valid]


def group_by_data_through_date(
    snapshots: Iterable[DiscoveredSnapshot],
) -> dict[str, list[DiscoveredSnapshot]]:
    grouped: dict[str, list[DiscoveredSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.data_through_date, []).append(snapshot)
    return grouped


def select_preferred_snapshot(
    snapshots_for_date: Sequence[DiscoveredSnapshot],
) -> DiscoveredSnapshot | None:
    """Apply the same-date precedence rule from the module docstring to a
    set of VALID snapshots sharing one `data_through_date`. Callers must
    pre-filter to valid snapshots -- this function does not check validity.
    """
    if not snapshots_for_date:
        return None
    non_backfill = [
        s for s in snapshots_for_date if s.snapshot_type != SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL
    ]
    candidates = non_backfill if non_backfill else list(snapshots_for_date)
    return max(candidates, key=lambda s: (s.generated_at or "", s.directory_name))


def resolve_preferred_snapshots(
    snapshots: Iterable[DiscoveredSnapshot],
) -> dict[str, DiscoveredSnapshot]:
    """One preferred snapshot per `data_through_date`, valid snapshots only.
    A date whose only snapshot(s) are invalid has no entry at all -- never a
    silent fallback to an invalid snapshot.
    """
    grouped = group_by_data_through_date(valid_snapshots(snapshots))
    result: dict[str, DiscoveredSnapshot] = {}
    for date, group in grouped.items():
        preferred = select_preferred_snapshot(group)
        if preferred is not None:
            result[date] = preferred
    return result


def resolve_latest_snapshot(snapshots: Iterable[DiscoveredSnapshot]) -> DiscoveredSnapshot | None:
    """The dashboard's single "latest" snapshot -- see module docstring
    "Latest-snapshot resolution."
    """
    preferred = resolve_preferred_snapshots(snapshots)
    eligible = {
        date: snap
        for date, snap in preferred.items()
        if snap.snapshot_type != SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL
    }
    if not eligible:
        return None
    return eligible[max(eligible)]


@dataclass(frozen=True)
class SnapshotHistoryEntry:
    """One row of "browse previous snapshots by date": the preferred
    snapshot for that date, plus every other valid snapshot found for the
    same date (superseded corrections, or a backfill that lost precedence)
    so the status page can show them without implying they are the current
    view.
    """

    data_through_date: str
    preferred: DiscoveredSnapshot | None
    all_valid_snapshots: tuple[DiscoveredSnapshot, ...] = field(default_factory=tuple)


def build_snapshot_history(snapshots: Iterable[DiscoveredSnapshot]) -> list[SnapshotHistoryEntry]:
    """All dates with at least one valid snapshot, newest first."""
    grouped = group_by_data_through_date(valid_snapshots(snapshots))
    entries = []
    for date in sorted(grouped, reverse=True):
        group = grouped[date]
        entries.append(
            SnapshotHistoryEntry(
                data_through_date=date,
                preferred=select_preferred_snapshot(group),
                all_valid_snapshots=tuple(sorted(group, key=lambda s: s.directory_name)),
            )
        )
    return entries
