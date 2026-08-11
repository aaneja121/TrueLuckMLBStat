"""Contact Luck: durable archival of official Version 1.1 prospective
snapshots to Cloudflare R2.

THE PROBLEM THIS SOLVES: `outputs/prospective/v1_1/` and
`artifacts/prospective/v1_1/` are gitignored (see CLAUDE.md "Version 1.1")
and, when a snapshot is produced by a GitHub Actions run, exist only on
that run's disposable runner -- they vanish when the job ends. This module
copies a completed local snapshot's files, byte for byte, into a durable
R2 bucket, keyed by data-through date (and snapshot label), so an official
snapshot survives runner destruction.

WHAT THIS MODULE DOES NOT DO: it does not score anything
(`prospective/run_v1_1_2026_scoring.py` owns that), does not decide
whether a snapshot is valid or complete (Version 1.1's own guards already
ran before this is ever called), and does not change what the dashboard
reads (the dashboard -- `dashboard/snapshot_data.py` -- continues to read
only from local disk; this module is a durability backstop, not a new
runtime dependency of the dashboard -- see "Recovery" below for the one
place that changes).

## Archive layout

Mirrors the existing two-namespace local contract exactly (never
inventing a reduced representation) -- for a snapshot directory named
`<snapshot_dir_name>` (`YYYY-MM-DD`, or `YYYY-MM-DD__<label>` exactly
matching `_snapshot_dir_name` in `run_v1_1_2026_scoring.py` and
`parse_snapshot_directory_name` in `dashboard/snapshot_data.py`):

    prospective/<season>/<snapshot_dir_name>/outputs/<every file under
        outputs/prospective/v1_1/<snapshot_dir_name>/, enumerated
        dynamically -- never a hardcoded filename list, so a future
        Version 1.1 output file is archived automatically>
    prospective/<season>/<snapshot_dir_name>/artifacts/<every file under
        artifacts/prospective/v1_1/<snapshot_dir_name>/, same way>

`<season>` comes from the snapshot's own `manifest.json` (`prospective_season`),
not a hardcoded "2026", so this generalizes to a future prospective version's
own season without a code change here.

## Write-once / identity semantics

`artifacts/integrity_hashes.json` already contains a sha256 per output file
plus the manifest's own hash (written by `run_v1_1_2026_scoring.py`) -- this
module reuses it AS the identity anchor rather than inventing a second
hashing scheme:

1. If the archive has no `artifacts/integrity_hashes.json` yet for this
   `<snapshot_dir_name>`: this is a new archive entry. Upload every local
   file, then re-fetch the just-written `integrity_hashes.json` from R2 and
   confirm it matches byte-for-byte what was intended (catches a corrupted/
   incomplete upload before declaring success).
2. If the archive already has one: fetch it and compare (as parsed JSON,
   not raw bytes, so formatting differences don't cause a false conflict)
   against the local copy.
   - Identical -> no-op (`ArchiveOutcome.NO_OP_IDENTICAL`). This is what
     makes a rerun of an already-archived date safe.
   - Different -> `ArchiveConflictError` -- NEVER silently overwritten.
     A corrected/refreshed snapshot (a distinct `snapshot_label`, e.g.
     `2026-08-08__refreshed`) is a DIFFERENT `<snapshot_dir_name>` and so
     archives as its own, separate, additional entry -- exactly mirroring
     how the local directories and the dashboard's own precedence rules
     (`dashboard/snapshot_data.py`) already treat it. This module never
     decides precedence between them; it only stores both.

## Recovery

`restore_snapshot()` / `--restore` is the explicit, on-demand reverse
operation for ONE known snapshot: given a data-through date (and optional
label) plus its season, downloads that snapshot's files from R2 back into
the normal local `outputs/prospective/v1_1/<snapshot_dir_name>/` /
`artifacts/prospective/v1_1/<snapshot_dir_name>/` paths, verifying every
downloaded file against the archived `integrity_hashes.json` before
declaring success, and refusing (never silently overwriting) if different
local files already exist at that path.

## History sync (repopulating a disposable CI runner)

A GitHub Actions runner starts from a fresh git checkout -- `outputs/
prospective/v1_1/`/`artifacts/prospective/v1_1/` are gitignored, so a CI job
that just scored today's date has ONLY today's snapshot locally. Without
more, `dashboard/build.py`'s season-to-date trend charts would show a
single point on every CI-built dashboard, no matter how much history is
sitting in R2. `list_archived_snapshots()` / `sync_missing_snapshots()` /
`--sync-history` close that gap:

- `list_archived_snapshots(client, season=...)` enumerates every candidate
  `<snapshot_dir_name>` under `prospective/<season>/` (never a hardcoded
  date list). A key only becomes a CANDIDATE at all if the path segment
  right after `<season>/` has the SHAPE of a real snapshot directory name
  (`YYYY-MM-DD` or `YYYY-MM-DD__<label>`, with the date portion a genuine
  calendar date) -- a key that reaches the right depth but whose name
  doesn't look like a snapshot at all (some unrelated object that happens
  to sit under this prefix) is treated as genuinely unrelated and silently
  ignored, never reported. Every candidate that DOES pass the shape check
  then gets a COMPLETENESS determination: complete only if its `artifacts/
  integrity_hashes.json` object exists, parses as JSON, AND every relative
  path it lists is also actually present as an object in the archive -- see
  `ArchivedSnapshotInfo.complete`/`.problem`.
- `sync_missing_snapshots(season=...)` restores every COMPLETE archived
  snapshot that is missing locally, using the exact same `restore_snapshot()`
  machinery (same hash verification, same "never overwrite" guarantee). For
  a `<snapshot_dir_name>` that already exists locally, it checks whether the
  local copy is identical to the archive (by comparing `integrity_hashes.json`
  AND re-hashing every referenced local file) -- identical is a safe no-op;
  anything else (missing `integrity_hashes.json`, a hash mismatch, a
  referenced file that's missing or corrupted) raises
  `HistorySyncConflictError` rather than silently overwriting or attempting
  automatic repair. A candidate that passed the shape check (so it IS a real
  snapshot directory, by name) but failed the completeness check raises
  `HistorySyncIncompleteArchiveError` -- a broken OFFICIAL snapshot entry is
  never silently skipped, it stops the sync so a human investigates. Only a
  genuinely UNRELATED key (never became a candidate in the first place) is
  ignored without blocking anything else from syncing down.

This is the ONLY way R2 data reaches this repository's local filesystem --
nothing else auto-restores, and the dashboard build and the prospective
scoring guards are UNCHANGED by this module: they still only ever read
local disk (see `dashboard/snapshot_data.py`). `scripts/publish_snapshot.sh`
runs history sync as an explicit pipeline stage, after archiving and before
the dashboard build, in BOTH dry-run and production modes (dry-run performs
only R2 reads -- no archive writes, no Cloudflare Pages deploy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROSPECTIVE_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "prospective" / "v1_1"
PROSPECTIVE_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "prospective" / "v1_1"

DEFAULT_ARCHIVE_PREFIX = "prospective"
INTEGRITY_HASHES_FILENAME = "integrity_hashes.json"
MANIFEST_FILENAME = "manifest.json"

__all__ = [
    "ArchiveClient",
    "ArchiveConflictError",
    "ArchiveError",
    "ArchiveOutcome",
    "ArchiveResult",
    "ArchivedSnapshotInfo",
    "HistorySyncConflictError",
    "HistorySyncIncompleteArchiveError",
    "HistorySyncResult",
    "InMemoryArchiveClient",
    "archive_snapshot",
    "build_snapshot_dir_name",
    "discover_local_snapshot_files",
    "list_archived_snapshots",
    "restore_snapshot",
    "split_snapshot_dir_name",
    "sync_missing_snapshots",
]


class ArchiveError(Exception):
    """Base class for archival failures -- any of these must prevent a
    subsequent deploy (see scripts/publish_snapshot.sh's ordering).
    """


class ArchiveConflictError(ArchiveError):
    """A DIFFERENT snapshot already exists in the archive at this
    snapshot_dir_name. Never overwritten -- this is the write-once
    guarantee. Resolve by using a distinct --snapshot-label, exactly as
    prospective/run_v1_1_2026_scoring.py's own SnapshotConflictError
    already requires for the LOCAL snapshot this was derived from.
    """


class UploadVerificationError(ArchiveError):
    """The archive was just written, but re-fetching it did not match what
    was sent -- treated as a failure rather than trusting a silent partial
    upload.
    """


class RestoreConflictError(ArchiveError):
    """restore_snapshot() found a local snapshot already at the target
    path that differs from the archived one -- never silently overwritten.
    """


class HistorySyncConflictError(ArchiveError):
    """sync_missing_snapshots() found a local snapshot directory that
    already exists for an archived <snapshot_dir_name> but is partial,
    corrupt, or does not match the archive -- never auto-repaired or
    overwritten. Resolve by hand (move/remove the local directory) before
    retrying.
    """


class HistorySyncIncompleteArchiveError(ArchiveError):
    """sync_missing_snapshots() found a candidate whose name has the SHAPE
    of a real snapshot directory (YYYY-MM-DD or YYYY-MM-DD__<label>) but
    failed list_archived_snapshots()'s completeness check -- a broken
    OFFICIAL archive entry, not a genuinely unrelated key. Never silently
    skipped: this stops the whole sync so a human investigates the archive
    directly, rather than quietly proceeding with a historical record
    that's missing a real date.
    """


class ArchiveOutcome(StrEnum):
    WRITTEN = "written"
    NO_OP_IDENTICAL = "no_op_identical"


@dataclass(frozen=True)
class ArchiveResult:
    outcome: ArchiveOutcome
    snapshot_dir_name: str
    season: int
    keys_written: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchivedSnapshotInfo:
    """One candidate <snapshot_dir_name> discovered under prospective/<season>/.

    Only reaches this dataclass at all if its name has the SHAPE of a real
    snapshot directory (YYYY-MM-DD or YYYY-MM-DD__<label>, a genuine
    calendar date) -- a key that doesn't even look like a snapshot is
    treated as unrelated and never becomes an ArchivedSnapshotInfo. Given
    that, `complete=False` means "this IS meant to be an official snapshot,
    but it's broken" -- missing artifacts/integrity_hashes.json, a
    malformed/unparseable one, or one that references a file not actually
    present in the archive. sync_missing_snapshots() treats that as a hard
    failure (HistorySyncIncompleteArchiveError), never a silent skip.
    """

    snapshot_dir_name: str
    season: int
    complete: bool
    problem: str | None = None


@dataclass(frozen=True)
class HistorySyncResult:
    season: int
    restored: tuple[str, ...] = ()
    already_present: tuple[str, ...] = ()


class ArchiveClient(Protocol):
    """The operations this module needs from an object store -- kept
    intentionally small so tests can implement it with a plain in-memory
    dict (`InMemoryArchiveClient` below) instead of contacting real
    Cloudflare R2 or requiring a mocking library.
    """

    def object_exists(self, key: str) -> bool: ...

    def get_object_bytes(self, key: str) -> bytes: ...

    def put_object_bytes(self, key: str, data: bytes) -> None: ...

    def list_keys(self, prefix: str) -> list[str]: ...


class InMemoryArchiveClient:
    """Test double -- an ArchiveClient backed by a plain dict, never
    touching the network. Used by tests/test_archive_snapshot.py and
    tests/test_archive_history_sync.py; never used in production.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def object_exists(self, key: str) -> bool:
        return key in self._objects

    def get_object_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def put_object_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self._objects if key.startswith(prefix))


def make_r2_client(
    *, bucket: str, account_id: str, access_key_id: str, secret_access_key: str
) -> ArchiveClient:
    """The real, production ArchiveClient -- boto3's S3 client pointed at
    Cloudflare R2's S3-compatible endpoint. Imports boto3 lazily so that
    importing this module (e.g. for its pure functions, or in tests that
    only use InMemoryArchiveClient) never requires boto3 to be installed
    unless this specific factory is actually called.
    """
    import boto3  # noqa: PLC0415 -- deliberately lazy, see docstring
    from botocore.config import Config  # noqa: PLC0415
    from botocore.exceptions import ClientError  # noqa: PLC0415

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    class _R2Client:
        def object_exists(self, key: str) -> bool:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    return False
                raise

        def get_object_bytes(self, key: str) -> bytes:
            response: Any = s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()

        def put_object_bytes(self, key: str, data: bytes) -> None:
            s3.put_object(Bucket=bucket, Key=key, Body=data)

        def list_keys(self, prefix: str) -> list[str]:
            keys: list[str] = []
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                response: Any = s3.list_objects_v2(**kwargs)
                keys.extend(obj["Key"] for obj in response.get("Contents", []))
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")
            return keys

    return _R2Client()


def build_snapshot_dir_name(data_through_date: str, snapshot_label: str | None) -> str:
    """Mirrors `_snapshot_dir_name` in run_v1_1_2026_scoring.py exactly,
    without importing it (same read-only-boundary reasoning as
    dashboard/snapshot_data.py's own copy of this formatting rule).
    """
    return f"{data_through_date}__{snapshot_label}" if snapshot_label else data_through_date


def split_snapshot_dir_name(snapshot_dir_name: str) -> tuple[str, str | None]:
    """The inverse of build_snapshot_dir_name() -- mirrors
    dashboard/snapshot_data.py's parse_snapshot_directory_name() exactly,
    without importing it.
    """
    if "__" in snapshot_dir_name:
        date_part, label_part = snapshot_dir_name.split("__", 1)
        return date_part, label_part
    return snapshot_dir_name, None


def discover_local_snapshot_files(outputs_dir: Path, artifacts_dir: Path) -> dict[str, Path]:
    """Every file actually present under the two local snapshot
    directories, mapped to its archive-relative key suffix
    (`outputs/<name>` / `artifacts/<name>`). Enumerated dynamically -- NOT
    a hardcoded filename list -- so a future Version 1.1 output file is
    archived automatically without a code change here.
    """
    files: dict[str, Path] = {}
    for label, root in (("outputs", outputs_dir), ("artifacts", artifacts_dir)):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_file():
                files[f"{label}/{path.name}"] = path
    return files


def _r2_key(prefix: str, season: int, snapshot_dir_name: str, relative: str) -> str:
    return f"{prefix}/{season}/{snapshot_dir_name}/{relative}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


_SNAPSHOT_DIR_NAME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:__.+)?$")


def _is_valid_snapshot_dir_name_shape(name: str) -> bool:
    """True only if `name` has the exact shape a real snapshot directory
    name would (YYYY-MM-DD or YYYY-MM-DD__<label>) AND its date portion is
    a genuine calendar date -- this is the line between "an unrelated key
    that happens to sit under prospective/<season>/" (safe to ignore) and
    "this IS supposed to be an official snapshot" (an incomplete/malformed
    one of these must fail loudly, never be silently skipped -- see
    HistorySyncIncompleteArchiveError).
    """
    match = _SNAPSHOT_DIR_NAME_PATTERN.match(name)
    if not match:
        return False
    try:
        date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return True


def _parse_snapshot_dir_names(keys: list[str], *, prefix: str, season: int) -> set[str]:
    """Every distinct <snapshot_dir_name> segment implied by a set of R2
    keys under prospective/<season>/ -- NOT a hardcoded date list. A key
    is excluded (treated as genuinely unrelated, never reported) if either:
    it doesn't reach a <snapshot_dir_name>/<category>/<file> depth, or its
    <snapshot_dir_name> segment doesn't have the shape of a real snapshot
    directory name (see _is_valid_snapshot_dir_name_shape). Anything that
    DOES pass both checks is a genuine snapshot candidate from here on --
    list_archived_snapshots()/sync_missing_snapshots() then decide whether
    it's complete, but never whether it counts as a snapshot at all.
    """
    season_prefix = f"{prefix}/{season}/"
    names: set[str] = set()
    for key in keys:
        if not key.startswith(season_prefix):
            continue
        remainder = key[len(season_prefix) :]
        parts = remainder.split("/", 2)
        if len(parts) < 3:
            # Doesn't reach <snapshot_dir_name>/<outputs|artifacts>/<file>
            # depth -- not a key this archive contract ever writes, so it's
            # not evidence of a real snapshot. Ignored, not an error: see
            # "R2 listing safety" in the module docstring.
            continue
        candidate = parts[0]
        if not _is_valid_snapshot_dir_name_shape(candidate):
            # Reaches the right depth but doesn't look like a snapshot
            # directory at all -- genuinely unrelated, safe to ignore.
            continue
        names.add(candidate)
    return names


def list_archived_snapshots(
    client: ArchiveClient, *, season: int, prefix: str = DEFAULT_ARCHIVE_PREFIX
) -> list[ArchivedSnapshotInfo]:
    """Enumerate every candidate <snapshot_dir_name> under
    prospective/<season>/ and determine which are COMPLETE (safe to
    restore) -- see "History sync" in the module docstring for the exact
    completeness rule. Never contacts anything outside this one season
    prefix, and never hardcodes a specific date.
    """
    season_prefix = f"{prefix}/{season}/"
    all_keys = client.list_keys(season_prefix)
    all_keys_set = set(all_keys)
    candidate_names = _parse_snapshot_dir_names(all_keys, prefix=prefix, season=season)

    results: list[ArchivedSnapshotInfo] = []
    for name in sorted(candidate_names):
        integrity_key = _r2_key(prefix, season, name, f"artifacts/{INTEGRITY_HASHES_FILENAME}")
        if integrity_key not in all_keys_set:
            results.append(
                ArchivedSnapshotInfo(
                    snapshot_dir_name=name,
                    season=season,
                    complete=False,
                    problem=f"missing artifacts/{INTEGRITY_HASHES_FILENAME}",
                )
            )
            continue

        try:
            integrity = json.loads(client.get_object_bytes(integrity_key))
        except json.JSONDecodeError as exc:
            results.append(
                ArchivedSnapshotInfo(
                    snapshot_dir_name=name,
                    season=season,
                    complete=False,
                    problem=f"{INTEGRITY_HASHES_FILENAME} is not valid JSON: {exc}",
                )
            )
            continue

        missing_files = sorted(
            relative
            for relative in integrity
            if _r2_key(prefix, season, name, relative) not in all_keys_set
        )
        if missing_files:
            results.append(
                ArchivedSnapshotInfo(
                    snapshot_dir_name=name,
                    season=season,
                    complete=False,
                    problem=f"missing archived file(s): {missing_files}",
                )
            )
            continue

        results.append(ArchivedSnapshotInfo(snapshot_dir_name=name, season=season, complete=True))

    return results


def archive_snapshot(
    *,
    data_through_date: str,
    snapshot_label: str | None,
    client: ArchiveClient,
    outputs_root: Path = PROSPECTIVE_OUTPUTS_ROOT,
    artifacts_root: Path = PROSPECTIVE_ARTIFACTS_ROOT,
    prefix: str = DEFAULT_ARCHIVE_PREFIX,
) -> ArchiveResult:
    """Archive one already-completed local snapshot. Raises ArchiveError
    (never returns a "failed" result silently) if anything is wrong --
    see module docstring for the write-once/identity rules.
    """
    snapshot_dir_name = build_snapshot_dir_name(data_through_date, snapshot_label)
    outputs_dir = outputs_root / snapshot_dir_name
    artifacts_dir = artifacts_root / snapshot_dir_name

    manifest_path = artifacts_dir / MANIFEST_FILENAME
    integrity_path = artifacts_dir / INTEGRITY_HASHES_FILENAME
    if not manifest_path.exists() or not integrity_path.exists():
        raise ArchiveError(
            f"Local snapshot {snapshot_dir_name} is missing {MANIFEST_FILENAME}/"
            f"{INTEGRITY_HASHES_FILENAME} under {artifacts_dir} -- refusing to archive an "
            "incomplete snapshot."
        )

    manifest = _load_json(manifest_path)
    season = int(manifest["prospective_season"])
    local_integrity = _load_json(integrity_path)
    integrity_key = _r2_key(
        prefix, season, snapshot_dir_name, f"artifacts/{INTEGRITY_HASHES_FILENAME}"
    )

    if client.object_exists(integrity_key):
        remote_integrity = json.loads(client.get_object_bytes(integrity_key))
        if remote_integrity == local_integrity:
            return ArchiveResult(
                outcome=ArchiveOutcome.NO_OP_IDENTICAL,
                snapshot_dir_name=snapshot_dir_name,
                season=season,
            )
        raise ArchiveConflictError(
            f"An archived snapshot already exists at prefix={prefix} season={season} "
            f"snapshot_dir_name={snapshot_dir_name} and its integrity hashes DIFFER from the "
            "local snapshot -- refusing to overwrite. Use a distinct --snapshot-label to "
            "archive this as a separate, additional entry."
        )

    local_files = discover_local_snapshot_files(outputs_dir, artifacts_dir)
    if not local_files:
        raise ArchiveError(
            f"No local files found for snapshot {snapshot_dir_name} -- nothing to archive."
        )

    keys_written = []
    for relative, local_path in local_files.items():
        key = _r2_key(prefix, season, snapshot_dir_name, relative)
        client.put_object_bytes(key, local_path.read_bytes())
        keys_written.append(key)

    round_trip = json.loads(client.get_object_bytes(integrity_key))
    if round_trip != local_integrity:
        raise UploadVerificationError(
            f"Uploaded {integrity_key} but re-fetching it did not match what was sent -- "
            "treating this as a failed archive rather than trusting a silent partial upload."
        )

    return ArchiveResult(
        outcome=ArchiveOutcome.WRITTEN,
        snapshot_dir_name=snapshot_dir_name,
        season=season,
        keys_written=tuple(sorted(keys_written)),
    )


def restore_snapshot(
    *,
    data_through_date: str,
    snapshot_label: str | None,
    season: int,
    client: ArchiveClient,
    outputs_root: Path = PROSPECTIVE_OUTPUTS_ROOT,
    artifacts_root: Path = PROSPECTIVE_ARTIFACTS_ROOT,
    prefix: str = DEFAULT_ARCHIVE_PREFIX,
) -> ArchiveResult:
    """The explicit, on-demand reverse of archive_snapshot() -- see module
    docstring "Recovery". `season` must be supplied by the caller (unlike
    archive_snapshot(), there is no local manifest.json to read it from
    when nothing local exists yet -- that is exactly the case this
    function is for).
    """
    snapshot_dir_name = build_snapshot_dir_name(data_through_date, snapshot_label)
    integrity_relative = f"artifacts/{INTEGRITY_HASHES_FILENAME}"
    integrity_key = _r2_key(prefix, season, snapshot_dir_name, integrity_relative)
    if not client.object_exists(integrity_key):
        raise ArchiveError(
            f"No archived snapshot found at prefix={prefix} season={season} "
            f"snapshot_dir_name={snapshot_dir_name}."
        )
    integrity_bytes = client.get_object_bytes(integrity_key)
    remote_integrity: dict[str, str] = json.loads(integrity_bytes)

    outputs_dir = outputs_root / snapshot_dir_name
    artifacts_dir = artifacts_root / snapshot_dir_name

    # Every file the archive knows about for this snapshot: the files
    # LISTED INSIDE integrity_hashes.json, plus integrity_hashes.json
    # itself (which, being the anchor, never lists its own hash) -- both
    # must come back for the restored directory to pass
    # dashboard/snapshot_data.py's own integrity check, exactly as it would
    # for a snapshot that was never archived at all.
    all_relative_paths = [*remote_integrity.keys(), integrity_relative]

    for relative in all_relative_paths:
        dest_root = outputs_dir if relative.startswith("outputs/") else artifacts_dir
        local_path = dest_root / Path(relative).name
        if local_path.exists():
            raise RestoreConflictError(
                f"{local_path} already exists locally -- restore_snapshot() never "
                "overwrites existing local files. Move or remove it first if you intend "
                "to replace it with the archived copy."
            )

    written: list[Path] = []
    for relative in all_relative_paths:
        dest_root = outputs_dir if relative.startswith("outputs/") else artifacts_dir
        dest_root.mkdir(parents=True, exist_ok=True)
        dest_path = dest_root / Path(relative).name
        data = (
            integrity_bytes
            if relative == integrity_relative
            else client.get_object_bytes(_r2_key(prefix, season, snapshot_dir_name, relative))
        )
        dest_path.write_bytes(data)
        written.append(dest_path)

    for relative, expected_hash in remote_integrity.items():
        dest_root = outputs_dir if relative.startswith("outputs/") else artifacts_dir
        dest_path = dest_root / Path(relative).name
        actual_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ArchiveError(
                f"Restored {dest_path} does not match its archived hash -- restore failed "
                "verification."
            )

    return ArchiveResult(
        outcome=ArchiveOutcome.WRITTEN,
        snapshot_dir_name=snapshot_dir_name,
        season=season,
        keys_written=tuple(sorted(str(p) for p in written)),
    )


def _local_snapshot_matches_archive(
    *, outputs_dir: Path, artifacts_dir: Path, remote_integrity: dict[str, str]
) -> bool:
    """True only if a LOCAL snapshot directory that already exists is
    byte-for-byte identical to what the archive has -- both its own
    integrity_hashes.json content AND every file it references actually
    present locally with a matching hash. Anything else (missing
    integrity_hashes.json, a content mismatch, a referenced file missing or
    corrupted) returns False, which sync_missing_snapshots() treats as a
    conflict to fail loudly on, never to repair.
    """
    local_integrity_path = artifacts_dir / INTEGRITY_HASHES_FILENAME
    if not local_integrity_path.exists():
        return False
    try:
        local_integrity = json.loads(local_integrity_path.read_text())
    except json.JSONDecodeError:
        return False
    if local_integrity != remote_integrity:
        return False
    for relative, expected_hash in local_integrity.items():
        dest_root = outputs_dir if relative.startswith("outputs/") else artifacts_dir
        dest_path = dest_root / Path(relative).name
        if not dest_path.exists():
            return False
        if hashlib.sha256(dest_path.read_bytes()).hexdigest() != expected_hash:
            return False
    return True


def sync_missing_snapshots(
    *,
    season: int,
    client: ArchiveClient,
    outputs_root: Path = PROSPECTIVE_OUTPUTS_ROOT,
    artifacts_root: Path = PROSPECTIVE_ARTIFACTS_ROOT,
    prefix: str = DEFAULT_ARCHIVE_PREFIX,
) -> HistorySyncResult:
    """Restore every COMPLETE archived snapshot for `season` that is
    missing locally -- see "History sync" in the module docstring. Never
    overwrites or repairs an existing local snapshot: identical -> no-op;
    anything else -> HistorySyncConflictError, raised immediately (the
    whole sync stops at the first conflict, exactly like archive_snapshot()
    stopping at the first problem it finds -- a conflict means something is
    genuinely wrong and needs a human, not a partial sync papering over it).
    A candidate whose NAME has the shape of a real snapshot directory but
    fails the completeness check raises HistorySyncIncompleteArchiveError,
    same fail-immediately treatment -- a broken official snapshot entry is
    never silently skipped. Only a key that never became a candidate at all
    (genuinely unrelated -- see list_archived_snapshots()) is ignored
    without blocking anything else from syncing down.
    """
    archived = list_archived_snapshots(client, season=season, prefix=prefix)

    restored: list[str] = []
    already_present: list[str] = []

    for info in archived:
        if not info.complete:
            raise HistorySyncIncompleteArchiveError(
                f"Archived snapshot {info.snapshot_dir_name} (season={season}, "
                f"prefix={prefix}) has the name of a real snapshot but is incomplete or "
                f"malformed: {info.problem}. Refusing to sync history with a broken "
                "official archive entry -- investigate directly in R2 before retrying."
            )

        outputs_dir = outputs_root / info.snapshot_dir_name
        artifacts_dir = artifacts_root / info.snapshot_dir_name

        if outputs_dir.exists() or artifacts_dir.exists():
            integrity_key = _r2_key(
                prefix, season, info.snapshot_dir_name, f"artifacts/{INTEGRITY_HASHES_FILENAME}"
            )
            remote_integrity = json.loads(client.get_object_bytes(integrity_key))
            if _local_snapshot_matches_archive(
                outputs_dir=outputs_dir,
                artifacts_dir=artifacts_dir,
                remote_integrity=remote_integrity,
            ):
                already_present.append(info.snapshot_dir_name)
                continue
            raise HistorySyncConflictError(
                f"Local snapshot directory already exists for {info.snapshot_dir_name} "
                f"(under {outputs_dir} / {artifacts_dir}) but does not match the archive -- "
                "it is either partial, corrupt, or genuinely different. Refusing to overwrite "
                "or auto-repair; resolve by hand before retrying."
            )

        data_through_date, snapshot_label = split_snapshot_dir_name(info.snapshot_dir_name)
        restore_snapshot(
            data_through_date=data_through_date,
            snapshot_label=snapshot_label,
            season=season,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
            prefix=prefix,
        )
        restored.append(info.snapshot_dir_name)

    return HistorySyncResult(
        season=season,
        restored=tuple(restored),
        already_present=tuple(already_present),
    )


def _client_from_env() -> ArchiveClient:
    bucket = os.environ.get("R2_BUCKET_NAME")
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    missing = [
        name
        for name, value in (
            ("R2_BUCKET_NAME", bucket),
            ("R2_ACCOUNT_ID", account_id),
            ("R2_ACCESS_KEY_ID", access_key_id),
            ("R2_SECRET_ACCESS_KEY", secret_access_key),
        )
        if not value
    ]
    if missing:
        raise ArchiveError(
            f"Missing required environment variable(s) for R2 archival: {', '.join(missing)}"
        )
    assert bucket and account_id and access_key_id and secret_access_key  # narrows for mypy
    return make_r2_client(
        bucket=bucket,
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-through",
        default=None,
        help="YYYY-MM-DD, matching the local snapshot. Required unless --sync-history.",
    )
    parser.add_argument("--snapshot-label", default=None)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Download ONE archived snapshot back to local disk instead of archiving one.",
    )
    parser.add_argument(
        "--sync-history",
        action="store_true",
        help=(
            "Restore every archived snapshot for --season that is missing locally, then exit. "
            "Mutually exclusive with --restore/archiving; does not need --data-through."
        ),
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Required with --restore or --sync-history (no local manifest.json to read it from).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    bucket_for_log = None
    try:
        bucket_for_log = os.environ.get("R2_BUCKET_NAME")
        client = _client_from_env()

        if args.sync_history:
            if args.season is None:
                raise ArchiveError("--sync-history requires --season (see --help).")
            sync_result = sync_missing_snapshots(season=args.season, client=client)
            print(
                f"bucket={bucket_for_log} sync season={sync_result.season} "
                f"restored={len(sync_result.restored)} already_present={len(sync_result.already_present)}"
            )
            for name in sync_result.restored:
                print(f"  restored: {name}")
            return 0

        if args.restore:
            if args.season is None:
                raise ArchiveError("--restore requires --season (see --help).")
            if args.data_through is None:
                raise ArchiveError("--restore requires --data-through (see --help).")
            result = restore_snapshot(
                data_through_date=args.data_through,
                snapshot_label=args.snapshot_label,
                season=args.season,
                client=client,
            )
        else:
            if args.data_through is None:
                raise ArchiveError(
                    "--data-through is required unless --sync-history is used (see --help)."
                )
            result = archive_snapshot(
                data_through_date=args.data_through,
                snapshot_label=args.snapshot_label,
                client=client,
            )
    except ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"bucket={bucket_for_log} outcome={result.outcome.value} "
        f"snapshot_dir_name={result.snapshot_dir_name} season={result.season} "
        f"keys={len(result.keys_written)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
