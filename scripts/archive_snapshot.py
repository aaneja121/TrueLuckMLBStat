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
operation: given a data-through date (and optional label), downloads that
snapshot's files from R2 back into the normal local
`outputs/prospective/v1_1/<snapshot_dir_name>/` /
`artifacts/prospective/v1_1/<snapshot_dir_name>/` paths, verifying every
downloaded file against the archived `integrity_hashes.json` before
declaring success, and refusing (never silently overwriting) if different
local files already exist at that path. This is the ONLY way R2 data
reaches this repository's local filesystem -- nothing else auto-restores.
The dashboard build and the prospective scoring guards are UNCHANGED by
this module: they still only ever see local disk. If a future version
wants the dashboard (or CI) to pull automatically from R2 instead of
requiring this explicit step, that is a real architectural change
(a new runtime dependency on remote state) and deserves its own explicit
decision -- not something this module does quietly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
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
    "InMemoryArchiveClient",
    "archive_snapshot",
    "build_snapshot_dir_name",
    "discover_local_snapshot_files",
    "restore_snapshot",
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


class ArchiveOutcome(StrEnum):
    WRITTEN = "written"
    NO_OP_IDENTICAL = "no_op_identical"


@dataclass(frozen=True)
class ArchiveResult:
    outcome: ArchiveOutcome
    snapshot_dir_name: str
    season: int
    keys_written: tuple[str, ...] = ()


class ArchiveClient(Protocol):
    """The only three operations this module needs from an object store --
    kept intentionally tiny so tests can implement it with a plain
    in-memory dict (`InMemoryArchiveClient` below) instead of contacting
    real Cloudflare R2 or requiring a mocking library.
    """

    def object_exists(self, key: str) -> bool: ...

    def get_object_bytes(self, key: str) -> bytes: ...

    def put_object_bytes(self, key: str, data: bytes) -> None: ...


class InMemoryArchiveClient:
    """Test double -- an ArchiveClient backed by a plain dict, never
    touching the network. Used by tests/test_archive_snapshot.py; never
    used in production.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def object_exists(self, key: str) -> bool:
        return key in self._objects

    def get_object_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def put_object_bytes(self, key: str, data: bytes) -> None:
        self._objects[key] = data


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

    return _R2Client()


def build_snapshot_dir_name(data_through_date: str, snapshot_label: str | None) -> str:
    """Mirrors `_snapshot_dir_name` in run_v1_1_2026_scoring.py exactly,
    without importing it (same read-only-boundary reasoning as
    dashboard/snapshot_data.py's own copy of this formatting rule).
    """
    return f"{data_through_date}__{snapshot_label}" if snapshot_label else data_through_date


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
        "--data-through", required=True, help="YYYY-MM-DD, matching the local snapshot."
    )
    parser.add_argument("--snapshot-label", default=None)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Download an archived snapshot back to local disk instead of archiving one.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Required with --restore only (no local manifest.json to read it from).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    bucket_for_log = None
    try:
        bucket_for_log = os.environ.get("R2_BUCKET_NAME")
        client = _client_from_env()
        if args.restore:
            if args.season is None:
                raise ArchiveError("--restore requires --season (see --help).")
            result = restore_snapshot(
                data_through_date=args.data_through,
                snapshot_label=args.snapshot_label,
                season=args.season,
                client=client,
            )
        else:
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
