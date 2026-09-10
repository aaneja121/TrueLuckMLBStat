"""Read-only provenance audit of the R2 snapshot archive and the Actions runs
that might still hold a scored snapshot.

## Why this is a separate module and not a flag on `archive_snapshot.py`

`archive_snapshot.py`'s `ArchiveClient` declares `put_object_bytes`, and its
production implementation calls `s3.put_object`. Adding a `--list` flag there
would mean the audit ran through an object that CAN write, and "we did not
call the write path" is an assurance about intent rather than about capability.

This module's reader exposes `list_keys`, `get_bytes` and `head` and nothing
else. There is no put, upload, copy, delete or multipart operation anywhere in
it, and it never constructs `archive_snapshot`'s write-capable client. Only
three boto3 operations are reachable, named in `READ_ONLY_S3_OPERATIONS` and
asserted by `tests/test_audit_r2_snapshot_history.py`.

What IS reused from `archive_snapshot` is deliberately only the pure key
convention and the "what counts as a snapshot directory" rule. Re-deriving
either here would let the audit disagree with the archive about where an
object lives, and report a snapshot as missing because it looked in the wrong
place.

## What it answers

For each date the caller names: does the archive hold a complete snapshot, a
partial one, or nothing -- and if not, does an exact scored snapshot survive
as a GitHub Actions artifact, so the date could be recovered WITHOUT
re-scoring. Re-scoring produces a different manifest once the repository has
moved on (it records `repository_commit` and `generated_at`), so "recoverable
without re-scoring" and "re-creatable" are different questions and this
answers the first.

It parses no baseball data. Every file is treated as opaque bytes to be
hashed; nothing reads a row.

## Fail-closed

An unreadable manifest, a malformed integrity file, a missing side, an
orphaned object: all classify AMBIGUOUS_OR_PARTIAL. Nothing is reported
complete unless every file the snapshot's own integrity manifest names is
present AND hashes to the recorded value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pure helpers only -- key shaping and the snapshot-directory rule. Nothing
# imported here touches the network or writes anything. The test suite pins
# this allowlist so a future import cannot quietly widen it.
from archive_snapshot import (  # noqa: E402
    DEFAULT_ARCHIVE_PREFIX,
    INTEGRITY_HASHES_FILENAME,
    MANIFEST_FILENAME,
    _parse_snapshot_dir_names,
    _r2_key,
)

#: The ONLY S3 operations this module may perform. Any boto3 call outside this
#: set is a bug, and the tests assert the module's source contains no other.
READ_ONLY_S3_OPERATIONS: tuple[str, ...] = ("list_objects_v2", "get_object", "head_object")

#: Statuses a date can receive. `EXACT_PRIOR_SNAPSHOT_RECOVERABLE` never comes
#: from R2 -- it means R2 lacks the snapshot but an Actions artifact still
#: holds the scored one.
STATUS_ALREADY_ARCHIVED = "ALREADY_ARCHIVED"
STATUS_RECOVERABLE = "EXACT_PRIOR_SNAPSHOT_RECOVERABLE"
STATUS_NONE = "NO_SNAPSHOT_EXISTS"
STATUS_AMBIGUOUS = "AMBIGUOUS_OR_PARTIAL"

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Artifact names that carry a SCORED SNAPSHOT rather than only build output.
#: The publish workflow uploads `snapshot-manifest-<date>` (manifest +
#: integrity only) and `dashboard-dist-<date>` (the built site). Neither is
#: the snapshot itself, which is why this distinction is reported rather
#: than assumed.
_SNAPSHOT_ARTIFACT_HINTS = ("snapshot-manifest-", "prospective-snapshot-", "snapshot-")
_BUILD_ONLY_ARTIFACT_HINTS = ("dashboard-dist-",)


class AuditError(RuntimeError):
    """The audit cannot proceed. Never raised for a finding -- a missing or
    broken snapshot is a RESULT, reported as a status."""


@dataclass
class ReadOnlyR2Reader:
    """A deliberately narrow R2 reader: list, get, head. Nothing else.

    There is no `put`, `upload`, `copy`, `delete` or multipart method on this
    class, so an R2 write is not something this audit declines to do -- it is
    something it has no way to express.
    """

    bucket: str
    _s3: Any = field(repr=False)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._s3.list_objects_v2(**kwargs)
            keys.extend(item["Key"] for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return sorted(keys)
            token = response.get("NextContinuationToken")

    def head(self, key: str) -> dict[str, Any]:
        response = self._s3.head_object(Bucket=self.bucket, Key=key)
        return {
            "bytes": int(response["ContentLength"]),
            "last_modified": response["LastModified"].isoformat(),
        }

    def get_bytes(self, key: str) -> bytes:
        return self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()


def make_read_only_reader() -> ReadOnlyR2Reader:
    """Build the reader from the environment.

    Deliberately NOT `archive_snapshot.make_r2_client`, whose client can
    write. Credentials are read and handed straight to boto3; they are never
    logged, echoed, stored, or included in the report.
    """
    import boto3  # noqa: PLC0415 -- lazy, so importing this module needs no boto3

    required = ("R2_ACCOUNT_ID", "R2_BUCKET_NAME", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        # Names only. Never a value, not even a prefix.
        raise AuditError(f"missing required environment variable(s): {', '.join(missing)}")

    account_id = os.environ["R2_ACCOUNT_ID"]
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return ReadOnlyR2Reader(bucket=os.environ["R2_BUCKET_NAME"], _s3=s3)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_r2_date(
    reader: ReadOnlyR2Reader,
    *,
    date: str,
    season: int,
    prefix: str = DEFAULT_ARCHIVE_PREFIX,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Inventory every archived object for one date, and say whether it is
    a complete snapshot.

    Reports every snapshot directory whose name starts with the date, so a
    labelled variant (`<date>__refreshed`) is found rather than missed.
    """
    if not _DATE_PATTERN.match(date):
        raise AuditError(f"date {date!r} is not YYYY-MM-DD")

    all_keys = reader.list_keys(f"{prefix}/{season}/")
    candidates = sorted(
        name
        for name in _parse_snapshot_dir_names(all_keys, prefix=prefix, season=season)
        if name == date or name.startswith(f"{date}__")
    )

    if not candidates:
        return {
            "date": date,
            "status": STATUS_NONE,
            "snapshot_dir_names": [],
            "any_objects_exist": False,
            "object_keys": [],
            "detail": "no object under this season prefix names this date",
        }

    directories = [
        _audit_one_directory(
            reader,
            name=name,
            season=season,
            prefix=prefix,
            all_keys=all_keys,
            verify_hashes=verify_hashes,
        )
        for name in candidates
    ]
    complete = [d for d in directories if d["complete"]]
    status = STATUS_ALREADY_ARCHIVED if len(complete) == 1 else STATUS_AMBIGUOUS
    if len(complete) > 1:
        detail = f"{len(complete)} complete snapshot directories claim this date"
    elif complete:
        detail = "one complete, hash-verified snapshot"
    else:
        detail = "objects exist for this date but no directory is complete"
    return {
        "date": date,
        "status": status,
        "snapshot_dir_names": candidates,
        "any_objects_exist": True,
        "object_keys": sorted(k for d in directories for k in d["object_keys"]),
        "directories": directories,
        "detail": detail,
    }


def _audit_one_directory(
    reader: ReadOnlyR2Reader,
    *,
    name: str,
    season: int,
    prefix: str,
    all_keys: list[str],
    verify_hashes: bool,
) -> dict[str, Any]:
    """One snapshot directory, inventoried and verified. Fails closed."""
    dir_prefix = f"{prefix}/{season}/{name}/"
    keys = [k for k in all_keys if k.startswith(dir_prefix)]
    relatives = {k[len(dir_prefix) :] for k in keys}

    record: dict[str, Any] = {
        "snapshot_dir_name": name,
        "object_keys": sorted(keys),
        "n_objects": len(keys),
        "outputs_side_present": any(r.startswith("outputs/") for r in relatives),
        "artifacts_side_present": any(r.startswith("artifacts/") for r in relatives),
        "manifest_present": f"artifacts/{MANIFEST_FILENAME}" in relatives,
        "integrity_present": f"artifacts/{INTEGRITY_HASHES_FILENAME}" in relatives,
        "complete": False,
        "problems": [],
    }

    if not record["integrity_present"]:
        record["problems"].append(f"missing artifacts/{INTEGRITY_HASHES_FILENAME}")
        return record
    if not record["manifest_present"]:
        record["problems"].append(f"missing artifacts/{MANIFEST_FILENAME}")

    integrity_key = _r2_key(prefix, season, name, f"artifacts/{INTEGRITY_HASHES_FILENAME}")
    try:
        integrity = json.loads(reader.get_bytes(integrity_key))
    except json.JSONDecodeError as exc:
        record["problems"].append(f"{INTEGRITY_HASHES_FILENAME} is not valid JSON: {exc}")
        return record
    if not isinstance(integrity, dict) or not integrity:
        record["problems"].append(f"{INTEGRITY_HASHES_FILENAME} records no files")
        return record
    record["n_files_recorded"] = len(integrity)

    # Manifest provenance -- read as JSON metadata only, never as data rows.
    if record["manifest_present"]:
        manifest_key = _r2_key(prefix, season, name, f"artifacts/{MANIFEST_FILENAME}")
        manifest_bytes = reader.get_bytes(manifest_key)
        record["manifest_sha256"] = _sha256(manifest_bytes)
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            record["problems"].append(f"{MANIFEST_FILENAME} is not valid JSON: {exc}")
            return record
        for wanted in ("repository_commit", "generated_at", "data_through_date", "snapshot_label"):
            record[wanted] = manifest.get(wanted)
        recorded_manifest_hash = integrity.get(f"artifacts/{MANIFEST_FILENAME}")
        record["manifest_matches_integrity_record"] = (
            recorded_manifest_hash == record["manifest_sha256"]
        )
        if recorded_manifest_hash and not record["manifest_matches_integrity_record"]:
            record["problems"].append("manifest.json does not match its own integrity record")

    missing = sorted(r for r in integrity if r not in relatives)
    if missing:
        record["problems"].append(f"{len(missing)} recorded file(s) absent: {missing[:5]}")
    # The integrity file cannot record its own hash, so it is never in its
    # own list and must never count as an orphan. Without this exclusion
    # EVERY genuine snapshot classifies AMBIGUOUS, and the audit reports a
    # healthy archive as partial -- which is the exact wrong direction for a
    # tool whose output decides whether a backfill happens.
    self_excluded = {f"artifacts/{INTEGRITY_HASHES_FILENAME}"}
    orphaned = sorted(r for r in relatives if r not in integrity and r not in self_excluded)
    record["orphaned_objects"] = orphaned
    if orphaned:
        record["problems"].append(f"{len(orphaned)} object(s) not named by the integrity file")

    if verify_hashes and not missing:
        mismatched = []
        verified_bytes = 0
        for relative, expected in sorted(integrity.items()):
            data = reader.get_bytes(_r2_key(prefix, season, name, relative))
            verified_bytes += len(data)
            if _sha256(data) != expected:
                mismatched.append(relative)
        record["all_integrity_hashes_validate"] = not mismatched
        record["bytes_verified"] = verified_bytes
        if mismatched:
            record["problems"].append(f"hash mismatch: {mismatched[:5]}")
    elif missing:
        record["all_integrity_hashes_validate"] = False
    else:
        record["all_integrity_hashes_validate"] = None

    record["complete"] = (
        not record["problems"]
        and record["all_integrity_hashes_validate"] is True
        and record["outputs_side_present"]
        and record["artifacts_side_present"]
    )
    return record


# ---------------------------------------------------------------------------
# GitHub Actions recoverability -- read-only REST, no writes of any kind
# ---------------------------------------------------------------------------


def _github_get(path: str, token: str) -> dict[str, Any]:
    request = Request(  # noqa: S310 -- fixed api.github.com host
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "contact-luck-snapshot-audit",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def audit_actions_recoverability(
    *, repository: str, token: str, dates: list[str], run_limit: int = 60
) -> dict[str, Any]:
    """Do any retained Actions artifacts still hold a scored snapshot?

    The publish workflow uploads artifacts only on non-deploying runs, and
    what it uploads is the snapshot MANIFEST plus integrity hashes, not the
    snapshot. That distinction decides whether a date is recoverable without
    re-scoring, so it is reported explicitly rather than inferred from a name.
    """
    runs = _github_get(f"/repos/{repository}/actions/runs?per_page={run_limit}", token).get(
        "workflow_runs", []
    )
    findings: list[dict[str, Any]] = []
    for run in runs:
        artifacts = _github_get(
            f"/repos/{repository}/actions/runs/{run['id']}/artifacts", token
        ).get("artifacts", [])
        matched = [
            a
            for a in artifacts
            if any(d in a["name"] for d in dates) and not a.get("expired", False)
        ]
        if not matched:
            continue
        findings.append(
            {
                "run_id": run["id"],
                "workflow_name": run.get("name"),
                "event": run.get("event"),
                "conclusion": run.get("conclusion"),
                "created_at": run.get("created_at"),
                "head_sha": run.get("head_sha"),
                "artifacts": [
                    {
                        "name": a["name"],
                        "bytes": a.get("size_in_bytes"),
                        "expired": a.get("expired"),
                        "expires_at": a.get("expires_at"),
                        "carries_scored_snapshot": _artifact_carries_snapshot(a["name"]),
                        "build_output_only": any(
                            h in a["name"] for h in _BUILD_ONLY_ARTIFACT_HINTS
                        ),
                    }
                    for a in matched
                ],
            }
        )
    per_date = {}
    for date in dates:
        carrying = [
            {"run_id": f["run_id"], "artifact": a["name"], "expires_at": a["expires_at"]}
            for f in findings
            for a in f["artifacts"]
            if date in a["name"] and a["carries_scored_snapshot"]
        ]
        per_date[date] = {
            "exact_snapshot_recoverable_without_rescoring": bool(carrying),
            "carrying_artifacts": carrying,
            "note": (
                "The publish workflow uploads manifest+integrity metadata, not the scored "
                "snapshot itself. Metadata alone proves what WAS scored; it does not let "
                "the snapshot be reconstructed without re-scoring."
            ),
        }
    return {
        "runs_inspected": len(runs),
        "runs_with_matching_artifacts": findings,
        "by_date": per_date,
    }


def _artifact_carries_snapshot(name: str) -> bool:
    """Conservative: a name must look like a snapshot artifact AND not look
    like build output. `snapshot-manifest-<date>` is metadata, and metadata is
    not the snapshot -- so it does NOT count as recoverable."""
    if any(h in name for h in _BUILD_ONLY_ARTIFACT_HINTS):
        return False
    if "snapshot-manifest-" in name:
        return False
    return any(h in name for h in _SNAPSHOT_ARTIFACT_HINTS)


def combine_status(r2_record: dict[str, Any], actions_record: dict[str, Any]) -> str:
    """One status per date, from both sources. R2 wins when it holds a
    complete snapshot; otherwise a retained scored artifact makes the date
    recoverable; otherwise partial beats absent, because a partial archive is
    a thing someone must look at."""
    if r2_record["status"] == STATUS_ALREADY_ARCHIVED:
        return STATUS_ALREADY_ARCHIVED
    if actions_record.get("exact_snapshot_recoverable_without_rescoring"):
        return STATUS_RECOVERABLE
    if r2_record["status"] == STATUS_AMBIGUOUS:
        return STATUS_AMBIGUOUS
    return STATUS_NONE


def build_report(
    *,
    dates: list[str],
    season: int,
    reader: ReadOnlyR2Reader | None,
    repository: str | None,
    github_token: str | None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if not dates:
        raise AuditError("at least one --date is required; this audit never guesses a date")

    r2: dict[str, Any] = {}
    if reader is not None:
        for date in dates:
            r2[date] = audit_r2_date(reader, date=date, season=season, verify_hashes=verify_hashes)
    actions: dict[str, Any] = {"by_date": {}}
    if repository and github_token:
        actions = audit_actions_recoverability(
            repository=repository, token=github_token, dates=dates
        )

    summary = {}
    for date in dates:
        r2_record = r2.get(date, {"status": STATUS_NONE, "detail": "R2 not inspected"})
        actions_record = actions["by_date"].get(date, {})
        summary[date] = combine_status(r2_record, actions_record)

    return {
        "audit_version": "r2_snapshot_history_audit_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "s3_operations_used": list(READ_ONLY_S3_OPERATIONS),
        "season": season,
        "dates_requested": dates,
        "status_by_date": summary,
        "r2": r2,
        "github_actions": actions,
        "contains_no_credentials": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="YYYY-MM-DD to inspect. Repeatable. REQUIRED -- never defaulted.",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", default="r2-snapshot-history-audit.json")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Inventory only. Reports all_integrity_hashes_validate as null rather than true.",
    )
    args = parser.parse_args(argv)

    if not args.date:
        print("error: at least one --date is required", file=sys.stderr)
        return 2

    try:
        reader = make_read_only_reader()
        report = build_report(
            dates=args.date,
            season=args.season,
            reader=reader,
            repository=args.repository,
            github_token=os.environ.get("GITHUB_TOKEN"),
            verify_hashes=not args.skip_hash_verification,
        )
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    for date, status in report["status_by_date"].items():
        print(f"[audit] {date}: {status}")
    print(f"[audit] wrote {args.output}")
    print("[audit] READ-ONLY: no object was created, modified or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
