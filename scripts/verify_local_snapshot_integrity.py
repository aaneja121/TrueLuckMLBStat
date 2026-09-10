"""Verify that a LOCAL prospective snapshot is byte-identical to what its
own `integrity_hashes.json` says it should be.

Written for the feature-only build path
(`scripts/publish_snapshot.sh --build-from-existing-snapshot`), which does
not score anything: it restores an already-archived snapshot from R2 and
builds the site from that. In that path this script is the ONLY thing
standing between "a directory with the right name exists" and "the exact
archived production snapshot is on disk", so it is deliberately
independent of the restore itself.

`scripts/archive_snapshot.py` already verifies each file as it downloads.
This re-checks the result afterwards, from the snapshot's own manifest,
as a post-condition. That redundancy is the point: a restore that
half-succeeded, a truncated write, a stale local directory that was never
replaced, or a hand-edited file between stages all produce a snapshot that
LOOKS present and is not the archived one -- and the feature-only path
would otherwise publish it.

Reads only. Writes nothing, contacts no network, and never repairs: a
mismatch is reported and the process exits non-zero, because "repair" here
would mean deciding which of two disagreeing copies is production, and
that is not a decision a build step gets to make.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "prospective" / "v1_1"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "prospective" / "v1_1"

INTEGRITY_HASHES_FILENAME = "integrity_hashes.json"
MANIFEST_FILENAME = "manifest.json"

#: `integrity_hashes.json` keys are "<half>/<tail>", where <half> is the
#: snapshot half the file lives in. A snapshot spans two trees, so the key
#: prefix is what says which one -- see the module docstring of
#: `scripts/archive_snapshot.py`.
_HALVES = ("outputs", "artifacts")


class SnapshotIntegrityError(Exception):
    """The local snapshot is absent, incomplete, or does not match its own
    recorded hashes. Never raised for a difference this script could
    plausibly fix -- it fixes nothing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(
    relative: str, directory_name: str, *, outputs_root: Path, artifacts_root: Path
) -> Path:
    half, _, tail = relative.partition("/")
    if half not in _HALVES or not tail:
        raise SnapshotIntegrityError(
            f"integrity_hashes.json key {relative!r} is not '<outputs|artifacts>/<path>'"
        )
    root = outputs_root if half == "outputs" else artifacts_root
    return root / directory_name / tail


def verify_snapshot(
    directory_name: str,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> dict[str, object]:
    """Raise `SnapshotIntegrityError` unless every file the snapshot's own
    `integrity_hashes.json` names is present and hashes to the recorded
    value. Returns a small report on success."""
    artifacts_dir = artifacts_root / directory_name
    integrity_path = artifacts_dir / INTEGRITY_HASHES_FILENAME
    manifest_path = artifacts_dir / MANIFEST_FILENAME

    if not artifacts_dir.is_dir():
        raise SnapshotIntegrityError(
            f"no local snapshot directory {directory_name!r} under {artifacts_root}. "
            "The feature-only path builds from an ALREADY-ARCHIVED snapshot; it does not "
            "score one. If the archive has no snapshot for this date, that is the answer, "
            "not something to work around."
        )
    for required in (integrity_path, manifest_path):
        if not required.is_file():
            raise SnapshotIntegrityError(
                f"{required} is missing -- the snapshot is partial, so it cannot be "
                "verified and must not be published."
            )

    try:
        recorded: dict[str, str] = json.loads(integrity_path.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotIntegrityError(f"{integrity_path} is not valid JSON: {exc}") from exc
    if not recorded:
        raise SnapshotIntegrityError(f"{integrity_path} records no files at all")

    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted(recorded.items()):
        path = resolve_relative(
            relative, directory_name, outputs_root=outputs_root, artifacts_root=artifacts_root
        )
        if not path.is_file():
            missing.append(relative)
            continue
        if _sha256(path) != expected:
            mismatched.append(relative)

    if missing or mismatched:
        lines = [f"local snapshot {directory_name!r} does not match its own integrity_hashes.json."]
        if missing:
            lines.append(f"  missing ({len(missing)}): {', '.join(missing)}")
        if mismatched:
            lines.append(f"  hash mismatch ({len(mismatched)}): {', '.join(mismatched)}")
        lines.append(
            "  Refusing to continue. This snapshot is not the archived production snapshot, "
            "and a feature-only build must publish that one or nothing."
        )
        raise SnapshotIntegrityError("\n".join(lines))

    return {
        "directory_name": directory_name,
        "files_verified": len(recorded),
        "manifest_sha256": recorded.get(f"artifacts/{MANIFEST_FILENAME}", ""),
        "data_through_date": json.loads(manifest_path.read_text()).get("data_through_date"),
        "repository_commit": json.loads(manifest_path.read_text()).get("repository_commit"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-through", required=True, help="YYYY-MM-DD of the snapshot.")
    parser.add_argument("--snapshot-label", default=None)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    args = parser.parse_args(argv)

    directory_name = args.data_through
    if args.snapshot_label:
        directory_name = f"{args.data_through}__{args.snapshot_label}"

    try:
        report = verify_snapshot(
            directory_name, outputs_root=args.outputs_root, artifacts_root=args.artifacts_root
        )
    except SnapshotIntegrityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"[verify_local_snapshot_integrity] snapshot={report['directory_name']}")
    print(f"[verify_local_snapshot_integrity] files_verified={report['files_verified']}")
    print(f"[verify_local_snapshot_integrity] data_through={report['data_through_date']}")
    print(
        f"[verify_local_snapshot_integrity] snapshot_repository_commit={report['repository_commit']}"
    )
    print(f"[verify_local_snapshot_integrity] manifest_sha256={report['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
