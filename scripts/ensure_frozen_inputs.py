"""Contact Luck: portability fix -- obtain the frozen 2021-2024 development
input on a fresh runner that has no local `data/processed/` cache.

THE PROBLEM THIS SOLVES: `data/processed/cleaned_development_data_with_
sprint_speed.parquet` is the frozen 2021-2024 development input Version
0.11/0.12 (and, by extension, the sealed Version 1.0 final evaluation and
every Version 1.1 prospective snapshot) train against -- see
`evaluation.run_v1_final_evaluation.DEVELOPMENT_INPUT_PATH`, imported here
unchanged rather than re-declared, so this module can never point at a
different path than the real pipeline reads from. Per CLAUDE.md's "Never
commit datasets" rule, it is gitignored and has always been produced/kept
locally by whoever ran the Version 0.x data-cleaning pipeline -- a
maintainer's own long-lived machine already has it, but a disposable GitHub
Actions runner starts from a clean checkout and does not. The first
scheduled prospective dry-run failed exactly here: it downloaded and scored
2026 data successfully, then had nothing to train the frozen component
models against.

WHAT THIS MODULE DOES NOT DO: it does not regenerate, reclean, or
resynthesize the development dataset from raw Statcast data under any
circumstance -- there is no code path here that reads `data/raw/` or calls
any cleaning/feature-engineering function. The ONLY two ways this file can
end up on disk are (1) it was already there (verified against the pinned
hash, never blindly trusted) or (2) it was downloaded byte-for-byte from a
durable R2 object and verified against the same pinned hash before being
written. A local file that exists but does NOT match the pinned hash is
left untouched and this module fails loudly -- it is never silently
redownloaded over, and never silently accepted as "close enough."

## Storage layout

Deliberately a SEPARATE top-level prefix from `scripts.archive_snapshot`'s
`prospective/<season>/...` snapshot archive, even though both live in the
same R2 bucket and reuse the same `R2_ACCOUNT_ID`/`R2_ACCESS_KEY_ID`/
`R2_SECRET_ACCESS_KEY`/`R2_BUCKET_NAME` credentials -- this is a
conceptually different kind of object (one immutable, versioned INPUT
artifact, never a per-date snapshot) and must never be discoverable by, or
confused with, `list_archived_snapshots()`/`--sync-history`'s
`prospective/<season>/` enumeration:

    frozen-inputs/v1/cleaned_development_data_with_sprint_speed.parquet

The `v1` segment is a version marker for this ARCHIVED INPUT ARTIFACT
itself (not the Version 1.x product versioning) -- if the frozen
development input ever legitimately changes (a new Version 0.x
data-cleaning run, never a routine event), that becomes a new
`frozen-inputs/v2/...` key with its own pinned hash below, never an
in-place overwrite of `v1`.

## Identity: a single pinned SHA256, not a manifest

Unlike the snapshot archive (which has a whole `integrity_hashes.json` per
snapshot), there is exactly ONE file here, so its identity is a single
pinned constant, `FROZEN_INPUT_SHA256` below. That hash was computed
directly from the real, currently-in-use local file and independently
cross-checked against `outputs/final_evaluation/v1/v1_final_report.json`'s
own `manifest.artifact_hashes` entry for this same path (recorded when the
real, sealed Version 1.0 evaluation actually ran against it) -- both agree
exactly. If the pinned hash and the local file's actual hash ever disagree,
something is wrong (a stale file, a bad download, a bit-flip) and this
module fails loudly rather than guessing which one is right.

## The two entry points

`ensure_frozen_development_input()` (and the CLI's default mode) is the
ONLY thing `scripts/publish_snapshot.sh` ever calls, always BEFORE
`prospective/run_v1_1_2026_scoring.py`. It is READ-ONLY from R2's
perspective and, in the common case (the file is already present and
correct -- true on a maintainer's own machine and on any CI runner after
the one-time upload has happened), touches R2 not at all -- no client is
even constructed unless the local file is actually missing, so this
introduces no new network dependency for local development.

`upload_frozen_input_to_r2()` / `--upload` is the SEPARATE, one-time
seeding action that populates the R2 object in the first place. It is
NEVER called by `ensure_frozen_development_input()` or by
`scripts/publish_snapshot.sh` -- it exists only to be run by hand, once,
with explicit human intent, exactly like the analogous one-time snapshot
archival bootstrap in `scripts/archive_snapshot.py`. It refuses to upload
a local file that doesn't match the pinned hash, and refuses to overwrite
an existing R2 object with different content (write-once, mirroring
`scripts.archive_snapshot`'s own `ArchiveConflictError` philosophy).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from archive_snapshot import ArchiveClient, make_r2_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mirrors prospective/prospective_config.py's own sys.path insertion exactly
# (same reasoning documented there): a direct CLI invocation only gets this
# script's own directory (scripts/) on sys.path automatically, so the
# sibling evaluation/ directory is added explicitly, at import time, to
# reach the single source of truth for the frozen input's path.
_EVALUATION_DIR = str(PROJECT_ROOT / "evaluation")
if _EVALUATION_DIR not in sys.path:
    sys.path.insert(0, _EVALUATION_DIR)

from run_v1_final_evaluation import DEVELOPMENT_INPUT_PATH  # noqa: E402

__all__ = [
    "FROZEN_INPUT_LOCAL_PATH",
    "FROZEN_INPUT_R2_KEY",
    "FROZEN_INPUT_SHA256",
    "FROZEN_INPUT_SIZE_BYTES",
    "FrozenInputError",
    "FrozenInputLocalHashMismatchError",
    "FrozenInputMissingRemoteObjectError",
    "FrozenInputOutcome",
    "FrozenInputRemoteHashMismatchError",
    "FrozenInputResult",
    "ensure_frozen_development_input",
    "upload_frozen_input_to_r2",
]

#: The exact path the real pipeline reads from -- imported, never
#: re-declared, so this module can never silently drift from it.
FROZEN_INPUT_LOCAL_PATH: Path = DEVELOPMENT_INPUT_PATH

#: Deliberately separate from scripts.archive_snapshot.DEFAULT_ARCHIVE_PREFIX
#: ("prospective") -- see module docstring "Storage layout".
FROZEN_INPUT_R2_KEY = "frozen-inputs/v1/cleaned_development_data_with_sprint_speed.parquet"

#: Computed directly from the real local file (`shasum -a 256`) and
#: cross-checked against outputs/final_evaluation/v1/v1_final_report.json's
#: own manifest.artifact_hashes entry for this same path, recorded when the
#: real sealed Version 1.0 evaluation ran -- both agree exactly. Update this
#: constant ONLY alongside a deliberate, human-reviewed change to the frozen
#: development input itself (a new Version 0.x data-cleaning run), never as
#: a way to make a failing check pass.
FROZEN_INPUT_SHA256 = "f791415d218334aa578fd8104e7c3aec5f52716931fc69287aa2ee05b12443f8"

#: Informational only (logged, never used as a substitute for the hash
#: check) -- the real file's size in bytes at the time FROZEN_INPUT_SHA256
#: was computed.
FROZEN_INPUT_SIZE_BYTES = 111_445_297


class FrozenInputError(Exception):
    """Base class for frozen-input-portability failures. Every one of
    these must prevent prospective scoring from ever running against
    missing, wrong, or unverified development data -- see
    scripts/publish_snapshot.sh's ordering (this check runs first).
    """


class FrozenInputLocalHashMismatchError(FrozenInputError):
    """A local file already exists at FROZEN_INPUT_LOCAL_PATH but its
    sha256 does not match FROZEN_INPUT_SHA256. Never silently redownloaded
    or overwritten -- this could be a stale file, a corrupted one, or
    something else entirely at that path; a human must investigate.
    """


class FrozenInputMissingRemoteObjectError(FrozenInputError):
    """The local file is missing AND the pinned R2 key does not exist
    either. Never a reason to fall back to regenerating the dataset from
    raw data -- see module docstring.
    """


class FrozenInputRemoteHashMismatchError(FrozenInputError):
    """A file WAS downloaded from R2, but its sha256 does not match
    FROZEN_INPUT_SHA256. Never written to FROZEN_INPUT_LOCAL_PATH -- the
    downloaded bytes are discarded, not kept as a "best effort" copy.
    """


class FrozenInputUploadConflictError(FrozenInputError):
    """upload_frozen_input_to_r2() found an existing R2 object at
    FROZEN_INPUT_R2_KEY whose content does NOT match FROZEN_INPUT_SHA256.
    Never overwritten -- this key is meant to be immutable; a genuinely
    different frozen input needs a new versioned key (frozen-inputs/v2/...).
    """


class FrozenInputUploadVerificationError(FrozenInputError):
    """upload_frozen_input_to_r2() uploaded the file, but re-fetching it
    from R2 did not match what was sent -- treated as a failed upload, not
    a partial success.
    """


class FrozenInputOutcome(StrEnum):
    REUSED_LOCAL = "reused_local"
    DOWNLOADED_FROM_R2 = "downloaded_from_r2"


class FrozenInputUploadOutcome(StrEnum):
    UPLOADED = "uploaded"
    NO_OP_IDENTICAL = "no_op_identical"


@dataclass(frozen=True)
class FrozenInputResult:
    outcome: FrozenInputOutcome
    path: Path
    sha256: str
    size_bytes: int


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Writes `data` to `path` via a temp-file-then-rename so a failure or
    interruption mid-write can never leave a partial/corrupt file sitting
    at `path` looking like a legitimate, hash-verified artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _client_from_env() -> ArchiveClient:
    """Deliberately duplicated (not imported) from
    scripts.archive_snapshot's identically-named private helper -- both are
    ~10 lines reading the same 4 well-established env var names, and
    keeping this module able to construct its own client independently
    avoids reaching into another module's private (`_`-prefixed) API. See
    that module's docstring for what these four variables need to be; both
    modules read the exact same env var names on purpose so CI needs no new
    secrets for this fix (see module docstring).
    """
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
        raise FrozenInputError(
            f"Missing required environment variable(s) for R2 access: {', '.join(missing)}"
        )
    assert bucket and account_id and access_key_id and secret_access_key  # narrows for mypy
    return make_r2_client(
        bucket=bucket,
        account_id=account_id,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )


def ensure_frozen_development_input(
    *,
    local_path: Path = FROZEN_INPUT_LOCAL_PATH,
    expected_sha256: str = FROZEN_INPUT_SHA256,
    r2_key: str = FROZEN_INPUT_R2_KEY,
    client_factory: Callable[[], ArchiveClient] | None = None,
) -> FrozenInputResult:
    """Reuse-if-valid, fetch-if-missing, verify-always. Never regenerates
    or substitutes the development dataset -- the only two possible
    outcomes are REUSED_LOCAL (already correct on disk) and
    DOWNLOADED_FROM_R2 (fetched and verified just now); every other case
    is a raised FrozenInputError.

    Args:
        client_factory: Builds an ArchiveClient ONLY if actually needed
            (the local file is missing) -- defaults to `_client_from_env`,
            the real R2-backed client. Tests inject a factory returning an
            `InMemoryArchiveClient` instead, so this function never
            contacts real R2 or requires real credentials under test.
    """
    if local_path.exists():
        actual = _sha256_of_file(local_path)
        if actual == expected_sha256:
            return FrozenInputResult(
                outcome=FrozenInputOutcome.REUSED_LOCAL,
                path=local_path,
                sha256=actual,
                size_bytes=local_path.stat().st_size,
            )
        raise FrozenInputLocalHashMismatchError(
            f"{local_path} exists but its sha256 ({actual}) does not match the pinned frozen "
            f"development input hash ({expected_sha256}). Refusing to silently overwrite or "
            "redownload it -- this file may be stale, corrupted, or unrelated entirely. "
            "Investigate by hand before retrying (move it aside if it should be replaced)."
        )

    client = client_factory() if client_factory is not None else _client_from_env()

    if not client.object_exists(r2_key):
        raise FrozenInputMissingRemoteObjectError(
            f"{local_path} is missing locally, and R2 object {r2_key!r} does not exist "
            "either -- cannot obtain the frozen development input. This is never a reason "
            "to regenerate it from raw data automatically (see CLAUDE.md's frozen "
            "development-data rules); the one-time seeding upload "
            "(scripts/ensure_frozen_inputs.py --upload) may not have been run yet."
        )

    data = client.get_object_bytes(r2_key)
    actual = _sha256_of_bytes(data)
    if actual != expected_sha256:
        raise FrozenInputRemoteHashMismatchError(
            f"Downloaded {r2_key!r} from R2 but its sha256 ({actual}) does not match the "
            f"pinned hash ({expected_sha256}). NOT writing it to {local_path} -- this could "
            "mean a corrupted upload, a bit-flip in transit, or the wrong object at this key. "
            "Investigate the R2 object directly before retrying."
        )

    _atomic_write(local_path, data)
    return FrozenInputResult(
        outcome=FrozenInputOutcome.DOWNLOADED_FROM_R2,
        path=local_path,
        sha256=actual,
        size_bytes=len(data),
    )


def upload_frozen_input_to_r2(
    *,
    local_path: Path = FROZEN_INPUT_LOCAL_PATH,
    expected_sha256: str = FROZEN_INPUT_SHA256,
    r2_key: str = FROZEN_INPUT_R2_KEY,
    client: ArchiveClient,
) -> FrozenInputUploadOutcome:
    """ONE-TIME seeding action -- never called by
    `ensure_frozen_development_input()` or `scripts/publish_snapshot.sh`.
    Run by hand, once, with explicit human intent (`--upload`).
    """
    if not local_path.exists():
        raise FrozenInputError(f"{local_path} does not exist -- nothing to upload.")

    actual = _sha256_of_file(local_path)
    if actual != expected_sha256:
        raise FrozenInputLocalHashMismatchError(
            f"Refusing to upload {local_path}: its sha256 ({actual}) does not match the "
            f"pinned hash ({expected_sha256}). Update the pinned hash only after deliberately "
            "confirming this is the intended frozen input, never as a way to make an upload "
            "succeed."
        )

    if client.object_exists(r2_key):
        existing_hash = _sha256_of_bytes(client.get_object_bytes(r2_key))
        if existing_hash == expected_sha256:
            return FrozenInputUploadOutcome.NO_OP_IDENTICAL
        raise FrozenInputUploadConflictError(
            f"R2 object {r2_key!r} already exists with a DIFFERENT sha256 ({existing_hash}) "
            f"than the pinned hash ({expected_sha256}). Refusing to overwrite -- this key is "
            "meant to be immutable. Use a new versioned key (e.g. frozen-inputs/v2/...) for a "
            "genuinely different frozen input."
        )

    client.put_object_bytes(r2_key, local_path.read_bytes())

    reread_hash = _sha256_of_bytes(client.get_object_bytes(r2_key))
    if reread_hash != expected_sha256:
        raise FrozenInputUploadVerificationError(
            f"Uploaded {r2_key!r} but re-fetching it gave sha256 {reread_hash}, not the "
            f"expected {expected_sha256} -- treat this as a failed upload, not a partial "
            "success. Investigate before retrying."
        )
    return FrozenInputUploadOutcome.UPLOADED


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "ONE-TIME seeding action: upload the local frozen input to R2 (only if it "
            "matches the pinned hash, and only if no conflicting object already exists "
            "there). Never invoked by scripts/publish_snapshot.sh."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.upload:
            client = _client_from_env()
            outcome = upload_frozen_input_to_r2(client=client)
            print(f"upload outcome={outcome.value} key={FROZEN_INPUT_R2_KEY}")
            return 0

        result = ensure_frozen_development_input()
        print(
            f"outcome={result.outcome.value} path={result.path} "
            f"sha256={result.sha256} size_bytes={result.size_bytes}"
        )
        return 0
    except FrozenInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
