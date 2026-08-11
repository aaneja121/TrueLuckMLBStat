"""Contact Luck: portability fix -- obtain every gitignored frozen artifact
Version 1.1 prospective scoring (and its manifest generation) requires, on a
fresh runner that has no local `data/processed/`/`outputs/tables/` cache.

THE PROBLEM THIS SOLVES: Version 1.1's manifest freezes its inputs by
hashing every path in `evaluation.v1_final_evaluation_manifest.
FROZEN_ARTIFACT_RELATIVE_PATHS` (`prospective.prospective_manifest.
build_snapshot_manifest` calls `build_artifact_hashes(repo_root,
relative_paths=FROZEN_ARTIFACT_RELATIVE_PATHS)` directly -- the SAME list
the sealed Version 1.0 evaluation uses, never a second, prospective-specific
list). Of that list's 32 entries, 27 are `src/mlb_luck_score/*.py` source
files (already git-tracked -- present on any fresh checkout automatically,
nothing to do here) and exactly FOUR are gitignored, local-only data/detail
files a maintainer's own machine already has but a disposable CI runner
does not:

  - data/processed/cleaned_development_data_with_sprint_speed.parquet
  - outputs/tables/opportunity_model_comparison_detail.json
  - outputs/tables/infield_opportunity_detail.json
  - outputs/tables/advancement_detail.json

The first scheduled dry-run failed on the parquet alone; once that was
fixed (this module originally covered only the parquet), the SECOND
scheduled dry-run got past scoring and failed on the remaining three at
manifest-build time. `FROZEN_INPUT_BUNDLE` below is the complete audited
set -- confirmed exhaustive by cross-referencing every entry in
`FROZEN_ARTIFACT_RELATIVE_PATHS` against `git check-ignore` (see
`tests/test_ensure_frozen_inputs.py::TestBundleCoversEveryGitignoredFrozenArtifact`,
which fails loudly if a FUTURE frozen artifact is ever added to that list
without also being added to this bundle).

WHAT THIS MODULE DOES NOT DO: it does not regenerate, reclean, retrain, or
resynthesize ANY of these four files from raw data or from a fresh model
-selection run under any circumstance -- there is no code path here that
imports cleaning/feature-engineering/model-comparison/Statcast-download
code. The ONLY two ways any one of these files can end up on disk are (1)
it was already there (verified against its pinned hash, never blindly
trusted) or (2) it was downloaded byte-for-byte from a durable R2 object
and verified against the same pinned hash before being written. A local
file that exists but does NOT match its pinned hash is left untouched and
this module fails loudly -- it is never silently redownloaded over, and
never silently accepted as "close enough."

## Storage layout: one bundle, relative-path-preserving keys

Every object lives under the `frozen-inputs/v1/` prefix (still a SEPARATE
top-level prefix from `scripts.archive_snapshot`'s `prospective/<season>/`
snapshot archive, and still the SAME R2 bucket/credentials -- see that
module's docstring). The three newly-added detail JSON files use keys that
preserve their local relative path exactly, e.g.:

    frozen-inputs/v1/outputs/tables/opportunity_model_comparison_detail.json

The parquet keeps its ALREADY-LIVE key unchanged
(`frozen-inputs/v1/cleaned_development_data_with_sprint_speed.parquet`,
flat, no `data/processed/` prefix) rather than being migrated to match the
newer convention -- that object was already uploaded and independently
verified against real R2 before this bundle existed, and re-keying it here
would silently orphan a working, already-verified production object for no
functional benefit. `FROZEN_INPUT_BUNDLE`'s `r2_key` field is per-entry and
explicit specifically so this kind of historical exception is visible in
one place, not implied by a blanket path-transformation rule.

The `v1` segment versions the ARCHIVED INPUT BUNDLE itself (not the
Version 1.x product versioning) -- if any frozen artifact in the bundle
ever legitimately changes (a new Version 0.x data-cleaning or model
-selection run, never a routine event), that becomes a new
`frozen-inputs/v2/...` key with its own pinned hash, never an in-place
overwrite of `v1`.

## Identity: one pinned SHA256 per bundle entry

Every `FrozenInputSpec.sha256` was computed directly from the real,
currently-in-use local file and independently cross-checked against
`outputs/final_evaluation/v1/v1_final_report.json`'s own
`manifest.artifact_hashes` entry for that same path (recorded when the
real, sealed Version 1.0 evaluation actually ran against it) -- every
entry agrees exactly. If a pinned hash and a local file's actual hash ever
disagree, something is wrong (a stale file, a bad download, a bit-flip)
and this module fails loudly rather than guessing which one is right.

## The two entry points

`ensure_frozen_input_bundle()` (and the CLI's default mode) is the ONLY
thing `scripts/publish_snapshot.sh` ever calls, always BEFORE
`prospective/run_v1_1_2026_scoring.py`. For each bundle entry it reuses
the local file if present and hash-valid, otherwise fetches it from R2 and
verifies it -- built on top of the single-file `ensure_frozen_development_
input()` primitive (unchanged since this module covered only the parquet;
every existing single-file test still exercises it directly). Processing
is fail-fast, in bundle order: the first entry that cannot be satisfied
raises immediately, and no later entry is attempted. An R2 client is
constructed AT MOST ONCE per call, shared across every entry that actually
needs one, and never constructed at all if every local file is already
present and valid -- true on a maintainer's own machine, false on a fresh
CI runner (before the one-time upload) or on any runner missing part of
the bundle.

`upload_frozen_input_bundle_to_r2()` / `--upload` is the SEPARATE, one-time
seeding action that populates every R2 object in the bundle. It is NEVER
called by `ensure_frozen_input_bundle()` or by `scripts/publish_snapshot.
sh` -- it exists only to be run by hand, once per bundle entry that hasn't
been seeded yet, with explicit human intent. It is safe to re-run over an
already-seeded entry (write-once: refuses to overwrite a DIFFERENT object,
no-ops on an identical one -- see `upload_frozen_input_to_r2`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from archive_snapshot import ArchiveClient, make_r2_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Deliberately a plain string literal here, NOT an import of
#: `evaluation.run_v1_final_evaluation.DEVELOPMENT_INPUT_PATH` (as an
#: earlier, parquet-only version of this module did) -- `FROZEN_INPUT_
#: BUNDLE` below is cross-checked against the full `FROZEN_ARTIFACT_
#: RELATIVE_PATHS` list by a dedicated structural test (`tests/
#: test_ensure_frozen_inputs.py::
#: TestBundleCoversEveryGitignoredFrozenArtifact`), which catches drift
#: across ALL four bundle entries, not just this one -- a strictly
#: stronger guarantee than importing a single constant, without this
#: module needing evaluation/ on its import-time sys.path at all.

__all__ = [
    "FROZEN_INPUT_BUNDLE",
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
    "FrozenInputSpec",
    "ensure_frozen_development_input",
    "ensure_frozen_input_bundle",
    "upload_frozen_input_bundle_to_r2",
    "upload_frozen_input_to_r2",
]


@dataclass(frozen=True)
class FrozenInputSpec:
    """One gitignored frozen artifact the bundle knows how to obtain.

    `local_relative_path` is relative to the repository root and is
    EXACTLY the same string `evaluation.v1_final_evaluation_manifest.
    FROZEN_ARTIFACT_RELATIVE_PATHS` uses for this file -- see
    `tests/test_ensure_frozen_inputs.py::
    TestBundleCoversEveryGitignoredFrozenArtifact` for the structural
    guarantee that these two lists never silently diverge.
    """

    local_relative_path: str
    r2_key: str
    sha256: str
    size_bytes: int


#: The complete, audited set of gitignored frozen artifacts Version 1.1
#: prospective scoring's manifest generation requires -- see module
#: docstring for how this was derived and verified. Order matters only in
#: that `ensure_frozen_input_bundle`/`upload_frozen_input_bundle_to_r2`
#: process entries in this order and fail fast on the first problem.
FROZEN_INPUT_BUNDLE: tuple[FrozenInputSpec, ...] = (
    FrozenInputSpec(
        local_relative_path="data/processed/cleaned_development_data_with_sprint_speed.parquet",
        r2_key="frozen-inputs/v1/cleaned_development_data_with_sprint_speed.parquet",
        sha256="f791415d218334aa578fd8104e7c3aec5f52716931fc69287aa2ee05b12443f8",
        size_bytes=111_445_297,
    ),
    FrozenInputSpec(
        local_relative_path="outputs/tables/opportunity_model_comparison_detail.json",
        r2_key="frozen-inputs/v1/outputs/tables/opportunity_model_comparison_detail.json",
        sha256="a694f6f65f0f94b7ed30fd785144a363d5f075a4a465f6a5b9a0eda2237a1d38",
        size_bytes=49_619,
    ),
    FrozenInputSpec(
        local_relative_path="outputs/tables/infield_opportunity_detail.json",
        r2_key="frozen-inputs/v1/outputs/tables/infield_opportunity_detail.json",
        sha256="2b879a72af11618b8d9f8939d900120c325c20990698261d7f4dcbb95b8141f0",
        size_bytes=85_434,
    ),
    FrozenInputSpec(
        local_relative_path="outputs/tables/advancement_detail.json",
        r2_key="frozen-inputs/v1/outputs/tables/advancement_detail.json",
        sha256="8826bae37c6b51489098018e95154d47ec0e1f05cd7c52bd29a633b20d01e222",
        size_bytes=18_174,
    ),
)

#: Convenience aliases for the bundle's first (parquet) entry -- kept for
#: backward compatibility with code/tests written when this module covered
#: only the parquet. New code should prefer FROZEN_INPUT_BUNDLE directly.
FROZEN_INPUT_LOCAL_PATH: Path = PROJECT_ROOT / FROZEN_INPUT_BUNDLE[0].local_relative_path
FROZEN_INPUT_R2_KEY = FROZEN_INPUT_BUNDLE[0].r2_key
FROZEN_INPUT_SHA256 = FROZEN_INPUT_BUNDLE[0].sha256
FROZEN_INPUT_SIZE_BYTES = FROZEN_INPUT_BUNDLE[0].size_bytes


class FrozenInputError(Exception):
    """Base class for frozen-input-portability failures. Every one of
    these must prevent prospective scoring from ever running against
    missing, wrong, or unverified frozen data -- see
    scripts/publish_snapshot.sh's ordering (this check runs first).
    """


class FrozenInputLocalHashMismatchError(FrozenInputError):
    """A local file already exists at the expected path but its sha256
    does not match the pinned hash. Never silently redownloaded or
    overwritten -- this could be a stale file, a corrupted one, or
    something else entirely at that path; a human must investigate.
    """


class FrozenInputMissingRemoteObjectError(FrozenInputError):
    """The local file is missing AND the pinned R2 key does not exist
    either. Never a reason to fall back to regenerating/retraining -- see
    module docstring.
    """


class FrozenInputRemoteHashMismatchError(FrozenInputError):
    """A file WAS downloaded from R2, but its sha256 does not match the
    pinned hash. Never written to the local path -- the downloaded bytes
    are discarded, not kept as a "best effort" copy.
    """


class FrozenInputUploadConflictError(FrozenInputError):
    """upload_frozen_input_to_r2() found an existing R2 object at the
    target key whose content does NOT match the pinned hash. Never
    overwritten -- this key is meant to be immutable; a genuinely
    different frozen input needs a new versioned key
    (frozen-inputs/v2/...).
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
    local_path: Path,
    expected_sha256: str,
    r2_key: str,
    client_factory: Callable[[], ArchiveClient] | None = None,
) -> FrozenInputResult:
    """Reuse-if-valid, fetch-if-missing, verify-always, for ONE file. Never
    regenerates or substitutes it -- the only two possible outcomes are
    REUSED_LOCAL (already correct on disk) and DOWNLOADED_FROM_R2 (fetched
    and verified just now); every other case is a raised FrozenInputError.

    This is the single-file primitive `ensure_frozen_input_bundle()` below
    calls once per `FrozenInputSpec` -- kept as its own function (rather
    than inlined) because it is independently useful and independently
    tested (see `tests/test_ensure_frozen_inputs.py`'s single-file test
    classes, unchanged since this module covered only the parquet).

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
            f"input hash ({expected_sha256}). Refusing to silently overwrite or redownload "
            "it -- this file may be stale, corrupted, or unrelated entirely. Investigate by "
            "hand before retrying (move it aside if it should be replaced)."
        )

    client = client_factory() if client_factory is not None else _client_from_env()

    if not client.object_exists(r2_key):
        raise FrozenInputMissingRemoteObjectError(
            f"{local_path} is missing locally, and R2 object {r2_key!r} does not exist "
            "either -- cannot obtain this frozen input. This is never a reason to "
            "regenerate/retrain it automatically (see CLAUDE.md's frozen development-data "
            "rules); the one-time seeding upload "
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


def ensure_frozen_input_bundle(
    specs: Sequence[FrozenInputSpec] = FROZEN_INPUT_BUNDLE,
    *,
    project_root: Path = PROJECT_ROOT,
    client_factory: Callable[[], ArchiveClient] | None = None,
) -> tuple[FrozenInputResult, ...]:
    """Ensure EVERY entry in `specs` (default: the complete audited
    bundle) is present locally and hash-verified, in order, failing fast
    on the first entry that cannot be satisfied. This is the function
    `scripts/publish_snapshot.sh` (via this module's CLI) and Version
    1.1's manifest generation ultimately depend on being satisfied before
    scoring begins.

    An R2 client is built AT MOST ONCE across the whole call (shared by
    every spec that actually needs one) -- never built at all if every
    local file is already present and valid.
    """
    shared_client: list[ArchiveClient] = []

    def _shared_client_factory() -> ArchiveClient:
        if not shared_client:
            shared_client.append(
                client_factory() if client_factory is not None else _client_from_env()
            )
        return shared_client[0]

    results = []
    for spec in specs:
        result = ensure_frozen_development_input(
            local_path=project_root / spec.local_relative_path,
            expected_sha256=spec.sha256,
            r2_key=spec.r2_key,
            client_factory=_shared_client_factory,
        )
        results.append(result)
    return tuple(results)


def upload_frozen_input_to_r2(
    *,
    local_path: Path,
    expected_sha256: str,
    r2_key: str,
    client: ArchiveClient,
) -> FrozenInputUploadOutcome:
    """ONE-TIME seeding action for ONE file -- never called by
    `ensure_frozen_development_input()`/`ensure_frozen_input_bundle()` or
    `scripts/publish_snapshot.sh`. Run by hand, with explicit human
    intent, via `upload_frozen_input_bundle_to_r2()`/`--upload`.
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


def upload_frozen_input_bundle_to_r2(
    specs: Sequence[FrozenInputSpec] = FROZEN_INPUT_BUNDLE,
    *,
    project_root: Path = PROJECT_ROOT,
    client: ArchiveClient,
) -> tuple[tuple[str, FrozenInputUploadOutcome], ...]:
    """Upload every entry in `specs` that isn't already seeded, in order,
    failing fast on the first conflict/verification failure. Safe to
    re-run over an already-fully-seeded bundle (every entry no-ops).
    Returns `(r2_key, outcome)` per entry, in order.
    """
    results = []
    for spec in specs:
        outcome = upload_frozen_input_to_r2(
            local_path=project_root / spec.local_relative_path,
            expected_sha256=spec.sha256,
            r2_key=spec.r2_key,
            client=client,
        )
        results.append((spec.r2_key, outcome))
    return tuple(results)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "ONE-TIME seeding action: upload every bundle entry to R2 that isn't already "
            "seeded (only if the local file matches its pinned hash, and only if no "
            "conflicting object already exists at its key). Never invoked by "
            "scripts/publish_snapshot.sh."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.upload:
            client = _client_from_env()
            for r2_key, outcome in upload_frozen_input_bundle_to_r2(client=client):
                print(f"upload outcome={outcome.value} key={r2_key}")
            return 0

        for result in ensure_frozen_input_bundle():
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
