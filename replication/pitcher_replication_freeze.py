"""Version 0.14: builds, writes and validates the write-once freeze artifact
for the one-time 2025 held-out Pitcher Contact Luck replication.

This module RECORDS state. It computes no pitcher value, opens no season,
reads no parquet, and makes no network call. Its only file writes go to the
isolated `artifacts/pitcher_replication/v0_14/` namespace.

## What the freeze artifact contains

1. The pre-registered specification (`pitcher_replication_spec.spec_payload`)
   verbatim, plus its own content hash.
2. The frozen numeric constants this specification depends on, RESOLVED from
   the real modules at freeze time rather than restated by hand -- pitcher
   qualification threshold sets, the bootstrap interval design, the negated
   column list, the role/board presentation constants.
3. A SHA-256 per frozen source file (`FROZEN_SOURCE_RELATIVE_PATHS`), so any
   later change to the scoring, aggregation, interval or pitcher-side code
   invalidates the freeze instead of silently changing what "the frozen
   procedure" means.
4. The repository commit and working-tree state at freeze time.
5. A pre-outcome attestation that no 2025 pitcher output existed when the
   freeze was written, evidenced by the replication namespace being empty.

## Write-once

`write_freeze` never overwrites a different payload. Re-running it with an
unchanged specification is idempotent; re-running it after ANY spec or
source change fails loudly and points at `amend_freeze`. `amend_freeze`
writes a new numbered revision beside the original and appends to an
amendment log -- it never mutates or deletes an existing revision, and it
refuses unless the caller attests the amendment is being recorded BEFORE any
2025 access.

## Authorization is recorded elsewhere, and binds by hash

This module does not itself authorize anything. 2025 is
`FINAL_TEST_SEASONS`; `RESEARCH_RULES.md` permits it exactly once, through
the sealed Version 1.0 entry point, which has been used, so a second sealed
evaluation needs the maintainer's explicit sign-off as a separate decision.

That sign-off lives in `pitcher_replication_authorization`, recorded AFTER
the freeze and deliberately outside `FROZEN_SOURCE_RELATIVE_PATHS` -- the
frozen spec's own `AUTHORIZATION_STATUS` still reads False because it
records the state at freeze time, which is the evidence that the questions
were fixed before the sign-off. `resolve_authorization` binds a recorded
authorization to a freeze by BOTH content hashes, so a sign-off can never
transfer to a different specification.

`assert_ready_for_2025` consults it and still enforces every other
precondition, so this namespace cannot become the second code path by
accident.

`evaluation/` is a plain script directory, not part of the installed
`mlb_luck_score` package, and `replication/` follows it: not covered by
`make check`'s default `ruff/mypy src tests`, so run `ruff format/check
replication` and `mypy replication` explicitly. Tests import it via
`pyproject.toml`'s pytest `pythonpath`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pitcher_replication_spec import (
    BOARD_DISPLAY_MINIMUM_BBE,
    REPLICATION_SEASON,
    ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE,
    ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE,
    SPEC_VERSION,
    spec_payload,
)

FREEZE_VERSION = "0.14.0"

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Isolated namespace, mirroring `artifacts/final_evaluation/v1/` and
#: `artifacts/prospective/v1_1/`. Gitignored, exactly like those, so a run's
#: own output can never dirty the working tree for a later run.
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "pitcher_replication" / "v0_14"
FREEZE_PATH = ARTIFACTS_DIR / "pitcher_replication_freeze.json"
AMENDMENT_LOG_PATH = ARTIFACTS_DIR / "amendments.jsonl"

#: Where a future authorized 2025 replication would write. Must be empty (or
#: absent) at freeze time -- that emptiness is the pre-outcome evidence.
REPLICATION_DATA_DIR = REPO_ROOT / "data" / "pitcher_replication" / "2025"
REPLICATION_OUTPUTS_DIR = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"

#: Every file whose CONTENT this freeze pins. A change to any of these
#: changes what "the frozen procedure" means, so the freeze must be rebuilt
#: (before 2025 is opened) or the replication abandoned.
#:
#: As of Version 0.14 this set contains RESEARCH code only. The question-E
#: estimators moved to `replication/pitcher_replication_estimators.py` and
#: question F is implemented in `replication/pitcher_split_half.py`, so
#: `demo/build_pitcher_prototype_fixture.py` -- a presentation-side fixture
#: generator -- is deliberately NOT pinned here. It is now a consumer of the
#: estimator module rather than its home, so a purely presentational edit can
#: no longer invalidate this freeze.
#:
#: `models/evaluate_aggregation_stability.py` IS pinned: question F imports
#: its split rules and inclusion threshold, which makes it a frozen input.
FROZEN_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "src/mlb_luck_score/config.py",
    "src/mlb_luck_score/eligibility.py",
    "src/mlb_luck_score/scoring/run_values.py",
    "src/mlb_luck_score/scoring/contact_luck.py",
    "src/mlb_luck_score/scoring/attribution_ledger.py",
    "src/mlb_luck_score/scoring/aggregate_attribution.py",
    "src/mlb_luck_score/scoring/aggregation_uncertainty.py",
    "src/mlb_luck_score/scoring/qualification.py",
    "src/mlb_luck_score/scoring/run_season_aggregation.py",
    "src/mlb_luck_score/scoring/pitching_contact_luck.py",
    "src/mlb_luck_score/scoring/run_pitching_contact_luck.py",
    "src/mlb_luck_score/models/evaluate_aggregation_stability.py",
    "replication/pitcher_replication_spec.py",
    "replication/pitcher_replication_estimators.py",
    "replication/pitcher_split_half.py",
)

#: The Version 0.13.1 freeze tied question E to a presentation-side fixture
#: generator. Version 0.14 resolved that; this records the resolution so the
#: history is legible from the artifact itself.
ESTIMATOR_IMPLEMENTATION_CAVEAT = (
    "RESOLVED in Version 0.14. Question E's estimators were extracted from "
    "demo/build_pitcher_prototype_fixture.py into "
    "replication/pitcher_replication_estimators.py, verified to reproduce every Version "
    "0.13.1 value exactly (k=70.588; 199/311/554/1246; 0.508/0.4486/0.0/0.5007; the >=1 "
    "BBE artifact pair 1.9386 -> 0.8766 over 73 zero-width rows). The generator now "
    "IMPORTS that module, so the committed fixture and the replication are computed by "
    "the same functions, and the freeze pins research code only. Question F is likewise "
    "implemented in replication/pitcher_split_half.py, a port whose batter-side control "
    "reproduces the committed Version 0.11 Phase 6 numbers to 0.0 absolute difference."
)


class FreezeError(RuntimeError):
    """Raised when a freeze cannot be built, written, or validated."""


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    """A stable SHA-256 over a JSON payload, canonicalized with sorted keys
    so re-serialization order can never change the hash.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_source_hashes(
    repo_root: Path = REPO_ROOT,
    *,
    relative_paths: tuple[str, ...] = FROZEN_SOURCE_RELATIVE_PATHS,
) -> dict[str, str]:
    """`{relative_path: sha256}`. A missing frozen source fails loudly -- it
    is never silently skipped, because a skipped file is an unpinned one.
    """
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for rel_path in relative_paths:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            missing.append(rel_path)
            continue
        hashes[rel_path] = compute_file_sha256(full_path)
    if missing:
        raise FreezeError(f"Frozen source file(s) missing, cannot build freeze: {missing}")
    return hashes


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def get_git_commit_hash(repo_root: Path = REPO_ROOT) -> str:
    commit = _git(repo_root, "rev-parse", "HEAD")
    if not commit:
        raise FreezeError("git rev-parse HEAD returned nothing")
    return commit


def working_tree_status(repo_root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    """`(is_clean, dirty_paths)` from `git status --porcelain`.

    Reported rather than enforced at build time, because the freeze is
    written in the same change that introduces the specification and the
    tree is necessarily dirty then. `assert_ready_for_2025` is where a clean
    tree becomes mandatory.
    """
    porcelain = _git(repo_root, "status", "--porcelain")
    dirty = [line for line in porcelain.splitlines() if line.strip()]
    return (not dirty, dirty)


# ---------------------------------------------------------------------------
# 2025 guards
# ---------------------------------------------------------------------------


def assert_2025_protection_intact() -> dict[str, Any]:
    """Verify, from `mlb_luck_score.config` itself, that 2025 is still
    sealed and unreachable by ordinary development paths.

    This reads CONSTANTS, never data. It is the guard that would catch a
    freeze being built in a repository where someone had already quietly
    opened 2025 up.

    Raises:
        FreezeError: if 2025 has been added to the development season lists,
            the development date ranges, or removed from the final-test set.
    """
    from mlb_luck_score.config import (
        DEVELOPMENT_SEASONS,
        FINAL_TEST_SEASONS,
        MLB_REGULAR_SEASON_DATE_RANGES,
        PROSPECTIVE_SEASONS,
        TRAIN_SEASONS,
        VALIDATION_SEASONS,
    )

    problems: list[str] = []
    if REPLICATION_SEASON not in FINAL_TEST_SEASONS:
        problems.append(f"{REPLICATION_SEASON} is no longer in FINAL_TEST_SEASONS")
    if REPLICATION_SEASON in DEVELOPMENT_SEASONS:
        problems.append(f"{REPLICATION_SEASON} has been added to DEVELOPMENT_SEASONS")
    if REPLICATION_SEASON in TRAIN_SEASONS or REPLICATION_SEASON in VALIDATION_SEASONS:
        problems.append(f"{REPLICATION_SEASON} has been added to a development season list")
    if REPLICATION_SEASON in MLB_REGULAR_SEASON_DATE_RANGES:
        problems.append(f"{REPLICATION_SEASON} has been added to MLB_REGULAR_SEASON_DATE_RANGES")
    if problems:
        raise FreezeError(
            "2025 protection is NOT intact -- refusing to build a pre-2025 freeze: "
            + "; ".join(problems)
        )

    return {
        "final_test_seasons": list(FINAL_TEST_SEASONS),
        "prospective_seasons": list(PROSPECTIVE_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "replication_season_in_development_lists": False,
        "replication_season_has_development_date_range": False,
        "enforcement_point": "mlb_luck_score.config.assert_seasons_allowed",
        "allow_final_evaluation_required_for_2025": True,
    }


def display_path(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise. Never raises: these
    directories are module-level constants that a test (or a future runner
    with a relocated namespace) may point outside the repository, and a
    reporting helper must not be the thing that fails.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _listing(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(display_path(p) for p in directory.rglob("*") if p.is_file())


def assert_no_replication_outputs_exist() -> dict[str, Any]:
    """Verify the 2025 replication namespace holds nothing yet.

    This is the pre-outcome attestation's evidence: a freeze claiming to
    predate the result is only meaningful if no result exists.

    Raises:
        FreezeError: if any file is present in either replication namespace.
    """
    found = _listing(REPLICATION_DATA_DIR) + _listing(REPLICATION_OUTPUTS_DIR)
    if found:
        raise FreezeError(
            "2025 replication namespace is NOT empty -- a freeze cannot claim to precede "
            f"an outcome that already exists: {found}"
        )
    return {
        "data_dir": display_path(REPLICATION_DATA_DIR),
        "outputs_dir": display_path(REPLICATION_OUTPUTS_DIR),
        "data_dir_exists": REPLICATION_DATA_DIR.exists(),
        "outputs_dir_exists": REPLICATION_OUTPUTS_DIR.exists(),
        "files_present": [],
    }


# ---------------------------------------------------------------------------
# Resolved frozen constants
# ---------------------------------------------------------------------------


def _resolved_pitcher_thresholds() -> dict[str, dict[str, Any]]:
    from mlb_luck_score.scoring.pitching_contact_luck import (
        PITCHER_QUALIFICATION_THRESHOLD_SETS,
    )

    return {
        name: asdict(thresholds)
        for name, thresholds in PITCHER_QUALIFICATION_THRESHOLD_SETS.items()
    }


def _resolved_batter_thresholds() -> dict[str, dict[str, Any]]:
    from mlb_luck_score.scoring.qualification import QUALIFICATION_THRESHOLD_SETS

    return {name: asdict(t) for name, t in QUALIFICATION_THRESHOLD_SETS.items()}


def _resolved_interval_design() -> dict[str, Any]:
    from mlb_luck_score.scoring.aggregation_uncertainty import BootstrapDesign

    design = BootstrapDesign()
    return {
        "method": design.method,
        "resampling_unit": design.resampling_unit,
        "n_reps": design.n_reps,
        "seed": design.seed,
        "alpha": design.alpha,
        "refits_models": design.refits_models,
    }


def _resolved_signed_value_columns() -> list[str]:
    from mlb_luck_score.scoring.pitching_contact_luck import SIGNED_VALUE_COLUMNS

    return list(SIGNED_VALUE_COLUMNS)


def _resolved_ambiguous_outcome_events() -> list[str]:
    """The events excluded from the resolved-eligible denominator, read from
    `eligibility.py` rather than restated, so the freeze records what the
    code actually does.
    """
    from mlb_luck_score import eligibility

    return sorted(eligibility._AMBIGUOUS_OUTCOME_EVENTS)


def resolved_frozen_constants() -> dict[str, Any]:
    """Every frozen numeric/structural constant the specification depends
    on, read from its canonical module at freeze time.
    """
    return {
        "pitcher_qualification_threshold_sets": _resolved_pitcher_thresholds(),
        "batter_qualification_threshold_sets": _resolved_batter_thresholds(),
        "interval_design": _resolved_interval_design(),
        "signed_value_columns": _resolved_signed_value_columns(),
        "ambiguous_outcome_events_excluded_from_denominator": (
            _resolved_ambiguous_outcome_events()
        ),
        "presentation_constants": {
            "starter_like_min_bbe_per_appearance": ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE,
            "reliever_like_max_bbe_per_appearance": ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE,
            "board_display_minimum_bbe": BOARD_DISPLAY_MINIMUM_BBE,
        },
        "deterministic_seeds": {"model_random_seed": 42, "bootstrap_seed": 42},
        "train_seasons": [2021, 2022, 2023],
    }


# ---------------------------------------------------------------------------
# The freeze artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PitcherReplicationFreeze:
    freeze_version: str
    spec_version: str
    revision: int
    specification: dict[str, Any]
    spec_content_hash: str
    resolved_frozen_constants: dict[str, Any]
    source_hashes: dict[str, str]
    repository_commit: str
    working_tree_clean: bool
    working_tree_dirty_paths: list[str]
    season_protection: dict[str, Any]
    pre_outcome_attestation: dict[str, Any]
    estimator_implementation_caveat: str
    frozen_at_utc: str
    amendment: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PitcherReplicationFreeze:
        return cls(**data)

    def freeze_content_hash(self) -> str:
        """Hash of everything except the volatile recording fields, so the
        same specification frozen at the same commit hashes identically
        across runs (and write-once comparison is meaningful).
        """
        payload = self.to_dict()
        for volatile in ("frozen_at_utc", "revision", "amendment", "notes"):
            payload.pop(volatile, None)
        return content_hash(payload)


def build_freeze(
    repo_root: Path = REPO_ROOT,
    *,
    revision: int = 1,
    amendment: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> PitcherReplicationFreeze:
    """Assemble the freeze. Runs both 2025 guards first, so a freeze can
    never be produced in a repository where 2025 protection has lapsed or a
    2025 result already exists.
    """
    season_protection = assert_2025_protection_intact()
    namespace_state = assert_no_replication_outputs_exist()

    spec = spec_payload()
    is_clean, dirty = working_tree_status(repo_root)

    return PitcherReplicationFreeze(
        freeze_version=FREEZE_VERSION,
        spec_version=SPEC_VERSION,
        revision=revision,
        specification=spec,
        spec_content_hash=content_hash(spec),
        resolved_frozen_constants=resolved_frozen_constants(),
        source_hashes=build_source_hashes(repo_root),
        repository_commit=get_git_commit_hash(repo_root),
        working_tree_clean=is_clean,
        working_tree_dirty_paths=dirty,
        season_protection=season_protection,
        pre_outcome_attestation={
            "replication_season": REPLICATION_SEASON,
            "any_2025_data_read_while_building_this_freeze": False,
            "any_2025_pitcher_output_computed": False,
            "any_2026_data_read_while_building_this_freeze": False,
            "freeze_precedes_evaluation": True,
            "evidence": namespace_state,
            "method": (
                "This module imports only constants and hashing utilities. It calls no "
                "data loader, no downloader and no scoring runner, so there is no code "
                "path from building this freeze to a season's rows."
            ),
        },
        estimator_implementation_caveat=ESTIMATOR_IMPLEMENTATION_CAVEAT,
        frozen_at_utc=datetime.now(UTC).isoformat(),
        amendment=amendment,
        notes=notes or [],
    )


def _serialize(freeze: PitcherReplicationFreeze) -> str:
    return json.dumps(freeze.to_dict(), indent=2, sort_keys=True, default=str) + "\n"


def read_freeze(path: Path = FREEZE_PATH) -> PitcherReplicationFreeze:
    if not path.is_file():
        raise FreezeError(f"No freeze artifact at {path}")
    return PitcherReplicationFreeze.from_dict(json.loads(path.read_text()))


def write_freeze(
    freeze: PitcherReplicationFreeze,
    *,
    path: Path = FREEZE_PATH,
    rebuild_provisional: bool = False,
) -> tuple[Path, str]:
    """Write the freeze WRITE-ONCE.

    Re-writing an identical freeze is idempotent and returns the existing
    path. Re-writing a DIFFERENT one raises: an established freeze is never
    overwritten, because a freeze that can be rewritten in place is not a
    freeze.

    `rebuild_provisional` is the one narrow escape hatch, for the window in
    which the freeze is still being CONSTRUCTED. A freeze built on a dirty
    tree is marked provisional (`working_tree_clean` False) and is not yet a
    real freeze -- it authorizes nothing, because `assert_ready_for_2025`
    rejects it outright. While the existing artifact is provisional AND no
    2025 result exists, it may be superseded: the outgoing copy is archived
    under `superseded/<content hash>.json`, never discarded. Once a freeze
    has been written from a clean tree this flag stops working and the only
    remaining route is `amend_freeze`.

    Returns:
        `(path, freeze_content_hash)`.

    Raises:
        FreezeError: if a different freeze already exists and this is not an
            eligible provisional rebuild.
    """
    new_hash = freeze.freeze_content_hash()
    if path.exists():
        existing = read_freeze(path)
        existing_hash = existing.freeze_content_hash()
        if existing_hash == new_hash:
            return (path, existing_hash)
        if not rebuild_provisional or existing.working_tree_clean:
            detail = (
                " rebuild_provisional does not apply: the existing freeze was built from "
                "a CLEAN tree and is established, not provisional."
                if rebuild_provisional
                else ""
            )
            raise FreezeError(
                f"A DIFFERENT freeze already exists at {path} (existing content hash "
                f"{existing_hash}, new {new_hash}). The freeze is write-once. If the "
                "specification genuinely must change AND no 2025 data has been opened, "
                "record a pre-outcome amendment with amend_freeze() -- which writes a "
                "new numbered revision beside this one and never mutates it." + detail
            )
        assert_no_replication_outputs_exist()
        superseded = path.parent / "superseded" / f"{existing_hash}.json"
        superseded.parent.mkdir(parents=True, exist_ok=True)
        if not superseded.exists():
            superseded.write_text(_serialize(existing))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(freeze))
    return (path, new_hash)


def amend_freeze(
    *,
    reason: str,
    recorded_before_any_2025_access: bool,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, str]:
    """Record a PRE-OUTCOME amendment as a new numbered revision.

    Never mutates or deletes an existing revision. Refuses outright unless
    the caller attests the amendment precedes any 2025 access, and re-runs
    `assert_no_replication_outputs_exist` to check that attestation against
    the filesystem rather than taking it on trust.

    Raises:
        FreezeError: if the attestation is false, the reason is empty, or a
            2025 result already exists.
    """
    if not recorded_before_any_2025_access:
        raise FreezeError(
            "Refusing to amend: an amendment recorded after 2025 access is not an "
            "amendment, it is a post-hoc revision of a pre-registration. Report the "
            "replication under the frozen specification instead."
        )
    if not reason.strip():
        raise FreezeError("Refusing to amend without a stated reason.")

    assert_no_replication_outputs_exist()

    base = read_freeze() if FREEZE_PATH.exists() else None
    next_revision = (base.revision + 1) if base else 1
    amendment = {
        "reason": reason.strip(),
        "recorded_before_any_2025_access": True,
        "amends_revision": base.revision if base else None,
        "amends_spec_content_hash": base.spec_content_hash if base else None,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    amended = build_freeze(repo_root, revision=next_revision, amendment=amendment)

    revision_path = ARTIFACTS_DIR / f"pitcher_replication_freeze.rev{next_revision}.json"
    if revision_path.exists():
        raise FreezeError(f"Revision file already exists, refusing to overwrite: {revision_path}")
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    revision_path.write_text(_serialize(amended))

    with AMENDMENT_LOG_PATH.open("a") as handle:
        handle.write(
            json.dumps(
                {**amendment, "revision": next_revision, "path": str(revision_path.name)},
                sort_keys=True,
            )
            + "\n"
        )
    return (revision_path, amended.freeze_content_hash())


def validate_freeze(
    freeze: PitcherReplicationFreeze, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Re-derive everything the freeze pinned and compare. Returns a report;
    raises on any mismatch.

    Raises:
        FreezeError: if the specification, any frozen source file, or the
            required source set has changed since the freeze was written.
    """
    problems: list[str] = []

    current_spec_hash = content_hash(spec_payload())
    if current_spec_hash != freeze.spec_content_hash:
        problems.append(
            f"specification changed (frozen {freeze.spec_content_hash}, now {current_spec_hash})"
        )

    frozen_paths = set(freeze.source_hashes)
    required_paths = set(FROZEN_SOURCE_RELATIVE_PATHS)
    added = sorted(required_paths - frozen_paths)
    removed = sorted(frozen_paths - required_paths)
    if added or removed:
        problems.append(
            f"the frozen source set changed -- newly required: {added or 'none'}; no "
            f"longer required: {removed or 'none'}"
        )

    current = build_source_hashes(repo_root, relative_paths=tuple(frozen_paths))
    changed = sorted(p for p, h in freeze.source_hashes.items() if current.get(p) != h)
    if changed:
        problems.append(f"frozen source file(s) changed: {changed}")

    if problems:
        raise FreezeError("Freeze validation FAILED: " + "; ".join(problems))

    return {
        "spec_content_hash": current_spec_hash,
        "freeze_content_hash": freeze.freeze_content_hash(),
        "source_files_verified": len(frozen_paths),
        "repository_commit_at_freeze": freeze.repository_commit,
        "valid": True,
    }


def resolve_authorization(freeze: PitcherReplicationFreeze) -> tuple[bool, list[str]]:
    """Is there a recorded maintainer sign-off that applies to THIS freeze?

    The authorization lives in `pitcher_replication_authorization`, recorded
    after (and deliberately outside) the freeze -- see that module's
    docstring for why editing the frozen spec to flip a flag would have
    destroyed the pre-registration evidence.

    Returns:
        `(authorized, reasons)`. `reasons` explains every failure to bind,
        so a refusal can say precisely why rather than just "no".
    """
    try:
        from pitcher_replication_authorization import authorization_binds_to
    except ImportError:  # pragma: no cover -- the module is committed alongside this one
        return (False, ["no authorization record exists in this repository"])
    return authorization_binds_to(freeze)


def assert_ready_for_2025(
    freeze: PitcherReplicationFreeze,
    *,
    maintainer_authorized_second_sealed_evaluation: bool | None = None,
    repo_root: Path = REPO_ROOT,
) -> None:
    """The gate a 2025 runner must pass.

    `RESEARCH_RULES.md` permits 2025 exactly once, through the sealed
    Version 1.0 entry point, which has been used. A pitcher replication is a
    SECOND sealed evaluation through a second code path, and needs its own
    explicit maintainer sign-off.

    Authorization resolution:
      - `None` (the default) consults the recorded authorization via
        `resolve_authorization`, which binds by freeze AND spec content hash
        so a sign-off can never transfer to a different specification.
      - `False` forces refusal regardless of what is recorded.
      - `True` asserts the sign-off directly, for a caller that has it out
        of band. It does not skip any other check below.

    Every other precondition is unchanged and still mandatory: a clean
    working tree, a non-provisional freeze, intact 2025 protection, an empty
    replication namespace (which is what makes the authorization one-time),
    and a fully validating freeze.

    Raises:
        FreezeError: if authorization is absent or does not bind, the tree
            is dirty, the freeze is provisional, 2025 protection has lapsed,
            a 2025 result already exists, or the freeze does not validate.
    """
    if maintainer_authorized_second_sealed_evaluation is None:
        authorized, reasons = resolve_authorization(freeze)
    else:
        authorized, reasons = (
            bool(maintainer_authorized_second_sealed_evaluation),
            ["authorization explicitly withheld by the caller"],
        )
    if not authorized:
        raise FreezeError(
            "A 2025 pitcher replication is a SECOND sealed evaluation through a SECOND "
            "code path. RESEARCH_RULES.md requires the maintainer's explicit sign-off "
            "for that as a separate decision, recorded against this exact freeze. It "
            "does not apply here, so this freeze authorizes no 2025 access: " + "; ".join(reasons)
        )
    is_clean, dirty = working_tree_status(repo_root)
    if not is_clean:
        raise FreezeError(f"Working tree is not clean; refusing to proceed toward 2025: {dirty}")
    if not freeze.working_tree_clean:
        raise FreezeError(
            "This freeze was built on a dirty tree and is PROVISIONAL. Rebuild it from a "
            "committed, clean tree before any 2025 access, so the recorded commit is "
            "authoritative."
        )
    assert_2025_protection_intact()
    assert_no_replication_outputs_exist()
    validate_freeze(freeze, repo_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build, write and validate the write-once 2025 pitcher-replication freeze. "
            "Reads no season's data."
        )
    )
    parser.add_argument(
        "--rebuild-provisional",
        action="store_true",
        help=(
            "Supersede an existing PROVISIONAL freeze (one built on a dirty tree, while "
            "the specification is still being constructed). Archives the outgoing copy "
            "under superseded/. Refused once a freeze has been written from a clean tree."
        ),
    )
    args = parser.parse_args(argv)

    freeze = build_freeze()
    path, digest = write_freeze(freeze, rebuild_provisional=args.rebuild_provisional)
    report = validate_freeze(freeze)
    print(f"=== Wrote {path} ===")
    print(f"freeze_content_hash = {digest}")
    print(f"spec_content_hash   = {freeze.spec_content_hash}")
    print(f"repository_commit   = {freeze.repository_commit}")
    print(f"working_tree_clean  = {freeze.working_tree_clean}")
    if not freeze.working_tree_clean:
        print(
            "NOTE: this freeze is PROVISIONAL (dirty tree). It authorizes nothing. "
            "Rebuild it from a committed, clean tree before any 2025 access."
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
