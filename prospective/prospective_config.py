"""Contact Luck v1.1: prospective 2026 scoring -- isolated namespaces and guards.

Versions 0.2-1.0 are FROZEN and this module changes none of them. See
CLAUDE.md "Version 1.1: prospective 2026 scoring" for the policy this package
implements: 2026 may be SCORED by the exact frozen Version 1.0 system, never
used to tune, retrain, select features, or pick thresholds.

## Why this lives outside `src/mlb_luck_score/`

Mirrors `evaluation/`'s own rationale exactly: `prospective/` is a plain
script directory, not part of the installed `mlb_luck_score` package, and not
covered by `make check`'s default `ruff format/check src tests` / `mypy src`.
Tests reach it via `tests/conftest.py`'s `pythonpath` pytest setting
(`pyproject.toml`), the same mechanism already used for `evaluation/`.

## Namespace isolation

Every 2026 raw/derived file lives under `PROSPECTIVE_RAW_DIR`/
`PROSPECTIVE_OUTPUTS_DIR`/`PROSPECTIVE_ARTIFACTS_DIR` -- never `data/raw`,
`data/processed`, `outputs/tables`, or `artifacts/`, so no development runner
can discover 2026 data by accident (mirrors the Version 1.0 2025 namespace
rule). Conversely, this package must never read the sealed Version 1.0
namespaces (`SEALED_V1_NAMESPACE_ROOTS`) as model inputs -- the only
permitted touch is an optional, read-only integrity check that the v1.0 seal
file still exists and is unchanged (`prospective.prospective_manifest.
verify_v1_seal_unchanged`), never a consumer of 2025 data itself.

## Cross-importing the frozen Version 1.0 modules

Several modules here (`prospective_manifest`, `prospective_ingestion`,
`prospective_scoring`, `run_v1_1_2026_scoring`) reuse pure, generic functions
directly from `evaluation/` (e.g. `v1_final_evaluation_manifest.
build_artifact_hashes`, `run_v1_final_evaluation.
assert_raw_statcast_schema_compatible`) rather than duplicating them. Under
pytest this works because `pyproject.toml` puts both `evaluation` and
`prospective` on `pythonpath`; a direct CLI invocation (`python prospective/
run_v1_1_2026_scoring.py ...`) only gets `prospective/` on `sys.path`
automatically (Python adds the running script's own directory), so this
module inserts the sibling `evaluation/` directory itself, at import time,
below -- mirroring notebook 14's manual `sys.path.insert(0, "../evaluation")`
pattern, just automatic instead of copy-pasted into every entry point.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_EVALUATION_DIR = str(PROJECT_ROOT / "evaluation")
if _EVALUATION_DIR not in sys.path:
    sys.path.insert(0, _EVALUATION_DIR)

#: The single prospective season this package is ever authorized to touch.
#: Matches `mlb_luck_score.config.PROSPECTIVE_SEASONS` (which already
#: excludes 2026 from `DEVELOPMENT_SEASONS`/`MLB_REGULAR_SEASON_DATE_RANGES`
#: -- that omission is a second layer of defense this package relies on but
#: does not itself implement).
PROSPECTIVE_SEASON = 2026

#: Isolated namespace for every 2026 file this package reads or writes.
PROSPECTIVE_RAW_DIR = PROJECT_ROOT / "data" / "prospective" / "2026"
PROSPECTIVE_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "prospective" / "v1_1"
PROSPECTIVE_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "prospective" / "v1_1"

#: The sealed Version 1.0 namespaces -- read-only-at-most (seal-integrity
#: check only), NEVER a source of model input for prospective scoring.
SEALED_V1_RAW_DIR = PROJECT_ROOT / "data" / "final_evaluation" / "2025"
SEALED_V1_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "final_evaluation" / "v1"
SEALED_V1_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "final_evaluation" / "v1"
SEALED_V1_NAMESPACE_ROOTS: tuple[Path, ...] = (
    SEALED_V1_RAW_DIR,
    SEALED_V1_OUTPUTS_DIR,
    SEALED_V1_ARTIFACTS_DIR,
)
SEALED_V1_SEAL_PATH = SEALED_V1_ARTIFACTS_DIR / "seal.json"

#: VERIFIED 2026 MLB championship-season opening date -- see CLAUDE.md
#: "Version 1.1: prospective 2026 scoring" for the policy this satisfies.
#: Source: MLB's official 2026 schedule release. The 2026 championship
#: season began Wednesday, March 25, 2026, with Opening Night: the New York
#: Yankees at the San Francisco Giants; the official schedule confirms that
#: game was played. This was confirmed via an explicit maintainer-provided
#: citation of the official MLB schedule -- Claude Code did not independently
#: fetch or cross-check a live schedule source for this date (this
#: repository's tooling never guesses or generates a schedule URL on its
#: own -- see CLAUDE.md's URL-generation rule). If this date is ever wrong or
#: needs revision for a future season, update the date, the source citation,
#: and the verification date together -- never change one without the others.
PROSPECTIVE_2026_SEASON_START_DATE = date(2026, 3, 25)
PROSPECTIVE_2026_SEASON_START_SOURCE = (
    "MLB official 2026 championship season schedule -- Opening Night: New York Yankees at "
    "San Francisco Giants, 2026-03-25 (maintainer-provided citation, not independently "
    "fetched by this repository's tooling)."
)
PROSPECTIVE_2026_SEASON_START_VERIFIED_AT = "2026-08-06"
PROSPECTIVE_2026_SEASON_START_VERIFIED = True


class NamespaceViolationError(ValueError):
    """Raised when a path would read/write outside an isolated namespace --
    catches both a typo'd path and a path-traversal attempt (`..` segments).
    """


class SealedNamespaceAccessError(ValueError):
    """Raised when prospective code would read a sealed Version 1.0 namespace
    as something other than the one permitted read-only integrity check.
    """


def _assert_within_namespace(path: Path, namespace_root: Path, *, label: str) -> None:
    resolved = path.resolve()
    root = namespace_root.resolve()
    if not resolved.is_relative_to(root):
        raise NamespaceViolationError(
            f"{label} path {resolved} is not inside the isolated Version 1.1 prospective "
            f"namespace {root} -- refusing to read or write outside data/prospective/2026, "
            "outputs/prospective/v1_1, or artifacts/prospective/v1_1. This guard exists so a "
            "typo'd or maliciously crafted path (including '..' traversal) can never make 2026 "
            "scoring touch a development cache or the sealed Version 1.0 namespace."
        )


def assert_not_sealed_v1_namespace(path: Path, *, label: str) -> None:
    """Refuse any path that resolves inside one of the three sealed Version
    1.0 namespaces -- the ONE exception (a read-only seal-integrity check of
    `SEALED_V1_SEAL_PATH` itself) is implemented separately in
    `prospective.prospective_manifest.verify_v1_seal_unchanged` and never
    routes through this guard, since that check intentionally reads inside
    `SEALED_V1_ARTIFACTS_DIR`.
    """
    resolved = path.resolve()
    for root in SEALED_V1_NAMESPACE_ROOTS:
        if resolved.is_relative_to(root.resolve()):
            raise SealedNamespaceAccessError(
                f"{label} path {resolved} resolves inside the sealed Version 1.0 namespace "
                f"{root} -- Version 1.1 must never read sealed 2025 final-evaluation data, "
                "outputs, or artifacts as a model input. The only permitted touch is the "
                "read-only seal-integrity check in prospective.prospective_manifest."
            )
