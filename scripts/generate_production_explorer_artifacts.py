"""Contact Luck v1.4.0 Phase 5: production Play Explorer browser-artifact
generation, wired to a REAL Version 1.1 prospective snapshot.

    outputs/prospective/v1_1/<snapshot>/play_ledger.parquet
    outputs/prospective/v1_1/<snapshot>/play_ledger_metadata.json
    outputs/prospective/v1_1/<snapshot>/public_score.json
            v
    THIS SCRIPT (integrity check, then projection)
            v
    <ephemeral output dir>/players.json + players/<id>.json +
        games/<game_pk>.json + explore-metadata.json

This is orchestration glue ONLY -- it does not score anything, does not
re-derive any Contact Luck value, and does not reimplement the projection
logic. Every actual row-projection/version-check rule lives in `demo/
build_play_explorer_fixture.generate_explorer_artifacts` (imported and
called directly, never duplicated -- same convention `demo/
build_play_explorer_dev_ledger.py` already established for local/dev use of
that same generator).

## Canonical source integrity: reused, not reinvented

Per CLAUDE.md's "any 'fit on train, apply to all' statistic"/"a second,
independent check" precedents, this script does NOT implement its own
sha256-vs-integrity_hashes.json comparison. It reuses `dashboard/
snapshot_data.py`'s `discover_snapshots()` -- the EXACT function
`dashboard/build.py` itself already relies on to decide whether a snapshot
is trustworthy (it hashes every file `integrity_hashes.json` lists,
including `outputs/play_ledger.parquet` and
`outputs/play_ledger_metadata.json`, against the actual bytes on disk) --
and refuses to proceed unless `snapshot.integrity.valid`. The `play_ledger_
version == "2.0"` requirement is enforced a SECOND time, independently, by
the generator's own `require_compatible_play_ledger_version` (see that
module's docstring) -- this script never bypasses it.

Importing `dashboard/snapshot_data.py` from here (a `scripts/` module, not
a `dashboard/` one) does not weaken `dashboard/`'s own read-only-boundary
guarantee (`tests/test_dashboard_isolation.py` only inspects each file
UNDER `dashboard/`'s own direct imports; `snapshot_data.py` itself imports
nothing scoring-related) -- see that test module's docstring.

## Where `batter_name` comes from

Same-snapshot `public_score.json` (already name-resolved by
`prospective.prospective_player_names.apply_player_name_overlay` at scoring
time -- see CLAUDE.md "Player name resolution is presentation-only") --
NEVER a fresh call to the MLB Stats API `/people` endpoint. This is the one
concrete difference from `demo/build_play_explorer_dev_ledger.py`, which
DOES call the live API (explicitly, loudly, for local/dev fixture
generation only) because a bounded dev ledger has no accompanying
`public_score.json` of its own to read names from.

## Output location: never the canonical snapshot

`output_dir` is caller-supplied and, by convention (see
`scripts/publish_snapshot.sh`), lives under `outputs/explorer_build/`, a
namespace entirely separate from `outputs/prospective/v1_1/`/`artifacts/
prospective/v1_1/`. This script never writes into the snapshot directories
it reads from -- `generate_explorer_artifacts()` itself only ever writes
under the `output_dir` it's given. `scripts/archive_snapshot.py` only ever
enumerates files under `outputs/prospective/v1_1/<snapshot>/`/`artifacts/
prospective/v1_1/<snapshot>/`, so browser artifacts written elsewhere are
structurally never archived into the canonical R2 snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "demo"))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import build_play_explorer_fixture as explorer_gen  # noqa: E402
import snapshot_data as sd  # noqa: E402

__all__ = [
    "ProductionExplorerGenerationError",
    "generate_production_explorer_artifacts",
]

DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "prospective" / "v1_1"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "prospective" / "v1_1"

_REQUIRED_INTEGRITY_KEYS: tuple[str, ...] = (
    "outputs/play_ledger.parquet",
    "outputs/play_ledger_metadata.json",
)


class ProductionExplorerGenerationError(Exception):
    """Raised when the requested snapshot cannot safely become production
    Play Explorer browser artifacts -- never a reason to fall back to a
    stale, unverifiable, or version-incompatible source.
    """


def build_snapshot_dir_name(data_through_date: str, snapshot_label: str | None) -> str:
    """Mirrors `scripts.archive_snapshot.build_snapshot_dir_name`/
    `prospective.run_v1_1_2026_scoring._snapshot_dir_name` exactly, without
    importing either (same duplicate-a-small-pure-function convention
    `dashboard/snapshot_data.py`'s own `parse_snapshot_directory_name`
    already established for this exact string).
    """
    return f"{data_through_date}__{snapshot_label}" if snapshot_label else data_through_date


def _find_snapshot(
    directory_name: str, *, outputs_root: Path, artifacts_root: Path
) -> sd.DiscoveredSnapshot:
    for snapshot in sd.discover_snapshots(outputs_root=outputs_root, artifacts_root=artifacts_root):
        if snapshot.directory_name == directory_name:
            return snapshot
    raise ProductionExplorerGenerationError(
        f"No local snapshot directory named {directory_name!r} found under {outputs_root} "
        f"/ {artifacts_root}."
    )


def _load_batter_names_from_public_score(public_score_path: Path) -> dict[int, str]:
    """The same-snapshot presentation-name overlay, built from THIS
    snapshot's own already-resolved `public_score.json` -- never a fresh
    network call. Mirrors `demo/build_play_explorer_fixture.
    load_batter_names`'s `{batter_id: name}` shape (`None`/unresolved names
    are simply omitted, never fabricated).
    """
    if not public_score_path.exists():
        raise ProductionExplorerGenerationError(
            f"public_score.json not found at {public_score_path} -- cannot build the "
            "same-snapshot batter-name overlay."
        )
    rows = json.loads(public_score_path.read_text())
    names: dict[int, str] = {}
    for row in rows:
        name = row.get("batter_name")
        if name is not None:
            names[int(row["batter_id"])] = str(name)
    return names


def generate_production_explorer_artifacts(
    *,
    snapshot_directory_name: str,
    output_dir: Path,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
) -> dict[str, Any]:
    """Verify `snapshot_directory_name`'s integrity, then project its
    canonical `play_ledger.parquet` into Play Explorer browser artifacts
    under `output_dir`.

    Raises:
        ProductionExplorerGenerationError: if the snapshot cannot be found,
            fails its own integrity check (`dashboard/snapshot_data.py`'s
            hash-verified discovery), is missing `play_ledger.parquet`/
            `play_ledger_metadata.json`/`public_score.json`, or
            `integrity_hashes.json` does not cover the ledger/metadata
            files.
        IncompatiblePlayLedgerVersionError: if `play_ledger_version` is not
            exactly `"2.0"` (raised by the generator's own fail-closed
            check -- see `demo/build_play_explorer_fixture.py`).
    """
    snapshot = _find_snapshot(
        snapshot_directory_name, outputs_root=outputs_root, artifacts_root=artifacts_root
    )
    if not snapshot.integrity.valid:
        raise ProductionExplorerGenerationError(
            f"Snapshot {snapshot_directory_name!r} failed its own integrity check -- refusing "
            f"to generate Play Explorer artifacts from it: {list(snapshot.integrity.errors)}"
        )

    play_ledger_path = snapshot.outputs_dir / "play_ledger.parquet"
    play_ledger_metadata_path = snapshot.outputs_dir / "play_ledger_metadata.json"
    public_score_path = snapshot.outputs_dir / "public_score.json"
    for required in (play_ledger_path, play_ledger_metadata_path, public_score_path):
        if not required.exists():
            raise ProductionExplorerGenerationError(
                f"required snapshot file missing for {snapshot_directory_name!r}: {required}"
            )

    integrity_hashes = json.loads((snapshot.artifacts_dir / "integrity_hashes.json").read_text())
    missing_keys = [key for key in _REQUIRED_INTEGRITY_KEYS if key not in integrity_hashes]
    if missing_keys:
        raise ProductionExplorerGenerationError(
            f"snapshot {snapshot_directory_name!r}'s integrity_hashes.json does not cover "
            f"{missing_keys} -- refusing to generate Play Explorer artifacts from an "
            "unverifiable canonical source. (Every hash in integrity_hashes.json was already "
            "independently re-verified against the actual bytes on disk by "
            "dashboard/snapshot_data.py's discover_snapshots() above; this only confirms the "
            "ledger/metadata files were among the files it checked.)"
        )

    names = _load_batter_names_from_public_score(public_score_path)
    with tempfile.TemporaryDirectory() as tmp:
        names_path = Path(tmp) / "names.json"
        names_path.write_text(json.dumps({str(k): v for k, v in names.items()}))
        result = explorer_gen.generate_explorer_artifacts(
            play_ledger_path=play_ledger_path,
            play_ledger_metadata_path=play_ledger_metadata_path,
            output_dir=output_dir,
            names_path=names_path,
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-through", required=True, help="YYYY-MM-DD, matching the target snapshot."
    )
    parser.add_argument("--snapshot-label", default=None)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Ephemeral directory to write the browser artifacts into -- never a canonical "
        "snapshot directory.",
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    snapshot_directory_name = build_snapshot_dir_name(args.data_through, args.snapshot_label)
    result = generate_production_explorer_artifacts(
        snapshot_directory_name=snapshot_directory_name,
        output_dir=args.output_dir,
        outputs_root=args.outputs_root,
        artifacts_root=args.artifacts_root,
    )
    print(
        f"[generate_production_explorer_artifacts] play_ledger_version={result['play_ledger_version']}"
    )
    print(
        f"[generate_production_explorer_artifacts] rows={result['row_count']} "
        f"players={result['player_count']} games={result['game_count']}"
    )
    print(
        "[generate_production_explorer_artifacts] source_play_ledger_sha256="
        f"{result['explore_metadata']['source_play_ledger_sha256']}"
    )
    print(f"[generate_production_explorer_artifacts] wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
