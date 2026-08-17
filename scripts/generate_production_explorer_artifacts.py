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

## Version 1.4.1: Showcase Plays sensitivity artifacts

After the main artifacts (players/games/showcase/metadata) are written
unconditionally, this script ALSO attempts to generate `showcase
-sensitivity/<play_id>.json` "What if?" grids for up to 6 favorable + 6
unfavorable interactive-eligible showcase plays (`demo/
build_showcase_sensitivity.py`, which owns the mandatory reconciliation
gate). This retrains the frozen `baseline_v02` contact model a second time
in this process, from the SAME frozen dev-data file already required by
the normal publish path (`scripts/ensure_frozen_inputs.py`'s
`FROZEN_INPUT_LOCAL_PATH`) -- no new frozen input, no second season
rescore, no R2 write. If sensitivity generation fails its reconciliation
gate for ANY selected play, the exception propagates UNCAUGHT: the main
Explorer artifacts (already written) are left in place, but the CALLER's
own top-level failure (a non-zero exit under `scripts/publish_snapshot.sh`'s
`set -euo pipefail`) blocks the rest of that publish run, exactly like any
other stage failure -- see `demo/build_showcase_sensitivity.py`'s own
docstring for why a genuine reconciliation failure should stop a release
rather than silently ship a downgraded interactive layer.

Two cases are a deliberate, LOGGED no-op rather than a failure: no showcase
row is interactive-eligible at all, or the frozen dev-data file
(`scripts/ensure_frozen_inputs.py`'s `FROZEN_INPUT_LOCAL_PATH`) is not
present locally (e.g. this script invoked standalone, without that earlier
pipeline stage having run). In both cases `showcase.json` simply keeps its
input-eligibility-only `interactive_available` flags un-patched, and the
Showcase ranking itself is entirely unaffected. A MISSING dev-data file is
only ever a soft skip HERE, at this orchestration layer -- `demo/
build_showcase_sensitivity.train_frozen_contact_model` itself still raises
loudly if called directly with a path that doesn't exist, e.g. from a
context that DOES expect the frozen bundle to already be present.
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
import build_showcase_sensitivity as sensitivity_gen  # noqa: E402
import pandas as pd  # noqa: E402
import snapshot_data as sd  # noqa: E402
from ensure_frozen_inputs import FROZEN_INPUT_LOCAL_PATH  # noqa: E402

__all__ = [
    "ProductionExplorerGenerationError",
    "generate_production_explorer_artifacts",
]

DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "prospective" / "v1_1"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "prospective" / "v1_1"
DEFAULT_DEV_DATA_PATH = FROZEN_INPUT_LOCAL_PATH

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
    dev_data_path: Path = DEFAULT_DEV_DATA_PATH,
    generate_sensitivity: bool = True,
) -> dict[str, Any]:
    """Verify `snapshot_directory_name`'s integrity, then project its
    canonical `play_ledger.parquet` into Play Explorer browser artifacts
    (including `showcase.json`) under `output_dir`, then -- if
    `generate_sensitivity` and at least one showcase row is interactive
    -eligible and `dev_data_path` exists -- generate `showcase-sensitivity/
    <play_id>.json` "What if?" grids for up to 6 favorable + 6 unfavorable
    of them (see module docstring "Version 1.4.1: Showcase Plays
    sensitivity artifacts").

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
        ShowcaseSensitivityReconciliationError: if `generate_sensitivity`
            and a selected interactive candidate fails the mandatory
            reconciliation gate -- propagates uncaught (see module
            docstring); the main artifacts are already written by this
            point and are left in place.
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

    sensitivity_result: dict[str, Any] = {"interactive_play_ids": [], "output_dir": None}
    if generate_sensitivity:
        showcase_path = output_dir / "showcase.json"
        showcase_rows = json.loads(showcase_path.read_text())
        candidates = sensitivity_gen.select_interactive_candidates(showcase_rows)
        if not candidates:
            pass  # no showcase row is interactive-eligible -- nothing to generate.
        elif not dev_data_path.exists():
            print(
                f"[generate_production_explorer_artifacts] NOTE: skipping showcase sensitivity "
                f"generation -- frozen dev-data file not found at {dev_data_path} (run "
                "scripts/ensure_frozen_inputs.py first for the normal publish path)."
            )
        else:
            play_ledger = pd.read_parquet(play_ledger_path)
            sensitivity_result = sensitivity_gen.generate_showcase_sensitivity_artifacts(
                play_ledger=play_ledger,
                showcase_rows=showcase_rows,
                dev_data_path=dev_data_path,
                output_dir=output_dir / "showcase-sensitivity",
            )
            explorer_gen.patch_showcase_interactive_flags(
                output_dir, set(sensitivity_result["interactive_play_ids"])
            )

    result["showcase_interactive_play_ids"] = sensitivity_result["interactive_play_ids"]
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
    parser.add_argument("--dev-data-path", type=Path, default=DEFAULT_DEV_DATA_PATH)
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip showcase-sensitivity/<play_id>.json 'What if?' grid generation entirely "
        "(showcase.json's ranking is unaffected either way).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    snapshot_directory_name = build_snapshot_dir_name(args.data_through, args.snapshot_label)
    result = generate_production_explorer_artifacts(
        snapshot_directory_name=snapshot_directory_name,
        output_dir=args.output_dir,
        outputs_root=args.outputs_root,
        artifacts_root=args.artifacts_root,
        dev_data_path=args.dev_data_path,
        generate_sensitivity=not args.skip_sensitivity,
    )
    print(
        f"[generate_production_explorer_artifacts] play_ledger_version={result['play_ledger_version']}"
    )
    print(
        f"[generate_production_explorer_artifacts] rows={result['row_count']} "
        f"players={result['player_count']} games={result['game_count']}"
    )
    print(
        f"[generate_production_explorer_artifacts] showcase favorable="
        f"{result['showcase_favorable_count']} unfavorable={result['showcase_unfavorable_count']} "
        f"interactive={len(result['showcase_interactive_play_ids'])}"
    )
    print(
        "[generate_production_explorer_artifacts] source_play_ledger_sha256="
        f"{result['explore_metadata']['source_play_ledger_sha256']}"
    )
    print(f"[generate_production_explorer_artifacts] wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
