"""Version 1.4.0 Phase 4 (revised Phase 4.2): the offline Play Explorer
browser-artifact generator.

    canonical play_ledger.parquet
            v
    offline browser-artifact generator   <-- THIS MODULE
            v
    static players.json + per-player play index + per-game detail JSON
            v
    dashboard/static site

PROJECTION/PRESENTATION ONLY. This module never scores a play, never
recomputes a probability, never derives `observed_final_run_value` (Rf),
`expected_run_value` (E0), or `contact_luck_runs` (Rf - E0) -- every
scoring-truth field in its output is copied verbatim from the canonical
`play_ledger.parquet` it is given. It imports NOTHING from `mlb_luck_score`
except the two pure-validation modules that already ship with the ledger
producer (`play_ledger_schema`, `play_ledger_metadata`) -- no model, no
training code, no `predict_proba_ordered` anywhere in this file (see
`tests/test_play_explorer_fixture_generator.py::
test_generator_imports_no_model_or_training_code` for the structural
guarantee, mirroring `tests/test_dashboard_isolation.py`'s pattern for
`dashboard/`).

## Version safety: fail closed

`require_compatible_play_ledger_version` is called before ANY row is
projected. It accepts ONLY the exact currently-frozen `PLAY_LEDGER_VERSION`
("2.0" as of Phase 3.1) -- a missing version, the sealed v1.0 artifact
(2026-08-15), or any future unknown version string is rejected with
`IncompatiblePlayLedgerVersionError`, never silently accepted or
downgraded/upgraded in place. This is what makes it structurally
impossible to publish the archived 2026-08-15 v1.0 ledger through the
Explorer.

## Where this script's OWN input comes from

This module is deliberately agnostic about where `play_ledger.parquet`/
`play_ledger_metadata.json` came from -- a real prospective snapshot's
`outputs/prospective/v1_1/<date>/` directory (production, not yet wired as
of this writing) or a local, bounded, real-2021-2023/2024-data ledger
built by `demo/build_play_explorer_dev_ledger.py` (local development only,
never 2025, never a prospective 2026 run). Both are just files on disk to
this module -- it never trains or downloads anything itself.

## Phase 4.2: sharded, player-first browser artifacts

Phase 4.1 shipped a single monolithic `search-index.json` containing every
scored play. At full 2024 development-data scale that file measured
~29.3 MiB -- over Cloudflare Pages' 25 MiB per-asset limit -- and the
Explorer UI loaded the entire season's plays on every page visit even
though a visitor only ever looks at one hitter at a time. Phase 4.2
replaces it with four artifact kinds, all still produced by this ONE
generator from the SAME canonical ledger:

- `players.json`: a small catalog, one row per batter represented in the
  scored play corpus (`batter_id`, `batter_name`, `play_count`), sorted
  deterministically by `batter_id`. This is the only file the Explorer
  landing page loads up front.
- `players/<batter_id>.json`: that batter's own scored plays ONLY, in the
  same compact per-play shape Phase 4.1's search-index row used (`play_id,
  game_pk, game_date, batter_id, batter_name, outcome_class, launch_speed,
  launch_angle, expected_run_value, contact_luck_runs`). The dashboard
  fetches exactly one of these, only after a visitor selects that hitter --
  see `dashboard/static/explore.js`'s module docstring.
- `games/<game_pk>.json`: unchanged from Phase 4.1 -- full per-play detail
  for the individual play page, still partitioned by game.
- `explore-metadata.json`: unchanged in spirit, extended with `player_count`
  (see below).

Every play appears in EXACTLY one `players/<batter_id>.json` file (a play
has exactly one batter) and EXACTLY one `games/<game_pk>.json` file -- there
is no overlap or duplication between shards of the same kind.

## Compact per-player index vs. per-game detail split

`build_players_catalog_rows` returns the minimal catalog row set needed for
hitter search (`batter_id, batter_name, play_count`), one row per batter
with at least one scored play, sorted by `batter_id`.
`build_player_play_index_rows` partitions by `batter_id` and returns, per
batter, the compact per-play row set needed to find/filter/sort that
batter's plays (`play_id, game_pk, game_date, batter_id, batter_name,
outcome_class, launch_speed, launch_angle, expected_run_value,
contact_luck_runs`), each batter's rows sorted by `(game_date, game_pk,
play_id)`. `build_per_game_detail_rows` is unchanged from Phase 4.1: it
partitions by `game_pk` and returns, per play, every canonical v2.0 field
the individual play page needs, plus the same `batter_name` overlay. All
three are scoped to SCORED rows only (`is_scored=True` / `resolved_rows`) --
an Explorer that shows a play with no outcome/Contact Luck to display is
not useful; unresolved rows are simply omitted from these artifacts (they
remain fully present in the canonical `play_ledger.parquet` itself,
unaffected by this projection).

`batter_name` is a PRESENTATION OVERLAY (see `play_ledger_schema.py`'s own
module docstring on why `batter_name` is not a canonical field) -- supplied
to this module as a plain `{batter_id: name}` dict, never derived from
canonical scoring data, and left `None` for any unresolved id (never
fabricated). The canonical Parquet itself is never modified to add names.

## Provenance: `explore-metadata.json`

A small, file-level (never per-row, never per-shard) artifact --
`explorer_artifact_version`, `play_ledger_version`, `season`,
`data_through_date` (verbatim from the source ledger metadata; `None` stays
`None`, never invented -- the local dev ledger genuinely has no
data-through date, unlike a real prospective snapshot), `play_count`,
`player_count` (Phase 4.2: the number of `players/<batter_id>.json` files
written, i.e. `len(players.json)`), `game_count`, and
`source_play_ledger_sha256` (the ACTUAL SHA-256 of the input Parquet's raw
bytes, computed by this module, never copied from elsewhere). No generation
timestamp -- deterministic for identical inputs, same convention as
`play_ledger_metadata.py`/`dashboard/demo_counterfactual_grid.json`. This is
what lets `dashboard/explore_content.py` enforce a SECOND, independent
compatibility check at build time (`explorer_artifact_version` +
`play_ledger_version`), after this generator's own -- the same two-layer
"never trust one boundary alone" pattern CLAUDE.md's v1.1.2 raw-Statcast
-cache fix established.

`EXPLORER_ARTIFACT_VERSION` was bumped from `"1.0"` to `"2.0"` for this
Phase -- the on-disk STRUCTURE changed (monolithic search-index.json
replaced by players.json + players/<id>.json), independent of
`PLAY_LEDGER_VERSION` (unchanged; the canonical scoring contract this
generator reads did not change). `dashboard/explore_content.py`'s
`SUPPORTED_EXPLORER_ARTIFACT_VERSION` was bumped alongside it.

## Determinism

All output artifacts are sorted/serialized deterministically (`players.json`
by `batter_id`; each per-player file's plays by `(game_date, game_pk,
play_id)`; each per-game file's plays by `play_id`; all JSON with
`sort_keys=True`, no timestamp field anywhere), so re-running this generator
against the same canonical ledger + name mapping reproduces byte-identical
output -- see `tests/test_play_explorer_fixture_generator.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_luck_score.scoring.play_ledger_metadata import validate_play_ledger_metadata
from mlb_luck_score.scoring.play_ledger_schema import (
    PLAY_LEDGER_VERSION,
    resolved_rows,
    validate_play_ledger,
)

__all__ = [
    "EXPLORER_ARTIFACT_VERSION",
    "SHOWCASE_CANONICAL_FIELDS",
    "SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS",
    "SHOWCASE_TOP_N",
    "IncompatiblePlayLedgerVersionError",
    "build_explore_metadata",
    "build_per_game_detail_rows",
    "build_player_play_index_rows",
    "build_players_catalog_rows",
    "build_showcase_rows",
    "compute_file_sha256",
    "generate_explorer_artifacts",
    "is_showcase_interactive_eligible",
    "load_batter_names",
    "patch_showcase_interactive_flags",
    "require_compatible_play_ledger_version",
]

#: Bump on any change to the Explorer browser-artifact STRUCTURE (new/
#: renamed/retyped field, a changed file layout) -- independent of
#: `PLAY_LEDGER_VERSION` (which versions the CANONICAL scoring contract
#: this generator reads, not the presentation artifacts it writes).
#: `dashboard/explore_content.py` pins its own
#: `SUPPORTED_EXPLORER_ARTIFACT_VERSION` and checks it independently at
#: build time -- see that module's docstring. Bumped to "2.0" for Phase
#: 4.2's sharded players.json + players/<id>.json replacement of the
#: monolithic search-index.json. Bumped to "3.0" for Version 1.4.1's
#: `showcase.json` addition (see `build_showcase_rows` below) -- a new
#: required file, so a pre-Showcase ("2.0") artifact set is correctly
#: rejected as incompatible rather than silently rendering an Explorer with
#: no Showcase section.
EXPLORER_ARTIFACT_VERSION = "3.0"

#: Version 1.4.1: how many of the TRUE season extremes (by `contact_luck_
#: runs`, scored rows only, no qualification filter, no per-player cap) go
#: into each Showcase group. Deliberately small and editorial -- this is a
#: curated entry point into the Explorer, not another large table (see
#: module docstring "Showcase Plays" section below).
SHOWCASE_TOP_N = 12

#: The presentation fields copied verbatim into each showcase row (real
#: `play_ledger.parquet` columns only, same overlay convention as
#: `PLAYER_INDEX_CANONICAL_FIELDS` -- `batter_name` is attached separately).
SHOWCASE_CANONICAL_FIELDS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "game_date",
    "batter_id",
    "outcome_class",
    "launch_speed",
    "launch_angle",
    "expected_run_value",
    "observed_run_value",
    "contact_luck_runs",
)

#: The frozen `baseline_v02` contact model's own required inputs -- a
#: showcase play only qualifies for a `showcase-sensitivity/<play_id>.json`
#: "What if?" grid (see `demo/build_showcase_sensitivity.py`) if EVERY one
#: of these is non-null for that play. A true season extreme with missing
#: raw Statcast measurements (real gap in the ledger -- see the showcase
#: report for real 2026 examples) still appears on the showcase with
#: `interactive_available=False`; it is NEVER dropped from the ranking for
#: this reason (see module docstring "Interactive eligibility").
SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "bb_type",
    "stand",
)

#: The players catalog field set -- enough for hitter search/selection
#: without loading any batter's play data.
PLAYERS_CATALOG_FIELDS: tuple[str, ...] = ("batter_id", "batter_name", "play_count")

#: The compact per-player play index field set -- enough for client-side
#: find/filter/sort of one batter's plays without loading per-play detail.
#: Every one of these is a real canonical `play_ledger.parquet` column
#: (copied verbatim below); `batter_name` is a SEPARATE presentation
#: overlay, attached afterward from `batter_names`, never read from the
#: ledger itself (it has no such column -- see module docstring /
#: `play_ledger_schema.py`).
PLAYER_INDEX_CANONICAL_FIELDS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "game_date",
    "batter_id",
    "outcome_class",
    "launch_speed",
    "launch_angle",
    "expected_run_value",
    "contact_luck_runs",
)
#: Full field set as it appears in each emitted per-player index row
#: (canonical fields, in their canonical order, plus the overlay field) --
#: used only by tests that want the exact output shape without duplicating
#: it inline.
PLAYER_INDEX_FIELDS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "game_date",
    "batter_id",
    "batter_name",
    "outcome_class",
    "launch_speed",
    "launch_angle",
    "expected_run_value",
    "contact_luck_runs",
)

#: Canonical v2.0 fields copied verbatim into each per-game detail record
#: (real `play_ledger.parquet` columns only -- `batter_name` is attached
#: separately, same overlay pattern as above). Deliberately excludes
#: defensive/advancement component fields (out of v1.4.0 scope) and any
#: raw Statcast `des` text (site-wide data-attribution gap, unresolved).
PER_GAME_DETAIL_CANONICAL_FIELDS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "batter_id",
    "stand",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "spray_angle_approx",
    "hit_distance_sc",
    "outcome_class",
    "observed_run_value",
    "p_out",
    "p_single",
    "p_double",
    "p_triple",
    "p_home_run",
    "expected_run_value",
    "contact_luck_runs",
)
#: Full field set as it appears in each emitted per-game detail record
#: (canonical fields, in their canonical order, plus the overlay field).
PER_GAME_DETAIL_FIELDS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "batter_id",
    "batter_name",
    "stand",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "spray_angle_approx",
    "hit_distance_sc",
    "outcome_class",
    "observed_run_value",
    "p_out",
    "p_single",
    "p_double",
    "p_triple",
    "p_home_run",
    "expected_run_value",
    "contact_luck_runs",
)


class IncompatiblePlayLedgerVersionError(ValueError):
    """Raised when a candidate canonical ledger's `play_ledger_version` is
    not exactly the currently-frozen `PLAY_LEDGER_VERSION` -- fail closed,
    never silently accept a missing, older, or unknown-future version.
    """


def require_compatible_play_ledger_version(metadata: dict[str, Any]) -> None:
    """Fail closed unless `metadata["play_ledger_version"] ==
    PLAY_LEDGER_VERSION` exactly.

    Rejects: a missing key, the sealed v1.0 artifact, and any unrecognized
    future version string -- there is no allow-list, no best-effort
    coercion, and no fallback rendering path. This is what makes it
    structurally impossible for the archived 2026-08-15 v1.0 snapshot's
    play ledger to ever reach the Explorer.
    """
    version = metadata.get("play_ledger_version")
    if version != PLAY_LEDGER_VERSION:
        raise IncompatiblePlayLedgerVersionError(
            f"play_ledger_version={version!r} is not compatible with this Play Explorer "
            f"generator (requires exactly {PLAY_LEDGER_VERSION!r}) -- refusing to publish. "
            "This is a fail-closed check: a missing version, an older version (e.g. the "
            "sealed 2026-08-15 v1.0 snapshot), and any unrecognized future version are all "
            "rejected, never silently accepted."
        )


def _clean_scalar(value: Any) -> Any:
    """`pd.NA`/`NaN` -> `None` for JSON serialization; everything else
    passed through as a plain Python scalar (`.item()` for numpy scalars).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def build_players_catalog_rows(
    play_ledger: pd.DataFrame, batter_names: dict[int, str]
) -> list[dict[str, Any]]:
    """Project `play_ledger` (already `validate_play_ledger`-passing) into
    the small players catalog: one row per batter with at least one scored
    play (`batter_id, batter_name, play_count`), sorted deterministically
    by `batter_id`.
    """
    scored = resolved_rows(play_ledger)
    counts = scored["batter_id"].astype("int64").value_counts()
    rows: list[dict[str, Any]] = []
    for batter_id in sorted(int(b) for b in counts.index):
        rows.append(
            {
                "batter_id": batter_id,
                "batter_name": batter_names.get(batter_id),
                "play_count": int(counts.loc[batter_id]),
            }
        )
    return rows


def build_player_play_index_rows(
    play_ledger: pd.DataFrame, batter_names: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Project `play_ledger` into per-batter compact play-index records,
    scored rows only, partitioned by `str(batter_id)`, each batter's plays
    sorted by `(game_date, game_pk, play_id)`. Every field is copied
    verbatim from the canonical ledger (see `PLAYER_INDEX_CANONICAL_
    FIELDS`); `batter_name` is the same presentation overlay
    `build_players_catalog_rows`/`build_per_game_detail_rows` apply.
    """
    scored = resolved_rows(play_ledger).copy()
    scored["_batter_id_key"] = scored["batter_id"].astype("int64").astype(str)
    partitions: dict[str, list[dict[str, Any]]] = {}
    for batter_id_key, group in scored.groupby("_batter_id_key", sort=True):
        ordered = group.sort_values(["game_date", "game_pk", "play_id"], kind="stable")
        records: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            batter_id = int(row["batter_id"])
            record = {field: _clean_scalar(row[field]) for field in PLAYER_INDEX_CANONICAL_FIELDS}
            record["batter_id"] = batter_id
            record["batter_name"] = batter_names.get(batter_id)
            records.append(record)
        partitions[str(batter_id_key)] = records
    return partitions


def build_per_game_detail_rows(
    play_ledger: pd.DataFrame, batter_names: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Project `play_ledger` into per-game detail records, scored rows
    only, partitioned by `str(game_pk)`, each game's plays sorted by
    `play_id`. Every canonical v2.0 field the play page needs is copied
    verbatim (see `PER_GAME_DETAIL_FIELDS`); `batter_name` is the same
    presentation overlay `build_players_catalog_rows` applies.
    """
    scored = resolved_rows(play_ledger).copy()
    scored["_game_pk_key"] = scored["game_pk"].astype("int64").astype(str)
    partitions: dict[str, list[dict[str, Any]]] = {}
    for game_pk_key, group in scored.groupby("_game_pk_key", sort=True):
        ordered = group.sort_values("play_id", kind="stable")
        records: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            batter_id = int(row["batter_id"])
            record = {
                field: _clean_scalar(row[field]) for field in PER_GAME_DETAIL_CANONICAL_FIELDS
            }
            record["batter_id"] = batter_id
            record["batter_name"] = batter_names.get(batter_id)
            records.append(record)
        partitions[str(game_pk_key)] = records
    return partitions


def is_showcase_interactive_eligible(row: pd.Series) -> bool:
    """True only if every `SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS` value is
    non-null for `row` -- the frozen `baseline_v02` contact model's own
    required inputs (see `demo/build_showcase_sensitivity.py`, which
    additionally requires the model's OWN-COORDINATE score to reconcile
    with this row's canonical `p_out`..`p_home_run`/`expected_run_value`
    before a play actually gets a sensitivity grid; this function only
    checks the input side).
    """
    return all(pd.notna(row[field]) for field in SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS)


def build_showcase_rows(
    play_ledger: pd.DataFrame, batter_names: dict[int, str]
) -> list[dict[str, Any]]:
    """Project `play_ledger` into the Showcase Plays list: the `SHOWCASE_
    TOP_N` biggest favorable and `SHOWCASE_TOP_N` biggest unfavorable
    `contact_luck_runs` breaks among SCORED rows -- the TRUE season
    extremes. No qualification filter, no one-play-per-player cap: if one
    batter genuinely owns multiple season extremes, every one of their
    rows appears (see module docstring "Showcase selection").

    Deterministic ordering: `contact_luck_runs` (descending for favorable,
    ascending for unfavorable) is the primary key; ties break on
    `game_date`, then `game_pk`, then `play_id`, all ascending -- a total
    order over the scored population, so re-running this generator against
    an unchanged ledger always reproduces the identical 24 rows in the
    identical order. `kind="mergesort"` (stable) so pandas' own sort is
    deterministic across runs, not just the key tuple.

    Each row carries `group` (`"favorable"`/`"unfavorable"`), `rank`
    (1-`SHOWCASE_TOP_N` within its group), the `SHOWCASE_CANONICAL_FIELDS`
    copied verbatim from the ledger, the same `batter_name` presentation
    overlay every other artifact in this module applies, and
    `interactive_available` (see `is_showcase_interactive_eligible`).
    """
    scored = resolved_rows(play_ledger)
    sort_keys = ["contact_luck_runs", "game_date", "game_pk", "play_id"]

    favorable = scored.sort_values(
        sort_keys, ascending=[False, True, True, True], kind="mergesort"
    ).head(SHOWCASE_TOP_N)
    unfavorable = scored.sort_values(
        sort_keys, ascending=[True, True, True, True], kind="mergesort"
    ).head(SHOWCASE_TOP_N)

    rows: list[dict[str, Any]] = []
    for group, ordered in (("favorable", favorable), ("unfavorable", unfavorable)):
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            batter_id = int(row["batter_id"])
            record = {field: _clean_scalar(row[field]) for field in SHOWCASE_CANONICAL_FIELDS}
            record["batter_id"] = batter_id
            record["batter_name"] = batter_names.get(batter_id)
            record["group"] = group
            record["rank"] = rank
            record["interactive_available"] = bool(is_showcase_interactive_eligible(row))
            rows.append(record)
    return rows


def load_batter_names(path: Path | None) -> dict[int, str]:
    """Load a `{batter_id: name}` presentation-overlay mapping from a JSON
    file (`{"12345": "Some Player", ...}`). Returns `{}` if `path` is
    `None` -- an absent name mapping means every play renders with no
    resolved name (the site's existing `"Player {id}"` fallback pattern
    applies at the presentation layer, not here).
    """
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items() if v is not None}


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of `path`'s raw bytes -- mirrors `prospective_manifest.
    compute_file_sha256`/`scripts.archive_snapshot.compute_file_sha256`'s
    identical implementation (duplicated, not imported -- this module stays
    outside the `mlb_luck_score`/prospective-orchestration import graph
    entirely, per the module docstring)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_explore_metadata(
    *,
    play_ledger_path: Path,
    play_ledger_metadata: dict[str, Any],
    play_count: int,
    player_count: int,
    game_count: int,
    showcase_favorable_count: int,
    showcase_unfavorable_count: int,
    showcase_interactive_count: int,
) -> dict[str, Any]:
    """Build the file-level Explorer browser-artifact provenance dict --
    NEVER repeated per row or per shard (see module docstring).
    `data_through_date` is copied verbatim from the source ledger's own
    metadata; a local dev ledger genuinely has none, so it stays `None`
    rather than being invented. `source_play_ledger_sha256` is the ACTUAL
    hash of the input Parquet file this run read, computed here, not copied
    from anywhere. Deterministic: no generation timestamp.

    Version 1.4.1: `showcase_favorable_count`/`showcase_unfavorable_count`
    let `dashboard/build.py` reconcile `showcase.json`'s actual row counts
    against this provenance the same way it already does for `play_count`/
    `player_count`/`game_count` -- a mismatch is a hard build failure, never
    silently skipped. `showcase_interactive_count` is the number of those
    rows with a `showcase-sensitivity/<play_id>.json` grid actually written
    (0 if none, e.g. slider generation was skipped or deferred -- see
    `demo/build_showcase_sensitivity.py`).
    """
    return {
        "explorer_artifact_version": EXPLORER_ARTIFACT_VERSION,
        "play_ledger_version": play_ledger_metadata["play_ledger_version"],
        "season": play_ledger_metadata["season"],
        "data_through_date": play_ledger_metadata.get("data_through_date"),
        "play_count": play_count,
        "player_count": player_count,
        "game_count": game_count,
        "showcase_favorable_count": showcase_favorable_count,
        "showcase_unfavorable_count": showcase_unfavorable_count,
        "showcase_interactive_count": showcase_interactive_count,
        "source_play_ledger_sha256": compute_file_sha256(play_ledger_path),
    }


def generate_explorer_artifacts(
    *,
    play_ledger_path: Path,
    play_ledger_metadata_path: Path,
    output_dir: Path,
    names_path: Path | None = None,
) -> dict[str, Any]:
    """Read a canonical `play_ledger.parquet` + its metadata sidecar,
    validate them (including the fail-closed version check), and write
    `players.json` + `players/<batter_id>.json` + `games/<game_pk>.json` +
    `explore-metadata.json` into `output_dir`.

    Raises:
        IncompatiblePlayLedgerVersionError: if the metadata's
            `play_ledger_version` is not the current `PLAY_LEDGER_VERSION`.
        PlayLedgerValidationError / PlayLedgerMetadataError: if the
            ledger/metadata otherwise fail their own frozen validation
            contracts.
    """
    metadata = json.loads(play_ledger_metadata_path.read_text())
    require_compatible_play_ledger_version(metadata)

    play_ledger = pd.read_parquet(play_ledger_path)
    validate_play_ledger(play_ledger)
    validate_play_ledger_metadata(metadata, play_ledger)

    batter_names = load_batter_names(names_path)
    players_catalog = build_players_catalog_rows(play_ledger, batter_names)
    player_play_indexes = build_player_play_index_rows(play_ledger, batter_names)
    per_game = build_per_game_detail_rows(play_ledger, batter_names)
    showcase_rows = build_showcase_rows(play_ledger, batter_names)
    total_play_count = sum(row["play_count"] for row in players_catalog)

    showcase_favorable = [r for r in showcase_rows if r["group"] == "favorable"]
    showcase_unfavorable = [r for r in showcase_rows if r["group"] == "unfavorable"]
    # `interactive_available` here reflects raw MODEL-INPUT eligibility only
    # (is_showcase_interactive_eligible) -- the actual count of plays that
    # get a real showcase-sensitivity/<play_id>.json grid is a strict
    # subset of this (capped at 6 per group, and only ever written after
    # passing the mandatory reconciliation gate -- see `demo/
    # build_showcase_sensitivity.py`). `patch_showcase_interactive_flags`
    # below is the SECOND, later pass that downgrades this field to ground
    # truth once that generation has actually run (or left unpatched, i.e.
    # input-eligible-only, for a caller that never attempts sensitivity
    # generation at all -- see that function's own docstring).
    showcase_interactive_count = sum(1 for r in showcase_rows if r["interactive_available"])

    explore_metadata = build_explore_metadata(
        play_ledger_path=play_ledger_path,
        play_ledger_metadata=metadata,
        play_count=total_play_count,
        player_count=len(players_catalog),
        game_count=len(per_game),
        showcase_favorable_count=len(showcase_favorable),
        showcase_unfavorable_count=len(showcase_unfavorable),
        showcase_interactive_count=showcase_interactive_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    players_dir = output_dir / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "players.json").write_text(
        json.dumps(players_catalog, sort_keys=True, separators=(",", ":"))
    )
    for batter_id, records in player_play_indexes.items():
        (players_dir / f"{batter_id}.json").write_text(
            json.dumps(records, sort_keys=True, separators=(",", ":"))
        )
    for game_pk, records in per_game.items():
        (games_dir / f"{game_pk}.json").write_text(
            json.dumps(records, sort_keys=True, separators=(",", ":"))
        )
    (output_dir / "showcase.json").write_text(
        json.dumps(showcase_rows, sort_keys=True, separators=(",", ":"))
    )
    (output_dir / "explore-metadata.json").write_text(
        json.dumps(explore_metadata, sort_keys=True, separators=(",", ":"))
    )

    return {
        "play_ledger_version": metadata["play_ledger_version"],
        "row_count": total_play_count,
        "player_count": len(players_catalog),
        "game_count": len(per_game),
        "showcase_favorable_count": len(showcase_favorable),
        "showcase_unfavorable_count": len(showcase_unfavorable),
        "showcase_interactive_count": showcase_interactive_count,
        "output_dir": str(output_dir),
        "explore_metadata": explore_metadata,
    }


def patch_showcase_interactive_flags(output_dir: Path, interactive_play_ids: set[str]) -> None:
    """SECOND pass, run AFTER `generate_explorer_artifacts` and AFTER
    sensitivity-grid generation has actually completed (see `demo/
    build_showcase_sensitivity.py`): rewrites `showcase.json`'s
    `interactive_available` field to ground truth -- `True` only for a
    `play_id` in `interactive_play_ids` (i.e. a real `showcase-sensitivity/
    <play_id>.json` grid was actually written and passed the mandatory
    reconciliation gate), `False` for every other row, INCLUDING a row that
    was input-eligible but excluded only by the per-group interactive cap.
    Also updates `explore-metadata.json`'s `showcase_interactive_count` to
    `len(interactive_play_ids)` so the two files never disagree.

    Never called at all if a caller does not attempt sensitivity
    generation -- `showcase.json` then keeps `generate_explorer_artifacts`'s
    original, input-eligibility-only flag, which is still an honest (if
    more optimistic) statement: "this play's raw model inputs are present,"
    not "a grid exists for it." `dashboard/explore_content.py`'s own
    validation only ever trusts `interactive_available` to gate whether a
    `showcase-sensitivity/<play_id>.json` is EXPECTED to exist for a row
    marked `True` -- see that module's docstring.
    """
    showcase_path = output_dir / "showcase.json"
    metadata_path = output_dir / "explore-metadata.json"
    rows = json.loads(showcase_path.read_text())
    for row in rows:
        row["interactive_available"] = row["play_id"] in interactive_play_ids
    showcase_path.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")))

    metadata = json.loads(metadata_path.read_text())
    metadata["showcase_interactive_count"] = len(interactive_play_ids)
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--play-ledger", required=True, type=Path, help="Path to a canonical play_ledger.parquet."
    )
    parser.add_argument(
        "--play-ledger-metadata",
        required=True,
        type=Path,
        help="Path to that ledger's play_ledger_metadata.json sidecar.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write the browser artifacts."
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=None,
        help="Optional {batter_id: name} JSON presentation-overlay mapping.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = generate_explorer_artifacts(
        play_ledger_path=args.play_ledger,
        play_ledger_metadata_path=args.play_ledger_metadata,
        output_dir=args.output_dir,
        names_path=args.names,
    )
    print(f"[build_play_explorer_fixture] play_ledger_version={result['play_ledger_version']}")
    print(
        f"[build_play_explorer_fixture] rows={result['row_count']} "
        f"players={result['player_count']} games={result['game_count']}"
    )
    print(
        f"[build_play_explorer_fixture] showcase favorable={result['showcase_favorable_count']} "
        f"unfavorable={result['showcase_unfavorable_count']} "
        f"interactive_eligible={result['showcase_interactive_count']}"
    )
    print(
        "[build_play_explorer_fixture] source_play_ledger_sha256="
        f"{result['explore_metadata']['source_play_ledger_sha256']}"
    )
    print(f"[build_play_explorer_fixture] wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
