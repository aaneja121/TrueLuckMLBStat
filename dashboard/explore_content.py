"""Contact Luck v1.4.0 (Phase 4.2): view-model/validation layer for the Play
Explorer.

Reads ONLY the committed sharded browser-artifact fixture (`players.json` +
`players/<batter_id>.json` + `games/<game_pk>.json`), produced OFFLINE by
`demo/build_play_explorer_fixture.py` (which imports `mlb_luck_score`
freely -- this module must never do that, see
`tests/test_dashboard_isolation.py`). Every value here is copied verbatim
from those already-generated files; the only computation this module
performs is a pure-arithmetic reconciliation check on numbers already
present in the fixture (mirroring `demo_content.py`'s precedent for
`/demo/`) -- no scoring formula is reimplemented, no probability is invoked
or recomputed.

Phase 4.2 replaced the Phase 4.1 monolithic `search-index.json` (~29.3 MiB
at full 2024 development-data scale -- over Cloudflare Pages' 25 MiB
per-asset limit) with a small `players.json` catalog plus one
`players/<batter_id>.json` per batter. `load_explore_catalog` loads and
cross-validates ALL of these at build time (this is offline build-time
validation, not the browser's runtime behavior -- the browser itself never
fetches more than one player index at a time, see
`dashboard/static/explore.js`'s module docstring).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SUPPORTED_EXPLORER_ARTIFACT_VERSION",
    "SUPPORTED_PLAY_LEDGER_VERSION",
    "ExploreContentError",
    "ExploreIndexRow",
    "ExploreLoadedData",
    "ExploreMetadata",
    "PlayDetail",
    "PlayerCatalogEntry",
    "load_explore_catalog",
    "load_explore_metadata",
    "load_play_detail",
    "load_player_play_index",
    "load_players_catalog",
]

_RECONCILIATION_TOLERANCE = 1e-6
_PROBABILITY_SUM_TOLERANCE = 1e-3

#: Phase 4.1: the SECOND, independent fail-closed compatibility boundary,
#: after `demo/build_play_explorer_fixture.py`'s own `require_compatible_
#: play_ledger_version` check. Pinned HERE, duplicated rather than
#: imported (this module must never import `mlb_luck_score` -- see module
#: docstring), so a bug or stale value in the generator's own check can
#: never be the only thing standing between a v1.0 (or any incompatible)
#: ledger and the published site. Bump `SUPPORTED_PLAY_LEDGER_VERSION`/
#: `SUPPORTED_EXPLORER_ARTIFACT_VERSION` together with `mlb_luck_score.
#: scoring.play_ledger_schema.PLAY_LEDGER_VERSION`/`demo.
#: build_play_explorer_fixture.EXPLORER_ARTIFACT_VERSION` whenever those
#: change -- never silently widen either contract by leaving this stale.
#: `SUPPORTED_EXPLORER_ARTIFACT_VERSION` was bumped to "2.0" for Phase 4.2's
#: sharded players.json + players/<id>.json replacement of the monolithic
#: search-index.json -- an old "1.0" fixture is now correctly rejected as
#: incompatible (it has no players.json to load).
SUPPORTED_PLAY_LEDGER_VERSION = "2.0"
SUPPORTED_EXPLORER_ARTIFACT_VERSION = "2.0"

_EXPLORE_METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "explorer_artifact_version",
    "play_ledger_version",
    "season",
    "data_through_date",
    "play_count",
    "player_count",
    "game_count",
    "source_play_ledger_sha256",
)

_PLAYERS_CATALOG_REQUIRED_KEYS: tuple[str, ...] = ("batter_id", "batter_name", "play_count")

_PLAYER_INDEX_ROW_REQUIRED_KEYS: tuple[str, ...] = (
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

_PLAY_DETAIL_REQUIRED_KEYS: tuple[str, ...] = (
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

_PROBABILITY_KEYS: tuple[str, ...] = ("p_out", "p_single", "p_double", "p_triple", "p_home_run")


class ExploreContentError(Exception):
    """Raised when the committed Play Explorer fixture is missing, malformed,
    or internally inconsistent -- never a reason to fabricate a placeholder
    row or silently drop a play.
    """


@dataclass(frozen=True)
class PlayerCatalogEntry:
    batter_id: int
    batter_name: str | None
    play_count: int


@dataclass(frozen=True)
class ExploreIndexRow:
    play_id: str
    game_pk: int
    game_date: str
    batter_id: int
    batter_name: str | None
    outcome_class: str
    launch_speed: float | None
    launch_angle: float | None
    expected_run_value: float
    contact_luck_runs: float


@dataclass(frozen=True)
class ExploreLoadedData:
    players: list[PlayerCatalogEntry]
    players_dir: Path
    games_dir: Path
    total_play_count: int
    game_pks: frozenset[int]


@dataclass(frozen=True)
class ExploreMetadata:
    explorer_artifact_version: str
    play_ledger_version: str
    season: int
    data_through_date: str | None
    play_count: int
    player_count: int
    game_count: int
    source_play_ledger_sha256: str


@dataclass(frozen=True)
class PlayDetail:
    play_id: str
    game_pk: int
    at_bat_number: int
    pitch_number: int
    game_date: str
    season: int
    batter_id: int
    batter_name: str | None
    stand: str
    launch_speed: float | None
    launch_angle: float | None
    bb_type: str | None
    spray_angle_approx: float | None
    hit_distance_sc: float | None
    outcome_class: str
    observed_run_value: float
    p_out: float
    p_single: float
    p_double: float
    p_triple: float
    p_home_run: float
    expected_run_value: float
    contact_luck_runs: float


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExploreContentError(f"failed to load {path}: {exc}") from exc


def _validate_player_index_row(raw: dict[str, Any], position: int, source: Path) -> ExploreIndexRow:
    missing = [k for k in _PLAYER_INDEX_ROW_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ExploreContentError(f"{source} row {position} missing key(s): {missing}")
    return ExploreIndexRow(
        play_id=raw["play_id"],
        game_pk=int(raw["game_pk"]),
        game_date=raw["game_date"],
        batter_id=int(raw["batter_id"]),
        batter_name=raw.get("batter_name"),
        outcome_class=raw["outcome_class"],
        launch_speed=None if raw["launch_speed"] is None else float(raw["launch_speed"]),
        launch_angle=None if raw["launch_angle"] is None else float(raw["launch_angle"]),
        expected_run_value=float(raw["expected_run_value"]),
        contact_luck_runs=float(raw["contact_luck_runs"]),
    )


def load_explore_metadata(path: Path) -> ExploreMetadata:
    """Load and validate the committed `explore-metadata.json` -- the
    SECOND, independent fail-closed compatibility check (see
    `SUPPORTED_PLAY_LEDGER_VERSION`'s docstring above).

    Raises:
        ExploreContentError: if the file is missing/malformed, missing a
            required key, or either `play_ledger_version` or
            `explorer_artifact_version` is not exactly the version this
            dashboard build supports.
    """
    if not path.exists():
        raise ExploreContentError(
            f"Play Explorer metadata not found at {path}. It is committed browser-artifact "
            "provenance produced by `demo/build_play_explorer_fixture.py` -- a Play Explorer "
            "fixture with no accompanying explore-metadata.json is a malformed/incomplete "
            "fixture, never silently treated as absent."
        )
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ExploreContentError(f"{path} must contain a JSON object, got {type(raw)}")

    missing = [k for k in _EXPLORE_METADATA_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ExploreContentError(f"explore-metadata.json missing key(s): {missing}")

    if raw["play_ledger_version"] != SUPPORTED_PLAY_LEDGER_VERSION:
        raise ExploreContentError(
            f"explore-metadata.json play_ledger_version={raw['play_ledger_version']!r} is not "
            f"supported by this dashboard build (requires exactly "
            f"{SUPPORTED_PLAY_LEDGER_VERSION!r}) -- refusing to build the Play Explorer from "
            "this artifact. This is a fail-closed check: a missing version, an older version "
            "(e.g. the sealed 2026-08-15 v1.0 snapshot), and any unrecognized future version "
            "are all rejected, never silently accepted."
        )
    if raw["explorer_artifact_version"] != SUPPORTED_EXPLORER_ARTIFACT_VERSION:
        raise ExploreContentError(
            f"explore-metadata.json explorer_artifact_version="
            f"{raw['explorer_artifact_version']!r} is not supported by this dashboard build "
            f'(requires exactly {SUPPORTED_EXPLORER_ARTIFACT_VERSION!r}). A Phase 4.1 ("1.0") '
            "monolithic search-index.json fixture is correctly rejected here -- it has no "
            "players.json/players/<id>.json to load."
        )

    return ExploreMetadata(
        explorer_artifact_version=raw["explorer_artifact_version"],
        play_ledger_version=raw["play_ledger_version"],
        season=int(raw["season"]),
        data_through_date=raw.get("data_through_date"),
        play_count=int(raw["play_count"]),
        player_count=int(raw["player_count"]),
        game_count=int(raw["game_count"]),
        source_play_ledger_sha256=raw["source_play_ledger_sha256"],
    )


def load_players_catalog(path: Path) -> list[PlayerCatalogEntry]:
    """Load and validate the committed `players.json` catalog.

    Raises:
        ExploreContentError: if the file is missing/malformed, a row is
            missing a required key, or `batter_id` is duplicated.
    """
    if not path.exists():
        raise ExploreContentError(
            f"Play Explorer players catalog not found at {path}. It is committed "
            "browser-artifact data produced by `demo/build_play_explorer_fixture.py` -- it is "
            "never regenerated by the dashboard build."
        )
    raw_rows = _load_json(path)
    if not isinstance(raw_rows, list):
        raise ExploreContentError(
            f"{path} must contain a JSON array of row objects, got {type(raw_rows)}"
        )

    entries: list[PlayerCatalogEntry] = []
    for position, raw in enumerate(raw_rows):
        missing = [k for k in _PLAYERS_CATALOG_REQUIRED_KEYS if k not in raw]
        if missing:
            raise ExploreContentError(f"players.json row {position} missing key(s): {missing}")
        entries.append(
            PlayerCatalogEntry(
                batter_id=int(raw["batter_id"]),
                batter_name=raw.get("batter_name"),
                play_count=int(raw["play_count"]),
            )
        )

    batter_ids = [e.batter_id for e in entries]
    if len(set(batter_ids)) != len(batter_ids):
        counts = Counter(batter_ids)
        dupes = sorted(bid for bid, n in counts.items() if n > 1)
        raise ExploreContentError(f"duplicate batter_id(s) in players.json: {dupes}")

    return entries


def load_player_play_index(players_dir: Path, batter_id: int) -> list[ExploreIndexRow]:
    """Load and validate one batter's `players/<batter_id>.json` play
    index.

    Raises:
        ExploreContentError: if the file is missing/malformed, a row is
            missing a required key, `play_id` is duplicated within this
            file, or a row's `batter_id` does not match `batter_id`.
    """
    path = players_dir / f"{batter_id}.json"
    if not path.exists():
        raise ExploreContentError(
            f"Play Explorer player index not found at {path}. It is committed browser-artifact "
            "data produced by `demo/build_play_explorer_fixture.py` -- it is never regenerated "
            "by the dashboard build."
        )
    raw_rows = _load_json(path)
    if not isinstance(raw_rows, list):
        raise ExploreContentError(
            f"{path} must contain a JSON array of row objects, got {type(raw_rows)}"
        )

    rows = [_validate_player_index_row(r, i, path) for i, r in enumerate(raw_rows)]

    play_ids = [r.play_id for r in rows]
    if len(set(play_ids)) != len(play_ids):
        counts = Counter(play_ids)
        dupes = sorted(pid for pid, n in counts.items() if n > 1)
        raise ExploreContentError(f"duplicate play_id(s) in {path}: {dupes}")

    mismatched = sorted({r.play_id for r in rows if r.batter_id != batter_id})
    if mismatched:
        raise ExploreContentError(
            f"{path} contains row(s) whose batter_id does not match its own filename "
            f"({batter_id}): play_id(s) {mismatched}"
        )

    return rows


def load_explore_catalog(
    players_path: Path, players_dir: Path, games_dir: Path
) -> ExploreLoadedData:
    """Load and cross-validate the full committed Play Explorer fixture:
    `players.json`, every `players/<batter_id>.json` it references, and
    every `games/<game_pk>.json` any of those rows reference.

    This is offline build-time validation -- it loads every shard once to
    prove global invariants hold (no play_id duplicated across batters, no
    referenced game_pk missing its detail file). It is NOT what the browser
    does at runtime: the Explorer only ever fetches `players.json` plus,
    after a hitter is selected, that ONE hitter's `players/<batter_id>.json`
    -- see `dashboard/static/explore.js`'s module docstring.

    Raises:
        ExploreContentError: if any shard is missing/malformed, a
            catalog entry's `play_count` disagrees with its own index
            file's row count, `play_id` is duplicated across two different
            batters' indexes, or any referenced `game_pk` has no
            corresponding `games/<game_pk>.json` file present.
    """
    players = load_players_catalog(players_path)

    all_play_ids: list[str] = []
    game_pks: set[int] = set()
    for entry in players:
        rows = load_player_play_index(players_dir, entry.batter_id)
        if len(rows) != entry.play_count:
            raise ExploreContentError(
                f"players.json play_count={entry.play_count} for batter_id={entry.batter_id} "
                f"does not match the actual players/{entry.batter_id}.json row count "
                f"({len(rows)})"
            )
        all_play_ids.extend(r.play_id for r in rows)
        game_pks.update(r.game_pk for r in rows)

    if len(set(all_play_ids)) != len(all_play_ids):
        counts = Counter(all_play_ids)
        dupes = sorted(pid for pid, n in counts.items() if n > 1)
        raise ExploreContentError(f"duplicate play_id(s) across player indexes: {dupes}")

    missing_game_files = sorted({pk for pk in game_pks if not (games_dir / f"{pk}.json").exists()})
    if missing_game_files:
        raise ExploreContentError(
            f"player indexes reference game_pk(s) with no per-game detail file under "
            f"{games_dir}: {missing_game_files}"
        )

    return ExploreLoadedData(
        players=players,
        players_dir=players_dir,
        games_dir=games_dir,
        total_play_count=len(all_play_ids),
        game_pks=frozenset(game_pks),
    )


def _build_play_detail(raw: dict[str, Any]) -> PlayDetail:
    missing = [k for k in _PLAY_DETAIL_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ExploreContentError(f"play detail record missing key(s): {missing}")

    probabilities = {k: float(raw[k]) for k in _PROBABILITY_KEYS}
    for cls, p in probabilities.items():
        if not isinstance(p, int | float) or p != p or p in (float("inf"), float("-inf")):
            raise ExploreContentError(f"{raw.get('play_id')}: probability {cls!r} not finite")
        if p < 0.0 or p > 1.0:
            raise ExploreContentError(f"{raw.get('play_id')}: probability {cls!r} out of [0, 1]")
    total = sum(probabilities.values())
    if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise ExploreContentError(
            f"{raw.get('play_id')}: probabilities sum to {total!r}, expected ~1.0"
        )

    observed = float(raw["observed_run_value"])
    expected = float(raw["expected_run_value"])
    contact_luck = float(raw["contact_luck_runs"])
    gap = abs((observed - expected) - contact_luck)
    if gap > _RECONCILIATION_TOLERANCE:
        raise ExploreContentError(
            f"{raw.get('play_id')}: observed_run_value - expected_run_value does not "
            f"reconcile with contact_luck_runs (gap={gap!r})"
        )

    return PlayDetail(
        play_id=raw["play_id"],
        game_pk=int(raw["game_pk"]),
        at_bat_number=int(raw["at_bat_number"]),
        pitch_number=int(raw["pitch_number"]),
        game_date=raw["game_date"],
        season=int(raw["season"]),
        batter_id=int(raw["batter_id"]),
        batter_name=raw.get("batter_name"),
        stand=raw["stand"],
        launch_speed=None if raw["launch_speed"] is None else float(raw["launch_speed"]),
        launch_angle=None if raw["launch_angle"] is None else float(raw["launch_angle"]),
        bb_type=raw.get("bb_type"),
        spray_angle_approx=(
            None if raw.get("spray_angle_approx") is None else float(raw["spray_angle_approx"])
        ),
        hit_distance_sc=(
            None if raw.get("hit_distance_sc") is None else float(raw["hit_distance_sc"])
        ),
        outcome_class=raw["outcome_class"],
        observed_run_value=observed,
        p_out=probabilities["p_out"],
        p_single=probabilities["p_single"],
        p_double=probabilities["p_double"],
        p_triple=probabilities["p_triple"],
        p_home_run=probabilities["p_home_run"],
        expected_run_value=expected,
        contact_luck_runs=contact_luck,
    )


def load_play_detail(games_dir: Path, game_pk: int, play_id: str) -> PlayDetail:
    """Load one play's full detail record from its per-game JSON file.

    Raises:
        ExploreContentError: if the per-game file is missing/malformed, or
            `play_id` is not present in it, or the record fails its own
            probability/reconciliation validation.
    """
    path = games_dir / f"{game_pk}.json"
    if not path.exists():
        raise ExploreContentError(f"per-game detail file not found: {path}")
    raw_records = _load_json(path)
    if not isinstance(raw_records, list):
        raise ExploreContentError(f"{path} must contain a JSON array of play records")
    for raw in raw_records:
        if raw.get("play_id") == play_id:
            return _build_play_detail(raw)
    raise ExploreContentError(f"play_id {play_id!r} not found in {path}")
