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
    "SensitivityGrid",
    "ShowcaseRow",
    "load_explore_catalog",
    "load_explore_metadata",
    "load_play_detail",
    "load_player_play_index",
    "load_players_catalog",
    "load_showcase",
    "load_showcase_sensitivity",
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
#: incompatible (it has no players.json to load). Bumped to "3.0" for
#: Version 1.4.1's `showcase.json` addition -- a "2.0" artifact set (no
#: showcase.json, no showcase_*_count metadata) is now correctly rejected
#: as incompatible too.
SUPPORTED_PLAY_LEDGER_VERSION = "2.0"
SUPPORTED_EXPLORER_ARTIFACT_VERSION = "3.0"

_EXPLORE_METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "explorer_artifact_version",
    "play_ledger_version",
    "season",
    "data_through_date",
    "play_count",
    "player_count",
    "game_count",
    "showcase_favorable_count",
    "showcase_unfavorable_count",
    "showcase_interactive_count",
    "source_play_ledger_sha256",
)

_SHOWCASE_ROW_REQUIRED_KEYS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "game_date",
    "batter_id",
    "batter_name",
    "outcome_class",
    "launch_speed",
    "launch_angle",
    "expected_run_value",
    "observed_run_value",
    "contact_luck_runs",
    "group",
    "rank",
    "interactive_available",
)

_SHOWCASE_GROUPS: tuple[str, ...] = ("favorable", "unfavorable")

_SENSITIVITY_ARTIFACT_REQUIRED_KEYS: tuple[str, ...] = (
    "play_id",
    "model_configuration",
    "counterfactual_semantics",
    "fixed_context",
    "original_exit_velocity_mph",
    "original_launch_angle_deg",
    "original_probabilities",
    "original_expected_run_value",
    "original_grid_index",
    "exit_velocity_values",
    "launch_angle_values",
    "grid_shape",
    "grid",
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

#: Mirrors `mlb_luck_score.config.CLASS_ORDER` exactly (duplicated, not
#: imported -- this module must never import `mlb_luck_score`, see module
#: docstring). Used only to key into `showcase-sensitivity/<play_id>.json`'s
#: `original_probabilities`/per-cell `grid` arrays, which -- unlike
#: `PlayDetail`'s `p_out`..`p_home_run` columns -- use the bare class label,
#: matching `demo/build_counterfactual_grid.py`'s own artifact shape.
_SENSITIVITY_CLASS_ORDER: tuple[str, ...] = ("out", "single", "double", "triple", "home_run")


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
    showcase_favorable_count: int
    showcase_unfavorable_count: int
    showcase_interactive_count: int
    source_play_ledger_sha256: str


@dataclass(frozen=True)
class ShowcaseRow:
    play_id: str
    game_pk: int
    game_date: str
    batter_id: int
    batter_name: str | None
    outcome_class: str
    launch_speed: float | None
    launch_angle: float | None
    expected_run_value: float
    observed_run_value: float
    contact_luck_runs: float
    group: str
    rank: int
    interactive_available: bool


@dataclass(frozen=True)
class SensitivityGrid:
    """Loaded, validated `showcase-sensitivity/<play_id>.json` -- the
    per-cell `grid`/axis arrays are exposed verbatim (the browser does the
    direct lookup, this dataclass just proves the file is well-formed and
    internally consistent before it ships -- see `load_showcase_sensitivity`).
    """

    play_id: str
    fixed_context: dict[str, Any]
    original_exit_velocity_mph: float
    original_launch_angle_deg: float
    original_probabilities: dict[str, float]
    original_expected_run_value: float
    original_grid_index: dict[str, int]
    exit_velocity_values: list[float]
    launch_angle_values: list[float]
    grid_shape: dict[str, int]
    grid: list[list[float]]


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
        showcase_favorable_count=int(raw["showcase_favorable_count"]),
        showcase_unfavorable_count=int(raw["showcase_unfavorable_count"]),
        showcase_interactive_count=int(raw["showcase_interactive_count"]),
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


def _validate_showcase_row(raw: dict[str, Any], position: int, source: Path) -> ShowcaseRow:
    missing = [k for k in _SHOWCASE_ROW_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ExploreContentError(f"{source} row {position} missing key(s): {missing}")
    if raw["group"] not in _SHOWCASE_GROUPS:
        raise ExploreContentError(
            f"{source} row {position}: group={raw['group']!r} is not one of {_SHOWCASE_GROUPS}"
        )
    observed = float(raw["observed_run_value"])
    expected = float(raw["expected_run_value"])
    contact_luck = float(raw["contact_luck_runs"])
    gap = abs((observed - expected) - contact_luck)
    if gap > _RECONCILIATION_TOLERANCE:
        raise ExploreContentError(
            f"{source} row {position} (play_id={raw.get('play_id')!r}): observed_run_value - "
            f"expected_run_value does not reconcile with contact_luck_runs (gap={gap!r})"
        )
    return ShowcaseRow(
        play_id=raw["play_id"],
        game_pk=int(raw["game_pk"]),
        game_date=raw["game_date"],
        batter_id=int(raw["batter_id"]),
        batter_name=raw.get("batter_name"),
        outcome_class=raw["outcome_class"],
        launch_speed=None if raw["launch_speed"] is None else float(raw["launch_speed"]),
        launch_angle=None if raw["launch_angle"] is None else float(raw["launch_angle"]),
        expected_run_value=expected,
        observed_run_value=observed,
        contact_luck_runs=contact_luck,
        group=raw["group"],
        rank=int(raw["rank"]),
        interactive_available=bool(raw["interactive_available"]),
    )


def load_showcase(path: Path, metadata: ExploreMetadata) -> list[ShowcaseRow]:
    """Load and validate the committed `showcase.json` (Version 1.4.1
    Showcase Plays): the `SHOWCASE_TOP_N` biggest favorable and biggest
    unfavorable `contact_luck_runs` breaks, no qualification filter, no
    one-play-per-player cap -- see `demo/build_play_explorer_fixture.
    build_showcase_rows`'s docstring for the exact selection/ordering rule
    this validates against.

    Raises:
        ExploreContentError: if the file is missing/malformed, a row is
            missing a required key or fails its own reconciliation check,
            `group` is not `"favorable"`/`"unfavorable"`, `rank` within
            either group is not the contiguous sequence `1..N` in ascending
            `rank` order, `play_id` is duplicated within the showcase, or
            either group's row count disagrees with `metadata`'s own
            `showcase_favorable_count`/`showcase_unfavorable_count`.
    """
    if not path.exists():
        raise ExploreContentError(
            f"Showcase Plays artifact not found at {path}. It is committed browser-artifact "
            "data produced by `demo/build_play_explorer_fixture.py` -- it is never regenerated "
            "by the dashboard build."
        )
    raw_rows = _load_json(path)
    if not isinstance(raw_rows, list):
        raise ExploreContentError(
            f"{path} must contain a JSON array of row objects, got {type(raw_rows)}"
        )

    rows = [_validate_showcase_row(r, i, path) for i, r in enumerate(raw_rows)]

    play_ids = [r.play_id for r in rows]
    if len(set(play_ids)) != len(play_ids):
        counts = Counter(play_ids)
        dupes = sorted(pid for pid, n in counts.items() if n > 1)
        raise ExploreContentError(f"duplicate play_id(s) in {path}: {dupes}")

    expected_counts = {
        "favorable": metadata.showcase_favorable_count,
        "unfavorable": metadata.showcase_unfavorable_count,
    }
    for group in _SHOWCASE_GROUPS:
        group_rows = sorted((r for r in rows if r.group == group), key=lambda r: r.rank)
        if len(group_rows) != expected_counts[group]:
            raise ExploreContentError(
                f"{path}: group={group!r} has {len(group_rows)} row(s), but "
                f"explore-metadata.json declares {expected_counts[group]}"
            )
        actual_ranks = [r.rank for r in group_rows]
        expected_ranks = list(range(1, len(group_rows) + 1))
        if actual_ranks != expected_ranks:
            raise ExploreContentError(
                f"{path}: group={group!r} ranks are {actual_ranks}, expected a contiguous "
                f"{expected_ranks}"
            )

    return rows


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


def load_showcase_sensitivity(path: Path) -> SensitivityGrid:
    """Load and structurally validate one `showcase-sensitivity/
    <play_id>.json` "What if?" grid (Version 1.4.1) -- produced by `demo/
    build_showcase_sensitivity.py`, which owns the SCIENTIFIC reconciliation
    gate (this dashboard module never imports it or re-runs it -- see
    module docstring's read-only boundary). What this function DOES check,
    as a build-time self-consistency guard: the file has every required
    key, `grid`'s length matches `grid_shape.n_ev * grid_shape.n_la`, both
    axis arrays are non-empty, `original_grid_index` is within bounds, AND
    -- exactly mirroring `demo/build_counterfactual_grid.py`'s own
    "one full-precision cell per reference play" invariant -- a direct
    lookup into `grid` at `original_grid_index` reproduces
    `original_probabilities` exactly. A corrupted or hand-edited artifact
    that fails any of these is rejected here, before it ever ships.

    Raises:
        ExploreContentError: if the file is missing/malformed, missing a
            required key, has an inconsistent grid shape, or its own
            reference-coordinate cell does not exactly match
            `original_probabilities`.
    """
    if not path.exists():
        raise ExploreContentError(f"showcase sensitivity artifact not found: {path}")
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ExploreContentError(f"{path} must contain a JSON object, got {type(raw)}")

    missing = [k for k in _SENSITIVITY_ARTIFACT_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ExploreContentError(f"{path} missing key(s): {missing}")

    ev_values = raw["exit_velocity_values"]
    la_values = raw["launch_angle_values"]
    grid = raw["grid"]
    grid_shape = raw["grid_shape"]
    if not ev_values or not la_values:
        raise ExploreContentError(f"{path}: exit_velocity_values/launch_angle_values must be non-empty")
    if grid_shape["n_ev"] != len(ev_values) or grid_shape["n_la"] != len(la_values):
        raise ExploreContentError(
            f"{path}: grid_shape {grid_shape!r} does not match axis array lengths "
            f"(n_ev={len(ev_values)}, n_la={len(la_values)})"
        )
    if len(grid) != len(ev_values) * len(la_values):
        raise ExploreContentError(
            f"{path}: grid has {len(grid)} row(s), expected "
            f"{len(ev_values)} * {len(la_values)} = {len(ev_values) * len(la_values)}"
        )

    original_grid_index = raw["original_grid_index"]
    ev_index = int(original_grid_index["ev_index"])
    la_index = int(original_grid_index["la_index"])
    if not (0 <= ev_index < len(ev_values)) or not (0 <= la_index < len(la_values)):
        raise ExploreContentError(
            f"{path}: original_grid_index {original_grid_index!r} is out of bounds for a "
            f"{len(ev_values)}x{len(la_values)} grid"
        )

    row_index = ev_index * len(la_values) + la_index
    cell = grid[row_index]
    original_probabilities = raw["original_probabilities"]
    for i, cls in enumerate(_SENSITIVITY_CLASS_ORDER):
        gap = abs(float(cell[i]) - float(original_probabilities[cls]))
        if gap > _RECONCILIATION_TOLERANCE:
            raise ExploreContentError(
                f"{path}: direct grid lookup at original_grid_index does not reproduce "
                f"original_probabilities[{cls!r}] exactly (gap={gap!r})"
            )

    return SensitivityGrid(
        play_id=raw["play_id"],
        fixed_context=raw["fixed_context"],
        original_exit_velocity_mph=float(raw["original_exit_velocity_mph"]),
        original_launch_angle_deg=float(raw["original_launch_angle_deg"]),
        original_probabilities={k: float(v) for k, v in original_probabilities.items()},
        original_expected_run_value=float(raw["original_expected_run_value"]),
        original_grid_index=original_grid_index,
        exit_velocity_values=ev_values,
        launch_angle_values=la_values,
        grid_shape=grid_shape,
        grid=grid,
    )
