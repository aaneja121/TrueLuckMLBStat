"""Contact Luck v1.4.0 (Phase 4.2) dashboard: `explore_content.py`
view-model tests.

Uses small synthetic fixture files (never the real committed Play Explorer
fixture -- that's covered end-to-end by `tests/test_dashboard_explore_build.
py`) to exercise `load_players_catalog`/`load_player_play_index`/
`load_explore_catalog`/`load_play_detail`'s validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import explore_content as ec
import pytest


def _catalog_entry(**overrides: Any) -> dict[str, Any]:
    row = {"batter_id": 12345, "batter_name": "Test Player", "play_count": 1}
    row.update(overrides)
    return row


def _index_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "game_date": "2024-06-01",
        "batter_id": 12345,
        "batter_name": "Test Player",
        "outcome_class": "home_run",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "expected_run_value": 0.17,
        "contact_luck_runs": 1.23,
    }
    row.update(overrides)
    return row


def _explore_metadata(**overrides: Any) -> dict[str, Any]:
    row = {
        "explorer_artifact_version": ec.SUPPORTED_EXPLORER_ARTIFACT_VERSION,
        "play_ledger_version": ec.SUPPORTED_PLAY_LEDGER_VERSION,
        "season": 2024,
        "data_through_date": None,
        "play_count": 1,
        "player_count": 1,
        "game_count": 1,
        "showcase_favorable_count": 0,
        "showcase_unfavorable_count": 0,
        "showcase_interactive_count": 0,
        "source_play_ledger_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def _write_explore_metadata(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "explore-metadata.json"
    path.write_text(json.dumps(_explore_metadata(**overrides)))
    return path


def _play_detail(**overrides: Any) -> dict[str, Any]:
    row = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "at_bat_number": 10,
        "pitch_number": 3,
        "game_date": "2024-06-01",
        "season": 2024,
        "batter_id": 12345,
        "batter_name": "Test Player",
        "stand": "L",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "outcome_class": "home_run",
        "observed_run_value": 1.4,
        "p_out": 0.05,
        "p_single": 0.05,
        "p_double": 0.05,
        "p_triple": 0.05,
        "p_home_run": 0.8,
        "expected_run_value": 0.17,
        "contact_luck_runs": 1.23,
    }
    row.update(overrides)
    return row


def _write_players_json(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "players.json"
    path.write_text(json.dumps(rows))
    return path


def _write_player_index_file(players_dir: Path, batter_id: int, rows: list[dict[str, Any]]) -> None:
    players_dir.mkdir(parents=True, exist_ok=True)
    (players_dir / f"{batter_id}.json").write_text(json.dumps(rows))


def _write_game_file(games_dir: Path, game_pk: int, records: list[dict[str, Any]]) -> None:
    games_dir.mkdir(parents=True, exist_ok=True)
    (games_dir / f"{game_pk}.json").write_text(json.dumps(records))


# ---------------------------------------------------------------------------
# load_players_catalog
# ---------------------------------------------------------------------------


def test_valid_players_catalog_loads(tmp_path: Path):
    path = _write_players_json(tmp_path, [_catalog_entry()])
    entries = ec.load_players_catalog(path)
    assert len(entries) == 1
    assert entries[0].batter_id == 12345
    assert entries[0].batter_name == "Test Player"
    assert entries[0].play_count == 1


def test_missing_players_catalog_raises(tmp_path: Path):
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_players_catalog(tmp_path / "missing.json")


def test_players_catalog_must_be_a_json_array(tmp_path: Path):
    path = tmp_path / "players.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ec.ExploreContentError, match="JSON array"):
        ec.load_players_catalog(path)


def test_players_catalog_row_missing_key_raises(tmp_path: Path):
    bad_row = _catalog_entry()
    del bad_row["play_count"]
    path = _write_players_json(tmp_path, [bad_row])
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_players_catalog(path)


def test_duplicate_batter_id_in_catalog_raises(tmp_path: Path):
    path = _write_players_json(tmp_path, [_catalog_entry(), _catalog_entry()])
    with pytest.raises(ec.ExploreContentError, match="duplicate batter_id"):
        ec.load_players_catalog(path)


def test_players_catalog_null_batter_name_allowed(tmp_path: Path):
    path = _write_players_json(tmp_path, [_catalog_entry(batter_name=None)])
    entries = ec.load_players_catalog(path)
    assert entries[0].batter_name is None


# ---------------------------------------------------------------------------
# load_player_play_index
# ---------------------------------------------------------------------------


def test_valid_player_play_index_loads(tmp_path: Path):
    players_dir = tmp_path / "players"
    _write_player_index_file(players_dir, 12345, [_index_row()])
    rows = ec.load_player_play_index(players_dir, 12345)
    assert len(rows) == 1
    assert rows[0].play_id == "700001-10-3"
    assert rows[0].batter_name == "Test Player"


def test_missing_player_index_file_raises(tmp_path: Path):
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_player_play_index(tmp_path / "players", 999999)


def test_player_index_must_be_a_json_array(tmp_path: Path):
    players_dir = tmp_path / "players"
    players_dir.mkdir(parents=True)
    (players_dir / "12345.json").write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ec.ExploreContentError, match="JSON array"):
        ec.load_player_play_index(players_dir, 12345)


def test_player_index_row_missing_key_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    bad_row = _index_row()
    del bad_row["outcome_class"]
    _write_player_index_file(players_dir, 12345, [bad_row])
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_player_play_index(players_dir, 12345)


def test_duplicate_play_id_within_player_index_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    _write_player_index_file(players_dir, 12345, [_index_row(), _index_row()])
    with pytest.raises(ec.ExploreContentError, match="duplicate play_id"):
        ec.load_player_play_index(players_dir, 12345)


def test_player_index_row_batter_id_mismatch_raises(tmp_path: Path):
    """A row inside players/12345.json whose own batter_id field disagrees
    with the filename it lives in is a malformed/misplaced shard -- never
    silently trusted."""
    players_dir = tmp_path / "players"
    _write_player_index_file(players_dir, 12345, [_index_row(batter_id=999)])
    with pytest.raises(ec.ExploreContentError, match="batter_id"):
        ec.load_player_play_index(players_dir, 12345)


def test_player_index_row_includes_expected_run_value(tmp_path: Path):
    players_dir = tmp_path / "players"
    _write_player_index_file(players_dir, 12345, [_index_row()])
    rows = ec.load_player_play_index(players_dir, 12345)
    assert rows[0].expected_run_value == pytest.approx(0.17)


def test_player_index_row_missing_expected_run_value_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    bad_row = _index_row()
    del bad_row["expected_run_value"]
    _write_player_index_file(players_dir, 12345, [bad_row])
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_player_play_index(players_dir, 12345)


# ---------------------------------------------------------------------------
# load_explore_catalog (full cross-shard validation)
# ---------------------------------------------------------------------------


def test_load_explore_catalog_valid(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    players_path = _write_players_json(tmp_path, [_catalog_entry()])
    _write_player_index_file(players_dir, 12345, [_index_row()])
    _write_game_file(games_dir, 700001, [_play_detail()])

    data = ec.load_explore_catalog(players_path, players_dir, games_dir)
    assert len(data.players) == 1
    assert data.total_play_count == 1
    assert data.game_pks == frozenset({700001})


def test_load_explore_catalog_play_count_mismatch_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    players_path = _write_players_json(tmp_path, [_catalog_entry(play_count=5)])
    _write_player_index_file(players_dir, 12345, [_index_row()])
    _write_game_file(games_dir, 700001, [_play_detail()])

    with pytest.raises(ec.ExploreContentError, match="play_count"):
        ec.load_explore_catalog(players_path, players_dir, games_dir)


def test_load_explore_catalog_duplicate_play_id_across_players_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    players_path = _write_players_json(
        tmp_path, [_catalog_entry(batter_id=1), _catalog_entry(batter_id=2)]
    )
    # Same play_id copy-pasted into two different batters' indexes.
    _write_player_index_file(players_dir, 1, [_index_row(batter_id=1)])
    _write_player_index_file(players_dir, 2, [_index_row(batter_id=2, play_id="700001-10-3")])
    _write_game_file(games_dir, 700001, [_play_detail()])

    with pytest.raises(ec.ExploreContentError, match="duplicate play_id"):
        ec.load_explore_catalog(players_path, players_dir, games_dir)


def test_load_explore_catalog_missing_game_file_for_referenced_game_pk_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    games_dir.mkdir(parents=True)
    players_path = _write_players_json(tmp_path, [_catalog_entry()])
    _write_player_index_file(players_dir, 12345, [_index_row()])
    # No games/700001.json written.

    with pytest.raises(ec.ExploreContentError, match="no per-game detail file"):
        ec.load_explore_catalog(players_path, players_dir, games_dir)


def test_load_explore_catalog_missing_player_file_raises(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    players_dir.mkdir(parents=True)
    players_path = _write_players_json(tmp_path, [_catalog_entry()])
    # No players/12345.json written.

    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_explore_catalog(players_path, players_dir, games_dir)


def test_load_explore_catalog_total_play_count_sums_across_players(tmp_path: Path):
    players_dir = tmp_path / "players"
    games_dir = tmp_path / "games"
    players_path = _write_players_json(
        tmp_path,
        [_catalog_entry(batter_id=1, play_count=2), _catalog_entry(batter_id=2, play_count=1)],
    )
    _write_player_index_file(
        players_dir,
        1,
        [
            _index_row(batter_id=1, play_id="700001-1-1"),
            _index_row(batter_id=1, play_id="700001-2-1"),
        ],
    )
    _write_player_index_file(
        players_dir, 2, [_index_row(batter_id=2, play_id="700002-1-1", game_pk=700002)]
    )
    _write_game_file(
        games_dir, 700001, [_play_detail(play_id="700001-1-1"), _play_detail(play_id="700001-2-1")]
    )
    _write_game_file(games_dir, 700002, [_play_detail(play_id="700002-1-1", game_pk=700002)])

    data = ec.load_explore_catalog(players_path, players_dir, games_dir)
    assert data.total_play_count == 3
    assert data.game_pks == frozenset({700001, 700002})


# ---------------------------------------------------------------------------
# load_play_detail
# ---------------------------------------------------------------------------


def test_valid_play_detail_loads(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(games_dir, 700001, [_play_detail()])
    detail = ec.load_play_detail(games_dir, 700001, "700001-10-3")
    assert detail.play_id == "700001-10-3"
    assert detail.batter_name == "Test Player"
    assert detail.outcome_class == "home_run"
    assert detail.contact_luck_runs == pytest.approx(1.23)


def test_play_detail_missing_game_file_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    games_dir.mkdir(parents=True)
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_play_detail(games_dir, 999999, "999999-1-1")


def test_play_detail_unknown_play_id_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(games_dir, 700001, [_play_detail()])
    with pytest.raises(ec.ExploreContentError, match="not found in"):
        ec.load_play_detail(games_dir, 700001, "700001-99-9")


def test_play_detail_missing_key_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    bad = _play_detail()
    del bad["stand"]
    _write_game_file(games_dir, 700001, [bad])
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_play_detail(games_dir, 700001, "700001-10-3")


def test_play_detail_probability_out_of_range_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(games_dir, 700001, [_play_detail(p_out=1.5)])
    with pytest.raises(ec.ExploreContentError, match=r"\[0, 1\]"):
        ec.load_play_detail(games_dir, 700001, "700001-10-3")


def test_play_detail_probabilities_not_summing_to_one_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(
        games_dir,
        700001,
        [_play_detail(p_out=0.9, p_single=0.9, p_double=0.0, p_triple=0.0, p_home_run=0.0)],
    )
    with pytest.raises(ec.ExploreContentError, match="sum to"):
        ec.load_play_detail(games_dir, 700001, "700001-10-3")


def test_play_detail_reconciliation_mismatch_raises(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(games_dir, 700001, [_play_detail(contact_luck_runs=999.0)])
    with pytest.raises(ec.ExploreContentError, match="does not.*reconcile"):
        ec.load_play_detail(games_dir, 700001, "700001-10-3")


def test_play_detail_null_batter_name_allowed(tmp_path: Path):
    games_dir = tmp_path / "games"
    _write_game_file(games_dir, 700001, [_play_detail(batter_name=None)])
    detail = ec.load_play_detail(games_dir, 700001, "700001-10-3")
    assert detail.batter_name is None


# ---------------------------------------------------------------------------
# load_explore_metadata
# ---------------------------------------------------------------------------


def test_valid_explore_metadata_loads(tmp_path: Path):
    path = _write_explore_metadata(tmp_path)
    metadata = ec.load_explore_metadata(path)
    assert metadata.explorer_artifact_version == ec.SUPPORTED_EXPLORER_ARTIFACT_VERSION
    assert metadata.play_ledger_version == ec.SUPPORTED_PLAY_LEDGER_VERSION
    assert metadata.play_count == 1
    assert metadata.player_count == 1
    assert metadata.game_count == 1
    assert metadata.data_through_date is None
    assert metadata.source_play_ledger_sha256 == "a" * 64


def test_explore_metadata_missing_file_raises(tmp_path: Path):
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_explore_metadata(tmp_path / "missing-explore-metadata.json")


def test_explore_metadata_not_a_dict_raises(tmp_path: Path):
    path = tmp_path / "explore-metadata.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ec.ExploreContentError, match="JSON object"):
        ec.load_explore_metadata(path)


def test_explore_metadata_missing_required_key_raises(tmp_path: Path):
    data = _explore_metadata()
    del data["source_play_ledger_sha256"]
    path = tmp_path / "explore-metadata.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_explore_metadata(path)


def test_explore_metadata_missing_player_count_raises(tmp_path: Path):
    data = _explore_metadata()
    del data["player_count"]
    path = tmp_path / "explore-metadata.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_explore_metadata(path)


def test_explore_metadata_wrong_play_ledger_version_raises(tmp_path: Path):
    path = _write_explore_metadata(tmp_path, play_ledger_version="1.0")
    with pytest.raises(ec.ExploreContentError, match="play_ledger_version"):
        ec.load_explore_metadata(path)


def test_explore_metadata_wrong_explorer_artifact_version_raises(tmp_path: Path):
    path = _write_explore_metadata(tmp_path, explorer_artifact_version="99.0")
    with pytest.raises(ec.ExploreContentError, match="explorer_artifact_version"):
        ec.load_explore_metadata(path)


def test_explore_metadata_phase_4_1_v1_0_artifact_rejected(tmp_path: Path):
    """The Phase 4.1 monolithic-search-index fixture's explorer_artifact_
    version ("1.0") must be rejected by this Phase 4.2 dashboard build --
    it has no players.json/players/<id>.json to load."""
    path = _write_explore_metadata(tmp_path, explorer_artifact_version="1.0")
    with pytest.raises(ec.ExploreContentError, match="explorer_artifact_version"):
        ec.load_explore_metadata(path)


# ---------------------------------------------------------------------------
# Version 1.4.1: load_showcase
# ---------------------------------------------------------------------------


def _showcase_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "game_date": "2024-06-01",
        "batter_id": 12345,
        "batter_name": "Test Player",
        "outcome_class": "home_run",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "expected_run_value": 0.17,
        "observed_run_value": 1.4,
        "contact_luck_runs": 1.23,
        "group": "favorable",
        "rank": 1,
        "interactive_available": False,
    }
    row.update(overrides)
    return row


def _write_showcase(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "showcase.json"
    path.write_text(json.dumps(rows))
    return path


def _metadata_for_showcase(
    favorable: int, unfavorable: int, interactive: int = 0
) -> ec.ExploreMetadata:
    raw = _explore_metadata(
        showcase_favorable_count=favorable,
        showcase_unfavorable_count=unfavorable,
        showcase_interactive_count=interactive,
    )
    return ec.ExploreMetadata(
        explorer_artifact_version=raw["explorer_artifact_version"],
        play_ledger_version=raw["play_ledger_version"],
        season=raw["season"],
        data_through_date=raw["data_through_date"],
        play_count=raw["play_count"],
        player_count=raw["player_count"],
        game_count=raw["game_count"],
        showcase_favorable_count=favorable,
        showcase_unfavorable_count=unfavorable,
        showcase_interactive_count=interactive,
        source_play_ledger_sha256=raw["source_play_ledger_sha256"],
    )


def test_load_showcase_valid_rows(tmp_path: Path) -> None:
    rows = [_showcase_row(play_id="700001-1-1", rank=1)]
    path = _write_showcase(tmp_path, rows)
    metadata = _metadata_for_showcase(favorable=1, unfavorable=0)
    loaded = ec.load_showcase(path, metadata)
    assert len(loaded) == 1
    assert loaded[0].play_id == "700001-1-1"
    assert loaded[0].group == "favorable"


def test_load_showcase_missing_file_raises(tmp_path: Path) -> None:
    metadata = _metadata_for_showcase(favorable=0, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_showcase(tmp_path / "does_not_exist.json", metadata)


def test_load_showcase_missing_key_raises(tmp_path: Path) -> None:
    row = _showcase_row()
    del row["contact_luck_runs"]
    path = _write_showcase(tmp_path, [row])
    metadata = _metadata_for_showcase(favorable=1, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_showcase(path, metadata)


def test_load_showcase_invalid_group_raises(tmp_path: Path) -> None:
    path = _write_showcase(tmp_path, [_showcase_row(group="sideways")])
    metadata = _metadata_for_showcase(favorable=1, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="group"):
        ec.load_showcase(path, metadata)


def test_load_showcase_reconciliation_failure_raises(tmp_path: Path) -> None:
    path = _write_showcase(
        tmp_path,
        [_showcase_row(observed_run_value=1.4, expected_run_value=0.17, contact_luck_runs=99.0)],
    )
    metadata = _metadata_for_showcase(favorable=1, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="reconcile"):
        ec.load_showcase(path, metadata)


def test_load_showcase_duplicate_play_id_raises(tmp_path: Path) -> None:
    rows = [
        _showcase_row(play_id="700001-1-1", rank=1),
        _showcase_row(play_id="700001-1-1", rank=2, group="unfavorable"),
    ]
    path = _write_showcase(tmp_path, rows)
    metadata = _metadata_for_showcase(favorable=1, unfavorable=1)
    with pytest.raises(ec.ExploreContentError, match="duplicate play_id"):
        ec.load_showcase(path, metadata)


def test_load_showcase_favorable_count_mismatch_raises(tmp_path: Path) -> None:
    path = _write_showcase(tmp_path, [_showcase_row(play_id="700001-1-1", rank=1)])
    metadata = _metadata_for_showcase(favorable=2, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="favorable"):
        ec.load_showcase(path, metadata)


def test_load_showcase_non_contiguous_ranks_raise(tmp_path: Path) -> None:
    rows = [
        _showcase_row(play_id="700001-1-1", rank=1),
        _showcase_row(play_id="700001-2-1", rank=3),
    ]
    path = _write_showcase(tmp_path, rows)
    metadata = _metadata_for_showcase(favorable=2, unfavorable=0)
    with pytest.raises(ec.ExploreContentError, match="ranks"):
        ec.load_showcase(path, metadata)


def test_load_showcase_batter_name_none_when_unresolved(tmp_path: Path) -> None:
    path = _write_showcase(tmp_path, [_showcase_row(batter_name=None)])
    metadata = _metadata_for_showcase(favorable=1, unfavorable=0)
    loaded = ec.load_showcase(path, metadata)
    assert loaded[0].batter_name is None


# ---------------------------------------------------------------------------
# Version 1.4.1: load_showcase_sensitivity
# ---------------------------------------------------------------------------


def _sensitivity_artifact(**overrides: Any) -> dict[str, Any]:
    # A tiny 2x2 grid: real coordinate at (ev_index=0, la_index=0).
    grid = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]
    payload = {
        "play_id": "700001-10-3",
        "model_configuration": {"run_value_table": {}},
        "counterfactual_semantics": {"type": "ceteris_paribus_model_sensitivity"},
        "fixed_context": {
            "hit_distance_sc": 413.0,
            "spray_angle_approx": -5.9,
            "bb_type": "fly_ball",
            "stand": "L",
        },
        "original_exit_velocity_mph": 107.6,
        "original_launch_angle_deg": 33.0,
        "original_probabilities": {
            "out": 1.0,
            "single": 0.0,
            "double": 0.0,
            "triple": 0.0,
            "home_run": 0.0,
        },
        "original_expected_run_value": -0.25,
        "original_grid_index": {"ev_index": 0, "la_index": 0},
        "exit_velocity_values": [107.6, 110],
        "launch_angle_values": [33.0, 40],
        "grid_shape": {"n_ev": 2, "n_la": 2},
        "grid": grid,
    }
    payload.update(overrides)
    return payload


def _write_sensitivity(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / f"{payload['play_id']}.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_showcase_sensitivity_valid(tmp_path: Path) -> None:
    path = _write_sensitivity(tmp_path, _sensitivity_artifact())
    grid = ec.load_showcase_sensitivity(path)
    assert grid.play_id == "700001-10-3"
    assert grid.grid_shape == {"n_ev": 2, "n_la": 2}
    assert grid.original_probabilities["out"] == 1.0


def test_load_showcase_sensitivity_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ec.ExploreContentError, match="not found"):
        ec.load_showcase_sensitivity(tmp_path / "does_not_exist.json")


def test_load_showcase_sensitivity_missing_key_raises(tmp_path: Path) -> None:
    payload = _sensitivity_artifact()
    del payload["grid_shape"]
    path = _write_sensitivity(tmp_path, payload)
    with pytest.raises(ec.ExploreContentError, match="missing key"):
        ec.load_showcase_sensitivity(path)


def test_load_showcase_sensitivity_grid_shape_mismatch_raises(tmp_path: Path) -> None:
    payload = _sensitivity_artifact(grid_shape={"n_ev": 3, "n_la": 2})
    path = _write_sensitivity(tmp_path, payload)
    with pytest.raises(ec.ExploreContentError, match="grid_shape"):
        ec.load_showcase_sensitivity(path)


def test_load_showcase_sensitivity_wrong_grid_length_raises(tmp_path: Path) -> None:
    payload = _sensitivity_artifact()
    payload["grid"] = payload["grid"][:3]  # drop one row
    path = _write_sensitivity(tmp_path, payload)
    with pytest.raises(ec.ExploreContentError, match="grid has"):
        ec.load_showcase_sensitivity(path)


def test_load_showcase_sensitivity_out_of_bounds_index_raises(tmp_path: Path) -> None:
    payload = _sensitivity_artifact(original_grid_index={"ev_index": 9, "la_index": 0})
    path = _write_sensitivity(tmp_path, payload)
    with pytest.raises(ec.ExploreContentError, match="out of bounds"):
        ec.load_showcase_sensitivity(path)


def test_load_showcase_sensitivity_reference_cell_mismatch_raises(tmp_path: Path) -> None:
    """The direct grid lookup at original_grid_index must reproduce
    original_probabilities exactly -- a corrupted/hand-edited grid cell at
    that position is rejected."""
    payload = _sensitivity_artifact()
    payload["grid"][0] = [0.9, 0.1, 0.0, 0.0, 0.0]  # no longer matches original_probabilities
    path = _write_sensitivity(tmp_path, payload)
    with pytest.raises(ec.ExploreContentError, match="does not reproduce"):
        ec.load_showcase_sensitivity(path)
