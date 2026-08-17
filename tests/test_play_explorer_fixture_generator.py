"""Version 1.4.0 Phase 4 (revised Phase 4.2): tests for `demo/
build_play_explorer_fixture.py`, the projection-only Play Explorer
browser-artifact generator.

Pure synthetic hand-built ledgers -- no model training, no real data, no
network. Covers: fail-closed version rejection (v1.0, missing, unknown
future), deterministic output, players-catalog/per-player-index row
identity, per-game partitioning correctness, canonical field passthrough,
and the structural "no model/scoring imports" guarantee (mirroring
`tests/test_dashboard_isolation.py`'s pattern for `dashboard/`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import build_play_explorer_fixture as gen
import pandas as pd
import pytest

from mlb_luck_score.scoring.play_ledger_metadata import build_play_ledger_metadata
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_COLUMNS, PLAY_LEDGER_VERSION

_OBSERVED = -0.25491574127584554
_EXPECTED = 1.290656107968476


def _resolved_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "at_bat_number": 10,
        "pitch_number": 3,
        "game_date": "2024-06-01",
        "season": 2024,
        "batter_id": 12345,
        "stand": "L",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": _OBSERVED,
        "p_out": 0.05692444081479338,
        "p_single": 0.0020191686532899677,
        "p_double": 0.019210357764228986,
        "p_triple": 0.003476039197145578,
        "p_home_run": 0.9183699935705423,
        "expected_run_value": _EXPECTED,
    }
    row["contact_luck_runs"] = row["observed_run_value"] - row["expected_run_value"]
    row.update(overrides)
    return row


def _unresolved_row(**overrides: Any) -> dict[str, Any]:
    row = _resolved_row(
        play_id="700001-11-1",
        is_scored=False,
        outcome_class=pd.NA,
        observed_run_value=pd.NA,
        contact_luck_runs=pd.NA,
        p_out=pd.NA,
        p_single=pd.NA,
        p_double=pd.NA,
        p_triple=pd.NA,
        p_home_run=pd.NA,
        expected_run_value=pd.NA,
    )
    row.update(overrides)
    return row


def _df(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[list(PLAY_LEDGER_COLUMNS)]


def _write_ledger(
    tmp_path: Path, df: pd.DataFrame, *, version: str | None = PLAY_LEDGER_VERSION
) -> tuple[Path, Path]:
    ledger_path = tmp_path / "play_ledger.parquet"
    df.to_parquet(ledger_path, index=False)
    metadata = build_play_ledger_metadata(
        df, season=2024, score_version="0.2", model_versions={"contact": "x"}
    )
    if version is None:
        del metadata["play_ledger_version"]
    else:
        metadata["play_ledger_version"] = version
    metadata_path = tmp_path / "play_ledger_metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    return ledger_path, metadata_path


# ---------------------------------------------------------------------------
# Version safety: fail closed
# ---------------------------------------------------------------------------


def test_v2_ledger_accepted(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df, version="2.0")
    out_dir = tmp_path / "out"
    result = gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    assert result["play_ledger_version"] == "2.0"
    assert (out_dir / "players.json").exists()


def test_v1_ledger_rejected(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df, version="1.0")
    with pytest.raises(gen.IncompatiblePlayLedgerVersionError, match="1.0"):
        gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=tmp_path / "out",
        )


def test_missing_ledger_version_rejected(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df, version=None)
    with pytest.raises(gen.IncompatiblePlayLedgerVersionError, match="None"):
        gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=tmp_path / "out",
        )


def test_unknown_future_version_rejected(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df, version="3.7")
    with pytest.raises(gen.IncompatiblePlayLedgerVersionError, match="3.7"):
        gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=tmp_path / "out",
        )


def test_require_compatible_play_ledger_version_rejects_wrong_type():
    with pytest.raises(gen.IncompatiblePlayLedgerVersionError):
        gen.require_compatible_play_ledger_version({"play_ledger_version": 2.0})  # float, not "2.0"


def test_require_compatible_play_ledger_version_accepts_current():
    gen.require_compatible_play_ledger_version({"play_ledger_version": PLAY_LEDGER_VERSION})


def test_explorer_artifact_version_is_3_0():
    """Version 1.4.1 bumped the STRUCTURE version again -- showcase.json
    added -- independent of PLAY_LEDGER_VERSION."""
    assert gen.EXPLORER_ARTIFACT_VERSION == "3.0"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generated_artifacts_are_byte_deterministic(tmp_path: Path):
    df = _df(_resolved_row(), _unresolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    names_path = tmp_path / "names.json"
    names_path.write_text(json.dumps({"12345": "Test Player"}))

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path,
        play_ledger_metadata_path=metadata_path,
        output_dir=out_a,
        names_path=names_path,
    )
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path,
        play_ledger_metadata_path=metadata_path,
        output_dir=out_b,
        names_path=names_path,
    )
    assert (out_a / "players.json").read_bytes() == (out_b / "players.json").read_bytes()
    assert (out_a / "players" / "12345.json").read_bytes() == (
        out_b / "players" / "12345.json"
    ).read_bytes()
    assert (out_a / "games" / "700001.json").read_bytes() == (
        out_b / "games" / "700001.json"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Players catalog / per-player index row identity and partitioning
# ---------------------------------------------------------------------------


def test_players_catalog_row_identity_preserved(tmp_path: Path):
    df = _df(_resolved_row())
    rows = gen.build_players_catalog_rows(df, {12345: "Test Player"})
    assert len(rows) == 1
    row = rows[0]
    assert row["batter_id"] == 12345
    assert row["batter_name"] == "Test Player"
    assert row["play_count"] == 1


def test_players_catalog_one_row_per_represented_batter(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="700001-1-1", batter_id=1),
        _resolved_row(play_id="700001-2-1", batter_id=1),
        _resolved_row(play_id="700001-3-1", batter_id=2),
    )
    rows = gen.build_players_catalog_rows(df, {})
    by_id = {r["batter_id"]: r for r in rows}
    assert set(by_id) == {1, 2}
    assert by_id[1]["play_count"] == 2
    assert by_id[2]["play_count"] == 1


def test_players_catalog_sorted_by_batter_id(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="700001-1-1", batter_id=300),
        _resolved_row(play_id="700001-2-1", batter_id=100),
        _resolved_row(play_id="700001-3-1", batter_id=200),
    )
    rows = gen.build_players_catalog_rows(df, {})
    assert [r["batter_id"] for r in rows] == [100, 200, 300]


def test_players_catalog_excludes_batters_with_no_scored_plays(tmp_path: Path):
    df = _df(_resolved_row(batter_id=1), _unresolved_row(batter_id=2))
    rows = gen.build_players_catalog_rows(df, {})
    assert [r["batter_id"] for r in rows] == [1]


def test_player_index_fields_match_documented_schema():
    fields = list(gen.PLAYER_INDEX_FIELDS)
    assert fields == [
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
    ]


def test_player_index_row_identity_preserved(tmp_path: Path):
    """Every field in a per-player index row must equal the SAME field on
    the source canonical ledger row -- verbatim, no transformation beyond
    the documented batter_name overlay attachment."""
    df = _df(_resolved_row())
    partitions = gen.build_player_play_index_rows(df, {12345: "Test Player"})
    rows = partitions["12345"]
    assert len(rows) == 1
    row = rows[0]
    source = df.iloc[0]
    assert row["play_id"] == source["play_id"]
    assert row["game_pk"] == int(source["game_pk"])
    assert row["game_date"] == source["game_date"]
    assert row["batter_id"] == int(source["batter_id"])
    assert row["batter_name"] == "Test Player"
    assert row["outcome_class"] == source["outcome_class"]
    assert row["expected_run_value"] == pytest.approx(float(source["expected_run_value"]))
    assert row["contact_luck_runs"] == pytest.approx(float(source["contact_luck_runs"]))
    assert row["launch_speed"] == pytest.approx(float(source["launch_speed"]))
    assert row["launch_angle"] == pytest.approx(float(source["launch_angle"]))


def test_player_index_partitions_contain_only_that_batters_plays(tmp_path: Path):
    row_batter_a_1 = _resolved_row(play_id="800001-1-1", game_pk=800001, batter_id=1)
    row_batter_a_2 = _resolved_row(play_id="800001-2-1", game_pk=800001, batter_id=1)
    row_batter_b_1 = _resolved_row(play_id="800002-1-1", game_pk=800002, batter_id=2)
    df = _df(row_batter_a_1, row_batter_a_2, row_batter_b_1)
    partitions = gen.build_player_play_index_rows(df, {})

    assert set(partitions.keys()) == {"1", "2"}
    assert {r["play_id"] for r in partitions["1"]} == {"800001-1-1", "800001-2-1"}
    assert all(r["batter_id"] == 1 for r in partitions["1"])
    assert {r["play_id"] for r in partitions["2"]} == {"800002-1-1"}
    assert all(r["batter_id"] == 2 for r in partitions["2"])


def test_player_index_rows_sorted_by_date_game_play(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="700003-1-1", game_pk=700003, game_date="2024-06-03", batter_id=1),
        _resolved_row(play_id="700001-2-1", game_pk=700001, game_date="2024-06-01", batter_id=1),
        _resolved_row(play_id="700002-1-1", game_pk=700002, game_date="2024-06-02", batter_id=1),
    )
    partitions = gen.build_player_play_index_rows(df, {})
    dates = [r["game_date"] for r in partitions["1"]]
    assert dates == sorted(dates)


def test_unresolved_rows_excluded_from_players_catalog_and_index(tmp_path: Path):
    df = _df(_resolved_row(), _unresolved_row())
    catalog = gen.build_players_catalog_rows(df, {})
    assert sum(r["play_count"] for r in catalog) == 1
    partitions = gen.build_player_play_index_rows(df, {})
    all_play_ids = [r["play_id"] for records in partitions.values() for r in records]
    assert all_play_ids == ["700001-10-3"]


def test_unresolved_rows_excluded_from_per_game_details(tmp_path: Path):
    df = _df(_resolved_row(), _unresolved_row())
    partitions = gen.build_per_game_detail_rows(df, {})
    all_play_ids = [r["play_id"] for records in partitions.values() for r in records]
    assert all_play_ids == ["700001-10-3"]


def test_per_game_partitions_contain_correct_play_ids(tmp_path: Path):
    row_game_a_1 = _resolved_row(play_id="800001-1-1", game_pk=800001)
    row_game_a_2 = _resolved_row(play_id="800001-2-1", game_pk=800001)
    row_game_b_1 = _resolved_row(play_id="800002-1-1", game_pk=800002)
    df = _df(row_game_a_1, row_game_a_2, row_game_b_1)
    partitions = gen.build_per_game_detail_rows(df, {})

    assert set(partitions.keys()) == {"800001", "800002"}
    assert [r["play_id"] for r in partitions["800001"]] == ["800001-1-1", "800001-2-1"]
    assert [r["play_id"] for r in partitions["800002"]] == ["800002-1-1"]


def test_no_duplicate_play_id_across_game_partitions(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="800001-1-1", game_pk=800001),
        _resolved_row(play_id="800002-1-1", game_pk=800002),
        _resolved_row(play_id="800003-1-1", game_pk=800003),
    )
    partitions = gen.build_per_game_detail_rows(df, {})
    all_play_ids = [r["play_id"] for records in partitions.values() for r in records]
    assert len(all_play_ids) == len(set(all_play_ids))


def test_no_duplicate_play_id_across_player_partitions(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="800001-1-1", batter_id=1),
        _resolved_row(play_id="800002-1-1", batter_id=2),
        _resolved_row(play_id="800003-1-1", batter_id=3),
    )
    partitions = gen.build_player_play_index_rows(df, {})
    all_play_ids = [r["play_id"] for records in partitions.values() for r in records]
    assert len(all_play_ids) == len(set(all_play_ids))


def test_union_of_player_index_play_ids_equals_canonical_scored_set(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="800001-1-1", batter_id=1),
        _resolved_row(play_id="800002-1-1", batter_id=2),
        _unresolved_row(play_id="800003-1-1", batter_id=3),
    )
    partitions = gen.build_player_play_index_rows(df, {})
    union_ids = {r["play_id"] for records in partitions.values() for r in records}
    assert union_ids == {"800001-1-1", "800002-1-1"}


def test_per_game_detail_canonical_fields_copied_exactly(tmp_path: Path):
    df = _df(_resolved_row())
    partitions = gen.build_per_game_detail_rows(df, {12345: "Test Player"})
    record = partitions["700001"][0]
    source = df.iloc[0]
    for field in gen.PER_GAME_DETAIL_CANONICAL_FIELDS:
        expected = source[field]
        actual = record[field]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), field
        else:
            assert actual == expected, field
    assert record["batter_name"] == "Test Player"


def test_batter_name_null_when_unresolved(tmp_path: Path):
    df = _df(_resolved_row())
    rows = gen.build_players_catalog_rows(df, {})
    assert rows[0]["batter_name"] is None
    partitions = gen.build_player_play_index_rows(df, {})
    assert partitions["12345"][0]["batter_name"] is None


# ---------------------------------------------------------------------------
# End-to-end via generate_explorer_artifacts
# ---------------------------------------------------------------------------


def test_generate_explorer_artifacts_full_stack_scoring_fields_reconcile(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    detail = json.loads((out_dir / "games" / "700001.json").read_text())[0]
    gap = abs(
        (detail["observed_run_value"] - detail["expected_run_value"]) - detail["contact_luck_runs"]
    )
    assert gap < 1e-9


def test_expected_run_value_and_contact_luck_runs_copied_exactly_into_player_index(
    tmp_path: Path,
):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    row = json.loads((out_dir / "players" / "12345.json").read_text())[0]
    source = df.iloc[0]
    assert row["expected_run_value"] == pytest.approx(float(source["expected_run_value"]))
    assert row["contact_luck_runs"] == pytest.approx(float(source["contact_luck_runs"]))


def test_generate_explorer_artifacts_writes_no_monolithic_search_index(tmp_path: Path):
    """The Phase 4.1 monolithic search-index.json must never be produced
    again -- only the sharded players.json + players/<id>.json +
    games/<pk>.json + explore-metadata.json."""
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    top_level = {p.name for p in out_dir.iterdir()}
    assert top_level == {
        "players.json",
        "players",
        "games",
        "explore-metadata.json",
        "showcase.json",
    }
    assert "search-index.json" not in top_level
    assert (out_dir / "players").is_dir()
    assert (out_dir / "games").is_dir()
    player_files = {p.name for p in (out_dir / "players").iterdir()}
    assert player_files == {"12345.json"}
    game_files = {p.name for p in (out_dir / "games").iterdir()}
    assert game_files == {"700001.json"}


# ---------------------------------------------------------------------------
# explore-metadata.json provenance
# ---------------------------------------------------------------------------


def test_compute_file_sha256_matches_real_hash(tmp_path: Path):
    import hashlib

    path = tmp_path / "sample.bin"
    path.write_bytes(b"some canonical parquet bytes")
    expected = hashlib.sha256(b"some canonical parquet bytes").hexdigest()
    assert gen.compute_file_sha256(path) == expected


def test_build_explore_metadata_contains_required_keys_and_real_sha256(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    ledger_metadata = json.loads(metadata_path.read_text())
    metadata = gen.build_explore_metadata(
        play_ledger_path=ledger_path,
        play_ledger_metadata=ledger_metadata,
        play_count=1,
        player_count=1,
        game_count=1,
        showcase_favorable_count=1,
        showcase_unfavorable_count=1,
        showcase_interactive_count=0,
    )
    assert set(metadata.keys()) == {
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
    }
    assert metadata["explorer_artifact_version"] == gen.EXPLORER_ARTIFACT_VERSION
    assert metadata["play_ledger_version"] == "2.0"
    assert metadata["play_count"] == 1
    assert metadata["player_count"] == 1
    assert metadata["game_count"] == 1
    assert metadata["showcase_favorable_count"] == 1
    assert metadata["showcase_unfavorable_count"] == 1
    assert metadata["showcase_interactive_count"] == 0
    assert metadata["source_play_ledger_sha256"] == gen.compute_file_sha256(ledger_path)


def test_build_explore_metadata_data_through_date_null_when_absent(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    ledger_metadata = json.loads(metadata_path.read_text())
    assert ledger_metadata.get("data_through_date") is None
    metadata = gen.build_explore_metadata(
        play_ledger_path=ledger_path,
        play_ledger_metadata=ledger_metadata,
        play_count=1,
        player_count=1,
        game_count=1,
        showcase_favorable_count=1,
        showcase_unfavorable_count=1,
        showcase_interactive_count=0,
    )
    assert metadata["data_through_date"] is None


def test_generate_explorer_artifacts_writes_explore_metadata_matching_output(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    result = gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    on_disk = json.loads((out_dir / "explore-metadata.json").read_text())
    assert on_disk == result["explore_metadata"]
    assert on_disk["play_count"] == 1
    assert on_disk["player_count"] == 1
    assert on_disk["game_count"] == 1
    assert on_disk["source_play_ledger_sha256"] == gen.compute_file_sha256(ledger_path)


def test_explore_metadata_counts_reconcile_with_generated_artifacts(tmp_path: Path):
    df = _df(
        _resolved_row(play_id="800001-1-1", game_pk=800001, batter_id=1),
        _resolved_row(play_id="800002-1-1", game_pk=800002, batter_id=2),
        _resolved_row(play_id="800002-2-1", game_pk=800002, batter_id=1),
    )
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    metadata = json.loads((out_dir / "explore-metadata.json").read_text())
    player_files = list((out_dir / "players").glob("*.json"))
    game_files = list((out_dir / "games").glob("*.json"))
    total_rows = sum(len(json.loads(p.read_text())) for p in player_files)

    assert metadata["play_count"] == total_rows == 3
    assert metadata["player_count"] == len(player_files) == 2
    assert metadata["game_count"] == len(game_files) == 2


def test_explore_metadata_is_byte_deterministic(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_a
    )
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_b
    )
    assert (out_a / "explore-metadata.json").read_bytes() == (
        out_b / "explore-metadata.json"
    ).read_bytes()


def test_explore_metadata_has_no_generation_timestamp(tmp_path: Path):
    df = _df(_resolved_row())
    ledger_path, metadata_path = _write_ledger(tmp_path, df)
    out_dir = tmp_path / "out"
    gen.generate_explorer_artifacts(
        play_ledger_path=ledger_path, play_ledger_metadata_path=metadata_path, output_dir=out_dir
    )
    on_disk = json.loads((out_dir / "explore-metadata.json").read_text())
    assert "generated_at" not in on_disk
    assert not any("time" in key.lower() or "timestamp" in key.lower() for key in on_disk)


def test_load_batter_names_none_path_returns_empty_dict():
    assert gen.load_batter_names(None) == {}


def test_load_batter_names_skips_null_values(tmp_path: Path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"1": "Alice", "2": None}))
    names = gen.load_batter_names(path)
    assert names == {1: "Alice"}


# ---------------------------------------------------------------------------
# Structural: no model/scoring imports (mirrors test_dashboard_isolation.py)
# ---------------------------------------------------------------------------

_BANNED_IMPORT_NAMES = (
    "train_contact_model",
    "train_opportunity_model",
    "train_advancement_model",
    "compare_infield_opportunity",
    "compare_advancement_models",
    "compare_near_wall_models",
    "attribution_ledger",
    "pybaseball",
    "download_statcast",
)


def test_generator_imports_no_model_or_training_code():
    path = Path(__file__).resolve().parents[1] / "demo" / "build_play_explorer_fixture.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
            imported.add(node.module.split(".")[0])
    violations = [name for name in _BANNED_IMPORT_NAMES if name in imported]
    assert not violations, (
        f"demo/build_play_explorer_fixture.py must never import training/scoring code, "
        f"found: {violations}"
    )


def test_generator_never_calls_predict_proba():
    """Checks for an actual CALL (`predict_proba_ordered(`/`.predict_proba(`/
    `.fit(`), not merely the substring -- this module's own docstring
    legitimately mentions `predict_proba_ordered` in prose (documenting
    that it does NOT call it), which a bare substring check would flag."""
    path = Path(__file__).resolve().parents[1] / "demo" / "build_play_explorer_fixture.py"
    text = path.read_text()
    assert "predict_proba_ordered(" not in text
    assert ".predict_proba(" not in text
    assert ".fit(" not in text
