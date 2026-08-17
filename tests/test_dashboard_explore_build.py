"""Contact Luck v1.4.0 dashboard: end-to-end Play Explorer build tests.

Follows `tests/test_dashboard_build.py`'s exact skeleton -- seed synthetic
snapshot(s) via `dashboard_snapshot_fixtures.write_snapshot`, seed a
synthetic, SHARDED Play Explorer fixture (players.json +
players/<batter_id>.json + games/<pk>.json + explore-metadata.json, never
the real committed one), call `build_dashboard(...)` with the injectable
explore-fixture path overrides, and assert on rendered output.

Phase 4.2 note: the Phase 4.1 monolithic `search-index.json` was replaced
by a small `players.json` catalog plus one `players/<batter_id>.json` per
batter -- see `demo/build_play_explorer_fixture.py`'s module docstring for
why (a full 2024-season search-index.json measured ~29.3 MiB, over
Cloudflare Pages' 25 MiB per-asset limit).

Phase 4.1 routing note: there is exactly ONE static play-page shell
(`dist/plays/index.html`), never one directory per play_id. Direct-play
resolution (query-param parsing, per-game JSON fetch, not-found handling)
happens entirely client-side in `dashboard/static/play.js`, which this test
suite has no JS runtime to execute -- so those behaviors are verified
structurally, against the actual shipped `dist/static/play.js`/`explore.js`
source, exactly as `test_play_explorer_fixture_generator.py` verifies
generator behavior structurally (e.g. `test_generator_never_calls_
predict_proba`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import build as dashboard_build
import explore_content as ec
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot


def _seed_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01",
        data_through_date="2026-01-01",
        snapshot_label=None,
        generated_at="2026-01-01T00:00:00+00:00",
        players=[
            default_player_record(
                batter_id=1, batter_name="Alice Alpha", score=5.0, lower=1.0, upper=9.0
            )
        ],
    )
    return out_root, art_root


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


def _seed_explore_fixture(
    tmp_path: Path,
    *,
    index_rows: list[dict[str, Any]] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    showcase_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Seeds a synthetic, SHARDED players.json + players/<id>.json +
    games/<pk>.json + explore-metadata.json + showcase.json, all internally
    consistent by default. `showcase_rows` defaults to an EMPTY showcase
    (`[]`) -- a valid, self-consistent showcase with zero rows, not a
    special case `ec.load_showcase` needs to accommodate; tests that
    specifically exercise Showcase Plays content pass their own rows (see
    `_showcase_row`). Returns
    `(players_path, players_dir, games_dir, metadata_path, showcase_path)`.
    """
    explore_root = tmp_path / "explore_fixture"
    players_dir = explore_root / "players"
    games_dir = explore_root / "games"
    players_dir.mkdir(parents=True)
    games_dir.mkdir(parents=True)

    rows = index_rows if index_rows is not None else [_index_row()]
    by_batter: dict[int, list[dict[str, Any]]] = {}
    by_game: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_batter.setdefault(row["batter_id"], []).append(row)
        by_game.setdefault(row["game_pk"], []).append(
            _play_detail(
                play_id=row["play_id"],
                game_pk=row["game_pk"],
                batter_id=row["batter_id"],
                batter_name=row.get("batter_name"),
            )
        )

    players_catalog = [
        {
            "batter_id": batter_id,
            "batter_name": batter_rows[0].get("batter_name"),
            "play_count": len(batter_rows),
        }
        for batter_id, batter_rows in sorted(by_batter.items())
    ]
    players_path = explore_root / "players.json"
    players_path.write_text(json.dumps(players_catalog))
    for batter_id, batter_rows in by_batter.items():
        (players_dir / f"{batter_id}.json").write_text(json.dumps(batter_rows))
    for game_pk, records in by_game.items():
        (games_dir / f"{game_pk}.json").write_text(json.dumps(records))

    showcase = showcase_rows if showcase_rows is not None else []
    showcase_path = explore_root / "showcase.json"
    showcase_path.write_text(json.dumps(showcase))
    showcase_favorable_count = sum(1 for r in showcase if r["group"] == "favorable")
    showcase_unfavorable_count = sum(1 for r in showcase if r["group"] == "unfavorable")
    showcase_interactive_count = sum(1 for r in showcase if r["interactive_available"])

    metadata = {
        "explorer_artifact_version": ec.SUPPORTED_EXPLORER_ARTIFACT_VERSION,
        "play_ledger_version": ec.SUPPORTED_PLAY_LEDGER_VERSION,
        "season": 2024,
        "data_through_date": None,
        "play_count": len(rows),
        "player_count": len(players_catalog),
        "game_count": len(by_game),
        "showcase_favorable_count": showcase_favorable_count,
        "showcase_unfavorable_count": showcase_unfavorable_count,
        "showcase_interactive_count": showcase_interactive_count,
        "source_play_ledger_sha256": "b" * 64,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    metadata_path = explore_root / "explore-metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    return players_path, players_dir, games_dir, metadata_path, showcase_path


def _build(
    tmp_path: Path,
    out_root: Path,
    art_root: Path,
    players_path: Path,
    players_dir: Path,
    games_dir: Path,
    metadata_path: Path,
    showcase_path: Path,
    *,
    out_dir: Path | None = None,
):
    return dashboard_build.build_dashboard(
        out_dir=out_dir or (tmp_path / "dist"),
        outputs_root=out_root,
        artifacts_root=art_root,
        explore_players_path=players_path,
        explore_players_dir=players_dir,
        explore_games_dir=games_dir,
        explore_metadata_path=metadata_path,
        explore_showcase_path=showcase_path,
        build_timestamp="2026-01-01T12:00:00+00:00",
    )


class TestExploreLandingPage:
    def test_explore_page_renders_when_fixture_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "explore" / "index.html").read_text()
        assert "Explore Plays" in html
        assert (tmp_path / "dist" / "explore" / "players.json").exists()
        assert (tmp_path / "dist" / "explore" / "players" / "12345.json").exists()
        assert (tmp_path / "dist" / "explore" / "games" / "700001.json").exists()
        assert (tmp_path / "dist" / "explore" / "explore-metadata.json").exists()

    def test_no_monolithic_search_index_generated(self, tmp_path: Path) -> None:
        """The Phase 4.1 monolithic search-index.json must never appear in
        the built site again."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        assert not (tmp_path / "dist" / "explore" / "search-index.json").exists()

    def test_players_catalog_one_entry_per_represented_batter(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [
            _index_row(play_id="700001-10-3", batter_id=1, game_pk=700001),
            _index_row(play_id="700001-11-1", batter_id=1, game_pk=700001),
            _index_row(play_id="700002-1-1", batter_id=2, game_pk=700002),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        catalog = json.loads((tmp_path / "dist" / "explore" / "players.json").read_text())
        assert {e["batter_id"] for e in catalog} == {1, 2}
        assert len(catalog) == len(set(e["batter_id"] for e in catalog))

    def test_each_batter_file_contains_only_that_batters_plays(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [
            _index_row(play_id="700001-10-3", batter_id=1, game_pk=700001),
            _index_row(play_id="700002-1-1", batter_id=2, game_pk=700002),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        batter_1 = json.loads((tmp_path / "dist" / "explore" / "players" / "1.json").read_text())
        batter_2 = json.loads((tmp_path / "dist" / "explore" / "players" / "2.json").read_text())
        assert {r["play_id"] for r in batter_1} == {"700001-10-3"}
        assert all(r["batter_id"] == 1 for r in batter_1)
        assert {r["play_id"] for r in batter_2} == {"700002-1-1"}
        assert all(r["batter_id"] == 2 for r in batter_2)

    def test_no_duplicate_play_id_across_player_files(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [
            _index_row(play_id="700001-10-3", batter_id=1, game_pk=700001),
            _index_row(play_id="700001-11-1", batter_id=1, game_pk=700001),
            _index_row(play_id="700002-1-1", batter_id=2, game_pk=700002),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        catalog = json.loads((tmp_path / "dist" / "explore" / "players.json").read_text())
        all_play_ids: list[str] = []
        for entry in catalog:
            batter_rows = json.loads(
                (
                    tmp_path / "dist" / "explore" / "players" / f"{entry['batter_id']}.json"
                ).read_text()
            )
            all_play_ids.extend(r["play_id"] for r in batter_rows)
        assert len(all_play_ids) == len(set(all_play_ids))

    def test_union_of_player_index_play_ids_equals_canonical_scored_set(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [
            _index_row(play_id="700001-10-3", batter_id=1, game_pk=700001),
            _index_row(play_id="700001-11-1", batter_id=1, game_pk=700001),
            _index_row(play_id="700002-1-1", batter_id=2, game_pk=700002),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        catalog = json.loads((tmp_path / "dist" / "explore" / "players.json").read_text())
        union_ids: set[str] = set()
        for entry in catalog:
            batter_rows = json.loads(
                (
                    tmp_path / "dist" / "explore" / "players" / f"{entry['batter_id']}.json"
                ).read_text()
            )
            union_ids.update(r["play_id"] for r in batter_rows)
        assert union_ids == {row["play_id"] for row in rows}

    def test_expected_run_value_and_contact_luck_runs_copied_exactly(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [_index_row(expected_run_value=0.4242, contact_luck_runs=-1.9191)]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        batter_rows = json.loads(
            (tmp_path / "dist" / "explore" / "players" / "12345.json").read_text()
        )
        assert batter_rows[0]["expected_run_value"] == pytest.approx(0.4242)
        assert batter_rows[0]["contact_luck_runs"] == pytest.approx(-1.9191)

    def test_explore_table_renders_expected_rv_column(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "explore" / "index.html").read_text()
        assert "Expected RV" in html

    def test_explore_metadata_copied_byte_for_byte(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        assert (tmp_path / "dist" / "explore" / "explore-metadata.json").read_bytes() == (
            metadata_path.read_bytes()
        )

    def test_explore_nav_link_present_when_fixture_available(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        homepage = (tmp_path / "dist" / "index.html").read_text()
        assert 'href="/explore/"' in homepage
        assert "Explore Plays" in homepage

    def test_explore_section_skipped_when_fixture_absent(self, tmp_path: Path) -> None:
        """No fixture at the given path -> the whole Explorer is skipped,
        the nav link is omitted, and every OTHER page still builds fine."""
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            explore_players_path=tmp_path / "does_not_exist.json",
            explore_players_dir=tmp_path / "does_not_exist_players",
            explore_games_dir=tmp_path / "does_not_exist_games",
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert not (tmp_path / "dist" / "explore").exists()
        assert not (tmp_path / "dist" / "plays").exists()
        homepage = (tmp_path / "dist" / "index.html").read_text()
        assert "Explore Plays" not in homepage
        assert "Alice Alpha" in homepage  # leaderboard content still present


class TestExploreMetadataBuildGuard:
    """The SECOND, independent fail-closed compatibility boundary (after
    the generator's own): a present-but-incompatible/inconsistent
    explore-metadata.json is a hard build failure, never silently skipped
    like a genuinely absent fixture."""

    def test_build_rejects_wrong_play_ledger_version(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"play_ledger_version": "1.0"}
        )
        with pytest.raises(ec.ExploreContentError, match="play_ledger_version"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_build_rejects_v1_0_explorer_artifact_version(self, tmp_path: Path) -> None:
        """The Phase 4.1 monolithic-search-index fixture's explorer_
        artifact_version ("1.0") is rejected -- it structurally cannot be
        the sharded schema this build reads."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"explorer_artifact_version": "1.0"}
        )
        with pytest.raises(ec.ExploreContentError, match="explorer_artifact_version"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_build_rejects_wrong_explorer_artifact_version(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"explorer_artifact_version": "99.0"}
        )
        with pytest.raises(ec.ExploreContentError, match="explorer_artifact_version"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_build_rejects_play_count_mismatch(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"play_count": 99}
        )
        with pytest.raises(dashboard_build.DashboardBuildError, match="play_count"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_build_rejects_player_count_mismatch(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"player_count": 99}
        )
        with pytest.raises(dashboard_build.DashboardBuildError, match="player_count"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_build_rejects_game_count_mismatch(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, metadata_overrides={"game_count": 99}
        )
        with pytest.raises(dashboard_build.DashboardBuildError, match="game_count"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )


class TestPlayPageRouting:
    def test_exactly_one_play_page_shell_generated(self, tmp_path: Path) -> None:
        """Regardless of how many plays are in the fixture, `dist/plays/`
        contains exactly one file: index.html. No per-play_id directory or
        file of any kind."""
        out_root, art_root = _seed_snapshot(tmp_path)
        rows = [
            _index_row(play_id="700001-10-3", game_pk=700001),
            _index_row(play_id="700001-11-1", game_pk=700001),
            _index_row(play_id="700002-1-1", game_pk=700002, batter_id=54321),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, index_rows=rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        plays_dir = tmp_path / "dist" / "plays"
        entries = list(plays_dir.iterdir())
        assert [p.name for p in entries] == ["index.html"]

    def test_play_shell_embeds_no_play_specific_data(self, tmp_path: Path) -> None:
        """The single static shell must be fully generic -- no play_id,
        game_pk, or batter_name baked into the server-rendered HTML. All of
        that is resolved client-side from the `?id=` query parameter."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert "700001-10-3" not in html
        assert "Test Player" not in html
        assert "data-play-id" not in html
        assert "data-game-pk" not in html

    def test_play_page_contains_controlled_not_found_scaffold(self, tmp_path: Path) -> None:
        """Every play page ships the not-found DOM state (hidden by
        default) that play.js reveals if the per-game JSON fetch fails or
        the play_id isn't found within it -- see dashboard/static/play.js.
        """
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert 'data-role="play-not-found"' in html
        assert "Play not found" in html

    def test_explore_results_link_to_query_param_route(self, tmp_path: Path) -> None:
        """`explore.js` must build play links as `/plays/?id=<play_id>`,
        never a per-play directory path -- checked against the actual
        shipped source, since link construction happens client-side and
        this suite has no JS runtime to execute it."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "explore.js").read_text()
        assert '"/plays/?id="' in js
        assert '"/plays/" +' not in js

    def test_play_js_reads_query_param_and_validates_play_id_shape(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "play.js").read_text()
        assert "window.location.search" in js
        assert 'params.get("id")' in js
        # A play_id format guard exists and is checked BEFORE the fetch --
        # i.e. the regex-match code appears earlier in source order than
        # the fetch() call that reaches the network.
        match_pos = js.index("PLAY_ID_PATTERN.exec")
        fetch_pos = js.index("fetch(")
        assert match_pos < fetch_pos

    def test_play_js_never_fetches_full_players_catalog_or_player_index(
        self, tmp_path: Path
    ) -> None:
        """A direct play-page visit must fetch only its own per-game JSON,
        never players.json and never any players/<batter_id>.json."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "play.js").read_text()
        assert 'fetch("players.json")' not in js
        assert "/explore/players/" not in js
        assert "/explore/games/" in js

    def test_play_js_malformed_id_never_reaches_fetch(self, tmp_path: Path) -> None:
        """Structural proof that a failed regex match returns/exits before
        any fetch() call, mirroring the generator's own call-site
        (not-substring) checking pattern in test_play_explorer_fixture_
        generator.py::test_generator_never_calls_predict_proba."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "play.js").read_text()
        collapsed = re.sub(r"\s+", " ", js)
        assert re.search(r"if \(!match\)\s*\{.*?showNotFound\(page\);\s*return;", collapsed)


class TestExplorerNetworkBehavior:
    """Structural checks on the shipped explore.js source (this test suite
    has no JS runtime to execute it) proving the player-first fetch
    contract: only players.json loads up front, and a hitter's own play
    index loads exactly once, only after that hitter is selected."""

    def test_explore_js_fetches_players_catalog_on_load(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "explore.js").read_text()
        assert "fetch(PLAYERS_URL)" in js
        assert 'var PLAYERS_URL = "players.json"' in js

    def test_explore_js_does_not_preload_all_player_files(self, tmp_path: Path) -> None:
        """No loop over the loaded playersCatalog issues a fetch -- the
        per-player fetch call lives only inside the single-selection
        function, never inside a `.forEach`/`for` over the catalog."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "explore.js").read_text()
        assert js.count('fetch("players/"') == 1
        assert "playersCatalog.forEach(function" not in js.replace(" ", "")

    def test_explore_js_player_fetch_only_inside_select_function(self, tmp_path: Path) -> None:
        """The per-player fetch call is scoped inside `selectPlayer`,
        triggered by exactly one user action (a suggestion click), never at
        module/page-load scope."""
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "explore.js").read_text()
        select_fn_start = js.index("function selectPlayer(")
        select_fn_end = js.index("\n    }", select_fn_start)
        select_fn_body = js[select_fn_start:select_fn_end]
        assert 'fetch("players/"' in select_fn_body


class TestTerminology:
    def test_play_page_uses_recorded_result_not_final_result(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert "Recorded result" in html
        assert "Final result" not in html
        assert "final result" not in html

    def test_play_page_labels_observed_run_value_as_final_observed_rv(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert "Final observed RV" in html

    def test_play_page_explains_recorded_result_vs_final_rv_are_not_guaranteed_identical(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        # Rendered HTML preserves the template's own line-wrapping, so
        # whitespace (including newlines) between words is collapsed
        # before matching rather than requiring one exact contiguous
        # substring.
        collapsed = re.sub(r"\s+", " ", html)
        assert "not always identical" in collapsed or "not guaranteed" in collapsed.lower()

    def test_explore_page_uses_recorded_result_column_label(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "explore" / "index.html").read_text()
        assert "Recorded result" in html
        assert "Final result" not in html


class TestSourceAttribution:
    def test_status_page_names_statcast_and_access_mechanism(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "status" / "index.html").read_text()
        assert "Statcast" in html
        assert "pybaseball" in html


class TestExistingPagesUnaffected:
    def test_leaderboard_player_demo_methodology_pages_unaffected_by_explore_fixture(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        result = _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        assert (tmp_path / "dist" / "index.html").exists()
        assert (tmp_path / "dist" / "players" / "1" / "index.html").exists()
        assert (tmp_path / "dist" / "demo" / "index.html").exists()
        assert (tmp_path / "dist" / "methodology" / "index.html").exists()
        assert (tmp_path / "dist" / "status" / "index.html").exists()

        player_html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert "Alice Alpha" in player_html
        assert result.manifest.player_count == 1

    def test_build_never_writes_into_snapshot_or_source_fixture_directories(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        explore_root = players_path.parent

        def _snapshot_files(*roots: Path) -> dict[str, bytes]:
            files: dict[str, bytes] = {}
            for root in roots:
                for p in root.rglob("*"):
                    if p.is_file():
                        files[str(p)] = p.read_bytes()
            return files

        before = _snapshot_files(out_root, art_root, explore_root)
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        after = _snapshot_files(out_root, art_root, explore_root)
        assert before == after


class TestDeterminism:
    def test_explore_and_play_pages_are_deterministic(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )

        dist_a = tmp_path / "dist_a"
        dist_b = tmp_path / "dist_b"
        for dist in (dist_a, dist_b):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
                out_dir=dist,
            )
        assert (dist_a / "explore" / "index.html").read_bytes() == (
            dist_b / "explore" / "index.html"
        ).read_bytes()
        assert (dist_a / "plays" / "index.html").read_bytes() == (
            dist_b / "plays" / "index.html"
        ).read_bytes()
        assert (dist_a / "explore" / "players.json").read_bytes() == (
            dist_b / "explore" / "players.json"
        ).read_bytes()


class TestShowcaseSection:
    def test_showcase_json_copied_to_dist(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [
            _showcase_row(play_id="700001-10-3", rank=1, group="favorable"),
            _showcase_row(play_id="700002-1-1", rank=1, group="unfavorable", batter_id=54321),
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, showcase_rows=showcase_rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        assert (tmp_path / "dist" / "explore" / "showcase.json").exists()
        copied = json.loads((tmp_path / "dist" / "explore" / "showcase.json").read_text())
        assert len(copied) == 2

    def test_showcase_favorable_count_mismatch_raises(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [_showcase_row(play_id="700001-10-3", rank=1, group="favorable")]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path,
            showcase_rows=showcase_rows,
            metadata_overrides={"showcase_favorable_count": 99},
        )
        with pytest.raises(ec.ExploreContentError, match="favorable"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_showcase_reconciliation_failure_raises(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [
            _showcase_row(
                play_id="700001-10-3",
                rank=1,
                group="favorable",
                observed_run_value=1.4,
                expected_run_value=0.17,
                contact_luck_runs=99.0,
            )
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, showcase_rows=showcase_rows
        )
        with pytest.raises(ec.ExploreContentError, match="reconcile"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_interactive_row_with_missing_sensitivity_file_raises(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [
            _showcase_row(
                play_id="700001-10-3", rank=1, group="favorable", interactive_available=True
            )
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path,
            showcase_rows=showcase_rows,
            metadata_overrides={"showcase_interactive_count": 1},
        )
        sensitivity_dir = showcase_path.parent / "showcase-sensitivity"
        sensitivity_dir.mkdir()
        # No 700001-10-3.json written -- must fail closed.
        with pytest.raises(ec.ExploreContentError, match="not found"):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / "dist",
                outputs_root=out_root,
                artifacts_root=art_root,
                explore_players_path=players_path,
                explore_players_dir=players_dir,
                explore_games_dir=games_dir,
                explore_metadata_path=metadata_path,
                explore_showcase_path=showcase_path,
                explore_showcase_sensitivity_dir=sensitivity_dir,
                build_timestamp="2026-01-01T12:00:00+00:00",
            )

    def test_showcase_interactive_count_without_sensitivity_dir_raises(
        self, tmp_path: Path
    ) -> None:
        """explore-metadata.json claims interactive plays exist, but no
        explore_showcase_sensitivity_dir was passed at all -- must fail
        closed, never silently show a card with a broken/missing slider."""
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [
            _showcase_row(
                play_id="700001-10-3", rank=1, group="favorable", interactive_available=True
            )
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path,
            showcase_rows=showcase_rows,
            metadata_overrides={"showcase_interactive_count": 1},
        )
        with pytest.raises(dashboard_build.DashboardBuildError, match="showcase_interactive_count"):
            _build(
                tmp_path,
                out_root,
                art_root,
                players_path,
                players_dir,
                games_dir,
                metadata_path,
                showcase_path,
            )

    def test_sensitivity_dir_copied_to_dist_when_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [
            _showcase_row(
                play_id="700001-10-3", rank=1, group="favorable", interactive_available=True
            )
        ]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path,
            showcase_rows=showcase_rows,
            metadata_overrides={"showcase_interactive_count": 1},
        )
        sensitivity_dir = showcase_path.parent / "showcase-sensitivity"
        sensitivity_dir.mkdir()
        (sensitivity_dir / "700001-10-3.json").write_text(
            json.dumps(
                {
                    "play_id": "700001-10-3",
                    "model_configuration": {"run_value_table": {}},
                    "counterfactual_semantics": {},
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
                    "exit_velocity_values": [107.6],
                    "launch_angle_values": [33.0],
                    "grid_shape": {"n_ev": 1, "n_la": 1},
                    "grid": [[1.0, 0.0, 0.0, 0.0, 0.0]],
                }
            )
        )
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            explore_players_path=players_path,
            explore_players_dir=players_dir,
            explore_games_dir=games_dir,
            explore_metadata_path=metadata_path,
            explore_showcase_path=showcase_path,
            explore_showcase_sensitivity_dir=sensitivity_dir,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert (
            tmp_path / "dist" / "explore" / "showcase-sensitivity" / "700001-10-3.json"
        ).exists()

    def test_showcase_section_present_in_rendered_explore_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        showcase_rows = [_showcase_row(play_id="700001-10-3", rank=1, group="favorable")]
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path, showcase_rows=showcase_rows
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "explore" / "index.html").read_text()
        assert "Showcase Plays" in html
        assert 'data-role="showcase-section"' in html
        # Showcase markup appears BEFORE the player-search markup in source order.
        assert html.index('data-role="showcase-section"') < html.index(
            'data-role="explore-player-search"'
        )


class TestShowcaseAndWhatIfClientBehavior:
    """Structural checks on the shipped explore.js/showcase_whatif.js
    source (no JS runtime in this suite) -- mirrors the existing structural
    -check pattern already used for explore.js/play.js elsewhere in this
    file."""

    def test_showcase_card_links_use_plays_query_param_route(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "explore.js").read_text()
        assert '"/plays/?id="' in js

    def test_whatif_script_shipped_and_linked_from_play_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        assert (tmp_path / "dist" / "static" / "showcase_whatif.js").exists()
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert "showcase_whatif.js" in html

    def test_whatif_section_below_factual_accounting_in_play_page_source(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        html = (tmp_path / "dist" / "plays" / "index.html").read_text()
        assert html.index('data-role="play-luck-figure"') < html.index(
            'data-role="play-whatif-section"'
        )

    def test_whatif_js_fetches_sensitivity_json_not_players_or_games(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
            tmp_path
        )
        _build(
            tmp_path,
            out_root,
            art_root,
            players_path,
            players_dir,
            games_dir,
            metadata_path,
            showcase_path,
        )
        js = (tmp_path / "dist" / "static" / "showcase_whatif.js").read_text()
        assert "/explore/showcase-sensitivity/" in js
        assert 'fetch("players.json")' not in js

    def test_whatif_js_never_imports_or_runs_model_code(self, tmp_path: Path) -> None:
        js_path = Path("dashboard/static/showcase_whatif.js")
        js = js_path.read_text()
        for banned in ("predict_proba", "train_model", "sklearn", "LogisticRegression"):
            assert banned not in js

    def test_whatif_js_never_computes_a_hypothetical_observed_or_luck_value(self) -> None:
        """Checks actual code identifiers (camelCase variable/property
        names), not bare substring matches -- the module's own comments
        legitimately use the English words "observed"/"luck" in prose
        explaining what this code deliberately does NOT do."""
        js_path = Path("dashboard/static/showcase_whatif.js")
        js = js_path.read_text()
        for banned in ("observedRv", "observedRunValue", "luckFigure", "contactLuck", "luckBadge"):
            assert banned not in js
