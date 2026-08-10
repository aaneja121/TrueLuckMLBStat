"""Contact Luck v1.2 dashboard: end-to-end static-site build tests.

Builds against synthetic snapshot fixtures only (`dashboard_snapshot_fixtures`)
-- never touches this repo's real `outputs/prospective`/`artifacts/prospective`
directories, and never invokes any model-training/scoring code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build as dashboard_build
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot


def _players() -> list[dict]:
    return [
        default_player_record(
            batter_id=1,
            batter_name="Alice Alpha",
            score=5.25,
            lower=1.1,
            upper=9.4,
            bbe=310,
            games=101,
        ),
        default_player_record(
            batter_id=2,
            batter_name="Bob Beta",
            score=-3.5,
            lower=-7.2,
            upper=1.3,
            bbe=280,
            games=95,
        ),
        default_player_record(
            batter_id=3,
            batter_name="Carl Gamma",
            score=1.0,
            lower=-4.0,
            upper=6.0,
            bbe=40,
            games=15,
            qualification_status="small_sample",
        ),
        default_player_record(
            batter_id=4, batter_name=None, score=2.0, lower=-1.0, upper=5.0, bbe=200, games=80
        ),
    ]


def _seed_two_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01",
        data_through_date="2026-01-01",
        snapshot_label=None,
        generated_at="2026-01-01T00:00:00+00:00",
        players=_players(),
    )
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-02",
        data_through_date="2026-01-02",
        snapshot_label=None,
        generated_at="2026-01-02T00:00:00+00:00",
        players=_players(),
    )
    return out_root, art_root


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TestBuildErrors:
    def test_raises_when_no_snapshot_exists(self, tmp_path: Path) -> None:
        with pytest.raises(dashboard_build.DashboardBuildError):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / "dist",
                outputs_root=tmp_path / "outputs",
                artifacts_root=tmp_path / "artifacts",
            )

    def test_raises_when_only_invalid_snapshots_exist(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
            corrupt="bad_hash",
        )
        with pytest.raises(dashboard_build.DashboardBuildError):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / "dist", outputs_root=out_root, artifacts_root=art_root
            )


class TestBuildContent:
    def test_leaderboard_shows_only_qualified_players(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        # Exclude the embedded global-search index -- Carl belongs there (any
        # player is searchable) but must never appear as a ranked table row.
        table_html = html.split('id="player-index-data">', 1)[0]
        assert "Alice Alpha" in table_html
        assert "Bob Beta" in table_html
        assert "Carl Gamma" not in table_html  # small_sample -- never on the ranked leaderboard

    def test_player_names_displayed_including_null_fallback(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        unresolved_html = (tmp_path / "dist" / "players" / "4" / "index.html").read_text()
        assert "Player 4" in unresolved_html

    def test_every_interval_bar_carries_the_point_estimate_alongside_it(
        self, tmp_path: Path
    ) -> None:
        """Phase 4: intervals must never be shown without the point estimate
        right there with them -- check the SVG title text pattern directly.
        """
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert "<title>5.25 runs/100 (95% interval: 1.10 to 9.40)</title>" in html

    def test_component_status_reason_codes_propagate_to_player_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert "near_wall_provisional" in html
        assert "calibrated_with_limited_subgroup_evidence" in html

    def test_trend_chart_present_across_two_stored_snapshots(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert 'class="trend-chart"' in html
        assert "2026-01-01" in html and "2026-01-02" in html

    def test_status_page_shows_latest_data_through_date(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "status" / "index.html").read_text()
        assert "2026-01-02" in html

    def test_sorted_view_and_official_rank_column_both_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        assert 'data-sort-key="officialRank"' in html
        assert 'id="sorted-label-favorable"' in html

    def test_global_search_index_includes_non_qualified_players(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        marker = html.split('id="player-index-data">', 1)[1].split("</script>", 1)[0]
        index = json.loads(marker)
        names = {row["batter_name"] for row in index}
        assert "Carl Gamma" in names  # searchable even though not on the ranked leaderboard

    def test_stable_player_urls_are_keyed_by_batter_id(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        assert (tmp_path / "dist" / "players" / "1" / "index.html").exists()
        assert (tmp_path / "dist" / "players" / "4" / "index.html").exists()


class TestResponsiveBasics:
    def test_viewport_meta_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        assert 'name="viewport"' in html
        assert 'class="overflow-x"' in html

    def test_stylesheet_has_a_media_query(self) -> None:
        css = (
            Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
        ).read_text()
        assert "@media" in css


class TestDeterminismAndMutation:
    def test_build_is_deterministic_given_a_fixed_timestamp(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_a",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_b",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        hashes_a = _sha256_tree(tmp_path / "dist_a")
        hashes_b = _sha256_tree(tmp_path / "dist_b")
        assert hashes_a == hashes_b

    def test_build_never_writes_into_snapshot_directories(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        before = _sha256_tree(out_root) | _sha256_tree(art_root)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        after = _sha256_tree(out_root) | _sha256_tree(art_root)
        assert before == after

    def test_build_manifest_records_expected_provenance(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        assert result.manifest.preferred_snapshot_directory_name == "2026-01-02"
        assert result.manifest.data_through_date == "2026-01-02"
        assert result.manifest.player_count == 4
        assert result.manifest.qualified_count == 3
        assert result.invalid_snapshot_count == 0

        manifest_on_disk = json.loads(
            (tmp_path / "dist" / "data" / "dashboard_build_manifest.json").read_text()
        )
        assert manifest_on_disk["preferred_snapshot_directory_name"] == "2026-01-02"

    def test_invalid_snapshot_alongside_a_valid_one_is_excluded_but_does_not_block_build(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-02",
            data_through_date="2026-01-02",
            snapshot_label=None,
            generated_at="2026-01-02T00:00:00+00:00",
            players=_players(),
            corrupt="bad_hash",
        )
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        assert result.manifest.preferred_snapshot_directory_name == "2026-01-01"
        assert result.invalid_snapshot_count == 1
