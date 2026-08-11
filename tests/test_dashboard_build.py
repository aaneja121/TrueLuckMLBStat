"""Contact Luck v1.2 dashboard: end-to-end static-site build tests.

Builds against synthetic snapshot fixtures only (`dashboard_snapshot_fixtures`)
-- never touches this repo's real `outputs/prospective`/`artifacts/prospective`
directories, and never invokes any model-training/scoring code.
"""

from __future__ import annotations

import hashlib
import json
import re
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
        # Raw codes still exist -- inside the technical disclosure, not the
        # primary table.
        assert "near_wall_provisional" in html
        assert "calibrated_with_limited_subgroup_evidence" in html
        assert '<details class="technical-disclosure">' in html
        assert "View technical reason codes" in html

    def test_component_status_shows_plain_language_summary_first(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        primary_table_html = html.split('<details class="technical-disclosure">', 1)[0]
        assert "Provisional" in primary_table_html
        assert "Limited subgroup evidence" in primary_table_html
        # The raw status token must not leak into the primary (non-disclosure)
        # table -- only the plain-language summary belongs there.
        assert "calibrated_with_limited_subgroup_evidence" not in primary_table_html

    def test_share_from_provisional_components_label_is_plain_language(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert "Share from provisional components" in html
        assert "Provisional-component share" not in html

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


class TestIntervalColumnHeaderAlignment:
    """Lightweight regression coverage for the "95% interval" header
    appearing left of the bars it labels: the header `<th>` and every
    interval `<td>` in BOTH leaderboard tables must carry the same
    `interval-col` class (so they share one column width/alignment), and
    the stylesheet must actually centre that class and the fixed-width
    compact bar within it. This does not touch, and is not a substitute
    for, `tests/test_dashboard_visuals.py`'s interval-domain/geometry
    tests -- it only guards the header/cell presentation from silently
    drifting apart again.
    """

    def test_header_and_every_interval_cell_share_the_interval_col_class(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()

        headers = re.findall(r'<th class="interval-col">95% interval</th>', html)
        assert len(headers) == 2, (
            "expected one '95% interval' header per table (favorable + unfavorable)"
        )

        cells = re.findall(r'<td class="interval-col"><svg class="interval-bar', html)
        rows = re.findall(r"<tr data-official-rank=", html)
        assert len(cells) == len(rows) > 0
        qualified_count = sum(1 for p in _players() if p["qualification_status"] == "qualified")
        assert len(cells) == 2 * qualified_count  # every qualified player, in both tables

    def test_stylesheet_centers_the_interval_col_class_and_the_compact_bar(self) -> None:
        css = (
            Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
        ).read_text()
        assert re.search(
            r"table\.leaderboard\s+(td|th)\.interval-col,\s*"
            r"table\.leaderboard\s+(th|td)\.interval-col\s*\{[^}]*text-align:\s*center",
            css,
        ), "the interval-col header/cell rule must center-align (grep-visible in style.css)"
        assert re.search(r"\.interval-bar-compact\s*\{[^}]*margin:\s*0 auto", css), (
            "the compact bar itself must be centered within its (now text-align:center) cell"
        )


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


def _bar_positions(svg: str) -> dict[str, float]:
    return {
        "zero": float(re.search(r'interval-zero-line" x1="([\d.-]+)"', svg).group(1)),
        "lower": float(re.search(r'interval-range[^"]*" x1="([\d.-]+)"', svg).group(1)),
        "upper": float(
            re.search(r'interval-range[^"]*" x1="[\d.-]+" y1="[\d.-]+" x2="([\d.-]+)"', svg).group(
                1
            )
        ),
        "point": float(re.search(r'interval-point[^"]*" cx="([\d.-]+)"', svg).group(1)),
    }


def _row_svg(html: str, table_id: str, batter_name: str) -> str:
    table_html = html.split(f'id="lb-{table_id}"', 1)[1]
    row_start = table_html.find(f'data-player-name="{batter_name}"')
    assert row_start != -1, f"{batter_name!r} not found in table {table_id!r}"
    return re.search(
        r'<svg class="interval-bar interval-bar-compact".*?</svg>', table_html[row_start:]
    ).group(0)


class TestIntervalDomainConsistency:
    """Regression coverage for the interval-bar scale bug: every mini bar in
    the SAME leaderboard view (both tables) must share one x-domain, AND a
    player's own detail-page bar must land on the exact same scale as their
    leaderboard row -- not a separately (re-)padded domain. See
    `dashboard/build.py`'s `qualified_intervals`/`_player_view` and
    `tests/test_dashboard_visuals.py` for the underlying transform's own
    numeric tests.
    """

    def test_all_leaderboard_rows_in_both_tables_share_one_zero_position(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        zero_positions = set(re.findall(r'interval-zero-line" x1="([\d.-]+)"', html))
        assert len(zero_positions) == 1, (
            f"expected one shared zero position across every row in both "
            f"tables, found: {zero_positions}"
        )

    def test_domain_reflects_both_tables_even_when_they_diverge(self, tmp_path: Path) -> None:
        """Alice is the most favorable player and would be cut from the
        unfavorable table by a `top_n=1` slice, while Bob (the most
        unfavorable) would be cut from the favorable table the same way --
        each table alone is missing one extreme. The shared domain must
        still be built from the UNION of what both tables display, not from
        whichever table happens to be used to compute it -- so Bob's
        negative CI in the favorable-derived view must not be clipped
        against a domain sized only for Alice's positive range.
        """
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
            favorable_top_n=1,
            unfavorable_top_n=1,
        )
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()

        alice_svg = _row_svg(html, "favorable", "Alice Alpha")
        bob_svg = _row_svg(html, "unfavorable", "Bob Beta")
        alice_pos = _bar_positions(alice_svg)
        bob_pos = _bar_positions(bob_svg)

        # Same shared domain -> identical zero position in both tables,
        # even though Bob never appears in the favorable table this domain
        # was historically (buggily) derived from alone.
        assert alice_pos["zero"] == bob_pos["zero"]

        # Bob's CI is [-7.2, 1.3] -- entirely on the negative side except
        # for a sliver above zero. If the domain were sized only for
        # Alice's [1.1, 9.4] range, Bob's lower bound (-7.2) would fall
        # outside domain_min and get clamped flush to the left margin,
        # identical to where a much-less-negative value would also clamp.
        # Confirm no clamping happened: Bob's lower bound must sit strictly
        # inside the drawable area, not pinned to the margin.
        compact_width, compact_margin = 220, 10
        assert bob_pos["lower"] > compact_margin + 0.5
        assert bob_pos["lower"] < compact_width - compact_margin

        # And the point estimate must sit at its mathematically correct
        # proportional location within Bob's own CI.
        value_fraction = (-3.5 - (-7.2)) / (1.3 - (-7.2))
        pixel_fraction = (bob_pos["point"] - bob_pos["lower"]) / (
            bob_pos["upper"] - bob_pos["lower"]
        )
        assert pixel_fraction == pytest.approx(value_fraction, abs=1e-3)

    def test_player_detail_page_bar_uses_the_same_scale_as_its_leaderboard_row(
        self, tmp_path: Path
    ) -> None:
        """Regression test for the specific bug this task fixed: the player
        detail page used to re-pad the already-padded leaderboard domain
        (double padding), landing the SAME player's SAME numbers at a
        different fractional position than their leaderboard row. Both
        views must now agree, to floating-point rounding only.
        """
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        index_html = (tmp_path / "dist" / "index.html").read_text()
        player_html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()

        row_svg = _row_svg(index_html, "favorable", "Alice Alpha")
        player_svg = re.search(
            r'<svg class="interval-bar interval-bar-full".*?</svg>', player_html
        ).group(0)

        row_pos = _bar_positions(row_svg)
        player_pos = _bar_positions(player_svg)

        compact_width, compact_margin = 220, 10
        full_width, full_margin = 480, 28

        def fractions(pos: dict[str, float], width: int, margin: int) -> dict[str, float]:
            inner_w = width - 2 * margin
            return {k: (v - margin) / inner_w for k, v in pos.items()}

        row_fractions = fractions(row_pos, compact_width, compact_margin)
        player_fractions = fractions(player_pos, full_width, full_margin)

        for key in ("zero", "lower", "point", "upper"):
            assert row_fractions[key] == pytest.approx(player_fractions[key], abs=1e-3), (
                f"{key} fraction differs between leaderboard row "
                f"({row_fractions[key]:.4f}) and player detail page "
                f"({player_fractions[key]:.4f}) -- they must share one scale"
            )


class TestScoredGamesLabelClarification:
    """Presentation-only clarification: the public "Games" label was renamed
    to "Scored Games" everywhere on the dashboard, defined as "games with at
    least one outcome-resolved eligible batted ball" -- NOT official MLB
    games played. The underlying `games` value itself (`aggregate_
    attribution.aggregate_to_batter_season`'s computation, unchanged by this
    task) must still display byte-for-byte identically under the new label.
    See dashboard/templates/{_macros.html,player.html,methodology.html} and
    `mlb_luck_score.scoring.public_score_schema`'s "games" field description
    (tested separately in `tests/test_public_score_schema.py`).
    """

    def test_leaderboard_header_says_scored_games_not_bare_games(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        assert html.count(">Scored Games<") == 2  # one header per table (favorable + unfavorable)
        assert ">Games<" not in html

    def test_player_page_stat_label_says_scored_games(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        assert ">Scored Games<" in html
        assert ">Games<" not in html

    def test_methodology_page_defines_scored_games(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "methodology" / "index.html").read_text()
        assert "<strong>Scored Games</strong>" in html
        assert "outcome-resolved eligible batted ball" in html
        assert "This is <em>not</em> the" in html
        assert "official MLB games played" in html

    def test_games_numeric_value_is_unchanged_in_leaderboard_row_and_player_page(
        self, tmp_path: Path
    ) -> None:
        """Alice Alpha's fixture `games=101` (see `_players()`/
        `default_player_record`) is untouched by this task -- confirm the
        renamed label still displays the EXACT same numeric value as
        before, both in the leaderboard row's data attribute and on her own
        player page's stat card.
        """
        out_root, art_root = _seed_two_snapshots(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-02T12:00:00+00:00",
        )
        index_html = (tmp_path / "dist" / "index.html").read_text()
        assert 'data-games="101"' in index_html

        player_html = (tmp_path / "dist" / "players" / "1" / "index.html").read_text()
        stat_value = re.search(
            r'<div class="stat-label"[^>]*>Scored Games</div>\s*'
            r'<div class="stat-value">(\d+)</div>',
            player_html,
        )
        assert stat_value is not None, "could not find the Scored Games stat card"
        assert stat_value.group(1) == "101"
