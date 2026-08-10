"""Contact Luck v1.2 dashboard: content.py view-model tests -- exact
preservation of stored values, qualified-only ranking, player-name display,
component status propagation, and trend gap/no-interpolation behavior.
"""

from __future__ import annotations

from pathlib import Path

import content as c
import snapshot_data as sd

from dashboard_snapshot_fixtures import default_player_record, write_snapshot


def _players() -> list[dict]:
    return [
        default_player_record(
            batter_id=1, batter_name="Alice", score=5.25, lower=1.1, upper=9.4, bbe=310, games=101
        ),
        default_player_record(
            batter_id=2, batter_name="Bob", score=-3.5, lower=-7.2, upper=1.3, bbe=280, games=95
        ),
        default_player_record(
            batter_id=3,
            batter_name="Carl",
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


def _one_snapshot(tmp_path: Path) -> sd.DiscoveredSnapshot:
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
    snaps = sd.discover_snapshots(out_root, art_root)
    latest = sd.resolve_latest_snapshot(snaps)
    assert latest is not None
    return latest


class TestLeaderboards:
    def test_only_qualified_players_are_ranked(self, tmp_path: Path) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        favorable = c.build_favorable_leaderboard(payloads)
        assert {r.batter_id for r in favorable} == {1, 2, 4}  # not Carl (small_sample)

    def test_favorable_leaderboard_preserves_stored_rank_order(self, tmp_path: Path) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        favorable = c.build_favorable_leaderboard(payloads)
        ranks = [r.official_rank for r in favorable]
        assert ranks == sorted(ranks)
        assert favorable[0].batter_id == 1  # Alice, highest score

    def test_leaderboard_values_are_byte_identical_to_stored_json(self, tmp_path: Path) -> None:
        """No recomputation -- every displayed number is copied verbatim."""
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        favorable = c.build_favorable_leaderboard(payloads)
        alice = next(r for r in favorable if r.batter_id == 1)
        assert alice.contact_luck_runs_per_100 == 5.25
        assert alice.lower_95_interval == 1.1
        assert alice.upper_95_interval == 9.4
        assert alice.eligible_batted_balls == 310
        assert alice.games == 101

    def test_unresolved_player_name_falls_back_gracefully(self, tmp_path: Path) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        favorable = c.build_favorable_leaderboard(payloads)
        unresolved = next(r for r in favorable if r.batter_id == 4)
        assert unresolved.batter_name is None  # template layer supplies the "Player <id>" fallback


class TestPlayerDetail:
    def test_component_values_and_status_codes_propagate(self, tmp_path: Path) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        record = c.find_player_record(payloads, 1)
        assert record is not None
        detail = c.build_player_detail(record)
        assert detail.batter_name == "Alice"
        assert detail.qualification_status == "qualified"
        assert detail.components.contact_per_100 == record["contact_component_per_100"]
        assert detail.components.share_of_value_from_provisional_components == 0.4
        assert "outfield_defense" in detail.components.component_status_reason_codes
        assert detail.components.component_status_reason_codes["outfield_defense"][
            "reason_codes"
        ] == ["near_wall_provisional"]

    def test_small_sample_player_has_no_official_rank(self, tmp_path: Path) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        record = c.find_player_record(payloads, 3)
        assert record is not None
        detail = c.build_player_detail(record)
        assert detail.qualification_status == "small_sample"
        assert detail.official_rank_favorable is None
        assert detail.official_rank_unfavorable is None

    def test_player_index_includes_every_player_regardless_of_qualification(
        self, tmp_path: Path
    ) -> None:
        payloads = c.load_snapshot_payloads(_one_snapshot(tmp_path))
        index = c.build_player_index(payloads)
        assert {e.batter_id for e in index} == {1, 2, 3, 4}


class TestTrend:
    def test_trend_skips_missing_dates_without_interpolation(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        for day, score in (("2026-01-01", 2.0), ("2026-01-02", 2.5), ("2026-01-04", 3.0)):
            write_snapshot(
                out_root,
                art_root,
                directory_name=day,
                data_through_date=day,
                snapshot_label=None,
                generated_at=f"{day}T00:00:00+00:00",
                players=[
                    default_player_record(
                        batter_id=1,
                        batter_name="Alice",
                        score=score,
                        lower=score - 3,
                        upper=score + 3,
                    )
                ],
            )
        snaps = sd.discover_snapshots(out_root, art_root)
        history = sd.build_snapshot_history(snaps)
        trend = c.build_player_trend(history, 1)
        dates = [p.data_through_date for p in trend]
        assert dates == ["2026-01-01", "2026-01-02", "2026-01-04"]
        assert "2026-01-03" not in dates
        assert [p.contact_luck_runs_per_100 for p in trend] == [2.0, 2.5, 3.0]

    def test_trend_skips_dates_where_player_has_no_row(self, tmp_path: Path) -> None:
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
                    batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                )
            ],
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-02",
            data_through_date="2026-01-02",
            snapshot_label=None,
            generated_at="2026-01-02T00:00:00+00:00",
            players=[
                default_player_record(
                    batter_id=2, batter_name="Bob", score=1.0, lower=-2.0, upper=4.0
                )
            ],
        )
        snaps = sd.discover_snapshots(out_root, art_root)
        history = sd.build_snapshot_history(snaps)
        trend = c.build_player_trend(history, 1)
        assert [p.data_through_date for p in trend] == ["2026-01-01"]

    def test_trend_uses_preferred_corrected_snapshot_not_superseded_one(
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
            players=[
                default_player_record(
                    batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                )
            ],
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01__refreshed",
            data_through_date="2026-01-01",
            snapshot_label="refreshed",
            generated_at="2026-01-01T12:00:00+00:00",
            players=[
                default_player_record(
                    batter_id=1, batter_name="Alice", score=2.5, lower=-0.5, upper=5.5
                )
            ],
        )
        snaps = sd.discover_snapshots(out_root, art_root)
        history = sd.build_snapshot_history(snaps)
        trend = c.build_player_trend(history, 1)
        assert len(trend) == 1
        assert trend[0].contact_luck_runs_per_100 == 2.5
        assert trend[0].snapshot_type == sd.SNAPSHOT_TYPE_CORRECTED

    def test_trend_points_retain_snapshot_type_metadata(self, tmp_path: Path) -> None:
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
                    batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                )
            ],
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-05__retrospective_backfill",
            data_through_date="2026-01-05",
            snapshot_label="retrospective_backfill",
            generated_at="2026-06-01T00:00:00+00:00",
            players=[
                default_player_record(
                    batter_id=1, batter_name="Alice", score=3.0, lower=0.0, upper=6.0
                )
            ],
        )
        snaps = sd.discover_snapshots(out_root, art_root)
        history = sd.build_snapshot_history(snaps)
        trend = c.build_player_trend(history, 1)
        assert [p.snapshot_type for p in trend] == [
            sd.SNAPSHOT_TYPE_GENUINE,
            sd.SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL,
        ]


class TestStatusPageData:
    def test_status_reflects_snapshot_and_scorecard(self, tmp_path: Path) -> None:
        latest = _one_snapshot(tmp_path)
        payloads = c.load_snapshot_payloads(latest)
        history = sd.build_snapshot_history(
            sd.discover_snapshots(latest.outputs_dir.parent, latest.artifacts_dir.parent)
        )
        status = c.build_status_page_data(payloads, history)
        assert status.data_through_date == "2026-01-01"
        assert status.qualified_count == 3  # Alice, Bob, unresolved-name player
        assert status.integrity_valid is True
        assert status.snapshot_type == sd.SNAPSHOT_TYPE_GENUINE
