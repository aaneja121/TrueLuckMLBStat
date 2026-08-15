"""Contact Luck v1.2 dashboard: Phase 1 snapshot discovery/integrity/
precedence tests. All fixtures are synthetic (`dashboard_snapshot_fixtures`)
-- no test here reads the real repo's `outputs/prospective`/
`artifacts/prospective` directories.
"""

from __future__ import annotations

from pathlib import Path

import snapshot_data as sd

from dashboard_snapshot_fixtures import default_player_record, write_snapshot


def _players() -> list[dict]:
    return [
        default_player_record(batter_id=1, batter_name="Alice", score=5.0, lower=1.0, upper=9.0),
        default_player_record(batter_id=2, batter_name="Bob", score=-3.0, lower=-7.0, upper=1.0),
        default_player_record(
            batter_id=3,
            batter_name="Carl",
            score=1.0,
            lower=-4.0,
            upper=6.0,
            qualification_status="small_sample",
        ),
    ]


class TestClassification:
    def test_no_label_is_genuine(self) -> None:
        assert (
            sd.classify_snapshot_type(snapshot_label=None, integrity_valid=True)
            == sd.SNAPSHOT_TYPE_GENUINE
        )

    def test_arbitrary_label_is_corrected(self) -> None:
        assert (
            sd.classify_snapshot_type(snapshot_label="refreshed", integrity_valid=True)
            == sd.SNAPSHOT_TYPE_CORRECTED
        )

    def test_reserved_backfill_label_is_backfill(self) -> None:
        assert (
            sd.classify_snapshot_type(snapshot_label="retrospective_backfill", integrity_valid=True)
            == sd.SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL
        )

    def test_failed_integrity_is_always_invalid_regardless_of_label(self) -> None:
        assert (
            sd.classify_snapshot_type(snapshot_label=None, integrity_valid=False)
            == sd.SNAPSHOT_TYPE_INVALID
        )
        assert (
            sd.classify_snapshot_type(
                snapshot_label="retrospective_backfill", integrity_valid=False
            )
            == sd.SNAPSHOT_TYPE_INVALID
        )

    def test_parse_snapshot_directory_name(self) -> None:
        assert sd.parse_snapshot_directory_name("2026-08-08") == ("2026-08-08", None)
        assert sd.parse_snapshot_directory_name("2026-08-08__refreshed") == (
            "2026-08-08",
            "refreshed",
        )
        assert sd.parse_snapshot_directory_name("2026-08-08__a__b") == ("2026-08-08", "a__b")


class TestDiscoveryAndIntegrity:
    def test_valid_snapshot_discovered_and_marked_valid(self, tmp_path: Path) -> None:
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
        snapshots = sd.discover_snapshots(out_root, art_root)
        assert len(snapshots) == 1
        assert snapshots[0].integrity.valid
        assert snapshots[0].snapshot_type == sd.SNAPSHOT_TYPE_GENUINE

    def test_snapshot_without_play_ledger_files_remains_valid(self, tmp_path: Path) -> None:
        """Version 1.4.0 Phase 3, Section 7 (backward compatibility): a
        snapshot predating play-ledger persistence -- `write_snapshot`
        (this whole file's shared fixture) never writes `play_ledger.
        parquet`/`play_ledger_metadata.json` -- must still validate, since
        `_load_and_validate_snapshot`'s integrity check iterates whatever
        `integrity_hashes.json` actually lists, never a hardcoded expected
        -filename set requiring these two files to be present. Older
        canonical snapshots stay valid under their historical schema; a
        play ledger is a forward-only addition, never retroactively
        required."""
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
        assert not (out_root / "2026-01-01" / "play_ledger.parquet").exists()
        assert not (out_root / "2026-01-01" / "play_ledger_metadata.json").exists()

        snapshots = sd.discover_snapshots(out_root, art_root)
        assert len(snapshots) == 1
        assert snapshots[0].integrity.valid
        assert snapshots[0].snapshot_type == sd.SNAPSHOT_TYPE_GENUINE

    def test_missing_manifest_is_invalid_and_excluded(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
            corrupt="missing_manifest",
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        assert not snapshots[0].integrity.valid
        assert snapshots[0].snapshot_type == sd.SNAPSHOT_TYPE_INVALID
        assert sd.valid_snapshots(snapshots) == []
        assert sd.resolve_latest_snapshot(snapshots) is None

    def test_bad_hash_is_invalid_and_excluded(self, tmp_path: Path) -> None:
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
        snapshots = sd.discover_snapshots(out_root, art_root)
        assert not snapshots[0].integrity.valid
        assert any("hash mismatch" in e for e in snapshots[0].integrity.errors)
        assert sd.resolve_latest_snapshot(snapshots) is None

    def test_mismatched_label_is_invalid(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
            corrupt="mismatched_label",
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        assert not snapshots[0].integrity.valid
        assert any(
            "does not match manifest snapshot_label" in e for e in snapshots[0].integrity.errors
        )

    def test_malformed_manifest_json_is_invalid(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
            corrupt="malformed_manifest_json",
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        assert not snapshots[0].integrity.valid

    def test_a_bad_newer_snapshot_never_becomes_latest(self, tmp_path: Path) -> None:
        """An invalid, newer-dated snapshot must never be silently preferred
        over a valid, older one -- `resolve_latest_snapshot` should fall
        back to the newest VALID date, not the newest date on disk.
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
        snapshots = sd.discover_snapshots(out_root, art_root)
        latest = sd.resolve_latest_snapshot(snapshots)
        assert latest is not None
        assert latest.directory_name == "2026-01-01"


class TestPrecedence:
    def test_corrected_snapshot_supersedes_genuine_for_same_date(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T10:00:00+00:00",
            players=_players(),
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01__refreshed",
            data_through_date="2026-01-01",
            snapshot_label="refreshed",
            generated_at="2026-01-01T12:00:00+00:00",
            players=_players(),
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        preferred = sd.resolve_preferred_snapshots(snapshots)
        assert preferred["2026-01-01"].directory_name == "2026-01-01__refreshed"
        assert preferred["2026-01-01"].snapshot_type == sd.SNAPSHOT_TYPE_CORRECTED

        history = sd.build_snapshot_history(snapshots)
        assert len(history) == 1
        assert {s.directory_name for s in history[0].all_valid_snapshots} == {
            "2026-01-01",
            "2026-01-01__refreshed",
        }

    def test_earlier_generated_correction_does_not_win(self, tmp_path: Path) -> None:
        """Precedence is by generated_at, not by directory-name order."""
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01__early_fix",
            data_through_date="2026-01-01",
            snapshot_label="early_fix",
            generated_at="2026-01-01T09:00:00+00:00",
            players=_players(),
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T10:00:00+00:00",
            players=_players(),
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        preferred = sd.resolve_preferred_snapshots(snapshots)
        assert preferred["2026-01-01"].directory_name == "2026-01-01"

    def test_retrospective_backfill_never_supersedes_a_genuine_snapshot(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T10:00:00+00:00",
            players=_players(),
        )
        # Backfill generated LATER than the genuine snapshot -- must still lose.
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01__retrospective_backfill",
            data_through_date="2026-01-01",
            snapshot_label="retrospective_backfill",
            generated_at="2026-06-01T00:00:00+00:00",
            players=_players(),
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        preferred = sd.resolve_preferred_snapshots(snapshots)
        assert preferred["2026-01-01"].directory_name == "2026-01-01"
        assert preferred["2026-01-01"].snapshot_type == sd.SNAPSHOT_TYPE_GENUINE

    def test_retrospective_backfill_used_only_when_sole_snapshot_for_date(
        self, tmp_path: Path
    ) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01__retrospective_backfill",
            data_through_date="2026-01-01",
            snapshot_label="retrospective_backfill",
            generated_at="2026-06-01T00:00:00+00:00",
            players=_players(),
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        preferred = sd.resolve_preferred_snapshots(snapshots)
        assert preferred["2026-01-01"].snapshot_type == sd.SNAPSHOT_TYPE_RETROSPECTIVE_BACKFILL

    def test_resolve_latest_snapshot_never_returns_a_backfill(self, tmp_path: Path) -> None:
        """Even if a backfill's date is numerically the newest on disk, it
        must never be chosen as the site's 'latest' snapshot.
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
        )
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-05__retrospective_backfill",
            data_through_date="2026-01-05",
            snapshot_label="retrospective_backfill",
            generated_at="2026-06-01T00:00:00+00:00",
            players=_players(),
        )
        snapshots = sd.discover_snapshots(out_root, art_root)
        latest = sd.resolve_latest_snapshot(snapshots)
        assert latest is not None
        assert latest.directory_name == "2026-01-01"

    def test_gap_dates_have_no_history_entry(self, tmp_path: Path) -> None:
        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        for day in ("2026-01-01", "2026-01-02", "2026-01-04"):
            write_snapshot(
                out_root,
                art_root,
                directory_name=day,
                data_through_date=day,
                snapshot_label=None,
                generated_at=f"{day}T00:00:00+00:00",
                players=_players(),
            )
        snapshots = sd.discover_snapshots(out_root, art_root)
        dates = [entry.data_through_date for entry in sd.build_snapshot_history(snapshots)]
        assert dates == ["2026-01-04", "2026-01-02", "2026-01-01"]
        assert "2026-01-03" not in dates
