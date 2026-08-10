"""Tests for scripts/archive_snapshot.py -- durable R2 archival of official
Version 1.1 prospective snapshots. Every test uses InMemoryArchiveClient (a
plain dict-backed test double) -- none contacts real Cloudflare R2, and
none scores a new date or reads this repo's real outputs/artifacts
directories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import archive_snapshot as a
import pytest


def _write_snapshot(
    outputs_root: Path,
    artifacts_root: Path,
    *,
    snapshot_dir_name: str,
    season: int = 2026,
    output_files: dict[str, bytes] | None = None,
) -> None:
    """A minimal, synthetic local snapshot -- archive_snapshot.py only
    cares about manifest.json's prospective_season field and
    integrity_hashes.json's own content, plus whatever files are actually
    present on disk, so this fixture is deliberately much smaller than
    tests/dashboard_snapshot_fixtures.py's full snapshot builder.
    """
    output_files = (
        output_files if output_files is not None else {"public_score.json": b'[{"batter_id": 1}]'}
    )
    out_dir = outputs_root / snapshot_dir_name
    art_dir = artifacts_root / snapshot_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    for name, content in output_files.items():
        (out_dir / name).write_bytes(content)

    (art_dir / "manifest.json").write_text(json.dumps({"prospective_season": season}))

    integrity = {
        f"outputs/{name}": hashlib.sha256(content).hexdigest()
        for name, content in output_files.items()
    }
    (art_dir / "manifest.json").write_text(json.dumps({"prospective_season": season}))
    integrity["artifacts/manifest.json"] = hashlib.sha256(
        (art_dir / "manifest.json").read_bytes()
    ).hexdigest()
    (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity, indent=2))


class TestBuildSnapshotDirName:
    def test_no_label(self) -> None:
        assert a.build_snapshot_dir_name("2026-08-09", None) == "2026-08-09"

    def test_with_label(self) -> None:
        assert a.build_snapshot_dir_name("2026-08-08", "refreshed") == "2026-08-08__refreshed"


class TestDiscoverLocalSnapshotFiles:
    def test_enumerates_both_roots_without_a_hardcoded_list(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(
            outputs_root,
            artifacts_root,
            snapshot_dir_name="2026-08-09",
            output_files={"a.json": b"1", "b.csv": b"2", "brand_new_future_output.parquet": b"3"},
        )
        found = a.discover_local_snapshot_files(
            outputs_root / "2026-08-09", artifacts_root / "2026-08-09"
        )
        assert set(found) == {
            "outputs/a.json",
            "outputs/b.csv",
            "outputs/brand_new_future_output.parquet",
            "artifacts/manifest.json",
            "artifacts/integrity_hashes.json",
        }

    def test_missing_directory_yields_no_files_not_an_error(self, tmp_path: Path) -> None:
        found = a.discover_local_snapshot_files(
            tmp_path / "nope-outputs", tmp_path / "nope-artifacts"
        )
        assert found == {}


class TestArchiveSnapshot:
    def test_fresh_write_uploads_every_local_file(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")
        client = a.InMemoryArchiveClient()

        result = a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

        assert result.outcome == a.ArchiveOutcome.WRITTEN
        assert result.season == 2026
        assert set(result.keys_written) == {
            "prospective/2026/2026-08-09/outputs/public_score.json",
            "prospective/2026/2026-08-09/artifacts/manifest.json",
            "prospective/2026/2026-08-09/artifacts/integrity_hashes.json",
        }

    def test_labeled_snapshot_gets_its_own_key_prefix(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-08__refreshed")
        client = a.InMemoryArchiveClient()

        result = a.archive_snapshot(
            data_through_date="2026-08-08",
            snapshot_label="refreshed",
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        assert result.snapshot_dir_name == "2026-08-08__refreshed"
        assert any("2026-08-08__refreshed" in key for key in result.keys_written)

    def test_identical_rerun_is_a_no_op(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")
        client = a.InMemoryArchiveClient()
        kwargs = {
            "data_through_date": "2026-08-09",
            "snapshot_label": None,
            "client": client,
            "outputs_root": outputs_root,
            "artifacts_root": artifacts_root,
        }

        first = a.archive_snapshot(**kwargs)
        second = a.archive_snapshot(**kwargs)

        assert first.outcome == a.ArchiveOutcome.WRITTEN
        assert second.outcome == a.ArchiveOutcome.NO_OP_IDENTICAL
        assert second.keys_written == ()

    def test_conflicting_existing_archive_entry_is_refused(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")
        client = a.InMemoryArchiveClient()
        a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

        # Tamper with the archived integrity hashes to simulate a genuine
        # conflict (e.g. a differently-coded snapshot for the same date).
        key = "prospective/2026/2026-08-09/artifacts/integrity_hashes.json"
        tampered = json.loads(client.get_object_bytes(key))
        tampered["outputs/public_score.json"] = "0" * 64
        client.put_object_bytes(key, json.dumps(tampered).encode())

        with pytest.raises(a.ArchiveConflictError):
            a.archive_snapshot(
                data_through_date="2026-08-09",
                snapshot_label=None,
                client=client,
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )
        # Never silently overwritten -- the tampered value is still there.
        assert json.loads(client.get_object_bytes(key)) == tampered

    def test_missing_manifest_or_integrity_hashes_refuses_to_archive(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        (outputs_root / "2026-08-09").mkdir(parents=True)
        (outputs_root / "2026-08-09" / "public_score.json").write_text("[]")
        # No artifacts/2026-08-09/ directory at all.
        client = a.InMemoryArchiveClient()

        with pytest.raises(a.ArchiveError):
            a.archive_snapshot(
                data_through_date="2026-08-09",
                snapshot_label=None,
                client=client,
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )

    def test_upload_verification_failure_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "outputs", tmp_path / "artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")

        class CorruptingClient(a.InMemoryArchiveClient):
            """Simulates a put that silently didn't take -- the read-back
            never matches what was sent."""

            def put_object_bytes(self, key: str, data: bytes) -> None:
                if key.endswith("integrity_hashes.json"):
                    data = b'{"corrupted": true}'
                super().put_object_bytes(key, data)

        with pytest.raises(a.UploadVerificationError):
            a.archive_snapshot(
                data_through_date="2026-08-09",
                snapshot_label=None,
                client=CorruptingClient(),
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )


class TestRestoreSnapshot:
    def test_round_trip_restores_byte_identical_files(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "src-outputs", tmp_path / "src-artifacts"
        _write_snapshot(
            outputs_root,
            artifacts_root,
            snapshot_dir_name="2026-08-09",
            output_files={
                "public_score.json": b'[{"batter_id": 1}]',
                "scorecard.json": b'{"row_count": 1}',
            },
        )
        client = a.InMemoryArchiveClient()
        a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

        dest_outputs, dest_artifacts = tmp_path / "dst-outputs", tmp_path / "dst-artifacts"
        result = a.restore_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            season=2026,
            client=client,
            outputs_root=dest_outputs,
            artifacts_root=dest_artifacts,
        )

        assert result.outcome == a.ArchiveOutcome.WRITTEN
        restored_score = (dest_outputs / "2026-08-09" / "public_score.json").read_bytes()
        assert restored_score == (outputs_root / "2026-08-09" / "public_score.json").read_bytes()
        # integrity_hashes.json itself must come back too, or the restored
        # directory fails dashboard/snapshot_data.py's own integrity check.
        assert (dest_artifacts / "2026-08-09" / "integrity_hashes.json").exists()
        assert (dest_artifacts / "2026-08-09" / "manifest.json").exists()

    def test_restore_passes_the_dashboards_own_integrity_check(self, tmp_path: Path) -> None:
        """Uses the SAME full snapshot fixture the dashboard test suite
        validates against (tests/dashboard_snapshot_fixtures.py), not the
        minimal one above, specifically so this exercises the real
        dashboard/snapshot_data.py validator faithfully -- a restored
        snapshot must satisfy every field that validator checks, not just
        the fields archive_snapshot.py itself happens to touch.
        """
        import snapshot_data as sd

        from dashboard_snapshot_fixtures import default_player_record
        from dashboard_snapshot_fixtures import write_snapshot as write_full_snapshot

        outputs_root, artifacts_root = tmp_path / "src-outputs", tmp_path / "src-artifacts"
        write_full_snapshot(
            outputs_root,
            artifacts_root,
            directory_name="2026-08-09",
            data_through_date="2026-08-09",
            snapshot_label=None,
            generated_at="2026-08-09T00:00:00+00:00",
            players=[
                default_player_record(
                    batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                )
            ],
        )
        client = a.InMemoryArchiveClient()
        a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

        dest_outputs, dest_artifacts = tmp_path / "dst-outputs", tmp_path / "dst-artifacts"
        a.restore_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            season=2026,
            client=client,
            outputs_root=dest_outputs,
            artifacts_root=dest_artifacts,
        )

        discovered = sd.discover_snapshots(dest_outputs, dest_artifacts)
        assert len(discovered) == 1
        assert discovered[0].integrity.valid, discovered[0].integrity.errors

    def test_missing_archive_entry_raises(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        with pytest.raises(a.ArchiveError):
            a.restore_snapshot(
                data_through_date="2099-01-01",
                snapshot_label=None,
                season=2026,
                client=client,
                outputs_root=tmp_path / "outputs",
                artifacts_root=tmp_path / "artifacts",
            )

    def test_refuses_to_overwrite_existing_local_files(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "src-outputs", tmp_path / "src-artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")
        client = a.InMemoryArchiveClient()
        a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

        dest_outputs, dest_artifacts = tmp_path / "dst-outputs", tmp_path / "dst-artifacts"
        (dest_outputs / "2026-08-09").mkdir(parents=True)
        (dest_outputs / "2026-08-09" / "public_score.json").write_text("already here")

        with pytest.raises(a.RestoreConflictError):
            a.restore_snapshot(
                data_through_date="2026-08-09",
                snapshot_label=None,
                season=2026,
                client=client,
                outputs_root=dest_outputs,
                artifacts_root=dest_artifacts,
            )
        # Never silently overwritten.
        assert (dest_outputs / "2026-08-09" / "public_score.json").read_text() == "already here"

    def test_corrupted_archived_object_fails_hash_verification(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = tmp_path / "src-outputs", tmp_path / "src-artifacts"
        _write_snapshot(outputs_root, artifacts_root, snapshot_dir_name="2026-08-09")
        client = a.InMemoryArchiveClient()
        a.archive_snapshot(
            data_through_date="2026-08-09",
            snapshot_label=None,
            client=client,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        # Corrupt the archived output file WITHOUT updating integrity_hashes.json
        # -- simulates bit rot / a partial write on the remote side.
        client.put_object_bytes(
            "prospective/2026/2026-08-09/outputs/public_score.json", b"corrupted bytes"
        )

        with pytest.raises(a.ArchiveError):
            a.restore_snapshot(
                data_through_date="2026-08-09",
                snapshot_label=None,
                season=2026,
                client=client,
                outputs_root=tmp_path / "dst-outputs",
                artifacts_root=tmp_path / "dst-artifacts",
            )


class TestCliMain:
    def test_missing_env_vars_exits_2_with_clear_message(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("R2_BUCKET_NAME", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            monkeypatch.delenv(var, raising=False)

        exit_code = a.main(["--data-through", "2026-08-09"])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "R2_BUCKET_NAME" in captured.err
        assert "R2_ACCOUNT_ID" in captured.err

    def test_restore_without_season_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("R2_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("R2_ACCOUNT_ID", "test-account")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")

        exit_code = a.main(["--data-through", "2026-08-09", "--restore"])

        assert exit_code == 2
