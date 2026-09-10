"""`scripts/verify_local_snapshot_integrity.py` -- the post-condition that
makes the feature-only build path trustworthy.

The feature-only path (`publish_snapshot.sh --build-from-existing-snapshot`)
never scores: it restores an already-archived snapshot and builds the site
from it. Between "a directory with the right name exists" and "the exact
archived production snapshot is on disk" sits exactly this script, so its
failure modes are the interesting part -- a verifier that passes on a
half-restored snapshot is worse than no verifier, because it converts a
loud failure into a published one.

Fully offline and synthetic: every snapshot below is built by hand in a
tmp_path. No real snapshot, no network, no R2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from verify_local_snapshot_integrity import (
    SnapshotIntegrityError,
    main,
    verify_snapshot,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def snapshot(tmp_path: Path) -> tuple[str, Path, Path]:
    """A well-formed synthetic snapshot spanning both halves."""
    directory = "2026-09-08"
    outputs_root = tmp_path / "outputs" / "prospective" / "v1_1"
    artifacts_root = tmp_path / "artifacts" / "prospective" / "v1_1"

    files = {
        "outputs/public_score.json": '[{"batter_id": 1}]',
        "outputs/scorecard.json": '{"rows": 1}',
        "artifacts/manifest.json": json.dumps(
            {"data_through_date": "2026-09-08", "repository_commit": "abc123"}
        ),
    }
    recorded = {}
    for relative, text in files.items():
        half, _, tail = relative.partition("/")
        root = outputs_root if half == "outputs" else artifacts_root
        _write(root / directory / tail, text)
        recorded[relative] = hashlib.sha256(text.encode()).hexdigest()

    _write(
        artifacts_root / directory / "integrity_hashes.json",
        json.dumps(recorded, indent=2),
    )
    return directory, outputs_root, artifacts_root


class TestAGoodSnapshotVerifies:
    def test_it_accepts_a_complete_matching_snapshot(self, snapshot) -> None:
        directory, outputs_root, artifacts_root = snapshot
        report = verify_snapshot(
            directory, outputs_root=outputs_root, artifacts_root=artifacts_root
        )
        assert report["files_verified"] == 3
        assert report["data_through_date"] == "2026-09-08"
        assert report["repository_commit"] == "abc123"

    def test_it_reports_the_manifest_hash_the_dashboard_will_cite(self, snapshot) -> None:
        """`content.build_dashboard_manifest` publishes this same value as
        `snapshot_manifest_hash_reference`, so an operator can match the
        built site against the snapshot it claims to be built from."""
        directory, outputs_root, artifacts_root = snapshot
        report = verify_snapshot(
            directory, outputs_root=outputs_root, artifacts_root=artifacts_root
        )
        recorded = json.loads((artifacts_root / directory / "integrity_hashes.json").read_text())
        assert report["manifest_sha256"] == recorded["artifacts/manifest.json"]


class TestItRefusesEverythingItShould:
    def test_a_missing_snapshot_refuses(self, tmp_path: Path) -> None:
        """The archive having no snapshot for a date is an answer, not an
        obstacle -- so this must fail rather than build from nothing."""
        with pytest.raises(SnapshotIntegrityError, match="no local snapshot directory"):
            verify_snapshot(
                "2026-09-08",
                outputs_root=tmp_path / "outputs",
                artifacts_root=tmp_path / "artifacts",
            )

    def test_a_hash_mismatch_refuses(self, snapshot) -> None:
        """The core case: a file that exists but is not the archived byte
        sequence. Silently publishing it is exactly the rollback class of
        failure this path exists to prevent."""
        directory, outputs_root, artifacts_root = snapshot
        (outputs_root / directory / "public_score.json").write_text('[{"batter_id": 999}]')
        with pytest.raises(SnapshotIntegrityError, match="hash mismatch"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)

    def test_a_partially_restored_snapshot_refuses(self, snapshot) -> None:
        directory, outputs_root, artifacts_root = snapshot
        (outputs_root / directory / "scorecard.json").unlink()
        with pytest.raises(SnapshotIntegrityError, match="missing"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)

    def test_a_snapshot_with_no_integrity_file_refuses(self, snapshot) -> None:
        directory, outputs_root, artifacts_root = snapshot
        (artifacts_root / directory / "integrity_hashes.json").unlink()
        with pytest.raises(SnapshotIntegrityError, match="integrity_hashes.json is missing"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)

    def test_a_snapshot_with_no_manifest_refuses(self, snapshot) -> None:
        directory, outputs_root, artifacts_root = snapshot
        (artifacts_root / directory / "manifest.json").unlink()
        with pytest.raises(SnapshotIntegrityError, match="manifest.json is missing"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)

    def test_an_empty_integrity_file_refuses(self, snapshot) -> None:
        """Verifying zero files would 'pass' vacuously."""
        directory, outputs_root, artifacts_root = snapshot
        (artifacts_root / directory / "integrity_hashes.json").write_text("{}")
        with pytest.raises(SnapshotIntegrityError, match="records no files"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)

    def test_a_corrupt_integrity_file_refuses(self, snapshot) -> None:
        directory, outputs_root, artifacts_root = snapshot
        (artifacts_root / directory / "integrity_hashes.json").write_text("{not json")
        with pytest.raises(SnapshotIntegrityError, match="not valid JSON"):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)


class TestItNeverRepairs:
    def test_a_mismatch_leaves_every_file_exactly_as_it_found_it(self, snapshot) -> None:
        """'Repair' here would mean deciding which of two disagreeing copies
        is production. A build step does not get to decide that."""
        directory, outputs_root, artifacts_root = snapshot
        target = outputs_root / directory / "public_score.json"
        target.write_text('[{"batter_id": 999}]')
        before = {
            p: p.read_bytes()
            for p in sorted(outputs_root.rglob("*")) + sorted(artifacts_root.rglob("*"))
            if p.is_file()
        }
        with pytest.raises(SnapshotIntegrityError):
            verify_snapshot(directory, outputs_root=outputs_root, artifacts_root=artifacts_root)
        after = {
            p: p.read_bytes()
            for p in sorted(outputs_root.rglob("*")) + sorted(artifacts_root.rglob("*"))
            if p.is_file()
        }
        assert before == after


class TestTheCommandLine:
    def test_it_exits_zero_on_a_good_snapshot(self, snapshot, capsys) -> None:
        directory, outputs_root, artifacts_root = snapshot
        code = main(
            [
                "--data-through",
                "2026-09-08",
                "--outputs-root",
                str(outputs_root),
                "--artifacts-root",
                str(artifacts_root),
            ]
        )
        assert code == 0
        assert "files_verified=3" in capsys.readouterr().out

    def test_it_exits_nonzero_on_a_bad_snapshot(self, snapshot, capsys) -> None:
        """The shell path is `set -e`, so a non-zero exit is what actually
        stops the build."""
        directory, outputs_root, artifacts_root = snapshot
        (outputs_root / directory / "public_score.json").write_text("tampered")
        code = main(
            [
                "--data-through",
                "2026-09-08",
                "--outputs-root",
                str(outputs_root),
                "--artifacts-root",
                str(artifacts_root),
            ]
        )
        assert code == 2
        assert "hash mismatch" in capsys.readouterr().err

    def test_a_snapshot_label_selects_the_labelled_directory(self, tmp_path: Path) -> None:
        outputs_root = tmp_path / "outputs"
        artifacts_root = tmp_path / "artifacts"
        directory = "2026-08-08__refreshed"
        text = '{"data_through_date": "2026-08-08"}'
        _write(artifacts_root / directory / "manifest.json", text)
        _write(
            artifacts_root / directory / "integrity_hashes.json",
            json.dumps({"artifacts/manifest.json": hashlib.sha256(text.encode()).hexdigest()}),
        )
        code = main(
            [
                "--data-through",
                "2026-08-08",
                "--snapshot-label",
                "refreshed",
                "--outputs-root",
                str(outputs_root),
                "--artifacts-root",
                str(artifacts_root),
            ]
        )
        assert code == 0
