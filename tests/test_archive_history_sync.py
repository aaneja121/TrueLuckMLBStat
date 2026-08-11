"""Tests for scripts/archive_snapshot.py's history-sync capability --
list_archived_snapshots(), sync_missing_snapshots(), and the end-to-end
scenario this exists to fix: a disposable CI runner that starts with only
today's snapshot locally must be able to repopulate the rest of the
season's history from R2 before the dashboard is built, or its
season-to-date trend charts would show a single point forever. Every test
uses InMemoryArchiveClient -- none contacts real Cloudflare R2, none scores
real MLB data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import archive_snapshot as a
import pytest


def _write_local_snapshot(
    outputs_root: Path,
    artifacts_root: Path,
    *,
    snapshot_dir_name: str,
    season: int = 2026,
    content: bytes | None = None,
) -> None:
    out_dir = outputs_root / snapshot_dir_name
    art_dir = artifacts_root / snapshot_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    content = content if content is not None else f"data for {snapshot_dir_name}".encode()

    (out_dir / "public_score.json").write_bytes(content)
    (art_dir / "manifest.json").write_text(json.dumps({"prospective_season": season}))
    integrity = {
        "outputs/public_score.json": hashlib.sha256(content).hexdigest(),
        "artifacts/manifest.json": hashlib.sha256(
            (art_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity, indent=2))


def _archive(
    client: a.InMemoryArchiveClient,
    outputs_root: Path,
    artifacts_root: Path,
    snapshot_dir_name: str,
    *,
    season: int = 2026,
    content: bytes | None = None,
) -> None:
    """Write a local snapshot, archive it, and leave nothing local -- most
    tests below want a client pre-populated with archive data but an empty
    (or selectively populated) local disk, mimicking a disposable CI runner
    that has never seen these dates.
    """
    _write_local_snapshot(
        outputs_root,
        artifacts_root,
        snapshot_dir_name=snapshot_dir_name,
        season=season,
        content=content,
    )
    data_through, label = a.split_snapshot_dir_name(snapshot_dir_name)
    a.archive_snapshot(
        data_through_date=data_through,
        snapshot_label=label,
        client=client,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )


class TestListArchivedSnapshots:
    def test_lists_multiple_archived_dates(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        for name in ["2026-08-05", "2026-08-06", "2026-08-09"]:
            _archive(client, maint_out, maint_art, name)

        infos = a.list_archived_snapshots(client, season=2026)
        assert sorted(i.snapshot_dir_name for i in infos) == [
            "2026-08-05",
            "2026-08-06",
            "2026-08-09",
        ]
        assert all(i.complete for i in infos)

    def test_labeled_corrected_snapshot_discovered_as_its_own_entry(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-08", content=b"original")
        _archive(client, maint_out, maint_art, "2026-08-08__refreshed", content=b"corrected")

        infos = {i.snapshot_dir_name: i for i in a.list_archived_snapshots(client, season=2026)}
        assert set(infos) == {"2026-08-08", "2026-08-08__refreshed"}
        assert infos["2026-08-08"].complete
        assert infos["2026-08-08__refreshed"].complete

    def test_missing_integrity_hashes_is_incomplete_not_silently_valid(
        self, tmp_path: Path
    ) -> None:
        client = a.InMemoryArchiveClient()
        # Only an outputs file was ever uploaded -- no artifacts/integrity_hashes.json at all.
        client.put_object_bytes("prospective/2026/2026-08-07/outputs/public_score.json", b"partial")

        infos = a.list_archived_snapshots(client, season=2026)
        assert len(infos) == 1
        assert infos[0].snapshot_dir_name == "2026-08-07"
        assert infos[0].complete is False
        assert "integrity_hashes.json" in (infos[0].problem or "")

    def test_integrity_hashes_referencing_a_missing_file_is_incomplete(
        self, tmp_path: Path
    ) -> None:
        client = a.InMemoryArchiveClient()
        # integrity_hashes.json claims a file that was never actually uploaded.
        client.put_object_bytes(
            "prospective/2026/2026-08-07/artifacts/integrity_hashes.json",
            json.dumps({"outputs/public_score.json": "deadbeef"}).encode(),
        )
        infos = a.list_archived_snapshots(client, season=2026)
        assert len(infos) == 1
        assert infos[0].complete is False
        assert "missing archived file" in (infos[0].problem or "")

    def test_malformed_json_integrity_hashes_is_incomplete(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        client.put_object_bytes(
            "prospective/2026/2026-08-07/artifacts/integrity_hashes.json", b"{not valid json"
        )
        infos = a.list_archived_snapshots(client, season=2026)
        assert infos[0].complete is False
        assert "not valid JSON" in (infos[0].problem or "")

    def test_different_season_is_never_conflated(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-05", season=2026)

        assert a.list_archived_snapshots(client, season=2027) == []

    def test_unrelated_key_shape_is_ignored_not_treated_as_a_snapshot(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        # Doesn't reach <snapshot_dir_name>/<category>/<file> depth.
        client.put_object_bytes("prospective/2026/README.txt", b"not a snapshot")
        assert a.list_archived_snapshots(client, season=2026) == []

    def test_full_depth_key_with_non_date_shaped_name_is_ignored(self, tmp_path: Path) -> None:
        """Reaches the right depth (<name>/<category>/<file>), but <name>
        doesn't look like a snapshot directory at all -- must be treated as
        genuinely unrelated, not reported as an incomplete snapshot.
        """
        client = a.InMemoryArchiveClient()
        client.put_object_bytes("prospective/2026/not-a-real-date/outputs/whatever.json", b"x")
        assert a.list_archived_snapshots(client, season=2026) == []

    def test_date_shaped_but_invalid_calendar_date_is_ignored(self, tmp_path: Path) -> None:
        """2026-13-40 has the right SHAPE (digits-digits-digits) but isn't a
        real calendar date -- also treated as unrelated, not a broken
        snapshot.
        """
        client = a.InMemoryArchiveClient()
        client.put_object_bytes("prospective/2026/2026-13-40/outputs/whatever.json", b"x")
        assert a.list_archived_snapshots(client, season=2026) == []


class TestSyncMissingSnapshots:
    def test_restores_every_missing_snapshot(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        for name in ["2026-08-05", "2026-08-06", "2026-08-08__refreshed"]:
            _archive(client, maint_out, maint_art, name)

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"  # nothing local yet
        result = a.sync_missing_snapshots(
            season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
        )

        assert sorted(result.restored) == ["2026-08-05", "2026-08-06", "2026-08-08__refreshed"]
        assert result.already_present == ()
        for name in result.restored:
            assert (ci_out / name / "public_score.json").exists()
            assert (ci_art / name / "integrity_hashes.json").exists()

    def test_already_valid_local_snapshot_is_a_clean_no_op(self, tmp_path: Path) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-09")

        # CI already scored today's date locally with IDENTICAL content.
        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        _write_local_snapshot(ci_out, ci_art, snapshot_dir_name="2026-08-09")

        before = (ci_out / "2026-08-09" / "public_score.json").read_bytes()
        result = a.sync_missing_snapshots(
            season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
        )

        assert result.restored == ()
        assert result.already_present == ("2026-08-09",)
        after = (ci_out / "2026-08-09" / "public_score.json").read_bytes()
        assert before == after, "already-present snapshot must never be rewritten"

    def test_conflicting_local_snapshot_fails_loudly_and_is_never_touched(
        self, tmp_path: Path
    ) -> None:
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-05", content=b"archived version")

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        _write_local_snapshot(
            ci_out, ci_art, snapshot_dir_name="2026-08-05", content=b"DIFFERENT local version"
        )

        with pytest.raises(a.HistorySyncConflictError):
            a.sync_missing_snapshots(
                season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
            )

        assert (
            ci_out / "2026-08-05" / "public_score.json"
        ).read_bytes() == b"DIFFERENT local version"

    def test_local_snapshot_missing_integrity_hashes_is_a_conflict_not_a_repair_target(
        self, tmp_path: Path
    ) -> None:
        """A local directory that exists but is missing integrity_hashes.json
        (partial/corrupt) must fail loudly, never be silently "repaired" by
        overwriting it with the archived copy.
        """
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-05")

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        (ci_out / "2026-08-05").mkdir(parents=True)
        (ci_out / "2026-08-05" / "public_score.json").write_bytes(
            b"partial, no integrity_hashes.json"
        )
        (ci_art / "2026-08-05").mkdir(
            parents=True
        )  # exists but genuinely empty -- no integrity_hashes.json

        with pytest.raises(a.HistorySyncConflictError):
            a.sync_missing_snapshots(
                season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
            )

    def test_malformed_valid_shaped_snapshot_raises_and_is_never_restored(
        self, tmp_path: Path
    ) -> None:
        """A candidate whose NAME looks like a real snapshot (2026-08-07)
        but is incomplete/malformed must FAIL the whole sync, not be
        silently skipped -- it's a broken OFFICIAL entry, not unrelated
        noise. Entries alphabetically before it may already have been
        restored (sync is not required to be atomic -- same non-atomic
        fail-fast precedent as HistorySyncConflictError), but the broken
        entry itself must never be restored.
        """
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        _archive(client, maint_out, maint_art, "2026-08-05")
        # A broken entry, sorted after the good one above.
        client.put_object_bytes(
            "prospective/2026/2026-08-07/outputs/public_score.json", b"partial upload"
        )

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        with pytest.raises(a.HistorySyncIncompleteArchiveError, match="2026-08-07"):
            a.sync_missing_snapshots(
                season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
            )

        assert (ci_out / "2026-08-05").exists(), "entries before the broken one may still land"
        assert not (ci_out / "2026-08-07").exists(), "the broken entry itself is never restored"

    def test_unrelated_key_does_not_block_sync_of_valid_snapshots(self, tmp_path: Path) -> None:
        """Unlike a broken OFFICIAL entry (above), a key that never even
        looks like a snapshot directory must be silently ignored and must
        NOT prevent genuinely valid snapshots from syncing.
        """
        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        for name in ["2026-08-05", "2026-08-06"]:
            _archive(client, maint_out, maint_art, name)
        client.put_object_bytes("prospective/2026/not-a-real-date/outputs/whatever.json", b"x")

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        result = a.sync_missing_snapshots(
            season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
        )

        assert sorted(result.restored) == ["2026-08-05", "2026-08-06"]
        assert not (ci_out / "not-a-real-date").exists()

    def test_restored_snapshots_pass_the_real_dashboard_integrity_validator(
        self, tmp_path: Path
    ) -> None:
        """Uses the SAME full snapshot fixture the dashboard test suite
        validates against, exactly like
        test_archive_snapshot.py::test_restore_passes_the_dashboards_own_integrity_check,
        so this exercises the real validator faithfully after a multi-snapshot sync.
        """
        import snapshot_data as sd

        from dashboard_snapshot_fixtures import default_player_record
        from dashboard_snapshot_fixtures import write_snapshot as write_full_snapshot

        client = a.InMemoryArchiveClient()
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        for day in ("2026-08-05", "2026-08-06"):
            write_full_snapshot(
                maint_out,
                maint_art,
                directory_name=day,
                data_through_date=day,
                snapshot_label=None,
                generated_at=f"{day}T00:00:00+00:00",
                players=[
                    default_player_record(
                        batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                    )
                ],
            )
            data_through, label = a.split_snapshot_dir_name(day)
            a.archive_snapshot(
                data_through_date=data_through,
                snapshot_label=label,
                client=client,
                outputs_root=maint_out,
                artifacts_root=maint_art,
            )

        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        a.sync_missing_snapshots(
            season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
        )

        discovered = sd.discover_snapshots(ci_out, ci_art)
        assert len(discovered) == 2
        for snap in discovered:
            assert snap.integrity.valid, (snap.directory_name, snap.integrity.errors)


class TestDashboardSeesFullHistoryAfterSync:
    def test_dashboard_discovery_sees_full_history_not_only_todays_snapshot(
        self, tmp_path: Path
    ) -> None:
        """THE representative regression scenario this whole capability
        exists for: several historical snapshots exist ONLY in the archive
        (as they would after prior CI runs whose runners were destroyed),
        plus today's snapshot was just scored locally (as a fresh CI runner
        would have it). Before sync, dashboard discovery would see only
        today. After sync, it must see the full date history.

        Uses the FULL snapshot fixture (not the minimal `_write_local_snapshot`
        helper above) specifically so this exercises
        dashboard/snapshot_data.py's real validity/precedence logic
        (build_snapshot_history, resolve_latest_snapshot) faithfully -- those
        filter on integrity.valid, which the minimal fixture's bare-bones
        manifest.json doesn't satisfy.
        """
        import snapshot_data as sd

        from dashboard_snapshot_fixtures import default_player_record
        from dashboard_snapshot_fixtures import write_snapshot as write_full_snapshot

        def archive_full(outputs_root: Path, artifacts_root: Path, snapshot_dir_name: str) -> None:
            data_through, label = a.split_snapshot_dir_name(snapshot_dir_name)
            write_full_snapshot(
                outputs_root,
                artifacts_root,
                directory_name=snapshot_dir_name,
                data_through_date=data_through,
                snapshot_label=label,
                generated_at=f"{data_through}T00:00:00+00:00",
                players=[
                    default_player_record(
                        batter_id=1, batter_name="Alice", score=2.0, lower=-1.0, upper=5.0
                    )
                ],
            )
            a.archive_snapshot(
                data_through_date=data_through,
                snapshot_label=label,
                client=client,
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )

        client = a.InMemoryArchiveClient()

        # Simulate several PRIOR days' snapshots that exist only in the
        # durable archive (their local copies are long gone -- e.g. each
        # was produced by a different, now-destroyed CI runner).
        maint_out, maint_art = tmp_path / "maint-out", tmp_path / "maint-art"
        historical_dates = ["2026-08-05", "2026-08-06", "2026-08-08__refreshed"]
        for name in historical_dates:
            archive_full(maint_out, maint_art, name)

        # Simulate today's disposable CI runner: freshly checked out repo,
        # today's snapshot was just scored and archived, nothing else local.
        ci_out, ci_art = tmp_path / "ci-out", tmp_path / "ci-art"
        archive_full(ci_out, ci_art, "2026-08-09")  # archives AND leaves it local, like a real run

        # Sanity check: before sync, CI only has today.
        before_sync = sd.discover_snapshots(ci_out, ci_art)
        assert [s.directory_name for s in before_sync] == ["2026-08-09"]

        sync_result = a.sync_missing_snapshots(
            season=2026, client=client, outputs_root=ci_out, artifacts_root=ci_art
        )
        assert sorted(sync_result.restored) == sorted(historical_dates)

        after_sync = sd.discover_snapshots(ci_out, ci_art)
        after_dates = sorted(s.directory_name for s in after_sync)
        assert after_dates == sorted([*historical_dates, "2026-08-09"])
        assert all(s.integrity.valid for s in after_sync), [
            (s.directory_name, s.integrity.errors) for s in after_sync
        ]

        # And the dashboard's own latest-snapshot/history resolution logic
        # (not just raw discovery) sees the complete, correctly-precedented
        # series -- 2026-08-08 is superseded by 2026-08-08__refreshed, and
        # the latest overall is still 2026-08-09.
        history = sd.build_snapshot_history(after_sync)
        history_dates = [entry.data_through_date for entry in history]
        assert history_dates == ["2026-08-09", "2026-08-08", "2026-08-06", "2026-08-05"]
        preferred_for_08 = next(e for e in history if e.data_through_date == "2026-08-08").preferred
        assert preferred_for_08 is not None
        assert preferred_for_08.directory_name == "2026-08-08__refreshed"

        latest = sd.resolve_latest_snapshot(after_sync)
        assert latest is not None
        assert latest.directory_name == "2026-08-09"
