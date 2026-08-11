"""Tests for scripts/ensure_frozen_inputs.py -- the portability fix that
lets a fresh CI runner obtain every gitignored frozen artifact Version 1.1
prospective scoring's manifest generation requires (the frozen 2021-2024
development parquet plus three model-comparison detail JSON files) without
ever regenerating any of them. All tests use `InMemoryArchiveClient`
(imported from `scripts.archive_snapshot`); none contact real Cloudflare
R2, none require real credentials, and none touch the real frozen files on
this machine's disk (except the read-only cross-checks in
`TestBundleCoversEveryGitignoredFrozenArtifact`, which only ever read,
never write) -- every synthetic test uses tiny bytes and its own
`tmp_path`-scoped local path/hash pair.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

import ensure_frozen_inputs as efi
import pytest
from archive_snapshot import InMemoryArchiveClient

SCRIPTS_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CONTENT = b"synthetic frozen development input bytes, not the real 106MB parquet"
CONTENT_HASH = _sha256(CONTENT)
OTHER_CONTENT = b"a completely different file that happens to sit at the same key"
OTHER_HASH = _sha256(OTHER_CONTENT)


class TestValidLocalArtifactReuse:
    def test_reuses_local_file_when_hash_matches_without_any_client(self, tmp_path: Path) -> None:
        local_path = tmp_path / "cleaned_development_data_with_sprint_speed.parquet"
        local_path.write_bytes(CONTENT)

        def _fail_if_called() -> InMemoryArchiveClient:
            raise AssertionError("client_factory must never be called when local file is valid")

        result = efi.ensure_frozen_development_input(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key="frozen-inputs/v1/irrelevant.parquet",
            client_factory=_fail_if_called,
        )

        assert result.outcome == efi.FrozenInputOutcome.REUSED_LOCAL
        assert result.path == local_path
        assert result.sha256 == CONTENT_HASH
        assert result.size_bytes == len(CONTENT)
        # File is byte-for-byte untouched.
        assert local_path.read_bytes() == CONTENT


class TestMissingLocalArtifactFetchedFromFakeR2:
    def test_downloads_from_r2_when_local_file_is_missing(self, tmp_path: Path) -> None:
        local_path = tmp_path / "does_not_exist_yet.parquet"
        r2_key = "frozen-inputs/v1/cleaned_development_data_with_sprint_speed.parquet"

        client = InMemoryArchiveClient()
        client.put_object_bytes(r2_key, CONTENT)

        result = efi.ensure_frozen_development_input(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key=r2_key,
            client_factory=lambda: client,
        )

        assert result.outcome == efi.FrozenInputOutcome.DOWNLOADED_FROM_R2
        assert local_path.exists()
        assert local_path.read_bytes() == CONTENT

    def test_downloads_into_a_local_directory_that_does_not_exist_yet(self, tmp_path: Path) -> None:
        local_path = tmp_path / "nested" / "does_not_exist" / "input.parquet"
        r2_key = "frozen-inputs/v1/input.parquet"
        client = InMemoryArchiveClient()
        client.put_object_bytes(r2_key, CONTENT)

        result = efi.ensure_frozen_development_input(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key=r2_key,
            client_factory=lambda: client,
        )
        assert result.outcome == efi.FrozenInputOutcome.DOWNLOADED_FROM_R2
        assert local_path.read_bytes() == CONTENT


class TestDownloadedHashVerification:
    def test_result_sha256_and_size_match_the_pinned_hash_and_real_content(
        self, tmp_path: Path
    ) -> None:
        local_path = tmp_path / "input.parquet"
        r2_key = "frozen-inputs/v1/input.parquet"
        client = InMemoryArchiveClient()
        client.put_object_bytes(r2_key, CONTENT)

        result = efi.ensure_frozen_development_input(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key=r2_key,
            client_factory=lambda: client,
        )
        assert result.sha256 == CONTENT_HASH
        assert result.size_bytes == len(CONTENT)


class TestLocalHashMismatchFails:
    def test_raises_and_never_touches_the_mismatched_local_file(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(OTHER_CONTENT)  # wrong content already sitting here

        def _fail_if_called() -> InMemoryArchiveClient:
            raise AssertionError("must never contact R2 when a local file already exists")

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            efi.ensure_frozen_development_input(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,  # pinned hash, does NOT match OTHER_CONTENT
                r2_key="frozen-inputs/v1/irrelevant.parquet",
                client_factory=_fail_if_called,
            )

        # Never silently overwritten, never silently redownloaded over.
        assert local_path.read_bytes() == OTHER_CONTENT

    def test_never_falls_back_to_downloading_after_a_local_mismatch(self, tmp_path: Path) -> None:
        """Even if a client WOULD have the correct file available, a local
        mismatch must fail loudly rather than silently prefer the remote
        copy -- the human needs to know a stale/wrong file was sitting
        there, not have it quietly papered over.
        """
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(OTHER_CONTENT)
        r2_key = "frozen-inputs/v1/input.parquet"
        client = InMemoryArchiveClient()
        client.put_object_bytes(r2_key, CONTENT)  # the "correct" content, available remotely

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            efi.ensure_frozen_development_input(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key=r2_key,
                client_factory=lambda: client,
            )
        assert local_path.read_bytes() == OTHER_CONTENT


class TestRemoteHashMismatchFails:
    def test_raises_and_does_not_write_a_local_file(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        r2_key = "frozen-inputs/v1/input.parquet"
        client = InMemoryArchiveClient()
        client.put_object_bytes(r2_key, OTHER_CONTENT)  # wrong content at the pinned key

        with pytest.raises(efi.FrozenInputRemoteHashMismatchError):
            efi.ensure_frozen_development_input(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key=r2_key,
                client_factory=lambda: client,
            )

        # A corrupted/wrong download must never be left on disk looking
        # like a valid, verified artifact.
        assert not local_path.exists()


class TestMissingRemoteObjectFails:
    def test_raises_when_neither_local_nor_remote_has_the_file(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        client = InMemoryArchiveClient()  # empty -- nothing uploaded yet

        with pytest.raises(efi.FrozenInputMissingRemoteObjectError):
            efi.ensure_frozen_development_input(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key="frozen-inputs/v1/input.parquet",
                client_factory=lambda: client,
            )
        assert not local_path.exists()


class TestNoRegenerationFallback:
    """Structural + behavioral guarantee that this module can NEVER
    produce the frozen input by regenerating/resynthesizing it from raw
    data -- only reuse-if-valid or fetch-and-verify-from-R2.
    """

    def test_module_imports_no_cleaning_download_or_eligibility_code(self) -> None:
        tree = ast.parse((SCRIPTS_SOURCE_ROOT / "ensure_frozen_inputs.py").read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        banned = {
            "mlb_luck_score",  # the model-training/scoring/cleaning package
            "prospective_scoring",
            "prospective_ingestion",  # performs the real Statcast download
            "pybaseball",  # the Statcast download library itself
            "clean_development_data",
            "clean_batted_balls",
        }
        violations = imported & banned
        assert not violations, f"ensure_frozen_inputs.py must never import: {violations}"

    def test_the_only_two_possible_success_outcomes_are_reuse_or_verified_download(self) -> None:
        assert set(efi.FrozenInputOutcome) == {
            efi.FrozenInputOutcome.REUSED_LOCAL,
            efi.FrozenInputOutcome.DOWNLOADED_FROM_R2,
        }

    def test_missing_everywhere_raises_rather_than_returning_any_result(
        self, tmp_path: Path
    ) -> None:
        local_path = tmp_path / "input.parquet"
        client = InMemoryArchiveClient()
        with pytest.raises(efi.FrozenInputError):
            efi.ensure_frozen_development_input(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key="frozen-inputs/v1/input.parquet",
                client_factory=lambda: client,
            )
        assert not local_path.exists()  # no file conjured from nowhere


class TestUploadSeedingAction:
    """The separate, never-automatically-invoked one-time upload path."""

    def test_uploads_when_local_matches_pin_and_remote_is_empty(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(CONTENT)
        client = InMemoryArchiveClient()

        outcome = efi.upload_frozen_input_to_r2(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key="frozen-inputs/v1/input.parquet",
            client=client,
        )
        assert outcome == efi.FrozenInputUploadOutcome.UPLOADED
        assert client.get_object_bytes("frozen-inputs/v1/input.parquet") == CONTENT

    def test_refuses_to_upload_a_local_file_that_does_not_match_the_pin(
        self, tmp_path: Path
    ) -> None:
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(OTHER_CONTENT)
        client = InMemoryArchiveClient()

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            efi.upload_frozen_input_to_r2(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key="frozen-inputs/v1/input.parquet",
                client=client,
            )
        assert client.list_keys("frozen-inputs/") == []

    def test_no_op_when_identical_object_already_archived(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(CONTENT)
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/input.parquet", CONTENT)

        outcome = efi.upload_frozen_input_to_r2(
            local_path=local_path,
            expected_sha256=CONTENT_HASH,
            r2_key="frozen-inputs/v1/input.parquet",
            client=client,
        )
        assert outcome == efi.FrozenInputUploadOutcome.NO_OP_IDENTICAL

    def test_refuses_to_overwrite_a_different_existing_remote_object(self, tmp_path: Path) -> None:
        local_path = tmp_path / "input.parquet"
        local_path.write_bytes(CONTENT)
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/input.parquet", OTHER_CONTENT)

        with pytest.raises(efi.FrozenInputUploadConflictError):
            efi.upload_frozen_input_to_r2(
                local_path=local_path,
                expected_sha256=CONTENT_HASH,
                r2_key="frozen-inputs/v1/input.parquet",
                client=client,
            )
        # Never overwritten -- the original, different object survives.
        assert client.get_object_bytes("frozen-inputs/v1/input.parquet") == OTHER_CONTENT


class TestPinnedConfig:
    def test_r2_key_uses_a_prefix_separate_from_the_snapshot_archive(self) -> None:
        assert efi.FROZEN_INPUT_R2_KEY.startswith("frozen-inputs/")
        assert not efi.FROZEN_INPUT_R2_KEY.startswith("prospective/")

    def test_local_path_matches_the_real_pipelines_frozen_input_path(self) -> None:
        # FROZEN_INPUT_LOCAL_PATH is a backward-compatible alias for
        # FROZEN_INPUT_BUNDLE[0] (the parquet entry) -- this assertion
        # confirms that alias still points at the right file, not a
        # different bundle entry, after the bundle refactor.
        assert (
            efi.FROZEN_INPUT_LOCAL_PATH.name == "cleaned_development_data_with_sprint_speed.parquet"
        )
        assert efi.FROZEN_INPUT_LOCAL_PATH.parent.name == "processed"

    def test_pinned_hash_is_a_well_formed_sha256_hex_digest(self) -> None:
        assert len(efi.FROZEN_INPUT_SHA256) == 64
        int(efi.FROZEN_INPUT_SHA256, 16)  # raises ValueError if not valid hex

    def test_every_bundle_entry_has_a_well_formed_sha256_and_a_frozen_inputs_key(self) -> None:
        for spec in efi.FROZEN_INPUT_BUNDLE:
            assert len(spec.sha256) == 64
            int(spec.sha256, 16)  # raises ValueError if not valid hex
            assert spec.r2_key.startswith("frozen-inputs/v1/")
            assert spec.size_bytes > 0

    def test_bundle_has_exactly_four_entries_matching_the_known_incident(self) -> None:
        # Not a hard architectural limit -- just documents the exact
        # incident this bundle was built to close (parquet + the 3 detail
        # JSONs the second dry-run failed on) so a future addition/removal
        # here is a deliberate, visible change to this test, not silent.
        local_paths = {spec.local_relative_path for spec in efi.FROZEN_INPUT_BUNDLE}
        assert local_paths == {
            "data/processed/cleaned_development_data_with_sprint_speed.parquet",
            "outputs/tables/opportunity_model_comparison_detail.json",
            "outputs/tables/infield_opportunity_detail.json",
            "outputs/tables/advancement_detail.json",
        }


class TestBundleReuseLocally:
    def test_complete_bundle_reused_locally_without_any_client(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec(
                "nested/b.json", "frozen-inputs/v1/nested/b.json", _sha256(b"BBBB"), 4
            ),
        )
        (tmp_path / "a.bin").write_bytes(b"AAA")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "b.json").write_bytes(b"BBBB")

        def _fail_if_called() -> InMemoryArchiveClient:
            raise AssertionError("must never contact R2 when every local file is already valid")

        results = efi.ensure_frozen_input_bundle(
            specs, project_root=tmp_path, client_factory=_fail_if_called
        )

        assert len(results) == 2
        assert all(r.outcome == efi.FrozenInputOutcome.REUSED_LOCAL for r in results)
        assert results[0].sha256 == _sha256(b"AAA")
        assert results[1].sha256 == _sha256(b"BBBB")

    def test_the_real_local_bundle_is_fully_reused_with_zero_r2_contact(self) -> None:
        """Exercises the DEFAULT bundle (real project paths) against
        whatever is actually on this machine's disk right now -- skipped
        (not failed) if any entry is missing, since that's a valid state
        on a fresh checkout before the one-time seed/first `ensure` run.
        """
        for spec in efi.FROZEN_INPUT_BUNDLE:
            if not (efi.PROJECT_ROOT / spec.local_relative_path).exists():
                pytest.skip("real frozen-input bundle not fully present on this machine")

        def _fail_if_called() -> InMemoryArchiveClient:
            raise AssertionError("must never contact R2 when every local file is already valid")

        results = efi.ensure_frozen_input_bundle(client_factory=_fail_if_called)
        assert len(results) == len(efi.FROZEN_INPUT_BUNDLE)
        assert all(r.outcome == efi.FrozenInputOutcome.REUSED_LOCAL for r in results)


class TestBundleMultipleMissingFetchedCorrectly:
    def test_fetches_every_missing_entry_from_r2(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec(
                "nested/b.json", "frozen-inputs/v1/nested/b.json", _sha256(b"BBBB"), 4
            ),
        )
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"AAA")
        client.put_object_bytes("frozen-inputs/v1/nested/b.json", b"BBBB")

        results = efi.ensure_frozen_input_bundle(
            specs, project_root=tmp_path, client_factory=lambda: client
        )

        assert all(r.outcome == efi.FrozenInputOutcome.DOWNLOADED_FROM_R2 for r in results)
        assert (tmp_path / "a.bin").read_bytes() == b"AAA"
        assert (tmp_path / "nested" / "b.json").read_bytes() == b"BBBB"

    def test_fetches_only_the_missing_entries_and_reuses_the_rest(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        (tmp_path / "a.bin").write_bytes(b"AAA")  # already present locally
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/b.bin", b"BBB")  # only b needs fetching

        results = efi.ensure_frozen_input_bundle(
            specs, project_root=tmp_path, client_factory=lambda: client
        )
        assert results[0].outcome == efi.FrozenInputOutcome.REUSED_LOCAL
        assert results[1].outcome == efi.FrozenInputOutcome.DOWNLOADED_FROM_R2

    def test_shares_exactly_one_client_across_the_whole_bundle(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"AAA")
        client.put_object_bytes("frozen-inputs/v1/b.bin", b"BBB")

        call_count = 0

        def factory() -> InMemoryArchiveClient:
            nonlocal call_count
            call_count += 1
            return client

        efi.ensure_frozen_input_bundle(specs, project_root=tmp_path, client_factory=factory)
        assert call_count == 1


class TestBundleOneMissingRemoteArtifactFails:
    def test_stops_at_the_first_missing_remote_entry(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"AAA")
        # b.bin was never uploaded.

        with pytest.raises(efi.FrozenInputMissingRemoteObjectError):
            efi.ensure_frozen_input_bundle(
                specs, project_root=tmp_path, client_factory=lambda: client
            )

        assert (tmp_path / "a.bin").exists()  # the earlier entry still succeeded
        assert not (tmp_path / "b.bin").exists()


class TestBundleOneCorruptedLocalArtifactFails:
    def test_stops_at_the_first_corrupted_local_entry_before_any_r2_contact(
        self, tmp_path: Path
    ) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        (tmp_path / "a.bin").write_bytes(b"AAA")
        (tmp_path / "b.bin").write_bytes(b"WRONG")  # corrupted/stale

        def _fail_if_called() -> InMemoryArchiveClient:
            raise AssertionError("a corrupted local entry must fail before any R2 contact")

        with pytest.raises(efi.FrozenInputLocalHashMismatchError):
            efi.ensure_frozen_input_bundle(
                specs, project_root=tmp_path, client_factory=_fail_if_called
            )
        assert (tmp_path / "b.bin").read_bytes() == b"WRONG"  # never overwritten


class TestBundleOneCorruptedRemoteArtifactFails:
    def test_stops_at_the_first_corrupted_remote_entry(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"AAA")
        client.put_object_bytes("frozen-inputs/v1/b.bin", b"CORRUPTED")

        with pytest.raises(efi.FrozenInputRemoteHashMismatchError):
            efi.ensure_frozen_input_bundle(
                specs, project_root=tmp_path, client_factory=lambda: client
            )
        assert (tmp_path / "a.bin").exists()
        assert not (tmp_path / "b.bin").exists()  # corrupted download never written


class TestBundleCoversEveryGitignoredFrozenArtifact:
    """THE structural regression test this task asks for: a future frozen
    artifact added to `evaluation.v1_final_evaluation_manifest.FROZEN_
    ARTIFACT_RELATIVE_PATHS` (the exact list `prospective_manifest.
    build_snapshot_manifest` hashes) without also being added to `FROZEN_
    INPUT_BUNDLE` here must fail this test -- exactly the gap that caused
    the real second-dry-run incident (three detail JSONs were in the
    frozen-artifact list but not yet covered by this module).
    """

    def test_every_gitignored_frozen_artifact_path_is_in_the_bundle(self) -> None:
        from v1_final_evaluation_manifest import FROZEN_ARTIFACT_RELATIVE_PATHS

        result = subprocess.run(
            ["git", "check-ignore", *FROZEN_ARTIFACT_RELATIVE_PATHS],
            cwd=efi.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,  # exit 1 just means "none of these are ignored" -- not an error
        )
        gitignored = {line for line in result.stdout.splitlines() if line}
        bundle_paths = {spec.local_relative_path for spec in efi.FROZEN_INPUT_BUNDLE}
        uncovered = gitignored - bundle_paths
        assert not uncovered, (
            "gitignored frozen artifact(s) in FROZEN_ARTIFACT_RELATIVE_PATHS not covered by "
            f"scripts/ensure_frozen_inputs.py's FROZEN_INPUT_BUNDLE: {sorted(uncovered)}"
        )

    def test_bundle_has_no_stale_entries_outside_the_real_frozen_artifact_list(self) -> None:
        """The reverse direction -- a bundle entry no longer part of the
        real frozen-artifact list is dead weight, not a safety hole, but
        worth catching so the bundle stays an accurate mirror.
        """
        from v1_final_evaluation_manifest import FROZEN_ARTIFACT_RELATIVE_PATHS

        frozen_paths = set(FROZEN_ARTIFACT_RELATIVE_PATHS)
        bundle_paths = {spec.local_relative_path for spec in efi.FROZEN_INPUT_BUNDLE}
        stale = bundle_paths - frozen_paths
        assert not stale, f"bundle entry no longer in FROZEN_ARTIFACT_RELATIVE_PATHS: {stale}"

    def test_every_bundle_entrys_pinned_hash_matches_the_real_local_file(self) -> None:
        """Since the real files are present on a machine that has run the
        production pipeline, directly cross-check every pinned hash
        against reality -- catches a transcription error in the bundle
        immediately, not only at ensure-time on some other machine.
        Skips (never fails) an entry that isn't present here.
        """
        for spec in efi.FROZEN_INPUT_BUNDLE:
            full_path = efi.PROJECT_ROOT / spec.local_relative_path
            if not full_path.exists():
                continue
            actual = hashlib.sha256(full_path.read_bytes()).hexdigest()
            assert actual == spec.sha256, f"{spec.local_relative_path}: pinned hash is stale"
            assert full_path.stat().st_size == spec.size_bytes


class TestUploadBundleSeedingAction:
    def test_uploads_every_entry_and_reports_per_key_outcome(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        (tmp_path / "a.bin").write_bytes(b"AAA")
        (tmp_path / "b.bin").write_bytes(b"BBB")
        client = InMemoryArchiveClient()

        results = efi.upload_frozen_input_bundle_to_r2(specs, project_root=tmp_path, client=client)

        assert results == (
            ("frozen-inputs/v1/a.bin", efi.FrozenInputUploadOutcome.UPLOADED),
            ("frozen-inputs/v1/b.bin", efi.FrozenInputUploadOutcome.UPLOADED),
        )
        assert client.get_object_bytes("frozen-inputs/v1/a.bin") == b"AAA"
        assert client.get_object_bytes("frozen-inputs/v1/b.bin") == b"BBB"

    def test_rerunning_over_a_fully_seeded_bundle_is_all_no_ops(self, tmp_path: Path) -> None:
        specs = (efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),)
        (tmp_path / "a.bin").write_bytes(b"AAA")
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"AAA")

        results = efi.upload_frozen_input_bundle_to_r2(specs, project_root=tmp_path, client=client)
        assert results == (
            ("frozen-inputs/v1/a.bin", efi.FrozenInputUploadOutcome.NO_OP_IDENTICAL),
        )

    def test_stops_at_the_first_upload_conflict(self, tmp_path: Path) -> None:
        specs = (
            efi.FrozenInputSpec("a.bin", "frozen-inputs/v1/a.bin", _sha256(b"AAA"), 3),
            efi.FrozenInputSpec("b.bin", "frozen-inputs/v1/b.bin", _sha256(b"BBB"), 3),
        )
        (tmp_path / "a.bin").write_bytes(b"AAA")
        (tmp_path / "b.bin").write_bytes(b"BBB")
        client = InMemoryArchiveClient()
        client.put_object_bytes("frozen-inputs/v1/a.bin", b"SOMETHING ELSE ENTIRELY")

        with pytest.raises(efi.FrozenInputUploadConflictError):
            efi.upload_frozen_input_bundle_to_r2(specs, project_root=tmp_path, client=client)
        # b was never attempted.
        assert not client.object_exists("frozen-inputs/v1/b.bin")
