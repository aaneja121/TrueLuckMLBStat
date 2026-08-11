"""Tests for scripts/ensure_frozen_inputs.py -- the portability fix that
lets a fresh CI runner obtain the frozen 2021-2024 development input
without ever regenerating it. All tests use `InMemoryArchiveClient`
(imported from `scripts.archive_snapshot`); none contact real Cloudflare
R2, none require real credentials, and none touch the real 106MB frozen
parquet on this machine's disk -- every test uses tiny synthetic bytes and
its own `tmp_path`-scoped local path/hash pair.
"""

from __future__ import annotations

import ast
import hashlib
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
        # Imported (not re-declared) from evaluation.run_v1_final_evaluation
        # -- this assertion is really just confirming the import wiring
        # didn't silently fall back to some other path.
        assert (
            efi.FROZEN_INPUT_LOCAL_PATH.name == "cleaned_development_data_with_sprint_speed.parquet"
        )
        assert efi.FROZEN_INPUT_LOCAL_PATH.parent.name == "processed"

    def test_pinned_hash_is_a_well_formed_sha256_hex_digest(self) -> None:
        assert len(efi.FROZEN_INPUT_SHA256) == 64
        int(efi.FROZEN_INPUT_SHA256, 16)  # raises ValueError if not valid hex
