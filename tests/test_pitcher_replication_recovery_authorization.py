"""Version 0.14: the maintainer's recovery authorization for the
already-exposed 2025 pitcher replication.

No test here re-opens 2025, parses the held-out artifacts, runs modeling, or
reads 2026. The authorization is checked by hash and by behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pitcher_replication_execution as pre
import pitcher_replication_freeze as prf
import pitcher_replication_incident as inc
import pitcher_replication_recovery as rec
import pitcher_replication_recovery_authorization as auth
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestBindsToTheExactRecoveryManifest:
    SUPERSEDED = rec.SUPERSEDED_RECOVERY_MANIFEST_PATHS[0]

    def test_it_names_the_revision_2_manifest_on_disk(self) -> None:
        """The authorization matches the artifact it was granted against --
        revision 2, read from disk, not copied from a request.
        """
        assert auth.AUTHORIZED_RECOVERY_MANIFEST_REVISION == 2
        live = rec.read_recovery_manifest().manifest_content_hash()
        assert live == auth.AUTHORIZED_RECOVERY_MANIFEST_HASH

    def test_it_binds_to_revision_2(self) -> None:
        live = rec.read_recovery_manifest().manifest_content_hash()
        authorized, reasons = auth.recovery_authorization_binds_to(live)
        assert authorized is True
        assert reasons == []

    def test_the_recovery_gate_now_resolves(self) -> None:
        live = rec.read_recovery_manifest().manifest_content_hash()
        authorized, reasons = rec.resolve_recovery_authorization(live)
        assert authorized is True, reasons

    def test_it_fails_closed_against_revision_1(self) -> None:
        """The superseded sign-off cannot be replayed by presenting the old
        manifest.
        """
        v1 = rec.read_recovery_manifest(self.SUPERSEDED).manifest_content_hash()
        assert v1 == auth.SUPERSEDED_RECOVERY_MANIFEST_HASH
        authorized, reasons = auth.recovery_authorization_binds_to(v1)
        assert authorized is False
        assert any("REVISION 1" in r and "superseded" in r for r in reasons)

    def test_it_fails_closed_against_any_future_revision(self) -> None:
        for other in ("7" * 64, "0" * 64, "deadbeef", ""):
            authorized, reasons = auth.recovery_authorization_binds_to(other)
            assert authorized is False
            assert any("does not transfer" in r for r in reasons)

    def test_it_fails_closed_on_any_other_manifest(self) -> None:
        for other in ("0" * 64, "deadbeef", "", "24e2ea817e741178fefdb5bd844603f67651983b"):
            authorized, reasons = auth.recovery_authorization_binds_to(other)
            assert authorized is False
            assert any("does not transfer" in r for r in reasons)

    def test_a_changed_manifest_revokes_the_authorization(self) -> None:
        """Changing anything the manifest pins -- recovery code or cached
        2025 bytes -- changes its hash, and the sign-off stops applying
        rather than following it.
        """
        manifest = rec.read_recovery_manifest()
        object.__setattr__(manifest, "corrected_runner_sha256", "9" * 64)
        authorized, _ = rec.resolve_recovery_authorization(manifest.manifest_content_hash())
        assert authorized is False


class TestTheAuthorizationRecord:
    def test_it_pins_every_required_identity(self) -> None:
        record = auth.recovery_authorization_record()
        manifest = rec.read_recovery_manifest()
        assert record["original_execution_id"] == "8edc32d6ca8830ce"
        assert record["original_receipt_sha256"] == manifest.original_receipt_sha256
        assert (
            record["original_execution_manifest_hash"] == manifest.original_execution_manifest_hash
        )
        assert record["incident_record_sha256"] == manifest.incident_record_sha256
        assert record["freeze_content_hash"] == manifest.freeze_content_hash
        assert record["spec_content_hash"] == manifest.spec_content_hash
        assert record["correction_commit"] == manifest.correction_commit
        assert record["recovery_control_commit"] == manifest.repository_commit
        assert (
            record["original_authorization_content_hash"]
            == manifest.original_authorization_content_hash
        )
        assert json.dumps(record)

    def test_it_records_the_verbatim_authorization_and_timestamp(self) -> None:
        record = auth.recovery_authorization_record()
        assert (
            "I explicitly authorize ONE technical recovery attempt" in record["authorization_text"]
        )
        assert "NOT a new first look" in record["authorization_text"]
        assert record["authorized_at_utc"].endswith("Z")

    def test_it_declares_the_exposure_and_the_first_look_prohibition(self) -> None:
        declarations = auth.AUTHORIZATION_DECLARATIONS
        assert declarations["season_2025_already_exposed"] is True
        assert declarations["is_a_first_look"] is False
        assert "never be reported" in declarations["never_describe_as_a_first_look"]

    def test_it_declares_one_attempt_and_no_further_entitlement(self) -> None:
        declarations = auth.AUTHORIZATION_DECLARATIONS
        assert declarations["one_recovery_attempt_only"] is True
        assert declarations["original_authorization_consumed"] is True
        assert (
            "new explicit incident review" in declarations["creates_no_further_retry_entitlement"]
        )

    def test_it_withholds_redownload_and_2026(self) -> None:
        declarations = auth.AUTHORIZATION_DECLARATIONS
        assert declarations["no_permission_to_redownload_2025"] is True
        assert "2026 remains prospective" in declarations["no_2026_authorization"]
        for withheld in (
            "redownload 2025",
            "inspect or use 2026",
            "train on 2025",
            "add 2024 to training",
            "deploy anything",
        ):
            assert withheld in auth.AUTHORIZATION_EXCLUSIONS

    def test_it_permits_only_the_frozen_work(self) -> None:
        permits = " ".join(auth.AUTHORIZATION_PERMITS)
        assert "2021-2023 only" in permits
        assert "preregistered questions A-G" in permits
        assert "REPLICATED / REVISE / NO_GO" in permits


class TestOrderingAndIsolation:
    def test_the_authorization_is_outside_the_recovery_manifest_source_set(self) -> None:
        """Recording the sign-off must not alter the manifest it authorizes."""
        assert (
            "replication/pitcher_replication_recovery_authorization.py"
            not in rec.RECOVERY_SOURCE_RELATIVE_PATHS
        )

    def test_revision_2_was_sealed_beside_revision_1_not_over_it(self) -> None:
        v1_path = rec.SUPERSEDED_RECOVERY_MANIFEST_PATHS[0]
        assert v1_path.is_file()
        assert rec.RECOVERY_MANIFEST_PATH.is_file()
        assert v1_path != rec.RECOVERY_MANIFEST_PATH
        assert rec.read_recovery_manifest(v1_path).manifest_content_hash() == (
            auth.SUPERSEDED_RECOVERY_MANIFEST_HASH
        )
        assert rec.validate_recovery_manifest(rec.read_recovery_manifest())["valid"] is True

    def test_the_research_freeze_was_not_rebuilt(self) -> None:
        report = prf.validate_freeze(prf.read_freeze())
        assert report["valid"] is True
        assert report["source_files_verified"] == 15

    def test_the_original_receipt_is_still_byte_identical(self) -> None:
        hashes = rec.assert_originals_preserved()
        assert hashes["execution_start_receipt"] == auth.ORIGINAL_RECEIPT_SHA256

    def test_the_authorization_records_the_incident_chain(self) -> None:
        records = inc.read_failure_records()
        assert len(records) == auth.INCIDENT_CHAIN_RECORD_COUNT
        assert max(r.get("sequence", 1) for r in records) == auth.INCIDENT_CHAIN_LATEST_SEQUENCE

    def test_it_declares_the_supersession_and_unconsumed_attempt(self) -> None:
        declarations = auth.AUTHORIZATION_DECLARATIONS
        assert declarations["revision"] == 2
        assert "SUPERSEDED" in declarations["supersedes_revision_1_authorization"]
        assert (
            "NOT consumed"
            in (
                declarations["previous_recovery_launch_did_not_consume_the_attempt"].replace(
                    "did not consume", "NOT consumed"
                )
            )
            or "before recovery_receipt.json"
            in (declarations["previous_recovery_launch_did_not_consume_the_attempt"])
        )

    def test_the_incident_log_grew_by_append_not_by_edit(self) -> None:
        """A second incident was appended, so the log hash necessarily moved.
        The ORIGINAL record must still be there, unchanged, as entry 0.
        """
        records = inc.read_failure_records()
        assert len(records) >= 2
        assert records[0]["incident"] == "execution_failure_after_held_out_ingestion"
        assert records[0]["execution_id"] == auth.ORIGINAL_EXECUTION_ID


@pytest.fixture
def clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a clean working tree.

    The clean-tree check runs first in `assert_ready_for_recovery`, correctly,
    but this suite runs while its own files may still be uncommitted.
    Stubbing it isolates the gate under test;
    `test_readiness_still_refuses_on_a_dirty_tree` covers the real check, and
    the committed end state is verified by the CLI readiness run.
    """
    monkeypatch.setattr(rec, "working_tree_status", lambda repo_root=None: (True, []))


class TestRecoveryReadiness:
    """Revision 2 is authorized, so readiness now PASSES -- while every other
    guard keeps its teeth.
    """

    def test_readiness_now_passes(self, clean_tree: None) -> None:
        report = rec.assert_ready_for_recovery()
        assert report["ready"] is True
        assert report["valid"] is True
        assert report["recovery_sources_verified"] == 7
        assert report["artifacts_2025_verified"] == 3
        assert report["originals_preserved"] is True
        assert report["recovery_manifest_hash"] == auth.AUTHORIZED_RECOVERY_MANIFEST_HASH

    def test_readiness_still_refuses_once_a_recovery_receipt_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_tree: None
    ) -> None:
        """Authorization does not bypass the one-attempt guard."""
        receipt = tmp_path / "recovery_receipt.json"
        receipt.write_text(json.dumps({"recovery_id": "x", "reopened_at_utc": "t"}))
        monkeypatch.setattr(rec, "RECOVERY_RECEIPT_PATH", receipt)
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.assert_ready_for_recovery()

    def test_readiness_still_refuses_on_a_dirty_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rec, "working_tree_status", lambda repo_root=None: (False, ["M x"]))
        with pytest.raises(rec.RecoveryError, match="not clean"):
            rec.assert_ready_for_recovery()

    def test_readiness_still_refuses_if_2025_artifacts_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_tree: None
    ) -> None:
        decoy = tmp_path / "2025"
        decoy.mkdir()
        for name in (
            "statcast_2025_regular_season.parquet",
            "game_metadata_2025.parquet",
            "sprint_speed_2025.parquet",
        ):
            (decoy / name).write_bytes(b"tampered")
        monkeypatch.setattr(rec, "REPLICATION_DATA_DIR", decoy)
        with pytest.raises(rec.RecoveryError, match="bytes changed since the incident"):
            rec.assert_ready_for_recovery()

    def test_revision_1_remains_preserved_and_still_does_not_validate(self) -> None:
        """Superseded, never resealed, never deleted."""
        v1_path = rec.SUPERSEDED_RECOVERY_MANIFEST_PATHS[0]
        assert v1_path.is_file()
        v1 = rec.read_recovery_manifest(v1_path)
        assert v1.manifest_content_hash() == auth.SUPERSEDED_RECOVERY_MANIFEST_HASH
        with pytest.raises(rec.RecoveryError):
            rec.validate_recovery_manifest(v1)


class TestNothingHasBeenExecuted:
    def test_no_recovery_receipt_exists(self) -> None:
        assert not rec.RECOVERY_RECEIPT_PATH.exists()

    def test_no_replication_result_exists(self) -> None:
        outputs = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
        files = (
            [p for p in outputs.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            if outputs.exists()
            else []
        )
        assert files == []

    def test_the_original_receipt_still_records_the_only_exposure(self) -> None:
        receipt = json.loads(pre.EXECUTION_RECEIPT_PATH.read_text())
        assert receipt["execution_id"] == auth.ORIGINAL_EXECUTION_ID

    def test_the_incident_still_records_that_no_model_was_fit(self) -> None:
        record = inc.read_failure_records()[-1]
        assert record["exposure"]["model_parameters_fit"] is False
        assert record["exposure"]["questions_a_to_g_computed"] is False
        assert record["exposure"]["classification_computed"] is False
