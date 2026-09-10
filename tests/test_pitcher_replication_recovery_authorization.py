"""Version 0.14: the maintainer's recovery authorization for the
already-exposed 2025 pitcher replication.

No test here re-opens 2025, parses the held-out artifacts, runs modeling, or
reads 2026. The authorization is checked by hash and by behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pitcher_replication_execution as pre
import pitcher_replication_freeze as prf
import pitcher_replication_incident as inc
import pitcher_replication_recovery as rec
import pitcher_replication_recovery_authorization as auth
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    """Byte hash of a sealed artifact. Reads bytes, never parses a result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


@pytest.fixture
def outputs_sealed_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the completed-output and seal lookups at an empty directory.

    The sealed run means `assert_no_completed_results` is now the FIRST gate
    to refuse, permanently. It would mask every later gate, so tests that
    exist to prove a LATER guard still has teeth isolate it this way -- the
    same stubbing convention this suite already uses for the clean-tree
    check. Nothing real is moved or written.
    """
    empty = tmp_path / "outputs"
    empty.mkdir()
    monkeypatch.setattr(rec, "REPLICATION_OUTPUTS_DIR", empty)
    monkeypatch.setattr(rec, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return empty


class TestRecoveryReadiness:
    """RETARGETED after the sealed run.

    Revision 2 was authorized and the one permitted recovery RAN. Readiness
    therefore no longer passes and never will again: the completed result is
    now itself a permanent refusal. The earlier "readiness now passes" form
    was correct only in the window between the sign-off and the run.
    """

    def test_readiness_now_permanently_refuses_because_the_recovery_completed(
        self, clean_tree: None
    ) -> None:
        with pytest.raises(rec.RecoveryError, match="completed 2025 replication result") as excinfo:
            rec.assert_ready_for_recovery()
        assert "may never overwrite a finished replication" in str(excinfo.value)

    def test_the_authorization_itself_still_binds_and_was_not_revoked(self) -> None:
        """The refusal above is a STATE refusal, not an authorization failure.

        Preserved from the pre-execution suite: revision 2 is still the
        authorized manifest. Recovery is closed because it has been used, not
        because the sign-off went missing.
        """
        current = rec.read_recovery_manifest().manifest_content_hash()
        assert current == auth.AUTHORIZED_RECOVERY_MANIFEST_HASH
        authorized, reasons = rec.resolve_recovery_authorization(current)
        assert authorized is True, reasons

    def test_readiness_still_refuses_once_a_recovery_receipt_exists(
        self, outputs_sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        """Authorization does not bypass the one-attempt guard.

        The real receipt now exists, so this guard fires on the REAL path --
        no fixture receipt is needed any more.
        """
        assert rec.RECOVERY_RECEIPT_PATH.is_file()
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.assert_ready_for_recovery()

    def test_readiness_still_refuses_on_a_dirty_tree(
        self, outputs_sealed_elsewhere: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


class TestTheAuthorizedRecoveryHasBeenExecuted:
    """RETARGETED after the sealed run.

    This class asserted that nothing had been executed yet. The one
    authorized recovery has since run to completion, so the invariant it must
    now protect is the mirror image: the receipt and the result exist, are
    unchanged, and close the path behind them.
    """

    def test_the_recovery_receipt_exists_and_records_the_consumed_attempt(self) -> None:
        assert rec.RECOVERY_RECEIPT_PATH.is_file()
        receipt = json.loads(rec.RECOVERY_RECEIPT_PATH.read_text())
        assert receipt["recovery_id"] == "780aa2e2b8ec43f5"
        assert receipt["one_recovery_attempt_only"] is True
        assert receipt["is_a_first_look"] is False
        assert receipt["supersedes_execution_id"] == auth.ORIGINAL_EXECUTION_ID
        assert receipt["recovery_manifest_hash"] == auth.AUTHORIZED_RECOVERY_MANIFEST_HASH

    def test_the_receipt_pins_the_original_receipt_it_superseded(self) -> None:
        """The recovery receipt records the original's hash, so deleting or
        rewriting the original to fake a first look is detectable.
        """
        receipt = json.loads(rec.RECOVERY_RECEIPT_PATH.read_text())
        assert receipt["original_receipt_sha256"] == _sha256(pre.EXECUTION_RECEIPT_PATH)

    def test_a_second_recovery_is_refused_on_the_real_receipt(self) -> None:
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.assert_no_recovery_receipt()

    def test_the_sealed_replication_result_exists_and_matches_its_seal(self) -> None:
        outputs = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
        results = outputs / "pitcher_replication_2025_results.json"
        provenance = outputs / "pitcher_replication_2025_provenance.json"
        seal_path = REPO_ROOT / "artifacts" / "pitcher_replication" / "v0_14"
        seal_path = seal_path / "pitcher_replication_2025_seal.json"
        assert results.is_file() and provenance.is_file() and seal_path.is_file()

        seal = json.loads(seal_path.read_text())
        assert seal["results_sha256"] == _sha256(results)
        assert seal["classification"] == "REPLICATED"
        assert seal["primary_disagreements"] == []
        assert seal["execution_id"] == "780aa2e2b8ec43f5"

    def test_the_original_receipt_still_records_the_only_exposure(self) -> None:
        receipt = json.loads(pre.EXECUTION_RECEIPT_PATH.read_text())
        assert receipt["execution_id"] == auth.ORIGINAL_EXECUTION_ID

    def test_the_incident_still_records_that_no_model_was_fit(self) -> None:
        record = inc.read_failure_records()[-1]
        assert record["exposure"]["model_parameters_fit"] is False
        assert record["exposure"]["questions_a_to_g_computed"] is False
        assert record["exposure"]["classification_computed"] is False
