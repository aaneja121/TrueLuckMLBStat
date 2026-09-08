"""Version 0.14: the permanent post-execution invariants of the sealed 2025
pitcher replication.

The one-time held-out 2025 replication ran, completed, and was sealed
REPLICATED. Every earlier guard in this suite was written to protect a season
that had not yet been opened; those have been retargeted in place. This module
is the standing record of what must hold FOREVER now that it has been opened
and finished.

Nothing here re-opens 2025, re-runs modeling, recomputes A-G, or reads 2026.
Integrity is proven by byte hashes, filesystem state, and reading already
-written JSON.

The sealed artifacts live in gitignored namespaces, so these tests describe a
machine that actually holds the completed replication -- the same assumption
the existing post-exposure tests already make.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pitcher_replication_execution as pre
import pitcher_replication_freeze as prf
import pitcher_replication_incident as inc
import pitcher_replication_recovery as rec
import pytest

from mlb_luck_score.config import FINAL_TEST_SEASONS, PROSPECTIVE_SEASONS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
RESULTS_PATH = OUTPUTS_DIR / "pitcher_replication_2025_results.json"
PROVENANCE_PATH = OUTPUTS_DIR / "pitcher_replication_2025_provenance.json"
SEAL_PATH = prf.ARTIFACTS_DIR / "pitcher_replication_2025_seal.json"

#: The sealed identity of the completed run. Pinned so that any later edit to
#: a control file -- however well intentioned -- fails loudly instead of
#: quietly rewriting the scientific record.
SEALED_IDENTITY = {
    "execution_id": "780aa2e2b8ec43f5",
    "superseded_execution_id": "8edc32d6ca8830ce",
    "repository_commit": "30b4d8210127724f664d54ac8771ccf5561fc474",
    "freeze_content_hash": "58d8235b8d8324d55fe296875c1c8cdbd9857c5375be887f273822d1b1d40714",
    "spec_content_hash": "23320d4654b930112575fa7915b77cac618ff567ed3dabe5c314bf00002d0069",
    "recovery_manifest_hash": "b05b8f1b273eebfc26d6a62fba3608fabfb2cf41929c63bef82270dcb1925eb6",
    "results_sha256": "d483342a2549653845e88153c73e51329d0a9a1da8d41fa1be9d6d0c16d5b041",
}

#: Byte hashes of the immutable control files, as sealed.
IMMUTABLE_CONTROL_FILES = {
    "execution_start_receipt.json": (
        "40a236a8b169d8fad016490ebddec86acfe82473aab5e8d0e9f6d30437e238f2"
    ),
    "pitcher_replication_execution_manifest.json": (
        "06f19c43d23821ff7b6185cd0f990689927533e65053dab98229db341a235cd3"
    ),
    "recovery_receipt.json": ("4159ed40d1532ab99b6421524715f22e351cb89a5fdc37633b39c8bed0e4b53a"),
    "recovery_manifest.json": ("355fe492eaef5774eab4652321916b6690db010ee7cf2af16bb218c4232c449f"),
    "recovery_manifest_rev2.json": (
        "dfe895b4328a7f2c6759b8207f398a25a6e9027e71ecbca86abf06f16c9cc35c"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


@pytest.fixture(scope="module")
def seal() -> dict:
    return json.loads(SEAL_PATH.read_text())


class TestTheSealedResultExistsAndIsIntact:
    """States 3-8: the successful evaluation's own artifacts."""

    def test_results_provenance_and_seal_all_exist(self) -> None:
        assert RESULTS_PATH.is_file()
        assert PROVENANCE_PATH.is_file()
        assert SEAL_PATH.is_file()

    def test_the_results_hash_matches_the_seal(self, seal: dict) -> None:
        """The seal's whole purpose: the result cannot be edited unnoticed."""
        assert seal["results_sha256"] == SEALED_IDENTITY["results_sha256"]
        assert _sha256(RESULTS_PATH) == seal["results_sha256"]

    def test_the_classification_remains_replicated(self, seal: dict, results: dict) -> None:
        assert seal["classification"] == "REPLICATED"
        assert results["classification"]["classification"] == "REPLICATED"

    def test_primary_disagreements_remain_empty(self, seal: dict, results: dict) -> None:
        assert seal["primary_disagreements"] == []
        assert results["classification"]["primary_disagreements"] == []

    def test_the_seal_carries_the_sealed_identity(self, seal: dict) -> None:
        for field in (
            "execution_id",
            "repository_commit",
            "freeze_content_hash",
            "spec_content_hash",
        ):
            assert seal[field] == SEALED_IDENTITY[field]
        assert seal["execution_manifest_hash"] == SEALED_IDENTITY["recovery_manifest_hash"]

    def test_every_primary_question_agreed(self, results: dict) -> None:
        for question in (
            "A_population_and_centering",
            "B_opportunity_heterogeneity",
            "C_totals_vs_rate",
            "D_reliever_single_play_dominance",
            "E_rate_precision",
        ):
            assert results["questions"][question]["agrees"] is True


class TestTheEvidenceOfExposureIsPreserved:
    """States 1, 2, 13-16: nothing that records how 2025 was opened may move."""

    @pytest.mark.parametrize("name,expected", sorted(IMMUTABLE_CONTROL_FILES.items()))
    def test_control_file_is_byte_identical_to_its_sealed_state(
        self, name: str, expected: str
    ) -> None:
        path = prf.ARTIFACTS_DIR / name
        assert path.is_file(), f"{name} must never be deleted"
        assert _sha256(path) == expected, f"{name} changed after sealing"

    def test_the_first_exposure_receipt_is_the_one_the_recovery_superseded(self) -> None:
        original = json.loads(pre.EXECUTION_RECEIPT_PATH.read_text())
        receipt = json.loads(rec.RECOVERY_RECEIPT_PATH.read_text())
        assert original["execution_id"] == SEALED_IDENTITY["superseded_execution_id"]
        assert receipt["supersedes_execution_id"] == original["execution_id"]
        assert receipt["original_receipt_sha256"] == _sha256(pre.EXECUTION_RECEIPT_PATH)

    def test_the_successful_recovery_is_not_recorded_as_a_first_look(self) -> None:
        receipt = json.loads(rec.RECOVERY_RECEIPT_PATH.read_text())
        assert receipt["is_a_first_look"] is False
        assert receipt["recovery_id"] == SEALED_IDENTITY["execution_id"]
        assert "RESUMPTION after exposure" in receipt["statement"]

    def test_both_recovery_manifest_revisions_are_kept(self) -> None:
        """Revision 1 is superseded, never deleted; revision 2 is the one the
        sign-off covers and the one the seal names.
        """
        v1 = rec.SUPERSEDED_RECOVERY_MANIFEST_PATHS[0]
        assert v1.is_file()
        assert rec.RECOVERY_MANIFEST_PATH.is_file()
        assert v1 != rec.RECOVERY_MANIFEST_PATH
        live = rec.read_recovery_manifest().manifest_content_hash()
        assert live == SEALED_IDENTITY["recovery_manifest_hash"]

    def test_the_incident_history_is_append_only_and_complete(self) -> None:
        records = inc.read_failure_records()
        assert len(records) >= 2
        assert records[0]["incident"] == "execution_failure_after_held_out_ingestion"
        assert records[0]["execution_id"] == SEALED_IDENTITY["superseded_execution_id"]
        assert records[-1]["incident"] == "recovery_readiness_failure_before_receipt"
        # Neither recorded failure ever fit a model or computed a classification.
        for record in records:
            assert record["exposure"]["model_parameters_fit"] is False
            assert record["exposure"]["classification_computed"] is False


class TestEveryExecutionPathIsNowClosed:
    """States 9-11: the successful run made the repository MORE closed."""

    def test_the_ordinary_first_look_namespace_guard_refuses(self) -> None:
        with pytest.raises(pre.ExecutionError, match="already populated"):
            pre.assert_output_namespace_available()

    def test_the_first_look_readiness_gate_refuses(self) -> None:
        with pytest.raises(prf.FreezeError, match="NOT empty"):
            prf.assert_no_replication_outputs_exist()

    def test_a_second_execution_receipt_is_refused(self) -> None:
        with pytest.raises(pre.ExecutionError, match="already been opened"):
            pre.assert_no_execution_receipt()

    def test_the_recovery_gate_refuses_on_the_completed_result(self) -> None:
        with pytest.raises(rec.RecoveryError, match="completed 2025 replication result"):
            rec.assert_no_completed_results()

    def test_a_second_recovery_receipt_is_refused(self) -> None:
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.assert_no_recovery_receipt()

    def test_full_recovery_readiness_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even from a clean tree, end to end."""
        monkeypatch.setattr(rec, "working_tree_status", lambda repo_root=None: (True, []))
        with pytest.raises(rec.RecoveryError):
            rec.assert_ready_for_recovery()

    def test_no_automatic_retry_path_exists(self) -> None:
        """A failure after this point needs a new human decision, not a loop."""
        source = Path(rec.__file__).read_text()
        assert "no_automatic_retry" in source
        assert "for attempt in" not in source
        assert "while True" not in source
        receipt = json.loads(rec.RECOVERY_RECEIPT_PATH.read_text())
        assert "no_automatic_retry" in receipt


class TestTheResearchFreezeStillHolds:
    """State 12: the frozen science did not move when the result landed."""

    def test_the_freeze_validates_15_of_15_with_zero_drift(self) -> None:
        report = prf.validate_freeze(prf.read_freeze())
        assert report["valid"] is True
        assert report["source_files_verified"] == 15
        assert report["freeze_content_hash"] == SEALED_IDENTITY["freeze_content_hash"]
        assert report["spec_content_hash"] == SEALED_IDENTITY["spec_content_hash"]

    def test_the_sealed_result_was_produced_under_this_exact_freeze(self, results: dict) -> None:
        assert results["freeze_content_hash"] == SEALED_IDENTITY["freeze_content_hash"]
        assert results["spec_content_hash"] == SEALED_IDENTITY["spec_content_hash"]


class TestTheQuestionFErratumDidNotMutateTheResult:
    """State 17: the erratum is documentation, never a rewrite.

    The sealed result records `implementation_trustworthy = false` because the
    batter-side reproduction control was measured against the wrong season's
    artifacts. The post-replication audit cleared the implementation but must
    NOT edit the sealed result to say so -- the record has to keep saying what
    it said at execution time.
    """

    def test_the_sealed_result_still_reports_the_failed_control(self, results: dict) -> None:
        question_f = results["questions"]["F_persistence"]
        assert question_f["implementation_trustworthy"] is False
        assert question_f["batter_side_reproduction"]["reproduces"] is False

    def test_question_f_still_agrees_and_stays_secondary(self, results: dict) -> None:
        question_f = results["questions"]["F_persistence"]
        assert question_f["agrees"] is True
        assert question_f["cannot_override_package_classification"] is True
        assert question_f["is_a_success_criterion_on_its_own"] is False

    def test_the_failed_control_never_entered_the_classification(self, results: dict) -> None:
        assert results["classification"]["secondary_disagreements"] == []
        assert results["classification"]["classification"] == "REPLICATED"

    def test_the_erratum_is_tracked_documentation_not_an_artifact_edit(self) -> None:
        erratum = REPO_ROOT / "docs" / "pitcher_replication_2025_question_f_erratum.md"
        record = REPO_ROOT / "docs" / "pitcher_replication_2025_decision_record.md"
        assert erratum.is_file()
        assert record.is_file()
        # The decision record indexes the sealed artifacts by hash rather than
        # copying them into git.
        assert SEALED_IDENTITY["results_sha256"] in record.read_text()

    def test_the_result_hash_is_unchanged_by_the_audit(self, seal: dict) -> None:
        assert _sha256(RESULTS_PATH) == seal["results_sha256"]


class Test2026RemainsProspectiveAndUnopened:
    """State 18: the pitcher line's next season is still untouched."""

    def test_2026_is_configured_as_prospective_not_final_test(self) -> None:
        assert PROSPECTIVE_SEASONS == (2026,)
        assert FINAL_TEST_SEASONS == (2025,)
        assert 2026 not in FINAL_TEST_SEASONS

    def test_the_sealed_replication_covers_2025_only(self, results: dict) -> None:
        assert results["replication_season"] == 2025
        start, end = results["date_range"]
        assert start.startswith("2025-") and end.startswith("2025-")

    def test_no_recorded_incident_ever_read_2026(self) -> None:
        for record in inc.read_failure_records():
            exposure = record["exposure"]
            if "season_2026_read" in exposure:
                assert exposure["season_2026_read"] is False

    def test_the_replication_namespace_holds_no_2026_data(self) -> None:
        """Filesystem names only -- no 2026 file is opened or read."""
        for directory in (prf.REPLICATION_DATA_DIR, OUTPUTS_DIR, prf.ARTIFACTS_DIR):
            if not directory.exists():
                continue
            assert not [p for p in directory.rglob("*2026*") if p.is_file()]
