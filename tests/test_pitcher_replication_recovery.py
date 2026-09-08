"""Version 0.14: recovery control for the failed 2025 pitcher replication.

No test here re-opens 2025, parses the held-out artifacts, runs modeling,
computes A-G, or reads 2026. Integrity is proven by byte hashes and
filesystem state only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pitcher_replication_execution as pre
import pitcher_replication_freeze as prf
import pitcher_replication_incident as inc
import pitcher_replication_recovery as rec
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sealed_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the recovery manifest/receipt into `tmp_path`, so no test can
    write into the real artifacts namespace.
    """
    monkeypatch.setattr(rec, "RECOVERY_MANIFEST_PATH", tmp_path / "recovery_manifest.json")
    monkeypatch.setattr(rec, "RECOVERY_RECEIPT_PATH", tmp_path / "recovery_receipt.json")
    return tmp_path


@pytest.fixture
def clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec, "working_tree_status", lambda repo_root=None: (True, []))


class TestRecoveryAuthorizationGate:
    """The maintainer granted the recovery authorization on 2026-09-08. These
    assert the gate's post-authorization contract; the earlier
    "no authorization exists" versions were correct only before it.
    """

    def test_the_recorded_authorization_binds_to_the_current_revision(self) -> None:
        """The revision-2 sign-off covers the manifest now on disk."""
        current = rec.read_recovery_manifest().manifest_content_hash()
        authorized, reasons = rec.resolve_recovery_authorization(current)
        assert authorized is True, reasons

    def test_it_does_not_cover_the_superseded_revision_1(self) -> None:
        """Presenting the old manifest cannot replay the spent sign-off."""
        v1 = rec.read_recovery_manifest(
            rec.SUPERSEDED_RECOVERY_MANIFEST_PATHS[0]
        ).manifest_content_hash()
        authorized, reasons = rec.resolve_recovery_authorization(v1)
        assert authorized is False
        assert any("REVISION 1" in r and "superseded" in r for r in reasons)

    def test_it_still_refuses_any_manifest_it_does_not_name(self) -> None:
        """Fail-closed: the sign-off is bound to one exact manifest hash."""
        authorized, reasons = rec.resolve_recovery_authorization("0" * 64)
        assert authorized is False
        assert any("does not transfer" in r for r in reasons)

    def test_the_original_authorization_is_not_reused(self) -> None:
        """Behavioural, not textual: the ORIGINAL authorization still binds to
        the freeze, yet the recovery gate is still False. So the recovery is
        not riding on the spent sign-off.
        """
        binds, _ = prf.resolve_authorization(prf.read_freeze())
        assert binds is True, "the original authorization still binds to the freeze"
        authorized, _ = rec.resolve_recovery_authorization("any-hash")
        assert authorized is False, (
            "a spent one-time authorization must not satisfy the recovery gate"
        )

    def test_readiness_refuses_a_manifest_the_authorization_does_not_cover(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        """A manifest sealed elsewhere has a different hash, so the sign-off
        does not apply and readiness refuses -- naming the spent original.
        """
        rec.write_recovery_manifest(rec.build_recovery_manifest(), path=rec.RECOVERY_MANIFEST_PATH)
        with pytest.raises(rec.RecoveryError) as excinfo:
            rec.assert_ready_for_recovery()
        message = str(excinfo.value)
        assert "No maintainer authorization for this recovery" in message
        assert "CONSUMED by execution" in message
        assert "may not be reused as though unspent" in message

    def test_the_spent_original_authorization_is_never_consulted(self) -> None:
        """Behavioural: the ORIGINAL authorization still binds to the freeze,
        but it is a different mechanism and cannot satisfy the recovery gate.
        """
        binds, _ = prf.resolve_authorization(prf.read_freeze())
        assert binds is True
        authorized, _ = rec.resolve_recovery_authorization("not-the-recovery-manifest")
        assert authorized is False


class Test2025ArtifactIntegrity:
    def test_all_three_artifacts_match_the_incident_record(self) -> None:
        verified = rec.verify_2025_artifacts()
        assert set(verified) == {
            "statcast_2025_regular_season.parquet",
            "game_metadata_2025.parquet",
            "sprint_speed_2025.parquet",
        }
        assert all(entry["matches_incident_record"] for entry in verified.values())

    def test_verification_never_parses_the_held_out_contents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integrity is a byte question. Opening the data to prove the
        recovery is possible would itself be another look.
        """

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("2025 contents must not be parsed during recovery prep")

        for name in ("read_parquet", "read_csv", "read_json", "read_feather"):
            monkeypatch.setattr(pd, name, explode)
        assert rec.verify_2025_artifacts()

    def test_changed_artifact_bytes_are_refused(self, tmp_path: Path) -> None:
        decoy = tmp_path / "2025"
        decoy.mkdir()
        for name in (
            "statcast_2025_regular_season.parquet",
            "game_metadata_2025.parquet",
            "sprint_speed_2025.parquet",
        ):
            (decoy / name).write_bytes(b"tampered")
        with pytest.raises(rec.RecoveryError, match="bytes changed since the incident"):
            rec.verify_2025_artifacts(decoy)

    def test_a_missing_artifact_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "2025"
        empty.mkdir()
        with pytest.raises(rec.RecoveryError, match="is missing"):
            rec.verify_2025_artifacts(empty)


class TestNoNetworkRequired:
    def test_ingestion_skips_download_when_the_artifact_exists(self) -> None:
        """Each ingest function downloads only when its target path is
        absent, so a resumption over the existing files performs no network
        operation.
        """
        source = Path(REPO_ROOT / "replication" / "run_pitcher_replication_2025.py").read_text()
        for fn in ("ingest_2025_raw", "ingest_2025_game_metadata", "ingest_2025_sprint_speed"):
            body = source.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
            assert "if not path.exists():" in body, f"{fn} must guard its download"
            download_at = min(
                body.index(token)
                for token in (
                    "download_statcast_range(",
                    "_build_2025_game_metadata(",
                    "fetch_season_sprint_speed(",
                )
                if token in body
            )
            assert body.index("if not path.exists():") < download_at, (
                f"{fn}'s download must sit inside the not-exists guard"
            )

    def test_all_three_target_paths_are_already_present(self) -> None:
        """The concrete precondition for a no-network resumption."""
        from mlb_luck_score.config import (
            development_raw_path,
            game_metadata_path,
            sprint_speed_path,
        )

        for resolver in (development_raw_path, game_metadata_path, sprint_speed_path):
            assert resolver(prf.REPLICATION_DATA_DIR, 2025).is_file()

    def test_verification_completes_with_network_methods_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("recovery preparation must not touch the network")

        for name in ("get", "post", "put", "patch", "delete", "head", "request", "Session"):
            if hasattr(requests, name):
                monkeypatch.setattr(requests, name, explode)
        assert rec.verify_2025_artifacts()
        assert rec.assert_originals_preserved()


class TestRecoveryManifest:
    def test_it_pins_everything_the_incident_requires(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        assert manifest.original_execution_id == "8edc32d6ca8830ce"
        assert manifest.correction_commit == rec.CORRECTION_COMMIT
        assert manifest.failed_run_commit == rec.FAILED_RUN_COMMIT
        assert len(manifest.artifacts_2025) == 3
        assert manifest.working_tree_clean is True
        assert manifest.correction_diff_sha256
        assert "isin(TRAIN_SEASONS)" in manifest.correction_diff
        for key in (
            "original_receipt_sha256",
            "original_execution_manifest_sha256",
            "incident_record_sha256",
            "freeze_content_hash",
            "spec_content_hash",
            "original_authorization_content_hash",
            "corrected_runner_sha256",
        ):
            assert getattr(manifest, key)

    def test_it_records_the_required_statements(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        statements = rec.build_recovery_manifest().statements
        assert statements["artifacts_downloaded_during_failed_first_exposure"] is True
        assert statements["no_contact_luck_result_computed_before_failure"] is True
        assert statements["recovery_is_technical_not_methodological"] is True
        assert statements["season_2025_already_exposed"] is True
        assert statements["is_a_first_look"] is False
        assert statements["season_2026_exposed"] is False
        assert statements["original_authorization_is_spent"] is True
        assert statements["one_recovery_attempt_only"] is True

    def test_identical_reseal_is_idempotent(self, sealed_elsewhere: Path, clean_tree: None) -> None:
        manifest = rec.build_recovery_manifest()
        path, first = rec.write_recovery_manifest(manifest, path=rec.RECOVERY_MANIFEST_PATH)
        before = path.read_text()
        _, second = rec.write_recovery_manifest(manifest, path=rec.RECOVERY_MANIFEST_PATH)
        assert second == first
        assert path.read_text() == before

    def test_changed_recovery_code_refuses_overwrite(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        rec.write_recovery_manifest(rec.build_recovery_manifest(), path=rec.RECOVERY_MANIFEST_PATH)
        drifted = rec.build_recovery_manifest()
        hashes = dict(drifted.recovery_source_hashes)
        hashes["replication/run_pitcher_replication_2025.py"] = "0" * 64
        object.__setattr__(drifted, "recovery_source_hashes", hashes)
        with pytest.raises(rec.RecoveryError, match="write-once"):
            rec.write_recovery_manifest(drifted, path=rec.RECOVERY_MANIFEST_PATH)

    def test_there_is_no_amendment_mechanism(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        """No amend function exists, and the refusal directs a changed runner
        to a NEW manifest rather than an in-place edit. Asserted against the
        RAISED message, not the source text.
        """
        assert "def amend_" not in Path(rec.__file__).read_text()

        rec.write_recovery_manifest(rec.build_recovery_manifest(), path=rec.RECOVERY_MANIFEST_PATH)
        drifted = rec.build_recovery_manifest()
        object.__setattr__(drifted, "corrected_runner_sha256", "6" * 64)
        with pytest.raises(rec.RecoveryError) as excinfo:
            rec.write_recovery_manifest(drifted, path=rec.RECOVERY_MANIFEST_PATH)
        message = str(excinfo.value)
        assert "write-once and has no amendment path" in message
        assert "seal a NEW recovery manifest" in message

    def test_recovery_source_drift_is_detected(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        hashes = dict(manifest.recovery_source_hashes)
        hashes["replication/pitcher_replication_questions.py"] = "1" * 64
        object.__setattr__(manifest, "recovery_source_hashes", hashes)
        with pytest.raises(rec.RecoveryError, match="changed since sealing"):
            rec.validate_recovery_manifest(manifest)

    def test_a_changed_original_receipt_is_detected(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        object.__setattr__(manifest, "original_receipt_sha256", "2" * 64)
        with pytest.raises(rec.RecoveryError, match="ORIGINAL execution-start receipt changed"):
            rec.validate_recovery_manifest(manifest)

    def test_a_changed_original_execution_manifest_is_detected(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        object.__setattr__(manifest, "original_execution_manifest_sha256", "3" * 64)
        with pytest.raises(rec.RecoveryError, match="ORIGINAL execution manifest changed"):
            rec.validate_recovery_manifest(manifest)

    def test_a_changed_incident_record_is_detected(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        object.__setattr__(manifest, "incident_record_sha256", "4" * 64)
        with pytest.raises(rec.RecoveryError, match="incident record changed"):
            rec.validate_recovery_manifest(manifest)

    def test_changed_2025_artifact_bytes_are_detected(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        artifacts = {
            name: {**entry, "sha256": "5" * 64} for name, entry in manifest.artifacts_2025.items()
        }
        object.__setattr__(manifest, "artifacts_2025", artifacts)
        with pytest.raises(rec.RecoveryError, match="2025 artifact bytes changed since sealing"):
            rec.validate_recovery_manifest(manifest)

    def test_sealing_requires_a_clean_tree(
        self, sealed_elsewhere: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rec, "working_tree_status", lambda repo_root=None: (False, ["M x"]))
        with pytest.raises(rec.RecoveryError, match="not clean"):
            rec.build_recovery_manifest()


class TestOriginalsArePreserved:
    def test_the_original_receipt_and_manifest_still_exist(self) -> None:
        hashes = rec.assert_originals_preserved()
        assert set(hashes) == {
            "execution_start_receipt",
            "execution_manifest",
            "incident_record",
        }

    def test_the_original_receipt_still_records_the_first_exposure(self) -> None:
        receipt = json.loads(pre.EXECUTION_RECEIPT_PATH.read_text())
        assert receipt["execution_id"] == "8edc32d6ca8830ce"
        assert receipt["one_time_use"] is True

    def test_the_incident_record_is_intact(self) -> None:
        """The ORIGINAL incident is record 0 and never moves as more are
        appended.
        """
        records = inc.read_failure_records()
        assert records
        assert records[0]["exception_type"] == "FinalEvaluationError"
        assert records[0]["exposure"]["model_parameters_fit"] is False


class TestOneRecoveryAttempt:
    def test_no_recovery_receipt_exists_yet(self) -> None:
        assert not rec.RECOVERY_RECEIPT_PATH.exists()

    def test_a_second_attempt_is_refused(self, sealed_elsewhere: Path, clean_tree: None) -> None:
        manifest = rec.build_recovery_manifest()
        rec.write_recovery_receipt(manifest, path=rec.RECOVERY_RECEIPT_PATH)
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.assert_no_recovery_receipt(rec.RECOVERY_RECEIPT_PATH)
        with pytest.raises(rec.RecoveryError, match="permitted ONE attempt"):
            rec.write_recovery_receipt(manifest, path=rec.RECOVERY_RECEIPT_PATH)

    def test_the_receipt_is_separate_from_the_original(
        self, sealed_elsewhere: Path, clean_tree: None
    ) -> None:
        manifest = rec.build_recovery_manifest()
        receipt = rec.write_recovery_receipt(manifest, path=rec.RECOVERY_RECEIPT_PATH)
        assert receipt["supersedes_execution_id"] == "8edc32d6ca8830ce"
        assert receipt["is_a_first_look"] is False
        assert receipt["one_recovery_attempt_only"] is True
        assert rec.RECOVERY_RECEIPT_PATH != pre.EXECUTION_RECEIPT_PATH
        assert pre.EXECUTION_RECEIPT_PATH.exists()

    def test_no_automatic_retry_exists(self) -> None:
        source = Path(rec.__file__).read_text()
        assert "no_automatic_retry" in source
        assert "for attempt in" not in source
        assert "while True" not in source


class TestScientificStateDidNotMove:
    def test_the_research_freeze_still_validates_15_of_15(self) -> None:
        report = prf.validate_freeze(prf.read_freeze())
        assert report["valid"] is True
        assert report["source_files_verified"] == 15

    def test_every_frozen_scientific_setting_is_unchanged(self) -> None:
        import pitcher_replication_spec as spec
        from pitcher_replication_estimators import (
            K_FIT_MIN_BBE,
            NORMAL_95_Z,
            PRACTICAL_WORKLOAD_FLOORS,
        )
        from pitcher_split_half import MIN_ELIGIBLE_EACH_HALF, SPLITS

        assert spec.PLAY_LEVEL_DEFINITION["formula"].endswith(
            "expected_run_value - observed_run_value"
        )
        assert spec.DENOMINATOR["redefined_for_the_pitcher_stage"] is False
        assert spec.PRIMARY_QUANTITY["name"] == "cumulative_contact_luck_runs"
        assert spec.SECONDARY_QUANTITY["may_be_the_official_ranking_key"] is False
        assert spec.INTERVAL_PROCEDURE["may_be_changed_after_seeing_2025"] is False
        assert spec.ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE == 10.0
        assert spec.ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE == 8.0
        assert spec.BOARD_DISPLAY_MINIMUM_BBE == 60
        assert spec.CLASSIFICATION_VALUES == ("REPLICATED", "REVISE", "NO_GO")
        assert PRACTICAL_WORKLOAD_FLOORS == (60, 150, 300, 450)
        assert NORMAL_95_Z == 1.96
        assert K_FIT_MIN_BBE == 30
        assert MIN_ELIGIBLE_EACH_HALF == 20
        assert SPLITS == ("calendar", "odd_even")

    def test_training_seasons_are_still_2021_to_2023(self) -> None:
        from mlb_luck_score.config import TRAIN_SEASONS

        assert TRAIN_SEASONS == (2021, 2022, 2023)

    def test_all_seven_questions_are_still_present(self) -> None:
        import pitcher_replication_questions as prq

        assert prq.QUESTION_KEYS == (
            "A_population_and_centering",
            "B_opportunity_heterogeneity",
            "C_totals_vs_rate",
            "D_reliever_single_play_dominance",
            "E_rate_precision",
            "F_persistence",
            "G_real_play_sanity_check",
        )

    def test_no_replication_result_exists(self) -> None:
        outputs = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
        files = (
            [p for p in outputs.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            if outputs.exists()
            else []
        )
        assert files == []


class TestTwoSeparateStateMachines:
    """The first-look gate and the recovery gate are structurally distinct.

    The 2026-09-08 recovery-readiness bug was that the corrected runner still
    routed through `assert_ready_for_2025`, whose namespace-empty condition is
    a pre-first-look guard and is incompatible with a resumption over cached
    inputs.
    """

    def test_the_first_look_guard_still_refuses_when_any_2025_artifact_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """UNCHANGED and still protecting ordinary first executions.

        The clean-tree check runs first, so it is stubbed to isolate the
        namespace-empty condition under test.
        """
        monkeypatch.setattr(prf, "working_tree_status", lambda repo_root=None: (True, []))
        with pytest.raises(prf.FreezeError, match="NOT empty"):
            prf.assert_ready_for_2025(prf.read_freeze())

    def test_the_first_look_guard_was_not_weakened(self) -> None:
        source = Path(prf.__file__).read_text()
        gate = source.split("def assert_ready_for_2025", 1)[1].split("\ndef ", 1)[0]
        assert "assert_no_replication_outputs_exist()" in gate
        assert "ignore_namespace" not in source
        assert "skip_namespace" not in source

    def test_the_recovery_gate_does_not_call_the_first_look_gate(self) -> None:
        """Comments may NAME the first-look gate to explain the separation;
        what must not appear is a CALL to it.
        """
        source = Path(rec.__file__).read_text()
        gate = source.split("def assert_ready_for_recovery", 1)[1].split("\ndef ", 1)[0]
        code = "\n".join(line for line in gate.splitlines() if not line.lstrip().startswith("#"))
        assert "assert_ready_for_2025(" not in code
        assert "assert_no_replication_outputs_exist(" not in code

    def test_recovery_execution_does_not_call_the_first_look_gate(self) -> None:
        source = Path(rec.__file__).read_text()
        body = source.split("def run_recovery(", 1)[1].split("\ndef ", 1)[0]
        assert "run_recovery_readiness_checks" in body
        assert "assert_ready_for_2025" not in body
        assert "run_readiness_checks(" not in body

    def test_first_look_execution_still_calls_the_first_look_gate(self) -> None:
        runner = Path(REPO_ROOT / "replication" / "run_pitcher_replication_2025.py").read_text()
        body = runner.split("def run_replication(", 1)[1].split("\ndef ", 1)[0]
        assert "run_readiness_checks(" in body

    def test_both_paths_share_the_same_execution_stages(self) -> None:
        """Only the gate differs; the science is identical either way."""
        runner = Path(REPO_ROOT / "replication" / "run_pitcher_replication_2025.py").read_text()
        assert "def execute_replication_stages(" in runner
        first = runner.split("def run_replication(", 1)[1].split("\ndef ", 1)[0]
        assert "execute_replication_stages(" in first
        recovery = Path(rec.__file__).read_text().split("def run_recovery(", 1)[1]
        assert "execute_replication_stages(" in recovery

    def test_the_recovery_mint_has_exactly_one_call_site(self) -> None:
        source = Path(rec.__file__).read_text()
        assert source.count("mint_authorization_after_recovery_gate") == 2  # import + call
        gate = source.split("def run_recovery_readiness_checks", 1)[1].split("\ndef ", 1)[0]
        assert "mint_authorization_after_recovery_gate(" in gate


class TestCachedInputsVersusCompletedOutputs:
    """The distinction the bug conflated."""

    def test_cached_inputs_are_required_not_refused(self) -> None:
        verified = rec.assert_cached_inputs_exact()
        assert len(verified) == 3

    def test_a_missing_cached_input_refuses(self, tmp_path: Path) -> None:
        empty = tmp_path / "2025"
        empty.mkdir()
        with pytest.raises(rec.RecoveryError, match="is missing"):
            rec.assert_cached_inputs_exact(empty)

    def test_changed_cached_bytes_refuse(self, tmp_path: Path) -> None:
        decoy = tmp_path / "2025"
        decoy.mkdir()
        for name in (
            "statcast_2025_regular_season.parquet",
            "game_metadata_2025.parquet",
            "sprint_speed_2025.parquet",
        ):
            (decoy / name).write_bytes(b"tampered")
        with pytest.raises(rec.RecoveryError, match="bytes changed since the incident"):
            rec.assert_cached_inputs_exact(decoy)

    def test_an_extra_unapproved_cached_artifact_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        staged = tmp_path / "2025"
        staged.mkdir()
        for entry in inc.read_failure_records()[0]["artifacts_2025"]:
            name = entry["path"].split("/")[-1]
            (staged / name).write_bytes((prf.REPLICATION_DATA_DIR / name).read_bytes())
        (staged / "sneaky_extra.parquet").write_bytes(b"unapproved")
        with pytest.raises(rec.RecoveryError, match="Unapproved file"):
            rec.assert_cached_inputs_exact(staged)

    def test_no_completed_result_currently_exists(self) -> None:
        report = rec.assert_no_completed_results()
        assert report["completed_results"] == []
        assert report["run_seal_exists"] is False

    def test_a_completed_result_refuses(self, tmp_path: Path) -> None:
        outputs = tmp_path / "v0_14"
        outputs.mkdir()
        (outputs / "pitcher_replication_2025_results.json").write_text("{}")
        with pytest.raises(rec.RecoveryError, match="completed 2025 replication result"):
            rec.assert_no_completed_results(outputs)

    def test_completed_results_check_ignores_the_input_directory(self, tmp_path: Path) -> None:
        """Cached inputs must never be mistaken for a finished result."""
        empty_outputs = tmp_path / "v0_14"
        empty_outputs.mkdir()
        assert rec.assert_no_completed_results(empty_outputs)["completed_results"] == []


class TestTheFailedRecoveryAttemptWasNotConsumed:
    def test_no_recovery_receipt_was_created(self) -> None:
        assert not rec.RECOVERY_RECEIPT_PATH.exists()

    def test_the_second_incident_is_recorded(self) -> None:
        records = inc.read_failure_records()
        second = [r for r in records if r.get("sequence") == 2]
        assert second, "the recovery-readiness failure must be recorded"
        exposure = second[-1]["exposure"]
        assert exposure["recovery_receipt_existed"] is False
        assert exposure["authorized_recovery_attempt_consumed"] is False
        assert exposure["season_2025_reread_during_this_attempt"] is False
        assert exposure["season_2026_read"] is False

    def test_the_incident_log_is_append_only_and_kept_both(self) -> None:
        records = inc.read_failure_records()
        assert len(records) >= 2
        assert records[0]["incident"] == "execution_failure_after_held_out_ingestion"
        assert records[-1]["incident"] == "recovery_readiness_failure_before_receipt"
