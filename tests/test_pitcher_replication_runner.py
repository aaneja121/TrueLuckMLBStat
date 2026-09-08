"""Version 0.14: guards for the one-time held-out 2025 pitcher-replication
runner.

**No test here opens 2025 or 2026.** Several exist specifically to prove
that: `TestZeroImportTimeDataAccess` imports the runner with every pandas
reader and every `requests` method armed to raise, and
`TestReadinessPrecedesDataAccess` instruments the first attempted data
access and asserts readiness ran before it.

The correlations and classification are exercised on synthetic frames built
in-process, never on a real season.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pitcher_replication_execution as pre
import pitcher_replication_freeze as prf
import pitcher_replication_questions as prq
import pytest
import run_pitcher_replication_2025 as runner

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every control path into `tmp_path`, so no test can write to
    the real artifacts or replication namespace.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for module in (pre, prf):
        monkeypatch.setattr(module, "ARTIFACTS_DIR", artifacts, raising=False)
        monkeypatch.setattr(
            module, "REPLICATION_DATA_DIR", tmp_path / "data" / "2025", raising=False
        )
        monkeypatch.setattr(
            module, "REPLICATION_OUTPUTS_DIR", tmp_path / "outputs" / "v0_14", raising=False
        )
    monkeypatch.setattr(
        pre, "EXECUTION_MANIFEST_PATH", artifacts / "pitcher_replication_execution_manifest.json"
    )
    monkeypatch.setattr(pre, "EXECUTION_RECEIPT_PATH", artifacts / "execution_start_receipt.json")
    return artifacts


@pytest.fixture
def clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a clean working tree.

    `build_execution_manifest` requires one, correctly -- but this suite runs
    while the runner itself is uncommitted. Stubbing it here isolates the
    manifest behaviour under test; `test_sealing_requires_a_clean_tree`
    covers the real check.
    """
    monkeypatch.setattr(pre, "working_tree_status", lambda repo_root=None: (True, []))


def _synthetic_pitcher_frame(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """A pitcher-season frame with the shape the questions expect. Entirely
    generated; no season is read.
    """
    rng = np.random.default_rng(seed)
    bbe = np.concatenate(
        [
            rng.integers(1, 40, n // 3),
            rng.integers(60, 300, n // 3),
            rng.integers(300, 620, n - 2 * (n // 3)),
        ]
    )
    appearances = np.where(
        bbe >= 250, rng.integers(28, 34, len(bbe)), rng.integers(30, 70, len(bbe))
    )
    total = rng.normal(0, 1.0, len(bbe)) * np.sqrt(bbe) / 10.0
    per_100 = total / np.maximum(bbe, 1) * 100.0
    half = 70.6 / np.sqrt(np.maximum(bbe, 1))
    return pd.DataFrame(
        {
            "pitcher_id": np.arange(100, 100 + len(bbe)),
            "name": [f"Pitcher {i}" for i in range(len(bbe))],
            "role_bucket": [
                prq.role_bucket(int(b), int(a)) for b, a in zip(bbe, appearances, strict=True)
            ],
            "cumulative_contact_luck_runs": total,
            "eligible_batted_balls": bbe,
            "appearances": appearances,
            "contact_luck_per_100": per_100,
            "per_100_ci_low": per_100 - half,
            "per_100_ci_high": per_100 + half,
            "largest_play_share_of_net": rng.uniform(0.1, 2.0, len(bbe)),
            "largest_favorable_play": [{"play_id": "x"}] * len(bbe),
            "largest_unfavorable_play": [{"play_id": "y"}] * len(bbe),
        }
    )


class TestZeroImportTimeDataAccess:
    def test_importing_the_runner_reads_nothing_and_calls_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("import-time data or network access")

        for name in (
            "read_parquet",
            "read_csv",
            "read_json",
            "read_feather",
            "read_pickle",
            "read_excel",
            "read_sql",
            "read_sql_table",
        ):
            if hasattr(pd, name):
                monkeypatch.setattr(pd, name, explode)
        for name in ("get", "post", "put", "patch", "delete", "head", "request", "Session"):
            if hasattr(requests, name):
                monkeypatch.setattr(requests, name, explode)

        for module_name in (
            "run_pitcher_replication_2025",
            "pitcher_replication_execution",
            "pitcher_replication_questions",
        ):
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)

    def test_runner_source_defers_every_data_import(self) -> None:
        """Data-layer imports live inside functions, so importing the module
        cannot pull in a downloader.
        """
        source = Path(runner.__file__).read_text()
        header = source.split("def ingest_2025_raw", 1)[0]
        for forbidden in (
            "download_statcast",
            "clean_development_data",
            "fetch_season_sprint_speed",
        ):
            assert forbidden not in header, f"{forbidden} must not be imported at module level"


class TestDateRangeIsLocalAndAuthoritative:
    def test_range_matches_the_repositorys_verified_2025_range(self) -> None:
        from run_v1_final_evaluation import FINAL_EVALUATION_2025_DATE_RANGE

        assert runner.PITCHER_REPLICATION_2025_DATE_RANGE == FINAL_EVALUATION_2025_DATE_RANGE

    def test_2025_is_absent_from_every_development_configuration(self) -> None:
        from mlb_luck_score.config import (
            DEVELOPMENT_SEASONS,
            MLB_REGULAR_SEASON_DATE_RANGES,
            TRAIN_SEASONS,
            VALIDATION_SEASONS,
        )

        for collection in (DEVELOPMENT_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS):
            assert 2025 not in collection
        assert 2025 not in MLB_REGULAR_SEASON_DATE_RANGES

    def test_2026_can_never_be_selected(self) -> None:
        """2026 may appear in prose ("never touches 2026"); it must never
        appear as a season value or a date the runner could fetch.
        """
        from mlb_luck_score.config import PROSPECTIVE_SEASONS

        assert runner.REPLICATION_SEASON == 2025
        assert 2025 not in PROSPECTIVE_SEASONS
        assert all(d.startswith("2025-") for d in runner.PITCHER_REPLICATION_2025_DATE_RANGE)

        source = Path(runner.__file__).read_text()
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        for selector in ("2026-", "= 2026", "[2026]", "(2026", "2026,"):
            assert selector not in code, f"2026 must never appear as a selector ({selector})"

    def test_the_runner_cannot_reach_a_prospective_season(self) -> None:
        """Even with a minted token, the guard only ever unlocks 2025."""
        source = Path(pre.__file__).read_text()
        assert "assert_seasons_allowed([REPLICATION_SEASON], allow_final_evaluation=True)" in source
        assert runner.REPLICATION_SEASON == 2025


class TestAuthorizationTokenIsUnforgeable:
    def test_a_bare_true_is_refused(self) -> None:
        with pytest.raises(pre.ExecutionError, match="bare boolean is never accepted"):
            pre.require_authorization(True, "test")

    def test_a_fabricated_token_is_refused(self) -> None:
        forged = pre.ReplicationAuthorization(
            token="deadbeef", freeze_content_hash="x", execution_manifest_hash="y", granted_at="z"
        )
        with pytest.raises(pre.ExecutionError, match="fabricated token"):
            pre.require_authorization(forged, "test")

    def test_final_evaluation_permission_is_unreachable_from_development_code(self) -> None:
        """`allow_final_evaluation=True` appears only behind the token gate."""
        from mlb_luck_score.config import ProtectedSeasonError, assert_seasons_allowed

        with pytest.raises(ProtectedSeasonError):
            assert_seasons_allowed([2025])

        # The permission is minted in exactly one place...
        gate = Path(pre.__file__).read_text().split("def require_authorization", 1)[1]
        assert "allow_final_evaluation=True" in gate

        # ...and every OTHER use in the runner sits inside a function whose
        # first statement is a require_authorization call.
        source = Path(runner.__file__).read_text()
        for block in source.split("\ndef ")[1:]:
            if "allow_final_evaluation=True" not in block:
                continue
            body = block.split('"""', 2)[-1]
            assert "require_authorization(" in body, (
                f"allow_final_evaluation=True used in {block.split('(')[0]} without "
                "requiring a minted authorization first"
            )

    def test_ingestion_refuses_without_a_minted_authorization(self) -> None:
        for fn in (
            runner.ingest_2025_raw,
            runner.ingest_2025_game_metadata,
            runner.ingest_2025_sprint_speed,
            runner.build_2025_dataset,
        ):
            with pytest.raises(pre.ExecutionError):
                fn(authorization=True)  # type: ignore[arg-type]


class TestNamespaceIsolation:
    @pytest.mark.parametrize(
        "escape",
        [
            "data/processed",
            "data/final_evaluation/2025",
            "outputs/prospective/v1_1",
            "data/forecast",
            "dashboard",
        ],
    )
    def test_writes_outside_the_namespace_are_refused(self, escape: str) -> None:
        with pytest.raises(pre.ExecutionError, match="outside the isolated"):
            pre.assert_within_namespace(REPO_ROOT / escape, prf.REPLICATION_DATA_DIR, label="test")

    def test_path_traversal_is_refused(self) -> None:
        with pytest.raises(pre.ExecutionError, match="outside the isolated"):
            pre.assert_within_namespace(
                prf.REPLICATION_DATA_DIR / ".." / ".." / "processed",
                prf.REPLICATION_DATA_DIR,
                label="test",
            )

    def test_a_path_inside_the_namespace_is_accepted(self) -> None:
        pre.assert_within_namespace(
            prf.REPLICATION_DATA_DIR / "raw.parquet", prf.REPLICATION_DATA_DIR, label="test"
        )

    def test_a_populated_output_namespace_refuses(self, isolated_namespace: Path) -> None:
        pre.REPLICATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (pre.REPLICATION_OUTPUTS_DIR / "results.json").write_text("{}")
        with pytest.raises(pre.ExecutionError, match="already populated"):
            pre.assert_output_namespace_available()

    def test_control_files_alone_do_not_count_as_populated(self, isolated_namespace: Path) -> None:
        (isolated_namespace / prf.FREEZE_PATH.name).write_text("{}")
        (isolated_namespace / "amendments.jsonl").write_text("")
        assert pre.assert_output_namespace_available()["artifacts_dir_has_only_control_files"]


class TestExecutionManifest:
    def _manifest(self) -> pre.ExecutionManifest:
        return pre.build_execution_manifest()

    def test_sealing_requires_a_clean_tree(
        self, isolated_namespace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pre, "working_tree_status", lambda repo_root=None: (False, ["M x"]))
        with pytest.raises(pre.ExecutionError, match="not clean"):
            pre.build_execution_manifest()

    def test_missing_manifest_refuses_readiness(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        with pytest.raises(pre.ExecutionError, match="No pre-execution manifest"):
            pre.read_execution_manifest()

    def test_identical_reseal_is_idempotent(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        path, first = pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        before = path.read_text()
        _, second = pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        assert second == first
        assert path.read_text() == before

    def test_changed_execution_code_refuses_overwrite(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        drifted = self._manifest()
        hashes = dict(drifted.execution_source_hashes)
        hashes["replication/run_pitcher_replication_2025.py"] = "0" * 64
        object.__setattr__(drifted, "execution_source_hashes", hashes)
        with pytest.raises(pre.ExecutionError, match="write-once"):
            pre.write_execution_manifest(drifted, path=pre.EXECUTION_MANIFEST_PATH)

    def test_no_amendment_path_can_rewrite_the_original(self) -> None:
        source = Path(pre.__file__).read_text()
        assert "def amend_execution_manifest" not in source
        assert "NO amendment mechanism that rewrites the original" in source

    def test_source_drift_after_sealing_is_detected(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        hashes = dict(manifest.execution_source_hashes)
        hashes["replication/pitcher_replication_questions.py"] = "1" * 64
        object.__setattr__(manifest, "execution_source_hashes", hashes)
        with pytest.raises(pre.ExecutionError, match="changed since sealing"):
            pre.validate_execution_manifest(manifest)

    def test_a_changed_freeze_invalidates_the_execution_manifest(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        object.__setattr__(manifest, "freeze_content_hash", "2" * 64)
        with pytest.raises(pre.ExecutionError, match="freeze changed"):
            pre.validate_execution_manifest(manifest)

    def test_a_changed_authorization_invalidates_the_execution_manifest(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        object.__setattr__(manifest, "authorization_content_hash", "3" * 64)
        with pytest.raises(pre.ExecutionError, match="authorization record changed"):
            pre.validate_execution_manifest(manifest)

    def test_the_manifest_pins_the_runner_and_the_authorization(self) -> None:
        pinned = set(pre.EXECUTION_SOURCE_RELATIVE_PATHS)
        assert "replication/run_pitcher_replication_2025.py" in pinned
        assert "replication/pitcher_replication_questions.py" in pinned
        assert "replication/pitcher_replication_authorization.py" in pinned
        assert "Makefile" in pinned

    def test_the_manifest_records_the_required_preconditions(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = self._manifest()
        pre_ = manifest.preconditions
        assert pre_["freeze_validates"] is True
        assert pre_["frozen_sources_verified"] == 15
        assert pre_["authorization_resolves"] is True
        assert pre_["season_2025_opened"] is False
        assert pre_["season_2026_opened"] is False
        assert pre_["receipt_exists"] is False
        assert manifest.working_tree_clean is True
        assert manifest.date_range == list(runner.PITCHER_REPLICATION_2025_DATE_RANGE)


class TestOneTimeExecutionReceipt:
    def test_receipt_records_the_full_binding(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = pre.build_execution_manifest()
        pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        auth = pre._mint_authorization("f", manifest.manifest_content_hash())
        receipt = pre.write_execution_start_receipt(auth, manifest)
        for key in (
            "execution_id",
            "opened_at_utc",
            "freeze_content_hash",
            "spec_content_hash",
            "authorization_content_hash",
            "execution_manifest_hash",
            "repository_commit",
            "statement",
            "recovery_procedure",
        ):
            assert key in receipt
        assert receipt["one_time_use"] is True
        assert "no 2025 data had been read" in receipt["statement"]

    def test_a_second_execution_refuses(self, isolated_namespace: Path, clean_tree: None) -> None:
        manifest = pre.build_execution_manifest()
        pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        auth = pre._mint_authorization("f", manifest.manifest_content_hash())
        pre.write_execution_start_receipt(auth, manifest)

        with pytest.raises(pre.ExecutionError, match="already been opened"):
            pre.assert_no_execution_receipt()
        with pytest.raises(pre.ExecutionError, match="Refusing a second look"):
            pre.write_execution_start_receipt(auth, manifest)

    def test_the_refusal_names_the_recovery_procedure(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = pre.build_execution_manifest()
        pre.write_execution_manifest(manifest, path=pre.EXECUTION_MANIFEST_PATH)
        auth = pre._mint_authorization("f", manifest.manifest_content_hash())
        pre.write_execution_start_receipt(auth, manifest)
        with pytest.raises(pre.ExecutionError, match="Do NOT delete the receipt"):
            pre.assert_no_execution_receipt()

    def test_there_is_no_automatic_retry(self) -> None:
        source = Path(runner.__file__).read_text()
        for pattern in (
            "for attempt in",
            "while True",
            "except Exception:\n        return run_replication",
        ):
            assert pattern not in source, "the runner must never silently reopen 2025"


class TestReadinessPrecedesDataAccess:
    def test_readiness_runs_before_the_first_data_access(
        self, isolated_namespace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instrument the first attempted data access and prove readiness
        already ran. Readiness is made to FAIL here, so the run aborts before
        any ingestion -- which is exactly the property under test.
        """
        events: list[str] = []

        def record_and_explode(*args: Any, **kwargs: Any) -> Any:
            events.append("DATA_ACCESS")
            raise AssertionError("data access attempted")

        monkeypatch.setattr(pd, "read_parquet", record_and_explode)

        real_readiness = pre.run_readiness_checks

        def spy(*args: Any, **kwargs: Any) -> Any:
            events.append("READINESS")
            return real_readiness(*args, **kwargs)

        monkeypatch.setattr(runner, "run_readiness_checks", spy)

        with pytest.raises((pre.ExecutionError, prf.FreezeError)):
            runner.run_replication()

        assert events, "run_replication did nothing at all"
        assert events[0] == "READINESS", f"readiness must come first, got {events}"
        assert "DATA_ACCESS" not in events, "no data may be touched when readiness fails"

    def test_readiness_failure_writes_no_receipt(self, isolated_namespace: Path) -> None:
        with pytest.raises((pre.ExecutionError, prf.FreezeError)):
            runner.run_replication()
        assert not pre.EXECUTION_RECEIPT_PATH.exists()

    def test_run_replication_calls_readiness_before_anything_else(self) -> None:
        source = Path(runner.__file__).read_text()
        body = source.split("def run_replication", 1)[1]
        readiness_at = body.index("run_readiness_checks(")
        for later in ("write_execution_start_receipt(", "ingest_2025_raw(", "train_and_score_2025"):
            assert body.index(later) > readiness_at, f"{later} must follow readiness"

    def test_the_receipt_precedes_every_ingestion_call(self) -> None:
        body = Path(runner.__file__).read_text().split("def run_replication", 1)[1]
        receipt_at = body.index("write_execution_start_receipt(")
        for ingest in (
            "ingest_2025_raw(",
            "ingest_2025_game_metadata(",
            "ingest_2025_sprint_speed(",
        ):
            assert body.index(ingest) > receipt_at, f"{ingest} must follow the receipt"


class TestFrozenQuestionsAndClassification:
    def test_all_seven_questions_are_answered(self) -> None:
        frame = _synthetic_pitcher_frame()
        answers = {
            "A_population_and_centering": prq.question_a(frame, 0.0),
            "B_opportunity_heterogeneity": prq.question_b(frame),
            "C_totals_vs_rate": prq.question_c(frame),
            "D_reliever_single_play_dominance": prq.question_d(frame),
            "E_rate_precision": prq.question_e(frame),
            "F_persistence": {"agrees": True},
            "G_real_play_sanity_check": prq.question_g(frame),
        }
        report = {"questions": answers, "classification": prq.classify(answers)}
        prq.assert_report_covers_every_question(report)
        assert set(answers) == set(prq.QUESTION_KEYS)

    def test_an_extra_question_is_refused(self) -> None:
        report = {"questions": dict.fromkeys(prq.QUESTION_KEYS, {}) | {"H_new_idea": {}}}
        with pytest.raises(prq.QuestionError, match="unexpected"):
            prq.assert_report_covers_every_question(report)

    def test_a_missing_question_is_refused(self) -> None:
        keys = list(prq.QUESTION_KEYS)[:-1]
        with pytest.raises(prq.QuestionError, match="missing"):
            prq.assert_report_covers_every_question({"questions": dict.fromkeys(keys, {})})

    def test_question_e_reports_the_one_bbe_artifact_as_an_artifact(self) -> None:
        answer = prq.question_e(_synthetic_pitcher_frame())
        artifact = answer["one_bbe_floor_artifact"]
        assert artifact["is_an_artifact_never_evidence_of_signal"] is True
        assert "raw" in artifact and "excluding_zero_width" in artifact
        assert answer["practical_floors"] == [60, 150, 300, 450]

    def test_question_g_is_reported_not_scored(self) -> None:
        answer = prq.question_g(_synthetic_pitcher_frame())
        assert answer["agrees"] is None
        assert answer["is_qualitative"] is True

    def test_all_primary_agreeing_gives_replicated(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        answers["G_real_play_sanity_check"] = {"agrees": None}
        assert prq.classify(answers)["classification"] == "REPLICATED"

    def test_a_presentation_failure_gives_revise(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        answers["D_reliever_single_play_dominance"] = {"agrees": False}
        result = prq.classify(answers)
        assert result["classification"] == "REVISE"
        assert "D_reliever_single_play_dominance" in result["primary_disagreements"]

    def test_a_coherence_failure_gives_no_go(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        answers["E_rate_precision"] = {"agrees": False}
        assert prq.classify(answers)["classification"] == "NO_GO"

    def test_question_f_alone_cannot_change_the_verdict(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        answers["F_persistence"] = {"agrees": False}
        result = prq.classify(answers)
        assert result["classification"] == "REPLICATED"
        assert result["secondary_disagreements"] == ["F_persistence"]

    def test_every_disagreement_is_reported_even_under_replicated(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        answers["F_persistence"] = {"agrees": False}
        result = prq.classify(answers)
        assert result["all_disagreements_reported"] is True
        assert result["secondary_disagreements"]

    def test_classification_does_not_use_significance_testing(self) -> None:
        answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
        result = prq.classify(answers)
        assert result["significance_testing_was_not_used"] is True
        assert result["decided_on_the_package_not_one_statistic"] is True
        assert "p_value" not in Path(prq.__file__).read_text()

    def test_the_verdict_is_always_a_frozen_value(self) -> None:
        from pitcher_replication_spec import CLASSIFICATION_VALUES

        for failing in prq.QUESTION_KEYS:
            answers = {k: {"agrees": True} for k in prq.QUESTION_KEYS}
            answers[failing] = {"agrees": False}
            assert prq.classify(answers)["classification"] in CLASSIFICATION_VALUES


class TestOutputSchema:
    def test_the_results_schema_is_complete(self, isolated_namespace: Path) -> None:
        """Assemble the real output payload from synthetic answers."""
        frame = _synthetic_pitcher_frame()
        answers = {
            "A_population_and_centering": prq.question_a(frame, 0.0),
            "B_opportunity_heterogeneity": prq.question_b(frame),
            "C_totals_vs_rate": prq.question_c(frame),
            "D_reliever_single_play_dominance": prq.question_d(frame),
            "E_rate_precision": prq.question_e(frame),
            "F_persistence": {"agrees": True, "question": "F_persistence"},
            "G_real_play_sanity_check": prq.question_g(frame),
        }
        report = {"questions": answers, "classification": prq.classify(answers)}
        freeze = prf.read_freeze()
        results = {
            "replication_season": 2025,
            "spec_version": freeze.spec_version,
            "freeze_content_hash": freeze.freeze_content_hash(),
            "spec_content_hash": freeze.spec_content_hash,
            "pitcher_season_rows": len(frame),
            **report,
            "classification_is_frozen_before_any_design_change": True,
        }
        for key in (
            "replication_season",
            "spec_version",
            "freeze_content_hash",
            "spec_content_hash",
            "questions",
            "classification",
            "classification_is_frozen_before_any_design_change",
        ):
            assert key in results
        assert json.dumps(results, default=str)

    def test_intended_output_paths_are_inside_the_namespace(self) -> None:
        for path in (runner.RESULTS_PATH, runner.PROVENANCE_PATH):
            assert prf.REPLICATION_OUTPUTS_DIR in path.parents
        assert prf.ARTIFACTS_DIR in runner.SEAL_PATH.parents


class TestControlPathsResolveAtCallTime:
    """Regression guard.

    These paths were once definition-time defaults, so redirecting the module
    attribute silently had no effect and a test wrote a receipt into the REAL
    artifacts namespace -- which would have permanently blocked the authorized
    run. They must resolve from the module attribute when called.
    """

    def test_receipt_writes_honour_a_redirected_path(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        manifest = pre.build_execution_manifest()
        pre.write_execution_manifest(manifest)
        auth = pre._mint_authorization("f", manifest.manifest_content_hash())
        pre.write_execution_start_receipt(auth, manifest)

        assert (isolated_namespace / "execution_start_receipt.json").exists()
        real = (
            Path(__file__).resolve().parent.parent
            / "artifacts/pitcher_replication/v0_14/execution_start_receipt.json"
        )
        # A real run has since written a receipt here. The guard is that a test
        # must neither create nor MUTATE it -- absence is no longer the check.
        assert real.read_bytes() == self._real_receipt_bytes, (
            "a test must never write to, or overwrite, the real receipt"
        )

    @pytest.fixture(autouse=True)
    def _capture_real_receipt(self) -> None:
        real = (
            Path(__file__).resolve().parent.parent
            / "artifacts/pitcher_replication/v0_14/execution_start_receipt.json"
        )
        self._real_receipt_bytes = real.read_bytes() if real.exists() else None

    def test_manifest_writes_honour_a_redirected_path(
        self, isolated_namespace: Path, clean_tree: None
    ) -> None:
        pre.write_execution_manifest(pre.build_execution_manifest())
        assert (isolated_namespace / "pitcher_replication_execution_manifest.json").exists()


class TestPostExposureInvariants:
    """2025 was opened on 2026-09-08 and the run failed inside the frozen
    scorer. These assert what must hold AFTER that exposure -- the earlier
    "nothing exists yet" versions of these tests were correct only before it.
    """

    def test_the_execution_receipt_is_preserved(self) -> None:
        """The receipt records that the one-time authorization is spent. It
        must never be deleted to make a rerun look like a first look.
        """
        assert pre.EXECUTION_RECEIPT_PATH.exists()
        receipt = json.loads(pre.EXECUTION_RECEIPT_PATH.read_text())
        assert receipt["one_time_use"] is True
        assert receipt["execution_id"]

    def test_a_rerun_is_refused_while_the_receipt_stands(self) -> None:
        with pytest.raises(pre.ExecutionError, match="already been opened"):
            pre.assert_no_execution_receipt()

    def test_no_replication_result_was_produced(self) -> None:
        """Ingestion happened; scoring did not. No A-G, no classification."""
        outputs = REPO_ROOT / "outputs" / "pitcher_replication" / "v0_14"
        files = (
            [p for p in outputs.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            if outputs.exists()
            else []
        )
        assert files == []
        assert not (prf.ARTIFACTS_DIR / "pitcher_replication_2025_seal.json").exists()

    def test_the_research_freeze_survived_the_incident(self) -> None:
        assert prf.validate_freeze(prf.read_freeze())["valid"] is True
