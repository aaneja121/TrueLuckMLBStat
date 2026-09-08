"""Version 0.14: guards for the pre-registered 2025 pitcher-replication freeze.

Every test here is offline and touches no season's data. Several of them
exist specifically to prove that -- see `TestNoSeasonAccess`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pitcher_replication_freeze as prf
import pitcher_replication_spec as spec
import pytest


@pytest.fixture
def isolated_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every write path at `tmp_path`, so no test can touch the real
    artifact or the real replication namespace.
    """
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(prf, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(prf, "FREEZE_PATH", artifacts / "pitcher_replication_freeze.json")
    monkeypatch.setattr(prf, "AMENDMENT_LOG_PATH", artifacts / "amendments.jsonl")
    monkeypatch.setattr(prf, "REPLICATION_DATA_DIR", tmp_path / "data" / "2025")
    monkeypatch.setattr(prf, "REPLICATION_OUTPUTS_DIR", tmp_path / "outputs" / "v0_14")
    monkeypatch.setattr(prf, "REPO_ROOT", prf.Path(__file__).resolve().parent.parent)
    return artifacts


class TestSpecificationIsFrozenAndCoherent:
    def test_seasons_are_development_2024_and_replication_2025(self) -> None:
        assert spec.DEVELOPMENT_SEASON == 2024
        assert spec.REPLICATION_SEASON == 2025

    def test_2026_is_prohibited(self) -> None:
        assert 2026 in spec.PROHIBITED_SEASONS

    def test_play_level_definition_is_expected_minus_observed(self) -> None:
        formula = spec.PLAY_LEVEL_DEFINITION["formula"]
        assert "expected_run_value - observed_run_value" in formula
        assert spec.PLAY_LEVEL_DEFINITION["trains_a_pitcher_specific_model"] is False
        assert spec.PLAY_LEVEL_DEFINITION["reuses_frozen_architecture"] is True

    def test_primary_quantity_is_the_cumulative_total(self) -> None:
        assert spec.PRIMARY_QUANTITY["name"] == "cumulative_contact_luck_runs"
        assert spec.PRIMARY_QUANTITY["is_workload_sensitive_by_design"] is True
        for banned in ("pitcher skill", "talent", "persistence", "a forecast"):
            assert banned in spec.PRIMARY_QUANTITY["is_not"]

    def test_per_100_is_secondary_and_never_the_ranking_key(self) -> None:
        assert spec.SECONDARY_QUANTITY["status"] == "secondary_only"
        assert spec.SECONDARY_QUANTITY["may_be_the_official_ranking_key"] is False
        assert spec.PRESENTATION_RULES["per_100_may_be_primary_ordering"] is False
        assert spec.PRESENTATION_RULES["primary_ordering"] == "cumulative_contact_luck_runs"

    def test_per_100_must_be_paired_with_exposure_and_an_interval(self) -> None:
        paired = " ".join(spec.SECONDARY_QUANTITY["must_always_appear_with"])
        assert "BBE" in paired
        assert "95%" in paired

    def test_denominator_is_resolved_eligible_bbe_and_is_not_redefined(self) -> None:
        assert spec.DENOMINATOR["meaning"] == "OUTCOME-RESOLVED eligible batted balls"
        assert spec.DENOMINATOR["redefined_for_the_pitcher_stage"] is False
        assert "notna()" in spec.DENOMINATOR["resolved_test"]

    def test_denominator_records_field_error_and_fielders_choice_handling(self) -> None:
        handling = spec.DENOMINATOR["unresolved_row_handling"]
        assert "field_error" in handling
        assert "fielders_choice" in handling
        assert "never enters this denominator" in handling["field_error"]
        assert "fielders_choice_out" in handling["fielders_choice"]

    def test_role_grouping_boundaries_match_the_2024_derivation(self) -> None:
        assert spec.ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE == 10.0
        assert spec.ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE == 8.0
        assert spec.ROLE_LIKE_GROUPING["is_official_role_metadata"] is False
        assert spec.ROLE_LIKE_GROUPING["is_a_qualification_system"] is False

    def test_closer_and_official_role_labels_are_banned(self) -> None:
        for banned in ("closer", "starter", "reliever"):
            assert banned in spec.ROLE_LIKE_GROUPING["banned_labels"]

    def test_board_minimum_is_display_only_not_qualification(self) -> None:
        assert spec.BOARD_DISPLAY_MINIMUM_BBE == 60
        assert spec.PRESENTATION_RULES["board_display_minimum_is_display_only"] is True
        assert spec.PRESENTATION_RULES["board_display_minimum_is_qualification"] is False

    def test_boards_are_never_combined_and_carry_no_player_card_rank(self) -> None:
        assert spec.PRESENTATION_RULES["single_combined_official_ranked_board"] is False
        assert spec.PRESENTATION_RULES["starter_like_and_reliever_like_analysed_separately"]
        assert spec.PRESENTATION_RULES["board_rank_shown_on_player_card"] is False

    def test_bbe_is_always_visible_and_largest_plays_are_surfaced(self) -> None:
        assert spec.PRESENTATION_RULES["bbe_always_visible_beside_the_total"] is True
        assert spec.PRESENTATION_RULES["largest_favorable_and_unfavorable_plays_surfaced"]

    def test_presentation_rules_may_not_be_reopened_without_a_reported_outcome(self) -> None:
        assert "REVISE or NO-GO" in spec.PRESENTATION_RULES["reopening_rule"]

    def test_interval_procedure_is_frozen_against_2025(self) -> None:
        assert spec.INTERVAL_PROCEDURE["may_be_changed_after_seeing_2025"] is False


class TestReplicationQuestions:
    EXPECTED = (
        "A_population_and_centering",
        "B_opportunity_heterogeneity",
        "C_totals_vs_rate",
        "D_reliever_single_play_dominance",
        "E_rate_precision",
        "F_persistence",
        "G_real_play_sanity_check",
    )

    def test_all_seven_questions_are_present(self) -> None:
        assert tuple(spec.REPLICATION_QUESTIONS) == self.EXPECTED

    @pytest.mark.parametrize("key", EXPECTED)
    def test_every_question_states_what_it_reports(self, key: str) -> None:
        assert spec.REPLICATION_QUESTIONS[key]["report"]

    @pytest.mark.parametrize("key", EXPECTED)
    def test_every_question_has_a_prespecified_structural_hypothesis(self, key: str) -> None:
        assert spec.REPLICATION_QUESTIONS[key]["structural_hypothesis"].strip()

    @pytest.mark.parametrize("key", EXPECTED)
    def test_every_question_says_what_disagreement_looks_like(self, key: str) -> None:
        assert spec.REPLICATION_QUESTIONS[key]["disagreement_looks_like"].strip()

    @pytest.mark.parametrize("key", EXPECTED)
    def test_no_question_requires_exact_numeric_replication(self, key: str) -> None:
        assert spec.REPLICATION_QUESTIONS[key]["requires_exact_numeric_replication"] is False

    def test_question_e_carries_the_zero_width_artifact_guard(self) -> None:
        guard = spec.REPLICATION_QUESTIONS["E_rate_precision"]["mandatory_guard"]
        assert "zero-width" in guard
        assert "one-appearance" in guard
        assert "1.94" in guard and "0.877" in guard
        assert spec.FROZEN_ESTIMATORS["resolving_power"]["excluded_floor"] == 1

    def test_question_e_floors_exclude_the_one_bbe_floor(self) -> None:
        floors = spec.FROZEN_ESTIMATORS["resolving_power"]["workload_floors"]
        assert 1 not in floors
        assert floors == (60, 150, 300, 450)

    def test_question_f_is_now_implemented_and_frozen(self) -> None:
        question = spec.REPLICATION_QUESTIONS["F_persistence"]
        assert question["procedure_status"] == "IMPLEMENTED_AND_FROZEN"
        assert question["implementation_module"] == "replication.pitcher_split_half"
        assert question["procedure"]["splits_are_game_clustered"] is True
        assert question["procedure"]["min_eligible_each_half"] == 20
        assert question["procedure"]["denominator_unchanged"] is True

    def test_question_f_cannot_override_the_package_classification(self) -> None:
        question = spec.REPLICATION_QUESTIONS["F_persistence"]
        assert question["is_a_success_criterion_on_its_own"] is False
        assert question["cannot_override_package_classification"] is True

    def test_frozen_estimator_forms_are_recorded_literally(self) -> None:
        precision = spec.FROZEN_ESTIMATORS["rate_precision_requirement"]
        assert precision["form"] == "half_width_runs_per_100 = k / sqrt(resolved_bbe)"
        assert precision["fitted_on_rows_with_min_bbe"] == 30
        assert precision["development_reference_bbe_required"]["plus_minus_2"] == 1246
        resolving = spec.FROZEN_ESTIMATORS["resolving_power"]
        assert resolving["measurement_sd_per_row"] == "(ci_high - ci_low) / (2 * 1.96)"
        assert resolving["negative_signal_variance_is_reported_raw"] is True


class TestClassificationRule:
    def test_three_values_only(self) -> None:
        assert spec.CLASSIFICATION_VALUES == ("REPLICATED", "REVISE", "NO_GO")

    def test_replicated_requires_every_structural_conclusion(self) -> None:
        required = spec.CLASSIFICATION_RULE["REPLICATED"]["requires_all_of"]
        assert len(required) == 5
        joined = " ".join(required).lower()
        assert "retrospective accounting" in joined
        assert "uncertainty-limited" in joined
        assert "individual plays" in joined

    def test_significance_on_one_statistic_is_not_the_rule(self) -> None:
        assert spec.CLASSIFICATION_RULE["significance_testing_is_not_the_rule"] is True
        note = spec.CLASSIFICATION_RULE["significance_testing_note"]
        assert "must NOT be the classification rule" in note

    def test_every_disagreement_must_be_reported(self) -> None:
        assert spec.CLASSIFICATION_RULE["every_disagreement_must_be_reported"] is True

    def test_classification_precedes_any_design_change(self) -> None:
        assert spec.CLASSIFICATION_RULE["classification_precedes_any_design_change"] is True

    def test_decision_is_on_the_package_of_findings(self) -> None:
        assert "AS A PACKAGE" in spec.CLASSIFICATION_RULE["decision_procedure"]


class TestAuthorizationIsNotGranted:
    def test_second_sealed_evaluation_is_not_authorized(self) -> None:
        assert spec.AUTHORIZATION_STATUS["second_sealed_2025_evaluation_authorized"] is False

    def test_assert_ready_for_2025_refuses_without_signoff(self, isolated_namespace: Path) -> None:
        freeze = prf.build_freeze()
        with pytest.raises(prf.FreezeError, match="SECOND sealed evaluation"):
            prf.assert_ready_for_2025(freeze)

    def test_assert_ready_for_2025_refuses_a_provisional_dirty_tree_freeze(
        self, isolated_namespace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        freeze = prf.build_freeze()
        monkeypatch.setattr(prf, "working_tree_status", lambda repo_root=None: (True, []))
        if freeze.working_tree_clean:
            pytest.skip("tree is clean; the provisional-freeze branch is unreachable here")
        with pytest.raises(prf.FreezeError, match="PROVISIONAL"):
            prf.assert_ready_for_2025(freeze, maintainer_authorized_second_sealed_evaluation=True)


class TestSeasonProtectionGuards:
    def test_2025_protection_is_intact(self) -> None:
        report = prf.assert_2025_protection_intact()
        assert report["final_test_seasons"] == [2025]
        assert 2025 not in report["development_seasons"]
        assert report["allow_final_evaluation_required_for_2025"] is True

    def test_guard_fires_if_2025_enters_the_development_seasons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlb_luck_score.config as config

        monkeypatch.setattr(config, "DEVELOPMENT_SEASONS", (2021, 2022, 2023, 2024, 2025))
        with pytest.raises(prf.FreezeError, match="DEVELOPMENT_SEASONS"):
            prf.assert_2025_protection_intact()

    def test_guard_fires_if_2025_gains_a_development_date_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlb_luck_score.config as config

        ranges = dict(config.MLB_REGULAR_SEASON_DATE_RANGES)
        ranges[2025] = ("2025-03-27", "2025-09-28")
        monkeypatch.setattr(config, "MLB_REGULAR_SEASON_DATE_RANGES", ranges)
        with pytest.raises(prf.FreezeError, match="MLB_REGULAR_SEASON_DATE_RANGES"):
            prf.assert_2025_protection_intact()

    def test_guard_fires_if_2025_leaves_the_final_test_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlb_luck_score.config as config

        monkeypatch.setattr(config, "FINAL_TEST_SEASONS", ())
        with pytest.raises(prf.FreezeError, match="FINAL_TEST_SEASONS"):
            prf.assert_2025_protection_intact()

    def test_empty_replication_namespace_is_the_pre_outcome_evidence(
        self, isolated_namespace: Path
    ) -> None:
        report = prf.assert_no_replication_outputs_exist()
        assert report["files_present"] == []

    def test_a_pre_existing_2025_result_blocks_the_freeze(self, isolated_namespace: Path) -> None:
        prf.REPLICATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (prf.REPLICATION_OUTPUTS_DIR / "pitcher_2025.json").write_text("{}")
        with pytest.raises(prf.FreezeError, match="NOT empty"):
            prf.assert_no_replication_outputs_exist()
        with pytest.raises(prf.FreezeError, match="NOT empty"):
            prf.build_freeze()


class TestNoSeasonAccess:
    """The freeze must be constructible with every data door nailed shut."""

    def test_building_the_freeze_reads_no_dataframe_and_makes_no_request(
        self, isolated_namespace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pandas as pd
        import requests

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the freeze must not read data or hit the network")

        for name in ("read_parquet", "read_csv", "read_json"):
            monkeypatch.setattr(pd, name, explode)
        for name in ("get", "post", "request"):
            monkeypatch.setattr(requests, name, explode)

        freeze = prf.build_freeze()
        assert freeze.spec_content_hash

    def test_freeze_modules_reference_no_season_data_path(self) -> None:
        for module in (
            Path(prf.__file__),
            Path(spec.__file__),
        ):
            text = module.read_text()
            for forbidden in ("read_parquet", "download_statcast", "pybaseball", "requests."):
                assert forbidden not in text, f"{module.name} references {forbidden}"

    def test_attestation_records_that_no_season_was_opened(self, isolated_namespace: Path) -> None:
        attestation = prf.build_freeze().pre_outcome_attestation
        assert attestation["any_2025_data_read_while_building_this_freeze"] is False
        assert attestation["any_2025_pitcher_output_computed"] is False
        assert attestation["any_2026_data_read_while_building_this_freeze"] is False
        assert attestation["freeze_precedes_evaluation"] is True


class TestResolvedConstantsMatchTheRealModules:
    def test_pitcher_thresholds_come_from_the_scoring_module(self) -> None:
        from mlb_luck_score.scoring.pitching_contact_luck import (
            PITCHER_QUALIFICATION_THRESHOLD_SETS,
        )

        resolved = prf.resolved_frozen_constants()["pitcher_qualification_threshold_sets"]
        assert set(resolved) == set(PITCHER_QUALIFICATION_THRESHOLD_SETS)
        assert resolved["pitcher_primary"]["min_eligible_batted_balls"] == 450
        assert resolved["pitcher_primary"]["min_games"] == 15

    def test_interval_design_comes_from_bootstrap_design(self) -> None:
        design = prf.resolved_frozen_constants()["interval_design"]
        assert design["method"] == "percentile"
        assert design["resampling_unit"] == "game_pk_within_batter_season"
        assert design["alpha"] == 0.05
        assert design["refits_models"] is False

    def test_ambiguous_events_excluded_from_the_denominator_are_read_from_eligibility(
        self,
    ) -> None:
        excluded = prf.resolved_frozen_constants()[
            "ambiguous_outcome_events_excluded_from_denominator"
        ]
        assert excluded == ["field_error", "fielders_choice"]

    def test_presentation_constants_match_the_specification(self) -> None:
        presentation = prf.resolved_frozen_constants()["presentation_constants"]
        assert presentation["board_display_minimum_bbe"] == spec.BOARD_DISPLAY_MINIMUM_BBE
        assert (
            presentation["starter_like_min_bbe_per_appearance"]
            == spec.ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE
        )


class TestWriteOnce:
    def test_first_write_creates_the_artifact(self, isolated_namespace: Path) -> None:
        freeze = prf.build_freeze()
        path, digest = prf.write_freeze(freeze, path=prf.FREEZE_PATH)
        assert path.is_file()
        assert digest == freeze.freeze_content_hash()
        assert json.loads(path.read_text())["spec_version"] == spec.SPEC_VERSION

    def test_rewriting_an_identical_freeze_is_idempotent(self, isolated_namespace: Path) -> None:
        first = prf.build_freeze()
        path, digest = prf.write_freeze(first, path=prf.FREEZE_PATH)
        before = path.read_text()

        second = prf.build_freeze()
        path_again, digest_again = prf.write_freeze(second, path=prf.FREEZE_PATH)

        assert (path_again, digest_again) == (path, digest)
        assert path.read_text() == before, "an idempotent write must not rewrite the file"

    def test_writing_a_different_freeze_is_refused(self, isolated_namespace: Path) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        mutated = prf.build_freeze(notes=["unrelated"])
        object.__setattr__(mutated, "spec_content_hash", "0" * 64)
        with pytest.raises(prf.FreezeError, match="write-once"):
            prf.write_freeze(mutated, path=prf.FREEZE_PATH)

    def test_a_provisional_freeze_may_be_superseded_during_construction(
        self, isolated_namespace: Path
    ) -> None:
        first = prf.build_freeze()
        if first.working_tree_clean:
            pytest.skip("tree is clean; there is no provisional freeze to supersede")
        path, first_hash = prf.write_freeze(first, path=prf.FREEZE_PATH)

        second = prf.build_freeze(notes=["spec reformatted during construction"])
        object.__setattr__(second, "spec_content_hash", "4" * 64)
        _, second_hash = prf.write_freeze(second, path=prf.FREEZE_PATH, rebuild_provisional=True)

        assert second_hash != first_hash
        on_disk = json.loads(path.read_text())
        assert on_disk["spec_content_hash"] == "4" * 64
        assert on_disk["notes"] == ["spec reformatted during construction"]

    def test_an_unchanged_rebuild_is_still_idempotent(self, isolated_namespace: Path) -> None:
        first = prf.build_freeze()
        path, first_hash = prf.write_freeze(first, path=prf.FREEZE_PATH)
        before = path.read_text()

        # `notes` is deliberately excluded from the content hash, so a rebuild
        # that differs only in notes is the SAME freeze and must not rewrite.
        second = prf.build_freeze(notes=["cosmetic"])
        _, second_hash = prf.write_freeze(second, path=prf.FREEZE_PATH, rebuild_provisional=True)

        assert second_hash == first_hash
        assert path.read_text() == before

    def test_superseding_archives_the_outgoing_copy(self, isolated_namespace: Path) -> None:
        first = prf.build_freeze()
        if first.working_tree_clean:
            pytest.skip("tree is clean; there is no provisional freeze to supersede")
        path, _ = prf.write_freeze(first, path=prf.FREEZE_PATH)
        object.__setattr__(first, "spec_content_hash", "2" * 64)
        prf.write_freeze(first, path=prf.FREEZE_PATH, rebuild_provisional=True)

        archived = list((path.parent / "superseded").glob("*.json"))
        assert len(archived) == 1, "the superseded provisional freeze must be archived"

    def test_an_established_clean_tree_freeze_cannot_be_rebuilt(
        self, isolated_namespace: Path
    ) -> None:
        established = prf.build_freeze()
        object.__setattr__(established, "working_tree_clean", True)
        object.__setattr__(established, "working_tree_dirty_paths", [])
        prf.write_freeze(established, path=prf.FREEZE_PATH)

        replacement = prf.build_freeze(notes=["late change"])
        object.__setattr__(replacement, "spec_content_hash", "3" * 64)
        with pytest.raises(prf.FreezeError, match="established, not provisional"):
            prf.write_freeze(replacement, path=prf.FREEZE_PATH, rebuild_provisional=True)

    def test_the_refusal_points_at_the_amendment_path(self, isolated_namespace: Path) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        mutated = prf.build_freeze()
        object.__setattr__(mutated, "spec_content_hash", "1" * 64)
        with pytest.raises(prf.FreezeError, match="amend_freeze"):
            prf.write_freeze(mutated, path=prf.FREEZE_PATH)


class TestAmendment:
    def test_amendment_without_a_pre_outcome_attestation_is_refused(
        self, isolated_namespace: Path
    ) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        with pytest.raises(prf.FreezeError, match="post-hoc revision"):
            prf.amend_freeze(reason="looks better", recorded_before_any_2025_access=False)

    def test_amendment_without_a_reason_is_refused(self, isolated_namespace: Path) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        with pytest.raises(prf.FreezeError, match="stated reason"):
            prf.amend_freeze(reason="   ", recorded_before_any_2025_access=True)

    def test_amendment_writes_a_new_revision_without_mutating_the_original(
        self, isolated_namespace: Path
    ) -> None:
        path, _ = prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        original = path.read_text()

        revision_path, _ = prf.amend_freeze(
            reason="freeze a split-half implementation for question F",
            recorded_before_any_2025_access=True,
        )

        assert revision_path.is_file()
        assert revision_path != path
        assert path.read_text() == original, "the original revision must never be mutated"
        amended = json.loads(revision_path.read_text())
        assert amended["revision"] == 2
        assert amended["amendment"]["recorded_before_any_2025_access"] is True
        assert "split-half" in amended["amendment"]["reason"]

    def test_amendment_is_appended_to_the_log(self, isolated_namespace: Path) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        prf.amend_freeze(reason="first", recorded_before_any_2025_access=True)
        entries = [
            json.loads(line)
            for line in prf.AMENDMENT_LOG_PATH.read_text().splitlines()
            if line.strip()
        ]
        assert len(entries) == 1
        assert entries[0]["reason"] == "first"
        assert entries[0]["revision"] == 2

    def test_amendment_is_refused_once_a_2025_result_exists(self, isolated_namespace: Path) -> None:
        prf.write_freeze(prf.build_freeze(), path=prf.FREEZE_PATH)
        prf.REPLICATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (prf.REPLICATION_OUTPUTS_DIR / "result.json").write_text("{}")
        with pytest.raises(prf.FreezeError, match="NOT empty"):
            prf.amend_freeze(reason="too late", recorded_before_any_2025_access=True)


class TestValidation:
    def test_a_fresh_freeze_validates(self, isolated_namespace: Path) -> None:
        report = prf.validate_freeze(prf.build_freeze())
        assert report["valid"] is True
        assert report["source_files_verified"] == len(prf.FROZEN_SOURCE_RELATIVE_PATHS)

    def test_a_changed_specification_invalidates_the_freeze(self, isolated_namespace: Path) -> None:
        freeze = prf.build_freeze()
        object.__setattr__(freeze, "spec_content_hash", "0" * 64)
        with pytest.raises(prf.FreezeError, match="specification changed"):
            prf.validate_freeze(freeze)

    def test_a_changed_frozen_source_invalidates_the_freeze(self, isolated_namespace: Path) -> None:
        freeze = prf.build_freeze()
        hashes = dict(freeze.source_hashes)
        hashes["src/mlb_luck_score/scoring/pitching_contact_luck.py"] = "0" * 64
        object.__setattr__(freeze, "source_hashes", hashes)
        with pytest.raises(prf.FreezeError, match="frozen source file"):
            prf.validate_freeze(freeze)

    def test_a_changed_frozen_source_set_invalidates_the_freeze(
        self, isolated_namespace: Path
    ) -> None:
        freeze = prf.build_freeze()
        hashes = dict(freeze.source_hashes)
        hashes.pop("src/mlb_luck_score/scoring/pitching_contact_luck.py")
        object.__setattr__(freeze, "source_hashes", hashes)
        with pytest.raises(prf.FreezeError, match="frozen source set changed"):
            prf.validate_freeze(freeze)

    def test_every_frozen_source_path_exists(self) -> None:
        for rel_path in prf.FROZEN_SOURCE_RELATIVE_PATHS:
            assert (prf.REPO_ROOT / rel_path).is_file(), rel_path

    def test_the_pitcher_side_and_spec_modules_are_both_pinned(self) -> None:
        pinned = set(prf.FROZEN_SOURCE_RELATIVE_PATHS)
        assert "src/mlb_luck_score/scoring/pitching_contact_luck.py" in pinned
        assert "replication/pitcher_replication_spec.py" in pinned
        assert "replication/pitcher_replication_estimators.py" in pinned
        assert "replication/pitcher_split_half.py" in pinned
        # Question F imports its split rules from here, so it is a frozen input.
        assert "src/mlb_luck_score/models/evaluate_aggregation_stability.py" in pinned

    def test_the_freeze_no_longer_pins_presentation_code(self) -> None:
        """Version 0.14 decoupled question E from the fixture generator."""
        pinned = set(prf.FROZEN_SOURCE_RELATIVE_PATHS)
        assert "demo/build_pitcher_prototype_fixture.py" not in pinned
        assert not any(p.startswith("demo/") or p.startswith("dashboard/") for p in pinned)
        assert "RESOLVED in Version 0.14" in prf.ESTIMATOR_IMPLEMENTATION_CAVEAT

    def test_content_hash_is_key_order_independent(self) -> None:
        assert prf.content_hash({"a": 1, "b": 2}) == prf.content_hash({"b": 2, "a": 1})

    def test_content_hash_changes_when_content_changes(self) -> None:
        assert prf.content_hash({"a": 1}) != prf.content_hash({"a": 2})


class TestCommittedArtifactOnDisk:
    """If the real freeze artifact is present, it must still be valid. It is
    gitignored, so its absence is not a failure.
    """

    def test_the_real_freeze_still_validates(self) -> None:
        if not prf.FREEZE_PATH.is_file():
            pytest.skip("no freeze artifact written in this checkout")
        freeze = prf.read_freeze()
        assert freeze.spec_version == spec.SPEC_VERSION
        assert prf.validate_freeze(freeze)["valid"] is True
