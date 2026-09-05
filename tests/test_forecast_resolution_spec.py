"""Contact Forecast: the prespecified end-of-season 2026 resolution pass.

The resolution pass is a SECOND look at a season whose first look was already
reported and found inconclusive. That is precisely the setting in which a study
drifts: keep resolving windows, keep re-testing, stop when an interval clears
zero. These tests pin the structure that prevents it -- the primary cohort is
the only unopened data, the pooled cohort can never decide, there is no minimum
sample and no stopping rule, and the specification admits its own error rate is
above nominal.

Synthetic inputs where possible; the frozen artifacts are read read-only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from forecast.phase2 import resolution_spec as rs
from forecast.phase2.phase2_config import SUCCESS_CLASSIFICATION


def base_key_results(**overrides: Any) -> dict[str, Any]:
    record = {
        "spec_version": rs.RESOLUTION_SPEC_VERSION,
        "is_a_second_look": True,
        "primary_cohort": "incremental",
        "classification_applies_to": "the incremental cohort only",
        "predictions_regenerated": False,
        "end_of_season_outcomes_opened": False,
        "full_season_may_decide": False,
        "minimum_sample": None,
        "stopping_rule": None,
        "secondary_may_reclassify": False,
        "success_classification_keys": sorted(SUCCESS_CLASSIFICATION),
        "n_frozen_ledgers": 2,
        "gate_requirements": 7,
        "season_end_date": "2026-09-27",
        "season_end_verified": True,
        "cohorts_receiving_the_classification": ["incremental"],
        "n_amendments": 2,
        "re_frozen_before_any_incremental_outcome": True,
        "amendment_cohort_outcomes_opened": [],
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------
# No outcome may exist when this is frozen
# --------------------------------------------------------------------------


def test_the_specification_carries_no_outcome() -> None:
    rs.assert_no_resolution_outcome(rs.build_resolution_specification())


def test_a_leaked_numeric_outcome_is_refused() -> None:
    spec = rs.build_resolution_specification()
    spec["metrics"]["delta_mae"] = -0.0739
    with pytest.raises(rs.ResolutionSpecError, match="numeric"):
        rs.assert_no_resolution_outcome(spec)


def test_naming_an_outcome_in_prose_is_allowed() -> None:
    """The sign convention describes `delta_mae` without carrying one."""
    spec = rs.build_resolution_specification()
    assert "delta_mae" in spec["metrics"]["sign_convention"]
    rs.assert_no_resolution_outcome(spec)


def test_outcomes_already_opened_is_refused() -> None:
    spec = rs.build_resolution_specification()
    spec["season_discipline"]["end_of_season_outcomes_opened"] = True
    with pytest.raises(rs.ResolutionSpecError, match="before the second look"):
        rs.assert_no_resolution_outcome(spec)


# --------------------------------------------------------------------------
# Only unopened data may decide
# --------------------------------------------------------------------------


def test_the_primary_cohort_is_the_incremental_one() -> None:
    spec = rs.build_resolution_specification()
    assert spec["primary_cohort"] == "incremental"
    incremental = spec["cohorts"]["incremental"]
    assert incremental["may_decide_the_conclusion"] is True
    assert incremental["independent_of_the_first_look"] is True


def test_the_pooled_cohort_can_never_decide() -> None:
    full_season = rs.build_resolution_specification()["cohorts"]["full_season"]
    assert full_season["may_decide_the_conclusion"] is False
    assert full_season["independent_of_the_first_look"] is False
    assert "NOT INDEPENDENT" in full_season["mandatory_label"]


def test_a_pooled_primary_cohort_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="Only unopened data"):
        rs.assert_resolution_specification(base_key_results(primary_cohort="full_season"))


def test_letting_the_pooled_cohort_decide_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="may never do so"):
        rs.assert_resolution_specification(base_key_results(full_season_may_decide=True))


def test_hitters_who_never_complete_are_never_evaluated() -> None:
    cohort = rs.build_resolution_specification()["cohorts"]["never_completed"]
    assert cohort["four_way_classification_applied"] is False
    assert cohort["may_decide_the_conclusion"] is False
    assert "never imputed" in cohort["rule"]


# --------------------------------------------------------------------------
# The predictions are attached to, never regenerated
# --------------------------------------------------------------------------


def test_predictions_are_never_regenerated() -> None:
    spec = rs.build_resolution_specification()
    assert spec["predictions_regenerated"] is False
    assert "never recomputed" in spec["how_outcomes_are_attached"]
    assert "regenerating forecasts at an end-of-season cutoff" in spec["not_authorized"]


def test_regenerating_predictions_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="regenerating"):
        rs.assert_resolution_specification(base_key_results(predictions_regenerated=True))


def test_every_frozen_ledger_names_the_manifest_that_seals_it() -> None:
    for ledger in rs.FROZEN_PREDICTION_LEDGERS.values():
        assert ledger["pending_ledger"].endswith(".parquet")
        assert ledger["result_manifest"].endswith(".json")
        assert ledger["n_pending_at_first_look"] > 0


# --------------------------------------------------------------------------
# No stopping rule, no rescue
# --------------------------------------------------------------------------


def test_there_is_no_minimum_sample_and_no_stopping_rule() -> None:
    spec = rs.build_resolution_specification()
    assert spec["minimum_sample"] is None
    assert spec["stopping_rule"] is None
    assert spec["exploratory_power_estimate_is_a_rule"] is False


def test_a_stopping_rule_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="stopping rule"):
        rs.assert_resolution_specification(base_key_results(stopping_rule="stop at 95%"))
    with pytest.raises(rs.ResolutionSpecError, match="minimum sample"):
        rs.assert_resolution_specification(base_key_results(minimum_sample=82))


def test_an_underpowered_result_is_reported_not_rescued() -> None:
    rule = rs.build_resolution_specification()["underpowered_rule"]
    assert "never grounds for falling back to the full-season cohort" in rule


def test_secondary_metrics_may_not_reclassify() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="secondary metrics"):
        rs.assert_resolution_specification(base_key_results(secondary_may_reclassify=True))


def test_the_four_way_rule_is_the_prespecified_one() -> None:
    spec = rs.build_resolution_specification()
    assert spec["success_classification"] == SUCCESS_CLASSIFICATION
    with pytest.raises(rs.ResolutionSpecError, match="four-way classification"):
        rs.assert_resolution_specification(
            base_key_results(success_classification_keys=["only_one"])
        )


# --------------------------------------------------------------------------
# The multiplicity disclosure
# --------------------------------------------------------------------------


def test_the_pass_admits_it_is_a_second_unadjusted_look() -> None:
    disclosure = rs.MULTIPLICITY_DISCLOSURE
    assert "SECOND unadjusted look" in disclosure["headline"]
    assert disclosure["first_look"]["alpha_spent"].startswith("none reserved")
    assert "above 5%" in disclosure["consequence"]
    assert "does NOT claim to restore" in disclosure["what_this_specification_does_about_it"]


def test_confirmation_language_is_forbidden() -> None:
    forbidden = rs.MULTIPLICITY_DISCLOSURE["what_may_never_be_claimed"]
    assert "confirms the first look" in forbidden
    assert "no third pass" in rs.MULTIPLICITY_DISCLOSURE["no_further_looks"].lower()


def test_dropping_the_second_look_admission_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="second look"):
        rs.assert_resolution_specification(base_key_results(is_a_second_look=False))


def test_the_conclusion_refuses_to_be_called_a_result() -> None:
    conclusion = rs.RESOLUTION_CONCLUSION
    assert "no end-of-season 2026 outcome has been opened" in conclusion["verdict"].lower()
    assert any(
        "confirmation of the first look" in c for c in conclusion["must_not_be_described_as"]
    )
    assert conclusion["preserved_negative_results"]


def test_the_limitations_name_the_selection_effect() -> None:
    ids = {item["id"] for item in rs.RESOLUTION_LIMITATIONS}
    assert "second_unadjusted_look" in ids
    assert "decision_to_look_again_followed_the_first_result" in ids
    assert "incremental_cohort_is_not_exchangeable" in ids


# --------------------------------------------------------------------------
# Season discipline and the gate
# --------------------------------------------------------------------------


def test_2025_stays_sealed_and_2026_is_never_fitted_on() -> None:
    discipline = rs.build_resolution_specification()["season_discipline"]
    assert discipline["sealed"] == [2025]
    assert discipline["fitted_on_2026"] is False
    assert 2025 not in discipline["training_seasons"]
    assert 2026 not in discipline["training_seasons"]


def test_the_gate_requires_a_verified_season_end_date() -> None:
    gate = rs.build_resolution_specification()["gate"]
    assert any("verified" in item and "final date" in item for item in gate)
    assert any("authorized the second look" in item for item in gate)


def test_the_first_look_results_are_untouchable() -> None:
    spec = rs.build_resolution_specification()
    assert spec["first_look_results_modified"] is False
    assert spec["deployed"] is False
    assert any("first-look results" in item for item in spec["frozen_and_untouchable"])


# --------------------------------------------------------------------------
# The written specification
# --------------------------------------------------------------------------


@pytest.fixture
def frozen_spec() -> dict[str, Any]:
    path = rs.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json"
    if not path.exists():
        pytest.skip("resolution specification has not been generated in this environment")
    return json.loads(path.read_text())


def test_the_written_specification_opens_no_outcome(frozen_spec: dict) -> None:
    assert frozen_spec["season_discipline"]["end_of_season_outcomes_opened"] is False
    rs.assert_no_resolution_outcome(frozen_spec)


def test_the_written_specification_verifies_against_its_freeze() -> None:
    from forecast.stage_freeze import verify_stage

    if not (rs.RESOLUTION_OUTPUTS_DIR / "resolution_spec_freeze_manifest.json").exists():
        pytest.skip("resolution specification has not been frozen in this environment")
    result = verify_stage(rs.resolution_freeze_spec(), outputs_dir=rs.RESOLUTION_OUTPUTS_DIR)
    assert result["matches_freeze"] is True, result["drift"]


def test_the_report_leads_with_the_second_look_warning() -> None:
    path = rs.RESOLUTION_OUTPUTS_DIR / "resolution_specification_report.md"
    if not path.exists():
        pytest.skip("resolution specification report has not been generated")
    report = path.read_text()
    assert "No end-of-season outcome has been opened. Nothing below is a result." in report
    assert "second look" in report.lower()
    assert "may never be claimed" in report.lower()


# --------------------------------------------------------------------------
# The externally verified season end date
# --------------------------------------------------------------------------


def test_the_season_end_date_is_verified_and_loaded_not_restated() -> None:
    from prospective.prospective_config import (
        PROSPECTIVE_2026_SEASON_END_DATE,
        PROSPECTIVE_2026_SEASON_END_SOURCE,
        PROSPECTIVE_2026_SEASON_END_VERIFIED,
    )

    verification = rs.build_resolution_specification()["season_end_verification"]
    assert verification["regular_season_end_date"] == PROSPECTIVE_2026_SEASON_END_DATE.isoformat()
    assert verification["regular_season_end_date"] == "2026-09-27"
    assert verification["verified"] is PROSPECTIVE_2026_SEASON_END_VERIFIED is True
    assert verification["source"] == PROSPECTIVE_2026_SEASON_END_SOURCE
    assert "MLB official 2026 schedule announcement" in verification["source"]


def test_the_end_date_records_that_tooling_did_not_fetch_it() -> None:
    """The same honesty the verified START date carries."""
    verification = rs.build_resolution_specification()["season_end_verification"]
    assert verification["independently_fetched_by_tooling"] is False
    assert "maintainer-provided citation" in verification["source"]
    assert verification["verified_at"]


def test_the_end_date_uses_the_same_mechanism_as_the_start_date() -> None:
    """Four parallel constants: date, source, verified-at, verified flag."""
    from prospective import prospective_config as pc

    for suffix in ("_DATE", "_SOURCE", "_VERIFIED_AT", "_VERIFIED"):
        assert hasattr(pc, f"PROSPECTIVE_2026_SEASON_START{suffix}")
        assert hasattr(pc, f"PROSPECTIVE_2026_SEASON_END{suffix}")
    assert pc.PROSPECTIVE_2026_SEASON_END_DATE > pc.PROSPECTIVE_2026_SEASON_START_DATE


def test_an_unverified_season_end_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="not marked verified"):
        rs.assert_resolution_specification(base_key_results(season_end_verified=False))


def test_a_drifted_season_end_date_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="no longer"):
        rs.assert_resolution_specification(base_key_results(season_end_date="2026-10-05"))


def test_the_gate_requires_the_snapshot_to_reach_the_verified_end_date() -> None:
    gate = rs.build_resolution_specification()["gate"]
    assert any("data-through date is on or after" in item for item in gate)


# --------------------------------------------------------------------------
# The amendment, and when it happened
# --------------------------------------------------------------------------


def test_the_amendment_records_that_no_outcome_was_open() -> None:
    amendment = rs.RESOLUTION_AMENDMENTS[0]
    assert amendment["re_freeze_occurred_before_any_incremental_cohort_outcome_was_opened"] is True
    assert not any(amendment["cohort_outcomes_opened_at_amendment_time"].values())
    assert amendment["content_otherwise_unchanged"] is True
    assert amendment["first_look_results_modified"] is False
    assert amendment["evaluation_run_early"] is False


def test_the_amendment_chains_to_the_manifest_it_replaced() -> None:
    amendment = rs.RESOLUTION_AMENDMENTS[0]
    assert len(amendment["previous_freeze_manifest_sha256"]) == 64
    assert "GATE CONDITION" in amendment["why_this_was_not_a_specification_change"]


def test_amending_after_an_outcome_was_opened_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="before any incremental"):
        rs.assert_resolution_specification(
            base_key_results(re_frozen_before_any_incremental_outcome=False)
        )
    with pytest.raises(rs.ResolutionSpecError, match="already open"):
        rs.assert_resolution_specification(
            base_key_results(amendment_cohort_outcomes_opened=["incremental"])
        )


def test_the_next_evaluation_is_not_run_early() -> None:
    amendment = rs.RESOLUTION_AMENDMENTS[0]
    assert "after 2026-09-27" in amendment["next_evaluation"]
    assert "not run early" in amendment["next_evaluation"]


def test_no_resolution_outcome_artifact_exists_yet() -> None:
    """The amendment must not have opened anything."""
    if not rs.RESOLUTION_OUTPUTS_DIR.exists():
        pytest.skip("resolution namespace has not been generated in this environment")
    assert not list(rs.RESOLUTION_OUTPUTS_DIR.glob("*.parquet"))


# --------------------------------------------------------------------------
# Amendment 2: the classification flags may never disagree again
# --------------------------------------------------------------------------


def test_only_the_primary_cohort_is_flagged_for_classification() -> None:
    spec = rs.build_resolution_specification()
    flagged = sorted(
        name for name, c in spec["cohorts"].items() if c["four_way_classification_applied"]
    )
    assert flagged == [spec["primary_cohort"]] == ["incremental"]
    assert spec["classification_applies_to"] == "the incremental cohort only"


def test_the_pooled_cohort_gets_an_interval_instead_of_a_class() -> None:
    full_season = rs.build_resolution_specification()["cohorts"]["full_season"]
    assert full_season["four_way_classification_applied"] is False
    assert "no classification label" in full_season["receives_instead"]


def test_a_specification_that_classifies_a_non_deciding_cohort_is_refused() -> None:
    with pytest.raises(rs.ResolutionSpecError, match="may never disagree"):
        rs.assert_resolution_specification(
            base_key_results(cohorts_receiving_the_classification=["full_season", "incremental"])
        )


def test_amendment_two_records_that_no_outcome_was_open() -> None:
    amendment = rs.RESOLUTION_AMENDMENTS[1]
    assert amendment["amendment"] == 2
    assert amendment["re_freeze_occurred_before_any_incremental_cohort_outcome_was_opened"] is True
    assert not any(amendment["cohort_outcomes_opened_at_amendment_time"].values())
    assert amendment["resolved_toward"].startswith("the stricter reading")
    assert amendment["content_otherwise_unchanged"] is True


def test_amendment_two_chains_to_the_manifest_it_replaced() -> None:
    assert (
        rs.RESOLUTION_AMENDMENTS[1]["previous_freeze_manifest_sha256"]
        == "52da1f6f33556e8cb9f05acd430aa12a8e1662118b4311316a41c5629aa0e91e"
    )
