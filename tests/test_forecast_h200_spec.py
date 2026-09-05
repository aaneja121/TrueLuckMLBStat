"""Contact Forecast: the frozen Phase 2 H=200 specification.

The specification exists because the exploratory 100 -> 200 result came from a
REFIT, not from the frozen H=100 model. These tests pin that answer, pin the
guards that keep 2026 outcomes closed until the freeze is in place, and pin
the fence around the exploratory power estimate so it can never become a
decision rule.

Synthetic inputs where possible; the frozen artifacts are read read-only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from forecast.phase2 import h200_spec as h2
from forecast.phase2.phase2_config import SUCCESS_CLASSIFICATION


def base_key_results(**overrides: Any) -> dict[str, Any]:
    record = {
        "upstream_copies_match_live_manifests": {"r1_freeze_manifest.json": True},
        "provenance_answer": "REFIT",
        "case": 2,
        "prospective_outcomes_opened": False,
        "power_estimate_is_a_threshold": False,
        "success_classification_keys": sorted(SUCCESS_CLASSIFICATION),
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------
# The provenance answer
# --------------------------------------------------------------------------


def test_the_answer_is_refit_case_two() -> None:
    answer = h2.H200_PROVENANCE_ANSWER
    assert answer["answer"] == "REFIT"
    assert answer["case"] == 2
    assert answer["frozen_h100_coefficients_reused"] is False
    assert answer["frozen_h100_preprocessing_reused"] is False
    assert answer["frozen_h100_intercept_reused"] is False
    assert answer["inherited_from_the_frozen_ridge"] == ["feature list", "alpha value"]


def test_the_answer_records_the_evidence_for_the_refit() -> None:
    evidence = h2.H200_PROVENANCE_ANSWER["evidence"]
    assert evidence["coefficients_identical"] is False
    assert evidence["scaler_means_identical"] is False
    assert evidence["intercept_h100"] != evidence["intercept_h200"]
    assert evidence["training_rows_h200"] < evidence["training_rows_h100"]


def test_the_forbidden_descriptions_are_recorded() -> None:
    forbidden = h2.H200_PROVENANCE_ANSWER["must_not_be_called"]
    assert any("frozen Contact Forecast measured at 200" in item for item in forbidden)
    assert any("prespecified in R1" in item for item in forbidden)
    assert h2.H200_PROVENANCE_ANSWER["frozen_h100_result_unchanged"] is True


def test_a_changed_provenance_answer_refuses_the_freeze() -> None:
    with pytest.raises(h2.H200SpecError, match="no longer REFIT"):
        h2.assert_h200_specification(base_key_results(provenance_answer="FROZEN_REUSE"))
    with pytest.raises(h2.H200SpecError, match="no longer REFIT"):
        h2.assert_h200_specification(base_key_results(case=1))


# --------------------------------------------------------------------------
# 2026 outcomes stay closed
# --------------------------------------------------------------------------


def test_opening_2026_outcomes_refuses_the_freeze() -> None:
    with pytest.raises(h2.H200SpecError, match="must be frozen"):
        h2.assert_h200_specification(base_key_results(prospective_outcomes_opened=True))


def test_the_prospective_audit_carries_no_target() -> None:
    h2.assert_no_prospective_target({"n_reaching_100_resolved_bbe": 380, "nested": {"a": 1}})
    with pytest.raises(h2.H200SpecError, match="must not be opened"):
        h2.assert_no_prospective_target({"target_realized_rv_per_100": 4.2})
    with pytest.raises(h2.H200SpecError, match="must not be opened"):
        h2.assert_no_prospective_target({"deep": [{"target_deserved_rv_per_100": 1.0}]})


def test_a_stale_upstream_copy_refuses_the_freeze() -> None:
    with pytest.raises(h2.H200SpecError, match="no longer match the live"):
        h2.assert_h200_specification(
            base_key_results(
                upstream_copies_match_live_manifests={"ridge_freeze_manifest.json": False}
            )
        )


# --------------------------------------------------------------------------
# The power estimate is fenced off
# --------------------------------------------------------------------------


def test_the_power_estimate_is_not_a_threshold_or_stopping_rule() -> None:
    estimate = h2.EXPLORATORY_POWER_ESTIMATE
    assert estimate["is_a_success_threshold"] is False
    assert estimate["is_a_stopping_rule"] is False
    assert estimate["status"].startswith("EXPLORATORY")
    assert "never" in estimate["note"]


def test_turning_the_power_estimate_into_a_threshold_refuses_the_freeze() -> None:
    with pytest.raises(h2.H200SpecError, match="success threshold"):
        h2.assert_h200_specification(base_key_results(power_estimate_is_a_threshold=True))


def test_the_success_classification_never_references_the_power_estimate() -> None:
    for entry in SUCCESS_CLASSIFICATION.values():
        assert "82" not in entry["condition"]
        assert "window" not in entry["condition"]


def test_the_classification_is_the_prespecified_four_way_rule() -> None:
    assert set(SUCCESS_CLASSIFICATION) == {
        "established_incremental_success",
        "promising_but_inconclusive",
        "no_evidence_of_incremental_improvement",
        "evidence_against_incremental_value",
    }
    with pytest.raises(h2.H200SpecError, match="four-way rule"):
        h2.assert_h200_specification(base_key_results(success_classification_keys=["only_one"]))


# --------------------------------------------------------------------------
# Season discipline
# --------------------------------------------------------------------------


def test_the_scope_refuses_sealed_seasons() -> None:
    from forecast.phase2.phase2_config import Phase2SeasonError

    with pytest.raises(Phase2SeasonError, match="sealed"):
        h2.build_development_windows(seasons=(2022, 2025))


def test_the_frozen_h100_result_is_untouched() -> None:
    """Freezing H=200 must not disturb the sealed H=100 stages."""
    from forecast.freeze_hgb import HGB_FREEZE_SPEC
    from forecast.freeze_ridge import RIDGE_FREEZE_SPEC
    from forecast.stage_freeze import verify_stage

    for spec in (RIDGE_FREEZE_SPEC, HGB_FREEZE_SPEC):
        result = verify_stage(spec)
        assert result["matches_freeze"] is True, result["drift"]


# --------------------------------------------------------------------------
# The written specification
# --------------------------------------------------------------------------


@pytest.fixture
def frozen_spec() -> dict[str, Any]:
    path = h2.H200_OUTPUTS_DIR / "h200_specification.json"
    if not path.exists():
        pytest.skip("H=200 specification has not been generated in this environment")
    return json.loads(path.read_text())


def test_the_specification_contains_every_required_element(frozen_spec: dict) -> None:
    assert frozen_spec["design"]["cutoff"].startswith("first 100")
    assert "next 200" in frozen_spec["design"]["target"]
    model = frozen_spec["forecast_model_being_evaluated"]
    assert model["distinct_from_the_frozen_h100_model"] is True
    assert model["alpha"] > 0
    assert model["coefficients"]
    assert frozen_spec["benchmarks"]["primary"] == "shrunk_deserved_persistence"
    assert set(frozen_spec["benchmarks"]["all"]) == {
        "league_mean",
        "shrunk_realized_persistence",
        "shrunk_deserved_persistence",
    }
    assert frozen_spec["metrics"]["primary"] == "MAE"
    assert frozen_spec["metrics"]["secondary_may_reclassify"] is False
    for required in (
        "survivorship audit",
        "stability and outlier sensitivity",
        "distribution-shift diagnostics",
    ):
        assert required in frozen_spec["required_analyses"]


def test_the_specification_selected_alpha_without_2024_or_2026(frozen_spec: dict) -> None:
    selection = frozen_spec["alpha_selection"]
    assert selection["fit_seasons"] == [2022]
    assert selection["validation_season"] == 2023
    assert selection["evaluation_season_used_in_selection"] is False
    assert selection["prospective_season_used_in_selection"] is False


def test_the_specification_states_the_generalization_limit(frozen_spec: dict) -> None:
    assert (
        "conditional on the hitter subsequently accumulating 200"
        in (frozen_spec["design"]["conditional_interpretation"])
    )
    assert frozen_spec["season_discipline"]["prospective_outcomes_opened"] is False
    assert frozen_spec["season_discipline"]["sealed"] == [2025]


def test_the_report_leads_with_the_refit_answer() -> None:
    path = h2.H200_OUTPUTS_DIR / "h200_specification_report.md"
    if not path.exists():
        pytest.skip("H=200 report has not been generated in this environment")
    report = path.read_text()
    assert "**Answer: REFIT (case 2).**" in report
    assert "was **refit** against next-200-BBE targets" in report
    assert "NOT the frozen" in report
    assert "2026 H=200 outcomes opened: False" in report


def test_the_survivorship_audit_reports_completion_and_differences() -> None:
    path = h2.H200_OUTPUTS_DIR / "h200_development_survivorship.json"
    if not path.exists():
        pytest.skip("H=200 survivorship has not been generated in this environment")
    audit = json.loads(path.read_text())["pooled"]
    assert (
        audit["n_reaching_100_resolved_bbe"]
        > audit["n_subsequently_reaching_300_total_resolved_bbe"]
    )
    assert 0.0 < audit["completion_rate"] < 1.0
    assert audit["largest_standardized_differences"]
    assert "generalizes ONLY to hitters" in audit["generalization_note"]
