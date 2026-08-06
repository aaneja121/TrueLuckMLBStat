"""Contact Luck v1.0, Task 33 category 10: outcome classification.

Verifies `evaluation.v1_final_report.classify_final_evaluation_outcome`
produces exactly the three predeclared outcomes, that a structural/
accounting failure always produces `final_evaluation_failed` regardless of
which OTHER inputs look fine, and that the classification is a pure
function of its inputs (no runtime/global state it could be swayed by).
"""

from __future__ import annotations

import dataclasses

import pytest
import v1_final_report as report_mod

_ALL_PASS_KWARGS = {
    "accounting_passed": True,
    "routing_passed": True,
    "reproducibility_passed": True,
    "schema_and_public_contract_passed": True,
    "frozen_artifacts_reproducible": True,
    "critical_input_or_schema_issue": False,
    "any_required_component_gate_severely_failed": False,
    "has_provisional_or_subgroup_limitations": False,
}


def _inputs(**overrides: bool) -> report_mod.OutcomeInputs:
    kwargs = {**_ALL_PASS_KWARGS, **overrides}
    return report_mod.OutcomeInputs(**kwargs)


def test_all_pass_yields_validated_as_frozen() -> None:
    outcome, reasons = report_mod.classify_final_evaluation_outcome(_inputs())
    assert outcome == report_mod.OUTCOME_VALIDATED_AS_FROZEN
    assert reasons


def test_limitations_with_no_failure_yields_validated_with_documented_limitations() -> None:
    outcome, reasons = report_mod.classify_final_evaluation_outcome(
        _inputs(has_provisional_or_subgroup_limitations=True)
    )
    assert outcome == report_mod.OUTCOME_VALIDATED_WITH_DOCUMENTED_LIMITATIONS
    assert reasons


@pytest.mark.parametrize(
    ("failure_kwarg", "expected_reason_substring"),
    [
        ({"accounting_passed": False}, "accounting"),
        ({"routing_passed": False}, "routing"),
        ({"reproducibility_passed": False}, "reproduce"),
        ({"frozen_artifacts_reproducible": False}, "manifest"),
        ({"critical_input_or_schema_issue": True}, "schema"),
        ({"any_required_component_gate_severely_failed": True}, "component"),
        ({"schema_and_public_contract_passed": False}, "ranking or qualification contract"),
    ],
)
def test_each_individual_failure_flag_yields_final_evaluation_failed(
    failure_kwarg: dict[str, bool], expected_reason_substring: str
) -> None:
    outcome, reasons = report_mod.classify_final_evaluation_outcome(_inputs(**failure_kwarg))
    assert outcome == report_mod.OUTCOME_FINAL_EVALUATION_FAILED
    assert any(expected_reason_substring in r for r in reasons), reasons


def test_structural_accounting_failure_dominates_even_with_no_limitations_flag() -> None:
    """A structural/accounting failure must produce `final_evaluation_
    failed` regardless of whether has_provisional_or_subgroup_limitations
    is also True or False -- failure always outranks "limitations."
    """
    for limitations_flag in (True, False):
        outcome, _reasons = report_mod.classify_final_evaluation_outcome(
            _inputs(
                accounting_passed=False,
                has_provisional_or_subgroup_limitations=limitations_flag,
            )
        )
        assert outcome == report_mod.OUTCOME_FINAL_EVALUATION_FAILED


def test_multiple_simultaneous_failures_are_all_reported() -> None:
    outcome, reasons = report_mod.classify_final_evaluation_outcome(
        _inputs(accounting_passed=False, routing_passed=False)
    )
    assert outcome == report_mod.OUTCOME_FINAL_EVALUATION_FAILED
    assert len(reasons) >= 2


def test_only_three_outcome_values_exist() -> None:
    assert set(report_mod.OUTCOME_VALUES) == {
        report_mod.OUTCOME_VALIDATED_AS_FROZEN,
        report_mod.OUTCOME_VALIDATED_WITH_DOCUMENTED_LIMITATIONS,
        report_mod.OUTCOME_FINAL_EVALUATION_FAILED,
    }
    for outcome, _reasons in (
        report_mod.classify_final_evaluation_outcome(_inputs()),
        report_mod.classify_final_evaluation_outcome(
            _inputs(has_provisional_or_subgroup_limitations=True)
        ),
        report_mod.classify_final_evaluation_outcome(_inputs(accounting_passed=False)),
    ):
        assert outcome in report_mod.OUTCOME_VALUES


# ---------------------------------------------------------------------------
# The rule is a pure function of its declared inputs, not runtime data
# ---------------------------------------------------------------------------


def test_outcome_inputs_is_frozen_and_cannot_be_mutated_after_construction() -> None:
    inputs = _inputs()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.accounting_passed = False  # type: ignore[misc]


def test_classification_depends_only_on_its_declared_inputs() -> None:
    """Two `OutcomeInputs` built from the exact same field values, at
    different times, with different notes, classify identically -- nothing
    about the function's own behavior can be swayed by anything other than
    the fields `OutcomeInputs` declares.
    """
    first = _inputs()
    second = report_mod.OutcomeInputs(**{**_ALL_PASS_KWARGS}, notes=["some diagnostic note"])
    outcome_first, _ = report_mod.classify_final_evaluation_outcome(first)
    outcome_second, _ = report_mod.classify_final_evaluation_outcome(second)
    assert outcome_first == outcome_second


def test_classify_function_has_no_side_effects_on_repeated_calls() -> None:
    inputs = _inputs(has_provisional_or_subgroup_limitations=True)
    first_call = report_mod.classify_final_evaluation_outcome(inputs)
    second_call = report_mod.classify_final_evaluation_outcome(inputs)
    assert first_call == second_call
