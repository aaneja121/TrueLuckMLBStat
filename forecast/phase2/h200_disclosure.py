"""Contact Forecast H=200: the 2026 exposure disclosure, recorded before outcomes.

Written and hashed BEFORE the H=200 evaluation opens a single 2026 outcome, so
that the record of what the study had already seen cannot be assembled after
the answer is known.

This module is deliberately NOT part of the frozen `h200_spec` source set.
Editing a frozen source module would drift the H=200 specification freeze, and
a disclosure is not a specification change: it adds no element, removes none,
and modifies none. It states what was inspected on the cutoff side of 2026
while the specification was being frozen, and asserts -- in a form a reader
can check against the artifacts -- that none of it touched an outcome.
"""

from __future__ import annotations

from typing import Any

#: The disclosure, as recorded before any H=200 outcome was opened.
H200_2026_EXPOSURE_DISCLOSURE: dict[str, Any] = {
    "headline": ("2026 should be described as OUTCOME-HELD-OUT, not completely unseen."),
    "what_was_inspected_before_outcomes_were_opened": (
        "number of hitters reaching the required 300 resolved BBE",
        "completion rate",
        "cutoff-side predictor/features for the survivorship audit",
    ),
    "what_was_not_inspected": (
        "2026 future realized-RV target",
        "2026 prediction error",
        "2026 benchmark comparison",
        "2026 model coefficient",
        "2026 performance metric",
    ),
    "statement": (
        "Prior to opening H=200 performance outcomes, the study inspected only the "
        "number of hitters reaching the required 300 resolved BBE, the completion "
        "rate, and cutoff-side predictor/features for the survivorship audit. No "
        "2026 future realized-RV target, prediction error, benchmark comparison, "
        "model coefficient, or performance metric was inspected."
    ),
    "readiness_observations_did_not_modify": (
        "Model D architecture",
        "feature list",
        "alpha",
        "preprocessing procedure",
        "target definition",
        "benchmarks",
        "primary metric",
        "CI procedure",
        "success classification",
    ),
    "recorded_before_any_2026_h200_outcome_was_opened": True,
    "source_artifact": "h200_prospective_cutoff_audit.json",
    "verification": (
        "The cutoff audit is the only 2026 artifact produced before this "
        "evaluation, `h200_spec.assert_no_prospective_target` proves it carries no "
        "target, and every frozen element listed above is read out of the verified "
        "h200_spec freeze manifest rather than restated -- so a readiness "
        "observation had nothing it could have changed."
    ),
    "must_be_preserved_in_the_final_report": True,
}

#: The frozen elements the disclosure asserts were untouched, in the form the
#: evaluation re-checks them against the sealed manifest.
DISCLOSURE_PROTECTED_ELEMENTS: tuple[str, ...] = (
    "Model D architecture",
    "feature list",
    "alpha",
    "preprocessing procedure",
    "target definition",
    "benchmarks",
    "primary metric",
    "CI procedure",
    "success classification",
)


def render_disclosure_markdown() -> str:
    """The disclosure as report prose, preserved verbatim in the final report."""
    disclosure = H200_2026_EXPOSURE_DISCLOSURE
    lines = [
        "## Disclosure: 2026 is outcome-held-out, not completely unseen",
        "",
        "Recorded and hashed BEFORE any H=200 outcome was opened.",
        "",
        disclosure["headline"],
        "",
        "Prior to opening H=200 performance outcomes, the study inspected only:",
        "",
    ]
    lines += [f"- {item}" for item in disclosure["what_was_inspected_before_outcomes_were_opened"]]
    lines += [
        "",
        "No 2026 future realized-RV target, prediction error, benchmark comparison, "
        "model coefficient, or performance metric was inspected.",
        "",
        "Those readiness/survivorship observations did not modify:",
        "",
    ]
    lines += [f"- {item}" for item in disclosure["readiness_observations_did_not_modify"]]
    lines += ["", disclosure["verification"], ""]
    return "\n".join(lines)
