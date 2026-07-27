"""Version 0.2 additive raw contact luck, in fixed context-neutral run values.

    expected_run_value    = sum(probability[outcome] * run_value[outcome])
    actual_run_value       = run_value[observed_outcome]
    raw_contact_luck_runs  = actual_run_value - expected_run_value

Uses the fixed `mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`
(FanGraphs Guts-derived, averaged over 2021-2024) instead of the Version 0.1
ordinal placeholder in `mlb_luck_score.scoring.raw_luck` (out=0/single=1/
double=2/triple=3/home_run=4). `raw_luck.py` is now LEGACY -- see its module
docstring -- and is kept only for backward compatibility, not as the
default. Units here are RUNS, not an arbitrary ordinal scale.

These run values are GENERALIZED and CONTEXT-NEUTRAL: they represent each
outcome's average run value across a league-average plate appearance in a
given season, independent of the actual base/out state, score, win
probability, or RE24 context of any specific play. A bases-loaded triple and
a bases-empty triple get the exact same fixed run value here. This is a
deliberate Version 0.2 scope choice, not an oversight -- do NOT use actual
base/out state, runs scored, win probability, or RE24 to compute this value.
Context-aware run values are a documented future extension (see README.md
"Future work"), not implemented in Version 0.2.

Do not silently redefine this formula or the default run-value map
elsewhere -- see CLAUDE.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP


class ContactLuckValidationError(ValueError):
    """Raised when inputs to the Version 0.2 raw-contact-luck calculation are invalid."""


def _validate_run_value_map(run_value_map: Mapping[str, float], class_order: Sequence[str]) -> None:
    missing = [c for c in class_order if c not in run_value_map]
    if missing:
        raise ContactLuckValidationError(f"run_value_map is missing class(es): {missing}")


def _validate_probabilities(
    probabilities: Mapping[str, float], class_order: Sequence[str], *, atol: float = 1e-3
) -> None:
    missing = [c for c in class_order if c not in probabilities]
    if missing:
        raise ContactLuckValidationError(
            f"predicted_probability is missing class(es): {missing}. "
            "Every class in class_order must have a probability -- raw contact luck "
            "cannot be computed from a partial distribution."
        )

    values = [probabilities[c] for c in class_order]
    if not all(math.isfinite(v) for v in values):
        raise ContactLuckValidationError(
            f"predicted_probability contains non-finite value(s): {values}"
        )
    if any(v < 0 for v in values):
        raise ContactLuckValidationError(
            f"predicted_probability contains negative value(s): {values}"
        )

    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=atol):
        raise ContactLuckValidationError(
            f"predicted_probability must sum to ~1.0 (got {total:.6f}); "
            "this looks like an invalid or partial probability distribution"
        )


def compute_expected_run_value(
    probabilities: Mapping[str, float],
    *,
    run_value_map: Mapping[str, float] = DEFAULT_RUN_VALUE_MAP,
    class_order: Sequence[str] = CLASS_ORDER,
) -> float:
    """Expected run value: sum(p(outcome) * run_value(outcome)).

    Validates that `probabilities` covers every class in `class_order`, is
    finite, non-negative, and sums to ~1. Result does not depend on the
    iteration order of `class_order` (summation is commutative) or on
    dictionary key order in `probabilities`/`run_value_map`.
    """
    _validate_run_value_map(run_value_map, class_order)
    _validate_probabilities(probabilities, class_order)
    return sum(probabilities[c] * run_value_map[c] for c in class_order)


def compute_raw_contact_luck_runs(
    probabilities: Mapping[str, float],
    observed_outcome: str,
    *,
    run_value_map: Mapping[str, float] = DEFAULT_RUN_VALUE_MAP,
    class_order: Sequence[str] = CLASS_ORDER,
) -> float:
    """Version 0.2 additive raw contact luck for one play, in runs.

    Args:
        probabilities: Predicted probability for every class in
            `class_order`. Must be finite, non-negative, and sum to ~1.
        observed_outcome: The actual outcome class observed on the play.
            Must be a key in `run_value_map`.
        run_value_map: Fixed run value assigned to each outcome class.
            Defaults to `mlb_luck_score.scoring.run_values.
            DEFAULT_RUN_VALUE_MAP` (FanGraphs Guts-derived, 2021-2024
            average).
        class_order: The class ordering used to validate `probabilities`.
            Result is independent of this ordering.

    Returns:
        `actual_run_value - expected_run_value`, in runs. Positive means
        the batter-runner did better than the model expected (favorable
        luck); negative means worse (unfavorable luck); near zero means the
        outcome matched expectation. This value is additive: summing it
        across plays yields a meaningful total (see
        `mlb_luck_score.scoring.aggregation`).

    Raises:
        ContactLuckValidationError: If `observed_outcome` is unknown, or if
            `probabilities` is missing a class, contains a non-finite or
            negative value, or does not sum to ~1.
    """
    _validate_run_value_map(run_value_map, class_order)
    if observed_outcome not in run_value_map:
        raise ContactLuckValidationError(
            f"observed_outcome '{observed_outcome}' is not a known class in {list(run_value_map)}"
        )

    expected_run_value = compute_expected_run_value(
        probabilities, run_value_map=run_value_map, class_order=class_order
    )
    actual_run_value = run_value_map[observed_outcome]
    return actual_run_value - expected_run_value
