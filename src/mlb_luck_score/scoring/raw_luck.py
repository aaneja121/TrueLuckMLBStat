"""LEGACY (Version 0.1): ordinal-scale additive raw-luck calculation.

> **Superseded by Version 0.2.** The default Contact Luck raw-luck
> calculation is now `mlb_luck_score.scoring.contact_luck.
> compute_raw_contact_luck_runs`, which uses fixed, empirically-grounded
> run values (FanGraphs Guts-derived, see `mlb_luck_score.scoring.
> run_values`) instead of the arbitrary 0-4 ordinal scale below. This
> module is kept ONLY for backward compatibility and explicit
> Version-0.1-vs-0.2 comparisons (see notebook `04_luck_score_demo.ipynb`'s
> "Legacy Version 0.1 comparison" section) -- do not use it as the default
> for new work.

    expected_value = sum(predicted_probability[outcome] * value[outcome])
    actual_value    = value[observed_outcome]
    raw_luck        = actual_value - expected_value

This treats outcomes as an ordinal scale (see
`mlb_luck_score.config.DEFAULT_VALUE_MAP`) and measures how much better or
worse the actual result was than the model's expectation, in those ordinal
units. It is deliberately simple:

* It ignores base/out state, park, and win-expectancy context.
* The ordinal value mapping (0/1/2/3/4) is a provisional research choice,
  not a validated run-value model -- a double is not "twice as good" as a
  single in run-scoring terms. Version 0.2 fixes this with real run values.
* Raw luck is additive by construction (differences of expectations are
  additive), but the public-facing -100..+100 score derived from it is NOT
  additive -- see `mlb_luck_score.scoring.public_score` (also legacy; see
  `mlb_luck_score.scoring.empirical_score` for the Version 0.2 replacement).

Do not silently redefine this formula or the default value map elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from mlb_luck_score.config import CLASS_ORDER, DEFAULT_VALUE_MAP


class RawLuckValidationError(ValueError):
    """Raised when inputs to the raw-luck calculation are invalid."""


def _validate_value_map(value_map: Mapping[str, float], class_order: Sequence[str]) -> None:
    missing = [c for c in class_order if c not in value_map]
    if missing:
        raise RawLuckValidationError(f"value_map is missing class(es): {missing}")


def _validate_probabilities(
    probabilities: Mapping[str, float], class_order: Sequence[str], *, atol: float = 1e-3
) -> None:
    missing = [c for c in class_order if c not in probabilities]
    if missing:
        raise RawLuckValidationError(
            f"predicted_probability is missing class(es): {missing}. "
            "Every class in class_order must have a probability -- raw luck "
            "cannot be computed from a partial distribution."
        )

    values = [probabilities[c] for c in class_order]
    if not all(math.isfinite(v) for v in values):
        raise RawLuckValidationError(
            f"predicted_probability contains non-finite value(s): {values}"
        )
    if any(v < 0 for v in values):
        raise RawLuckValidationError(f"predicted_probability contains negative value(s): {values}")

    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=atol):
        raise RawLuckValidationError(
            f"predicted_probability must sum to ~1.0 (got {total:.6f}); "
            "this looks like an invalid or partial probability distribution"
        )


def compute_expected_value(
    probabilities: Mapping[str, float],
    *,
    value_map: Mapping[str, float] = DEFAULT_VALUE_MAP,
    class_order: Sequence[str] = CLASS_ORDER,
) -> float:
    """Expected ordinal value: sum(p(outcome) * value(outcome)).

    Validates that `probabilities` covers every class in `class_order`, is
    finite, non-negative, and sums to ~1. Result does not depend on the
    iteration order of `class_order` (summation is commutative).
    """
    _validate_value_map(value_map, class_order)
    _validate_probabilities(probabilities, class_order)
    return sum(probabilities[c] * value_map[c] for c in class_order)


def compute_raw_luck(
    probabilities: Mapping[str, float],
    observed_outcome: str,
    *,
    value_map: Mapping[str, float] = DEFAULT_VALUE_MAP,
    class_order: Sequence[str] = CLASS_ORDER,
) -> float:
    """Preliminary additive raw luck for one play: actual - expected ordinal value.

    Args:
        probabilities: Predicted probability for every class in
            `class_order`. Must be finite, non-negative, and sum to ~1.
        observed_outcome: The actual outcome class observed on the play.
            Must be a key in `value_map`.
        value_map: Ordinal value assigned to each outcome class. Defaults to
            the Version 0.1 provisional mapping in
            `mlb_luck_score.config.DEFAULT_VALUE_MAP`.
        class_order: The class ordering used to validate `probabilities`.
            Result is independent of this ordering.

    Returns:
        `actual_value - expected_value`. Positive means the batter-runner
        did better than the model expected (favorable luck); negative means
        worse (unfavorable luck); near zero means the outcome matched
        expectation.

    Raises:
        RawLuckValidationError: If `observed_outcome` is unknown, or if
            `probabilities` is missing a class, contains a non-finite or
            negative value, or does not sum to ~1.
    """
    _validate_value_map(value_map, class_order)
    if observed_outcome not in value_map:
        raise RawLuckValidationError(
            f"observed_outcome '{observed_outcome}' is not a known class in {list(value_map)}"
        )

    expected_value = compute_expected_value(
        probabilities, value_map=value_map, class_order=class_order
    )
    actual_value = value_map[observed_outcome]
    return actual_value - expected_value
