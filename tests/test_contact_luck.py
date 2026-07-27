from __future__ import annotations

import math

import pytest

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.scoring.aggregation import total_raw_contact_luck_runs
from mlb_luck_score.scoring.contact_luck import (
    ContactLuckValidationError,
    compute_expected_run_value,
    compute_raw_contact_luck_runs,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

EVEN_PROBS = {"out": 0.2, "single": 0.2, "double": 0.2, "triple": 0.2, "home_run": 0.2}


def test_expected_run_value_matches_manual_weighted_sum():
    probs = {"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}
    expected = sum(probs[c] * DEFAULT_RUN_VALUE_MAP[c] for c in CLASS_ORDER)
    assert compute_expected_run_value(probs) == pytest.approx(expected)


def test_expected_run_value_even_distribution_equals_mean_run_value():
    expected = sum(DEFAULT_RUN_VALUE_MAP.values()) / 5
    assert compute_expected_run_value(EVEN_PROBS) == pytest.approx(expected)


def test_favorable_raw_luck_is_positive():
    probs = {"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}
    luck = compute_raw_contact_luck_runs(probs, "home_run")
    assert luck > 0


def test_unfavorable_raw_luck_is_negative():
    probs = {"out": 0.05, "single": 0.1, "double": 0.15, "triple": 0.2, "home_run": 0.5}
    luck = compute_raw_contact_luck_runs(probs, "out")
    assert luck < 0


def test_raw_luck_units_are_runs_matching_run_value_table():
    # A model 100% certain of the observed outcome has raw luck of exactly 0
    # (no surprise); the units throughout are the fixed run-value table's
    # runs, not an arbitrary ordinal scale.
    certain = {"out": 0.0, "single": 0.0, "double": 0.0, "triple": 0.0, "home_run": 1.0}
    assert compute_raw_contact_luck_runs(certain, "home_run") == pytest.approx(0.0, abs=1e-9)


def test_invalid_probability_distribution_fails_clearly():
    bad_sum = {"out": 0.5, "single": 0.5, "double": 0.5, "triple": 0.0, "home_run": 0.0}
    with pytest.raises(ContactLuckValidationError, match="sum to ~1"):
        compute_raw_contact_luck_runs(bad_sum, "out")

    non_finite = dict(EVEN_PROBS, out=float("nan"))
    with pytest.raises(ContactLuckValidationError, match="non-finite"):
        compute_raw_contact_luck_runs(non_finite, "out")

    negative = dict(EVEN_PROBS, out=-0.1, single=0.3)
    with pytest.raises(ContactLuckValidationError, match="negative"):
        compute_raw_contact_luck_runs(negative, "out")


def test_missing_classes_fail_clearly():
    incomplete = {"out": 0.5, "single": 0.5}
    with pytest.raises(ContactLuckValidationError, match="missing class"):
        compute_raw_contact_luck_runs(incomplete, "out")


def test_unknown_observed_outcome_fails_clearly():
    with pytest.raises(ContactLuckValidationError, match="not a known class"):
        compute_raw_contact_luck_runs(EVEN_PROBS, "grand_slam_walk_off")


def test_class_order_does_not_affect_result():
    reversed_order = tuple(reversed(CLASS_ORDER))
    a = compute_raw_contact_luck_runs(EVEN_PROBS, "single", class_order=CLASS_ORDER)
    b = compute_raw_contact_luck_runs(EVEN_PROBS, "single", class_order=reversed_order)
    assert math.isclose(a, b)


def test_dictionary_insertion_order_does_not_affect_result():
    probs_a = {"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}
    probs_b = {k: probs_a[k] for k in reversed(list(probs_a))}
    assert compute_raw_contact_luck_runs(probs_a, "single") == pytest.approx(
        compute_raw_contact_luck_runs(probs_b, "single")
    )
    run_value_a = dict(DEFAULT_RUN_VALUE_MAP)
    run_value_b = {k: run_value_a[k] for k in reversed(list(run_value_a))}
    assert compute_raw_contact_luck_runs(
        probs_a, "single", run_value_map=run_value_a
    ) == pytest.approx(compute_raw_contact_luck_runs(probs_a, "single", run_value_map=run_value_b))


def test_raw_contact_luck_is_additive_across_plays():
    plays = [
        ({"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}, "home_run"),
        ({"out": 0.05, "single": 0.1, "double": 0.15, "triple": 0.2, "home_run": 0.5}, "out"),
        (EVEN_PROBS, "single"),
    ]
    individual_values = [compute_raw_contact_luck_runs(p, o) for p, o in plays]
    assert total_raw_contact_luck_runs(individual_values) == pytest.approx(sum(individual_values))


def test_custom_run_value_map_scales_result_proportionally():
    probs = {"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}
    default_luck = compute_raw_contact_luck_runs(probs, "home_run")
    doubled_map = {k: v * 2 for k, v in DEFAULT_RUN_VALUE_MAP.items()}
    doubled_luck = compute_raw_contact_luck_runs(probs, "home_run", run_value_map=doubled_map)
    assert doubled_luck == pytest.approx(default_luck * 2)
