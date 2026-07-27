from __future__ import annotations

import math

import pandas as pd
import pytest

from mlb_luck_score.config import CLASS_ORDER, DEFAULT_VALUE_MAP
from mlb_luck_score.scoring.confidence import (
    HIGH_LABEL,
    LOW_LABEL,
    MEDIUM_LABEL,
    compute_confidence,
)
from mlb_luck_score.scoring.public_score import (
    PublicScoreValidationError,
    raw_luck_to_public_score,
)
from mlb_luck_score.scoring.raw_luck import RawLuckValidationError, compute_raw_luck

EVEN_PROBS = {"out": 0.2, "single": 0.2, "double": 0.2, "triple": 0.2, "home_run": 0.2}


def test_better_than_expected_outcome_is_positive_luck():
    probs = {"out": 0.7, "single": 0.15, "double": 0.1, "triple": 0.03, "home_run": 0.02}
    luck = compute_raw_luck(probs, "home_run")
    assert luck > 0


def test_worse_than_expected_outcome_is_negative_luck():
    probs = {"out": 0.05, "single": 0.1, "double": 0.15, "triple": 0.2, "home_run": 0.5}
    luck = compute_raw_luck(probs, "out")
    assert luck < 0


def test_outcome_matching_expectation_is_near_zero():
    # With even probabilities, expected value = mean of value map = 2.0 (double).
    luck = compute_raw_luck(EVEN_PROBS, "double")
    assert math.isclose(luck, 0.0, abs_tol=1e-9)


def test_invalid_probability_distribution_fails_clearly():
    bad_sum = {"out": 0.5, "single": 0.5, "double": 0.5, "triple": 0.0, "home_run": 0.0}
    with pytest.raises(RawLuckValidationError, match="sum to ~1"):
        compute_raw_luck(bad_sum, "out")

    non_finite = dict(EVEN_PROBS, out=float("nan"))
    with pytest.raises(RawLuckValidationError, match="non-finite"):
        compute_raw_luck(non_finite, "out")

    negative = dict(EVEN_PROBS, out=-0.1, single=0.3)
    with pytest.raises(RawLuckValidationError, match="negative"):
        compute_raw_luck(negative, "out")


def test_missing_classes_fail_clearly():
    incomplete = {"out": 0.5, "single": 0.5}
    with pytest.raises(RawLuckValidationError, match="missing class"):
        compute_raw_luck(incomplete, "out")


def test_unknown_observed_outcome_fails_clearly():
    with pytest.raises(RawLuckValidationError, match="not a known class"):
        compute_raw_luck(EVEN_PROBS, "grand_slam_walk_off")


def test_class_ordering_does_not_alter_raw_luck_result():
    reversed_order = tuple(reversed(CLASS_ORDER))
    a = compute_raw_luck(EVEN_PROBS, "single", class_order=CLASS_ORDER)
    b = compute_raw_luck(EVEN_PROBS, "single", class_order=reversed_order)
    assert math.isclose(a, b)


def test_raw_luck_uses_configurable_value_map():
    custom_map = dict(DEFAULT_VALUE_MAP, home_run=10.0)
    luck_default = compute_raw_luck(EVEN_PROBS, "home_run")
    luck_custom = compute_raw_luck(EVEN_PROBS, "home_run", value_map=custom_map)
    assert luck_custom > luck_default


# -- public score -------------------------------------------------------


def test_public_score_preserves_sign():
    assert raw_luck_to_public_score(1.5) > 0
    assert raw_luck_to_public_score(-1.5) < 0
    assert raw_luck_to_public_score(0.0) == 0.0


def test_public_score_is_monotonic():
    xs = [-4.0, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 4.0]
    scores = [raw_luck_to_public_score(x) for x in xs]
    assert scores == sorted(scores)


def test_public_score_is_clipped_to_valid_range():
    assert -100.0 <= raw_luck_to_public_score(1000.0) <= 100.0
    assert -100.0 <= raw_luck_to_public_score(-1000.0) <= 100.0


def test_public_score_rejects_invalid_input():
    with pytest.raises(PublicScoreValidationError):
        raw_luck_to_public_score(float("nan"))
    with pytest.raises(PublicScoreValidationError):
        raw_luck_to_public_score(1.0, scale=0.0)


# -- confidence -----------------------------------------------------------


def test_confidence_reports_high_completeness_when_all_core_fields_present():
    row = pd.Series(
        {
            "launch_speed": 95.0,
            "launch_angle": 12.0,
            "hit_distance_sc": 300.0,
            "hc_x": 120.0,
            "hc_y": 140.0,
            "bb_type": "line_drive",
            "stand": "R",
            "venue": "Test Park",
        }
    )
    report = compute_confidence(row)
    assert report.label == HIGH_LABEL
    assert report.required_fields_present_pct == 100.0
    assert report.spray_direction_available is True
    assert report.park_available is True


def test_confidence_reports_low_completeness_when_most_fields_missing():
    row = pd.Series({"launch_speed": 95.0})
    report = compute_confidence(row)
    assert report.label in (LOW_LABEL, MEDIUM_LABEL)
    assert report.required_fields_present_pct < 100.0
    assert report.spray_direction_available is False
    assert report.park_available is False


def test_confidence_does_not_require_all_fields_to_avoid_crashing():
    report = compute_confidence({})
    assert report.required_fields_present_pct == 0.0
    assert report.label == LOW_LABEL
