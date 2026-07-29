"""Tests for the Contact Luck v0.7B defensive-execution scoring.

Pure-function tests only -- no network, no real Statcast data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.scoring.defensive_execution import (
    DefensiveExecutionError,
    compute_defensive_execution,
)


def test_out_made_on_low_probability_opportunity_is_strong_positive_execution():
    actual = pd.Series([1])
    p_out = pd.Series([0.15])
    result = compute_defensive_execution(actual, p_out)
    assert result["defensive_execution"].iloc[0] == pytest.approx(0.85)


def test_out_missed_on_high_probability_opportunity_is_strong_negative_execution():
    actual = pd.Series([0])
    p_out = pd.Series([0.9])
    result = compute_defensive_execution(actual, p_out)
    assert result["defensive_execution"].iloc[0] == pytest.approx(-0.9)


def test_routine_catch_is_near_neutral_execution():
    actual = pd.Series([1])
    p_out = pd.Series([0.95])
    result = compute_defensive_execution(actual, p_out)
    assert result["defensive_execution"].iloc[0] == pytest.approx(0.05, abs=1e-9)


def test_batter_favorable_circumstance_is_negated_execution():
    actual = pd.Series([1, 0])
    p_out = pd.Series([0.2, 0.8])
    result = compute_defensive_execution(actual, p_out)
    pd.testing.assert_series_equal(
        result["batter_favorable_defensive_circumstance"],
        -result["defensive_execution"],
        check_names=False,
    )


def test_strong_defense_is_unfavorable_for_the_batter():
    # Out made on a low-probability opportunity: strong positive defense,
    # which must be NEGATIVE (unfavorable) from the batter's perspective.
    actual = pd.Series([1])
    p_out = pd.Series([0.1])
    result = compute_defensive_execution(actual, p_out)
    assert result["defensive_execution"].iloc[0] > 0
    assert result["batter_favorable_defensive_circumstance"].iloc[0] < 0


def test_poor_defense_is_favorable_for_the_batter():
    # High-probability opportunity missed: strong negative defense, which
    # must be POSITIVE (favorable) from the batter's perspective.
    actual = pd.Series([0])
    p_out = pd.Series([0.9])
    result = compute_defensive_execution(actual, p_out)
    assert result["defensive_execution"].iloc[0] < 0
    assert result["batter_favorable_defensive_circumstance"].iloc[0] > 0


def test_rejects_mismatched_index():
    actual = pd.Series([1], index=[0])
    p_out = pd.Series([0.5], index=[1])
    with pytest.raises(DefensiveExecutionError, match="identical index"):
        compute_defensive_execution(actual, p_out)


def test_rejects_p_out_outside_unit_interval():
    actual = pd.Series([1])
    p_out = pd.Series([1.5])
    with pytest.raises(DefensiveExecutionError, match=r"\[0, 1\]"):
        compute_defensive_execution(actual, p_out)


def test_rejects_non_binary_actual_indicator():
    actual = pd.Series([2])
    p_out = pd.Series([0.5])
    with pytest.raises(DefensiveExecutionError, match="0/1"):
        compute_defensive_execution(actual, p_out)


def test_vectorized_over_multiple_rows():
    actual = pd.Series([1, 0, 1, 0])
    p_out = pd.Series([0.2, 0.3, 0.8, 0.9])
    result = compute_defensive_execution(actual, p_out)
    assert len(result) == 4
    assert list(result.columns) == [
        "defensive_execution",
        "batter_favorable_defensive_circumstance",
    ]
