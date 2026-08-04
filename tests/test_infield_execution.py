"""Tests for the Contact Luck v0.8 infield-execution scoring.

Pure-function tests only -- no network, no real Statcast data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.scoring.infield_execution import (
    InfieldExecutionError,
    compute_infield_execution,
)


def test_out_made_on_low_probability_opportunity_is_strong_positive_execution():
    actual = pd.Series([1])
    p_out = pd.Series([0.15])
    result = compute_infield_execution(actual, p_out)
    assert result["defensive_execution_probability"].iloc[0] == pytest.approx(0.85)


def test_out_missed_on_high_probability_opportunity_is_strong_negative_execution():
    actual = pd.Series([0])
    p_out = pd.Series([0.95])
    result = compute_infield_execution(actual, p_out)
    assert result["defensive_execution_probability"].iloc[0] == pytest.approx(-0.95)


def test_routine_out_is_near_neutral_execution():
    actual = pd.Series([1])
    p_out = pd.Series([0.95])
    result = compute_infield_execution(actual, p_out)
    assert result["defensive_execution_probability"].iloc[0] == pytest.approx(0.05, abs=1e-9)


def test_batter_perspective_is_negated_execution():
    actual = pd.Series([1, 0])
    p_out = pd.Series([0.2, 0.8])
    result = compute_infield_execution(actual, p_out)
    pd.testing.assert_series_equal(
        result["batter_perspective_infield_execution"],
        -result["defensive_execution_probability"],
        check_names=False,
    )


def test_strong_defense_is_unfavorable_for_the_batter():
    actual = pd.Series([1])
    p_out = pd.Series([0.1])
    result = compute_infield_execution(actual, p_out)
    assert result["defensive_execution_probability"].iloc[0] > 0
    assert result["batter_perspective_infield_execution"].iloc[0] < 0


def test_poor_defense_is_favorable_for_the_batter():
    actual = pd.Series([0])
    p_out = pd.Series([0.9])
    result = compute_infield_execution(actual, p_out)
    assert result["defensive_execution_probability"].iloc[0] < 0
    assert result["batter_perspective_infield_execution"].iloc[0] > 0


def test_rejects_mismatched_index():
    actual = pd.Series([1], index=[0])
    p_out = pd.Series([0.5], index=[1])
    with pytest.raises(InfieldExecutionError, match="identical index"):
        compute_infield_execution(actual, p_out)


def test_rejects_p_out_outside_unit_interval():
    actual = pd.Series([1])
    p_out = pd.Series([1.5])
    with pytest.raises(InfieldExecutionError, match=r"\[0, 1\]"):
        compute_infield_execution(actual, p_out)


def test_rejects_non_binary_actual_indicator():
    actual = pd.Series([2])
    p_out = pd.Series([0.5])
    with pytest.raises(InfieldExecutionError, match="0/1"):
        compute_infield_execution(actual, p_out)


def test_vectorized_over_multiple_rows():
    actual = pd.Series([1, 0, 1, 0])
    p_out = pd.Series([0.2, 0.3, 0.8, 0.9])
    result = compute_infield_execution(actual, p_out)
    assert len(result) == 4
    assert list(result.columns) == [
        "defensive_execution_probability",
        "batter_perspective_infield_execution",
    ]
