"""Contact Luck v0.5.1: controlled feature perturbations for physical-plausibility testing.

Version 0.5's adoption process caught two real bugs precisely because it
checked whether the model's WEATHER effects made physical sense, not just
whether aggregate calibration improved (see `mlb_luck_score.scoring.
weather_attribution` and README.md "Weather and air density"). This module
generalizes that idea into a reusable, directly-testable mechanism:
override specific feature columns to controlled values across a row sample,
predict with an already-trained model, and check whether the AGGREGATE
predicted probability of a given outcome moves in the physically expected
direction.

IMPORTANT -- the same lesson that produced the Version 0.5 bug applies here:
every categorical override MUST be a value the model actually saw during
training. `OneHotEncoder(handle_unknown="ignore")` silently encodes any
novel category as all zeros -- a pattern the fitted model never learned to
interpret -- rather than raising an error, so an unseen category produces a
silently wrong prediction, not a crash. Callers of `override_columns` are
responsible for this (see `mlb_luck_score.models.compare_weather_variants`
for the concrete, real-category overrides used for each scenario).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlb_luck_score.models.train_contact_model import TrainedModel, predict_proba_ordered


def override_columns(
    df: pd.DataFrame, overrides: dict[str, Any], *, mask: pd.Series | None = None
) -> pd.DataFrame:
    """Return a copy of `df` with `overrides` applied to masked rows (default: all rows).

    Every non-overridden column (including every non-weather feature) is
    left EXACTLY as-is -- this is the same "hold everything else constant"
    mechanism `mlb_luck_score.features.build_contact_features.
    generate_standardized_environment_rows` uses, generalized to arbitrary
    override values instead of one fixed standardized environment.
    """
    out = df.copy()
    active_mask = mask if mask is not None else pd.Series(True, index=out.index)
    for col, value in overrides.items():
        if col in out.columns:
            out.loc[active_mask, col] = value
    return out


def mean_predicted_probability(
    trained: TrainedModel, df: pd.DataFrame, outcome_class: str
) -> float:
    """Mean predicted probability of `outcome_class` across every row of `df`."""
    feature_cols = trained.numeric_features + trained.categorical_features
    proba = predict_proba_ordered(trained, df[feature_cols])
    return float(proba[outcome_class].mean())


@dataclass(frozen=True)
class DirectionalCheckResult:
    """Result of comparing mean predicted probability under two controlled scenarios.

    Attributes:
        label: Human-readable description of what's being tested (e.g.
            `"air_density: low vs high"`).
        outcome_class: Which outcome class's probability was compared.
        low_label / high_label: Descriptions of the two scenarios.
        mean_prob_low / mean_prob_high: Mean predicted probability of
            `outcome_class` under each scenario.
        delta: `mean_prob_high - mean_prob_low`.
        expect_high_greater: The physically expected direction (`True` if
            the "high" scenario should show a HIGHER probability).
        passed: Whether `delta`'s sign matches `expect_high_greater`.
        sample_size: Number of rows the comparison was computed over.
    """

    label: str
    outcome_class: str
    low_label: str
    high_label: str
    mean_prob_low: float
    mean_prob_high: float
    delta: float
    expect_high_greater: bool
    passed: bool
    sample_size: int


def check_directional_effect(
    trained: TrainedModel,
    df: pd.DataFrame,
    *,
    low_overrides: dict[str, Any],
    high_overrides: dict[str, Any],
    expect_high_greater: bool,
    outcome_class: str = "home_run",
    mask: pd.Series | None = None,
    label: str = "",
    low_label: str = "low",
    high_label: str = "high",
) -> DirectionalCheckResult:
    """Compare mean predicted `outcome_class` probability under two controlled scenarios.

    Args:
        trained: An already-fitted model (see `mlb_luck_score.models.
            train_contact_model.train_model`).
        df: Real rows to perturb (typically a validation set, or a venue
            -specific subset -- see `mlb_luck_score.models.
            compare_weather_variants` for concrete usage).
        low_overrides / high_overrides: Column overrides for each scenario
            (see `override_columns` -- every value must be a real,
            previously-seen category for any categorical column).
        expect_high_greater: The physically expected direction.
        outcome_class: Which outcome class to compare (default `"home_run"`,
            the class most directly affected by carry/air-density effects).
        mask: Which rows to perturb (default: all rows in `df`).
        label, low_label, high_label: Human-readable descriptions for
            reporting.

    Returns:
        A `DirectionalCheckResult`. `passed=True` means the model's
        aggregate behavior under these two scenarios matches physical
        expectation -- NOT proof the model is correct in general, only that
        it didn't get this specific, checkable direction backwards.
    """
    low_df = override_columns(df, low_overrides, mask=mask)
    high_df = override_columns(df, high_overrides, mask=mask)
    mean_low = mean_predicted_probability(trained, low_df, outcome_class)
    mean_high = mean_predicted_probability(trained, high_df, outcome_class)
    delta = mean_high - mean_low
    passed = (delta > 0) if expect_high_greater else (delta < 0)
    return DirectionalCheckResult(
        label=label,
        outcome_class=outcome_class,
        low_label=low_label,
        high_label=high_label,
        mean_prob_low=mean_low,
        mean_prob_high=mean_high,
        delta=delta,
        expect_high_greater=expect_high_greater,
        passed=passed,
        sample_size=int(len(df) if mask is None else int(mask.sum())),
    )
