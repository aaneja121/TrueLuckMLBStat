"""Contact Luck v0.5: per-play weather/environmental run-value attribution.

Only meaningful once a weather-aware candidate has passed the Version 0.5
adoption rule (`mlb_luck_score.models.compare_weather_aware.
recommend_weather_adoption`) -- this module provides the reusable
computation, but a caller should not treat its output as a validated,
headline Contact Luck component unless that gate has actually been passed
(see README.md "Weather attribution (Version 0.5)"). Building and testing
this infrastructure does not itself constitute adoption.

    expected_run_value_actual_environment   = E[run_value | actual effective weather]
    expected_run_value_standard_environment = E[run_value | standardized weather,
                                                             every non-weather feature held constant]
    weather_run_value_effect = expected_run_value_actual_environment
                              - expected_run_value_standard_environment

Positive `weather_run_value_effect` means the ACTUAL weather that day made
contact more favorable to the batter than a fixed, neutral reference
atmosphere would have (see `mlb_luck_score.features.build_contact_features.
generate_standardized_environment_rows` for the exact standardized values);
negative means less favorable.

This is DELIBERATELY KEPT SEPARATE from `mlb_luck_score.scoring.
contact_luck.compute_raw_contact_luck_runs` (`raw_contact_luck_runs =
actual_run_value - expected_run_value`, using whichever model is the
selected production baseline's own prediction, actual weather included if
weather were ever adopted into that baseline). Do NOT add
`weather_run_value_effect` on top of `raw_contact_luck_runs` -- that would
double-count weather's contribution (once implicitly, through the baseline
model's own prediction if it uses weather features; once explicitly, through
this attribution). If a weather-aware candidate is ever adopted as the
production baseline, `raw_contact_luck_runs` computed from IT already
reflects actual weather -- `weather_run_value_effect` from THIS module is a
separate, additional decomposition ("how much of the play's expected value
came from weather specifically"), not an addend to luck.

Uses the SAME fixed run-value formula as `mlb_luck_score.scoring.
contact_luck.compute_expected_run_value` (`mlb_luck_score.scoring.
run_values.DEFAULT_RUN_VALUE_MAP`), vectorized over a DataFrame of
predictions rather than one play's dict of probabilities -- this is the
identical formula `mlb_luck_score.models.build_reference_score` already
uses vectorized this way, not a redefinition (see CLAUDE.md "Never silently
redefine the Luck Score").
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP


class WeatherAttributionError(ValueError):
    """Raised when inputs to the weather-attribution calculation are invalid."""


def compute_expected_run_value_vectorized(
    proba_df: pd.DataFrame, run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP
) -> pd.Series:
    """Vectorized `sum(p(outcome) * run_value(outcome))` over every row of `proba_df`.

    Same formula as `mlb_luck_score.scoring.contact_luck.
    compute_expected_run_value`, applied to a whole probability DataFrame at
    once (columns in `CLASS_ORDER`) instead of one play's probability dict.
    """
    if list(proba_df.columns) != list(CLASS_ORDER):
        raise WeatherAttributionError(
            f"Expected proba_df columns {CLASS_ORDER}, got {list(proba_df.columns)}"
        )
    run_value_series = pd.Series(run_value_map).reindex(list(CLASS_ORDER))
    if run_value_series.isna().any():
        missing = run_value_series[run_value_series.isna()].index.tolist()
        raise WeatherAttributionError(f"run_value_map is missing class(es): {missing}")
    values = proba_df[list(CLASS_ORDER)].to_numpy() @ run_value_series.to_numpy()
    return pd.Series(values, index=proba_df.index)


def compute_weather_attribution(
    actual_proba_df: pd.DataFrame,
    standardized_proba_df: pd.DataFrame,
    *,
    run_value_map: dict[str, float] = DEFAULT_RUN_VALUE_MAP,
) -> pd.DataFrame:
    """Per-play weather run-value attribution.

    Args:
        actual_proba_df: Predicted probabilities (columns in `CLASS_ORDER`)
            using each play's ACTUAL effective weather features -- every
            other (non-weather) feature identical to `standardized_proba_df`'s
            inputs.
        standardized_proba_df: Predicted probabilities using the Version 0.5
            standardized environment (see `mlb_luck_score.features.
            build_contact_features.generate_standardized_environment_rows`)
            for the SAME rows, in the SAME order.
        run_value_map: Fixed run-value table. Defaults to
            `mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`.

    Returns:
        A DataFrame (same index as the inputs) with
        `expected_run_value_actual_environment`,
        `expected_run_value_standard_environment`, and
        `weather_run_value_effect` columns.

    Raises:
        WeatherAttributionError: If the two inputs have different indices/
            lengths (a mismatched pair cannot be meaningfully compared).
    """
    if len(actual_proba_df) != len(standardized_proba_df) or not actual_proba_df.index.equals(
        standardized_proba_df.index
    ):
        raise WeatherAttributionError(
            "actual_proba_df and standardized_proba_df must have the identical index/row order "
            "-- they must be predictions for the SAME rows, differing only in weather features."
        )

    expected_actual = compute_expected_run_value_vectorized(actual_proba_df, run_value_map)
    expected_standard = compute_expected_run_value_vectorized(standardized_proba_df, run_value_map)

    return pd.DataFrame(
        {
            "expected_run_value_actual_environment": expected_actual,
            "expected_run_value_standard_environment": expected_standard,
            "weather_run_value_effect": expected_actual - expected_standard,
        },
        index=actual_proba_df.index,
    )
