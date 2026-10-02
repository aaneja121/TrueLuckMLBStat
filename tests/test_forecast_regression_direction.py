"""Contact Forecast R1: the regression-direction experiment and its placebo.

The claim these tests exist to protect is narrow and easy to overstate. A
negative association between `contact_result_surprise` and subsequent change
follows from the two quantities SHARING a `current_realized` term, with no
player-specific contact information involved. So the tests build both worlds
explicitly:

  - a world where `deserved` is pure noise -- the mechanical coupling still
    produces a negative slope, and the placebo reproduces it;
  - a world where `deserved` genuinely tracks hitter skill -- and only there
    does the real signal separate from the placebo.

Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import regression_direction as rd


def make_cell(
    *,
    n: int = 300,
    seed: int = 3,
    deserved_carries_skill: bool,
    skill_sd: float = 5.0,
    luck_sd: float = 6.0,
    deserved_noise_sd: float = 1.0,
    season: int = 2024,
    cutoff: int = 100,
    horizon: int = 100,
) -> pd.DataFrame:
    """Hitters with a true skill, luck-contaminated realized rates, and a
    `deserved` read that either tracks skill or is unrelated noise."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(4.0, skill_sd, n)
    current_realized = skill + rng.normal(0.0, luck_sd, n)
    future_realized = skill + rng.normal(0.0, luck_sd, n)
    if deserved_carries_skill:
        deserved = skill + rng.normal(0.0, deserved_noise_sd, n)
    else:
        deserved = rng.normal(4.0, skill_sd, n)
    return pd.DataFrame(
        {
            "batter": np.arange(n),
            "season": season,
            "cutoff": cutoff,
            "horizon": horizon,
            rd.CURRENT_REALIZED: current_realized,
            rd.CURRENT_DESERVED: deserved,
            rd.CURRENT_SURPRISE: current_realized - deserved,
            rd.FUTURE_REALIZED: future_realized,
        }
    )


# --------------------------------------------------------------------------
# The mechanical coupling
# --------------------------------------------------------------------------


def test_mechanical_coupling_alone_produces_a_negative_surprise_slope() -> None:
    """No player-specific deserved information, yet the slope is still negative."""
    cell = make_cell(deserved_carries_skill=False, seed=1)
    result = rd.run_regression_direction_cell(cell, permutations=200, seed=1)
    assert result["real_surprise_regression"]["slope_surprise"] < 0


def test_the_placebo_reproduces_that_negative_slope() -> None:
    """The null is centred on a negative slope, because the coupling survives it."""
    cell = make_cell(deserved_carries_skill=False, seed=1)
    result = rd.run_regression_direction_cell(cell, permutations=300, seed=1)
    assert result["placebo"]["surprise_slope"]["placebo_mean"] < 0


def test_without_real_signal_the_real_slope_sits_inside_the_placebo_null() -> None:
    """The honest negative result: significant-looking, but not beyond placebo."""
    cell = make_cell(deserved_carries_skill=False, seed=2)
    result = rd.run_regression_direction_cell(cell, permutations=500, seed=2)
    placebo = result["placebo"]
    assert placebo["surprise_slope"]["p_lower_real_more_negative"] > 0.05
    assert placebo["incremental_deserved_slope"]["p_upper_real_more_positive"] > 0.05
    assert placebo["delta_r_squared"]["p_upper_real_more_positive"] > 0.05


def test_a_real_deserved_signal_separates_from_the_placebo() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=3)
    result = rd.run_regression_direction_cell(cell, permutations=500, seed=3)
    placebo = result["placebo"]
    assert placebo["surprise_slope"]["p_lower_real_more_negative"] < 0.01
    assert result["real_incremental_regression"]["slope_current_deserved"] > 0
    assert placebo["incremental_deserved_slope"]["p_upper_real_more_positive"] < 0.01
    assert placebo["delta_r_squared"]["p_upper_real_more_positive"] < 0.01


def test_the_mean_regression_baseline_is_reported_and_negative() -> None:
    """Extreme current performance regresses even with no contact information."""
    cell = make_cell(deserved_carries_skill=False, seed=4)
    result = rd.run_regression_direction_cell(cell, permutations=50, seed=4)
    baseline = result["mean_regression_baseline"]
    assert baseline["model"] == "outcome ~ current_realized"
    assert baseline["slope_current_realized"] < 0


def test_delta_r_squared_is_measured_against_that_baseline() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=5)
    result = rd.run_regression_direction_cell(cell, permutations=50, seed=5)
    incremental = result["real_incremental_regression"]
    baseline = result["mean_regression_baseline"]
    assert incremental["delta_r_squared_vs_mean_regression_baseline"] == pytest.approx(
        incremental["r_squared"] - baseline["r_squared"]
    )
    assert incremental["delta_r_squared_vs_mean_regression_baseline"] > 0


def test_prespecified_directions_are_recorded_with_the_result() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=6)
    result = rd.run_regression_direction_cell(cell, permutations=20, seed=6)
    directions = result["placebo"]["prespecified_directions"]
    assert "NEGATIVE" in directions["surprise_slope"]
    assert "POSITIVE" in directions["incremental_deserved_slope"]
    # Both tails are always reported, so a direction cannot be chosen post hoc.
    for statistic in ("surprise_slope", "incremental_deserved_slope", "delta_r_squared"):
        entry = result["placebo"][statistic]
        assert entry["p_lower_real_more_negative"] is not None
        assert entry["p_upper_real_more_positive"] is not None


def test_a_permutation_p_value_is_never_exactly_zero() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=7)
    result = rd.run_regression_direction_cell(cell, permutations=100, seed=7)
    p = result["placebo"]["surprise_slope"]["p_lower_real_more_negative"]
    assert p >= 1.0 / 101.0


# --------------------------------------------------------------------------
# Construction contracts
# --------------------------------------------------------------------------


def test_the_surprise_identity_is_enforced() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=8)
    cell[rd.CURRENT_SURPRISE] = cell[rd.CURRENT_SURPRISE] + 1.0
    with pytest.raises(rd.ForecastRegressionDirectionError, match="mechanical-coupling"):
        rd.run_regression_direction_cell(cell, permutations=5, seed=8)


def test_the_outcome_is_future_minus_current() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=9, n=50)
    prepared = rd.prepare_cell(cell)
    assert np.allclose(
        prepared["outcome_change_rv_per_100"],
        prepared[rd.FUTURE_REALIZED] - prepared[rd.CURRENT_REALIZED],
    )


def test_the_placebo_loop_matches_fit_ols_on_the_same_permuted_data() -> None:
    """The fast numpy path and `fit_ols` are the same arithmetic."""
    cell = rd.prepare_cell(make_cell(deserved_carries_skill=True, seed=10, n=120))
    rng = np.random.default_rng(0)
    permuted = rng.permutation(cell[rd.CURRENT_DESERVED].to_numpy(dtype=float))
    realized = cell[rd.CURRENT_REALIZED].to_numpy(dtype=float)
    y = cell["outcome_change_rv_per_100"].to_numpy(dtype=float)
    ones = np.ones(len(cell))

    beta_fast, _ = rd._least_squares(np.column_stack([ones, realized, permuted]), y)
    shuffled = cell.copy()
    shuffled[rd.CURRENT_DESERVED] = permuted
    slow = rd.fit_ols(
        shuffled,
        outcome="outcome_change_rv_per_100",
        predictors=[rd.CURRENT_REALIZED, rd.CURRENT_DESERVED],
    )
    assert beta_fast[1] == pytest.approx(slow.coefficients[rd.CURRENT_REALIZED])
    assert beta_fast[2] == pytest.approx(slow.coefficients[rd.CURRENT_DESERVED])


def test_too_few_windows_is_reported_not_fitted() -> None:
    cell = make_cell(deserved_carries_skill=True, seed=11, n=5)
    result = rd.run_regression_direction_cell(cell, permutations=5, seed=11)
    assert result["status"] == "insufficient_windows"


def test_ols_recovers_a_known_slope() -> None:
    frame = pd.DataFrame({"x": np.arange(50.0), "y": 3.0 + 2.0 * np.arange(50.0)})
    fit = rd.fit_ols(frame, outcome="y", predictors=["x"])
    assert fit.coefficients["x"] == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(3.0)
    assert fit.r_squared == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Probabilistic direction prediction
# --------------------------------------------------------------------------


def make_walk_forward_windows(*, seed: int = 12) -> pd.DataFrame:
    frames = []
    for offset, season in enumerate((2022, 2023, 2024)):
        frame = make_cell(n=200, seed=seed + offset, deserved_carries_skill=True, season=season)
        frame["batter"] = frame["batter"] + 1000 * offset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_direction_probability_reports_proper_scoring_rules_not_accuracy_alone() -> None:
    result = rd.run_direction_probability(make_walk_forward_windows())
    fold = result["cells"]["K100_H100_eval2024"]
    assert fold["status"] == "fitted"
    for model in fold["models"].values():
        assert np.isfinite(model["brier_score"])
        assert np.isfinite(model["log_loss"])
        assert 0.0 <= model["event_prevalence"] <= 1.0
        assert "majority_class_accuracy" in model
        assert "accuracy_is_insufficient" in model


def test_every_prediction_lands_in_exactly_one_calibration_bin() -> None:
    result = rd.run_direction_probability(make_walk_forward_windows())
    fold = result["cells"]["K100_H100_eval2024"]
    for model in fold["models"].values():
        binned = sum(row["n"] for row in model["calibration_by_bin"])
        assert binned == model["n"]


def test_the_base_rate_model_is_the_training_prevalence() -> None:
    windows = make_walk_forward_windows()
    result = rd.run_direction_probability(windows)
    fold = result["cells"]["K100_H100_eval2023"]
    base = fold["models"]["base_rate"]
    # A constant predictor: every prediction sits in one bin.
    occupied = [row for row in base["calibration_by_bin"] if row["n"] > 0]
    assert len(occupied) == 1
    assert occupied[0]["mean_predicted"] == pytest.approx(fold["train_event_prevalence"])


def test_every_fold_is_fit_on_strictly_earlier_seasons() -> None:
    result = rd.run_direction_probability(make_walk_forward_windows())
    for key, fold in result["cells"].items():
        if fold.get("status") != "fitted":
            continue
        assert all(s < fold["evaluate_season"] for s in fold["fit_seasons"]), key


def test_a_non_causal_fold_is_refused() -> None:
    windows = make_walk_forward_windows()
    with pytest.raises(rd.ForecastRegressionDirectionError, match="strictly earlier"):
        rd.run_direction_probability(windows, folds=(((2023, 2024), 2024),))


def test_the_training_rule_is_recorded_with_the_result() -> None:
    result = rd.run_direction_probability(make_walk_forward_windows())
    assert "strictly earlier seasons" in result["training_rule"]
    assert set(result["models"]) == {"base_rate", "realized_only", "realized_plus_deserved"}
