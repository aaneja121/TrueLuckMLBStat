"""Contact Forecast R1: the baseline ladder and matched shrinkage.

The point of these tests is that shrinkage is genuinely MATCHED -- the same
estimator, differing only in which column it reads -- and genuinely CAUSAL,
with every parameter fit on strictly earlier seasons.

Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from forecast import baselines as bl
from forecast.forecast_config import BASELINE_LADDER


def make_events(
    *,
    seasons: tuple[int, ...] = (2021, 2022),
    n_batters: int = 60,
    n_events: int = 200,
    seed: int = 3,
    true_sd: float = 0.10,
    noise_sd: float = 0.50,
) -> pd.DataFrame:
    """Per-batted-ball rows with a known between-hitter signal.

    Each hitter has a true rate drawn with SD `true_sd`; observed values add
    per-event noise with SD `noise_sd`. Deserved is built to be a LESS noisy
    read on the same true rate, which is the real asymmetry the matched
    shrinkage exists to handle.
    """
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for season in seasons:
        for batter in range(n_batters):
            true_rate = rng.normal(0.05, true_sd)
            observed = rng.normal(true_rate, noise_sd, n_events)
            deserved = rng.normal(true_rate, noise_sd * 0.4, n_events)
            rows.append(
                pd.DataFrame(
                    {
                        "batter": batter,
                        "season": season,
                        "observed_contact_result_run_value": observed,
                        "baseline_expected_contact_run_value": deserved,
                        "contact_result_surprise": observed - deserved,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def make_windows(season: int = 2023, n: int = 40, cutoff: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(n)],
            "batter": np.arange(n),
            "season": season,
            "cutoff": cutoff,
            "horizon": 100,
            "n_resolved_bbe_through_cutoff": float(cutoff),
            "std_realized_rv_per_100": rng.normal(5.0, 8.0, n),
            "std_deserved_rv_per_100": rng.normal(5.0, 5.0, n),
            "target_realized_rv_per_100": rng.normal(5.0, 7.0, n),
        }
    )


# --------------------------------------------------------------------------
# Matched: the same estimator for both quantities
# --------------------------------------------------------------------------


def test_realized_and_deserved_use_the_identical_estimator() -> None:
    """Matched means one function; only the column read differs.

    Fitting both on the SAME column must produce identical parameters -- if
    the two paths differed in any way, this would fail.
    """
    events = make_events()

    as_realized = bl.fit_shrinkage(
        events, value_column="observed_contact_result_run_value", quantity="a"
    )
    as_deserved = bl.fit_shrinkage(
        events, value_column="observed_contact_result_run_value", quantity="b"
    )

    assert as_realized.prior_mean == as_deserved.prior_mean
    assert as_realized.within_variance == as_deserved.within_variance
    assert as_realized.between_variance == as_deserved.between_variance


def test_the_less_noisy_quantity_shrinks_less() -> None:
    """The scientific mechanism: higher reliability means less shrinkage.

    Deserved is constructed here as a less noisy read on the same true rate,
    so its reliability must be higher. This is what a real advantage would
    look like -- and why comparing RAW realized against RAW deserved cannot
    distinguish it from generic regression to the mean.
    """
    events = make_events()

    realized = bl.fit_shrinkage(
        events, value_column="observed_contact_result_run_value", quantity="realized"
    )
    deserved = bl.fit_shrinkage(
        events, value_column="baseline_expected_contact_run_value", quantity="deserved"
    )

    assert deserved.reliability(100) > realized.reliability(100)
    assert 0.0 <= realized.reliability(100) <= 1.0
    assert 0.0 <= deserved.reliability(100) <= 1.0


def test_reliability_increases_with_sample_size() -> None:
    fit = bl.fit_shrinkage(
        make_events(), value_column="observed_contact_result_run_value", quantity="realized"
    )
    reliabilities = [fit.reliability(n) for n in (50, 100, 150, 200)]
    assert reliabilities == sorted(reliabilities)


def test_shrinkage_pulls_toward_the_prior_mean() -> None:
    fit = bl.fit_shrinkage(
        make_events(), value_column="observed_contact_result_run_value", quantity="realized"
    )
    rates = pd.Series([50.0, -50.0])
    counts = pd.Series([100.0, 100.0])

    shrunk = fit.apply(rates, counts)

    assert abs(shrunk.iloc[0] - fit.prior_mean) < abs(rates.iloc[0] - fit.prior_mean)
    assert abs(shrunk.iloc[1] - fit.prior_mean) < abs(rates.iloc[1] - fit.prior_mean)


def test_no_detectable_between_hitter_signal_shrinks_fully_to_the_mean() -> None:
    """A negative method-of-moments estimate is a finding, not a negative variance."""
    events = make_events(true_sd=0.0, noise_sd=0.8, seed=17)

    fit = bl.fit_shrinkage(
        events, value_column="observed_contact_result_run_value", quantity="realized"
    )

    assert fit.between_variance >= 0.0
    assert fit.reliability(100) < 0.25


def test_shrinkage_preserves_rank_within_a_cell() -> None:
    """Within a cell every hitter has the same n, so shrinkage is affine.

    Affine maps preserve order, so Spearman is IDENTICAL for raw and shrunk.
    The report must therefore never claim shrinkage improved rank agreement.
    """
    fit = bl.fit_shrinkage(
        make_events(), value_column="observed_contact_result_run_value", quantity="realized"
    )
    windows = make_windows()
    raw = windows["std_realized_rv_per_100"]
    shrunk = fit.apply(raw, windows["n_resolved_bbe_through_cutoff"])

    assert spearmanr(raw, windows["target_realized_rv_per_100"]).statistic == pytest.approx(
        spearmanr(shrunk, windows["target_realized_rv_per_100"]).statistic
    )


# --------------------------------------------------------------------------
# Causal: fit on strictly prior seasons only
# --------------------------------------------------------------------------


def test_shrinkage_fit_records_the_seasons_it_saw() -> None:
    fit = bl.fit_shrinkage(
        make_events(seasons=(2021, 2022)),
        value_column="observed_contact_result_run_value",
        quantity="realized",
    )
    assert fit.fit_seasons == (2021, 2022)


def test_causality_guard_rejects_a_fit_that_saw_the_predicted_season() -> None:
    fit = bl.fit_shrinkage(
        make_events(seasons=(2021, 2022, 2023)),
        value_column="observed_contact_result_run_value",
        quantity="realized",
    )
    with pytest.raises(bl.ForecastBaselineError, match="not strictly earlier"):
        bl.assert_fit_is_causal(fit, 2023)


def test_causality_guard_accepts_a_strictly_prior_fit() -> None:
    fit = bl.fit_shrinkage(
        make_events(seasons=(2021, 2022)),
        value_column="observed_contact_result_run_value",
        quantity="realized",
    )
    bl.assert_fit_is_causal(fit, 2023)


def test_building_the_ladder_refuses_prior_events_from_the_predicted_season() -> None:
    events = make_events(seasons=(2021, 2022, 2023))
    with pytest.raises(bl.ForecastBaselineError, match="not strictly earlier"):
        bl.build_baseline_predictions(
            make_windows(season=2023), prior_events=events, predicted_season=2023
        )


def test_ladder_refuses_windows_spanning_several_seasons() -> None:
    windows = pd.concat([make_windows(2023), make_windows(2024)], ignore_index=True)
    with pytest.raises(bl.ForecastBaselineError, match="fit a separate ladder per season"):
        bl.build_baseline_predictions(windows, prior_events=make_events(), predicted_season=2023)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_ladder_produces_every_frozen_rung() -> None:
    result = bl.build_baseline_predictions(
        make_windows(season=2023), prior_events=make_events(), predicted_season=2023
    )
    for rung in BASELINE_LADDER:
        assert rung in result.columns
        assert result[rung].notna().all()


def test_league_mean_rung_is_constant_and_comes_from_prior_seasons() -> None:
    events = make_events()
    result = bl.build_baseline_predictions(
        make_windows(season=2023), prior_events=events, predicted_season=2023
    )

    assert result["league_mean"].nunique() == 1
    expected = 100.0 * events["observed_contact_result_run_value"].mean()
    assert result["league_mean"].iloc[0] == pytest.approx(expected)


def test_raw_rungs_are_the_unshrunk_season_to_date_rates() -> None:
    windows = make_windows(season=2023)
    result = bl.build_baseline_predictions(
        windows, prior_events=make_events(), predicted_season=2023
    )
    pd.testing.assert_series_equal(
        result["raw_realized_persistence"],
        windows["std_realized_rv_per_100"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["raw_deserved_persistence"],
        windows["std_deserved_rv_per_100"],
        check_names=False,
    )


def test_shrunk_rungs_sit_between_the_raw_value_and_the_league_mean() -> None:
    result = bl.build_baseline_predictions(
        make_windows(season=2023), prior_events=make_events(), predicted_season=2023
    )
    for raw_col, shrunk_col in (
        ("raw_realized_persistence", "shrunk_realized_persistence"),
        ("raw_deserved_persistence", "shrunk_deserved_persistence"),
    ):
        low = np.minimum(result[raw_col], result["league_mean"])
        high = np.maximum(result[raw_col], result["league_mean"])
        assert (result[shrunk_col] >= low - 1e-9).all()
        assert (result[shrunk_col] <= high + 1e-9).all()


# --------------------------------------------------------------------------
# Rescaling
# --------------------------------------------------------------------------


def test_rescaling_is_fit_on_prior_seasons_and_recorded() -> None:
    prior = make_windows(season=2022, n=60)
    rescaling = bl.fit_linear_rescaling(
        prior, predictor="std_deserved_rv_per_100", target="target_realized_rv_per_100"
    )
    assert rescaling.fit_seasons == (2022,)
    bl.assert_rescaling_is_causal(rescaling, 2023)


def test_rescaling_causality_guard_rejects_a_scored_season_fit() -> None:
    rescaling = bl.fit_linear_rescaling(
        make_windows(season=2023, n=60),
        predictor="std_deserved_rv_per_100",
        target="target_realized_rv_per_100",
    )
    with pytest.raises(bl.ForecastBaselineError, match="not strictly earlier"):
        bl.assert_rescaling_is_causal(rescaling, 2023)


def test_rescaling_recovers_a_known_linear_relationship() -> None:
    prior = pd.DataFrame(
        {
            "season": 2022,
            "std_deserved_rv_per_100": np.arange(50, dtype=float),
            "target_realized_rv_per_100": 3.0 + 2.0 * np.arange(50, dtype=float),
        }
    )
    rescaling = bl.fit_linear_rescaling(
        prior, predictor="std_deserved_rv_per_100", target="target_realized_rv_per_100"
    )
    assert rescaling.intercept == pytest.approx(3.0)
    assert rescaling.slope == pytest.approx(2.0)


def test_rescaling_with_a_positive_slope_preserves_rank() -> None:
    """Rescaling changes MAE/RMSE, never Spearman -- it cannot rescue rank."""
    rng = np.random.default_rng(4)
    predictor = rng.normal(5.0, 5.0, 60)
    prior = pd.DataFrame(
        {
            "season": 2022,
            "std_deserved_rv_per_100": predictor,
            "target_realized_rv_per_100": 1.0 + 1.4 * predictor + rng.normal(0, 2.0, 60),
        }
    )
    rescaling = bl.fit_linear_rescaling(
        prior, predictor="std_deserved_rv_per_100", target="target_realized_rv_per_100"
    )
    assert rescaling.preserves_rank

    current = make_windows(season=2023, n=60)
    rescaled = rescaling.apply(current["std_deserved_rv_per_100"])

    assert spearmanr(
        current["std_deserved_rv_per_100"], current["target_realized_rv_per_100"]
    ).statistic == pytest.approx(
        spearmanr(rescaled, current["target_realized_rv_per_100"]).statistic
    )


def test_a_negative_rescaling_slope_is_flagged_as_anti_predictive() -> None:
    """A negative slope inverts rank and means the predictor ran the wrong way.

    It must be surfaced explicitly, never applied quietly -- discovering it
    later by noticing a sign would mean every downstream number had already
    been read the wrong way round.
    """
    predictor = np.arange(40, dtype=float)
    prior = pd.DataFrame(
        {
            "season": 2022,
            "std_deserved_rv_per_100": predictor,
            "target_realized_rv_per_100": 10.0 - 1.5 * predictor,
        }
    )

    rescaling = bl.fit_linear_rescaling(
        prior, predictor="std_deserved_rv_per_100", target="target_realized_rv_per_100"
    )
    record = bl.build_rescaling_record(rescaling)

    assert rescaling.slope < 0
    assert rescaling.preserves_rank is False
    assert record["slope_is_negative"] is True


def test_rescaling_refuses_a_zero_variance_predictor() -> None:
    prior = pd.DataFrame(
        {
            "season": 2022,
            "std_deserved_rv_per_100": np.zeros(10),
            "target_realized_rv_per_100": np.arange(10, dtype=float),
        }
    )
    with pytest.raises(bl.ForecastBaselineError, match="zero variance"):
        bl.fit_linear_rescaling(
            prior, predictor="std_deserved_rv_per_100", target="target_realized_rv_per_100"
        )


def test_shrinkage_fit_record_is_manifest_ready() -> None:
    fit = bl.fit_shrinkage(
        make_events(), value_column="baseline_expected_contact_run_value", quantity="deserved"
    )
    record = bl.shrinkage_fit_record(fit)
    assert record["quantity"] == "deserved"
    assert record["fit_seasons"] == [2021, 2022]
    for key in (
        "prior_mean_per_100",
        "within_hitter_variance",
        "between_hitter_variance",
        "reliability_at_100_bbe",
    ):
        assert key in record


def test_fitting_on_no_prior_events_is_refused() -> None:
    empty = make_events().iloc[0:0]
    with pytest.raises(bl.ForecastBaselineError, match="No prior-season events"):
        bl.fit_shrinkage(
            empty, value_column="observed_contact_result_run_value", quantity="realized"
        )
