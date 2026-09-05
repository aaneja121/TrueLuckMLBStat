"""Contact Forecast R1: the metric layer and its frozen sign convention.

The point of these tests is that the sign convention is fixed, checkable, and
impossible for a presentation layer to reverse; that an undefined correlation
stays undefined rather than becoming a zero; and that the macro-average and
the batter-balanced weighting actually change what they claim to change.

Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from forecast import metrics as mt
from forecast.forecast_config import PRIMARY_COMPARISON

REFERENCE, CHALLENGER = PRIMARY_COMPARISON


def make_frame(
    *,
    n_batters: int = 40,
    seasons: tuple[int, ...] = (2023, 2024),
    seed: int = 11,
    challenger_noise: float = 3.0,
    reference_noise: float = 6.0,
) -> pd.DataFrame:
    """Windows where the challenger is a genuinely less noisy read on the target."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for season in seasons:
        for batter in range(n_batters):
            truth = rng.normal(4.0, 5.0)
            rows.append(
                {
                    "window_id": f"{batter}-{season}",
                    "batter": batter,
                    "season": season,
                    "cutoff": 100,
                    "horizon": 100,
                    "target": truth + rng.normal(0.0, 7.0),
                    REFERENCE: truth + rng.normal(0.0, reference_noise),
                    CHALLENGER: truth + rng.normal(0.0, challenger_noise),
                    "league_mean": 4.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The sign convention
# --------------------------------------------------------------------------


def test_delta_is_challenger_minus_reference() -> None:
    frame = make_frame()
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    assert delta.delta_mae == pytest.approx(delta.mae_challenger - delta.mae_reference)
    assert delta.delta_rmse == pytest.approx(delta.rmse_challenger - delta.rmse_reference)


def test_negative_delta_means_the_challenger_is_better() -> None:
    """The whole convention in one assertion: less noise -> negative delta."""
    frame = make_frame(challenger_noise=1.0, reference_noise=9.0)
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    assert delta.mae_challenger < delta.mae_reference
    assert delta.delta_mae < 0
    assert delta.direction_mae == mt.CHALLENGER_BETTER


def test_positive_delta_means_the_reference_is_better() -> None:
    frame = make_frame(challenger_noise=9.0, reference_noise=1.0)
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    assert delta.delta_mae > 0
    assert delta.direction_mae == mt.REFERENCE_BETTER


def test_identical_predictors_give_exactly_zero_delta() -> None:
    frame = make_frame()
    frame[CHALLENGER] = frame[REFERENCE]
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    assert delta.delta_mae == 0.0
    assert delta.direction_mae == mt.NO_DIFFERENCE


def test_report_cannot_reverse_the_sign_convention() -> None:
    """A flipped delta fails the re-derivation, whatever its label says."""
    frame = make_frame()
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    mt.assert_delta_convention(delta)

    flipped = type(delta)(**{**delta.__dict__, "delta_mae": -delta.delta_mae})
    with pytest.raises(mt.ForecastMetricError, match="does not equal"):
        mt.assert_delta_convention(flipped)


def test_a_mislabelled_direction_is_caught() -> None:
    frame = make_frame(challenger_noise=1.0, reference_noise=9.0)
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    mislabelled = type(delta)(**{**delta.__dict__, "direction_mae": mt.REFERENCE_BETTER})
    with pytest.raises(mt.ForecastMetricError, match="direction_mae"):
        mt.assert_delta_convention(mislabelled)


def test_percentage_shares_the_delta_sign() -> None:
    """An inverted "improvement" percentage is the exact failure mode banned."""
    frame = make_frame(challenger_noise=1.0, reference_noise=9.0)
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    assert delta.pct_change_mae is not None
    assert np.sign(delta.pct_change_mae) == np.sign(delta.delta_mae)

    inverted = type(delta)(**{**delta.__dict__, "pct_change_mae": -delta.pct_change_mae})
    with pytest.raises(mt.ForecastMetricError, match="does not share the sign"):
        mt.assert_delta_convention(inverted)


def test_sign_convention_is_machine_readable_and_states_all_three_cases() -> None:
    convention = mt.DELTA_SIGN_CONVENTION
    assert convention["reference"] == REFERENCE
    assert convention["challenger"] == CHALLENGER
    assert "deserved is better" in convention["negative"]
    assert convention["zero"] == "no difference"
    assert "realized is better" in convention["positive"]


# --------------------------------------------------------------------------
# Undefined correlations
# --------------------------------------------------------------------------


def test_constant_predictor_correlations_are_undefined_not_zero() -> None:
    frame = make_frame()
    result = mt.evaluate_predictor(frame, target="target", predictor="league_mean")
    assert result.pearson is None
    assert result.spearman is None
    assert result.pearson_status == mt.CORRELATION_UNDEFINED_CONSTANT_PREDICTOR
    assert result.spearman_status == mt.CORRELATION_UNDEFINED_CONSTANT_PREDICTOR


def test_an_undefined_correlation_never_becomes_a_failure_score() -> None:
    """MAE and RMSE stay valid; the undefined correlation is not coerced to 0."""
    frame = make_frame()
    result = mt.evaluate_predictor(frame, target="target", predictor="league_mean")
    assert np.isfinite(result.mae) and result.mae > 0
    assert np.isfinite(result.rmse) and result.rmse > 0
    record = result.as_record()
    assert record["pearson"] is None
    assert record["spearman"] is None
    assert 0.0 not in (record["pearson"], record["spearman"])


def test_defined_correlations_match_scipy() -> None:
    frame = make_frame()
    result = mt.evaluate_predictor(frame, target="target", predictor=CHALLENGER)
    expected_pearson = pearsonr(frame["target"], frame[CHALLENGER]).statistic
    expected_spearman = spearmanr(frame["target"], frame[CHALLENGER]).statistic
    assert result.pearson == pytest.approx(expected_pearson)
    assert result.spearman == pytest.approx(expected_spearman)
    assert result.pearson_status == mt.CORRELATION_DEFINED


def test_spearman_matches_scipy_under_ties() -> None:
    y_true = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0])
    y_pred = np.array([1.0, 1.0, 3.0, 3.0, 4.0, 6.0, 6.0])
    value, status = mt.spearman(y_true, y_pred)
    assert status == mt.CORRELATION_DEFINED
    assert value == pytest.approx(spearmanr(y_true, y_pred).statistic)


def test_a_single_row_has_no_defined_correlation() -> None:
    value, status = mt.pearson(np.array([1.0]), np.array([2.0]))
    assert value is None
    assert status == mt.CORRELATION_UNDEFINED_INSUFFICIENT_SAMPLE


# --------------------------------------------------------------------------
# Paired-complete-case discipline
# --------------------------------------------------------------------------


def test_metrics_refuse_a_null_rather_than_dropping_it() -> None:
    frame = make_frame()
    frame.loc[0, CHALLENGER] = np.nan
    with pytest.raises(mt.ForecastMetricError, match="null/non-finite"):
        mt.evaluate_predictor(frame, target="target", predictor=CHALLENGER)


def test_a_paired_delta_needs_identical_row_counts() -> None:
    frame = make_frame()
    reference = mt.evaluate_predictor(frame, target="target", predictor=REFERENCE)
    challenger = mt.evaluate_predictor(frame.head(10), target="target", predictor=CHALLENGER)
    with pytest.raises(mt.ForecastMetricError, match="paired delta requires"):
        mt.build_paired_delta(reference, challenger)


def test_evaluate_ladder_skips_a_wholly_missing_rung_without_inventing_one() -> None:
    frame = make_frame()
    frame["shrunk_deserved_persistence"] = np.nan
    measured = mt.evaluate_ladder(
        frame,
        target="target",
        predictors=[REFERENCE, "shrunk_deserved_persistence", "absent_column"],
    )
    assert [m.predictor for m in measured] == [REFERENCE]


def test_evaluate_ladder_measures_each_rung_on_its_own_complete_cases() -> None:
    frame = make_frame()
    frame.loc[frame.index[:5], CHALLENGER] = np.nan
    measured = {
        m.predictor: m
        for m in mt.evaluate_ladder(frame, target="target", predictors=[REFERENCE, CHALLENGER])
    }
    assert measured[REFERENCE].n == len(frame)
    assert measured[CHALLENGER].n == len(frame) - 5


# --------------------------------------------------------------------------
# Season structure and the macro-average
# --------------------------------------------------------------------------


def test_macro_average_is_the_mean_of_the_season_deltas() -> None:
    frame = make_frame()
    season_deltas = mt.per_season_deltas(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    macro = mt.macro_average_delta(season_deltas)
    assert macro["macro_delta_mae"] == pytest.approx(
        float(np.mean([d.delta_mae for d in season_deltas.values()]))
    )
    assert macro["weighting"] == "equal weight per season"


def test_a_large_season_cannot_dominate_the_macro_average() -> None:
    """One huge season swings the pooled figure; the macro-average resists it."""
    small = make_frame(n_batters=10, seasons=(2023,), seed=1, challenger_noise=8.0)
    large = make_frame(n_batters=300, seasons=(2024,), seed=2, challenger_noise=0.5)
    large["batter"] = large["batter"] + 1000
    frame = pd.concat([small, large], ignore_index=True)

    pooled = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    season_deltas = mt.per_season_deltas(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    macro = mt.macro_average_delta(season_deltas)

    assert season_deltas[2023].delta_mae > season_deltas[2024].delta_mae
    # The pooled figure sits close to the 300-hitter season; the macro sits
    # between the two seasons because each gets one vote.
    assert abs(pooled.delta_mae - season_deltas[2024].delta_mae) < abs(
        macro["macro_delta_mae"] - season_deltas[2024].delta_mae
    )


def test_macro_average_records_each_seasons_direction() -> None:
    frame = make_frame()
    season_deltas = mt.per_season_deltas(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    macro = mt.macro_average_delta(season_deltas)
    assert set(macro["season_directions_mae"]) == {"2023", "2024"}


def test_an_empty_season_set_gives_an_explicit_empty_macro() -> None:
    macro = mt.macro_average_delta({})
    assert macro["n_seasons"] == 0
    assert macro["macro_delta_mae"] is None


# --------------------------------------------------------------------------
# Batter-balanced weighting
# --------------------------------------------------------------------------


def test_batter_balanced_equals_window_weighted_with_one_window_per_batter() -> None:
    frame = make_frame(seasons=(2024,))
    pooled = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    balanced = mt.batter_balanced_delta(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    assert balanced.delta_mae == pytest.approx(pooled.delta_mae)


def test_batter_balanced_gives_a_repeated_batter_one_vote() -> None:
    """A hitter with many windows stops carrying many windows' worth of weight."""
    rng = np.random.default_rng(5)
    rows = []
    # One hitter with 50 easy windows, ten hitters with one hard window each.
    for i in range(50):
        rows.append(
            {
                "batter": 1,
                "season": 2024,
                "target": 0.0,
                REFERENCE: 0.0,
                CHALLENGER: 0.0,
                "window_id": f"a{i}",
            }
        )
    for b in range(10):
        rows.append(
            {
                "batter": 100 + b,
                "season": 2024,
                "target": 10.0,
                REFERENCE: 0.0,
                CHALLENGER: 5.0,
                "window_id": f"b{b}",
            }
        )
    frame = pd.DataFrame(rows)
    del rng

    pooled = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    balanced = mt.batter_balanced_delta(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    # Window-weighted, the 50 tied windows dilute the challenger's advantage;
    # batter-balanced, the ten distinct hitters carry 10/11 of the weight.
    assert balanced.delta_mae < pooled.delta_mae
    assert balanced.n == 11


def test_batter_balanced_keeps_the_same_sign_convention() -> None:
    frame = make_frame(challenger_noise=1.0, reference_noise=9.0)
    balanced = mt.batter_balanced_delta(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER
    )
    mt.assert_delta_convention(balanced)
    assert balanced.delta_mae < 0
    assert balanced.direction_mae == mt.CHALLENGER_BETTER


def test_batter_balanced_rmse_is_the_root_of_the_mean_within_batter_mse() -> None:
    frame = make_frame(seasons=(2024,), n_batters=6, seed=7)
    _, balanced_rmse, n = mt.batter_balanced_errors(frame, target="target", predictor=CHALLENGER)
    per_batter = ((frame[CHALLENGER] - frame["target"]) ** 2).groupby(frame["batter"]).mean()
    assert balanced_rmse == pytest.approx(float(np.sqrt(per_batter.mean())))
    assert n == 6


# --------------------------------------------------------------------------
# Practical magnitude
# --------------------------------------------------------------------------


def test_practical_magnitude_expresses_the_delta_against_the_target_spread() -> None:
    frame = make_frame()
    delta = mt.paired_delta(frame, target="target", reference=REFERENCE, challenger=CHALLENGER)
    described = mt.describe_practical_magnitude(delta, target_sd=float(frame["target"].std(ddof=1)))
    assert described["abs_delta_mae_as_fraction_of_target_sd"] == pytest.approx(
        abs(delta.delta_mae) / float(frame["target"].std(ddof=1))
    )
    assert described["delta_mae"] == delta.delta_mae


# --------------------------------------------------------------------------
# Regression: a constant column's SD is not exactly zero in floating point
# --------------------------------------------------------------------------


def test_a_constant_predictor_is_undefined_even_though_its_sd_is_not_zero() -> None:
    """`np.std` of ~300 identical floats returns ~9e-16, not 0.0.

    Testing constancy as `std == 0` therefore reported `league_mean` as having
    a DEFINED Pearson correlation of about -1e-17 -- a meaningless number that
    does not read as undefined. Constancy is tested as `max == min` instead.
    """
    constant = np.full(280, 3.941521768980823)
    assert np.std(constant) > 0.0  # the exact condition that broke the old test
    assert constant.max() == constant.min()

    target = np.random.default_rng(0).normal(size=280)
    assert mt.pearson(target, constant) == (None, mt.CORRELATION_UNDEFINED_CONSTANT_PREDICTOR)
    assert mt.spearman(target, constant) == (None, mt.CORRELATION_UNDEFINED_CONSTANT_PREDICTOR)


def test_a_predictor_with_any_real_spread_keeps_its_correlation() -> None:
    """The fix must not silently swallow a genuinely tiny but real spread."""
    almost_constant = np.full(200, 4.0)
    almost_constant[0] = 4.0 + 1e-9
    target = np.random.default_rng(1).normal(size=200)
    value, status = mt.pearson(target, almost_constant)
    assert status == mt.CORRELATION_DEFINED
    assert value is not None
