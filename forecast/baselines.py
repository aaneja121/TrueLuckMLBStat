"""Contact Forecast R1: the baseline ladder and matched shrinkage.

## Why matched shrinkage is the whole point

Deserved is a less variable quantity than realized by construction -- the
contact model's expectation is smoothed where the observed outcome is not
(measured on real windows: SD 5.54 versus 7.49 at cutoff 50 / horizon 50).
A raw comparison of realized-as-predictor against deserved-as-predictor
therefore rewards deserved partly for being smoother, which is generic
regression to the mean, not a Contact-Luck-specific signal.

The fix is to shrink BOTH through the identical estimator and compare the
shrunk versions. `shrink` is one function; realized and deserved differ only
in which column is passed to it. Any difference that survives is a
difference in the reliability of the two quantities, which is the actual
scientific question.

## The estimator

Standard empirical-Bayes / James-Stein shrinkage toward a prior mean:

    shrunk_i = mu + B_i * (x_i - mu),    B_i = tau^2 / (tau^2 + sigma^2 / n_i)

  - `mu`: league mean rate.
  - `sigma^2`: within-hitter, per-batted-ball variance.
  - `tau^2`: between-hitter variance of true rates, by method of moments,
    `Var(observed hitter rates) - mean(sigma^2 / n_i)`, floored at zero.

`B_i` is the reliability of a hitter's `n_i`-batted-ball sample. A quantity
with more between-hitter signal relative to per-event noise shrinks less.

**Every one of `mu`, `sigma^2`, `tau^2` is estimated on strictly prior
seasons.** `fit_shrinkage` takes only prior-season events and records which
seasons it saw; `assert_fit_is_causal` re-proves that against the season
being predicted.

## Shrinkage cannot change rank WITHIN a cell -- and that is expected

Within one (cutoff, horizon) cell every hitter has exactly `n_i = K`
resolved batted balls, so `B_i` is constant and shrinkage is an affine map.
Affine maps preserve order, so Spearman is IDENTICAL for raw and shrunk
within a cell. Shrinkage changes MAE and RMSE, not rank. The report must not
present a rank improvement from shrinkage, because there cannot be one --
`test_shrinkage_preserves_rank_within_a_cell` pins this.

## Rescaling is a separate, also-causal transform

`fit_linear_rescaling` maps a predictor onto the target's scale by ordinary
least squares fit on PRIOR-SEASON windows only. It exists because raw-unit
MAE penalises a predictor for being on the wrong scale rather than for
carrying the wrong information. Raw-unit results are always reported
alongside; rescaling is never a way to make an unfavorable raw result
disappear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    BASELINE_LADDER,
    DESERVED_COLUMN,
    OBSERVED_COLUMN,
    RATE_SCALE,
)


class ForecastBaselineError(ValueError):
    """Raised when a baseline or shrinkage fit violates its causal contract."""


@dataclass(frozen=True)
class ShrinkageFit:
    """Shrinkage parameters and the seasons they were estimated from.

    `fit_seasons` is carried on the object rather than tracked by the caller
    so `assert_fit_is_causal` can verify causality from the fit itself, and
    so a manifest can record exactly what each number saw.
    """

    quantity: str
    prior_mean: float
    within_variance: float
    between_variance: float
    fit_seasons: tuple[int, ...]
    n_fit_hitter_seasons: int
    n_fit_events: int

    def reliability(self, n: float) -> float:
        """`B = tau^2 / (tau^2 + sigma^2 / n)` for a sample of `n` batted balls."""
        if n <= 0:
            return 0.0
        denominator = self.between_variance + self.within_variance / float(n)
        if denominator <= 0.0:
            # No between-hitter signal detectable: shrink fully to the mean.
            return 0.0
        return float(self.between_variance / denominator)

    def apply(self, rates_per_100: pd.Series, n_events: pd.Series) -> pd.Series:
        """Shrink observed per-100 rates toward the prior mean."""
        reliabilities = np.array(
            [self.reliability(float(n)) for n in n_events.to_numpy()], dtype=float
        )
        return pd.Series(
            self.prior_mean + reliabilities * (rates_per_100.to_numpy() - self.prior_mean),
            index=rates_per_100.index,
        )


def fit_shrinkage(
    prior_events: pd.DataFrame,
    *,
    value_column: str,
    quantity: str,
    min_events_per_hitter: int = 50,
) -> ShrinkageFit:
    """Estimate shrinkage parameters from PRIOR-SEASON batted balls only.

    The identical function fits realized and deserved -- only `value_column`
    changes. That is what makes the two ladders' shrinkage matched rather
    than merely similar.

    Args:
        prior_events: Per-batted-ball rows from strictly earlier seasons,
            already restricted to resolved batted balls.
        value_column: The per-batted-ball run-value column to fit on.
        quantity: A label recorded on the fit ("realized" / "deserved").
        min_events_per_hitter: Hitter-seasons below this contribute to the
            within-hitter variance but not to the between-hitter estimate,
            where a handful of batted balls would otherwise inflate the
            observed spread of rates.

    Raises:
        ForecastBaselineError: If `prior_events` is empty or lacks the column.
    """
    if prior_events.empty:
        raise ForecastBaselineError(f"No prior-season events to fit {quantity} shrinkage on")
    if value_column not in prior_events.columns:
        raise ForecastBaselineError(f"Prior events lack value column {value_column!r}")

    per_event = pd.to_numeric(prior_events[value_column], errors="coerce")
    if per_event.isna().any():
        raise ForecastBaselineError(
            f"{value_column!r} contains nulls; resolve or drop them before fitting"
        )

    # mu on the per-100 scale, so it is directly comparable with the rates
    # it will shrink.
    prior_mean = float(RATE_SCALE * per_event.mean())
    # sigma^2 of a per-100 RATE built from n events is RATE_SCALE^2 * Var(per event) / n;
    # store the per-event piece scaled once here so `reliability` only divides by n.
    within_variance = float((RATE_SCALE**2) * per_event.var(ddof=1))

    grouped = prior_events.assign(_value=per_event).groupby(["batter", "season"])["_value"]
    hitter_rates = RATE_SCALE * grouped.mean()
    hitter_counts = grouped.size()

    usable = hitter_counts >= int(min_events_per_hitter)
    if int(usable.sum()) < 2:
        raise ForecastBaselineError(
            f"Fewer than two hitter-seasons with >= {min_events_per_hitter} batted balls; "
            "cannot estimate between-hitter variance"
        )

    observed_variance = float(hitter_rates[usable].var(ddof=1))
    expected_noise = float((within_variance / hitter_counts[usable]).mean())
    # Method of moments. Floored at zero: a negative estimate means the
    # observed spread is fully explained by sampling noise, i.e. no
    # detectable between-hitter signal -- which is a real finding, reported
    # as zero reliability rather than as a negative variance.
    between_variance = max(0.0, observed_variance - expected_noise)

    return ShrinkageFit(
        quantity=quantity,
        prior_mean=prior_mean,
        within_variance=within_variance,
        between_variance=between_variance,
        fit_seasons=tuple(sorted(int(s) for s in prior_events["season"].unique())),
        n_fit_hitter_seasons=int(usable.sum()),
        n_fit_events=int(len(prior_events)),
    )


def assert_fit_is_causal(fit: ShrinkageFit, predicted_season: int) -> None:
    """Re-prove that a fit saw only seasons strictly earlier than the target.

    Raises:
        ForecastBaselineError: If the fit saw the predicted season or later.
    """
    offending = [s for s in fit.fit_seasons if s >= int(predicted_season)]
    if offending:
        raise ForecastBaselineError(
            f"{fit.quantity} shrinkage was fit on season(s) {offending}, which are not "
            f"strictly earlier than the predicted season {predicted_season}"
        )


@dataclass(frozen=True)
class LinearRescaling:
    """An intercept/slope map from a predictor onto a target's scale."""

    predictor: str
    intercept: float
    slope: float
    fit_seasons: tuple[int, ...]
    n_fit_windows: int

    @property
    def preserves_rank(self) -> bool:
        """Whether applying this map leaves rank order intact.

        A positive slope is order-preserving, so Spearman is unchanged. A
        NEGATIVE slope inverts the order -- and means the predictor ran the
        wrong way on the prior seasons it was fit on, i.e. it was
        anti-predictive there. That is a substantive finding about the
        predictor, never something to apply quietly, which is why
        `build_rescaling_record` surfaces it and the report must state it.
        """
        return self.slope > 0.0

    def apply(self, values: pd.Series) -> pd.Series:
        return pd.Series(self.intercept + self.slope * values.to_numpy(), index=values.index)


def fit_linear_rescaling(
    prior_windows: pd.DataFrame, *, predictor: str, target: str
) -> LinearRescaling:
    """Least-squares map of `predictor` onto `target`, fit on prior seasons only.

    Raises:
        ForecastBaselineError: If there are too few prior windows, or the
            predictor has no variance to fit a slope against.
    """
    if predictor not in prior_windows.columns or target not in prior_windows.columns:
        raise ForecastBaselineError(f"Prior windows lack {predictor!r} or {target!r}")

    usable = prior_windows[[predictor, target, "season"]].dropna(subset=[predictor, target])
    if len(usable) < 3:
        raise ForecastBaselineError(
            f"Only {len(usable)} prior window(s) available to fit the {predictor!r} rescaling"
        )

    x = usable[predictor].to_numpy(dtype=float)
    y = usable[target].to_numpy(dtype=float)
    if float(np.var(x)) <= 0.0:
        raise ForecastBaselineError(f"{predictor!r} has zero variance in the prior windows")

    slope, intercept = np.polyfit(x, y, deg=1)
    return LinearRescaling(
        predictor=predictor,
        intercept=float(intercept),
        slope=float(slope),
        fit_seasons=tuple(sorted(int(s) for s in usable["season"].unique())),
        n_fit_windows=int(len(usable)),
    )


def assert_rescaling_is_causal(rescaling: LinearRescaling, predicted_season: int) -> None:
    """Re-prove that a rescaling saw only strictly earlier seasons."""
    offending = [s for s in rescaling.fit_seasons if s >= int(predicted_season)]
    if offending:
        raise ForecastBaselineError(
            f"{rescaling.predictor} rescaling was fit on season(s) {offending}, which are "
            f"not strictly earlier than the predicted season {predicted_season}"
        )


def build_baseline_predictions(
    windows: pd.DataFrame,
    *,
    prior_events: pd.DataFrame,
    predicted_season: int,
) -> pd.DataFrame:
    """Every rung of the frozen baseline ladder, for one season's windows.

    The ladder, in report order:

      1. `league_mean` -- the prior-season league rate, identical for
         everyone. The floor any predictor must clear.
      2. `raw_realized_persistence` -- season-to-date realized rate, unshrunk.
      3. `raw_deserved_persistence` -- season-to-date deserved rate, unshrunk.
      4. `shrunk_realized_persistence` -- rung 2 through the matched estimator.
      5. `shrunk_deserved_persistence` -- rung 3 through the SAME estimator.

    Rungs 4 and 5 are the frozen primary comparison; 2 and 3 are kept so a
    reader can see how much of any difference is shrinkage rather than
    signal.

    Args:
        windows: Windows for `predicted_season`, already carrying the
            `std_realized_rv_per_100` / `std_deserved_rv_per_100` features
            and `n_resolved_bbe_through_cutoff`.
        prior_events: Per-batted-ball rows from strictly earlier seasons.
        predicted_season: The season being predicted.

    Returns:
        `windows` with one column per ladder rung added.
    """
    if windows.empty:
        return windows.copy()

    seasons = {int(s) for s in windows["season"].unique()}
    if seasons != {int(predicted_season)}:
        raise ForecastBaselineError(
            f"Windows span seasons {sorted(seasons)} but predicted_season is "
            f"{predicted_season}; fit a separate ladder per season"
        )

    realized_fit = fit_shrinkage(prior_events, value_column=OBSERVED_COLUMN, quantity="realized")
    deserved_fit = fit_shrinkage(prior_events, value_column=DESERVED_COLUMN, quantity="deserved")
    assert_fit_is_causal(realized_fit, predicted_season)
    assert_fit_is_causal(deserved_fit, predicted_season)

    out = windows.copy()
    n_events = out["n_resolved_bbe_through_cutoff"]

    out["league_mean"] = realized_fit.prior_mean
    out["raw_realized_persistence"] = out["std_realized_rv_per_100"]
    out["raw_deserved_persistence"] = out["std_deserved_rv_per_100"]
    out["shrunk_realized_persistence"] = realized_fit.apply(
        out["std_realized_rv_per_100"], n_events
    )
    out["shrunk_deserved_persistence"] = deserved_fit.apply(
        out["std_deserved_rv_per_100"], n_events
    )

    missing = [rung for rung in BASELINE_LADDER if rung not in out.columns]
    if missing:
        raise ForecastBaselineError(f"Baseline ladder is incomplete: missing {missing}")
    return out


def build_rescaling_record(rescaling: LinearRescaling) -> dict[str, object]:
    """A manifest-ready record of a rescaling, flagging an inverted fit.

    `slope_is_negative` is reported explicitly rather than buried in the
    coefficient: a negative slope means the predictor was anti-predictive on
    the prior seasons, which changes how every downstream number should be
    read and must never be discovered only by noticing a sign.
    """
    return {
        "predictor": rescaling.predictor,
        "intercept": rescaling.intercept,
        "slope": rescaling.slope,
        "slope_is_negative": not rescaling.preserves_rank,
        "preserves_rank": rescaling.preserves_rank,
        "fit_seasons": list(rescaling.fit_seasons),
        "n_fit_windows": rescaling.n_fit_windows,
    }


def shrinkage_fit_record(fit: ShrinkageFit) -> dict[str, object]:
    """A manifest-ready record of a shrinkage fit, including what it saw."""
    return {
        "quantity": fit.quantity,
        "prior_mean_per_100": fit.prior_mean,
        "within_hitter_variance": fit.within_variance,
        "between_hitter_variance": fit.between_variance,
        "fit_seasons": list(fit.fit_seasons),
        "n_fit_hitter_seasons": fit.n_fit_hitter_seasons,
        "n_fit_events": fit.n_fit_events,
        "reliability_at_50_bbe": fit.reliability(50),
        "reliability_at_100_bbe": fit.reliability(100),
        "reliability_at_150_bbe": fit.reliability(150),
        "reliability_at_200_bbe": fit.reliability(200),
    }
