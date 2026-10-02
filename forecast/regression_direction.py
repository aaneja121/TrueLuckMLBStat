"""Contact Forecast R1: the regression-direction experiment and its placebo.

## The mechanical coupling this experiment must control for

The regression-direction outcome is

    outcome = future_realized - current_realized

and the contact-stage predictor is

    contact_result_surprise = current_realized - current_deserved

The outcome contains `-current_realized`; the predictor contains
`+current_realized`. That shared term alone produces a NEGATIVE association
between them, with no player-specific information in `current_deserved`
whatsoever. "Positive current surprise predicts future decline" is therefore
not evidence of anything on its own -- it is very close to an algebraic
identity plus regression to the mean.

## The placebo

Within each (scored season, cutoff, horizon) cell, `current_deserved` is
PERMUTED across hitters, and

    placebo_surprise = current_realized - permuted_current_deserved

is built. The permutation preserves the `current_realized` term, the marginal
distribution of `current_deserved`, and the cell's sample size exactly; it
destroys only the pairing between a hitter and HIS OWN deserved value. So the
placebo distribution is precisely "what this coefficient looks like when the
contact model carries no player-specific information", and the meaningful
result is not

    "positive current surprise predicts future decline"

but

    "the real contact-stage surprise predicts regression MORE STRONGLY than
     the mechanically induced placebo does."

## The mean-regression baseline

A third model regresses the outcome on `current_realized` alone. That is the
generic "extreme current performance tends to regress" effect with no contact
information at all. The incremental question -- does contact-stage
information add anything beyond it -- is answered by

    outcome ~ current_realized + current_deserved

where, under the null of no player-specific deserved signal, the coefficient
on `current_deserved` is zero and the R-squared gain over the baseline is
zero. Both are compared against the same permutation null.

## Prespecified directions

The alternative hypotheses are fixed BEFORE any coefficient is seen, so that
a p-value is not chosen to match the sign that turned up:

  - `surprise_slope`: real is MORE NEGATIVE than placebo (lower-tail test).
  - `incremental_deserved_slope`: real is MORE POSITIVE than placebo
    (upper-tail test) -- a hitter with better deserved contact, holding
    realized fixed, should decline less.
  - `delta_r_squared`: real is LARGER than placebo (upper-tail test).

Both tail probabilities are reported for every statistic regardless, along
with the placebo distribution's own quantiles, so a reader can see the whole
null rather than one number from it.

## Probabilistic direction prediction

Classification accuracy is not reported as a headline: it is threshold-
dependent and uninformative at an unbalanced base rate. Brier score, log
loss, calibration by probability bin and the event prevalence are reported
instead, and every fitted probability mapping is trained on TEMPORALLY
EARLIER seasons only (`forecast_config.SEASON_FORWARD_FOLDS`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from forecast.forecast_config import RANDOM_SEED, SEASON_FORWARD_FOLDS

#: Column names this experiment reads. `surprise` is not read from the
#: window table but rebuilt from its two components, so the identity
#: `surprise = realized - deserved` is enforced rather than assumed.
CURRENT_REALIZED = "std_realized_rv_per_100"
CURRENT_DESERVED = "std_deserved_rv_per_100"
CURRENT_SURPRISE = "std_surprise_rv_per_100"
FUTURE_REALIZED = "target_realized_rv_per_100"

#: Permutations of the placebo null. Large enough that the reported tail
#: probabilities have a resolution of 1/1001, small enough to stay cheap.
DEFAULT_PERMUTATIONS = 1000

#: Fixed calibration bin edges. Fixed rather than quantile-based so bins are
#: comparable across models, folds and cells -- a quantile binning would make
#: a confident model and a flat one look alike by construction.
CALIBRATION_BIN_EDGES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Tolerance for `surprise == realized - deserved` on per-100 rates.
_IDENTITY_ATOL = 1e-8


class ForecastRegressionDirectionError(ValueError):
    """Raised when the regression-direction inputs violate the design."""


def _least_squares(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """`(coefficients, residual sum of squares)` for an already-built design.

    The numeric core `fit_ols` and the permutation loop share, so the real
    fit and its placebo null are computed by the same arithmetic rather than
    by two implementations that could drift apart.
    """
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return beta, float(residual @ residual)


@dataclass(frozen=True)
class OLSFit:
    """A least-squares fit, with the pieces the placebo comparison needs."""

    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    n: int
    standard_errors: dict[str, float]


def fit_ols(frame: pd.DataFrame, *, outcome: str, predictors: list[str]) -> OLSFit:
    """Ordinary least squares with an intercept.

    Standard errors are the classical homoskedastic ones. Within a single
    (season, cutoff, horizon) cell each hitter contributes exactly one row,
    so there is no within-cell clustering for them to understate; they are
    NOT valid for a frame pooled across seasons or cells, and this module
    never fits one.
    """
    y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(frame))]
        + [pd.to_numeric(frame[p], errors="coerce").to_numpy(dtype=float) for p in predictors]
    )
    if not np.isfinite(y).all() or not np.isfinite(design).all():
        raise ForecastRegressionDirectionError(
            "Regression inputs carry null values; restrict to complete cases first"
        )
    if len(frame) <= design.shape[1]:
        raise ForecastRegressionDirectionError(
            f"{len(frame)} row(s) cannot support a {design.shape[1]}-parameter fit"
        )

    beta, ss_residual = _least_squares(design, y)
    ss_total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")

    degrees_of_freedom = len(frame) - design.shape[1]
    sigma_squared = ss_residual / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    try:
        covariance = sigma_squared * np.linalg.inv(design.T @ design)
        errors = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        errors = np.full(design.shape[1], np.nan)

    return OLSFit(
        coefficients={name: float(beta[i + 1]) for i, name in enumerate(predictors)},
        intercept=float(beta[0]),
        r_squared=float(r_squared),
        n=int(len(frame)),
        standard_errors={name: float(errors[i + 1]) for i, name in enumerate(predictors)},
    )


def assert_surprise_identity(frame: pd.DataFrame) -> None:
    """Prove `surprise == realized - deserved` on the pre-cutoff rates.

    The whole experiment is about a shared `current_realized` term. If the
    surprise column were not exactly that difference, the mechanical-coupling
    argument -- and hence the placebo's construction -- would not apply to it.

    Raises:
        ForecastRegressionDirectionError: If the identity fails.
    """
    if CURRENT_SURPRISE not in frame.columns:
        return
    residual = (
        pd.to_numeric(frame[CURRENT_REALIZED], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(frame[CURRENT_DESERVED], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(frame[CURRENT_SURPRISE], errors="coerce").to_numpy(dtype=float)
    )
    worst = float(np.nanmax(np.abs(residual))) if residual.size else 0.0
    if worst > _IDENTITY_ATOL:
        raise ForecastRegressionDirectionError(
            f"{CURRENT_SURPRISE} is not exactly {CURRENT_REALIZED} - {CURRENT_DESERVED} "
            f"(max absolute residual {worst:.3e}); the mechanical-coupling argument this "
            "experiment controls for would not apply as stated"
        )


def prepare_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Complete cases with the outcome and surprise rebuilt from components."""
    needed = [CURRENT_REALIZED, CURRENT_DESERVED, FUTURE_REALIZED]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ForecastRegressionDirectionError(f"Frame is missing column(s): {missing}")
    assert_surprise_identity(frame)

    cell = frame.dropna(subset=needed).copy()
    cell["outcome_change_rv_per_100"] = cell[FUTURE_REALIZED].to_numpy(dtype=float) - cell[
        CURRENT_REALIZED
    ].to_numpy(dtype=float)
    cell["surprise"] = cell[CURRENT_REALIZED].to_numpy(dtype=float) - cell[
        CURRENT_DESERVED
    ].to_numpy(dtype=float)
    return cell


def _tail_probabilities(real: float, placebo: np.ndarray) -> dict[str, object]:
    """Both one-sided permutation tail probabilities plus the null's shape.

    Uses the `(1 + count) / (1 + n)` estimator, which never reports an
    impossible p-value of exactly zero from a finite permutation set.
    """
    finite = placebo[np.isfinite(placebo)]
    n = int(finite.size)
    if n == 0 or not np.isfinite(real):
        return {
            "n_permutations": n,
            "p_lower_real_more_negative": None,
            "p_upper_real_more_positive": None,
            "placebo_mean": None,
            "placebo_sd": None,
            "placebo_p2_5": None,
            "placebo_p97_5": None,
            "z_vs_placebo": None,
        }
    placebo_sd = float(np.std(finite, ddof=1)) if n > 1 else float("nan")
    return {
        "n_permutations": n,
        "p_lower_real_more_negative": float((1 + int((finite <= real).sum())) / (n + 1)),
        "p_upper_real_more_positive": float((1 + int((finite >= real).sum())) / (n + 1)),
        "placebo_mean": float(np.mean(finite)),
        "placebo_sd": placebo_sd,
        "placebo_p2_5": float(np.percentile(finite, 2.5)),
        "placebo_p97_5": float(np.percentile(finite, 97.5)),
        "z_vs_placebo": (
            float((real - np.mean(finite)) / placebo_sd)
            if np.isfinite(placebo_sd) and placebo_sd > 0
            else None
        ),
    }


def run_regression_direction_cell(
    frame: pd.DataFrame,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """The full regression-direction experiment for ONE season/cutoff/horizon cell.

    Returns a record carrying, in order: the mean-regression baseline (no
    contact information at all), the real surprise regression, the real
    incremental-deserved regression, and the permutation null for both
    contact-stage statistics.
    """
    cell = prepare_cell(frame)
    if len(cell) < 10:
        return {
            "n": int(len(cell)),
            "status": "insufficient_windows",
            "note": "fewer than 10 complete cases; no regression was fit",
        }

    outcome = "outcome_change_rv_per_100"
    baseline = fit_ols(cell, outcome=outcome, predictors=[CURRENT_REALIZED])
    surprise_fit = fit_ols(cell, outcome=outcome, predictors=["surprise"])
    both_fit = fit_ols(cell, outcome=outcome, predictors=[CURRENT_REALIZED, CURRENT_DESERVED])
    real_delta_r2 = float(both_fit.r_squared - baseline.r_squared)

    rng = np.random.default_rng(seed)
    deserved = cell[CURRENT_DESERVED].to_numpy(dtype=float)
    realized = cell[CURRENT_REALIZED].to_numpy(dtype=float)
    y = cell[outcome].to_numpy(dtype=float)
    ones = np.ones(len(cell), dtype=float)
    ss_total = float(((y - y.mean()) ** 2).sum())
    placebo_surprise_slope = np.empty(permutations, dtype=float)
    placebo_deserved_slope = np.empty(permutations, dtype=float)
    placebo_delta_r2 = np.empty(permutations, dtype=float)

    # The permutation loop fits its two models directly in numpy rather than
    # through `fit_ols`. Rebuilding a DataFrame per permutation dominated the
    # runtime and bought nothing; `test_placebo_loop_matches_fit_ols` pins the
    # two paths to identical coefficients on real-shaped input.
    for i in range(permutations):
        permuted = rng.permutation(deserved)
        # The placebo PRESERVES the current-realized term and destroys only
        # the hitter-to-deserved pairing.
        placebo_surprise = realized - permuted
        beta_surprise, _ = _least_squares(np.column_stack([ones, placebo_surprise]), y)
        beta_both, ss_residual_both = _least_squares(np.column_stack([ones, realized, permuted]), y)
        placebo_surprise_slope[i] = beta_surprise[1]
        placebo_deserved_slope[i] = beta_both[2]
        placebo_r2 = 1.0 - ss_residual_both / ss_total if ss_total > 0 else float("nan")
        placebo_delta_r2[i] = placebo_r2 - baseline.r_squared

    return {
        "n": int(len(cell)),
        "status": "fitted",
        "event_definition": ("outcome = future_realized_rv_per_100 - current_realized_rv_per_100"),
        "mechanical_coupling_note": (
            "outcome contains -current_realized and surprise contains +current_realized, "
            "so a negative surprise slope can arise with no player-specific deserved "
            "information at all. The placebo below is the null that quantifies it."
        ),
        "mean_regression_baseline": {
            "model": "outcome ~ current_realized",
            "slope_current_realized": baseline.coefficients[CURRENT_REALIZED],
            "standard_error": baseline.standard_errors[CURRENT_REALIZED],
            "r_squared": baseline.r_squared,
        },
        "real_surprise_regression": {
            "model": "outcome ~ surprise",
            "slope_surprise": surprise_fit.coefficients["surprise"],
            "standard_error": surprise_fit.standard_errors["surprise"],
            "r_squared": surprise_fit.r_squared,
        },
        "real_incremental_regression": {
            "model": "outcome ~ current_realized + current_deserved",
            "slope_current_realized": both_fit.coefficients[CURRENT_REALIZED],
            "slope_current_deserved": both_fit.coefficients[CURRENT_DESERVED],
            "standard_error_current_deserved": both_fit.standard_errors[CURRENT_DESERVED],
            "r_squared": both_fit.r_squared,
            "delta_r_squared_vs_mean_regression_baseline": real_delta_r2,
        },
        "placebo": {
            "construction": (
                "current_deserved permuted across hitters within this cell; "
                "placebo_surprise = current_realized - permuted_current_deserved"
            ),
            "prespecified_directions": {
                "surprise_slope": "real more NEGATIVE than placebo (p_lower)",
                "incremental_deserved_slope": "real more POSITIVE than placebo (p_upper)",
                "delta_r_squared": "real LARGER than placebo (p_upper)",
            },
            "surprise_slope": _tail_probabilities(
                surprise_fit.coefficients["surprise"], placebo_surprise_slope
            ),
            "incremental_deserved_slope": _tail_probabilities(
                both_fit.coefficients[CURRENT_DESERVED], placebo_deserved_slope
            ),
            "delta_r_squared": _tail_probabilities(real_delta_r2, placebo_delta_r2),
        },
    }


# --------------------------------------------------------------------------
# Probabilistic direction prediction
# --------------------------------------------------------------------------


def _calibration_by_bin(y_true: np.ndarray, probabilities: np.ndarray) -> list[dict[str, object]]:
    """Observed frequency versus mean predicted probability, per fixed bin."""
    edges = np.asarray(CALIBRATION_BIN_EDGES, dtype=float)
    # `right=True` on all but the first bin, so 0.0 lands in bin 0 and 1.0 in
    # the last bin rather than falling outside the range.
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: list[dict[str, object]] = []
    for b in range(len(edges) - 1):
        mask = indices == b
        n = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": n,
                "mean_predicted": float(probabilities[mask].mean()) if n else None,
                "observed_rate": float(y_true[mask].mean()) if n else None,
            }
        )
    return rows


def _probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    """Brier, log loss, prevalence and calibration -- never accuracy alone.

    Accuracy at a 0.5 threshold IS reported, but only next to the base-rate
    accuracy a constant predictor would achieve, so it can never be read as
    skill on its own.
    """
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    prevalence = float(y_true.mean())
    majority_accuracy = max(prevalence, 1.0 - prevalence)
    return {
        "n": int(y_true.size),
        "event_prevalence": prevalence,
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "accuracy_at_0_5": float(((clipped >= 0.5).astype(int) == y_true).mean()),
        "majority_class_accuracy": float(majority_accuracy),
        "accuracy_is_insufficient": (
            "Accuracy is reported only beside the majority-class rate; Brier score, "
            "log loss and the calibration table are the evaluation."
        ),
        "calibration_by_bin": _calibration_by_bin(y_true, clipped),
    }


def run_direction_probability(
    windows: pd.DataFrame,
    *,
    folds: tuple[tuple[tuple[int, ...], int], ...] = SEASON_FORWARD_FOLDS,
) -> dict[str, object]:
    """Walk-forward probability of DECLINE, evaluated with proper scoring rules.

    The event is `future_realized < current_realized`. Every probability
    mapping -- including the constant base-rate reference -- is fit on
    strictly earlier seasons and evaluated on a later one, so no fold sees
    its own outcome distribution.

    Returns one record per `(cutoff, horizon, evaluated season)` cell, each
    carrying the three models' Brier score, log loss, calibration table and
    the event's base rate in both the training and evaluation folds.
    """
    prepared = prepare_cell(windows)
    prepared["declined"] = (prepared["outcome_change_rv_per_100"] < 0).astype(int)

    results: dict[str, object] = {}
    for cutoff in sorted(prepared["cutoff"].astype(int).unique()):
        for horizon in sorted(prepared["horizon"].astype(int).unique()):
            cell = prepared[
                (prepared["cutoff"].astype(int) == cutoff)
                & (prepared["horizon"].astype(int) == horizon)
            ]
            for fit_seasons, evaluate_season in folds:
                train = cell[cell["season"].astype(int).isin(fit_seasons)]
                test = cell[cell["season"].astype(int) == int(evaluate_season)]
                key = f"K{cutoff}_H{horizon}_eval{evaluate_season}"
                if len(train) < 30 or len(test) < 30:
                    results[key] = {
                        "status": "insufficient_windows",
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                    }
                    continue
                results[key] = _direction_fold(
                    train,
                    test,
                    cutoff=cutoff,
                    horizon=horizon,
                    fit_seasons=fit_seasons,
                    evaluate_season=int(evaluate_season),
                )
    return {
        "event": "decline: future_realized_rv_per_100 < current_realized_rv_per_100",
        "training_rule": (
            "every probability mapping is fit on strictly earlier seasons only "
            "(forecast_config.SEASON_FORWARD_FOLDS)"
        ),
        "models": {
            "base_rate": "constant = decline prevalence in the training fold",
            "realized_only": "logistic on current_realized",
            "realized_plus_deserved": "logistic on current_realized + current_deserved",
        },
        "cells": results,
    }


def _direction_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cutoff: int,
    horizon: int,
    fit_seasons: tuple[int, ...],
    evaluate_season: int,
) -> dict[str, object]:
    """One walk-forward fold of the decline-probability comparison."""
    later = [s for s in fit_seasons if int(s) >= evaluate_season]
    if later:
        raise ForecastRegressionDirectionError(
            f"Direction-probability fold would fit on season(s) {later}, which are not "
            f"strictly earlier than the evaluated season {evaluate_season}"
        )

    y_train = train["declined"].to_numpy(dtype=int)
    y_test = test["declined"].to_numpy(dtype=int)
    base_rate = float(y_train.mean())

    models: dict[str, object] = {
        "base_rate": _probability_metrics(y_test, np.full(len(test), base_rate)),
    }
    for name, predictors in (
        ("realized_only", [CURRENT_REALIZED]),
        ("realized_plus_deserved", [CURRENT_REALIZED, CURRENT_DESERVED]),
    ):
        # `C=np.inf` gives the unpenalised MLE, so the fit does not depend on
        # the arbitrary scale of a runs-per-100 predictor the way a default L2
        # penalty silently would. (Spelled as an infinite inverse-penalty
        # rather than `penalty=None`, which scikit-learn deprecated in 1.8.)
        estimator = LogisticRegression(C=np.inf, max_iter=1000)
        estimator.fit(train[predictors].to_numpy(dtype=float), y_train)
        probabilities = estimator.predict_proba(test[predictors].to_numpy(dtype=float))[:, 1]
        record = _probability_metrics(y_test, probabilities)
        record["coefficients"] = {
            name: float(value) for name, value in zip(predictors, estimator.coef_[0], strict=True)
        }
        record["intercept"] = float(estimator.intercept_[0])
        models[name] = record

    return {
        "status": "fitted",
        "cutoff": int(cutoff),
        "horizon": int(horizon),
        "fit_seasons": [int(s) for s in fit_seasons],
        "evaluate_season": int(evaluate_season),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_event_prevalence": base_rate,
        "test_event_prevalence": float(y_test.mean()),
        "models": models,
    }
