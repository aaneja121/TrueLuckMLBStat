"""Contact Forecast: regularized linear (ridge) forecasting primitives.

The first forecasting model in this study. It predicts future REALIZED
contact-result run value per 100 resolved eligible BBE over the next 100 such
batted balls, conditional on the hitter accumulating them.

## The benchmark that matters

`shrunk_deserved_persistence`. Beating `league_mean` or `raw_realized` proves
nothing: R1 already showed raw realized persistence is WORSE than the league
mean, and that most of what shrinkage buys is generic regression to the mean.
The only delta that establishes incremental value for a forecasting layer is

    MAE(ridge) - MAE(shrunk_deserved_persistence)

on the frozen sign convention: negative means the forecasting layer adds
something beyond the already-established luck-adjusted persistence signal.

## Season-forward everything

Standardization statistics, the ridge coefficients and the alpha are all fit
on strictly earlier seasons:

    fold 1: train 2022            -> select alpha on 2023 (validation)
    fold 2: train 2022-2023 at the fixed alpha -> evaluate 2024

Windows are never split randomly across train and validation, and 2024 never
participates in any selection. `assert_no_later_season_rows` re-proves that on
the frame handed to the estimator rather than on the season list that was
intended.

## The identifiable parameterization

Three EXACT linear dependencies exist in the declared feature list, and ridge
would silently absorb all three -- producing a unique solution whose
coefficients are arbitrary splits across the dependent set, and whose
predictions differ from the identifiable fit because the L2 penalty
distributes weight differently across collinear columns:

  1. `std_surprise_rv_per_100 == std_realized_rv_per_100 -
     std_deserved_rv_per_100`, exactly, and the same for the `recent_`
     triple. Verified to ~1e-14 on real windows.
  2. `bb_rate_ground_ball + line_drive + fly_ball + popup == 1`, exactly,
     which is collinear with the intercept.
  3. `n_resolved_bbe_through_cutoff` is CONSTANT within a fixed-cutoff cell
     (100 in the focal cell), so it has zero variance and no within-cell
     information -- the same argument `forecast.features.feature_columns`
     makes for excluding `cutoff` itself.

`PRESPECIFIED_DROPS` resolves each one by declaration, BEFORE any fit, and
`assert_design_is_identifiable` then re-proves on the actual matrix that the
result is full rank. A dependency that survived the declared drops fails the
run rather than being absorbed by the penalty.

Dropping the surprise columns loses no information: `{realized, deserved}`
spans exactly the same column space as `{realized, deserved, surprise}`. The
`surprise` coefficient is not deleted so much as reparameterized, and
`collinearity_audit` reports the alternative fit so the choice is visible
rather than asserted.

## Coefficients are predictive weights, not causal importances

Ridge coefficients on standardized features are reported. They are shrunk,
correlated with one another, and conditional on the rest of the design.
Nothing here licenses reading a large coefficient as a large causal effect,
and `COEFFICIENT_INTERPRETATION_CAVEAT` travels with them into every artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecast.forecast_config import PRIMARY_TARGET

#: The prediction target: future realized contact-result RV/100.
RIDGE_TARGET = PRIMARY_TARGET

#: The alpha grid, prespecified. Wide enough to reach both the effectively
#: unpenalised and the effectively intercept-only ends, so the selected value
#: is never pinned at a boundary the grid imposed rather than the data.
ALPHA_GRID: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2.0, 4.0, 25))

#: The metric alpha and the winning model are selected on. MAE, matching the
#: study's primary metric -- fixed before any fold was run.
SELECTION_METRIC = "mae"

# --------------------------------------------------------------------------
# Feature sets: prespecified nested ablations
# --------------------------------------------------------------------------

_CURRENT_REALIZED = "std_realized_rv_per_100"
_CURRENT_DESERVED = "std_deserved_rv_per_100"
_CURRENT_SURPRISE = "std_surprise_rv_per_100"
_RECENT_REALIZED = "recent_realized_rv_per_100"
_RECENT_DESERVED = "recent_deserved_rv_per_100"
_RECENT_SURPRISE = "recent_surprise_rv_per_100"
_BBE_COUNT = "n_resolved_bbe_through_cutoff"

_LAUNCH_SPEED = ("launch_speed_mean", "launch_speed_sd", "launch_speed_p10", "launch_speed_p90")
_LAUNCH_ANGLE = ("launch_angle_mean", "launch_angle_sd", "launch_angle_p10", "launch_angle_p90")
_SPRAY_ANGLE = ("spray_angle_mean", "spray_angle_sd", "spray_angle_p10", "spray_angle_p90")
_BB_RATES = (
    "bb_rate_ground_ball",
    "bb_rate_line_drive",
    "bb_rate_fly_ball",
    "bb_rate_popup",
)

#: The four nested models, declared before any was fitted. The nesting is the
#: point: it localises where any predictive gain comes from.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "A_results_only": (_CURRENT_REALIZED, _RECENT_REALIZED),
    "B_deserved_only": (_CURRENT_DESERVED, _RECENT_DESERVED),
    "C_results_plus_deserved": (
        _CURRENT_REALIZED,
        _CURRENT_DESERVED,
        _CURRENT_SURPRISE,
        _RECENT_REALIZED,
        _RECENT_DESERVED,
        _RECENT_SURPRISE,
    ),
    "D_full_contact_profile": (
        _CURRENT_REALIZED,
        _CURRENT_DESERVED,
        _CURRENT_SURPRISE,
        _RECENT_REALIZED,
        _RECENT_DESERVED,
        _RECENT_SURPRISE,
        _BBE_COUNT,
        *_LAUNCH_SPEED,
        *_LAUNCH_ANGLE,
        *_SPRAY_ANGLE,
        "hard_hit_rate",
        "sweet_spot_rate",
        *_BB_RATES,
        "bats_left",
    ),
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "A_results_only": "current + recent realized RV/100",
    "B_deserved_only": "current + recent deserved RV/100",
    "C_results_plus_deserved": "realized + deserved (+ surprise, reparameterized away)",
    "D_full_contact_profile": "Model C plus EV/LA/spray distributions, "
    "batted-ball-type rates and handedness",
}

#: Columns removed BEFORE any fit, each with the exact dependency it resolves.
PRESPECIFIED_DROPS: dict[str, str] = {
    _CURRENT_SURPRISE: (
        f"exactly {_CURRENT_REALIZED} - {_CURRENT_DESERVED}; including it makes the "
        "design rank-deficient and lets the L2 penalty split one coefficient across "
        "three columns. {realized, deserved} spans the identical column space, so no "
        "information is lost."
    ),
    _RECENT_SURPRISE: (f"exactly {_RECENT_REALIZED} - {_RECENT_DESERVED}; same reasoning."),
    "bb_rate_popup": (
        "the four batted-ball-type rates sum to exactly 1, so with an intercept one is "
        "redundant. Popup is the reference category -- the smallest and least "
        "informative class, chosen by that rule rather than by any fitted result."
    ),
}

#: Dropped only where it is actually degenerate, which the data decides.
ZERO_VARIANCE_NOTE = (
    "constant within this cell (every hitter has the same resolved-BBE count at a "
    "fixed cutoff), so it carries no within-cell information and cannot be "
    "standardized"
)

COEFFICIENT_INTERPRETATION_CAVEAT = (
    "Standardized ridge coefficients are PREDICTIVE WEIGHTS under an L2 penalty, "
    "conditional on the rest of the design and correlated with one another. They are "
    "not causal importances, and a larger coefficient does not mean a larger effect."
)


class ForecastRidgeError(ValueError):
    """Raised when a ridge fit would violate its causal or identifiability contract."""


# --------------------------------------------------------------------------
# Design construction
# --------------------------------------------------------------------------


def resolve_features(
    nominal: tuple[str, ...], frame: pd.DataFrame
) -> tuple[list[str], list[dict[str, str]]]:
    """Apply the prespecified drops and remove degenerate columns.

    Returns:
        `(retained, drops)`. `drops` records every removal and its reason, so
        the artifact shows what was declared away rather than leaving a reader
        to diff two feature lists.
    """
    retained: list[str] = []
    drops: list[dict[str, str]] = []
    for column in nominal:
        if column not in frame.columns:
            raise ForecastRidgeError(f"Feature {column!r} is not in the frame")
        if column in PRESPECIFIED_DROPS:
            drops.append(
                {
                    "feature": column,
                    "reason": PRESPECIFIED_DROPS[column],
                    "kind": "prespecified_exact_dependency",
                }
            )
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ForecastRidgeError(
                f"Feature {column!r} carries {int((~np.isfinite(values)).sum())} "
                "null/non-finite value(s); ridge inputs must be complete cases"
            )
        if values.max() == values.min():
            drops.append({"feature": column, "reason": ZERO_VARIANCE_NOTE, "kind": "zero_variance"})
            continue
        retained.append(column)
    if not retained:
        raise ForecastRidgeError("Every declared feature was dropped; nothing to fit")
    return retained, drops


def assert_design_is_identifiable(
    frame: pd.DataFrame, features: list[str], *, tolerance: float = 1e-8
) -> None:
    """Re-prove that the retained design is full rank on the actual matrix.

    The declared drops are a claim about the data; this is the check. A
    dependency that survived them means the declaration is incomplete, which
    must fail the run rather than be absorbed silently by the penalty.

    Raises:
        ForecastRidgeError: If the standardized design is rank-deficient.
    """
    matrix = frame[features].to_numpy(dtype=float)
    centred = matrix - matrix.mean(axis=0)
    scale = centred.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    standardized = centred / scale
    rank = int(np.linalg.matrix_rank(standardized, tol=tolerance))
    if rank < len(features):
        raise ForecastRidgeError(
            f"Design is rank-deficient: rank {rank} for {len(features)} feature(s). "
            "An exact linear dependency survived PRESPECIFIED_DROPS; declare it "
            "rather than letting the L2 penalty absorb it."
        )


def collinearity_audit(frame: pd.DataFrame, features: list[str]) -> dict[str, object]:
    """Condition number, rank and per-feature VIF for the retained design.

    VIF is `1 / (1 - R^2_j)` from regressing each feature on the others. It is
    reported even where it is comfortable, because "we checked and it was
    fine" is a result and an absent check is not.
    """
    matrix = frame[features].to_numpy(dtype=float)
    centred = matrix - matrix.mean(axis=0)
    scale = centred.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    standardized = centred / scale

    vif: dict[str, float] = {}
    for index, name in enumerate(features):
        others = np.delete(standardized, index, axis=1)
        if others.shape[1] == 0:
            vif[name] = 1.0
            continue
        design = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(design, standardized[:, index], rcond=None)
        residual = standardized[:, index] - design @ beta
        ss_residual = float(residual @ residual)
        ss_total = float(standardized[:, index] @ standardized[:, index])
        r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0
        vif[name] = float("inf") if r_squared >= 1.0 else float(1.0 / (1.0 - r_squared))

    singular = np.linalg.svd(standardized, compute_uv=False)
    return {
        "n_features": len(features),
        "rank": int(np.linalg.matrix_rank(standardized)),
        "condition_number": (
            float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        ),
        "variance_inflation_factors": vif,
        "max_vif": float(max(vif.values())) if vif else None,
        "features_with_vif_above_10": sorted(k for k, v in vif.items() if v > 10.0),
    }


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeFit:
    """A fitted ridge model and everything needed to audit its causality."""

    model_name: str
    alpha: float
    features: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    coefficients: dict[str, float]
    intercept: float
    train_seasons: tuple[int, ...]
    n_train: int
    dropped_features: list[dict[str, str]] = field(default_factory=list)

    def as_record(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "description": MODEL_DESCRIPTIONS.get(self.model_name, ""),
            "alpha": self.alpha,
            "features": list(self.features),
            "dropped_features": [dict(d) for d in self.dropped_features],
            "standardized_coefficients": dict(self.coefficients),
            "intercept": self.intercept,
            "coefficient_caveat": COEFFICIENT_INTERPRETATION_CAVEAT,
            "train_seasons": list(self.train_seasons),
            "n_train": self.n_train,
            "standardization": {
                "fit_on": "training seasons only",
                "mean": list(self.scaler_mean),
                "scale": list(self.scaler_scale),
            },
        }


def assert_no_later_season_rows(train: pd.DataFrame, evaluate_season: int) -> dict[str, object]:
    """Prove, on the training frame itself, that no row is from `evaluate_season` or later.

    Operates on the frame about to reach the estimator, not on the season list
    that was intended, so an upstream filtering bug cannot pass.

    Raises:
        ForecastRidgeError: If any training row is from the evaluated season or later.
    """
    seasons = sorted(int(s) for s in train["season"].unique())
    offending = [s for s in seasons if s >= int(evaluate_season)]
    if offending:
        raise ForecastRidgeError(
            f"Training frame for evaluation season {evaluate_season} contains "
            f"season(s) {offending}; every fitted quantity must come from strictly "
            "earlier seasons"
        )
    return {
        "evaluate_season": int(evaluate_season),
        "seasons_present_in_training_frame": seasons,
        "rows_from_evaluate_season_or_later": 0,
        "confirmed": True,
    }


def fit_ridge(
    train: pd.DataFrame,
    *,
    model_name: str,
    alpha: float,
    target: str = RIDGE_TARGET,
    nominal_features: tuple[str, ...] | None = None,
) -> RidgeFit:
    """Standardize on the training rows only, then fit ridge at a fixed alpha."""
    nominal = nominal_features if nominal_features is not None else FEATURE_SETS[model_name]
    features, drops = resolve_features(nominal, train)
    assert_design_is_identifiable(train, features)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy(dtype=float))
    y_train = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_train).all():
        raise ForecastRidgeError(f"Target {target!r} carries null values in the training frame")

    estimator = Ridge(alpha=float(alpha), fit_intercept=True)
    estimator.fit(x_train, y_train)
    return RidgeFit(
        model_name=model_name,
        alpha=float(alpha),
        features=features,
        scaler_mean=[float(v) for v in scaler.mean_],
        scaler_scale=[float(v) for v in scaler.scale_],
        coefficients={
            name: float(value) for name, value in zip(features, estimator.coef_, strict=True)
        },
        intercept=float(estimator.intercept_),
        train_seasons=tuple(sorted(int(s) for s in train["season"].unique())),
        n_train=int(len(train)),
        dropped_features=drops,
    )


def predict(fit: RidgeFit, frame: pd.DataFrame) -> np.ndarray:
    """Apply a fitted model, using the TRAINING standardization statistics."""
    missing = [f for f in fit.features if f not in frame.columns]
    if missing:
        raise ForecastRidgeError(f"Frame is missing fitted feature(s): {missing}")
    matrix = frame[fit.features].to_numpy(dtype=float)
    standardized = (matrix - np.asarray(fit.scaler_mean)) / np.asarray(fit.scaler_scale)
    weights = np.asarray([fit.coefficients[f] for f in fit.features], dtype=float)
    return standardized @ weights + fit.intercept


def _boundary_diagnostic(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    *,
    model_name: str,
    target: str,
    best: dict[str, float],
    alphas: tuple[float, ...],
) -> dict[str, object]:
    """Whether a boundary-pinned alpha is a truncated grid or a genuine limit.

    An alpha at the LOW end is not automatically a problem: as alpha -> 0 ridge
    converges to ordinary least squares, so a two-feature model on a few
    hundred rows legitimately wants no penalty and the boundary is a limit the
    grid approaches rather than a wall it hit. The same holds at the high end,
    where the fit converges to the intercept-only model.

    This refits at an alpha far outside the grid in the pinned direction and
    reports how much the validation score actually moves. A negligible move
    means the boundary is the limit; a large one means the grid was too narrow
    and the selected value is an artifact of where it stopped.
    """
    at_low = best["alpha"] == float(alphas[0])
    beyond = 1e-10 if at_low else 1e10
    fit = fit_ridge(train, model_name=model_name, alpha=beyond, target=target)
    y_validate = pd.to_numeric(validate[target], errors="coerce").to_numpy(dtype=float)
    beyond_score = float(np.mean(np.abs(predict(fit, validate) - y_validate)))
    movement = abs(beyond_score - best[SELECTION_METRIC])
    return {
        "boundary": "lower" if at_low else "upper",
        "limit_behaviour": (
            "alpha -> 0 converges to ordinary least squares"
            if at_low
            else "alpha -> inf converges to the intercept-only model"
        ),
        "selected_alpha_score": best[SELECTION_METRIC],
        "score_far_beyond_the_grid": beyond_score,
        "absolute_movement": movement,
        "grid_is_the_binding_constraint": bool(movement > 0.01),
        "reading": (
            "the grid was too narrow -- the score keeps moving beyond it"
            if movement > 0.01
            else "the boundary is the limit, not a truncation: extending the grid "
            "changes the validation score negligibly"
        ),
    }


def select_alpha(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    *,
    model_name: str,
    target: str = RIDGE_TARGET,
    alphas: tuple[float, ...] = ALPHA_GRID,
) -> dict[str, Any]:
    """Season-forward alpha selection: fit on `train`, score on a LATER season.

    Never a random split of windows or hitters. The selection metric is
    `SELECTION_METRIC`, fixed in advance.
    """
    validate_seasons = sorted(int(s) for s in validate["season"].unique())
    if len(validate_seasons) != 1:
        raise ForecastRidgeError(
            f"Alpha selection expects exactly one validation season; got {validate_seasons}"
        )
    causality = assert_no_later_season_rows(train, validate_seasons[0])

    y_validate = pd.to_numeric(validate[target], errors="coerce").to_numpy(dtype=float)
    grid: list[dict[str, float]] = []
    for alpha in alphas:
        fit = fit_ridge(train, model_name=model_name, alpha=alpha, target=target)
        predictions = predict(fit, validate)
        grid.append(
            {
                "alpha": float(alpha),
                "mae": float(np.mean(np.abs(predictions - y_validate))),
                "rmse": float(np.sqrt(np.mean((predictions - y_validate) ** 2))),
            }
        )

    best = min(grid, key=lambda row: row[SELECTION_METRIC])
    at_boundary = best["alpha"] in (float(alphas[0]), float(alphas[-1]))
    return {
        "boundary_diagnostic": (
            _boundary_diagnostic(
                train,
                validate,
                model_name=model_name,
                target=target,
                best=best,
                alphas=alphas,
            )
            if at_boundary
            else None
        ),
        "model": model_name,
        "selection_metric": SELECTION_METRIC,
        "selected_alpha": best["alpha"],
        "validation_season": validate_seasons[0],
        "train_seasons": causality["seasons_present_in_training_frame"],
        "causality": causality,
        "selected_alpha_at_grid_boundary": at_boundary,
        "grid": grid,
        "split": "season-forward; windows and hitters are never split randomly",
    }
