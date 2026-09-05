"""Contact Forecast: the single nonlinear candidate -- HistGradientBoosting.

Exactly one nonlinear model family is implemented. No random forest, no
XGBoost/LightGBM, no neural network, no ensemble search, no model zoo. The
question this stage answers is narrow:

    does nonlinear modeling extract information from the contact profile that
    the frozen full-profile ridge failed to capture?

## The feature set is inherited, not chosen

`forecast.ridge.FEATURE_SETS["D_full_contact_profile"]` is used verbatim,
passed through the SAME frozen `resolve_features` the ridge stage used. No
feature was added, removed, transformed or interacted on the basis of the
already-observed 2024 ridge results. In particular the large 2024 ridge
coefficients were NOT used to select a subset: those coefficients are
diagnostics, and treating them as a selection procedure would fit the
evaluation season through the back door.

Keeping the resolved list identical to ridge's is what makes this a
comparison of MODEL FAMILIES rather than of feature sets. One consequence
runs AGAINST this stage and is worth stating plainly: `surprise` is dropped
because it is exactly `realized - deserved`, which is a rank argument that
binds for a linear model and not for a tree. An axis-aligned splitter cannot
easily synthesise that difference, so excluding the column withholds
genuinely usable representation from HGB. The choice is therefore
conservative -- it cannot flatter the nonlinear model.

## Two prespecified formulations, one of them primary by necessity

**HGB-Direct** predicts `target_realized_rv_per_100` from the contact-profile
features. It is the PRIMARY formulation.

**HGB-Residual** predicts `target - shrunk_deserved_persistence` and adds the
prediction back to the baseline. It is the prespecified SENSITIVITY analysis.

The designation is forced by data availability, not by any result. The
residual target needs a causally-generated shrunk-deserved value, and 2022 has
none: its only strictly-earlier season is 2021, which has no walk-forward
contact model (frozen R1 limitation `primary_comparison_covers_2023_2024_only`).
So the residual arm cannot use the prescribed "fit 2022, select on 2023"
structure at all -- 2023 is the only training season available to it before
2024. It therefore inherits the hyperparameters selected for HGB-Direct and
has no validation estimate of its own. Both facts are recorded, and the
designation was fixed before either 2024 number was computed.

That inheritance carries a caveat which is NOT 2024 leakage but is still a
contamination: the hyperparameters were chosen by scoring on 2023, and the
residual arm then TRAINS on 2023. Configurations that happened to suit 2023's
noise are mildly favoured. It cannot touch the 2024 evaluation through the
target, and it is reported rather than smoothed over.

## The grid is small on purpose and fixed in advance

564 training windows do not justify aggressive tuning. `HGB_PARAMETER_GRID`
spans only defensible complexity controls, is written to disk BEFORE any
evaluation, and is never expanded after seeing a validation result.
`min_samples_leaf` is pinned rather than searched: below about 20 rows a leaf
on this sample is noise, which is a sample-size argument made in advance, not
a tuning outcome.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forecast.forecast_config import RANDOM_SEED
from forecast.ridge import FEATURE_SETS, ForecastRidgeError, resolve_features

#: The frozen ridge feature definition this stage inherits, unmodified.
HGB_FEATURE_SET_NAME = "D_full_contact_profile"

#: Prediction target of the direct formulation.
HGB_TARGET = "target_realized_rv_per_100"

#: The causal baseline the residual formulation corrects.
RESIDUAL_BASELINE = "shrunk_deserved_persistence"

#: Column name the residual target is materialised under.
RESIDUAL_TARGET = "residual_vs_shrunk_deserved"

#: The complete candidate grid, prespecified. Recorded machine-readably before
#: any evaluation runs. Never expanded after seeing a validation result.
HGB_PARAMETER_GRID: dict[str, tuple[Any, ...]] = {
    "learning_rate": (0.03, 0.1),
    "max_iter": (100, 300),
    "max_leaf_nodes": (7, 15),
    "l2_regularization": (0.0, 1.0),
}

#: Pinned rather than searched -- a sample-size argument, fixed in advance.
HGB_FIXED_PARAMETERS: dict[str, Any] = {
    "min_samples_leaf": 20,
    # Deterministic: an internal early-stopping split would be both random and
    # wasteful on a few hundred rows, and would make the fit depend on a split
    # seed rather than on the specification.
    "early_stopping": False,
    "loss": "squared_error",
    "random_state": RANDOM_SEED,
}

#: The metric candidates are selected on. Fixed in advance, matching the
#: study's primary metric and the ridge stage's selection metric.
HGB_SELECTION_METRIC = "mae"

#: Primary formulation, designated before either 2024 number existed.
PRIMARY_FORMULATION = "direct"
SENSITIVITY_FORMULATION = "residual"


class ForecastHGBError(ValueError):
    """Raised when an HGB fit would violate its causal or specification contract."""


def build_candidate_grid() -> list[dict[str, Any]]:
    """Every prespecified candidate configuration, in a deterministic order."""
    names = sorted(HGB_PARAMETER_GRID)
    combinations = itertools.product(*(HGB_PARAMETER_GRID[name] for name in names))
    return [
        {**HGB_FIXED_PARAMETERS, **dict(zip(names, values, strict=True))} for values in combinations
    ]


def grid_record() -> dict[str, Any]:
    """The machine-readable grid declaration, written before evaluation."""
    candidates = build_candidate_grid()
    return {
        "model_family": "sklearn.ensemble.HistGradientBoostingRegressor",
        "only_nonlinear_family_implemented": True,
        "searched_parameters": {k: list(v) for k, v in HGB_PARAMETER_GRID.items()},
        "fixed_parameters": dict(HGB_FIXED_PARAMETERS),
        "fixed_parameter_rationale": {
            "min_samples_leaf": (
                "pinned at 20: below roughly 20 rows a leaf on this sample is noise. "
                "A sample-size argument made in advance, not a tuning outcome."
            ),
            "early_stopping": (
                "disabled: an internal validation split would be random and wasteful "
                "on a few hundred rows and would make the fit depend on a split seed."
            ),
            "random_state": "fixed, so every fit in this stage is reproducible",
        },
        "n_candidates": len(candidates),
        "candidates": candidates,
        "selection_metric": HGB_SELECTION_METRIC,
        "expansion_rule": "the grid is never expanded after a validation result is seen",
    }


def hgb_features(frame: pd.DataFrame) -> tuple[list[str], list[dict[str, str]]]:
    """The frozen Model D feature list, resolved by the frozen ridge rules."""
    try:
        return resolve_features(FEATURE_SETS[HGB_FEATURE_SET_NAME], frame)
    except ForecastRidgeError as exc:
        raise ForecastHGBError(str(exc)) from exc


def assert_no_later_season_rows(train: pd.DataFrame, evaluate_season: int) -> dict[str, Any]:
    """Prove on the training frame that no row is from `evaluate_season` or later.

    Raises:
        ForecastHGBError: If any training row is from the evaluated season or later.
    """
    seasons = sorted(int(s) for s in train["season"].unique())
    offending = [s for s in seasons if s >= int(evaluate_season)]
    if offending:
        raise ForecastHGBError(
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


def assert_residual_baseline_is_causal(r1_metrics: dict[str, Any]) -> dict[str, Any]:
    """Prove the residual's shrunk-deserved baseline was built causally.

    Reads the FROZEN R1 metrics artifact and checks, per season, that the
    deserved shrinkage estimator saw only strictly-earlier seasons -- the same
    guarantee `forecast.baselines.assert_fit_is_causal` enforced when the
    column was produced, re-verified here from the artifact rather than
    assumed.

    Raises:
        ForecastHGBError: If any season's deserved shrinkage was fit on that
            season or later.
    """
    ladders = r1_metrics["assembly_audit"]["ladders"]
    confirmations: dict[str, Any] = {}
    for season_key, ladder in sorted(ladders.items()):
        season = int(season_key)
        record = ladder.get("deserved_shrinkage")
        if record is None:
            confirmations[season_key] = {
                "available": False,
                "reason": ladder["rung_availability"].get("reason"),
            }
            continue
        fit_seasons = [int(s) for s in record["fit_seasons"]]
        offending = [s for s in fit_seasons if s >= season]
        if offending:
            raise ForecastHGBError(
                f"The residual baseline for {season} was fit on season(s) {offending}, "
                "which are not strictly earlier. The residual target would not be causal."
            )
        confirmations[season_key] = {
            "available": True,
            "fit_seasons": fit_seasons,
            "strictly_earlier": True,
        }
    return {
        "baseline_column": RESIDUAL_BASELINE,
        "verified_from": "frozen r1_metrics.json assembly_audit.ladders",
        "by_season": confirmations,
    }


def build_residual_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `residual = target - shrunk_deserved`, on rows where it exists.

    Rows without a causal baseline (every 2022 window) are DROPPED, not
    imputed: a residual against an absent baseline is not a residual.
    """
    for column in (HGB_TARGET, RESIDUAL_BASELINE):
        if column not in frame.columns:
            raise ForecastHGBError(f"Frame has no {column!r} column")
    usable = frame.dropna(subset=[HGB_TARGET, RESIDUAL_BASELINE]).copy()
    usable[RESIDUAL_TARGET] = usable[HGB_TARGET].to_numpy(dtype=float) - usable[
        RESIDUAL_BASELINE
    ].to_numpy(dtype=float)
    return usable


@dataclass(frozen=True)
class HGBFit:
    """A fitted HGB model and the record needed to audit its causality."""

    formulation: str
    parameters: dict[str, Any]
    features: list[str]
    target_column: str
    train_seasons: tuple[int, ...]
    n_train: int
    model: HistGradientBoostingRegressor
    dropped_features: list[dict[str, str]]

    def as_record(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "parameters": dict(self.parameters),
            "features": list(self.features),
            "n_features": len(self.features),
            "dropped_features": [dict(d) for d in self.dropped_features],
            "target_column": self.target_column,
            "train_seasons": list(self.train_seasons),
            "n_train": self.n_train,
            "preprocessing": (
                "none -- HistGradientBoosting is scale-invariant and handles the raw "
                "feature matrix directly; no standardization is fitted, so there is no "
                "preprocessing statistic that could leak"
            ),
        }


def fit_hgb(
    train: pd.DataFrame,
    *,
    formulation: str,
    parameters: dict[str, Any],
    target_column: str = HGB_TARGET,
) -> HGBFit:
    """Fit one HGB configuration on a training frame."""
    features, drops = hgb_features(train)
    y = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ForecastHGBError(
            f"Target {target_column!r} carries null values in the training frame"
        )
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(train[features].to_numpy(dtype=float), y)
    return HGBFit(
        formulation=formulation,
        parameters=dict(parameters),
        features=features,
        target_column=target_column,
        train_seasons=tuple(sorted(int(s) for s in train["season"].unique())),
        n_train=int(len(train)),
        model=model,
        dropped_features=drops,
    )


def predict(fit: HGBFit, frame: pd.DataFrame) -> np.ndarray:
    """Apply a fitted HGB model to a frame."""
    missing = [f for f in fit.features if f not in frame.columns]
    if missing:
        raise ForecastHGBError(f"Frame is missing fitted feature(s): {missing}")
    return np.asarray(fit.model.predict(frame[fit.features].to_numpy(dtype=float)), dtype=float)


def select_hyperparameters(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    *,
    formulation: str = PRIMARY_FORMULATION,
    target_column: str = HGB_TARGET,
) -> dict[str, Any]:
    """Season-forward selection over the prespecified grid.

    Fits every candidate on `train` and scores it on a strictly LATER season.
    Never a random split of windows or hitters.
    """
    validate_seasons = sorted(int(s) for s in validate["season"].unique())
    if len(validate_seasons) != 1:
        raise ForecastHGBError(
            f"Selection expects exactly one validation season; got {validate_seasons}"
        )
    causality = assert_no_later_season_rows(train, validate_seasons[0])

    y_validate = pd.to_numeric(validate[target_column], errors="coerce").to_numpy(dtype=float)
    scored: list[dict[str, Any]] = []
    for parameters in build_candidate_grid():
        fit = fit_hgb(
            train,
            formulation=formulation,
            parameters=parameters,
            target_column=target_column,
        )
        predictions = predict(fit, validate)
        scored.append(
            {
                "parameters": {k: v for k, v in parameters.items() if k in HGB_PARAMETER_GRID},
                "mae": float(np.mean(np.abs(predictions - y_validate))),
                "rmse": float(np.sqrt(np.mean((predictions - y_validate) ** 2))),
            }
        )

    best = min(scored, key=lambda row: row[HGB_SELECTION_METRIC])
    scores = np.array([row[HGB_SELECTION_METRIC] for row in scored], dtype=float)
    return {
        "formulation": formulation,
        "selection_metric": HGB_SELECTION_METRIC,
        "validation_season": validate_seasons[0],
        "train_seasons": causality["seasons_present_in_training_frame"],
        "causality": causality,
        "selected_parameters": {**HGB_FIXED_PARAMETERS, **best["parameters"]},
        "selected_searched_parameters": best["parameters"],
        "selected_validation_mae": best["mae"],
        "n_candidates": len(scored),
        "grid_scores": scored,
        # How much of the winner's edge could be selection noise: on a few
        # hundred validation windows the spread across a 16-candidate grid is
        # small, and a reader is entitled to see it next to the winner.
        "validation_mae_spread": {
            "best": float(scores.min()),
            "worst": float(scores.max()),
            "median": float(np.median(scores)),
            "range": float(scores.max() - scores.min()),
        },
        "split": "season-forward; windows and hitters are never split randomly",
    }
