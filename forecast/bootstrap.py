"""Contact Forecast R1: the paired, globally batter-clustered bootstrap.

## Why the cluster is the BATTER, globally

The window table repeats hitters twice over. One hitter-season supplies a
window at every `(cutoff, horizon)` cell it qualifies for, and those windows
share the same history events; and the same hitter recurs across 2022, 2023
and 2024. Treating windows -- or even batter-seasons -- as independent draws
would understate the sampling variance of every interval in this study.

So the resampling unit is `forecast_config.BOOTSTRAP_CLUSTER_COLUMN`
(`batter`), and it is drawn ONCE per replicate for the whole study: when a
batter is resampled, all of his seasons, all of his cutoffs, all of his
horizons and both sides of every paired prediction travel together. Season-
level metrics and the macro-average are then recomputed INSIDE the
replicate, from the resampled rows, rather than being averaged across
replicates afterwards.

## Paired, not two independent bootstraps

Both predictors are carried on the same rows and differenced inside each
replicate. The interval is therefore an interval on the DIFFERENCE, which is
what a comparison needs: two separately-bootstrapped intervals that happen to
overlap say nothing about whether the difference is distinguishable from
zero, because the two predictors' errors are strongly positively correlated
across the same hitters.

## The multiplicity identity (why there is no row-index loop)

Resampling clusters with replacement and summing over the resampled rows is
algebraically identical to a multiplicity-weighted sum over the ORIGINAL
rows:

    sum over resample of f(row) = sum over clusters m_c * sum over rows in c of f(row)

where `m_c` is how many times cluster `c` was drawn. This module uses that
identity -- one multinomial draw and a handful of `np.bincount` calls per
replicate -- because building an explicit row index per replicate is
needlessly slow. `_replicate_row_index` implements the literal, obvious
construction, and `test_multiplicity_weights_match_an_explicit_resample`
proves the two agree exactly, so the optimisation is checked rather than
asserted.

## What is bootstrapped, and what is not

MAE and RMSE, and the paired deltas built from them. Pearson and Spearman
are NOT bootstrapped here: they are undefined for a constant predictor
(`forecast.metrics`), so a replicate distribution over them would be a
distribution over a partly-undefined quantity. They are reported as point
estimates with their definedness status instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    BOOTSTRAP_CLUSTER_COLUMN,
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
)
from forecast.metrics import DELTA_SIGN_CONVENTION, _direction


class ForecastBootstrapError(ValueError):
    """Raised when a bootstrap input violates the paired-cluster contract."""


@dataclass(frozen=True)
class BootstrapDesign:
    """Everything about the resampling scheme, recorded in every artifact."""

    cluster_column: str = BOOTSTRAP_CLUSTER_COLUMN
    reps: int = DEFAULT_BOOTSTRAP_REPS
    seed: int = DEFAULT_BOOTSTRAP_SEED
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA

    def as_record(self) -> dict[str, object]:
        return {
            "cluster_column": self.cluster_column,
            "clustering": (
                "global: one draw of batters per replicate, shared across every "
                "season, cutoff, horizon and both sides of every paired prediction"
            ),
            "paired": True,
            "reps": self.reps,
            "seed": self.seed,
            "alpha": self.alpha,
            "interval": f"{100 * (1 - self.alpha):.0f}% percentile interval",
            "recomputed_inside_each_replicate": [
                "season-level MAE and RMSE",
                "pooled MAE and RMSE",
                "macro-average of the season-specific paired differences",
            ],
        }


def _replicate_row_index(cluster_rows: list[np.ndarray], drawn_clusters: np.ndarray) -> np.ndarray:
    """The literal resampled row index: every row of every drawn cluster.

    The obvious, slow construction. Used by the tests to prove the
    multiplicity-weighted fast path is exactly equivalent; not used in the
    hot loop.
    """
    return np.concatenate([cluster_rows[c] for c in drawn_clusters])


def _percentile_interval(values: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    """A two-sided percentile interval, or `(None, None)` if nothing is finite."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    lower = float(np.percentile(finite, 100.0 * alpha / 2.0))
    upper = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _summarize(values: np.ndarray, *, point: float, alpha: float) -> dict[str, object]:
    """A point estimate plus its replicate distribution, on the frozen convention.

    `share_replicates_favouring_reference` is the share of replicates in
    which the delta was >= 0, i.e. in which the challenger did NOT come out
    ahead. Reported alongside the interval because a reader deciding whether
    an advantage is robust wants the mass, not only the endpoints.
    """
    finite = values[np.isfinite(values)]
    lower, upper = _percentile_interval(values, alpha)
    crosses_zero = None if lower is None or upper is None else bool(lower <= 0.0 <= upper)
    return {
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_crosses_zero": crosses_zero,
        "bootstrap_mean": float(np.mean(finite)) if finite.size else None,
        "bootstrap_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "share_replicates_favouring_reference": (
            float(np.mean(finite >= 0.0)) if finite.size else None
        ),
        "n_valid_replicates": int(finite.size),
        "direction": _direction(point),
    }


@dataclass(frozen=True)
class _PreparedFrame:
    """Row-level arrays a replicate loop needs, built once."""

    cluster_codes: np.ndarray
    n_clusters: int
    group_codes: np.ndarray
    group_keys: list[tuple[int, int, int]]
    abs_reference: np.ndarray
    sq_reference: np.ndarray
    abs_challenger: np.ndarray
    sq_challenger: np.ndarray
    cluster_rows: list[np.ndarray]


def _prepare(
    frame: pd.DataFrame,
    *,
    target: str,
    reference: str,
    challenger: str,
    cluster_column: str,
) -> _PreparedFrame:
    """Validate the paired frame and precompute the per-row error arrays."""
    required = [
        target,
        reference,
        challenger,
        cluster_column,
        "season",
        "cutoff",
        "horizon",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ForecastBootstrapError(f"Paired frame is missing column(s): {missing}")
    if frame.empty:
        raise ForecastBootstrapError("Paired frame is empty")

    values = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        for name in (target, reference, challenger)
    }
    for name, array in values.items():
        if not np.isfinite(array).all():
            raise ForecastBootstrapError(
                f"{name!r} carries {int((~np.isfinite(array)).sum())} null value(s). "
                "Restrict to paired complete cases before bootstrapping -- a row "
                "present for one predictor and absent for the other is not paired"
            )

    clusters, cluster_codes = np.unique(frame[cluster_column].to_numpy(), return_inverse=True)
    group_frame = frame[["season", "cutoff", "horizon"]].astype(int)
    group_tuples = list(map(tuple, group_frame.to_numpy()))
    unique_groups = sorted(set(group_tuples))
    group_lookup = {key: i for i, key in enumerate(unique_groups)}
    group_codes = np.array([group_lookup[key] for key in group_tuples], dtype=np.intp)

    cluster_rows = [np.flatnonzero(cluster_codes == code) for code in range(len(clusters))]
    return _PreparedFrame(
        cluster_codes=cluster_codes.astype(np.intp),
        n_clusters=int(len(clusters)),
        group_codes=group_codes,
        group_keys=[(int(a), int(b), int(c)) for a, b, c in unique_groups],
        abs_reference=np.abs(values[reference] - values[target]),
        sq_reference=(values[reference] - values[target]) ** 2,
        abs_challenger=np.abs(values[challenger] - values[target]),
        sq_challenger=(values[challenger] - values[target]) ** 2,
        cluster_rows=cluster_rows,
    )


def _replicate_group_sums(prepared: _PreparedFrame, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Multiplicity-weighted per-(season, cutoff, horizon) sums for one replicate."""
    row_weights = weights[prepared.cluster_codes]
    n_groups = len(prepared.group_keys)
    return {
        "count": np.bincount(prepared.group_codes, weights=row_weights, minlength=n_groups),
        "abs_reference": np.bincount(
            prepared.group_codes,
            weights=row_weights * prepared.abs_reference,
            minlength=n_groups,
        ),
        "sq_reference": np.bincount(
            prepared.group_codes,
            weights=row_weights * prepared.sq_reference,
            minlength=n_groups,
        ),
        "abs_challenger": np.bincount(
            prepared.group_codes,
            weights=row_weights * prepared.abs_challenger,
            minlength=n_groups,
        ),
        "sq_challenger": np.bincount(
            prepared.group_codes,
            weights=row_weights * prepared.sq_challenger,
            minlength=n_groups,
        ),
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division, emitting NaN where the denominator is zero.

    A replicate can, in principle, draw no batter belonging to some
    (season, cutoff, horizon) group. That group's metric is genuinely
    undefined in that replicate and is carried as NaN, then excluded from the
    percentile -- never silently treated as zero error.
    """
    out = np.full(numerator.shape, np.nan, dtype=float)
    nonzero = denominator > 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def run_paired_bootstrap(
    frame: pd.DataFrame,
    *,
    target: str,
    reference: str,
    challenger: str,
    design: BootstrapDesign | None = None,
) -> dict[str, object]:
    """Paired, globally batter-clustered percentile intervals for every cell.

    Args:
        frame: Paired complete cases spanning every season, cutoff and
            horizon the comparison covers. Carries `batter`, `season`,
            `cutoff`, `horizon`, the target and both predictors.
        target: The target column.
        reference: The predictor the delta is measured AGAINST.
        challenger: The predictor whose advantage (or lack of one) is being
            measured. `delta = challenger - reference`, so negative favours
            the challenger.
        design: Resampling design; defaults to the frozen configuration.

    Returns:
        A JSON-ready record carrying the design, the sign convention, and one
        entry per `(cutoff, horizon)` cell with pooled, macro-average and
        per-season intervals for `delta_mae` and `delta_rmse`.
    """
    design = design or BootstrapDesign()
    prepared = _prepare(
        frame,
        target=target,
        reference=reference,
        challenger=challenger,
        cluster_column=design.cluster_column,
    )
    n_groups = len(prepared.group_keys)
    rng = np.random.default_rng(design.seed)
    probabilities = np.full(prepared.n_clusters, 1.0 / prepared.n_clusters)

    counts = np.empty((design.reps, n_groups), dtype=float)
    sums = {
        key: np.empty((design.reps, n_groups), dtype=float)
        for key in ("abs_reference", "sq_reference", "abs_challenger", "sq_challenger")
    }
    for rep in range(design.reps):
        multiplicities = rng.multinomial(prepared.n_clusters, probabilities).astype(float)
        replicate = _replicate_group_sums(prepared, multiplicities)
        counts[rep] = replicate["count"]
        for key in sums:
            sums[key][rep] = replicate[key]

    observed_weights = np.ones(prepared.n_clusters, dtype=float)
    observed = _replicate_group_sums(prepared, observed_weights)

    cells: dict[str, object] = {}
    for cutoff, horizon in sorted(
        {(k, h) for _, k, h in prepared.group_keys}, key=lambda pair: (pair[0], pair[1])
    ):
        member_indices = [
            i for i, (_, k, h) in enumerate(prepared.group_keys) if k == cutoff and h == horizon
        ]
        seasons = [prepared.group_keys[i][0] for i in member_indices]
        cells[f"K{cutoff}_H{horizon}"] = {
            "cutoff": int(cutoff),
            "horizon": int(horizon),
            "seasons": seasons,
            "n_windows": int(observed["count"][member_indices].sum()),
            "pooled": _cell_pooled(counts, sums, observed, member_indices, alpha=design.alpha),
            "macro_average_of_season_deltas": _cell_macro(
                counts, sums, observed, member_indices, alpha=design.alpha
            ),
            "by_season": {
                str(prepared.group_keys[i][0]): _cell_pooled(
                    counts, sums, observed, [i], alpha=design.alpha
                )
                for i in member_indices
            },
        }

    return {
        "design": design.as_record(),
        "sign_convention": DELTA_SIGN_CONVENTION,
        "reference": reference,
        "challenger": challenger,
        "target": target,
        "n_clusters_resampled": prepared.n_clusters,
        "n_paired_windows": int(len(frame)),
        "cells": cells,
    }


def _deltas_from_sums(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled `delta_mae` and `delta_rmse` over `indices`, per replicate.

    Pooling is over the RAW SUMS, so a season with more windows contributes
    proportionally -- which is exactly what "pooled" means and exactly why the
    macro-average is reported next to it.
    """
    pooled_count = counts[..., indices].sum(axis=-1)
    mae_reference = _safe_divide(sums["abs_reference"][..., indices].sum(axis=-1), pooled_count)
    mae_challenger = _safe_divide(sums["abs_challenger"][..., indices].sum(axis=-1), pooled_count)
    rmse_reference = np.sqrt(
        _safe_divide(sums["sq_reference"][..., indices].sum(axis=-1), pooled_count)
    )
    rmse_challenger = np.sqrt(
        _safe_divide(sums["sq_challenger"][..., indices].sum(axis=-1), pooled_count)
    )
    return mae_challenger - mae_reference, rmse_challenger - rmse_reference


def _cell_pooled(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    indices: list[int],
    *,
    alpha: float,
) -> dict[str, object]:
    """Window-weighted pooled deltas with their bootstrap intervals."""
    replicate_mae, replicate_rmse = _deltas_from_sums(counts, sums, indices)
    point_mae, point_rmse = _deltas_from_sums(
        observed["count"][np.newaxis, :],
        {key: observed[key][np.newaxis, :] for key in sums},
        indices,
    )
    return {
        "weighting": "window-weighted (pooled sums)",
        "delta_mae": _summarize(replicate_mae, point=float(point_mae[0]), alpha=alpha),
        "delta_rmse": _summarize(replicate_rmse, point=float(point_rmse[0]), alpha=alpha),
    }


def _cell_macro(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    indices: list[int],
    *,
    alpha: float,
) -> dict[str, object]:
    """Macro-average of the SEASON-specific deltas, recomputed per replicate.

    Each season's delta is computed from that season's own resampled rows,
    then the seasons are averaged with equal weight -- inside the replicate,
    so the interval propagates the resampling through the macro-average
    rather than around it.
    """
    per_season_mae = []
    per_season_rmse = []
    point_season_mae = []
    point_season_rmse = []
    for index in indices:
        replicate_mae, replicate_rmse = _deltas_from_sums(counts, sums, [index])
        per_season_mae.append(replicate_mae)
        per_season_rmse.append(replicate_rmse)
        point_mae, point_rmse = _deltas_from_sums(
            observed["count"][np.newaxis, :],
            {key: observed[key][np.newaxis, :] for key in sums},
            [index],
        )
        point_season_mae.append(float(point_mae[0]))
        point_season_rmse.append(float(point_rmse[0]))

    macro_mae = np.mean(np.vstack(per_season_mae), axis=0)
    macro_rmse = np.mean(np.vstack(per_season_rmse), axis=0)
    return {
        "weighting": "equal weight per season",
        "n_seasons": len(indices),
        "delta_mae": _summarize(macro_mae, point=float(np.mean(point_season_mae)), alpha=alpha),
        "delta_rmse": _summarize(macro_rmse, point=float(np.mean(point_season_rmse)), alpha=alpha),
    }
