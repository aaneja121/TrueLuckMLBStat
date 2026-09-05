"""Contact Forecast R1: evaluation metrics and the FROZEN sign convention.

This module owns three things and nothing else: how a predictor's error is
measured, how two predictors' errors are differenced, and what the sign of
that difference means. No model, no bootstrap, no report text.

## The sign convention is frozen here and nowhere else

For every paired comparison in this study:

    delta_metric = metric(challenger) - metric(reference)

with the primary comparison's challenger and reference fixed by
`forecast_config.PRIMARY_COMPARISON`:

    delta_MAE  = MAE(shrunk_deserved_persistence)  - MAE(shrunk_realized_persistence)
    delta_RMSE = RMSE(shrunk_deserved_persistence) - RMSE(shrunk_realized_persistence)

Therefore, for MAE and RMSE (both lower-is-better):

    negative -> DESERVED is better
    zero     -> no difference
    positive -> REALIZED is better

`DELTA_SIGN_CONVENTION` carries that statement in machine-readable form and
is written verbatim into every artifact, so a reader never has to infer it
from a variable name.

**Report code may not reverse a sign for presentation.** Every `PairedDelta`
carries the two component metrics it was built from, and
`assert_delta_convention` re-derives the delta from them. A presentation
layer that flipped a sign would have to flip the components too, which the
check catches. `test_report_cannot_reverse_the_sign_convention` pins it.

## Percentage improvement stays on the same convention

`pct_change_mae = 100 * delta_mae / mae_reference` deliberately shares the
delta's sign: -3.0 means the challenger reduces MAE by 3.0%. A separate
"percent improvement" quantity with the OPPOSITE sign is exactly the
presentational sign-flip this module exists to prevent, so it is not
provided.

## Undefined correlations are undefined, never zero

A constant predictor -- `league_mean` is constant by construction -- has no
Pearson or Spearman correlation with anything: the denominator is zero. That
is recorded as `None` with an explanatory status string, NEVER as 0.0.
Coercing it to zero would silently convert "this metric does not apply" into
"this predictor has no rank skill", which is a different and false claim.
MAE and RMSE remain perfectly well defined for such a predictor and are
reported normally.

## Paired means paired

Every function here refuses NaN in the columns it reads. Silently dropping a
row from one predictor but not the other would unpair the comparison, which
is the failure mode that makes a paired bootstrap meaningless. Callers
restrict to complete cases ONCE, explicitly, and pass the result down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.forecast_config import BOOTSTRAP_CLUSTER_COLUMN, PRIMARY_COMPARISON


class ForecastMetricError(ValueError):
    """Raised when a metric input violates the paired-complete-case contract."""


#: The frozen sign convention, written verbatim into every artifact.
DELTA_SIGN_CONVENTION: dict[str, str] = {
    "definition": "delta_metric = metric(challenger) - metric(reference)",
    "reference": PRIMARY_COMPARISON[0],
    "challenger": PRIMARY_COMPARISON[1],
    "delta_mae": ("MAE(shrunk_deserved_persistence) - MAE(shrunk_realized_persistence)"),
    "delta_rmse": ("RMSE(shrunk_deserved_persistence) - RMSE(shrunk_realized_persistence)"),
    "negative": "deserved is better (lower error)",
    "zero": "no difference",
    "positive": "realized is better (lower error)",
    "percentage": (
        "pct_change_mae = 100 * delta_mae / mae_reference, on the SAME sign "
        "convention: -3.0 means deserved reduces MAE by 3.0%"
    ),
    "presentation_rule": (
        "Report code must never reverse these signs. assert_delta_convention "
        "re-derives every delta from its stored component metrics."
    ),
}

#: Direction labels for a delta on a lower-is-better metric.
CHALLENGER_BETTER = "challenger_better"
REFERENCE_BETTER = "reference_better"
NO_DIFFERENCE = "no_difference"

#: Correlation status strings. A correlation is either a number with status
#: `defined`, or `None` with one of the others -- never a number with an
#: "undefined" status, and never 0.0 standing in for one of these.
CORRELATION_DEFINED = "defined"
CORRELATION_UNDEFINED_CONSTANT_PREDICTOR = "undefined_constant_predictor"
CORRELATION_UNDEFINED_CONSTANT_TARGET = "undefined_constant_target"
CORRELATION_UNDEFINED_INSUFFICIENT_SAMPLE = "undefined_insufficient_sample"


def _is_constant(values: np.ndarray) -> bool:
    """Whether every element is bit-identical, tested as `max == min`.

    Deliberately NOT `np.std(values) == 0`. Summation rounding inside
    `np.mean` over a few hundred identical floats leaves a residual of order
    1e-16, so a genuinely constant column reports a SD of ~9e-16 and its
    correlation comes back as a "defined" -1e-17 -- a number that means
    nothing but does not read as undefined. `max == min` has no such failure
    mode, and a predictor with any real spread, however small, still counts
    as non-constant and keeps its (unstable) correlation.
    """
    return bool(values.size and values.max() == values.min())


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A float view of `name`, refusing missing columns and NaN."""
    if name not in frame.columns:
        raise ForecastMetricError(f"Frame has no column {name!r}")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ForecastMetricError(
            f"{name!r} has {n_bad} null/non-finite value(s). Restrict to complete "
            "cases once, explicitly, before measuring -- dropping rows inside a "
            "metric would unpair the comparison"
        )
    return values


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _correlation_status(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Whether a correlation is defined for this pair, and if not, why."""
    if y_true.size < 2:
        return CORRELATION_UNDEFINED_INSUFFICIENT_SAMPLE
    if _is_constant(y_pred):
        return CORRELATION_UNDEFINED_CONSTANT_PREDICTOR
    if _is_constant(y_true):
        return CORRELATION_UNDEFINED_CONSTANT_TARGET
    return CORRELATION_DEFINED


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, str]:
    """Pearson correlation, or `(None, reason)` when it is undefined."""
    status = _correlation_status(y_true, y_pred)
    if status != CORRELATION_DEFINED:
        return None, status
    return float(np.corrcoef(y_true, y_pred)[0, 1]), status


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, str]:
    """Spearman rank correlation, or `(None, reason)` when it is undefined.

    Computed as the Pearson correlation of average ranks, which is the
    definition and matches `scipy.stats.spearmanr` including under ties.
    Implemented here rather than imported so this package's runtime needs
    nothing beyond its declared dependencies.
    """
    status = _correlation_status(y_true, y_pred)
    if status != CORRELATION_DEFINED:
        return None, status
    true_ranks = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    # Ranks of a non-constant vector are themselves non-constant, so this
    # denominator cannot be zero once the status check above has passed.
    return float(np.corrcoef(true_ranks, pred_ranks)[0, 1]), status


@dataclass(frozen=True)
class PredictorMetrics:
    """One predictor's error and rank agreement on one set of rows."""

    predictor: str
    n: int
    mae: float
    rmse: float
    pearson: float | None
    pearson_status: str
    spearman: float | None
    spearman_status: str
    mean_prediction: float
    sd_prediction: float

    def as_record(self) -> dict[str, object]:
        """A JSON-ready record. `None` correlations stay `None`, never 0.0."""
        return {
            "predictor": self.predictor,
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "pearson": self.pearson,
            "pearson_status": self.pearson_status,
            "spearman": self.spearman,
            "spearman_status": self.spearman_status,
            "mean_prediction": self.mean_prediction,
            "sd_prediction": self.sd_prediction,
        }


def evaluate_predictor(frame: pd.DataFrame, *, target: str, predictor: str) -> PredictorMetrics:
    """Measure one predictor against one target on complete cases only.

    Raises:
        ForecastMetricError: If either column is missing or carries NaN.
    """
    y_true = _column(frame, target)
    y_pred = _column(frame, predictor)
    if y_true.size == 0:
        raise ForecastMetricError(f"No rows to evaluate {predictor!r} on")

    pearson_value, pearson_status = pearson(y_true, y_pred)
    spearman_value, spearman_status = spearman(y_true, y_pred)
    return PredictorMetrics(
        predictor=predictor,
        n=int(y_true.size),
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        pearson=pearson_value,
        pearson_status=pearson_status,
        spearman=spearman_value,
        spearman_status=spearman_status,
        mean_prediction=float(np.mean(y_pred)),
        sd_prediction=float(np.std(y_pred, ddof=1)) if y_pred.size > 1 else 0.0,
    )


def evaluate_ladder(
    frame: pd.DataFrame, *, target: str, predictors: tuple[str, ...] | list[str]
) -> list[PredictorMetrics]:
    """Measure every rung of a comparison ladder on the same rows.

    Each rung is measured on ITS OWN complete cases, and the restriction is
    made here, explicitly, rather than inside a metric. Rungs can therefore
    have different `n` -- which is why this is a descriptive ladder and NOT a
    comparison: `paired_delta` is the only thing entitled to compare two
    rungs, and it works on rows where both are present.

    A rung whose column is entirely absent or entirely NaN on these rows is
    SKIPPED rather than imputed -- see `forecast.assemble` for the one place
    that legitimately happens (2022 has no causal prior-season deserved
    values, so it has no shrunk-deserved rung). Callers report the omission;
    this function never invents a number for it.
    """
    out: list[PredictorMetrics] = []
    for predictor in predictors:
        if predictor not in frame.columns:
            continue
        complete = frame.dropna(subset=[target, predictor])
        if complete.empty:
            continue
        out.append(evaluate_predictor(complete, target=target, predictor=predictor))
    return out


@dataclass(frozen=True)
class PairedDelta:
    """A paired difference between two predictors, on the frozen convention.

    Carries the component metrics it was built from so
    `assert_delta_convention` can re-derive the delta and prove no
    presentation layer flipped a sign.
    """

    reference: str
    challenger: str
    n: int
    mae_reference: float
    mae_challenger: float
    delta_mae: float
    pct_change_mae: float | None
    rmse_reference: float
    rmse_challenger: float
    delta_rmse: float
    pct_change_rmse: float | None
    direction_mae: str
    direction_rmse: str

    def as_record(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "challenger": self.challenger,
            "n": self.n,
            "mae_reference": self.mae_reference,
            "mae_challenger": self.mae_challenger,
            "delta_mae": self.delta_mae,
            "pct_change_mae": self.pct_change_mae,
            "rmse_reference": self.rmse_reference,
            "rmse_challenger": self.rmse_challenger,
            "delta_rmse": self.delta_rmse,
            "pct_change_rmse": self.pct_change_rmse,
            "direction_mae": self.direction_mae,
            "direction_rmse": self.direction_rmse,
            "sign_convention": DELTA_SIGN_CONVENTION["negative"],
        }


def _direction(delta: float) -> str:
    """Which side a delta favours, on a lower-is-better metric."""
    if delta < 0.0:
        return CHALLENGER_BETTER
    if delta > 0.0:
        return REFERENCE_BETTER
    return NO_DIFFERENCE


def _pct_change(delta: float, reference: float) -> float | None:
    """`delta` as a percentage of the reference metric, same sign as `delta`."""
    if reference == 0.0:
        return None
    return float(100.0 * delta / reference)


def paired_delta(
    frame: pd.DataFrame, *, target: str, reference: str, challenger: str
) -> PairedDelta:
    """The frozen `challenger - reference` difference on identical rows.

    Both predictors are measured on the SAME rows, which is what makes the
    difference paired. The caller is responsible for having restricted to
    complete cases; `_column` refuses NaN rather than dropping it.
    """
    reference_metrics = evaluate_predictor(frame, target=target, predictor=reference)
    challenger_metrics = evaluate_predictor(frame, target=target, predictor=challenger)
    return build_paired_delta(reference_metrics, challenger_metrics)


def build_paired_delta(
    reference_metrics: PredictorMetrics, challenger_metrics: PredictorMetrics
) -> PairedDelta:
    """Assemble a `PairedDelta` from two already-measured predictors.

    Raises:
        ForecastMetricError: If the two were not measured on the same number
            of rows, which would mean they are not paired.
    """
    if reference_metrics.n != challenger_metrics.n:
        raise ForecastMetricError(
            f"{challenger_metrics.predictor!r} was measured on {challenger_metrics.n} "
            f"row(s) but {reference_metrics.predictor!r} on {reference_metrics.n}; a "
            "paired delta requires identical rows"
        )
    delta_mae = float(challenger_metrics.mae - reference_metrics.mae)
    delta_rmse = float(challenger_metrics.rmse - reference_metrics.rmse)
    return PairedDelta(
        reference=reference_metrics.predictor,
        challenger=challenger_metrics.predictor,
        n=int(reference_metrics.n),
        mae_reference=reference_metrics.mae,
        mae_challenger=challenger_metrics.mae,
        delta_mae=delta_mae,
        pct_change_mae=_pct_change(delta_mae, reference_metrics.mae),
        rmse_reference=reference_metrics.rmse,
        rmse_challenger=challenger_metrics.rmse,
        delta_rmse=delta_rmse,
        pct_change_rmse=_pct_change(delta_rmse, reference_metrics.rmse),
        direction_mae=_direction(delta_mae),
        direction_rmse=_direction(delta_rmse),
    )


def assert_delta_convention(delta: PairedDelta, *, atol: float = 1e-12) -> None:
    """Re-derive a delta from its components and prove the sign was not flipped.

    The guard against a presentation layer "helpfully" reorienting a
    difference so that bigger looks better. Also re-checks the direction
    label and the percentage's sign.

    Raises:
        ForecastMetricError: On any inconsistency.
    """
    expected_mae = delta.mae_challenger - delta.mae_reference
    if abs(expected_mae - delta.delta_mae) > atol:
        raise ForecastMetricError(
            f"delta_mae {delta.delta_mae!r} does not equal MAE(challenger) - "
            f"MAE(reference) = {expected_mae!r}. The frozen convention is "
            f"{DELTA_SIGN_CONVENTION['definition']}"
        )
    expected_rmse = delta.rmse_challenger - delta.rmse_reference
    if abs(expected_rmse - delta.delta_rmse) > atol:
        raise ForecastMetricError(
            f"delta_rmse {delta.delta_rmse!r} does not equal RMSE(challenger) - "
            f"RMSE(reference) = {expected_rmse!r}"
        )
    for label, delta_value, direction in (
        ("mae", delta.delta_mae, delta.direction_mae),
        ("rmse", delta.delta_rmse, delta.direction_rmse),
    ):
        if _direction(delta_value) != direction:
            raise ForecastMetricError(
                f"direction_{label} is {direction!r} but delta_{label}="
                f"{delta_value!r} implies {_direction(delta_value)!r}"
            )
    for label, delta_value, pct in (
        ("mae", delta.delta_mae, delta.pct_change_mae),
        ("rmse", delta.delta_rmse, delta.pct_change_rmse),
    ):
        if pct is None:
            continue
        if delta_value != 0.0 and np.sign(pct) != np.sign(delta_value):
            raise ForecastMetricError(
                f"pct_change_{label}={pct!r} does not share the sign of "
                f"delta_{label}={delta_value!r}; percentage improvement is reported "
                "on the SAME convention as the delta, never inverted"
            )


# --------------------------------------------------------------------------
# Season structure: per-season deltas and the macro-average
# --------------------------------------------------------------------------


def per_season_deltas(
    frame: pd.DataFrame, *, target: str, reference: str, challenger: str
) -> dict[int, PairedDelta]:
    """The paired delta computed separately within each season."""
    out: dict[int, PairedDelta] = {}
    for season in sorted(int(s) for s in frame["season"].unique()):
        season_rows = frame[frame["season"].astype(int) == season]
        if season_rows.empty:
            continue
        out[season] = paired_delta(
            season_rows, target=target, reference=reference, challenger=challenger
        )
    return out


def macro_average_delta(season_deltas: dict[int, PairedDelta]) -> dict[str, object]:
    """Equal-weight average of the SEASON-SPECIFIC paired differences.

    Pooling windows across seasons lets the season with the most qualifying
    windows dominate purely through sample size. The macro-average gives each
    season one vote, so a pooled and a macro result that disagree are telling
    you the effect is not uniform across seasons -- which is a finding, not a
    nuisance, and both are therefore reported side by side.

    The average is taken over the season-level DELTAS, not over separately
    averaged component metrics, so it stays on the frozen sign convention.
    """
    if not season_deltas:
        return {
            "n_seasons": 0,
            "seasons": [],
            "macro_delta_mae": None,
            "macro_delta_rmse": None,
            "macro_pct_change_mae": None,
            "macro_pct_change_rmse": None,
            "direction_mae": None,
            "direction_rmse": None,
            "season_directions_mae": {},
            "weighting": "equal weight per season",
        }
    seasons = sorted(season_deltas)
    delta_maes = [season_deltas[s].delta_mae for s in seasons]
    delta_rmses = [season_deltas[s].delta_rmse for s in seasons]
    # Built with an explicit loop rather than a filtered comprehension so the
    # `None` exclusion is visible in the type as well as at runtime: a season
    # whose reference error is exactly zero has no percentage change, and
    # averaging it in as a zero would understate the macro figure.
    pct_maes: list[float] = []
    pct_rmses: list[float] = []
    for season in seasons:
        delta = season_deltas[season]
        if delta.pct_change_mae is not None:
            pct_maes.append(delta.pct_change_mae)
        if delta.pct_change_rmse is not None:
            pct_rmses.append(delta.pct_change_rmse)
    macro_mae = float(np.mean(delta_maes))
    macro_rmse = float(np.mean(delta_rmses))
    return {
        "n_seasons": len(seasons),
        "seasons": seasons,
        "macro_delta_mae": macro_mae,
        "macro_delta_rmse": macro_rmse,
        "macro_pct_change_mae": float(np.mean(pct_maes)) if pct_maes else None,
        "macro_pct_change_rmse": float(np.mean(pct_rmses)) if pct_rmses else None,
        "direction_mae": _direction(macro_mae),
        "direction_rmse": _direction(macro_rmse),
        "season_directions_mae": {str(s): season_deltas[s].direction_mae for s in seasons},
        "weighting": "equal weight per season",
    }


# --------------------------------------------------------------------------
# Batter-balanced weighting
# --------------------------------------------------------------------------


def batter_balanced_errors(
    frame: pd.DataFrame,
    *,
    target: str,
    predictor: str,
    cluster_column: str = BOOTSTRAP_CLUSTER_COLUMN,
) -> tuple[float, float, int]:
    """`(MAE, RMSE, n_unique_batters)` with every unique batter weighted equally.

    Loss is averaged WITHIN each batter first, then across batters. A hitter
    who appears in three seasons therefore counts once, not three times --
    which is the point: the ordinary window-weighted figure lets durable
    everyday hitters, who are also the ones most likely to clear
    `cutoff + horizon` in several seasons, carry more of the result than
    hitters who qualified once.

    RMSE is `sqrt(mean over batters of mean squared error within batter)`,
    the batter-balanced analogue of the pooled definition.
    """
    y_true = _column(frame, target)
    y_pred = _column(frame, predictor)
    if cluster_column not in frame.columns:
        raise ForecastMetricError(f"Frame has no cluster column {cluster_column!r}")

    per_row = pd.DataFrame(
        {
            "cluster": frame[cluster_column].to_numpy(),
            "abs_error": np.abs(y_pred - y_true),
            "sq_error": (y_pred - y_true) ** 2,
        }
    )
    per_batter = per_row.groupby("cluster")[["abs_error", "sq_error"]].mean()
    return (
        float(per_batter["abs_error"].mean()),
        float(np.sqrt(per_batter["sq_error"].mean())),
        int(len(per_batter)),
    )


def batter_balanced_delta(
    frame: pd.DataFrame,
    *,
    target: str,
    reference: str,
    challenger: str,
    cluster_column: str = BOOTSTRAP_CLUSTER_COLUMN,
) -> PairedDelta:
    """The frozen `challenger - reference` delta under batter-balanced weighting.

    Same convention, same guard, different weights: negative still means
    deserved is better.
    """
    ref_mae, ref_rmse, n_ref = batter_balanced_errors(
        frame, target=target, predictor=reference, cluster_column=cluster_column
    )
    chal_mae, chal_rmse, n_chal = batter_balanced_errors(
        frame, target=target, predictor=challenger, cluster_column=cluster_column
    )
    if n_ref != n_chal:
        raise ForecastMetricError(
            "Batter-balanced comparison saw different batter counts for the two "
            f"predictors ({n_ref} vs {n_chal}); the rows are not paired"
        )
    delta_mae = float(chal_mae - ref_mae)
    delta_rmse = float(chal_rmse - ref_rmse)
    return PairedDelta(
        reference=reference,
        challenger=challenger,
        n=n_ref,
        mae_reference=ref_mae,
        mae_challenger=chal_mae,
        delta_mae=delta_mae,
        pct_change_mae=_pct_change(delta_mae, ref_mae),
        rmse_reference=ref_rmse,
        rmse_challenger=chal_rmse,
        delta_rmse=delta_rmse,
        pct_change_rmse=_pct_change(delta_rmse, ref_rmse),
        direction_mae=_direction(delta_mae),
        direction_rmse=_direction(delta_rmse),
    )


def describe_practical_magnitude(delta: PairedDelta, *, target_sd: float) -> dict[str, object]:
    """Express a delta in units a reader can judge, alongside its raw size.

    Statistical distinguishability is not magnitude. A delta is reported as a
    fraction of the TARGET's own spread and as a percentage of the reference
    error, so a difference that is real but trivially small is described as
    trivially small rather than as a win.
    """
    fraction_of_sd = float(abs(delta.delta_mae) / target_sd) if target_sd > 0.0 else None
    return {
        "delta_mae": delta.delta_mae,
        "delta_rmse": delta.delta_rmse,
        "pct_change_mae": delta.pct_change_mae,
        "target_sd": target_sd,
        "abs_delta_mae_as_fraction_of_target_sd": fraction_of_sd,
        "direction_mae": delta.direction_mae,
    }
