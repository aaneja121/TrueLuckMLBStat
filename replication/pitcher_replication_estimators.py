"""Version 0.14: the frozen research estimators for replication question E
(rate precision and resolving power).

Extracted from `demo/build_pitcher_prototype_fixture.py` so the replication
freeze depends on RESEARCH code, never on presentation code. The formulas
are unchanged -- this module is a relocation with tests, not a rewrite. The
fixture generator now imports from here, so the dashboard fixture and the
replication are guaranteed to be computed by the same functions rather than
by two copies that can drift.

## The frozen formulas

Every one of these is fixed and may not be changed in response to 2025.

    half_width      = (ci_high - ci_low) / 2
    measurement_sd  = (ci_high - ci_low) / (2 * 1.96)
    var_observed    = var(per_100, ddof=1)
    var_measurement = mean(measurement_sd ** 2)
    var_signal      = var_observed - var_measurement          # NEVER clipped
    resolving_power = sqrt(var_signal) / sqrt(var_measurement)

    half_width      = k / sqrt(bbe)
    k               = median(half_width * sqrt(bbe)) over rows with >=30 BBE
    required_bbe    = round((k / target_half_width) ** 2)

**95% is already represented by the bootstrap interval endpoints.** The
percentile bootstrap produces the endpoints directly, so nothing here
multiplies by 1.96 a second time. The single 1.96 in `measurement_sd` runs
the other way: it converts an already-95% interval BACK to the normal
-approximation SD the variance decomposition needs.

## Negative signal variance is a result, not an error

`var_signal` is returned raw, including when it is negative. A negative
estimate means the observed spread is no wider than measurement error alone
would produce -- which is the finding. Clipping it to zero and presenting
the result as near-zero-but-real would be a different, and false, claim.
`resolving_power` reports 0.0 in that case and sets
`signal_variance_is_negative`, so a reader is never shown a positive-looking
number derived from a negative variance.

## The >=1 BBE artifact, guarded here rather than remembered

The bootstrap resamples GAMES within a pitcher-season. A season with one
appearance has one game to resample, so every replicate reproduces the point
estimate exactly, the interval has zero width, and its measurement variance
records as zero. The decomposition then books all of that row's (enormous)
spread as signal.

On 2024 this produced a spurious resolving power of 1.94 at a >=1 BBE floor;
excluding the 73 zero-width rows gives 0.877. `PRACTICAL_WORKLOAD_FLOORS`
therefore excludes 1, every result carries `n_zero_width_intervals`, and
`resolving_power_report` refuses to omit that count. **A >=1 BBE result must
never be used as evidence of signal.**
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Converts an already-95% interval back to the normal-approximation SD it
#: implies. NOT a widening factor -- see the module docstring.
NORMAL_95_Z = 1.96

#: Workload floors the resolving-power table reports. 1 is deliberately
#: absent; see the module docstring.
PRACTICAL_WORKLOAD_FLOORS: tuple[int, ...] = (60, 150, 300, 450)

#: Rows below this exposure are excluded when fitting `k`, so the fit is not
#: dominated by seasons whose interval is essentially unbounded.
K_FIT_MIN_BBE = 30

#: Half-widths, in runs per 100, that `required_bbe` is reported for.
PRECISION_TARGETS: tuple[int, ...] = (5, 4, 3, 2)

#: An interval narrower than this is treated as zero-width. Bootstrap
#: endpoints for a single-game season are bit-identical, so this only has to
#: absorb float round-trips through JSON.
ZERO_WIDTH_TOLERANCE = 1e-9


class EstimatorError(ValueError):
    """Raised when an estimator's inputs cannot support the statistic."""


def interval_half_width(ci_low: Any, ci_high: Any) -> Any:
    """`(ci_high - ci_low) / 2`, elementwise."""
    return (np.asarray(ci_high, dtype=float) - np.asarray(ci_low, dtype=float)) / 2.0


def measurement_sd(ci_low: Any, ci_high: Any) -> Any:
    """The normal-approximation SD a 95% interval implies:
    `(ci_high - ci_low) / (2 * 1.96)`.
    """
    width = np.asarray(ci_high, dtype=float) - np.asarray(ci_low, dtype=float)
    return width / (2.0 * NORMAL_95_Z)


def count_zero_width_intervals(ci_low: Any, ci_high: Any) -> int:
    """How many intervals have (effectively) zero width -- the artifact
    counter question E must always disclose.
    """
    width = np.abs(np.asarray(ci_high, dtype=float) - np.asarray(ci_low, dtype=float))
    return int(np.sum(width < ZERO_WIDTH_TOLERANCE))


def decompose_variance(values: Any, ci_low: Any, ci_high: Any) -> dict[str, float]:
    """`var_observed = var_signal + mean(var_measurement)`, solved for the
    signal term.

    Returns:
        `var_observed`, `var_measurement`, `var_signal` as raw floats.
        `var_signal` is NEVER clipped.

    Raises:
        EstimatorError: if fewer than two rows are supplied (a sample
            variance with `ddof=1` is undefined).
    """
    observed = np.asarray(values, dtype=float)
    if observed.size < 2:
        raise EstimatorError("variance decomposition needs at least 2 rows")
    sd = measurement_sd(ci_low, ci_high)
    var_measurement = float(np.mean(np.asarray(sd, dtype=float) ** 2))
    var_observed = float(np.var(observed, ddof=1))
    return {
        "var_observed": var_observed,
        "var_measurement": var_measurement,
        "var_signal": var_observed - var_measurement,
    }


def resolving_power(values: Any, ci_low: Any, ci_high: Any) -> dict[str, Any]:
    """`sqrt(var_signal) / sqrt(var_measurement)`.

    Below 1, the typical difference between two rows' true values is smaller
    than the error bar on either of them, so an ordering on the quantity is
    substantially an ordering of noise.

    Returns `resolving_power` 0.0 when `var_signal` is non-positive, with
    `signal_variance_is_negative` set and the raw (possibly negative)
    `var_signal` still reported.
    """
    parts = decompose_variance(values, ci_low, ci_high)
    var_signal = parts["var_signal"]
    var_measurement = parts["var_measurement"]
    is_negative = var_signal <= 0
    if is_negative or var_measurement <= 0:
        power = 0.0
    else:
        power = float(np.sqrt(var_signal) / np.sqrt(var_measurement))
    return {
        **parts,
        "resolving_power": power,
        "signal_variance_is_negative": bool(is_negative),
    }


def estimate_k(half_width: Any, bbe: Any, *, min_bbe: int = K_FIT_MIN_BBE) -> float:
    """`k = median(half_width * sqrt(bbe))` over rows with `bbe >= min_bbe`.

    A description of the supplied season's own measured intervals under the
    sqrt-n form a per-100 rate obeys -- never a model, and never fitted by
    optimization.

    Raises:
        EstimatorError: if no row clears `min_bbe`.
    """
    widths = np.asarray(half_width, dtype=float)
    exposure = np.asarray(bbe, dtype=float)
    keep = exposure >= min_bbe
    if not keep.any():
        raise EstimatorError(f"no rows with >= {min_bbe} BBE to fit k on")
    return float(np.median(widths[keep] * np.sqrt(exposure[keep])))


def required_bbe(k: float, target_half_width: float) -> int:
    """`round((k / target_half_width) ** 2)` -- the resolved BBE at which the
    fitted relationship reaches `target_half_width` runs per 100.

    Raises:
        EstimatorError: if `target_half_width` is not positive.
    """
    if target_half_width <= 0:
        raise EstimatorError("target_half_width must be positive")
    return int(round((k / target_half_width) ** 2))


# ---------------------------------------------------------------------------
# Frame-level report shapes
#
# These preserve the exact keys and 4-decimal rounding the Version 0.13.1
# research report already used, so the committed fixture and the research
# report are unchanged by the extraction.
# ---------------------------------------------------------------------------


def resolving_power_report(
    frame: pd.DataFrame,
    *,
    min_bbe: int,
    drop_zero_width: bool = False,
    bbe_column: str = "eligible_batted_balls",
    value_column: str = "contact_luck_per_100",
    ci_low_column: str = "per_100_ci_low",
    ci_high_column: str = "per_100_ci_high",
) -> dict[str, Any]:
    """Resolving power for rows at or above `min_bbe`, always disclosing the
    zero-width interval count for that slice.

    `drop_zero_width=True` produces the companion figure the >=1 BBE artifact
    guard requires: the same statistic with the zero-width (one-appearance)
    rows removed.
    """
    rows = frame[frame[bbe_column] >= min_bbe]
    n_zero_width = count_zero_width_intervals(rows[ci_low_column], rows[ci_high_column])
    if drop_zero_width:
        width = (rows[ci_high_column] - rows[ci_low_column]).abs()
        rows = rows[width >= ZERO_WIDTH_TOLERANCE]
    if len(rows) < 2:
        return {
            "min_bbe": min_bbe,
            "n_rows": int(len(rows)),
            "n_zero_width_intervals": n_zero_width,
            "zero_width_intervals_dropped": drop_zero_width,
            "resolving_power": None,
        }
    result = resolving_power(rows[value_column], rows[ci_low_column], rows[ci_high_column])
    return {
        "min_bbe": min_bbe,
        "n_rows": int(len(rows)),
        "n_zero_width_intervals": n_zero_width,
        "zero_width_intervals_dropped": drop_zero_width,
        "sd_observed_per_100": round(float(np.sqrt(result["var_observed"])), 4),
        "mean_sd_measurement_per_100": round(float(np.sqrt(result["var_measurement"])), 4),
        "var_signal_per_100": round(result["var_signal"], 4),
        "resolving_power": round(result["resolving_power"], 4),
        "signal_variance_is_negative": result["signal_variance_is_negative"],
    }


def rate_precision_report(
    frame: pd.DataFrame,
    *,
    min_bbe: int = K_FIT_MIN_BBE,
    targets: tuple[int, ...] = PRECISION_TARGETS,
    bbe_column: str = "eligible_batted_balls",
    ci_low_column: str = "per_100_ci_low",
    ci_high_column: str = "per_100_ci_high",
) -> dict[str, Any]:
    """How many resolved BBE a season needs before its 95% rate interval is
    narrower than each target half-width.
    """
    usable = frame[frame[bbe_column] >= min_bbe]
    half_width = interval_half_width(usable[ci_low_column], usable[ci_high_column])
    k = estimate_k(half_width, usable[bbe_column], min_bbe=0)
    return {
        "form": "half_width_runs_per_100 = k / sqrt(bbe)",
        "k": round(k, 3),
        "fitted_on_rows_with_min_bbe": min_bbe,
        "n_rows": int(len(usable)),
        "bbe_required": {f"plus_minus_{target}": required_bbe(k, target) for target in targets},
    }
