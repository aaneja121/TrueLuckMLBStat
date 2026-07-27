"""Season-level aggregation of Version 0.2 raw contact luck (runs).

Only raw contact luck (`mlb_luck_score.scoring.contact_luck.
compute_raw_contact_luck_runs`, additive, in runs) is ever summed or
averaged here. **Never sum or average public display scores**
(`mlb_luck_score.scoring.empirical_score`) to construct a season metric --
the signed-percentile mapping is non-linear and computed per play, so a
season total or rate has no defined meaning in score units. If you need a
season-level number, compute it from raw luck values, not from scores.

These are aggregation PRIMITIVES, not a leaderboard: no qualification
threshold (e.g. a minimum number of eligible events for inclusion) is
defined or enforced here yet -- that is deliberately deferred to future
work (see README.md "Future work").
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class AggregationValidationError(ValueError):
    """Raised when raw-luck values passed to an aggregation function are invalid."""


def _as_array(raw_luck_values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(raw_luck_values), dtype=float)
    if arr.size and not np.isfinite(arr).all():
        raise AggregationValidationError("raw_luck_values contains non-finite value(s)")
    return arr


def eligible_event_count(raw_luck_values: Iterable[float]) -> int:
    """Count of eligible events (rows) contributing raw-luck values."""
    return int(_as_array(raw_luck_values).size)


def total_raw_contact_luck_runs(raw_luck_values: Iterable[float]) -> float:
    """Sum of raw contact luck across events, in runs. Additive by construction."""
    return float(_as_array(raw_luck_values).sum())


def positive_raw_luck_runs(raw_luck_values: Iterable[float]) -> float:
    """Sum of only the favorable (positive) raw-luck values, in runs."""
    arr = _as_array(raw_luck_values)
    return float(arr[arr > 0].sum())


def negative_raw_luck_runs(raw_luck_values: Iterable[float]) -> float:
    """Sum of only the unfavorable (negative) raw-luck values, in runs."""
    arr = _as_array(raw_luck_values)
    return float(arr[arr < 0].sum())


def raw_contact_luck_runs_per_100_eligible_events(raw_luck_values: Iterable[float]) -> float:
    """Rate-normalized total raw contact luck, in runs per 100 eligible events.

    Raises:
        AggregationValidationError: If `raw_luck_values` is empty (a rate
            with zero events in the denominator is undefined).
    """
    arr = _as_array(raw_luck_values)
    if arr.size == 0:
        raise AggregationValidationError(
            "Cannot compute a per-100-event rate with zero eligible events"
        )
    return float(arr.sum() / arr.size * 100.0)
