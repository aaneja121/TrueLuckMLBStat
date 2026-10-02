"""Contact Forecast R1: the target contract.

`forecast.windows.build_windows` COMPUTES the two targets as part of window
construction. This module owns their DEFINITION, their conditional
interpretation, and an independent re-derivation used to cross-check the
window arithmetic by a second code path.

## The two targets

Both are rates per 100 RESOLVED eligible batted balls over the target window
only, and both are contact-stage quantities:

  - **Target A**, `target_realized_rv_per_100`: observed contact-result run
    value (Rc). The principal user-facing forecasting target at the primary
    horizon.
  - **Target B**, `target_deserved_rv_per_100`: baseline expected contact
    run value (E0) under the frozen contact model. Persistence of underlying
    contact quality, less contaminated by future defensive and outcome noise.

Neither is the published Contact Luck metric (Rf - E0). See
`forecast_config.CONTACT_STAGE_DISCLAIMER`.

## Stage consistency is a hard requirement

Predictors and targets must come from the same stage. A contact-stage
predictor paired with a full-telescoping Rf target would be comparing two
different quantities and would silently attribute advancement-stage variance
to the contact model. `assert_stage_consistency` enforces this on the column
names actually in play, so the pairing cannot drift as the codebase grows.

## Targets are conditional, not projections of playing time

A target exists only for hitters who went on to accumulate the horizon's
worth of additional resolved eligible batted balls. Nothing here forecasts
whether a hitter will reach that horizon -- see
`forecast_config.conditional_forecast_language`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    BANNED_R1_METRIC_PHRASES,
    DESERVED_COLUMN,
    OBSERVED_COLUMN,
    PRIMARY_HORIZON,
    PRIMARY_TARGET,
    RATE_SCALE,
    SURPRISE_COLUMN,
    conditional_forecast_language,
)

#: Target column -> the per-batted-ball ledger column it aggregates.
TARGET_SOURCE_COLUMNS: dict[str, str] = {
    "target_realized_rv_per_100": OBSERVED_COLUMN,
    "target_deserved_rv_per_100": DESERVED_COLUMN,
}

#: Human-readable target labels for the report. Deliberately spell out
#: "contact-result" and "expected contact" so no figure caption can quietly
#: become "Contact Luck".
TARGET_LABELS: dict[str, str] = {
    "target_realized_rv_per_100": (
        "Future realized contact-result run value per 100 resolved BBE (Target A)"
    ),
    "target_deserved_rv_per_100": (
        "Future expected contact run value per 100 resolved BBE (Target B)"
    ),
}

#: Every ledger column that belongs to the contact stage. Any target or
#: predictor must be built from one of these -- never from a full
#: telescoping (Rf) column.
CONTACT_STAGE_COLUMNS: frozenset[str] = frozenset(
    {OBSERVED_COLUMN, DESERVED_COLUMN, SURPRISE_COLUMN}
)

#: Full-telescoping column names. Present here ONLY so their appearance can
#: be detected and refused; this package never computes them.
FULL_TELESCOPING_COLUMNS: frozenset[str] = frozenset(
    {
        "observed_final_run_value",
        "final_result_surprise",
        "observed_run_value",
        "contact_luck_runs",
        "expected_run_value",
    }
)


class ForecastTargetError(ValueError):
    """Raised when a target definition or stage-consistency rule is violated."""


def assert_stage_consistency(columns: list[str] | tuple[str, ...]) -> None:
    """Refuse any mix of contact-stage and full-telescoping quantities.

    Args:
        columns: Ledger column names a predictor or target is built from.

    Raises:
        ForecastTargetError: If a full-telescoping (Rf) column appears. R1
            is a contact-stage experiment end to end; mixing stages would
            attribute advancement variance to the contact model.
    """
    offending = sorted(set(columns) & FULL_TELESCOPING_COLUMNS)
    if offending:
        raise ForecastTargetError(
            f"Full-telescoping column(s) {offending} cannot be used in R1: predictors "
            "and targets are contact-stage quantities (Rc, E0, Rc - E0) end to end. "
            "Mixing stages would attribute batter-runner advancement variance to the "
            "contact model."
        )


def assert_no_banned_metric_language(text: str) -> None:
    """Refuse report text that calls an R1 quantity the published metric.

    Mirrors `mlb_luck_score.scoring.public_labels`' banned-phrase approach:
    the guard lives in code so a caption or conclusion cannot quietly
    upgrade `contact_result_surprise` into Contact Luck.

    Raises:
        ForecastTargetError: If any banned phrase appears (case-insensitive).
    """
    lowered = text.lower()
    found = [phrase for phrase in BANNED_R1_METRIC_PHRASES if phrase in lowered]
    if found:
        raise ForecastTargetError(
            f"Text uses banned phrase(s) {found}. R1 measures the contact stage "
            "(Rc - E0), not the published Contact Luck metric (Rf - E0); say so "
            "explicitly rather than eliding the difference."
        )


def target_columns() -> tuple[str, ...]:
    """Both target column names, primary first."""
    return (PRIMARY_TARGET, *(t for t in TARGET_SOURCE_COLUMNS if t != PRIMARY_TARGET))


def is_primary_cell(cutoff: int, horizon: int, target: str) -> bool:
    """Whether a (cutoff, horizon, target) cell belongs to the frozen primary family.

    Every cutoff at the primary horizon is primary; no single cutoff may be
    promoted after results are visible. `cutoff` is accepted so callers read
    naturally and so a future narrowing has one place to change.
    """
    del cutoff
    return int(horizon) == PRIMARY_HORIZON and target == PRIMARY_TARGET


def describe_target(target: str, horizon: int) -> str:
    """The target's label plus its mandatory conditional-interpretation sentence."""
    if target not in TARGET_LABELS:
        raise ForecastTargetError(f"Unknown target {target!r}")
    return f"{TARGET_LABELS[target]}. {conditional_forecast_language(horizon)}"


def recompute_targets_independently(ordered: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Re-derive both targets by direct per-window slicing, not cumulative sums.

    `build_windows` computes targets from group cumulative sums, which is
    fast but easy to get subtly wrong at a boundary. This re-derives the same
    numbers the slow, obvious way -- take each window's rows and average them
    -- so `assert_targets_match_independent_recomputation` can compare two
    genuinely independent code paths rather than re-running one.

    Returns:
        `windows`' key columns plus `recomputed_realized`/`recomputed_deserved`.
    """
    indexed = ordered.set_index(["batter", "season", "bbe_index"]).sort_index()

    recomputed_realized: list[float] = []
    recomputed_deserved: list[float] = []
    for batter, season, start, end in zip(
        windows["batter"].tolist(),
        windows["season"].tolist(),
        windows["target_start_index"].astype(int).tolist(),
        windows["target_end_index"].astype(int).tolist(),
        strict=True,
    ):
        block = indexed.loc[(batter, season, slice(start, end)), :]
        recomputed_realized.append(RATE_SCALE * float(block[OBSERVED_COLUMN].mean()))
        recomputed_deserved.append(RATE_SCALE * float(block[DESERVED_COLUMN].mean()))

    return pd.DataFrame(
        {
            "window_id": windows["window_id"].to_numpy(),
            "recomputed_realized": recomputed_realized,
            "recomputed_deserved": recomputed_deserved,
        }
    )


def assert_targets_match_independent_recomputation(
    ordered: pd.DataFrame, windows: pd.DataFrame, *, atol: float = 1e-9
) -> None:
    """Cross-check `build_windows`' targets against a direct re-derivation.

    Raises:
        ForecastTargetError: If any window's target differs by more than
            `atol`, naming the offenders.
    """
    if windows.empty:
        return
    recomputed = recompute_targets_independently(ordered, windows)
    merged = windows.merge(recomputed, on="window_id", validate="one_to_one")

    for target, recomputed_col in (
        ("target_realized_rv_per_100", "recomputed_realized"),
        ("target_deserved_rv_per_100", "recomputed_deserved"),
    ):
        delta = np.abs(merged[target].to_numpy() - merged[recomputed_col].to_numpy())
        bad = delta > atol
        if bad.any():
            offenders = merged.loc[bad, "window_id"].head(5).tolist()
            raise ForecastTargetError(
                f"{target} disagrees with an independent re-derivation for "
                f"{int(bad.sum())} window(s) (max delta {float(delta.max()):.3e}); "
                f"e.g. {offenders}"
            )


def summarize_repeated_window_structure(windows: pd.DataFrame) -> dict[str, object]:
    """Report how much the window table repeats hitters.

    One hitter contributes several overlapping (cutoff, horizon) windows and
    may appear in several seasons, so the row count is NOT a count of
    independent observations. Reporting these three numbers together, plus
    the concentration of windows per batter, is what stops 6,773 windows
    from being read as 6,773 independent draws.
    """
    if windows.empty:
        return {
            "n_windows": 0,
            "n_unique_batter_seasons": 0,
            "n_unique_batters": 0,
            "windows_per_batter_mean": float("nan"),
            "windows_per_batter_max": 0,
            "share_of_windows_from_top_decile_of_batters": float("nan"),
        }

    per_batter = windows.groupby("batter").size().sort_values(ascending=False)
    top_decile_size = max(1, int(round(0.10 * len(per_batter))))
    return {
        "n_windows": int(len(windows)),
        "n_unique_batter_seasons": int(windows.groupby(["batter", "season"]).ngroups),
        "n_unique_batters": int(windows["batter"].nunique()),
        "windows_per_batter_mean": float(per_batter.mean()),
        "windows_per_batter_max": int(per_batter.max()),
        "share_of_windows_from_top_decile_of_batters": float(
            per_batter.head(top_decile_size).sum() / len(windows)
        ),
    }
