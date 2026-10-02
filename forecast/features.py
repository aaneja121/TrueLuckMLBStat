"""Contact Forecast R1: pre-cutoff features.

Every feature is computed from a hitter's batted balls at `bbe_index <= K`
and from strictly prior seasons. Nothing here may read a single event at or
beyond the cutoff.

## How that is enforced rather than asserted

Feature code never slices the ordered ledger itself. It receives the slice
from `forecast.windows.history_events`, which is the one supported way to
obtain pre-cutoff rows, and `build_window_features` re-checks the slice's
`bbe_index` maximum before computing anything. The corresponding test
rewrites every event at or beyond the cutoff and proves the feature table
does not move.

## Prior-season features and their asymmetric availability

Prior-season REALIZED performance needs no model -- it is the frozen
run-value table applied to the recorded outcome -- so it is available for
every analysis season including 2022 (whose prior season is 2021).

Prior-season DESERVED performance needs a scored ledger, and 2021 is a
feature-source season with no strictly-earlier season to fit a causal
contact model on. It is therefore MISSING for every 2022 window and present
for 2023 and 2024 windows. That asymmetry is real and is surfaced as an
explicit `has_prior_deserved` indicator rather than papered over with an
imputed value that would look like evidence.

Missing values are left as NaN here. Imputation belongs to the modeling
stage, where it must be fit on training folds only -- imputing in the
feature builder would fit a statistic across the whole pool and leak.

## Stage consistency

Predictors are contact-stage quantities (Rc, E0, Rc - E0) exactly like the
targets. `forecast.targets.assert_stage_consistency` is called on the source
columns so a full-telescoping column can never enter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    DESERVED_COLUMN,
    OBSERVED_COLUMN,
    RATE_SCALE,
    SURPRISE_COLUMN,
)
from forecast.targets import assert_stage_consistency
from forecast.windows import history_events

#: Trailing window, in batted balls, for the "recent form" features. Both
#: are strictly inside the history window: at the smallest cutoff (50) the
#: 25-BBE window is the hitter's second half of observed contact, never a
#: reach past the cutoff.
RECENT_WINDOW_BBE = 25

#: Launch speed at or above which contact is counted "hard". The standard
#: public Statcast threshold; a documented convention, not a fitted value.
HARD_HIT_MPH = 95.0

#: Launch-angle band conventionally associated with productive contact.
#: Documented convention, not fitted -- no threshold in this study is chosen
#: by looking at an outcome.
SWEET_SPOT_ANGLE_RANGE = (8.0, 32.0)

#: Batted-ball types whose rates become features.
BB_TYPES: tuple[str, ...] = ("ground_ball", "line_drive", "fly_ball", "popup")


class ForecastFeatureError(ValueError):
    """Raised when feature construction inputs violate the pre-cutoff rule."""


def _rate_per_100(values: pd.Series) -> float:
    """Mean of a per-batted-ball run value, expressed per 100 batted balls."""
    return float(RATE_SCALE * values.mean()) if len(values) else float("nan")


def _distribution_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    """Mean/SD/percentile summary of a contact-quality distribution."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_sd": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_sd": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        f"{prefix}_p10": float(clean.quantile(0.10)),
        f"{prefix}_p90": float(clean.quantile(0.90)),
    }


def _hitter_season_features(history: pd.DataFrame) -> dict[str, float | str]:
    """Every pre-cutoff feature for ONE hitter-season at one cutoff."""
    observed = history[OBSERVED_COLUMN]
    deserved = history[DESERVED_COLUMN]
    surprise = history[SURPRISE_COLUMN]
    recent = history.tail(RECENT_WINDOW_BBE)

    launch_speed = pd.to_numeric(history["launch_speed"], errors="coerce")
    launch_angle = pd.to_numeric(history["launch_angle"], errors="coerce")

    features: dict[str, float | str] = {
        # Season-to-date contact-stage rates through the cutoff.
        "std_realized_rv_per_100": _rate_per_100(observed),
        "std_deserved_rv_per_100": _rate_per_100(deserved),
        "std_surprise_rv_per_100": _rate_per_100(surprise),
        "n_resolved_bbe_through_cutoff": float(len(history)),
        # Recent form, still strictly inside the history window.
        "recent_realized_rv_per_100": _rate_per_100(recent[OBSERVED_COLUMN]),
        "recent_deserved_rv_per_100": _rate_per_100(recent[DESERVED_COLUMN]),
        "recent_surprise_rv_per_100": _rate_per_100(recent[SURPRISE_COLUMN]),
        # Contact-quality distributions.
        **_distribution_summary(history["launch_speed"], "launch_speed"),
        **_distribution_summary(history["launch_angle"], "launch_angle"),
        **_distribution_summary(history["spray_angle_approx"], "spray_angle"),
        "hard_hit_rate": float((launch_speed >= HARD_HIT_MPH).mean()),
        "sweet_spot_rate": float(
            launch_angle.between(*SWEET_SPOT_ANGLE_RANGE, inclusive="both").mean()
        ),
        # Handedness, as a 0/1 indicator so the feature table stays numeric.
        "bats_left": float(str(history["stand"].iloc[0]).upper() == "L"),
    }

    bb_type = history["bb_type"].astype(str)
    for kind in BB_TYPES:
        features[f"bb_rate_{kind}"] = float((bb_type == kind).mean())

    return features


def build_window_features(ordered: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Pre-cutoff features for every hitter-season that reached `cutoff`.

    Args:
        ordered: Output of `forecast.windows.order_eligible_events`, already
            restricted to resolved batted balls.
        cutoff: The observation cutoff `K`.

    Returns:
        One row per (batter, season) reaching the cutoff, with a `cutoff`
        column and every feature above.

    Raises:
        ForecastFeatureError: If the history slice somehow contains an event
            at or beyond the cutoff.
    """
    assert_stage_consistency([OBSERVED_COLUMN, DESERVED_COLUMN, SURPRISE_COLUMN])

    history = history_events(ordered, cutoff)
    if not history.empty and int(history["bbe_index"].max()) > int(cutoff):
        raise ForecastFeatureError(
            f"History slice reaches bbe_index {int(history['bbe_index'].max())} for "
            f"cutoff {cutoff}; features must never see an event at or beyond the cutoff"
        )

    # Only hitter-seasons that actually REACHED the cutoff get features. A
    # hitter with 40 batted balls has no 50-BBE observation, and averaging
    # his 40 would silently invent one.
    reached = history.groupby(["batter", "season"])["bbe_index"].max()
    eligible_keys = set(reached[reached >= int(cutoff)].index)

    rows: list[dict[str, Any]] = []
    for group_key, group in history.groupby(["batter", "season"], sort=False):
        batter, season = group_key
        if (batter, season) not in eligible_keys:
            continue
        record: dict[str, Any] = {
            "batter": batter,
            "season": int(str(season)),
            "cutoff": int(cutoff),
        }
        record.update(_hitter_season_features(group))
        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=["batter", "season", "cutoff"])
    return pd.DataFrame(rows)


def build_prior_season_table(
    scored_ledgers: dict[int, pd.DataFrame],
    realized_only_ledgers: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Full-season contact-stage rates per (batter, season), for use as PRIOR-season features.

    Args:
        scored_ledgers: season -> walk-forward ledger with both Rc and E0.
        realized_only_ledgers: season -> frame carrying Rc but no E0, for a
            season this study never scored (2021). Its deserved rate is NaN,
            which is the honest representation of "no causal contact model
            exists for that season", not a gap to fill.

    Returns:
        One row per (batter, season) with `prior_source_season`,
        `prior_n_resolved_bbe`, `prior_realized_rv_per_100`,
        `prior_deserved_rv_per_100`, and `has_prior_deserved`.
    """
    frames: list[pd.DataFrame] = []

    for season, ledger in sorted(scored_ledgers.items()):
        grouped = ledger.groupby("batter").agg(
            prior_n_resolved_bbe=(OBSERVED_COLUMN, "size"),
            observed_sum=(OBSERVED_COLUMN, "sum"),
            deserved_sum=(DESERVED_COLUMN, "sum"),
        )
        frame = grouped.reset_index()
        frame["prior_source_season"] = int(season)
        frame["prior_realized_rv_per_100"] = (
            RATE_SCALE * frame["observed_sum"] / frame["prior_n_resolved_bbe"]
        )
        frame["prior_deserved_rv_per_100"] = (
            RATE_SCALE * frame["deserved_sum"] / frame["prior_n_resolved_bbe"]
        )
        frames.append(frame.drop(columns=["observed_sum", "deserved_sum"]))

    for season, ledger in sorted((realized_only_ledgers or {}).items()):
        grouped = ledger.groupby("batter").agg(
            prior_n_resolved_bbe=(OBSERVED_COLUMN, "size"),
            observed_sum=(OBSERVED_COLUMN, "sum"),
        )
        frame = grouped.reset_index()
        frame["prior_source_season"] = int(season)
        frame["prior_realized_rv_per_100"] = (
            RATE_SCALE * frame["observed_sum"] / frame["prior_n_resolved_bbe"]
        )
        frame["prior_deserved_rv_per_100"] = np.nan
        frames.append(frame.drop(columns=["observed_sum"]))

    if not frames:
        return pd.DataFrame(
            columns=[
                "batter",
                "prior_source_season",
                "prior_n_resolved_bbe",
                "prior_realized_rv_per_100",
                "prior_deserved_rv_per_100",
                "has_prior_deserved",
            ]
        )

    table = pd.concat(frames, ignore_index=True)
    table["has_prior_deserved"] = table["prior_deserved_rv_per_100"].notna().astype(float)
    return table


def attach_prior_season_features(
    features: pd.DataFrame, prior_season_table: pd.DataFrame
) -> pd.DataFrame:
    """Join each row's IMMEDIATELY PRIOR season's full-season rates.

    The join key is `prior_source_season == season - 1`, so a window in
    season S can only ever see season S-1, never S itself and never a later
    one. A hitter with no prior season (a rookie, or one who did not appear)
    gets NaN and `has_prior_deserved == 0`.
    """
    if features.empty:
        return features.copy()

    enriched = features.copy()
    enriched["prior_source_season"] = enriched["season"].astype(int) - 1

    merged = enriched.merge(
        prior_season_table,
        on=["batter", "prior_source_season"],
        how="left",
        validate="many_to_one",
    )
    merged["has_prior_season"] = merged["prior_n_resolved_bbe"].notna().astype(float)
    merged["has_prior_deserved"] = merged["has_prior_deserved"].fillna(0.0)

    late = merged["prior_source_season"] >= merged["season"]
    if bool(late.any()):
        raise ForecastFeatureError(
            "Prior-season join produced a source season not strictly earlier than the "
            "window's own season"
        )
    return merged


def feature_columns(features: pd.DataFrame) -> list[str]:
    """The modelable feature columns of a built feature table, in stable order.

    Excludes identifiers and join bookkeeping. `cutoff` is excluded on
    purpose: within a (cutoff, horizon) cell it is constant, so it carries
    no within-cell information and would only matter in a pooled fit, where
    it would act as a proxy for playing time rather than contact quality.
    """
    excluded = {
        "batter",
        "season",
        "cutoff",
        "prior_source_season",
        "window_id",
        "horizon",
    }
    return [
        column
        for column in features.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]
