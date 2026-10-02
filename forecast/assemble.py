"""Contact Forecast R1: assembling the paired prediction table.

Turns the walk-forward ledgers `forecast.score_development_seasons` wrote
into ONE table with, per (batter, season, cutoff, horizon) window: the two
targets, the pre-cutoff features, and every rung of the frozen baseline
ladder. Everything downstream -- metrics, bootstrap, sensitivity, the
regression-direction experiment -- reads this table and nothing else, so
there is exactly one place where a window's predictors are decided.

## The 2022 shrunk-deserved gap is structural and is NOT imputed

`forecast.baselines.fit_shrinkage` estimates a quantity's prior mean,
within-hitter variance and between-hitter variance from STRICTLY EARLIER
seasons. For the deserved quantity that requires a prior season carrying
`baseline_expected_contact_run_value`, i.e. a walk-forward-scored season:

    2023 -> fit on 2022    (scored)
    2024 -> fit on 2022-23 (scored)
    2022 -> fit on 2021    (NOT scored, and never can be)

2021 is a feature-source season with no strictly-earlier development season
to fit a causal contact model on, so it has no deserved values at all. It is
not that they are missing from a file -- they do not exist, and inventing
them by fitting a 2021 model on 2021 would make every 2022 deserved number
partly in-sample.

The consequence is stated rather than patched: **2022 has no
`shrunk_deserved_persistence` rung**, so the primary comparison
(shrunk deserved versus shrunk realized) covers 2023 and 2024 only, and its
macro-average is over two seasons. 2022 still contributes `league_mean`,
`raw_realized_persistence`, `raw_deserved_persistence` and
`shrunk_realized_persistence` -- all four need only 2021's REALIZED run
values, which need no model, just the frozen run-value table applied to the
recorded outcome. `RUNG_AVAILABILITY` carries this per season into every
artifact, and the affected cells are reported as unavailable, never dropped
from a table.

## Matched shrinkage means the same fit sample

Realized and deserved shrinkage are fit on the SAME prior events for 2023
and 2024, so the two ladders differ only in which column the identical
estimator read. 2022's realized-only fit necessarily uses a different prior
sample (2021), which is one more reason its deserved counterpart is absent
rather than approximated from somewhere else.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.baselines import (
    ShrinkageFit,
    assert_fit_is_causal,
    build_baseline_predictions,
    fit_shrinkage,
    shrinkage_fit_record,
)
from forecast.features import build_window_features
from forecast.forecast_config import (
    BASELINE_LADDER,
    DESERVED_COLUMN,
    FORECAST_ANALYSIS_SEASONS,
    FORECAST_CUTOFFS,
    FORECAST_DATA_DIR,
    FORECAST_HORIZONS,
    OBSERVED_COLUMN,
    assert_forecast_seasons_allowed,
    assert_path_outside_forbidden_namespaces,
)
from forecast.targets import assert_targets_match_independent_recomputation
from forecast.windows import (
    build_windows,
    order_eligible_events,
    select_scored_eligible_events,
    summarize_inclusion,
)

logger = logging.getLogger(__name__)

#: Which prior seasons each analysis season's shrinkage may be fit on, and
#: whether a deserved fit is possible there at all.
SHRINKAGE_PRIOR_SEASONS: dict[int, tuple[int, ...]] = {
    2022: (2021,),
    2023: (2022,),
    2024: (2022, 2023),
}

#: The one structurally unavailable rung, recorded per season so no artifact
#: has to infer it from a NaN.
RUNG_AVAILABILITY: dict[int, dict[str, object]] = {
    2022: {
        "shrunk_deserved_persistence": "unavailable",
        "reason": (
            "2022's only strictly-earlier season is 2021, which has no causal "
            "contact model and therefore no baseline_expected_contact_run_value. "
            "The deserved shrinkage parameters cannot be estimated causally and "
            "are NOT imputed."
        ),
    },
    2023: {"shrunk_deserved_persistence": "available"},
    2024: {"shrunk_deserved_persistence": "available"},
}

#: Cached 2021 realized-only events, so the 111 MB development dataset is
#: read at most once across runs.
PRIOR_2021_CACHE = FORECAST_DATA_DIR / "prior_2021_realized_events.parquet"


class ForecastAssemblyError(RuntimeError):
    """Raised when the prediction table cannot be assembled as specified."""


def load_ledger(season: int, *, data_dir: Path = FORECAST_DATA_DIR) -> pd.DataFrame:
    """Read one walk-forward ledger, refusing an unauthorized season or path."""
    assert_forecast_seasons_allowed(int(season))
    path = data_dir / f"walk_forward_ledger_{int(season)}.parquet"
    assert_path_outside_forbidden_namespaces(path)
    if not path.exists():
        raise ForecastAssemblyError(
            f"No walk-forward ledger for {season} at {path}. Run "
            "`python -m forecast.score_development_seasons` first."
        )
    return pd.read_parquet(path)


def load_prior_2021_events(
    *, data_dir: Path = FORECAST_DATA_DIR, development_input: Path | None = None
) -> pd.DataFrame:
    """2021's per-batted-ball REALIZED run values -- no model involved.

    Cached to Parquet on first build. Delegates to
    `forecast.score_development_seasons.build_realized_only_reference`, which
    is the one place that knows how a season with no causal contact model may
    still supply realized values.
    """
    cache = data_dir / PRIOR_2021_CACHE.name
    if cache.exists():
        return pd.read_parquet(cache)

    # Imported lazily: this pulls in the frozen scoring package and the full
    # development dataset, which the rest of the analysis never needs.
    from forecast.score_development_seasons import (
        DEFAULT_DEVELOPMENT_INPUT,
        _prepare_development_frame,
        build_realized_only_reference,
    )

    input_path = development_input or DEFAULT_DEVELOPMENT_INPUT
    logger.info("Building 2021 realized-only prior events from %s", input_path)
    development_df = _prepare_development_frame(input_path)
    events = build_realized_only_reference(development_df, 2021)
    cache.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cache, index=False)
    return events


def prior_events_for(
    season: int,
    *,
    ledgers: dict[int, pd.DataFrame],
    prior_2021: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The strictly-earlier per-batted-ball rows a season's shrinkage is fit on.

    Returns resolved rows only. For 2022 this is 2021's realized-only frame,
    which carries no deserved column -- which is exactly why 2022 has no
    shrunk-deserved rung.
    """
    prior_seasons = SHRINKAGE_PRIOR_SEASONS[int(season)]
    later = [s for s in prior_seasons if s >= int(season)]
    if later:
        raise ForecastAssemblyError(
            f"Shrinkage prior seasons {later} are not strictly earlier than {season}"
        )

    if prior_seasons == (2021,):
        if prior_2021 is None:
            raise ForecastAssemblyError(
                "Season 2022 needs 2021's realized-only events; none were supplied"
            )
        return prior_2021.copy()

    frames = []
    for prior in prior_seasons:
        ledger = ledgers.get(int(prior))
        if ledger is None:
            raise ForecastAssemblyError(f"No ledger loaded for prior season {prior}")
        if OBSERVED_COLUMN not in ledger.columns:
            raise ForecastAssemblyError(
                f"Prior-season {prior} ledger has no {OBSERVED_COLUMN!r} column; it is "
                "not a walk-forward ledger and cannot fit a shrinkage estimator"
            )
        frames.append(ledger[ledger[OBSERVED_COLUMN].notna()])
    return pd.concat(frames, ignore_index=True)


def build_season_windows(
    ledger: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = FORECAST_CUTOFFS,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """`(ordered events, windows with features, audit)` for one season.

    Runs the study's own causality and target cross-checks on the way
    through: `build_windows` re-proves the window structure, and
    `assert_targets_match_independent_recomputation` re-derives both targets
    by a second, independent code path.
    """
    resolved, exclusion = select_scored_eligible_events(ledger)
    ordered = order_eligible_events(resolved)
    windows = build_windows(ordered, cutoffs=cutoffs, horizons=horizons)
    assert_targets_match_independent_recomputation(ordered, windows)

    feature_frames = [build_window_features(ordered, cutoff) for cutoff in cutoffs]
    features = pd.concat([f for f in feature_frames if not f.empty], ignore_index=True)

    # `many_to_one`: features are one row per (batter, season, cutoff), while a
    # window exists per (batter, season, cutoff, HORIZON) -- so one feature row
    # legitimately serves every horizon sharing that cutoff.
    merged = windows.merge(
        features,
        on=["batter", "season", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    unmatched = int(merged["std_realized_rv_per_100"].isna().sum())
    if unmatched:
        raise ForecastAssemblyError(
            f"{unmatched} window(s) found no pre-cutoff feature row; a window cannot "
            "exist without the history that defines it"
        )

    audit = {
        "resolved_event_exclusion": exclusion,
        "inclusion_by_cell": summarize_inclusion(
            ordered, cutoffs=cutoffs, horizons=horizons
        ).to_dict(orient="records"),
    }
    return ordered, merged, audit


def build_ladder_for_season(
    windows: pd.DataFrame, *, prior_events: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every AVAILABLE rung of the frozen ladder, plus what was unavailable and why.

    When the prior events carry a deserved column the whole ladder is built
    by `forecast.baselines.build_baseline_predictions` -- the same matched
    estimator for both quantities. When they do not (2022), the four
    realized-only rungs are built from the same primitives and
    `shrunk_deserved_persistence` is emitted as an all-NaN column with a
    recorded reason, so downstream tables show a gap rather than a number.
    """
    has_deserved = DESERVED_COLUMN in prior_events.columns

    if has_deserved:
        out = build_baseline_predictions(
            windows, prior_events=prior_events, predicted_season=int(season)
        )
        realized_fit = fit_shrinkage(
            prior_events, value_column=OBSERVED_COLUMN, quantity="realized"
        )
        deserved_fit: ShrinkageFit | None = fit_shrinkage(
            prior_events, value_column=DESERVED_COLUMN, quantity="deserved"
        )
    else:
        realized_fit = fit_shrinkage(
            prior_events, value_column=OBSERVED_COLUMN, quantity="realized"
        )
        assert_fit_is_causal(realized_fit, int(season))
        deserved_fit = None
        out = windows.copy()
        out["league_mean"] = realized_fit.prior_mean
        out["raw_realized_persistence"] = out["std_realized_rv_per_100"]
        out["raw_deserved_persistence"] = out["std_deserved_rv_per_100"]
        out["shrunk_realized_persistence"] = realized_fit.apply(
            out["std_realized_rv_per_100"], out["n_resolved_bbe_through_cutoff"]
        )
        # Structurally unavailable, not missing data. Emitted so every
        # artifact has the column and can report the gap explicitly.
        out["shrunk_deserved_persistence"] = np.nan

    missing = [rung for rung in BASELINE_LADDER if rung not in out.columns]
    if missing:
        raise ForecastAssemblyError(f"Ladder is incomplete for {season}: missing {missing}")

    record: dict[str, Any] = {
        "season": int(season),
        "shrinkage_prior_seasons": list(SHRINKAGE_PRIOR_SEASONS[int(season)]),
        "matched_fit_sample": has_deserved,
        "rung_availability": RUNG_AVAILABILITY[int(season)],
        "realized_shrinkage": shrinkage_fit_record(realized_fit),
        "deserved_shrinkage": (
            shrinkage_fit_record(deserved_fit) if deserved_fit is not None else None
        ),
    }
    return out, record


def build_prediction_table(
    *,
    seasons: tuple[int, ...] = FORECAST_ANALYSIS_SEASONS,
    data_dir: Path = FORECAST_DATA_DIR,
    cutoffs: tuple[int, ...] = FORECAST_CUTOFFS,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    development_input: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The one table every downstream analysis reads, plus its assembly audit."""
    assert_forecast_seasons_allowed(list(seasons))
    # Ledgers for the analysis seasons AND for the strictly-earlier seasons
    # their shrinkage is fit on. 2021 is excluded here on purpose: it has no
    # ledger and never will (see the module docstring).
    needed_ledgers = set(int(s) for s in seasons)
    for season in seasons:
        needed_ledgers.update(p for p in SHRINKAGE_PRIOR_SEASONS[int(season)] if p != 2021)
    ledgers = {s: load_ledger(s, data_dir=data_dir) for s in sorted(needed_ledgers)}
    needs_2021 = any(SHRINKAGE_PRIOR_SEASONS[int(s)] == (2021,) for s in seasons)
    prior_2021 = (
        load_prior_2021_events(data_dir=data_dir, development_input=development_input)
        if needs_2021
        else None
    )

    frames: list[pd.DataFrame] = []
    audit: dict[str, Any] = {"seasons": {}, "ladders": {}}
    for season in seasons:
        ordered, windows, season_audit = build_season_windows(
            ledgers[int(season)], cutoffs=cutoffs, horizons=horizons
        )
        del ordered
        prior_events = prior_events_for(int(season), ledgers=ledgers, prior_2021=prior_2021)
        with_ladder, ladder_record = build_ladder_for_season(
            windows, prior_events=prior_events, season=int(season)
        )
        frames.append(with_ladder)
        audit["seasons"][str(season)] = season_audit
        audit["ladders"][str(season)] = ladder_record
        logger.info("Season %d: %d windows assembled", season, len(with_ladder))

    table = pd.concat(frames, ignore_index=True)
    audit["rung_availability_by_season"] = {str(s): RUNG_AVAILABILITY[int(s)] for s in seasons}
    audit["n_windows"] = int(len(table))
    return table, audit
