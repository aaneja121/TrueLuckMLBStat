"""Contact Forecast R1: assembling the paired prediction table.

The load-bearing claim here is negative: 2022 has NO causal prior-season
deserved values, so it gets no `shrunk_deserved_persistence` rung, and that
gap is carried into every artifact as an explicit unavailability rather than
filled with a number. The rest of the tests check that the ladder is built
from strictly earlier seasons and that a window's features are joined to it
correctly across horizons.

Synthetic data only -- no ledger file is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import assemble as asm
from forecast.forecast_config import (
    BASELINE_LADDER,
    FORECAST_ANALYSIS_SEASONS,
    OBSERVED_COLUMN,
)

_BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")


def make_prior_events(
    *,
    seasons: tuple[int, ...],
    n_batters: int = 40,
    n_events: int = 220,
    seed: int = 2,
    with_deserved: bool,
) -> pd.DataFrame:
    """Per-batted-ball prior-season rows, with or without a deserved column."""
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        for batter in range(n_batters):
            truth = rng.normal(0.05, 0.10)
            observed = rng.normal(truth, 0.5, n_events)
            data = {
                "batter": batter,
                "season": season,
                OBSERVED_COLUMN: observed,
            }
            if with_deserved:
                data["baseline_expected_contact_run_value"] = rng.normal(truth, 0.2, n_events)
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def make_windows(*, season: int, n: int = 60, cutoff: int = 100, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "window_id": [f"w{i}-{season}" for i in range(n)],
            "batter": np.arange(n),
            "season": season,
            "cutoff": cutoff,
            "horizon": 100,
            "n_resolved_bbe_through_cutoff": float(cutoff),
            "std_realized_rv_per_100": rng.normal(5.0, 8.0, n),
            "std_deserved_rv_per_100": rng.normal(5.0, 5.0, n),
            "target_realized_rv_per_100": rng.normal(5.0, 7.0, n),
        }
    )


# --------------------------------------------------------------------------
# The 2022 gap
# --------------------------------------------------------------------------


def test_2022_gets_no_shrunk_deserved_rung() -> None:
    """2021 has no contact model, so the rung cannot exist -- and is not invented."""
    windows = make_windows(season=2022)
    prior = make_prior_events(seasons=(2021,), with_deserved=False)
    built, record = asm.build_ladder_for_season(windows, prior_events=prior, season=2022)
    assert built["shrunk_deserved_persistence"].isna().all()
    assert record["rung_availability"]["shrunk_deserved_persistence"] == "unavailable"
    assert "no causal contact model" in record["rung_availability"]["reason"]
    assert record["deserved_shrinkage"] is None
    assert record["matched_fit_sample"] is False


def test_2022_still_gets_every_rung_that_needs_only_realized_values() -> None:
    windows = make_windows(season=2022)
    prior = make_prior_events(seasons=(2021,), with_deserved=False)
    built, _ = asm.build_ladder_for_season(windows, prior_events=prior, season=2022)
    for rung in BASELINE_LADDER:
        assert rung in built.columns
    for rung in (
        "league_mean",
        "raw_realized_persistence",
        "raw_deserved_persistence",
        "shrunk_realized_persistence",
    ):
        assert built[rung].notna().all(), rung


def test_the_missing_rung_is_never_filled_with_the_league_mean() -> None:
    """The obvious wrong fix -- imputing the mean -- would look like evidence."""
    windows = make_windows(season=2022)
    prior = make_prior_events(seasons=(2021,), with_deserved=False)
    built, _ = asm.build_ladder_for_season(windows, prior_events=prior, season=2022)
    assert not (built["shrunk_deserved_persistence"] == built["league_mean"]).any()
    assert not built["shrunk_deserved_persistence"].notna().any()


def test_availability_is_recorded_for_every_analysis_season() -> None:
    assert set(asm.RUNG_AVAILABILITY) == set(FORECAST_ANALYSIS_SEASONS)
    assert asm.RUNG_AVAILABILITY[2023]["shrunk_deserved_persistence"] == "available"
    assert asm.RUNG_AVAILABILITY[2024]["shrunk_deserved_persistence"] == "available"


# --------------------------------------------------------------------------
# Seasons with a scored prior season
# --------------------------------------------------------------------------


def test_a_scored_prior_season_gives_the_full_matched_ladder() -> None:
    windows = make_windows(season=2023)
    prior = make_prior_events(seasons=(2022,), with_deserved=True)
    built, record = asm.build_ladder_for_season(windows, prior_events=prior, season=2023)
    for rung in BASELINE_LADDER:
        assert built[rung].notna().all(), rung
    assert record["matched_fit_sample"] is True
    assert record["realized_shrinkage"]["fit_seasons"] == [2022]
    assert record["deserved_shrinkage"]["fit_seasons"] == [2022]


def test_both_shrinkage_fits_see_the_identical_prior_sample() -> None:
    """Matched shrinkage means one fit sample, not two similar ones."""
    windows = make_windows(season=2024)
    prior = make_prior_events(seasons=(2022, 2023), with_deserved=True)
    _, record = asm.build_ladder_for_season(windows, prior_events=prior, season=2024)
    assert (
        record["realized_shrinkage"]["n_fit_events"] == record["deserved_shrinkage"]["n_fit_events"]
    )
    assert (
        record["realized_shrinkage"]["fit_seasons"] == record["deserved_shrinkage"]["fit_seasons"]
    )


def test_a_ladder_fit_on_the_scored_season_is_refused() -> None:
    windows = make_windows(season=2023)
    prior = make_prior_events(seasons=(2023,), with_deserved=True)
    with pytest.raises(Exception, match="strictly earlier|not strictly earlier"):
        asm.build_ladder_for_season(windows, prior_events=prior, season=2023)


# --------------------------------------------------------------------------
# Prior-season routing
# --------------------------------------------------------------------------


def test_every_shrinkage_prior_season_is_strictly_earlier() -> None:
    for season, priors in asm.SHRINKAGE_PRIOR_SEASONS.items():
        assert all(p < season for p in priors), (season, priors)


def test_shrinkage_priors_cover_exactly_the_analysis_seasons() -> None:
    assert set(asm.SHRINKAGE_PRIOR_SEASONS) == set(FORECAST_ANALYSIS_SEASONS)


def test_2022_requires_the_2021_realized_only_frame() -> None:
    with pytest.raises(asm.ForecastAssemblyError, match="2021"):
        asm.prior_events_for(2022, ledgers={}, prior_2021=None)


def test_a_missing_prior_ledger_is_an_error_not_a_silent_skip() -> None:
    """2024 needs both 2022 and 2023; having only one is not "close enough"."""
    ledger_2022 = pd.DataFrame({"batter": [1], "season": [2022], OBSERVED_COLUMN: [0.1]})
    with pytest.raises(asm.ForecastAssemblyError, match="No ledger loaded"):
        asm.prior_events_for(2024, ledgers={2022: ledger_2022}, prior_2021=None)


def test_a_prior_frame_without_run_values_is_refused() -> None:
    with pytest.raises(asm.ForecastAssemblyError, match="not a walk-forward ledger"):
        asm.prior_events_for(2023, ledgers={2022: pd.DataFrame()}, prior_2021=None)


def test_prior_events_drop_unresolved_rows() -> None:
    ledger = pd.DataFrame(
        {
            "batter": [1, 1, 2],
            "season": 2022,
            OBSERVED_COLUMN: [0.1, np.nan, 0.2],
        }
    )
    events = asm.prior_events_for(2023, ledgers={2022: ledger}, prior_2021=None)
    assert len(events) == 2
    assert events[OBSERVED_COLUMN].notna().all()


def test_loading_an_unauthorized_season_is_refused() -> None:
    from forecast.forecast_config import ForecastSeasonError

    with pytest.raises(ForecastSeasonError, match="sealed"):
        asm.load_ledger(2025)
