"""Tests for the v1.2 per-venue station-wind check (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models import venue_wind_support as vws

MPS_PER_MPH = 0.44704


def _synthetic_rows(
    *,
    venue_id: int = 1,
    ft_per_mph: float = 2.0,
    n_games: int = 60,
    rows_per_game: int = 10,
    noise_ft: float = 5.0,
    seed: int = 0,
    season: int = 2022,
) -> pd.DataFrame:
    """Fly balls whose distance depends on wind by a known slope.

    Wind and density vary by game (as real weather does); launch speed/angle
    vary by row and shift distance, so the cell fixed effects matter.
    """
    rng = np.random.default_rng(seed)
    n = n_games * rows_per_game
    game = np.repeat(np.arange(n_games) + venue_id * 100_000, rows_per_game)
    wind_mps = np.repeat(rng.normal(0.0, 3.0, n_games), rows_per_game)
    density = np.repeat(rng.normal(1.18, 0.03, n_games), rows_per_game)
    ls = rng.uniform(96.0, 100.0, n)  # two 2-mph cells
    la = rng.uniform(26.0, 30.0, n)  # two 2-degree cells
    dist = (
        300.0
        + 4.0 * (ls - 96.0)
        + 2.0 * (la - 26.0)
        + ft_per_mph * wind_mps / MPS_PER_MPH
        - 145.0 * (density - 1.18)
        + rng.normal(0.0, noise_ft, n)
    )
    return pd.DataFrame(
        {
            "event_id": [f"{venue_id}-{i}" for i in range(n)],
            "season": season,
            "game_pk": game,
            "venue_id": venue_id,
            "venue_name": f"Venue {venue_id}",
            "bb_type": "fly_ball",
            "eligible_for_training": True,
            "hit_distance_sc": dist,
            "launch_speed": ls,
            "launch_angle": la,
            "roof_status": "outdoor_open_air",
            "air_density_kg_m3": density,
            "following_wind_mps": wind_mps,
        }
    )


# --- row selection -----------------------------------------------------------


def test_select_rows_applies_every_frozen_filter() -> None:
    base = _synthetic_rows(n_games=2, rows_per_game=1)
    row = base.iloc[[0]]
    variants = {
        "keep": row.assign(),
        "ground_ball": row.assign(bb_type="ground_ball"),
        "ineligible": row.assign(eligible_for_training=False),
        "roof_closed": row.assign(roof_status="retractable_roof_closed"),
        "indoor": row.assign(roof_status="fixed_indoor"),
        "roof_unknown": row.assign(roof_status="roof_status_unknown"),
        "no_distance": row.assign(hit_distance_sc=np.nan),
        "no_density": row.assign(air_density_kg_m3=np.nan),
        "no_wind": row.assign(following_wind_mps=np.nan),
        "slow": row.assign(launch_speed=89.9),
        "fast": row.assign(launch_speed=115.1),
        "low": row.assign(launch_angle=19.9),
        "high": row.assign(launch_angle=40.1),
        "roof_open": row.assign(roof_status="retractable_roof_open"),
    }
    df = pd.concat([v.assign(event_id=k) for k, v in variants.items()], ignore_index=True)
    kept = set(vws.select_wind_check_rows(df)["event_id"])
    assert kept == {"keep", "roof_open"}


def test_select_rows_handles_nullable_eligibility() -> None:
    df = _synthetic_rows(n_games=3, rows_per_game=1)
    df["eligible_for_training"] = pd.array([True, pd.NA, False], dtype="boolean")
    assert len(vws.select_wind_check_rows(df)) == 1


@pytest.mark.parametrize("season", [2025, 2026])
def test_select_rows_refuses_non_development_seasons(season: int) -> None:
    df = _synthetic_rows(n_games=2, rows_per_game=1, season=season)
    with pytest.raises(ValueError):
        vws.select_wind_check_rows(df)


# --- estimator ------------------------------------------------------------------


def test_estimator_recovers_known_wind_slope_despite_cell_effects() -> None:
    df = vws.select_wind_check_rows(_synthetic_rows(ft_per_mph=2.0, n_games=400, noise_ft=1.0))
    est = vws.estimate_venue_coefficients(df)
    assert est["ft_per_mph_following_wind"] == pytest.approx(2.0, abs=0.1)
    assert est["ft_per_001_density"] == pytest.approx(-1.45, abs=0.1)


def test_estimator_drops_cells_with_fewer_than_min_rows() -> None:
    df = vws.select_wind_check_rows(_synthetic_rows(n_games=50))
    lone = df.iloc[[0]].assign(event_id="lone", launch_speed=112.5, launch_angle=38.5)
    demeaned = vws.demean_within_cells(pd.concat([df, lone], ignore_index=True))
    assert "lone" not in set(demeaned["event_id"])
    assert len(demeaned) == len(df)


# --- classification ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_plays", "n_games", "lo", "hi", "expected"),
    [
        (500, 50, 0.1, 0.9, vws.STATUS_WIND_SUPPORTED),
        (500, 50, -0.9, -0.1, vws.STATUS_WIND_CONTRADICTED),
        (500, 50, -0.1, 0.9, vws.STATUS_INSUFFICIENT_EVIDENCE),
        (500, 50, 0.0, 0.9, vws.STATUS_INSUFFICIENT_EVIDENCE),
        (99, 50, 0.1, 0.9, vws.STATUS_INSUFFICIENT_EVIDENCE),
        (500, 19, 0.1, 0.9, vws.STATUS_INSUFFICIENT_EVIDENCE),
        (500, 50, float("nan"), float("nan"), vws.STATUS_INSUFFICIENT_EVIDENCE),
    ],
)
def test_classify(n_plays: int, n_games: int, lo: float, hi: float, expected: str) -> None:
    assert vws.classify_venue_wind(n_plays, n_games, lo, hi) == expected


def test_only_supported_status_makes_the_check_applicable() -> None:
    assert vws.wind_check_applies(vws.STATUS_WIND_SUPPORTED)
    assert not vws.wind_check_applies(vws.STATUS_WIND_CONTRADICTED)
    assert not vws.wind_check_applies(vws.STATUS_INSUFFICIENT_EVIDENCE)


# --- end to end -------------------------------------------------------------------


def test_run_classifies_synthetic_venues_by_direction() -> None:
    df = pd.concat(
        [
            _synthetic_rows(venue_id=1, ft_per_mph=3.0, seed=1),
            _synthetic_rows(venue_id=2, ft_per_mph=-3.0, seed=2),
            _synthetic_rows(venue_id=3, ft_per_mph=0.0, noise_ft=40.0, n_games=25, seed=3),
            _synthetic_rows(venue_id=4, ft_per_mph=3.0, n_games=5, seed=4),
        ],
        ignore_index=True,
    )
    table = vws.run_venue_wind_check(df, n_reps=200).set_index("venue_id")
    assert table.loc[1, "status"] == vws.STATUS_WIND_SUPPORTED
    assert table.loc[2, "status"] == vws.STATUS_WIND_CONTRADICTED
    assert table.loc[3, "status"] == vws.STATUS_INSUFFICIENT_EVIDENCE
    assert table.loc[4, "status"] == vws.STATUS_INSUFFICIENT_EVIDENCE  # 5 games < 20
    assert bool(table.loc[1, "wind_check_applies"]) is True
    assert not table.loc[[2, 3, 4], "wind_check_applies"].any()
    for col in [
        "venue_name",
        "n_plays",
        "n_games",
        "ft_per_mph_following_wind",
        "ci_low",
        "ci_high",
        "bootstrap_se",
        "bonferroni_ci_low",
        "bonferroni_ci_high",
    ]:
        assert col in table.columns


def test_run_is_deterministic_for_a_fixed_seed() -> None:
    df = _synthetic_rows(ft_per_mph=1.0, noise_ft=20.0)
    a = vws.run_venue_wind_check(df, n_reps=100)
    b = vws.run_venue_wind_check(df, n_reps=100)
    pd.testing.assert_frame_equal(a, b)


def test_bonferroni_interval_is_wider_than_the_primary_one_with_many_venues() -> None:
    df = pd.concat(
        [_synthetic_rows(venue_id=v, ft_per_mph=1.0, noise_ft=20.0, seed=v) for v in range(1, 6)],
        ignore_index=True,
    )
    t = vws.run_venue_wind_check(df, n_reps=200)
    assert (t["bonferroni_ci_high"] - t["bonferroni_ci_low"] > t["ci_high"] - t["ci_low"]).all()
