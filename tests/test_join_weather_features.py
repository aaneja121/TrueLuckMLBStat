"""Tests for the Contact Luck v0.5 per-play weather-feature join pipeline.

No test touches the network -- synthetic game-weather-shaped fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.data.join_weather_features import (
    JoinWeatherFeaturesError,
    add_venue_air_density_anomaly,
    build_weather_join_report,
    compute_venue_air_density_baseline,
    join_weather_features,
)


def _game_weather_row(**overrides) -> dict:
    base = {
        "game_pk": 1,
        "weather_status": "ok",
        "weather_match_quality": "good",
        "weather_time_offset_minutes": 10.0,
        "station_distance_km": 7.3,
        "roof_status": "outdoor_open_air",
        "effective_temperature_c": 24.0,
        "effective_relative_humidity_pct": 60.0,
        "effective_pressure_hpa": 1005.0,
        "effective_wind_speed_mps": 3.5,
        "effective_wind_movement_bearing_degrees": 0.0,
        "air_density_kg_m3": 1.18,
    }
    base.update(overrides)
    return base


@pytest.fixture
def play_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 3, 4],
            "spray_angle_approx": [0.0, 30.0, -10.0, 0.0, np.nan],
            "season": [2024] * 5,
            "venue_name": ["Fenway Park"] * 2 + ["Tropicana Field", "Rogers Centre", "Fenway Park"],
        }
    )


@pytest.fixture
def game_weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _game_weather_row(game_pk=1),
            _game_weather_row(
                game_pk=2,
                weather_status="effective_conditions_unavailable_indoor",
                weather_match_quality=None,
                roof_status="fixed_indoor",
                effective_temperature_c=None,
                effective_relative_humidity_pct=None,
                effective_pressure_hpa=None,
                effective_wind_speed_mps=0.0,
                effective_wind_movement_bearing_degrees=None,
                air_density_kg_m3=None,
            ),
            _game_weather_row(
                game_pk=3,
                weather_status="venue_environment_unavailable",
                weather_match_quality=None,
                roof_status="venue_environment_unavailable",
                effective_temperature_c=None,
                effective_relative_humidity_pct=None,
                effective_pressure_hpa=None,
                effective_wind_speed_mps=None,
                effective_wind_movement_bearing_degrees=None,
                air_density_kg_m3=None,
                station_distance_km=None,
            ),
            _game_weather_row(game_pk=4),
        ]
    )


def test_join_preserves_every_row(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    assert len(joined) == len(play_df)


def test_join_produces_no_duplicate_rows(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    assert len(joined) == len(play_df)
    assert joined.index.is_unique


def test_ok_row_has_effective_weather_and_wind_components(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row = joined.iloc[0]  # game_pk=1, spray_angle=0, wind bearing=0 -> pure following
    assert row["has_weather_data"]
    assert row["has_effective_weather"]
    assert row["following_wind_mps"] == pytest.approx(3.5)
    assert row["headwind_mps"] == pytest.approx(-3.5)
    assert row["crosswind_mps"] == pytest.approx(0.0, abs=1e-9)


def test_wind_components_vary_with_spray_angle(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row0 = joined.iloc[0]  # spray_angle=0
    row1 = joined.iloc[1]  # spray_angle=30, same game/wind
    assert row0["following_wind_mps"] != row1["following_wind_mps"]
    assert row1["crosswind_mps"] != 0.0


def test_indoor_row_suppresses_wind_and_marks_uncertain(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row = joined.iloc[2]  # game_pk=2, Tropicana, indoor
    assert row["indoor_indicator"]
    assert row["following_wind_mps"] == 0.0
    assert row["headwind_mps"] == 0.0
    assert row["crosswind_mps"] == 0.0
    assert not row["has_effective_weather"]
    assert row["weather_uncertain"]


def test_venue_unavailable_row_not_marked_uncertain(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row = joined.iloc[3]  # game_pk=3, venue_environment_unavailable
    assert not row["has_weather_data"]
    assert not row["weather_uncertain"]  # not applicable, not "uncertain"


def test_missing_spray_angle_leaves_wind_components_unavailable(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row = joined.iloc[4]  # game_pk=4, missing spray_angle_approx
    assert pd.isna(row["following_wind_mps"])
    assert pd.isna(row["headwind_mps"])
    assert pd.isna(row["crosswind_mps"])


def test_air_density_deviation_from_reference(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    row = joined.iloc[0]
    assert row["air_density_deviation_from_reference"] == pytest.approx(1.18 - 1.225, abs=1e-6)


def test_missing_required_columns_raises():
    df = pd.DataFrame({"game_pk": [1]})
    with pytest.raises(JoinWeatherFeaturesError, match="spray_angle_approx"):
        join_weather_features(df, pd.DataFrame({"game_pk": [1]}))


def test_duplicate_game_pk_in_game_weather_rejected(play_df, game_weather_df):
    dup = pd.concat([game_weather_df, game_weather_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(JoinWeatherFeaturesError, match="duplicate"):
        join_weather_features(play_df, dup)


def test_coverage_report(play_df, game_weather_df):
    joined = join_weather_features(play_df, game_weather_df)
    report = build_weather_join_report(joined)
    assert report.total_rows == 5
    # game_pk 1 (x2 rows) and game_pk 4 (missing spray angle, but still "ok" at
    # the game level) all have effective weather -- 3 of 5 rows.
    assert report.rows_with_effective_weather == 3
    assert report.effective_coverage_rate == pytest.approx(3 / 5)
    assert sum(report.counts_by_status.values()) == 5


# ---------------------------------------------------------------------------
# Version 0.5.1: per-venue air-density baseline / anomaly
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_season_density_df() -> pd.DataFrame:
    rows = []
    # Venue 1: consistently low density (Coors-like) across training seasons.
    for season in (2021, 2022, 2023):
        for i in range(40):
            rows.append(
                {
                    "venue_id": 1,
                    "season": season,
                    "has_effective_weather": True,
                    "air_density_kg_m3": 0.98 + 0.01 * (i % 5 - 2) / 5,
                }
            )
    # Venue 2: consistently high density.
    for season in (2021, 2022, 2023):
        for i in range(40):
            rows.append(
                {
                    "venue_id": 2,
                    "season": season,
                    "has_effective_weather": True,
                    "air_density_kg_m3": 1.20 + 0.01 * (i % 5 - 2) / 5,
                }
            )
    # Venue 3: too few training samples to be reliable.
    for _i in range(5):
        rows.append(
            {
                "venue_id": 3,
                "season": 2021,
                "has_effective_weather": True,
                "air_density_kg_m3": 1.10,
            }
        )
    # Validation-season rows for venue 1 -- must NOT influence the baseline.
    for _i in range(10):
        rows.append(
            {
                "venue_id": 1,
                "season": 2024,
                "has_effective_weather": True,
                "air_density_kg_m3": 0.50,
            }
        )
    return pd.DataFrame(rows)


def test_venue_baseline_computed_from_training_seasons_only(multi_season_density_df):
    baseline = compute_venue_air_density_baseline(multi_season_density_df, (2021, 2022, 2023))
    # If 2024's density=0.50 rows leaked in, venue 1's mean would be pulled
    # far below ~0.98 -- confirms the baseline is train-only.
    assert baseline[1] == pytest.approx(0.98, abs=0.01)
    assert baseline[2] == pytest.approx(1.20, abs=0.01)


def test_venue_baseline_excludes_low_sample_venues(multi_season_density_df):
    baseline = compute_venue_air_density_baseline(
        multi_season_density_df, (2021, 2022, 2023), min_samples=30
    )
    assert 3 not in baseline


def test_venue_baseline_includes_venue_with_enough_samples(multi_season_density_df):
    baseline = compute_venue_air_density_baseline(
        multi_season_density_df, (2021, 2022, 2023), min_samples=30
    )
    assert 1 in baseline
    assert 2 in baseline


def test_venue_anomaly_is_zero_at_the_baseline():
    df = pd.DataFrame({"venue_id": [1, 1], "air_density_kg_m3": [0.98, 1.20]})
    out = add_venue_air_density_anomaly(df, {1: 0.98})
    assert out["air_density_venue_anomaly_kg_m3"].iloc[0] == pytest.approx(0.0)
    assert out["air_density_venue_anomaly_kg_m3"].iloc[1] == pytest.approx(0.22)


def test_venue_anomaly_is_nan_for_venue_without_baseline():
    df = pd.DataFrame({"venue_id": [1, 2], "air_density_kg_m3": [0.98, 1.20]})
    out = add_venue_air_density_anomaly(df, {1: 0.98})  # venue 2 not in baseline
    assert pd.isna(out["air_density_venue_anomaly_kg_m3"].iloc[1])


def test_venue_anomaly_is_nan_when_density_missing():
    df = pd.DataFrame({"venue_id": [1], "air_density_kg_m3": [np.nan]})
    out = add_venue_air_density_anomaly(df, {1: 0.98})
    assert pd.isna(out["air_density_venue_anomaly_kg_m3"].iloc[0])


def test_venue_anomaly_isolates_day_specific_weather_from_altitude(multi_season_density_df):
    # End-to-end check of the design intent: venue 1's raw density is always
    # low (~0.98, persistent altitude signature); its ANOMALY should be
    # small/centered (day-specific variation), not itself persistently low.
    baseline = compute_venue_air_density_baseline(multi_season_density_df, (2021, 2022, 2023))
    out = add_venue_air_density_anomaly(multi_season_density_df, baseline)
    venue_1_train = out[(out["venue_id"] == 1) & (out["season"] != 2024)]
    assert venue_1_train["air_density_venue_anomaly_kg_m3"].abs().max() < 0.05
    assert venue_1_train["air_density_kg_m3"].mean() < 1.0  # raw density stays persistently low
