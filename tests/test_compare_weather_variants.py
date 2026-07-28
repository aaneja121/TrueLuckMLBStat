"""Tests for the Contact Luck v0.5.1 corrected weather-variant comparison.

No test touches the network -- synthetic multi-season, weather-joined
fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import (
    CLASS_ORDER,
    LEAKAGE_COLUMNS,
    ProtectedSeasonError,
    assert_seasons_allowed,
)
from mlb_luck_score.data.join_weather_features import join_weather_features
from mlb_luck_score.features.build_contact_features import (
    WEATHER_COMPONENTS_ONLY_CATEGORICAL_FEATURES,
    WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES,
    WEATHER_DENSITY_ANOMALY_CATEGORICAL_FEATURES,
    WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES,
    WEATHER_DENSITY_ONLY_CATEGORICAL_FEATURES,
    WEATHER_DENSITY_ONLY_NUMERIC_FEATURES,
)
from mlb_luck_score.models.compare_geometry_aware import compute_paired_bootstrap
from mlb_luck_score.models.compare_weather_variants import (
    ALL_V051_CANDIDATES,
    VARIANT_COMPONENTS_ONLY,
    VARIANT_DENSITY_ANOMALY,
    VARIANT_DENSITY_ONLY,
    VARIANT_SELECTED_PRODUCTION_BASELINE,
    _prepare_variant_columns,
    check_standardized_stability_for_candidate,
    recommend_variant_adoption,
    run_perturbation_checks,
    run_weather_variant_comparison,
)
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult


def _synthetic_venue_joined_season_df(
    season: int, rng: np.random.Generator, *, n_per_class: int = 40
) -> pd.DataFrame:
    profiles = {
        "out": (75.0, 20.0, 100.0),
        "single": (92.0, 8.0, 180.0),
        "double": (98.0, 18.0, 300.0),
        "triple": (100.0, 15.0, 340.0),
        "home_run": (105.0, 28.0, 400.0),
    }
    rows = []
    for outcome, (speed, angle, dist) in profiles.items():
        for i in range(n_per_class):
            day = 1 + (i % 27)
            game_pk = season * 1000 + day
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": f"{season}-06-{day:02d}",
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.uniform(-44.0, 44.0),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "fly_ball" if outcome in ("home_run", "double") else "line_drive",
                    "stand": "R" if i % 2 == 0 else "L",
                    "venue": "Fenway Park",
                    "venue_id": 3.0,
                    "venue_name": "Fenway Park",
                    "has_venue_metadata": True,
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def weather_joined_df() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    venue_df = pd.concat(
        [_synthetic_venue_joined_season_df(s, rng) for s in (2021, 2022, 2023, 2024)],
        ignore_index=True,
    )
    game_pks = venue_df["game_pk"].unique()
    gw_rng = np.random.default_rng(9)
    is_ok = gw_rng.choice([True, False], size=len(game_pks), p=[0.85, 0.15])
    game_weather = pd.DataFrame(
        {
            "game_pk": game_pks,
            "weather_status": np.where(is_ok, "ok", "effective_conditions_unavailable_indoor"),
            "weather_match_quality": np.where(is_ok, "good", None),
            "weather_time_offset_minutes": 10.0,
            "station_distance_km": 7.3,
            "roof_status": np.where(is_ok, "outdoor_open_air", "fixed_indoor"),
            "effective_temperature_c": np.where(is_ok, gw_rng.normal(22, 8, len(game_pks)), np.nan),
            "effective_relative_humidity_pct": np.where(
                is_ok, gw_rng.uniform(30, 90, len(game_pks)), np.nan
            ),
            "effective_pressure_hpa": np.where(
                is_ok, gw_rng.normal(1013, 8, len(game_pks)), np.nan
            ),
            "effective_wind_speed_mps": np.where(is_ok, gw_rng.uniform(0, 8, len(game_pks)), 0.0),
            "effective_wind_movement_bearing_degrees": np.where(
                is_ok, gw_rng.uniform(-90, 90, len(game_pks)), np.nan
            ),
            "air_density_kg_m3": np.where(is_ok, gw_rng.normal(1.18, 0.04, len(game_pks)), np.nan),
        }
    )
    return join_weather_features(venue_df, game_weather)


def test_weather_variant_features_are_not_leakage_columns():
    for feats in (
        WEATHER_DENSITY_ONLY_NUMERIC_FEATURES,
        WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES,
        WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES,
        WEATHER_DENSITY_ONLY_CATEGORICAL_FEATURES,
        WEATHER_COMPONENTS_ONLY_CATEGORICAL_FEATURES,
        WEATHER_DENSITY_ANOMALY_CATEGORICAL_FEATURES,
    ):
        assert set(feats).isdisjoint(LEAKAGE_COLUMNS)


def test_candidates_are_mutually_exclusive_on_density_representation():
    # density_only has air_density_kg_m3 but no temp/humidity/pressure;
    # components_only has temp/humidity/pressure but no air_density;
    # density_anomaly has neither raw density nor raw components.
    assert "air_density_kg_m3" in WEATHER_DENSITY_ONLY_NUMERIC_FEATURES
    assert "temperature_c" not in WEATHER_DENSITY_ONLY_NUMERIC_FEATURES
    assert "temperature_c" in WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES
    assert "air_density_kg_m3" not in WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES
    assert "air_density_venue_anomaly_kg_m3" in WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES
    assert "air_density_kg_m3" not in WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES
    assert "temperature_c" not in WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES


def test_comparison_rejects_missing_weather_columns():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="temperature_c"):
        run_weather_variant_comparison(df)


def test_all_variants_present(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba, _baseline = run_weather_variant_comparison(weather_joined_df)
    assert set(comparison.keys()) == {VARIANT_SELECTED_PRODUCTION_BASELINE, *ALL_V051_CANDIDATES}


def test_candidates_trained_independently_not_cumulatively(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba, _baseline = run_weather_variant_comparison(weather_joined_df)
    density_only_features = set(comparison[VARIANT_DENSITY_ONLY]["features"])
    components_only_features = set(comparison[VARIANT_COMPONENTS_ONLY]["features"])
    density_anomaly_features = set(comparison[VARIANT_DENSITY_ANOMALY]["features"])
    # No candidate's feature set should be a superset of another's weather
    # features -- they are independent hypotheses, not cumulative.
    assert "temperature_c" not in density_only_features
    assert "air_density_kg_m3" not in components_only_features
    assert "air_density_kg_m3" not in density_anomaly_features
    assert "temperature_c" not in density_anomaly_features


def test_identical_train_validation_rows_across_variants(
    weather_joined_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_weather_variants as cwv

    seen_row_counts: set[int] = set()
    seen_seasons: set[int] = set()
    original = cwv.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_row_counts.add(len(train_df))
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cwv, "train_model", _tracking_train_model)
    run_weather_variant_comparison(weather_joined_df)

    assert len(seen_row_counts) == 1
    assert seen_seasons == {2021, 2022, 2023}


def test_2025_still_rejected_for_variant_training():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed((2021, 2022, 2023, 2025))


def test_probability_columns_class_ordered_and_sum_to_one(weather_joined_df: pd.DataFrame):
    _comparison, _trained, proba_by_variant, _baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    for proba_df in proba_by_variant.values():
        assert list(proba_df.columns) == list(CLASS_ORDER)
        assert np.allclose(proba_df.to_numpy().sum(axis=1), 1.0, atol=1e-6)
        assert np.isfinite(proba_df.to_numpy()).all()


def test_venue_baseline_returned_and_reasonable(weather_joined_df: pd.DataFrame):
    _comparison, _trained, _proba, venue_baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    assert isinstance(venue_baseline, dict)
    # Fenway (the only venue in this synthetic fixture) should have a
    # reliable baseline given 3 training seasons x 5 classes x ~34 rows each.
    assert 3 in venue_baseline
    assert 1.0 < venue_baseline[3] < 1.4  # plausible air density range


# ---------------------------------------------------------------------------
# Controlled-perturbation checks
# ---------------------------------------------------------------------------


def test_run_perturbation_checks_returns_density_and_wind_for_density_only(weather_joined_df):
    comparison, trained_models, _proba, venue_baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    training_eligible = weather_joined_df[weather_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    from mlb_luck_score.data.join_weather_features import add_venue_air_density_anomaly

    val_df = add_venue_air_density_anomaly(val_df, venue_baseline)
    val_df = _prepare_variant_columns(val_df)

    results = run_perturbation_checks(
        trained_models[VARIANT_DENSITY_ONLY], val_df, VARIANT_DENSITY_ONLY
    )
    assert "density_direction" in results
    assert "wind_direction" in results
    assert all(isinstance(r, DirectionalCheckResult) for r in results.values())


def test_run_perturbation_checks_components_only_uses_temp_pressure_proxy(weather_joined_df):
    comparison, trained_models, _proba, venue_baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    training_eligible = weather_joined_df[weather_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    from mlb_luck_score.data.join_weather_features import add_venue_air_density_anomaly

    val_df = add_venue_air_density_anomaly(val_df, venue_baseline)
    val_df = _prepare_variant_columns(val_df)

    results = run_perturbation_checks(
        trained_models[VARIANT_COMPONENTS_ONLY], val_df, VARIANT_COMPONENTS_ONLY
    )
    assert (
        "temperature_c" in results["density_direction"].label
        or "implied-density" in results["density_direction"].label
    )
    # components_only has no single density column, so no Coors-specific check.
    assert "coors_field_effect" not in results


def test_check_standardized_stability_uses_genuine_standardized_predictions(weather_joined_df):
    comparison, trained_models, _proba, venue_baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    training_eligible = weather_joined_df[weather_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    from mlb_luck_score.data.join_weather_features import add_venue_air_density_anomaly

    val_df = add_venue_air_density_anomaly(val_df, venue_baseline)
    val_df = _prepare_variant_columns(val_df)

    stable = check_standardized_stability_for_candidate(
        trained_models[VARIANT_DENSITY_ONLY], val_df
    )
    assert stable is True


# ---------------------------------------------------------------------------
# Adoption rule: AND-combination, never solely log-loss
# ---------------------------------------------------------------------------


def _base_summary(log_loss: float) -> dict:
    return {
        "multiclass_log_loss": log_loss,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "weather_subgroups": {},
        "weather_coverage_rate": 0.9,
    }


def _passing_bootstrap() -> dict:
    return {
        "log_loss": {
            "metric": "log_loss",
            "point_estimate": -0.0003,
            "ci_low": -0.0006,
            "ci_high": -0.0001,
            "n_reps": 10,
            "seed": 1,
            "resampling_unit": "game_pk",
        }
    }


def test_adoption_requires_perturbation_checks_even_if_log_loss_improves():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)  # tiny but real improvement, like real v0.5 findings
    comparison = {
        VARIANT_SELECTED_PRODUCTION_BASELINE: baseline,
        VARIANT_DENSITY_ONLY: candidate,
    }
    bootstrap = {VARIANT_DENSITY_ONLY: _passing_bootstrap()}
    # Perturbation check FAILS (backwards direction) -- must block adoption
    # even though log loss + bootstrap both look favorable.
    failing_check = DirectionalCheckResult(
        label="density direction",
        outcome_class="home_run",
        low_label="low",
        high_label="high",
        mean_prob_low=0.03,
        mean_prob_high=0.05,
        delta=0.02,
        expect_high_greater=False,
        passed=False,
        sample_size=1000,
    )
    perturbation = {VARIANT_DENSITY_ONLY: {"density_direction": failing_check}}
    standardized_stable = {VARIANT_DENSITY_ONLY: True}

    rec = recommend_variant_adoption(
        comparison,
        bootstrap,
        perturbation,
        standardized_stable,
        candidate_variants=(VARIANT_DENSITY_ONLY,),
    )
    assert rec["per_candidate"][VARIANT_DENSITY_ONLY]["passes_all_automated_criteria"] is False
    assert (
        "density_direction" in rec["per_candidate"][VARIANT_DENSITY_ONLY]["perturbation_failures"]
    )
    assert rec["recommend_adopt_any_v051_candidate"] is False


def test_adoption_passes_when_everything_including_perturbation_checks_out():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)
    comparison = {
        VARIANT_SELECTED_PRODUCTION_BASELINE: baseline,
        VARIANT_DENSITY_ANOMALY: candidate,
    }
    bootstrap = {VARIANT_DENSITY_ANOMALY: _passing_bootstrap()}
    passing_check = DirectionalCheckResult(
        label="density direction",
        outcome_class="home_run",
        low_label="low",
        high_label="high",
        mean_prob_low=0.05,
        mean_prob_high=0.03,
        delta=-0.02,
        expect_high_greater=False,
        passed=True,
        sample_size=1000,
    )
    perturbation = {
        VARIANT_DENSITY_ANOMALY: {
            "density_direction": passing_check,
            "wind_direction": passing_check,
        }
    }
    standardized_stable = {VARIANT_DENSITY_ANOMALY: True}

    rec = recommend_variant_adoption(
        comparison,
        bootstrap,
        perturbation,
        standardized_stable,
        candidate_variants=(VARIANT_DENSITY_ANOMALY,),
    )
    assert rec["per_candidate"][VARIANT_DENSITY_ANOMALY]["passes_all_automated_criteria"] is True
    assert rec["recommend_adopt_any_v051_candidate"] is True
    assert rec["best_candidate"] == VARIANT_DENSITY_ANOMALY


def test_adoption_requires_standardized_stability():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)
    comparison = {VARIANT_SELECTED_PRODUCTION_BASELINE: baseline, VARIANT_DENSITY_ONLY: candidate}
    bootstrap = {VARIANT_DENSITY_ONLY: _passing_bootstrap()}
    passing_check = DirectionalCheckResult(
        label="x",
        outcome_class="home_run",
        low_label="l",
        high_label="h",
        mean_prob_low=0.05,
        mean_prob_high=0.03,
        delta=-0.02,
        expect_high_greater=False,
        passed=True,
        sample_size=1000,
    )
    perturbation = {VARIANT_DENSITY_ONLY: {"density_direction": passing_check}}
    standardized_stable = {VARIANT_DENSITY_ONLY: False}  # instability

    rec = recommend_variant_adoption(
        comparison,
        bootstrap,
        perturbation,
        standardized_stable,
        candidate_variants=(VARIANT_DENSITY_ONLY,),
    )
    assert rec["per_candidate"][VARIANT_DENSITY_ONLY]["passes_all_automated_criteria"] is False


def test_adoption_never_true_without_bootstrap_support():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.699)
    comparison = {VARIANT_SELECTED_PRODUCTION_BASELINE: baseline, VARIANT_DENSITY_ONLY: candidate}
    bootstrap = {
        VARIANT_DENSITY_ONLY: {
            "log_loss": {
                "metric": "log_loss",
                "point_estimate": -0.001,
                "ci_low": -0.005,
                "ci_high": 0.002,
                "n_reps": 10,
                "seed": 1,
                "resampling_unit": "game_pk",
            }
        }
    }
    passing_check = DirectionalCheckResult(
        label="x",
        outcome_class="home_run",
        low_label="l",
        high_label="h",
        mean_prob_low=0.05,
        mean_prob_high=0.03,
        delta=-0.02,
        expect_high_greater=False,
        passed=True,
        sample_size=1000,
    )
    perturbation = {VARIANT_DENSITY_ONLY: {"density_direction": passing_check}}
    standardized_stable = {VARIANT_DENSITY_ONLY: True}
    rec = recommend_variant_adoption(
        comparison,
        bootstrap,
        perturbation,
        standardized_stable,
        candidate_variants=(VARIANT_DENSITY_ONLY,),
    )
    assert rec["per_candidate"][VARIANT_DENSITY_ONLY]["passes_all_automated_criteria"] is False


def test_bootstrap_determinism_reused_from_geometry_module(weather_joined_df):
    _comparison, _trained, proba_by_variant, _baseline = run_weather_variant_comparison(
        weather_joined_df
    )
    training_eligible = weather_joined_df[weather_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    y_true = val_df["outcome_class"].astype(str)
    kwargs = dict(
        y_true=y_true,
        baseline_proba=proba_by_variant[VARIANT_SELECTED_PRODUCTION_BASELINE],
        candidate_proba=proba_by_variant[VARIANT_DENSITY_ONLY],
        game_pks=val_df["game_pk"],
        n_reps=15,
        seed=7,
    )
    result_a = compute_paired_bootstrap(**kwargs)
    result_b = compute_paired_bootstrap(**kwargs)
    for metric in result_a:
        assert result_a[metric]["ci_low"] == result_b[metric]["ci_low"]
