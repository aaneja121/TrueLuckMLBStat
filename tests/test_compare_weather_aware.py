"""Tests for the Contact Luck v0.5 weather-aware model comparison, standardized
-environment generation, and weather attribution.

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
    WEATHER_BASIC_CATEGORICAL_FEATURES,
    WEATHER_BASIC_NUMERIC_FEATURES,
    WEATHER_VECTOR_CATEGORICAL_FEATURES,
    WEATHER_VECTOR_NUMERIC_FEATURES,
    add_weather_interaction_features,
    generate_standardized_environment_rows,
)
from mlb_luck_score.models.compare_geometry_aware import compute_paired_bootstrap
from mlb_luck_score.models.compare_weather_aware import (
    VARIANT_SELECTED_PRODUCTION_BASELINE,
    VARIANT_WEATHER_BASIC_V05_CANDIDATE,
    VARIANT_WEATHER_VECTOR_V05_CANDIDATE,
    WEATHER_CANDIDATE_VARIANTS_NO_GEOMETRY,
    _prepare_weather_columns,
    check_standardized_predictions_stable,
    find_material_subgroup_regressions,
    recommend_weather_adoption,
    run_weather_aware_comparison,
)
from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model
from mlb_luck_score.scoring.weather_attribution import (
    WeatherAttributionError,
    compute_weather_attribution,
)


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
    rng = np.random.default_rng(21)
    venue_df = pd.concat(
        [_synthetic_venue_joined_season_df(s, rng) for s in (2021, 2022, 2023, 2024)],
        ignore_index=True,
    )
    game_pks = venue_df["game_pk"].unique()
    gw_rng = np.random.default_rng(5)
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


def test_weather_features_are_not_leakage_columns():
    assert set(WEATHER_BASIC_NUMERIC_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert set(WEATHER_VECTOR_NUMERIC_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert set(WEATHER_BASIC_CATEGORICAL_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert set(WEATHER_VECTOR_CATEGORICAL_FEATURES).isdisjoint(LEAKAGE_COLUMNS)


def test_comparison_rejects_missing_weather_columns():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="temperature_c"):
        run_weather_aware_comparison(df)


def test_variants_present_without_geometry(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    assert set(comparison.keys()) == {
        VARIANT_SELECTED_PRODUCTION_BASELINE,
        VARIANT_WEATHER_BASIC_V05_CANDIDATE,
        VARIANT_WEATHER_VECTOR_V05_CANDIDATE,
    }


def test_include_geometry_raises_without_geometry_columns(weather_joined_df: pd.DataFrame):
    with pytest.raises(ValueError, match="geometry"):
        run_weather_aware_comparison(weather_joined_df, include_geometry=True)


def test_baseline_has_no_weather_features(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    baseline_features = set(comparison[VARIANT_SELECTED_PRODUCTION_BASELINE]["features"])
    assert baseline_features.isdisjoint(WEATHER_VECTOR_NUMERIC_FEATURES)
    assert baseline_features.isdisjoint(WEATHER_VECTOR_CATEGORICAL_FEATURES)


def test_identical_train_validation_rows_across_variants(
    weather_joined_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_weather_aware as cwa

    seen_row_counts: set[int] = set()
    seen_seasons: set[int] = set()
    original = cwa.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_row_counts.add(len(train_df))
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cwa, "train_model", _tracking_train_model)
    run_weather_aware_comparison(weather_joined_df, include_geometry=False)

    assert len(seen_row_counts) == 1
    assert seen_seasons == {2021, 2022, 2023}


def test_validation_uses_only_2024(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    expected_n = 40 * 5
    for summary in comparison.values():
        assert summary["sample_count"] == expected_n


def test_2025_still_rejected_for_weather_aware_training():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed((2021, 2022, 2023, 2025))


def test_probability_columns_class_ordered_and_sum_to_one(weather_joined_df: pd.DataFrame):
    _comparison, _trained, proba_by_variant = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    for proba_df in proba_by_variant.values():
        assert list(proba_df.columns) == list(CLASS_ORDER)
        assert np.allclose(proba_df.to_numpy().sum(axis=1), 1.0, atol=1e-6)
        assert np.isfinite(proba_df.to_numpy()).all()


def test_complete_case_and_all_row_both_reported(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    candidate = comparison[VARIANT_WEATHER_BASIC_V05_CANDIDATE]
    assert "complete_case" in candidate
    assert candidate["complete_case"]["sample_count"] <= candidate["sample_count"]
    assert "weather_coverage_rate" in candidate


def test_weather_subgroups_present(weather_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    subgroups = comparison[VARIANT_WEATHER_BASIC_V05_CANDIDATE]["weather_subgroups"]
    assert "roof_outdoor_open_air" in subgroups
    assert "roof_fixed_indoor" in subgroups
    assert "complete_case_weather_available" in subgroups


# ---------------------------------------------------------------------------
# Subgroup regression detection
# ---------------------------------------------------------------------------


def test_find_material_subgroup_regressions_flags_large_delta():
    baseline = {"roof_outdoor_open_air": {"ece": 0.02, "sample_count": 200}}
    candidate = {"roof_outdoor_open_air": {"ece": 0.05, "sample_count": 200}}
    assert find_material_subgroup_regressions(baseline, candidate) == ["roof_outdoor_open_air"]


def test_find_material_subgroup_regressions_ignores_small_sample():
    baseline = {"roof_fixed_indoor": {"ece": 0.02, "sample_count": 5}}
    candidate = {"roof_fixed_indoor": {"ece": 0.5, "sample_count": 5}}
    assert find_material_subgroup_regressions(baseline, candidate, min_sample_size=30) == []


def test_find_material_subgroup_regressions_no_flag_for_small_change():
    baseline = {"roof_outdoor_open_air": {"ece": 0.05, "sample_count": 200}}
    candidate = {"roof_outdoor_open_air": {"ece": 0.052, "sample_count": 200}}
    assert find_material_subgroup_regressions(baseline, candidate) == []


def test_find_material_subgroup_regressions_skips_missing_candidate_label():
    baseline = {"roof_fixed_indoor": {"ece": 0.02, "sample_count": 200}}
    assert find_material_subgroup_regressions(baseline, {}) == []


# ---------------------------------------------------------------------------
# Paired bootstrap (reused from compare_geometry_aware) + adoption rule
# ---------------------------------------------------------------------------


@pytest.fixture
def bootstrap_inputs(weather_joined_df: pd.DataFrame):
    comparison, _trained, proba_by_variant = run_weather_aware_comparison(
        weather_joined_df, include_geometry=False
    )
    training_eligible = weather_joined_df[weather_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    val_df = _prepare_weather_columns(val_df)
    y_true = val_df["outcome_class"].astype(str)
    return comparison, proba_by_variant, y_true, val_df


def test_bootstrap_is_deterministic_given_seed(bootstrap_inputs):
    _comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    kwargs = dict(
        y_true=y_true,
        baseline_proba=proba_by_variant[VARIANT_SELECTED_PRODUCTION_BASELINE],
        candidate_proba=proba_by_variant[VARIANT_WEATHER_BASIC_V05_CANDIDATE],
        game_pks=val_df["game_pk"],
        n_reps=15,
        seed=7,
    )
    result_a = compute_paired_bootstrap(**kwargs)
    result_b = compute_paired_bootstrap(**kwargs)
    for metric in result_a:
        assert result_a[metric]["ci_low"] == result_b[metric]["ci_low"]
        assert result_a[metric]["point_estimate"] == result_b[metric]["point_estimate"]


def test_recommend_weather_adoption_structure(bootstrap_inputs):
    comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    bootstrap_by_candidate = {
        cand: compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_SELECTED_PRODUCTION_BASELINE],
            proba_by_variant[cand],
            val_df["game_pk"],
            n_reps=10,
            seed=1,
        )
        for cand in WEATHER_CANDIDATE_VARIANTS_NO_GEOMETRY
    }
    rec = recommend_weather_adoption(comparison, bootstrap_by_candidate)
    assert set(rec["per_candidate"].keys()) == set(WEATHER_CANDIDATE_VARIANTS_NO_GEOMETRY)
    assert isinstance(rec["recommend_adopt_any_v05_candidate"], bool)
    for result in rec["per_candidate"].values():
        assert result["requires_manual_review_physical_plausibility"] is True


def test_recommend_weather_adoption_never_true_without_bootstrap_support():
    baseline = {
        "multiclass_log_loss": 0.70,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "weather_subgroups": {},
    }
    candidate = {
        "multiclass_log_loss": 0.69,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "weather_subgroups": {},
        "weather_coverage_rate": 0.9,
    }
    comparison = {
        VARIANT_SELECTED_PRODUCTION_BASELINE: baseline,
        VARIANT_WEATHER_BASIC_V05_CANDIDATE: candidate,
        VARIANT_WEATHER_VECTOR_V05_CANDIDATE: candidate,
    }
    bootstrap = {
        "log_loss": {
            "metric": "log_loss",
            "point_estimate": -0.01,
            "ci_low": -0.05,
            "ci_high": 0.02,  # does NOT support improvement
            "n_reps": 10,
            "seed": 1,
            "resampling_unit": "game_pk",
        }
    }
    rec = recommend_weather_adoption(
        comparison,
        {
            VARIANT_WEATHER_BASIC_V05_CANDIDATE: bootstrap,
            VARIANT_WEATHER_VECTOR_V05_CANDIDATE: bootstrap,
        },
    )
    assert rec["recommend_adopt_any_v05_candidate"] is False


def test_check_standardized_predictions_stable_valid():
    proba = pd.DataFrame([[0.2, 0.2, 0.2, 0.2, 0.2]], columns=list(CLASS_ORDER))
    assert check_standardized_predictions_stable(proba) is True


def test_check_standardized_predictions_stable_invalid():
    proba = pd.DataFrame([[0.5, 0.5, 0.5, 0.5, 0.5]], columns=list(CLASS_ORDER))
    assert check_standardized_predictions_stable(proba) is False


# ---------------------------------------------------------------------------
# Standardized-environment generation
# ---------------------------------------------------------------------------


def test_standardized_environment_non_weather_features_identical(weather_joined_df: pd.DataFrame):
    cast = _prepare_weather_columns(weather_joined_df)
    standardized = generate_standardized_environment_rows(cast)
    non_weather_cols = [
        "launch_speed",
        "launch_angle",
        "spray_angle_approx",
        "hit_distance_sc",
        "bb_type",
    ]
    for col in non_weather_cols:
        pd.testing.assert_series_equal(cast[col], standardized[col])


def test_standardized_environment_overrides_weather_only_for_available_rows(
    weather_joined_df: pd.DataFrame,
):
    cast = _prepare_weather_columns(weather_joined_df)
    standardized = generate_standardized_environment_rows(cast)
    available_mask = cast["has_effective_weather"]
    assert (standardized.loc[available_mask, "wind_speed_mps"] == 0.0).all()
    assert np.allclose(standardized.loc[available_mask, "air_density_kg_m3"], 1.225)
    # Rows without weather to begin with are left untouched (still unavailable).
    unavailable = ~available_mask
    if unavailable.any():
        assert standardized.loc[unavailable, "air_density_kg_m3"].isna().all()


def test_standardized_environment_is_noop_without_weather_columns():
    df = pd.DataFrame({"launch_speed": [95.0]})
    result = generate_standardized_environment_rows(df)
    pd.testing.assert_frame_equal(df, result)


def test_standardized_environment_categorical_overrides_are_realistic_categories():
    # Regression test: an earlier version set weather_match_quality to a
    # synthetic "standardized" label never seen during training, which
    # OneHotEncoder(handle_unknown="ignore") one-hot-encodes as all zeros --
    # a pattern the fitted model never learned to interpret, verified to
    # produce physically backwards counterfactual predictions (Coors Field's
    # thin actual air scoring WORSE than the denser standardized reference).
    # Every categorical override must be a value that genuinely appears in
    # real weather-joined data.
    from mlb_luck_score.features.build_contact_features import (
        STANDARD_ENVIRONMENT_MATCH_QUALITY,
        STANDARD_ENVIRONMENT_ROOF_STATUS,
    )

    real_roof_statuses = {
        "outdoor_open_air",
        "retractable_roof_open",
        "retractable_roof_closed",
        "fixed_indoor",
        "roof_status_unknown",
    }
    real_match_qualities = {"good", "fair"}
    assert STANDARD_ENVIRONMENT_ROOF_STATUS in real_roof_statuses
    assert STANDARD_ENVIRONMENT_MATCH_QUALITY in real_match_qualities


def test_standardized_predictions_stable_uses_genuine_standardized_predictions(
    weather_joined_df: pd.DataFrame,
):
    # Regression test: the CLI's standardized-predictions check must be run
    # against predictions on ACTUAL standardized rows, not the model's
    # regular actual-environment predictions (which are already validated
    # elsewhere and would make this criterion a no-op).
    df = _prepare_weather_columns(weather_joined_df)
    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin((2021, 2022, 2023))]
    val_df = training_eligible[training_eligible["season"] == 2024].copy()

    trained = train_model(
        train_df,
        class_weight=None,
        extra_numeric_features=WEATHER_VECTOR_NUMERIC_FEATURES,
        extra_categorical_features=WEATHER_VECTOR_CATEGORICAL_FEATURES,
    )
    feature_cols = trained.numeric_features + trained.categorical_features
    standardized_val_df = generate_standardized_environment_rows(val_df)
    standardized_proba = predict_proba_ordered(trained, standardized_val_df[feature_cols])

    assert check_standardized_predictions_stable(standardized_proba) is True
    assert list(standardized_proba.columns) == list(CLASS_ORDER)


def test_add_weather_interaction_features_is_noop_without_weather_columns():
    df = pd.DataFrame({"launch_speed": [95.0]})
    result = add_weather_interaction_features(df)
    pd.testing.assert_frame_equal(df, result)


# ---------------------------------------------------------------------------
# Weather attribution
# ---------------------------------------------------------------------------


def test_weather_attribution_sign_positive_when_actual_more_favorable():
    actual = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    standard = pd.DataFrame([[0.6, 0.2, 0.1, 0.04, 0.06]], columns=list(CLASS_ORDER))
    result = compute_weather_attribution(actual, standard)
    # actual has more home_run mass (0.1 vs 0.06) -- more favorable to batter.
    assert result["weather_run_value_effect"].iloc[0] > 0


def test_weather_attribution_sign_negative_when_actual_less_favorable():
    actual = pd.DataFrame([[0.6, 0.2, 0.1, 0.04, 0.06]], columns=list(CLASS_ORDER))
    standard = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    result = compute_weather_attribution(actual, standard)
    assert result["weather_run_value_effect"].iloc[0] < 0


def test_weather_attribution_zero_when_identical():
    proba = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    result = compute_weather_attribution(proba, proba.copy())
    assert result["weather_run_value_effect"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_weather_attribution_equals_difference_of_expected_values():
    actual = pd.DataFrame(
        [[0.5, 0.2, 0.15, 0.05, 0.1], [0.3, 0.3, 0.2, 0.1, 0.1]], columns=list(CLASS_ORDER)
    )
    standard = pd.DataFrame(
        [[0.55, 0.2, 0.13, 0.04, 0.08], [0.3, 0.3, 0.2, 0.1, 0.1]], columns=list(CLASS_ORDER)
    )
    result = compute_weather_attribution(actual, standard)
    computed_diff = (
        result["expected_run_value_actual_environment"]
        - result["expected_run_value_standard_environment"]
    )
    pd.testing.assert_series_equal(
        result["weather_run_value_effect"], computed_diff, check_names=False
    )


def test_weather_attribution_rejects_mismatched_index():
    actual = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER), index=[0])
    standard = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER), index=[1])
    with pytest.raises(WeatherAttributionError, match="identical index"):
        compute_weather_attribution(actual, standard)


def test_weather_attribution_never_uses_leakage_columns_in_its_module():
    # Sanity check that the attribution module's only inputs are already-
    # predicted probabilities and the fixed run-value table -- no raw
    # outcome/leakage columns are referenced anywhere in its computation.
    import inspect

    from mlb_luck_score.scoring import weather_attribution as wa

    source = inspect.getsource(wa)
    for leaked_col in LEAKAGE_COLUMNS:
        # Quoted-literal check (e.g. df["des"]), not a bare substring search
        # -- short column names like "des" are common substrings of ordinary
        # English words ("provides", "described", ...) that would otherwise
        # false-positive.
        assert f'"{leaked_col}"' not in source
        assert f"'{leaked_col}'" not in source


def test_weather_run_value_effect_not_added_to_raw_contact_luck_by_default():
    # Contract check: mlb_luck_score.scoring.contact_luck's formula does not
    # import or reference weather_attribution at all -- the two are
    # independent computations, never summed together automatically.
    import inspect

    from mlb_luck_score.scoring import contact_luck

    source = inspect.getsource(contact_luck)
    assert "weather_attribution" not in source
    assert "weather_run_value_effect" not in source


def test_trained_model_can_predict_on_standardized_rows(weather_joined_df: pd.DataFrame):
    # End-to-end smoke test: train a weather-vector model, predict on both
    # actual and standardized rows for the same plays, and confirm the
    # attribution pipeline runs without error.
    df = _prepare_weather_columns(weather_joined_df)
    training_eligible = df[df["eligible_for_training"].astype(bool)]
    train_df = training_eligible[training_eligible["season"].isin((2021, 2022, 2023))]
    val_df = training_eligible[training_eligible["season"] == 2024].copy()

    trained = train_model(
        train_df,
        class_weight=None,
        extra_numeric_features=WEATHER_VECTOR_NUMERIC_FEATURES,
        extra_categorical_features=WEATHER_VECTOR_CATEGORICAL_FEATURES,
    )
    feature_cols = trained.numeric_features + trained.categorical_features

    standardized_val_df = generate_standardized_environment_rows(val_df)

    actual_proba = predict_proba_ordered(trained, val_df[feature_cols])
    standardized_proba = predict_proba_ordered(trained, standardized_val_df[feature_cols])

    attribution = compute_weather_attribution(actual_proba, standardized_proba)
    assert len(attribution) == len(val_df)
    assert np.isfinite(attribution["weather_run_value_effect"]).all()
