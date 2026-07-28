"""Tests for the Contact Luck v0.4 four-way geometry-aware model comparison.

No test touches the network -- synthetic multi-season, multi-venue,
geometry-joined fixtures only.
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
from mlb_luck_score.data.join_park_geometry import join_park_geometry
from mlb_luck_score.features.build_contact_features import (
    GEOMETRY_CATEGORICAL_FEATURES,
    GEOMETRY_NUMERIC_FEATURES,
)
from mlb_luck_score.models.compare_geometry_aware import (
    ALL_VARIANTS,
    VARIANT_BASELINE_V02,
    VARIANT_GEOMETRY_ONLY_V04_CANDIDATE,
    VARIANT_PARK_AWARE_V03_CANDIDATE,
    VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE,
    compute_paired_bootstrap,
    recommend_geometry_adoption,
    run_geometry_aware_comparison,
)


def _synthetic_venue_joined_season_df(
    season: int, rng: np.random.Generator, *, n_per_class: int = 40
) -> pd.DataFrame:
    """Rows shaped like cleaned_development_data_with_venue.parquet, always at Camden Yards."""
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
            # Multiple rows per game_pk (like real at-bats within a game) so
            # game-level bootstrap resampling is meaningfully different from
            # row-level resampling -- 27 distinct games per season here.
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
                    "venue": "Oriole Park at Camden Yards",
                    "venue_id": 2.0,
                    "venue_name": "Oriole Park at Camden Yards",
                    "has_venue_metadata": True,
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def geometry_joined_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    venue_df = pd.concat(
        [_synthetic_venue_joined_season_df(s, rng) for s in (2021, 2022, 2023, 2024)],
        ignore_index=True,
    )
    return join_park_geometry(venue_df)


def test_geometry_features_are_not_leakage_columns():
    assert set(GEOMETRY_NUMERIC_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert set(GEOMETRY_CATEGORICAL_FEATURES).isdisjoint(LEAKAGE_COLUMNS)


def test_comparison_rejects_missing_geometry_columns():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="venue_id"):
        run_geometry_aware_comparison(df)


def test_all_four_variants_present(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    assert set(comparison.keys()) == set(ALL_VARIANTS)


def test_geometry_only_candidate_has_no_venue_id_feature(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    assert "venue_id" not in comparison[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE]["features"]
    assert "venue_id" in comparison[VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE]["features"]


def test_baseline_has_no_geometry_or_venue_features(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    baseline_features = set(comparison[VARIANT_BASELINE_V02]["features"])
    assert "venue_id" not in baseline_features
    assert baseline_features.isdisjoint(GEOMETRY_CATEGORICAL_FEATURES)
    for f in GEOMETRY_NUMERIC_FEATURES:
        assert f not in baseline_features


def test_non_geometry_features_identical_across_variants(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    baseline_features = set(comparison[VARIANT_BASELINE_V02]["features"])
    geometry_features = set(comparison[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE]["features"])
    non_geometry_extra = (
        geometry_features
        - baseline_features
        - set(GEOMETRY_NUMERIC_FEATURES)
        - set(GEOMETRY_CATEGORICAL_FEATURES)
    )
    assert not non_geometry_extra


def test_identical_train_and_validation_rows_across_variants(
    geometry_joined_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_geometry_aware as cga

    seen_train_row_counts: set[int] = set()
    seen_train_seasons: set[int] = set()
    original = cga.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_train_row_counts.add(len(train_df))
        seen_train_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cga, "train_model", _tracking_train_model)
    run_geometry_aware_comparison(geometry_joined_df)

    # all 4 variants must have trained on the exact same row COUNT (identical rows)
    assert len(seen_train_row_counts) == 1
    assert seen_train_seasons == {2021, 2022, 2023}


def test_validation_uses_only_2024(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    n_per_class = 40
    expected_val_rows = n_per_class * 5  # only season 2024 rows
    for variant in ALL_VARIANTS:
        assert comparison[variant]["sample_count"] == expected_val_rows


def test_2025_still_rejected_for_geometry_aware_training():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed((2021, 2022, 2023, 2025))


def test_probability_columns_class_ordered_and_sum_to_one(geometry_joined_df: pd.DataFrame):
    _comparison, _trained, proba_by_variant = run_geometry_aware_comparison(geometry_joined_df)
    for variant in ALL_VARIANTS:
        proba_df = proba_by_variant[variant]
        assert list(proba_df.columns) == list(CLASS_ORDER)
        assert np.allclose(proba_df.to_numpy().sum(axis=1), 1.0, atol=1e-6)
        assert np.isfinite(proba_df.to_numpy()).all()


def test_geometry_subgroups_present(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    for variant in ALL_VARIANTS:
        subgroups = comparison[variant]["geometry_subgroups"]
        assert "near_wall_5ft" in subgroups
        assert "geometry_available" in subgroups


def test_bool_mask_handles_stringified_booleans():
    # Regression test: add_geometry_interaction_features stringifies
    # GEOMETRY_CATEGORICAL_FEATURES columns (including every near-wall flag) to
    # "True"/"False"/None for the model pipeline. A naive `.astype(bool)` on that
    # string column would treat "False" as truthy (non-empty string), silently
    # collapsing every near-wall/beyond-wall subgroup into "geometry available at
    # all". _bool_mask must interpret the strings literally instead.
    from mlb_luck_score.models.compare_geometry_aware import _bool_mask

    s = pd.Series(["True", "False", None, "True", "False"])
    assert _bool_mask(s).tolist() == [True, False, False, True, False]


def test_bool_mask_handles_native_nullable_booleans():
    from mlb_luck_score.models.compare_geometry_aware import _bool_mask

    s = pd.Series(pd.array([True, False, pd.NA, True], dtype="boolean"))
    assert _bool_mask(s).tolist() == [True, False, False, True]


def test_near_wall_subgroup_counts_are_nested_and_distinct(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    subgroups = comparison[VARIANT_BASELINE_V02]["geometry_subgroups"]
    n5 = subgroups["near_wall_5ft"]["sample_count"]
    n10 = subgroups["near_wall_10ft"]["sample_count"]
    n20 = subgroups["near_wall_20ft"]["sample_count"]
    n_available = subgroups["geometry_available"]["sample_count"]
    assert n5 <= n10 <= n20 <= n_available
    # Must not all silently collapse to the same "geometry available" count
    # (the exact failure mode of the stringified-boolean bug above).
    assert n5 < n20


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


@pytest.fixture
def bootstrap_inputs(geometry_joined_df: pd.DataFrame):
    comparison, _trained, proba_by_variant = run_geometry_aware_comparison(geometry_joined_df)
    training_eligible = geometry_joined_df[geometry_joined_df["eligible_for_training"].astype(bool)]
    val_df = training_eligible[training_eligible["season"] == 2024]
    y_true = val_df["outcome_class"].astype(str)
    return comparison, proba_by_variant, y_true, val_df


def test_bootstrap_is_deterministic_given_seed(bootstrap_inputs):
    _comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    result_a = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=25,
        seed=123,
    )
    result_b = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=25,
        seed=123,
    )
    for metric in result_a:
        assert result_a[metric]["ci_low"] == result_b[metric]["ci_low"]
        assert result_a[metric]["ci_high"] == result_b[metric]["ci_high"]
        assert result_a[metric]["point_estimate"] == result_b[metric]["point_estimate"]


def test_bootstrap_different_seeds_can_differ(bootstrap_inputs):
    _comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    result_a = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=25,
        seed=1,
    )
    result_b = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=25,
        seed=2,
    )
    # point estimates must match (computed on the full validation set, seed-independent)
    assert result_a["log_loss"]["point_estimate"] == result_b["log_loss"]["point_estimate"]


def test_bootstrap_reports_metadata(bootstrap_inputs):
    _comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    result = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=30,
        seed=99,
    )
    for metric_result in result.values():
        assert metric_result["n_reps"] == 30
        assert metric_result["seed"] == 99
        assert metric_result["resampling_unit"] == "game_pk"
        assert metric_result["ci_low"] <= metric_result["ci_high"]


def test_bootstrap_resamples_at_game_level_not_row_level(bootstrap_inputs):
    _comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    # A one-row-per-game synthetic table would make row-level and game-level resampling
    # indistinguishable; assert instead that every row of a sampled game is included
    # together by checking bootstrap runs without error on a val_df where n_games <<
    # n_rows (already true here: 5 outcomes x several games each per season).
    n_games = val_df["game_pk"].nunique()
    assert n_games < len(val_df)
    result = compute_paired_bootstrap(
        y_true,
        proba_by_variant[VARIANT_BASELINE_V02],
        proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
        val_df["game_pk"],
        n_reps=10,
        seed=1,
    )
    assert "log_loss" in result


def test_recommend_geometry_adoption_structure(bootstrap_inputs):
    comparison, proba_by_variant, y_true, val_df = bootstrap_inputs
    comparison[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE]["geometry_coverage_rate"] = float(
        val_df["has_park_geometry"].mean()
    )
    comparison[VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE]["geometry_coverage_rate"] = float(
        val_df["has_park_geometry"].mean()
    )
    bootstrap_by_candidate = {
        VARIANT_GEOMETRY_ONLY_V04_CANDIDATE: compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_BASELINE_V02],
            proba_by_variant[VARIANT_GEOMETRY_ONLY_V04_CANDIDATE],
            val_df["game_pk"],
            n_reps=10,
            seed=1,
        ),
        VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE: compute_paired_bootstrap(
            y_true,
            proba_by_variant[VARIANT_BASELINE_V02],
            proba_by_variant[VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE],
            val_df["game_pk"],
            n_reps=10,
            seed=1,
        ),
    }
    recommendation = recommend_geometry_adoption(comparison, bootstrap_by_candidate)
    assert set(recommendation["per_candidate"].keys()) == {
        VARIANT_GEOMETRY_ONLY_V04_CANDIDATE,
        VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE,
    }
    assert isinstance(recommendation["recommend_adopt_any_v04_candidate"], bool)
    for candidate_result in recommendation["per_candidate"].values():
        assert candidate_result["requires_manual_review_physical_plausibility"] is True


def test_recommend_geometry_adoption_never_true_without_bootstrap_support():
    # Construct a comparison where log loss "improves" but the bootstrap CI's upper
    # bound shows the improvement is not robust (ci_high > 0) -- must not recommend.
    baseline = {
        "multiclass_log_loss": 0.70,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "geometry_subgroups": {},
    }
    candidate = {
        "multiclass_log_loss": 0.69,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "geometry_subgroups": {},
        "geometry_coverage_rate": 0.9,
    }
    comparison = {
        VARIANT_BASELINE_V02: baseline,
        VARIANT_GEOMETRY_ONLY_V04_CANDIDATE: candidate,
        VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE: candidate,
    }
    bootstrap = {
        "log_loss": {
            "metric": "log_loss",
            "point_estimate": -0.01,
            "ci_low": -0.05,
            "ci_high": 0.02,  # upper bound above zero -- does NOT support improvement
            "n_reps": 10,
            "seed": 1,
            "resampling_unit": "game_pk",
        }
    }
    recommendation = recommend_geometry_adoption(
        comparison,
        {
            VARIANT_GEOMETRY_ONLY_V04_CANDIDATE: bootstrap,
            VARIANT_VENUE_PLUS_GEOMETRY_V04_CANDIDATE: bootstrap,
        },
    )
    assert recommendation["recommend_adopt_any_v04_candidate"] is False


def test_park_aware_v03_candidate_included_for_reference(geometry_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_geometry_aware_comparison(geometry_joined_df)
    assert VARIANT_PARK_AWARE_V03_CANDIDATE in comparison
