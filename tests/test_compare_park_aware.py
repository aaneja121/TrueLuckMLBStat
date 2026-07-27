"""Tests for the Contact Luck v0.3 park-aware model comparison.

No test touches the network or real data -- synthetic multi-season,
multi-venue fixtures only.
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
from mlb_luck_score.models.build_reference_score import build_reference_artifact
from mlb_luck_score.models.compare_park_aware import (
    VARIANT_BASELINE_V02,
    VARIANT_PARK_AWARE_V03_CANDIDATE,
    find_material_venue_regressions,
    recommend_park_aware_adoption,
    run_park_aware_comparison,
)
from mlb_luck_score.models.train_contact_model import (
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.scoring.empirical_score import load_reference_artifact, save_reference_artifact


def _synthetic_season_df(
    season: int,
    rng: np.random.Generator,
    *,
    venues: tuple[str, ...],
    n_per_class: int = 30,
) -> pd.DataFrame:
    profiles = {
        "out": (75.0, 20.0, 100.0),
        "single": (92.0, 8.0, 180.0),
        "double": (98.0, 18.0, 300.0),
        "triple": (100.0, 15.0, 340.0),
        "home_run": (105.0, 28.0, 410.0),
    }
    rows = []
    for outcome, (speed, angle, dist) in profiles.items():
        for i in range(n_per_class):
            venue_id = venues[i % len(venues)]
            # Give one venue ("2" -- a launching-pad park) a real home-run boost so
            # the park-aware model has an actual signal to pick up on.
            hr_boost = 6.0 if (venue_id == venues[-1] and outcome == "home_run") else 0.0
            rows.append(
                {
                    "launch_speed": speed + rng.normal(0, 1.5) + hr_boost,
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.normal(0, 10),
                    "hit_distance_sc": dist + rng.normal(0, 5) + hr_boost * 10,
                    "bb_type": "fly_ball" if outcome in ("home_run", "double") else "line_drive",
                    "stand": "R" if i % 2 == 0 else "L",
                    "venue": "Synthetic Park",
                    "venue_id": venue_id,
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def joined_multi_venue_df() -> pd.DataFrame:
    rng = np.random.default_rng(33)
    venues = ("101", "102", "103")
    return pd.concat(
        [_synthetic_season_df(s, rng, venues=venues) for s in (2021, 2022, 2023, 2024)],
        ignore_index=True,
    )


def test_venue_id_is_not_a_leakage_column():
    assert "venue_id" not in LEAKAGE_COLUMNS


def test_feature_inclusion_venue_id_only_in_park_aware(joined_multi_venue_df: pd.DataFrame):
    comparison = run_park_aware_comparison(joined_multi_venue_df)
    assert "venue_id" not in comparison[VARIANT_BASELINE_V02]["features"]
    assert "venue_id" in comparison[VARIANT_PARK_AWARE_V03_CANDIDATE]["features"]
    # every other feature must be identical between the two variants
    baseline_features = set(comparison[VARIANT_BASELINE_V02]["features"])
    candidate_features = set(comparison[VARIANT_PARK_AWARE_V03_CANDIDATE]["features"]) - {
        "venue_id"
    }
    assert baseline_features == candidate_features


def test_probability_columns_are_class_ordered_and_sum_to_one(joined_multi_venue_df: pd.DataFrame):
    training_eligible = joined_multi_venue_df[joined_multi_venue_df["eligible_for_training"]]
    train_df = training_eligible[training_eligible["season"].isin((2021, 2022, 2023))].copy()
    train_df["venue_id"] = train_df["venue_id"].astype("string")
    trained = train_model(train_df, class_weight=None, extra_categorical_features=("venue_id",))
    feature_cols = trained.numeric_features + trained.categorical_features
    proba_df = predict_proba_ordered(trained, train_df[feature_cols].head(5))
    assert list(proba_df.columns) == list(CLASS_ORDER)
    validate_probabilities(proba_df)


def test_train_validation_seasons_are_separated(
    joined_multi_venue_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_park_aware as cpa

    seen_train_seasons: set[int] = set()
    original = cpa.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_train_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cpa, "train_model", _tracking_train_model)
    run_park_aware_comparison(joined_multi_venue_df)

    assert seen_train_seasons == {2021, 2022, 2023}


def test_comparison_rejects_missing_venue_id_column():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="venue_id"):
        run_park_aware_comparison(df)


def test_calibration_by_venue_reports_sample_counts(joined_multi_venue_df: pd.DataFrame):
    comparison = run_park_aware_comparison(joined_multi_venue_df, min_venue_samples=10)
    by_venue = comparison[VARIANT_PARK_AWARE_V03_CANDIDATE]["calibration_by_venue"]
    assert len(by_venue) >= 1
    total_venue_samples = sum(row["sample_count"] for row in by_venue)
    assert total_venue_samples == comparison[VARIANT_PARK_AWARE_V03_CANDIDATE]["sample_count"]


def test_home_run_and_subgroup_calibration_present(joined_multi_venue_df: pd.DataFrame):
    comparison = run_park_aware_comparison(joined_multi_venue_df)
    for variant in (VARIANT_BASELINE_V02, VARIANT_PARK_AWARE_V03_CANDIDATE):
        summary = comparison[variant]
        assert "home_run_ece" in summary
        assert summary["fly_ball_subgroup"]["sample_count"] > 0
        assert summary["high_distance_subgroup"]["sample_count"] > 0


def test_find_material_venue_regressions_flags_only_bad_and_reliable_venues():
    baseline = [
        {
            "venue_id": "A",
            "sample_count": 500,
            "ece_overall": 0.05,
            "ece_home_run": 0.05,
            "reliable": True,
        },
        {
            "venue_id": "B",
            "sample_count": 5,
            "ece_overall": 0.05,
            "ece_home_run": 0.05,
            "reliable": False,
        },
    ]
    candidate = [
        {
            "venue_id": "A",
            "sample_count": 500,
            "ece_overall": 0.25,
            "ece_home_run": 0.25,
            "reliable": True,
        },
        {
            "venue_id": "B",
            "sample_count": 5,
            "ece_overall": 0.30,
            "ece_home_run": 0.30,
            "reliable": False,
        },
    ]
    flagged = find_material_venue_regressions(baseline, candidate)
    assert flagged == ["A"]  # B is not reliable in baseline, so never flagged


def test_recommend_park_aware_adoption_structure(joined_multi_venue_df: pd.DataFrame):
    comparison = run_park_aware_comparison(joined_multi_venue_df)
    recommendation = recommend_park_aware_adoption(comparison)
    assert set(recommendation.keys()) == {
        "recommend_adopt_park_aware_v03",
        "improves_log_loss",
        "improves_ece",
        "log_loss_delta",
        "ece_delta",
        "material_venue_regressions",
    }
    assert isinstance(recommendation["recommend_adopt_park_aware_v03"], bool)


def test_2025_still_rejected_for_park_aware_training():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed((2021, 2022, 2023, 2025))


def test_reference_artifact_version_separation(joined_multi_venue_df: pd.DataFrame, tmp_path):
    df = joined_multi_venue_df.copy()
    df["venue_id"] = df["venue_id"].astype("string")

    v02_artifact = build_reference_artifact(df, scoring_version="0.2.0")
    v03_artifact = build_reference_artifact(
        df, scoring_version="0.3.0", extra_categorical_features=("venue_id",)
    )

    assert v02_artifact.scoring_version == "0.2.0"
    assert v03_artifact.scoring_version == "0.3.0"
    assert (
        "venue_id" not in v02_artifact.run_value_table
    )  # run values are outcome-keyed, unaffected
    assert v02_artifact != v03_artifact

    v02_path = tmp_path / "reference_score_v0.2.0.json"
    v03_path = tmp_path / "reference_score_v0.3.0.json"
    save_reference_artifact(v02_artifact, v02_path)
    save_reference_artifact(v03_artifact, v03_path)

    assert v02_path.exists()
    assert v03_path.exists()
    assert load_reference_artifact(v02_path).scoring_version == "0.2.0"
    assert load_reference_artifact(v03_path).scoring_version == "0.3.0"
    # building the v0.3 artifact must never have overwritten the v0.2 file
    assert load_reference_artifact(v02_path) == v02_artifact
