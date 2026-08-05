"""Tests for the Contact Luck v0.6 alignment-aware positioning comparison.

No test touches the network -- synthetic multi-season, venue-joined
fixtures only. The synthetic data deliberately encodes a real physical
relationship (shifted infield alignment -> fewer pull-side ground-ball
singles; strategic outfield alignment -> fewer pull-side fly-ball doubles)
so the controlled-perturbation directional checks have real signal to
detect, mirroring `tests/test_compare_weather_variants.py`'s approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import CLASS_ORDER, LEAKAGE_COLUMNS
from mlb_luck_score.features.build_contact_features import (
    ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES,
    ALIGNMENT_INTERACTION_NUMERIC_FEATURES,
    ALIGNMENT_LABELS_CATEGORICAL_FEATURES,
    add_alignment_interaction_features,
    generate_typical_alignment_rows,
)
from mlb_luck_score.models.compare_alignment_aware import (
    ALIGNMENT_CANDIDATE_VARIANTS,
    VARIANT_ALIGNMENT_INTERACTIONS_V06,
    VARIANT_ALIGNMENT_LABELS_V06,
    check_positioning_counterfactual_stable,
    find_most_changed_plays,
    override_alignment_and_recompute,
    recommend_alignment_adoption,
    run_alignment_aware_comparison,
    run_alignment_perturbation_checks,
)
from mlb_luck_score.models.compare_park_aware import VARIANT_BASELINE_V02
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult
from mlb_luck_score.scoring.positioning_attribution import (
    PositioningAttributionError,
    compute_positioning_attribution,
)


def _base_rows(season: int, rng: np.random.Generator, *, n_per_class: int = 30) -> list[dict]:
    profiles = {
        "out": (75.0, 20.0, 100.0, "line_drive"),
        "single": (92.0, 8.0, 180.0, "line_drive"),
        "double": (98.0, 18.0, 300.0, "fly_ball"),
        "triple": (100.0, 15.0, 340.0, "fly_ball"),
        "home_run": (105.0, 28.0, 400.0, "fly_ball"),
    }
    rows = []
    for outcome, (speed, angle, dist, bb_type) in profiles.items():
        for i in range(n_per_class):
            day = 1 + (i % 27)
            rows.append(
                {
                    "game_pk": season * 1000 + day,
                    "game_date": f"{season}-06-{day:02d}",
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.uniform(-8.0, 8.0),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": bb_type,
                    "stand": "R" if i % 2 == 0 else "L",
                    "spray_sector": "center",
                    "is_pull": False,
                    "is_opposite_field": False,
                    "on_1b": None,
                    "on_2b": None,
                    "on_3b": None,
                    "if_fielding_alignment": "Standard",
                    "of_fielding_alignment": "Standard",
                    "venue": "Fenway Park",
                    "venue_id": 3.0,
                    "has_venue_metadata": True,
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return rows


def _infield_shift_signal_rows(season: int, rng: np.random.Generator, *, n: int = 60) -> list[dict]:
    rows = []
    for i in range(n):
        day = 1 + (i % 27)
        standard = i % 2 == 0
        alignment = "Standard" if standard else "Infield shift"
        outcome = "single" if standard else "out"
        rows.append(
            {
                "game_pk": season * 2000 + day,
                "game_date": f"{season}-07-{day:02d}",
                "launch_speed": 90.0 + rng.normal(0, 1.5),
                "launch_angle": 5.0 + rng.normal(0, 1.5),
                "spray_angle_approx": -30.0 + rng.normal(0, 2),
                "hit_distance_sc": 120.0 + rng.normal(0, 5),
                "bb_type": "ground_ball",
                "stand": "R",
                "spray_sector": "left",
                "is_pull": True,
                "is_opposite_field": False,
                "on_1b": None,
                "on_2b": None,
                "on_3b": None,
                "if_fielding_alignment": alignment,
                "of_fielding_alignment": "Standard",
                "venue": "Fenway Park",
                "venue_id": 3.0,
                "has_venue_metadata": True,
                "outcome_class": outcome,
                "season": season,
                "eligible_for_training": True,
            }
        )
    return rows


def _outfield_shift_signal_rows(
    season: int, rng: np.random.Generator, *, n: int = 60
) -> list[dict]:
    rows = []
    for i in range(n):
        day = 1 + (i % 27)
        standard = i % 2 == 0
        alignment = "Standard" if standard else "Strategic"
        outcome = "double" if standard else "out"
        rows.append(
            {
                "game_pk": season * 3000 + day,
                "game_date": f"{season}-08-{day:02d}",
                "launch_speed": 96.0 + rng.normal(0, 1.5),
                "launch_angle": 25.0 + rng.normal(0, 1.5),
                "spray_angle_approx": -30.0 + rng.normal(0, 2),
                "hit_distance_sc": 320.0 + rng.normal(0, 5),
                "bb_type": "fly_ball",
                "stand": "R",
                "spray_sector": "left",
                "is_pull": True,
                "is_opposite_field": False,
                "on_1b": None,
                "on_2b": None,
                "on_3b": None,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": alignment,
                "venue": "Fenway Park",
                "venue_id": 3.0,
                "has_venue_metadata": True,
                "outcome_class": outcome,
                "season": season,
                "eligible_for_training": True,
            }
        )
    return rows


@pytest.fixture
def alignment_joined_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        rows += _base_rows(season, rng)
        rows += _infield_shift_signal_rows(season, rng)
        rows += _outfield_shift_signal_rows(season, rng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_alignment_features_are_not_leakage_columns():
    for feats in (
        ALIGNMENT_LABELS_CATEGORICAL_FEATURES,
        ALIGNMENT_INTERACTION_NUMERIC_FEATURES,
        ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES,
    ):
        assert set(feats).isdisjoint(LEAKAGE_COLUMNS)


def test_add_alignment_interaction_features_is_noop_without_alignment_columns():
    df = pd.DataFrame({"launch_speed": [95.0]})
    result = add_alignment_interaction_features(df)
    pd.testing.assert_frame_equal(df, result)


def test_add_alignment_interaction_features_computes_shift_indicators():
    df = pd.DataFrame(
        {
            "if_fielding_alignment": ["Standard", "Infield shift", None],
            "of_fielding_alignment": ["Standard", "Strategic", "Standard"],
            "launch_angle": [10.0, 20.0, 30.0],
            "hit_distance_sc": [100.0, 200.0, 300.0],
            "is_pull": [True, True, False],
            "bb_type": ["ground_ball", "ground_ball", "fly_ball"],
            "stand": ["R", "L", "R"],
            "spray_sector": ["left", "left", "center"],
        }
    )
    out = add_alignment_interaction_features(df)
    assert out["if_alignment_shift_indicator"].tolist()[:2] == [0.0, 1.0]
    assert pd.isna(out["if_alignment_shift_indicator"].iloc[2])
    assert out["of_alignment_shift_indicator"].tolist() == [0.0, 1.0, 0.0]
    assert out["of_shift_x_launch_angle"].iloc[1] == pytest.approx(1.0 * 20.0)
    assert out["if_shift_x_pull_groundball"].iloc[1] == pytest.approx(1.0)  # shift AND pull AND GB
    assert out["if_shift_x_pull_groundball"].iloc[0] == pytest.approx(0.0)  # standard, not shifted
    assert out["if_alignment_x_stand"].iloc[0] == "Standard_R"
    assert out["if_alignment_x_spray_sector"].iloc[1] == "Infield shift_left"
    assert out["if_alignment_x_stand"].iloc[2] is None  # missing if_fielding_alignment


def test_generate_typical_alignment_rows_overrides_to_standard_and_recomputes():
    df = pd.DataFrame(
        {
            "if_fielding_alignment": ["Infield shift"],
            "of_fielding_alignment": ["Strategic"],
            "launch_angle": [10.0],
            "hit_distance_sc": [150.0],
            "is_pull": [True],
            "bb_type": ["ground_ball"],
            "stand": ["R"],
            "spray_sector": ["left"],
        }
    )
    typical = generate_typical_alignment_rows(df)
    assert typical["if_fielding_alignment"].iloc[0] == "Standard"
    assert typical["of_fielding_alignment"].iloc[0] == "Standard"
    # Derived columns must be RECOMPUTED, not stale.
    assert typical["if_alignment_shift_indicator"].iloc[0] == 0.0
    assert typical["of_alignment_shift_indicator"].iloc[0] == 0.0
    assert typical["if_shift_x_pull_groundball"].iloc[0] == 0.0
    assert typical["if_alignment_x_stand"].iloc[0] == "Standard_R"


def test_generate_typical_alignment_rows_leaves_missing_alignment_rows_untouched():
    df = pd.DataFrame(
        {
            "if_fielding_alignment": [None],
            "of_fielding_alignment": [None],
            "launch_angle": [10.0],
            "hit_distance_sc": [150.0],
            "is_pull": [True],
            "bb_type": ["ground_ball"],
            "stand": ["R"],
            "spray_sector": ["left"],
        }
    )
    typical = generate_typical_alignment_rows(df)
    assert pd.isna(typical["if_fielding_alignment"].iloc[0])
    assert pd.isna(typical["of_fielding_alignment"].iloc[0])


def test_override_alignment_and_recompute_respects_mask():
    df = pd.DataFrame(
        {
            "if_fielding_alignment": ["Standard", "Standard"],
            "of_fielding_alignment": ["Standard", "Standard"],
            "launch_angle": [10.0, 10.0],
            "hit_distance_sc": [150.0, 150.0],
            "is_pull": [True, True],
            "bb_type": ["ground_ball", "ground_ball"],
            "stand": ["R", "R"],
            "spray_sector": ["left", "left"],
        }
    )
    mask = pd.Series([True, False])
    out = override_alignment_and_recompute(df, if_alignment="Infield shift", mask=mask)
    assert out["if_fielding_alignment"].tolist() == ["Infield shift", "Standard"]
    assert out["if_alignment_shift_indicator"].tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------


def test_comparison_rejects_missing_alignment_columns():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="if_fielding_alignment"):
        run_alignment_aware_comparison(df)


def test_all_variants_present(alignment_joined_df: pd.DataFrame):
    comparison, _trained, _proba = run_alignment_aware_comparison(alignment_joined_df)
    assert set(comparison.keys()) == {VARIANT_BASELINE_V02, *ALIGNMENT_CANDIDATE_VARIANTS}


def test_alignment_interactions_candidate_is_cumulative_over_labels(alignment_joined_df):
    comparison, _trained, _proba = run_alignment_aware_comparison(alignment_joined_df)
    labels_features = set(comparison[VARIANT_ALIGNMENT_LABELS_V06]["features"])
    interactions_features = set(comparison[VARIANT_ALIGNMENT_INTERACTIONS_V06]["features"])
    assert labels_features <= interactions_features
    assert "if_alignment_x_stand" in interactions_features
    assert "if_alignment_x_stand" not in labels_features


def test_identical_train_validation_rows_across_variants(
    alignment_joined_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_alignment_aware as caa

    seen_row_counts: set[int] = set()
    seen_seasons: set[int] = set()
    original = caa.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_row_counts.add(len(train_df))
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(caa, "train_model", _tracking_train_model)
    run_alignment_aware_comparison(alignment_joined_df)

    assert len(seen_row_counts) == 1
    assert seen_seasons == {2021, 2022, 2023}


def test_probability_columns_class_ordered_and_sum_to_one(alignment_joined_df: pd.DataFrame):
    _comparison, _trained, proba_by_variant = run_alignment_aware_comparison(alignment_joined_df)
    for proba_df in proba_by_variant.values():
        assert list(proba_df.columns) == list(CLASS_ORDER)
        assert np.allclose(proba_df.to_numpy().sum(axis=1), 1.0, atol=1e-6)
        assert np.isfinite(proba_df.to_numpy()).all()


def test_alignment_subgroups_cover_required_evaluation_groups(alignment_joined_df):
    comparison, _trained, _proba = run_alignment_aware_comparison(alignment_joined_df)
    subgroups = comparison[VARIANT_ALIGNMENT_INTERACTIONS_V06]["alignment_subgroups"]
    for expected in (
        "bb_type_ground_ball",
        "bb_type_line_drive",
        "bb_type_fly_ball",
        "pull_side",
        "opposite_field",
        "shifted_alignment",
        "standard_alignment",
        "lhb",
        "rhb",
        "bases_empty",
        "runners_on",
    ):
        assert expected in subgroups


def test_most_changed_plays_reported_for_each_candidate(alignment_joined_df):
    comparison, _trained, _proba = run_alignment_aware_comparison(alignment_joined_df)
    for candidate in ALIGNMENT_CANDIDATE_VARIANTS:
        plays = comparison[candidate]["most_changed_plays"]
        assert isinstance(plays, list)
        assert len(plays) > 0
        assert "probability_shift_tvd" in plays[0]
        # Sorted descending by shift magnitude.
        shifts = [p["probability_shift_tvd"] for p in plays]
        assert shifts == sorted(shifts, reverse=True)


def test_find_most_changed_plays_rejects_mismatched_index():
    a = pd.DataFrame([[0.2] * 5], columns=list(CLASS_ORDER), index=[0])
    b = pd.DataFrame([[0.2] * 5], columns=list(CLASS_ORDER), index=[1])
    with pytest.raises(ValueError, match="same row index"):
        find_most_changed_plays(a, b, pd.DataFrame({"game_pk": [1]}))


# ---------------------------------------------------------------------------
# Controlled-perturbation checks
# ---------------------------------------------------------------------------


def _prepared_val_df(alignment_joined_df: pd.DataFrame) -> pd.DataFrame:
    from mlb_luck_score.models.compare_alignment_aware import _prepare_alignment_columns
    from mlb_luck_score.models.compare_park_aware import _prepare_venue_column

    training_eligible = alignment_joined_df[
        alignment_joined_df["eligible_for_training"].astype(bool)
    ]
    val_df = training_eligible[training_eligible["season"] == 2024].copy()
    val_df = _prepare_alignment_columns(val_df)
    return _prepare_venue_column(val_df)


def test_run_alignment_perturbation_checks_detects_correct_direction(alignment_joined_df):
    _comparison, trained_models, _proba = run_alignment_aware_comparison(alignment_joined_df)
    val_df = _prepared_val_df(alignment_joined_df)

    results = run_alignment_perturbation_checks(
        trained_models[VARIANT_ALIGNMENT_INTERACTIONS_V06], val_df
    )
    assert "infield_shift_reduces_pull_groundball_singles" in results
    assert "outfield_strategic_reduces_pull_flyball_doubles" in results
    assert all(isinstance(r, DirectionalCheckResult) for r in results.values())
    # The synthetic fixture deterministically encodes both relationships, so
    # a model trained on the interaction features should learn them.
    for result in results.values():
        assert result.passed is True


def test_run_alignment_perturbation_checks_skipped_without_alignment_columns():
    df = pd.DataFrame({"launch_speed": [95.0]})
    results = run_alignment_perturbation_checks(None, df)  # trained never used
    assert results == {}


# ---------------------------------------------------------------------------
# Positioning counterfactual stability
# ---------------------------------------------------------------------------


def test_check_positioning_counterfactual_stable(alignment_joined_df):
    _comparison, trained_models, _proba = run_alignment_aware_comparison(alignment_joined_df)
    val_df = _prepared_val_df(alignment_joined_df)

    result = check_positioning_counterfactual_stable(
        trained_models[VARIANT_ALIGNMENT_INTERACTIONS_V06], val_df
    )
    assert result["typical_predictions_well_formed"] is True
    assert result["positioning_effect_zero_at_standard_alignment"] is True
    assert result["stable"] is True


# ---------------------------------------------------------------------------
# Adoption rule: AND-combination, never solely log-loss
# ---------------------------------------------------------------------------


def _base_summary(log_loss: float) -> dict:
    return {
        "multiclass_log_loss": log_loss,
        "expected_calibration_error": 0.02,
        "home_run_ece": 0.01,
        "calibration_by_venue": [],
        "alignment_subgroups": {},
        "alignment_coverage_rate": 0.9,
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


def _passing_check() -> DirectionalCheckResult:
    return DirectionalCheckResult(
        label="x",
        outcome_class="single",
        low_label="Standard",
        high_label="Infield shift",
        mean_prob_low=0.3,
        mean_prob_high=0.1,
        delta=-0.2,
        expect_high_greater=False,
        passed=True,
        sample_size=100,
    )


def _failing_check() -> DirectionalCheckResult:
    check = _passing_check()
    return DirectionalCheckResult(**{**check.__dict__, "passed": False, "delta": 0.2})


def test_adoption_requires_perturbation_checks_even_if_log_loss_improves():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)
    comparison = {VARIANT_BASELINE_V02: baseline, VARIANT_ALIGNMENT_LABELS_V06: candidate}
    bootstrap = {VARIANT_ALIGNMENT_LABELS_V06: _passing_bootstrap()}
    perturbation = {VARIANT_ALIGNMENT_LABELS_V06: {"infield_shift": _failing_check()}}
    stability = {VARIANT_ALIGNMENT_LABELS_V06: {"stable": True}}

    rec = recommend_alignment_adoption(
        comparison,
        bootstrap,
        perturbation,
        stability,
        candidate_variants=(VARIANT_ALIGNMENT_LABELS_V06,),
    )
    assert (
        rec["per_candidate"][VARIANT_ALIGNMENT_LABELS_V06]["passes_all_automated_criteria"] is False
    )
    assert rec["recommend_adopt_any_v06_candidate"] is False


def test_adoption_passes_when_everything_checks_out():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)
    comparison = {VARIANT_BASELINE_V02: baseline, VARIANT_ALIGNMENT_INTERACTIONS_V06: candidate}
    bootstrap = {VARIANT_ALIGNMENT_INTERACTIONS_V06: _passing_bootstrap()}
    perturbation = {
        VARIANT_ALIGNMENT_INTERACTIONS_V06: {
            "infield_shift": _passing_check(),
            "outfield_strategic": _passing_check(),
        }
    }
    stability = {VARIANT_ALIGNMENT_INTERACTIONS_V06: {"stable": True}}

    rec = recommend_alignment_adoption(
        comparison,
        bootstrap,
        perturbation,
        stability,
        candidate_variants=(VARIANT_ALIGNMENT_INTERACTIONS_V06,),
    )
    assert (
        rec["per_candidate"][VARIANT_ALIGNMENT_INTERACTIONS_V06]["passes_all_automated_criteria"]
        is True
    )
    assert rec["recommend_adopt_any_v06_candidate"] is True
    assert rec["best_candidate"] == VARIANT_ALIGNMENT_INTERACTIONS_V06


def test_adoption_requires_counterfactual_stability():
    baseline = _base_summary(0.70)
    candidate = _base_summary(0.6997)
    comparison = {VARIANT_BASELINE_V02: baseline, VARIANT_ALIGNMENT_LABELS_V06: candidate}
    bootstrap = {VARIANT_ALIGNMENT_LABELS_V06: _passing_bootstrap()}
    perturbation = {VARIANT_ALIGNMENT_LABELS_V06: {"infield_shift": _passing_check()}}
    stability = {VARIANT_ALIGNMENT_LABELS_V06: {"stable": False}}

    rec = recommend_alignment_adoption(
        comparison,
        bootstrap,
        perturbation,
        stability,
        candidate_variants=(VARIANT_ALIGNMENT_LABELS_V06,),
    )
    assert (
        rec["per_candidate"][VARIANT_ALIGNMENT_LABELS_V06]["passes_all_automated_criteria"] is False
    )


def test_adoption_regresses_on_subgroup_failure():
    baseline = _base_summary(0.70)
    baseline["alignment_subgroups"] = {
        "bb_type_ground_ball": {"ece": 0.02, "sample_count": 500, "log_loss": 0.5}
    }
    candidate = _base_summary(0.6997)
    candidate["alignment_subgroups"] = {
        "bb_type_ground_ball": {"ece": 0.10, "sample_count": 500, "log_loss": 0.5}
    }
    comparison = {VARIANT_BASELINE_V02: baseline, VARIANT_ALIGNMENT_LABELS_V06: candidate}
    bootstrap = {VARIANT_ALIGNMENT_LABELS_V06: _passing_bootstrap()}
    perturbation = {VARIANT_ALIGNMENT_LABELS_V06: {"infield_shift": _passing_check()}}
    stability = {VARIANT_ALIGNMENT_LABELS_V06: {"stable": True}}

    rec = recommend_alignment_adoption(
        comparison,
        bootstrap,
        perturbation,
        stability,
        candidate_variants=(VARIANT_ALIGNMENT_LABELS_V06,),
    )
    result = rec["per_candidate"][VARIANT_ALIGNMENT_LABELS_V06]
    assert result["passes_all_automated_criteria"] is False
    assert "bb_type_ground_ball" in result["material_subgroup_regressions"]


# ---------------------------------------------------------------------------
# Positioning attribution
# ---------------------------------------------------------------------------


def test_positioning_attribution_sign_positive_when_actual_more_favorable():
    actual = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    typical = pd.DataFrame([[0.6, 0.2, 0.1, 0.04, 0.06]], columns=list(CLASS_ORDER))
    result = compute_positioning_attribution(actual, typical)
    assert result["positioning_effect"].iloc[0] > 0


def test_positioning_attribution_sign_negative_when_actual_less_favorable():
    actual = pd.DataFrame([[0.6, 0.2, 0.1, 0.04, 0.06]], columns=list(CLASS_ORDER))
    typical = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    result = compute_positioning_attribution(actual, typical)
    assert result["positioning_effect"].iloc[0] < 0


def test_positioning_attribution_zero_when_identical():
    proba = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER))
    result = compute_positioning_attribution(proba, proba.copy())
    assert result["positioning_effect"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_positioning_attribution_equals_difference_of_expected_values():
    actual = pd.DataFrame(
        [[0.5, 0.2, 0.15, 0.05, 0.1], [0.3, 0.3, 0.2, 0.1, 0.1]], columns=list(CLASS_ORDER)
    )
    typical = pd.DataFrame(
        [[0.55, 0.2, 0.13, 0.04, 0.08], [0.3, 0.3, 0.2, 0.1, 0.1]], columns=list(CLASS_ORDER)
    )
    result = compute_positioning_attribution(actual, typical)
    computed_diff = (
        result["expected_run_value_actual_alignment"]
        - result["expected_run_value_typical_alignment"]
    )
    pd.testing.assert_series_equal(result["positioning_effect"], computed_diff, check_names=False)


def test_positioning_attribution_rejects_mismatched_index():
    actual = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER), index=[0])
    typical = pd.DataFrame([[0.5, 0.2, 0.15, 0.05, 0.1]], columns=list(CLASS_ORDER), index=[1])
    with pytest.raises(PositioningAttributionError, match="identical index"):
        compute_positioning_attribution(actual, typical)


def test_positioning_attribution_never_uses_leakage_columns_in_its_module():
    import inspect

    from mlb_luck_score.scoring import positioning_attribution as pa

    source = inspect.getsource(pa)
    for leaked_col in LEAKAGE_COLUMNS:
        # Quoted-literal check (e.g. df["des"]), not a bare substring search
        # -- short column names like "des" are common substrings of ordinary
        # English words ("provides", "described", ...) that would otherwise
        # false-positive.
        assert f'"{leaked_col}"' not in source
        assert f"'{leaked_col}'" not in source


def test_positioning_effect_not_added_to_raw_contact_luck_by_default():
    import inspect

    from mlb_luck_score.scoring import contact_luck

    source = inspect.getsource(contact_luck)
    assert "positioning_attribution" not in source
    assert "positioning_effect" not in source


def test_end_to_end_positioning_attribution_from_trained_model(alignment_joined_df):
    from mlb_luck_score.models.train_contact_model import predict_proba_ordered

    _comparison, trained_models, _proba = run_alignment_aware_comparison(alignment_joined_df)
    val_df = _prepared_val_df(alignment_joined_df)
    trained = trained_models[VARIANT_ALIGNMENT_INTERACTIONS_V06]
    feature_cols = trained.numeric_features + trained.categorical_features

    typical_val_df = generate_typical_alignment_rows(val_df)
    actual_proba = predict_proba_ordered(trained, val_df[feature_cols])
    typical_proba = predict_proba_ordered(trained, typical_val_df[feature_cols])

    attribution = compute_positioning_attribution(actual_proba, typical_proba)
    assert len(attribution) == len(val_df)
    assert np.isfinite(attribution["positioning_effect"]).all()
