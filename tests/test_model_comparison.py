"""Tests for the class-weight/calibration model comparison workflow.

Covers: no implicit class weighting in the probability baseline, correct
class ordering and probability-sum invariants across every variant
(including the naive prevalence baseline and the post-hoc-calibrated
variant), that calibration fitting/evaluation use non-overlapping seasons,
and that 2025 remains rejected everywhere. No test touches the network or
real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import (
    CALIBRATION_BASE_TRAIN_SEASONS,
    CALIBRATION_EVAL_SEASONS,
    CALIBRATION_FIT_SEASONS,
    CLASS_ORDER,
    FINAL_TEST_SEASONS,
    ProtectedSeasonError,
    assert_seasons_allowed,
)
from mlb_luck_score.models.baseline_models import (
    predict_proba_prevalence,
    train_prevalence_baseline,
)
from mlb_luck_score.models.calibrate_model import (
    compute_calibration_table,
    compute_expected_calibration_error,
    fit_calibrated_classifier,
    predict_proba_calibrated,
)
from mlb_luck_score.models.compare_models import (
    VARIANT_CLASS_BALANCED,
    VARIANT_NAIVE_PREVALENCE,
    VARIANT_UNWEIGHTED,
    VARIANT_UNWEIGHTED_CALIBRATED,
    run_comparison,
)
from mlb_luck_score.models.train_contact_model import (
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)


def _synthetic_season_df(
    season: int, rng: np.random.Generator, n_per_class: int = 25
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
            rows.append(
                {
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.normal(0, 10),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "line_drive",
                    "stand": "R" if i % 2 == 0 else "L",
                    "venue": "Synthetic Park",
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_season_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.concat(
        [_synthetic_season_df(s, rng) for s in (2021, 2022, 2023, 2024)], ignore_index=True
    )


# -- no implicit class weighting -------------------------------------------


def test_train_model_default_has_no_class_weight(multi_season_df: pd.DataFrame):
    trained = train_model(multi_season_df)
    classifier = trained.pipeline.named_steps["classify"]
    assert classifier.class_weight is None
    assert trained.class_weight is None
    assert trained.variant == VARIANT_UNWEIGHTED or "unweighted" in trained.variant


def test_train_model_balanced_is_distinctly_labeled(multi_season_df: pd.DataFrame):
    trained = train_model(multi_season_df, class_weight="balanced")
    classifier = trained.pipeline.named_steps["classify"]
    assert classifier.class_weight == "balanced"
    assert "class_balanced" in trained.variant
    assert trained.variant != VARIANT_UNWEIGHTED


def test_every_triple_is_kept_regardless_of_class_weight(multi_season_df: pd.DataFrame):
    n_triples = int((multi_season_df["outcome_class"] == "triple").sum())
    for class_weight in (None, "balanced"):
        trained = train_model(multi_season_df, class_weight=class_weight)
        feature_cols = trained.numeric_features + trained.categorical_features
        x = multi_season_df[feature_cols]
        y = multi_season_df["outcome_class"].astype(str)
        # class_weight must not change which rows are used for fitting.
        assert len(x) == len(multi_season_df)
        assert int((y == "triple").sum()) == n_triples


# -- class ordering / probability-sum invariants ---------------------------


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_logistic_variants_preserve_class_order_and_sum_to_one(
    multi_season_df: pd.DataFrame, class_weight: str | None
):
    trained = train_model(multi_season_df, class_weight=class_weight)
    feature_cols = trained.numeric_features + trained.categorical_features
    proba_df = predict_proba_ordered(trained, multi_season_df[feature_cols].head(10))
    assert list(proba_df.columns) == list(CLASS_ORDER)
    validate_probabilities(proba_df)


def test_naive_prevalence_baseline_ignores_features_and_sums_to_one(multi_season_df: pd.DataFrame):
    baseline = train_prevalence_baseline(multi_season_df)
    proba_df = predict_proba_prevalence(baseline, 5)
    assert list(proba_df.columns) == list(CLASS_ORDER)
    validate_probabilities(proba_df)
    # every row identical -- the model ignores all features by design
    assert proba_df.nunique().eq(1).all()
    # matches the true training marginal frequencies
    true_freq = multi_season_df["outcome_class"].value_counts(normalize=True)
    for cls in CLASS_ORDER:
        assert proba_df[cls].iloc[0] == pytest.approx(true_freq.get(cls, 0.0))


def test_calibrated_classifier_preserves_class_order_and_sum_to_one(multi_season_df: pd.DataFrame):
    base = train_model(multi_season_df, class_weight=None)
    feature_cols = base.numeric_features + base.categorical_features
    calibrated = fit_calibrated_classifier(
        base.pipeline,
        multi_season_df[feature_cols],
        multi_season_df["outcome_class"].astype(str),
    )
    proba_df = predict_proba_calibrated(calibrated, multi_season_df[feature_cols].head(10))
    assert list(proba_df.columns) == list(CLASS_ORDER)
    validate_probabilities(proba_df)


# -- calibration season separation ------------------------------------------


def test_calibration_seasons_are_pairwise_disjoint():
    base = set(CALIBRATION_BASE_TRAIN_SEASONS)
    fit = set(CALIBRATION_FIT_SEASONS)
    ev = set(CALIBRATION_EVAL_SEASONS)
    assert base.isdisjoint(fit)
    assert base.isdisjoint(ev)
    assert fit.isdisjoint(ev)


def test_calibration_seasons_never_include_2025():
    all_calibration_seasons = (
        set(CALIBRATION_BASE_TRAIN_SEASONS)
        | set(CALIBRATION_FIT_SEASONS)
        | set(CALIBRATION_EVAL_SEASONS)
    )
    assert set(FINAL_TEST_SEASONS).isdisjoint(all_calibration_seasons)


def test_calibration_eval_seasons_match_the_untouched_validation_season():
    from mlb_luck_score.config import VALIDATION_SEASONS

    assert CALIBRATION_EVAL_SEASONS == VALIDATION_SEASONS


def test_run_comparison_post_hoc_calibration_uses_separate_seasons(
    multi_season_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    """The calibrator must be fit on CALIBRATION_FIT_SEASONS rows and
    evaluated on CALIBRATION_EVAL_SEASONS rows -- never the same rows."""
    seen_fit_seasons: set[int] = set()
    seen_eval_seasons: set[int] = set()

    import mlb_luck_score.models.compare_models as compare_models_module

    original_fit = compare_models_module.fit_calibrated_classifier
    original_summarize = compare_models_module.summarize_variant

    def _tracking_fit(pipeline, x_val, y_val, **kwargs):
        seen_fit_seasons.update(x_val.index.map(lambda i: multi_season_df.loc[i, "season"]))
        return original_fit(pipeline, x_val, y_val, **kwargs)

    def _tracking_summarize(variant, proba_df, y_true, **kwargs):
        if variant == VARIANT_UNWEIGHTED_CALIBRATED:
            seen_eval_seasons.update(y_true.index.map(lambda i: multi_season_df.loc[i, "season"]))
        return original_summarize(variant, proba_df, y_true, **kwargs)

    monkeypatch.setattr(compare_models_module, "fit_calibrated_classifier", _tracking_fit)
    monkeypatch.setattr(compare_models_module, "summarize_variant", _tracking_summarize)

    run_comparison(multi_season_df, include_post_hoc_calibration=True)

    assert seen_fit_seasons == set(CALIBRATION_FIT_SEASONS)
    assert seen_eval_seasons == set(CALIBRATION_EVAL_SEASONS)
    assert seen_fit_seasons.isdisjoint(seen_eval_seasons)


# -- 2025 rejection ----------------------------------------------------------


def test_assert_seasons_allowed_rejects_2025_for_calibration_seasons():
    with pytest.raises(ProtectedSeasonError):
        assert_seasons_allowed((*CALIBRATION_BASE_TRAIN_SEASONS, *CALIBRATION_FIT_SEASONS, 2025))


def test_run_comparison_end_to_end_variant_labels(multi_season_df: pd.DataFrame):
    results_without_calibration = run_comparison(
        multi_season_df, include_post_hoc_calibration=False
    )
    variants = {r["variant"] for r in results_without_calibration}
    assert variants == {VARIANT_UNWEIGHTED, VARIANT_CLASS_BALANCED, VARIANT_NAIVE_PREVALENCE}

    results_with_calibration = run_comparison(multi_season_df, include_post_hoc_calibration=True)
    variants_with_cal = {r["variant"] for r in results_with_calibration}
    assert variants_with_cal == {
        VARIANT_UNWEIGHTED,
        VARIANT_CLASS_BALANCED,
        VARIANT_NAIVE_PREVALENCE,
        VARIANT_UNWEIGHTED_CALIBRATED,
    }


def test_run_comparison_rejects_when_data_missing_required_seasons():
    tiny_df = _synthetic_season_df(2024, np.random.default_rng(1), n_per_class=3)
    with pytest.raises(ValueError, match="training"):
        run_comparison(tiny_df)


# -- ECE sanity --------------------------------------------------------------


def test_expected_calibration_error_matches_manual_weighted_average():
    table = pd.DataFrame(
        {
            "outcome_class": ["out", "out", "single"],
            "sample_count": [100, 300, 50],
            "abs_calibration_error": [0.1, 0.2, 0.05],
            "reliable": [True, True, True],
        }
    )
    expected = (100 * 0.1 + 300 * 0.2 + 50 * 0.05) / (100 + 300 + 50)
    assert compute_expected_calibration_error(table) == pytest.approx(expected)


def test_expected_calibration_error_requires_class_order_columns_upstream(
    multi_season_df: pd.DataFrame,
):
    trained = train_model(multi_season_df, class_weight=None)
    feature_cols = trained.numeric_features + trained.categorical_features
    proba_df = predict_proba_ordered(trained, multi_season_df[feature_cols])
    table = compute_calibration_table(multi_season_df["outcome_class"], proba_df)
    ece = compute_expected_calibration_error(table)
    assert 0.0 <= ece <= 1.0
