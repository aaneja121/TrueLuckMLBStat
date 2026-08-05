"""Tests for the Contact Luck v0.8 infield opportunity model.

No test touches the network -- synthetic infield-ground-ball-shaped
fixtures only. The synthetic data deliberately encodes a real physical
relationship (a faster batter-runner is LESS likely to be retired) so the
model and perturbation checks have real, learnable signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import CALIBRATION_BASE_TRAIN_SEASONS, CALIBRATION_EVAL_SEASONS
from mlb_luck_score.features.build_contact_features import INFIELD_OPPORTUNITY_TARGET_COLUMN
from mlb_luck_score.models.compare_infield_opportunity import (
    INFIELD_SELECTION_CANDIDATES,
    VARIANT_INFIELD_FINAL,
    InfieldPerturbationSuite,
    ReachedOnErrorComparison,
    build_infield_subgroup_masks,
    build_infield_venue_masks,
    compute_grouped_sprint_speed_response,
    compute_launch_speed_partial_dependence,
    compute_reached_on_error_comparison,
    compute_sprint_speed_buckets,
    get_infield_rows,
    run_infield_final_comparison,
    run_infield_model_selection,
    run_infield_perturbation_checks,
    summarize_infield_calibration,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    OVERALL_STATUS_NOT_CALIBRATED,
    SUBGROUP_STATUS_CALIBRATED,
    SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
    SUBGROUP_STATUS_NOT_CALIBRATED,
    SubgroupCalibrationEvidence,
)
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult


def _synthetic_infield_rows(season: int, rng: np.random.Generator, *, n: int = 200) -> list[dict]:
    rows = []
    for i in range(n):
        # A faster batter is genuinely less likely to be retired.
        sprint_speed = rng.uniform(22.0, 30.0)
        p_safe = 0.15 + 0.6 * (sprint_speed - 22.0) / 8.0
        retired = rng.random() > p_safe
        rows.append(
            {
                "game_pk": season * 1000 + i,
                "season": season,
                "bb_type": "ground_ball",
                "events": "field_out" if retired else "single",
                "des": "Ground ball.",
                "hit_location": 4 + (i % 3),  # 4, 5, 6
                "launch_speed": 88.0 + rng.normal(0, 5),
                "launch_angle": -3.0 + rng.normal(0, 2),
                "spray_angle_approx": rng.uniform(-20, 20),
                "hit_distance_sc": 70.0 + rng.normal(0, 8),
                "pitcher": 1,
                "fielder_2": 202,
                "fielder_3": 303,
                "fielder_4": 404,
                "fielder_5": 505,
                "fielder_6": 606,
                "sprint_speed": sprint_speed,
                "hp_to_1b": 4.5 - (sprint_speed - 22.0) * 0.05,
                "outs_when_up": i % 3,
                "on_1b": None if i % 2 == 0 else 1,
                "stand": "R" if i % 2 == 0 else "L",
                "if_fielding_alignment": "Standard",
                "surface_type": "Grass",
                "spray_sector": ("left", "left_center", "center", "right_center", "right")[i % 5],
                "venue_id": str(1 + (i % 2)),
            }
        )
    return rows


@pytest.fixture
def infield_raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    rows: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        rows += _synthetic_infield_rows(season, rng)
    return pd.DataFrame(rows)


@pytest.fixture
def infield_df(infield_raw_df: pd.DataFrame) -> pd.DataFrame:
    return get_infield_rows(infield_raw_df)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def test_get_infield_rows_only_keeps_eligible_rows(infield_raw_df):
    result = get_infield_rows(infield_raw_df)
    assert len(result) > 0
    assert result["infield_opportunity_eligible"].all()
    assert len(result) <= len(infield_raw_df)


def test_get_infield_rows_has_infield_target_column(infield_df):
    assert INFIELD_OPPORTUNITY_TARGET_COLUMN in infield_df.columns
    pd.testing.assert_series_equal(
        infield_df[INFIELD_OPPORTUNITY_TARGET_COLUMN], infield_df["y_out"], check_names=False
    )


def test_get_infield_rows_does_not_write_the_outfield_target_column(infield_df):
    # Version 0.10 hardening: the infield builder must never write the
    # outfield's target column name -- the two domains no longer share one.
    assert "outfield_converted_to_out" not in infield_df.columns
    assert "converted_to_out" not in infield_df.columns


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_model_selection_picks_a_registered_candidate(infield_df):
    winner, selection_metrics, trained_by_candidate = run_infield_model_selection(infield_df)
    assert winner in INFIELD_SELECTION_CANDIDATES
    assert set(selection_metrics.keys()) == set(INFIELD_SELECTION_CANDIDATES)
    assert set(trained_by_candidate.keys()) == set(INFIELD_SELECTION_CANDIDATES)


def test_model_selection_uses_correct_season_split(infield_df, monkeypatch: pytest.MonkeyPatch):
    import mlb_luck_score.models.compare_infield_opportunity as cio

    seen_seasons: set[int] = set()
    original = cio.train_opportunity_model

    def _tracking_train(train_df, **kwargs):
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cio, "train_opportunity_model", _tracking_train)
    run_infield_model_selection(infield_df)
    assert seen_seasons == set(CALIBRATION_BASE_TRAIN_SEASONS)


def test_model_selection_rejects_empty_seasons():
    df = pd.DataFrame({"season": [2024] * 10})
    with pytest.raises(ValueError, match="fit"):
        run_infield_model_selection(df)


# ---------------------------------------------------------------------------
# Perturbation checks
# ---------------------------------------------------------------------------


def _fit_winner(infield_df: pd.DataFrame):
    fit_df = infield_df[infield_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    from mlb_luck_score.features.build_contact_features import select_infield_opportunity_features

    numeric_features, categorical_features = select_infield_opportunity_features(fit_df)
    return train_opportunity_model(
        fit_df,
        class_weight=None,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=INFIELD_OPPORTUNITY_TARGET_COLUMN,
    )


def test_sprint_speed_perturbation_detects_correct_direction(infield_df):
    trained = _fit_winner(infield_df)
    result = run_infield_perturbation_checks(trained, infield_df)
    assert isinstance(result, InfieldPerturbationSuite)
    assert "sprint_speed_direction" in result.required
    check = result.required["sprint_speed_direction"]
    assert isinstance(check, DirectionalCheckResult)
    # The synthetic fixture deterministically encodes "faster -> less likely
    # retired", so the model should learn it correctly.
    assert check.passed is True
    assert check.expect_high_greater is False
    assert result.timing_margin_check_status == "not_applicable"


def test_launch_speed_partial_dependence_is_descriptive_only(infield_df):
    trained = _fit_winner(infield_df)
    diagnostic = compute_launch_speed_partial_dependence(trained, infield_df)
    assert len(diagnostic.mean_p_out) == len(diagnostic.grid)
    assert isinstance(diagnostic.flagged_as_erratic, bool)


def test_sprint_speed_buckets_produce_four_groups():
    speeds = pd.Series(np.linspace(22.0, 30.0, 100))
    buckets = compute_sprint_speed_buckets(speeds)
    assert set(buckets.dropna().unique()) == {"q1_slowest", "q2", "q3", "q4_fastest"}


def test_grouped_sprint_speed_response_uses_only_real_rows(infield_df):
    trained = _fit_winner(infield_df)
    bins = compute_grouped_sprint_speed_response(trained, infield_df)
    assert len(bins) > 0
    real_speeds = set(infield_df["sprint_speed"].round(6))
    for b in bins:
        assert b.sample_size > 0
        assert 0.0 <= b.mean_predicted_p_out <= 1.0
        assert any(abs(b.mean_sprint_speed_fts - s) < 10 for s in real_speeds)


# ---------------------------------------------------------------------------
# Subgroup / venue masks
# ---------------------------------------------------------------------------


def test_subgroup_masks_cover_required_dimensions(infield_df):
    masks = build_infield_subgroup_masks(infield_df)
    n = len(infield_df)
    assert any(k.startswith("position_") for k in masks)
    assert "batter_stand_R" in masks
    assert "batter_stand_L" in masks
    assert any(k.startswith("sprint_speed_") for k in masks)
    assert "alignment_standard" in masks
    assert any(k.startswith("ground_ball_") for k in masks)
    assert any(k.startswith("spray_sector_") for k in masks)
    for mask in masks.values():
        assert mask.shape == (n,)


def test_venue_masks_one_per_distinct_venue(infield_df):
    masks = build_infield_venue_masks(infield_df["venue_id"])
    assert set(masks.keys()) == {f"venue_{v}" for v in infield_df["venue_id"].unique()}


def test_venue_masks_empty_when_no_venue_column():
    assert build_infield_venue_masks(None) == {}


# ---------------------------------------------------------------------------
# Final comparison / reached-on-error comparison
# ---------------------------------------------------------------------------


def test_final_comparison_reports_expected_keys(infield_df):
    trained = _fit_winner(infield_df)
    comparison, final_df, p_out = run_infield_final_comparison(infield_df, trained)
    assert comparison["variant"] == VARIANT_INFIELD_FINAL
    assert comparison["sample_count"] == int(
        (infield_df["season"].isin(CALIBRATION_EVAL_SEASONS)).sum()
    )
    assert len(final_df) == comparison["sample_count"]
    assert len(p_out) == comparison["sample_count"]
    assert 0.0 <= comparison["outcome_prevalence"] <= 1.0


def test_final_comparison_rejects_empty_final_seasons():
    df = pd.DataFrame({"season": [2021] * 10})
    with pytest.raises(ValueError, match="final-comparison"):
        run_infield_final_comparison(df, None)  # type: ignore[arg-type]


def test_reached_on_error_comparison_reports_sample_sizes(infield_df):
    trained = _fit_winner(infield_df)
    _, final_df, p_out = run_infield_final_comparison(infield_df, trained)
    result = compute_reached_on_error_comparison(final_df, p_out)
    assert isinstance(result, ReachedOnErrorComparison)
    assert result.reached_on_error_n == 0  # this fixture has no field_error rows
    assert result.reached_on_error_mean_predicted_p_out is None
    assert result.non_error_safe_n > 0


# ---------------------------------------------------------------------------
# Calibration-gate summary status logic
# ---------------------------------------------------------------------------


def _evidence(label: str, status: str) -> SubgroupCalibrationEvidence:
    return SubgroupCalibrationEvidence(
        label=label,
        n_plays=500,
        n_games=100,
        n_positive=200,
        n_negative=300,
        has_adequate_support=status != SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
        specialist_log_loss=0.5,
        specialist_brier_score=0.2,
        specialist_adaptive_ece=0.02,
        specialist_calibration_intercept=0.0,
        specialist_calibration_slope=1.0,
        specialist_ece_ci_low=0.01,
        specialist_ece_ci_high=0.03,
        baseline_log_loss=0.5,
        baseline_brier_score=0.2,
        baseline_adaptive_ece=0.02,
        paired_log_loss_delta=0.0,
        paired_log_loss_delta_ci_low=0.0,
        paired_log_loss_delta_ci_high=0.0,
        paired_ece_delta=0.0,
        paired_ece_delta_ci_low=0.0,
        paired_ece_delta_ci_high=0.0,
        status=status,
        status_reason="test fixture",
    )


def _passing_check() -> DirectionalCheckResult:
    return DirectionalCheckResult(
        label="infield: batter sprint speed slow vs fast",
        outcome_class="out",
        low_label="low",
        high_label="high",
        mean_prob_low=0.6,
        mean_prob_high=0.3,
        delta=-0.3,
        expect_high_greater=False,
        passed=True,
        sample_size=100,
    )


def _failing_check() -> DirectionalCheckResult:
    check = _passing_check()
    return DirectionalCheckResult(**{**check.__dict__, "passed": False})


def _suite(passed: bool) -> InfieldPerturbationSuite:
    return InfieldPerturbationSuite(
        required={"sprint_speed_direction": _passing_check() if passed else _failing_check()},
        partial_dependence={},
    )


def _reached_on_error() -> ReachedOnErrorComparison:
    return ReachedOnErrorComparison(
        reached_on_error_n=10,
        reached_on_error_mean_predicted_p_out=0.5,
        non_error_safe_n=100,
        non_error_safe_mean_predicted_p_out=0.5,
    )


def test_calibrated_true_when_ece_ok_perturbation_passes_and_no_bad_subgroups():
    comparison = {"expected_calibration_error": 0.01}
    subgroup_evidence = [_evidence("position_4", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_infield_calibration(
        comparison, subgroup_evidence, [], _suite(passed=True), _reached_on_error()
    )
    assert summary["overall_status"] == OVERALL_STATUS_CALIBRATED
    assert summary["calibrated_infield_opportunity"] is True


def test_limited_evidence_status_when_some_subgroups_lack_evidence():
    comparison = {"expected_calibration_error": 0.01}
    subgroup_evidence = [
        _evidence("position_4", SUBGROUP_STATUS_CALIBRATED),
        _evidence("position_5", SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE),
    ]
    summary = summarize_infield_calibration(
        comparison, subgroup_evidence, [], _suite(passed=True), _reached_on_error()
    )
    assert summary["overall_status"] == OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE
    assert summary["calibrated_infield_opportunity"] is False


def test_not_calibrated_when_a_subgroup_is_credibly_not_calibrated():
    comparison = {"expected_calibration_error": 0.01}
    subgroup_evidence = [_evidence("position_4", SUBGROUP_STATUS_NOT_CALIBRATED)]
    summary = summarize_infield_calibration(
        comparison, subgroup_evidence, [], _suite(passed=True), _reached_on_error()
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert "position_4" in summary["not_calibrated_groups"]


def test_not_calibrated_when_overall_ece_exceeds_threshold():
    comparison = {"expected_calibration_error": 0.5}
    summary = summarize_infield_calibration(
        comparison, [], [], _suite(passed=True), _reached_on_error()
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert summary["overall_ece_within_threshold"] is False


def test_not_calibrated_when_required_perturbation_fails_despite_good_ece():
    comparison = {"expected_calibration_error": 0.01}
    summary = summarize_infield_calibration(
        comparison, [], [], _suite(passed=False), _reached_on_error()
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert "sprint_speed_direction" in summary["perturbation_failures"]
