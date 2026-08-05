"""Tests for the Contact Luck v0.9 batter-runner advancement model comparison.

No test touches the network -- synthetic advancement-shaped fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import CALIBRATION_BASE_TRAIN_SEASONS, CALIBRATION_EVAL_SEASONS
from mlb_luck_score.eligibility import ADVANCEMENT_LABELS
from mlb_luck_score.models.compare_advancement_models import (
    ADVANCEMENT_SELECTION_CANDIDATES,
    VARIANT_ADVANCEMENT_EMPIRICAL,
    AdvancementDirectionalCheckResult,
    build_advancement_subgroup_masks,
    build_advancement_venue_masks,
    compute_expected_advancement_index,
    compute_sprint_speed_buckets,
    fit_empirical_advancement_baseline,
    predict_empirical_baseline_proba,
    run_advancement_final_comparison,
    run_advancement_model_selection,
    run_advancement_perturbation_checks,
    summarize_advancement_calibration,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    OVERALL_STATUS_CALIBRATED,
    SUBGROUP_STATUS_CALIBRATED,
    SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
    SUBGROUP_STATUS_NOT_CALIBRATED,
    SubgroupCalibrationEvidence,
)


def _synthetic_advancement_rows(
    season: int, rng: np.random.Generator, *, n: int = 150
) -> list[dict]:
    rows = []
    hit_types = [("single_or_error", 1), ("double", 2), ("triple", 3)]
    for i in range(n):
        hit_type_group, floor = hit_types[i % 3]
        sprint_speed = rng.uniform(22.0, 30.0)
        # Faster batters advance further, deterministically enough to learn.
        p_extra = 0.1 + 0.5 * (sprint_speed - 22.0) / 8.0
        advances = rng.random() < p_extra
        final_base = min(floor + (1 if advances else 0), 4)
        label = {
            1: "held_at_first",
            2: "advanced_to_second",
            3: "advanced_to_third",
            4: "inside_the_park_home_run",
        }[final_base]
        rows.append(
            {
                "game_pk": season * 1000 + i,
                "season": season,
                "launch_speed": 95.0 + rng.normal(0, 3),
                "launch_angle": 20.0 + rng.normal(0, 3),
                "spray_angle_approx": rng.uniform(-15, 15),
                "hit_distance_sc": 300.0 + rng.normal(0, 10),
                "estimated_hang_time_s": 4.0 + rng.normal(0, 0.3),
                "landing_x_ft": rng.normal(0, 10),
                "landing_y_ft": 300.0,
                "wall_distance_in_spray_direction": 380.0,
                "absolute_distance_to_wall": 80.0,
                "wall_height_in_spray_direction": 8.0,
                "outs_when_up": i % 3,
                "on_1b_occupied": 0,
                "on_2b_occupied": 0,
                "on_3b_occupied": 0,
                "assigned_outfield_position": ("7", "8", "9")[i % 3],
                "of_fielding_alignment": "Standard",
                "stand": "R" if i % 2 == 0 else "L",
                "spray_sector": ("left", "left_center", "center", "right_center", "right")[i % 5],
                "venue_id": str(1 + (i % 2)),
                "sprint_speed": sprint_speed,
                "hit_type_group": hit_type_group,
                "hit_type_implied_floor_base": floor,
                "contact_p_out": 0.3,
                "contact_p_single": 0.3,
                "contact_p_double": 0.2,
                "contact_p_triple": 0.1,
                "contact_p_home_run": 0.1,
                "batter_final_base": label,
            }
        )
    return rows


@pytest.fixture
def advancement_raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        rows += _synthetic_advancement_rows(season, rng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_model_selection_picks_a_registered_candidate(advancement_raw_df):
    winner, selection_metrics, trained_by_candidate = run_advancement_model_selection(
        advancement_raw_df
    )
    assert winner in ADVANCEMENT_SELECTION_CANDIDATES
    assert set(selection_metrics.keys()) == set(ADVANCEMENT_SELECTION_CANDIDATES)
    assert set(trained_by_candidate.keys()) == set(ADVANCEMENT_SELECTION_CANDIDATES)


def test_model_selection_rejects_empty_seasons():
    df = pd.DataFrame({"season": [2024] * 10})
    with pytest.raises(ValueError, match="fit"):
        run_advancement_model_selection(df)


# ---------------------------------------------------------------------------
# Empirical baseline
# ---------------------------------------------------------------------------


def test_sprint_speed_buckets_produce_four_groups():
    speeds = pd.Series(np.linspace(22.0, 30.0, 100))
    buckets = compute_sprint_speed_buckets(speeds)
    assert set(buckets.dropna().unique()) == {"q1_slowest", "q2", "q3", "q4_fastest"}


def test_empirical_baseline_probabilities_sum_to_one(advancement_raw_df):
    fit_df = advancement_raw_df[advancement_raw_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    baseline = fit_empirical_advancement_baseline(fit_df)
    proba = predict_empirical_baseline_proba(baseline, advancement_raw_df)
    assert list(proba.columns) == list(ADVANCEMENT_LABELS)
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_empirical_baseline_falls_back_to_overall_for_unseen_group(advancement_raw_df):
    fit_df = advancement_raw_df[advancement_raw_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    baseline = fit_empirical_advancement_baseline(fit_df)
    unseen = pd.DataFrame({"hit_type_implied_floor_base": [999], "sprint_speed": [float("nan")]})
    proba = predict_empirical_baseline_proba(baseline, unseen)
    row = proba.iloc[0].to_dict()
    assert row == pytest.approx(baseline.overall)


# ---------------------------------------------------------------------------
# Perturbation check and expected-advancement-index
# ---------------------------------------------------------------------------


def test_expected_advancement_index_uses_base_value_scale():
    proba = pd.DataFrame(
        {
            "held_at_first": [1.0],
            "advanced_to_second": [0.0],
            "advanced_to_third": [0.0],
            "inside_the_park_home_run": [0.0],
            "retired_while_advancing": [0.0],
        }
    )
    idx = compute_expected_advancement_index(proba)
    assert idx.iloc[0] == pytest.approx(1.0)


def test_expected_advancement_index_scores_retired_below_held_at_first():
    proba = pd.DataFrame(
        {
            "held_at_first": [0.0],
            "advanced_to_second": [0.0],
            "advanced_to_third": [0.0],
            "inside_the_park_home_run": [0.0],
            "retired_while_advancing": [1.0],
        }
    )
    idx = compute_expected_advancement_index(proba)
    assert idx.iloc[0] == pytest.approx(0.0)


def _fit_speed_candidate(advancement_raw_df: pd.DataFrame):
    winner, selection_metrics, trained_by_candidate = run_advancement_model_selection(
        advancement_raw_df
    )
    return trained_by_candidate[winner], winner


def test_sprint_speed_perturbation_detects_correct_direction(advancement_raw_df):
    trained, _ = _fit_speed_candidate(advancement_raw_df)
    results = run_advancement_perturbation_checks(trained, advancement_raw_df)
    assert "sprint_speed_direction" in results
    check = results["sprint_speed_direction"]
    assert isinstance(check, AdvancementDirectionalCheckResult)
    # The synthetic fixture deterministically encodes "faster -> more
    # advancement", so the model should learn it correctly (non-negative delta).
    assert check.passed is True
    assert check.delta >= 0


# ---------------------------------------------------------------------------
# Subgroup / venue masks
# ---------------------------------------------------------------------------


def test_subgroup_masks_cover_required_dimensions(advancement_raw_df):
    masks = build_advancement_subgroup_masks(advancement_raw_df)
    n = len(advancement_raw_df)
    assert any(k.startswith("sprint_speed_") for k in masks)
    assert any(k.startswith("contact_type_") for k in masks)
    assert any(k.startswith("spray_sector_") for k in masks)
    assert any(k.startswith("outfielder_position_") for k in masks)
    assert any(k.startswith("wall_proximity_") for k in masks)
    for mask in masks.values():
        assert mask.shape == (n,)


def test_venue_masks_one_per_distinct_venue(advancement_raw_df):
    masks = build_advancement_venue_masks(advancement_raw_df["venue_id"])
    assert set(masks.keys()) == {f"venue_{v}" for v in advancement_raw_df["venue_id"].unique()}


def test_venue_masks_empty_when_no_venue_column():
    assert build_advancement_venue_masks(None) == {}


# ---------------------------------------------------------------------------
# Final comparison
# ---------------------------------------------------------------------------


def test_final_comparison_reports_winner_and_empirical_baseline(advancement_raw_df):
    winner_trained, winner = _fit_speed_candidate(advancement_raw_df)
    fit_df = advancement_raw_df[advancement_raw_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    baseline = fit_empirical_advancement_baseline(fit_df)
    comparison, final_df, winner_proba = run_advancement_final_comparison(
        advancement_raw_df, winner_trained, baseline
    )
    assert set(comparison.keys()) == {winner, VARIANT_ADVANCEMENT_EMPIRICAL}
    assert comparison[winner]["sample_count"] == int(
        (advancement_raw_df["season"].isin(CALIBRATION_EVAL_SEASONS)).sum()
    )
    assert len(final_df) == comparison[winner]["sample_count"]
    assert list(winner_proba.columns) == list(ADVANCEMENT_LABELS)


def test_final_comparison_rejects_empty_final_seasons():
    df = pd.DataFrame({"season": [2021] * 10})
    with pytest.raises(ValueError, match="final-comparison"):
        run_advancement_final_comparison(df, None, None)  # type: ignore[arg-type]


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
        specialist_log_loss=0.3,
        specialist_brier_score=0.1,
        specialist_adaptive_ece=0.01,
        specialist_calibration_intercept=0.0,
        specialist_calibration_slope=1.0,
        specialist_ece_ci_low=0.005,
        specialist_ece_ci_high=0.02,
        baseline_log_loss=0.3,
        baseline_brier_score=0.1,
        baseline_adaptive_ece=0.01,
        paired_log_loss_delta=0.0,
        paired_log_loss_delta_ci_low=0.0,
        paired_log_loss_delta_ci_high=0.0,
        paired_ece_delta=0.0,
        paired_ece_delta_ci_low=0.0,
        paired_ece_delta_ci_high=0.0,
        status=status,
        status_reason="test fixture",
    )


def _passing_perturbation() -> dict[str, AdvancementDirectionalCheckResult]:
    return {
        "sprint_speed_direction": AdvancementDirectionalCheckResult(
            label="x",
            low_label="slow",
            high_label="fast",
            mean_expected_advancement_low=1.0,
            mean_expected_advancement_high=1.5,
            delta=0.5,
            passed=True,
            sample_size=100,
        )
    }


def _failing_perturbation() -> dict[str, AdvancementDirectionalCheckResult]:
    check = _passing_perturbation()["sprint_speed_direction"]
    return {
        "sprint_speed_direction": AdvancementDirectionalCheckResult(
            **{**check.__dict__, "passed": False}
        )
    }


def _good_ece_comparison(winner: str, beats_baseline: bool = True) -> dict:
    winner_ll = 0.15
    baseline_ll = 0.18 if beats_baseline else 0.10
    return {
        winner: {
            "multiclass_log_loss": winner_ll,
            "per_class_ece": {label: 0.01 for label in ADVANCEMENT_LABELS},
        },
        VARIANT_ADVANCEMENT_EMPIRICAL: {"multiclass_log_loss": baseline_ll},
    }


def test_calibrated_true_when_beats_baseline_and_no_bad_pairs():
    winner = "advancement_speed_v09"
    comparison = _good_ece_comparison(winner)
    evidence = [_evidence("held_at_first__class_x", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_advancement_calibration(
        comparison, winner, evidence, _passing_perturbation()
    )
    assert summary["overall_status"] == OVERALL_STATUS_CALIBRATED
    assert summary["calibrated_advancement_model"] is True
    assert summary["beats_empirical_baseline"] is True


def test_not_calibrated_when_worse_than_empirical_baseline():
    winner = "advancement_speed_v09"
    comparison = _good_ece_comparison(winner, beats_baseline=False)
    summary = summarize_advancement_calibration(comparison, winner, [], _passing_perturbation())
    assert summary["beats_empirical_baseline"] is False
    assert summary["calibrated_advancement_model"] is False


def test_limited_evidence_when_some_pairs_lack_evidence():
    winner = "advancement_speed_v09"
    comparison = _good_ece_comparison(winner)
    evidence = [
        _evidence("a", SUBGROUP_STATUS_CALIBRATED),
        _evidence("b", SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE),
    ]
    summary = summarize_advancement_calibration(
        comparison, winner, evidence, _passing_perturbation()
    )
    assert summary["overall_status"] == "calibrated_with_limited_subgroup_evidence"
    assert summary["calibrated_advancement_model"] is False


def test_not_calibrated_when_a_pair_is_credibly_not_calibrated():
    winner = "advancement_speed_v09"
    comparison = _good_ece_comparison(winner)
    evidence = [_evidence("a", SUBGROUP_STATUS_NOT_CALIBRATED)]
    summary = summarize_advancement_calibration(
        comparison, winner, evidence, _passing_perturbation()
    )
    assert summary["overall_status"] == "not_calibrated"
    assert "a" in summary["not_calibrated_pairs"]


def test_not_calibrated_when_perturbation_fails_despite_good_ece():
    winner = "advancement_speed_v09"
    comparison = _good_ece_comparison(winner)
    summary = summarize_advancement_calibration(comparison, winner, [], _failing_perturbation())
    assert summary["overall_status"] == "not_calibrated"
    assert "sprint_speed_direction" in summary["perturbation_failures"]


def test_not_calibrated_when_ece_exceeds_threshold():
    winner = "advancement_speed_v09"
    comparison = {
        winner: {
            "multiclass_log_loss": 0.1,
            "per_class_ece": {label: 0.5 for label in ADVANCEMENT_LABELS},
        },
        VARIANT_ADVANCEMENT_EMPIRICAL: {"multiclass_log_loss": 0.2},
    }
    summary = summarize_advancement_calibration(comparison, winner, [], _passing_perturbation())
    assert summary["overall_ece_within_threshold"] is False
    assert summary["overall_status"] == "not_calibrated"
