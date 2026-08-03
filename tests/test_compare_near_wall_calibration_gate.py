"""Tests for the Contact Luck v0.7D sample-size-aware calibration gate.

No test touches the network or a trained model -- every scenario here is built
from synthetic `y_true`/`p_out`/`game_pk` arrays directly, since the whole
point of this module is the STATISTICAL evaluation layer sitting on top of
whatever probabilities a model produces, not the model itself (see
`tests/test_compare_near_wall_models.py` for near-wall MODEL tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    MIN_CLASS_SUPPORT,
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
    OVERALL_STATUS_CALIBRATED,
    OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE,
    OVERALL_STATUS_NOT_CALIBRATED,
    SUBGROUP_STATUS_CALIBRATED,
    SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
    SUBGROUP_STATUS_NOT_CALIBRATED,
    SubgroupCalibrationEvidence,
    build_required_subgroup_masks,
    build_venue_masks,
    classify_subgroup_status,
    compute_adaptive_calibration_table,
    compute_calibration_intercept_slope,
    compute_subgroup_bootstrap,
    evaluate_subgroup_calibration,
    has_adequate_support,
    plot_subgroup_reliability_diagrams,
    summarize_calibration_gate,
)
from mlb_luck_score.models.compare_near_wall_models import (
    VARIANT_NEAR_WALL_FINAL,
    VARIANT_OPEN_FIELD,
)
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _tiled_subgroup(
    n_games_per_tile: int,
    plays_per_game: int,
    tiles: int,
    true_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(y_true, game_pks)` built by tiling one base block of games `tiles` times.

    Tiling guarantees the point-estimate outcome rate (and therefore ECE
    against a fixed `p_out`) is EXACTLY identical regardless of `tiles` --
    only the number of distinct games (and therefore bootstrap CI width)
    changes. This is the mechanism behind `test_identical_ece_...`.
    """
    rng = np.random.default_rng(seed)
    n_base = n_games_per_tile * plays_per_game
    base_y = rng.binomial(1, true_rate, size=n_base)
    game_ids_base = np.repeat(np.arange(n_games_per_tile), plays_per_game)
    y_all = np.tile(base_y, tiles)
    game_ids_all = np.concatenate([game_ids_base + t * n_games_per_tile for t in range(tiles)])
    return y_all, game_ids_all


# ---------------------------------------------------------------------------
# Adaptive-bin ECE
# ---------------------------------------------------------------------------


def test_adaptive_calibration_table_bin_count_scales_with_sample_size():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, size=1000)
    y = rng.binomial(1, p)
    small_table = compute_adaptive_calibration_table(y[:100], pd.Series(p[:100]))
    large_table = compute_adaptive_calibration_table(y, pd.Series(p))
    assert len(small_table) <= 5  # 100 // 20 = 5
    assert len(large_table) == 10  # capped at ADAPTIVE_ECE_MAX_BINS


def test_adaptive_calibration_table_single_bin_for_constant_probability():
    y = np.array([0, 1, 1, 0, 1] * 40)
    p = pd.Series([0.5] * len(y))
    table = compute_adaptive_calibration_table(y, p)
    assert len(table) == 1
    assert table.iloc[0]["sample_count"] == len(y)


def test_adaptive_calibration_table_empty_input():
    table = compute_adaptive_calibration_table(np.array([]), pd.Series([], dtype=float))
    assert table.empty


# ---------------------------------------------------------------------------
# Calibration intercept/slope
# ---------------------------------------------------------------------------


def test_calibration_intercept_slope_estimable_for_adequate_varied_data():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, size=200)
    y = rng.binomial(1, p)
    intercept, slope = compute_calibration_intercept_slope(y, p)
    assert intercept is not None
    assert slope is not None


def test_calibration_intercept_slope_not_estimable_below_min_samples():
    y = np.array([0, 1] * 5)
    p = np.array([0.3, 0.7] * 5)
    intercept, slope = compute_calibration_intercept_slope(y, p, min_samples=100)
    assert intercept is None
    assert slope is None


def test_calibration_intercept_slope_not_estimable_single_class():
    y = np.ones(200, dtype=int)
    p = np.linspace(0.1, 0.9, 200)
    intercept, slope = compute_calibration_intercept_slope(y, p)
    assert intercept is None
    assert slope is None


def test_calibration_intercept_slope_not_estimable_constant_probability():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.5, size=200)
    p = np.full(200, 0.5)
    intercept, slope = compute_calibration_intercept_slope(y, p)
    assert intercept is None
    assert slope is None


# ---------------------------------------------------------------------------
# Minimum-support gate
# ---------------------------------------------------------------------------


def test_has_adequate_support_true_when_all_thresholds_met():
    assert has_adequate_support(
        n_plays=MIN_SUBGROUP_PLAYS,
        n_games=MIN_SUBGROUP_GAMES,
        n_positive=MIN_CLASS_SUPPORT,
        n_negative=MIN_CLASS_SUPPORT,
    )


def test_has_adequate_support_false_below_each_threshold_independently():
    ok = dict(
        n_plays=MIN_SUBGROUP_PLAYS,
        n_games=MIN_SUBGROUP_GAMES,
        n_positive=MIN_CLASS_SUPPORT,
        n_negative=MIN_CLASS_SUPPORT,
    )
    assert not has_adequate_support(**{**ok, "n_plays": MIN_SUBGROUP_PLAYS - 1})
    assert not has_adequate_support(**{**ok, "n_games": MIN_SUBGROUP_GAMES - 1})
    assert not has_adequate_support(**{**ok, "n_positive": MIN_CLASS_SUPPORT - 1})
    assert not has_adequate_support(**{**ok, "n_negative": MIN_CLASS_SUPPORT - 1})


# ---------------------------------------------------------------------------
# Status classification (task item 11: "all three status classifications")
# ---------------------------------------------------------------------------


def test_classify_status_insufficient_evidence_when_support_inadequate():
    status, reason = classify_subgroup_status(
        has_adequate_support=False,
        specialist_ece_ci_low=0.01,
        specialist_ece_ci_high=0.02,
        paired_log_loss_delta_ci_low=-0.1,
    )
    assert status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE
    assert "support" in reason


def test_classify_status_calibrated_when_ece_ci_entirely_below_threshold():
    status, _reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=0.01,
        specialist_ece_ci_high=0.03,
        paired_log_loss_delta_ci_low=-0.05,
    )
    assert status == SUBGROUP_STATUS_CALIBRATED


def test_classify_status_not_calibrated_when_ece_ci_entirely_above_threshold():
    status, reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=0.08,
        specialist_ece_ci_high=0.15,
        paired_log_loss_delta_ci_low=None,
    )
    assert status == SUBGROUP_STATUS_NOT_CALIBRATED
    assert "ECE confidence interval" in reason


def test_classify_status_not_calibrated_when_paired_regression_credible():
    status, reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=0.01,
        specialist_ece_ci_high=0.03,
        paired_log_loss_delta_ci_low=0.02,  # entirely above zero -- credible regression
    )
    assert status == SUBGROUP_STATUS_NOT_CALIBRATED
    assert "regression" in reason


def test_classify_status_insufficient_evidence_when_ci_straddles_threshold():
    status, reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=0.02,
        specialist_ece_ci_high=0.08,  # straddles 0.05
        paired_log_loss_delta_ci_low=-0.1,
    )
    assert status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE
    assert "straddles" in reason


def test_classify_status_insufficient_evidence_when_ci_missing():
    status, reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=None,
        specialist_ece_ci_high=None,
        paired_log_loss_delta_ci_low=None,
    )
    assert status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE
    assert "could not be computed" in reason


def test_classify_status_does_not_fail_merely_from_point_estimate_alone():
    """Task item 4: a subgroup must not be marked not_calibrated purely because
    a point estimate (not represented by these args at all -- only the CI is)
    exceeds the threshold; only a CI entirely above it can trigger that."""
    # Point estimate could be arbitrarily bad; classify_subgroup_status never
    # even receives it -- only the CI bounds. A CI whose lower bound is at or
    # below threshold cannot yield not_calibrated no matter how bad ci_high is.
    status, _reason = classify_subgroup_status(
        has_adequate_support=True,
        specialist_ece_ci_low=0.03,
        specialist_ece_ci_high=0.9,
        paired_log_loss_delta_ci_low=None,
    )
    assert status != SUBGROUP_STATUS_NOT_CALIBRATED


# ---------------------------------------------------------------------------
# Game-clustered bootstrap
# ---------------------------------------------------------------------------


def test_subgroup_bootstrap_deterministic_given_seed():
    y, g = _tiled_subgroup(25, 8, 2, 0.4, seed=11)
    p = np.full(len(y), 0.4)
    baseline = np.full(len(y), 0.4)
    result_a = compute_subgroup_bootstrap(y, p, baseline, g, n_reps=100, seed=99)
    result_b = compute_subgroup_bootstrap(y, p, baseline, g, n_reps=100, seed=99)
    assert result_a["specialist_ece"]["ci_low"] == result_b["specialist_ece"]["ci_low"]
    assert result_a["specialist_ece"]["ci_high"] == result_b["specialist_ece"]["ci_high"]


def test_subgroup_bootstrap_resamples_at_game_level_not_row_level():
    """With only 2 distinct games, every bootstrap replicate must be one of the
    4 possible (with-replacement) combinations of those 2 games -- so the
    point estimate of a replicate using both copies of game 0 must exactly
    equal game 0's own metric, never some blend only reachable by resampling
    individual ROWS across games."""
    game0_y = np.array([1, 1, 1, 0, 0])
    game1_y = np.array([0, 0, 0, 0, 1])
    y = np.concatenate([game0_y, game1_y])
    g = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)])
    p = np.full(10, 0.5)
    baseline = np.full(10, 0.5)

    result = compute_subgroup_bootstrap(y, p, baseline, g, n_reps=50, seed=1)
    # Only 2 unique games -- reported faithfully in the bootstrap metadata.
    assert result["specialist_ece"]["n_unique_games"] == 2


def test_subgroup_bootstrap_paired_delta_detects_specialist_worse():
    y, g = _tiled_subgroup(30, 10, 1, 0.75, seed=5)
    specialist_p = np.full(len(y), 0.5)  # badly miscalibrated
    baseline_p = np.full(len(y), 0.75)  # well calibrated
    result = compute_subgroup_bootstrap(y, specialist_p, baseline_p, g, n_reps=300, seed=42)
    paired = result["paired_log_loss_delta"]
    assert paired["point_estimate"] > 0
    assert paired["ci_low"] > 0  # credibly worse, not just point-estimate worse


def test_subgroup_bootstrap_paired_delta_detects_specialist_better():
    y, g = _tiled_subgroup(30, 10, 1, 0.75, seed=5)
    specialist_p = np.full(len(y), 0.75)  # well calibrated
    baseline_p = np.full(len(y), 0.5)  # badly miscalibrated
    result = compute_subgroup_bootstrap(y, specialist_p, baseline_p, g, n_reps=300, seed=42)
    paired = result["paired_log_loss_delta"]
    assert paired["point_estimate"] < 0
    assert paired["ci_high"] < 0  # credibly better


# ---------------------------------------------------------------------------
# End-to-end: evaluate_subgroup_calibration scenarios (task item 11)
# ---------------------------------------------------------------------------


def test_identical_ece_different_sample_size_yields_different_certainty():
    """Task item 11: identical point-estimate ECE, different sample sizes ->
    different certainty. Tiling the SAME base block guarantees an EXACTLY
    identical point-estimate ECE; only the number of distinct games differs.
    """
    small_y, small_g = _tiled_subgroup(25, 4, 1, 0.56, seed=7)
    large_y, large_g = _tiled_subgroup(25, 4, 10, 0.56, seed=7)

    p_small = pd.Series([0.5] * len(small_y))
    p_large = pd.Series([0.5] * len(large_y))

    small_evidence = evaluate_subgroup_calibration(
        "small",
        np.ones(len(small_y), dtype=bool),
        small_y,
        p_small,
        p_small,
        pd.Series(small_g),
        n_reps=500,
        seed=42,
    )
    large_evidence = evaluate_subgroup_calibration(
        "large",
        np.ones(len(large_y), dtype=bool),
        large_y,
        p_large,
        p_large,
        pd.Series(large_g),
        n_reps=500,
        seed=42,
    )

    # Identical point-estimate ECE by construction.
    assert small_evidence.specialist_adaptive_ece == pytest.approx(
        large_evidence.specialist_adaptive_ece
    )
    # But the small subgroup's evidence is inconclusive while the large one's
    # is credible -- exactly the "different certainty" the task describes.
    small_width = small_evidence.specialist_ece_ci_high - small_evidence.specialist_ece_ci_low
    large_width = large_evidence.specialist_ece_ci_high - large_evidence.specialist_ece_ci_low
    assert large_width < small_width
    assert small_evidence.status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE
    assert large_evidence.status == SUBGROUP_STATUS_NOT_CALIBRATED


def test_insufficient_outcome_support_marks_insufficient_evidence():
    y, g = _tiled_subgroup(30, 20, 1, 0.02, seed=9)  # rare positive outcomes
    p = pd.Series([0.02] * len(y))
    evidence = evaluate_subgroup_calibration(
        "rare_positive", np.ones(len(y), dtype=bool), y, p, p, pd.Series(g)
    )
    assert evidence.n_positive < MIN_CLASS_SUPPORT
    assert evidence.has_adequate_support is False
    assert evidence.status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE


def test_calibrated_status_with_adequate_evidence():
    y, g = _tiled_subgroup(30, 20, 3, 0.5, seed=3)
    p = pd.Series([0.5] * len(y))
    evidence = evaluate_subgroup_calibration(
        "well_calibrated", np.ones(len(y), dtype=bool), y, p, p, pd.Series(g)
    )
    assert evidence.has_adequate_support is True
    assert evidence.status == SUBGROUP_STATUS_CALIBRATED


def test_paired_regression_flags_not_calibrated_even_with_ok_absolute_ece():
    """Task item 6: compare specialist against baseline on the SAME rows --
    a credible paired regression alone (not necessarily a bad absolute ECE)
    is sufficient for not_calibrated (task item 5's second bullet)."""
    y, g = _tiled_subgroup(30, 10, 1, 0.75, seed=5)
    specialist_p = pd.Series([0.5] * len(y))
    baseline_p = pd.Series([0.75] * len(y))
    evidence = evaluate_subgroup_calibration(
        "regressed", np.ones(len(y), dtype=bool), y, specialist_p, baseline_p, pd.Series(g)
    )
    assert evidence.paired_log_loss_delta_ci_low is not None
    assert evidence.paired_log_loss_delta_ci_low > 0
    assert evidence.status == SUBGROUP_STATUS_NOT_CALIBRATED


def test_empty_mask_is_insufficient_evidence():
    y = np.array([0, 1, 0, 1])
    p = pd.Series([0.5] * 4)
    g = pd.Series([1, 1, 2, 2])
    evidence = evaluate_subgroup_calibration("empty", np.zeros(4, dtype=bool), y, p, p, g)
    assert evidence.n_plays == 0
    assert evidence.status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Subgroup/venue mask construction
# ---------------------------------------------------------------------------


def _synthetic_final_df(n: int = 400, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "game_pk": rng.integers(1, 40, size=n),
            "venue_id": rng.choice(["1", "2", "3", pd.NA], size=n),
            "near_wall_5ft": rng.choice(["True", "False"], size=n),
            "near_wall_10ft": rng.choice(["True", "False"], size=n),
            "near_wall_20ft": rng.choice(["True", "False"], size=n),
            "wall_height_in_spray_direction": rng.uniform(6.0, 20.0, size=n),
            "spray_sector": rng.choice(
                ["left", "left_center", "center", "right_center", "right"], size=n
            ),
            "estimated_hang_time_s": rng.uniform(2.0, 6.0, size=n),
        }
    )


def test_build_required_subgroup_masks_covers_all_required_categories():
    df = _synthetic_final_df()
    masks = build_required_subgroup_masks(df)
    for expected in (
        "near_wall_5ft",
        "near_wall_10ft",
        "near_wall_20ft",
        "wall_height_short",
        "wall_height_medium",
        "wall_height_tall",
        "spray_sector_left",
        "spray_sector_center",
        "spray_sector_right",
        "opportunity_time_q1_shortest",
        "opportunity_time_q4_longest",
    ):
        assert expected in masks
        assert masks[expected].dtype == bool
        assert len(masks[expected]) == len(df)


def test_build_venue_masks_groups_missing_venue():
    df = _synthetic_final_df()
    masks = build_venue_masks(df["venue_id"])
    assert any(k.endswith("__missing_venue__") for k in masks)
    total_covered = sum(m.sum() for m in masks.values())
    assert total_covered == len(df)


# ---------------------------------------------------------------------------
# Overall gate: three overall statuses (task items 9-10)
# ---------------------------------------------------------------------------


def _passing_perturbation_check() -> DirectionalCheckResult:
    return DirectionalCheckResult(
        label="x",
        outcome_class="out",
        low_label="low",
        high_label="high",
        mean_prob_low=0.3,
        mean_prob_high=0.6,
        delta=0.3,
        expect_high_greater=True,
        passed=True,
        sample_size=1000,
    )


class _FakeSuite:
    def __init__(self, required):
        self.required = required


def _summary(log_loss: float) -> dict:
    return {"binary_log_loss": log_loss, "expected_calibration_error": 0.01}


def _evidence(label: str, status: str) -> SubgroupCalibrationEvidence:
    return SubgroupCalibrationEvidence(
        label=label,
        n_plays=500,
        n_games=50,
        n_positive=250,
        n_negative=250,
        has_adequate_support=status != SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
        specialist_log_loss=0.5,
        specialist_brier_score=0.2,
        specialist_adaptive_ece=0.02,
        specialist_calibration_intercept=0.0,
        specialist_calibration_slope=1.0,
        specialist_ece_ci_low=0.01,
        specialist_ece_ci_high=0.03,
        baseline_log_loss=0.6,
        baseline_brier_score=0.25,
        baseline_adaptive_ece=0.1,
        paired_log_loss_delta=-0.1,
        paired_log_loss_delta_ci_low=-0.15,
        paired_log_loss_delta_ci_high=-0.05,
        paired_ece_delta=-0.08,
        paired_ece_delta_ci_low=-0.12,
        paired_ece_delta_ci_high=-0.04,
        status=status,
        status_reason="synthetic fixture",
    )


def _base_args():
    comparison = {
        VARIANT_OPEN_FIELD: _summary(0.6),
        VARIANT_NEAR_WALL_FINAL: _summary(0.4),
    }
    aggregate_bootstrap = {"log_loss_delta": {"ci_high": -0.05}}
    perturbation_suite = _FakeSuite({"wall_distance_direction": _passing_perturbation_check()})
    open_field_regression_check = {"predictions_identical_via_gate": True}
    return comparison, aggregate_bootstrap, perturbation_suite, open_field_regression_check


def test_overall_status_calibrated_when_every_group_calibrated():
    comparison, bootstrap, suite, regression_check = _base_args()
    subgroup_evidence = [_evidence("a", SUBGROUP_STATUS_CALIBRATED)]
    venue_evidence = [_evidence("venue_1", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_calibration_gate(
        comparison, bootstrap, subgroup_evidence, venue_evidence, suite, regression_check
    )
    assert summary["overall_status"] == OVERALL_STATUS_CALIBRATED
    assert summary["near_wall_specialist_calibrated"] is True


def test_overall_status_calibrated_with_limited_evidence_when_some_groups_lack_evidence():
    comparison, bootstrap, suite, regression_check = _base_args()
    subgroup_evidence = [
        _evidence("a", SUBGROUP_STATUS_CALIBRATED),
        _evidence("b", SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE),
    ]
    venue_evidence = [_evidence("venue_1", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_calibration_gate(
        comparison, bootstrap, subgroup_evidence, venue_evidence, suite, regression_check
    )
    assert summary["overall_status"] == OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE
    # Task item 10: insufficient evidence must never be reported as a pass.
    assert summary["near_wall_specialist_calibrated"] is False


def test_overall_status_not_calibrated_when_any_group_credibly_fails():
    comparison, bootstrap, suite, regression_check = _base_args()
    subgroup_evidence = [
        _evidence("a", SUBGROUP_STATUS_CALIBRATED),
        _evidence("b", SUBGROUP_STATUS_NOT_CALIBRATED),
    ]
    venue_evidence = []
    summary = summarize_calibration_gate(
        comparison, bootstrap, subgroup_evidence, venue_evidence, suite, regression_check
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert summary["near_wall_specialist_calibrated"] is False
    assert "b" in summary["not_calibrated_groups"]


def test_overall_status_not_calibrated_when_perturbation_check_fails():
    comparison, bootstrap, _suite, regression_check = _base_args()
    failing_check = DirectionalCheckResult(
        label="x",
        outcome_class="out",
        low_label="low",
        high_label="high",
        mean_prob_low=0.6,
        mean_prob_high=0.3,
        delta=-0.3,
        expect_high_greater=True,
        passed=False,
        sample_size=1000,
    )
    suite = _FakeSuite({"wall_distance_direction": failing_check})
    subgroup_evidence = [_evidence("a", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_calibration_gate(
        comparison, bootstrap, subgroup_evidence, [], suite, regression_check
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert summary["near_wall_specialist_calibrated"] is False


def test_overall_status_not_calibrated_when_open_field_regression_check_fails():
    comparison, bootstrap, suite, _regression_check = _base_args()
    subgroup_evidence = [_evidence("a", SUBGROUP_STATUS_CALIBRATED)]
    summary = summarize_calibration_gate(
        comparison,
        bootstrap,
        subgroup_evidence,
        [],
        suite,
        {"predictions_identical_via_gate": False},
    )
    assert summary["overall_status"] == OVERALL_STATUS_NOT_CALIBRATED
    assert summary["near_wall_specialist_calibrated"] is False


def test_summary_reports_honest_note_distinguishing_failure_from_evidence_gap():
    comparison, bootstrap, suite, regression_check = _base_args()
    summary = summarize_calibration_gate(
        comparison,
        bootstrap,
        [_evidence("a", SUBGROUP_STATUS_CALIBRATED)],
        [],
        suite,
        regression_check,
    )
    assert (
        "insufficient_evidence" in summary["measured_miscalibration_vs_insufficient_evidence_note"]
    )
    assert "not_calibrated" in summary["measured_miscalibration_vs_insufficient_evidence_note"]


# ---------------------------------------------------------------------------
# Reliability plots
# ---------------------------------------------------------------------------


def test_plot_subgroup_reliability_diagrams_writes_one_file_per_nonempty_mask(tmp_path):
    rng = np.random.default_rng(6)
    n = 200
    y = rng.binomial(1, 0.5, size=n)
    p = pd.Series(rng.uniform(0, 1, size=n))
    masks = {
        "group_a": np.ones(n, dtype=bool),
        "group_b_empty": np.zeros(n, dtype=bool),
    }
    written = plot_subgroup_reliability_diagrams(y, p, masks, tmp_path)
    assert len(written) == 1
    assert written[0].exists()
    assert written[0].name == "reliability_group_a.png"
