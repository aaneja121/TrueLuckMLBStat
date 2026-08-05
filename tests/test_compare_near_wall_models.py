"""Tests for the Contact Luck v0.7C near-wall opportunity specialist.

No test touches the network -- synthetic near-wall-shaped fixtures only.
The synthetic data deliberately encodes a real physical relationship (a
closer wall makes a play LESS catchable -- more likely a home run/wall-ball
extra-base hit than a routine out) so the model and perturbation checks
have real, learnable signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import (
    CALIBRATION_BASE_TRAIN_SEASONS,
    CALIBRATION_EVAL_SEASONS,
)
from mlb_luck_score.models.compare_near_wall_models import (
    BAND_SHORT_OF_WALL,
    NEAR_WALL_SELECTION_CANDIDATES,
    VARIANT_NEAR_WALL_FINAL,
    VARIANT_NEAR_WALL_LOGISTIC,
    VARIANT_OPEN_FIELD,
    BandedEffectResult,
    NearWallPerturbationSuite,
    OpportunityTimeResponseBin,
    _override_launch_angle_and_recompute_hang_time,
    _override_launch_angle_matched_trajectory,
    _override_wall_distance_and_recompute,
    compute_grouped_opportunity_time_response,
    compute_near_wall_paired_bootstrap,
    compute_near_wall_subgroups,
    compute_wall_height_buckets,
    find_material_subgroup_issues,
    run_near_wall_final_comparison,
    run_near_wall_model_selection,
    run_near_wall_perturbation_checks,
    summarize_near_wall_validation,
)
from mlb_luck_score.models.train_opportunity_model import (
    train_opportunity_model,
)
from mlb_luck_score.models.weather_perturbation import DirectionalCheckResult


def _synthetic_near_wall_rows(season: int, rng: np.random.Generator, *, n: int = 200) -> list[dict]:
    rows = []
    for i in range(n):
        day = 1 + (i % 27)
        close_wall = i % 2 == 0
        wall_distance = 320.0 + rng.normal(0, 3) if close_wall else 400.0 + rng.normal(0, 3)
        # Ball travels roughly the same distance regardless of wall -- a
        # close wall means it clears (home run / not an out); a far wall
        # means the same ball is caught (out).
        hit_distance = (
            wall_distance - rng.uniform(2, 8) if close_wall else wall_distance - rng.uniform(15, 25)
        )
        converted = 0 if close_wall else 1
        outcome_class = "home_run" if close_wall else "out"
        rows.append(
            {
                "game_pk": season * 1000 + day,
                "season": season,
                "launch_speed": 100.0 + rng.normal(0, 2),
                "launch_angle": 26.0 + rng.normal(0, 2),
                "hit_distance_sc": hit_distance,
                "estimated_hang_time_s": 4.0 + rng.normal(0, 0.3),
                "landing_x_ft": rng.normal(0, 10),
                "landing_y_ft": hit_distance,
                "wall_distance_in_spray_direction": wall_distance,
                "absolute_distance_to_wall": abs(hit_distance - wall_distance),
                "wall_height_in_spray_direction": 10.0 + rng.normal(0, 1),
                "projected_distance_to_wall_margin": hit_distance - wall_distance,
                "bb_type": "fly_ball",
                "of_fielding_alignment": "Standard",
                "assigned_outfield_position": "8",
                "wall_segment_label": "CF",
                "spray_sector": "center",
                "near_wall_5ft": "False",
                "near_wall_10ft": "True",
                "near_wall_20ft": "True",
                "venue_id": "3",
                "outcome_class": outcome_class,
                "outfield_converted_to_out": converted,
            }
        )
    return rows


@pytest.fixture
def near_wall_df() -> pd.DataFrame:
    rng = np.random.default_rng(55)
    rows: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        rows += _synthetic_near_wall_rows(season, rng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dependent-feature recomputation on override (the bug this caught)
# ---------------------------------------------------------------------------


def test_override_wall_distance_recomputes_margin_and_absolute_distance():
    df = pd.DataFrame(
        {
            "wall_distance_in_spray_direction": [350.0],
            "hit_distance_sc": [340.0],
            "absolute_distance_to_wall": [999.0],  # stale sentinel
            "projected_distance_to_wall_margin": [999.0],
        }
    )
    out = _override_wall_distance_and_recompute(df, 300.0)
    assert out["wall_distance_in_spray_direction"].iloc[0] == 300.0
    assert out["projected_distance_to_wall_margin"].iloc[0] == pytest.approx(340.0 - 300.0)
    assert out["absolute_distance_to_wall"].iloc[0] == pytest.approx(abs(340.0 - 300.0))


def test_override_launch_angle_recomputes_hang_time():
    df = pd.DataFrame(
        {
            "launch_angle": [10.0],
            "launch_speed": [95.0],
            "estimated_hang_time_s": [999.0],  # stale sentinel
        }
    )
    out = _override_launch_angle_and_recompute_hang_time(df, 45.0)
    assert out["launch_angle"].iloc[0] == 45.0
    assert out["estimated_hang_time_s"].iloc[0] != 999.0
    assert out["estimated_hang_time_s"].iloc[0] > 0


def test_override_launch_angle_matched_trajectory_preserves_distance_holds_context_fixed():
    df = pd.DataFrame(
        {
            "launch_angle": [10.0],
            "launch_speed": [95.0],
            "hit_distance_sc": [340.0],
            "estimated_hang_time_s": [999.0],  # stale sentinel
            "wall_distance_in_spray_direction": [350.0],
            "projected_distance_to_wall_margin": [-10.0],
            "absolute_distance_to_wall": [10.0],
            "landing_x_ft": [5.0],
            "landing_y_ft": [339.9],
        }
    )
    out = _override_launch_angle_matched_trajectory(df, 40.0)
    assert out["launch_angle"].iloc[0] == 40.0
    # Exit velocity is SOLVED, not left at the real row's value.
    assert out["launch_speed"].iloc[0] != 95.0
    assert out["estimated_hang_time_s"].iloc[0] != 999.0
    assert out["estimated_hang_time_s"].iloc[0] > 0
    # Landing distance and every wall-geometry feature derived from it are
    # UNCHANGED -- the whole point of a trajectory-matched counterfactual.
    assert out["hit_distance_sc"].iloc[0] == 340.0
    assert out["wall_distance_in_spray_direction"].iloc[0] == 350.0
    assert out["projected_distance_to_wall_margin"].iloc[0] == -10.0
    assert out["absolute_distance_to_wall"].iloc[0] == 10.0
    assert out["landing_x_ft"].iloc[0] == 5.0
    assert out["landing_y_ft"].iloc[0] == 339.9


def test_override_launch_angle_matched_trajectory_higher_angle_has_more_hang_time():
    df = pd.DataFrame({"launch_angle": [15.0], "launch_speed": [95.0], "hit_distance_sc": [360.0]})
    low_df = _override_launch_angle_matched_trajectory(df, 18.0)
    high_df = _override_launch_angle_matched_trajectory(df, 40.0)
    assert high_df["estimated_hang_time_s"].iloc[0] > low_df["estimated_hang_time_s"].iloc[0]


# ---------------------------------------------------------------------------
# Wall height buckets / subgroup calibration
# ---------------------------------------------------------------------------


def test_wall_height_buckets_produces_three_groups():
    heights = pd.Series(np.linspace(6.0, 20.0, 100))
    buckets = compute_wall_height_buckets(heights)
    assert set(buckets.dropna().unique()) == {"short", "medium", "tall"}


def test_compute_near_wall_subgroups_covers_required_dimensions(near_wall_df):
    y_true = near_wall_df["outfield_converted_to_out"].to_numpy()
    p_out = pd.Series(0.5, index=near_wall_df.index)
    subgroups = compute_near_wall_subgroups(y_true, p_out, near_wall_df)
    for expected in (
        "near_wall_10ft",
        "near_wall_20ft",
        "wall_height_short",
        "spray_sector_center",
    ):
        assert expected in subgroups


def test_find_material_subgroup_issues_flags_high_ece():
    subgroups = {"a": {"ece": 0.2, "sample_count": 500}, "b": {"ece": 0.01, "sample_count": 500}}
    flagged = find_material_subgroup_issues(subgroups, absolute_threshold=0.05, min_sample_size=100)
    assert flagged == ["a"]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_model_selection_picks_a_registered_candidate(near_wall_df):
    winner, selection_metrics, trained_by_candidate = run_near_wall_model_selection(near_wall_df)
    assert winner in NEAR_WALL_SELECTION_CANDIDATES
    assert set(selection_metrics.keys()) == set(NEAR_WALL_SELECTION_CANDIDATES)
    assert set(trained_by_candidate.keys()) == set(NEAR_WALL_SELECTION_CANDIDATES)


def test_model_selection_uses_correct_season_split(near_wall_df, monkeypatch: pytest.MonkeyPatch):
    import mlb_luck_score.models.compare_near_wall_models as cnw

    seen_seasons: set[int] = set()
    original = cnw.train_opportunity_model

    def _tracking_train(train_df, **kwargs):
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(cnw, "train_opportunity_model", _tracking_train)
    run_near_wall_model_selection(near_wall_df)
    assert seen_seasons == set(CALIBRATION_BASE_TRAIN_SEASONS)


def test_model_selection_rejects_empty_seasons():
    df = pd.DataFrame({"season": [2024] * 10})
    with pytest.raises(ValueError, match="fit"):
        run_near_wall_model_selection(df)


# ---------------------------------------------------------------------------
# Perturbation checks
# ---------------------------------------------------------------------------


def test_wall_distance_perturbation_detects_correct_direction(near_wall_df):
    fit_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    from mlb_luck_score.features.build_contact_features import (
        NEAR_WALL_CATEGORICAL_FEATURES,
        NEAR_WALL_NUMERIC_FEATURES,
    )

    # Logistic (not HGB) here: HGB's tree splits return a near-constant
    # prediction when extrapolated outside the narrow margin range this
    # small synthetic fixture's near-wall rows span (real-data behavior is
    # verified separately against production data, not re-derived from a
    # minimal unit fixture) -- logistic regression extrapolates linearly,
    # so the override+recompute mechanics are still exercised meaningfully.
    trained = train_opportunity_model(
        fit_df,
        class_weight=None,
        numeric_features=NEAR_WALL_NUMERIC_FEATURES,
        categorical_features=NEAR_WALL_CATEGORICAL_FEATURES,
    )
    results = run_near_wall_perturbation_checks(trained, near_wall_df)
    assert "wall_distance_direction" in results.required
    assert "trajectory_matched_opportunity_time_short_of_wall" in results.required
    assert all(isinstance(r, DirectionalCheckResult) for r in results.required.values())
    # The synthetic fixture deterministically encodes the wall-distance
    # relationship, so the model should learn it correctly.
    assert results.required["wall_distance_direction"].passed is True
    assert results.required["wall_distance_direction"].sample_size == len(near_wall_df)
    # The ORIGINAL (unmatched) launch-angle proxy check is demoted to
    # descriptive -- reported, but no longer required to pass.
    assert "launch_angle_proxy_short_of_wall" in results.descriptive
    assert isinstance(results.descriptive["launch_angle_proxy_short_of_wall"], BandedEffectResult)
    # The trajectory-match invariant (higher matched angle -> strictly more
    # estimated opportunity time) must hold by construction.
    assert BAND_SHORT_OF_WALL in results.trajectory_match_invariants
    invariant = results.trajectory_match_invariants[BAND_SHORT_OF_WALL]
    assert invariant.verified is True
    assert invariant.mean_hang_time_high_angle_s > invariant.mean_hang_time_low_angle_s
    # Real-data grouped opportunity-time response curve is populated.
    assert len(results.opportunity_time_response) > 0
    assert all(isinstance(b, OpportunityTimeResponseBin) for b in results.opportunity_time_response)


def test_grouped_opportunity_time_response_uses_only_real_rows(near_wall_df):
    from mlb_luck_score.features.build_contact_features import (
        NEAR_WALL_CATEGORICAL_FEATURES,
        NEAR_WALL_NUMERIC_FEATURES,
    )

    fit_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    trained = train_opportunity_model(
        fit_df,
        class_weight=None,
        numeric_features=NEAR_WALL_NUMERIC_FEATURES,
        categorical_features=NEAR_WALL_CATEGORICAL_FEATURES,
    )
    bins = compute_grouped_opportunity_time_response(trained, near_wall_df)
    assert len(bins) > 0
    real_distances = set(near_wall_df["hit_distance_sc"].round(6))
    for b in bins:
        assert b.band in ("short_of_wall", "at_wall", "beyond_wall")
        assert b.bb_type == "fly_ball"  # the only bb_type in this synthetic fixture
        assert b.sample_size > 0
        assert 0.0 <= b.mean_predicted_p_out <= 1.0
        # Real-data only -- every bin's mean distance is a genuine average of
        # REAL rows' hit_distance_sc, never a synthetic override.
        assert any(abs(b.mean_hit_distance_ft - d) < 50 for d in real_distances)


# ---------------------------------------------------------------------------
# Final comparison / paired bootstrap
# ---------------------------------------------------------------------------


def test_final_comparison_reports_both_variants(near_wall_df):
    from mlb_luck_score.features.build_contact_features import (
        NEAR_WALL_CATEGORICAL_FEATURES,
        NEAR_WALL_NUMERIC_FEATURES,
        OPPORTUNITY_CATEGORICAL_FEATURES,
        OPPORTUNITY_NUMERIC_FEATURES,
    )

    fit_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)]
    open_field_trained = train_opportunity_model(
        fit_df,
        class_weight=None,
        numeric_features=OPPORTUNITY_NUMERIC_FEATURES,
        categorical_features=OPPORTUNITY_CATEGORICAL_FEATURES,
    )
    winner_trained = train_opportunity_model(
        fit_df,
        class_weight=None,
        numeric_features=NEAR_WALL_NUMERIC_FEATURES,
        categorical_features=NEAR_WALL_CATEGORICAL_FEATURES,
    )
    comparison = run_near_wall_final_comparison(
        near_wall_df, open_field_trained, VARIANT_NEAR_WALL_LOGISTIC, winner_trained
    )
    assert set(comparison.keys()) == {VARIANT_OPEN_FIELD, VARIANT_NEAR_WALL_FINAL}
    assert comparison[VARIANT_NEAR_WALL_FINAL]["sample_count"] == int(
        (near_wall_df["season"].isin(CALIBRATION_EVAL_SEASONS)).sum()
    )


def test_final_comparison_rejects_empty_final_seasons():
    df = pd.DataFrame({"season": [2021] * 10})
    with pytest.raises(ValueError, match="final-comparison"):
        run_near_wall_final_comparison(df, None, VARIANT_NEAR_WALL_LOGISTIC, None)  # type: ignore[arg-type]


def test_paired_bootstrap_deterministic():
    y_true = np.array([0, 1] * 50)
    baseline = pd.Series([0.5] * 100)
    candidate = pd.Series(np.linspace(0.1, 0.9, 100))
    game_pks = pd.Series(list(range(50)) * 2)
    kwargs = dict(
        y_true=y_true,
        baseline_p_out=baseline,
        candidate_p_out=candidate,
        game_pks=game_pks,
        n_reps=15,
        seed=3,
    )
    result_a = compute_near_wall_paired_bootstrap(**kwargs)
    result_b = compute_near_wall_paired_bootstrap(**kwargs)
    assert result_a["log_loss_delta"]["ci_low"] == result_b["log_loss_delta"]["ci_low"]


# ---------------------------------------------------------------------------
# Validation summary: AND-combination, never adopted on log-loss alone
# ---------------------------------------------------------------------------


def _passing_check(label="x") -> DirectionalCheckResult:
    return DirectionalCheckResult(
        label=label,
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


def _failing_check(label="x") -> DirectionalCheckResult:
    check = _passing_check(label)
    return DirectionalCheckResult(**{**check.__dict__, "passed": False})


def _summary(log_loss: float) -> dict:
    return {
        "binary_log_loss": log_loss,
        "expected_calibration_error": 0.01,
        "near_wall_subgroups": {},
        "calibration_by_venue": [],
    }


def _passing_suite() -> NearWallPerturbationSuite:
    return NearWallPerturbationSuite(
        required={
            "wall_distance_direction": _passing_check(),
            "trajectory_matched_opportunity_time_short_of_wall": _passing_check(),
        },
        descriptive={},
        partial_dependence={},
    )


def test_calibrated_true_when_everything_passes():
    comparison = {VARIANT_OPEN_FIELD: _summary(0.6), VARIANT_NEAR_WALL_FINAL: _summary(0.4)}
    bootstrap = {"log_loss_delta": {"ci_high": -0.05}}
    perturbation = _passing_suite()
    regression_check = {"predictions_identical_via_gate": True}
    summary = summarize_near_wall_validation(comparison, bootstrap, perturbation, regression_check)
    assert summary["near_wall_specialist_calibrated"] is True


def test_calibrated_false_when_perturbation_fails_despite_log_loss_win():
    comparison = {VARIANT_OPEN_FIELD: _summary(0.6), VARIANT_NEAR_WALL_FINAL: _summary(0.4)}
    bootstrap = {"log_loss_delta": {"ci_high": -0.05}}
    perturbation = NearWallPerturbationSuite(
        required={
            "wall_distance_direction": _passing_check(),
            "trajectory_matched_opportunity_time_short_of_wall": _failing_check(),
        },
        descriptive={},
        partial_dependence={},
    )
    regression_check = {"predictions_identical_via_gate": True}
    summary = summarize_near_wall_validation(comparison, bootstrap, perturbation, regression_check)
    assert summary["near_wall_specialist_calibrated"] is False
    assert "trajectory_matched_opportunity_time_short_of_wall" in summary["perturbation_failures"]


def test_calibrated_false_when_bootstrap_does_not_support_improvement():
    comparison = {VARIANT_OPEN_FIELD: _summary(0.6), VARIANT_NEAR_WALL_FINAL: _summary(0.599)}
    bootstrap = {"log_loss_delta": {"ci_high": 0.01}}  # crosses zero
    perturbation = _passing_suite()
    regression_check = {"predictions_identical_via_gate": True}
    summary = summarize_near_wall_validation(comparison, bootstrap, perturbation, regression_check)
    assert summary["near_wall_specialist_calibrated"] is False


def test_calibrated_false_when_subgroup_issue_present():
    comparison = {VARIANT_OPEN_FIELD: _summary(0.6), VARIANT_NEAR_WALL_FINAL: _summary(0.4)}
    comparison[VARIANT_NEAR_WALL_FINAL]["near_wall_subgroups"] = {
        "near_wall_5ft": {"ece": 0.3, "sample_count": 500}
    }
    bootstrap = {"log_loss_delta": {"ci_high": -0.05}}
    perturbation = _passing_suite()
    regression_check = {"predictions_identical_via_gate": True}
    summary = summarize_near_wall_validation(comparison, bootstrap, perturbation, regression_check)
    assert summary["near_wall_specialist_calibrated"] is False
    assert "near_wall_5ft" in summary["wall_band_ece_issues"]


def test_reporting_rule_text_matches_task_wording():
    comparison = {VARIANT_OPEN_FIELD: _summary(0.6), VARIANT_NEAR_WALL_FINAL: _summary(0.4)}
    summary = summarize_near_wall_validation(
        comparison,
        {"log_loss_delta": {"ci_high": -0.05}},
        NearWallPerturbationSuite(required={}, descriptive={}, partial_dependence={}),
        {"predictions_identical_via_gate": True},
    )
    assert "provisional or unavailable for wall-adjacent" in summary["reporting_rule"]
