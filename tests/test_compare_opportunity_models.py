"""Tests for the Contact Luck v0.7A outfield-opportunity model evaluation.

No test touches the network -- synthetic venue+geometry-joined fixtures
only. The synthetic data deliberately encodes a real physical relationship
(short hang time/shallow landing -> almost always converted; deep/long hang
time -> rarely converted) so the model has real signal to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import LEAKAGE_COLUMNS
from mlb_luck_score.features.build_contact_features import (
    OPPORTUNITY_CATEGORICAL_FEATURES,
    OPPORTUNITY_NUMERIC_FEATURES,
)
from mlb_luck_score.models.compare_opportunity_models import (
    ALL_V07A_CANDIDATES,
    MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE,
    VARIANT_MEASURED_CONTACT_ONLY,
    compute_binary_calibration_by_venue,
    compute_binary_calibration_table,
    compute_binary_ece,
    compute_binary_subgroup_calibration,
    compute_defender_subgroup_calibration,
    compute_opportunity_bootstrap,
    compute_opportunity_subgroups,
    compute_opportunity_time_buckets,
    find_material_subgroup_issues,
    run_opportunity_model_evaluation,
    summarize_opportunity_validation,
)


def _synthetic_air_ball_rows(season: int, rng: np.random.Generator, *, n: int = 200) -> list[dict]:
    rows = []
    for i in range(n):
        deep = i % 2 == 0
        day = 1 + (i % 27)
        distance = 380.0 + rng.normal(0, 10) if deep else 150.0 + rng.normal(0, 10)
        hang_time_signal = 5.5 if deep else 2.0
        converted = 0 if deep else 1
        outcome_class = "double" if (deep and converted == 0) else "out"
        # ~15% chance of a genuine hit even on a "shallow" opportunity, and
        # vice versa, so the model has some realistic noise, not perfect
        # separability.
        if rng.random() < 0.1:
            converted = 1 - converted
            outcome_class = "out" if converted == 1 else "single"
        position = 8 if abs(rng.normal(0, 1)) < 1.5 else (7 if i % 2 == 0 else 9)
        rows.append(
            {
                "game_pk": season * 1000 + day,
                "season": season,
                "eligible_for_training": True,
                "bb_type": "fly_ball",
                "events": "field_out" if converted == 1 else "double",
                "outcome_class": outcome_class,
                "launch_speed": 95.0 + rng.normal(0, 2),
                "launch_angle": 28.0 + rng.normal(0, 2) + (hang_time_signal - 3.5) * 2,
                "hit_distance_sc": distance,
                "spray_angle_approx": rng.uniform(-40, 40),
                "hit_location": position,
                "of_fielding_alignment": "Standard",
                "if_fielding_alignment": "Standard",
                "fielder_7": 111,
                "fielder_8": 222,
                "fielder_9": 333,
                "venue": "Fenway Park",
                "venue_id": 3.0,
                "has_venue_metadata": True,
            }
        )
    return rows


@pytest.fixture
def opportunity_joined_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        rows += _synthetic_air_ball_rows(season, rng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binary calibration primitives
# ---------------------------------------------------------------------------


def test_binary_calibration_table_perfect_calibration():
    y_true = np.array([0, 0, 1, 1] * 10)
    p_out = pd.Series([0.5] * 40)
    table = compute_binary_calibration_table(y_true, p_out, n_bins=2)
    assert not table.empty
    ece = compute_binary_ece(table)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_binary_ece_empty_table_returns_nan():
    assert np.isnan(compute_binary_ece(pd.DataFrame()))


def test_binary_subgroup_calibration_empty_mask_returns_none_metrics():
    y_true = np.array([0, 1, 0, 1])
    p_out = pd.Series([0.5, 0.5, 0.5, 0.5])
    mask = np.array([False, False, False, False])
    result = compute_binary_subgroup_calibration(y_true, p_out, mask, "empty")
    assert result["sample_count"] == 0
    assert result["ece"] is None


def test_opportunity_time_buckets_produces_four_groups():
    hang_times = pd.Series(np.linspace(1.0, 6.0, 100))
    buckets = compute_opportunity_time_buckets(hang_times)
    assert set(buckets.dropna().unique()) == {"q1_shortest", "q2", "q3", "q4_longest"}


def test_binary_calibration_by_venue_flags_reliable_venues():
    y_true = np.array([0, 1] * 60)
    p_out = pd.Series([0.5] * 120)
    venues = pd.Series(["A"] * 100 + ["B"] * 20)
    table = compute_binary_calibration_by_venue(y_true, p_out, venues, min_reliable_samples=50)
    reliable = table[table["reliable"]]
    assert set(reliable["venue_id"]) == {"A"}


def test_defender_subgroup_calibration_excludes_small_samples_and_handles_missing_ids():
    y_true = np.array([1, 0] * 60)
    p_out = pd.Series([0.5] * 120)
    fielder_ids = pd.Series([111] * 60 + [222] * 55 + [pd.NA] * 5)
    table = compute_defender_subgroup_calibration(y_true, p_out, fielder_ids, min_samples=60)
    assert set(table["responsible_outfielder_id"]) == {111}


# ---------------------------------------------------------------------------
# Feature-set sanity
# ---------------------------------------------------------------------------


def test_opportunity_feature_sets_disjoint_from_leakage():
    assert set(OPPORTUNITY_NUMERIC_FEATURES).isdisjoint(LEAKAGE_COLUMNS)
    assert set(OPPORTUNITY_CATEGORICAL_FEATURES).isdisjoint(LEAKAGE_COLUMNS)


def test_only_one_v07a_candidate_registered():
    # typical_position_proxy_v07 was investigated and not built -- see
    # module docstring. This test documents/locks in that current state so
    # a future change to ALL_V07A_CANDIDATES is a deliberate, visible diff.
    assert ALL_V07A_CANDIDATES == (VARIANT_MEASURED_CONTACT_ONLY,)


# ---------------------------------------------------------------------------
# End-to-end evaluation pipeline
# ---------------------------------------------------------------------------


def test_comparison_rejects_missing_alignment_columns():
    df = pd.DataFrame({"eligible_for_training": [True], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(ValueError, match="if_fielding_alignment"):
        run_opportunity_model_evaluation(df)


def test_run_opportunity_model_evaluation_produces_valid_probabilities(opportunity_joined_df):
    comparison, trained_models, p_out_by_variant = run_opportunity_model_evaluation(
        opportunity_joined_df
    )
    assert set(comparison.keys()) == set(ALL_V07A_CANDIDATES)
    p_out = p_out_by_variant[VARIANT_MEASURED_CONTACT_ONLY]
    assert np.isfinite(p_out.to_numpy()).all()
    assert (p_out >= 0).all() and (p_out <= 1).all()


def test_run_opportunity_model_evaluation_reports_required_subgroups(opportunity_joined_df):
    comparison, _trained, _proba = run_opportunity_model_evaluation(opportunity_joined_df)
    subgroups = comparison[VARIANT_MEASURED_CONTACT_ONLY]["opportunity_subgroups"]
    for expected in (
        "bb_type_fly_ball",
        "left_field",
        "center_field",
        "right_field",
        "opportunity_time_q1_shortest",
    ):
        assert expected in subgroups
    assert (
        comparison[VARIANT_MEASURED_CONTACT_ONLY]["movement_direction_subgroup_note"]
        == MOVEMENT_DIRECTION_SUBGROUP_UNAVAILABLE
    )


def test_run_opportunity_model_evaluation_reports_venue_and_defender_calibration(
    opportunity_joined_df,
):
    comparison, _trained, _proba = run_opportunity_model_evaluation(opportunity_joined_df)
    summary = comparison[VARIANT_MEASURED_CONTACT_ONLY]
    assert "calibration_by_venue" in summary
    assert "calibration_by_defender" in summary


def test_identical_train_validation_rows_across_variants(
    opportunity_joined_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.compare_opportunity_models as com

    seen_row_counts: set[int] = set()
    seen_seasons: set[int] = set()
    original = com.train_opportunity_model

    def _tracking_train(train_df, **kwargs):
        seen_row_counts.add(len(train_df))
        seen_seasons.update(train_df["season"].unique().tolist())
        return original(train_df, **kwargs)

    monkeypatch.setattr(com, "train_opportunity_model", _tracking_train)
    run_opportunity_model_evaluation(opportunity_joined_df)

    assert len(seen_row_counts) == 1
    assert seen_seasons == {2021, 2022, 2023}


def test_opportunity_bootstrap_returns_log_loss_and_ece_ci(opportunity_joined_df):
    comparison, _trained, p_out_by_variant = run_opportunity_model_evaluation(opportunity_joined_df)
    training_eligible = opportunity_joined_df  # already all eligible in this fixture
    from mlb_luck_score.models.compare_opportunity_models import _prepare_opportunity_columns

    df = _prepare_opportunity_columns(training_eligible)
    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]
    val_df = eligible[eligible["season"] == 2024]
    y_true = val_df["converted_to_out"].astype(int).to_numpy()

    result = compute_opportunity_bootstrap(
        y_true, p_out_by_variant[VARIANT_MEASURED_CONTACT_ONLY], val_df["game_pk"], n_reps=20
    )
    assert "log_loss" in result and "ece" in result
    assert (
        result["log_loss"]["ci_low"]
        <= result["log_loss"]["point_estimate"]
        <= result["log_loss"]["ci_high"]
    )


def test_opportunity_bootstrap_deterministic(opportunity_joined_df):
    _comparison, _trained, p_out_by_variant = run_opportunity_model_evaluation(
        opportunity_joined_df
    )
    from mlb_luck_score.models.compare_opportunity_models import _prepare_opportunity_columns

    df = _prepare_opportunity_columns(opportunity_joined_df)
    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]
    val_df = eligible[eligible["season"] == 2024]
    y_true = val_df["converted_to_out"].astype(int).to_numpy()

    kwargs = dict(
        y_true=y_true,
        p_out=p_out_by_variant[VARIANT_MEASURED_CONTACT_ONLY],
        game_pks=val_df["game_pk"],
        n_reps=15,
        seed=7,
    )
    result_a = compute_opportunity_bootstrap(**kwargs)
    result_b = compute_opportunity_bootstrap(**kwargs)
    assert result_a["log_loss"]["ci_low"] == result_b["log_loss"]["ci_low"]


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------


def test_find_material_subgroup_issues_flags_high_ece():
    subgroups = {
        "a": {"ece": 0.1, "sample_count": 500},
        "b": {"ece": 0.01, "sample_count": 500},
        "c": {"ece": 0.2, "sample_count": 10},  # too few samples, not flagged
    }
    flagged = find_material_subgroup_issues(subgroups, absolute_threshold=0.05, min_sample_size=100)
    assert flagged == ["a"]


def test_summarize_opportunity_validation_documents_option_b_absence():
    comparison = {
        VARIANT_MEASURED_CONTACT_ONLY: {
            "expected_calibration_error": 0.01,
            "opportunity_subgroups": {},
            "calibration_by_venue": [],
        }
    }
    bootstrap = {VARIANT_MEASURED_CONTACT_ONLY: {}}
    summary = summarize_opportunity_validation(comparison, bootstrap)
    assert summary["typical_position_proxy_v07_built"] is False
    assert (
        summary["per_candidate"][VARIANT_MEASURED_CONTACT_ONLY]["passes_basic_validation"] is True
    )


def test_summarize_opportunity_validation_fails_on_subgroup_issue():
    comparison = {
        VARIANT_MEASURED_CONTACT_ONLY: {
            "expected_calibration_error": 0.01,
            "opportunity_subgroups": {"near_wall_5ft": {"ece": 0.3, "sample_count": 500}},
            "calibration_by_venue": [],
        }
    }
    bootstrap = {VARIANT_MEASURED_CONTACT_ONLY: {}}
    summary = summarize_opportunity_validation(comparison, bootstrap)
    result = summary["per_candidate"][VARIANT_MEASURED_CONTACT_ONLY]
    assert result["passes_basic_validation"] is False
    assert "near_wall_5ft" in result["subgroup_ece_issues"]


def test_compute_opportunity_subgroups_never_references_movement_data(opportunity_joined_df):
    # Sanity check the documented gap: no forward/lateral/backward movement
    # subgroup keys are ever produced.
    from mlb_luck_score.models.compare_opportunity_models import _prepare_opportunity_columns

    df = _prepare_opportunity_columns(opportunity_joined_df)
    eligible = df[df["outfield_opportunity_eligible"].astype(bool)]
    y_true = eligible["converted_to_out"].astype(int).to_numpy()
    p_out = pd.Series(0.5, index=eligible.index)
    subgroups = compute_opportunity_subgroups(y_true, p_out, eligible)
    assert not any("movement" in key for key in subgroups)
    assert not any("forward" in key or "lateral" in key or "backward" in key for key in subgroups)
