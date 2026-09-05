"""Contact Forecast R1: pre-cutoff features and the target contract.

The load-bearing tests are the two mutation tests -- rewriting events at or
beyond the cutoff must not move a single feature -- and the independent
target re-derivation, which checks `build_windows`' cumulative-sum
arithmetic against a slow, obvious second code path.

Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import features as ft
from forecast import targets as tg
from forecast import windows as w
from forecast.forecast_config import (
    BASELINE_LADDER,
    PRIMARY_COMPARISON,
    PRIMARY_HORIZON,
    PRIMARY_METRICS,
    PRIMARY_TARGET,
)

_BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")


def make_ledger(
    *,
    batter: int = 501,
    season: int = 2023,
    n_events: int = 300,
    seed: int = 5,
    stand: str = "R",
) -> pd.DataFrame:
    """A synthetic scored ledger carrying every column features need."""
    rng = np.random.default_rng(seed)
    index = np.arange(1, n_events + 1)
    game_offset = (index - 1) // 4
    observed = rng.normal(0.05, 0.5, n_events)
    deserved = observed * 0.4 + rng.normal(0.02, 0.2, n_events)
    return pd.DataFrame(
        {
            "event_id": [f"{batter}-{season}-{i}" for i in index],
            "batter": batter,
            "season": season,
            "game_date": (
                pd.Timestamp(f"{season}-04-01") + pd.to_timedelta(game_offset, unit="D")
            ).strftime("%Y-%m-%d"),
            "game_pk": 900000 + game_offset,
            "at_bat_number": ((index - 1) % 4) + 1,
            "pitch_number": 1,
            "stand": stand,
            "launch_speed": rng.normal(89.0, 11.0, n_events),
            "launch_angle": rng.normal(13.0, 21.0, n_events),
            "spray_angle_approx": rng.normal(0.0, 25.0, n_events),
            "bb_type": rng.choice(_BB_TYPES, n_events),
            "outcome_class": "out",
            "observed_contact_result_run_value": observed,
            "baseline_expected_contact_run_value": deserved,
            "contact_result_surprise": observed - deserved,
        }
    )


@pytest.fixture
def ordered() -> pd.DataFrame:
    return w.order_eligible_events(make_ledger())


# --------------------------------------------------------------------------
# Features: the pre-cutoff rule
# --------------------------------------------------------------------------


def test_mutating_events_at_or_after_the_cutoff_cannot_move_a_feature() -> None:
    """The strong pre-cutoff check: rewrite the unseeable rows, expect no change."""
    baseline = ft.build_window_features(w.order_eligible_events(make_ledger()), 100)

    tampered = make_ledger()
    at_or_after = tampered.index >= 100  # 0-based: events 101..300
    for column in ("launch_speed", "launch_angle", "spray_angle_approx"):
        tampered.loc[at_or_after, column] = 999.0
    tampered.loc[at_or_after, "bb_type"] = "popup"
    tampered.loc[at_or_after, "observed_contact_result_run_value"] = 5.0
    tampered.loc[at_or_after, "baseline_expected_contact_run_value"] = -5.0
    tampered.loc[at_or_after, "contact_result_surprise"] = 10.0
    mutated = ft.build_window_features(w.order_eligible_events(tampered), 100)

    pd.testing.assert_frame_equal(baseline, mutated)


def test_features_change_when_pre_cutoff_events_change() -> None:
    """Companion check: the mutation test above is not vacuously true."""
    baseline = ft.build_window_features(w.order_eligible_events(make_ledger()), 100)

    tampered = make_ledger()
    tampered.loc[tampered.index < 100, "launch_speed"] = 105.0
    mutated = ft.build_window_features(w.order_eligible_events(tampered), 100)

    assert baseline["launch_speed_mean"].iloc[0] != mutated["launch_speed_mean"].iloc[0]
    assert baseline["hard_hit_rate"].iloc[0] != mutated["hard_hit_rate"].iloc[0]


def test_recent_form_window_stays_inside_the_history_window(ordered: pd.DataFrame) -> None:
    """At the smallest cutoff the 25-BBE recent window is still fully pre-cutoff."""
    assert min(50, 100) > ft.RECENT_WINDOW_BBE
    features = ft.build_window_features(ordered, 50)
    assert features["recent_realized_rv_per_100"].notna().all()


def test_season_to_date_rate_matches_a_hand_computation(ordered: pd.DataFrame) -> None:
    features = ft.build_window_features(ordered, 100)
    history = ordered[ordered["bbe_index"] <= 100]

    expected = 100.0 * history["observed_contact_result_run_value"].mean()

    assert features["std_realized_rv_per_100"].iloc[0] == pytest.approx(expected)


def test_surprise_feature_equals_realized_minus_deserved(ordered: pd.DataFrame) -> None:
    features = ft.build_window_features(ordered, 150).iloc[0]
    assert features["std_surprise_rv_per_100"] == pytest.approx(
        features["std_realized_rv_per_100"] - features["std_deserved_rv_per_100"]
    )


def test_hitter_who_never_reached_the_cutoff_gets_no_features() -> None:
    """Averaging 40 batted balls would silently invent a 50-BBE observation."""
    ordered = w.order_eligible_events(make_ledger(n_events=40))
    assert ft.build_window_features(ordered, 50).empty


def test_handedness_is_encoded_for_both_hands() -> None:
    right = ft.build_window_features(w.order_eligible_events(make_ledger(stand="R")), 100)
    left = ft.build_window_features(
        w.order_eligible_events(make_ledger(stand="L", batter=502)), 100
    )
    assert right["bats_left"].iloc[0] == 0.0
    assert left["bats_left"].iloc[0] == 1.0


def test_batted_ball_type_rates_sum_to_one(ordered: pd.DataFrame) -> None:
    features = ft.build_window_features(ordered, 200).iloc[0]
    total = sum(features[f"bb_rate_{kind}"] for kind in ft.BB_TYPES)
    assert total == pytest.approx(1.0)


def test_feature_columns_excludes_identifiers_and_cutoff(ordered: pd.DataFrame) -> None:
    """`cutoff` is constant within a cell and would proxy playing time if pooled."""
    features = ft.build_window_features(ordered, 100)
    columns = ft.feature_columns(features)
    for excluded in ("batter", "season", "cutoff"):
        assert excluded not in columns
    assert "std_deserved_rv_per_100" in columns


# --------------------------------------------------------------------------
# Prior-season features and their asymmetric availability
# --------------------------------------------------------------------------


def test_prior_season_deserved_is_missing_for_an_unscored_prior_season() -> None:
    """2021 has no causal contact model, so its deserved rate must stay NaN."""
    realized_only = make_ledger(season=2021, batter=501)
    table = ft.build_prior_season_table(
        scored_ledgers={}, realized_only_ledgers={2021: realized_only}
    )

    row = table.iloc[0]
    assert np.isfinite(row["prior_realized_rv_per_100"])
    assert np.isnan(row["prior_deserved_rv_per_100"])
    assert row["has_prior_deserved"] == 0.0


def test_prior_season_join_only_ever_sees_the_immediately_prior_season() -> None:
    prior = ft.build_prior_season_table(scored_ledgers={2022: make_ledger(season=2022)})
    features = ft.build_window_features(w.order_eligible_events(make_ledger(season=2023)), 100)

    joined = ft.attach_prior_season_features(features, prior)

    assert (joined["prior_source_season"] == 2022).all()
    assert (joined["prior_source_season"] < joined["season"]).all()
    assert joined["has_prior_season"].iloc[0] == 1.0
    assert joined["has_prior_deserved"].iloc[0] == 1.0


def test_hitter_with_no_prior_season_is_flagged_not_imputed() -> None:
    """A rookie gets NaN and a flag -- never a filled value that looks like evidence."""
    prior = ft.build_prior_season_table(scored_ledgers={2022: make_ledger(season=2022, batter=999)})
    features = ft.build_window_features(w.order_eligible_events(make_ledger(season=2023)), 100)

    joined = ft.attach_prior_season_features(features, prior)

    assert joined["has_prior_season"].iloc[0] == 0.0
    assert np.isnan(joined["prior_realized_rv_per_100"].iloc[0])


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def test_targets_match_an_independent_recomputation(ordered: pd.DataFrame) -> None:
    """Cross-check the cumulative-sum arithmetic against direct slicing."""
    windows = w.build_windows(ordered)
    tg.assert_targets_match_independent_recomputation(ordered, windows)


def test_independent_recomputation_catches_a_corrupted_target(
    ordered: pd.DataFrame,
) -> None:
    windows = w.build_windows(ordered)
    windows.loc[0, "target_realized_rv_per_100"] += 1.0

    with pytest.raises(tg.ForecastTargetError, match="independent re-derivation"):
        tg.assert_targets_match_independent_recomputation(ordered, windows)


def test_full_telescoping_columns_are_refused() -> None:
    """R1 is contact-stage end to end; an Rf column must never enter."""
    with pytest.raises(tg.ForecastTargetError, match="contact-stage quantities"):
        tg.assert_stage_consistency(["observed_contact_result_run_value", "contact_luck_runs"])


def test_contact_stage_columns_are_accepted() -> None:
    tg.assert_stage_consistency(
        [
            "observed_contact_result_run_value",
            "baseline_expected_contact_run_value",
            "contact_result_surprise",
        ]
    )


@pytest.mark.parametrize(
    "text",
    [
        "this proves Contact Luck predicts future performance",
        "R1 validates the published Contact Luck metric",
        "the official Contact Luck number improves forecasts",
    ],
)
def test_banned_metric_language_is_refused(text: str) -> None:
    with pytest.raises(tg.ForecastTargetError, match="banned phrase"):
        tg.assert_no_banned_metric_language(text)


def test_honest_contact_stage_language_is_accepted() -> None:
    tg.assert_no_banned_metric_language(
        "The luck-adjusted contact signal (Rc - E0) predicts future realized "
        "contact-result run value better than realized production does."
    )


def test_target_description_carries_the_conditional_language() -> None:
    described = tg.describe_target(PRIMARY_TARGET, 100)
    assert "CONDITIONAL ON" in described
    assert "does not predict continued playing time" in described


def test_primary_cell_designation_is_frozen_to_the_primary_horizon() -> None:
    assert tg.is_primary_cell(100, PRIMARY_HORIZON, PRIMARY_TARGET) is True
    assert tg.is_primary_cell(100, 50, PRIMARY_TARGET) is False
    assert tg.is_primary_cell(100, PRIMARY_HORIZON, "target_deserved_rv_per_100") is False


def test_primary_specification_is_the_one_that_was_frozen() -> None:
    """Pins the frozen choices so a later edit cannot quietly relocate them."""
    assert PRIMARY_TARGET == "target_realized_rv_per_100"
    assert PRIMARY_HORIZON == 100
    assert PRIMARY_COMPARISON == (
        "shrunk_realized_persistence",
        "shrunk_deserved_persistence",
    )
    assert set(PRIMARY_METRICS) == {"mae", "rmse", "spearman"}
    assert BASELINE_LADDER == (
        "league_mean",
        "raw_realized_persistence",
        "raw_deserved_persistence",
        "shrunk_realized_persistence",
        "shrunk_deserved_persistence",
    )


# --------------------------------------------------------------------------
# Repeated-window structure
# --------------------------------------------------------------------------


def test_repeated_window_structure_separates_windows_from_batters() -> None:
    """6,773 windows are not 6,773 independent observations."""
    ledger = pd.concat(
        [make_ledger(batter=b, season=s, seed=b + s) for b in (1, 2, 3) for s in (2023, 2024)],
        ignore_index=True,
    )
    ledger["event_id"] = [f"e{i}" for i in range(len(ledger))]
    windows = w.build_windows(w.order_eligible_events(ledger))

    summary = tg.summarize_repeated_window_structure(windows)

    assert summary["n_windows"] > summary["n_unique_batter_seasons"]
    assert summary["n_unique_batter_seasons"] > summary["n_unique_batters"]
    assert summary["n_unique_batters"] == 3
    assert summary["windows_per_batter_max"] >= summary["windows_per_batter_mean"]


def test_repeated_window_structure_handles_an_empty_table() -> None:
    summary = tg.summarize_repeated_window_structure(pd.DataFrame(columns=["batter", "season"]))
    assert summary["n_windows"] == 0
