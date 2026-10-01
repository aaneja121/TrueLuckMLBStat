"""Tests for the v1.2 `gated_geometry_v12` candidate and its frozen evaluation.

Synthetic data only: batted balls placed at real venues/dates so the real v1.2
geometry join runs, with outcomes generated from a known rule (a ball that clears
the wall is usually a home run). No real Statcast data, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models import evaluate_gated_geometry_v12 as ev

_VENUES = (3, 17, 2681, 4705, 5325, 2889)  # Fenway, Wrigley, CBP, Truist, Globe Life, Busch


def _synthetic(
    n_games_per_season: int = 40, rows_per_game: int = 30, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_pk = 0
    for season in (2021, 2022, 2023, 2024):
        for g in range(n_games_per_season):
            game_pk += 1
            venue = _VENUES[g % len(_VENUES)]
            for _ in range(rows_per_game):
                bb = rng.choice(
                    ["fly_ball", "line_drive", "ground_ball", "popup"], p=[0.4, 0.25, 0.3, 0.05]
                )
                la = {"fly_ball": rng.uniform(25, 45), "line_drive": rng.uniform(10, 25)}.get(
                    bb, rng.uniform(-20, 10) if bb == "ground_ball" else rng.uniform(50, 70)
                )
                ev_mph = rng.uniform(70, 112)
                spray = rng.uniform(-44, 44)
                dist = max(
                    5.0,
                    ev_mph * 4.2 * np.sin(np.radians(2 * max(la, 1))) ** 0.5
                    - 30
                    + rng.normal(0, 15),
                )
                rows.append(
                    {
                        "event_id": f"e{len(rows)}",
                        "season": season,
                        "game_pk": game_pk,
                        "game_date": f"{season}-06-15",
                        "venue_id": float(venue),
                        "has_venue_metadata": True,
                        "launch_speed": ev_mph,
                        "launch_angle": la,
                        "spray_angle_approx": spray,
                        "hit_distance_sc": dist,
                        "bb_type": bb,
                        "stand": rng.choice(["L", "R"]),
                        "eligible_for_training": True,
                    }
                )
    df = pd.DataFrame(rows)
    df = ev.join_v12_geometry(df)
    # Outcome rule: beyond the wall and over its height -> mostly HR; deep short -> mix.
    margin = df["projected_distance_to_wall_margin"].fillna(-999)
    height = df["wall_height_in_spray_direction"].fillna(8.0)
    p_hr = np.clip(1 / (1 + np.exp(-(margin - 0.4 * height) / 4)), 0, 1)
    u = rng.uniform(size=len(df))
    out = np.where(u < p_hr, "home_run", "out")
    deep = (margin > -30) & (out == "out")
    out = np.where(deep & (rng.uniform(size=len(df)) < 0.3), "double", out)
    short = margin <= -30
    r = rng.uniform(size=len(df))
    out = np.where(short & (r < 0.25), "single", out)
    out = np.where(short & (r > 0.97), "triple", out)
    df["outcome_class"] = out
    return df


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return _synthetic()


# --- gate --------------------------------------------------------------------


def test_gate_includes_only_air_balls_with_geometry_at_or_beyond_minus_20() -> None:
    df = pd.DataFrame(
        {
            "geometry_status": ["ok", "ok", "ok", "ok", "outside_modeled_angular_range", "ok"],
            "bb_type": ["fly_ball", "line_drive", "fly_ball", "ground_ball", "fly_ball", "popup"],
            "projected_distance_to_wall_margin": [-20.0, 15.0, -20.01, 5.0, 5.0, 5.0],
        }
    )
    assert ev.gate_mask(df).tolist() == [True, True, False, False, False, False]


def test_gate_treats_missing_margin_as_outside() -> None:
    df = pd.DataFrame(
        {
            "geometry_status": ["ok"],
            "bb_type": ["fly_ball"],
            "projected_distance_to_wall_margin": [np.nan],
        }
    )
    assert ev.gate_mask(df).tolist() == [False]


# --- derived-feature recomputation ---------------------------------------------


def test_recompute_derived_keeps_every_geometry_feature_consistent(synth: pd.DataFrame) -> None:
    df = synth[ev.gate_mask(synth)].head(50).copy()
    df["wall_distance_in_spray_direction"] += 10.0
    df["wall_height_in_spray_direction"] = 16.0
    df["hit_distance_sc"] -= 3.0
    out = ev.recompute_geometry_derived(df)
    margin = out["hit_distance_sc"] - out["wall_distance_in_spray_direction"]
    np.testing.assert_allclose(out["projected_distance_to_wall_margin"], margin)
    np.testing.assert_allclose(out["absolute_distance_to_wall"], margin.abs())
    np.testing.assert_allclose(out["wall_margin_x_launch_angle"], margin * out["launch_angle"])
    np.testing.assert_allclose(out["launch_angle_x_wall_height"], out["launch_angle"] * 16.0)


# --- folds -----------------------------------------------------------------------


def test_folds_are_leave_one_season_out_within_2021_2023_plus_2024_confirmation() -> None:
    folds = ev.evaluation_folds()
    assert folds["primary"] == [
        ((2022, 2023), 2021),
        ((2021, 2023), 2022),
        ((2021, 2022), 2023),
    ]
    assert folds["confirmation"] == [((2021, 2022, 2023), 2024)]
    for train, held_out in folds["primary"] + folds["confirmation"]:
        assert held_out not in train


@pytest.mark.parametrize("season", [2025, 2026])
def test_non_development_seasons_are_refused(synth: pd.DataFrame, season: int) -> None:
    bad = synth.head(5).assign(season=season)
    with pytest.raises(ValueError):
        ev.prepare_rows(pd.concat([synth, bad], ignore_index=True))


# --- fitting and prediction ---------------------------------------------------------


def test_specialist_must_keep_all_five_geometry_features(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    train = rows[rows["season"].isin((2021, 2022))].copy()
    train["wall_height_in_spray_direction"] = np.nan  # would be silently dropped by the trainer
    train = ev.recompute_geometry_derived(train)
    with pytest.raises(ValueError, match="wall_height_in_spray_direction"):
        ev.fit_fold(train)


def test_open_field_rows_are_bit_identical_to_the_baseline(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    models = ev.fit_fold(rows[rows["season"].isin((2021, 2022))])
    held = rows[rows["season"] == 2023]
    pred = ev.predict_fold(models, held)
    gate = pred["gate"].to_numpy()
    assert gate.any() and (~gate).any()
    base = pred["baseline"].to_numpy()
    cand = pred["candidate"].to_numpy()
    assert np.array_equal(base[~gate], cand[~gate])
    assert not np.allclose(base[gate], cand[gate])
    np.testing.assert_allclose(cand.sum(axis=1), 1.0, atol=1e-9)


def test_run_is_deterministic(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    a = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    b = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_pooled_predictions_cover_each_held_out_row_once(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    res = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    held = rows[rows["season"].isin((2021, 2022, 2023))]
    assert sorted(res.predictions["event_id"]) == sorted(held["event_id"])


# --- perturbations ---------------------------------------------------------------------


def test_perturbations_recover_the_synthetic_physics(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    res = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    checks = ev.summarize_perturbations(res.perturbations)
    assert checks["farther_wall"]["passed"]
    assert checks["longer_ball"]["passed"]
    assert checks["farther_wall"]["delta_p_hr"] < 0
    assert checks["longer_ball"]["delta_p_hr"] > 0
    assert checks["taller_wall"]["n_rows"] > 0


# --- criteria -------------------------------------------------------------------------------


def test_material_ece_rule_matches_v03() -> None:
    assert not ev.material_ece_regression(0.020, 0.025)  # +0.005, +25%
    assert ev.material_ece_regression(0.020, 0.031)  # +0.011 absolute
    assert ev.material_ece_regression(0.004, 0.0065)  # +62.5% relative
    assert not ev.material_ece_regression(0.020, 0.010)


def test_wall_height_bins_are_fixed_and_cover_unknown() -> None:
    s = pd.Series([4.0, 8.0, 8.5, 14.9, 15.0, 37.0, np.nan])
    assert ev.wall_height_bin(s).tolist() == [
        "short",
        "short",
        "medium",
        "medium",
        "tall",
        "tall",
        "unknown",
    ]


def _passing_summary() -> dict:
    return {
        "log_loss": {"ci_high": -0.001},
        "calibration": {
            "overall_material_regression": False,
            "home_run_material_regression": False,
        },
        "groups": [
            {"group": "venue_3", "adequate_support": True, "paired_log_loss_ci_low": -0.01},
            {"group": "venue_9", "adequate_support": False, "paired_log_loss_ci_low": 0.5},
        ],
        "perturbations": {
            "farther_wall": {"passed": True},
            "taller_wall": {"passed": True},
            "longer_ball": {"passed": True},
        },
        "open_field_identical": True,
        "confirmation_2024": {"credible_regression": False},
    }


def test_recommendation_passes_only_when_every_criterion_holds() -> None:
    rec = ev.recommend_v12_adoption(_passing_summary())
    assert rec["recommend_adopt"] is True
    assert rec["requires_maintainer_decision"] is True


@pytest.mark.parametrize(
    ("path", "value", "criterion"),
    [
        (("log_loss", "ci_high"), 0.0, "log_loss"),
        (("calibration", "home_run_material_regression"), True, "calibration"),
        (("perturbations", "taller_wall", "passed"), False, "perturbations"),
        (("open_field_identical",), False, "open_field_identical"),
        (("confirmation_2024", "credible_regression"), True, "confirmation_2024"),
    ],
)
def test_recommendation_fails_on_any_single_criterion(
    path: tuple, value: object, criterion: str
) -> None:
    s = _passing_summary()
    target = s
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    rec = ev.recommend_v12_adoption(s)
    assert rec["recommend_adopt"] is False
    assert rec["criteria"][criterion] is False


def test_a_credible_group_regression_fails_but_insufficient_evidence_does_not() -> None:
    s = _passing_summary()
    s["groups"][0]["paired_log_loss_ci_low"] = 0.001
    assert ev.recommend_v12_adoption(s)["criteria"]["no_group_regression"] is False
    s = _passing_summary()  # venue_9 has a positive CI but inadequate support
    assert ev.recommend_v12_adoption(s)["criteria"]["no_group_regression"] is True


def test_full_evaluation_runs_end_to_end_on_synthetic_data(synth: pd.DataFrame) -> None:
    rows = ev.prepare_rows(synth)
    folds = ev.evaluation_folds()
    primary = ev.run_folds(rows, folds["primary"])
    confirmation = ev.run_folds(rows, folds["confirmation"])
    rng = np.random.default_rng(1)
    weather = pd.DataFrame(
        {
            "event_id": rows["event_id"],
            "roof_status": "outdoor_open_air",
            "air_density_kg_m3": rng.normal(1.18, 0.03, len(rows)),
            "following_wind_mps": rng.normal(0, 3, len(rows)),
        }
    )
    summary = ev.evaluate(primary, confirmation, weather, n_reps=20, seed=1)
    import json

    json.dumps(summary, default=float)  # serializable as written by main()
    assert set(summary["recommendation"]["criteria"]) == {
        "log_loss",
        "calibration",
        "no_group_regression",
        "perturbations",
        "open_field_identical",
        "confirmation_2024",
    }
    assert summary["n_gated"] > 0
    assert "air_density_kg_m3" in summary["reported_only"]["weather_residual"]
    assert any(g["group"].startswith("venue_") for g in summary["groups"])
