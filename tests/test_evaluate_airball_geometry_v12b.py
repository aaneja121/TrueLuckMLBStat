"""Tests for the v1.2b outcome-free air-ball candidate (`airball_geometry_v12b`).

Synthetic data only: air balls at real venues, with outcomes generated from a known
carry rule (carry from launch speed/angle vs the real v1.2 wall), so geometry should
help and the physical directions are known. No real data, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models import evaluate_airball_geometry_v12b as ev

_VENUES = (3.0, 17.0, 2681.0, 4705.0, 5325.0, 2889.0)


def _synthetic(
    n_games_per_season: int = 36, rows_per_game: int = 30, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_pk = 0
    for season in (2021, 2022, 2023, 2024):
        for g in range(n_games_per_season):
            game_pk += 1
            for _ in range(rows_per_game):
                bb = rng.choice(
                    ["fly_ball", "line_drive", "ground_ball", "popup"], p=[0.45, 0.25, 0.25, 0.05]
                )
                la = {"fly_ball": rng.uniform(22, 45), "line_drive": rng.uniform(10, 22)}.get(
                    bb, rng.uniform(-20, 8) if bb == "ground_ball" else rng.uniform(50, 70)
                )
                rows.append(
                    {
                        "event_id": f"e{len(rows)}",
                        "season": season,
                        "game_pk": game_pk,
                        "game_date": f"{season}-06-15",
                        "venue_id": _VENUES[g % len(_VENUES)],
                        "has_venue_metadata": True,
                        "launch_speed": rng.uniform(75, 112),
                        "launch_angle": la,
                        "spray_angle_approx": rng.uniform(-44, 44),
                        "bb_type": bb,
                        "stand": rng.choice(["L", "R"]),
                        "eligible_for_training": True,
                    }
                )
    df = pd.DataFrame(rows)
    carry = (
        4.1
        * df["launch_speed"]
        * np.sin(np.radians(np.clip(2 * df["launch_angle"], 1, 179))) ** 0.6
        - 25
    )
    df["hit_distance_sc"] = carry + rng.normal(0, 10, len(df))
    df = ev.join_v12_geometry(df)
    wall = df["wall_distance_in_spray_direction"].fillna(400.0)
    height = df["wall_height_in_spray_direction"].fillna(8.0)
    p_hr = 1 / (1 + np.exp(-(carry - wall - 0.8 * height) / 6))
    u = rng.uniform(size=len(df))
    out = np.where(u < p_hr, "home_run", "out")
    r = rng.uniform(size=len(df))
    out = np.where((out == "out") & (r < 0.15), "single", out)
    out = np.where((out == "out") & (r > 0.92), "double", out)
    out = np.where((out == "out") & (r > 0.915) & (r <= 0.92), "triple", out)
    df["outcome_class"] = out
    return df


@pytest.fixture(scope="module")
def synth() -> pd.DataFrame:
    return _synthetic()


@pytest.fixture(scope="module")
def rows(synth: pd.DataFrame) -> pd.DataFrame:
    return ev.prepare_rows(synth)


# --- scope and inputs -----------------------------------------------------------------


def test_scope_is_air_balls_with_geometry() -> None:
    df = pd.DataFrame(
        {
            "geometry_status": ["ok", "ok", "ok", "ok", "outside_modeled_angular_range"],
            "bb_type": ["fly_ball", "line_drive", "ground_ball", "popup", "fly_ball"],
        }
    )
    assert ev.scope_mask(df).tolist() == [True, True, False, False, False]


def test_inputs_contain_no_fielding_distance() -> None:
    forbidden = {
        "hit_distance_sc",
        "projected_distance_to_wall_margin",
        "absolute_distance_to_wall",
        "wall_margin_x_launch_angle",
        "near_wall_5ft",
        "near_wall_10ft",
        "near_wall_20ft",
        "projected_beyond_wall",
    }
    assert not forbidden & set(ev.CANDIDATE_FEATURES)
    assert not forbidden & set(ev.REFERENCE_FEATURES)
    assert set(ev.CANDIDATE_FEATURES) - set(ev.REFERENCE_FEATURES) == set(ev.WALL_FEATURES)


def test_in_scope_predictions_do_not_depend_on_hit_distance(rows: pd.DataFrame) -> None:
    models = ev.fit_fold(rows[rows["season"].isin((2021, 2022))])
    held = rows[rows["season"] == 2023]
    a = ev.predict_fold(models, held)
    b = ev.predict_fold(models, held.assign(hit_distance_sc=held["hit_distance_sc"] + 50.0))
    scope = a["scope"].to_numpy()
    np.testing.assert_array_equal(
        a["candidate"].to_numpy()[scope], b["candidate"].to_numpy()[scope]
    )
    np.testing.assert_array_equal(
        a["reference"].to_numpy()[scope], b["reference"].to_numpy()[scope]
    )


def test_unknown_wall_height_reaches_the_model_as_missing() -> None:
    pipe = ev.build_airball_model(ev.CANDIDATE_FEATURES)
    x = pd.DataFrame(
        {
            "launch_speed": [100.0, 95.0],
            "launch_angle": [28.0, 30.0],
            "spray_angle_approx": [0.0, 10.0],
            "wall_distance_in_spray_direction": [400.0, 380.0],
            "wall_height_in_spray_direction": [np.nan, 8.0],
            "stand": ["L", "R"],
            "bb_type": ["fly_ball", "line_drive"],
        }
    )
    transformed = pipe.named_steps["preprocess"].fit_transform(x)
    assert np.isnan(np.asarray(transformed, dtype=float)).any()


# --- fitting and prediction --------------------------------------------------------------


def test_out_of_scope_rows_are_bit_identical_to_baseline(rows: pd.DataFrame) -> None:
    models = ev.fit_fold(rows[rows["season"].isin((2021, 2022))])
    pred = ev.predict_fold(models, rows[rows["season"] == 2023])
    scope = pred["scope"].to_numpy()
    assert scope.any() and (~scope).any()
    base, cand = pred["baseline"].to_numpy(), pred["candidate"].to_numpy()
    assert np.array_equal(base[~scope], cand[~scope])
    np.testing.assert_allclose(cand.sum(axis=1), 1.0, atol=1e-9)


def test_folds_never_train_on_the_held_out_season(rows: pd.DataFrame) -> None:
    res = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    held = rows[rows["season"].isin((2021, 2022, 2023))]
    assert sorted(res.predictions["event_id"]) == sorted(held["event_id"])


def test_run_is_deterministic(rows: pd.DataFrame) -> None:
    a = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    b = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


@pytest.mark.parametrize("season", [2025, 2026])
def test_non_development_seasons_are_refused(synth: pd.DataFrame, season: int) -> None:
    with pytest.raises(ValueError):
        ev.prepare_rows(pd.concat([synth, synth.head(3).assign(season=season)], ignore_index=True))


# --- perturbations ----------------------------------------------------------------------------


def test_perturbations_recover_the_synthetic_physics(rows: pd.DataFrame) -> None:
    res = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    checks = ev.summarize_perturbations(res.perturbations)
    assert set(checks) == {"farther_wall", "taller_wall", "harder_hit"}
    assert checks["farther_wall"]["delta_p_hr"] < 0 and checks["farther_wall"]["passed"]
    assert checks["harder_hit"]["delta_p_hr"] > 0 and checks["harder_hit"]["passed"]
    assert checks["taller_wall"]["n_rows"] > 0


# --- criteria ----------------------------------------------------------------------------------


def _passing() -> dict:
    return {
        "geometry_vs_reference": {"log_loss": {"ci_high": -0.001}},
        "calibration": {
            "overall_material_regression": False,
            "home_run_material_regression": False,
        },
        "groups": [
            {"group": "venue_3", "adequate_support": True, "paired_log_loss_ci_low": -0.01},
            {"group": "venue_9", "adequate_support": False, "paired_log_loss_ci_low": 0.3},
        ],
        "perturbations": {
            k: {"passed": True} for k in ("farther_wall", "taller_wall", "harder_hit")
        },
        "out_of_scope_identical": True,
        "confirmation_2024": {"credible_regression": False},
    }


def test_recommendation_passes_only_when_every_criterion_holds() -> None:
    rec = ev.recommend_v12b_adoption(_passing())
    assert rec["recommend_adopt"] is True and rec["requires_maintainer_decision"] is True


@pytest.mark.parametrize(
    ("path", "value", "criterion"),
    [
        (("geometry_vs_reference", "log_loss", "ci_high"), 0.0, "geometry_adds"),
        (("calibration", "overall_material_regression"), True, "calibration"),
        (("perturbations", "harder_hit", "passed"), False, "perturbations"),
        (("out_of_scope_identical",), False, "out_of_scope_identical"),
        (("confirmation_2024", "credible_regression"), True, "confirmation_2024"),
    ],
)
def test_any_single_failure_blocks(path: tuple, value: object, criterion: str) -> None:
    s = _passing()
    target = s
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    rec = ev.recommend_v12b_adoption(s)
    assert rec["recommend_adopt"] is False and rec["criteria"][criterion] is False


def test_group_regression_needs_adequate_support() -> None:
    s = _passing()
    s["groups"][0]["paired_log_loss_ci_low"] = 0.002
    assert ev.recommend_v12b_adoption(s)["criteria"]["no_group_regression"] is False
    assert ev.recommend_v12b_adoption(_passing())["criteria"]["no_group_regression"] is True


def test_full_evaluation_runs_end_to_end(rows: pd.DataFrame) -> None:
    folds = ev.evaluation_folds()
    primary = ev.run_folds(rows, folds["primary"])
    confirmation = ev.run_folds(rows, folds["confirmation"])
    rng = np.random.default_rng(2)
    weather = pd.DataFrame(
        {
            "event_id": rows["event_id"],
            "roof_status": "outdoor_open_air",
            "air_density_kg_m3": rng.normal(1.18, 0.03, len(rows)),
            "following_wind_mps": rng.normal(0, 3, len(rows)),
        }
    )
    summary = ev.evaluate(primary, confirmation, weather, n_reps=20, seed=1)
    json.dumps(summary, default=float)
    assert set(summary["recommendation"]["criteria"]) == {
        "geometry_adds",
        "calibration",
        "no_group_regression",
        "perturbations",
        "out_of_scope_identical",
        "confirmation_2024",
    }
    names = {g["group"] for g in summary["groups"]}
    assert any(n.startswith("venue_") for n in names)
    assert not any(n.startswith("band_") for n in names)  # bands are not in the v1.2b spec
    assert "leak_gap_vs_baseline" in summary["reported_only"]
    assert "robbery_zone" in summary["reported_only"]
