"""Tests for the v1.3 outcome-free contact candidate (`outcome_free_v13`).

Synthetic data only: batted balls at real venues with outcomes from a known rule in
which carry grows with launch speed and falls with air density, so the physical
directions are known. No real data, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.models import evaluate_outcome_free_v13 as ev

_VENUES = (3.0, 17.0, 19.0, 4705.0, 5325.0, 2889.0)


def _synthetic(
    n_games_per_season: int = 36, rows_per_game: int = 30, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    game_pk = 0
    for season in (2021, 2022, 2023, 2024):
        for g in range(n_games_per_season):
            game_pk += 1
            venue = _VENUES[g % len(_VENUES)]
            roof = "fixed_indoor" if venue == 5325.0 and game_pk % 2 else "outdoor_open_air"
            density = np.nan if roof == "fixed_indoor" else rng.normal(1.15, 0.05)
            for _ in range(rows_per_game):
                bb = rng.choice(
                    ["fly_ball", "line_drive", "ground_ball", "popup"], p=[0.4, 0.25, 0.3, 0.05]
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
                        "venue_id": venue,
                        "has_venue_metadata": True,
                        "launch_speed": rng.uniform(70, 112),
                        "launch_angle": la,
                        "spray_angle_approx": rng.uniform(-44, 44),
                        "bb_type": bb,
                        "stand": rng.choice(["L", "R"]),
                        "eligible_for_training": True,
                        "roof_status": roof,
                        "air_density_kg_m3": density,
                    }
                )
    df = pd.DataFrame(rows)
    dens = df["air_density_kg_m3"].fillna(1.18)
    carry = (
        4.1
        * df["launch_speed"]
        * np.sin(np.radians(np.clip(2 * df["launch_angle"], 1, 179))) ** 0.6
        - 25
        - 400 * (dens - 1.15)
    )
    df["hit_distance_sc"] = carry + rng.normal(0, 10, len(df))
    p_hr = 1 / (1 + np.exp(-(carry - 395) / 6))
    u = rng.uniform(size=len(df))
    out = np.where(u < p_hr, "home_run", "out")
    r = rng.uniform(size=len(df))
    gb = df["bb_type"] == "ground_ball"
    out = np.where((out == "out") & (r < np.where(gb, 0.25, 0.15)), "single", out)
    out = np.where((out == "out") & (r > 0.93), "double", out)
    out = np.where((out == "out") & (r > 0.925) & (r <= 0.93), "triple", out)
    df["outcome_class"] = out
    return df


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return ev.prepare_rows(_synthetic())


@pytest.fixture(scope="module")
def primary(rows: pd.DataFrame) -> ev.FoldRunResult:
    return ev.run_folds(rows, ev.evaluation_folds()["primary"])


# --- inputs ------------------------------------------------------------------------


def test_inputs_contain_no_distance_geometry_or_venue() -> None:
    forbidden = {
        "hit_distance_sc",
        "projected_distance_to_wall_margin",
        "wall_distance_in_spray_direction",
        "wall_height_in_spray_direction",
        "venue_id",
        "venue",
    }
    for features in (ev.CANDIDATE_FEATURES, ev.NO_DENSITY_FEATURES, ev.NO_SPRAY_FEATURES):
        assert not forbidden & set(features)
    assert set(ev.CANDIDATE_FEATURES) - set(ev.NO_DENSITY_FEATURES) == {"air_density_kg_m3"}
    assert set(ev.CANDIDATE_FEATURES) - set(ev.NO_SPRAY_FEATURES) == {"spray_angle_approx"}


def test_predictions_ignore_distance_and_geometry(rows: pd.DataFrame) -> None:
    models = ev.fit_fold(rows[rows["season"].isin((2021, 2022))])
    held = rows[rows["season"] == 2023]
    a = ev.predict_fold(models, held)["candidate"].to_numpy()
    changed = held.assign(
        hit_distance_sc=held["hit_distance_sc"] + 60.0,
        wall_distance_in_spray_direction=held["wall_distance_in_spray_direction"] - 30.0,
        wall_height_in_spray_direction=40.0,
    )
    b = ev.predict_fold(models, changed)["candidate"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_every_row_is_scored_and_probabilities_sum_to_one(
    rows: pd.DataFrame, primary: ev.FoldRunResult
) -> None:
    pred = primary.predictions
    held = rows[rows["season"].isin((2021, 2022, 2023))]
    assert sorted(pred["event_id"]) == sorted(held["event_id"])
    cand = pred[[f"cand_{c}" for c in ev.CLASS_ORDER]].to_numpy()
    np.testing.assert_allclose(cand.sum(axis=1), 1.0, atol=1e-9)


def test_roof_groups_cover_every_state() -> None:
    s = pd.Series(
        [
            "outdoor_open_air",
            "retractable_roof_open",
            "retractable_roof_closed",
            "fixed_indoor",
            "roof_status_unknown",
            None,
        ]
    )
    assert ev.roof_group(s).tolist() == [
        "outdoor",
        "roof_open",
        "closed_or_indoor",
        "closed_or_indoor",
        "unknown",
        "unknown",
    ]


@pytest.mark.parametrize("season", [2025, 2026])
def test_non_development_seasons_are_refused(season: int) -> None:
    df = _synthetic(n_games_per_season=2, rows_per_game=3)
    with pytest.raises(ValueError):
        ev.prepare_rows(df.assign(season=season))


def test_run_is_deterministic(rows: pd.DataFrame, primary: ev.FoldRunResult) -> None:
    again = ev.run_folds(rows, ev.evaluation_folds()["primary"])
    pd.testing.assert_frame_equal(primary.predictions, again.predictions)


# --- physics -----------------------------------------------------------------------------


def test_perturbations_recover_the_synthetic_physics(primary: ev.FoldRunResult) -> None:
    checks = ev.summarize_perturbations(primary.perturbations)
    assert set(checks) == {"harder_hit", "thinner_air"}
    assert checks["harder_hit"]["delta_p_hr"] > 0 and checks["harder_hit"]["passed"]
    assert checks["thinner_air"]["delta_p_hr"] > 0 and checks["thinner_air"]["passed"]
    assert (
        checks["thinner_air"]["n_rows"] < checks["harder_hit"]["n_rows"]
    )  # density-known rows only


# --- criteria ---------------------------------------------------------------------------------


def _passing() -> dict:
    return {
        "vs_baseline": {"log_loss": {"ci_high": -0.01}},
        "calibration": {
            "overall_material_regression": False,
            "home_run_material_regression": False,
        },
        "groups": [
            {"group": "venue_3", "adequate_support": True, "paired_log_loss_ci_low": -0.02},
            {"group": "venue_9", "adequate_support": False, "paired_log_loss_ci_low": 0.4},
        ],
        "perturbations": {"harder_hit": {"passed": True}, "thinner_air": {"passed": True}},
        "confirmation_2024": {"credible_regression": False},
    }


def test_recommendation_passes_only_when_every_gate_holds() -> None:
    rec = ev.recommend_v13_adoption(_passing())
    assert rec["recommend_adopt"] is True and rec["requires_maintainer_decision"] is True
    assert rec["real_test"] == "pre-registered 2027 prospective comparison"


@pytest.mark.parametrize(
    ("path", "value", "gate"),
    [
        (("vs_baseline", "log_loss", "ci_high"), 0.0, "log_loss"),
        (("calibration", "home_run_material_regression"), True, "calibration"),
        (("perturbations", "thinner_air", "passed"), False, "perturbations"),
        (("confirmation_2024", "credible_regression"), True, "confirmation_2024"),
    ],
)
def test_any_single_failure_blocks(path: tuple, value: object, gate: str) -> None:
    s = _passing()
    target = s
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    rec = ev.recommend_v13_adoption(s)
    assert rec["recommend_adopt"] is False and rec["criteria"][gate] is False


def test_group_regression_needs_adequate_support() -> None:
    s = _passing()
    s["groups"][0]["paired_log_loss_ci_low"] = 0.001
    assert ev.recommend_v13_adoption(s)["criteria"]["no_group_regression"] is False
    assert ev.recommend_v13_adoption(_passing())["criteria"]["no_group_regression"] is True


def test_full_evaluation_runs_end_to_end(rows: pd.DataFrame, primary: ev.FoldRunResult) -> None:
    confirmation = ev.run_folds(rows, ev.evaluation_folds()["confirmation"])
    rng = np.random.default_rng(3)
    weather = pd.DataFrame(
        {
            "event_id": rows["event_id"],
            "roof_status": rows["roof_status"],
            "air_density_kg_m3": rows["air_density_kg_m3"],
            "following_wind_mps": rng.normal(0, 3, len(rows)),
        }
    )
    summary = ev.evaluate(primary, confirmation, weather, n_reps=20, seed=1)
    json.dumps(summary, default=float)
    assert set(summary["recommendation"]["criteria"]) == {
        "log_loss",
        "calibration",
        "no_group_regression",
        "perturbations",
        "confirmation_2024",
    }
    names = {g["group"] for g in summary["groups"]}
    assert {"roof_outdoor", "bb_type_ground_ball"} <= names
    rep = summary["reported_only"]
    for key in (
        "density_ablation",
        "spray_ablation",
        "park_residual",
        "robbery_zone",
        "per_class_ece",
        "weather_residual",
    ):
        assert key in rep
    assert "ground_ball" in rep["spray_ablation"]
