"""The pre-registered 2027 v1.3 vs v1.1 comparison (docs/plans/v1_3_2027_preregistration.md).

Synthetic data only: "2027" rows are generated, never downloaded; guards, ingestion and
the dataset build are replaced with fakes. The classification rule is tested exhaustively.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import run_v1_3_2027_comparison as cmp
import v1_3_2027_config as cfg

from mlb_luck_score.config import CLASS_ORDER

_VENUES = (3.0, 17.0, 19.0, 4705.0)


def _rows(season: int, n_games: int = 40, per_game: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + season)
    rows = []
    for g in range(n_games):
        density = rng.normal(1.15, 0.05) if g % 5 else np.nan
        for _ in range(per_game):
            bb = rng.choice(
                ["fly_ball", "line_drive", "ground_ball", "popup"], p=[0.4, 0.25, 0.3, 0.05]
            )
            la = {"fly_ball": rng.uniform(22, 45), "line_drive": rng.uniform(10, 22)}.get(
                bb, rng.uniform(-20, 8) if bb == "ground_ball" else rng.uniform(50, 70)
            )
            rows.append(
                {
                    "event_id": f"{season}-{len(rows)}",
                    "season": season,
                    "game_pk": season * 1000 + g,
                    "game_date": f"{season}-06-15",
                    "venue_id": _VENUES[g % len(_VENUES)],
                    "has_venue_metadata": True,
                    "launch_speed": rng.uniform(70, 112),
                    "launch_angle": la,
                    "spray_angle_approx": rng.uniform(-44, 44),
                    "bb_type": bb,
                    "stand": rng.choice(["L", "R"]),
                    "eligible_for_training": True,
                    "roof_status": "outdoor_open_air" if g % 5 else "fixed_indoor",
                    "air_density_kg_m3": density,
                    "following_wind_mps": rng.normal(0, 3),
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
    p = 1 / (1 + np.exp(-(carry - 395) / 6))
    out = np.where(rng.uniform(size=len(df)) < p, "home_run", "out")
    r = rng.uniform(size=len(df))
    out = np.where((out == "out") & (r < 0.2), "single", out)
    out = np.where((out == "out") & (r > 0.93), "double", out)
    out = np.where((out == "out") & (r > 0.925) & (r <= 0.93), "triple", out)
    df["outcome_class"] = out
    return df


@pytest.fixture(scope="module")
def dev() -> pd.DataFrame:
    return pd.concat([_rows(s) for s in (2021, 2022, 2023, 2024)], ignore_index=True)


@pytest.fixture(scope="module")
def season_2027() -> pd.DataFrame:
    return _rows(2027, n_games=60)


# --- classification ------------------------------------------------------------------------


def _gates(**over):
    base = {
        "n_population": 120_000,
        "primary_ci": (-0.02, -0.01),
        "calibration_ok": True,
        "no_venue_regression": True,
        "directions_ok": True,
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({}, "confirmed"),
        ({"n_population": 59_999}, "inconclusive"),
        ({"n_population": 59_999, "primary_ci": (0.01, 0.02)}, "inconclusive"),
        ({"primary_ci": (0.001, 0.02)}, "not_confirmed"),
        ({"primary_ci": (-0.01, 0.01)}, "inconclusive"),
        ({"calibration_ok": False}, "not_confirmed"),
        ({"no_venue_regression": False}, "not_confirmed"),
        ({"directions_ok": False}, "not_confirmed"),
        ({"primary_ci": (-0.01, 0.01), "directions_ok": False}, "not_confirmed"),
        ({"primary_ci": (-0.02, 0.0)}, "inconclusive"),
    ],
)
def test_classification_follows_the_preregistration(over: dict, expected: str) -> None:
    assert cmp.classify(**_gates(**over)) == expected


# --- training and scoring ----------------------------------------------------------------------


def test_models_train_on_2021_2023_only(dev: pd.DataFrame) -> None:
    train = cmp.training_rows(dev)
    assert sorted(train["season"].unique()) == [2021, 2022, 2023]


def test_2027_rows_must_be_2027_only(season_2027: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        cmp.prepare_2027_rows(pd.concat([season_2027, season_2027.head(3).assign(season=2026)]))


def test_scoring_produces_three_models_that_ignore_distance(
    dev: pd.DataFrame, season_2027: pd.DataFrame
) -> None:
    models = cmp.fit_models(cmp.training_rows(dev))
    rows = cmp.prepare_2027_rows(season_2027)
    pred = cmp.score(models, rows)
    for prefix in ("base_", "cand_", "nod_"):
        np.testing.assert_allclose(
            pred[[f"{prefix}{c}" for c in CLASS_ORDER]].sum(axis=1), 1.0, atol=1e-9
        )
    moved = cmp.score(models, rows.assign(hit_distance_sc=rows["hit_distance_sc"] + 80))
    np.testing.assert_array_equal(pred["cand_home_run"], moved["cand_home_run"])


def test_compare_reports_every_preregistered_quantity(
    dev: pd.DataFrame, season_2027: pd.DataFrame
) -> None:
    models = cmp.fit_models(cmp.training_rows(dev))
    rows = cmp.prepare_2027_rows(season_2027)
    weather = rows[["event_id", "roof_status", "air_density_kg_m3", "following_wind_mps"]]
    result = cmp.compare(models, rows, weather, n_reps=20, seed=1)
    json.dumps(result, default=float)
    assert result["classification"] in {"confirmed", "not_confirmed", "inconclusive"}
    assert result["n_population"] == len(rows)
    assert result["classification"] == "inconclusive"  # synthetic population < 60,000
    for key in (
        "primary",
        "calibration",
        "venues",
        "directions",
        "density_test",
        "weather_residual",
    ):
        assert key in result
    assert result["density_test"]["status"] in {"supported", "not_supported"}
    assert set(result["directions"]) == {"harder_hit", "thinner_air"}


# --- orchestration: one run, guarded, inside the namespace ----------------------------------------


@pytest.fixture
def wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev: pd.DataFrame, season_2027: pd.DataFrame
):
    out = tmp_path / "outputs" / "prospective" / "v1_3"
    monkeypatch.setattr(cfg, "NAMESPACE_ROOTS", (tmp_path / "raw", out, tmp_path / "art"))
    monkeypatch.setattr(cfg, "COMPARISON_PATH", out / "comparison_2027.json")
    calls: list[str] = []

    def guards(**kw):
        calls.append("guards")
        cfg.assert_not_already_run(cfg.COMPARISON_PATH)
        return {"season_end_date": "2027-09-26"}

    monkeypatch.setattr(cmp, "run_guards", guards)
    monkeypatch.setattr(
        cmp, "ingest_2027", lambda start, end: calls.append("ingest") or {"statcast": {"rows": 1}}
    )
    monkeypatch.setattr(
        cmp,
        "build_scoring_dataset",
        lambda: (calls.append("build") or season_2027.copy(), {"rows": len(season_2027)}),
    )
    monkeypatch.setattr(cmp, "load_development_rows", lambda: dev.copy())
    monkeypatch.setattr(cmp, "current_commit", lambda: "abc123")
    monkeypatch.setattr(cmp, "N_BOOTSTRAP_REPS", 20)
    return calls


def test_main_runs_guards_first_and_writes_once(wired: list[str]) -> None:
    assert cmp.main([]) == 0
    assert wired == ["guards", "ingest", "build"]
    written = json.loads(cfg.COMPARISON_PATH.read_text())
    assert written["code_commit"] == "abc123"
    assert written["preregistration"] == cfg.PREREGISTRATION
    assert written["result"]["classification"] == "inconclusive"
    with pytest.raises(cfg.AlreadyRunError):
        cmp.main([])


def test_main_stops_before_any_download_when_guards_fail(
    wired: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**kw):
        raise cfg.SeasonNotRecordedError("not recorded")

    monkeypatch.setattr(cmp, "run_guards", refuse)
    with pytest.raises(cfg.SeasonNotRecordedError):
        cmp.main([])
    assert "ingest" not in wired and not cfg.COMPARISON_PATH.exists()


def test_the_writer_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(cfg, "NAMESPACE_ROOTS", (out,))
    path = out / "comparison_2027.json"
    cmp.write_once(path, {"a": 1})
    with pytest.raises(cfg.AlreadyRunError):
        cmp.write_once(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_a_row_without_a_venue_cannot_crash_the_one_run(
    dev: pd.DataFrame, season_2027: pd.DataFrame
) -> None:
    models = cmp.fit_models(cmp.training_rows(dev))
    rows = cmp.prepare_2027_rows(season_2027)
    rows.loc[rows.index[:3], "venue_id"] = np.nan
    weather = rows[["event_id", "roof_status", "air_density_kg_m3", "following_wind_mps"]]
    result = cmp.compare(models, rows, weather, n_reps=5, seed=1)
    assert result["n_population"] == len(rows)
