"""Contact Forecast R1: season-policy and Phase-2-gate guards.

These tests exist to make two claims mechanically checkable rather than
merely documented: that 2025 is unreachable from this package under any
argument, and that Phase 1 contains no door to 2026 at all.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from forecast import forecast_config as cfg
from mlb_luck_score.config import TRAIN_SEASONS


def test_sealed_2025_is_refused() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="permanently sealed"):
        cfg.assert_forecast_seasons_allowed(2025)


def test_sealed_2025_is_refused_inside_a_mixed_season_list() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="permanently sealed"):
        cfg.assert_forecast_seasons_allowed([2022, 2023, 2025])


def test_phase_2_season_2026_is_refused_and_names_the_gate() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="Phase 2"):
        cfg.assert_forecast_seasons_allowed(2026)


def test_assert_forecast_seasons_allowed_has_no_override_parameter() -> None:
    """The Phase 2 gate is a door that was never built, not a flag set False.

    An `allow_phase_2=True`-style parameter would be a weaker guard than no
    parameter at all: it turns opening 2026 into a one-keyword change that
    review could miss. Phase 2 must add its own separately-named entry
    point instead. This test fails the moment someone adds the keyword.
    """
    signature = inspect.signature(cfg.assert_forecast_seasons_allowed)
    assert list(signature.parameters) == ["seasons"]


def test_phase_1_declares_no_phase_2_entry_point() -> None:
    assert cfg.PHASE_2_ENTRY_POINT_EXISTS is False


def test_phase_2_gate_requires_the_spec_to_be_frozen_first() -> None:
    """The immutability requirement: everything is frozen BEFORE 2026 opens."""
    gate = cfg.PHASE_2_GATE
    assert "forecast_spec.json" in gate[0]
    frozen = [item for item in gate if item.startswith("frozen:")]
    for expected in (
        "feature definitions",
        "target definitions",
        "model class",
        "matched-shrinkage",
        "rescaling/calibration",
        "evaluation metric set",
        "exclusion rules",
        "report template",
    ):
        assert any(expected in item for item in frozen), f"gate does not freeze {expected!r}"
    assert "authorized" in gate[-1]


@pytest.mark.parametrize("season", cfg.FORECAST_DEVELOPMENT_SEASONS)
def test_development_seasons_are_allowed(season: int) -> None:
    cfg.assert_forecast_seasons_allowed(season)


def test_unknown_season_is_refused() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="not Contact Forecast development"):
        cfg.assert_forecast_seasons_allowed(2019)


def test_non_integer_season_is_refused() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="not an integer year"):
        cfg.assert_forecast_seasons_allowed("not-a-year")


def test_2021_is_a_feature_source_but_never_an_analysis_season() -> None:
    assert 2021 in cfg.FORECAST_DEVELOPMENT_SEASONS
    assert 2021 not in cfg.FORECAST_ANALYSIS_SEASONS
    assert 2021 not in cfg.WALK_FORWARD_TRAIN_SEASONS


@pytest.mark.parametrize("analysis_season", cfg.FORECAST_ANALYSIS_SEASONS)
def test_walk_forward_training_seasons_are_strictly_earlier(analysis_season: int) -> None:
    train = cfg.walk_forward_train_seasons(analysis_season)
    assert train, "every analysis season needs at least one training season"
    assert max(train) < analysis_season


def test_2024_walk_forward_config_equals_production_train_seasons() -> None:
    """The 2024 slice of this study is scored by exactly production's config.

    This is the claim that lets the report say the 2024 results are not an
    artifact of the walk-forward departure -- it holds only while these two
    tuples stay equal.
    """
    assert cfg.walk_forward_train_seasons(2024) == TRAIN_SEASONS


def test_walk_forward_rejects_a_non_analysis_season() -> None:
    with pytest.raises(cfg.ForecastSeasonError, match="not a Contact Forecast analysis season"):
        cfg.walk_forward_train_seasons(2021)


def test_walk_forward_detects_look_ahead_introduced_by_a_later_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future edit that lets a season train on itself must fail loudly."""
    monkeypatch.setitem(cfg.WALK_FORWARD_TRAIN_SEASONS, 2023, (2021, 2022, 2023))
    with pytest.raises(cfg.ForecastSeasonError, match="Look-ahead"):
        cfg.walk_forward_train_seasons(2023)


@pytest.mark.parametrize(
    "forbidden",
    [
        Path("data/final_evaluation/2025/statcast.parquet"),
        Path("outputs/final_evaluation/v1/report.json"),
        Path("artifacts/final_evaluation/v1/seal.json"),
        Path("data/prospective/2026/raw.parquet"),
        Path("outputs/prospective/v1_1/2026-09-01/play_ledger.parquet"),
        Path("artifacts/prospective/v1_1/2026-09-01/manifest.json"),
    ],
)
def test_sealed_and_phase_2_namespaces_are_refused(forbidden: Path) -> None:
    with pytest.raises(cfg.ForecastNamespaceError, match="forbidden namespace"):
        cfg.assert_path_outside_forbidden_namespaces(cfg.PROJECT_ROOT / forbidden)


@pytest.mark.parametrize(
    "allowed",
    [
        Path("data/processed/cleaned_development_data_with_sprint_speed.parquet"),
        Path("data/forecast/walk_forward_ledger_2024.parquet"),
        Path("outputs/forecast_research/baseline_results.json"),
    ],
)
def test_development_and_forecast_namespaces_are_allowed(allowed: Path) -> None:
    cfg.assert_path_outside_forbidden_namespaces(cfg.PROJECT_ROOT / allowed)


def test_forecast_artifacts_never_land_in_a_production_namespace() -> None:
    for directory in (cfg.FORECAST_DATA_DIR, cfg.FORECAST_OUTPUTS_DIR, cfg.FORECAST_SPEC_PATH):
        cfg.assert_path_outside_forbidden_namespaces(directory)


def test_bootstrap_clusters_on_batter_not_batter_season() -> None:
    """The same hitter recurs across seasons and across overlapping cells."""
    assert cfg.BOOTSTRAP_CLUSTER_COLUMN == "batter"


def test_window_grid_is_deterministic_and_complete() -> None:
    grid = cfg.window_grid()
    assert grid == tuple((k, h) for k in cfg.FORECAST_CUTOFFS for h in cfg.FORECAST_HORIZONS)
    assert len(grid) == len(cfg.FORECAST_CUTOFFS) * len(cfg.FORECAST_HORIZONS)
    assert cfg.window_grid() == grid


def test_season_forward_folds_never_fit_on_a_later_season() -> None:
    for fit_seasons, evaluate_season in cfg.SEASON_FORWARD_FOLDS:
        assert max(fit_seasons) < evaluate_season
        cfg.assert_forecast_seasons_allowed([*fit_seasons, evaluate_season])
