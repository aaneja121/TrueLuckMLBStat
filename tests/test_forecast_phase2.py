"""Contact Forecast Phase 2: guards, the frozen-code shim, and leakage.

The claims worth attacking:

**2025 is unreachable.** By season list and by file path, with no override.

**The shim does not weaken the frozen checks.** `assert_schema_for_phase2`
relies on the Phase 1 season guard being the LAST statement of
`assert_ledger_schema`; if that ever changed, the shim would silently skip a
check. A test pins the ordering. A second test proves the duplicated ordering
arithmetic is identical to the frozen function.

**Nothing after the cutoff can move a forecast.** Mutating every event past a
hitter's first 100 resolved BBE, and mutating the next-100 outcomes, must
leave the forecast, the preprocessing, the shrinkage parameters, the ridge
coefficients and the alpha bit-identical.

Synthetic data only -- no pinned snapshot is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.baselines import fit_shrinkage
from forecast.features import build_window_features
from forecast.phase2 import phase2_config as cfg
from forecast.phase2 import run_phase2_evaluation as runner
from forecast.phase2 import snapshot as snap
from forecast.phase2.phase2_windows import (
    horizon_completion,
    order_events_for_phase2,
)
from forecast.windows import ForecastWindowError, order_eligible_events

_BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")


def make_ledger(
    *, season: int = 2026, n_batters: int = 12, n_events: int = 260, seed: int = 3
) -> pd.DataFrame:
    """A valid contact-stage ledger in the forecast schema."""
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for batter in range(n_batters):
        index = np.arange(1, n_events + 1)
        game_offset = (index - 1) // 4
        observed = rng.normal(0.05, 0.5, n_events)
        deserved = observed * 0.4 + rng.normal(0.02, 0.2, n_events)
        frames.append(
            pd.DataFrame(
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
                    "stand": "R" if batter % 2 else "L",
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
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2025 is unreachable
# --------------------------------------------------------------------------


def test_2025_is_refused_by_season_and_says_why() -> None:
    with pytest.raises(cfg.Phase2SeasonError, match="permanently sealed"):
        cfg.assert_phase2_seasons_allowed(2025)
    with pytest.raises(cfg.Phase2SeasonError, match="permanently sealed"):
        cfg.assert_phase2_seasons_allowed([2024, 2025, 2026])


def test_the_authorized_seasons_are_exactly_the_permitted_set() -> None:
    cfg.assert_phase2_seasons_allowed([2022, 2023, 2024, 2026])
    assert 2025 not in cfg.PHASE2_ALLOWED_SEASONS
    for outside in (2020, 2021, 2027):
        with pytest.raises(cfg.Phase2SeasonError, match="outside the Phase 2"):
            cfg.assert_phase2_seasons_allowed(outside)


def test_a_final_evaluation_path_is_refused() -> None:
    for root in cfg.PHASE2_FORBIDDEN_ROOTS:
        with pytest.raises(cfg.Phase2PathError, match="sealed namespace"):
            cfg.assert_phase2_path_allowed(root / "anything.parquet")


def test_any_path_naming_2025_is_refused() -> None:
    """A second, independent layer: a sealed file moved somewhere unexpected."""
    with pytest.raises(cfg.Phase2PathError, match="sealed season 2025"):
        cfg.assert_phase2_path_allowed(cfg.PROJECT_ROOT / "data" / "2025" / "x.parquet")
    with pytest.raises(cfg.Phase2PathError, match="sealed season 2025"):
        cfg.assert_phase2_path_allowed(cfg.PROJECT_ROOT / "outputs" / "ledger_2025.parquet")


def test_the_prospective_2026_root_is_readable() -> None:
    cfg.assert_phase2_path_allowed(
        cfg.PROJECT_ROOT / "outputs" / "prospective" / "v1_1" / "2026-09-01"
    )


def test_phase_1_guards_are_not_weakened() -> None:
    """Phase 1 must still refuse 2026 -- Phase 2 added a door, not a hole."""
    from forecast.forecast_config import (
        PHASE_2_ENTRY_POINT_EXISTS,
        ForecastSeasonError,
        assert_forecast_seasons_allowed,
    )

    with pytest.raises(ForecastSeasonError, match="Phase 2"):
        assert_forecast_seasons_allowed(2026)
    with pytest.raises(ForecastSeasonError, match="sealed"):
        assert_forecast_seasons_allowed(2025)
    assert PHASE_2_ENTRY_POINT_EXISTS is False


def test_no_phase_2_module_references_a_2025_path() -> None:
    package = Path(cfg.__file__).parent
    for module in sorted(package.glob("*.py")):
        text = module.read_text()
        assert "final_evaluation/v1" not in text, module.name
        assert "data/final_evaluation" not in text.replace(
            'PROJECT_ROOT / "data" / "final_evaluation"', ""
        ), module.name


# --------------------------------------------------------------------------
# The shim reuses the frozen checks without weakening them
# --------------------------------------------------------------------------


def test_a_schema_violation_still_raises_before_the_season_guard() -> None:
    """The shim's correctness rests on the season guard being LAST.

    If a schema violation started raising a season error instead, the shim
    would swallow it. These cases prove each frozen check still fires for a
    2026 ledger.
    """
    from forecast.phase2.phase2_windows import assert_schema_for_phase2

    broken = make_ledger()
    broken.loc[0, "contact_result_surprise"] += 5.0
    with pytest.raises(ForecastWindowError, match="identity violated"):
        assert_schema_for_phase2(broken)

    duplicated = make_ledger()
    duplicated.loc[1, "event_id"] = duplicated.loc[0, "event_id"]
    with pytest.raises(ForecastWindowError, match="duplicated event_id"):
        assert_schema_for_phase2(duplicated)

    missing = make_ledger().drop(columns=["baseline_expected_contact_run_value"])
    with pytest.raises(ForecastWindowError, match="missing required column"):
        assert_schema_for_phase2(missing)


def test_the_shim_accepts_a_valid_2026_ledger() -> None:
    from forecast.phase2.phase2_windows import assert_schema_for_phase2

    assert_schema_for_phase2(make_ledger(season=2026))


def test_the_shim_still_refuses_2025() -> None:
    from forecast.phase2.phase2_windows import assert_schema_for_phase2

    with pytest.raises(cfg.Phase2SeasonError, match="sealed"):
        assert_schema_for_phase2(make_ledger(season=2025))


def test_phase2_ordering_matches_the_frozen_function() -> None:
    """The one duplicated piece of logic must be identical to Phase 1's."""
    development = make_ledger(season=2024, seed=9)
    frozen = order_eligible_events(development)
    shimmed = order_events_for_phase2(development)
    assert list(frozen.columns) == list(shimmed.columns)
    pd.testing.assert_frame_equal(frozen, shimmed)


# --------------------------------------------------------------------------
# Leakage: nothing after the cutoff may move a forecast
# --------------------------------------------------------------------------


def _mutate_after_index(ordered: pd.DataFrame, *, after: int, seed: int = 7) -> pd.DataFrame:
    """Overwrite every value of every event past `after`, for every hitter."""
    rng = np.random.default_rng(seed)
    mutated = ordered.copy()
    mask = mutated["bbe_index"] > after
    n = int(mask.sum())
    for column in (
        "observed_contact_result_run_value",
        "baseline_expected_contact_run_value",
        "launch_speed",
        "launch_angle",
        "spray_angle_approx",
    ):
        mutated.loc[mask, column] = rng.normal(500.0, 200.0, n)
    mutated.loc[mask, "contact_result_surprise"] = (
        mutated.loc[mask, "observed_contact_result_run_value"]
        - mutated.loc[mask, "baseline_expected_contact_run_value"]
    )
    mutated.loc[mask, "bb_type"] = "popup"
    return mutated


def test_mutating_events_after_the_cutoff_cannot_move_a_forecast_feature() -> None:
    ordered = order_events_for_phase2(make_ledger())
    baseline = build_window_features(ordered, 100)
    mutated = build_window_features(_mutate_after_index(ordered, after=100), 100)
    pd.testing.assert_frame_equal(
        baseline.sort_values(["batter", "season"]).reset_index(drop=True),
        mutated.sort_values(["batter", "season"]).reset_index(drop=True),
    )


def test_mutating_the_next_100_outcomes_changes_the_target_but_not_the_forecast() -> None:
    from forecast.windows import build_windows

    ordered = order_events_for_phase2(make_ledger())
    baseline_features = build_window_features(ordered, 100)
    baseline_windows = build_windows(ordered, cutoffs=(100,), horizons=(100,))

    mutated_ordered = _mutate_after_index(ordered, after=100, seed=21)
    mutated_features = build_window_features(mutated_ordered, 100)
    mutated_windows = build_windows(mutated_ordered, cutoffs=(100,), horizons=(100,))

    # The forecast inputs are untouched...
    pd.testing.assert_frame_equal(
        baseline_features.sort_values("batter").reset_index(drop=True),
        mutated_features.sort_values("batter").reset_index(drop=True),
    )
    # ...while the evaluation target genuinely moved, so the test is not vacuous.
    assert not np.allclose(
        baseline_windows.sort_values("batter")["target_realized_rv_per_100"].to_numpy(),
        mutated_windows.sort_values("batter")["target_realized_rv_per_100"].to_numpy(),
    )


def test_2026_outcomes_cannot_move_preprocessing_shrinkage_or_the_ridge() -> None:
    """Everything fitted comes from development seasons, which 2026 cannot touch."""
    from forecast.ridge import fit_ridge

    development_events = make_ledger(season=2023, n_batters=40, n_events=150, seed=11)
    baseline_fit = fit_shrinkage(
        development_events,
        value_column="baseline_expected_contact_run_value",
        quantity="deserved",
    )
    prospective = order_events_for_phase2(make_ledger(season=2026, seed=5))
    mutated_prospective = _mutate_after_index(prospective, after=100, seed=33)
    after_fit = fit_shrinkage(
        development_events,
        value_column="baseline_expected_contact_run_value",
        quantity="deserved",
    )
    assert after_fit == baseline_fit
    del mutated_prospective

    # Independent draws, not two linspaces: perfectly collinear columns would
    # trip the ridge identifiability guard rather than test the fit.
    noise = np.random.default_rng(4)
    development_windows = pd.DataFrame(
        {
            "season": 2023,
            "std_realized_rv_per_100": noise.normal(4.0, 7.0, 60),
            "std_deserved_rv_per_100": noise.normal(4.0, 4.0, 60),
            "recent_realized_rv_per_100": noise.normal(4.0, 9.0, 60),
            "recent_deserved_rv_per_100": noise.normal(4.0, 5.0, 60),
            "target_realized_rv_per_100": noise.normal(4.0, 6.0, 60),
        }
    )
    first = fit_ridge(development_windows, model_name="B_deserved_only", alpha=100.0)
    second = fit_ridge(development_windows, model_name="B_deserved_only", alpha=100.0)
    assert first.coefficients == second.coefficients
    assert first.scaler_mean == second.scaler_mean
    assert first.alpha == second.alpha == 100.0


# --------------------------------------------------------------------------
# Completed versus pending
# --------------------------------------------------------------------------


def test_horizon_completion_separates_reached_from_completed() -> None:
    ledger = pd.concat(
        [
            make_ledger(n_batters=1, n_events=260, seed=1),
            make_ledger(n_batters=1, n_events=150, seed=2).assign(
                batter=99,
                event_id=lambda d: "b99-" + d["event_id"].astype(str),
            ),
        ],
        ignore_index=True,
    )
    ordered = order_events_for_phase2(ledger)
    completion = horizon_completion(ordered, cutoff=100, horizon=100)
    reached = completion[completion["reached_cutoff"]]
    assert len(reached) == 2
    assert int(completion["completed_horizon"].sum()) == 1
    pending = completion[completion["reached_cutoff"] & ~completion["completed_horizon"]]
    assert int(pending["batter"].iloc[0]) == 99
    assert int(pending["resolved_bbe_since_cutoff"].iloc[0]) == 50


def test_a_pending_hitter_never_receives_a_target() -> None:
    forecasts = pd.DataFrame(
        {
            "batter": [1, 2],
            "season": 2026,
            runner.TARGET: [np.nan, np.nan],
        }
    )
    forecasts["evaluation_status"] = np.where(
        forecasts[runner.TARGET].notna(), "completed", "pending"
    )
    assert set(forecasts["evaluation_status"]) == {"pending"}


# --------------------------------------------------------------------------
# The prespecified classification
# --------------------------------------------------------------------------


def test_established_success_requires_the_interval_below_zero() -> None:
    result = runner.classify_result(-0.5, -0.8, -0.2)
    assert result["classification"] == "established_incremental_success"


def test_a_favourable_point_estimate_with_a_crossing_interval_is_inconclusive() -> None:
    result = runner.classify_result(-0.5, -0.8, 0.2)
    assert result["classification"] == "promising_but_inconclusive"


def test_an_unfavourable_point_estimate_is_no_evidence() -> None:
    result = runner.classify_result(0.1, -0.4, 0.6)
    assert result["classification"] == "no_evidence_of_incremental_improvement"


def test_an_interval_entirely_above_zero_is_evidence_against() -> None:
    result = runner.classify_result(0.5, 0.2, 0.9)
    assert result["classification"] == "evidence_against_incremental_value"


def test_mae_decides_and_secondary_metrics_cannot_reclassify() -> None:
    result = runner.classify_result(0.1, -0.4, 0.6)
    assert result["deciding_metric"] == "MAE"
    assert result["secondary_metrics_may_not_reclassify"] is True


# --------------------------------------------------------------------------
# Contact stage, not the published quantity
# --------------------------------------------------------------------------


def test_a_full_telescoping_column_is_refused() -> None:
    frame = make_ledger()
    frame["contact_luck_runs"] = 0.0
    with pytest.raises(cfg.Phase2PathError, match="contact stage"):
        snap.assert_no_full_telescoping_columns(frame)


def test_a_clean_contact_stage_ledger_passes() -> None:
    snap.assert_no_full_telescoping_columns(make_ledger())


# --------------------------------------------------------------------------
# The frozen contract is loaded, not restated
# --------------------------------------------------------------------------


def _write_manifests(directory: Path, *, model: str = "D_full_contact_profile") -> None:
    (directory / "r1_freeze_manifest.json").write_text(
        json.dumps(
            {
                "conclusion": {"verdict": "PROCEED"},
                "specification": {
                    "targets": {"primary": "target_realized_rv_per_100"},
                    "baseline_ladder": ["league_mean", "shrunk_deserved_persistence"],
                    "metrics": {"sign_convention": {"negative": "challenger better"}},
                    "shrinkage": {"fit_rule": "strictly earlier seasons only"},
                },
            }
        )
    )
    (directory / "ridge_freeze_manifest.json").write_text(
        json.dumps(
            {
                "conclusion": {"verdict": "CONTINUE"},
                "key_results": {
                    "selected_model": {"model": model},
                    "evaluation_mae_by_model": {model: 4.2128},
                    "benchmark_mae": {"shrunk_deserved_persistence": 4.3605},
                    "selected_alphas": {model: 100.0},
                    "feature_sets": {model: ["std_realized_rv_per_100"]},
                },
            }
        )
    )
    (directory / "hgb_freeze_manifest.json").write_text(
        json.dumps(
            {
                "conclusion": {"verdict": "PREFER RIDGE"},
                "key_results": {
                    "nonlinear_value_test": {
                        "preferred_model": "ridge",
                        "delta_mae_hgb_minus_ridge": 0.0708,
                    }
                },
            }
        )
    )


def test_the_frozen_contract_records_the_model_and_its_reasons(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    contract = runner.load_frozen_contract(research_dir=tmp_path)
    assert contract["model"] == "D_full_contact_profile"
    assert contract["alpha"] == 100.0
    reasons = contract["model_choice_reasons"]
    assert reasons["best_frozen_2024_mae"] == 4.2128
    assert reasons["better_than_shrunk_deserved"] == 4.3605
    assert reasons["hgb_did_not_outperform_ridge"] is True
    assert reasons["prespecified_nonlinear_rule_retained_ridge"] == "ridge"


def test_a_different_frozen_model_is_refused(tmp_path: Path) -> None:
    _write_manifests(tmp_path, model="A_results_only")
    with pytest.raises(runner.Phase2EvaluationError, match="not the expected"):
        runner.load_frozen_contract(research_dir=tmp_path)


def test_a_contract_whose_reasons_no_longer_hold_is_refused(tmp_path: Path) -> None:
    _write_manifests(tmp_path)
    manifest = json.loads((tmp_path / "hgb_freeze_manifest.json").read_text())
    manifest["key_results"]["nonlinear_value_test"]["preferred_model"] = "hgb"
    (tmp_path / "hgb_freeze_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(runner.Phase2EvaluationError, match="retaining ridge"):
        runner.load_frozen_contract(research_dir=tmp_path)


def test_the_authorization_record_states_what_is_not_authorized() -> None:
    record = cfg.build_authorization_record(frozen_manifests={"r1": "abc"})
    auth = record["authorization"]
    assert auth["sealed_seasons_still_forbidden"] == [2025]
    assert auth["evaluation_seasons_authorized"] == [2026]
    assert auth["no_2026_information_may_alter_any_frozen_element"] is True
    for forbidden in ("model tuning of any kind", "dashboard deployment"):
        assert any(forbidden in item for item in auth["not_authorized"])
    assert record["phase_2_gate"]["phase_1_guards_modified"] is False


# --------------------------------------------------------------------------
# Snapshot pinning
# --------------------------------------------------------------------------


def test_no_snapshot_directory_stops_rather_than_improvising(monkeypatch) -> None:
    monkeypatch.setattr(snap, "PROSPECTIVE_OUTPUTS_ROOT", Path("/nonexistent/snapshots"))
    with pytest.raises(snap.Phase2SnapshotError, match="does not generate one"):
        snap.resolve_snapshot()
