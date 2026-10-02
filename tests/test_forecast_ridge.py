"""Contact Forecast: the ridge forecasting layer.

Three things are load-bearing and are attacked here.

**Identifiability.** Three exact linear dependencies exist in the declared
feature list. They must be resolved by declaration before fitting, and a
dependency that survives the declared drops must FAIL the fit rather than be
absorbed by the L2 penalty.

**Causality, by mutation.** Rewriting the evaluation season -- its features,
its targets, or both -- must not move a single fitted quantity: not the
standardization statistics, not the selected alpha, not a coefficient. The
tests rewrite those rows with garbage and assert the fitted model is
bit-identical.

**Sealed seasons.** 2025 and 2026 cannot reach an estimator, including through
a hand-written parquet.

Synthetic data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast import ridge as rg
from forecast import run_ridge_forecast as runner
from forecast.forecast_config import (
    FOCAL_PRESENTATION_CUTOFF,
    PRIMARY_HORIZON,
    ForecastSeasonError,
)

TARGET = rg.RIDGE_TARGET


def make_focal_table(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024),
    n_per_season: int = 120,
    seed: int = 17,
) -> pd.DataFrame:
    """A focal-cell prediction table carrying every declared feature and benchmark."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    batter = 0
    for season in seasons:
        for _ in range(n_per_season):
            batter += 1
            skill = rng.normal(4.0, 5.0)
            realized = skill + rng.normal(0.0, 7.0)
            deserved = skill + rng.normal(0.0, 3.0)
            recent_realized = skill + rng.normal(0.0, 9.0)
            recent_deserved = skill + rng.normal(0.0, 4.0)
            ground = float(rng.uniform(0.30, 0.50))
            line = float(rng.uniform(0.15, 0.25))
            fly = float(rng.uniform(0.20, 0.35))
            popup = max(0.01, 1.0 - ground - line - fly)
            total = ground + line + fly + popup
            rows.append(
                {
                    "window_id": f"w{batter}",
                    "batter": batter,
                    "season": season,
                    "cutoff": FOCAL_PRESENTATION_CUTOFF,
                    "horizon": PRIMARY_HORIZON,
                    TARGET: skill + rng.normal(0.0, 6.0),
                    "std_realized_rv_per_100": realized,
                    "std_deserved_rv_per_100": deserved,
                    "std_surprise_rv_per_100": realized - deserved,
                    "recent_realized_rv_per_100": recent_realized,
                    "recent_deserved_rv_per_100": recent_deserved,
                    "recent_surprise_rv_per_100": recent_realized - recent_deserved,
                    "n_resolved_bbe_through_cutoff": 100.0,
                    "launch_speed_mean": rng.normal(89.0, 3.0),
                    "launch_speed_sd": rng.normal(14.0, 1.5),
                    "launch_speed_p10": rng.normal(70.0, 4.0),
                    "launch_speed_p90": rng.normal(104.0, 3.0),
                    "launch_angle_mean": rng.normal(12.0, 5.0),
                    "launch_angle_sd": rng.normal(25.0, 2.0),
                    "launch_angle_p10": rng.normal(-20.0, 5.0),
                    "launch_angle_p90": rng.normal(42.0, 5.0),
                    "spray_angle_mean": rng.normal(0.0, 6.0),
                    "spray_angle_sd": rng.normal(25.0, 2.0),
                    "spray_angle_p10": rng.normal(-30.0, 4.0),
                    "spray_angle_p90": rng.normal(30.0, 4.0),
                    "hard_hit_rate": float(rng.uniform(0.25, 0.55)),
                    "sweet_spot_rate": float(rng.uniform(0.25, 0.45)),
                    "bb_rate_ground_ball": ground / total,
                    "bb_rate_line_drive": line / total,
                    "bb_rate_fly_ball": fly / total,
                    "bb_rate_popup": popup / total,
                    "bats_left": float(rng.integers(0, 2)),
                    "league_mean": 4.0 + 0.1 * (season - 2022),
                    "shrunk_realized_persistence": 4.0 + 0.35 * (realized - 4.0),
                    "shrunk_deserved_persistence": 4.0 + 0.60 * (deserved - 4.0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Identifiability
# --------------------------------------------------------------------------


def test_the_surprise_columns_are_dropped_as_exact_dependencies() -> None:
    table = make_focal_table()
    retained, drops = rg.resolve_features(rg.FEATURE_SETS["C_results_plus_deserved"], table)
    assert "std_surprise_rv_per_100" not in retained
    assert "recent_surprise_rv_per_100" not in retained
    dropped = {d["feature"]: d for d in drops}
    assert dropped["std_surprise_rv_per_100"]["kind"] == "prespecified_exact_dependency"
    assert "no information is lost" in dropped["std_surprise_rv_per_100"]["reason"]


def test_the_surprise_identity_that_justifies_the_drop_actually_holds() -> None:
    """The drop is only sound because the identity is exact."""
    table = make_focal_table()
    residual = (
        table["std_realized_rv_per_100"]
        - table["std_deserved_rv_per_100"]
        - table["std_surprise_rv_per_100"]
    )
    assert float(np.abs(residual).max()) < 1e-12


def test_the_batted_ball_reference_category_is_dropped() -> None:
    table = make_focal_table()
    retained, drops = rg.resolve_features(rg.FEATURE_SETS["D_full_contact_profile"], table)
    assert "bb_rate_popup" not in retained
    assert "bb_rate_ground_ball" in retained
    reason = next(d for d in drops if d["feature"] == "bb_rate_popup")["reason"]
    assert "sum to exactly 1" in reason


def test_a_constant_feature_is_dropped_with_a_recorded_reason() -> None:
    """BBE count is constant at a fixed cutoff and cannot be standardized."""
    table = make_focal_table()
    _, drops = rg.resolve_features(rg.FEATURE_SETS["D_full_contact_profile"], table)
    zero_variance = [d for d in drops if d["kind"] == "zero_variance"]
    assert [d["feature"] for d in zero_variance] == ["n_resolved_bbe_through_cutoff"]


def test_every_declared_model_yields_a_full_rank_design() -> None:
    table = make_focal_table()
    for model_name, nominal in rg.FEATURE_SETS.items():
        retained, _ = rg.resolve_features(nominal, table)
        rg.assert_design_is_identifiable(table, retained)
        audit = rg.collinearity_audit(table, retained)
        assert audit["rank"] == audit["n_features"], model_name


def test_a_surviving_exact_dependency_fails_the_fit() -> None:
    """The declared drops are a claim; this proves the check is real."""
    table = make_focal_table()
    table["duplicate_of_deserved"] = table["std_deserved_rv_per_100"]
    with pytest.raises(rg.ForecastRidgeError, match="rank-deficient"):
        rg.fit_ridge(
            table,
            model_name="B_deserved_only",
            alpha=1.0,
            nominal_features=(
                "std_deserved_rv_per_100",
                "recent_deserved_rv_per_100",
                "duplicate_of_deserved",
            ),
        )


def test_the_collinearity_audit_reports_vif_and_condition_number() -> None:
    table = make_focal_table()
    retained, _ = rg.resolve_features(rg.FEATURE_SETS["D_full_contact_profile"], table)
    audit = rg.collinearity_audit(table, retained)
    assert set(audit["variance_inflation_factors"]) == set(retained)
    assert audit["condition_number"] > 1.0
    assert audit["max_vif"] >= 1.0


# --------------------------------------------------------------------------
# Causality: mutation tests
# --------------------------------------------------------------------------


def _mutate(frame: pd.DataFrame, mask: pd.Series, *, seed: int = 99) -> pd.DataFrame:
    """Overwrite every numeric column of the masked rows with garbage."""
    rng = np.random.default_rng(seed)
    mutated = frame.copy()
    numeric = [
        c
        for c in mutated.columns
        if pd.api.types.is_numeric_dtype(mutated[c])
        and c not in {"batter", "season", "cutoff", "horizon"}
    ]
    for column in numeric:
        mutated.loc[mask, column] = rng.normal(500.0, 250.0, int(mask.sum()))
    return mutated


def test_mutating_the_evaluation_season_cannot_move_the_fitted_model() -> None:
    table = make_focal_table()
    train = table[table["season"].isin([2022, 2023])]

    baseline = rg.fit_ridge(train, model_name="D_full_contact_profile", alpha=10.0)
    mutated_table = _mutate(table, table["season"] == 2024)
    mutated_train = mutated_table[mutated_table["season"].isin([2022, 2023])]
    after = rg.fit_ridge(mutated_train, model_name="D_full_contact_profile", alpha=10.0)

    assert after.scaler_mean == baseline.scaler_mean
    assert after.scaler_scale == baseline.scaler_scale
    assert after.coefficients == baseline.coefficients
    assert after.intercept == baseline.intercept


def test_mutating_the_evaluation_season_targets_cannot_move_the_selected_alpha() -> None:
    table = make_focal_table()
    train = table[table["season"] == 2022]
    validate = table[table["season"] == 2023]

    baseline = rg.select_alpha(train, validate, model_name="C_results_plus_deserved")
    mutated = table.copy()
    mutated.loc[mutated["season"] == 2024, TARGET] = -999.0
    after = rg.select_alpha(
        mutated[mutated["season"] == 2022],
        mutated[mutated["season"] == 2023],
        model_name="C_results_plus_deserved",
    )
    assert after["selected_alpha"] == baseline["selected_alpha"]
    assert after["grid"] == baseline["grid"]


def test_mutating_a_held_out_season_cannot_move_anything() -> None:
    """A season in neither the training nor the validation fold is inert."""
    table = make_focal_table(seasons=(2022, 2023, 2024))
    train = table[table["season"] == 2022]
    validate = table[table["season"] == 2023]
    baseline = rg.select_alpha(train, validate, model_name="A_results_only")
    baseline_fit = rg.fit_ridge(
        train, model_name="A_results_only", alpha=baseline["selected_alpha"]
    )

    mutated = _mutate(table, table["season"] == 2024, seed=7)
    after = rg.select_alpha(
        mutated[mutated["season"] == 2022],
        mutated[mutated["season"] == 2023],
        model_name="A_results_only",
    )
    after_fit = rg.fit_ridge(
        mutated[mutated["season"] == 2022],
        model_name="A_results_only",
        alpha=after["selected_alpha"],
    )
    assert after["selected_alpha"] == baseline["selected_alpha"]
    assert after_fit.coefficients == baseline_fit.coefficients
    assert after_fit.scaler_mean == baseline_fit.scaler_mean


def test_mutating_the_training_season_DOES_move_the_model() -> None:
    """The mutation tests would be vacuous if nothing ever moved the fit."""
    table = make_focal_table()
    train = table[table["season"].isin([2022, 2023])]
    baseline = rg.fit_ridge(train, model_name="B_deserved_only", alpha=10.0)

    mutated = _mutate(table, table["season"] == 2023)
    after = rg.fit_ridge(
        mutated[mutated["season"].isin([2022, 2023])],
        model_name="B_deserved_only",
        alpha=10.0,
    )
    assert after.coefficients != baseline.coefficients


def test_training_on_the_evaluation_season_is_refused() -> None:
    table = make_focal_table()
    with pytest.raises(rg.ForecastRidgeError, match="strictly earlier"):
        rg.assert_no_later_season_rows(table[table["season"].isin([2022, 2024])], 2024)


def test_alpha_selection_refuses_a_non_forward_split() -> None:
    table = make_focal_table()
    with pytest.raises(rg.ForecastRidgeError, match="strictly earlier"):
        rg.select_alpha(
            table[table["season"] == 2024],
            table[table["season"] == 2023],
            model_name="A_results_only",
        )


def test_alpha_selection_refuses_more_than_one_validation_season() -> None:
    table = make_focal_table()
    with pytest.raises(rg.ForecastRidgeError, match="exactly one validation season"):
        rg.select_alpha(
            table[table["season"] == 2022],
            table[table["season"].isin([2023, 2024])],
            model_name="A_results_only",
        )


# --------------------------------------------------------------------------
# Fitting and prediction
# --------------------------------------------------------------------------


def test_prediction_uses_the_training_standardization_not_a_refit() -> None:
    """Refitting the scaler on the evaluation frame would leak its distribution."""
    table = make_focal_table()
    train = table[table["season"] == 2022]
    evaluate = table[table["season"] == 2024].copy()
    fit = rg.fit_ridge(train, model_name="B_deserved_only", alpha=1.0)

    baseline = rg.predict(fit, evaluate)
    # Shift the evaluation frame's location. A model that restandardized on
    # these rows would absorb the shift; one using training statistics moves.
    shifted = evaluate.copy()
    for column in fit.features:
        shifted[column] = shifted[column] + 100.0
    assert not np.allclose(baseline, rg.predict(fit, shifted))


def test_ridge_recovers_a_known_linear_signal() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "season": 2022,
            "std_deserved_rv_per_100": x,
            "recent_deserved_rv_per_100": rng.normal(0.0, 1.0, n),
            TARGET: 3.0 + 2.0 * x + rng.normal(0.0, 0.1, n),
        }
    )
    fit = rg.fit_ridge(frame, model_name="B_deserved_only", alpha=1e-6)
    # Standardized coefficient ~ slope * SD(x).
    assert fit.coefficients["std_deserved_rv_per_100"] == pytest.approx(
        2.0 * float(np.std(x)), rel=0.05
    )
    assert fit.intercept == pytest.approx(float(frame[TARGET].mean()), rel=0.05)


def test_a_larger_alpha_shrinks_the_coefficients() -> None:
    table = make_focal_table()
    train = table[table["season"] == 2022]
    small = rg.fit_ridge(train, model_name="D_full_contact_profile", alpha=0.01)
    large = rg.fit_ridge(train, model_name="D_full_contact_profile", alpha=10000.0)
    assert sum(abs(v) for v in large.coefficients.values()) < sum(
        abs(v) for v in small.coefficients.values()
    )


def test_the_selection_metric_is_fixed_in_advance() -> None:
    assert rg.SELECTION_METRIC == "mae"
    assert rg.ALPHA_GRID[0] < rg.ALPHA_GRID[-1]
    assert len(rg.ALPHA_GRID) >= 10


def test_the_fit_record_carries_the_coefficient_caveat() -> None:
    table = make_focal_table()
    fit = rg.fit_ridge(table[table["season"] == 2022], model_name="A_results_only", alpha=1.0)
    record = fit.as_record()
    assert "not causal importances" in record["coefficient_caveat"]
    assert record["standardization"]["fit_on"] == "training seasons only"
    assert record["train_seasons"] == [2022]


# --------------------------------------------------------------------------
# The experiment end to end
# --------------------------------------------------------------------------


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    make_focal_table().to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    return tmp_path


def test_the_experiment_runs_season_forward_and_evaluates_on_2024(
    experiment_dir: Path,
) -> None:
    results = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    assert results["fold_1_validation"]["train_seasons"] == [2022]
    assert results["fold_1_validation"]["evaluate_season"] == 2023
    assert results["fold_2_evaluation"]["train_seasons"] == [2022, 2023]
    assert results["fold_2_evaluation"]["evaluate_season"] == 2024
    assert results["fold_2_evaluation"]["causality"]["confirmed"] is True


def test_every_prespecified_ablation_is_evaluated_and_reported(
    experiment_dir: Path,
) -> None:
    results = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    for model_name in rg.FEATURE_SETS:
        assert f"ridge_{model_name}" in results["fold_2_evaluation"]["metrics"]
        assert model_name in results["selected_alphas"]
        assert model_name in results["diagnostics"]["collinearity_by_model"]


def test_the_key_delta_is_measured_against_shrunk_deserved(
    experiment_dir: Path,
) -> None:
    results = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    key = results["key_delta"]
    assert "shrunk_deserved_persistence" in key["definition"]
    assert key["delta_mae"] == pytest.approx(key["mae_best_ridge"] - key["mae_shrunk_deserved"])
    assert key["evaluation_season"] == 2024


def test_model_selection_uses_no_evaluation_season_information(
    experiment_dir: Path,
) -> None:
    """Rewriting 2024 must not change which model is selected, nor any alpha."""
    baseline = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    table = pd.read_parquet(experiment_dir / "r1_prediction_table.parquet")
    _mutate(table, table["season"] == 2024).to_parquet(
        experiment_dir / "r1_prediction_table.parquet", index=False
    )
    after = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    assert after["selected_model"]["model"] == baseline["selected_model"]["model"]
    assert after["selected_alphas"] == baseline["selected_alphas"]
    assert (
        after["fold_2_evaluation"]["fits"]["D_full_contact_profile"]["standardized_coefficients"]
        == baseline["fold_2_evaluation"]["fits"]["D_full_contact_profile"][
            "standardized_coefficients"
        ]
    )


def test_the_experiment_records_that_clustering_is_a_no_op_here(
    experiment_dir: Path,
) -> None:
    results = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    no_op = results["diagnostics"]["clustering_is_a_no_op_here"]
    assert no_op["one_window_per_hitter"] is True
    assert "coincides with an ordinary" in no_op["consequence"]
    metrics = results["fold_2_evaluation"]["metrics"]["ridge_B_deserved_only"]
    assert metrics["batter_balanced_mae"] == pytest.approx(metrics["mae"])


def test_a_sealed_season_in_the_table_is_refused(tmp_path: Path) -> None:
    table = make_focal_table()
    table.loc[table.index[:10], "season"] = 2025
    table.to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(ForecastSeasonError, match="sealed"):
        runner.load_focal_table(outputs_dir=tmp_path, verify_freeze_hash=False)


def test_a_phase_two_season_in_the_table_is_refused(tmp_path: Path) -> None:
    table = make_focal_table()
    table.loc[table.index[:10], "season"] = 2026
    table.to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(ForecastSeasonError, match="Phase 2"):
        runner.load_focal_table(outputs_dir=tmp_path, verify_freeze_hash=False)


def test_a_drifted_r1_package_is_refused(tmp_path: Path) -> None:
    """Ridge results built on a drifted R1 package are not comparable."""
    make_focal_table().to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    (tmp_path / "r1_freeze_manifest.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {"r1_prediction_table.parquet": "0" * 64},
                "manifest_sha256": "x",
                "conclusion": {"verdict": "PROCEED"},
            }
        )
    )
    with pytest.raises(runner.ForecastRidgeRunError, match="drifted from its freeze"):
        runner.load_focal_table(outputs_dir=tmp_path, verify_freeze_hash=True)


def test_running_without_a_freeze_is_refused(tmp_path: Path) -> None:
    make_focal_table().to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(runner.ForecastRidgeRunError, match="must be frozen"):
        runner.load_focal_table(outputs_dir=tmp_path, verify_freeze_hash=True)


def test_the_report_renders_every_model_including_losers(experiment_dir: Path) -> None:
    from forecast.ridge_report import render_ridge_report

    results = runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    report = render_ridge_report(results)
    for model_name in rg.FEATURE_SETS:
        assert f"`ridge_{model_name}`" in report
    assert "ridge (regularized linear) only" in report
    assert "not causal importances" in report
    # Deltas are printed as stored, under the inherited convention.
    for benchmark, entries in results["deltas_on_evaluation_season"].items():
        for record in entries.values():
            assert f"| {record['delta']['delta_mae']:.4f} |" in report, benchmark
