"""Contact Forecast: the HistGradientBoosting stage.

What is attacked here:

**Anti-overfitting.** The candidate grid is fixed and recorded before any
evaluation; the feature list is inherited verbatim from the frozen ridge Model
D rather than chosen; and no 2024 information may reach hyperparameter
selection, the preprocessing, or the fitted 2022-2023 model. The mutation
tests rewrite 2024 with garbage and assert every fitted quantity is identical.

**Residual causality.** The shrunk-deserved baseline the residual target is
built against must itself have been fit on strictly earlier seasons, and rows
without a causal baseline must be dropped rather than imputed.

**The preference rule.** A numerically lower point estimate must not be enough
to call HGB superior.

Synthetic data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.test_forecast_ridge import make_focal_table

from forecast import hgb
from forecast import run_hgb_forecast as runner
from forecast.forecast_config import ForecastSeasonError
from forecast.ridge import FEATURE_SETS


def make_table(**kwargs: object) -> pd.DataFrame:
    """The ridge stage's focal table, with 2022 lacking a causal baseline."""
    table = make_focal_table(**kwargs)  # type: ignore[arg-type]
    # Mirror the real R1 package: 2022 has no causal shrunk-deserved value.
    table.loc[table["season"] == 2022, "shrunk_deserved_persistence"] = np.nan
    return table


# --------------------------------------------------------------------------
# One family, one grid, fixed in advance
# --------------------------------------------------------------------------


def test_only_one_nonlinear_family_is_implemented() -> None:
    record = hgb.grid_record()
    assert record["model_family"].endswith("HistGradientBoostingRegressor")
    assert record["only_nonlinear_family_implemented"] is True


def test_the_grid_is_compact_and_covers_only_complexity_controls() -> None:
    searched = set(hgb.HGB_PARAMETER_GRID)
    assert searched <= {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    }
    candidates = hgb.build_candidate_grid()
    assert len(candidates) == 16
    assert len({json.dumps(c, sort_keys=True) for c in candidates}) == 16


def test_the_grid_record_states_that_it_is_never_expanded() -> None:
    record = hgb.grid_record()
    assert "never expanded" in record["expansion_rule"]
    assert record["n_candidates"] == len(record["candidates"])
    assert record["selection_metric"] == "mae"


def test_pinned_parameters_carry_their_rationale() -> None:
    record = hgb.grid_record()
    assert record["fixed_parameters"]["min_samples_leaf"] == 20
    assert record["fixed_parameters"]["early_stopping"] is False
    assert "sample-size argument" in record["fixed_parameter_rationale"]["min_samples_leaf"]


def test_every_candidate_is_deterministic() -> None:
    for candidate in hgb.build_candidate_grid():
        assert candidate["random_state"] is not None
        assert candidate["early_stopping"] is False


# --------------------------------------------------------------------------
# The feature set is inherited, not chosen
# --------------------------------------------------------------------------


def test_the_feature_set_is_the_frozen_ridge_model_d() -> None:
    assert hgb.HGB_FEATURE_SET_NAME == "D_full_contact_profile"
    table = make_table()
    features, _ = hgb.hgb_features(table)
    from forecast.ridge import resolve_features

    ridge_features, _ = resolve_features(FEATURE_SETS["D_full_contact_profile"], table)
    assert features == ridge_features


def test_no_feature_subset_was_taken_from_large_ridge_coefficients() -> None:
    """The full resolved Model D list is used, not a coefficient-ranked subset."""
    from forecast.ridge import PRESPECIFIED_DROPS

    table = make_table()
    features, drops = hgb.hgb_features(table)
    declared = list(FEATURE_SETS["D_full_contact_profile"])
    dropped = {d["feature"] for d in drops}

    # The ONLY omissions are the declared exact dependencies plus the constant
    # BBE count -- nothing was pruned on the basis of a 2024 ridge coefficient.
    assert dropped == set(PRESPECIFIED_DROPS) | {"n_resolved_bbe_through_cutoff"}
    assert set(features) == set(declared) - dropped
    # Features carrying SMALL 2024 ridge coefficients are still present, which
    # is the point: coefficients were not used as a selection procedure.
    for retained in ("spray_angle_sd", "bb_rate_line_drive", "bats_left", "launch_speed_p10"):
        assert retained in features


# --------------------------------------------------------------------------
# Residual construction and its causality
# --------------------------------------------------------------------------


def test_the_residual_target_is_target_minus_the_causal_baseline() -> None:
    table = make_table()
    built = hgb.build_residual_target(table[table["season"] == 2024])
    assert np.allclose(
        built[hgb.RESIDUAL_TARGET],
        built[hgb.HGB_TARGET] - built[hgb.RESIDUAL_BASELINE],
    )


def test_rows_without_a_causal_baseline_are_dropped_not_imputed() -> None:
    table = make_table()
    built = hgb.build_residual_target(table)
    assert 2022 not in set(built["season"])
    assert built[hgb.RESIDUAL_BASELINE].notna().all()
    assert len(built) == int((table["season"] != 2022).sum())


def test_the_residual_baseline_causality_is_verified_from_the_frozen_artifact() -> None:
    artifact = {
        "assembly_audit": {
            "ladders": {
                "2022": {
                    "deserved_shrinkage": None,
                    "rung_availability": {"reason": "no causal contact model for 2021"},
                },
                "2023": {"deserved_shrinkage": {"fit_seasons": [2022]}},
                "2024": {"deserved_shrinkage": {"fit_seasons": [2022, 2023]}},
            }
        }
    }
    record = hgb.assert_residual_baseline_is_causal(artifact)
    assert record["by_season"]["2023"]["strictly_earlier"] is True
    assert record["by_season"]["2024"]["fit_seasons"] == [2022, 2023]
    assert record["by_season"]["2022"]["available"] is False


def test_a_non_causal_residual_baseline_is_refused() -> None:
    artifact = {
        "assembly_audit": {
            "ladders": {"2024": {"deserved_shrinkage": {"fit_seasons": [2022, 2024]}}}
        }
    }
    with pytest.raises(hgb.ForecastHGBError, match="not strictly earlier"):
        hgb.assert_residual_baseline_is_causal(artifact)


# --------------------------------------------------------------------------
# Causality: mutation tests
# --------------------------------------------------------------------------


def _mutate(frame: pd.DataFrame, mask: pd.Series, *, seed: int = 5) -> pd.DataFrame:
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
        mutated.loc[mask, column] = rng.normal(400.0, 200.0, int(mask.sum()))
    return mutated


def test_mutating_2024_cannot_alter_hyperparameter_selection() -> None:
    table = make_table()
    baseline = hgb.select_hyperparameters(
        table[table["season"] == 2022], table[table["season"] == 2023]
    )
    mutated = _mutate(table, table["season"] == 2024)
    after = hgb.select_hyperparameters(
        mutated[mutated["season"] == 2022], mutated[mutated["season"] == 2023]
    )
    assert after["selected_parameters"] == baseline["selected_parameters"]
    assert after["grid_scores"] == baseline["grid_scores"]


def test_mutating_2024_targets_cannot_alter_the_fitted_2022_2023_model() -> None:
    table = make_table()
    train = table[table["season"].isin([2022, 2023])]
    parameters = hgb.build_candidate_grid()[0]
    baseline = hgb.fit_hgb(train, formulation="direct", parameters=parameters)

    mutated = table.copy()
    mutated.loc[mutated["season"] == 2024, hgb.HGB_TARGET] = -999.0
    after = hgb.fit_hgb(
        mutated[mutated["season"].isin([2022, 2023])],
        formulation="direct",
        parameters=parameters,
    )
    probe = train.head(50)
    assert np.array_equal(hgb.predict(baseline, probe), hgb.predict(after, probe))
    assert baseline.features == after.features
    assert baseline.model.n_iter_ == after.model.n_iter_


def test_mutating_2024_features_cannot_alter_the_fitted_model() -> None:
    table = make_table()
    parameters = hgb.build_candidate_grid()[3]
    train = table[table["season"].isin([2022, 2023])]
    baseline = hgb.fit_hgb(train, formulation="direct", parameters=parameters)

    mutated = _mutate(table, table["season"] == 2024, seed=11)
    after = hgb.fit_hgb(
        mutated[mutated["season"].isin([2022, 2023])],
        formulation="direct",
        parameters=parameters,
    )
    probe = train.head(50)
    assert np.array_equal(hgb.predict(baseline, probe), hgb.predict(after, probe))


def test_mutating_2024_cannot_alter_the_residual_arm() -> None:
    table = make_table()
    parameters = hgb.build_candidate_grid()[0]
    residual_train = hgb.build_residual_target(table[table["season"] == 2023])
    baseline = hgb.fit_hgb(
        residual_train,
        formulation="residual",
        parameters=parameters,
        target_column=hgb.RESIDUAL_TARGET,
    )
    mutated = _mutate(table, table["season"] == 2024, seed=13)
    after = hgb.fit_hgb(
        hgb.build_residual_target(mutated[mutated["season"] == 2023]),
        formulation="residual",
        parameters=parameters,
        target_column=hgb.RESIDUAL_TARGET,
    )
    probe = residual_train.head(40)
    assert np.array_equal(hgb.predict(baseline, probe), hgb.predict(after, probe))


def test_mutating_the_training_season_DOES_move_the_model() -> None:
    """The mutation tests would be vacuous if nothing ever moved the fit."""
    table = make_table()
    parameters = hgb.build_candidate_grid()[0]
    train = table[table["season"].isin([2022, 2023])]
    baseline = hgb.fit_hgb(train, formulation="direct", parameters=parameters)
    mutated = _mutate(table, table["season"] == 2023, seed=17)
    after = hgb.fit_hgb(
        mutated[mutated["season"].isin([2022, 2023])],
        formulation="direct",
        parameters=parameters,
    )
    probe = table[table["season"] == 2024].head(40)
    assert not np.array_equal(hgb.predict(baseline, probe), hgb.predict(after, probe))


def test_training_on_the_evaluation_season_is_refused() -> None:
    table = make_table()
    with pytest.raises(hgb.ForecastHGBError, match="strictly earlier"):
        hgb.assert_no_later_season_rows(table[table["season"].isin([2022, 2024])], 2024)


def test_selection_refuses_a_non_forward_split() -> None:
    table = make_table()
    with pytest.raises(hgb.ForecastHGBError, match="strictly earlier"):
        hgb.select_hyperparameters(table[table["season"] == 2024], table[table["season"] == 2023])


def test_there_is_no_fitted_preprocessing_to_leak() -> None:
    table = make_table()
    fit = hgb.fit_hgb(
        table[table["season"] == 2022],
        formulation="direct",
        parameters=hgb.build_candidate_grid()[0],
    )
    assert "no" in fit.as_record()["preprocessing"].lower()


# --------------------------------------------------------------------------
# The preference rule
# --------------------------------------------------------------------------


def _verdict(delta_mae: float, crosses_zero: bool) -> dict[str, object]:
    cell = f"K{100}_H{100}"
    deltas = {
        runner.RIDGE_PREDICTION_COLUMN: {
            runner.HGB_DIRECT_COLUMN: {"delta_mae": delta_mae, "pct_change_mae": 1.0}
        }
    }
    bootstraps = {
        f"{runner.HGB_DIRECT_COLUMN}_vs_{runner.RIDGE_PREDICTION_COLUMN}": {
            "cells": {
                cell: {
                    "pooled": {
                        "delta_mae": {
                            "ci_lower": -1.0,
                            "ci_upper": 1.0,
                            "ci_crosses_zero": crosses_zero,
                        }
                    }
                }
            }
        }
    }
    return runner._nonlinear_value_verdict(deltas, bootstraps)


def test_a_lower_point_estimate_alone_does_not_make_hgb_superior() -> None:
    verdict = _verdict(-0.5, crosses_zero=True)
    assert verdict["point_estimate_favours_hgb"] is True
    assert verdict["interval_excludes_zero"] is False
    assert verdict["preferred_model"] == "ridge"
    assert verdict["answer"].startswith("No")


def test_hgb_is_preferred_only_when_the_interval_also_excludes_zero() -> None:
    verdict = _verdict(-0.5, crosses_zero=False)
    assert verdict["preferred_model"] == "hgb"
    assert verdict["answer"].startswith("Yes")


def test_a_worse_point_estimate_prefers_ridge() -> None:
    assert _verdict(0.3, crosses_zero=False)["preferred_model"] == "ridge"


def test_the_rule_was_declared_before_the_comparison() -> None:
    assert runner.NONLINEAR_VALUE_RULE["declared_before_the_comparison_ran"] is True
    assert "not sufficient" in runner.NONLINEAR_VALUE_RULE["explicitly_insufficient"].lower()


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    """A fresh directory, for tests that mutate the table and re-run."""
    make_table().to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    return tmp_path


@pytest.fixture(scope="module")
def shared_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One end-to-end run, shared by every read-only assertion.

    The experiment fits 16 candidates plus both formulations, so re-running it
    per test dominated the suite's runtime for no added coverage.
    """
    directory = tmp_path_factory.mktemp("hgb_shared")
    make_table().to_parquet(directory / "r1_prediction_table.parquet", index=False)
    return runner.run_experiment(outputs_dir=directory, reps=20, verify_freeze_hash=False)


@pytest.fixture(scope="module")
def shared_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory the shared run wrote its artifacts into."""
    directory = tmp_path_factory.mktemp("hgb_shared_artifacts")
    make_table().to_parquet(directory / "r1_prediction_table.parquet", index=False)
    runner.run_experiment(outputs_dir=directory, reps=20, verify_freeze_hash=False)
    return directory


def test_the_grid_is_written_before_any_evaluation(shared_dir: Path) -> None:
    grid = json.loads((shared_dir / "hgb_candidate_grid.json").read_text())
    assert grid["n_candidates"] == 16
    specification = json.loads((shared_dir / "hgb_specification.json").read_text())
    assert specification["frozen_before_2024_was_touched"] is True
    assert specification["selection"]["evaluation_season_used_in_selection"] is False


def test_the_frozen_specification_does_not_depend_on_2024(experiment_dir: Path) -> None:
    runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    baseline = json.loads((experiment_dir / "hgb_specification.json").read_text())

    table = pd.read_parquet(experiment_dir / "r1_prediction_table.parquet")
    _mutate(table, table["season"] == 2024).to_parquet(
        experiment_dir / "r1_prediction_table.parquet", index=False
    )
    runner.run_experiment(outputs_dir=experiment_dir, reps=20, verify_freeze_hash=False)
    after = json.loads((experiment_dir / "hgb_specification.json").read_text())

    for record in (baseline, after):
        record.pop("frozen_at_utc")
    assert after == baseline


def test_both_formulations_are_evaluated_and_reported(shared_results: dict) -> None:
    results = shared_results
    metrics = results["fold_2_evaluation"]["metrics"]
    assert runner.HGB_DIRECT_COLUMN in metrics
    assert runner.HGB_RESIDUAL_COLUMN in metrics
    for benchmark in runner.BENCHMARKS:
        assert benchmark in metrics


def test_the_residual_arm_trains_only_on_the_season_with_a_causal_baseline(
    shared_results: dict,
) -> None:
    results = shared_results
    fold2 = results["fold_2_evaluation"]
    assert fold2["residual_train_seasons"] == [2023]
    assert fold2["n_train_residual"] < fold2["n_train_direct"]


def test_residual_diagnostics_are_complete(shared_results: dict) -> None:
    results = shared_results
    diagnostics = results["residual_diagnostics"]
    for field in (
        "residual_target_sd",
        "residual_target_mean",
        "mae_predicting_residual_zero",
        "mae_hgb_predicted_residual",
        "correlation_predicted_vs_actual_residual",
        "fraction_of_benchmark_error_recovered",
    ):
        assert field in diagnostics
    # Predicting residual zero IS the benchmark, by construction.
    benchmark_mae = results["fold_2_evaluation"]["metrics"]["shrunk_deserved_persistence"]["mae"]
    assert diagnostics["mae_predicting_residual_zero"] == pytest.approx(benchmark_mae)


def test_stability_reports_tails_and_trimmed_means(shared_results: dict) -> None:
    results = shared_results
    stability = results["stability"]["vs_shrunk_deserved"]
    for field in ("median", "q1", "q3", "p5", "p95", "fraction_of_hitters_improved"):
        assert field in stability
    assert stability["trimmed_means"]
    assert len(stability["top_5_most_improved"]) == 5
    assert len(stability["bottom_5_most_worsened"]) == 5


def test_permutation_importance_is_labelled_exploratory_and_avoids_2024(
    shared_results: dict,
) -> None:
    results = shared_results
    importance = results["permutation_importance"]
    assert importance["status"].startswith("EXPLORATORY")
    assert importance["not_computed_on_2024"] is True
    assert importance["used_to_revise_the_model"] is False
    assert "2023 validation" in importance["computed_on"]


def test_the_experiment_is_deterministic(experiment_dir: Path) -> None:
    first = runner.run_experiment(outputs_dir=experiment_dir, reps=30, verify_freeze_hash=False)
    second = runner.run_experiment(outputs_dir=experiment_dir, reps=30, verify_freeze_hash=False)
    assert first["fold_2_evaluation"]["metrics"] == second["fold_2_evaluation"]["metrics"]
    assert first["nonlinear_value_test"] == second["nonlinear_value_test"]


def test_a_sealed_season_never_reaches_the_estimator(tmp_path: Path) -> None:
    table = make_table()
    table.loc[table.index[:10], "season"] = 2025
    table.to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(ForecastSeasonError, match="sealed"):
        runner.run_experiment(outputs_dir=tmp_path, reps=10, verify_freeze_hash=False)


def test_a_phase_two_season_never_reaches_the_estimator(tmp_path: Path) -> None:
    table = make_table()
    table.loc[table.index[:10], "season"] = 2026
    table.to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(ForecastSeasonError, match="Phase 2"):
        runner.run_experiment(outputs_dir=tmp_path, reps=10, verify_freeze_hash=False)


def test_running_without_a_sealed_ridge_stage_is_refused(tmp_path: Path) -> None:
    make_table().to_parquet(tmp_path / "r1_prediction_table.parquet", index=False)
    with pytest.raises(runner.ForecastHGBRunError, match="must be\\s+sealed|sealed before"):
        runner._verify_ridge_freeze(tmp_path)


def test_the_report_renders_both_formulations_and_the_verdict(
    shared_results: dict,
) -> None:
    from forecast.hgb_report import render_hgb_report

    results = shared_results
    report = render_hgb_report(results)
    assert "`hgb_direct`" in report
    assert "`hgb_residual`" in report
    assert "EXPLORATORY" in report
    assert results["nonlinear_value_test"]["answer"] in report
    for benchmark, entries in results["deltas_on_evaluation_season"].items():
        for delta in entries.values():
            assert f"| {delta['delta_mae']:.4f} |" in report, benchmark
