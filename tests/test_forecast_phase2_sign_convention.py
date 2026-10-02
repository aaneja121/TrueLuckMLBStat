"""Every Phase 2 report derives its sign convention from its own comparison.

The H=200 report originally displayed `forecast.metrics.DELTA_SIGN_CONVENTION`
verbatim. That shared R1 constant describes a DIFFERENT pair -- shrunk deserved
measured against shrunk realized -- so the block named the wrong challenger and
reference beside a delta computed from the right one. The H=100 report named the
right pair, but as hardcoded prose that no artifact could contradict.

These tests pin the property that fixes both: the displayed names are parsed out
of the comparison record's own `definition`, so a report cannot print a
convention that disagrees with the delta beside it. The synthetic definitions
below deliberately name pairs that appear nowhere in the codebase -- prose
copied from the shared constant, or hardcoded, could not reproduce them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from forecast.metrics import DELTA_SIGN_CONVENTION
from forecast.phase2.h200_disclosure import render_disclosure_markdown
from forecast.phase2.h200_spec import H200_OUTPUTS_DIR
from forecast.phase2.phase2_config import PHASE2_OUTPUTS_DIR
from forecast.phase2.phase2_report import (
    Phase2ReportError,
    parse_comparison_definition,
    render_phase2_report,
    render_sign_convention,
)
from forecast.phase2.run_h200_report import render_h200_report

#: A pair that exists nowhere in the codebase, so only derivation can print it.
SYNTHETIC_DEFINITION = "MAE(a_challenger_model) - MAE(a_reference_model)"


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


def test_the_names_come_from_the_definition() -> None:
    challenger, reference = parse_comparison_definition({"definition": SYNTHETIC_DEFINITION})
    assert challenger == "a_challenger_model"
    assert reference == "a_reference_model"


def test_the_block_names_the_pair_the_definition_names() -> None:
    block = "\n".join(render_sign_convention({"definition": SYNTHETIC_DEFINITION}))
    assert "delta_MAE = MAE(a_challenger_model) - MAE(a_reference_model)" in block
    assert "negative -> a_challenger_model better" in block
    assert "positive -> a_reference_model better" in block
    assert "zero     -> tied" in block


def test_a_report_may_not_display_a_convention_it_cannot_derive() -> None:
    with pytest.raises(Phase2ReportError, match="no `definition`"):
        render_sign_convention({})
    with pytest.raises(Phase2ReportError, match="not of the form"):
        render_sign_convention({"definition": "lower is better"})


def test_the_shared_r1_constant_describes_a_different_pair() -> None:
    """Why deriving matters: the shared constant is about the ladder, not this."""
    assert DELTA_SIGN_CONVENTION["challenger"] == "shrunk_deserved_persistence"
    assert DELTA_SIGN_CONVENTION["reference"] == "shrunk_realized_persistence"
    assert "contact_forecast" not in json.dumps(DELTA_SIGN_CONVENTION)


# --------------------------------------------------------------------------
# Both renderers, driven by a definition no hardcoded prose could produce
# --------------------------------------------------------------------------


def _metrics(definition: str) -> dict[str, Any]:
    predictor = {
        "mae": 1.0,
        "rmse": 2.0,
        "r_squared": 0.5,
        "pearson": 0.5,
        "pearson_status": "defined",
        "spearman": 0.5,
        "spearman_status": "defined",
    }
    return {
        "evaluation_season": 2026,
        "n_completed_windows": 3,
        "n_unique_hitters": 3,
        "one_window_per_hitter": True,
        "clustering_note": "note",
        # Present, and deliberately NOT what the report should display.
        "sign_convention": DELTA_SIGN_CONVENTION,
        "by_predictor": {"contact_forecast": predictor, "league_mean": predictor},
        "contact_forecast_vs_shrunk_deserved": {
            "definition": definition,
            "delta_mae": -0.1,
            "pct_change_mae": -1.0,
            "delta_rmse": -0.1,
            "pct_change_rmse": -1.0,
            "mae_contact_forecast": 1.0,
            "mae_shrunk_deserved": 1.1,
            "rmse_contact_forecast": 2.0,
            "rmse_shrunk_deserved": 2.1,
            "direction_mae": "challenger_better",
            "ci_delta_mae": [-0.3, 0.1],
            "ci_delta_rmse": [-0.3, 0.1],
            "ci_crosses_zero": True,
            "share_replicates_favouring_shrunk_deserved": 0.3,
            "share_replicates_favouring_contact_forecast": 0.7,
        },
        "prespecified_classification": {
            "classification": "promising_but_inconclusive",
            "condition": "condition",
            "meaning": "meaning",
        },
        "forecast_bias_diagnostics": {
            "mean_prediction": 1.0,
            "mean_realized_target": 1.0,
            "mean_prediction_error": 0.0,
            "median_prediction_error": 0.0,
            "sd_prediction": 1.0,
            "sd_realized_target": 1.0,
            "calibration_regression": {
                "model": "realized ~ intercept + slope * prediction",
                "slope": 1.0,
                "intercept": 0.0,
                "perfect_calibration_would_be": {"slope": 1.0, "intercept": 0.0},
            },
            "status": "DIAGNOSTIC ONLY",
        },
        "pending_forecasts": {"n_pending": 1, "note": "note"},
        "provenance": {
            "authorization_sha256": "0" * 64,
            "frozen_stage_manifests": {"r1": "a" * 64, "ridge": "b" * 64, "hgb": "c" * 64},
            "h200_spec_manifest_sha256": "d" * 64,
            "snapshot_label": "2026-09-01",
            "snapshot_data_through": "2026-09-01",
            "frozen_contract": {
                "spec_version": "v1",
                "model": "D_full_contact_profile",
                "alpha": 1.0,
                "features": ["f"],
                "trained_on_seasons": [2022],
                "n_train": 1,
                "target_definition": "target",
                "benchmarks": ["league_mean"],
                "primary_benchmark": "shrunk_deserved_persistence",
                "primary_metric": "MAE",
                "model_choice_reasons": {
                    "best_frozen_2024_mae": 1.0,
                    "better_than_shrunk_deserved": 1.1,
                    "hgb_vs_ridge_point_estimate_favoured_ridge": 0.01,
                    "prespecified_nonlinear_rule_retained_ridge": "ridge",
                },
                "declared_features": ["f"],
                "stage_verdicts": {"r1": "v", "ridge": "v", "hgb": "v"},
            },
        },
    }


def _snapshot_manifest() -> dict[str, Any]:
    return {
        "pinned_snapshot": {
            "snapshot_label": "2026-09-01",
            "data_through_date": "2026-09-01",
            "integrity_hashes_verified": ["public_score.parquet"],
        },
        "single_snapshot_used_for_entire_evaluation": True,
        "refreshed_during_analysis": False,
        "counts": {
            "n_events_2026": 1,
            "n_resolved_events_2026": 1,
            "n_hitters_reaching_100_resolved_bbe": 1,
            "n_hitters_reaching_200_resolved_bbe": 1,
            "n_hitters_reaching_300_resolved_bbe": 1,
            "n_evaluable_completed_100_to_100_windows": 1,
            "n_evaluable_completed_100_to_200_windows": 1,
            "n_forecasts_with_pending_outcomes": 1,
        },
    }


def _stability() -> dict[str, Any]:
    return {
        "quantity": "q",
        "sign_meaning": "s",
        "n_hitters": 3,
        "fraction_of_hitters_improved": 0.5,
        "mean": 0.0,
        "median": 0.0,
        "p10": 0.0,
        "q1": 0.0,
        "q3": 0.0,
        "p90": 0.0,
        "five_largest_improvements": [],
        "five_largest_deteriorations": [],
        "outlier_sensitivity": {
            "delta_mae_excluding_5_most_improved": 0.0,
            "delta_mae_excluding_10_most_improved": 0.0,
            "status": "SENSITIVITY ONLY",
        },
    }


def _shift() -> dict[str, Any]:
    return {
        "status": "EXPLANATORY ONLY",
        "material_shift_threshold_standardized": 0.25,
        "n_features": 1,
        "features_with_material_shift": [],
        "n_features_with_material_shift": 0,
        "by_feature": [],
    }


def _survivorship() -> dict[str, Any]:
    group = {
        "n_reaching_100_resolved_bbe": 1,
        "n_subsequently_reaching_300_total_resolved_bbe": 1,
        "completion_rate": 1.0,
        "largest_standardized_differences": [],
    }
    return {
        "status": "DESCRIPTIVE ONLY",
        "development_2022_2024_pooled": group,
        "prospective_2026": {
            **group,
            "snapshot_label": "2026-09-01",
            "snapshot_data_through": "2026-09-01",
        },
        "completion_rate_gap_2026_minus_development": 0.0,
        "cutoff_side_differences_side_by_side": [],
        "generalization_note": "note",
    }


def _authorization() -> dict[str, Any]:
    return {
        "authorization": {
            "scope": "scope",
            "not_authorized": ["tuning"],
            "evaluation_seasons_authorized": [2026],
            "sealed_seasons_still_forbidden": [2025],
            "permitted_training_seasons": [2022, 2023, 2024],
        },
        "conditional_forecast": "conditional on the hitter reaching the horizon",
        "record_sha256": "e" * 64,
        "freeze_chain_verification": {
            "manifest_sha256": {"r1": "a" * 64, "ridge": "b" * 64, "hgb": "c" * 64}
        },
    }


def render_h100(definition: str) -> str:
    return render_phase2_report(
        authorization=_authorization(),
        snapshot_manifest=_snapshot_manifest(),
        metrics=_metrics(definition),
        stability=_stability(),
        shift=_shift(),
    )


def render_h200(definition: str) -> str:
    return render_h200_report(
        authorization=_authorization(),
        snapshot_manifest=_snapshot_manifest(),
        metrics=_metrics(definition),
        stability=_stability(),
        shift=_shift(),
        survivorship=_survivorship(),
        disclosure_markdown=render_disclosure_markdown(),
    )


@pytest.mark.parametrize("render", [render_h100, render_h200], ids=["h100", "h200"])
def test_every_report_derives_its_sign_convention_from_its_own_comparison(
    render: Any,
) -> None:
    report = render(SYNTHETIC_DEFINITION)
    assert "delta_MAE = MAE(a_challenger_model) - MAE(a_reference_model)" in report
    assert "negative -> a_challenger_model better" in report
    assert "positive -> a_reference_model better" in report


@pytest.mark.parametrize("render", [render_h100, render_h200], ids=["h100", "h200"])
def test_no_report_displays_the_shared_r1_constant(render: Any) -> None:
    """The constant is in the metrics payload; no report may put it on the page."""
    report = render(SYNTHETIC_DEFINITION)
    assert "delta_metric = metric(challenger) - metric(reference)" not in report
    assert "'reference': 'shrunk_realized_persistence'" not in report
    assert "deserved is better (lower error)" not in report


@pytest.mark.parametrize("render", [render_h100, render_h200], ids=["h100", "h200"])
def test_a_report_refuses_a_convention_it_cannot_derive(render: Any) -> None:
    with pytest.raises(Phase2ReportError):
        render("lower is better")


# --------------------------------------------------------------------------
# The written reports
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outputs_dir", "name"),
    [
        (PHASE2_OUTPUTS_DIR, "phase2_2026_report.md"),
        (H200_OUTPUTS_DIR, "h200_2026_report.md"),
    ],
    ids=["h100", "h200"],
)
def test_the_written_report_names_the_actual_comparison(outputs_dir: Any, name: str) -> None:
    path = outputs_dir / name
    if not path.exists():
        pytest.skip(f"{name} has not been generated in this environment")
    report = path.read_text()
    assert "delta_MAE = MAE(contact_forecast) - MAE(shrunk_deserved_persistence)" in report
    assert "negative -> contact_forecast better" in report
    assert "delta_metric = metric(challenger) - metric(reference)" not in report


@pytest.mark.parametrize(
    ("outputs_dir", "name"),
    [
        (PHASE2_OUTPUTS_DIR, "phase2_2026_report_erratum.json"),
        (H200_OUTPUTS_DIR, "h200_2026_report_erratum.json"),
    ],
    ids=["h100", "h200"],
)
def test_the_erratum_records_a_presentation_only_correction(outputs_dir: Any, name: str) -> None:
    path = outputs_dir / name
    if not path.exists():
        pytest.skip(f"{name} has not been generated in this environment")
    erratum = json.loads(path.read_text())
    assert erratum["scope"] == "PRESENTATION ONLY"
    assert erratum["evaluation_rerun"] is False
    assert erratum["original_report_sha256"] != erratum["corrected_report_sha256"]
    assert erratum["integrity_recheck"]["all_numerical_artifacts_unchanged"] is True
    assert erratum["integrity_recheck"]["frozen_result_manifest_preserved"] is True
    assert erratum["exact_text_changed"]
