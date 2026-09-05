"""Contact Forecast R1: the freeze.

A freeze is only worth the guarantees it enforces, so these tests attack it:
prose that no longer matches its own numbers must be refused, a silently
rewritten specification must be refused, and drift in any artifact, source
module or input ledger must be detected rather than tolerated.

Synthetic artifacts only -- no real R1 output is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forecast import freeze_r1 as fz
from forecast.forecast_config import (
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_ANALYSIS_SEASONS,
    PRIMARY_COMPARISON,
    PRIMARY_CUTOFFS,
    PRIMARY_HORIZON,
    PRIMARY_TARGET,
)

REFERENCE, CHALLENGER = PRIMARY_COMPARISON
BOOT_KEY = f"{PRIMARY_TARGET}|{CHALLENGER}_vs_{REFERENCE}"


def make_comparison(*, delta_mae: float, pct: float) -> dict[str, Any]:
    return {
        "status": "measured",
        "seasons_in_comparison": [2023, 2024],
        "seasons_excluded_for_missing_rung": [2022],
        "n_paired_windows": 567,
        "pooled_window_weighted": {
            "delta_mae": delta_mae,
            "pct_change_mae": pct,
            "delta_rmse": delta_mae * 1.3,
            "direction_mae": "challenger_better" if delta_mae < 0 else "reference_better",
        },
        "macro_average_of_season_deltas": {"macro_delta_mae": delta_mae},
        "batter_balanced": {"delta": {"delta_mae": delta_mae * 1.17}},
        "practical_magnitude": {"abs_delta_mae_as_fraction_of_target_sd": 0.029},
    }


def write_artifacts(
    outputs_dir: Path,
    *,
    delta_mae: float = -0.1715,
    pct: float = -3.90,
    seasons: list[int] | None = None,
    excluded: list[int] | None = None,
) -> None:
    """The minimum artifact shape `extract_key_results` reads."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    comparison = make_comparison(delta_mae=delta_mae, pct=pct)
    if seasons is not None:
        comparison["seasons_in_comparison"] = seasons
    if excluded is not None:
        comparison["seasons_excluded_for_missing_rung"] = excluded

    metrics = {
        "cells": {
            f"{PRIMARY_TARGET}|K{cutoff}_H{PRIMARY_HORIZON}": {
                "comparisons": {CHALLENGER: comparison}
            }
            for cutoff in PRIMARY_CUTOFFS
        }
    }
    boot_cell = {
        "pooled": {
            "delta_mae": {"ci_lower": -0.3144, "ci_upper": -0.0258, "ci_crosses_zero": False}
        },
        "by_season": {
            "2023": {"delta_mae": {"ci_crosses_zero": True}},
            "2024": {"delta_mae": {"ci_crosses_zero": True}},
        },
    }
    bootstrap = {
        "comparisons": {
            BOOT_KEY: {
                "cells": {f"K{cutoff}_H{PRIMARY_HORIZON}": boot_cell for cutoff in PRIMARY_CUTOFFS}
            }
        }
    }
    regression = {
        "regression_direction_by_cell": {
            "K100_H100_season2024": {
                "status": "fitted",
                "real_surprise_regression": {"slope_surprise": -0.994},
                "placebo": {
                    "surprise_slope": {
                        "placebo_mean": -0.437,
                        "p_lower_real_more_negative": 0.001,
                    }
                },
            }
        }
    }
    (outputs_dir / "r1_metrics.json").write_text(json.dumps(metrics))
    (outputs_dir / "r1_bootstrap.json").write_text(json.dumps(bootstrap))
    (outputs_dir / "r1_regression_direction.json").write_text(json.dumps(regression))
    (outputs_dir / "r1_report.md").write_text("# report\n")
    (outputs_dir / "r1_prediction_table.parquet").write_bytes(b"not-a-real-parquet")


def write_ledgers(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for season in FORECAST_ANALYSIS_SEASONS:
        (data_dir / f"walk_forward_ledger_{season}.parquet").write_bytes(b"ledger")
        (data_dir / f"walk_forward_manifest_{season}.json").write_text("{}")


@pytest.fixture
def package(tmp_path: Path) -> tuple[Path, Path]:
    outputs_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    write_artifacts(outputs_dir)
    write_ledgers(data_dir)
    return outputs_dir, data_dir


# --------------------------------------------------------------------------
# The conclusion must match the data it describes
# --------------------------------------------------------------------------


def test_a_freeze_records_the_conservative_verdict(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    manifest = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert manifest["conclusion"]["verdict"] == "PROCEED"
    assert manifest["conclusion"]["label"] == "modest positive development evidence"
    assert manifest["conclusion"]["evidence_class"] == "development-season evidence only"


def test_the_conclusion_refuses_to_claim_the_published_metric_predicts() -> None:
    """The must-not list is part of the frozen record, not a footnote."""
    forbidden = fz.R1_CONCLUSION["must_not_be_described_as"]
    assert any("proof that the published metric predicts" in item for item in forbidden)
    assert any("no forecasting model was fitted" in item for item in forbidden)
    assert any("2025 is sealed" in item for item in forbidden)


def test_a_drifted_percentage_refuses_the_freeze(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    write_artifacts(outputs_dir, delta_mae=-0.9, pct=-19.0)
    with pytest.raises(fz.ForecastFreezeError, match="approximately -3.9"):
        fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)


def test_a_flipped_direction_refuses_the_freeze(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    # Same magnitude, opposite sign: prose says deserved wins, data says realized.
    write_artifacts(outputs_dir, delta_mae=0.1715, pct=-3.90)
    with pytest.raises(fz.ForecastFreezeError, match="deserved outperforms"):
        fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)


def test_wrong_season_coverage_refuses_the_freeze(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    write_artifacts(outputs_dir, seasons=[2022, 2023, 2024], excluded=[])
    with pytest.raises(fz.ForecastFreezeError, match="covers 2023-2024"):
        fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)


def test_a_missing_artifact_refuses_the_freeze(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    (outputs_dir / "r1_report.md").unlink()
    with pytest.raises(fz.ForecastFreezeError, match="is missing"):
        fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)


# --------------------------------------------------------------------------
# The specification is written once
# --------------------------------------------------------------------------


def test_the_specification_is_written_and_hashed(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    manifest = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    spec_path = outputs_dir / "forecast_spec.json"
    assert spec_path.exists()
    assert fz.hash_json(json.loads(spec_path.read_text())) == manifest["specification_sha256"]


def test_refreezing_identical_content_is_allowed(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    first = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    second = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert first["specification_sha256"] == second["specification_sha256"]


def test_a_changed_specification_is_refused_without_an_explicit_flag(
    package: tuple[Path, Path],
) -> None:
    outputs_dir, data_dir = package
    fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    spec_path = outputs_dir / "forecast_spec.json"
    tampered = json.loads(spec_path.read_text())
    tampered["primary_family"]["cutoffs"] = [100]
    spec_path.write_text(json.dumps(tampered))
    with pytest.raises(fz.ForecastFreezeError, match="written once"):
        fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)


def test_an_explicit_respecification_is_permitted(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    spec_path = outputs_dir / "forecast_spec.json"
    spec_path.write_text(json.dumps({"spec_version": "tampered"}))
    manifest = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir, force_respecify=True)
    assert manifest["specification"]["spec_version"] == fz.FREEZE_VERSION


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------


def test_a_fresh_freeze_verifies(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    result = fz.verify_freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert result["matches_freeze"] is True
    assert result["drift"] == {}


def test_a_changed_artifact_is_detected(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    (outputs_dir / "r1_report.md").write_text("# tampered\n")
    result = fz.verify_freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert result["matches_freeze"] is False
    assert result["drift"]["artifacts"] == ["r1_report.md"]


def test_a_changed_input_ledger_is_detected(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    (data_dir / "walk_forward_ledger_2024.parquet").write_bytes(b"different")
    result = fz.verify_freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert result["drift"]["input_ledgers"] == ["walk_forward_ledger_2024.parquet"]


def test_verifying_without_a_freeze_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(fz.ForecastFreezeError, match="No freeze manifest"):
        fz.verify_freeze(outputs_dir=tmp_path, data_dir=tmp_path)


def test_every_source_module_is_hashed(package: tuple[Path, Path]) -> None:
    outputs_dir, data_dir = package
    manifest = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    assert set(manifest["source_module_sha256"]) == set(fz.R1_SOURCE_MODULES)
    assert "forecast/metrics.py" in manifest["source_module_sha256"]


# --------------------------------------------------------------------------
# The documented record
# --------------------------------------------------------------------------


def test_all_five_required_limitations_are_documented() -> None:
    ids = {item["id"] for item in fz.R1_DOCUMENTED_LIMITATIONS}
    assert ids == {
        "primary_comparison_covers_2023_2024_only",
        "no_2021_cross_fitting_will_be_added",
        "focal_view_is_not_a_specification_narrowing",
        "every_prespecified_cell_remains_visible",
        "conditional_on_reaching_the_horizon",
    }


def test_the_rejected_amendment_is_recorded_with_its_evidence() -> None:
    amendment = fz.SPECIFICATION_AMENDMENTS[0]
    assert amendment["outcome"] == "NOT ADOPTED"
    assert "repository history" in amendment["test_applied"]
    assert any("git log" in item for item in amendment["evidence_examined"])
    assert "focal presentation view" in amendment["resolution"]


def test_the_specification_keeps_all_four_cutoffs() -> None:
    specification = fz.build_specification()
    assert specification["primary_family"]["cutoffs"] == list(PRIMARY_CUTOFFS)
    assert specification["primary_family"]["none_promoted"] is True
    assert specification["focal_presentation_view"]["cutoff"] == FOCAL_PRESENTATION_CUTOFF
    assert "NOT a narrowing" in specification["focal_presentation_view"]["status"]


def test_the_specification_covers_every_phase_2_gate_item() -> None:
    specification = fz.build_specification()
    assert specification["phase_2_gate"]
    assert specification["season_policy"]["sealed"] == [2025]
    assert specification["season_policy"]["phase_2_gated"] == [2026]


def test_the_rendered_conclusion_carries_the_verdict_and_the_limits(
    package: tuple[Path, Path],
) -> None:
    outputs_dir, data_dir = package
    manifest = fz.freeze(outputs_dir=outputs_dir, data_dir=data_dir)
    rendered = (outputs_dir / "r1_frozen_conclusion.md").read_text()
    assert "PROCEED -- modest positive development evidence" in rendered
    assert "must NOT be described as" in rendered
    assert "no_2021_cross_fitting_will_be_added" in rendered
    assert "NOT ADOPTED" in rendered
    assert manifest["manifest_sha256"] in rendered
