"""Contact Forecast: the staged freezes for the ridge and HGB stages.

A freeze is only worth the guarantees it enforces. These tests make the
recorded prose fail when the artifact moves underneath it, make the provenance
chain fail when an upstream stage is missing or re-frozen, and make drift in
any artifact or source module detectable.

Synthetic artifacts only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forecast import freeze_hgb as fh
from forecast import freeze_ridge as fr
from forecast.freeze_r1 import ForecastFreezeError, hash_json
from forecast.stage_freeze import freeze_stage, verify_stage


def write_upstream(outputs_dir: Path, name: str, *, verdict: str = "PROCEED") -> str:
    """A well-formed upstream manifest carrying a correct self-hash."""
    manifest: dict[str, Any] = {
        "stage": name.split("_")[0],
        "conclusion": {"verdict": verdict},
        "key_results": {},
    }
    manifest["manifest_sha256"] = hash_json(manifest)
    (outputs_dir / name).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest["manifest_sha256"]


def write_ridge_results(
    outputs_dir: Path,
    *,
    delta_mae: float = -0.1477,
    pct: float = -3.39,
    ci: tuple[float, float] = (-0.3381, 0.0332),
    crosses_zero: bool = True,
    b_to_c_step: float = 0.0138,
) -> None:
    model_mae = {
        "A_results_only": 4.5297,
        "B_deserved_only": 4.3285,
        "C_results_plus_deserved": 4.3285 + b_to_c_step,
        "D_full_contact_profile": 4.2128,
    }
    results = {
        "feature_sets": {name: {"declared_features": ["f1", "f2"]} for name in model_mae},
        "selected_alphas": dict.fromkeys(model_mae, 100.0),
        "selected_model": {"model": "D_full_contact_profile"},
        "protocol": {"selection_metric": "mae"},
        "r1_provenance": {"r1_conclusion": "PROCEED"},
        "fold_2_evaluation": {
            "benchmarks_available": ["shrunk_deserved_persistence"],
            "metrics": {
                **{f"ridge_{k}": {"mae": v} for k, v in model_mae.items()},
                "shrunk_deserved_persistence": {"mae": 4.3605},
            },
        },
        "deltas_on_evaluation_season": {
            "shrunk_deserved_persistence": {
                f"ridge_{k}": {"delta": {"delta_mae": v - 4.3605}} for k, v in model_mae.items()
            }
        },
        "key_delta": {
            "definition": "MAE(ridge) - MAE(shrunk_deserved_persistence)",
            "evaluation_season": 2024,
            "n_evaluation_windows": 283,
            "best_ridge_model": "D_full_contact_profile",
            "mae_best_ridge": 4.2128,
            "mae_shrunk_deserved": 4.3605,
            "delta_mae": delta_mae,
            "pct_change_mae": pct,
            "ci_delta_mae": list(ci),
            "ci_crosses_zero": crosses_zero,
            "share_replicates_favouring_benchmark": 0.059,
        },
    }
    (outputs_dir / "ridge_results.json").write_text(json.dumps(results))
    (outputs_dir / "ridge_report.md").write_text("# ridge report\n")


def write_hgb_results(
    outputs_dir: Path,
    *,
    delta_vs_ridge: float = 0.0708,
    pct: float = 1.68,
    ci: tuple[float, float] = (-0.0420, 0.1891),
    crosses_zero: bool = True,
    preferred: str = "ridge",
    direct_mae: float = 4.2836,
    residual_mae: float = 4.3081,
    fraction_improved: float = 0.519,
) -> None:
    results = {
        "nonlinear_value_test": {
            "question": "does nonlinear modeling extract more?",
            "answer": "No",
            "preferred_model": preferred,
            "delta_mae_hgb_minus_ridge": delta_vs_ridge,
            "pct_change_mae": pct,
            "ci_delta_mae": list(ci),
            "ci_crosses_zero": crosses_zero,
        },
        "fold_2_evaluation": {
            "evaluate_season": 2024,
            "n_evaluate": 283,
            "n_train_direct": 564,
            "n_train_residual": 284,
            "metrics": {
                "hgb_direct": {"mae": direct_mae},
                "hgb_residual": {"mae": residual_mae},
                "shrunk_deserved_persistence": {"mae": 4.3605},
            },
        },
        "deltas_on_evaluation_season": {
            "shrunk_deserved_persistence": {
                "hgb_direct": {"delta_mae": -0.0769},
                "hgb_residual": {"delta_mae": -0.0524},
            }
        },
        "fold_1_selection": {
            "selected_searched_parameters": {"learning_rate": 0.03},
            "validation_mae_spread": {"range": 0.54},
        },
        "candidate_grid": {
            "n_candidates": 16,
            "searched_parameters": {"learning_rate": [0.03, 0.1]},
            "fixed_parameters": {"min_samples_leaf": 20},
        },
        "specification": {
            "feature_set": "D_full_contact_profile",
            "feature_list": ["f"] * 22,
            "feature_provenance": "inherited",
        },
        "residual_diagnostics": {"fraction_of_benchmark_error_recovered": 0.012},
        "stability": {
            "vs_shrunk_deserved": {
                "mean": 0.0769,
                "median": 0.1475,
                "fraction_of_hitters_improved": fraction_improved,
                "trimmed_means": {"mean_excluding_top_5_most_improved": 0.0127},
            }
        },
        "provenance": {"ridge": {"ridge_verdict": "CONTINUE"}},
    }
    (outputs_dir / "hgb_results.json").write_text(json.dumps(results))
    (outputs_dir / "hgb_report.md").write_text("# hgb report\n")
    (outputs_dir / "hgb_candidate_grid.json").write_text("{}")
    (outputs_dir / "hgb_specification.json").write_text("{}")


@pytest.fixture
def ridge_dir(tmp_path: Path) -> Path:
    write_upstream(tmp_path, "r1_freeze_manifest.json")
    write_ridge_results(tmp_path)
    return tmp_path


@pytest.fixture
def hgb_dir(tmp_path: Path) -> Path:
    write_upstream(tmp_path, "r1_freeze_manifest.json")
    write_upstream(tmp_path, "ridge_freeze_manifest.json", verdict="CONTINUE")
    write_hgb_results(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------
# The ridge stage's recorded conclusion
# --------------------------------------------------------------------------


def test_the_ridge_conclusion_is_recorded_verbatim() -> None:
    verdict = fr.RIDGE_CONCLUSION["verdict"]
    assert verdict.startswith("CONTINUE -- promising incremental development signal")
    assert "statistically inconclusive" in verdict
    focal = fr.RIDGE_CONCLUSION["focal_comparison"]
    assert "-0.1477" in focal
    assert "-3.39%" in focal
    assert "[-0.3381, +0.0332]" in focal


def test_the_ridge_conclusion_refuses_the_word_established() -> None:
    forbidden = fr.RIDGE_CONCLUSION["must_not_be_described_as"]
    assert any("established superiority" in item for item in forbidden)
    assert any("crosses zero" in item for item in forbidden)


def test_the_negative_ablation_is_preserved_in_the_record() -> None:
    preserved = fr.RIDGE_CONCLUSION["preserved_negative_results"]
    assert any(
        "adding realized performance on top of deserved" in item.lower() for item in preserved
    )


def test_a_ridge_freeze_records_the_verdict_and_the_ablation(ridge_dir: Path) -> None:
    manifest = freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)
    assert manifest["conclusion"]["verdict"].startswith("CONTINUE")
    steps = manifest["key_results"]["ablation_steps_evaluation_mae"]
    assert steps["B_to_C_adding_realized_on_top_of_deserved"] > 0


def test_a_drifted_ridge_delta_refuses_the_freeze(ridge_dir: Path) -> None:
    write_ridge_results(ridge_dir, delta_mae=-0.9, pct=-20.0)
    with pytest.raises(ForecastFreezeError, match="delta_MAE = -0.1477"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)


def test_a_drifted_ridge_interval_refuses_the_freeze(ridge_dir: Path) -> None:
    write_ridge_results(ridge_dir, ci=(-0.9, -0.5))
    with pytest.raises(ForecastFreezeError, match="CI of"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)


def test_an_interval_that_stops_crossing_zero_refuses_the_freeze(ridge_dir: Path) -> None:
    """The prose calls the result inconclusive; it must not outlive that."""
    write_ridge_results(ridge_dir, crosses_zero=False)
    with pytest.raises(ForecastFreezeError, match="no longer crosses zero"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)


def test_an_ablation_that_turns_positive_refuses_the_freeze(ridge_dir: Path) -> None:
    write_ridge_results(ridge_dir, b_to_c_step=-0.05)
    with pytest.raises(ForecastFreezeError, match="NEGATIVE ablation"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)


# --------------------------------------------------------------------------
# The HGB stage's recorded conclusion
# --------------------------------------------------------------------------


def test_the_hgb_conclusion_prefers_ridge_and_says_why() -> None:
    verdict = fh.HGB_CONCLUSION["verdict"]
    assert verdict.startswith("PREFER RIDGE")
    assert "did not extract information" in verdict


def test_the_hgb_conclusion_is_not_a_refutation_of_nonlinear_modeling() -> None:
    forbidden = fh.HGB_CONCLUSION["must_not_be_described_as"]
    assert any("cannot add value" in item for item in forbidden)
    assert any("close the nonlinear direction permanently" in item for item in forbidden)
    assert any("also\ncrosses zero" in item or "crosses zero" in item for item in forbidden)


def test_the_hgb_negative_results_are_preserved() -> None:
    preserved = fh.HGB_CONCLUSION["preserved_negative_results"]
    assert any("WORSE than the frozen full-profile ridge" in item for item in preserved)
    assert any("HGB-Residual is worse than HGB-Direct" in item for item in preserved)
    assert any("51.9% of hitters improved" in item for item in preserved)


def test_an_hgb_freeze_records_the_specification(hgb_dir: Path) -> None:
    manifest = freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    specification = manifest["specification"]
    assert specification["model_family"].endswith("HistGradientBoostingRegressor")
    assert "XGBoost" in specification["families_explicitly_not_implemented"]
    assert specification["selection_structure"]["evaluation_season_used_in_any_selection"] is False
    assert manifest["specification_sha256"]


def test_a_preference_flip_refuses_the_hgb_freeze(hgb_dir: Path) -> None:
    write_hgb_results(hgb_dir, preferred="hgb")
    with pytest.raises(ForecastFreezeError, match="PREFER RIDGE"):
        freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)


def test_a_residual_arm_that_beats_direct_refuses_the_hgb_freeze(hgb_dir: Path) -> None:
    write_hgb_results(hgb_dir, residual_mae=4.0)
    with pytest.raises(ForecastFreezeError, match="HGB-Residual is worse"):
        freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)


def test_a_broad_improvement_refuses_the_hgb_freeze(hgb_dir: Path) -> None:
    """The record says the advantage is not broad; that must stay true."""
    write_hgb_results(hgb_dir, fraction_improved=0.80)
    with pytest.raises(ForecastFreezeError, match="not broad"):
        freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)


# --------------------------------------------------------------------------
# The provenance chain
# --------------------------------------------------------------------------


def test_a_missing_upstream_stage_refuses_the_freeze(tmp_path: Path) -> None:
    write_ridge_results(tmp_path)
    with pytest.raises(ForecastFreezeError, match="upstream manifest"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=tmp_path)


def test_an_upstream_manifest_that_fails_its_own_hash_is_refused(
    ridge_dir: Path,
) -> None:
    manifest = json.loads((ridge_dir / "r1_freeze_manifest.json").read_text())
    manifest["conclusion"]["verdict"] = "TAMPERED"
    (ridge_dir / "r1_freeze_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ForecastFreezeError, match="does not match its own recorded hash"):
        freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)


def test_the_chain_records_every_upstream_verdict(hgb_dir: Path) -> None:
    manifest = freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    chain = manifest["provenance_chain"]
    assert set(chain) == {"r1_freeze_manifest.json", "ridge_freeze_manifest.json"}
    assert chain["ridge_freeze_manifest.json"]["stage_conclusion"] == "CONTINUE"


def test_a_re_frozen_upstream_stage_is_detected_as_drift(hgb_dir: Path) -> None:
    freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    write_upstream(hgb_dir, "ridge_freeze_manifest.json", verdict="CHANGED")
    result = verify_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    assert result["matches_freeze"] is False
    assert "ridge_freeze_manifest.json re-frozen since" in result["drift"]["provenance_chain"]


# --------------------------------------------------------------------------
# Drift detection
# --------------------------------------------------------------------------


def test_a_fresh_stage_freeze_verifies(ridge_dir: Path) -> None:
    freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)
    result = verify_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)
    assert result["matches_freeze"] is True
    assert result["drift"] == {}


def test_a_changed_artifact_is_detected(ridge_dir: Path) -> None:
    freeze_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)
    (ridge_dir / "ridge_report.md").write_text("# tampered\n")
    result = verify_stage(fr.RIDGE_FREEZE_SPEC, outputs_dir=ridge_dir)
    assert result["drift"]["artifacts"] == ["ridge_report.md"]


def test_verifying_without_a_freeze_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ForecastFreezeError, match="No freeze manifest"):
        verify_stage(fh.HGB_FREEZE_SPEC, outputs_dir=tmp_path)


def test_every_stage_source_module_is_hashed(hgb_dir: Path) -> None:
    manifest = freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    assert set(manifest["source_module_sha256"]) == set(fh.HGB_SOURCE_MODULES)


def test_the_rendered_conclusion_carries_the_negatives(hgb_dir: Path) -> None:
    freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    rendered = (hgb_dir / "hgb_frozen_conclusion.md").read_text()
    assert "PREFER RIDGE" in rendered
    assert "Negative results preserved" in rendered
    assert "must NOT be described as" in rendered
    assert "Provenance chain" in rendered


def test_season_protection_travels_with_every_stage_freeze(hgb_dir: Path) -> None:
    manifest = freeze_stage(fh.HGB_FREEZE_SPEC, outputs_dir=hgb_dir)
    assert manifest["sealed_seasons"] == [2025]
    assert manifest["phase_2_entry_point_exists"] is False
    assert manifest["phase_2_gate_requirements"]
