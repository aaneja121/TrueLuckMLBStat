"""Contact Luck v1.0, Task 34: notebook 14 structure/execution test.

Executes `notebooks/14_v1_final_evaluation_review.ipynb` for real (via
`nbclient`) against a SYNTHETIC sealed Version 1.0 fixture -- never against
real 2025 data, which does not exist in this offline test suite. Verifies:
the notebook runs cell-by-cell with zero execution errors when the seal is
valid, refuses to render substantive results (also with zero errors) when
no seal exists or the report has been tampered with post-seal, and never
performs a network call (blocked globally by `tests/conftest.py`).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
import pytest
import run_v1_final_evaluation as rv1
import v1_final_evaluation_manifest as manifest_mod
from nbclient import NotebookClient
from v1_final_report import assemble_final_report

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "14_v1_final_evaluation_review.ipynb"
)

_GIT_ENV_ARGS = [
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.com",
    "-c",
    "commit.gpgsign=false",
]


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *_GIT_ENV_ARGS, *args], cwd=repo_root, check=True, capture_output=True, text=True
    )


def _build_sealed_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A synthetic (never real-2025) sealed Version 1.0 fixture: a throwaway
    git repo with one frozen artifact, a manifest built from it, a minimal
    but schema-complete final report, and a genuine seal over that report.
    Returns `(repo_root, artifacts_dir, outputs_dir)`.
    """
    repo_root = root / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.py").write_text("X = 1\n")
    _git(repo_root, "add", "frozen.py")
    _git(repo_root, "commit", "-q", "-m", "init")

    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    manifest = manifest_mod.build_manifest(repo_root)

    artifacts_dir = root / "artifacts" / "final_evaluation" / "v1"
    outputs_dir = root / "outputs" / "final_evaluation" / "v1"
    artifacts_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    public_score_table = pd.DataFrame(
        [
            {
                "batter_id": 1001,
                "season": 2025,
                "qualification_status": "qualified",
                "contact_luck_runs_per_100": 3.2,
                "lower_95_interval": 0.5,
                "upper_95_interval": 5.9,
                "eligible_batted_balls": 250,
                "official_rank_favorable": 1,
                "official_rank_unfavorable": None,
                "interval_interpretation": "entirely_above_zero",
            },
            {
                "batter_id": 1002,
                "season": 2025,
                "qualification_status": "qualified",
                "contact_luck_runs_per_100": -2.1,
                "lower_95_interval": -4.5,
                "upper_95_interval": 0.3,
                "eligible_batted_balls": 210,
                "official_rank_favorable": None,
                "official_rank_unfavorable": 1,
                "interval_interpretation": "overlaps_zero",
            },
        ]
    )
    public_score_table.to_json(
        outputs_dir / "public_score_v1_final.json", orient="records", indent=2
    )
    public_score_table.head(1).to_json(
        outputs_dir / "favorable_leaderboard_v1_final.json", orient="records", indent=2
    )
    public_score_table.tail(1).to_json(
        outputs_dir / "unfavorable_leaderboard_v1_final.json", orient="records", indent=2
    )

    report: dict[str, Any] = assemble_final_report(
        manifest_dict=manifest.to_dict(),
        system_checks={"play_level_accounting": {"identity_holds_for_every_resolved_row": True}},
        component_metrics={
            "contact_model": {"log_loss": 0.65},
            "open_field_outfield_model": {"expected_calibration_error": 0.02},
            "infield_model": {"overall": {}},
            "advancement_model": {"overall_comparison": {}},
            "near_wall_specialist": {
                "status": "provisional",
                "status_reason": "Version 0.7 is frozen; never adopted from a single run",
            },
        },
        distribution_shift_report={"continuous_features": {}},
        distribution_shift_flags=["launch_speed: SMD medium (0.3)"],
        public_score_summary={
            "n_batter_seasons": 2,
            "n_qualified": 2,
            "share_qualified": 1.0,
            "interval_width_distribution": {"mean": 4.0, "median": 4.0, "min": 4.0, "max": 4.0},
            "qualified_interval_interpretation_shares": {
                "entirely_above_zero": 1,
                "overlaps_zero": 1,
                "entirely_below_zero": 0,
            },
        },
        stability_summary={"split_half_2025": {}, "retrospective_interpretation_note": "note"},
        outcome="validated_with_documented_limitations",
        outcome_reasons=["one or more provisional components remain"],
        limitations=["launch_speed: SMD medium (0.3)"],
    )
    report["ingestion_provenance"] = {"raw_statcast": {"row_count": 12345}}
    (outputs_dir / "v1_final_report.json").write_text(json.dumps(report, indent=2, default=str))

    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", artifacts_dir)
    rv1.write_seal(manifest=manifest, report=report, seal_path=artifacts_dir / "seal.json")

    return repo_root, artifacts_dir, outputs_dir


def _execute_notebook(
    artifacts_dir: Path, outputs_dir: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> nbformat.NotebookNode:
    monkeypatch.setenv("CONTACT_LUCK_V1_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("CONTACT_LUCK_V1_OUTPUTS_DIR", str(outputs_dir))
    monkeypatch.setenv("CONTACT_LUCK_V1_REPO_ROOT", str(repo_root))

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    return nb


def _cell_text(nb: nbformat.NotebookNode, index: int) -> str:
    outputs = nb["cells"][index].get("outputs", [])
    return "".join(o.get("text", "") for o in outputs)


def _assert_no_execution_errors(nb: nbformat.NotebookNode) -> None:
    for i, cell in enumerate(nb["cells"]):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"cell {i} raised {output.get('ename')}: {output.get('evalue')}"
            )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_notebook_uses_kernel_registered_for_this_project() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert nb["metadata"]["kernelspec"]["name"] == "mlb-luck-score"


def test_notebook_never_writes_or_downloads_anything() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    for forbidden in (
        ".to_parquet(",
        ".to_csv(",
        "requests.",
        "pybaseball",
        "download_",
        "train_model",
        "fit(",
    ):
        assert forbidden not in source, f"notebook source contains forbidden token {forbidden!r}"


def test_notebook_is_watermarked_as_final_evaluation_review() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    first_cell_source = "".join(nb["cells"][0]["source"])
    assert "WATERMARK" in first_cell_source
    assert "NOT A DEVELOPMENT NOTEBOOK" in first_cell_source


# ---------------------------------------------------------------------------
# Execution against a synthetic sealed fixture
# ---------------------------------------------------------------------------


def test_notebook_renders_results_with_a_valid_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, artifacts_dir, outputs_dir = _build_sealed_fixture(tmp_path, monkeypatch)
    nb = _execute_notebook(artifacts_dir, outputs_dir, repo_root, monkeypatch)

    _assert_no_execution_errors(nb)
    assert "SEAL VERIFIED" in _cell_text(nb, 3)
    assert "VALIDATED WITH DOCUMENTED LIMITATIONS" in _cell_text(nb, 5)
    assert "provisional" in _cell_text(nb, 11)


def test_notebook_refuses_to_render_with_no_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    outputs_dir = tmp_path / "outputs"
    repo_root = tmp_path / "repo"
    artifacts_dir.mkdir()
    outputs_dir.mkdir()
    repo_root.mkdir()

    nb = _execute_notebook(artifacts_dir, outputs_dir, repo_root, monkeypatch)

    _assert_no_execution_errors(nb)
    assert "SEAL INVALID OR ABSENT" in _cell_text(nb, 3)
    for index in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25):
        assert "Skipped -- no valid seal." in _cell_text(nb, index)


def test_notebook_refuses_to_render_a_tampered_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, artifacts_dir, outputs_dir = _build_sealed_fixture(tmp_path, monkeypatch)

    report_path = outputs_dir / "v1_final_report.json"
    report = json.loads(report_path.read_text())
    report["outcome_classification"]["outcome"] = "validated_as_frozen"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    nb = _execute_notebook(artifacts_dir, outputs_dir, repo_root, monkeypatch)

    _assert_no_execution_errors(nb)
    assert "SEAL INVALID OR ABSENT" in _cell_text(nb, 3)
    assert "does not match the sealed hash" in _cell_text(nb, 3)
