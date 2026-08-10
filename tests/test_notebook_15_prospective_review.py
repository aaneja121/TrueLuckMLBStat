"""Contact Luck v1.1: notebook 15 structure/execution test.

Executes `notebooks/15_prospective_snapshot_review.ipynb` for real (via
`nbclient`) against a SYNTHETIC completed-snapshot fixture -- never against
real 2026 data, which does not exist in this offline test suite. Mirrors
`tests/test_notebook_14_v1_review.py`'s pattern. Verifies: the notebook runs
cell-by-cell with zero execution errors, renders the snapshot's own content,
degrades gracefully (still zero errors) when no snapshot exists, and never
writes, fits, scores, or downloads anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "15_prospective_snapshot_review.ipynb"
)


def _build_snapshot_fixture(root: Path) -> tuple[Path, Path]:
    outputs_dir = root / "outputs" / "prospective" / "v1_1" / "2026-04-15"
    artifacts_dir = root / "artifacts" / "prospective" / "v1_1" / "2026-04-15"
    outputs_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    public_score_table = pd.DataFrame(
        [
            {
                "batter_id": 101,
                "batter_name": "Alex Álvarez",
                "season": 2026,
                "qualification_status": "qualified",
                "contact_luck_runs_per_100": 3.2,
                "lower_95_interval": 0.5,
                "upper_95_interval": 5.9,
                "eligible_batted_balls": 250,
                "official_rank_favorable": 1,
                "official_rank_unfavorable": None,
            },
            {
                "batter_id": 102,
                "batter_name": None,
                "season": 2026,
                "qualification_status": "qualified",
                "contact_luck_runs_per_100": -2.1,
                "lower_95_interval": -4.5,
                "upper_95_interval": 0.3,
                "eligible_batted_balls": 210,
                "official_rank_favorable": None,
                "official_rank_unfavorable": 1,
            },
        ]
    )
    public_score_table.to_json(outputs_dir / "public_score.json", orient="records", indent=2)
    public_score_table.head(1).to_json(
        outputs_dir / "favorable_leaderboard.json", orient="records", indent=2
    )
    public_score_table.tail(1).to_json(
        outputs_dir / "unfavorable_leaderboard.json", orient="records", indent=2
    )
    (outputs_dir / "scorecard.json").write_text(json.dumps({"row_count": 2}, indent=2))
    (outputs_dir / "coverage_and_schema_report.json").write_text(
        json.dumps({"missing_dates_within_range": []}, indent=2)
    )
    (outputs_dir / "component_status_summary.json").write_text(
        json.dumps({"component_model_status": {"outfield": "calibrated"}}, indent=2)
    )
    (outputs_dir / "name_resolution_report.json").write_text(
        json.dumps(
            {
                "requested_batter_count": 2,
                "resolved_batter_count": 1,
                "unresolved_batter_count": 1,
                "source": "MLB Stats API (/people)",
            },
            indent=2,
        )
    )

    manifest = {
        "manifest_version": "1.1.0",
        "repository_commit": "abc123",
        "working_tree_clean": True,
        "prospective_season": 2026,
        "data_through_date": "2026-04-15",
        "snapshot_label": None,
        "model_versions": {"contact": "baseline_v02"},
        "score_version": "0.12.0",
        "qualification_threshold_set": "primary",
        "v1_seal_verification": {"verified": True, "seal_exists": True},
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (artifacts_dir / "integrity_hashes.json").write_text(
        json.dumps({"outputs/public_score.json": "deadbeef"}, indent=2)
    )

    return outputs_dir, artifacts_dir


def _execute_notebook(
    outputs_dir: Path | None, artifacts_dir: Path | None, monkeypatch: pytest.MonkeyPatch
) -> nbformat.NotebookNode:
    if outputs_dir is not None:
        monkeypatch.setenv("CONTACT_LUCK_PROSPECTIVE_OUTPUTS_DIR", str(outputs_dir))
    if artifacts_dir is not None:
        monkeypatch.setenv("CONTACT_LUCK_PROSPECTIVE_ARTIFACTS_DIR", str(artifacts_dir))

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    return nb


def _assert_no_execution_errors(nb: nbformat.NotebookNode) -> None:
    for i, cell in enumerate(nb["cells"]):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"cell {i} raised {output.get('ename')}: {output.get('evalue')}"
            )


def _all_cell_text(nb: nbformat.NotebookNode) -> str:
    return "\n".join(
        "".join(o.get("text", "") for o in cell.get("outputs", [])) for cell in nb["cells"]
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_notebook_uses_kernel_registered_for_this_project() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert nb["metadata"]["kernelspec"]["name"] == "mlb-luck-score"


def test_notebook_never_writes_fits_scores_or_downloads_anything() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    for forbidden in (
        ".to_parquet(",
        ".to_csv(",
        ".to_json(",
        "requests.",
        "pybaseball",
        "download_",
        "train_model",
        "fit(",
        "ingest_2026",
        "run_prospective_snapshot",
    ):
        assert forbidden not in source, f"notebook source contains forbidden token {forbidden!r}"


def test_notebook_is_watermarked_as_read_only_review() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    first_cell_source = "".join(nb["cells"][0]["source"])
    assert "WATERMARK" in first_cell_source
    assert "NOT A DEVELOPMENT NOTEBOOK" in first_cell_source


# ---------------------------------------------------------------------------
# Execution against a synthetic snapshot fixture
# ---------------------------------------------------------------------------


def test_notebook_renders_a_completed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_dir, artifacts_dir = _build_snapshot_fixture(tmp_path)
    nb = _execute_notebook(outputs_dir, artifacts_dir, monkeypatch)

    _assert_no_execution_errors(nb)
    text = _all_cell_text(nb)
    assert "SNAPSHOT FOUND" in text
    assert "data_through_date: 2026-04-15" in text
    assert "row_count: 2" in text
    assert "requested_batter_count: 2" in text


def test_notebook_degrades_gracefully_with_no_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_outputs = tmp_path / "outputs" / "prospective" / "v1_1" / "missing"
    empty_artifacts = tmp_path / "artifacts" / "prospective" / "v1_1" / "missing"
    nb = _execute_notebook(empty_outputs, empty_artifacts, monkeypatch)

    _assert_no_execution_errors(nb)
    text = _all_cell_text(nb)
    assert "SNAPSHOT NOT FOUND" in text
    assert text.count("Skipped --") >= 6


def test_notebook_execution_creates_no_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_dir, artifacts_dir = _build_snapshot_fixture(tmp_path)
    before = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

    _execute_notebook(outputs_dir, artifacts_dir, monkeypatch)

    after_paths = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after_paths == set(before.keys())
    for path, mtime in before.items():
        assert path.stat().st_mtime == mtime
