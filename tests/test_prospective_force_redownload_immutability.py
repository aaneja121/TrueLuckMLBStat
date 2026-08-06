"""Contact Luck v1.1: `--force-redownload` must never overwrite or mutate an
already-completed immutable snapshot -- it may only refresh the shared,
mutable raw staging cache under `data/prospective/2026/`.

Exercises `run_prospective_snapshot` genuinely end-to-end (real
`build_snapshot_manifest`, `resolve_snapshot_conflict`, `_read_prior_
manifest`, and file-writing logic), with every NETWORK-touching or
heavy-model-training step replaced by a fast, deterministic, in-memory stub
-- this test never touches the network or trains a real model, but the
snapshot-immutability guarantee itself is exercised for real.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import prospective_manifest as pm
import pytest
import run_v1_1_2026_scoring as runner
import v1_final_evaluation_manifest as manifest_mod

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


def _init_clean_repo(root: Path) -> Path:
    repo_root = root / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.py").write_text("X = 1\n")
    _git(repo_root, "add", "frozen.py")
    _git(repo_root, "commit", "-q", "-m", "init")
    return repo_root


@dataclass(frozen=True)
class _FakeCompletenessResult:
    def to_dict(self) -> dict[str, Any]:
        return {"is_complete": True}


@dataclass(frozen=True)
class _FakeNameReport:
    retrieved_at: str = "2026-01-01T00:00:00+00:00"

    def to_dict(self) -> dict[str, Any]:
        return {"requested_batter_count": 2, "resolved_batter_count": 0}


def _make_public_score_table(distinguishing_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter_id": [101, 102],
            "batter_name": pd.array([None, None], dtype="string"),
            "season": [2026, 2026],
            "qualification_status": ["qualified", "qualified"],
            "contact_luck_runs_per_100": [distinguishing_value, -distinguishing_value],
            "model_version": [{"contact": "baseline_v02"}, {"contact": "baseline_v02"}],
            "generated_at": ["2026-01-01T00:00:00+00:00"] * 2,
            "data_through_date": ["2026-04-15"] * 2,
            "component_status_reason_codes": [{}, {}],
        }
    )


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, *, distinguishing_value: float) -> None:
    auth = SimpleNamespace(token="stub-token", granted_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(runner, "run_prospective_guards", lambda **kw: auth)
    monkeypatch.setattr(
        runner, "assert_data_through_date_is_complete", lambda *a, **kw: _FakeCompletenessResult()
    )

    monkeypatch.setattr(
        runner,
        "ingest_2026_raw_statcast",
        lambda **kw: {
            "schema_check": {"missing_required_columns": []},
            "missing_dates_within_range": [],
            "requested_date_range": ["2026-03-25", "2026-04-15"],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": f"raw-hash-{distinguishing_value}",
            "row_count": 100,
            "observed_date_coverage": ["2026-03-25", "2026-04-15"],
        },
    )
    monkeypatch.setattr(
        runner,
        "ingest_2026_game_metadata",
        lambda **kw: {
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": "game-metadata-hash",
            "game_count": 10,
        },
    )
    monkeypatch.setattr(
        runner,
        "ingest_2026_sprint_speed",
        lambda **kw: {
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": "sprint-speed-hash",
            "player_count": 5,
        },
    )

    scoring_df = pd.DataFrame({"game_date": ["2026-04-14", "2026-04-15"], "season": [2026, 2026]})
    monkeypatch.setattr(
        runner,
        "build_2026_scoring_dataset",
        lambda **kw: (
            scoring_df,
            {"venue_join": {}, "geometry_join": {}, "sprint_speed_join": {}},
        ),
    )

    fake_artifacts = SimpleNamespace(
        report={
            "model_selection_winners": {"infield": "infield_hgb_v08"},
            "component_model_status": {"infield": "calibrated"},
        }
    )
    monkeypatch.setattr(
        runner, "train_and_score_2026", lambda *a, **kw: (fake_artifacts, SimpleNamespace())
    )

    table = _make_public_score_table(distinguishing_value)
    monkeypatch.setattr(runner, "build_public_score_table", lambda artifacts: table)
    monkeypatch.setattr(runner, "assign_official_ranks", lambda t: t)
    monkeypatch.setattr(runner, "most_favorable_leaderboard", lambda t: t.head(1))
    monkeypatch.setattr(runner, "least_favorable_leaderboard", lambda t: t.tail(1))
    monkeypatch.setattr(runner, "fetch_player_names", lambda ids: pd.DataFrame())
    monkeypatch.setattr(
        runner, "apply_player_name_overlay", lambda t, names_df: (t, _FakeNameReport())
    )

    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))


def _snapshot_files(*roots: Path) -> dict[str, tuple[float, bytes]]:
    """Snapshot every file under the given roots (mtime + content) -- deliberately
    NEVER passed the git repo root, since `assert_clean_working_tree`'s `git
    status --porcelain` call can touch `.git/index`'s own mtime as an
    incidental side effect unrelated to snapshot-output immutability.
    """
    files: dict[str, tuple[float, bytes]] = {}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file():
                files[str(p)] = (p.stat().st_mtime, p.read_bytes())
    return files


@pytest.fixture
def _isolated_dirs(tmp_path: Path) -> tuple[Path, Path]:
    outputs_root = tmp_path / "outputs" / "prospective" / "v1_1"
    artifacts_root = tmp_path / "artifacts" / "prospective" / "v1_1"
    outputs_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return outputs_root, artifacts_root


def test_first_run_writes_a_new_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_pipeline(monkeypatch, distinguishing_value=1.0)

    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    assert result["status"] == "written"
    assert (outputs_root / "2026-04-15" / "public_score.json").exists()
    assert (artifacts_root / "2026-04-15" / "manifest.json").exists()


def test_force_redownload_with_identical_result_is_idempotent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)  # "redownload" -> same content
    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        force_redownload=True,
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    assert result["status"] == "idempotent_no_op"
    assert (
        _snapshot_files(outputs_root, artifacts_root) == before
    )  # byte-for-byte and mtime-for-mtime unchanged


def test_force_redownload_that_would_change_the_result_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    # Simulate a corrected/changed raw source: --force-redownload now produces
    # a DIFFERENT public score table for the SAME --data-through date.
    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    with pytest.raises(pm.SnapshotConflictError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15",
            force_redownload=True,
            repo_root=repo_root,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

    # The already-completed snapshot must be COMPLETELY untouched.
    assert _snapshot_files(outputs_root, artifacts_root) == before


def test_force_redownload_false_also_never_overwrites_a_differing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    """The conflict guard applies regardless of `--force-redownload` --
    confirms the immutability guarantee doesn't depend on that flag at all.
    """
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    with pytest.raises(pm.SnapshotConflictError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15",
            force_redownload=False,
            repo_root=repo_root,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

    assert _snapshot_files(outputs_root, artifacts_root) == before


def test_a_different_snapshot_label_writes_a_separate_directory_not_a_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        snapshot_label="corrected",
        force_redownload=True,
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    assert result["status"] == "written"
    assert (outputs_root / "2026-04-15__corrected" / "public_score.json").exists()
    # The FIRST snapshot's own files are still completely untouched.
    original_files = {k: v for k, v in before.items() if "2026-04-15__corrected" not in k}
    after = _snapshot_files(outputs_root, artifacts_root)
    for key, value in original_files.items():
        assert after[key] == value
