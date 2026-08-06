"""Contact Luck v1.1: snapshot immutability / idempotent-rerun tests.

Exercises the pure conflict-decision logic (`resolve_snapshot_conflict`) and
the manifest round-trip (`_read_prior_manifest`) directly -- without running
the full (network-requiring) ingestion/scoring pipeline, mirroring how
`tests/test_v1_sealing.py` tests Version 1.0's sealing semantics against
fake manifest objects rather than a real evaluation run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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


def _repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.py").write_text("X = 1\n")
    _git(repo_root, "add", "frozen.py")
    _git(repo_root, "commit", "-q", "-m", "init")
    return repo_root


def _manifest(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> pm.SnapshotManifest:
    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    kwargs: dict[str, object] = {
        "data_through_date": "2026-04-15",
        "requested_date_range": ("2026-03-26", "2026-04-15"),
        "observed_date_coverage": ["2026-03-26", "2026-04-15"],
        "snapshot_label": None,
        "retrieval_timestamps": {"raw_statcast": "2026-04-16T00:00:00+00:00"},
        "source_hashes": {"raw_statcast": "abc123"},
        "raw_row_counts": {"raw_statcast": 100},
        "processed_row_count": 90,
        "schema_checks": {"ok": True},
        "join_coverage": {},
        "qualification_threshold_set": "primary",
        "v1_seal_path": repo_root / "no_seal.json",
    }
    kwargs.update(overrides)
    return pm.build_snapshot_manifest(repo_root, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_snapshot_conflict: pure decision logic
# ---------------------------------------------------------------------------


def test_no_prior_manifest_means_new_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo(tmp_path)
    candidate = _manifest(repo_root, monkeypatch)
    status = runner.resolve_snapshot_conflict(
        None, candidate, dir_name="2026-04-15", artifacts_snapshot_dir=tmp_path / "snap"
    )
    assert status == "new"


def test_identical_rerun_is_an_idempotent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo(tmp_path)
    prior = _manifest(repo_root, monkeypatch)
    rerun = _manifest(repo_root, monkeypatch)  # same inputs, different generated_at
    assert prior.generated_at != rerun.generated_at
    status = runner.resolve_snapshot_conflict(
        prior, rerun, dir_name="2026-04-15", artifacts_snapshot_dir=tmp_path / "snap"
    )
    assert status == "idempotent_no_op"


def test_differing_rerun_raises_snapshot_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo(tmp_path)
    prior = _manifest(repo_root, monkeypatch)
    changed = _manifest(repo_root, monkeypatch, processed_row_count=12345)
    with pytest.raises(pm.SnapshotConflictError):
        runner.resolve_snapshot_conflict(
            prior, changed, dir_name="2026-04-15", artifacts_snapshot_dir=tmp_path / "snap"
        )


def test_two_different_data_through_dates_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo(tmp_path)
    day_one = _manifest(repo_root, monkeypatch, data_through_date="2026-04-15")
    day_two = _manifest(repo_root, monkeypatch, data_through_date="2026-04-16")
    assert day_one.deterministic_content_hash() != day_two.deterministic_content_hash()
    # Each is evaluated against its own (absent) prior -- both are "new".
    assert (
        runner.resolve_snapshot_conflict(
            None, day_one, dir_name="2026-04-15", artifacts_snapshot_dir=tmp_path / "a"
        )
        == "new"
    )
    assert (
        runner.resolve_snapshot_conflict(
            None, day_two, dir_name="2026-04-16", artifacts_snapshot_dir=tmp_path / "b"
        )
        == "new"
    )


# ---------------------------------------------------------------------------
# _read_prior_manifest: round trip + fail-safe on corruption
# ---------------------------------------------------------------------------


def test_read_prior_manifest_returns_none_when_absent(tmp_path: Path) -> None:
    assert runner._read_prior_manifest(tmp_path / "does_not_exist") is None


def test_read_prior_manifest_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = _repo(tmp_path)
    manifest = _manifest(repo_root, monkeypatch)
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), default=str))

    read_back = runner._read_prior_manifest(snapshot_dir)
    assert read_back == manifest


def test_read_prior_manifest_fails_safely_on_corrupted_json(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.json").write_text("{not valid json")
    with pytest.raises(pm.SnapshotConflictError):
        runner._read_prior_manifest(snapshot_dir)


def test_read_prior_manifest_fails_safely_on_schema_mismatch(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.json").write_text(json.dumps({"unexpected": "shape"}))
    with pytest.raises(pm.SnapshotConflictError):
        runner._read_prior_manifest(snapshot_dir)
