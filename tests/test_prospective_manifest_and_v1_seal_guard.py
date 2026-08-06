"""Contact Luck v1.1: snapshot manifest build/validate and the read-only,
non-fatal Version 1.0 seal-integrity check.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import prospective_manifest as pm
import pytest
import run_v1_final_evaluation as rv1
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


def _init_repo_with_frozen_file(root: Path, *, dirty: bool = False) -> Path:
    repo_root = root / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.py").write_text("X = 1\n")
    _git(repo_root, "add", "frozen.py")
    _git(repo_root, "commit", "-q", "-m", "init")
    if dirty:
        (repo_root / "untracked.txt").write_text("scratch\n")
    return repo_root


def _build_manifest(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frozen_paths: tuple[str, ...] = ("frozen.py",),
    **overrides: object,
) -> pm.SnapshotManifest:
    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", frozen_paths)
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", frozen_paths)
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
        "join_coverage": {"venue_join": {"match_rate": 1.0}},
        "qualification_threshold_set": "primary",
    }
    kwargs.update(overrides)
    return pm.build_snapshot_manifest(repo_root, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Build / round-trip
# ---------------------------------------------------------------------------


def test_build_snapshot_manifest_round_trips_through_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    manifest = _build_manifest(repo_root, monkeypatch)
    round_tripped = pm.SnapshotManifest.from_dict(manifest.to_dict())
    assert round_tripped == manifest


def test_manifest_records_prospective_season_2026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    manifest = _build_manifest(repo_root, monkeypatch)
    assert manifest.prospective_season == 2026


def test_build_snapshot_manifest_itself_only_records_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_snapshot_manifest` (this low-level builder) only RECORDS
    `working_tree_clean` -- it never raises on a dirty tree itself. The
    actual BLOCKING enforcement lives one layer up, in `prospective_
    ingestion.run_prospective_guards` (see `tests/
    test_prospective_working_tree_guard.py`), which every real snapshot run
    goes through before this function is ever called.
    """
    dirty_repo = _init_repo_with_frozen_file(tmp_path, dirty=True)
    manifest = _build_manifest(dirty_repo, monkeypatch)
    assert manifest.working_tree_clean is False  # recorded, never raised HERE


def test_manifest_reuses_v1_frozen_model_and_score_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    manifest = _build_manifest(repo_root, monkeypatch)
    assert manifest.model_versions == manifest_mod._model_versions()
    assert manifest.score_version == manifest_mod._score_version()


def test_manifest_raises_manifest_error_on_missing_frozen_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    with pytest.raises(pm.ManifestError):
        _build_manifest(repo_root, monkeypatch, frozen_paths=("does_not_exist.py",))


# ---------------------------------------------------------------------------
# deterministic_content_hash()
# ---------------------------------------------------------------------------


def test_deterministic_hash_ignores_wall_clock_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    a = _build_manifest(repo_root, monkeypatch)
    b = _build_manifest(
        repo_root, monkeypatch, retrieval_timestamps={"raw_statcast": "2099-01-01T00:00:00+00:00"}
    )
    assert a.generated_at != b.generated_at or a.retrieval_timestamps != b.retrieval_timestamps
    assert a.deterministic_content_hash() == b.deterministic_content_hash()


def test_deterministic_hash_ignores_output_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    manifest = _build_manifest(repo_root, monkeypatch)
    with_hashes = pm.with_output_hashes(manifest, {"outputs/public_score.json": "deadbeef"})
    assert manifest.deterministic_content_hash() == with_hashes.deterministic_content_hash()


def test_deterministic_hash_changes_on_real_content_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    a = _build_manifest(repo_root, monkeypatch)
    b = _build_manifest(repo_root, monkeypatch, processed_row_count=999)
    assert a.deterministic_content_hash() != b.deterministic_content_hash()


def test_content_hash_differs_from_deterministic_content_hash_when_timestamps_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    a = _build_manifest(repo_root, monkeypatch)
    b = _build_manifest(
        repo_root, monkeypatch, retrieval_timestamps={"raw_statcast": "2099-01-01T00:00:00+00:00"}
    )
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# verify_v1_seal_unchanged: read-only, never raises
# ---------------------------------------------------------------------------


class _FakeV1Manifest:
    repository_commit = "commit-abc123"

    def content_hash(self) -> str:
        return "manifest-hash-abc123"


def test_seal_check_reports_verified_true_for_an_intact_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seal_path = tmp_path / "seal.json"
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", tmp_path)
    rv1.write_seal(
        manifest=_FakeV1Manifest(),  # type: ignore[arg-type]
        report={"outcome_classification": {"outcome": "validated_as_frozen"}},
        seal_path=seal_path,
    )
    result = pm.verify_v1_seal_unchanged(seal_path)
    assert result.verified is True
    assert result.seal_exists is True


def test_seal_check_is_non_fatal_when_no_seal_exists(tmp_path: Path) -> None:
    result = pm.verify_v1_seal_unchanged(tmp_path / "no_such_seal.json")  # must not raise
    assert result.verified is False
    assert result.seal_exists is False


def test_seal_check_is_non_fatal_when_seal_is_corrupted(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{not valid json")
    result = pm.verify_v1_seal_unchanged(seal_path)  # must not raise
    assert result.verified is False
    assert result.seal_exists is True
    assert "failed validation" in result.detail


def test_seal_check_is_non_fatal_when_seal_is_missing_a_required_field(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text('{"manifest_content_hash": "a"}')
    result = pm.verify_v1_seal_unchanged(seal_path)  # must not raise
    assert result.verified is False


def test_manifest_embeds_v1_seal_verification_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    seal_path = tmp_path / "no_seal_here.json"
    manifest = _build_manifest(repo_root, monkeypatch, v1_seal_path=seal_path)
    assert manifest.v1_seal_verification["verified"] is False


def test_manifest_embeds_verified_true_for_an_intact_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_repo_with_frozen_file(tmp_path)
    seal_path = tmp_path / "seal.json"
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", tmp_path)
    rv1.write_seal(
        manifest=_FakeV1Manifest(),  # type: ignore[arg-type]
        report={"outcome_classification": {"outcome": "validated_as_frozen"}},
        seal_path=seal_path,
    )
    manifest = _build_manifest(repo_root, monkeypatch, v1_seal_path=seal_path)
    assert manifest.v1_seal_verification["verified"] is True
