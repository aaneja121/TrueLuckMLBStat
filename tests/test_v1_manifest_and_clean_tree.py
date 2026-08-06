"""Contact Luck v1.0, Task 33 categories 3+4: clean-tree/freeze guards and
manifest/artifact integrity.

Every test here builds its own throwaway git repository under `tmp_path` --
none of them touch this actual repository's working tree or history.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import v1_final_evaluation_manifest as manifest_mod

# ---------------------------------------------------------------------------
# Throwaway git repo helper
# ---------------------------------------------------------------------------

_GIT_ENV_ARGS = [
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.com",
    "-c",
    "commit.gpgsign=false",
]


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *_GIT_ENV_ARGS, *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo_with_one_frozen_artifact(tmp_path: Path) -> tuple[Path, str]:
    """A minimal git repo containing exactly one file, with `FROZEN_ARTIFACT_
    RELATIVE_PATHS` monkeypatched (by the caller) to that single path, so
    manifest tests don't depend on this actual repository's frozen files.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    frozen_file = repo_root / "frozen.txt"
    frozen_file.write_text("frozen content v1\n")
    _git(repo_root, "add", "frozen.txt")
    _git(repo_root, "commit", "-q", "-m", "initial commit")
    commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    return repo_root, commit


@pytest.fixture
def frozen_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo_root, commit = _init_repo_with_one_frozen_artifact(tmp_path)
    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.txt",))
    return repo_root, commit


# ---------------------------------------------------------------------------
# Category 3: clean-tree and freeze guards
# ---------------------------------------------------------------------------


def test_dirty_tracked_file_rejects_clean_tree_check(frozen_repo: tuple[Path, str]) -> None:
    repo_root, _ = frozen_repo
    (repo_root / "frozen.txt").write_text("modified, uncommitted\n")
    with pytest.raises(manifest_mod.ManifestError, match="not clean"):
        manifest_mod.assert_clean_working_tree(repo_root)


def test_untracked_non_ignored_file_rejects_clean_tree_check(
    frozen_repo: tuple[Path, str],
) -> None:
    repo_root, _ = frozen_repo
    (repo_root / "new_untracked_source.py").write_text("x = 1\n")
    with pytest.raises(manifest_mod.ManifestError, match="not clean"):
        manifest_mod.assert_clean_working_tree(repo_root)


def test_ignored_output_file_does_not_dirty_the_tree(frozen_repo: tuple[Path, str]) -> None:
    repo_root, _ = frozen_repo
    (repo_root / ".gitignore").write_text("outputs/final_evaluation/*\n")
    _git(repo_root, "add", ".gitignore")
    _git(repo_root, "commit", "-q", "-m", "add gitignore")

    ignored_dir = repo_root / "outputs" / "final_evaluation"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "v1_final_report.json").write_text("{}")

    # Must not raise -- the ignored output file must never be seen as dirt.
    manifest_mod.assert_clean_working_tree(repo_root)


def test_commit_hash_is_captured_correctly(frozen_repo: tuple[Path, str]) -> None:
    repo_root, expected_commit = frozen_repo
    assert manifest_mod.get_git_commit_hash(repo_root) == expected_commit


def test_detached_head_commit_hash_is_still_captured(frozen_repo: tuple[Path, str]) -> None:
    repo_root, expected_commit = frozen_repo
    _git(repo_root, "checkout", "-q", "--detach", expected_commit)
    assert manifest_mod.get_git_commit_hash(repo_root) == expected_commit


def test_missing_commit_information_fails_safely_no_commits_yet(tmp_path: Path) -> None:
    repo_root = tmp_path / "empty_repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.get_git_commit_hash(repo_root)


def test_missing_commit_information_fails_safely_not_a_git_repo(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.get_git_commit_hash(not_a_repo)


# ---------------------------------------------------------------------------
# Category 4: manifest and artifact integrity
# ---------------------------------------------------------------------------


def test_manifest_is_deterministic_apart_from_timestamp_fields(
    frozen_repo: tuple[Path, str],
) -> None:
    repo_root, _ = frozen_repo
    first = manifest_mod.build_manifest(repo_root)
    second = manifest_mod.build_manifest(repo_root)

    first_dict = first.to_dict()
    second_dict = second.to_dict()
    permitted_timestamp_fields = {"evaluation_date"}
    for key in first_dict:
        if key in permitted_timestamp_fields:
            continue
        assert first_dict[key] == second_dict[key], f"non-timestamp field {key!r} differs"


def test_all_required_frozen_artifacts_are_represented(frozen_repo: tuple[Path, str]) -> None:
    repo_root, _ = frozen_repo
    manifest = manifest_mod.build_manifest(repo_root)
    assert set(manifest.artifact_hashes.keys()) == set(manifest_mod.FROZEN_ARTIFACT_RELATIVE_PATHS)


def test_file_hashes_are_deterministic(frozen_repo: tuple[Path, str]) -> None:
    repo_root, _ = frozen_repo
    path = repo_root / "frozen.txt"
    assert manifest_mod.compute_file_sha256(path) == manifest_mod.compute_file_sha256(path)


def test_changed_artifact_content_fails_validation(frozen_repo: tuple[Path, str]) -> None:
    """Isolates the content-hash check specifically: the file changes but
    HEAD does not move (as if amending in place), so the commit-hash check
    passes and the artifact-hash check is what catches the drift.
    """
    repo_root, _ = frozen_repo
    manifest = manifest_mod.build_manifest(repo_root)

    (repo_root / "frozen.txt").write_text("changed content\n")

    with pytest.raises(manifest_mod.ManifestError, match="changed since manifest"):
        manifest_mod.validate_manifest_against_current_state(manifest, repo_root)


def test_missing_artifact_fails_manifest_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _ = _init_repo_with_one_frozen_artifact(tmp_path)
    monkeypatch.setattr(
        manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.txt", "does_not_exist.txt")
    )
    with pytest.raises(manifest_mod.ManifestError, match="missing"):
        manifest_mod.build_manifest(repo_root)


def test_additional_required_artifact_fails_validation(frozen_repo: tuple[Path, str]) -> None:
    """If a NEW file is added to `FROZEN_ARTIFACT_RELATIVE_PATHS` after a
    manifest was already built, re-validating that stale manifest against
    the current (larger) required set must fail -- not silently pass just
    because every artifact the OLD manifest already knew about still matches.
    The new file is added to the working tree WITHOUT a new commit, isolating
    this from the (separately-tested) commit-hash check.
    """
    repo_root, _ = frozen_repo
    manifest = manifest_mod.build_manifest(repo_root)

    (repo_root / "second_frozen.txt").write_text("a second frozen file\n")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.txt", "second_frozen.txt")
        )
        with pytest.raises(manifest_mod.ManifestError, match="set of frozen artifacts"):
            manifest_mod.validate_manifest_against_current_state(manifest, repo_root)


def test_removed_required_artifact_fails_validation(frozen_repo: tuple[Path, str]) -> None:
    """The inverse: a manifest built with an artifact list that has since
    shrunk (a file was decided not to be frozen anymore) must also fail --
    the manifest and the module's current expectations must match exactly.
    """
    repo_root, _ = frozen_repo
    manifest = manifest_mod.build_manifest(repo_root)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ())
        with pytest.raises(manifest_mod.ManifestError, match="set of frozen artifacts"):
            manifest_mod.validate_manifest_against_current_state(manifest, repo_root)


def test_frozen_versions_thresholds_seeds_and_schema_are_populated(
    frozen_repo: tuple[Path, str],
) -> None:
    repo_root, _ = frozen_repo
    manifest = manifest_mod.build_manifest(repo_root)

    assert manifest.model_versions
    assert manifest.score_version
    assert manifest.frozen_qualification_thresholds
    assert manifest.frozen_calibration_gate_rules
    assert manifest.frozen_bootstrap_settings
    assert manifest.deterministic_seeds == {"model_random_seed": 42, "bootstrap_seed": 42}


def test_manifest_validation_occurs_before_ingestion(
    frozen_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_pre_evaluation_guards` must raise (and never mint an
    authorization) when the tree is dirty -- proving manifest/clean-tree
    validation happens strictly before any ingestion could be authorized.
    """
    import run_v1_final_evaluation as rv1

    repo_root, _ = frozen_repo
    (repo_root / "frozen.txt").write_text("dirty\n")

    tokens_before = set(rv1._ISSUED_AUTHORIZATION_TOKENS)
    with pytest.raises(manifest_mod.ManifestError):
        rv1.run_pre_evaluation_guards(
            repo_root=repo_root,
            seal_path=repo_root / "seal.json",
            acknowledge_defect_fix=False,
        )
    assert tokens_before == rv1._ISSUED_AUTHORIZATION_TOKENS
