"""Contact Luck v1.1: working-tree-cleanliness guard.

A dirty working tree must BLOCK prospective scoring (not merely be recorded)
-- a snapshot generated from uncommitted code could differ from the commit
named in its manifest, making it impossible to reproduce exactly. Covers all
four cases the task requires:

  - a dirty TRACKED working tree blocks scoring
  - an untracked, non-ignored source/config/test file blocks scoring
  - ignored files inside the prospective raw/output/artifact namespaces do
    NOT falsely dirty the tree
  - the snapshot manifest records the clean committed HEAD

Every scenario builds its OWN throwaway git repo under `tmp_path` -- never
touches the real repository working tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import prospective_ingestion as pi
import pytest

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


# ---------------------------------------------------------------------------
# Case 1: clean tree -- guard passes
# ---------------------------------------------------------------------------


def test_clean_tree_passes_the_guard(tmp_path: Path) -> None:
    repo_root = _init_clean_repo(tmp_path)
    auth = pi.run_prospective_guards(repo_root=repo_root)  # must not raise
    assert isinstance(auth, pi.ProspectiveAuthorization)


# ---------------------------------------------------------------------------
# Case 2: dirty TRACKED file blocks scoring
# ---------------------------------------------------------------------------


def test_modified_tracked_file_blocks_scoring(tmp_path: Path) -> None:
    repo_root = _init_clean_repo(tmp_path)
    (repo_root / "frozen.py").write_text("X = 2\n")  # unstaged modification
    with pytest.raises(pi.WorkingTreeNotCleanError):
        pi.run_prospective_guards(repo_root=repo_root)


def test_staged_but_uncommitted_change_blocks_scoring(tmp_path: Path) -> None:
    repo_root = _init_clean_repo(tmp_path)
    (repo_root / "frozen.py").write_text("X = 3\n")
    _git(repo_root, "add", "frozen.py")  # staged, not committed
    with pytest.raises(pi.WorkingTreeNotCleanError):
        pi.run_prospective_guards(repo_root=repo_root)


# ---------------------------------------------------------------------------
# Case 3: untracked, non-ignored source/config/test file blocks scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["new_module.py", "config.toml", "test_something.py"],
)
def test_untracked_non_ignored_file_blocks_scoring(tmp_path: Path, filename: str) -> None:
    repo_root = _init_clean_repo(tmp_path)
    (repo_root / filename).write_text("# not committed\n")
    with pytest.raises(pi.WorkingTreeNotCleanError):
        pi.run_prospective_guards(repo_root=repo_root)


# ---------------------------------------------------------------------------
# Case 4: ignored files inside the prospective namespaces do NOT dirty the tree
# ---------------------------------------------------------------------------


def _init_repo_with_prospective_gitignore(root: Path) -> Path:
    """Mirrors the REAL repository's `.gitignore` "Version 1.1 prospective
    2026 scoring namespace" section, so this test genuinely exercises the
    same ignore rules the real guard relies on.
    """
    repo_root = root / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / ".gitignore").write_text(
        "data/prospective/*\n"
        "outputs/prospective/*\n"
        "!artifacts/prospective/\n"
        "artifacts/prospective/*\n"
        "!data/prospective/.gitkeep\n"
        "!outputs/prospective/.gitkeep\n"
        "!artifacts/prospective/.gitkeep\n"
    )
    (repo_root / "frozen.py").write_text("X = 1\n")
    for rel in (
        "data/prospective/.gitkeep",
        "outputs/prospective/.gitkeep",
        "artifacts/prospective/.gitkeep",
    ):
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-q", "-m", "init")
    return repo_root


def test_files_inside_gitignored_prospective_namespaces_do_not_dirty_the_tree(
    tmp_path: Path,
) -> None:
    repo_root = _init_repo_with_prospective_gitignore(tmp_path)
    for rel in (
        "data/prospective/2026/statcast_2026_regular_season.parquet",
        "outputs/prospective/v1_1/2026-04-15/public_score.json",
        "artifacts/prospective/v1_1/2026-04-15/manifest.json",
    ):
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("snapshot output content, presence is what matters")

    auth = pi.run_prospective_guards(repo_root=repo_root)  # must not raise
    assert isinstance(auth, pi.ProspectiveAuthorization)


def test_a_non_ignored_file_still_blocks_even_alongside_ignored_ones(tmp_path: Path) -> None:
    repo_root = _init_repo_with_prospective_gitignore(tmp_path)
    ignored = repo_root / "data" / "prospective" / "2026" / "cache.parquet"
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_text("ignored content")
    (repo_root / "stray_new_file.py").write_text("# not committed, not ignored\n")

    with pytest.raises(pi.WorkingTreeNotCleanError):
        pi.run_prospective_guards(repo_root=repo_root)


# ---------------------------------------------------------------------------
# Manifest records the clean committed HEAD
# ---------------------------------------------------------------------------


def test_manifest_commit_matches_the_clean_head_after_the_guard_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prospective_manifest as pm
    import v1_final_evaluation_manifest as manifest_mod

    repo_root = _init_clean_repo(tmp_path)
    pi.run_prospective_guards(repo_root=repo_root)  # confirms clean, must not raise

    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()

    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    manifest = pm.build_snapshot_manifest(
        repo_root,
        data_through_date="2026-04-15",
        requested_date_range=("2026-03-25", "2026-04-15"),
        observed_date_coverage=[],
        snapshot_label=None,
        retrieval_timestamps={},
        source_hashes={},
        raw_row_counts={},
        processed_row_count=0,
        schema_checks={},
        join_coverage={},
        qualification_threshold_set="primary",
        v1_seal_path=repo_root / "no_seal.json",
    )
    assert manifest.repository_commit == expected_commit
    assert manifest.working_tree_clean is True


def test_run_prospective_guards_default_repo_root_is_project_root() -> None:
    import inspect

    from prospective_config import PROJECT_ROOT

    sig = inspect.signature(pi.run_prospective_guards)
    assert sig.parameters["repo_root"].default == PROJECT_ROOT
