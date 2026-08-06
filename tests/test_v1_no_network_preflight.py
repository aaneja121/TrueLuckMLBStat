"""Contact Luck v1.0, Task 33 category 6: no-network preflight.

Verifies the manifest/integrity/clean-tree/prior-seal checks in
`run_pre_evaluation_guards` never call a network-facing function -- whether
they pass or fail -- and that the offline test suite itself is genuinely
blocked from real network access (see `tests/conftest.py`'s autouse
`_block_real_network_access` fixture, exercised directly here too).
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from unittest import mock

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


def _clean_frozen_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.txt").write_text("frozen\n")
    _git(repo_root, "add", "frozen.txt")
    _git(repo_root, "commit", "-q", "-m", "init")
    return repo_root


_NETWORK_FUNCTIONS = (
    "download_statcast_range",
    "_fetch_schedule_chunk",
    "fetch_venue_details",
    "fetch_season_sprint_speed",
)


def test_real_network_access_is_blocked_in_this_test_suite() -> None:
    """Exercises `tests/conftest.py`'s autouse `_block_real_network_access`
    fixture directly. Caught as `RuntimeError` (its base class) rather than
    the fixture's own exception class, since importing that class via a
    second module path (`tests.conftest` vs. pytest's own `conftest`) would
    create a distinct class object with the same name.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError, match="Real network access attempted"):
        sock.connect(("example.com", 80))


def test_successful_preflight_calls_zero_network_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _clean_frozen_repo(tmp_path)
    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.txt",))

    with mock.patch.multiple(rv1, **{name: mock.DEFAULT for name in _NETWORK_FUNCTIONS}) as mocks:
        manifest, prior_seal, authorization = rv1.run_pre_evaluation_guards(
            repo_root=repo_root,
            seal_path=tmp_path / "seal.json",
            acknowledge_defect_fix=False,
        )
        assert authorization is not None
        for name in _NETWORK_FUNCTIONS:
            mocks[name].assert_not_called()


def test_failed_preflight_dirty_tree_calls_zero_network_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _clean_frozen_repo(tmp_path)
    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.txt",))
    (repo_root / "frozen.txt").write_text("dirty now\n")

    with mock.patch.multiple(rv1, **{name: mock.DEFAULT for name in _NETWORK_FUNCTIONS}) as mocks:
        with pytest.raises(manifest_mod.ManifestError):
            rv1.run_pre_evaluation_guards(
                repo_root=repo_root,
                seal_path=tmp_path / "seal.json",
                acknowledge_defect_fix=False,
            )
        for name in _NETWORK_FUNCTIONS:
            mocks[name].assert_not_called()


def test_failed_preflight_prior_seal_calls_zero_network_functions(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(
        '{"manifest_content_hash": "a", "report_content_hash": "b", '
        '"sealed_at": "2025-01-01T00:00:00+00:00", "repository_commit": "c", '
        '"defect_fix_of": null}'
    )

    with mock.patch.multiple(rv1, **{name: mock.DEFAULT for name in _NETWORK_FUNCTIONS}) as mocks:
        with pytest.raises(rv1.SealError):
            rv1.run_pre_evaluation_guards(
                seal_path=seal_path,
                acknowledge_defect_fix=False,
            )
        for name in _NETWORK_FUNCTIONS:
            mocks[name].assert_not_called()
