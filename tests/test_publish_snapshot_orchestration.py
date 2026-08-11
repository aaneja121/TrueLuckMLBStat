"""Regression tests for scripts/publish_snapshot.sh's STAGE SEQUENCING --
specifically: does an ensure-frozen-inputs/archive/history-sync (or
scoring) failure actually prevent the dashboard build and the Cloudflare
Pages deploy from ever running, and does the frozen-input check actually
run BEFORE scoring (see `TestFailuresBlockLaterStages.
test_ensure_frozen_inputs_failure_prevents_everything_after_it`).

This deliberately does NOT rely only on "the script has `set -euo
pipefail`, so it must be fine" -- that's a reasonable design argument, but
not a test. Instead it runs the REAL `scripts/publish_snapshot.sh` (copied
byte-for-byte into an isolated fake project root, never a hand-written
duplicate of its logic) against fake `.venv/bin/python` and `npx`
executables that log every invocation (by clean stage name -- ensure,
score, archive, history_sync, build) and can be told to fail on command.
This proves the actual shell control flow, not just an assertion about it.

Never scores real MLB data, never contacts R2, never contacts Cloudflare
Pages: the fake `python` never runs real prospective/dashboard/archive/
ensure-frozen-inputs code (it only inspects its own argv to decide what to
log and whether to exit non-zero), and the fake `npx` never runs real
wrangler.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_snapshot.sh"

# Records every invocation as "python:<stage_name>" to $CALL_LOG, then exits
# 1 (rather than the actual scoring/archive/sync/build work) if the stage
# named by $FAIL_STAGE matches. The archive_snapshot.py script is called for
# TWO different stages (archive-upload and --sync-history) -- distinguished
# by scanning the full argv, not just $1, since both pass
# "scripts/archive_snapshot.py" as $1.
FAKE_PYTHON = """#!/usr/bin/env bash
set -euo pipefail
FIRST_ARG="${1:-}"
case "$FIRST_ARG" in
  scripts/ensure_frozen_inputs.py) STAGE_NAME="ensure" ;;
  prospective/run_v1_1_2026_scoring.py) STAGE_NAME="score" ;;
  scripts/archive_snapshot.py)
    if [[ "$*" == *"--sync-history"* ]]; then
      STAGE_NAME="history_sync"
    else
      STAGE_NAME="archive"
    fi
    ;;
  dashboard/build.py) STAGE_NAME="build" ;;
  *) STAGE_NAME="unknown" ;;
esac
echo "python:$STAGE_NAME" >> "$CALL_LOG"
if [[ "${FAIL_STAGE:-}" == "$STAGE_NAME" ]]; then
  echo "fake python: simulating failure for stage $STAGE_NAME" >&2
  exit 1
fi
exit 0
"""

# Records "npx:<args>" to $CALL_LOG and always succeeds -- publish_snapshot.sh
# calls `npx wrangler pages deploy ...` directly (not through $PYTHON), so
# this is the only way to observe whether a deploy was attempted.
FAKE_NPX = """#!/usr/bin/env bash
echo "npx:$*" >> "$CALL_LOG"
exit 0
"""


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """An isolated directory shaped just enough like the real repo for
    scripts/publish_snapshot.sh's own PROJECT_ROOT-detection
    (`dirname "${BASH_SOURCE[0]}"/..`) to work -- a copy of the REAL
    script, a fake `.venv/bin/python`, and a `fake-bin/npx` meant to be
    prepended to PATH.
    """
    root = tmp_path / "fake-repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "fake-bin").mkdir()

    copied_script = root / "scripts" / "publish_snapshot.sh"
    shutil.copy2(REAL_SCRIPT, copied_script)
    copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _make_executable(root / ".venv" / "bin" / "python", FAKE_PYTHON)
    _make_executable(root / "fake-bin" / "npx", FAKE_NPX)

    return root


def _run(
    fake_project: Path, call_log: Path, *extra_args: str, fail_stage: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{fake_project / 'fake-bin'}:{env['PATH']}"
    env["CALL_LOG"] = str(call_log)
    # Fake R2 credentials -- history sync now always runs (even dry-run),
    # so scripts/publish_snapshot.sh's own env-var presence expectations
    # must be satisfied; the fake python never actually uses them.
    env["R2_ACCOUNT_ID"] = "fake-account"
    env["R2_ACCESS_KEY_ID"] = "fake-key-id"
    env["R2_SECRET_ACCESS_KEY"] = "fake-secret"
    if fail_stage is not None:
        env["FAIL_STAGE"] = fail_stage
    else:
        env.pop("FAIL_STAGE", None)
    return subprocess.run(
        ["scripts/publish_snapshot.sh", "--data-through", "2026-08-09", "--yes", *extra_args],
        cwd=fake_project,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _log_lines(call_log: Path) -> list[str]:
    if not call_log.exists():
        return []
    return [line for line in call_log.read_text().splitlines() if line]


class TestFailuresBlockLaterStages:
    def test_ensure_frozen_inputs_failure_prevents_everything_after_it(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """THE key orchestration proof this task adds: the frozen-input
        check is the very first stage, and a failure there must prevent
        scoring (and everything after it) from ever running -- scoring
        cannot proceed without the frozen development input, so this
        stage failing must behave exactly like scoring itself failing.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="ensure")

        assert result.returncode != 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == ["python:ensure"]
        assert "python:score" not in lines
        assert not any(line.startswith("npx:") for line in lines), (
            "deploy must never be attempted after a failed frozen-input check"
        )

    def test_archive_failure_prevents_history_sync_build_and_deploy(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="archive")

        assert result.returncode != 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == ["python:ensure", "python:score", "python:archive"]
        assert "python:history_sync" not in lines
        assert not any(line.startswith("python:build") for line in lines), (
            "dashboard build must never run after a failed archive"
        )
        assert not any(line.startswith("npx:") for line in lines), (
            "deploy must never be attempted after a failed archive"
        )

    def test_history_sync_failure_prevents_build_and_deploy(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="history_sync")

        assert result.returncode != 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == [
            "python:ensure",
            "python:score",
            "python:archive",
            "python:history_sync",
        ]
        assert not any(line.startswith("python:build") for line in lines), (
            "dashboard build must never run after a failed history sync"
        )
        assert not any(line.startswith("npx:") for line in lines), (
            "deploy must never be attempted after a failed history sync"
        )

    def test_scoring_failure_prevents_everything_after_it(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Confirms the same guarantee holds transitively for every later
        stage, not just the one immediately before the dashboard build --
        the frozen-input check succeeds first, then scoring itself fails.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="score")

        assert result.returncode != 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == ["python:ensure", "python:score"]
        assert not any(line.startswith("npx:") for line in lines)

    def test_ensure_then_score_then_archive_then_history_sync_run_before_build_on_success(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Direct evidence of the documented ordering (ensure -> score ->
        archive -> history_sync -> build -> deploy), not just an assertion
        about it.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log)

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines.index("python:ensure") < lines.index("python:score")
        assert lines.index("python:score") < lines.index("python:archive")
        assert lines.index("python:archive") < lines.index("python:history_sync")
        assert lines.index("python:history_sync") < lines.index("python:build")


class TestSkipFlagSemantics:
    def test_skip_deploy_archives_syncs_and_builds_but_never_deploys(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, "--skip-deploy")

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == [
            "python:ensure",
            "python:score",
            "python:archive",
            "python:history_sync",
            "python:build",
        ]
        assert not any(line.startswith("npx:") for line in lines), "--skip-deploy must never deploy"

    def test_skip_archive_and_skip_deploy_still_runs_history_sync_read_only(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """The key behavior change this task adds: a full dry run
        (--skip-archive --skip-deploy) skips the archive WRITE and the
        Pages deploy, but the frozen-input check and history sync still
        run -- both only READ from R2 in this mode, so this validates the
        real CI scoring/dashboard-build behavior even in dry-run mode.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, "--skip-archive", "--skip-deploy")

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == ["python:ensure", "python:score", "python:history_sync", "python:build"]
        assert "python:archive" not in lines
        assert not any(line.startswith("npx:") for line in lines)

    def test_skip_archive_without_skip_deploy_is_rejected_before_any_stage_runs(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, "--skip-archive")

        assert result.returncode != 0
        assert "cannot be combined with a real deploy" in result.stderr
        assert _log_lines(call_log) == [], "refusal must happen before scoring is ever invoked"

    def test_full_success_reaches_deploy(self, fake_project: Path, tmp_path: Path) -> None:
        """Baseline happy path -- contrast for the failure tests above:
        when nothing fails and neither skip flag is passed, deploy IS
        reached, after ensure, archive, AND history sync.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log)

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines[:5] == [
            "python:ensure",
            "python:score",
            "python:archive",
            "python:history_sync",
            "python:build",
        ]
        assert any(line.startswith("npx:wrangler pages deploy") for line in lines)
