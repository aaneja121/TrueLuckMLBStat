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
score, archive, history_sync, generate_explorer, build) and can be told to
fail on command. This proves the actual shell control flow, not just an
assertion about it.

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
  scripts/verify_local_snapshot_integrity.py) STAGE_NAME="verify_snapshot" ;;
  scripts/generate_production_explorer_artifacts.py) STAGE_NAME="generate_explorer" ;;
  dashboard/build.py) STAGE_NAME="build" ;;
  *) STAGE_NAME="unknown" ;;
esac
echo "python:$STAGE_NAME" >> "$CALL_LOG"
# Full argv, to a SEPARATE file: `$CALL_LOG` is compared with `==` against
# exact stage sequences, so anything added to it would break every one of
# those assertions. Stage order lives there; stage ARGUMENTS live here.
if [[ -n "${ARGV_LOG:-}" ]]; then
  echo "$STAGE_NAME $*" >> "$ARGV_LOG"
fi
if [[ "${FAIL_STAGE:-}" == "$STAGE_NAME" ]]; then
  echo "fake python: simulating failure for stage $STAGE_NAME" >&2
  exit "${FAIL_EXIT_CODE:-1}"
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
    fake_project: Path,
    call_log: Path,
    *extra_args: str,
    fail_stage: str | None = None,
    fail_exit_code: int = 1,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{fake_project / 'fake-bin'}:{env['PATH']}"
    env["CALL_LOG"] = str(call_log)
    env["ARGV_LOG"] = str(call_log.parent / "argv.log")
    # Fake R2 credentials -- history sync now always runs (even dry-run),
    # so scripts/publish_snapshot.sh's own env-var presence expectations
    # must be satisfied; the fake python never actually uses them.
    env["R2_ACCOUNT_ID"] = "fake-account"
    env["R2_ACCESS_KEY_ID"] = "fake-key-id"
    env["R2_SECRET_ACCESS_KEY"] = "fake-secret"
    if fail_stage is not None:
        env["FAIL_STAGE"] = fail_stage
        env["FAIL_EXIT_CODE"] = str(fail_exit_code)
    else:
        env.pop("FAIL_STAGE", None)
        env.pop("FAIL_EXIT_CODE", None)
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
        assert "python:generate_explorer" not in lines, (
            "Explorer artifact generation must never run after a failed history sync"
        )
        assert not any(line.startswith("python:build") for line in lines), (
            "dashboard build must never run after a failed history sync"
        )
        assert not any(line.startswith("npx:") for line in lines), (
            "deploy must never be attempted after a failed history sync"
        )

    def test_generate_explorer_failure_prevents_build_and_deploy(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Version 1.4.0 Phase 5: Explorer artifact generation is its own
        pipeline stage, between history sync and the dashboard build -- a
        failure there (e.g. the just-scored snapshot fails its own
        integrity check, or is somehow not play_ledger_version 2.0) must
        prevent the dashboard from being built or deployed, exactly like
        every earlier stage's failure does.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="generate_explorer")

        assert result.returncode != 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == [
            "python:ensure",
            "python:score",
            "python:archive",
            "python:history_sync",
            "python:generate_explorer",
        ]
        assert not any(line.startswith("python:build") for line in lines), (
            "dashboard build must never run after a failed Explorer artifact generation"
        )
        assert not any(line.startswith("npx:") for line in lines), (
            "deploy must never be attempted after a failed Explorer artifact generation"
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

    def test_ensure_then_score_then_archive_then_history_sync_then_explore_run_before_build_on_success(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Direct evidence of the documented ordering (ensure -> score ->
        archive -> history_sync -> generate_explorer -> build -> deploy),
        not just an assertion about it.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log)

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines.index("python:ensure") < lines.index("python:score")
        assert lines.index("python:score") < lines.index("python:archive")
        assert lines.index("python:archive") < lines.index("python:history_sync")
        assert lines.index("python:history_sync") < lines.index("python:generate_explorer")
        assert lines.index("python:generate_explorer") < lines.index("python:build")


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
            "python:generate_explorer",
            "python:build",
        ]
        assert not any(line.startswith("npx:") for line in lines), "--skip-deploy must never deploy"

    def test_skip_archive_and_skip_deploy_still_runs_history_sync_and_explore_read_only(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """The key behavior change this task adds: a full dry run
        (--skip-archive --skip-deploy) skips the archive WRITE and the
        Pages deploy, but the frozen-input check, history sync, AND
        Explorer artifact generation still run -- none of the three touch
        the archive WRITE path (generation only reads this run's own local
        outputs), so this validates the real CI scoring/dashboard-build
        behavior even in dry-run mode.
        """
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, "--skip-archive", "--skip-deploy")

        assert result.returncode == 0, result.stderr
        lines = _log_lines(call_log)
        assert lines == [
            "python:ensure",
            "python:score",
            "python:history_sync",
            "python:generate_explorer",
            "python:build",
        ]
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
        assert lines[:6] == [
            "python:ensure",
            "python:score",
            "python:archive",
            "python:history_sync",
            "python:generate_explorer",
            "python:build",
        ]
        assert any(line.startswith("npx:wrangler pages deploy") for line in lines)


def _argv_lines(call_log: Path) -> list[str]:
    argv_log = call_log.parent / "argv.log"
    if not argv_log.exists():
        return []
    return [line for line in argv_log.read_text().splitlines() if line]


class TestTheBuildStagePublishesThePitcherSurface:
    """The production wiring, checked by RUNNING the publish path rather
    than by reading it.

    `dashboard/build.py` has no default pitcher fixture -- publication is
    one explicit flag on one line of this script. A string search would pass
    on a commented-out line or a flag attached to the wrong stage, so this
    asserts what the build stage was actually invoked with.
    """

    def _build_argv(self, fake_project: Path, tmp_path: Path) -> str:
        log = tmp_path / "calls.log"
        result = _run(fake_project, log, "--skip-deploy")
        assert result.returncode == 0, result.stderr
        build_lines = [ln for ln in _argv_lines(log) if ln.startswith("build ")]
        assert len(build_lines) == 1, _argv_lines(log)
        return build_lines[0]

    def test_the_build_stage_receives_the_pitcher_fixture(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        argv = self._build_argv(fake_project, tmp_path)
        assert "--pitcher-season-fixture" in argv
        assert "dashboard/pitcher_season_fixture.json" in argv

    def test_the_build_stage_still_receives_the_explorer_artifacts(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """The pitcher flag is added ALONGSIDE the explorer one, not in
        place of it -- a line-continuation edit is exactly where that gets
        lost."""
        argv = self._build_argv(fake_project, tmp_path)
        assert "--explore-artifacts-dir" in argv

    def test_no_other_stage_is_handed_the_pitcher_fixture(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--skip-deploy")
        for line in _argv_lines(log):
            if not line.startswith("build "):
                assert "--pitcher-season-fixture" not in line, line


class TestFeatureOnlyBuildFromExistingSnapshot:
    """`--build-from-existing-snapshot`: build the site from an
    ALREADY-ARCHIVED snapshot, without scoring anything.

    Why the mode exists, because the tests only make sense with it:
    a snapshot's manifest records `repository_commit`, `generated_at` and a
    hash of every frozen input. Re-scoring a date that is already archived
    therefore yields a DIFFERENT manifest as soon as the repository has
    moved on -- even when every scored value is identical -- and history
    sync then correctly refuses to reconcile the fresh copy with the
    archived one. That refusal is the archive guard working. This mode is
    the way to ship a PRESENTATION change without provoking it: it never
    re-scores the date it builds from.

    Everything below is a safety property, not a convenience: the mode must
    be incapable of scoring, incapable of writing to R2, and incapable of
    deploying, and it must refuse rather than silently downgrade a request
    for any of those.
    """

    def _lines(self, fake_project: Path, tmp_path: Path, *extra: str):
        log = tmp_path / "calls.log"
        result = _run(fake_project, log, "--build-from-existing-snapshot", *extra)
        return result, _log_lines(log)

    # ── it must never score ──────────────────────────────────────────────
    def test_the_scoring_script_is_never_invoked(self, fake_project: Path, tmp_path: Path) -> None:
        """THE point of the mode. Not 'scored and discarded' -- never run."""
        result, lines = self._lines(fake_project, tmp_path)
        assert result.returncode == 0, result.stderr
        assert not any(line.startswith("python:score") for line in lines), lines

    def test_no_stage_receives_the_scoring_entry_point(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--build-from-existing-snapshot")
        for line in _argv_lines(log):
            assert "run_v1_1_2026_scoring.py" not in line, line

    # ── it must never write to R2 ────────────────────────────────────────
    def test_the_archive_write_is_never_invoked(self, fake_project: Path, tmp_path: Path) -> None:
        """The harness distinguishes the archive WRITE from `--sync-history`
        by scanning argv, so this is a real separation, not a name match."""
        _, lines = self._lines(fake_project, tmp_path)
        assert not any(line.startswith("python:archive") for line in lines), lines

    def test_the_only_archive_call_is_the_read_only_history_sync(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--build-from-existing-snapshot")
        archive_calls = [ln for ln in _argv_lines(log) if "archive_snapshot.py" in ln]
        assert archive_calls, "the snapshot has to be retrieved from somewhere"
        for call in archive_calls:
            assert "--sync-history" in call, call

    # ── it must never deploy ─────────────────────────────────────────────
    def test_it_never_deploys(self, fake_project: Path, tmp_path: Path) -> None:
        _, lines = self._lines(fake_project, tmp_path)
        assert not any(line.startswith("npx:") for line in lines), lines

    def test_it_refuses_an_explicit_skip_deploy_rather_than_ignoring_it(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Refuse, not silently accept: a caller passing flags this mode
        forces anyway has a different model of what it does."""
        log = tmp_path / "calls.log"
        result = _run(fake_project, log, "--build-from-existing-snapshot", "--skip-deploy")
        assert result.returncode != 0
        assert "redundant" in result.stderr
        assert _log_lines(log) == []

    def test_it_refuses_an_explicit_skip_archive(self, fake_project: Path, tmp_path: Path) -> None:
        log = tmp_path / "calls.log"
        result = _run(fake_project, log, "--build-from-existing-snapshot", "--skip-archive")
        assert result.returncode != 0
        assert "meaningless" in result.stderr
        assert _log_lines(log) == []

    # ── it must verify what it retrieved ─────────────────────────────────
    def test_the_retrieved_snapshot_is_verified_before_anything_is_built(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        _, lines = self._lines(fake_project, tmp_path)
        assert "python:verify_snapshot" in lines, lines
        assert lines.index("python:verify_snapshot") < lines.index("python:generate_explorer")
        assert lines.index("python:verify_snapshot") < lines.index("python:build")

    def test_the_verifier_is_told_which_snapshot_to_verify(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--build-from-existing-snapshot")
        verify = [ln for ln in _argv_lines(log) if ln.startswith("verify_snapshot ")]
        assert len(verify) == 1, verify
        assert "--data-through 2026-08-09" in verify[0]

    def test_a_failed_verification_stops_the_build(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """A snapshot that is not the archived one must never be published,
        so the failure has to be terminal, not advisory."""
        log = tmp_path / "calls.log"
        result = _run(
            fake_project, log, "--build-from-existing-snapshot", fail_stage="verify_snapshot"
        )
        assert result.returncode != 0
        lines = _log_lines(log)
        assert "python:verify_snapshot" in lines
        assert not any(line.startswith("python:generate_explorer") for line in lines), lines
        assert not any(line.startswith("python:build") for line in lines), lines

    def test_a_failed_history_sync_stops_before_verification(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """If the archive has no such snapshot, or refuses to reconcile a
        local one, nothing downstream may run."""
        log = tmp_path / "calls.log"
        result = _run(
            fake_project, log, "--build-from-existing-snapshot", fail_stage="history_sync"
        )
        assert result.returncode != 0
        lines = _log_lines(log)
        assert not any(line.startswith("python:verify_snapshot") for line in lines), lines
        assert not any(line.startswith("python:build") for line in lines), lines

    # ── it must still build the real thing ───────────────────────────────
    def test_the_exact_stage_sequence(self, fake_project: Path, tmp_path: Path) -> None:
        _, lines = self._lines(fake_project, tmp_path)
        assert lines == [
            "python:ensure",
            "python:history_sync",
            "python:verify_snapshot",
            "python:generate_explorer",
            "python:build",
        ], lines

    def test_the_explorer_generator_receives_the_retrieved_snapshot(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--build-from-existing-snapshot")
        gen = [ln for ln in _argv_lines(log) if ln.startswith("generate_explorer ")]
        assert len(gen) == 1, gen
        assert "--data-through 2026-08-09" in gen[0]

    def test_the_build_receives_both_the_explorer_artifacts_and_the_pitcher_fixture(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--build-from-existing-snapshot")
        build = [ln for ln in _argv_lines(log) if ln.startswith("build ")]
        assert len(build) == 1, build
        assert "--explore-artifacts-dir" in build[0]
        assert "--pitcher-season-fixture dashboard/pitcher_season_fixture.json" in build[0]

    # ── the normal path must be untouched ────────────────────────────────
    def test_the_normal_dry_run_still_scores(self, fake_project: Path, tmp_path: Path) -> None:
        """The guard against this mode leaking into the scheduled loop."""
        log = tmp_path / "calls.log"
        _run(fake_project, log, "--skip-archive", "--skip-deploy")
        lines = _log_lines(log)
        assert "python:score" in lines
        assert "python:verify_snapshot" not in lines

    def test_the_normal_deploy_path_still_scores_and_archives(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        log = tmp_path / "calls.log"
        _run(fake_project, log)
        lines = _log_lines(log)
        assert "python:score" in lines
        assert "python:archive" in lines
        assert "python:verify_snapshot" not in lines


class TestNothingToScoreIsACleanStop:
    """Exit 3 from the scoring entry point means "this date completed no
    games" -- the season ended, or the slate was postponed. That is a normal
    outcome for the daily scheduled loop, not a fault, so the orchestrator
    must stop cleanly: publish nothing, and exit 0 so the workflow does not
    report a failure every morning of the off-season. See `prospective.
    prospective_ingestion.assert_data_through_date_has_completed_games`.
    """

    def test_exit_three_from_scoring_is_not_a_failure(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="score", fail_exit_code=3)
        assert result.returncode == 0, result.stderr

    def test_exit_three_says_plainly_why_it_stopped(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="score", fail_exit_code=3)
        assert "no completed games" in (result.stdout + result.stderr).lower()

    def test_exit_three_publishes_nothing(self, fake_project: Path, tmp_path: Path) -> None:
        """The whole point: no archive write, no history sync, no build, no
        deploy -- and above all no NEW snapshot for a day without baseball.
        """
        call_log = tmp_path / "calls.log"
        _run(fake_project, call_log, fail_stage="score", fail_exit_code=3)
        assert _log_lines(call_log) == ["python:ensure", "python:score"]

    def test_an_ordinary_scoring_failure_still_fails_the_run(
        self, fake_project: Path, tmp_path: Path
    ) -> None:
        """Exit 3 is special; every other non-zero code must stay a failure."""
        call_log = tmp_path / "calls.log"
        result = _run(fake_project, call_log, fail_stage="score", fail_exit_code=2)
        assert result.returncode != 0
        assert _log_lines(call_log) == ["python:ensure", "python:score"]
