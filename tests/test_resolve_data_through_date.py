"""Tests for scripts/resolve_data_through_date.sh -- the date-resolution
logic .github/workflows/publish-prospective.yml uses when no explicit
--data-through date is supplied.

GNU-date-only (see that script's own header): this only runs for real on
Linux (the `ubuntu-latest` GitHub Actions runner uses GNU coreutils; macOS
ships BSD date with different flag semantics). Skipped elsewhere rather
than attempting a Docker-based workaround, so the offline test suite never
needs network access to pull an image -- see CLAUDE.md/conftest.py's
network-blocking fixture. The GNU-date behavior asserted here was verified
manually during development via `docker run --rm ubuntu:24.04 ...` against
the same script.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_data_through_date.sh"

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="scripts/resolve_data_through_date.sh requires GNU date (Linux); "
    "this environment has BSD/other date semantics",
)


def _resolve(reference_now: str) -> str:
    result = subprocess.run(
        [str(SCRIPT)],
        env={"REFERENCE_NOW": reference_now, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111, "script must be executable"


def test_ordinary_day_during_edt() -> None:
    assert _resolve("2026-07-15 12:00") == "2026-07-14"


def test_day_after_spring_forward_dst_boundary() -> None:
    # DST began 2026-03-08 (America/New_York) -- the reference instant is
    # the first full day of EDT.
    assert _resolve("2026-03-09 12:00") == "2026-03-08"


def test_day_after_fall_back_dst_boundary() -> None:
    # DST ended 2026-11-01 (America/New_York) -- the reference instant is
    # the first full day back on EST.
    assert _resolve("2026-11-02 12:00") == "2026-11-01"


def test_early_utc_run_still_resolves_the_correct_new_york_date() -> None:
    """The whole point of TZ-aware resolution: a run that fires at 02:00
    UTC on July 16 is still 22:00 the EVENING OF JULY 15 in
    America/New_York (UTC-4 in July) -- so "today" in New York is the
    15th, and "yesterday" is the 14th. A naive UTC-calendar-date approach
    would compute "today" as the 16th and "yesterday" as the 15th --
    exactly the off-by-one this whole script exists to avoid.
    """
    assert _resolve("2026-07-16T02:00:00Z") == "2026-07-14"


def test_missing_tzdata_fails_loudly_rather_than_silently(tmp_path: Path) -> None:
    """A bare `ubuntu:24.04` container (unlike a real Ubuntu install or the
    actual GitHub Actions runner) does not ship `tzdata` by default, and a
    `TZ=America/New_York` conversion against a missing zoneinfo file fails
    SILENTLY (wrong date, no error) rather than raising -- discovered
    during development by testing against exactly that container. This
    points the script's overridable ZONEINFO_NEW_YORK check at a path that
    genuinely doesn't exist and confirms it fails loudly instead.
    """
    missing_path = tmp_path / "no-such-zoneinfo"
    result = subprocess.run(
        [str(SCRIPT)],
        env={
            "REFERENCE_NOW": "2026-07-15 12:00",
            "ZONEINFO_NEW_YORK": str(missing_path),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "tzdata" in result.stderr


def test_real_zoneinfo_path_is_used_by_default() -> None:
    """The override exists only for the test above -- real invocations
    (including the workflow) never set ZONEINFO_NEW_YORK, so this checks
    the unset default still resolves correctly.
    """
    assert _resolve("2026-07-15 12:00") == "2026-07-14"


def test_defaults_to_real_now_when_reference_not_set() -> None:
    result = subprocess.run(
        [str(SCRIPT)],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    date_str = result.stdout.strip()
    assert len(date_str) == len("YYYY-MM-DD")
    assert date_str.count("-") == 2
