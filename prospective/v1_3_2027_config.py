"""Season dates, namespaces and fail-fast guards for the one-shot 2027 v1.3 comparison.

Governance: `RESEARCH_RULES.md`, "2027: a pre-registered test season", and
`docs/plans/v1_3_2027_preregistration.md`. 2027 dates live here, never in
`mlb_luck_score.config` (a frozen v1.0 input whose `PROSPECTIVE_SEASONS` stays `(2026,)`).

The end date follows the 2026 rule: recorded only from a maintainer-provided citation of
the official schedule, with `_SOURCE` and `_VERIFIED_AT` updated together, never one
without the others. Until then `run_guards` refuses to run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from prospective_config import (
    PROJECT_ROOT,
    NamespaceViolationError,
    assert_not_sealed_v1_namespace,
)
from prospective_ingestion import (
    COMPLETED_GAME_STATUS_VALUES,
    POSTPONED_OR_CANCELLED_STATUS_VALUES,
    WorkingTreeNotCleanError,
    fetch_schedule_game_statuses_range,
    run_prospective_guards,
)

__all__ = ["NamespaceViolationError", "WorkingTreeNotCleanError"]

SEASON_2027 = 2027

#: Verified 2026-10-01 against MLB.com: "It is slated to begin on Wednesday, March 24,
#: with an Opening Night game streamed live on Netflix."
SEASON_2027_START_DATE = date(2027, 3, 24)
SEASON_2027_START_SOURCE = (
    "MLB.com, 'MLB releases 2027 regular-season schedule' (published 2026-07-16), "
    "https://www.mlb.com/news/mlb-2027-schedule-released"
)
SEASON_2027_START_VERIFIED_AT = "2026-10-01"

#: Recorded after the season ends, from a maintainer-provided official-schedule citation.
SEASON_2027_END_DATE: date | None = None
SEASON_2027_END_SOURCE: str | None = None
SEASON_2027_END_VERIFIED_AT: str | None = None

RAW_DIR = PROJECT_ROOT / "data" / "prospective" / "2027" / "v1_3"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "prospective" / "v1_3"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "prospective" / "v1_3"
NAMESPACE_ROOTS: tuple[Path, ...] = (RAW_DIR, OUTPUTS_DIR, ARTIFACTS_DIR)
COMPARISON_PATH = OUTPUTS_DIR / "comparison_2027.json"
PREREGISTRATION = "docs/plans/v1_3_2027_preregistration.md"

#: Pre-registration §4: below this the result is "inconclusive".
MIN_POPULATION_BATTED_BALLS = 60_000

#: How far past the recorded end to look for regular-season games that would make the
#: recorded date wrong (rainout makeups).
_END_LOOKAHEAD_DAYS = 45


class SeasonNotRecordedError(ValueError):
    """The 2027 season-end date (with source and verification date) is not recorded."""


class SeasonIncompleteError(ValueError):
    """The 2027 regular season is not verifiably complete."""


class AlreadyRunError(ValueError):
    """The one permitted comparison has already been written."""


def assert_season_end_recorded(
    end_date: date | None, source: str | None, verified_at: str | None
) -> date:
    if end_date is None or not source or not verified_at:
        raise SeasonNotRecordedError(
            "The 2027 regular-season end date must be recorded in v1_3_2027_config with an "
            "official-schedule citation and a verification date before the comparison may run."
        )
    return end_date


def check_season_complete(
    start: date,
    end: date,
    *,
    fetch_fn: Callable[[str, str], list[dict[str, Any]]] = fetch_schedule_game_statuses_range,
    today: date | None = None,
) -> dict[str, Any]:
    """Every regular-season game from `start` to `end` is final, postponed or cancelled,
    and none completed after `end`. Anything else (suspended, in progress, unknown) blocks."""
    today = today or date.today()
    if today <= end:
        raise SeasonIncompleteError(f"today ({today}) is not after the recorded end ({end})")
    lookahead = end + timedelta(days=_END_LOOKAHEAD_DAYS)
    games = fetch_fn(start.isoformat(), lookahead.isoformat())
    in_season = [g for g in games if g.get("game_date") and g["game_date"] <= end.isoformat()]
    after = [g for g in games if g.get("game_date") and g["game_date"] > end.isoformat()]
    final = [g for g in in_season if g.get("status_detailed_state") in COMPLETED_GAME_STATUS_VALUES]
    off = [
        g
        for g in in_season
        if g.get("status_detailed_state") in POSTPONED_OR_CANCELLED_STATUS_VALUES
    ]
    unfinished = [g for g in in_season if g not in final and g not in off]
    late = [g for g in after if g.get("status_detailed_state") in COMPLETED_GAME_STATUS_VALUES]
    if unfinished:
        raise SeasonIncompleteError(f"{len(unfinished)} game(s) not final, e.g. {unfinished[:3]}")
    if late:
        raise SeasonIncompleteError(
            f"{len(late)} regular-season game(s) completed after the recorded end date "
            f"{end}, e.g. {late[:3]}; the recorded date is wrong"
        )
    if not final:
        raise SeasonIncompleteError("no final regular-season games found")
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_final": len(final),
        "n_postponed_or_cancelled": len(off),
        "checked_on": today.isoformat(),
    }


def assert_not_already_run(path: Path = COMPARISON_PATH) -> None:
    if path.exists():
        raise AlreadyRunError(
            f"{path} exists. The pre-registered 2027 comparison runs once; a second run needs "
            "the maintainer's explicit sign-off (RESEARCH_RULES.md, 2027 rule 7)."
        )


def assert_in_namespace(path: Path) -> None:
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in NAMESPACE_ROOTS):
        raise NamespaceViolationError(
            f"{resolved} is outside the 2027 v1.3 namespace {[str(r) for r in NAMESPACE_ROOTS]}"
        )
    assert_not_sealed_v1_namespace(path, label="v1.3 2027")


def assert_clean_tree(repo_root: Path) -> None:
    """Reuses the v1.1 guard unchanged (raises `WorkingTreeNotCleanError`)."""
    run_prospective_guards(repo_root=repo_root)


def run_guards(
    *,
    end_date: date | None = SEASON_2027_END_DATE,
    end_source: str | None = SEASON_2027_END_SOURCE,
    end_verified_at: str | None = SEASON_2027_END_VERIFIED_AT,
    comparison_path: Path = COMPARISON_PATH,
    fetch_fn: Callable[[str, str], list[dict[str, Any]]] = fetch_schedule_game_statuses_range,
    today: date | None = None,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """All fail-fast checks, cheapest first. Returns a report for the provenance record."""
    assert_clean_tree(repo_root)
    end = assert_season_end_recorded(end_date, end_source, end_verified_at)
    assert_not_already_run(comparison_path)
    schedule = check_season_complete(SEASON_2027_START_DATE, end, fetch_fn=fetch_fn, today=today)
    return {
        "season_start_date": SEASON_2027_START_DATE.isoformat(),
        "season_start_source": SEASON_2027_START_SOURCE,
        "season_end_date": end.isoformat(),
        "season_end_source": end_source,
        "season_end_verified_at": end_verified_at,
        "schedule": schedule,
    }
