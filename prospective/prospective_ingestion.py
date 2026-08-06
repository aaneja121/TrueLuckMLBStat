"""Contact Luck v1.1: 2026 raw-data ingestion -- the ONLY code path authorized
to fetch prospective 2026 Statcast/game-metadata/sprint-speed data.

Mirrors `evaluation.run_v1_final_evaluation`'s `ingest_2025_*`/`build_2025_
evaluation_dataset` shape exactly, but for the isolated, repeatable 2026
prospective namespace (`prospective.prospective_config.PROSPECTIVE_RAW_DIR`)
instead of the sealed, one-time 2025 namespace. Every ingestion function
requires a `ProspectiveAuthorization` OBJECT (never a bare bool), minted only
by `run_prospective_guards`, and never touches `data/raw`, `data/processed`,
or any sealed Version 1.0 namespace.

Reuses genuinely generic pure functions directly rather than duplicating
them: `evaluation.run_v1_final_evaluation.assert_raw_statcast_schema_
compatible` (schema-generic) and `._build_2025_game_metadata` (despite its
name, takes `date_range` as a parameter and hardcodes no season -- see that
function's own definition). `mlb_luck_score.data.download_sprint_speed.
fetch_season_sprint_speed` is likewise reused as-is; only `download_sprint_
speed`'s own CLI-facing season list is off-limits, exactly as for Version 1.0.

## Pre-flight guards

`run_prospective_guards` now BLOCKS (raises `WorkingTreeNotCleanError`) on
any dirty working tree -- tracked or untracked-and-not-ignored -- before
minting an authorization token; see its own docstring and CLAUDE.md "Version
1.1" for why a snapshot's manifest must be reproducible from its recorded
commit alone. Separately, `assert_data_through_date_is_complete` refuses a
`--data-through` date that has any game not yet final (in progress,
suspended, or otherwise unresolved) -- see its own docstring for the
documented postponed/suspended-game rule.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from prospective_config import (
    PROJECT_ROOT,
    PROSPECTIVE_2026_SEASON_START_DATE,
    PROSPECTIVE_2026_SEASON_START_VERIFIED,
    PROSPECTIVE_RAW_DIR,
    PROSPECTIVE_SEASON,
    _assert_within_namespace,
)
from prospective_manifest import compute_file_sha256
from run_v1_final_evaluation import (
    _build_2025_game_metadata as _build_prospective_game_metadata,
)
from run_v1_final_evaluation import (
    assert_raw_statcast_schema_compatible,
)
from v1_final_evaluation_manifest import ManifestError as _V1ManifestError
from v1_final_evaluation_manifest import assert_clean_working_tree

from mlb_luck_score.config import (
    MLB_STATS_API_BASE_URL,
    development_raw_path,
    game_metadata_path,
    sprint_speed_path,
)
from mlb_luck_score.data.clean_development_data import clean_development_data
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_CHUNK_DAYS as _GAME_METADATA_DEFAULT_CHUNK_DAYS,
)
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_MAX_RETRIES as _GAME_METADATA_DEFAULT_MAX_RETRIES,
)
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_RETRY_BACKOFF_SECONDS as _GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
)
from mlb_luck_score.data.download_game_metadata import _get_json_with_retries
from mlb_luck_score.data.download_sprint_speed import (
    DEFAULT_MIN_OPPORTUNITIES,
    fetch_season_sprint_speed,
)
from mlb_luck_score.data.download_statcast import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_MAX_RETRIES,
    download_statcast_range,
    save_dataframe,
)
from mlb_luck_score.data.join_park_geometry import build_geometry_join_report, join_park_geometry
from mlb_luck_score.data.join_sprint_speed import (
    build_sprint_speed_join_report,
    join_sprint_speed,
    load_sprint_speed,
)
from mlb_luck_score.data.join_venue_metadata import (
    build_venue_join_report,
    join_venue_metadata,
    load_game_metadata,
)

logger = logging.getLogger(__name__)


class ProspectiveError(ValueError):
    """Raised when a Version 1.1 prospective-scoring precondition is violated."""


class ProspectiveConfigurationError(ProspectiveError):
    """Raised when a real 2026 ingestion would rely on an unverified
    configuration value.
    """


class WorkingTreeNotCleanError(ProspectiveError):
    """Raised when `run_prospective_guards` finds a dirty working tree --
    see its own docstring for exactly what counts as dirty.
    """


class IncompleteDataThroughDateError(ProspectiveError):
    """Raised when the requested `--data-through` date has one or more games
    that are not yet final -- see `assert_data_through_date_is_complete`'s
    docstring for the documented postponed/suspended-game rule.
    """


# ---------------------------------------------------------------------------
# Unforgeable-in-process authorization (mirrors Version 1.0's pattern)
# ---------------------------------------------------------------------------

_ISSUED_AUTHORIZATION_TOKENS: set[str] = set()


@dataclass(frozen=True)
class ProspectiveAuthorization:
    """An unforgeable-in-practice token proving `run_prospective_guards`
    actually ran. Ingestion functions require this OBJECT (never a bare
    `bool`). Process-local by design -- a within-process guard against a
    caller skipping the guards, not a cross-process security boundary.
    """

    token: str
    granted_at: str


def run_prospective_guards(*, repo_root: Path = PROJECT_ROOT) -> ProspectiveAuthorization:
    """Mint the one `ProspectiveAuthorization` this run will use, after
    confirming the working tree is clean.

    A dirty working tree BLOCKS this guard -- unlike Version 1.0's one-time
    seal, this doesn't check for a PRIOR seal (see `prospective.
    prospective_manifest` module docstring for why repeatable snapshots use a
    different immutability mechanism than a one-time seal), but a snapshot's
    manifest records the exact commit its code came from, and that record is
    only trustworthy if the tree was genuinely clean at run time -- an
    uncommitted change (even one never intended to affect scoring) means the
    code that actually ran cannot be reproduced from the commit hash alone.

    Reuses `evaluation.v1_final_evaluation_manifest.assert_clean_working_
    tree` UNCHANGED: it treats ANY `git status --porcelain` output as dirty
    -- both tracked changes (staged or unstaged) AND untracked-but-not
    -ignored files (an uncommitted new `.py`/config/test file counts). Files
    under `data/prospective/2026/`, `outputs/prospective/v1_1/`, and
    `artifacts/prospective/v1_1/` are gitignored (see `.gitignore`'s
    "Version 1.1 prospective 2026 scoring namespace" section) and therefore
    NEVER appear in `git status --porcelain` output at all -- a prior
    snapshot run's own output files can never falsely trip this guard.

    Raises:
        WorkingTreeNotCleanError: if the working tree is not clean.
    """
    try:
        assert_clean_working_tree(repo_root)
    except _V1ManifestError as exc:
        raise WorkingTreeNotCleanError(
            f"Refusing to run prospective scoring with a dirty working tree: {exc} A snapshot's "
            "manifest records the exact commit its code came from -- an uncommitted change "
            "(tracked or untracked-and-not-ignored) would make that record wrong, and the "
            "snapshot would not be reproducible from its own manifest. Commit or stash your "
            "changes first."
        ) from exc
    token = secrets.token_hex(16)
    _ISSUED_AUTHORIZATION_TOKENS.add(token)
    return ProspectiveAuthorization(token=token, granted_at=datetime.now(UTC).isoformat())


def _require_authorization(authorization: Any, fn_name: str) -> None:
    if not isinstance(authorization, ProspectiveAuthorization):
        raise ProspectiveError(
            f"{fn_name} requires a ProspectiveAuthorization instance minted by "
            f"run_prospective_guards -- got {type(authorization).__name__!r} (a bare boolean "
            "is never accepted, regardless of its value)."
        )
    if authorization.token not in _ISSUED_AUTHORIZATION_TOKENS:
        raise ProspectiveError(
            f"{fn_name} requires a ProspectiveAuthorization minted by run_prospective_guards -- "
            "a fabricated token, or an authorization from a different/earlier process, is never "
            "accepted."
        )


# ---------------------------------------------------------------------------
# Data-through-date completeness guard
# ---------------------------------------------------------------------------

#: MLB Stats API /schedule `status.detailedState` values that mean a game is
#: genuinely final. ANY other value -- including "In Progress", "Warmup",
#: "Pre-Game", "Delayed", "Delayed Start", "Scheduled", and "Suspended" -- is
#: treated as NOT complete. Fail-safe by design: an unrecognized status
#: string is never assumed final.
COMPLETED_GAME_STATUS_VALUES: frozenset[str] = frozenset({"Final", "Game Over", "Completed Early"})

#: A postponed or cancelled game never happened -- excluded entirely from
#: the "must be final" requirement below (there is no play to be final about,
#: so it never blocks the requested date), but always recorded separately in
#: `DataThroughDateCompletenessResult.postponed_or_cancelled_games` for
#: provenance -- never silently dropped.
POSTPONED_OR_CANCELLED_STATUS_VALUES: frozenset[str] = frozenset({"Postponed", "Cancelled"})

#: DOCUMENTED RULE for suspended games (explicitly required, not left
#: implicit): a "Suspended" game has real partial play but no final,
#: reconciled outcome -- it may resume later and its box score can still
#: change. It is deliberately EXCLUDED from `COMPLETED_GAME_STATUS_VALUES`
#: and NOT added to `POSTPONED_OR_CANCELLED_STATUS_VALUES` -- a suspended
#: game is treated exactly like an in-progress game: it blocks the requested
#: `--data-through` date until the game resumes and reaches a genuinely
#: final status. Never silently treat a suspended game as either "complete"
#: or "postponed" (those are different real-world outcomes with different
#: data implications).


@dataclass(frozen=True)
class DataThroughDateCompletenessResult:
    data_through_date: str
    is_complete: bool
    total_games_on_date: int
    incomplete_games: list[dict[str, Any]]
    postponed_or_cancelled_games: list[dict[str, Any]]
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_schedule_game_statuses(
    date_str: str,
    *,
    max_retries: int = _GAME_METADATA_DEFAULT_MAX_RETRIES,
    backoff_seconds: float = _GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch every game's status for ONE date via the MLB Stats API
    `/schedule` endpoint (the same public, no-key endpoint already used
    elsewhere in this repository for schedule/venue lookups -- see
    `mlb_luck_score.data.download_game_metadata`).

    Returns:
        A list of `{"game_pk", "game_date", "status_detailed_state"}` dicts,
        one per game scheduled on `date_str`.
    """
    data = _get_json_with_retries(
        f"{MLB_STATS_API_BASE_URL}/schedule",
        params={"sportId": 1, "gameType": "R", "startDate": date_str, "endDate": date_str},
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            games.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": date_str,
                    "status_detailed_state": (game.get("status") or {}).get("detailedState"),
                }
            )
    return games


def check_data_through_date_completeness(
    data_through_date: str,
    *,
    fetch_fn: Callable[[str], list[dict[str, Any]]] = fetch_schedule_game_statuses,
) -> DataThroughDateCompletenessResult:
    """Pure with respect to `fetch_fn` -- tests inject a synthetic fetch
    function so this (and everything built on it) is exercised without any
    real network access. A date with zero scheduled games (an off day) is
    trivially complete (`total_games_on_date == 0`).
    """
    games = fetch_fn(data_through_date)
    incomplete: list[dict[str, Any]] = []
    postponed_or_cancelled: list[dict[str, Any]] = []
    for game in games:
        status = game.get("status_detailed_state")
        if status in POSTPONED_OR_CANCELLED_STATUS_VALUES:
            postponed_or_cancelled.append(game)
        elif status not in COMPLETED_GAME_STATUS_VALUES:
            incomplete.append(game)
    return DataThroughDateCompletenessResult(
        data_through_date=data_through_date,
        is_complete=not incomplete,
        total_games_on_date=len(games),
        incomplete_games=incomplete,
        postponed_or_cancelled_games=postponed_or_cancelled,
        checked_at=datetime.now(UTC).isoformat(),
    )


def assert_data_through_date_is_complete(
    data_through_date: str,
    *,
    fetch_fn: Callable[[str], list[dict[str, Any]]] = fetch_schedule_game_statuses,
) -> DataThroughDateCompletenessResult:
    """Refuse a `--data-through` date with any game not yet final. Per the
    task's explicit instruction, this FAILS rather than silently walking
    backward to find "the preceding fully completed date" -- silently
    substituting a different date than the one requested is exactly the kind
    of silent repair this codebase avoids elsewhere (see CLAUDE.md's
    "fails silently" bug-class warnings); the caller must explicitly choose
    an earlier date and rerun.

    Raises:
        IncompleteDataThroughDateError: if any game on `data_through_date`
            is not final (including suspended games -- see module-level
            docstring above for the documented rule).
    """
    result = check_data_through_date_completeness(data_through_date, fetch_fn=fetch_fn)
    if not result.is_complete:
        statuses = [g.get("status_detailed_state") for g in result.incomplete_games]
        raise IncompleteDataThroughDateError(
            f"--data-through {data_through_date} has {len(result.incomplete_games)} game(s) not "
            f"yet final (statuses observed: {statuses}) -- refusing to score a date with games "
            "still in progress or suspended. Choose an earlier --data-through date -- the most "
            "recent date with no incomplete games -- and rerun."
        )
    return result


# ---------------------------------------------------------------------------
# 2026 ingestion
# ---------------------------------------------------------------------------


def ingest_2026_raw_statcast(
    *,
    authorization: ProspectiveAuthorization,
    data_through_date: str,
    season_start_date: str = PROSPECTIVE_2026_SEASON_START_DATE.isoformat(),
    output_dir: Path = PROSPECTIVE_RAW_DIR,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download raw 2026 Statcast rows via `download_statcast_range` directly
    -- never via `mlb_luck_score.data.download_development_data`, which has
    no (and must never gain) a 2026 entry in `MLB_REGULAR_SEASON_DATE_
    RANGES`. Returns a provenance record.

    `season_start_date` defaults to the VERIFIED `PROSPECTIVE_2026_SEASON_
    START_DATE` (2026-03-25, MLB Opening Night -- see `prospective_config`
    for the source citation and verification date). The `ProspectiveConfig
    urationError` guard below is retained as defense-in-depth for a future
    edit that resets `PROSPECTIVE_2026_SEASON_START_VERIFIED` to `False`
    without updating this default -- under the current, verified
    configuration it never fires.

    Raises:
        ProspectiveConfigurationError: if a REAL download (no existing cache,
            or `overwrite=True`) would be attempted while `season_start_date`
            matches the configured default AND `PROSPECTIVE_2026_SEASON_
            START_VERIFIED` is False.
    """
    _require_authorization(authorization, "ingest_2026_raw_statcast")
    _assert_within_namespace(
        output_dir, PROSPECTIVE_RAW_DIR, label="ingest_2026_raw_statcast output_dir"
    )

    path = development_raw_path(output_dir, PROSPECTIVE_SEASON)
    date_range = (season_start_date, data_through_date)
    start, end = date.fromisoformat(season_start_date), date.fromisoformat(data_through_date)
    if end < start:
        raise ProspectiveError(
            f"data_through_date {data_through_date} is before season_start_date "
            f"{season_start_date}."
        )
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2026 raw statcast cache already exists at %s; skipping download.", path)
    else:
        if (
            season_start_date == PROSPECTIVE_2026_SEASON_START_DATE.isoformat()
            and not PROSPECTIVE_2026_SEASON_START_VERIFIED
        ):
            raise ProspectiveConfigurationError(
                "Refusing a real 2026 download: PROSPECTIVE_2026_SEASON_START_VERIFIED is "
                "False for the configured season-start date. A maintainer must confirm the "
                "real 2026 Opening Day against the authoritative MLB schedule, then either "
                "pass a verified season_start_date explicitly or flip PROSPECTIVE_2026_SEASON_"
                "START_VERIFIED to True with a citation (see prospective_config.py)."
            )
        df = download_statcast_range(start, end, chunk_days=chunk_days, max_retries=max_retries)
        if df.empty:
            raise ProspectiveError(
                f"No rows downloaded for 2026 range {date_range} -- refusing to write an "
                "empty file."
            )
        save_dataframe(df, path)

    raw = pd.read_parquet(path)
    schema_check = assert_raw_statcast_schema_compatible(raw)

    observed_dates = set(pd.to_datetime(raw["game_date"]).dt.date.unique()) if len(raw) else set()
    expected_dates = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    missing_dates = sorted(d.isoformat() for d in expected_dates - observed_dates)

    return {
        "source": "pybaseball.statcast (Baseball Savant)",
        "requested_date_range": list(date_range),
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "row_count": int(len(raw)),
        "observed_date_coverage": (
            [str(min(observed_dates)), str(max(observed_dates))] if observed_dates else []
        ),
        "missing_dates_within_range": missing_dates,
        "schema_check": schema_check,
    }


def ingest_2026_game_metadata(
    *,
    authorization: ProspectiveAuthorization,
    data_through_date: str,
    season_start_date: str = PROSPECTIVE_2026_SEASON_START_DATE.isoformat(),
    output_dir: Path = PROSPECTIVE_RAW_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch 2026 per-game venue/roof/surface metadata directly from the MLB
    Stats API, reusing `_build_prospective_game_metadata` (a genuinely
    date-range-parameterized, season-agnostic function).
    """
    _require_authorization(authorization, "ingest_2026_game_metadata")
    _assert_within_namespace(
        output_dir, PROSPECTIVE_RAW_DIR, label="ingest_2026_game_metadata output_dir"
    )

    path = game_metadata_path(output_dir, PROSPECTIVE_SEASON)
    date_range = (season_start_date, data_through_date)
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2026 game-metadata cache already exists at %s; skipping fetch.", path)
    else:
        games_df = _build_prospective_game_metadata(
            date_range,
            chunk_days=_GAME_METADATA_DEFAULT_CHUNK_DAYS,
            max_retries=_GAME_METADATA_DEFAULT_MAX_RETRIES,
            backoff_seconds=_GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
        )
        if games_df.empty:
            raise ProspectiveError("No 2026 games found -- refusing to write an empty file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        games_df.to_parquet(path, index=False)

    saved = pd.read_parquet(path)
    return {
        "source": "MLB Stats API (/schedule, /venues)",
        "requested_date_range": list(date_range),
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "game_count": int(len(saved)),
    }


def ingest_2026_sprint_speed(
    *,
    authorization: ProspectiveAuthorization,
    output_dir: Path = PROSPECTIVE_RAW_DIR,
    min_opp: int = DEFAULT_MIN_OPPORTUNITIES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch the 2026 Sprint Speed leaderboard directly via `pybaseball.
    statcast_sprint_speed` -- `fetch_season_sprint_speed` has no season-list
    gate or protected-dict dependency, so it is reusable as-is.
    """
    _require_authorization(authorization, "ingest_2026_sprint_speed")
    _assert_within_namespace(
        output_dir, PROSPECTIVE_RAW_DIR, label="ingest_2026_sprint_speed output_dir"
    )

    path = sprint_speed_path(output_dir, PROSPECTIVE_SEASON)
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2026 sprint-speed cache already exists at %s; skipping fetch.", path)
    else:
        season_df = fetch_season_sprint_speed(PROSPECTIVE_SEASON, min_opp=min_opp)
        if season_df.empty:
            raise ProspectiveError(
                "No 2026 sprint-speed rows returned -- refusing to write an empty file."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(path, index=False)

    saved = pd.read_parquet(path)
    return {
        "source": "pybaseball.statcast_sprint_speed (Baseball Savant Sprint Speed leaderboard)",
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "player_count": int(len(saved)),
        "min_opp": min_opp,
    }


def build_2026_scoring_dataset(
    *, authorization: ProspectiveAuthorization, raw_dir: Path = PROSPECTIVE_RAW_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the isolated raw 2026 statcast file (reusing `clean_development_
    data` UNCHANGED, pointed at `raw_dir` so its output never touches
    `data/processed`) then join venue/geometry/sprint-speed exactly as the
    development pipeline does for 2021-2024.
    """
    _require_authorization(authorization, "build_2026_scoring_dataset")
    _assert_within_namespace(
        raw_dir, PROSPECTIVE_RAW_DIR, label="build_2026_scoring_dataset raw_dir"
    )

    cleaned = clean_development_data([PROSPECTIVE_SEASON], raw_dir)
    if cleaned.empty:
        raise ProspectiveError("Cleaned 2026 dataset is empty.")

    metadata = load_game_metadata(raw_dir, [PROSPECTIVE_SEASON])
    with_venue = join_venue_metadata(cleaned, metadata)
    venue_report = build_venue_join_report(with_venue, metadata)

    with_geometry = join_park_geometry(with_venue)
    geometry_report = build_geometry_join_report(with_geometry)

    sprint_speed_df = load_sprint_speed(raw_dir, [PROSPECTIVE_SEASON])
    with_sprint_speed = join_sprint_speed(with_geometry, sprint_speed_df)
    sprint_speed_report = build_sprint_speed_join_report(with_sprint_speed)

    provenance = {
        "venue_join": asdict(venue_report),
        "geometry_join": asdict(geometry_report),
        "sprint_speed_join": asdict(sprint_speed_report),
    }
    return with_sprint_speed, provenance
