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
documented postponed/suspended-game rule, and
`assert_data_through_date_has_completed_games` for the companion guard that
refuses a date which completed no games at all (the season-end case).
"""

from __future__ import annotations

import json
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
    PROSPECTIVE_2026_SEASON_END_DATE,
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
    build_date_chunks,
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


class NoCompletedGamesToScoreError(ProspectiveError):
    """Raised when the requested `--data-through` date completed NO games at
    all -- see `assert_data_through_date_has_completed_games`. Distinct from
    `IncompleteDataThroughDateError`: that one means "come back when the
    slate finishes," this one means "there was never anything here to score."
    """


class RecordedSeasonEndDateStaleError(ProspectiveError):
    """Raised when real games completed AFTER `PROSPECTIVE_2026_SEASON_END_
    DATE` -- the recorded fact and the live schedule disagree, so the
    recorded fact is stale. See `assert_data_through_date_agrees_with_
    recorded_season_end`.
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

    @property
    def completed_games_on_date(self) -> int:
        """How many games on this date genuinely finished.

        A derived PROPERTY, never a dataclass field, on purpose: `to_dict()`
        is `asdict()` and flows into every snapshot manifest's
        `schema_checks`, and manifests are content-hashed to tell an
        idempotent rerun from a conflict. Adding a field here would change
        the recorded shape for every already-archived snapshot, turning a
        legitimate rerun of an archived date into a `SnapshotConflictError`.
        A property stays out of `asdict()`. See `tests/
        test_prospective_no_completed_games_guard.py`.
        """
        return (
            self.total_games_on_date
            - len(self.incomplete_games)
            - len(self.postponed_or_cancelled_games)
        )

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


def assert_data_through_date_agrees_with_recorded_season_end(
    result: DataThroughDateCompletenessResult,
    *,
    season_end_date: date = PROSPECTIVE_2026_SEASON_END_DATE,
) -> None:
    """Cross-check the recorded season-finale date against what the schedule
    actually shows. Fires ONLY when the two disagree.

    `PROSPECTIVE_2026_SEASON_END_DATE` is a maintainer-cited fact, and facts
    recorded in source go stale. The failure this catches is a real game
    played after the recorded finale -- a rainout makeup, or simply a wrong
    constant.

    DELIBERATELY NOT A CUTOFF. Refusing every date after the recorded finale
    would be the obvious implementation and the wrong one: a makeup game
    played on the 28th is a genuine regular-season game whose plays belong in
    the season's totals, and a blind cutoff would drop it silently and
    forever. So this stays quiet about dates that completed no games -- the
    ordinary off-season case, owned by `assert_data_through_date_has_
    completed_games` and its clean exit 3 -- and speaks up only when real
    play exists past the recorded end.

    It is loud (the generic failure exit code 2, not the quiet exit 3)
    because a stale recorded fact needs a human: per the rule in
    `prospective_config`, the date, its source citation, and its verification
    date are updated together, and only a maintainer can supply the new
    citation.

    Raises:
        RecordedSeasonEndDateStaleError: if any game completed on a
            `data_through_date` later than `season_end_date`.
    """
    if result.completed_games_on_date <= 0:
        return
    requested = date.fromisoformat(result.data_through_date)
    if requested <= season_end_date:
        return
    raise RecordedSeasonEndDateStaleError(
        f"--data-through {result.data_through_date} completed "
        f"{result.completed_games_on_date} game(s), but the recorded regular-season finale is "
        f"{season_end_date.isoformat()} -- real play exists after the date this repository has "
        "on record, so PROSPECTIVE_2026_SEASON_END_DATE is stale. This is refused rather than "
        "guessed in either direction: scoring it would extend the season past a recorded fact, "
        "and skipping it would silently drop real games. Update PROSPECTIVE_2026_SEASON_END_"
        "DATE, its _SOURCE citation, and its _VERIFIED_AT together (never one without the "
        "others), then rerun."
    )


def assert_data_through_date_has_completed_games(
    result: DataThroughDateCompletenessResult,
) -> None:
    """Refuse a `--data-through` date on which NO game was actually played to
    completion.

    Takes the result `assert_data_through_date_is_complete` already returned
    rather than re-fetching: the schedule call this needs has, by
    construction, just been made. This guard costs no additional network
    request.

    WHY THIS EXISTS -- the season-end duplicate-snapshot defect. Nothing else
    in this pipeline notices that a season has ended:

      - `check_data_through_date_completeness` treats a date with zero
        scheduled games as trivially complete, which is correct (there is
        nothing unfinished about a day with no baseball) but means an off
        day and a post-season date both sail through it.
      - Both schedule fetches filter `gameType="R"`, so the postseason is
        invisible here -- October never registers as "games happening."
      - `assert_scoring_dataset_satisfies_coverage_contract` asks whether any
        COMPLETED date is missing from the data. On a date with no completed
        games there is nothing to be missing, so it passes.
      - The `observed_max_date > data_through` check in `run_v1_1_2026_
        scoring` only fires in the opposite direction (data PAST the cutoff).

    So every date after the final regular-season game would re-score
    byte-identical data into a NEW snapshot directory and a NEW write-once
    archive key, one per day, indefinitely -- and, with scheduled deploys
    enabled, publish a site whose `data_through_date` advances onto days no
    baseball was played.

    This FAILS rather than silently walking backward to the last date that
    did have games, exactly as its sibling guard does and for the same
    reason: the caller must choose the date explicitly. `run_v1_1_2026_
    scoring.main` maps it to a distinct exit code so the scheduled publish
    loop can treat "nothing new to score" as a clean stop instead of an
    error (see `scripts/publish_snapshot.sh`).

    A fully postponed slate is refused for the same reason -- a postponed
    game never happened, so such a date carries no new play either.

    Raises:
        NoCompletedGamesToScoreError: if no game on the date reached a final
            status.
    """
    if result.completed_games_on_date > 0:
        return
    postponed = len(result.postponed_or_cancelled_games)
    detail = (
        f"all {postponed} scheduled game(s) were postponed or cancelled"
        if postponed
        else "no games were scheduled"
    )
    raise NoCompletedGamesToScoreError(
        f"--data-through {result.data_through_date} completed no games ({detail}) -- there is "
        "no new play to score, and scoring it would duplicate the previous game date's "
        "snapshot under a new name. If the regular season has ended, this date is past it. "
        "Choose a --data-through date on which games were actually played."
    )


# ---------------------------------------------------------------------------
# Version 1.1.2: coverage-aware raw Statcast cache validation
#
# Incident this fixes: `data/prospective/2026/statcast_2026_regular_season.
# parquet` was downloaded once and then silently reused across three later
# `--data-through` requests (2026-08-06, the first 2026-08-08 attempt) whose
# ACTUAL coverage had moved on -- the old guard only asked "does the file
# exist", never "does it actually cover what was requested". Every
# function below exists so existence of the cache file is NEVER, by itself,
# sufficient to trust it -- see CLAUDE.md "Version 1.1.2" for the full
# incident writeup and the numbered rules this implements.
# ---------------------------------------------------------------------------


def fetch_schedule_game_statuses_range(
    start_date: str,
    end_date: str,
    *,
    chunk_days: int = _GAME_METADATA_DEFAULT_CHUNK_DAYS,
    max_retries: int = _GAME_METADATA_DEFAULT_MAX_RETRIES,
    backoff_seconds: float = _GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
) -> list[dict[str, Any]]:
    """Fetch every game's status across `[start_date, end_date]` (inclusive),
    chunked via the MLB Stats API `/schedule` endpoint -- generalizes
    `fetch_schedule_game_statuses` (single date) to a full range. Used ONLY
    by the cache-coverage validator and the final pre-scoring coverage
    assertion below -- NEVER a source of Statcast play-level data itself.
    """
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    games: list[dict[str, Any]] = []
    for chunk in build_date_chunks(start, end, chunk_days):
        data = _get_json_with_retries(
            f"{MLB_STATS_API_BASE_URL}/schedule",
            params={
                "sportId": 1,
                "gameType": "R",
                "startDate": chunk.start.isoformat(),
                "endDate": chunk.end.isoformat(),
            },
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        for day in data.get("dates", []):
            day_date = day.get("date")
            for game in day.get("games", []):
                games.append(
                    {
                        "game_pk": game.get("gamePk"),
                        "game_date": day_date,
                        "status_detailed_state": (game.get("status") or {}).get("detailedState"),
                    }
                )
    return games


def completed_mlb_game_dates(
    start_date: str,
    end_date: str,
    *,
    fetch_fn: Callable[[str, str], list[dict[str, Any]]] = fetch_schedule_game_statuses_range,
) -> set[str]:
    """The set of date strings within `[start_date, end_date]` that have AT
    LEAST ONE game with a status in `COMPLETED_GAME_STATUS_VALUES`. A date
    with zero games, or only postponed/cancelled/non-final games, is
    deliberately EXCLUDED -- rule 5: a date with no completed games is never
    a missing-data failure. This is DELIBERATELY more thorough than
    comparing `max(game_date)` alone (rule 4) -- it returns every individual
    qualifying date, so a caller can detect an INTERNAL gap (an earlier date
    missing from the cache even though a later date is present).
    """
    games = fetch_fn(start_date, end_date)
    return {
        g["game_date"]
        for g in games
        if g.get("game_date") and g.get("status_detailed_state") in COMPLETED_GAME_STATUS_VALUES
    }


def _raw_statcast_provenance_path(raw_path: Path) -> Path:
    return raw_path.parent / f"{raw_path.stem}.provenance.json"


class RawStatcastProvenanceError(ProspectiveError):
    """Raised by `read_raw_statcast_provenance` when the sidecar exists but
    cannot be parsed. Never swallowed by that function itself -- callers
    that want a refresh-on-malformed policy (`evaluate_raw_statcast_cache`)
    catch it explicitly.
    """


@dataclass(frozen=True)
class RawStatcastCacheProvenance:
    """Coverage provenance for the raw 2026 Statcast cache, persisted as a
    JSON sidecar next to the cached parquet file (`_raw_statcast_provenance_
    path`). `observed_game_dates` is the FULL set of distinct dates present
    in the cache, not just min/max -- rule 4 explicitly forbids validating
    coverage with `max(game_date)` alone, since a later date being present
    can otherwise hide an earlier internal gap.
    """

    requested_start_date: str
    requested_end_date: str
    observed_min_game_date: str | None
    observed_max_game_date: str | None
    observed_game_dates: list[str]
    retrieved_at: str
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawStatcastCacheProvenance:
        return cls(**data)


def build_raw_statcast_provenance(
    raw_path: Path, *, requested_start_date: str, requested_end_date: str, retrieved_at: str
) -> RawStatcastCacheProvenance:
    raw = pd.read_parquet(raw_path)
    observed_dates = (
        sorted({d.isoformat() for d in pd.to_datetime(raw["game_date"]).dt.date.unique()})
        if len(raw)
        else []
    )
    return RawStatcastCacheProvenance(
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        observed_min_game_date=observed_dates[0] if observed_dates else None,
        observed_max_game_date=observed_dates[-1] if observed_dates else None,
        observed_game_dates=observed_dates,
        retrieved_at=retrieved_at,
        row_count=int(len(raw)),
        sha256=compute_file_sha256(raw_path),
    )


def write_raw_statcast_provenance(
    provenance: RawStatcastCacheProvenance, provenance_path: Path
) -> None:
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance.to_dict(), indent=2))


def read_raw_statcast_provenance(provenance_path: Path) -> RawStatcastCacheProvenance | None:
    """`None` only when the sidecar file does not exist at all -- a sidecar
    that exists but cannot be parsed raises `RawStatcastProvenanceError`
    rather than being silently treated as absent (rule 3).
    """
    if not provenance_path.exists():
        return None
    try:
        data = json.loads(provenance_path.read_text())
        return RawStatcastCacheProvenance.from_dict(data)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RawStatcastProvenanceError(
            f"Raw statcast provenance sidecar at {provenance_path} exists but could not be "
            f"parsed into a RawStatcastCacheProvenance: {exc}"
        ) from exc


#: `CacheCoverageValidation.decision` values (rule 10: manifest records
#: "whether the cache was reused or refreshed").
CACHE_DECISION_REUSED = "reused"
CACHE_DECISION_REFRESHED = "refreshed"

#: `CacheCoverageValidation.reason` values (rule 10: "refresh reason when
#: applicable", also used for the reuse case).
REFRESH_REASON_FORCE_REDOWNLOAD = "force_redownload_requested"
REFRESH_REASON_NO_CACHE_FILE = "no_cached_raw_file"
REFRESH_REASON_NO_PROVENANCE = "no_provenance_sidecar"
REFRESH_REASON_MALFORMED_PROVENANCE = "provenance_sidecar_malformed"
REFRESH_REASON_HASH_MISMATCH = "raw_file_hash_does_not_match_provenance"
REFRESH_REASON_FORWARD_COVERAGE = "requested_data_through_beyond_cached_coverage"
REFRESH_REASON_COVERAGE_GAP = "completed_mlb_game_dates_missing_from_cached_coverage"
REUSE_REASON_VERIFIED = "cached_coverage_verified_sufficient"


@dataclass(frozen=True)
class CacheCoverageValidation:
    decision: str
    reason: str
    cached_provenance: dict[str, Any] | None
    missing_completed_game_dates: list[str]
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_raw_statcast_cache(
    *,
    raw_path: Path,
    provenance_path: Path,
    requested_start_date: str,
    requested_end_date: str,
    force_redownload: bool = False,
    completed_dates_fetch_fn: Callable[
        [str, str], list[dict[str, Any]]
    ] = fetch_schedule_game_statuses_range,
) -> CacheCoverageValidation:
    """Decide whether the cached raw Statcast file may be REUSED or must be
    REFRESHED. Existence of the file is NEVER sufficient on its own.

    Decision order (numbered rules from the Version 1.1.2 task):
      7. `force_redownload=True` -> REFRESH unconditionally.
      3. No cache file, no provenance sidecar, a malformed sidecar, or a
         sidecar whose recorded sha256 no longer matches the actual raw
         file -> REFRESH, never silently reused.
      2/4/5. A single unified check: every completed MLB game date from
         `requested_start_date` through `requested_end_date` (not just the
         tail -- rule 4) must be present in the cache's own `observed_game_
         dates`. A date with zero completed games is never counted as a
         missing date (rule 5). Deliberately NOT a separate "is requested_
         end_date > observed_max_game_date" shortcut: that naive comparison
         is WRONG whenever the requested cutoff date itself has zero
         completed games (postponed slate, a genuine off day) -- the cache's
         real `observed_max_game_date` legitimately lags the calendar in
         that case even though coverage is complete. Any date missing this
         way STILL triggers REFRESH -- reported as `REFRESH_REASON_FORWARD_
         COVERAGE` if every missing date is newer than the cache's own
         `observed_max_game_date` (the ordinary "cache needs to catch up"
         case), or `REFRESH_REASON_COVERAGE_GAP` if any missing date is
         older (an internal gap, hidden behind a newer date that IS
         present -- the actual real-world incident this fixes).
      1/6. Otherwise (no completed date is missing) -> REUSE.
      8. Cache AGE is never consulted here -- only coverage.
    """
    checked_at = datetime.now(UTC).isoformat()

    if force_redownload:
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, REFRESH_REASON_FORCE_REDOWNLOAD, None, [], checked_at
        )

    if not raw_path.exists():
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, REFRESH_REASON_NO_CACHE_FILE, None, [], checked_at
        )

    if not provenance_path.exists():
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, REFRESH_REASON_NO_PROVENANCE, None, [], checked_at
        )

    try:
        provenance = read_raw_statcast_provenance(provenance_path)
    except RawStatcastProvenanceError:
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, REFRESH_REASON_MALFORMED_PROVENANCE, None, [], checked_at
        )
    if provenance is None:  # pragma: no cover -- existence just checked above
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, REFRESH_REASON_NO_PROVENANCE, None, [], checked_at
        )

    actual_hash = compute_file_sha256(raw_path)
    if actual_hash != provenance.sha256:
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED,
            REFRESH_REASON_HASH_MISMATCH,
            provenance.to_dict(),
            [],
            checked_at,
        )

    completed_dates = completed_mlb_game_dates(
        requested_start_date, requested_end_date, fetch_fn=completed_dates_fetch_fn
    )
    missing = sorted(completed_dates - set(provenance.observed_game_dates))
    if missing:
        cached_max = provenance.observed_max_game_date or ""
        reason = (
            REFRESH_REASON_FORWARD_COVERAGE
            if all(d > cached_max for d in missing)
            else REFRESH_REASON_COVERAGE_GAP
        )
        return CacheCoverageValidation(
            CACHE_DECISION_REFRESHED, reason, provenance.to_dict(), missing, checked_at
        )

    return CacheCoverageValidation(
        CACHE_DECISION_REUSED, REUSE_REASON_VERIFIED, provenance.to_dict(), [], checked_at
    )


class CoverageContractViolationError(ProspectiveError):
    """Raised by the final, independent pre-scoring coverage check -- see
    `assert_scoring_dataset_satisfies_coverage_contract`.
    """


def assert_scoring_dataset_satisfies_coverage_contract(
    scoring_df: pd.DataFrame,
    *,
    season_start_date: str,
    data_through_date: str,
    completed_dates_fetch_fn: Callable[
        [str, str], list[dict[str, Any]]
    ] = fetch_schedule_game_statuses_range,
) -> dict[str, Any]:
    """Rule 9: immediately before model scoring, independently re-verify
    that the ACTUAL dataset about to be scored covers every completed MLB
    game date through the requested cutoff -- derived FROM `scoring_df`
    itself, with a FRESH schedule fetch. Deliberately does NOT read or trust
    any earlier `CacheCoverageValidation` result: this is a genuine second,
    independent check, so a bug (or an incorrectly mocked cache decision
    upstream, e.g. in a test) cannot silently let stale data reach scoring.
    """
    if scoring_df.empty:
        observed_dates: set[str] = set()
    else:
        observed_dates = {
            d.isoformat() for d in pd.to_datetime(scoring_df["game_date"]).dt.date.unique()
        }
    completed_dates = completed_mlb_game_dates(
        season_start_date, data_through_date, fetch_fn=completed_dates_fetch_fn
    )
    missing = sorted(completed_dates - observed_dates)
    if missing:
        raise CoverageContractViolationError(
            "Final pre-scoring coverage check failed: the scoring dataset is missing "
            f"{len(missing)} completed MLB game date(s) through {data_through_date}: {missing} "
            "-- refusing to score. This check is independent of, and does not trust, any "
            "earlier cache-validation decision."
        )
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "observed_game_date_count": len(observed_dates),
        "missing_completed_game_dates": [],
    }


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
    completed_dates_fetch_fn: Callable[
        [str, str], list[dict[str, Any]]
    ] = fetch_schedule_game_statuses_range,
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

    Version 1.1.2: whether the existing cache file may be REUSED is decided
    by `evaluate_raw_statcast_cache` -- coverage, not mere file existence,
    determines validity (see that function's docstring for the exact
    decision rules). `overwrite` (`--force-redownload`) still forces a
    refresh but is no longer required for an ordinary forward-moving
    request; the cache now refreshes itself automatically in that case.

    Raises:
        ProspectiveConfigurationError: if a REAL download (cache refresh)
            would be attempted while `season_start_date` matches the
            configured default AND `PROSPECTIVE_2026_SEASON_START_VERIFIED`
            is False.
    """
    _require_authorization(authorization, "ingest_2026_raw_statcast")
    _assert_within_namespace(
        output_dir, PROSPECTIVE_RAW_DIR, label="ingest_2026_raw_statcast output_dir"
    )

    path = development_raw_path(output_dir, PROSPECTIVE_SEASON)
    provenance_path = _raw_statcast_provenance_path(path)
    date_range = (season_start_date, data_through_date)
    start, end = date.fromisoformat(season_start_date), date.fromisoformat(data_through_date)
    if end < start:
        raise ProspectiveError(
            f"data_through_date {data_through_date} is before season_start_date "
            f"{season_start_date}."
        )
    retrieved_at = datetime.now(UTC).isoformat()

    cache_validation = evaluate_raw_statcast_cache(
        raw_path=path,
        provenance_path=provenance_path,
        requested_start_date=season_start_date,
        requested_end_date=data_through_date,
        force_redownload=overwrite,
        completed_dates_fetch_fn=completed_dates_fetch_fn,
    )

    if cache_validation.decision == CACHE_DECISION_REUSED:
        logger.info("Reusing verified raw statcast cache at %s (%s)", path, cache_validation.reason)
        final_provenance_dict = cache_validation.cached_provenance
    else:
        logger.info("Refreshing raw statcast cache at %s (%s)", path, cache_validation.reason)
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
        new_provenance = build_raw_statcast_provenance(
            path,
            requested_start_date=season_start_date,
            requested_end_date=data_through_date,
            retrieved_at=retrieved_at,
        )
        write_raw_statcast_provenance(new_provenance, provenance_path)
        final_provenance_dict = new_provenance.to_dict()

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
        "cache_coverage_validation": cache_validation.to_dict(),
        "cached_provenance_before_decision": cache_validation.cached_provenance,
        "final_raw_data_provenance": final_provenance_dict,
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
