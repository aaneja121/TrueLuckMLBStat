"""Look up player display names from the public MLB Stats API, by MLBAM ID.

General-purpose, reviewed infrastructure -- not season-gated, not a frozen
Version 0.2-1.0 artifact. Added to support Version 1.1's presentation-only
batter-ID-to-name join (`prospective.prospective_player_names`): the public
score table already carries a nullable `batter_name` column (see
`mlb_luck_score.scoring.public_score_schema`) that every earlier version left
unpopulated. This module only resolves a display name for an already-known
MLBAM batter ID -- it computes nothing, joins nothing into model features, and
has no season concept at all.

Uses the same public MLB Stats API (`mlb_luck_score.config.
MLB_STATS_API_BASE_URL`, no API key required) already used by
`mlb_luck_score.data.download_game_metadata` for schedule/venue lookups, via
the bulk `/people?personIds=...` endpoint. Retry/backoff pattern mirrors
`download_game_metadata._get_json_with_retries`.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from mlb_luck_score.config import MLB_STATS_API_BASE_URL

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30

#: The bulk /people endpoint accepts a comma-separated personIds list; kept
#: modest to avoid an overlong query string / server-side truncation.
DEFAULT_BATCH_SIZE = 100

PLAYER_NAME_SOURCE = "MLB Stats API (/people)"

PLAYER_NAME_COLUMNS: tuple[str, ...] = ("batter_id", "full_name", "retrieved_at", "source")


class DownloadPlayerNamesError(ValueError):
    """Raised when player-name lookup cannot proceed."""


def _get_json_with_retries(
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except requests.exceptions.RequestException as exc:  # noqa: PERF203
            last_error = exc
            logger.warning(
                "Request to %s failed on attempt %d/%d: %s", url, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts") from last_error


def _fetch_people_batch(person_ids: list[int], **retry_kwargs: Any) -> dict[int, str]:
    """One `/people?personIds=...` call -- `{person_id: full_name}` for every
    ID the API actually returned a person for. IDs the API has no record of
    are simply absent from the returned dict (never guessed).
    """
    if not person_ids:
        return {}
    data = _get_json_with_retries(
        f"{MLB_STATS_API_BASE_URL}/people",
        params={"personIds": ",".join(str(pid) for pid in person_ids)},
        **retry_kwargs,
    )
    resolved: dict[int, str] = {}
    for person in data.get("people", []):
        person_id = person.get("id")
        full_name = person.get("fullName")
        if person_id is not None and full_name:
            resolved[int(person_id)] = str(full_name)
    return resolved


def fetch_player_names(
    batter_ids: list[int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> pd.DataFrame:
    """Resolve display names for the given MLBAM batter IDs.

    Returns:
        A DataFrame with one row per UNIQUE input ID
        (`PLAYER_NAME_COLUMNS`: `batter_id`, `full_name`, `retrieved_at`,
        `source`). `full_name` is `None` (never guessed or fabricated) for
        any ID the API did not resolve -- callers must treat those as
        unresolved, not silently drop them.
    """
    unique_ids = sorted({int(b) for b in batter_ids})
    retrieved_at = datetime.now(UTC).isoformat()
    resolved: dict[int, str] = {}
    for start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[start : start + batch_size]
        resolved.update(
            _fetch_people_batch(batch, max_retries=max_retries, backoff_seconds=backoff_seconds)
        )

    return pd.DataFrame(
        {
            "batter_id": pd.array(unique_ids, dtype="int64"),
            "full_name": pd.array([resolved.get(bid) for bid in unique_ids], dtype="string"),
            "retrieved_at": retrieved_at,
            "source": PLAYER_NAME_SOURCE,
        }
    )
