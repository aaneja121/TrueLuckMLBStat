"""2027 data ingestion and scoring-dataset build for the one-shot v1.3 comparison.

Only `run_v1_3_2027_comparison` calls this, and only after `v1_3_2027_config.run_guards`
passes (RESEARCH_RULES.md, "2027: a pre-registered test season"). Every network fetcher
is injectable; tests never touch the network.

The pipeline reuses the development code unchanged: `clean_development_data` (pointed at
the 2027 namespace), `join_venue_metadata`, `build_game_weather_table` and
`join_weather_features`. The weather downloaders' season-level entry points read dates from
the frozen `MLB_REGULAR_SEASON_DATE_RANGES`, which must never gain 2027, so this module
calls their date-range fetchers directly, as the 2026 and sealed-2025 paths already do.

Venues missing from the frozen `venue_environment` table (in 2027: Sutter Health Park and
Las Vegas Ballpark) get no station observations, so their air density is missing. The
model handles missing density natively; `venues_without_weather_station` reports them.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import v1_3_2027_config as cfg
from prospective_ingestion import (
    _GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
    _GAME_METADATA_DEFAULT_CHUNK_DAYS,
    _GAME_METADATA_DEFAULT_MAX_RETRIES,
    _build_prospective_game_metadata,
)

from mlb_luck_score.config import (
    development_raw_path,
    game_metadata_path,
    schedule_weather_path,
    station_weather_path,
)
from mlb_luck_score.data.build_game_weather import (
    build_game_weather_table,
    load_roof_type_by_game,
    load_schedule_weather,
    load_station_observations,
)
from mlb_luck_score.data.clean_development_data import clean_development_data
from mlb_luck_score.data.download_historical_weather import (
    ASOS_REQUEST_SPACING_SECONDS,
    SCHEDULE_WEATHER_COLUMNS,
    _fetch_asos_csv,
    _fetch_schedule_weather_chunk,
    _parse_schedule_weather_game,
    parse_asos_csv,
)
from mlb_luck_score.data.download_statcast import build_date_chunks, download_statcast_range
from mlb_luck_score.data.join_venue_metadata import (
    build_venue_join_report,
    join_venue_metadata,
    load_game_metadata,
)
from mlb_luck_score.data.join_weather_features import (
    build_weather_join_report,
    join_weather_features,
)
from mlb_luck_score.data.venue_environment import VENUE_ENVIRONMENTS, get_venue_environment

logger = logging.getLogger(__name__)

SEASON = cfg.SEASON_2027


class IngestionError(ValueError):
    """Unrecoverable 2027 ingestion problem (never silently papered over)."""


# --- default (real) fetchers ---------------------------------------------------------


def _default_metadata(start: date, end: date) -> pd.DataFrame:
    return _build_prospective_game_metadata(
        (start.isoformat(), end.isoformat()),
        chunk_days=_GAME_METADATA_DEFAULT_CHUNK_DAYS,
        max_retries=_GAME_METADATA_DEFAULT_MAX_RETRIES,
        backoff_seconds=_GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
    )


def _default_schedule_weather(start: date, end: date) -> pd.DataFrame:
    games: list[dict[str, Any]] = []
    for chunk in build_date_chunks(start, end, _GAME_METADATA_DEFAULT_CHUNK_DAYS):
        games.extend(_fetch_schedule_weather_chunk(chunk.start, chunk.end))
    df = pd.DataFrame(
        [_parse_schedule_weather_game(g) for g in games], columns=list(SCHEDULE_WEATHER_COLUMNS)
    )
    return (
        df.drop_duplicates(subset=["game_pk"], keep="first")
        .sort_values("game_pk")
        .reset_index(drop=True)
    )


def _default_asos(station: str, start: date, end: date) -> pd.DataFrame:
    df = parse_asos_csv(_fetch_asos_csv(station, start, end))
    time.sleep(ASOS_REQUEST_SPACING_SECONDS)
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    cfg.assert_in_namespace(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {"path": str(path), "rows": len(df), "sha256": _sha256(path)}


def all_station_ids() -> list[str]:
    return sorted({v.weather_station_id for v in VENUE_ENVIRONMENTS})


def venues_without_weather_station(venue_ids: Sequence[int]) -> list[int]:
    return sorted(int(v) for v in set(venue_ids) if get_venue_environment(int(v)) is None)


def ingest_2027(
    start: date,
    end: date,
    *,
    raw_dir: Path = cfg.RAW_DIR,
    statcast_fn: Callable[[date, date], pd.DataFrame] = download_statcast_range,
    metadata_fn: Callable[[date, date], pd.DataFrame] = _default_metadata,
    schedule_weather_fn: Callable[[date, date], pd.DataFrame] = _default_schedule_weather,
    asos_fn: Callable[[str, date, date], pd.DataFrame] = _default_asos,
    stations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Download the 2027 regular season once into the namespace; return provenance."""
    cfg.assert_in_namespace(raw_dir)
    retrieved_at = datetime.now(UTC).isoformat()

    raw = statcast_fn(start, end)
    if raw.empty:
        raise IngestionError("no Statcast rows downloaded for 2027; refusing to continue")
    dates = pd.to_datetime(raw["game_date"]).dt.date
    keep = (dates >= start) & (dates <= end)
    if "game_type" in raw.columns:
        keep &= raw["game_type"].fillna("R") == "R"
    dropped = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    if raw.empty:
        raise IngestionError("no Statcast rows downloaded for 2027; refusing to continue")
    statcast = _write(raw, development_raw_path(raw_dir, SEASON))
    kept_dates = pd.to_datetime(raw["game_date"]).dt.date
    statcast.update(
        date_coverage=[kept_dates.min().isoformat(), kept_dates.max().isoformat()],
        rows_dropped_outside_window_or_not_regular=dropped,
        source="pybaseball.statcast (Baseball Savant)",
    )

    metadata = metadata_fn(start, end)
    if metadata.empty:
        raise IngestionError("no 2027 game metadata returned")
    game_metadata = _write(metadata, game_metadata_path(raw_dir, SEASON))

    sched = schedule_weather_fn(start, end)
    schedule_weather = _write(sched, schedule_weather_path(raw_dir, SEASON))

    asos: dict[str, Any] = {}
    for station in stations if stations is not None else all_station_ids():
        obs = asos_fn(station, start, end)
        if obs.empty:
            asos[station] = {
                "rows": 0,
                "note": "no observations; games at this station get no density",
            }
            continue
        asos[station] = _write(obs, station_weather_path(raw_dir, station, SEASON))

    return {
        "requested_range": [start.isoformat(), end.isoformat()],
        "retrieved_at": retrieved_at,
        "statcast": statcast,
        "game_metadata": game_metadata,
        "schedule_weather": schedule_weather,
        "asos": asos,
        "venues_without_weather_station": venues_without_weather_station(
            metadata["venue_id"].dropna().tolist()
        ),
    }


def build_scoring_dataset(raw_dir: Path = cfg.RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and join 2027 exactly as the development pipeline does for 2021-2024."""
    cfg.assert_in_namespace(raw_dir)
    cleaned = clean_development_data([SEASON], raw_dir)
    if cleaned.empty:
        raise IngestionError("cleaned 2027 dataset is empty")
    metadata = load_game_metadata(raw_dir, [SEASON])
    with_venue = join_venue_metadata(cleaned, metadata)
    game_weather = build_game_weather_table(
        load_schedule_weather(raw_dir, [SEASON]),
        load_station_observations(raw_dir, [SEASON]),
        load_roof_type_by_game(raw_dir, [SEASON]),
    )
    with_weather = join_weather_features(with_venue, game_weather)
    report = {
        "venue_join": build_venue_join_report(with_venue, metadata).__dict__,
        "weather_join": build_weather_join_report(with_weather).__dict__,
        "rows": len(with_weather),
        "training_eligible_rows": int(with_weather["eligible_for_training"].astype(bool).sum()),
    }
    return with_weather, report
