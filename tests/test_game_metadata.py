"""Tests for game-metadata download/join. No test touches the network --
schedule/venue HTTP calls are monkeypatched wherever exercised."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_luck_score.config import ProtectedSeasonError, game_metadata_path
from mlb_luck_score.data import download_game_metadata as dgm
from mlb_luck_score.data.join_venue_metadata import (
    JoinVenueMetadataError,
    build_venue_join_report,
    join_venue_metadata,
    load_game_metadata,
    validate_no_conflicting_venues,
)


def _metadata_row(game_pk, venue_id, venue_name, season, home_team="Team A", away_team="Team B"):
    return {
        "game_pk": game_pk,
        "game_date": f"{season}-05-01",
        "venue_id": venue_id,
        "venue_name": venue_name,
        "home_team": home_team,
        "away_team": away_team,
        "roof_type": "Open",
        "surface_type": "Grass",
        "is_neutral_site": False,
    }


# -- download_game_metadata --------------------------------------------------


def test_download_rejects_2025_without_flag(tmp_path: Path):
    with pytest.raises(ProtectedSeasonError):
        dgm.download_game_metadata([2024, 2025], tmp_path, dry_run=True)


def test_download_dry_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network should not be called during --dry-run")

    monkeypatch.setattr(dgm, "_fetch_schedule_chunk", _boom)
    results = dgm.download_game_metadata([2024], tmp_path, dry_run=True)
    assert results[0].status == dgm.STATUS_PLANNED


def test_download_skips_existing_season(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    existing_path = game_metadata_path(tmp_path, 2024)
    pd.DataFrame({"a": [1]}).to_parquet(existing_path, index=False)

    def _boom(*args, **kwargs):
        raise AssertionError("should not fetch a season whose file already exists")

    monkeypatch.setattr(dgm, "build_season_metadata", _boom)
    results = dgm.download_game_metadata([2024], tmp_path, overwrite=False)
    assert results[0].status == dgm.STATUS_SKIPPED_EXISTING


def test_build_season_metadata_deduplicates_by_game_pk(monkeypatch: pytest.MonkeyPatch):
    duplicate_games = [
        {
            "gamePk": 1,
            "officialDate": "2024-05-01",
            "venue": {"id": 10, "name": "Park A"},
            "teams": {"home": {"team": {"name": "Home"}}, "away": {"team": {"name": "Away"}}},
        },
        {
            "gamePk": 1,  # exact duplicate game_pk from an overlapping chunk boundary
            "officialDate": "2024-05-01",
            "venue": {"id": 10, "name": "Park A"},
            "teams": {"home": {"team": {"name": "Home"}}, "away": {"team": {"name": "Away"}}},
        },
        {
            "gamePk": 2,
            "officialDate": "2024-05-02",
            "venue": {"id": 20, "name": "Park B"},
            "teams": {"home": {"team": {"name": "Home2"}}, "away": {"team": {"name": "Away2"}}},
        },
    ]

    monkeypatch.setattr(dgm, "_fetch_schedule_chunk", lambda *a, **k: duplicate_games)
    monkeypatch.setattr(
        dgm,
        "fetch_venue_details",
        lambda venue_id, **k: {"roof_type": "Open", "surface_type": "Grass"},
    )

    result_df = dgm.build_season_metadata(2024)
    assert len(result_df) == 2
    assert sorted(result_df["game_pk"].tolist()) == [1, 2]


def test_neutral_site_flag_derivation():
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2024-04-01",
                "venue_id": 10,
                "venue_name": "Home Park",
                "home_team": "Cubs",
            },
            {
                "game_pk": 2,
                "game_date": "2024-04-08",
                "venue_id": 10,
                "venue_name": "Home Park",
                "home_team": "Cubs",
            },
            {
                "game_pk": 3,
                "game_date": "2024-06-15",
                "venue_id": 99,
                "venue_name": "London Stadium",
                "home_team": "Cubs",
            },
        ]
    )
    flags = dgm.compute_neutral_site_flags(games)
    assert flags.tolist() == [False, False, True]


# -- join_venue_metadata -------------------------------------------------------


def test_load_game_metadata_missing_season_raises_clear_error(tmp_path: Path):
    with pytest.raises(JoinVenueMetadataError, match="season 2021"):
        load_game_metadata(tmp_path, [2021])


def test_load_game_metadata_deduplicates_across_files(tmp_path: Path):
    df_a = pd.DataFrame(
        [_metadata_row(1, 10, "Park A", 2021), _metadata_row(2, 20, "Park B", 2021)]
    )
    df_b = pd.DataFrame([_metadata_row(2, 20, "Park B", 2021)])  # overlapping game_pk
    df_a.to_parquet(game_metadata_path(tmp_path, 2021), index=False)
    df_b.to_parquet(game_metadata_path(tmp_path, 2022), index=False)

    combined = load_game_metadata(tmp_path, [2021, 2022])
    assert sorted(combined["game_pk"].tolist()) == [1, 2]


def test_venue_join_correctness():
    cleaned = pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 3],
            "season": [2024, 2024, 2024, 2024],
            "outcome_class": ["out", "single", "double", "triple"],
        }
    )
    metadata = pd.DataFrame(
        [_metadata_row(1, 10, "Park A", 2024), _metadata_row(2, 20, "Park B", 2024)]
    )
    joined = join_venue_metadata(cleaned, metadata)
    assert len(joined) == 4
    assert joined.loc[joined["game_pk"] == 1, "venue_name"].unique().tolist() == ["Park A"]
    assert joined.loc[joined["game_pk"] == 2, "venue_name"].unique().tolist() == ["Park B"]


def test_missing_venue_rows_are_preserved_and_marked():
    cleaned = pd.DataFrame(
        {"game_pk": [1, 2], "season": [2024, 2024], "outcome_class": ["out", "single"]}
    )
    metadata = pd.DataFrame([_metadata_row(1, 10, "Park A", 2024)])  # game 2 has no metadata

    joined = join_venue_metadata(cleaned, metadata)
    assert len(joined) == 2  # row for game 2 is kept, not dropped
    unmatched = joined[joined["game_pk"] == 2]
    assert bool(unmatched["has_venue_metadata"].iloc[0]) is False
    assert pd.isna(unmatched["venue_id"].iloc[0])


def test_conflicting_venue_mapping_is_rejected():
    metadata = pd.DataFrame(
        [
            {**_metadata_row(1, 10, "Park A", 2024)},
            {**_metadata_row(1, 20, "Park B", 2024)},  # same game_pk, different venue
        ]
    )
    with pytest.raises(JoinVenueMetadataError, match="more than one venue"):
        validate_no_conflicting_venues(metadata)

    cleaned = pd.DataFrame({"game_pk": [1], "season": [2024], "outcome_class": ["out"]})
    with pytest.raises(JoinVenueMetadataError):
        join_venue_metadata(cleaned, metadata)


def test_venue_join_report_counts():
    cleaned = pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 3],
            "season": [2024, 2024, 2024, 2024],
            "outcome_class": ["out", "single", "double", "triple"],
        }
    )
    metadata = pd.DataFrame(
        [_metadata_row(1, 10, "Park A", 2024), _metadata_row(2, 20, "Park B", 2024)]
    )
    joined = join_venue_metadata(cleaned, metadata)
    report = build_venue_join_report(joined, metadata)

    assert report.total_games == 3
    assert report.total_event_rows == 4
    assert report.matched_rows == 3
    assert report.unmatched_rows == 1
    assert report.venue_match_rate == pytest.approx(0.75)
    assert report.unmatched_games == 1
    assert report.duplicate_mappings == 0
    assert report.counts_by_venue == {"Park A": 2, "Park B": 1}
