"""Tests for the full-development-dataset download/clean workflow.

No test in this module touches the network: `download_development_data`'s
per-chunk network call (`download_statcast_range`) is monkeypatched wherever
a download path is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_luck_score.config import (
    DEVELOPMENT_SEASONS,
    MLB_REGULAR_SEASON_DATE_RANGES,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    ProtectedSeasonError,
    development_raw_path,
)
from mlb_luck_score.data import download_development_data as ddd
from mlb_luck_score.data.clean_batted_balls import CleaningError, clean_batted_balls
from mlb_luck_score.data.clean_development_data import (
    clean_development_data,
    combine_cleaned_seasons,
)


def _synthetic_raw_rows(season: int, *, game_pk: int = 1, start_at_bat: int = 1) -> list[dict]:
    """Minimal raw-Statcast-shaped rows for one synthetic season."""
    rows = []
    at_bat = start_at_bat
    events_list = ["single", "double", "field_out"]
    for events in events_list:
        for _ in range(3):
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": f"{season}-05-01",
                    "at_bat_number": at_bat,
                    "pitch_number": 1,
                    "batter": 1,
                    "pitcher": 2,
                    "events": events,
                    "launch_speed": 90.0,
                    "launch_angle": 12.0,
                    "hc_x": 125.0,
                    "hc_y": 150.0,
                    "bb_type": "line_drive",
                    "stand": "R",
                    "venue": "Test Park",
                }
            )
            at_bat += 1
    # one ambiguous fielders_choice row, kept but excluded from training
    rows.append(
        {
            "game_pk": game_pk,
            "game_date": f"{season}-05-01",
            "at_bat_number": at_bat,
            "pitch_number": 1,
            "batter": 1,
            "pitcher": 2,
            "events": "fielders_choice",
            "launch_speed": 90.0,
            "launch_angle": 12.0,
            "hc_x": 125.0,
            "hc_y": 150.0,
            "bb_type": "line_drive",
            "stand": "R",
            "venue": "Test Park",
        }
    )
    return rows


def _write_season_raw(raw_dir: Path, season: int, *, game_pk: int = 1) -> Path:
    df = pd.DataFrame(_synthetic_raw_rows(season, game_pk=game_pk))
    path = development_raw_path(raw_dir, season)
    df.to_parquet(path, index=False)
    return path


# -- train/validation partition -------------------------------------------


def test_development_seasons_matches_train_plus_validation():
    assert set(DEVELOPMENT_SEASONS) == set(TRAIN_SEASONS) | set(VALIDATION_SEASONS)
    assert set(TRAIN_SEASONS) == {2021, 2022, 2023}
    assert set(VALIDATION_SEASONS) == {2024}
    assert 2025 not in DEVELOPMENT_SEASONS


def test_no_authoritative_date_range_configured_for_2025():
    # Defense in depth alongside assert_seasons_allowed.
    assert 2025 not in MLB_REGULAR_SEASON_DATE_RANGES
    for season in DEVELOPMENT_SEASONS:
        assert season in MLB_REGULAR_SEASON_DATE_RANGES


# -- download_development_data (network calls monkeypatched) ---------------


def test_download_rejects_2025_without_flag(tmp_path: Path):
    with pytest.raises(ProtectedSeasonError):
        ddd.download_development_data([2024, 2025], tmp_path, dry_run=True)


def test_download_dry_run_needs_no_network_and_plans_all_seasons(tmp_path: Path):
    results = ddd.download_development_data(list(DEVELOPMENT_SEASONS), tmp_path, dry_run=True)
    assert [r.season for r in results] == list(DEVELOPMENT_SEASONS)
    assert all(r.status == ddd.STATUS_PLANNED for r in results)
    assert all(r.rows > 0 for r in results)  # rows here = planned chunk count


def test_download_skips_existing_season_without_calling_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    existing_path = development_raw_path(tmp_path, 2021)
    pd.DataFrame({"a": [1]}).to_parquet(existing_path, index=False)

    def _boom(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("download_statcast_range should not be called for an existing season")

    monkeypatch.setattr(ddd, "download_statcast_range", _boom)

    results = ddd.download_development_data([2021], tmp_path, overwrite=False)
    assert results[0].status == ddd.STATUS_SKIPPED_EXISTING


def test_download_continues_after_one_season_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _fake_download(start, end, *, chunk_days, max_retries):  # noqa: ANN001
        if start.year == 2021:
            raise RuntimeError("simulated transient network failure")
        return pd.DataFrame(_synthetic_raw_rows(start.year))

    monkeypatch.setattr(ddd, "download_statcast_range", _fake_download)

    results = ddd.download_development_data([2021, 2022], tmp_path)
    by_season = {r.season: r for r in results}
    assert by_season[2021].status == ddd.STATUS_FAILED
    assert by_season[2022].status == ddd.STATUS_DOWNLOADED
    assert development_raw_path(tmp_path, 2022).exists()
    assert not development_raw_path(tmp_path, 2021).exists()


def test_download_never_writes_the_sample_filename(tmp_path: Path):
    path_2024 = development_raw_path(tmp_path, 2024)
    assert path_2024.name != "statcast_2024_sample.parquet"
    assert path_2024.name == "statcast_2024_regular_season.parquet"


# -- clean_development_data -------------------------------------------------


def test_missing_season_raises_clear_error_listing_all_missing(tmp_path: Path):
    _write_season_raw(tmp_path, 2021, game_pk=1)
    with pytest.raises(CleaningError, match=r"season 2022.*season 2023|season 2023.*season 2022"):
        clean_development_data([2021, 2022, 2023], tmp_path)


def test_clean_rejects_2025_without_flag(tmp_path: Path):
    _write_season_raw(tmp_path, 2024, game_pk=1)
    with pytest.raises(ProtectedSeasonError):
        clean_development_data([2024, 2025], tmp_path)


def test_clean_development_data_combines_multiple_seasons(tmp_path: Path):
    _write_season_raw(tmp_path, 2021, game_pk=1)
    _write_season_raw(tmp_path, 2022, game_pk=2)
    _write_season_raw(tmp_path, 2023, game_pk=3)

    combined = clean_development_data([2021, 2022, 2023], tmp_path)

    assert set(combined["season"].unique().tolist()) == {2021, 2022, 2023}
    # 9 unambiguous (3 outcomes x 3 rows) + 1 ambiguous fielders_choice per season
    assert len(combined) == 30
    # ambiguous fielders_choice rows preserved but excluded from training
    fc_rows = combined[combined["events"] == "fielders_choice"]
    assert len(fc_rows) == 3
    assert not fc_rows["eligible_for_training"].any()
    assert combined["eligible_for_training"].sum() == 27


def test_clean_development_data_logs_totals_and_exclusions_per_season(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    _write_season_raw(tmp_path, 2021, game_pk=1)
    _write_season_raw(tmp_path, 2022, game_pk=2)

    with caplog.at_level("INFO"):
        clean_development_data([2021, 2022], tmp_path)

    assert "[season 2021]" in caplog.text
    assert "[season 2022]" in caplog.text
    assert "training-exclusion reason counts" in caplog.text


# -- multi-file combination / cross-file dedup (unit-level) -----------------


def test_combine_cleaned_seasons_concatenates_and_tags_season():
    raw_2021 = pd.DataFrame(_synthetic_raw_rows(2021, game_pk=10))
    raw_2022 = pd.DataFrame(_synthetic_raw_rows(2022, game_pk=20))
    cleaned_2021 = clean_batted_balls(raw_2021)
    cleaned_2022 = clean_batted_balls(raw_2022)

    combined = combine_cleaned_seasons({2021: cleaned_2021, 2022: cleaned_2022})

    assert len(combined) == len(cleaned_2021) + len(cleaned_2022)
    assert set(combined["season"].unique().tolist()) == {2021, 2022}
    # deterministic sort: season then event_id
    assert combined["season"].is_monotonic_increasing


def test_combine_cleaned_seasons_deduplicates_shared_event_id():
    raw = pd.DataFrame(_synthetic_raw_rows(2021, game_pk=99))
    cleaned = clean_batted_balls(raw)
    # Simulate the same physical event appearing in two "season files" --
    # should not happen in practice (game_pk is globally unique) but the
    # combiner must handle it defensively.
    duplicated = combine_cleaned_seasons({2021: cleaned, 2022: cleaned.copy()})

    assert len(duplicated) == len(cleaned)  # deduped down to one copy


def test_combine_cleaned_seasons_handles_empty_frames():
    assert combine_cleaned_seasons({}).empty
    assert combine_cleaned_seasons({2021: pd.DataFrame()}).empty


# -- preserving the existing one-week sample workflow -----------------------


def test_development_workflow_does_not_collide_with_sample_paths(tmp_path: Path):
    sample_raw = tmp_path / "statcast_2024_sample.parquet"
    sample_cleaned = tmp_path / "cleaned_batted_balls.parquet"
    dev_raw_2024 = development_raw_path(tmp_path, 2024)
    dev_cleaned_default_name = "cleaned_development_data.parquet"

    assert dev_raw_2024 != sample_raw
    assert dev_cleaned_default_name != sample_cleaned.name


def test_existing_sample_file_untouched_by_development_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample_path = tmp_path / "statcast_2024_sample.parquet"
    sample_df = pd.DataFrame({"marker": ["do-not-touch"]})
    sample_df.to_parquet(sample_path, index=False)

    def _fake_download(start, end, *, chunk_days, max_retries):  # noqa: ANN001
        return pd.DataFrame(_synthetic_raw_rows(start.year))

    monkeypatch.setattr(ddd, "download_statcast_range", _fake_download)
    ddd.download_development_data([2024], tmp_path)

    # The sample file must be untouched -- development downloads write a
    # differently-named file in the same directory.
    reloaded = pd.read_parquet(sample_path)
    assert reloaded["marker"].tolist() == ["do-not-touch"]
    assert development_raw_path(tmp_path, 2024).exists()
