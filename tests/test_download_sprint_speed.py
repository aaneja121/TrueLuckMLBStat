"""Tests for the Version 0.8 Sprint Speed leaderboard download. No test
touches the network -- `pybaseball.statcast_sprint_speed` is monkeypatched
wherever exercised, or avoided entirely via `--dry-run`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_luck_score.config import ProtectedSeasonError, sprint_speed_path
from mlb_luck_score.data import download_sprint_speed as dss


def _leaderboard_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [100 + i for i in range(n)],
            "sprint_speed": [27.0 + i for i in range(n)],
            "hp_to_1b": [4.3 - 0.1 * i for i in range(n)],
        }
    )


def test_download_rejects_2025_without_flag(tmp_path: Path):
    with pytest.raises(ProtectedSeasonError):
        dss.download_sprint_speed([2024, 2025], tmp_path, dry_run=True)


def test_download_dry_run_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network should not be called during --dry-run")

    monkeypatch.setattr(dss, "fetch_season_sprint_speed", _boom)
    results = dss.download_sprint_speed([2024], tmp_path, dry_run=True)
    assert results[0].status == dss.STATUS_PLANNED


def test_download_skips_existing_season(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    existing_path = sprint_speed_path(tmp_path, 2024)
    pd.DataFrame({"a": [1]}).to_parquet(existing_path, index=False)

    def _boom(*args, **kwargs):
        raise AssertionError("should not fetch a season whose file already exists")

    monkeypatch.setattr(dss, "fetch_season_sprint_speed", _boom)
    results = dss.download_sprint_speed([2024], tmp_path, overwrite=False)
    assert results[0].status == dss.STATUS_SKIPPED_EXISTING


def test_download_writes_one_parquet_file_per_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(dss, "fetch_season_sprint_speed", lambda season, **k: _leaderboard_df())
    results = dss.download_sprint_speed([2021, 2022], tmp_path)

    assert [r.status for r in results] == [dss.STATUS_DOWNLOADED, dss.STATUS_DOWNLOADED]
    for season in (2021, 2022):
        path = sprint_speed_path(tmp_path, season)
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(dss.SPRINT_SPEED_COLUMNS)
        assert len(loaded) == 3


def test_download_overwrite_refetches_existing_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = sprint_speed_path(tmp_path, 2024)
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)

    monkeypatch.setattr(dss, "fetch_season_sprint_speed", lambda season, **k: _leaderboard_df(2))
    results = dss.download_sprint_speed([2024], tmp_path, overwrite=True)

    assert results[0].status == dss.STATUS_DOWNLOADED
    assert len(pd.read_parquet(path)) == 2


def test_download_refuses_to_write_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        dss,
        "fetch_season_sprint_speed",
        lambda season, **k: pd.DataFrame(columns=list(dss.SPRINT_SPEED_COLUMNS)),
    )
    results = dss.download_sprint_speed([2024], tmp_path)

    assert results[0].status == dss.STATUS_FAILED
    assert not sprint_speed_path(tmp_path, 2024).exists()


def test_download_records_failure_without_aborting_other_seasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def _flaky(season, **kwargs):
        if season == 2021:
            raise RuntimeError("Savant unavailable")
        return _leaderboard_df()

    monkeypatch.setattr(dss, "fetch_season_sprint_speed", _flaky)
    results = dss.download_sprint_speed([2021, 2022], tmp_path)

    statuses = {r.season: r.status for r in results}
    assert statuses[2021] == dss.STATUS_FAILED
    assert statuses[2022] == dss.STATUS_DOWNLOADED
    assert sprint_speed_path(tmp_path, 2022).exists()


def test_main_returns_nonzero_on_protected_season(tmp_path: Path):
    exit_code = dss.main(["--seasons", "2025", "--output-dir", str(tmp_path), "--dry-run"])
    assert exit_code == 2
