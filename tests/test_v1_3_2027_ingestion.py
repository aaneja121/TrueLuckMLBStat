"""2027 ingestion and scoring-dataset build for the one-shot v1.3 comparison.

Every network call is replaced by a synthetic fetcher (the suite blocks real sockets).
The dataset build runs the real cleaning, venue join and weather join on synthetic files.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v1_3_2027_config as cfg
import v1_3_2027_ingestion as ing

START, END = date(2027, 3, 24), date(2027, 3, 26)
TRUIST = 4705  # outdoor, station FTY in the frozen venue table


def _statcast(start: date, end: date) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for g, d in enumerate(("2027-03-24", "2027-03-25", "2027-03-26"), start=1):
        for ab in range(1, 21):
            ev = ["single", "field_out", "home_run", "double", "field_out"][ab % 5]
            rows.append(
                {
                    "game_pk": 900000 + g,
                    "game_date": d,
                    "at_bat_number": ab,
                    "pitch_number": 1,
                    "batter": 100 + ab,
                    "pitcher": 999,
                    "player_name": "Synthetic",
                    "home_team": "ATL",
                    "away_team": "NYM",
                    "inning": 3,
                    "inning_topbot": "Top",
                    "outs_when_up": 1,
                    "events": ev,
                    "description": "hit_into_play",
                    "launch_speed": float(rng.uniform(80, 110)),
                    "launch_angle": float(rng.uniform(-10, 40)),
                    "hit_distance_sc": float(rng.uniform(50, 420)),
                    "hit_location": 7,
                    "hc_x": float(rng.uniform(60, 190)),
                    "hc_y": float(rng.uniform(40, 160)),
                    "bb_type": "fly_ball" if ab % 2 else "ground_ball",
                    "stand": "R",
                    "p_throws": "R",
                    "game_type": "R",
                }
            )
    return pd.DataFrame(rows)


def _metadata(start: date, end: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [900001, 900002, 900003],
            "game_date": ["2027-03-24", "2027-03-25", "2027-03-26"],
            "venue_id": [TRUIST] * 3,
            "venue_name": ["Truist Park"] * 3,
            "home_team": ["ATL"] * 3,
            "away_team": ["NYM"] * 3,
            "roof_type": ["Open"] * 3,
            "surface_type": ["Grass"] * 3,
            "is_neutral_site": [False] * 3,
        }
    )


def _schedule_weather(start: date, end: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": [900001, 900002, 900003],
            "game_date": ["2027-03-24", "2027-03-25", "2027-03-26"],
            "scheduled_start_time_utc": [
                "2027-03-24T23:20:00Z",
                "2027-03-25T23:20:00Z",
                "2027-03-26T23:20:00Z",
            ],
            "venue_id": [TRUIST] * 3,
            "weather_condition": ["Sunny", "Clear", "Cloudy"],
            "weather_temp_f": ["72", "68", "65"],
            "weather_wind_raw": ["6 mph, Out To CF", "3 mph, In From LF", "0 mph, None"],
        }
    )


def _asos(station: str, start: date, end: date) -> pd.DataFrame:
    valid = pd.date_range("2027-03-24T00:00Z", "2027-03-27T23:00Z", freq="h")
    n = len(valid)
    return pd.DataFrame(
        {
            "station": station,
            "valid": valid,
            "tmpf": np.full(n, 70.0),
            "dwpf": np.full(n, 50.0),
            "relh": np.full(n, 50.0),
            "drct": np.full(n, 180.0),
            "sknt": np.full(n, 5.0),
            "alti": np.full(n, 30.0),
            "mslp": np.full(n, 1015.0),
        }
    )


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data" / "prospective" / "2027" / "v1_3"
    monkeypatch.setattr(cfg, "NAMESPACE_ROOTS", (d, tmp_path / "out", tmp_path / "art"))
    return d


def _ingest(raw_dir: Path, **kw):
    fetchers = {
        "statcast_fn": _statcast,
        "metadata_fn": _metadata,
        "schedule_weather_fn": _schedule_weather,
        "asos_fn": _asos,
        "stations": ["FTY"],
    }
    fetchers.update(kw)
    return ing.ingest_2027(START, END, raw_dir=raw_dir, **fetchers)


def test_ingest_writes_every_raw_file_inside_the_namespace_with_provenance(raw_dir: Path) -> None:
    prov = _ingest(raw_dir)
    for key in ("statcast", "game_metadata", "schedule_weather"):
        path = Path(prov[key]["path"])
        assert path.exists() and path.is_relative_to(raw_dir)
        assert len(prov[key]["sha256"]) == 64 and prov[key]["rows"] > 0
    assert prov["statcast"]["date_coverage"] == ["2027-03-24", "2027-03-26"]
    assert prov["requested_range"] == ["2027-03-24", "2027-03-26"]
    assert prov["asos"]["FTY"]["rows"] > 0


def test_ingest_refuses_a_directory_outside_the_namespace(tmp_path: Path) -> None:
    with pytest.raises(cfg.NamespaceViolationError):
        _ingest(tmp_path / "elsewhere")


def test_ingest_refuses_an_empty_statcast_download(raw_dir: Path) -> None:
    with pytest.raises(ing.IngestionError, match="no Statcast rows"):
        _ingest(raw_dir, statcast_fn=lambda s, e: _statcast(s, e).iloc[0:0])


def test_rows_outside_the_season_window_are_dropped_and_counted(raw_dir: Path) -> None:
    def with_spring(start: date, end: date) -> pd.DataFrame:
        df = _statcast(start, end)
        spring = df.head(5).assign(game_date="2027-03-01", game_type="S")
        return pd.concat([df, spring], ignore_index=True)

    prov = _ingest(raw_dir, statcast_fn=with_spring)
    assert prov["statcast"]["rows_dropped_outside_window_or_not_regular"] == 5
    assert prov["statcast"]["date_coverage"] == ["2027-03-24", "2027-03-26"]


def test_scoring_dataset_has_2027_rows_with_weather(raw_dir: Path) -> None:
    _ingest(raw_dir)
    df, report = ing.build_scoring_dataset(raw_dir)
    assert set(df["season"].unique()) == {2027}
    assert {"air_density_kg_m3", "roof_status", "venue_id"} <= set(df.columns)
    eligible = df[df["eligible_for_training"].astype(bool)]
    assert len(eligible) > 0
    assert eligible["air_density_kg_m3"].notna().any()
    assert (df["venue_id"] == TRUIST).all()
    assert "weather_join" in report and "venue_join" in report


def test_venues_without_a_station_are_reported() -> None:
    missing = ing.venues_without_weather_station([TRUIST, 2529, 5355])
    assert missing == [2529, 5355]
