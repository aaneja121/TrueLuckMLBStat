"""Project-wide configuration constants.

This module centralizes values that must stay consistent across the whole
pipeline: the outcome class order, the provisional ordinal value mapping used
by the raw-luck calculation, season roles for the time-based development
design, and the list of columns that are never allowed as model features
(target-leakage prevention).

Nothing in this module should be silently redefined elsewhere -- if a
downstream module needs a different value, change it here and update the
docstring explaining why.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

#: The five outcome classes predicted by the Version 0.1 contact model.
#: Order matters: every predicted-probability array, calibration table, and
#: raw-luck calculation must use this exact order so results stay comparable.
CLASS_ORDER: tuple[str, ...] = ("out", "single", "double", "triple", "home_run")

#: Version 0.1 provisional ordinal value mapping for the additive raw-luck
#: calculation. This is a research placeholder, NOT a validated run-value
#: model (it ignores base/out state, park, and win-expectancy context).
#: Keep it configurable -- callers may pass an alternate mapping to
#: `mlb_luck_score.scoring.raw_luck.compute_raw_luck`.
DEFAULT_VALUE_MAP: dict[str, float] = {
    "out": 0.0,
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
    "home_run": 4.0,
}

#: Columns that are computed from or that directly encode the play's result.
#: These must never be used as model input features -- doing so would leak
#: the target into the feature set. Note `bb_type` ("line_drive", "fly_ball",
#: ...) is deliberately NOT included: it describes the contact event itself
#: (pre-outcome) and is used as a categorical feature -- see
#: `features/build_contact_features.py` and the leakage check in
#: `tests/test_features.py`.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "events",
        "outcome_class",
        "description",
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "delta_home_win_exp",
        "delta_run_exp",
        "post_away_score",
        "post_home_score",
        "post_bat_score",
        "post_fld_score",
    }
)

#: Season roles for the time-based development design. 2025 is a protected,
#: untouched final test season -- see `assert_seasons_allowed` below and
#: CLAUDE.md for the enforced rule.
TRAIN_SEASONS: tuple[int, ...] = (2021, 2022, 2023)
VALIDATION_SEASONS: tuple[int, ...] = (2024,)
FINAL_TEST_SEASONS: tuple[int, ...] = (2025,)
PROSPECTIVE_SEASONS: tuple[int, ...] = (2026,)

#: Seasons for the time-ordered POST-HOC CALIBRATION design -- distinct from
#: `TRAIN_SEASONS`/`VALIDATION_SEASONS` above (which are used for the main
#: baseline model's train/validation split). A post-hoc calibration layer
#: (e.g. isotonic regression via `CalibratedClassifierCV`) must be fit on
#: data the base model never trained on, and evaluated on data neither the
#: base model nor the calibrator ever saw -- otherwise the evaluation is
#: optimistic. This gives three non-overlapping season groups:
#:   - `CALIBRATION_BASE_TRAIN_SEASONS`: train the base classifier only.
#:   - `CALIBRATION_FIT_SEASONS`: fit the calibration layer only (never used
#:     for base training).
#:   - `CALIBRATION_EVAL_SEASONS`: final evaluation only (never used for base
#:     training or calibration fitting). Deliberately equal to
#:     `VALIDATION_SEASONS` (2024) so calibrated and uncalibrated models are
#:     compared on the same untouched season.
#: See `mlb_luck_score.models.compare_models` and CLAUDE.md. Never fit a
#: calibrator and evaluate it on the same rows.
CALIBRATION_BASE_TRAIN_SEASONS: tuple[int, ...] = (2021, 2022)
CALIBRATION_FIT_SEASONS: tuple[int, ...] = (2023,)
CALIBRATION_EVAL_SEASONS: tuple[int, ...] = (2024,)

#: Small default sample window used by `make download-sample` and the
#: downloader CLI's defaults. Intentionally narrow (one week) so repository
#: bootstrap never triggers a full-season download.
DEFAULT_SAMPLE_START_DATE = "2024-04-01"
DEFAULT_SAMPLE_END_DATE = "2024-04-07"

#: All seasons used anywhere in the time-based development design (training
#: + validation). Deliberately excludes `FINAL_TEST_SEASONS` (2025) and
#: `PROSPECTIVE_SEASONS` (2026) -- this is the season list
#: `mlb_luck_score.data.download_development_data` and
#: `mlb_luck_score.data.clean_development_data` default to.
DEVELOPMENT_SEASONS: tuple[int, ...] = TRAIN_SEASONS + VALIDATION_SEASONS

#: Documented, best-effort MLB regular-season date ranges (commonly-cited
#: Opening Day through the last day of the 162-game slate) used by
#: `mlb_luck_score.data.download_development_data`. These are NOT scraped
#: from a live schedule API -- they are a Version 0.1 research choice.
#: Known simplification: 2024 excludes the earlier Seoul Series games
#: (2024-03-20/21), which most of the league did not play in.
#: Intentionally has NO entry for 2025 (defense in depth alongside
#: `assert_seasons_allowed`: even with `allow_final_evaluation=True`, this
#: dict must be extended before 2025 could be downloaded via this command).
MLB_REGULAR_SEASON_DATE_RANGES: dict[int, tuple[str, str]] = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
}

#: Filename template for per-season development raw files, shared by the
#: downloader and the cleaner so they always agree on where a season's raw
#: data lives. Deliberately distinct from the one-week bootstrap sample
#: filename (`statcast_2024_sample.parquet`) so the two workflows can never
#: collide or overwrite one another.
DEVELOPMENT_RAW_FILENAME_TEMPLATE = "statcast_{season}_regular_season.parquet"


def development_raw_path(raw_dir: Path, season: int) -> Path:
    """Path to a single development season's raw Statcast Parquet file."""
    return raw_dir / DEVELOPMENT_RAW_FILENAME_TEMPLATE.format(season=season)


#: Public MLB Stats API base URL used by
#: `mlb_luck_score.data.download_game_metadata` to recover venue/roof/
#: surface metadata per game (Statcast itself does not include a usable
#: venue field -- see CLAUDE.md). No API key required.
MLB_STATS_API_BASE_URL = "https://statsapi.mlb.com/api/v1"

#: Filename template for per-season game-metadata cache files, mirroring
#: `DEVELOPMENT_RAW_FILENAME_TEMPLATE`'s pattern so the two never collide.
GAME_METADATA_FILENAME_TEMPLATE = "game_metadata_{season}.parquet"


def game_metadata_path(raw_dir: Path, season: int) -> Path:
    """Path to a single season's cached game-metadata Parquet file."""
    return raw_dir / GAME_METADATA_FILENAME_TEMPLATE.format(season=season)


#: Public historical METAR/ASOS archive used by
#: `mlb_luck_score.data.download_historical_weather` to recover station-level
#: temperature/humidity/pressure/wind observations (no API key required).
#: See that module's docstring for the exact fields used and their units.
IOWA_MESONET_ASOS_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

#: Filename template for per-season MLB schedule-weather cache files
#: (temperature/wind/condition text hydrated from the `/schedule` endpoint),
#: mirroring `GAME_METADATA_FILENAME_TEMPLATE`'s pattern.
SCHEDULE_WEATHER_FILENAME_TEMPLATE = "schedule_weather_{season}.parquet"


def schedule_weather_path(raw_dir: Path, season: int) -> Path:
    """Path to a single season's cached MLB schedule-weather Parquet file."""
    return raw_dir / SCHEDULE_WEATHER_FILENAME_TEMPLATE.format(season=season)


#: Filename template for per-station-per-season historical ASOS observation
#: cache files.
STATION_WEATHER_FILENAME_TEMPLATE = "asos_weather_{station_id}_{season}.parquet"


def station_weather_path(raw_dir: Path, station_id: str, season: int) -> Path:
    """Path to a single station-season's cached ASOS observation Parquet file."""
    return raw_dir / STATION_WEATHER_FILENAME_TEMPLATE.format(station_id=station_id, season=season)


#: Filename template for per-season Baseball Savant Sprint Speed leaderboard
#: cache files (`mlb_luck_score.data.download_sprint_speed`) -- a SEASON
#: -LEVEL leaderboard (one row per qualified batter per season), not a
#: per-play field, joined by `batter`+`season` for the Version 0.8 infield
#: opportunity model.
SPRINT_SPEED_FILENAME_TEMPLATE = "sprint_speed_{season}.parquet"


def sprint_speed_path(raw_dir: Path, season: int) -> Path:
    """Path to a single season's cached Sprint Speed leaderboard Parquet file."""
    return raw_dir / SPRINT_SPEED_FILENAME_TEMPLATE.format(season=season)


class ProtectedSeasonError(ValueError):
    """Raised when a command would use a protected final-test season."""


def assert_seasons_allowed(
    seasons: int | str | list[int | str] | tuple[int | str, ...],
    *,
    allow_final_evaluation: bool = False,
) -> None:
    """Guard against accidental use of the 2025 final-test season.

    Never tune, iterate, or select features using 2025 results. This
    function is the single enforcement point for that rule; normal
    development commands (downloading, cleaning, training, calibrating) call
    it and must refuse to proceed unless the caller explicitly passes
    ``allow_final_evaluation=True``.

    Args:
        seasons: A single year (int or parseable string, e.g. a date string
            containing a year) or a collection of years to check.
        allow_final_evaluation: Must be explicitly set to True to permit use
            of a protected final-test season. Defaults to False.

    Raises:
        ProtectedSeasonError: If any protected season is present and
            ``allow_final_evaluation`` is False.
    """
    if isinstance(seasons, int | str):
        seasons = [seasons]

    years: set[int] = set()
    for value in seasons:
        if isinstance(value, int):
            years.add(value)
        else:
            # Accept plain years ("2025") or ISO date strings ("2025-04-01").
            years.add(int(str(value)[:4]))

    protected = years & set(FINAL_TEST_SEASONS)
    if protected and not allow_final_evaluation:
        raise ProtectedSeasonError(
            f"Season(s) {sorted(protected)} are reserved as the untouched final "
            "test set (see CLAUDE.md). Refusing to proceed. Pass "
            "allow_final_evaluation=True (CLI: --allow-final-evaluation) only "
            "for an intentional, one-time final evaluation -- never for "
            "tuning, iteration, or feature selection."
        )
