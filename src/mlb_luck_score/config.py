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

#: Small default sample window used by `make download-sample` and the
#: downloader CLI's defaults. Intentionally narrow (one week) so repository
#: bootstrap never triggers a full-season download.
DEFAULT_SAMPLE_START_DATE = "2024-04-01"
DEFAULT_SAMPLE_END_DATE = "2024-04-07"


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
