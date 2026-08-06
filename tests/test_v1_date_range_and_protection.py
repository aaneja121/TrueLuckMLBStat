"""Contact Luck v1.0, Task 33 category 1: date-range protection.

Verifies the reviewed `FINAL_EVALUATION_2025_DATE_RANGE` constant covers the
Tokyo Series, stays local to `evaluation.run_v1_final_evaluation` (never
copied into `mlb_luck_score.config`), and that the development-facing
config constants still exclude 2025 entirely.
"""

from __future__ import annotations

from datetime import date

import run_v1_final_evaluation as rv1

from mlb_luck_score import config


def test_date_range_matches_reviewed_range() -> None:
    assert rv1.FINAL_EVALUATION_2025_DATE_RANGE == ("2025-03-18", "2025-09-28")


def test_date_range_includes_tokyo_series_dates() -> None:
    start = date.fromisoformat(rv1.FINAL_EVALUATION_2025_DATE_RANGE[0])
    end = date.fromisoformat(rv1.FINAL_EVALUATION_2025_DATE_RANGE[1])
    for tokyo_date in (date(2025, 3, 18), date(2025, 3, 19)):
        assert start <= tokyo_date <= end


def test_date_range_end_is_final_scheduled_regular_season_date() -> None:
    assert rv1.FINAL_EVALUATION_2025_DATE_RANGE[1] == "2025-09-28"


def test_date_range_constant_not_defined_in_config_module() -> None:
    assert not hasattr(config, "FINAL_EVALUATION_2025_DATE_RANGE")


def test_config_date_ranges_still_exclude_2025() -> None:
    assert 2025 not in config.MLB_REGULAR_SEASON_DATE_RANGES


def test_config_development_seasons_still_exclude_2025() -> None:
    assert 2025 not in config.DEVELOPMENT_SEASONS


def test_2025_is_the_only_final_test_season() -> None:
    assert config.FINAL_TEST_SEASONS == (2025,)
