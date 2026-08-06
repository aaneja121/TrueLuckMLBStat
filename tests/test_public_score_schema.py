"""Tests for the Contact Luck v0.12 public score schema (Phase 1/9).

No test touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.public_score_schema import (
    INTERVAL_ABOVE_ZERO,
    INTERVAL_BELOW_ZERO,
    INTERVAL_OVERLAPS_ZERO,
    PUBLIC_SCORE_SCHEMA_VERSION,
    REQUIRED_PUBLIC_SCORE_COLUMNS,
    PublicScoreSchemaError,
    classify_interval,
    validate_public_score_table,
)


def _minimal_row(**overrides) -> dict:
    base = {col: None for col in REQUIRED_PUBLIC_SCORE_COLUMNS}
    base.update(
        {
            "batter_id": 1,
            "batter_name": None,
            "season": 2024,
            "games": 100,
            "eligible_batted_balls": 200,
            "total_contact_luck_runs": 5.0,
            "contact_luck_runs_per_100": 2.5,
            "lower_95_interval": 1.0,
            "upper_95_interval": 4.0,
            "interval_interpretation": INTERVAL_ABOVE_ZERO,
            "qualification_status": "qualified",
            "official_rank_eligible": True,
            "official_rank_favorable": 1,
            "official_rank_unfavorable": 50,
            "total_contact_component_runs": 1.0,
            "total_unexplained_residual_component_runs": 1.0,
            "total_defensive_execution_component_runs": 1.0,
            "total_advancement_component_runs": 1.0,
            "contact_component_per_100": 0.5,
            "unexplained_residual_component_per_100": 0.5,
            "defensive_execution_component_per_100": 0.5,
            "advancement_execution_component_per_100": 0.5,
            "share_of_value_from_provisional_components": 0.2,
            "defense_unavailable_play_count": 0,
            "advancement_unavailable_play_count": 0,
            "component_status_reason_codes": {},
            "model_version": {},
            "score_version": PUBLIC_SCORE_SCHEMA_VERSION,
            "generated_at": "2024-01-01T00:00:00+00:00",
            "data_through_date": "2024-09-29",
        }
    )
    base.update(overrides)
    return base


def _table(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_classify_interval_above_zero():
    assert classify_interval(1.0, 4.0) == INTERVAL_ABOVE_ZERO


def test_classify_interval_below_zero():
    assert classify_interval(-4.0, -1.0) == INTERVAL_BELOW_ZERO


def test_classify_interval_overlaps_zero():
    assert classify_interval(-1.0, 1.0) == INTERVAL_OVERLAPS_ZERO


def test_classify_interval_boundary_is_not_above_zero():
    # lower == 0 exactly must NOT be classified as "entirely above zero".
    assert classify_interval(0.0, 4.0) == INTERVAL_OVERLAPS_ZERO


def test_classify_interval_returns_none_for_null_bounds():
    assert classify_interval(float("nan"), 1.0) is None
    assert classify_interval(1.0, float("nan")) is None


def test_valid_table_passes_validation():
    table = _table(_minimal_row())
    validate_public_score_table(table)  # must not raise


def test_missing_column_fails_validation():
    table = _table(_minimal_row()).drop(columns=["contact_luck_runs_per_100"])
    with pytest.raises(PublicScoreSchemaError, match="missing column"):
        validate_public_score_table(table)


def test_duplicate_batter_season_fails_validation():
    table = _table(_minimal_row(batter_id=1, season=2024), _minimal_row(batter_id=1, season=2024))
    with pytest.raises(PublicScoreSchemaError, match="Duplicate"):
        validate_public_score_table(table)


def test_missing_batter_id_fails_validation():
    table = _table(_minimal_row(batter_id=None))
    with pytest.raises(PublicScoreSchemaError, match="never be null"):
        validate_public_score_table(table)


def test_missing_season_fails_validation():
    table = _table(_minimal_row(season=None))
    with pytest.raises(PublicScoreSchemaError, match="never be null"):
        validate_public_score_table(table)


def test_non_finite_official_value_fails_validation():
    table = _table(_minimal_row(contact_luck_runs_per_100=np.inf))
    with pytest.raises(PublicScoreSchemaError, match="non-finite"):
        validate_public_score_table(table)


def test_negative_infinite_total_fails_validation():
    table = _table(_minimal_row(total_contact_luck_runs=-np.inf))
    with pytest.raises(PublicScoreSchemaError, match="non-finite"):
        validate_public_score_table(table)


def test_null_rate_is_allowed_for_non_qualified_zero_play_row():
    table = _table(
        _minimal_row(
            batter_id=2,
            eligible_batted_balls=0,
            games=0,
            total_contact_luck_runs=0.0,
            contact_luck_runs_per_100=None,
            lower_95_interval=None,
            upper_95_interval=None,
            interval_interpretation=None,
            qualification_status="not_reportable",
            official_rank_eligible=False,
            official_rank_favorable=None,
            official_rank_unfavorable=None,
        )
    )
    validate_public_score_table(table)  # must not raise


def test_qualified_row_with_null_rate_fails_validation():
    table = _table(_minimal_row(contact_luck_runs_per_100=None))
    with pytest.raises(PublicScoreSchemaError, match="non-null contact_luck_runs_per_100"):
        validate_public_score_table(table)


def test_qualified_row_with_null_interval_fails_validation():
    table = _table(_minimal_row(lower_95_interval=None))
    with pytest.raises(PublicScoreSchemaError, match="non-null lower_95_interval"):
        validate_public_score_table(table)


def test_official_rank_eligible_inconsistent_with_qualification_status_fails():
    table = _table(_minimal_row(qualification_status="small_sample", official_rank_eligible=True))
    with pytest.raises(PublicScoreSchemaError, match="official_rank_eligible"):
        validate_public_score_table(table)


def test_rank_populated_for_non_rank_eligible_row_fails():
    table = _table(
        _minimal_row(
            qualification_status="small_sample",
            official_rank_eligible=False,
            official_rank_favorable=3,
        )
    )
    with pytest.raises(PublicScoreSchemaError, match="official_rank_favorable"):
        validate_public_score_table(table)
