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
    PUBLIC_SCORE_FIELDS,
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


class TestGamesFieldDescriptionClarification:
    """Regression coverage for the 'games' field's description text: it must
    accurately describe "distinct games with an outcome-resolved eligible
    batted ball" and must never again read as (or be silently reworded back
    to) "official MLB games played" -- see the dashboard's "Scored Games"
    presentation-only clarification, which relies on this same definition
    matching what the schema documents. This is a documentation-only
    contract: the underlying COLUMN NAME stays exactly "games" (the frozen
    public-score data contract is unchanged), only the description string
    changed.
    """

    def _games_field(self):
        (field,) = [f for f in PUBLIC_SCORE_FIELDS if f.name == "games"]
        return field

    def test_column_name_is_unchanged(self):
        # The frozen public-score JSON/parquet key must stay "games" --
        # only the human-readable description changed, never the contract.
        assert self._games_field().name == "games"
        assert self._games_field().dtype == "int64"
        assert self._games_field().nullable is False

    def test_description_states_outcome_resolved_definition(self):
        description = self._games_field().description
        assert "outcome-resolved" in description
        assert "eligible batted ball" in description

    def test_description_explicitly_disclaims_official_games_played(self):
        description = self._games_field().description
        assert "NOT official MLB games played" in description

    def test_description_no_longer_reads_as_bare_eligible_batted_ball(self):
        # The old, ambiguous wording ("games with an eligible batted ball")
        # doesn't distinguish a resolved batted ball from an ambiguous one
        # (field_error/fielders_choice) that never actually counts --
        # confirms the old phrasing is gone, not just that new text exists
        # alongside it.
        description = self._games_field().description
        assert "games with an eligible batted ball." not in description
