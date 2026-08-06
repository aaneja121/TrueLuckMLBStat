"""Tests for the Contact Luck v0.12 public score CLI and review tables (Phase 7-9).

No test touches the network -- uses the shared `v012_public_score_artifacts`
fixture (`tests/conftest.py`).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlb_luck_score.scoring.public_score_schema import validate_public_score_table
from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED
from mlb_luck_score.scoring.run_attribution_ledger import (
    SCRIPT_SEASONS,
    RunAttributionLedgerError,
    _validate_no_final_test_seasons,
)
from mlb_luck_score.scoring.run_public_score import (
    SINGLE_DIGIT_SAMPLE_THRESHOLD,
    RunPublicScoreError,
    _flatten_for_tabular_export,
    assert_no_single_digit_sample_players_ranked,
    build_arg_parser,
    build_review_tables,
    build_score_card,
    run,
)


@pytest.fixture(scope="module")
def results(v012_public_score_artifacts):
    return run(v012_public_score_artifacts)


def test_run_produces_all_expected_result_keys(results):
    assert set(results.keys()) == {
        "public_score_table",
        "favorable_leaderboard",
        "unfavorable_leaderboard",
        "score_card",
        "review_tables",
    }


def test_public_score_table_passes_schema_validation(results):
    validate_public_score_table(results["public_score_table"])  # must not raise


def test_no_single_digit_sample_players_are_ever_officially_ranked(results):
    assert_no_single_digit_sample_players_ranked(results["public_score_table"])  # must not raise


def test_single_digit_sample_detection_actually_fires_on_a_bad_table():
    table = pd.DataFrame(
        {
            "eligible_batted_balls": [5],
            "official_rank_favorable": [1],
            "official_rank_unfavorable": [pd.NA],
        }
    )
    with pytest.raises(RunPublicScoreError, match="single-digit-sample"):
        assert_no_single_digit_sample_players_ranked(table)


def test_single_digit_threshold_is_ten():
    assert SINGLE_DIGIT_SAMPLE_THRESHOLD == 10


def test_score_card_has_required_fields(results, v012_public_score_artifacts):
    card = build_score_card(results["public_score_table"], v012_public_score_artifacts)
    for key in (
        "score_version",
        "model_version",
        "generated_at",
        "data_through_date",
        "seasons",
        "row_count",
        "qualified_count",
        "qualification_counts",
        "ranking_method",
        "official_metric",
        "interval_method",
    ):
        assert key in card


def test_score_card_is_json_serializable(results, v012_public_score_artifacts):
    card = build_score_card(results["public_score_table"], v012_public_score_artifacts)
    json.dumps(card, default=str)  # must not raise


def test_review_tables_have_all_required_sections(results):
    review = results["review_tables"]
    for key in (
        "top_qualified_favorable",
        "bottom_qualified_unfavorable",
        "widest_qualified_intervals",
        "narrowest_qualified_intervals",
        "largest_provisional_component_shares",
        "qualified_intervals_crossing_zero",
        "non_qualified_extreme_point_estimates",
        "ranking_with_vs_without_provisional_components",
    ):
        assert key in review


def test_review_tables_favorable_and_unfavorable_only_include_qualified(
    v012_public_score_artifacts, results
):
    table = results["public_score_table"]
    review = build_review_tables(table, v012_public_score_artifacts)
    for record in review["top_qualified_favorable"]:
        assert record["qualification_status"] == STATUS_QUALIFIED
    for record in review["bottom_qualified_unfavorable"]:
        assert record["qualification_status"] == STATUS_QUALIFIED


def test_flatten_for_tabular_export_stringifies_dict_columns(results):
    table = results["public_score_table"]
    flat = _flatten_for_tabular_export(table)
    assert flat["component_status_reason_codes"].apply(lambda v: isinstance(v, str)).all()
    assert flat["model_version"].apply(lambda v: isinstance(v, str)).all()
    for cell in flat["model_version"]:
        json.loads(cell)  # must round-trip as valid JSON


def test_flatten_for_tabular_export_does_not_mutate_original(results):
    table = results["public_score_table"]
    _flatten_for_tabular_export(table)
    assert isinstance(table["model_version"].iloc[0], dict)


# ---------------------------------------------------------------------------
# 2025 protection
# ---------------------------------------------------------------------------


def test_run_public_score_argparser_has_no_final_evaluation_escape_hatch():
    parser = build_arg_parser()
    actions = {action.dest for action in parser._actions}
    assert "allow_final_evaluation" not in actions


def test_run_public_score_relies_on_the_same_2025_guard_as_run_attribution_ledger():
    df = pd.DataFrame({"season": [2021, 2024, 2025]})
    with pytest.raises(RunAttributionLedgerError, match="2025"):
        _validate_no_final_test_seasons(df)
    assert 2025 not in SCRIPT_SEASONS
