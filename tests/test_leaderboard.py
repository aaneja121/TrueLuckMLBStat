"""Tests for the Contact Luck v0.12 ranking policy (Phase 2/9).

No test touches the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.scoring.leaderboard import (
    LeaderboardError,
    assign_official_ranks,
    least_favorable_leaderboard,
    most_favorable_leaderboard,
)


def _table(**overrides) -> pd.DataFrame:
    n = 6
    base = {
        "batter_id": [1, 2, 3, 4, 5, 6],
        "batter_name": [None] * n,
        "season": [2024] * n,
        "contact_luck_runs_per_100": [5.0, 5.0, 3.0, -1.0, -1.0, 10.0],
        "lower_95_interval": [1.0, 1.0, 1.0, -5.0, -5.0, 8.0],
        "upper_95_interval": [9.0, 9.0, 5.0, 3.0, 3.0, 12.0],
        "interval_interpretation": ["overlaps_zero"] * n,
        "eligible_batted_balls": [200] * n,
        "games": [100] * n,
        "total_contact_luck_runs": [10.0] * n,
        "qualification_status": [
            "qualified",
            "qualified",
            "qualified",
            "qualified",
            "small_sample",
            "qualified",
        ],
        "official_rank_eligible": [True, True, True, True, False, True],
        "official_rank_favorable": [pd.NA] * n,
        "official_rank_unfavorable": [pd.NA] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_assign_official_ranks_uses_competition_ranking_with_gaps():
    ranked = assign_official_ranks(_table())
    favorable = dict(zip(ranked["batter_id"], ranked["official_rank_favorable"], strict=True))
    # batter 6 (10.0) rank 1; batters 1,2 (5.0, tied) rank 2; batter 3 (3.0) rank 4 (skips 3);
    # batter 4 (-1.0) rank 5; batter 5 (non-qualified) is null.
    assert favorable[6] == 1
    assert favorable[1] == 2
    assert favorable[2] == 2
    assert favorable[3] == 4
    assert favorable[4] == 5
    assert pd.isna(favorable[5])


def test_assign_official_ranks_unfavorable_direction_mirrors_favorable():
    ranked = assign_official_ranks(_table())
    unfavorable = dict(zip(ranked["batter_id"], ranked["official_rank_unfavorable"], strict=True))
    assert unfavorable[4] == 1
    assert unfavorable[3] == 2
    assert unfavorable[1] == 3
    assert unfavorable[2] == 3
    assert unfavorable[6] == 5
    assert pd.isna(unfavorable[5])


def test_ranking_is_deterministic_across_repeated_calls():
    table = _table()
    first = assign_official_ranks(table)
    second = assign_official_ranks(table)
    pd.testing.assert_frame_equal(first, second)


def test_non_qualified_row_never_gets_a_rank():
    ranked = assign_official_ranks(_table())
    non_qualified = ranked[ranked["batter_id"] == 5].iloc[0]
    assert pd.isna(non_qualified["official_rank_favorable"])
    assert pd.isna(non_qualified["official_rank_unfavorable"])


def test_non_qualified_row_retains_its_score_value():
    # Values are preserved for non-qualified rows even though they get no rank.
    ranked = assign_official_ranks(_table())
    non_qualified = ranked[ranked["batter_id"] == 5].iloc[0]
    assert non_qualified["contact_luck_runs_per_100"] == -1.0


def test_most_favorable_leaderboard_only_contains_qualified_rows():
    ranked = assign_official_ranks(_table())
    board = most_favorable_leaderboard(ranked)
    assert set(board["batter_id"]) == {1, 2, 3, 4, 6}
    assert 5 not in set(board["batter_id"])


def test_most_favorable_leaderboard_is_sorted_descending_by_score():
    ranked = assign_official_ranks(_table())
    board = most_favorable_leaderboard(ranked)
    scores = board["contact_luck_runs_per_100"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_least_favorable_leaderboard_is_sorted_ascending_by_score():
    ranked = assign_official_ranks(_table())
    board = least_favorable_leaderboard(ranked)
    scores = board["contact_luck_runs_per_100"].tolist()
    assert scores == sorted(scores)


def test_tied_rows_have_deterministic_display_order_by_batter_id():
    ranked = assign_official_ranks(_table())
    board = most_favorable_leaderboard(ranked)
    tied = board[board["contact_luck_runs_per_100"] == 5.0]
    assert tied["batter_id"].tolist() == sorted(tied["batter_id"].tolist())


def test_leaderboard_never_reorders_by_interval_endpoints():
    # Construct a case where sorting by upper_95_interval would give a
    # DIFFERENT order than sorting by contact_luck_runs_per_100 -- confirm
    # the leaderboard follows the score, not the interval.
    table = _table(
        contact_luck_runs_per_100=[1.0, 2.0, 3.0, 4.0, -1.0, 5.0],
        upper_95_interval=[100.0, 1.0, 1.0, 1.0, 3.0, 1.0],  # batter 1 has huge upper bound
    )
    ranked = assign_official_ranks(table)
    board = most_favorable_leaderboard(ranked)
    assert board.iloc[0]["batter_id"] == 6  # highest score, despite lowest interval upper bound
    assert board["batter_id"].tolist()[0] != 1


def test_top_n_limits_leaderboard_size():
    ranked = assign_official_ranks(_table())
    board = most_favorable_leaderboard(ranked, top_n=2)
    assert len(board) == 2


def test_assign_official_ranks_requires_official_rank_eligible_column():
    with pytest.raises(LeaderboardError, match="official_rank_eligible"):
        assign_official_ranks(pd.DataFrame({"contact_luck_runs_per_100": [1.0]}))


def test_leaderboard_requires_assign_official_ranks_to_have_run_first():
    table = _table().drop(columns=["official_rank_favorable", "official_rank_unfavorable"])
    with pytest.raises(LeaderboardError, match="assign_official_ranks"):
        most_favorable_leaderboard(table)


def test_zero_qualified_rows_yields_empty_leaderboard_not_a_crash():
    table = _table(
        qualification_status=["small_sample"] * 6,
        official_rank_eligible=[False] * 6,
    )
    ranked = assign_official_ranks(table)
    board = most_favorable_leaderboard(ranked)
    assert len(board) == 0
