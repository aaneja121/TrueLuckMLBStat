"""Contact Luck v1.1 Phase 6: presentation-only batter-ID-to-name overlay.

Confirms the join is keyed on `batter_id` only, never changes any score/
rank/qualification column, leaves unresolved names null with a reason code
in the SIDECAR report (never a new column on the frozen schema), and that
name resolution status never depends on -- or influences -- score ordering.
"""

from __future__ import annotations

import pandas as pd
import prospective_player_names as ppn
import pytest


def _base_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter_id": [101, 102, 103],
            "batter_name": pd.array([None, None, None], dtype="string"),
            "season": [2026, 2026, 2026],
            "contact_luck_runs_per_100": [3.1, -1.2, 0.4],
            "lower_95_interval": [0.5, -3.0, -1.0],
            "upper_95_interval": [5.7, 0.6, 1.8],
            "official_rank_favorable": pd.array([1, pd.NA, pd.NA], dtype="Int64"),
            "official_rank_unfavorable": pd.array([pd.NA, 1, pd.NA], dtype="Int64"),
            "qualification_status": ["qualified", "qualified", "not_qualified"],
        }
    )


def _names_df(*, include_unresolved_row: bool = True) -> pd.DataFrame:
    rows = [
        {
            "batter_id": 101,
            "full_name": "Alex Álvarez",
            "retrieved_at": "t0",
            "source": "MLB Stats API (/people)",
        },
        {
            "batter_id": 102,
            "full_name": None,
            "retrieved_at": "t0",
            "source": "MLB Stats API (/people)",
        },
    ]
    if not include_unresolved_row:
        rows.append(
            {
                "batter_id": 103,
                "full_name": "Third Player",
                "retrieved_at": "t0",
                "source": "MLB Stats API (/people)",
            }
        )
    return pd.DataFrame(rows)


def test_resolved_name_is_populated() -> None:
    overlaid, report = ppn.apply_player_name_overlay(_base_table(), _names_df())
    row = overlaid.set_index("batter_id").loc[101]
    assert row["batter_name"] == "Alex Álvarez"
    assert report.reason_codes_by_batter_id["101"] == ppn.REASON_RESOLVED


def test_batter_id_not_in_source_stays_null_with_reason_code() -> None:
    overlaid, report = ppn.apply_player_name_overlay(_base_table(), _names_df())
    row = overlaid.set_index("batter_id").loc[103]
    assert pd.isna(row["batter_name"])
    assert report.reason_codes_by_batter_id["103"] == ppn.REASON_NOT_FOUND_IN_SOURCE


def test_source_returned_null_name_stays_null_with_reason_code() -> None:
    overlaid, report = ppn.apply_player_name_overlay(_base_table(), _names_df())
    row = overlaid.set_index("batter_id").loc[102]
    assert pd.isna(row["batter_name"])
    assert report.reason_codes_by_batter_id["102"] == ppn.REASON_SOURCE_RETURNED_NULL_NAME


def test_no_reason_code_column_added_to_the_table() -> None:
    overlaid, _report = ppn.apply_player_name_overlay(_base_table(), _names_df())
    assert set(overlaid.columns) == set(_base_table().columns)


def test_only_batter_name_column_changes() -> None:
    before = _base_table()
    overlaid, _report = ppn.apply_player_name_overlay(before, _names_df())
    unchanged_cols = [c for c in before.columns if c != "batter_name"]
    assert before[unchanged_cols].equals(overlaid[unchanged_cols])


@pytest.mark.parametrize(
    "score_col",
    [
        "contact_luck_runs_per_100",
        "lower_95_interval",
        "upper_95_interval",
        "official_rank_favorable",
        "official_rank_unfavorable",
        "qualification_status",
    ],
)
def test_specific_score_and_rank_columns_are_never_altered(score_col: str) -> None:
    before = _base_table()
    overlaid, _report = ppn.apply_player_name_overlay(before, _names_df())
    pd.testing.assert_series_equal(before[score_col], overlaid[score_col])


def test_join_is_by_batter_id_not_by_any_name_field() -> None:
    """A batter with NO matching id in the names source stays unresolved even
    if some OTHER row in the names source happens to share a similar name --
    the join never falls back to name-based matching.
    """
    names_df = pd.DataFrame(
        [
            {
                "batter_id": 999,
                "full_name": "Totally Different Person",
                "retrieved_at": "t0",
                "source": "x",
            }
        ]
    )
    overlaid, report = ppn.apply_player_name_overlay(_base_table(), names_df)
    assert overlaid.set_index("batter_id").loc[101:103]["batter_name"].isna().all()
    assert report.unresolved_batter_count == 3


def test_resolving_a_name_does_not_change_rank_ordering() -> None:
    before = _base_table().sort_values("contact_luck_runs_per_100", ascending=False)
    overlaid, _report = ppn.apply_player_name_overlay(
        before, _names_df(include_unresolved_row=False)
    )
    assert list(before["batter_id"]) == list(overlaid["batter_id"])


def test_empty_public_score_table_returns_empty_report() -> None:
    empty = _base_table().iloc[0:0]
    overlaid, report = ppn.apply_player_name_overlay(empty, _names_df())
    assert overlaid.empty
    assert report.requested_batter_count == 0
