"""Tests for the Contact Luck v0.11 end-to-end aggregation CLI wiring (Phase 7/8).

No test touches the network, and none reads real data -- these specifically
target the 2025-protection precondition shared with `mlb_luck_score.
scoring.run_attribution_ledger` (Version 0.10), which `run_season_
aggregation` reuses unchanged rather than re-implementing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.config import FINAL_TEST_SEASONS
from mlb_luck_score.scoring.run_attribution_ledger import (
    SCRIPT_SEASONS,
    RunAttributionLedgerError,
    _validate_no_final_test_seasons,
)
from mlb_luck_score.scoring.run_season_aggregation import build_arg_parser


def test_final_test_seasons_are_outside_script_seasons():
    # SCRIPT_SEASONS is the hard allowlist BOTH run_attribution_ledger and
    # run_season_aggregation check input data against -- confirm 2025 is
    # genuinely excluded from it, not just absent from a specific test file.
    assert set(FINAL_TEST_SEASONS).isdisjoint(set(SCRIPT_SEASONS))


def test_validate_no_final_test_seasons_rejects_2025_rows():
    df = pd.DataFrame({"season": [2021, 2022, 2025]})
    with pytest.raises(RunAttributionLedgerError, match="2025"):
        _validate_no_final_test_seasons(df)


def test_validate_no_final_test_seasons_accepts_development_seasons():
    df = pd.DataFrame({"season": [2021, 2022, 2023, 2024]})
    observed = _validate_no_final_test_seasons(df)
    assert observed == [2021, 2022, 2023, 2024]


def test_run_season_aggregation_argparser_has_no_final_evaluation_escape_hatch():
    parser = build_arg_parser()
    actions = {action.dest for action in parser._actions}
    assert "allow_final_evaluation" not in actions
