"""Contact Luck v1.1: frozen-model/threshold preservation, accounting
identities, routing exclusivity, qualified-only rankings, interval pairing,
and provisional-status propagation.

Reuses `tests/conftest.py`'s session-scoped `v012_public_score_artifacts`
fixture (already a full frozen-pipeline `SeasonAggregationArtifacts`) rather
than building a second, redundant heavy synthetic pipeline -- the generic
accounting/routing/ranking checks in `evaluation.v1_system_evaluation`
operate on structural invariants of the OUTPUT, not on which season the rows
happen to carry, exactly as they already do for 2025 in the real Version 1.0
run. `train_and_score_2026`'s own season-validation guards (the part that
genuinely is 2026-specific) are tested directly and cheaply below, since
those checks run before any model is trained.
"""

from __future__ import annotations

import pandas as pd
import prospective_scoring as ps
import pytest
from v1_system_evaluation import (
    verify_play_level_accounting,
    verify_qualification_and_ranking_contract,
    verify_reproducibility,
    verify_routing_exclusivity,
    verify_season_level_accounting,
)

from mlb_luck_score.config import TRAIN_SEASONS
from mlb_luck_score.scoring.aggregation_uncertainty import bootstrap_batter_season_intervals
from mlb_luck_score.scoring.leaderboard import assign_official_ranks
from mlb_luck_score.scoring.public_score_table import build_public_score_table

# ---------------------------------------------------------------------------
# train_and_score_2026: season-validation guards (cheap, no model training)
# ---------------------------------------------------------------------------


def test_refuses_training_data_outside_train_seasons() -> None:
    bad_season = max(TRAIN_SEASONS) + 100
    development_df = pd.DataFrame({"season": [TRAIN_SEASONS[0], bad_season]})
    scoring_df = pd.DataFrame({"season": [ps.PROSPECTIVE_SEASON]})
    with pytest.raises(ps.ProspectiveScoringError, match="outside TRAIN_SEASONS"):
        ps.train_and_score_2026(development_df, scoring_df)


def test_refuses_scoring_data_that_is_not_exclusively_2026() -> None:
    development_df = pd.DataFrame({"season": list(TRAIN_SEASONS)})
    scoring_df = pd.DataFrame({"season": [ps.PROSPECTIVE_SEASON, 2024]})
    with pytest.raises(ps.ProspectiveScoringError, match="ONLY season"):
        ps.train_and_score_2026(development_df, scoring_df)


def test_refuses_scoring_data_from_a_different_single_season() -> None:
    development_df = pd.DataFrame({"season": list(TRAIN_SEASONS)})
    scoring_df = pd.DataFrame({"season": [2024]})
    with pytest.raises(ps.ProspectiveScoringError, match="ONLY season"):
        ps.train_and_score_2026(development_df, scoring_df)


def test_prospective_season_constant_is_2026() -> None:
    assert ps.PROSPECTIVE_SEASON == 2026


# ---------------------------------------------------------------------------
# Frozen-pipeline output invariants (season-agnostic checks, reused fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def public_score_table(v012_public_score_artifacts) -> pd.DataFrame:
    table = build_public_score_table(v012_public_score_artifacts)
    return assign_official_ranks(table)


def test_play_level_accounting_identity_holds(v012_public_score_artifacts) -> None:
    result = verify_play_level_accounting(v012_public_score_artifacts.ledger)
    assert result["identity_holds_for_every_resolved_row"] is True


def test_season_level_accounting_identity_holds(v012_public_score_artifacts) -> None:
    result = verify_season_level_accounting(v012_public_score_artifacts.player_season)
    assert result["identity_holds_for_every_row"] is True


def test_routing_is_mutually_exclusive(v012_public_score_artifacts) -> None:
    result = verify_routing_exclusivity(v012_public_score_artifacts.scoring_df)
    assert result["mutually_exclusive"] is True


def test_qualification_and_ranking_contract_holds(public_score_table: pd.DataFrame) -> None:
    result = verify_qualification_and_ranking_contract(public_score_table)
    assert result["schema_valid"] is True
    assert result["no_rank_on_non_qualified_rows"] is True
    assert result["point_estimates_always_paired_with_intervals"] is True


def test_bootstrap_reproducibility(v012_public_score_artifacts) -> None:
    result = verify_reproducibility(
        bootstrap_batter_season_intervals,
        v012_public_score_artifacts.scoring_df,
        v012_public_score_artifacts.ledger,
        n_reps=50,
        seed=42,
    )
    assert result["deterministic"] is True


def test_provisional_component_status_labels_propagate_unchanged(
    public_score_table: pd.DataFrame,
) -> None:
    """Confirms the frozen pipeline's own real status labels
    ("calibrated_with_limited_subgroup_evidence") survive into the public
    score table's `component_status_reason_codes` verbatim -- never silently
    re-labeled "calibrated" for a prospective snapshot.
    """
    observed_statuses: set[str] = set()
    for reason_codes in public_score_table["component_status_reason_codes"]:
        for component_detail in reason_codes.values():
            observed_statuses.update(component_detail.get("model_status_values", []))
    assert "calibrated_with_limited_subgroup_evidence" in observed_statuses
