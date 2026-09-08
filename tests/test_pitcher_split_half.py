"""Version 0.14: the question-F split-half implementation.

Offline and data-free. The correlations themselves need a scored season, so
these tests drive the real code path with a small SYNTHETIC
`SeasonAggregationArtifacts` stand-in rather than opening a season -- the
same convention the Version 1.0 evaluation-guard suite uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pitcher_split_half as psh
import pytest

from mlb_luck_score.models.evaluate_aggregation_stability import (
    MIN_ELIGIBLE_EACH_HALF,
    RANKING_METRIC,
)


@dataclass
class FakeArtifacts:
    scoring_df: pd.DataFrame
    ledger: pd.DataFrame
    confidence: pd.DataFrame


class TestTheProcedureIsImportedNotRestated:
    """The port's guarantee is reuse. These tests police that."""

    def test_inclusion_threshold_comes_from_the_accepted_module(
        self, request: pytest.FixtureRequest
    ) -> None:
        assert psh.MIN_ELIGIBLE_EACH_HALF is MIN_ELIGIBLE_EACH_HALF
        assert psh.MIN_ELIGIBLE_EACH_HALF == 20

    def test_metric_comes_from_the_accepted_module(self) -> None:
        assert psh.RANKING_METRIC is RANKING_METRIC
        assert psh.RANKING_METRIC == "observed_minus_expected_per_100"

    def test_split_helpers_are_the_accepted_modules_own(
        self, request: pytest.FixtureRequest
    ) -> None:
        from mlb_luck_score.models import evaluate_aggregation_stability as accepted

        assert psh._half_season_masks is accepted._half_season_masks
        assert psh._odd_even_masks is accepted._odd_even_masks

    def test_no_local_copy_of_the_split_rules(self) -> None:
        from pathlib import Path

        source = Path(psh.__file__).read_text()
        assert "def _half_season_masks" not in source
        assert "def _odd_even_masks" not in source
        assert "MIN_ELIGIBLE_EACH_HALF = " not in source, "the threshold must be imported"
        assert "RANKING_METRIC = " not in source, "the metric must be imported"


class TestFrozenDesign:
    def test_both_splits_are_frozen(self) -> None:
        assert psh.SPLITS == ("calendar", "odd_even")

    def test_the_workload_sweep_is_frozen(self) -> None:
        assert psh.MIN_ELIGIBLE_EACH_HALF_SWEEP == (20, 50, 100)
        assert psh.MIN_ELIGIBLE_EACH_HALF_SWEEP[0] == MIN_ELIGIBLE_EACH_HALF

    def test_an_unknown_split_is_refused(self) -> None:
        with pytest.raises(psh.SplitHalfError, match="Unknown split"):
            psh._masks(pd.DataFrame({"game_pk": [1]}), "random_halves")

    def test_an_unknown_group_is_refused(self) -> None:
        with pytest.raises(psh.SplitHalfError, match="Unknown group"):
            psh.compute_split_half_reliability(
                FakeArtifacts(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), group="team"
            )

    def test_the_committed_batter_reference_is_recorded(
        self, request: pytest.FixtureRequest
    ) -> None:
        ref = psh.BATTER_SIDE_2024_REFERENCE
        assert ref["calendar"]["n_players_compared"] == 399
        assert ref["odd_even"]["n_players_compared"] == 487
        assert ref["calendar"]["pearson_r"] == pytest.approx(0.050297542421047836)
        assert ref["odd_even"]["pearson_r"] == pytest.approx(0.09654304470474931)


def _synthetic_artifacts(request: pytest.FixtureRequest, *, one_pitcher: bool = False) -> Any:
    """Wrap the shared Version 0.11 synthetic ledger fixture as a
    `SeasonAggregationArtifacts` stand-in.

    Reusing `v011_full_ledger` rather than hand-rolling a ledger keeps this
    suite honest about the real schema and offline at the same time -- that
    fixture trains the frozen models on SYNTHETIC rows and opens no season.
    """
    df, ledger, confidence = request.getfixturevalue("v011_full_ledger")
    df = df.copy()
    # A deterministic pitcher assignment: one pitcher for the degenerate case,
    # otherwise spread across four so both halves have several to compare.
    df["pitcher"] = 100 if one_pitcher else 100 + (df.reset_index(drop=True).index % 4)
    return FakeArtifacts(df, ledger, confidence)


class TestRunsOnSyntheticData:
    def test_both_splits_produce_a_result_shape(self, request: pytest.FixtureRequest) -> None:
        artifacts = _synthetic_artifacts(request)
        for split in psh.SPLITS:
            result = psh.compute_split_half_reliability(artifacts, group="pitcher", split=split)
            assert result["group"] == "pitcher"
            assert result["split"] == split
            assert result["min_eligible_each_half"] == 20
            assert result["metric"] == RANKING_METRIC
            assert result["n_players_compared"] >= 0

    def test_too_few_rows_returns_null_correlations_not_an_error(
        self, request: pytest.FixtureRequest
    ) -> None:
        artifacts = _synthetic_artifacts(request, one_pitcher=True)
        result = psh.compute_split_half_reliability(artifacts, group="pitcher", split="odd_even")
        assert result["n_players_compared"] < 2
        assert result["pearson_r"] is None
        assert result["spearman_r"] is None

    def test_raising_the_inclusion_bar_never_grows_the_population(
        self, request: pytest.FixtureRequest
    ) -> None:
        artifacts = _synthetic_artifacts(request)
        loose = psh.compute_split_half_reliability(
            artifacts, group="pitcher", split="odd_even", min_eligible_each_half=20
        )
        tight = psh.compute_split_half_reliability(
            artifacts, group="pitcher", split="odd_even", min_eligible_each_half=100
        )
        assert tight["n_players_compared"] <= loose["n_players_compared"]

    def test_the_sweep_covers_every_frozen_floor(self, request: pytest.FixtureRequest) -> None:
        sweep = psh.compute_workload_sweep(_synthetic_artifacts(request), group="pitcher")
        assert set(sweep) == {
            f"min_eligible_each_half_{floor}" for floor in psh.MIN_ELIGIBLE_EACH_HALF_SWEEP
        }
        for splits in sweep.values():
            assert set(splits) == set(psh.SPLITS)

    def test_the_pitcher_sign_flip_does_not_change_the_correlation(
        self, request: pytest.FixtureRequest
    ) -> None:
        """A correlation is invariant under a common sign flip, so pitcher and
        batter groupings over an identical key must agree exactly.
        """
        artifacts = _synthetic_artifacts(request)
        # One pitcher per batter, so the two groupings partition identically.
        artifacts.scoring_df["batter"] = artifacts.scoring_df["pitcher"]
        pitcher = psh.compute_split_half_reliability(artifacts, group="pitcher", split="odd_even")
        batter = psh.compute_split_half_reliability(artifacts, group="batter", split="odd_even")
        assert pitcher["n_players_compared"] == batter["n_players_compared"]
        if pitcher["pearson_r"] is not None:
            assert pitcher["pearson_r"] == pytest.approx(batter["pearson_r"])
            assert pitcher["spearman_r"] == pytest.approx(batter["spearman_r"])


class TestQuestionFStaysSecondary:
    def test_the_report_declares_its_secondary_status(self, request: pytest.FixtureRequest) -> None:
        artifacts = _synthetic_artifacts(request)
        report = psh.build_question_f_report(artifacts)
        assert report["question"] == "F_persistence"
        assert report["status"] == "secondary_structural_question"
        assert report["is_a_success_criterion_on_its_own"] is False
        assert report["cannot_override_package_classification"] is True

    def test_the_report_carries_the_reproduction_control(
        self, request: pytest.FixtureRequest
    ) -> None:
        report = psh.build_question_f_report(_synthetic_artifacts(request))
        assert "batter_side_reproduction" in report
        assert "implementation_trustworthy" in report

    def test_the_spec_records_the_same_secondary_status(
        self, request: pytest.FixtureRequest
    ) -> None:
        import pitcher_replication_spec as spec

        question = spec.REPLICATION_QUESTIONS["F_persistence"]
        assert question["procedure_status"] == "IMPLEMENTED_AND_FROZEN"
        assert question["implementation_module"] == "replication.pitcher_split_half"
        assert question["is_a_success_criterion_on_its_own"] is False
        assert question["cannot_override_package_classification"] is True

    def test_the_spec_records_the_2024_result_and_its_discrepancy(
        self, request: pytest.FixtureRequest
    ) -> None:
        import pitcher_replication_spec as spec

        result = spec.REPLICATION_QUESTIONS["F_persistence"]["development_2024_result"]
        assert result["pitcher_min_each_half_20"]["calendar"]["n"] == 439
        assert result["pitcher_min_each_half_20"]["odd_even"]["n"] == 550
        assert "no exact prior value to reproduce" in result["known_discrepancy"].lower()
        assert "0.0 on every value" in result["batter_side_reproduction"]
