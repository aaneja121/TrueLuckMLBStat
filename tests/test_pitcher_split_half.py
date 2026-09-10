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


class TestTheReproductionControlIsScopedToItsOwnSeason:
    """The sealed 2025 run reported `implementation_trustworthy = false`.

    The post-replication audit traced that to the CONTROL's wiring, not to
    the procedure: `reproduce_batter_side` compares against the fixed
    `BATTER_SIDE_2024_REFERENCE`, but measures whatever artifacts its caller
    hands it. `build_question_f_report` is called with the 2025 evaluation
    artifacts during the replication, so the control compared a 2025
    measurement against a 2024 reference -- a season mismatch that cannot
    match regardless of whether the implementation is correct.

    Run on the artifacts it was designed for (the 2024 development season,
    via `make question-f-2024`) the same code reproduces the committed
    Version 0.11 Phase 6 numbers to an absolute difference of 0.0.

    These tests pin that diagnosis. They must not be "fixed" by editing
    `pitcher_split_half`, which is frozen source file 15/15 -- changing it
    would invalidate the freeze the sealed 2025 result is provenance-bound
    to. See `docs/pitcher_replication_2025_question_f_erratum.md`.
    """

    def test_the_reference_is_a_fixed_constant_with_no_season_parameter(self) -> None:
        """The control has no way to know which season it is measuring."""
        import inspect

        signature = inspect.signature(psh.reproduce_batter_side)
        assert "season" not in signature.parameters
        assert set(psh.BATTER_SIDE_2024_REFERENCE) == {"calendar", "odd_even"}

    def test_the_control_cannot_pass_on_artifacts_from_another_population(
        self, request: pytest.FixtureRequest
    ) -> None:
        """Foreign artifacts fail the control even though the procedure is
        the same one that reproduces 2024 exactly.
        """
        reproduction = psh.reproduce_batter_side(_synthetic_artifacts(request))
        assert reproduction["reproduces"] is False
        for split in psh.SPLITS:
            entry = reproduction["splits"][split]
            assert entry["matches"] is False
            # The reference side is untouched: only the measurement differs.
            assert entry["reference"] == psh.BATTER_SIDE_2024_REFERENCE[split]
            assert entry["measured"]["group"] == "batter"

    def test_a_failed_control_still_reports_the_pitcher_figures(
        self, request: pytest.FixtureRequest
    ) -> None:
        """`implementation_trustworthy` is a separate flag; it never blanks
        or alters the pitcher split-half numbers themselves.
        """
        artifacts = _synthetic_artifacts(request)
        report = psh.build_question_f_report(artifacts)
        assert report["implementation_trustworthy"] is False
        for split in psh.SPLITS:
            standalone = psh.compute_split_half_reliability(artifacts, group="pitcher", split=split)
            assert report["pitcher"][split] == standalone

    def test_the_trustworthy_flag_never_reaches_the_package_classification(self) -> None:
        """The classification reads `agrees` only. A false trustworthy flag
        cannot move the verdict -- which is why the sealed REPLICATED
        classification stands.
        """
        import pitcher_replication_questions as prq

        answers = {
            "A_population_and_centering": {"agrees": True},
            "B_opportunity_heterogeneity": {"agrees": True},
            "C_totals_vs_rate": {"agrees": True},
            "D_reliever_single_play_dominance": {"agrees": True},
            "E_rate_precision": {"agrees": True},
            "F_persistence": {"agrees": True, "implementation_trustworthy": False},
            "G_real_play_sanity_check": {"agrees": None},
        }
        verdict = prq.classify(answers)
        assert verdict["classification"] == "REPLICATED"
        assert verdict["primary_disagreements"] == []
        assert verdict["secondary_disagreements"] == []
