"""Version 0.14: the 2026-09-08 execution failure, and the proposed
training-frame correction.

Synthetic only. No test here reads the held-out 2025 artifacts, opens 2025 or
2026, or computes any replication result. The bug and the fix are both
demonstrated with hand-built season labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pitcher_replication_incident as inc
import pytest

from mlb_luck_score.config import TRAIN_SEASONS


def _development_like(seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)) -> pd.DataFrame:
    """A stand-in for the frozen development parquet: several seasons, a few
    rows each. Synthetic labels only.
    """
    rows = [{"season": season, "value": i} for season in seasons for i in range(5)]
    return pd.DataFrame(rows)


def _evaluation_like(season: int = 2025) -> pd.DataFrame:
    return pd.DataFrame([{"season": season, "value": i} for i in range(5)])


class TestReproducesTheBug:
    """The failure was a contract mismatch, and it is reproducible without
    any real data.
    """

    def test_the_unfiltered_frame_carries_2024(self) -> None:
        development_df = _development_like()
        assert 2024 in inc.training_seasons_of(development_df)

    def test_the_frozen_guard_refuses_a_training_frame_containing_2024(self) -> None:
        """The exact precondition `train_and_score_2025` applies, exercised
        directly rather than by running the frozen scorer.
        """
        development_df = _development_like()
        observed = sorted(inc.training_seasons_of(development_df))
        outside = [s for s in observed if s not in TRAIN_SEASONS]
        assert outside == [2024], "this is the season the real run tripped on"

    def test_the_guard_is_a_precondition_not_a_filter(self) -> None:
        """The heart of the defect: the callee validates, it does not filter,
        so the caller must filter.
        """
        source = Path("evaluation/run_v1_final_evaluation.py").resolve()
        fn = source.read_text().split("def train_and_score_2025(", 1)[1].split("\ndef ", 1)[0]
        guard = fn.index("Training data contains season(s) outside")
        assert "development_df[development_df" not in fn[:guard], (
            "train_and_score_2025 must not silently filter; it raises, and the caller "
            "is responsible for passing already-filtered training data"
        )

    def test_the_guard_precedes_every_model_fit(self) -> None:
        """Proves no model was fit on 2024 during the failed run."""
        source = Path("evaluation/run_v1_final_evaluation.py").resolve()
        fn = source.read_text().split("def train_and_score_2025(", 1)[1].split("\ndef ", 1)[0]
        guard = fn.index("Training data contains season(s) outside")
        assert guard < fn.index("_apply_eligibility_pipeline(development_df)")
        for fit_call in (
            "train_model(",
            "train_opportunity_model(",
            "run_infield_model_selection(",
            "fit_advancement_contact_model(",
            "run_advancement_model_selection(",
            "run_near_wall_model_selection(",
        ):
            assert fn.index(fit_call) > guard, f"{fit_call} must come after the guard"

    def test_the_runner_now_filters_before_calling_the_frozen_scorer(self) -> None:
        """The correction IS applied: the defect assertion is inverted.

        This is the regression guard. If the filter is ever removed the run
        would again pass 2024 into training, and the frozen guard would again
        refuse -- after 2025 had been re-opened.
        """
        runner = Path("replication/run_pitcher_replication_2025.py").resolve().read_text()
        body = runner.split("def run_replication", 1)[1]
        assert 'development_df["season"].isin(TRAIN_SEASONS)' in body
        assert "train_and_score_2025(training_df, evaluation_df)" in body
        assert "train_and_score_2025(development_df, evaluation_df)" not in body, (
            "the unfiltered call is the defect that failed execution 8edc32d6ca8830ce"
        )

    def test_the_runner_filters_before_it_calls(self) -> None:
        """Ordering, not just presence: the filter must precede the call."""
        body = (
            Path("replication/run_pitcher_replication_2025.py")
            .resolve()
            .read_text()
            .split("def run_replication", 1)[1]
        )
        assert body.index("isin(TRAIN_SEASONS)") < body.index("train_and_score_2025(training_df")


class TestTheAppliedCorrection:
    def test_filtering_yields_exactly_the_frozen_training_seasons(self) -> None:
        training = inc.select_training_frame(_development_like())
        assert inc.training_seasons_of(training) == {2021, 2022, 2023}

    def test_2024_cannot_enter_training(self) -> None:
        training = inc.select_training_frame(_development_like())
        assert 2024 not in inc.training_seasons_of(training)

    def test_2025_cannot_enter_training(self) -> None:
        contaminated = pd.concat([_development_like(), _evaluation_like()], ignore_index=True)
        training = inc.select_training_frame(contaminated)
        assert inc.training_seasons_of(training) == {2021, 2022, 2023}
        assert 2025 not in inc.training_seasons_of(training)

    def test_2026_cannot_enter_training(self) -> None:
        contaminated = pd.concat(
            [_development_like(), _evaluation_like(season=2026)], ignore_index=True
        )
        assert inc.training_seasons_of(inc.select_training_frame(contaminated)) == {
            2021,
            2022,
            2023,
        }

    def test_evaluation_frame_stays_2025_only(self) -> None:
        assert inc.training_seasons_of(_evaluation_like()) == {2025}

    def test_the_corrected_pair_satisfies_the_frozen_guard(self) -> None:
        """Both preconditions `train_and_score_2025` checks now hold."""
        training = inc.select_training_frame(_development_like())
        evaluation = _evaluation_like()
        assert not [s for s in inc.training_seasons_of(training) if s not in TRAIN_SEASONS]
        assert sorted(inc.training_seasons_of(evaluation)) == [2025]

    def test_the_correction_matches_version_1_0s_own_expression(self) -> None:
        """The fix is not invented: it is the filter the v1.0 caller already
        applies before the same call.
        """
        source = Path("evaluation/run_v1_final_evaluation.py").resolve().read_text()
        assert 'full_development_df[full_development_df["season"].isin(TRAIN_SEASONS)]' in source

    def test_the_frozen_scorer_still_performs_its_own_guard(self) -> None:
        """The correction does not remove the callee's precondition check --
        belt and braces, both must remain.
        """
        source = Path("evaluation/run_v1_final_evaluation.py").resolve()
        fn = source.read_text().split("def train_and_score_2025(", 1)[1].split("\ndef ", 1)[0]
        assert "Training data contains season(s) outside TRAIN_SEASONS" in fn
        assert "Evaluation dataset must contain ONLY season" in fn

    def test_the_raw_development_frame_may_still_contain_2021_to_2024(self) -> None:
        """The source parquet is unchanged; only what is PASSED changes."""
        assert inc.training_seasons_of(_development_like()) == {2021, 2022, 2023, 2024}

    def test_filtering_changes_no_scientific_setting(self) -> None:
        """Nothing in the frozen spec moves. The correction is plumbing."""
        import pitcher_replication_freeze as prf
        import pitcher_replication_spec as spec

        assert spec.ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE == 10.0
        assert spec.ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE == 8.0
        assert spec.BOARD_DISPLAY_MINIMUM_BBE == 60
        assert spec.PRIMARY_QUANTITY["name"] == "cumulative_contact_luck_runs"
        assert spec.SECONDARY_QUANTITY["may_be_the_official_ranking_key"] is False
        assert spec.DENOMINATOR["redefined_for_the_pitcher_stage"] is False
        assert spec.INTERVAL_PROCEDURE["may_be_changed_after_seeing_2025"] is False
        # And the research freeze itself is untouched by the incident.
        assert prf.validate_freeze(prf.read_freeze())["valid"] is True

    def test_an_empty_training_frame_is_refused(self) -> None:
        with pytest.raises(inc.IncidentError, match="no training rows"):
            inc.select_training_frame(_development_like(seasons=(2024,)))

    def test_a_frame_without_a_season_column_is_refused(self) -> None:
        with pytest.raises(inc.IncidentError, match="no 'season' column"):
            inc.select_training_frame(pd.DataFrame({"value": [1]}))


class TestIncidentRecord:
    def test_the_log_is_append_only(self, tmp_path: Path) -> None:
        path = tmp_path / "failures.jsonl"
        inc.append_failure_record({"incident": "first"}, path=path)
        inc.append_failure_record({"incident": "second"}, path=path)
        records = inc.read_failure_records(path)
        assert [r["incident"] for r in records] == ["first", "second"]

    def test_recording_never_mutates_the_receipt_or_manifest(self, tmp_path: Path) -> None:
        from pitcher_replication_execution import (
            EXECUTION_MANIFEST_PATH,
            EXECUTION_RECEIPT_PATH,
        )

        before = [
            p.read_bytes() for p in (EXECUTION_RECEIPT_PATH, EXECUTION_MANIFEST_PATH) if p.exists()
        ]
        inc.append_failure_record({"incident": "probe"}, path=tmp_path / "f.jsonl")
        after = [
            p.read_bytes() for p in (EXECUTION_RECEIPT_PATH, EXECUTION_MANIFEST_PATH) if p.exists()
        ]
        assert before == after

    def test_the_inventory_hashes_bytes_without_opening_contents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An incident record must never become a reason to read held-out
        results, so the inventory must not go through a dataframe reader.
        """

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the inventory must not read file CONTENTS")

        for name in ("read_parquet", "read_csv", "read_json"):
            monkeypatch.setattr(pd, name, explode)

        data_dir = tmp_path / "2025"
        data_dir.mkdir()
        (data_dir / "statcast_2025.parquet").write_bytes(b"synthetic-bytes")
        entries = inc.inventory_2025_artifacts(data_dir)
        assert len(entries) == 1
        assert entries[0]["bytes"] == len(b"synthetic-bytes")
        assert len(entries[0]["sha256"]) == 64

    def test_the_recorded_incident_is_present_and_well_formed(self) -> None:
        records = inc.read_failure_records()
        if not records:
            pytest.skip("no incident recorded in this checkout")
        record = records[-1]
        assert record["exception_type"] == "FinalEvaluationError"
        assert record["guard_fired_before_model_fitting"] is True
        assert record["is_a_first_look"] is False
        exposure = record["exposure"]
        assert exposure["receipt_written"] is True
        assert exposure["held_out_2025_ingested"] is True
        assert exposure["model_parameters_fit"] is False
        assert exposure["questions_a_to_g_computed"] is False
        assert exposure["classification_computed"] is False
        assert json.dumps(record, default=str)

    def test_no_replication_result_exists(self) -> None:
        outputs = Path("outputs/pitcher_replication/v0_14")
        files = (
            [p for p in outputs.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            if outputs.exists()
            else []
        )
        assert files == [], "no A-G, no classification, no results may exist"

    def test_the_receipt_is_still_present(self) -> None:
        from pitcher_replication_execution import EXECUTION_RECEIPT_PATH

        assert EXECUTION_RECEIPT_PATH.exists(), "the receipt must never be deleted"
