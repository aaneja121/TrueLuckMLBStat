"""Version 1.4.1: tests for `demo/build_showcase_sensitivity.py`, the
"What if?" sensitivity artifact generator and its mandatory reconciliation
gate.

Pure logic (eligibility, candidate selection) runs fully offline with
synthetic data. The reconciliation-gate/grid-construction tests need the
real, ~500MB, local-only 2021-2024 development parquet (never committed,
never fetched automatically -- see README.md "Full development dataset")
and are skipped when it's absent, exactly mirroring `tests/
test_counterfactual_grid.py`'s own real-data-gated pattern. This file never
touches 2025/2026 data and never contacts the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import build_showcase_sensitivity as sens
import pandas as pd
import pytest

from mlb_luck_score.config import CLASS_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DEVELOPMENT_DATA_PATH = (
    REPO_ROOT / "data" / "processed" / "cleaned_development_data_with_sprint_speed.parquet"
)

_requires_real_dev_data = pytest.mark.skipif(
    not REAL_DEVELOPMENT_DATA_PATH.exists(),
    reason="requires the real, local-only 2021-2024 development parquet (see README.md "
    "'Full development dataset') -- not present in CI",
)


def _play_row(**overrides: Any) -> pd.Series:
    row: dict[str, Any] = {
        "play_id": "700001-10-3",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "bb_type": "fly_ball",
        "stand": "L",
        "p_out": 0.5,
        "p_single": 0.2,
        "p_double": 0.15,
        "p_triple": 0.1,
        "p_home_run": 0.05,
        "expected_run_value": 0.0,
    }
    row.update(overrides)
    return pd.Series(row)


class TestIsInteractiveEligible:
    def test_all_present_is_eligible(self) -> None:
        assert sens.is_interactive_eligible(_play_row()) is True

    @pytest.mark.parametrize("field", sens.REQUIRED_SENSITIVITY_INPUT_COLUMNS)
    def test_any_missing_field_is_ineligible(self, field: str) -> None:
        assert sens.is_interactive_eligible(_play_row(**{field: pd.NA})) is False

    def test_required_columns_match_the_generators_own_contract(self) -> None:
        """Duplicated (not imported) contract -- see this module's own
        docstring on why -- but the two lists must never silently drift."""
        import build_play_explorer_fixture as explorer_gen

        assert set(sens.REQUIRED_SENSITIVITY_INPUT_COLUMNS) == set(
            explorer_gen.SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS
        )


class TestSelectInteractiveCandidates:
    def _rows(self, group: str, n: int, *, eligible_count: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(n):
            rows.append(
                {
                    "play_id": f"{group}-{i}",
                    "group": group,
                    "rank": i + 1,
                    "interactive_available": i < eligible_count,
                }
            )
        return rows

    def test_caps_at_six_per_group_by_default(self) -> None:
        rows = self._rows("favorable", 12, eligible_count=12) + self._rows(
            "unfavorable", 12, eligible_count=12
        )
        selected = sens.select_interactive_candidates(rows)
        favorable_selected = [r for r in selected if r["group"] == "favorable"]
        unfavorable_selected = [r for r in selected if r["group"] == "unfavorable"]
        assert len(favorable_selected) == 6
        assert len(unfavorable_selected) == 6

    def test_only_input_eligible_rows_selected(self) -> None:
        rows = self._rows("favorable", 12, eligible_count=3) + self._rows(
            "unfavorable", 12, eligible_count=12
        )
        selected = sens.select_interactive_candidates(rows)
        favorable_selected = [r for r in selected if r["group"] == "favorable"]
        assert len(favorable_selected) == 3
        assert all(r["interactive_available"] for r in favorable_selected)

    def test_selects_highest_ranked_eligible_rows_first(self) -> None:
        rows = self._rows("favorable", 12, eligible_count=12) + self._rows(
            "unfavorable", 12, eligible_count=12
        )
        selected = sens.select_interactive_candidates(rows)
        favorable_ranks = sorted(r["rank"] for r in selected if r["group"] == "favorable")
        assert favorable_ranks == [1, 2, 3, 4, 5, 6]

    def test_respects_custom_max_per_group(self) -> None:
        rows = self._rows("favorable", 12, eligible_count=12) + self._rows(
            "unfavorable", 12, eligible_count=12
        )
        selected = sens.select_interactive_candidates(rows, max_per_group=2)
        assert len([r for r in selected if r["group"] == "favorable"]) == 2

    def test_no_candidates_when_nothing_eligible(self) -> None:
        rows = self._rows("favorable", 12, eligible_count=0) + self._rows(
            "unfavorable", 12, eligible_count=0
        )
        assert sens.select_interactive_candidates(rows) == []


@_requires_real_dev_data
class TestRealDataReconciliationGate:
    """Uses the REAL frozen baseline_v02 contact model, retrained fresh
    from the real 2021-2024 development dataset -- the exact recipe
    production scoring uses. Never touches 2025/2026 data.
    """

    @pytest.fixture(scope="class")
    def trained_and_train_df(self):
        return sens.train_frozen_contact_model(REAL_DEVELOPMENT_DATA_PATH)

    def test_own_coordinate_score_reconciles_with_itself(self, trained_and_train_df) -> None:
        """The core safety claim this module exists to prove: scoring a
        real row at its own (EV, LA) and reconciling against a canonical
        row built from THAT SAME prediction must pass -- proves the full
        real-data plumbing (feature columns, grid construction, gate
        arithmetic) works end-to-end, not just on synthetic data.
        """
        trained, train_df = trained_and_train_df
        feature_cols = trained.numeric_features + trained.categorical_features
        real_row = train_df.iloc[0]

        from mlb_luck_score.models.train_contact_model import (
            predict_proba_ordered,
            validate_probabilities,
        )
        from mlb_luck_score.scoring.contact_luck import compute_expected_run_value
        from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

        x = pd.DataFrame([real_row[feature_cols]])[feature_cols]
        proba = predict_proba_ordered(trained, x)
        validate_probabilities(proba)
        canonical_probs = {cls: float(proba.iloc[0][cls]) for cls in CLASS_ORDER}
        canonical_erv = compute_expected_run_value(
            canonical_probs, run_value_map=DEFAULT_RUN_VALUE_MAP
        )

        canonical_row = pd.Series(
            {
                "play_id": "real-row-0",
                "launch_speed": float(real_row["launch_speed"]),
                "launch_angle": float(real_row["launch_angle"]),
                "spray_angle_approx": float(real_row["spray_angle_approx"]),
                "hit_distance_sc": float(real_row["hit_distance_sc"]),
                "bb_type": str(real_row["bb_type"]),
                "stand": str(real_row["stand"]),
                "expected_run_value": canonical_erv,
                **{f"p_{cls}": canonical_probs[cls] for cls in CLASS_ORDER},
            }
        )

        artifact = sens.build_sensitivity_artifact(trained, train_df, canonical_row)
        assert artifact["play_id"] == "real-row-0"
        assert artifact["original_expected_run_value"] == pytest.approx(canonical_erv, abs=1e-6)
        # Direct grid lookup at the real coordinate reproduces the exact
        # probabilities -- same "one full-precision cell" invariant
        # demo/build_counterfactual_grid.py establishes.
        ev_idx = artifact["original_grid_index"]["ev_index"]
        la_idx = artifact["original_grid_index"]["la_index"]
        row_index = ev_idx * len(artifact["launch_angle_values"]) + la_idx
        cell = artifact["grid"][row_index]
        for i, cls in enumerate(CLASS_ORDER):
            assert cell[i] == pytest.approx(canonical_probs[cls], abs=1e-9)

    def test_tampered_canonical_value_fails_the_gate(self, trained_and_train_df) -> None:
        trained, train_df = trained_and_train_df
        feature_cols = trained.numeric_features + trained.categorical_features
        real_row = train_df.iloc[1]

        from mlb_luck_score.models.train_contact_model import (
            predict_proba_ordered,
            validate_probabilities,
        )

        x = pd.DataFrame([real_row[feature_cols]])[feature_cols]
        proba = predict_proba_ordered(trained, x)
        validate_probabilities(proba)
        canonical_probs = {cls: float(proba.iloc[0][cls]) for cls in CLASS_ORDER}

        tampered_row = pd.Series(
            {
                "play_id": "tampered-row-1",
                "launch_speed": float(real_row["launch_speed"]),
                "launch_angle": float(real_row["launch_angle"]),
                "spray_angle_approx": float(real_row["spray_angle_approx"]),
                "hit_distance_sc": float(real_row["hit_distance_sc"]),
                "bb_type": str(real_row["bb_type"]),
                "stand": str(real_row["stand"]),
                # expected_run_value deliberately wrong by far more than tolerance.
                "expected_run_value": 999.0,
                **{f"p_{cls}": canonical_probs[cls] for cls in CLASS_ORDER},
            }
        )

        with pytest.raises(sens.ShowcaseSensitivityReconciliationError):
            sens.build_sensitivity_artifact(trained, train_df, tampered_row)

    def test_missing_required_input_raises_before_scoring(self, trained_and_train_df) -> None:
        trained, train_df = trained_and_train_df
        row = _play_row(launch_speed=pd.NA)
        with pytest.raises(sens.ShowcaseSensitivityError):
            sens.build_sensitivity_artifact(trained, train_df, row)

    def test_generate_showcase_sensitivity_artifacts_end_to_end(
        self, tmp_path: Path, trained_and_train_df
    ) -> None:
        trained, train_df = trained_and_train_df
        feature_cols = trained.numeric_features + trained.categorical_features

        from mlb_luck_score.models.train_contact_model import (
            predict_proba_ordered,
            validate_probabilities,
        )
        from mlb_luck_score.scoring.contact_luck import compute_expected_run_value
        from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

        real_rows = train_df.iloc[:2]
        ledger_rows = []
        showcase_rows = []
        for i, (_, real_row) in enumerate(real_rows.iterrows()):
            x = pd.DataFrame([real_row[feature_cols]])[feature_cols]
            proba = predict_proba_ordered(trained, x)
            validate_probabilities(proba)
            probs = {cls: float(proba.iloc[0][cls]) for cls in CLASS_ORDER}
            erv = compute_expected_run_value(probs, run_value_map=DEFAULT_RUN_VALUE_MAP)
            play_id = f"real-{i}"
            ledger_rows.append(
                {
                    "play_id": play_id,
                    "launch_speed": float(real_row["launch_speed"]),
                    "launch_angle": float(real_row["launch_angle"]),
                    "spray_angle_approx": float(real_row["spray_angle_approx"]),
                    "hit_distance_sc": float(real_row["hit_distance_sc"]),
                    "bb_type": str(real_row["bb_type"]),
                    "stand": str(real_row["stand"]),
                    "expected_run_value": erv,
                    **{f"p_{cls}": probs[cls] for cls in CLASS_ORDER},
                }
            )
            showcase_rows.append(
                {
                    "play_id": play_id,
                    "group": "favorable" if i == 0 else "unfavorable",
                    "rank": 1,
                    "interactive_available": True,
                }
            )

        play_ledger = pd.DataFrame(ledger_rows)
        output_dir = tmp_path / "showcase-sensitivity"
        result = sens.generate_showcase_sensitivity_artifacts(
            play_ledger=play_ledger,
            showcase_rows=showcase_rows,
            dev_data_path=REAL_DEVELOPMENT_DATA_PATH,
            output_dir=output_dir,
        )
        assert set(result["interactive_play_ids"]) == {"real-0", "real-1"}
        assert (output_dir / "real-0.json").exists()
        assert (output_dir / "real-1.json").exists()

    def test_generate_writes_nothing_if_any_candidate_fails_reconciliation(
        self, tmp_path: Path, trained_and_train_df
    ) -> None:
        """The orchestration-level atomicity claim: if even ONE of several
        selected candidates fails the gate, generate_showcase_sensitivity_
        artifacts must raise and leave NO files on disk -- not even for the
        candidate(s) that would have individually passed. This is the
        multi-candidate case; test_tampered_canonical_value_fails_the_gate
        above only exercises the single-artifact function directly.
        """
        trained, train_df = trained_and_train_df
        feature_cols = trained.numeric_features + trained.categorical_features

        from mlb_luck_score.models.train_contact_model import (
            predict_proba_ordered,
            validate_probabilities,
        )
        from mlb_luck_score.scoring.contact_luck import compute_expected_run_value
        from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

        real_rows = train_df.iloc[:2]
        ledger_rows = []
        showcase_rows = []
        for i, (_, real_row) in enumerate(real_rows.iterrows()):
            x = pd.DataFrame([real_row[feature_cols]])[feature_cols]
            proba = predict_proba_ordered(trained, x)
            validate_probabilities(proba)
            probs = {cls: float(proba.iloc[0][cls]) for cls in CLASS_ORDER}
            erv = compute_expected_run_value(probs, run_value_map=DEFAULT_RUN_VALUE_MAP)
            play_id = f"atomic-{i}"
            ledger_rows.append(
                {
                    "play_id": play_id,
                    "launch_speed": float(real_row["launch_speed"]),
                    "launch_angle": float(real_row["launch_angle"]),
                    "spray_angle_approx": float(real_row["spray_angle_approx"]),
                    "hit_distance_sc": float(real_row["hit_distance_sc"]),
                    "bb_type": str(real_row["bb_type"]),
                    "stand": str(real_row["stand"]),
                    # Row 0 (would-be-valid, first in candidate order) keeps
                    # its real canonical erv; row 1 is deliberately tampered
                    # so this specific candidate fails the gate.
                    "expected_run_value": erv if i == 0 else 999.0,
                    **{f"p_{cls}": probs[cls] for cls in CLASS_ORDER},
                }
            )
            showcase_rows.append(
                {
                    "play_id": play_id,
                    "group": "favorable" if i == 0 else "unfavorable",
                    "rank": 1,
                    "interactive_available": True,
                }
            )

        play_ledger = pd.DataFrame(ledger_rows)
        output_dir = tmp_path / "showcase-sensitivity"
        with pytest.raises(sens.ShowcaseSensitivityReconciliationError):
            sens.generate_showcase_sensitivity_artifacts(
                play_ledger=play_ledger,
                showcase_rows=showcase_rows,
                dev_data_path=REAL_DEVELOPMENT_DATA_PATH,
                output_dir=output_dir,
            )
        # Atomic: not even atomic-0.json (which would have individually
        # passed) exists -- no partial directory, no partial write.
        assert not output_dir.exists()

    def test_generate_returns_empty_when_no_candidates(self, tmp_path: Path) -> None:
        showcase_rows = [
            {"play_id": "x-1", "group": "favorable", "rank": 1, "interactive_available": False}
        ]
        result = sens.generate_showcase_sensitivity_artifacts(
            play_ledger=pd.DataFrame(),
            showcase_rows=showcase_rows,
            dev_data_path=REAL_DEVELOPMENT_DATA_PATH,
            output_dir=tmp_path / "showcase-sensitivity",
        )
        assert result["interactive_play_ids"] == []
        assert not (tmp_path / "showcase-sensitivity").exists()


class TestTrainFrozenContactModelMissingDevData:
    def test_raises_clear_error_when_dev_data_missing(self, tmp_path: Path) -> None:
        with pytest.raises(sens.ShowcaseSensitivityError, match="not found"):
            sens.train_frozen_contact_model(tmp_path / "does_not_exist.parquet")
