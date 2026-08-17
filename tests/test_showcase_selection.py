"""Version 1.4.1: tests for `demo/build_play_explorer_fixture.py`'s
Showcase Plays selection (`build_showcase_rows`/
`is_showcase_interactive_eligible`) and `patch_showcase_interactive_flags`.

Pure synthetic hand-built ledgers -- no model training, no real data, no
network -- following the exact same pattern `tests/
test_play_explorer_fixture_generator.py` already established for this
module.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import build_play_explorer_fixture as gen
import pandas as pd
import pytest

from mlb_luck_score.scoring.play_ledger_metadata import build_play_ledger_metadata
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_COLUMNS, PLAY_LEDGER_VERSION


def _resolved_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "at_bat_number": 10,
        "pitch_number": 3,
        "game_date": "2024-06-01",
        "season": 2024,
        "batter_id": 12345,
        "stand": "L",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": 0.0,
        "p_out": 0.5,
        "p_single": 0.2,
        "p_double": 0.15,
        "p_triple": 0.1,
        "p_home_run": 0.05,
        "expected_run_value": 0.0,
    }
    row["contact_luck_runs"] = row["observed_run_value"] - row["expected_run_value"]
    row.update(overrides)
    return row


def _unresolved_row(**overrides: Any) -> dict[str, Any]:
    row = _resolved_row(
        is_scored=False,
        outcome_class=pd.NA,
        observed_run_value=pd.NA,
        contact_luck_runs=pd.NA,
        p_out=pd.NA,
        p_single=pd.NA,
        p_double=pd.NA,
        p_triple=pd.NA,
        p_home_run=pd.NA,
        expected_run_value=pd.NA,
    )
    row.update(overrides)
    return row


def _df(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[list(PLAY_LEDGER_COLUMNS)]


def _many_rows(
    n: int, *, luck_start: float, luck_step: float, group_prefix: str
) -> list[dict[str, Any]]:
    """`n` distinct resolved rows with strictly increasing/decreasing
    `contact_luck_runs` (via distinct `expected_run_value`, `observed_run_value`
    fixed at 0.0) -- gives a clean, unambiguous ranking to assert against.
    """
    rows = []
    for i in range(n):
        luck = luck_start + i * luck_step
        rows.append(
            _resolved_row(
                play_id=f"{group_prefix}{i}-1-1",
                game_pk=800000 + i,
                batter_id=90000 + i,
                observed_run_value=0.0,
                expected_run_value=-luck,
                contact_luck_runs=luck,
                game_date=f"2024-0{(i % 9) + 1}-01",
            )
        )
    return rows


class TestOnlyScoredRowsRanked:
    def test_unresolved_rows_never_appear_in_showcase(self) -> None:
        rows = _many_rows(15, luck_start=1.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(15, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows, _unresolved_row(play_id="999999-1-1", batter_id=1))
        showcase = gen.build_showcase_rows(df, {})
        play_ids = {r["play_id"] for r in showcase}
        assert "999999-1-1" not in play_ids


class TestExactlyTopNSelected:
    def test_exactly_12_favorable_and_12_unfavorable(self) -> None:
        rows = _many_rows(20, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(20, luck_start=-5.0, luck_step=0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        favorable = [r for r in showcase if r["group"] == "favorable"]
        unfavorable = [r for r in showcase if r["group"] == "unfavorable"]
        assert len(favorable) == gen.SHOWCASE_TOP_N == 12
        assert len(unfavorable) == gen.SHOWCASE_TOP_N == 12

    def test_favorable_group_is_the_true_highest_luck_extremes(self) -> None:
        rows = _many_rows(20, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(5, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        favorable = sorted(
            (r for r in showcase if r["group"] == "favorable"), key=lambda r: r["rank"]
        )
        luck_values = [r["contact_luck_runs"] for r in favorable]
        assert luck_values == sorted(luck_values, reverse=True)
        assert luck_values[0] == pytest.approx(5.0)

    def test_unfavorable_group_is_the_true_lowest_luck_extremes(self) -> None:
        rows = _many_rows(5, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(20, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        unfavorable = sorted(
            (r for r in showcase if r["group"] == "unfavorable"), key=lambda r: r["rank"]
        )
        luck_values = [r["contact_luck_runs"] for r in unfavorable]
        assert luck_values == sorted(luck_values)
        assert luck_values[0] == pytest.approx(-1.0 - 19 * 0.1)


class TestDeterministicTieOrdering:
    def test_ties_break_on_game_date_then_game_pk_then_play_id(self) -> None:
        base = _many_rows(11, luck_start=5.0, luck_step=-0.5, group_prefix="f")
        # Three rows tied on contact_luck_runs -- must break on game_date,
        # then game_pk, then play_id, all ascending.
        tied = [
            _resolved_row(
                play_id="900002-1-1",
                game_pk=900002,
                batter_id=70000,
                observed_run_value=0.0,
                expected_run_value=-10.0,
                contact_luck_runs=10.0,
                game_date="2024-05-01",
            ),
            _resolved_row(
                play_id="900001-1-1",
                game_pk=900001,
                batter_id=70001,
                observed_run_value=0.0,
                expected_run_value=-10.0,
                contact_luck_runs=10.0,
                game_date="2024-05-01",
            ),
            _resolved_row(
                play_id="900001-1-1-b",
                game_pk=900001,
                batter_id=70002,
                observed_run_value=0.0,
                expected_run_value=-10.0,
                contact_luck_runs=10.0,
                game_date="2024-04-01",
            ),
        ]
        rows = _many_rows(9, luck_start=-1.0, luck_step=-0.1, group_prefix="u") + base + tied
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        favorable = sorted(
            (r for r in showcase if r["group"] == "favorable"), key=lambda r: r["rank"]
        )
        top_three_play_ids = [r["play_id"] for r in favorable[:3]]
        # Earliest game_date first ("2024-04-01" beats "2024-05-01"), then
        # lower game_pk (900001 beats 900002), then play_id ascending.
        assert top_three_play_ids == ["900001-1-1-b", "900001-1-1", "900002-1-1"]

    def test_ranks_are_contiguous_1_through_n_per_group(self) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        for group in ("favorable", "unfavorable"):
            ranks = sorted(r["rank"] for r in showcase if r["group"] == group)
            assert ranks == list(range(1, gen.SHOWCASE_TOP_N + 1))


class TestNoQualificationFilteringOrPerPlayerCap:
    def test_same_batter_appears_more_than_once_when_genuinely_extreme(self) -> None:
        rows = [
            _resolved_row(
                play_id=f"90000{i}-1-1",
                game_pk=900000 + i,
                batter_id=55555,
                observed_run_value=0.0,
                expected_run_value=-(5.0 - i * 0.1),
                contact_luck_runs=5.0 - i * 0.1,
                game_date="2024-05-01",
            )
            for i in range(3)
        ]
        rows += _many_rows(15, luck_start=4.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        batter_55555_rows = [r for r in showcase if r["batter_id"] == 55555]
        assert len(batter_55555_rows) == 3

    def test_no_qualification_related_logic_in_selection_source(self) -> None:
        """Structural guarantee: build_showcase_rows never references any
        qualification/eligibility FILTERING concept in its executable code
        -- it selects purely from scored rows by contact_luck_runs, nothing
        else. (The function's own docstring legitimately explains the
        ABSENCE of qualification filtering in prose, so this checks the
        function body with its docstring stripped, not a bare substring
        match on the whole source.)"""
        import ast
        import inspect
        import textwrap

        source = inspect.getsource(gen.build_showcase_rows)
        tree = ast.parse(textwrap.dedent(source))
        func_node = tree.body[0]
        assert isinstance(func_node, ast.FunctionDef)
        body_without_docstring = (
            func_node.body[1:] if ast.get_docstring(func_node) else func_node.body
        )
        body_source = "\n".join(ast.unparse(node) for node in body_without_docstring)
        for banned in ("qualif", "eligible_batted_balls", "min_bbe"):
            assert banned not in body_source.lower()


class TestCanonicalValuesAndNames:
    def test_canonical_fields_copied_exactly(self) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        by_id = {r["play_id"]: r for r in showcase}
        source_row = next(r for r in rows if r["play_id"] == "f0-1-1")
        showcase_row = by_id["f0-1-1"]
        for field in gen.SHOWCASE_CANONICAL_FIELDS:
            assert (
                showcase_row[field] == pytest.approx(source_row[field])
                if isinstance(source_row[field], float)
                else showcase_row[field] == source_row[field]
            )

    def test_names_from_same_snapshot_overlay(self) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        names = {90000: "Same Snapshot Hitter"}
        showcase = gen.build_showcase_rows(df, names)
        row = next(r for r in showcase if r["batter_id"] == 90000)
        assert row["batter_name"] == "Same Snapshot Hitter"
        other = next(r for r in showcase if r["batter_id"] != 90000)
        assert other["batter_name"] is None


class TestInteractiveEligibility:
    def test_all_required_inputs_present_is_eligible(self) -> None:
        row = pd.Series(_resolved_row())
        assert gen.is_showcase_interactive_eligible(row) is True

    @pytest.mark.parametrize("missing_field", gen.SHOWCASE_INTERACTIVE_ELIGIBILITY_FIELDS)
    def test_any_missing_required_field_is_ineligible(self, missing_field: str) -> None:
        row = pd.Series(_resolved_row(**{missing_field: pd.NA}))
        assert gen.is_showcase_interactive_eligible(row) is False

    def test_true_extreme_kept_even_when_ineligible_for_sliders(self) -> None:
        rows = [
            _resolved_row(
                play_id="f0-1-1",
                batter_id=1,
                launch_speed=pd.NA,
                observed_run_value=0.0,
                expected_run_value=-10.0,
                contact_luck_runs=10.0,
            )
        ]
        rows += _many_rows(15, luck_start=4.0, luck_step=-0.1, group_prefix="f2")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        showcase = gen.build_showcase_rows(df, {})
        row = next(r for r in showcase if r["play_id"] == "f0-1-1")
        assert row["interactive_available"] is False
        assert row["group"] == "favorable"  # still ranked, never dropped


class TestArtifactDeterminism:
    def test_showcase_json_is_byte_identical_across_regenerations(self, tmp_path: Path) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        ledger_path = tmp_path / "play_ledger.parquet"
        df.to_parquet(ledger_path, index=False)
        metadata = build_play_ledger_metadata(
            df, season=2024, score_version="0.2", model_versions={"contact": "x"}
        )
        metadata_path = tmp_path / "play_ledger_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        for out_dir in (out_a, out_b):
            gen.generate_explorer_artifacts(
                play_ledger_path=ledger_path,
                play_ledger_metadata_path=metadata_path,
                output_dir=out_dir,
            )
        assert (out_a / "showcase.json").read_bytes() == (out_b / "showcase.json").read_bytes()


class TestVersionSafety:
    def test_v1_0_ledger_never_produces_showcase(self, tmp_path: Path) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        ledger_path = tmp_path / "play_ledger.parquet"
        df.to_parquet(ledger_path, index=False)
        metadata = build_play_ledger_metadata(
            df, season=2024, score_version="0.2", model_versions={"contact": "x"}
        )
        metadata["play_ledger_version"] = "1.0"
        metadata_path = tmp_path / "play_ledger_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        out_dir = tmp_path / "out"
        with pytest.raises(gen.IncompatiblePlayLedgerVersionError):
            gen.generate_explorer_artifacts(
                play_ledger_path=ledger_path,
                play_ledger_metadata_path=metadata_path,
                output_dir=out_dir,
            )
        assert not out_dir.exists()

    def test_v2_0_ledger_produces_showcase(self, tmp_path: Path) -> None:
        assert PLAY_LEDGER_VERSION == "2.0"
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        ledger_path = tmp_path / "play_ledger.parquet"
        df.to_parquet(ledger_path, index=False)
        metadata = build_play_ledger_metadata(
            df, season=2024, score_version="0.2", model_versions={"contact": "x"}
        )
        metadata_path = tmp_path / "play_ledger_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        out_dir = tmp_path / "out"
        result = gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=out_dir,
        )
        assert (out_dir / "showcase.json").exists()
        assert result["showcase_favorable_count"] == 12
        assert result["showcase_unfavorable_count"] == 12


class TestPatchShowcaseInteractiveFlags:
    def test_patch_downgrades_to_ground_truth(self, tmp_path: Path) -> None:
        rows = _many_rows(12, luck_start=5.0, luck_step=-0.1, group_prefix="f")
        rows += _many_rows(12, luck_start=-1.0, luck_step=-0.1, group_prefix="u")
        df = _df(*rows)
        ledger_path = tmp_path / "play_ledger.parquet"
        df.to_parquet(ledger_path, index=False)
        metadata = build_play_ledger_metadata(
            df, season=2024, score_version="0.2", model_versions={"contact": "x"}
        )
        metadata_path = tmp_path / "play_ledger_metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        out_dir = tmp_path / "out"
        gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=out_dir,
        )
        showcase = json.loads((out_dir / "showcase.json").read_text())
        # Every row is input-eligible by construction (no missing fields).
        assert all(r["interactive_available"] for r in showcase)

        keep_ids = {showcase[0]["play_id"], showcase[1]["play_id"]}
        gen.patch_showcase_interactive_flags(out_dir, keep_ids)

        patched = json.loads((out_dir / "showcase.json").read_text())
        for row in patched:
            assert row["interactive_available"] == (row["play_id"] in keep_ids)

        patched_metadata = json.loads((out_dir / "explore-metadata.json").read_text())
        assert patched_metadata["showcase_interactive_count"] == len(keep_ids)


class TestGeneratorImportsNoModelCode:
    def test_module_never_imports_training_code(self) -> None:
        source = Path(gen.__file__).read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        banned = {"train_contact_model", "train_opportunity_model", "train_advancement_model"}
        assert not (imported & banned)
        assert "predict_proba_ordered(" not in source
