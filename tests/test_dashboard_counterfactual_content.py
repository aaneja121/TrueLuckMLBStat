"""Contact Luck v1.3.1 dashboard: `demo_counterfactual_content.py` view
-model tests.

Uses small synthetic grid dicts (not the real committed `dashboard/
demo_counterfactual_grid.json` -- that file's own scientific correctness
is covered end-to-end by `tests/test_counterfactual_grid.py`) to exercise
`load_counterfactual_grid_data`'s structural validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import demo_counterfactual_content as dcc
import pytest


def _reference_play(
    *,
    example_id: str = "hard_contact_out",
    ev_values: list[float] | None = None,
    la_values: list[float] | None = None,
    grid: list[list[float]] | None = None,
    ev_index: int = 1,
    la_index: int = 1,
    original_outcome_class: str = "out",
) -> dict[str, Any]:
    ev_values = ev_values if ev_values is not None else [70.0, 80.0, 90.0]
    la_values = la_values if la_values is not None else [10.0, 20.0, 30.0]
    n_ev, n_la = len(ev_values), len(la_values)
    if grid is None:
        grid = [[0.5, 0.2, 0.15, 0.1, 0.05] for _ in range(n_ev * n_la)]
    return {
        "example_id": example_id,
        "fixed_context": {
            "hit_distance_sc": 300.0,
            "spray_angle_approx": 0.0,
            "bb_type": "fly_ball",
            "stand": "R",
        },
        "original_exit_velocity_mph": ev_values[ev_index],
        "original_launch_angle_deg": la_values[la_index],
        "original_outcome_class": original_outcome_class,
        "original_grid_index": {"ev_index": ev_index, "la_index": la_index},
        "exit_velocity_values": ev_values,
        "launch_angle_values": la_values,
        "grid_shape": {"n_ev": n_ev, "n_la": n_la},
        "grid": grid,
    }


def _write_grid(tmp_path: Path, reference_plays: list[dict[str, Any]]) -> Path:
    path = tmp_path / "demo_counterfactual_grid.json"
    path.write_text(
        json.dumps(
            {
                "demo_counterfactual_grid_version": "1.3.1",
                "model_configuration": {
                    "run_value_table": {
                        "out": -0.25,
                        "single": 0.46,
                        "double": 0.76,
                        "triple": 1.03,
                        "home_run": 1.40,
                    }
                },
                "counterfactual_semantics": {
                    "type": "ceteris_paribus_model_sensitivity",
                    "varied_features": ["launch_speed", "launch_angle"],
                    "fixed_features": [
                        "hit_distance_sc",
                        "spray_angle_approx",
                        "bb_type",
                        "stand",
                    ],
                    "interpretation": (
                        "Model sensitivity only; not a physical ball-flight reconstruction."
                    ),
                },
                "generator_config_fingerprint": "deadbeef",
                "reference_plays": {rp["example_id"]: rp for rp in reference_plays},
            }
        )
    )
    return path


def _both_plays(**overrides: Any) -> list[dict[str, Any]]:
    hard = _reference_play(example_id="hard_contact_out")
    for k, v in overrides.items():
        hard[k] = v
    return [hard, _reference_play(example_id="weak_contact_single")]


class TestLoadCounterfactualGridData:
    def test_loads_valid_grid(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays())
        data = dcc.load_counterfactual_grid_data(path)
        assert set(data.reference_plays) == {"hard_contact_out", "weak_contact_single"}
        assert data.run_value_table["out"] == -0.25
        play = data.reference_plays["hard_contact_out"]
        assert play.exit_velocity_values == [70.0, 80.0, 90.0]
        assert play.original_exit_velocity_mph == 80.0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(tmp_path / "does_not_exist.json")

    def test_missing_required_reference_play_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, [_reference_play(example_id="hard_contact_out")])
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_unexpected_reference_play_id_raises(self, tmp_path: Path) -> None:
        path = _write_grid(
            tmp_path,
            [
                _reference_play(example_id="hard_contact_out"),
                _reference_play(example_id="not_a_real_example"),
            ],
        )
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_unsorted_axis_values_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays(exit_velocity_values=[90.0, 70.0, 80.0]))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_grid_shape_mismatch_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays(grid_shape={"n_ev": 99, "n_la": 3}))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_grid_length_mismatch_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays(grid=[[0.5, 0.2, 0.15, 0.1, 0.05]]))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_original_grid_index_pointing_at_wrong_ev_raises(self, tmp_path: Path) -> None:
        # Real value no longer matches exit_velocity_values[ev_index].
        path = _write_grid(tmp_path, _both_plays(original_exit_velocity_mph=999.0))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_original_grid_index_out_of_range_raises(self, tmp_path: Path) -> None:
        path = _write_grid(
            tmp_path, _both_plays(original_grid_index={"ev_index": 99, "la_index": 1})
        )
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_invalid_probability_row_raises(self, tmp_path: Path) -> None:
        bad_grid = [[2.0, 0.2, 0.15, 0.1, 0.05] for _ in range(9)]
        path = _write_grid(tmp_path, _both_plays(grid=bad_grid))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_missing_fixed_context_feature_raises(self, tmp_path: Path) -> None:
        path = _write_grid(
            tmp_path, _both_plays(fixed_context={"hit_distance_sc": 300.0, "bb_type": "fly_ball"})
        )
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_wrong_varied_features_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays())
        raw = json.loads(path.read_text())
        raw["counterfactual_semantics"]["varied_features"] = ["launch_speed"]
        path.write_text(json.dumps(raw))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)

    def test_missing_fixed_feature_name_in_semantics_raises(self, tmp_path: Path) -> None:
        path = _write_grid(tmp_path, _both_plays())
        raw = json.loads(path.read_text())
        raw["counterfactual_semantics"]["fixed_features"] = ["hit_distance_sc"]
        path.write_text(json.dumps(raw))
        with pytest.raises(dcc.CounterfactualContentError):
            dcc.load_counterfactual_grid_data(path)
