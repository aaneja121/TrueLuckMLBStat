"""Contact Luck v1.3.0 dashboard: `demo_content.py` view-model tests.

Uses small synthetic fixture dicts (not the real committed
`dashboard/demo_fixture.json` -- that's covered end-to-end by
`tests/test_demo_fixture.py`) to exercise `load_demo_page_data`'s
validation: missing/extra examples, invalid probability distributions, and
the observed/expected/Contact-Luck reconciliation check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import demo_content as dc
import pytest


def _example(
    *,
    example_id: str = "hard_contact_out",
    probabilities: dict[str, float] | None = None,
    expected_run_value: float = 1.0,
    observed_run_value: float = -0.25,
    contact_luck_runs: float | None = None,
    outcome_class: str = "out",
) -> dict[str, Any]:
    probs = probabilities or {
        "out": 0.9,
        "single": 0.05,
        "double": 0.03,
        "triple": 0.01,
        "home_run": 0.01,
    }
    if contact_luck_runs is None:
        contact_luck_runs = observed_run_value - expected_run_value
    return {
        "example_id": example_id,
        "narrative_label": "Test example",
        "contact_description": "Test contact",
        "provenance": {
            "batter_name": "Test Player",
            "batter_team": "AAA",
            "opponent_team": "BBB",
            "game_date": "2024-06-01",
            "description": "Test Player does a thing.",
        },
        "contact": {
            "exit_velocity_mph": 95.0,
            "launch_angle_deg": 20.0,
            "spray_angle_deg": 0.0,
            "bb_type": "line_drive",
            "hit_distance_ft": 300.0,
        },
        "expectation": {"probabilities": probs, "expected_run_value": expected_run_value},
        "reality": {
            "outcome_class": outcome_class,
            "outcome_label": outcome_class.capitalize(),
            "observed_run_value": observed_run_value,
        },
        "contact_luck_runs": contact_luck_runs,
    }


def _write_fixture(tmp_path: Path, examples: list[dict[str, Any]]) -> Path:
    path = tmp_path / "demo_fixture.json"
    path.write_text(
        json.dumps(
            {
                "demo_fixture_version": "1.3.0",
                "model_configuration": {"contact_model_variant": "unweighted_probability_baseline"},
                "generator_config_fingerprint": "deadbeef",
                "examples": examples,
            }
        )
    )
    return path


class TestLoadDemoPageData:
    def test_loads_valid_fixture(self, tmp_path: Path) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(
                    example_id="hard_contact_out", expected_run_value=1.0, observed_run_value=-0.25
                ),
                _example(
                    example_id="weak_contact_single",
                    expected_run_value=-0.25,
                    observed_run_value=0.46,
                ),
            ],
        )
        page_data = dc.load_demo_page_data(path)
        assert [e.example_id for e in page_data.examples] == [
            "hard_contact_out",
            "weak_contact_single",
        ]
        assert page_data.examples[0].explanation == (
            "The contact was worth more than the recorded outcome."
        )
        assert page_data.examples[1].explanation == (
            "The recorded outcome was worth more than the contact was expected to produce."
        )

    def test_probability_rows_are_ordered_and_flag_the_observed_outcome(
        self, tmp_path: Path
    ) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(example_id="hard_contact_out", outcome_class="home_run"),
                _example(example_id="weak_contact_single"),
            ],
        )
        page_data = dc.load_demo_page_data(path)
        rows = page_data.examples[0].probability_rows
        assert [r["outcome_class"] for r in rows] == [
            "out",
            "single",
            "double",
            "triple",
            "home_run",
        ]
        observed = [r for r in rows if r["is_observed"]]
        assert len(observed) == 1
        assert observed[0]["outcome_class"] == "home_run"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(tmp_path / "does_not_exist.json")

    def test_missing_required_example_raises(self, tmp_path: Path) -> None:
        path = _write_fixture(tmp_path, [_example(example_id="hard_contact_out")])
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(path)

    def test_unexpected_example_id_raises(self, tmp_path: Path) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(example_id="hard_contact_out"),
                _example(example_id="not_a_real_example"),
            ],
        )
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(path)

    def test_probabilities_not_summing_to_one_raises(self, tmp_path: Path) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(
                    example_id="hard_contact_out",
                    probabilities={
                        "out": 0.5,
                        "single": 0.5,
                        "double": 0.5,
                        "triple": 0.0,
                        "home_run": 0.0,
                    },
                ),
                _example(example_id="weak_contact_single"),
            ],
        )
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(path)

    def test_negative_probability_raises(self, tmp_path: Path) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(
                    example_id="hard_contact_out",
                    probabilities={
                        "out": 1.1,
                        "single": -0.1,
                        "double": 0.0,
                        "triple": 0.0,
                        "home_run": 0.0,
                    },
                ),
                _example(example_id="weak_contact_single"),
            ],
        )
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(path)

    def test_reconciliation_mismatch_raises(self, tmp_path: Path) -> None:
        path = _write_fixture(
            tmp_path,
            [
                _example(
                    example_id="hard_contact_out",
                    expected_run_value=1.0,
                    observed_run_value=-0.25,
                    contact_luck_runs=99.0,  # deliberately wrong
                ),
                _example(example_id="weak_contact_single"),
            ],
        )
        with pytest.raises(dc.DemoContentError):
            dc.load_demo_page_data(path)
