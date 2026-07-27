from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mlb_luck_score.scoring.aggregation import (
    AggregationValidationError,
    eligible_event_count,
    negative_raw_luck_runs,
    positive_raw_luck_runs,
    raw_contact_luck_runs_per_100_eligible_events,
    total_raw_contact_luck_runs,
)

VALUES = [0.5, -0.3, 0.2, -0.1, 0.0, 1.2, -0.9]


def test_eligible_event_count():
    assert eligible_event_count(VALUES) == len(VALUES)
    assert eligible_event_count([]) == 0


def test_total_raw_contact_luck_runs_uses_raw_runs():
    assert total_raw_contact_luck_runs(VALUES) == pytest.approx(sum(VALUES))


def test_positive_and_negative_raw_luck_runs_partition_the_total():
    positive = positive_raw_luck_runs(VALUES)
    negative = negative_raw_luck_runs(VALUES)
    zero_sum = sum(v for v in VALUES if v == 0.0)
    assert positive == pytest.approx(sum(v for v in VALUES if v > 0))
    assert negative == pytest.approx(sum(v for v in VALUES if v < 0))
    assert positive + negative + zero_sum == pytest.approx(total_raw_contact_luck_runs(VALUES))


def test_per_100_eligible_events_rate():
    total = total_raw_contact_luck_runs(VALUES)
    n = eligible_event_count(VALUES)
    assert raw_contact_luck_runs_per_100_eligible_events(VALUES) == pytest.approx(total / n * 100.0)


def test_per_100_events_raises_on_empty_input():
    with pytest.raises(AggregationValidationError, match="zero eligible events"):
        raw_contact_luck_runs_per_100_eligible_events([])


def test_rejects_non_finite_values():
    with pytest.raises(AggregationValidationError, match="non-finite"):
        total_raw_contact_luck_runs([1.0, float("nan")])


def test_additivity_matches_summing_contact_luck_directly():
    # Season aggregation must use raw contact-luck runs directly -- summing
    # per-play raw luck values must equal the total, with no other
    # transformation applied (e.g. no implicit scaling or clipping).
    plays = [0.186, -0.254916, 1.033068, -0.05]
    assert total_raw_contact_luck_runs(plays) == pytest.approx(sum(plays))


def test_aggregation_module_never_imports_empirical_score():
    # Static guard against regression: season aggregation must operate only
    # on raw contact-luck runs, never on public display scores.
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mlb_luck_score"
        / "scoring"
        / "aggregation.py"
    )
    tree = ast.parse(source_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert not any("empirical_score" in name or "public_score" in name for name in imported_modules)
