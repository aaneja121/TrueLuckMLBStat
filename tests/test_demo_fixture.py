"""Contact Luck v1.3.0: validates the COMMITTED `dashboard/demo_fixture.json`
against the CURRENT live `mlb_luck_score` constants and formulas.

This is the "future code changes cannot silently produce different demo
values" guard: everything below runs fully offline (no real Statcast/
development data needed, only the committed fixture file plus the frozen
scoring/config modules), and every check recomputes a real value from the
live code and compares it to what the fixture claims -- if `DEFAULT_RUN_
VALUE_MAP`, `CLASS_ORDER`, `TRAIN_SEASONS`, or the default `class_weight`
ever drift from what `demo/build_demo_fixture.py` recorded, these tests fail
rather than leaving the demo page silently showing stale, no-longer
-reproducible numbers. See `demo/build_demo_fixture.py`'s module docstring.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from mlb_luck_score.config import CLASS_ORDER, TRAIN_SEASONS
from mlb_luck_score.models.train_contact_model import RANDOM_SEED, VARIANT_UNWEIGHTED
from mlb_luck_score.scoring.contact_luck import (
    compute_expected_run_value,
    compute_raw_contact_luck_runs,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "demo_fixture.json"
REQUIRED_EXAMPLE_IDS = {"hard_contact_out", "weak_contact_single"}


@pytest.fixture(scope="module")
def fixture() -> dict:
    assert FIXTURE_PATH.exists(), (
        f"{FIXTURE_PATH} is missing. It is committed reference data produced by "
        "`demo/build_demo_fixture.py` -- see that module's docstring."
    )
    return json.loads(FIXTURE_PATH.read_text())


class TestFixtureShape:
    def test_has_exactly_the_two_required_examples(self, fixture: dict) -> None:
        ids = {e["example_id"] for e in fixture["examples"]}
        assert ids == REQUIRED_EXAMPLE_IDS

    def test_hard_contact_example_is_unfavorable_and_weak_contact_is_favorable(
        self, fixture: dict
    ) -> None:
        by_id = {e["example_id"]: e for e in fixture["examples"]}
        assert by_id["hard_contact_out"]["contact_luck_runs"] < 0
        assert by_id["weak_contact_single"]["contact_luck_runs"] > 0


class TestProbabilityDistributions:
    def test_every_example_has_a_valid_probability_distribution(self, fixture: dict) -> None:
        for example in fixture["examples"]:
            probs = example["expectation"]["probabilities"]
            assert set(probs) == set(CLASS_ORDER)
            values = list(probs.values())
            assert all(math.isfinite(v) for v in values)
            assert all(0.0 <= v <= 1.0 for v in values)
            assert math.isclose(sum(values), 1.0, abs_tol=1e-6)


class TestReconciliation:
    """observed_run_value - expected_run_value == contact_luck_runs, recomputed
    from the fixture's own stored probabilities via the LIVE, frozen scoring
    formula -- not merely re-checking the fixture's internal arithmetic.
    """

    def test_expected_run_value_matches_live_formula(self, fixture: dict) -> None:
        for example in fixture["examples"]:
            probs = example["expectation"]["probabilities"]
            recomputed = compute_expected_run_value(probs, run_value_map=DEFAULT_RUN_VALUE_MAP)
            assert recomputed == pytest.approx(
                example["expectation"]["expected_run_value"], abs=1e-9
            )

    def test_contact_luck_runs_matches_live_formula(self, fixture: dict) -> None:
        for example in fixture["examples"]:
            probs = example["expectation"]["probabilities"]
            outcome = example["reality"]["outcome_class"]
            recomputed = compute_raw_contact_luck_runs(
                probs, outcome, run_value_map=DEFAULT_RUN_VALUE_MAP
            )
            assert recomputed == pytest.approx(example["contact_luck_runs"], abs=1e-9)

    def test_observed_minus_expected_equals_contact_luck(self, fixture: dict) -> None:
        for example in fixture["examples"]:
            gap = (
                example["reality"]["observed_run_value"]
                - example["expectation"]["expected_run_value"]
            ) - example["contact_luck_runs"]
            assert abs(gap) < 1e-9

    def test_observed_run_value_matches_live_run_value_table(self, fixture: dict) -> None:
        for example in fixture["examples"]:
            outcome = example["reality"]["outcome_class"]
            assert example["reality"]["observed_run_value"] == pytest.approx(
                DEFAULT_RUN_VALUE_MAP[outcome], abs=1e-9
            )


class TestModelConfigurationMatchesLiveFrozenConstants:
    """If any of these drift from what the fixture recorded, the fixture is
    stale and `make build-demo-fixture` must be rerun (with a real, reviewed
    diff) before trusting the demo page's numbers again.
    """

    def test_variant_is_the_unweighted_probability_baseline(self, fixture: dict) -> None:
        config = fixture["model_configuration"]
        assert config["contact_model_variant"] == VARIANT_UNWEIGHTED
        assert config["class_weight"] is None

    def test_train_seasons_match_live_config(self, fixture: dict) -> None:
        assert fixture["model_configuration"]["train_seasons"] == list(TRAIN_SEASONS)

    def test_class_order_matches_live_config(self, fixture: dict) -> None:
        assert fixture["model_configuration"]["class_order"] == list(CLASS_ORDER)

    def test_random_seed_matches_live_training_code(self, fixture: dict) -> None:
        assert fixture["model_configuration"]["random_seed"] == RANDOM_SEED

    def test_run_value_table_matches_live_default_run_value_map(self, fixture: dict) -> None:
        recorded = fixture["model_configuration"]["run_value_table"]
        assert recorded.keys() == DEFAULT_RUN_VALUE_MAP.keys()
        for outcome, value in DEFAULT_RUN_VALUE_MAP.items():
            assert recorded[outcome] == pytest.approx(value, abs=1e-9)


class TestConfigFingerprintIsReproducible:
    def test_fingerprint_matches_a_fresh_hash_of_the_recorded_config(self, fixture: dict) -> None:
        canonical = json.dumps(
            fixture["model_configuration"], sort_keys=True, separators=(",", ":")
        )
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert recomputed == fixture["generator_config_fingerprint"]


class TestProvenance:
    def test_every_example_has_full_audit_provenance(self, fixture: dict) -> None:
        required_provenance_fields = {
            "event_id",
            "game_pk",
            "game_date",
            "season",
            "batter_id",
            "batter_name",
            "batter_team",
            "opponent_team",
            "description",
        }
        for example in fixture["examples"]:
            assert required_provenance_fields <= set(example["provenance"])
            assert example["provenance"]["batter_name"] in example["provenance"]["description"]
