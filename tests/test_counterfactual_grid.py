"""Contact Luck v1.3.1: validates the COMMITTED `dashboard/demo_counterfactual_grid.json`
against the CURRENT live `mlb_luck_score` constants/formulas, and validates
the generator's pure counterfactual-row-construction logic directly.

Follows the exact same pattern `tests/test_demo_fixture.py` established for
v1.3.0: everything here runs fully offline (no real Statcast/development
data needed -- only the committed artifact plus the frozen scoring/config
modules), so it never re-invokes `demo/build_counterfactual_grid.py`'s
actual model training (which needs the full, real, ~500MB local-only
2021-2024 development parquet not present in CI). What CAN be tested
offline without a trained model -- the pure `build_grid_dataframe` row
-construction function -- is exercised directly with synthetic inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from build_counterfactual_grid import (
    GRID_CELL_PROBABILITY_DECIMALS,
    GRID_FIXED_FEATURE_NAMES,
    GRID_VARIED_FEATURES,
    build_axis_values,
    build_grid_dataframe,
    compute_integer_slider_bounds,
    find_axis_index,
)

from mlb_luck_score.config import CLASS_ORDER, TRAIN_SEASONS
from mlb_luck_score.models.train_contact_model import RANDOM_SEED, VARIANT_UNWEIGHTED
from mlb_luck_score.scoring.contact_luck import (
    compute_expected_run_value,
    compute_raw_contact_luck_runs,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "dashboard" / "demo_counterfactual_grid.json"
FIXTURE_PATH = REPO_ROOT / "dashboard" / "demo_fixture.json"
REAL_DEVELOPMENT_DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_development_data.parquet"
REQUIRED_EXAMPLE_IDS = {"hard_contact_out", "weak_contact_single"}

# Cells are rounded to GRID_CELL_PROBABILITY_DECIMALS (6) decimals at
# generation time, so summing 5 independently-rounded probabilities can be
# off from 1.0 by at most 5 * 0.5*10^-6 = 2.5e-6. 1e-5 is comfortably above
# that real worst case (not an arbitrarily loose tolerance) while still
# being ~10x tighter than the previous 1e-4.
_GRID_CELL_TOLERANCE = 1e-5
assert 5 * (10**-GRID_CELL_PROBABILITY_DECIMALS) / 2 < _GRID_CELL_TOLERANCE, (
    "GRID_CELL_PROBABILITY_DECIMALS changed -- _GRID_CELL_TOLERANCE above is no longer "
    "justified by the actual rounding precision; revisit both together."
)
_ORIGINAL_POINT_TOLERANCE = 1e-9  # original_* fields keep full precision


@pytest.fixture(scope="module")
def grid() -> dict:
    assert GRID_PATH.exists(), (
        f"{GRID_PATH} is missing. It is committed reference data produced by "
        "`demo/build_counterfactual_grid.py` -- see that module's docstring."
    )
    return json.loads(GRID_PATH.read_text())


@pytest.fixture(scope="module")
def fixture() -> dict:
    assert FIXTURE_PATH.exists()
    return json.loads(FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Pure construction-logic tests (no real data / trained model needed)
# ---------------------------------------------------------------------------


class TestBuildGridDataframe:
    """`build_grid_dataframe` is the ONLY place a counterfactual row is
    assembled -- these tests are the "for every grid point, only the
    approved varied inputs differ from the reference context" guarantee.
    """

    def _fixed_context(self) -> dict[str, object]:
        return {
            "hit_distance_sc": 171.0,
            "spray_angle_approx": 46.452,
            "bb_type": "popup",
            "stand": "L",
        }

    def _feature_cols(self) -> list[str]:
        return [
            "launch_speed",
            "launch_angle",
            "spray_angle_approx",
            "hit_distance_sc",
            "bb_type",
            "stand",
        ]

    def test_only_launch_speed_and_launch_angle_vary_across_the_grid(self) -> None:
        fixed_context = self._fixed_context()
        df = build_grid_dataframe(
            fixed_context,
            ev_values=[50, 60, 70],
            la_values=[10, 20],
            feature_cols=self._feature_cols(),
        )
        assert len(df) == 3 * 2
        for name in GRID_FIXED_FEATURE_NAMES:
            assert df[name].nunique() == 1, f"{name} must be constant across every grid row"
            assert df[name].iloc[0] == fixed_context[name]

        assert set(df["launch_speed"].unique()) == {50.0, 60.0, 70.0}
        assert set(df["launch_angle"].unique()) == {10.0, 20.0}

    def test_hit_distance_sc_exactly_equals_reference_value_on_every_row(self) -> None:
        fixed_context = self._fixed_context()
        df = build_grid_dataframe(
            fixed_context,
            ev_values=list(range(40, 45)),
            la_values=list(range(20, 25)),
            feature_cols=self._feature_cols(),
        )
        assert (df["hit_distance_sc"] == fixed_context["hit_distance_sc"]).all()

    def test_spray_angle_approx_exactly_equals_reference_value_on_every_row(self) -> None:
        fixed_context = self._fixed_context()
        df = build_grid_dataframe(
            fixed_context,
            ev_values=list(range(40, 45)),
            la_values=list(range(20, 25)),
            feature_cols=self._feature_cols(),
        )
        assert (df["spray_angle_approx"] == fixed_context["spray_angle_approx"]).all()

    def test_bb_type_and_stand_exactly_equal_reference_value_on_every_row(self) -> None:
        fixed_context = self._fixed_context()
        df = build_grid_dataframe(
            fixed_context,
            ev_values=list(range(40, 45)),
            la_values=list(range(20, 25)),
            feature_cols=self._feature_cols(),
        )
        assert (df["bb_type"] == fixed_context["bb_type"]).all()
        assert (df["stand"] == fixed_context["stand"]).all()

    def test_row_major_index_convention(self) -> None:
        fixed_context = self._fixed_context()
        ev_values = [50, 51]
        la_values = [10, 11, 12]
        df = build_grid_dataframe(fixed_context, ev_values, la_values, self._feature_cols())
        # index = ev_index * len(la_values) + la_index
        assert df.iloc[0]["launch_speed"] == 50.0 and df.iloc[0]["launch_angle"] == 10.0
        assert df.iloc[1]["launch_speed"] == 50.0 and df.iloc[1]["launch_angle"] == 11.0
        assert df.iloc[3]["launch_speed"] == 51.0 and df.iloc[3]["launch_angle"] == 10.0


class TestComputeIntegerSliderBounds:
    def test_widens_outward_to_integers(self) -> None:
        lo, hi = compute_integer_slider_bounds(65.7, 110.6, 107.6)
        assert lo == 65
        assert hi == 111

    def test_widens_to_include_a_real_value_below_the_low_percentile(self) -> None:
        # Lindor's real case: real launch_angle (26) sits BELOW the popup
        # bb_type's own 1st percentile (33) -- bounds must still include it.
        lo, hi = compute_integer_slider_bounds(33.0, 87.0, 26.0)
        assert lo == 26
        assert hi == 87

    def test_real_value_always_within_returned_bounds(self) -> None:
        for p_low, p_high, real in [(10.0, 90.0, 50.0), (10.0, 90.0, 5.0), (10.0, 90.0, 95.0)]:
            lo, hi = compute_integer_slider_bounds(p_low, p_high, real)
            assert lo <= real <= hi


# ---------------------------------------------------------------------------
# Committed-artifact tests (offline, against the live mlb_luck_score code)
# ---------------------------------------------------------------------------


class TestArtifactShape:
    def test_has_exactly_the_two_required_reference_plays(self, grid: dict) -> None:
        assert set(grid["reference_plays"]) == REQUIRED_EXAMPLE_IDS

    def test_grid_length_matches_declared_shape(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            expected = rp["grid_shape"]["n_ev"] * rp["grid_shape"]["n_la"]
            assert len(rp["grid"]) == expected, example_id

    def test_grid_dimensions_match_len_of_axis_arrays(self, grid: dict) -> None:
        """Requirement #8: grid dimensions == len(exit_velocity_values) *
        len(launch_angle_values) -- the axis arrays, not an implied
        min/max/step range, are the authoritative shape source now.
        """
        for example_id, rp in grid["reference_plays"].items():
            n_ev = len(rp["exit_velocity_values"])
            n_la = len(rp["launch_angle_values"])
            assert rp["grid_shape"]["n_ev"] == n_ev, example_id
            assert rp["grid_shape"]["n_la"] == n_la, example_id
            assert len(rp["grid"]) == n_ev * n_la, example_id

    def test_axis_arrays_are_sorted_and_deduplicated(self, grid: dict) -> None:
        """Requirement #7: array ordering/index mapping is deterministic --
        a sorted, duplicate-free array is a precondition for a stable,
        reproducible index mapping.
        """
        for example_id, rp in grid["reference_plays"].items():
            for axis_name in ("exit_velocity_values", "launch_angle_values"):
                values = rp[axis_name]
                assert values == sorted(values), f"{example_id}/{axis_name} not sorted"
                assert len(values) == len(set(values)), f"{example_id}/{axis_name} has duplicates"

    def test_original_coordinate_is_within_axis_array_bounds(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            ev_values = rp["exit_velocity_values"]
            la_values = rp["launch_angle_values"]
            assert ev_values[0] <= rp["original_exit_velocity_mph"] <= ev_values[-1], example_id
            assert la_values[0] <= rp["original_launch_angle_deg"] <= la_values[-1], example_id


def _reference_row_index(rp: dict) -> int:
    return (
        rp["original_grid_index"]["ev_index"] * rp["grid_shape"]["n_la"]
        + rp["original_grid_index"]["la_index"]
    )


class TestGridCellProbabilities:
    def test_every_grid_cell_is_finite_and_in_0_1(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            for cell in rp["grid"]:
                assert len(cell) == len(CLASS_ORDER)
                assert all(math.isfinite(v) for v in cell), example_id
                assert all(0.0 <= v <= 1.0 for v in cell), example_id

    def test_ordinary_rounded_cells_sum_to_one_within_rounding_tolerance(self, grid: dict) -> None:
        """Requirement D (ordinary side): the rounded, display-only
        exploratory cells use the looser 1e-5 tolerance justified by
        GRID_CELL_PROBABILITY_DECIMALS' actual rounding error bound.
        """
        for example_id, rp in grid["reference_plays"].items():
            ref_row = _reference_row_index(rp)
            for i, cell in enumerate(rp["grid"]):
                if i == ref_row:
                    continue
                assert math.isclose(sum(cell), 1.0, abs_tol=_GRID_CELL_TOLERANCE), (
                    f"{example_id} row {i}"
                )

    def test_reference_cell_sums_to_one_within_tight_tolerance(self, grid: dict) -> None:
        """Requirement D (reference side): the ONE full-precision cell per
        reference play is held to the same tight tolerance as
        `original_probabilities` itself, since it IS that same value.
        """
        for example_id, rp in grid["reference_plays"].items():
            cell = rp["grid"][_reference_row_index(rp)]
            assert math.isclose(sum(cell), 1.0, abs_tol=1e-6), example_id

    def test_non_reference_cells_are_rounded_to_at_most_declared_decimals(self, grid: dict) -> None:
        """Requirement C: every grid cell EXCEPT the reference cell must be
        rounded to at most GRID_CELL_PROBABILITY_DECIMALS.
        """
        for example_id, rp in grid["reference_plays"].items():
            ref_row = _reference_row_index(rp)
            for i, cell in enumerate(rp["grid"]):
                if i == ref_row:
                    continue
                for v in cell:
                    assert round(v, GRID_CELL_PROBABILITY_DECIMALS) == v, (
                        f"{example_id} row {i}: {v!r} is not rounded to "
                        f"{GRID_CELL_PROBABILITY_DECIMALS} decimals"
                    )

    def test_reference_cell_carries_full_precision_not_rounded(self, grid: dict) -> None:
        """Negative control for the full-precision requirement: if the
        overwrite in `_build_reference_play` ever silently stopped
        happening, this reference cell would ALSO end up rounded to 6
        decimals like its neighbors -- this fails in that scenario. Neither
        Greene's nor Lindor's real probabilities happen to already be exact
        6-decimal round numbers, so "at least one class carries more
        precision than 6 decimals" is a meaningful, non-vacuous check.
        """
        for example_id, rp in grid["reference_plays"].items():
            cell = rp["grid"][_reference_row_index(rp)]
            assert any(round(v, GRID_CELL_PROBABILITY_DECIMALS) != v for v in cell), example_id

    def test_original_probabilities_are_finite_in_0_1_and_sum_to_one(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            probs = rp["original_probabilities"]
            assert set(probs) == set(CLASS_ORDER)
            values = list(probs.values())
            assert all(math.isfinite(v) for v in values), example_id
            assert all(0.0 <= v <= 1.0 for v in values), example_id
            assert math.isclose(sum(values), 1.0, abs_tol=1e-6), example_id


class TestReconciliation:
    def test_original_expected_run_value_matches_live_formula(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            recomputed = compute_expected_run_value(
                rp["original_probabilities"], run_value_map=DEFAULT_RUN_VALUE_MAP
            )
            assert recomputed == pytest.approx(
                rp["original_expected_run_value"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id

    def test_hypothetical_contact_luck_reconciles_for_every_outcome(self, grid: dict) -> None:
        """observed RV - expected RV == Contact Luck, for EVERY hypothetical
        realized outcome a visitor could pick -- not just the real one.
        """
        for example_id, rp in grid["reference_plays"].items():
            expected_rv = rp["original_expected_run_value"]
            for outcome in CLASS_ORDER:
                observed_rv = DEFAULT_RUN_VALUE_MAP[outcome]
                recomputed_luck = compute_raw_contact_luck_runs(
                    rp["original_probabilities"], outcome, run_value_map=DEFAULT_RUN_VALUE_MAP
                )
                assert recomputed_luck == pytest.approx(
                    observed_rv - expected_rv, abs=_ORIGINAL_POINT_TOLERANCE
                ), f"{example_id}/{outcome}"

    def test_original_coordinate_reproduces_the_reviewed_v1_3_0_fixture(
        self, grid: dict, fixture: dict
    ) -> None:
        fixture_by_id = {e["example_id"]: e for e in fixture["examples"]}
        for example_id, rp in grid["reference_plays"].items():
            fixture_example = fixture_by_id[example_id]
            for cls in CLASS_ORDER:
                assert rp["original_probabilities"][cls] == pytest.approx(
                    fixture_example["expectation"]["probabilities"][cls],
                    abs=_ORIGINAL_POINT_TOLERANCE,
                ), f"{example_id}/{cls}"
            assert rp["original_expected_run_value"] == pytest.approx(
                fixture_example["expectation"]["expected_run_value"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id

    def test_fixed_context_matches_the_reviewed_v1_3_0_fixture_contact_block(
        self, grid: dict, fixture: dict
    ) -> None:
        fixture_by_id = {e["example_id"]: e for e in fixture["examples"]}
        for example_id, rp in grid["reference_plays"].items():
            contact = fixture_by_id[example_id]["contact"]
            assert rp["fixed_context"]["spray_angle_approx"] == pytest.approx(
                contact["spray_angle_deg"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id
            assert rp["fixed_context"]["hit_distance_sc"] == pytest.approx(
                contact["hit_distance_ft"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id
            assert rp["fixed_context"]["bb_type"] == contact["bb_type"], example_id
            assert rp["original_exit_velocity_mph"] == pytest.approx(
                contact["exit_velocity_mph"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id
            assert rp["original_launch_angle_deg"] == pytest.approx(
                contact["launch_angle_deg"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id


class TestExplicitAxisArraysAndDirectLookup:
    """Requirements #1-6 from the Phase 2 amendment: the real reference
    coordinates -- non-integer for exit velocity -- are genuine points IN
    the explicit axis arrays, reachable by direct index lookup with no
    interpolation.
    """

    def test_greene_real_ev_107_6_exists_in_its_allowed_ev_array(self, grid: dict) -> None:
        rp = grid["reference_plays"]["hard_contact_out"]
        assert rp["original_exit_velocity_mph"] == 107.6
        assert any(math.isclose(v, 107.6, abs_tol=1e-9) for v in rp["exit_velocity_values"]), (
            "107.6 not found in Greene's exit_velocity_values"
        )

    def test_lindor_real_ev_60_4_exists_in_its_allowed_ev_array(self, grid: dict) -> None:
        rp = grid["reference_plays"]["weak_contact_single"]
        assert rp["original_exit_velocity_mph"] == 60.4
        assert any(math.isclose(v, 60.4, abs_tol=1e-9) for v in rp["exit_velocity_values"]), (
            "60.4 not found in Lindor's exit_velocity_values"
        )

    def test_each_reference_launch_angle_exists_in_its_allowed_la_array(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            real_la = rp["original_launch_angle_deg"]
            assert any(math.isclose(v, real_la, abs_tol=1e-9) for v in rp["launch_angle_values"]), (
                f"{example_id}: {real_la} not found in launch_angle_values"
            )

    def test_reset_coordinate_lookup_lands_on_the_exact_real_ev_la_pair(self, grid: dict) -> None:
        for example_id, rp in grid["reference_plays"].items():
            ev_values = rp["exit_velocity_values"]
            la_values = rp["launch_angle_values"]
            ev_index = rp["original_grid_index"]["ev_index"]
            la_index = rp["original_grid_index"]["la_index"]
            assert ev_values[ev_index] == pytest.approx(
                rp["original_exit_velocity_mph"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id
            assert la_values[la_index] == pytest.approx(
                rp["original_launch_angle_deg"], abs=_ORIGINAL_POINT_TOLERANCE
            ), example_id

    def test_find_axis_index_recomputes_the_same_original_grid_index(self, grid: dict) -> None:
        """`original_grid_index` isn't just self-consistent with the arrays
        it was stored alongside -- recompute it independently via
        `find_axis_index` and confirm agreement.
        """
        for example_id, rp in grid["reference_plays"].items():
            recomputed_ev_index = find_axis_index(
                rp["exit_velocity_values"], rp["original_exit_velocity_mph"]
            )
            recomputed_la_index = find_axis_index(
                rp["launch_angle_values"], rp["original_launch_angle_deg"]
            )
            assert recomputed_ev_index == rp["original_grid_index"]["ev_index"], example_id
            assert recomputed_la_index == rp["original_grid_index"]["la_index"], example_id

    def test_direct_grid_lookup_at_the_real_coordinate_reproduces_the_fixture(
        self, grid: dict, fixture: dict
    ) -> None:
        """Requirements A and B: reading `grid[row_index]` at each reference
        play's OWN real coordinate array index -- WITHOUT any
        interpolation, averaging, or nearest-neighbor approximation --
        reproduces `dashboard/demo_fixture.json`'s reviewed probabilities
        EXACTLY (within `_ORIGINAL_POINT_TOLERANCE`, not the looser
        rounded-cell tolerance), because that specific cell was overwritten
        with the full-precision value -- see
        `test_reference_cell_carries_full_precision_not_rounded` in
        `TestGridCellProbabilities` for the complementary check that this
        ISN'T true of any other cell.
        """
        fixture_by_id = {e["example_id"]: e for e in fixture["examples"]}
        for example_id, rp in grid["reference_plays"].items():
            grid_cell = rp["grid"][_reference_row_index(rp)]

            fixture_probs = fixture_by_id[example_id]["expectation"]["probabilities"]
            for i, cls in enumerate(CLASS_ORDER):
                assert grid_cell[i] == pytest.approx(
                    fixture_probs[cls], abs=_ORIGINAL_POINT_TOLERANCE
                ), f"{example_id}/{cls}: direct grid lookup does not exactly reproduce the fixture"

    def test_build_axis_values_inserts_the_real_value_only_when_not_already_integer(
        self,
    ) -> None:
        """Pure-function coverage (no real data needed): confirms
        `build_axis_values` behaves exactly as documented for both the
        "needs insertion" (Greene-like) and "already integer, no insertion"
        (both reference plays' launch angles) cases.
        """
        with_insertion = build_axis_values(65.7, 110.6, 107.6)
        assert 107.6 in with_insertion
        assert with_insertion == sorted(with_insertion)
        assert len(with_insertion) == len(set(with_insertion))
        # 65..111 inclusive is 47 integers; +1 for the inserted real value.
        assert len(with_insertion) == 47 + 1

        without_insertion = build_axis_values(22.0, 60.0, 33.0)
        assert without_insertion == list(range(22, 61))
        assert 33.0 in without_insertion
        assert all(isinstance(v, int) for v in without_insertion)

    def test_array_ordering_and_index_mapping_is_deterministic_across_rebuilds(self) -> None:
        """Requirement #7, at the pure-function level: calling
        `build_axis_values` twice with the same inputs produces byte
        -identical arrays (same order, same values) -- required for
        `find_axis_index` to mean the same thing on every rebuild.
        """
        first = build_axis_values(35.6, 98.1, 60.4)
        second = build_axis_values(35.6, 98.1, 60.4)
        assert first == second
        assert find_axis_index(first, 60.4) == find_axis_index(second, 60.4)


class TestModelConfigurationMatchesLiveFrozenConstants:
    def test_variant_is_the_unweighted_probability_baseline(self, grid: dict) -> None:
        config = grid["model_configuration"]
        assert config["contact_model_variant"] == VARIANT_UNWEIGHTED
        assert config["class_weight"] is None

    def test_train_seasons_match_live_config_and_exclude_2025(self, grid: dict) -> None:
        train_seasons = grid["model_configuration"]["train_seasons"]
        assert train_seasons == list(TRAIN_SEASONS)
        assert 2025 not in train_seasons

    def test_class_order_matches_live_config(self, grid: dict) -> None:
        assert grid["model_configuration"]["class_order"] == list(CLASS_ORDER)

    def test_random_seed_matches_live_training_code(self, grid: dict) -> None:
        assert grid["model_configuration"]["random_seed"] == RANDOM_SEED

    def test_run_value_table_matches_live_default_run_value_map(self, grid: dict) -> None:
        recorded = grid["model_configuration"]["run_value_table"]
        assert recorded.keys() == DEFAULT_RUN_VALUE_MAP.keys()
        for outcome, value in DEFAULT_RUN_VALUE_MAP.items():
            assert recorded[outcome] == pytest.approx(value, abs=1e-9)


class TestCounterfactualSemantics:
    def test_varied_features_are_exactly_launch_speed_and_launch_angle(self, grid: dict) -> None:
        assert grid["counterfactual_semantics"]["varied_features"] == list(GRID_VARIED_FEATURES)

    def test_fixed_features_include_the_expected_names(self, grid: dict) -> None:
        fixed = set(grid["counterfactual_semantics"]["fixed_features"])
        assert set(GRID_FIXED_FEATURE_NAMES) <= fixed

    def test_semantics_type_declares_ceteris_paribus_not_physics(self, grid: dict) -> None:
        assert grid["counterfactual_semantics"]["type"] == "ceteris_paribus_model_sensitivity"
        interpretation = grid["counterfactual_semantics"]["interpretation"].lower()
        assert "not a physical ball-flight reconstruction" in interpretation


class TestConfigFingerprintIsReproducible:
    def test_fingerprint_matches_a_fresh_hash_of_model_config_and_semantics(
        self, grid: dict
    ) -> None:
        payload = {
            "model_configuration": grid["model_configuration"],
            "counterfactual_semantics": grid["counterfactual_semantics"],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert recomputed == grid["generator_config_fingerprint"]


class TestByteDeterministicGeneration:
    """Requirement E: this is committed, reviewed reference data -- the SAME
    source data/configuration must produce byte-identical JSON on every
    regeneration. No generated-at/current-time field exists to ignore
    (removed entirely, not replaced with another runtime-dependent value --
    Git history already records when the reviewed artifact last changed).
    """

    def test_committed_artifact_has_no_timestamp_or_generated_at_field(self, grid: dict) -> None:
        assert "generated_at_utc" not in grid
        assert not any("generated_at" in k or "timestamp" in k.lower() for k in grid)

    @pytest.mark.skipif(
        not REAL_DEVELOPMENT_DATA_PATH.exists(),
        reason=(
            "requires the real, local-only 2021-2024 development dataset (see "
            "README.md 'Full development dataset') -- never present in CI, per "
            "CLAUDE.md's fully-offline test suite requirement"
        ),
    )
    def test_regenerating_from_identical_inputs_is_byte_identical(self) -> None:
        """The actual end-to-end check: build the artifact dict TWICE from
        the same real inputs and confirm identical canonical JSON
        serialization -- not merely identical Python dict equality (which
        wouldn't catch e.g. key-order-dependent downstream consumers).
        """
        from build_counterfactual_grid import build_counterfactual_grid

        first = build_counterfactual_grid(REAL_DEVELOPMENT_DATA_PATH, FIXTURE_PATH)
        second = build_counterfactual_grid(REAL_DEVELOPMENT_DATA_PATH, FIXTURE_PATH)

        first_canonical = json.dumps(first, sort_keys=True, separators=(",", ":"))
        second_canonical = json.dumps(second, sort_keys=True, separators=(",", ":"))
        assert first_canonical == second_canonical
        assert "generated_at_utc" not in first
