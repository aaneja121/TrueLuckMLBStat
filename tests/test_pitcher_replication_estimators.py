"""Version 0.14: the question-E estimators, extracted from the fixture
generator into research code.

Offline and data-free: every test builds its inputs by hand or reads the
committed 2024 development fixture. No season is opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pitcher_replication_estimators as est
import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "dashboard" / "pitcher_prototype_fixture.json"
)


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    if not FIXTURE_PATH.is_file():
        pytest.skip("pitcher prototype fixture not present")
    return pd.DataFrame(json.loads(FIXTURE_PATH.read_text())["pitchers"])


class TestFrozenFormulas:
    def test_half_width_is_half_the_interval(self) -> None:
        assert est.interval_half_width([0.0], [4.0])[0] == pytest.approx(2.0)

    def test_measurement_sd_divides_by_two_z_not_multiplies(self) -> None:
        # A 95% interval of width 3.92 implies SD 1.0 exactly.
        assert est.measurement_sd([-1.96], [1.96])[0] == pytest.approx(1.0)

    def test_the_only_z_runs_from_interval_to_sd(self) -> None:
        """95% is already in the endpoints; nothing multiplies by 1.96 again."""
        low, high = np.array([-10.0]), np.array([10.0])
        assert est.measurement_sd(low, high)[0] < est.interval_half_width(low, high)[0]
        assert est.NORMAL_95_Z == 1.96

    def test_variance_decomposition_is_observed_minus_measurement(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        parts = est.decompose_variance(values, [-1.96] * 4, [1.96] * 4)
        assert parts["var_observed"] == pytest.approx(np.var(values, ddof=1))
        assert parts["var_measurement"] == pytest.approx(1.0)
        assert parts["var_signal"] == pytest.approx(parts["var_observed"] - 1.0)

    def test_negative_signal_variance_is_returned_raw_not_clipped(self) -> None:
        # Tiny observed spread, enormous stated measurement error.
        parts = est.decompose_variance([1.0, 1.01], [-19.6, -19.6], [19.6, 19.6])
        assert parts["var_signal"] < 0, "a negative estimate must survive"

    def test_resolving_power_reports_zero_but_keeps_the_negative_variance(self) -> None:
        result = est.resolving_power([1.0, 1.01], [-19.6, -19.6], [19.6, 19.6])
        assert result["signal_variance_is_negative"] is True
        assert result["resolving_power"] == 0.0
        assert result["var_signal"] < 0, "the raw negative value must still be reported"

    def test_resolving_power_is_sd_signal_over_sd_measurement(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        result = est.resolving_power(values, [-1.96] * 4, [1.96] * 4)
        expected = float(np.sqrt(np.var(values, ddof=1) - 1.0) / 1.0)
        assert result["resolving_power"] == pytest.approx(expected)

    def test_k_is_the_median_of_half_width_times_sqrt_bbe(self) -> None:
        half_width = np.array([10.0, 20.0, 30.0])
        bbe = np.array([100.0, 100.0, 100.0])
        assert est.estimate_k(half_width, bbe, min_bbe=0) == pytest.approx(200.0)

    def test_k_fit_excludes_rows_below_the_floor(self) -> None:
        half_width = np.array([1000.0, 10.0, 10.0])
        bbe = np.array([1.0, 100.0, 100.0])
        assert est.estimate_k(half_width, bbe, min_bbe=30) == pytest.approx(100.0)

    def test_required_bbe_is_k_over_target_squared(self) -> None:
        assert est.required_bbe(70.588, 2) == 1246
        assert est.required_bbe(70.588, 3) == 554
        assert est.required_bbe(70.588, 4) == 311
        assert est.required_bbe(70.588, 5) == 199

    def test_required_bbe_rejects_a_non_positive_target(self) -> None:
        with pytest.raises(est.EstimatorError):
            est.required_bbe(70.0, 0)

    def test_variance_decomposition_needs_two_rows(self) -> None:
        with pytest.raises(est.EstimatorError):
            est.decompose_variance([1.0], [0.0], [1.0])

    def test_k_fit_fails_when_no_row_clears_the_floor(self) -> None:
        with pytest.raises(est.EstimatorError):
            est.estimate_k([1.0], [5.0], min_bbe=30)


class TestZeroWidthArtifactGuard:
    def test_practical_floors_exclude_the_one_bbe_floor(self) -> None:
        assert est.PRACTICAL_WORKLOAD_FLOORS == (60, 150, 300, 450)
        assert 1 not in est.PRACTICAL_WORKLOAD_FLOORS

    def test_zero_width_intervals_are_counted(self) -> None:
        assert est.count_zero_width_intervals([1.0, 2.0], [1.0, 5.0]) == 1

    def test_a_zero_width_row_records_zero_measurement_variance(self) -> None:
        """The mechanism behind the artifact, demonstrated rather than asserted."""
        parts = est.decompose_variance([0.0, 100.0], [0.0, 100.0], [0.0, 100.0])
        assert parts["var_measurement"] == 0.0
        assert parts["var_signal"] == parts["var_observed"], "all spread booked as signal"

    def test_every_report_discloses_the_zero_width_count(self, fixture_frame: pd.DataFrame) -> None:
        for floor in (1, *est.PRACTICAL_WORKLOAD_FLOORS):
            report = est.resolving_power_report(fixture_frame, min_bbe=floor)
            assert "n_zero_width_intervals" in report

    def test_the_2024_artifact_pair_reproduces(self, fixture_frame: pd.DataFrame) -> None:
        raw = est.resolving_power_report(fixture_frame, min_bbe=1)
        guarded = est.resolving_power_report(fixture_frame, min_bbe=1, drop_zero_width=True)
        assert raw["n_zero_width_intervals"] == 73
        assert raw["resolving_power"] == 1.9386
        assert guarded["resolving_power"] == 0.8766
        assert guarded["n_rows"] == raw["n_rows"] - 73


class TestReproducesVersion0131Exactly:
    """The extraction must not have changed a single published number."""

    def test_rate_precision_reproduces(self, fixture_frame: pd.DataFrame) -> None:
        report = est.rate_precision_report(fixture_frame)
        assert report["k"] == 70.588
        assert report["n_rows"] == 658
        assert report["fitted_on_rows_with_min_bbe"] == 30
        assert report["bbe_required"] == {
            "plus_minus_5": 199,
            "plus_minus_4": 311,
            "plus_minus_3": 554,
            "plus_minus_2": 1246,
        }

    @pytest.mark.parametrize(
        ("floor", "n_rows", "power", "negative"),
        [
            (60, 537, 0.508, False),
            (150, 305, 0.4486, False),
            (300, 129, 0.0, True),
            (450, 61, 0.5007, False),
        ],
    )
    def test_resolving_power_reproduces(
        self,
        fixture_frame: pd.DataFrame,
        floor: int,
        n_rows: int,
        power: float,
        negative: bool,
    ) -> None:
        report = est.resolving_power_report(fixture_frame, min_bbe=floor)
        assert report["n_rows"] == n_rows
        assert report["resolving_power"] == power
        assert report["signal_variance_is_negative"] is negative

    def test_the_300_floor_still_reports_its_negative_signal_variance(
        self, fixture_frame: pd.DataFrame
    ) -> None:
        report = est.resolving_power_report(fixture_frame, min_bbe=300)
        assert report["var_signal_per_100"] == -0.0024


class TestGeneratorIsNowAConsumer:
    def test_the_fixture_generator_imports_the_shared_estimators(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent / "demo" / "build_pitcher_prototype_fixture.py"
        ).read_text()
        assert "from pitcher_replication_estimators import" in source

    def test_the_generator_keeps_no_private_estimator_copy(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent / "demo" / "build_pitcher_prototype_fixture.py"
        ).read_text()
        assert "def _resolving_power(" not in source
        assert "def _rate_precision_requirements(" not in source
