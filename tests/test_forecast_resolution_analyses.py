"""The four analyses the resolution specification requires.

`required_analyses` names four, and the pass must produce all four:

    1. survivorship audit across incremental, first-look and never-completed
    2. stability and outlier sensitivity, on each classified cohort
    3. distribution-shift diagnostics
    4. forecast bias and calibration diagnostics

Three were already wired -- `evaluate_cohort` computes stability and bias
per cohort, and `build_survivorship` runs over all three populations. The
third was NOT: the resolution runner neither imported nor called
`build_distribution_shift`, and `REQUIRED_ARTIFACTS` did not name an
artifact for it, so the pass would have completed while silently omitting a
mandated analysis.

Finding that on 2026-09-27 would be the expensive version. Writing the
analysis afterward, with outcomes visible, is also exactly the unintended
flexibility a prespecified second look exists to avoid -- so the code and
its tests are built now, against SYNTHETIC cohorts, with no outcome opened.

Everything here is synthetic and offline. No 2026 outcome is read, no real
cohort is constructed, and nothing is written to any namespace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.phase2 import run_resolution_evaluation as R


@pytest.fixture(scope="module")
def features() -> list[str]:
    return R.frozen_model_features()


def _synthetic_frame(features: list[str], n: int, *, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(shift, 1.0, size=n) for f in features}
    data["batter_name"] = [f"Player {i}" for i in range(n)]
    return pd.DataFrame(data)


class TestTheFrozenFeatureListIsRead:
    """The feature list is on `frozen_and_untouchable`. Re-deriving it would
    agree today and diverge silently the day the code changes."""

    def test_it_comes_from_the_sealed_ridge_manifest(self, features: list[str]) -> None:
        assert len(features) == 26
        assert "std_realized_rv_per_100" in features
        assert "n_resolved_bbe_through_cutoff" in features

    def test_it_is_the_selected_model_s_set_not_another(self, features: list[str]) -> None:
        """Model D was selected on 2023 validation MAE, using no 2024
        information. The diagnostic must describe that model's inputs."""
        import json

        manifest = json.loads((R.RESOLUTION_OUTPUTS_DIR / "ridge_freeze_manifest.json").read_text())
        kr = manifest["key_results"]
        assert kr["selected_model"]["model"] == "D_full_contact_profile"
        assert features == list(kr["feature_sets"]["D_full_contact_profile"])

    def test_a_manifest_without_the_feature_set_is_refused(self, tmp_path) -> None:
        """No fallback to deriving one."""
        import json

        (tmp_path / "ridge_freeze_manifest.json").write_text(json.dumps({"key_results": {}}))
        with pytest.raises(R.ResolutionIntegrityError, match="does not name the selected model"):
            R.frozen_model_features(resolution_dir=tmp_path)


class TestDistributionShiftRunsPerCohort:
    """Per cohort, not pooled: the incremental cohort is the slower
    accumulators by construction and is explicitly not exchangeable with the
    first-look cohort, so one pooled diagnostic would describe a population
    no classification is made about."""

    def test_both_cohorts_are_evaluated(self, features: list[str]) -> None:
        cohorts = {
            "incremental": _synthetic_frame(features, 40, seed=1),
            "full_season": _synthetic_frame(features, 120, seed=2),
        }
        development = _synthetic_frame(features, 500, seed=3)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        assert shift["cohorts"]["incremental"]["evaluated"] is True
        assert shift["cohorts"]["incremental"]["n_windows"] == 40
        assert shift["cohorts"]["full_season"]["n_windows"] == 120
        assert shift["n_features"] == len(features)

    def test_every_frozen_feature_is_reported(self, features: list[str]) -> None:
        cohorts = {
            "incremental": _synthetic_frame(features, 30, seed=4),
            "full_season": pd.DataFrame(),
        }
        development = _synthetic_frame(features, 300, seed=5)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        reported = shift["cohorts"]["incremental"]["features"]["by_feature"]
        assert {row["feature"] for row in reported} == set(features)

    def test_a_real_shift_is_detected(self, features: list[str]) -> None:
        """The diagnostic has to be able to say 'these differ' or it is
        decoration."""
        cohorts = {
            "incremental": _synthetic_frame(features, 60, seed=6, shift=3.0),
            "full_season": pd.DataFrame(),
        }
        development = _synthetic_frame(features, 400, seed=7, shift=0.0)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        rows = shift["cohorts"]["incremental"]["features"]["by_feature"]
        shifts = [
            r["standardized_mean_shift"]
            for r in rows
            if r.get("standardized_mean_shift") is not None
        ]
        assert shifts, "no standardized shift reported"
        assert max(shifts) > 2.0

    def test_no_shift_reads_as_no_shift(self, features: list[str]) -> None:
        cohorts = {
            "incremental": _synthetic_frame(features, 200, seed=8),
            "full_season": pd.DataFrame(),
        }
        development = _synthetic_frame(features, 2000, seed=9)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        rows = shift["cohorts"]["incremental"]["features"]["by_feature"]
        shifts = [
            abs(r["standardized_mean_shift"])
            for r in rows
            if r.get("standardized_mean_shift") is not None
        ]
        assert max(shifts) < 0.75

    def test_an_empty_cohort_is_reported_not_crashed_on(self, features: list[str]) -> None:
        """The underpowered rule says a small cohort is classified mechanically
        anyway; a diagnostic that raised on one would block that."""
        cohorts = {
            "incremental": pd.DataFrame(),
            "full_season": _synthetic_frame(features, 10, seed=10),
        }
        development = _synthetic_frame(features, 100, seed=11)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        assert shift["cohorts"]["incremental"] == {"evaluated": False, "n_windows": 0}

    def test_a_cohort_missing_a_frozen_feature_is_refused(self, features: list[str]) -> None:
        """Never a silent comparison over whichever subset happens to exist."""
        frame = _synthetic_frame(features, 20, seed=12).drop(columns=[features[0]])
        cohorts = {"incremental": frame, "full_season": pd.DataFrame()}
        development = _synthetic_frame(features, 100, seed=13)
        with pytest.raises(R.ResolutionIntegrityError, match="missing frozen model features"):
            R.build_cohort_distribution_shift(cohorts, development, features=features)


class TestItStaysDiagnostic:
    def test_the_output_says_it_changes_nothing(self, features: list[str]) -> None:
        cohorts = {
            "incremental": _synthetic_frame(features, 25, seed=14),
            "full_season": pd.DataFrame(),
        }
        development = _synthetic_frame(features, 250, seed=15)
        shift = R.build_cohort_distribution_shift(cohorts, development, features=features)
        assert "DIAGNOSTIC ONLY" in shift["status"]
        assert "never grounds to adjust" in shift["generalization_note"]

    def test_it_matches_the_specification_s_own_prohibitions(self) -> None:
        """`not_authorized` forbids refitting, reweighting and exclusion. A
        shift diagnostic that recommended any of them would contradict the
        document it exists to satisfy."""
        import json

        spec = json.loads((R.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json").read_text())
        assert "refitting any model on 2026" in spec["not_authorized"]
        assert "removing, trimming or winsorizing individual hitters" in spec["not_authorized"]
        assert spec["survivorship"]["rule"].startswith("Descriptive only")


class TestAllFourRequiredAnalysesAreReachable:
    """The gap this file was written to close: the pass must produce all four,
    and `REQUIRED_ARTIFACTS` is what the runner checks itself against."""

    def test_the_distribution_shift_artifact_is_declared(self) -> None:
        assert "resolution_2026_distribution_shift.json" in R.REQUIRED_ARTIFACTS

    def test_the_survivorship_artifact_is_declared(self) -> None:
        assert "resolution_2026_survivorship.json" in R.REQUIRED_ARTIFACTS

    def test_stability_and_bias_are_computed_per_cohort(self) -> None:
        """They live inside `evaluate_cohort`, so they exist per cohort rather
        than once for the pass."""
        source = (
            __import__("pathlib").Path(R.__file__).read_text().split("def evaluate_cohort", 1)[1]
        )
        body = source.split("\ndef ", 1)[0]
        assert "build_stability(scored)" in body
        assert "_bias_diagnostics(scored)" in body

    def test_the_runner_calls_the_shift_analysis(self) -> None:
        source = __import__("pathlib").Path(R.__file__).read_text()
        assert "build_cohort_distribution_shift(" in source.split("def run(", 1)[1]


class TestTheReportShowsTheAnalysis:
    """A required analysis that is computed and never rendered is one nobody
    reads. The report is the artifact a human actually opens."""

    def _shift(self, features: list[str]) -> dict:
        cohorts = {
            "incremental": _synthetic_frame(features, 40, seed=21, shift=1.5),
            "full_season": _synthetic_frame(features, 90, seed=22),
        }
        development = _synthetic_frame(features, 400, seed=23)
        return R.build_cohort_distribution_shift(cohorts, development, features=features)

    def test_it_renders_a_section_per_cohort(self, features: list[str]) -> None:
        from forecast.phase2.run_resolution_report import _render_distribution_shift

        rendered = "\n".join(_render_distribution_shift(self._shift(features)))
        assert "## Distribution-shift diagnostics" in rendered
        assert "incremental" in rendered
        assert "full-season" in rendered

    def test_it_states_that_nothing_is_adjusted(self, features: list[str]) -> None:
        from forecast.phase2.run_resolution_report import _render_distribution_shift

        rendered = "\n".join(_render_distribution_shift(self._shift(features)))
        assert "EXPLANATORY ONLY" in rendered or "DIAGNOSTIC ONLY" in rendered
        assert "never grounds to adjust" in rendered

    def test_an_unevaluated_cohort_says_so_rather_than_vanishing(self, features: list[str]) -> None:
        from forecast.phase2.run_resolution_report import _render_distribution_shift

        shift = R.build_cohort_distribution_shift(
            {"incremental": pd.DataFrame(), "full_season": _synthetic_frame(features, 5, seed=24)},
            _synthetic_frame(features, 60, seed=25),
            features=features,
        )
        rendered = "\n".join(_render_distribution_shift(shift))
        assert "no windows to compare" in rendered

    def test_the_runner_passes_it_to_the_renderer(self) -> None:
        source = __import__("pathlib").Path(R.__file__).read_text()
        assert "distribution_shift=shift," in source
