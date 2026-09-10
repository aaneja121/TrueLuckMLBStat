"""Drive the resolution pass's decision logic on SYNTHETIC outcomes.

Every branch below will execute for the first time on real 2026 outcomes on
2026-09-27 unless it is exercised beforehand. Two of them are the ones most
likely to be reached and least likely to have been tried:

  * the UNDERPOWERED rule -- "If the incremental cohort is small, the result
    is reported as imprecise and classified mechanically anyway. A wide
    interval is a wide interval; it is never grounds for falling back to the
    full-season cohort, pooling the looks, or declining to report."
  * the COHORT DISAGREEMENT rule -- "The PRIMARY incremental cohort stands.
    The full-season cohort may never be used to upgrade a conclusion the
    incremental cohort did not support."

Both describe what happens when the result is inconvenient, which is
exactly when code that has never run gets improvised. Testing them now, with
no outcome opened, is also the only honest time to do it: writing this after
seeing the real numbers would be choosing the handling of a case whose
answer you already know.

Everything here is synthetic. No 2026 outcome is read, no snapshot is
pinned, no artifact is written, and `run()` itself is never called -- it is
gated on the season having ended and would correctly refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.bootstrap import BootstrapDesign
from forecast.phase2 import run_resolution_evaluation as R

CUTOFF = 100
HORIZON = 200
FAST = BootstrapDesign(reps=200, seed=42, alpha=0.05)


def _cohort_frame(
    n: int,
    *,
    seed: int,
    error_ratio: float,
    noise: float = 1.0,
) -> pd.DataFrame:
    """A synthetic completed-window cohort.

    `error_ratio` SCALES the forecast's error against the benchmark's, which
    is what MAE actually compares. Below 1 the forecast is genuinely closer;
    above 1 it is worse; 1.0 is a tie. Shifting the error instead would only
    move it off-centre and make |error| LARGER in both directions, which is
    the opposite of an advantage -- the mistake this generator made first.

    The delta convention is challenger - reference, so a ratio below 1
    produces a negative delta.
    """
    rng = np.random.default_rng(seed)
    target = rng.normal(0.0, 3.0, size=n)
    benchmark_error = rng.normal(0.0, noise, size=n)
    # INDEPENDENT, not a rescaling of the benchmark's error. Deriving one from
    # the other makes the two predictors almost perfectly correlated, the
    # paired delta nearly variance-free, and every cohort significant however
    # small -- which would make the inconclusive and underpowered cases
    # untestable.
    forecast_error = rng.normal(0.0, noise * error_ratio, size=n)
    return pd.DataFrame(
        {
            "batter": np.arange(1000, 1000 + n),
            "batter_name": [f"Player {i}" for i in range(n)],
            "season": 2026,
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            R.TARGET: target,
            R.FORECAST_COLUMN: target + forecast_error,
            "shrunk_deserved_persistence": target + benchmark_error,
            "shrunk_realized_persistence": target + benchmark_error * 1.1,
            "league_mean": np.full(n, float(target.mean())),
        }
    )


def _classify(frame: pd.DataFrame) -> dict:
    record = R.evaluate_cohort(
        frame, cohort="incremental", cutoff=CUTOFF, horizon=HORIZON, design=FAST, classify=True
    )
    return record["prespecified_classification"]


class TestTheFourWayClassificationIsMechanical:
    """MAE decides and nothing else may. The four cases were fixed before any
    2026 number existed, and each must actually be reachable."""

    def test_a_clear_forecast_advantage_is_established_success(self) -> None:
        # The forecast's error is smaller by a wide, consistent margin.
        result = _classify(_cohort_frame(300, seed=1, error_ratio=0.35, noise=1.0))
        assert result["classification"] == "established_incremental_success"
        assert result["delta_mae"] < 0
        assert result["ci_delta_mae"][1] < 0

    def test_a_clear_forecast_disadvantage_is_evidence_against(self) -> None:
        result = _classify(_cohort_frame(300, seed=2, error_ratio=2.2, noise=1.0))
        assert result["classification"] == "evidence_against_incremental_value"
        assert result["delta_mae"] > 0
        assert result["ci_delta_mae"][0] > 0

    def test_a_small_noisy_advantage_is_promising_but_inconclusive(self) -> None:
        """The first look's own outcome, and the one this pass exists to
        revisit -- it must not silently become a success."""
        result = _classify(_cohort_frame(35, seed=3, error_ratio=0.97, noise=1.0))
        assert result["classification"] == "promising_but_inconclusive"
        assert result["delta_mae"] < 0
        assert result["ci_delta_mae"][1] > 0, "interval must still cross zero"

    def test_no_advantage_is_no_evidence(self) -> None:
        """The fall-through branch: the delta does not favour the forecast and
        the interval crosses zero, so nothing is established either way."""
        result = _classify(_cohort_frame(60, seed=41, error_ratio=1.05, noise=1.0))
        assert result["classification"] == "no_evidence_of_incremental_improvement"
        assert result["delta_mae"] >= 0
        assert result["ci_delta_mae"][0] < 0 < result["ci_delta_mae"][1]

    def test_mae_decides_and_the_interval_only_sharpens_it(self) -> None:
        """Same sign of delta, different precision -> different class. That is
        the whole rule, and it must be driven by MAE and the interval alone."""
        wide = _classify(_cohort_frame(25, seed=55, error_ratio=0.92, noise=1.0))
        tight = _classify(_cohort_frame(1200, seed=55, error_ratio=0.92, noise=1.0))
        assert wide["delta_mae"] < 0 and tight["delta_mae"] < 0
        assert wide["classification"] == "promising_but_inconclusive"
        assert tight["classification"] == "established_incremental_success"

    def test_secondary_metrics_are_flagged_as_unable_to_reclassify(self) -> None:
        result = _classify(_cohort_frame(150, seed=6, error_ratio=0.5, noise=1.0))
        assert result["deciding_metric"] == "MAE"
        assert result["secondary_metrics_may_not_reclassify"] is True

    def test_all_four_classes_are_the_frozen_set(self) -> None:
        import json

        spec = json.loads((R.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json").read_text())
        assert set(spec["success_classification"]) == {
            "established_incremental_success",
            "evidence_against_incremental_value",
            "no_evidence_of_incremental_improvement",
            "promising_but_inconclusive",
        }


class TestTheUnderpoweredRule:
    """'A wide interval is a wide interval.' A tiny cohort is still
    classified, mechanically, and never triggers a fallback."""

    @pytest.mark.parametrize("n", [5, 8, 15])
    def test_a_tiny_cohort_is_still_classified(self, n: int) -> None:
        record = R.evaluate_cohort(
            _cohort_frame(n, seed=10 + n, error_ratio=0.85, noise=1.0),
            cohort="incremental",
            cutoff=CUTOFF,
            horizon=HORIZON,
            design=FAST,
            classify=True,
        )
        assert record["evaluated"] is not False
        assert "prespecified_classification" in record, "a small cohort must still be classified"
        assert record["prespecified_classification"]["classification"] in {
            "established_incremental_success",
            "evidence_against_incremental_value",
            "no_evidence_of_incremental_improvement",
            "promising_but_inconclusive",
        }

    def test_a_tiny_cohort_produces_a_wide_interval_and_says_so(self) -> None:
        small = _classify(_cohort_frame(6, seed=20, error_ratio=0.85, noise=1.0))
        large = _classify(_cohort_frame(400, seed=20, error_ratio=0.85, noise=1.0))
        small_width = small["ci_delta_mae"][1] - small["ci_delta_mae"][0]
        large_width = large["ci_delta_mae"][1] - large["ci_delta_mae"][0]
        assert small_width > large_width

    def test_an_empty_cohort_is_reported_not_classified(self) -> None:
        """No windows is not a class -- it is an absence, and inventing a
        label for it would be a result nobody measured."""
        record = R.evaluate_cohort(
            _cohort_frame(0, seed=21, error_ratio=1.0),
            cohort="incremental",
            cutoff=CUTOFF,
            horizon=HORIZON,
            design=FAST,
            classify=True,
        )
        assert record["evaluated"] is False
        assert record["n_windows"] == 0
        assert "prespecified_classification" not in record

    def test_the_spec_forbids_the_three_tempting_fallbacks(self) -> None:
        import json

        spec = json.loads((R.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json").read_text())
        rule = spec["underpowered_rule"]
        assert "falling back to the full-season cohort" in rule
        assert "pooling the looks" in rule
        assert "declining to report" in rule


class TestTheCohortDisagreementRule:
    """The full-season cohort adds precision, never independent confirmation,
    and may never upgrade a conclusion the incremental cohort did not
    support."""

    def test_the_full_season_cohort_is_never_classified(self) -> None:
        """The structural guarantee: it receives an estimate and an interval
        but no label a reader could quote as a result."""
        record = R.evaluate_cohort(
            _cohort_frame(400, seed=30, error_ratio=0.35, noise=1.0),
            cohort="full_season",
            cutoff=CUTOFF,
            horizon=HORIZON,
            design=FAST,
            classify=False,
        )
        # Present and explicitly None, with the reason recorded -- better than
        # an absent key, which would leave "unclassified" to be inferred.
        assert record["prespecified_classification"] is None
        assert record["may_decide_the_conclusion"] is False
        assert record["mandatory_label"]
        assert record["evaluated"] is not False

    def test_a_strong_full_season_cannot_upgrade_a_weak_incremental(self) -> None:
        """The disagreement case, constructed: incremental inconclusive,
        full-season a clear success. Only the incremental cohort carries a
        classification, so there is nothing for the stronger cohort to
        upgrade."""
        incremental = R.evaluate_cohort(
            _cohort_frame(35, seed=31, error_ratio=0.97, noise=1.0),
            cohort="incremental",
            cutoff=CUTOFF,
            horizon=HORIZON,
            design=FAST,
            classify=True,
        )
        full_season = R.evaluate_cohort(
            _cohort_frame(400, seed=32, error_ratio=0.35, noise=1.0),
            cohort="full_season",
            cutoff=CUTOFF,
            horizon=HORIZON,
            design=FAST,
            classify=False,
        )
        assert (
            incremental["prespecified_classification"]["classification"]
            == "promising_but_inconclusive"
        )
        assert full_season["prespecified_classification"] is None
        assert full_season["may_decide_the_conclusion"] is False
        assert full_season["contact_forecast_vs_shrunk_deserved"]["delta_mae"] < 0

    def test_the_spec_states_which_cohort_stands(self) -> None:
        import json

        spec = json.loads((R.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json").read_text())
        rule = spec["cohort_disagreement_rule"]
        assert "PRIMARY incremental cohort stands" in rule["if_they_disagree"]
        assert "never be used to upgrade" in rule["if_they_disagree"]
        assert "adds precision, not independent confirmation" in rule["if_they_agree"]
        assert "selecting whichever cohort reads better" in rule["forbidden"]


class TestTheSpecConsistencyGuard:
    """Amendment 2 narrowed classification to the primary cohort. The runner
    refuses a specification whose prose and flags disagree rather than
    picking a reading at run time."""

    def _spec(self, *, incremental: bool, full_season: bool, primary: str = "incremental") -> dict:
        return {
            "primary_cohort": primary,
            "classification_applies_to": "the incremental cohort only",
            "cohorts": {
                "incremental": {
                    "four_way_classification_applied": incremental,
                    "may_decide_the_conclusion": True,
                },
                "full_season": {
                    "four_way_classification_applied": full_season,
                    "may_decide_the_conclusion": False,
                },
                # never_completed is reported and never evaluated -- it is in
                # the real specification, so the guard must tolerate it.
                "never_completed": {
                    "four_way_classification_applied": False,
                    "may_decide_the_conclusion": False,
                },
            },
        }

    def test_the_real_frozen_spec_is_consistent(self) -> None:
        import json

        spec = json.loads((R.RESOLUTION_OUTPUTS_DIR / "resolution_specification.json").read_text())
        flags = R.assert_cohort_classification_is_consistent(spec)
        assert flags == {"incremental": True, "full_season": False, "never_completed": False}

    def test_classifying_the_full_season_cohort_is_refused(self) -> None:
        with pytest.raises(R.ResolutionGateError, match="must not carry a classification"):
            R.assert_cohort_classification_is_consistent(
                self._spec(incremental=True, full_season=True)
            )

    def test_classifying_no_cohort_is_refused(self) -> None:
        with pytest.raises(R.ResolutionGateError):
            R.assert_cohort_classification_is_consistent(
                self._spec(incremental=False, full_season=False)
            )

    def test_a_cohort_that_may_not_decide_may_not_be_classified(self) -> None:
        with pytest.raises(R.ResolutionGateError):
            R.assert_cohort_classification_is_consistent(
                self._spec(incremental=False, full_season=True, primary="full_season")
            )


class TestTheGateStillRefusesToday:
    """The dry run must not have made the pass runnable early."""

    def test_run_refuses_without_authorization(self) -> None:
        with pytest.raises(R.ResolutionGateError, match="authorize this second look"):
            R.run(authorized_by="")

    def test_the_season_gate_still_blocks(self) -> None:
        from datetime import date

        with pytest.raises(R.ResolutionGateError, match="does not run early"):
            R.assert_season_has_ended(today=date(2026, 9, 26))

    def test_it_opens_on_the_day_after_the_season_ends(self) -> None:
        from datetime import date

        assert R.assert_season_has_ended(today=date(2026, 9, 28))
