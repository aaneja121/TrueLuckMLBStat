"""Contact Forecast: the end-of-season resolution pass runner.

Written before the season ends and before any outcome exists, so these tests
pin the guards rather than a result. The load-bearing one is first: the pass
refuses to run early, and no flag overrides it.

Synthetic frames throughout. Nothing here reads a real 2026 outcome, and
nothing calls `run()` -- calling it today is exactly what the first test
proves impossible.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from prospective.prospective_config import PROSPECTIVE_2026_SEASON_END_DATE

from forecast.bootstrap import BootstrapDesign
from forecast.phase2 import run_resolution_evaluation as rr
from forecast.phase2.run_resolution_report import render_resolution_report

CUTOFF, HORIZON = 100, 200


# --------------------------------------------------------------------------
# The pass does not run early
# --------------------------------------------------------------------------


def test_it_refuses_to_run_before_the_season_ends() -> None:
    with pytest.raises(rr.ResolutionGateError, match="does not run early"):
        rr.assert_season_has_ended(today=date(2026, 9, 5))


def test_it_refuses_the_day_before_the_season_ends() -> None:
    with pytest.raises(rr.ResolutionGateError, match="Days remaining: 1"):
        rr.assert_season_has_ended(today=date(2026, 9, 26))


def test_it_runs_on_and_after_the_verified_end_date() -> None:
    for day in (date(2026, 9, 27), date(2026, 10, 3)):
        record = rr.assert_season_has_ended(today=day)
        assert record["season_has_ended"] is True
        assert record["verified_season_end_date"] == PROSPECTIVE_2026_SEASON_END_DATE.isoformat()


def test_the_guard_uses_the_verified_date_not_a_literal() -> None:
    record = rr.assert_season_has_ended(today=date(2026, 12, 1))
    assert record["verified_season_end_date"] == "2026-09-27"
    assert "maintainer-provided citation" in record["source"]


def test_a_snapshot_short_of_the_season_end_is_refused() -> None:
    class Snapshot:
        label = "2026-09-01"
        data_through_date = "2026-09-01"

    with pytest.raises(rr.ResolutionGateError, match="partial season"):
        rr.assert_snapshot_reaches_season_end(Snapshot())


def test_a_snapshot_reaching_the_season_end_is_accepted() -> None:
    class Snapshot:
        label = "2026-09-28"
        data_through_date = "2026-09-28"

    assert rr.assert_snapshot_reaches_season_end(Snapshot())["reaches_verified_season_end"]


def test_the_pass_requires_maintainer_authorization() -> None:
    for value in ("", "   "):
        with pytest.raises(rr.ResolutionGateError, match="authorize this second look"):
            rr.run(authorized_by=value)


def test_the_cli_makes_authorization_mandatory() -> None:
    parser = rr.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--authorized-by", "maintainer"])
    assert args.authorized_by == "maintainer"
    assert args.horizon_key == "H200"


# --------------------------------------------------------------------------
# Sealed predictions are verified, not trusted
# --------------------------------------------------------------------------


def test_a_moved_ledger_is_refused(tmp_path: Any, monkeypatch: Any) -> None:
    outputs = tmp_path / "sealed"
    outputs.mkdir()
    frame = pd.DataFrame({"batter": [1], "season": [2026]})
    frame.to_parquet(outputs / "pending.parquet", index=False)
    frame.to_parquet(outputs / "completed.parquet", index=False)
    (outputs / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "m" * 64,
                "artifact_sha256": {"pending.parquet": "0" * 64, "completed.parquet": "0" * 64},
            }
        )
    )
    monkeypatch.setitem(
        rr.FROZEN_PREDICTION_LEDGERS,
        "TEST",
        {
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            "outputs_dir": str(outputs),
            "pending_ledger": "pending.parquet",
            "completed_ledger": "completed.parquet",
            "result_manifest": "manifest.json",
            "n_pending_at_first_look": 1,
            "n_completed_at_first_look": 1,
        },
    )
    with pytest.raises(rr.ResolutionIntegrityError, match="predictions have moved"):
        rr.load_sealed_predictions("TEST")


def test_the_real_sealed_ledgers_are_declared_for_both_horizons() -> None:
    assert set(rr.FROZEN_PREDICTION_LEDGERS) == {"H100", "H200"}
    for ledger in rr.FROZEN_PREDICTION_LEDGERS.values():
        assert ledger["n_pending_at_first_look"] > 0


# --------------------------------------------------------------------------
# Outcomes are attached; predictions are never regenerated
# --------------------------------------------------------------------------


def _sealed_pending(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "batter": range(100, 100 + n),
            "batter_name": [f"H{i}" for i in range(n)],
            "season": 2026,
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            "contact_forecast": rng.normal(5, 1, n),
            "league_mean": 5.0,
            "shrunk_realized_persistence": rng.normal(5, 1, n),
            "shrunk_deserved_persistence": rng.normal(5, 1, n),
        }
    )


def test_attaching_outcomes_leaves_every_prediction_untouched(monkeypatch: Any) -> None:
    pending = _sealed_pending()
    targets = pd.DataFrame(
        {
            "batter": [100, 101, 102],
            "season": 2026,
            "window_id": ["w0", "w1", "w2"],
            "target_realized_rv_per_100": [4.0, 6.0, 5.0],
            "target_deserved_rv_per_100": [4.1, 6.1, 5.1],
            "target_start_date": "2026-06-01",
            "target_end_date": "2026-09-20",
        }
    )
    monkeypatch.setattr(rr, "build_windows", lambda *a, **k: targets)

    sealed = {"cutoff": CUTOFF, "horizon": HORIZON, "pending": pending}
    resolved, still_pending = rr.attach_end_of_season_outcomes(sealed, pd.DataFrame())

    assert len(resolved) == 3
    assert len(still_pending) == 3
    assert still_pending[rr.TARGET].isna().all()
    for column in rr.SEALED_PREDICTION_COLUMNS:
        merged = pd.concat([resolved, still_pending]).sort_values("batter")
        np.testing.assert_array_equal(
            merged[column].to_numpy(), pending.sort_values("batter")[column].to_numpy()
        )


def test_a_first_look_target_that_moved_is_refused(monkeypatch: Any) -> None:
    first_look = pd.DataFrame(
        {"batter": [1, 2], "season": 2026, "target_realized_rv_per_100": [5.0, 6.0]}
    )
    moved = pd.DataFrame(
        {"batter": [1, 2], "season": 2026, "target_realized_rv_per_100": [5.0, 6.5]}
    )
    monkeypatch.setattr(rr, "build_windows", lambda *a, **k: moved)
    sealed = {"cutoff": CUTOFF, "horizon": HORIZON, "first_look_completed": first_look}
    with pytest.raises(rr.ResolutionIntegrityError, match="target\\(s\\) changed"):
        rr.assert_first_look_targets_unchanged(sealed, pd.DataFrame())


def test_a_vanished_first_look_window_is_refused(monkeypatch: Any) -> None:
    first_look = pd.DataFrame(
        {"batter": [1, 2], "season": 2026, "target_realized_rv_per_100": [5.0, 6.0]}
    )
    monkeypatch.setattr(
        rr,
        "build_windows",
        lambda *a, **k: pd.DataFrame(
            {"batter": [1], "season": [2026], "target_realized_rv_per_100": [5.0]}
        ),
    )
    sealed = {"cutoff": CUTOFF, "horizon": HORIZON, "first_look_completed": first_look}
    with pytest.raises(rr.ResolutionIntegrityError, match="no longer resolve"):
        rr.assert_first_look_targets_unchanged(sealed, pd.DataFrame())


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------


def _cohort_frame(batters: range, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(batters)
    truth = rng.normal(5, 3, n)
    return pd.DataFrame(
        {
            "batter": list(batters),
            "batter_name": [f"H{i}" for i in batters],
            "season": 2026,
            "cutoff": CUTOFF,
            "horizon": HORIZON,
            "target_realized_rv_per_100": truth,
            "contact_forecast": truth + rng.normal(0, 2, n),
            "league_mean": 5.0,
            "shrunk_realized_persistence": truth + rng.normal(0, 3, n),
            "shrunk_deserved_persistence": truth + rng.normal(0, 2.5, n),
        }
    )


def test_the_incremental_cohort_must_be_disjoint_from_the_first_look() -> None:
    overlapping = _cohort_frame(range(0, 10), seed=1)
    sealed = {"first_look_completed": overlapping}
    with pytest.raises(rr.ResolutionIntegrityError, match="disjoint"):
        rr.build_cohorts(sealed, overlapping, overlapping.head(0))


def test_the_full_season_cohort_pools_first_look_and_incremental() -> None:
    first_look = _cohort_frame(range(0, 10), seed=1)
    incremental = _cohort_frame(range(10, 16), seed=2)
    never = _cohort_frame(range(16, 20), seed=3)
    cohorts = rr.build_cohorts({"first_look_completed": first_look}, incremental, never)
    counts = cohorts["counts"]
    assert counts["n_first_look_completed"] == 10
    assert counts["n_incremental"] == 6
    assert counts["n_full_season"] == 16
    assert counts["n_never_completed"] == 4
    assert counts["incremental_disjoint_from_first_look"] is True


# --------------------------------------------------------------------------
# Only the incremental cohort is classified
# --------------------------------------------------------------------------


def _evaluate(frame: pd.DataFrame, *, classify: bool, cohort: str) -> dict[str, Any]:
    return rr.evaluate_cohort(
        frame,
        cohort=cohort,
        cutoff=CUTOFF,
        horizon=HORIZON,
        design=BootstrapDesign(reps=200, seed=7),
        classify=classify,
    )


def test_the_incremental_cohort_is_classified() -> None:
    record = _evaluate(_cohort_frame(range(0, 40), seed=4), classify=True, cohort="incremental")
    assert record["evaluated"] is True
    assert record["may_decide_the_conclusion"] is True
    assert record["prespecified_classification"]["classification"] in {
        "established_incremental_success",
        "promising_but_inconclusive",
        "no_evidence_of_incremental_improvement",
        "evidence_against_incremental_value",
    }


def test_the_full_season_cohort_is_never_classified() -> None:
    record = _evaluate(_cohort_frame(range(0, 40), seed=5), classify=False, cohort="full_season")
    assert record["evaluated"] is True
    assert record["prespecified_classification"] is None
    assert record["may_decide_the_conclusion"] is False
    assert "NOT INDEPENDENT" in record["mandatory_label"]


def test_an_empty_cohort_is_reported_not_crashed() -> None:
    record = _evaluate(_cohort_frame(range(0, 0), seed=6), classify=True, cohort="incremental")
    assert record["evaluated"] is False
    assert record["n_windows"] == 0


def test_the_spec_contradiction_is_recorded_as_resolved() -> None:
    history = rr.COHORT_CLASSIFICATION_HISTORY
    assert history["was_a_contradiction"] is True
    assert history["resolved_by"] == "resolution_spec amendment 2"
    assert history["resolved_as"] == "the incremental cohort only"
    assert history["amended_before_any_outcome_was_opened"] is True
    assert history["runner_reads_the_flag_from_the_specification"] is True


def test_the_runner_reads_the_classification_flags_from_the_frozen_spec() -> None:
    from forecast.phase2.resolution_spec import build_resolution_specification

    flags = rr.assert_cohort_classification_is_consistent(build_resolution_specification())
    assert flags == {"incremental": True, "full_season": False, "never_completed": False}


def test_an_inconsistent_specification_stops_the_pass() -> None:
    from forecast.phase2.resolution_spec import build_resolution_specification

    spec = build_resolution_specification()
    spec["cohorts"]["full_season"]["four_way_classification_applied"] = True
    with pytest.raises(rr.ResolutionGateError, match="must not carry a classification"):
        rr.assert_cohort_classification_is_consistent(spec)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def _report(incremental: pd.DataFrame) -> str:
    metrics = {
        "cutoff": CUTOFF,
        "horizon": HORIZON,
        "primary_cohort": "incremental",
        "classification_applies_to": "the incremental cohort only",
        "cohort_classification_history": rr.COHORT_CLASSIFICATION_HISTORY,
        "incremental": _evaluate(incremental, classify=True, cohort="incremental"),
        "full_season": _evaluate(
            _cohort_frame(range(0, 50), seed=9), classify=False, cohort="full_season"
        ),
        "never_completed": {
            "cohort": "never_completed",
            "evaluated": False,
            "n_hitters": 12,
            "note": "Reported, never evaluated.",
        },
        "provenance": {
            "authorization_sha256": "a" * 64,
            "freeze_chain": {"r1": "b" * 64},
            "sealed_ledger_sha256": {"pending.parquet": "c" * 64},
            "snapshot_label": "2026-09-28",
            "snapshot_data_through": "2026-09-28",
        },
    }
    survivorship = {
        "status": "DESCRIPTIVE ONLY",
        "end_of_season_completion": {
            "n_hitters_with_a_forecast": 380,
            "n_completed_by_season_end": 200,
            "completion_rate": 0.526,
        },
        "largest_incremental_vs_first_look": [
            {
                "feature": "launch_speed_mean",
                "incremental_vs_first_look_standardized": -0.2,
                "incremental_vs_never_completed_standardized": 0.3,
            }
        ],
        "why_this_matters": "slower accumulators",
        "generalization_note": "conditional on reaching the horizon",
    }
    authorization = {
        "authorized_by": "maintainer",
        "authorized_at_utc": "2026-09-28T12:00:00+00:00",
        "acknowledged": "This is a SECOND unadjusted look at 2026.",
        "gate": {
            "season_has_ended": {"verified_season_end_date": "2026-09-27"},
            "snapshot_reaches_season_end": {
                "snapshot_label": "2026-09-28",
                "snapshot_data_through": "2026-09-28",
            },
        },
    }
    return render_resolution_report(
        authorization=authorization, metrics=metrics, survivorship=survivorship
    )


def test_the_warning_precedes_every_number() -> None:
    report = _report(_cohort_frame(range(0, 40), seed=8))
    warning = report.index("SECOND unadjusted look")
    assert warning < report.index("delta_MAE")
    assert warning < report.index("PRIMARY")


def test_the_report_gives_the_pooled_cohort_no_class_to_quote() -> None:
    report = _report(_cohort_frame(range(0, 40), seed=8))
    pooled = report[report.index("## SECONDARY") : report.index("## Reported, never evaluated")]
    assert "NOT INDEPENDENT" in pooled
    assert "No classification is assigned" in pooled
    for headline in ("PROMISING BUT INCONCLUSIVE", "ESTABLISHED INCREMENTAL SUCCESS"):
        assert headline not in pooled


def test_the_report_derives_its_sign_convention() -> None:
    report = _report(_cohort_frame(range(0, 40), seed=8))
    assert "delta_MAE = MAE(contact_forecast) - MAE(shrunk_deserved_persistence)" in report
    assert "delta_metric = metric(challenger) - metric(reference)" not in report


def test_an_empty_primary_cohort_does_not_fall_back() -> None:
    report = _report(_cohort_frame(range(0, 0), seed=8))
    assert "There is no primary result" in report
    assert "does not fall back to the full-season" in report
