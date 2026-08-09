"""Contact Luck v1.1.1 presentation patch: the name overlay must propagate
into BOTH leaderboard exports, not just `public_score.*`.

Regression coverage for the bug found in the real 2026-08-05 snapshot:
`favorable_leaderboard.json`/`unfavorable_leaderboard.json` showed
`batter_name = null` for every row even though `public_score.json` had
resolved names, because the leaderboards were sliced off `public_score_table`
BEFORE the name overlay ran. That snapshot is immutable and untouched by this
fix -- these tests only cover the CODE fix in `run_v1_1_2026_scoring.py` and
its underlying invariant (`resolve after ranks, before leaderboard slicing`).

Required guarantees (per the task that produced this patch):
  - resolved public-score names appear in both leaderboard files
  - batter_id, score, rank, interval, and row order remain unchanged
  - unresolved names remain null with a reason code
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import prospective_manifest as pm
import prospective_player_names as ppn
import pytest
import run_v1_1_2026_scoring as runner
import v1_final_evaluation_manifest as manifest_mod

from mlb_luck_score.scoring.leaderboard import (
    assign_official_ranks,
    least_favorable_leaderboard,
    most_favorable_leaderboard,
)
from mlb_luck_score.scoring.public_score_table import build_public_score_table

_GIT_ENV_ARGS = [
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.com",
    "-c",
    "commit.gpgsign=false",
]


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *_GIT_ENV_ARGS, *args], cwd=repo_root, check=True, capture_output=True, text=True
    )


def _init_clean_repo(root: Path) -> Path:
    repo_root = root / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-q")
    (repo_root / "frozen.py").write_text("X = 1\n")
    _git(repo_root, "add", "frozen.py")
    _git(repo_root, "commit", "-q", "-m", "init")
    return repo_root


def _partial_names_df(batter_ids: list[int]) -> pd.DataFrame:
    """Resolve every OTHER batter_id -- deliberately leaves some unresolved
    so the "unresolved stays null with a reason code" guarantee is exercised
    alongside the "resolved names propagate" guarantee, in the same table.
    """
    rows = []
    for i, batter_id in enumerate(sorted(set(batter_ids))):
        if i % 2 == 0:
            rows.append(
                {
                    "batter_id": batter_id,
                    "full_name": f"Test Player {batter_id}",
                    "retrieved_at": "2026-08-06T00:00:00+00:00",
                    "source": "MLB Stats API (/people)",
                }
            )
        # else: omitted entirely -> REASON_NOT_FOUND_IN_SOURCE
    return pd.DataFrame(rows, columns=["batter_id", "full_name", "retrieved_at", "source"])


# ---------------------------------------------------------------------------
# Unit level: the ordering itself, against the real frozen leaderboard
# functions and the real name-overlay function (no network, no stubbing)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ranked_table(v012_public_score_artifacts) -> pd.DataFrame:
    table = build_public_score_table(v012_public_score_artifacts)
    return assign_official_ranks(table)


def test_buggy_order_reproduces_the_null_name_leaderboard(_ranked_table: pd.DataFrame) -> None:
    """Documents WHY the ordering matters: building leaderboards from the
    table BEFORE applying the overlay leaves them permanently name-less, even
    though `public_score_table` itself gets names moments later -- exactly
    the real 2026-08-05 bug.
    """
    names_df = _partial_names_df(_ranked_table["batter_id"].tolist())

    buggy_favorable = most_favorable_leaderboard(_ranked_table)  # BEFORE overlay
    _overlaid_table, _report = ppn.apply_player_name_overlay(_ranked_table, names_df)

    assert buggy_favorable["batter_name"].isna().all()


def test_fixed_order_propagates_resolved_names_into_both_leaderboards(
    _ranked_table: pd.DataFrame,
) -> None:
    names_df = _partial_names_df(_ranked_table["batter_id"].tolist())
    overlaid_table, _report = ppn.apply_player_name_overlay(_ranked_table, names_df)

    favorable = most_favorable_leaderboard(overlaid_table)  # AFTER overlay
    unfavorable = least_favorable_leaderboard(overlaid_table)

    resolved_ids = set(names_df["batter_id"])
    for leaderboard in (favorable, unfavorable):
        resolved_rows = leaderboard[leaderboard["batter_id"].isin(resolved_ids)]
        if len(resolved_rows):
            assert resolved_rows["batter_name"].notna().all()
            for _, row in resolved_rows.iterrows():
                assert row["batter_name"] == f"Test Player {row['batter_id']}"


def test_unresolved_names_stay_null_with_a_reason_code_through_the_leaderboard_flow(
    _ranked_table: pd.DataFrame,
) -> None:
    names_df = _partial_names_df(_ranked_table["batter_id"].tolist())
    resolved_ids = set(names_df["batter_id"])
    overlaid_table, report = ppn.apply_player_name_overlay(_ranked_table, names_df)

    favorable = most_favorable_leaderboard(overlaid_table)
    unresolved_rows = favorable[~favorable["batter_id"].isin(resolved_ids)]
    if len(unresolved_rows):
        assert unresolved_rows["batter_name"].isna().all()
    for batter_id in favorable.loc[~favorable["batter_id"].isin(resolved_ids), "batter_id"]:
        assert (
            report.reason_codes_by_batter_id[str(int(batter_id))] == ppn.REASON_NOT_FOUND_IN_SOURCE
        )


def test_applying_the_overlay_before_leaderboard_slicing_never_changes_score_rank_interval_or_order(
    _ranked_table: pd.DataFrame,
) -> None:
    """The core safety guarantee: moving the overlay earlier changes ONLY
    `batter_name` -- batter_id, score, rank, interval, and row order are
    identical to the leaderboard built from the un-overlaid table.
    """
    names_df = _partial_names_df(_ranked_table["batter_id"].tolist())

    before_favorable = most_favorable_leaderboard(_ranked_table)
    before_unfavorable = least_favorable_leaderboard(_ranked_table)

    overlaid_table, _report = ppn.apply_player_name_overlay(_ranked_table, names_df)
    after_favorable = most_favorable_leaderboard(overlaid_table)
    after_unfavorable = least_favorable_leaderboard(overlaid_table)

    compare_cols = [
        "batter_id",
        "contact_luck_runs_per_100",
        "official_rank_favorable",
        "lower_95_interval",
        "upper_95_interval",
    ]
    pd.testing.assert_frame_equal(
        before_favorable[compare_cols].reset_index(drop=True),
        after_favorable[compare_cols].reset_index(drop=True),
    )
    compare_cols_unfav = [c.replace("_favorable", "_unfavorable") for c in compare_cols]
    pd.testing.assert_frame_equal(
        before_unfavorable[compare_cols_unfav].reset_index(drop=True),
        after_unfavorable[compare_cols_unfav].reset_index(drop=True),
    )
    # Row order itself (batter_id sequence) is untouched.
    assert list(before_favorable["batter_id"]) == list(after_favorable["batter_id"])
    assert list(before_unfavorable["batter_id"]) == list(after_unfavorable["batter_id"])


# ---------------------------------------------------------------------------
# Integration level: run_prospective_snapshot end-to-end (real
# build_public_score_table/assign_official_ranks/leaderboard/overlay
# functions; only network-touching steps and model training are stubbed)
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_dirs(tmp_path: Path) -> tuple[Path, Path]:
    outputs_root = tmp_path / "outputs" / "prospective" / "v1_1"
    artifacts_root = tmp_path / "artifacts" / "prospective" / "v1_1"
    outputs_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return outputs_root, artifacts_root


class _FakeCompletenessResult:
    def to_dict(self) -> dict[str, Any]:
        return {"is_complete": True}


def _stub_network_and_training_only(
    monkeypatch: pytest.MonkeyPatch, v012_public_score_artifacts
) -> None:
    """Unlike `test_prospective_force_redownload_immutability.py`, this
    leaves `build_public_score_table`, `assign_official_ranks`,
    `most_favorable_leaderboard`, `least_favorable_leaderboard`, and
    `apply_player_name_overlay` as the REAL, unstubbed frozen functions --
    exactly the functions whose call ORDER inside `run_prospective_snapshot`
    caused the bug this patch fixes.
    """
    auth = SimpleNamespace(token="stub-token", granted_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(runner, "run_prospective_guards", lambda **kw: auth)
    monkeypatch.setattr(
        runner, "assert_data_through_date_is_complete", lambda *a, **kw: _FakeCompletenessResult()
    )
    # v1.1.2: the final independent pre-scoring coverage check also hits the
    # network by default -- stub it here too, same as the other guards.
    monkeypatch.setattr(
        runner,
        "assert_scoring_dataset_satisfies_coverage_contract",
        lambda *a, **kw: {
            "verified_at": "2026-01-01T00:00:00+00:00",
            "observed_game_date_count": 0,
            "missing_completed_game_dates": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "ingest_2026_raw_statcast",
        lambda **kw: {
            "schema_check": {"missing_required_columns": []},
            "missing_dates_within_range": [],
            "requested_date_range": ["2026-03-25", "2026-08-05"],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": "raw-hash",
            "row_count": 100,
            "observed_date_coverage": ["2026-03-25", "2026-08-05"],
            "cache_coverage_validation": {
                "decision": "refreshed",
                "reason": "no_cached_raw_file",
                "cached_provenance": None,
                "missing_completed_game_dates": [],
                "checked_at": "2026-01-01T00:00:00+00:00",
            },
            "cached_provenance_before_decision": None,
            "final_raw_data_provenance": {
                "requested_start_date": "2026-03-25",
                "requested_end_date": "2026-08-05",
                "observed_min_game_date": "2026-03-25",
                "observed_max_game_date": "2026-08-05",
                "observed_game_dates": ["2026-03-25", "2026-08-05"],
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "row_count": 100,
                "sha256": "raw-hash",
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "ingest_2026_game_metadata",
        lambda **kw: {
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": "game-metadata-hash",
            "game_count": 10,
        },
    )
    monkeypatch.setattr(
        runner,
        "ingest_2026_sprint_speed",
        lambda **kw: {
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": "sprint-speed-hash",
            "player_count": 5,
        },
    )
    scoring_df = pd.DataFrame({"game_date": ["2026-08-04", "2026-08-05"], "season": [2026, 2026]})
    monkeypatch.setattr(
        runner,
        "build_2026_scoring_dataset",
        lambda **kw: (scoring_df, {"venue_join": {}, "geometry_join": {}, "sprint_speed_join": {}}),
    )
    # The shared `v012_public_score_artifacts` fixture's `.report` has
    # `model_selection_winners` but not `component_model_status` (other test
    # files don't need it) -- `run_prospective_snapshot` reads both, so this
    # stub augments a COPY of the fixture's report rather than the fixture
    # object itself (fixture is module/session-scoped and reused elsewhere).
    augmented_report = {**v012_public_score_artifacts.report, "component_model_status": {}}
    stub_artifacts = dataclasses.replace(v012_public_score_artifacts, report=augmented_report)
    monkeypatch.setattr(
        runner,
        "train_and_score_2026",
        lambda *a, **kw: (stub_artifacts, SimpleNamespace()),
    )

    def _fake_fetch_player_names(batter_ids: list[int]) -> pd.DataFrame:
        return _partial_names_df(batter_ids)

    monkeypatch.setattr(runner, "fetch_player_names", _fake_fetch_player_names)

    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))


def test_written_leaderboard_files_contain_the_resolved_names_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_dirs: tuple[Path, Path],
    v012_public_score_artifacts,
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_network_and_training_only(monkeypatch, v012_public_score_artifacts)

    result = runner.run_prospective_snapshot(
        data_through_date="2026-08-05",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    assert result["status"] == "written"

    snapshot_dir = outputs_root / "2026-08-05"
    public_score = {
        int(r["batter_id"]): r for r in json.loads((snapshot_dir / "public_score.json").read_text())
    }
    favorable = json.loads((snapshot_dir / "favorable_leaderboard.json").read_text())
    unfavorable = json.loads((snapshot_dir / "unfavorable_leaderboard.json").read_text())

    resolved_in_public_score = {
        bid: row["batter_name"] for bid, row in public_score.items() if row["batter_name"]
    }
    assert resolved_in_public_score, "fixture produced no resolved names -- test would be vacuous"

    for leaderboard in (favorable, unfavorable):
        for row in leaderboard:
            bid = int(row["batter_id"])
            if bid in resolved_in_public_score:
                assert row["batter_name"] == resolved_in_public_score[bid], (
                    f"batter_id={bid} has a resolved name in public_score.json but the "
                    "leaderboard export disagrees or is null -- the v1.1.1 regression"
                )


def test_written_leaderboard_rows_match_public_score_rank_and_score_for_shared_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_dirs: tuple[Path, Path],
    v012_public_score_artifacts,
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_network_and_training_only(monkeypatch, v012_public_score_artifacts)

    runner.run_prospective_snapshot(
        data_through_date="2026-08-05",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    snapshot_dir = outputs_root / "2026-08-05"
    public_score = {
        int(r["batter_id"]): r for r in json.loads((snapshot_dir / "public_score.json").read_text())
    }
    favorable = json.loads((snapshot_dir / "favorable_leaderboard.json").read_text())

    for row in favorable:
        bid = int(row["batter_id"])
        source = public_score[bid]
        assert row["contact_luck_runs_per_100"] == source["contact_luck_runs_per_100"]
        assert row["official_rank_favorable"] == source["official_rank_favorable"]
        assert row["lower_95_interval"] == source["lower_95_interval"]
        assert row["upper_95_interval"] == source["upper_95_interval"]

    # Row order is the frozen ranking order (favorable rank ascending), unrelated to names.
    ranks = [row["official_rank_favorable"] for row in favorable]
    assert ranks == sorted(ranks)


def test_unresolved_batter_names_are_null_in_the_written_leaderboards_with_a_reason_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_dirs: tuple[Path, Path],
    v012_public_score_artifacts,
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_network_and_training_only(monkeypatch, v012_public_score_artifacts)

    runner.run_prospective_snapshot(
        data_through_date="2026-08-05",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    snapshot_dir = outputs_root / "2026-08-05"
    favorable = json.loads((snapshot_dir / "favorable_leaderboard.json").read_text())
    name_report = json.loads((snapshot_dir / "name_resolution_report.json").read_text())

    unresolved_ids = set(name_report["unresolved_batter_ids"])
    assert unresolved_ids, "fixture produced no unresolved names -- test would be vacuous"

    for row in favorable:
        bid = int(row["batter_id"])
        if bid in unresolved_ids:
            assert row["batter_name"] is None
            assert (
                name_report["reason_codes_by_batter_id"][str(bid)] == ppn.REASON_NOT_FOUND_IN_SOURCE
            )
