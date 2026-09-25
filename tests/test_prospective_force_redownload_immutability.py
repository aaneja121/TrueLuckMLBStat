"""Contact Luck v1.1: `--force-redownload` must never overwrite or mutate an
already-completed immutable snapshot -- it may only refresh the shared,
mutable raw staging cache under `data/prospective/2026/`.

Exercises `run_prospective_snapshot` genuinely end-to-end (real
`build_snapshot_manifest`, `resolve_snapshot_conflict`, `_read_prior_
manifest`, and file-writing logic), with every NETWORK-touching or
heavy-model-training step replaced by a fast, deterministic, in-memory stub
-- this test never touches the network or trains a real model, but the
snapshot-immutability guarantee itself is exercised for real.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import prospective_ingestion as pi
import prospective_manifest as pm
import pytest
import run_v1_1_2026_scoring as runner
import v1_final_evaluation_manifest as manifest_mod

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


def _completeness_result() -> pi.DataThroughDateCompletenessResult:
    """A REAL `DataThroughDateCompletenessResult`, not a duck-typed stand-in.

    The stand-in this replaces carried only the attributes the guards of the
    day happened to read, so it silently went out of step every time a guard
    started reading a new one -- which is exactly how the season-end
    cross-check landed with two whole test files raising AttributeError.
    Constructing the real frozen dataclass makes that drift impossible: a new
    required field is a loud error here, at construction, instead of a
    surprise inside a monkeypatched call.

    `checked_at` is PINNED rather than real: it lands in the manifest's
    `schema_checks`, and `SnapshotManifest.deterministic_content_hash` pops
    only TOP-LEVEL wall-clock fields (`generated_at`, `retrieval_timestamps`),
    so a nested live timestamp would be hashed and make two otherwise
    identical runs look like a genuine conflict.

    2026-08-05 is an ordinary in-season date: a completed slate, at or before
    the recorded finale, so both season-end guards pass and these tests
    exercise the path they are actually about.
    """
    return pi.DataThroughDateCompletenessResult(
        data_through_date="2026-08-05",
        is_complete=True,
        total_games_on_date=15,
        incomplete_games=[],
        postponed_or_cancelled_games=[],
        checked_at="2026-01-01T00:00:00+00:00",
    )


@dataclass(frozen=True)
class _FakeNameReport:
    retrieved_at: str = "2026-01-01T00:00:00+00:00"

    def to_dict(self) -> dict[str, Any]:
        return {"requested_batter_count": 2, "resolved_batter_count": 0}


def _make_public_score_table(distinguishing_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter_id": [101, 102],
            "batter_name": pd.array([None, None], dtype="string"),
            "season": [2026, 2026],
            "qualification_status": ["qualified", "qualified"],
            "contact_luck_runs_per_100": [distinguishing_value, -distinguishing_value],
            "model_version": [{"contact": "baseline_v02"}, {"contact": "baseline_v02"}],
            "generated_at": ["2026-01-01T00:00:00+00:00"] * 2,
            "data_through_date": ["2026-04-15"] * 2,
            "component_status_reason_codes": [{}, {}],
        }
    )


def _stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    distinguishing_value: float,
    report_overrides: dict[str, Any] | None = None,
) -> None:
    """`report_overrides` is merged (top level) into the stub scoring
    report, so another module can reuse this same real-`run_prospective_
    snapshot` harness to exercise how a specific report field is persisted.
    """
    auth = SimpleNamespace(token="stub-token", granted_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(runner, "run_prospective_guards", lambda **kw: auth)
    monkeypatch.setattr(
        runner, "assert_data_through_date_is_complete", lambda *a, **kw: _completeness_result()
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
            "requested_date_range": ["2026-03-25", "2026-04-15"],
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "raw_file_sha256": f"raw-hash-{distinguishing_value}",
            "row_count": 100,
            "observed_date_coverage": ["2026-03-25", "2026-04-15"],
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
                "requested_end_date": "2026-04-15",
                "observed_min_game_date": "2026-03-25",
                "observed_max_game_date": "2026-04-15",
                "observed_game_dates": ["2026-03-25", "2026-04-15"],
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "row_count": 100,
                "sha256": f"raw-hash-{distinguishing_value}",
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

    scoring_df = pd.DataFrame({"game_date": ["2026-04-14", "2026-04-15"], "season": [2026, 2026]})
    monkeypatch.setattr(
        runner,
        "build_2026_scoring_dataset",
        lambda **kw: (
            scoring_df,
            {"venue_join": {}, "geometry_join": {}, "sprint_speed_join": {}},
        ),
    )

    # Version 1.4.0 Phase 3: run_prospective_snapshot now also builds a
    # canonical play ledger from artifacts.scoring_df/artifacts.ledger and
    # reads trained.contact_trained.variant for its metadata -- this stub
    # needs minimal-but-valid stand-ins for both (a single resolved row is
    # enough; this test's own focus is snapshot-immutability/hashing
    # behavior, not play-ledger content).
    fake_scoring_df = pd.DataFrame(
        {
            "game_pk": [700001],
            "at_bat_number": [10],
            "pitch_number": [3],
            "game_date": ["2026-04-15"],
            "season": [2026],
            "batter": [12345],
            "stand": ["L"],
            "launch_speed": [107.6],
            "launch_angle": [33.0],
            "bb_type": ["fly_ball"],
            "spray_angle_approx": [-5.9],
            "hit_distance_sc": [413.0],
            "outcome_class": ["out"],
        }
    )
    _fake_observed_rv = -0.25491574127584554
    _fake_expected_rv = 1.290656107968476
    fake_ledger = pd.DataFrame(
        {
            # Phase 3.1: play_ledger_export.py sources observed_run_value/
            # contact_luck_runs from the Rf columns (observed_final_run_
            # value/final_result_surprise), not the Rc columns -- this stub
            # only needs to be internally consistent (Rf - E0 ==
            # final_result_surprise), not to model any real Rc/Rf
            # divergence, since this test's focus is snapshot-immutability/
            # hashing, not play-ledger semantics.
            "observed_final_run_value": [_fake_observed_rv],
            "baseline_expected_contact_run_value": [_fake_expected_rv],
            "final_result_surprise": [_fake_observed_rv - _fake_expected_rv],
            "p_out": [0.05692444081479338],
            "p_single": [0.0020191686532899677],
            "p_double": [0.019210357764228986],
            "p_triple": [0.003476039197145578],
            "p_home_run": [0.9183699935705423],
        }
    )
    fake_artifacts = SimpleNamespace(
        report={
            "model_selection_winners": {
                "outfield": "measured_contact_only_v07",
                "infield": "infield_hgb_v08",
                "advancement": "advancement_speed_v09",
            },
            "component_model_status": {"infield": "calibrated"},
            **(report_overrides or {}),
        },
        scoring_df=fake_scoring_df,
        ledger=fake_ledger,
    )
    fake_trained = SimpleNamespace(
        contact_trained=SimpleNamespace(variant="unweighted_probability_baseline")
    )
    monkeypatch.setattr(
        runner, "train_and_score_2026", lambda *a, **kw: (fake_artifacts, fake_trained)
    )

    table = _make_public_score_table(distinguishing_value)
    monkeypatch.setattr(runner, "build_public_score_table", lambda artifacts: table)
    monkeypatch.setattr(runner, "assign_official_ranks", lambda t: t)
    monkeypatch.setattr(runner, "most_favorable_leaderboard", lambda t: t.head(1))
    monkeypatch.setattr(runner, "least_favorable_leaderboard", lambda t: t.tail(1))
    monkeypatch.setattr(runner, "fetch_player_names", lambda ids: pd.DataFrame())
    monkeypatch.setattr(
        runner, "apply_player_name_overlay", lambda t, names_df: (t, _FakeNameReport())
    )

    monkeypatch.setattr(manifest_mod, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))
    monkeypatch.setattr(pm, "FROZEN_ARTIFACT_RELATIVE_PATHS", ("frozen.py",))


def _snapshot_files(*roots: Path) -> dict[str, tuple[float, bytes]]:
    """Snapshot every file under the given roots (mtime + content) -- deliberately
    NEVER passed the git repo root, since `assert_clean_working_tree`'s `git
    status --porcelain` call can touch `.git/index`'s own mtime as an
    incidental side effect unrelated to snapshot-output immutability.
    """
    files: dict[str, tuple[float, bytes]] = {}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file():
                files[str(p)] = (p.stat().st_mtime, p.read_bytes())
    return files


@pytest.fixture
def _isolated_dirs(tmp_path: Path) -> tuple[Path, Path]:
    outputs_root = tmp_path / "outputs" / "prospective" / "v1_1"
    artifacts_root = tmp_path / "artifacts" / "prospective" / "v1_1"
    outputs_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    return outputs_root, artifacts_root


def test_first_run_writes_a_new_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_pipeline(monkeypatch, distinguishing_value=1.0)

    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    assert result["status"] == "written"
    assert (outputs_root / "2026-04-15" / "public_score.json").exists()
    assert (artifacts_root / "2026-04-15" / "manifest.json").exists()


def test_play_ledger_and_metadata_written_and_covered_by_generic_integrity_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    """Version 1.4.0 Phase 3, Section 6: play_ledger.parquet/play_ledger_
    metadata.json must be written into the snapshot AND automatically
    covered by the SAME generic, no-special-case output-hashing loop every
    other output file already gets -- never a dedicated archive code path.
    """
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)
    _stub_pipeline(monkeypatch, distinguishing_value=1.0)

    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    assert result["status"] == "written"

    play_ledger_path = outputs_root / "2026-04-15" / "play_ledger.parquet"
    play_ledger_metadata_path = outputs_root / "2026-04-15" / "play_ledger_metadata.json"
    assert play_ledger_path.exists()
    assert play_ledger_metadata_path.exists()

    # The manifest's own output_hashes (built by the SAME generic
    # `outputs_snapshot_dir.iterdir()` loop that hashes public_score.* etc.)
    # must include both new files -- proving no special-case archive code
    # was needed for them to participate in snapshot integrity.
    output_hashes = result["manifest"]["output_hashes"]
    assert "outputs/play_ledger.parquet" in output_hashes
    assert "outputs/play_ledger_metadata.json" in output_hashes
    assert output_hashes["outputs/play_ledger.parquet"] == pm.compute_file_sha256(play_ledger_path)
    assert output_hashes["outputs/play_ledger_metadata.json"] == pm.compute_file_sha256(
        play_ledger_metadata_path
    )

    # integrity_hashes.json (the artifact archive_snapshot.py/dashboard
    # validation actually reads) must ALSO cover both files, via the same
    # generic mechanism.
    integrity_hashes_path = artifacts_root / "2026-04-15" / "integrity_hashes.json"
    integrity_hashes = json.loads(integrity_hashes_path.read_text())
    assert "outputs/play_ledger.parquet" in integrity_hashes
    assert "outputs/play_ledger_metadata.json" in integrity_hashes


def test_force_redownload_with_identical_result_is_idempotent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)  # "redownload" -> same content
    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        force_redownload=True,
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    assert result["status"] == "idempotent_no_op"
    assert (
        _snapshot_files(outputs_root, artifacts_root) == before
    )  # byte-for-byte and mtime-for-mtime unchanged


def test_force_redownload_that_would_change_the_result_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    # Simulate a corrected/changed raw source: --force-redownload now produces
    # a DIFFERENT public score table for the SAME --data-through date.
    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    with pytest.raises(pm.SnapshotConflictError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15",
            force_redownload=True,
            repo_root=repo_root,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

    # The already-completed snapshot must be COMPLETELY untouched.
    assert _snapshot_files(outputs_root, artifacts_root) == before


def test_force_redownload_false_also_never_overwrites_a_differing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    """The conflict guard applies regardless of `--force-redownload` --
    confirms the immutability guarantee doesn't depend on that flag at all.
    """
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    with pytest.raises(pm.SnapshotConflictError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15",
            force_redownload=False,
            repo_root=repo_root,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )

    assert _snapshot_files(outputs_root, artifacts_root) == before


def test_a_different_snapshot_label_writes_a_separate_directory_not_a_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _isolated_dirs: tuple[Path, Path]
) -> None:
    outputs_root, artifacts_root = _isolated_dirs
    repo_root = _init_clean_repo(tmp_path)

    _stub_pipeline(monkeypatch, distinguishing_value=1.0)
    runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    before = _snapshot_files(outputs_root, artifacts_root)

    _stub_pipeline(monkeypatch, distinguishing_value=2.0)
    result = runner.run_prospective_snapshot(
        data_through_date="2026-04-15",
        snapshot_label="corrected",
        force_redownload=True,
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )

    assert result["status"] == "written"
    assert (outputs_root / "2026-04-15__corrected" / "public_score.json").exists()
    # The FIRST snapshot's own files are still completely untouched.
    original_files = {k: v for k, v in before.items() if "2026-04-15__corrected" not in k}
    after = _snapshot_files(outputs_root, artifacts_root)
    for key, value in original_files.items():
        assert after[key] == value
