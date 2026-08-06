"""Contact Luck v1.0, Task 33 category 5: isolated namespace.

Verifies every 2025 path resolves beneath the isolated `data/
final_evaluation/2025`, `outputs/final_evaluation/v1`, or `artifacts/
final_evaluation/v1` roots, that path traversal or a development-cache path
is rejected, and that a development runner cannot discover final-evaluation
data placed only in the isolated namespace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import run_v1_final_evaluation as rv1

from mlb_luck_score.config import RAW_DATA_DIR, development_raw_path, game_metadata_path
from mlb_luck_score.data.join_venue_metadata import JoinVenueMetadataError, load_game_metadata


def test_default_raw_dir_is_under_data_final_evaluation_2025() -> None:
    assert rv1.FINAL_EVALUATION_RAW_DIR.is_relative_to(rv1.PROJECT_ROOT / "data")
    assert rv1.FINAL_EVALUATION_RAW_DIR.parts[-3:] == ("data", "final_evaluation", "2025")


def test_default_outputs_dir_is_under_outputs_final_evaluation_v1() -> None:
    assert rv1.FINAL_EVALUATION_OUTPUTS_DIR.parts[-3:] == ("outputs", "final_evaluation", "v1")


def test_default_artifacts_dir_is_under_artifacts_final_evaluation_v1() -> None:
    assert rv1.FINAL_EVALUATION_ARTIFACTS_DIR.parts[-3:] == (
        "artifacts",
        "final_evaluation",
        "v1",
    )


def test_seal_path_resolves_inside_artifacts_namespace() -> None:
    assert rv1.SEAL_PATH.is_relative_to(rv1.FINAL_EVALUATION_ARTIFACTS_DIR)


def test_raw_statcast_path_resolves_inside_the_namespace() -> None:
    path = development_raw_path(rv1.FINAL_EVALUATION_RAW_DIR, 2025)
    assert path.is_relative_to(rv1.FINAL_EVALUATION_RAW_DIR)


def test_namespace_guard_accepts_a_path_within_the_root(tmp_path: Path) -> None:
    root = tmp_path / "final_evaluation" / "2025"
    root.mkdir(parents=True)
    nested = root / "subdir" / "file.parquet"
    rv1._assert_within_namespace(nested, root, label="test")  # must not raise


def test_namespace_guard_rejects_a_sibling_development_cache_path(tmp_path: Path) -> None:
    root = tmp_path / "final_evaluation" / "2025"
    root.mkdir(parents=True)
    outside = tmp_path / "raw"
    outside.mkdir()
    with pytest.raises(rv1.NamespaceViolationError):
        rv1._assert_within_namespace(outside, root, label="test")


def test_namespace_guard_rejects_path_traversal_escape(tmp_path: Path) -> None:
    root = tmp_path / "final_evaluation" / "2025"
    root.mkdir(parents=True)
    traversal = root / ".." / ".." / "raw"
    with pytest.raises(rv1.NamespaceViolationError):
        rv1._assert_within_namespace(traversal, root, label="test")


def test_ingest_raw_statcast_rejects_output_dir_outside_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeManifest:
        def content_hash(self) -> str:
            return "x"

    auth = rv1._mint_authorization(_FakeManifest())  # type: ignore[arg-type]
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.ingest_2025_raw_statcast(authorization=auth, output_dir=RAW_DATA_DIR)


def test_ingest_game_metadata_rejects_output_dir_outside_namespace() -> None:
    class _FakeManifest:
        def content_hash(self) -> str:
            return "x"

    auth = rv1._mint_authorization(_FakeManifest())  # type: ignore[arg-type]
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.ingest_2025_game_metadata(authorization=auth, output_dir=RAW_DATA_DIR)


def test_ingest_sprint_speed_rejects_output_dir_outside_namespace() -> None:
    class _FakeManifest:
        def content_hash(self) -> str:
            return "x"

    auth = rv1._mint_authorization(_FakeManifest())  # type: ignore[arg-type]
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.ingest_2025_sprint_speed(authorization=auth, output_dir=RAW_DATA_DIR)


def test_build_evaluation_dataset_rejects_raw_dir_outside_namespace() -> None:
    class _FakeManifest:
        def content_hash(self) -> str:
            return "x"

    auth = rv1._mint_authorization(_FakeManifest())  # type: ignore[arg-type]
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.build_2025_evaluation_dataset(authorization=auth, raw_dir=RAW_DATA_DIR)


def test_write_outputs_rejects_output_dir_outside_namespace() -> None:
    import pandas as pd

    empty = pd.DataFrame({"a": []})
    with pytest.raises(rv1.NamespaceViolationError):
        rv1._write_outputs(empty, empty, empty, {}, output_dir=RAW_DATA_DIR)


def test_write_seal_rejects_seal_path_outside_namespace(tmp_path: Path) -> None:
    class _FakeManifest:
        repository_commit = "abc"

        def content_hash(self) -> str:
            return "x"

    with pytest.raises(rv1.NamespaceViolationError):
        rv1.write_seal(
            manifest=_FakeManifest(),  # type: ignore[arg-type]
            report={},
            seal_path=tmp_path / "seal.json",
        )


def test_archive_prior_outputs_rejects_dirs_outside_namespace(tmp_path: Path) -> None:
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.archive_prior_outputs(outputs_dir=tmp_path, artifacts_dir=tmp_path)


def test_development_file_discovery_cannot_find_final_evaluation_data(tmp_path: Path) -> None:
    """2025 game metadata written ONLY into the isolated namespace must be
    invisible to a development runner pointed at the development raw cache.
    """
    isolated_root = tmp_path / "final_evaluation" / "2025"
    isolated_root.mkdir(parents=True)
    isolated_path = game_metadata_path(isolated_root, 2025)
    isolated_path.write_text("not a real parquet file, presence is what matters")

    dev_raw_dir = tmp_path / "raw"
    dev_raw_dir.mkdir()

    with pytest.raises(JoinVenueMetadataError, match="Missing game-metadata file"):
        load_game_metadata(dev_raw_dir, [2025])
